// =============================================================================
// Module  : npu_ctrl
// Project : NPU_prj
// Desc    : NPU top-level controller FSM with M×N Tile-Loop support.
//
//           Orchestrates M×N tile iterations:
//             for i in 0..M-1:
//               for j in 0..N-1:
//                 Load W[:,j] (K weights) → PPBuf_W
//                 Load A[i,:] (K activations) → PPBuf_A
//                 Compute: PE OS/WS for K elements
//                 Drain PE flush
//                 Write back C[i][j] → DRAM at R_BASE + (i*N+j)*4
//
//           When M=1, N=1: degenerates to original single-tile behavior.
//
//   States:
//     S_IDLE       - Wait for CPU start command
//     S_TILE_LOAD  - Load one tile's W and A into PPBufs
//     S_PRELOAD    - Wait 1 cycle for PPBuf swap to propagate
//     S_COMPUTE    - PE computing
//     S_DRAIN      - Assert pe_flush to drain accumulator
//     S_DRAIN2     - Wait for flush to propagate through pipeline
//     S_WRITE_BACK - Initiate DMA write-back for this tile's result
//     S_WB_WAIT    - Wait for DMA write-back to complete
//     S_DONE       - Assert done + IRQ, return to IDLE
// =============================================================================

`timescale 1ns/1ps

module npu_ctrl #(
    parameter ROWS   = 4,
    parameter COLS   = 4,
    parameter DATA_W = 16,
    parameter ACC_W  = 32
)(
    input  wire              clk,
    input  wire              rst_n,
    // Config from register file
    input  wire [31:0]       ctrl_reg,
    input  wire [31:0]       m_dim,
    input  wire [31:0]       n_dim,
    input  wire [31:0]       k_dim,
    input  wire [31:0]       w_addr,
    input  wire [31:0]       a_addr,
    input  wire [31:0]       r_addr,
    input  wire [7:0]        arr_cfg,
    // Status outputs
    output reg               busy,
    output reg               done,
    // DMA interface
    output reg               dma_w_start,
    input  wire              dma_w_done,
    output reg [31:0]        dma_w_addr,
    output reg [15:0]        dma_w_len,
    output reg               dma_a_start,
    input  wire              dma_a_done,
    output reg [31:0]        dma_a_addr,
    output reg [15:0]        dma_a_len,
    output reg               dma_r_start,
    input  wire              dma_r_done,
    output reg [31:0]        dma_r_addr,
    output reg [15:0]        dma_r_len,
    // PE array control
    output reg               pe_en,
    output reg               pe_flush,
    output reg               pe_mode,     // 0=INT8, 1=FP16
    output reg               pe_stat,     // 0=WS, 1=OS
    output reg               pe_load_w,   // WS mode: latch weight into PE
    // Ping-Pong Buffer status
    input  wire              w_ppb_ready,
    input  wire              w_ppb_empty,
    input  wire              a_ppb_ready,
    input  wire              a_ppb_empty,
    // Ping-Pong Buffer control
    output reg               w_ppb_swap,
    output reg               a_ppb_swap,
    output reg               w_ppb_clear,
    output reg               a_ppb_clear,
    // Result FIFO clear
    output reg               r_fifo_clear,
    // Interrupt
    output reg               irq
);

// ---------------------------------------------------------------------------
// FSM states
// ---------------------------------------------------------------------------
localparam S_IDLE       = 4'd0;
localparam S_TILE_LOAD  = 4'd1;   // Load W[:,j] and A[i,:] for one tile
localparam S_PRELOAD    = 4'd2;   // Wait 1 cycle for PPBuf swap
localparam S_COMPUTE    = 4'd3;   // PE computing
localparam S_DRAIN      = 4'd4;   // PE flush accumulators
localparam S_WRITE_BACK = 4'd5;   // Start DMA write-back
localparam S_WB_WAIT    = 4'd6;   // Wait for DMA write-back
localparam S_DONE       = 4'd7;
localparam S_DRAIN2     = 4'd9;   // Pipeline drain

reg [3:0] state;

wire cfg_start  = ctrl_reg[0];
wire cfg_abort  = ctrl_reg[1];
wire [1:0] cfg_mode   = ctrl_reg[3:2];   // 00=INT8, 10=FP16
wire [1:0] cfg_stat   = ctrl_reg[5:4];   // 0=WS, 1=OS

// Edge detect for start
reg cfg_start_d1;
always @(posedge clk) begin
    if (!rst_n) cfg_start_d1 <= 0;
    else cfg_start_d1 <= cfg_start;
end
wire cfg_start_rise = cfg_start && !cfg_start_d1;

// ---------------------------------------------------------------------------
// Mode decode (combinational)
// ---------------------------------------------------------------------------
always @(*) begin
    case (cfg_mode)
        2'b00: pe_mode = 1'b0;   // INT8
        2'b01: pe_mode = 1'b1;   // INT16
        2'b10: pe_mode = 1'b1;   // FP16
        default: pe_mode = 1'b0;
    endcase
end

// ---------------------------------------------------------------------------
// Data size (bytes per element)
// ---------------------------------------------------------------------------
wire [1:0] data_bytes = (cfg_mode == 2'b00) ? 2'd1 : 2'd2;

// ---------------------------------------------------------------------------
// Tile-loop counters: i ∈ [0, M-1], j ∈ [0, N-1]
// ---------------------------------------------------------------------------
reg [31:0] tile_i;  // current row tile index
reg [31:0] tile_j;  // current column tile index

// One tile's weight length = K * data_bytes
// One tile's activation length = K * data_bytes  (one row of A)
wire [15:0] tile_w_len = k_dim[15:0] * {14'b0, data_bytes};
wire [15:0] tile_a_len = k_dim[15:0] * {14'b0, data_bytes};

// Byte address for W[:,j] = W_BASE + j * K * data_bytes
// Byte address for A[i,:] = A_BASE + i * K * data_bytes
wire [31:0] cur_w_addr = w_addr + tile_j * {16'b0, tile_w_len};
wire [31:0] cur_a_addr = a_addr + tile_i * {16'b0, tile_a_len};

// Byte address for result C[i][j] = R_BASE + (i*N + j) * ACC_BYTES
wire [31:0] cur_r_addr = r_addr + (tile_i * n_dim + tile_j) * (ACC_W/8);

// Result DMA: always 1 word (one FP32/INT32 result per tile)
wire [15:0] tile_r_len = 16'd4;  // ACC_W/8 bytes

// ---------------------------------------------------------------------------
// Track DMA completion (latched)
// ---------------------------------------------------------------------------
reg dma_w_done_r, dma_a_done_r, dma_r_done_r;
wire dma_all_load_done = dma_w_done_r && dma_a_done_r;

always @(posedge clk) begin
    if (!rst_n) begin
        dma_w_done_r <= 0;
        dma_a_done_r <= 0;
        dma_r_done_r <= 0;
    end else begin
        if (dma_w_done) dma_w_done_r <= 1;
        if (dma_a_done) dma_a_done_r <= 1;
        if (dma_r_done) dma_r_done_r <= 1;
    end
end

// ---------------------------------------------------------------------------
// WS consume counter
// ---------------------------------------------------------------------------
reg [15:0] ws_consume_cnt;

// ---------------------------------------------------------------------------
// Main FSM
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        state       <= S_IDLE;
        busy        <= 0;
        done        <= 0;
        irq         <= 0;
        pe_en       <= 0;
        pe_flush    <= 0;
        pe_stat     <= 0;
        dma_w_start <= 0;
        dma_a_start <= 0;
        dma_r_start <= 0;
        dma_w_addr  <= 0;
        dma_a_addr  <= 0;
        dma_r_addr  <= 0;
        dma_w_len   <= 0;
        dma_a_len   <= 0;
        dma_r_len   <= 0;
        dma_w_done_r <= 0;
        dma_a_done_r <= 0;
        dma_r_done_r <= 0;
        w_ppb_swap  <= 0;
        a_ppb_swap  <= 0;
        w_ppb_clear <= 0;
        a_ppb_clear <= 0;
        r_fifo_clear <= 0;
        pe_load_w   <= 0;
        ws_consume_cnt <= 0;
        tile_i      <= 0;
        tile_j      <= 0;
    end else begin
        // Default: deassert pulse signals
        dma_w_start <= 0;
        dma_a_start <= 0;
        dma_r_start <= 0;
        w_ppb_swap  <= 0;
        a_ppb_swap  <= 0;
        w_ppb_clear <= 0;
        a_ppb_clear <= 0;
        r_fifo_clear <= 0;
        // done is cleared when CPU writes ctrl_reg=0
        if (!cfg_start) done <= 0;

        case (state)
            // ---------------------------------------------------------------
            S_IDLE: begin
                pe_en <= 0;
                pe_flush <= 0;
                pe_load_w <= 0;
                dma_w_done_r <= 0;
                dma_a_done_r <= 0;
                dma_r_done_r <= 0;

                if (cfg_start_rise) begin
                    busy    <= 1;
                    tile_i  <= 0;
                    tile_j  <= 0;
                    state   <= S_TILE_LOAD;
                    // Clear PPBufs and Result FIFO for first tile
                    w_ppb_clear  <= 1;
                    a_ppb_clear  <= 1;
                    r_fifo_clear <= 1;
                    // Launch DMA for tile (0,0)
                    dma_w_addr  <= w_addr;                    // B[:,0] start
                    dma_w_len   <= k_dim[15:0] * {14'b0, data_bytes};
                    dma_a_addr  <= a_addr;                    // A[0,:] start
                    dma_a_len   <= k_dim[15:0] * {14'b0, data_bytes};
                    dma_w_start <= 1;
                    dma_a_start <= 1;
                end
            end

            // ---------------------------------------------------------------
            // S_TILE_LOAD: wait for DMA to finish loading tile (i,j)
            // ---------------------------------------------------------------
            S_TILE_LOAD: begin
                if (dma_all_load_done) begin
                    // Swap PPBufs: make DMA-filled bank available to PE
                    w_ppb_swap <= 1;
                    a_ppb_swap <= 1;
                    // pe_en remains 0 here; S_COMPUTE first cycle will set pe_en=1.
                    // This avoids a spurious Stage-0 sample with stale weight_reg
                    // while pe_mode has already switched to the new dtype (e.g. FP16).
                    pe_stat <= cfg_stat[0];
                    state   <= S_PRELOAD;
                end
                if (cfg_abort) begin
                    state <= S_IDLE;
                    busy  <= 0;
                    pe_en <= 0;
                end
            end

            // ---------------------------------------------------------------
            // S_PRELOAD: 1-cycle wait for PPBuf swap + WS setup
            // ---------------------------------------------------------------
            S_PRELOAD: begin
                pe_en    <= 0;  // defer pe_en until data ready
                pe_flush <= 0;
                pe_load_w <= (cfg_stat[0] == 1'b0) ? 1'b1 : 1'b0;
                ws_consume_cnt <= 0;
                state    <= S_COMPUTE;
            end

            // ---------------------------------------------------------------
            // S_COMPUTE
            // ---------------------------------------------------------------
            S_COMPUTE: begin
                pe_flush <= 0;

                if (cfg_abort) begin
                    state <= S_IDLE;
                    busy  <= 0;
                    pe_en <= 0;
                    pe_load_w <= 0;
                end else if (cfg_stat[0] == 1'b0) begin
                    // ----- WS mode -----
                    // Feed K beats of weight+activation, then drain via S_DRAIN
                    pe_load_w <= (ws_consume_cnt < k_dim[15:0]) ? 1'b1 : 1'b0;
                    if (ws_consume_cnt < k_dim[15:0] + 16'd2) begin
                        pe_en <= 1;
                        ws_consume_cnt <= ws_consume_cnt + 1;
                    end else begin
                        pe_en <= 0;
                        pe_load_w <= 0;
                        state <= S_DRAIN;   // WS also uses flush path (same as OS)
                    end
                end else begin
                    // ----- OS mode -----
                    pe_en <= 1;
                    pe_load_w <= 0;
                    if (dma_all_load_done && w_ppb_empty && a_ppb_empty) begin
                        state <= S_DRAIN;
                    end
                end
            end

            // ---------------------------------------------------------------
            S_DRAIN: begin
                pe_en    <= 1;
                pe_flush <= 1;
                state    <= S_DRAIN2;
            end

            S_DRAIN2: begin
                pe_en    <= 1;
                pe_flush <= 0;
                state    <= S_WRITE_BACK;
            end

            // ---------------------------------------------------------------
            // S_WRITE_BACK: initiate write-back for current tile result
            // ---------------------------------------------------------------
            S_WRITE_BACK: begin
                pe_en       <= 1;
                pe_flush    <= 0;
                dma_r_addr  <= cur_r_addr;
                dma_r_len   <= tile_r_len;
                dma_r_start <= 1;
                state       <= S_WB_WAIT;
            end

            // ---------------------------------------------------------------
            // S_WB_WAIT: wait for DMA write-back, then advance tile
            // ---------------------------------------------------------------
            S_WB_WAIT: begin
                pe_en <= 1;
                dma_r_start <= 1;  // keep arming until DMA picks up
                if (dma_r_done_r) begin
                    dma_r_start <= 0;
                    pe_en <= 0;

                    // Advance to next tile
                    // Tile order: j increments first (row i fixed), then i increments
                    if (tile_j + 1 < n_dim) begin
                        // Move to next column in same row
                        tile_j <= tile_j + 1;
                        state  <= S_TILE_LOAD;
                        // Clear done flags and buffers for next tile
                        dma_w_done_r <= 0;
                        dma_a_done_r <= 0;
                        dma_r_done_r <= 0;
                        w_ppb_clear  <= 1;
                        a_ppb_clear  <= 1;
                        r_fifo_clear <= 1;
                        // Launch DMA for next tile: B[:,j+1], A[i,:] (same row)
                        dma_w_addr  <= w_addr + (tile_j + 1) * {16'b0, k_dim[15:0] * {14'b0, data_bytes}};
                        dma_w_len   <= k_dim[15:0] * {14'b0, data_bytes};
                        dma_a_addr  <= a_addr + tile_i * {16'b0, k_dim[15:0] * {14'b0, data_bytes}};
                        dma_a_len   <= k_dim[15:0] * {14'b0, data_bytes};
                        dma_w_start <= 1;
                        dma_a_start <= 1;
                    end else if (tile_i + 1 < m_dim) begin
                        // Move to next row, reset column to 0
                        tile_i <= tile_i + 1;
                        tile_j <= 0;
                        state  <= S_TILE_LOAD;
                        dma_w_done_r <= 0;
                        dma_a_done_r <= 0;
                        dma_r_done_r <= 0;
                        w_ppb_clear  <= 1;
                        a_ppb_clear  <= 1;
                        r_fifo_clear <= 1;
                        // Launch DMA for tile (i+1, 0): B[:,0], A[i+1,:]
                        dma_w_addr  <= w_addr;
                        dma_w_len   <= k_dim[15:0] * {14'b0, data_bytes};
                        dma_a_addr  <= a_addr + (tile_i + 1) * {16'b0, k_dim[15:0] * {14'b0, data_bytes}};
                        dma_a_len   <= k_dim[15:0] * {14'b0, data_bytes};
                        dma_w_start <= 1;
                        dma_a_start <= 1;
                    end else begin
                        // All tiles done
                        state <= S_DONE;
                    end
                end
            end

            // ---------------------------------------------------------------
            S_DONE: begin
                pe_en  <= 0;
                busy   <= 0;
                done   <= 1;
                irq    <= 1;
                state  <= S_IDLE;
                ws_consume_cnt <= 0;
                tile_i <= 0;
                tile_j <= 0;
            end

            default: state <= S_IDLE;
        endcase
    end
end

endmodule
