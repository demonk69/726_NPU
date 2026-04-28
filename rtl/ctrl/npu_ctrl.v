// =============================================================================
// Module  : npu_ctrl
// Project : NPU_prj
// Desc    : NPU top-level controller FSM — True Ping-Pong Overlap Edition
//
// ── Architecture ──────────────────────────────────────────────────────────
//
//   This controller implements a TRUE PIPELINE between DMA Load and PE Compute,
//   eliminating the serial "Load → Compute → WB" stall.
//
//   Tile execution pipeline (steady-state):
//
//     Cycle:  T0           T1           T2           T3
//             LOAD(0,0)    COMPUTE(0,0) COMPUTE(0,0) ...
//                          LOAD(0,1)    LOAD(0,1)    COMPUTE(0,1)
//                                                    LOAD(0,2)
//
//   Three execution phases:
//     Phase 0 – Warm-up  : Load tile(0,0) only. PE idle.
//     Phase 1 – Overlap  : PE computes tile(i,j) from Ping bank WHILE
//                          DMA loads tile(next_i, next_j) into Pong bank.
//     Phase 2 – Drain    : Final tile: compute + WB, no new load.
//
// ── FSM States ────────────────────────────────────────────────────────────
//
//   S_IDLE            - Wait for CPU start; latch all config registers.
//   S_WARMUP_LOAD     - Phase 0: wait for tile(0,0) DMA load to finish.
//   S_WARMUP_WAIT     - 1-cycle PPBuf swap propagation; launch prefetch.
//   S_OVERLAP_COMPUTE - Compute tile(i,j) while DMA prefetches next tile.
//   S_DRAIN           - Assert pe_flush 1 cycle.
//   S_DRAIN2          - Drain pipeline 2nd cycle.
//   S_WRITE_BACK      - Initiate DMA write-back.
//   S_WB_WAIT         - Wait DMA WB done; swap banks; launch next prefetch.
//   S_DONE            - Assert IRQ, reset counters, enter S_IDLE.
//
// ── IRQ Clear ─────────────────────────────────────────────────────────────
//
//   CPU clears IRQ by writing 1 to ctrl_reg[2] (IRQ_CLR bit).
//   npu_ctrl samples and immediately de-asserts irq. Bit is self-clearing.
//
// ── Address Latch ─────────────────────────────────────────────────────────
//
//   All config registers (m/n/k_dim, w/a/r_addr, mode, stat) are latched
//   on cfg_start_rise into shadow regs. FSM uses shadows only.
//
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
    // Config from AXI-Lite register file (live, may change during run)
    input  wire [31:0]       ctrl_reg,
    input  wire [31:0]       m_dim,
    input  wire [31:0]       n_dim,
    input  wire [31:0]       k_dim,
    input  wire [31:0]       w_addr,
    input  wire [31:0]       a_addr,
    input  wire [31:0]       r_addr,
    input  wire [7:0]        arr_cfg,
    // Shape configuration for reconfigurable array
    input  wire [1:0]        cfg_shape_in,
    output wire [1:0]        cfg_shape_latched,
    // 4x4 tile planner outputs (T2.3)
    output wire              tile_mode,
    output wire              vec_consume,
    output wire [31:0]       tile_m_base,
    output wire [31:0]       tile_n_base,
    output wire [3:0]        tile_row_valid,
    output wire [3:0]        tile_col_valid,
    output wire [2:0]        tile_active_rows,
    output wire [2:0]        tile_active_cols,
    output reg  [15:0]       tile_k_cycle,
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
    output reg               pe_stat,     // 0=WS,  1=OS
    output reg               pe_load_w,   // WS mode: latch weight into PE
    output reg               pe_swap_w,   // WS mode: swap dual weight regs
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
// FSM state encoding
// ---------------------------------------------------------------------------
localparam S_IDLE            = 4'd0;
localparam S_WARMUP_LOAD     = 4'd1;
localparam S_WARMUP_WAIT     = 4'd2;
localparam S_PRELOAD         = 4'd9;   // 1-cycle swap propagation before compute
localparam S_OVERLAP_COMPUTE = 4'd3;
localparam S_DRAIN           = 4'd4;
localparam S_DRAIN2          = 4'd5;
localparam S_WRITE_BACK      = 4'd6;
localparam S_WB_WAIT         = 4'd7;
localparam S_WAIT_PREFETCH   = 4'd10;  // Wait for prefetch DMA before swap+compute
localparam S_DONE            = 4'd8;

reg [3:0] state;

// ---------------------------------------------------------------------------
// ctrl_reg bit decode  (live, from AXI-Lite)
//   Address 0x00 CTRL register bit layout:
//     bit[0]   = start
//     bit[1]   = abort
//     bit[3:2] = data_mode (00=INT8, 10=FP16)
//     bit[5:4] = stat_mode (00=WS,   01=OS)
//     bit[6]   = irq_clr   (CPU writes 1 to acknowledge/clear IRQ; self-clearing)
// ---------------------------------------------------------------------------
wire        cfg_start   = ctrl_reg[0];
wire        cfg_abort   = ctrl_reg[1];
wire [1:0]  cfg_mode    = ctrl_reg[3:2]; // 00=INT8  10=FP16
wire [1:0]  cfg_stat    = ctrl_reg[5:4]; // 00=WS    01=OS
wire        cfg_irq_clr = ctrl_reg[6];   // CPU writes 1 → clear IRQ
wire [1:0]  cfg_data_bytes = (cfg_mode == 2'b00) ? 2'd1 : 2'd2;
wire [15:0] cfg_scalar_elem_bytes = {14'b0, cfg_data_bytes};
wire [15:0] cfg_vector_elem_bytes = cfg_scalar_elem_bytes << 2;
wire [15:0] cfg_start_tile_len = k_dim[15:0] *
                                  (arr_cfg[7] ? cfg_vector_elem_bytes
                                              : cfg_scalar_elem_bytes);

// Rising-edge detect for start
reg cfg_start_d1;
always @(posedge clk) begin
    if (!rst_n) cfg_start_d1 <= 1'b0;
    else        cfg_start_d1 <= cfg_start;
end
wire cfg_start_rise = cfg_start && !cfg_start_d1;

// ---------------------------------------------------------------------------
// Shadow (latched) configuration registers
// Latched once on cfg_start_rise; FSM exclusively uses these.
// ---------------------------------------------------------------------------
reg [31:0] lk_m_dim, lk_n_dim, lk_k_dim;
reg [31:0] lk_w_addr, lk_a_addr, lk_r_addr;
reg [1:0]  lk_mode, lk_stat;
reg [1:0]  lk_shape;   // latched cfg_shape
reg [7:0]  lk_arr_cfg;

assign cfg_shape_latched = lk_shape;
assign tile_mode = lk_arr_cfg[7];

always @(posedge clk) begin
    if (!rst_n) begin
        lk_m_dim  <= 32'd1; lk_n_dim  <= 32'd1; lk_k_dim  <= 32'd1;
        lk_w_addr <= 32'd0; lk_a_addr <= 32'd0; lk_r_addr <= 32'd0;
        lk_mode   <= 2'b10;                      // default FP16
        lk_stat   <= 2'b01;                      // default OS
        lk_shape  <= 2'b10;                      // default 16x16
        lk_arr_cfg <= 8'd0;
    end else if (cfg_start_rise) begin
        lk_m_dim  <= m_dim;
        lk_n_dim  <= n_dim;
        lk_k_dim  <= k_dim;
        lk_w_addr <= w_addr;
        lk_a_addr <= a_addr;
        lk_r_addr <= r_addr;
        lk_mode   <= cfg_mode;
        lk_stat   <= cfg_stat;
        lk_shape  <= cfg_shape_in;
        lk_arr_cfg <= arr_cfg;
    end
end

// ---------------------------------------------------------------------------
// Mode decode (combinational, uses shadow regs)
// ---------------------------------------------------------------------------
always @(*) begin
    case (lk_mode)
        2'b00:   pe_mode = 1'b0;   // INT8
        2'b10:   pe_mode = 1'b1;   // FP16
        default: pe_mode = 1'b1;
    endcase
end

// Bytes per element (from shadow)
wire [1:0] data_bytes = (lk_mode == 2'b00) ? 2'd1 : 2'd2;

// ---------------------------------------------------------------------------
// Tile-loop counters:
//   scalar mode: tile_i/tile_j are i/j for one C[i,j]
//   tile mode:   tile_i/tile_j are m_tile/n_tile for one 4x4 C tile
// ---------------------------------------------------------------------------
reg [31:0] tile_i;
reg [31:0] tile_j;
reg [2:0]  wb_row;

localparam [31:0] TILE_LANES = 32'd4;
wire [31:0] tile_m_tiles = tile_mode ? ((lk_m_dim + 32'd3) >> 2) : lk_m_dim;
wire [31:0] tile_n_tiles = tile_mode ? ((lk_n_dim + 32'd3) >> 2) : lk_n_dim;
wire [31:0] tile_iter_m_count = (tile_m_tiles == 32'd0) ? 32'd1 : tile_m_tiles;
wire [31:0] tile_iter_n_count = (tile_n_tiles == 32'd0) ? 32'd1 : tile_n_tiles;

assign tile_m_base = tile_mode ? (tile_i << 2) : tile_i;
assign tile_n_base = tile_mode ? (tile_j << 2) : tile_j;

wire [31:0] tile_row_rem = (lk_m_dim > tile_m_base) ? (lk_m_dim - tile_m_base) : 32'd0;
wire [31:0] tile_col_rem = (lk_n_dim > tile_n_base) ? (lk_n_dim - tile_n_base) : 32'd0;

assign tile_active_rows = !tile_mode ? 3'd1 :
                          (tile_row_rem >= TILE_LANES) ? 3'd4 :
                          tile_row_rem[2:0];
assign tile_active_cols = !tile_mode ? 3'd1 :
                          (tile_col_rem >= TILE_LANES) ? 3'd4 :
                          tile_col_rem[2:0];

assign tile_row_valid[0] = (tile_active_rows > 3'd0);
assign tile_row_valid[1] = (tile_active_rows > 3'd1);
assign tile_row_valid[2] = (tile_active_rows > 3'd2);
assign tile_row_valid[3] = (tile_active_rows > 3'd3);
assign tile_col_valid[0] = (tile_active_cols > 3'd0);
assign tile_col_valid[1] = (tile_active_cols > 3'd1);
assign tile_col_valid[2] = (tile_active_cols > 3'd2);
assign tile_col_valid[3] = (tile_active_cols > 3'd3);

// One-tile DMA byte length
wire [15:0] scalar_elem_bytes = {14'b0, data_bytes};
wire [15:0] vector_elem_bytes = scalar_elem_bytes << 2;
wire [15:0] tile_len = lk_k_dim[15:0] *
                       (tile_mode ? vector_elem_bytes : scalar_elem_bytes);

// Current-tile addresses (used for write-back address calc)
wire [31:0] comp_r_addr = lk_r_addr +
                          (tile_m_base * lk_n_dim + tile_n_base) * (ACC_W/8);
wire [31:0] comp_row_r_addr = lk_r_addr +
                              (((tile_m_base + {29'd0, wb_row}) * lk_n_dim) +
                               tile_n_base) * (ACC_W/8);
wire [15:0] tile_row_r_len = {13'd0, tile_active_cols} << 2;

// ── Last-tile flag ──
wire is_last_tile = (tile_i == tile_iter_m_count - 1) &&
                    (tile_j == tile_iter_n_count - 1);

// Result DMA: scalar mode writes one word; tile mode writes one row burst.
localparam [15:0] TILE_R_LEN = 16'd4;

wire [15:0] tile_compute_cycles = lk_k_dim[15:0] +
                                  {13'd0, tile_active_rows} - 16'd1;

assign vec_consume = tile_mode &&
                     pe_en &&
                     (state == S_OVERLAP_COMPUTE) &&
                     (lk_stat[0] == 1'b1) &&
                     (tile_k_cycle < lk_k_dim[15:0]);

// ---------------------------------------------------------------------------
// DMA completion latches
// ---------------------------------------------------------------------------
reg dma_w_done_r, dma_a_done_r, dma_r_done_r;
wire dma_load_done = dma_w_done_r && dma_a_done_r;

always @(posedge clk) begin
    if (!rst_n) begin
        dma_w_done_r <= 1'b0;
        dma_a_done_r <= 1'b0;
        dma_r_done_r <= 1'b0;
    end else begin
        if (dma_w_done) dma_w_done_r <= 1'b1;
        if (dma_a_done) dma_a_done_r <= 1'b1;
        if (dma_r_done) dma_r_done_r <= 1'b1;
    end
end

// ---------------------------------------------------------------------------
// WS consume counter
// ---------------------------------------------------------------------------
reg [15:0] ws_consume_cnt;

// ---------------------------------------------------------------------------
// Combinational: next-tile coordinates (after advancing current tile)
// These are the coordinates of the tile whose prefetch we need to issue.
// ---------------------------------------------------------------------------
wire [31:0] next_j_after_cur;   // j coord of tile AFTER current
wire [31:0] next_i_after_cur;   // i coord of tile AFTER current
assign next_j_after_cur = (tile_j + 1 < tile_iter_n_count) ? (tile_j + 1) : 32'd0;
assign next_i_after_cur = (tile_j + 1 < tile_iter_n_count) ? tile_i       : (tile_i + 1);

// "The tile after the next tile" — this is what we prefetch when sitting
// in S_WARMUP_WAIT (we've swapped tile(0,0) to Ping; we want to load tile(0,1))
// Since we're currently computing tile(0,0), next = tile(0,1):
//   pf_j = (0+1 < N) ? 1 : 0  etc.  →  already covered by next_j_after_cur/next_i_after_cur
//   at state entry tile_i=0, tile_j=0.

// Prefetch DMA addresses (next tile)
wire [31:0] pfetch_w_addr = lk_w_addr + next_j_after_cur * {16'b0, tile_len};
wire [31:0] pfetch_a_addr = lk_a_addr + next_i_after_cur * {16'b0, tile_len};

// Is the next tile the last tile?
wire next_is_last = (next_i_after_cur == tile_iter_m_count - 1) &&
                    (next_j_after_cur == tile_iter_n_count - 1);

// ---------------------------------------------------------------------------
// Main FSM
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        state          <= S_IDLE;
        busy           <= 1'b0;
        done           <= 1'b0;
        irq            <= 1'b0;
        pe_en          <= 1'b0;
        pe_flush       <= 1'b0;
        pe_stat        <= 1'b1;
        pe_load_w      <= 1'b0;
        pe_swap_w      <= 1'b0;
        dma_w_start    <= 1'b0;
        dma_a_start    <= 1'b0;
        dma_r_start    <= 1'b0;
        dma_w_addr     <= 32'd0;
        dma_a_addr     <= 32'd0;
        dma_r_addr     <= 32'd0;
        dma_w_len      <= 16'd0;
        dma_a_len      <= 16'd0;
        dma_r_len      <= 16'd0;
        dma_w_done_r   <= 1'b0;
        dma_a_done_r   <= 1'b0;
        dma_r_done_r   <= 1'b0;
        w_ppb_swap     <= 1'b0;
        a_ppb_swap     <= 1'b0;
        w_ppb_clear    <= 1'b0;
        a_ppb_clear    <= 1'b0;
        r_fifo_clear   <= 1'b0;
        ws_consume_cnt <= 16'd0;
        tile_k_cycle   <= 16'd0;
        tile_i         <= 32'd0;
        tile_j         <= 32'd0;
        wb_row         <= 3'd0;
    end else begin
        // ── Default: de-assert all one-cycle pulse signals ──
        dma_w_start  <= 1'b0;
        dma_a_start  <= 1'b0;
        dma_r_start  <= 1'b0;
        w_ppb_swap   <= 1'b0;
        a_ppb_swap   <= 1'b0;
        w_ppb_clear  <= 1'b0;
        a_ppb_clear  <= 1'b0;
        r_fifo_clear <= 1'b0;
        pe_swap_w    <= 1'b0;

        // ── IRQ Clear: CPU writes ctrl_reg[2] = 1 to acknowledge IRQ ──
        if (cfg_irq_clr) irq <= 1'b0;

        // ── done auto-clear when CPU de-asserts start ──
        if (!cfg_start) done <= 1'b0;

        case (state)

            // =================================================================
            // S_IDLE
            // =================================================================
            S_IDLE: begin
                pe_en    <= 1'b0;
                pe_flush <= 1'b0;
                pe_load_w <= 1'b0;
                dma_w_done_r <= 1'b0;
                dma_a_done_r <= 1'b0;
                dma_r_done_r <= 1'b0;

                if (cfg_start_rise) begin
                    busy           <= 1'b1;
                    tile_i         <= 32'd0;
                    tile_j         <= 32'd0;
                    ws_consume_cnt <= 16'd0;
                    tile_k_cycle   <= 16'd0;
                    wb_row         <= 3'd0;

                    // Clear all buffers for fresh layer start
                    w_ppb_clear  <= 1'b1;
                    a_ppb_clear  <= 1'b1;
                    r_fifo_clear <= 1'b1;

                    // Phase-0: launch warm-up load of tile(0,0)
                    // Use live cfg (shadow latched this same cycle)
                    dma_w_addr  <= w_addr;
                    dma_w_len   <= cfg_start_tile_len;
                    dma_a_addr  <= a_addr;
                    dma_a_len   <= cfg_start_tile_len;
                    dma_w_start <= 1'b1;
                    dma_a_start <= 1'b1;

                    state <= S_WARMUP_LOAD;
                end
            end

            // =================================================================
            // S_WARMUP_LOAD – wait for tile(0,0) DMA load done.
            // =================================================================
            S_WARMUP_LOAD: begin
                pe_en    <= 1'b0;
                pe_flush <= 1'b0;

                if (cfg_abort) begin
                    state <= S_IDLE; busy <= 1'b0;
                end else if (dma_load_done) begin
                    // Swap PPBuf Pong→Ping so PE can read tile(0,0)
                    w_ppb_swap   <= 1'b1;
                    a_ppb_swap   <= 1'b1;
                    dma_w_done_r <= 1'b0;
                    dma_a_done_r <= 1'b0;
                    state        <= S_WARMUP_WAIT;
                end
            end

            // =================================================================
            // S_WARMUP_WAIT – 1-cycle swap propagation.
            // Launch prefetch for tile(0,1) unless tile(0,0) is the last tile.
            // =================================================================
            S_WARMUP_WAIT: begin
                pe_stat        <= lk_stat[0];
                pe_load_w      <= (lk_stat[0] == 1'b0) ? 1'b1 : 1'b0;
                ws_consume_cnt <= 16'd0;
                tile_k_cycle   <= 16'd0;

                if (is_last_tile) begin
                    // Only one tile in the whole layer; no prefetch needed.
                    state <= S_PRELOAD;
                end else begin
                    // Prefetch next tile into Pong bank
                    dma_w_addr  <= pfetch_w_addr;
                    dma_w_len   <= tile_len;
                    dma_a_addr  <= pfetch_a_addr;
                    dma_a_len   <= tile_len;
                    dma_w_start <= 1'b1;
                    dma_a_start <= 1'b1;
                    dma_w_done_r <= 1'b0;
                    dma_a_done_r <= 1'b0;
                    state <= S_PRELOAD;
                end
            end

            // =================================================================
            // S_PRELOAD – 1-cycle wait for PPBuf swap to propagate.
            // After this cycle, rd_sel/wr_sel/rd_fill are stable for PE.
            // =================================================================
            S_PRELOAD: begin
                pe_en    <= 1'b0;
                pe_flush <= 1'b0;
                // pe_stat and pe_load_w are already set by previous state
                state <= S_OVERLAP_COMPUTE;
            end

            // =================================================================
            // S_OVERLAP_COMPUTE
            //
            //   Ping bank   → PE is computing tile(tile_i, tile_j)
            //   Pong bank   → DMA is loading next tile (already launched)
            //
            //   Exit when all Ping data consumed (OS) or K+2 beats (WS).
            //   Then go to S_DRAIN.
            // =================================================================
            S_OVERLAP_COMPUTE: begin
                pe_flush <= 1'b0;

                if (cfg_abort) begin
                    state <= S_IDLE; busy <= 1'b0;
                    pe_en <= 1'b0; pe_load_w <= 1'b0;

                end else if (lk_stat[0] == 1'b0) begin
                    // ─── WS mode ───
                    pe_load_w <= (ws_consume_cnt < lk_k_dim[15:0]) ? 1'b1 : 1'b0;
                    if (ws_consume_cnt < lk_k_dim[15:0] + 16'd2) begin
                        pe_en <= 1'b1;
                        ws_consume_cnt <= ws_consume_cnt + 1;
                    end else begin
                        pe_en     <= 1'b0;
                        pe_load_w <= 1'b0;
                        state     <= S_DRAIN;
                    end

                end else begin
                    // ─── OS mode ───
                    pe_load_w <= 1'b0;
                    if (tile_mode) begin
                        pe_en <= 1'b1;
                        if (!pe_en) begin
                            tile_k_cycle <= 16'd0;
                        end else if (tile_k_cycle + 16'd1 >= tile_compute_cycles) begin
                            pe_en        <= 1'b0;
                            tile_k_cycle <= 16'd0;
                            state        <= S_DRAIN;
                        end else begin
                            tile_k_cycle <= tile_k_cycle + 16'd1;
                        end
                    end else begin
                        pe_en <= 1'b1;
                        if (w_ppb_empty && a_ppb_empty) begin
                            state <= S_DRAIN;
                        end
                    end
                end
            end

            // =================================================================
            // S_DRAIN – Assert pe_flush for 1 cycle.
            // =================================================================
            S_DRAIN: begin
                pe_en    <= 1'b1;
                pe_flush <= 1'b1;
                state    <= S_DRAIN2;
            end

            // =================================================================
            // S_DRAIN2 – Flush pipeline 2nd cycle.
            // =================================================================
            S_DRAIN2: begin
                pe_en    <= 1'b1;
                pe_flush <= 1'b0;
                state    <= S_WRITE_BACK;
            end

            // =================================================================
            // S_WRITE_BACK – Initiate DMA result write-back.
            // =================================================================
            S_WRITE_BACK: begin
                pe_en        <= 1'b1;
                pe_flush     <= 1'b0;
                dma_r_addr   <= tile_mode ? comp_row_r_addr : comp_r_addr;
                dma_r_len    <= tile_mode ? tile_row_r_len  : TILE_R_LEN;
                dma_r_start  <= 1'b1;
                dma_r_done_r <= 1'b0;
                state        <= S_WB_WAIT;
            end

            // =================================================================
            // S_WB_WAIT – Wait DMA WB done; then check prefetch status.
            //
            //   After WB completes:
            //     - If IS last tile → go to S_DONE.
            //     - If NOT last tile:
            //         If prefetch DMA ALREADY done → swap + S_PRELOAD → compute.
            //         Else → go to S_WAIT_PREFETCH to wait for it.
            // =================================================================
            S_WB_WAIT: begin
                pe_en <= 1'b1;
                dma_r_start <= 1'b0;
                if (dma_r_done_r) begin
                    dma_r_start  <= 1'b0;
                    dma_r_done_r <= 1'b0;
                    pe_en        <= 1'b0;

                    if (tile_mode && (wb_row + 3'd1 < tile_active_rows)) begin
                        wb_row <= wb_row + 3'd1;
                        state  <= S_WRITE_BACK;
                    end else if (is_last_tile) begin
                        wb_row <= 3'd0;
                        state <= S_DONE;
                    end else begin
                        wb_row <= 3'd0;
                        // Compute next tile coordinates
                        if (tile_j + 1 < tile_iter_n_count) begin
                            tile_j <= tile_j + 1;
                        end else begin
                            tile_i <= tile_i + 1;
                            tile_j <= 32'd0;
                        end

                        // Decide what to do based on prefetch status
                        if (dma_load_done) begin
                            // Prefetch already finished — swap banks
                            w_ppb_swap   <= 1'b1;
                            a_ppb_swap   <= 1'b1;
                            r_fifo_clear <= 1'b1;
                            pe_stat        <= lk_stat[0];
                            pe_load_w      <= (lk_stat[0] == 1'b0) ? 1'b1 : 1'b0;
                            ws_consume_cnt <= 16'd0;
                            tile_k_cycle   <= 16'd0;
                            dma_w_done_r   <= 1'b0;
                            dma_a_done_r   <= 1'b0;
                            // Launch prefetch for the tile two steps ahead
                            if (!next_is_last) begin
                                dma_w_addr <= lk_w_addr +
                                    ( (next_j_after_cur + 1 < tile_iter_n_count) ? (next_j_after_cur + 1) : 32'd0 )
                                    * {16'b0, tile_len};
                                dma_a_addr <= lk_a_addr +
                                    ( (next_j_after_cur + 1 < tile_iter_n_count) ? next_i_after_cur : (next_i_after_cur + 1) )
                                    * {16'b0, tile_len};
                                dma_w_len  <= tile_len;
                                dma_a_len  <= tile_len;
                                dma_w_start <= 1'b1;
                                dma_a_start <= 1'b1;
                            end
                            state <= S_PRELOAD;  // 1-cycle swap propagation
                        end else begin
                            // Prefetch not yet done — wait in S_WAIT_PREFETCH
                            state <= S_WAIT_PREFETCH;
                        end
                    end
                end
            end

            // =================================================================
            // S_WAIT_PREFETCH – Wait for the in-flight prefetch DMA to finish,
            // then swap banks, wait S_PRELOAD, and enter compute.
            // =================================================================
            S_WAIT_PREFETCH: begin
                pe_en <= 1'b0;
                if (cfg_abort) begin
                    state <= S_IDLE; busy <= 1'b0;
                end else if (dma_load_done) begin
                    // Prefetch finished — swap and start compute
                    w_ppb_swap   <= 1'b1;
                    a_ppb_swap   <= 1'b1;
                    r_fifo_clear <= 1'b1;
                    pe_stat        <= lk_stat[0];
                    pe_load_w      <= (lk_stat[0] == 1'b0) ? 1'b1 : 1'b0;
                    ws_consume_cnt <= 16'd0;
                    tile_k_cycle   <= 16'd0;
                    dma_w_done_r   <= 1'b0;
                    dma_a_done_r   <= 1'b0;
                    // Launch prefetch for tile two steps ahead (from current new tile)
                    if (!next_is_last) begin
                        dma_w_addr <= lk_w_addr +
                            ( (next_j_after_cur + 1 < tile_iter_n_count) ? (next_j_after_cur + 1) : 32'd0 )
                            * {16'b0, tile_len};
                        dma_a_addr <= lk_a_addr +
                            ( (next_j_after_cur + 1 < tile_iter_n_count) ? next_i_after_cur : (next_i_after_cur + 1) )
                            * {16'b0, tile_len};
                        dma_w_len  <= tile_len;
                        dma_a_len  <= tile_len;
                        dma_w_start <= 1'b1;
                        dma_a_start <= 1'b1;
                    end
                    state <= S_PRELOAD;  // 1-cycle swap propagation
                end
            end

            // =================================================================
            // S_DONE – Layer complete. Assert IRQ. Return to S_IDLE.
            // =================================================================
            S_DONE: begin
                pe_en  <= 1'b0;
                busy   <= 1'b0;
                done   <= 1'b1;
                irq    <= 1'b1;
                // Reset counters; clear buffers for next layer
                ws_consume_cnt <= 16'd0;
                tile_k_cycle   <= 16'd0;
                tile_i         <= 32'd0;
                tile_j         <= 32'd0;
                wb_row         <= 3'd0;
                w_ppb_clear    <= 1'b1;
                a_ppb_clear    <= 1'b1;
                r_fifo_clear   <= 1'b1;
                state          <= S_IDLE;
            end

            default: state <= S_IDLE;

        endcase
    end
end

endmodule
