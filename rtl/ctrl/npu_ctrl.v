// =============================================================================
// Module  : npu_ctrl
// Project : NPU_prj
// Desc    : NPU top-level controller FSM (Ping-Pong aware).
//
//           Orchestrates: config → load W+A (overlapped with PE compute)
//                         → compute+drain → write-back → done.
//
//           Key difference from original: DMA load and PE compute can overlap
//           thanks to Ping-Pong Buffers. The PE starts consuming data as soon
//           as the first PPBuf bank reaches the THRESHOLD level.
//
//   States:
//     S_IDLE         - Wait for CPU start command
//     S_LOAD         - DMA loading W+A into PPBufs (PE starts when ready)
//     S_COMPUTE      - PE computing, DMA may still be filling PPBufs
//     S_DRAIN        - DMA done, PE draining remaining PPBuf data
//     S_WRITE_BACK   - Write PE results back to DRAM
//     S_DONE         - Assert done + IRQ, return to IDLE
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
    // Ping-Pong Buffer status
    input  wire              w_ppb_ready,     // Weight PPBuf has data for PE
    input  wire              w_ppb_empty,     // Weight PPBuf reader's bank empty
    input  wire              a_ppb_ready,     // Activation PPBuf has data for PE
    input  wire              a_ppb_empty,     // Activation PPBuf reader's bank empty
    // Ping-Pong Buffer control
    output reg               w_ppb_swap,      // Swap weight PPBuf bank
    output reg               a_ppb_swap,      // Swap activation PPBuf bank
    output reg               w_ppb_clear,     // Clear weight PPBuf
    output reg               a_ppb_clear,     // Clear activation PPBuf
    // Interrupt
    output reg               irq
);

// ---------------------------------------------------------------------------
// FSM states
// ---------------------------------------------------------------------------
localparam S_IDLE       = 4'd0;
localparam S_LOAD       = 4'd1;   // DMA loading W+A into PPBufs
localparam S_PRELOAD    = 4'd2;   // Wait for swap data to reach PE inputs (1 cycle)
localparam S_COMPUTE    = 4'd3;   // PE computing
localparam S_DRAIN      = 4'd4;   // PE flush accumulators
localparam S_WRITE_BACK = 4'd5;   // PE enabled, DMA start write-back
localparam S_WB_WAIT    = 4'd6;   // Wait for DMA write-back to complete
localparam S_DONE       = 4'd7;
localparam S_DRAIN2     = 4'd9;   // Pipeline drain (flush propagating through stages)

reg [3:0] state;

wire cfg_start  = ctrl_reg[0];
wire cfg_abort  = ctrl_reg[1];
wire [1:0] cfg_mode   = ctrl_reg[3:2];   // 00=INT8, 01=INT16, 10=FP16
wire [1:0] cfg_stat   = ctrl_reg[5:4];   // 0=WS, 1=OS

// Edge detect for start (rising edge only, avoids re-trigger from stale start bit)
reg cfg_start_d1;
always @(posedge clk) begin
    if (!rst_n) cfg_start_d1 <= 0;
    else cfg_start_d1 <= cfg_start;
end
wire cfg_start_rise = cfg_start && !cfg_start_d1;

// ---------------------------------------------------------------------------
// Mode decode
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
// Data size calculation (bytes per element)
// ---------------------------------------------------------------------------
wire [1:0] data_bytes = (cfg_mode == 2'b00) ? 2'd1 :   // INT8: 1 byte
                         2'd2;                          // INT16/FP16: 2 bytes

// ---------------------------------------------------------------------------
// Track DMA completion (latched)
// ---------------------------------------------------------------------------
reg dma_w_done_r, dma_a_done_r, dma_r_done_r;

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

wire dma_all_load_done = dma_w_done_r && dma_a_done_r;

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
    end else begin
        // Default: deassert pulse signals
        dma_w_start <= 0;
        dma_a_start <= 0;
        dma_r_start <= 0;
        w_ppb_swap  <= 0;
        a_ppb_swap  <= 0;
        w_ppb_clear <= 0;
        a_ppb_clear <= 0;
        // done is cleared when CPU writes ctrl_reg=0 (start bit cleared)
        if (!cfg_start) done <= 0;

        case (state)
            S_IDLE: begin
                pe_en <= 0;
                pe_flush <= 0;
                dma_w_done_r <= 0;
                dma_a_done_r <= 0;
                dma_r_done_r <= 0;

                if (cfg_start_rise) begin
                    busy <= 1;
                    state <= S_LOAD;
                    // Clear PPBufs before DMA starts filling
                    w_ppb_clear <= 1;
                    a_ppb_clear <= 1;
                    // Launch DMA for both W and A simultaneously
                    dma_w_addr  <= w_addr;
                    dma_w_len   <= n_dim * k_dim * data_bytes;
                    dma_a_addr  <= a_addr;
                    dma_a_len   <= m_dim * k_dim * data_bytes;
                    dma_w_start <= 1;
                    dma_a_start <= 1;
                end
            end

            S_LOAD: begin
                // Wait for BOTH DMA channels to complete, then swap + preload
                if (dma_w_done_r && dma_a_done_r) begin
                    // Swap PPBufs: transfer DMA-filled data to PE reader side
                    w_ppb_swap <= 1;
                    a_ppb_swap <= 1;
                    pe_en   <= 1;
                    pe_stat <= cfg_stat[0];
                    state   <= S_PRELOAD;
                end
                // Abort handling
                if (cfg_abort) begin
                    state <= S_IDLE;
                    busy  <= 0;
                    pe_en <= 0;
                end
            end

            S_PRELOAD: begin
                // Wait 1 cycle for PPBuf swap data to propagate to PE inputs
                pe_en    <= 1;
                pe_flush <= 0;
                state    <= S_COMPUTE;
            end

            S_COMPUTE: begin
                pe_en <= 1;
                pe_flush <= 0;

                if (cfg_abort) begin
                    state <= S_IDLE;
                    busy  <= 0;
                    pe_en <= 0;
                end
                // If both DMA channels are done AND both PPBufs are drained
                else if (dma_all_load_done && w_ppb_empty && a_ppb_empty) begin
                    // All data loaded and consumed, flush PE pipeline
                    state    <= S_DRAIN;
                end
            end

            S_DRAIN: begin
                pe_en    <= 1;
                pe_flush <= 1;  // flush accumulators
                state    <= S_DRAIN2;
            end

            S_DRAIN2: begin
                pe_en    <= 1;
                pe_flush <= 0;  // flush propagated to stage-0
                // Wait 2 more cycles for 3-stage pipeline to produce valid output
                state    <= S_WRITE_BACK;
            end

            S_WRITE_BACK: begin
                pe_en <= 1;  // Keep PE enabled for flush result to write into FIFO
                pe_flush <= 0;
                dma_r_addr <= r_addr;
                dma_r_len  <= m_dim * n_dim * (ACC_W/8);  // result bytes
                dma_r_start <= 1;
                state <= S_WB_WAIT;
            end

            S_WB_WAIT: begin
                pe_en <= 1;  // Keep PE enabled until DMA reads FIFO
                // Keep sending r_start until DMA picks it up (FIFO may be empty briefly)
                dma_r_start <= 1;
                // Wait for DMA result write-back to complete
                if (dma_r_done_r) begin
                    dma_r_start <= 0;
                    pe_en <= 0;
                    state <= S_DONE;
                end
            end

            S_DONE: begin
                pe_en  <= 0;
                busy   <= 0;
                done   <= 1;   // sticky: stays 1 until CPU clears ctrl_reg
                irq    <= 1;
                state  <= S_IDLE;
                // Note: done remains 1 until CPU writes 0 to ctrl_reg[0]
            end

            default: state <= S_IDLE;
        endcase
    end
end

endmodule
