// =============================================================================
// Module  : npu_ctrl
// Project : NPU_prj
// Desc    : NPU top-level controller FSM.
//           Orchestrates: config → weight load → compute → drain → done.
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
    output reg               pe_mode,     // 0=INT8, 1=INT16, 10=FP16
    output reg               pe_stat,
    // Interrupt
    output reg               irq
);

// ---------------------------------------------------------------------------
// FSM states
// ---------------------------------------------------------------------------
localparam S_IDLE       = 4'd0;
localparam S_LOAD_W     = 4'd1;
localparam S_LOAD_A     = 4'd2;
localparam S_COMPUTE    = 4'd3;
localparam S_DRAIN      = 4'd4;
localparam S_WRITE_BACK = 4'd5;
localparam S_DONE       = 4'd6;

reg [3:0] state;

wire cfg_start  = ctrl_reg[0];
wire cfg_abort  = ctrl_reg[1];
wire [1:0] cfg_mode   = ctrl_reg[3:2];   // 00=INT8, 01=INT16, 10=FP16
wire [1:0] cfg_stat   = ctrl_reg[5:4];   // 0=WS, 1=OS

// ---------------------------------------------------------------------------
// Mode decode (2-bit to 1-bit for PE)
// ---------------------------------------------------------------------------
always @(*) begin
    case (cfg_mode)
        2'b00: pe_mode = 1'b0;   // INT8
        2'b01: pe_mode = 1'b1;   // INT16 (treated as FP16 for now, expand later)
        2'b10: pe_mode = 1'b1;   // FP16
        default: pe_mode = 1'b0;
    endcase
end

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
    end else begin
        // Default: deassert pulse signals
        dma_w_start <= 0;
        dma_a_start <= 0;
        dma_r_start <= 0;
        done        <= 0;

        case (state)
            S_IDLE: begin
                pe_en <= 0;
                pe_flush <= 0;
                if (cfg_start) begin
                    busy <= 1;
                    state <= S_LOAD_W;
                    // Calculate DMA transfer sizes
                    dma_w_addr <= w_addr;
                    dma_w_len  <= n_dim * k_dim * (DATA_W/8);   // weight bytes
                    dma_w_start <= 1;
                end
            end

            S_LOAD_W: begin
                if (dma_w_done) begin
                    state <= S_LOAD_A;
                    dma_a_addr <= a_addr;
                    dma_a_len  <= m_dim * k_dim * (DATA_W/8);   // activation bytes
                    dma_a_start <= 1;
                end
            end

            S_LOAD_A: begin
                if (dma_a_done) begin
                    state <= S_COMPUTE;
                    pe_en <= 1;
                    pe_flush <= 0;
                    pe_stat <= cfg_stat[0];
                end
            end

            S_COMPUTE: begin
                // Compute for M + N + K - 1 cycles (systolic fill + drain)
                // Simplified: fixed wait
                pe_en <= 1;
                pe_flush <= 0;
                if (cfg_abort) begin
                    state <= S_IDLE;
                    busy <= 0;
                    pe_en <= 0;
                end else begin
                    state <= S_DRAIN;
                end
            end

            S_DRAIN: begin
                pe_en <= 1;
                pe_flush <= 1;  // flush accumulator
                state <= S_WRITE_BACK;
            end

            S_WRITE_BACK: begin
                pe_en <= 0;
                dma_r_addr <= r_addr;
                dma_r_len  <= m_dim * n_dim * (ACC_W/8);  // result bytes
                dma_r_start <= 1;
                // Directly transition after 1 cycle (simplified)
                state <= S_DONE;
            end

            S_DONE: begin
                pe_en <= 0;
                busy <= 0;
                done <= 1;
                irq  <= 1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
        endcase
    end
end

endmodule
