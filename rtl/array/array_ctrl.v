// =============================================================================
// Module  : array_ctrl
// Project : NPU_prj
// Desc    : Simplified PE array controller. Orchestrates weight loading,
//           activation feeding, and result draining for systolic PE array.
// =============================================================================

`timescale 1ns/1ps

module array_ctrl #(
    parameter ROWS   = 4,
    parameter COLS   = 4,
    parameter DATA_W = 16,
    parameter ACC_W  = 32
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  start,      // begin computation
    output reg                   busy,
    // control to PE array
    output reg                   pe_en,
    output reg                   pe_flush,
    output reg                   pe_mode,
    output reg                   pe_stat,
    // weight/activation/psum buses
    output reg  [COLS*DATA_W-1:0] pe_w_out,
    output reg  [ROWS*DATA_W-1:0] pe_act_out,
    output reg  [COLS*ACC_W-1:0]  pe_acc_out,
    input  wire [COLS*ACC_W-1:0]  pe_result,
    input  wire [COLS-1:0]        pe_valid
);

// Simple counter-based scheduler
reg [15:0] cycle_cnt;
wire       load_done   = (cycle_cnt >= ROWS + COLS);
wire       drain_done  = (cycle_cnt >= ROWS + COLS + 5);

always @(posedge clk) begin
    if (!rst_n) begin
        pe_en   <= 0;
        pe_flush <= 0;
        busy    <= 0;
        cycle_cnt <= 0;
    end else if (start) begin
        busy    <= 1;
        pe_en   <= 1;
        cycle_cnt <= cycle_cnt + 1;
        if (drain_done) begin
            busy <= 0;
            pe_en <= 0;
            cycle_cnt <= 0;
        end else if (load_done && !pe_flush) begin
            pe_flush <= 1;
        end
    end else begin
        pe_en   <= 0;
        pe_flush <= 0;
    end
end

endmodule
