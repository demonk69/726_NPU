// =============================================================================
// Module  : npu_power
// Project : NPU_prj
// Desc    : Power management control signal generation.
//           Keeps fabric clocking on clk and emits DFS/clock-gate enables.
// =============================================================================

`timescale 1ns/1ps

module npu_power #(
    parameter ROWS = 4,
    parameter COLS = 4
)(
    input  wire        clk,          // system clock
    input  wire        rst_n,
    // DFS configuration
    input  wire [2:0]  div_sel,      // 000=1x enable, 001=1/2, 010=1/4, 011=1/8
    // Clock gating control
    input  wire [ROWS-1:0] row_cg_en,  // per-row clock gate (1=gated/OFF)
    input  wire [COLS-1:0] col_cg_en,  // per-col clock gate (1=gated/OFF)
    // Historical port names: these are enables, not generated/gated clocks.
    output wire        npu_clk,
    output wire [ROWS-1:0] row_clk_gated,
    output wire [COLS-1:0] col_clk_gated
);

// ---------------------------------------------------------------------------
// DFS: clock-enable pulse generator. Do not generate fabric clocks here.
// ---------------------------------------------------------------------------
reg [2:0] dfs_cnt;
reg       dfs_ce;

always @(posedge clk) begin
    if (!rst_n) begin
        dfs_cnt  <= 0;
        dfs_ce   <= 1'b0;
    end else begin
        case (div_sel)
            3'b000: begin
                dfs_cnt <= 3'd0;
                dfs_ce  <= 1'b1;
            end
            3'b001: begin
                dfs_cnt <= dfs_cnt + 3'd1;
                dfs_ce  <= (dfs_cnt[0] == 1'b0);
            end
            3'b010: begin
                dfs_cnt <= dfs_cnt + 3'd1;
                dfs_ce  <= (dfs_cnt[1:0] == 2'b00);
            end
            3'b011: begin
                dfs_cnt <= dfs_cnt + 3'd1;
                dfs_ce  <= (dfs_cnt == 3'b000);
            end
            default: begin
                dfs_cnt <= 3'd0;
                dfs_ce  <= 1'b1;
            end
        endcase
    end
end

assign npu_clk = clk;

// ---------------------------------------------------------------------------
// Row enables
// ---------------------------------------------------------------------------
assign row_clk_gated = {ROWS{dfs_ce}} & ~row_cg_en;

// ---------------------------------------------------------------------------
// Column enables
// ---------------------------------------------------------------------------
assign col_clk_gated = {COLS{dfs_ce}} & ~col_cg_en;

endmodule
