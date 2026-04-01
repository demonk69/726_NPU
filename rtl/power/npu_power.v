// =============================================================================
// Module  : npu_power
// Project : NPU_prj
// Desc    : Power management: clock gating + dynamic frequency scaling (DFS).
//           Generates gated clocks for each row/column of the PE array.
// =============================================================================

`timescale 1ns/1ps

module npu_power #(
    parameter ROWS = 4,
    parameter COLS = 4
)(
    input  wire        clk,          // system clock
    input  wire        rst_n,
    // DFS configuration
    input  wire [2:0]  div_sel,      // 000=÷1, 001=÷2, 010=÷4, 011=÷8
    // Clock gating control
    input  wire [ROWS-1:0] row_cg_en,  // per-row clock gate (1=gated/OFF)
    input  wire [COLS-1:0] col_cg_en,  // per-col clock gate (1=gated/OFF)
    // Gated clock outputs
    output wire        npu_clk,          // divided clock
    output wire [ROWS-1:0] row_clk_gated,
    output wire [COLS-1:0] col_clk_gated
);

// ---------------------------------------------------------------------------
// DFS: Simple clock divider counter
// ---------------------------------------------------------------------------
reg [2:0] dfs_cnt;
reg       dfs_clk_r;

always @(posedge clk) begin
    if (!rst_n) begin
        dfs_cnt  <= 0;
        dfs_clk_r <= 0;
    end else begin
        case (div_sel)
            3'b000: dfs_clk_r <= clk;   // bypass
            3'b001: begin  // ÷2
                dfs_cnt <= dfs_cnt + 1'b1;
                dfs_clk_r <= ~dfs_cnt[0];
            end
            3'b010: begin  // ÷4
                dfs_cnt <= dfs_cnt + 1'b1;
                dfs_clk_r <= ~dfs_cnt[1];
            end
            3'b011: begin  // ÷8
                dfs_cnt <= dfs_cnt + 1'b1;
                dfs_clk_r <= ~dfs_cnt[2];
            end
            default: dfs_clk_r <= clk;
        endcase
    end
end

assign npu_clk = (div_sel == 3'b000) ? clk : dfs_clk_r;

// ---------------------------------------------------------------------------
// Row clock gating
// ---------------------------------------------------------------------------
genvar i;
generate
    for (i = 0; i < ROWS; i = i+1) begin : gen_row_cg
        // In FPGA: use BUFGCE or clock gate primitive
        // In ASIC: use integrated clock gating cell (ICG)
        // Here: behavioral model
        assign row_clk_gated[i] = row_cg_en[i] ? 1'b0 : npu_clk;
    end
endgenerate

// ---------------------------------------------------------------------------
// Col clock gating
// ---------------------------------------------------------------------------
genvar j;
generate
    for (j = 0; j < COLS; j = j+1) begin : gen_col_cg
        assign col_clk_gated[j] = col_cg_en[j] ? 1'b0 : npu_clk;
    end
endgenerate

endmodule
