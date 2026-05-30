`timescale 1ns/1ps

module tb_op_counter_perf;

localparam CLK_T = 10;

reg clk = 1'b0;
always #(CLK_T/2) clk = ~clk;

reg rst_n = 1'b0;
initial begin
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
end

reg        pe_en;
reg        pe_flush;
reg        ctrl_busy;
reg        ctrl_done;
reg        compute_valid;
wire [15:0] pe_valid = compute_valid ? 16'hFFFF : 16'h0000;

wire [63:0] total_mac_ops;
wire [63:0] total_ops;
wire [31:0] total_pe_cycles;
wire [31:0] total_busy_cycles;
wire [31:0] total_compute_cycles;
wire [31:0] total_dma_cycles;
wire [31:0] peak_ops_per_cycle;
wire [31:0] tops_x1e6;
wire [31:0] compute_util_bp;
wire [31:0] e2e_util_bp;

op_counter #(
    .ROWS(16),
    .COLS(16),
    .FREQ_MHZ(500),
    .ENABLE_DERIVED(1)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(1'b0),
    .pe_en(pe_en),
    .pe_flush(pe_flush),
    .ctrl_busy(ctrl_busy),
    .ctrl_done(ctrl_done),
    .dma_w_done(1'b0),
    .dma_a_done(1'b0),
    .dma_r_done(1'b0),
    .pe_valid(pe_valid),
    .m_dim(32'd16),
    .n_dim(32'd16),
    .k_dim(32'd16),
    .compute_valid(compute_valid),
    .active_rows(5'd16),
    .active_cols(6'd16),
    .simd_lanes(4'd1),
    .total_mac_ops(total_mac_ops),
    .total_pe_cycles(total_pe_cycles),
    .total_busy_cycles(total_busy_cycles),
    .total_compute_cycles(total_compute_cycles),
    .total_dma_cycles(total_dma_cycles),
    .total_ops(total_ops),
    .peak_ops_per_cycle(peak_ops_per_cycle),
    .tops_x1e6(tops_x1e6),
    .compute_util_bp(compute_util_bp),
    .e2e_util_bp(e2e_util_bp)
);

integer i;
integer pass_cnt;
integer fail_cnt;

task check32;
    input [127:0] label;
    input [31:0] got;
    input [31:0] exp;
    begin
        if (got === exp) begin
            pass_cnt = pass_cnt + 1;
        end else begin
            fail_cnt = fail_cnt + 1;
            $display("[FAIL] %0s got=%0d expected=%0d", label, got, exp);
        end
    end
endtask

task check64;
    input [127:0] label;
    input [63:0] got;
    input [63:0] exp;
    begin
        if (got === exp) begin
            pass_cnt = pass_cnt + 1;
        end else begin
            fail_cnt = fail_cnt + 1;
            $display("[FAIL] %0s got=%0d expected=%0d", label, got, exp);
        end
    end
endtask

initial begin
    pe_en = 1'b0;
    pe_flush = 1'b0;
    ctrl_busy = 1'b0;
    ctrl_done = 1'b0;
    compute_valid = 1'b0;
    pass_cnt = 0;
    fail_cnt = 0;

    @(posedge rst_n);
    @(negedge clk);

    pe_en = 1'b1;
    ctrl_busy = 1'b1;
    compute_valid = 1'b1;
    for (i = 0; i < 16; i = i + 1)
        @(posedge clk);

    @(negedge clk);
    pe_en = 1'b0;
    ctrl_busy = 1'b0;
    compute_valid = 1'b0;
    ctrl_done = 1'b1;
    @(posedge clk);
    @(negedge clk);
    ctrl_done = 1'b0;
    @(posedge clk);

    check64("MAC_OPS", total_mac_ops, 64'd4096);
    check64("OPS", total_ops, 64'd8192);
    check32("BUSY_CYCLES", total_busy_cycles, 32'd16);
    check32("COMPUTE_CYCLES", total_compute_cycles, 32'd16);
    check32("DMA_CYCLES", total_dma_cycles, 32'd0);
    check32("PEAK_OPS_CYCLE", peak_ops_per_cycle, 32'd512);
    check32("TOPS_X1E6", tops_x1e6, 32'd256000);
    check32("COMPUTE_UTIL_BP", compute_util_bp, 32'd10000);
    check32("E2E_UTIL_BP", e2e_util_bp, 32'd10000);

    if (fail_cnt == 0) begin
        $display("[PERF] op_counter MAC_OPS=%0d OPS=%0d BUSY_CYCLES=%0d COMPUTE_CYCLES=%0d TOPS_X1E6=%0d COMPUTE_UTIL_BP=%0d E2E_UTIL_BP=%0d PEAK_OPS_CYCLE=%0d",
                 total_mac_ops, total_ops, total_busy_cycles,
                 total_compute_cycles, tops_x1e6, compute_util_bp,
                 e2e_util_bp, peak_ops_per_cycle);
        $display("[PASS] tb_op_counter_perf: ALL %0d CHECKS PASSED", pass_cnt);
    end else begin
        $display("[FAIL] tb_op_counter_perf: %0d passed, %0d failed", pass_cnt, fail_cnt);
    end
    $finish;
end

endmodule
