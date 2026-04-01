// =============================================================================
// Module  : tb_pe_top
// Project : NPU_prj
// Desc    : Testbench for pe_top.
//           Tests:
//           1. INT8  Weight-Stationary  mode
//           2. INT8  Output-Stationary  mode
//           3. FP16  Weight-Stationary  mode
//           4. FP16  Output-Stationary  mode
// =============================================================================

`timescale 1ns/1ps

module tb_pe_top;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
parameter DATA_W = 16;
parameter ACC_W  = 32;
parameter CLK_PERIOD = 10; // 100 MHz

// ---------------------------------------------------------------------------
// DUT ports
// ---------------------------------------------------------------------------
reg               clk;
reg               rst_n;
reg               mode;
reg               stat_mode;
reg               en;
reg               flush;
reg  [DATA_W-1:0] w_in;
reg  [DATA_W-1:0] a_in;
reg  [ACC_W-1:0]  acc_in;
wire [ACC_W-1:0]  acc_out;
wire              valid_out;

// ---------------------------------------------------------------------------
// DUT instantiation
// ---------------------------------------------------------------------------
pe_top #(
    .DATA_W(DATA_W),
    .ACC_W (ACC_W)
) u_pe (
    .clk      (clk),
    .rst_n    (rst_n),
    .mode     (mode),
    .stat_mode(stat_mode),
    .en       (en),
    .flush    (flush),
    .w_in     (w_in),
    .a_in     (a_in),
    .acc_in   (acc_in),
    .acc_out  (acc_out),
    .valid_out(valid_out)
);

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------
initial clk = 0;
always #(CLK_PERIOD/2) clk = ~clk;

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------
task reset_dut;
begin
    rst_n = 0; en = 0; flush = 0;
    w_in = 0; a_in = 0; acc_in = 0;
    mode = 0; stat_mode = 0;
    repeat(4) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
end
endtask

// Drive one PE beat
task drive_beat;
    input [DATA_W-1:0] w;
    input [DATA_W-1:0] a;
    input [ACC_W-1:0]  acc;
    input              fl;
begin
    @(posedge clk);
    #1;
    w_in   = w;
    a_in   = a;
    acc_in = acc;
    flush  = fl;
    en     = 1;
end
endtask

// ---------------------------------------------------------------------------
// Test 1: INT8 Weight-Stationary
// ---------------------------------------------------------------------------
integer pass_cnt = 0;
integer fail_cnt = 0;

task check_result;
    input [ACC_W-1:0] got;
    input [ACC_W-1:0] exp;
    input [63:0]      test_id;
begin
    if (got === exp) begin
        $display("[PASS] Test %0d: got=%0d exp=%0d", test_id, $signed(got), $signed(exp));
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("[FAIL] Test %0d: got=%0d exp=%0d", test_id, $signed(got), $signed(exp));
        fail_cnt = fail_cnt + 1;
    end
end
endtask

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------
integer i;
reg signed [7:0]  iw, ia;
reg signed [31:0] expected_acc;

initial begin
    $dumpfile("sim/wave/tb_pe_top.vcd");
    $dumpvars(0, tb_pe_top);

    reset_dut;

    // -----------------------------------------------------------------------
    // Test 1: INT8 WS  -- weight=3, activations=1,2,3,4 accumulated via acc_in
    // -----------------------------------------------------------------------
    $display("\n--- Test 1: INT8 Weight-Stationary ---");
    mode      = 0;  // INT8
    stat_mode = 0;  // WS
    iw = 3;

    // Load weight (flush=1)
    drive_beat({{8{iw[7]}}, iw}, 16'd0, 32'd0, 1'b1);

    // Beat 1: a=1, acc_in=0
    drive_beat({{8{iw[7]}}, iw}, 16'd1, 32'd0, 1'b0);
    // Beat 2: a=2, acc_in = prev result (3)
    @(posedge clk); #1;
    drive_beat({{8{iw[7]}}, iw}, 16'd2, acc_out, 1'b0);
    // Beat 3: a=3, acc_in = prev result
    @(posedge clk); #1;
    drive_beat({{8{iw[7]}}, iw}, 16'd3, acc_out, 1'b0);
    // Beat 4: a=4
    @(posedge clk); #1;
    drive_beat({{8{iw[7]}}, iw}, 16'd4, acc_out, 1'b0);

    repeat(3) @(posedge clk);
    // Expected: 3*1+3*2+3*3+3*4 = 3+6+9+12 = 30
    check_result(acc_out, 32'd30, 1);

    // -----------------------------------------------------------------------
    // Test 2: INT8 OS  -- weight stream, fixed activation=2, 4 beats
    // -----------------------------------------------------------------------
    $display("\n--- Test 2: INT8 Output-Stationary ---");
    reset_dut;
    mode      = 0;  // INT8
    stat_mode = 1;  // OS
    ia = 2;

    // flush=1 on first beat to clear internal acc
    drive_beat(16'd1, {{8{ia[7]}}, ia}, 32'd0, 1'b1);
    @(posedge clk); #1;
    drive_beat(16'd2, {{8{ia[7]}}, ia}, 32'd0, 1'b0);
    @(posedge clk); #1;
    drive_beat(16'd3, {{8{ia[7]}}, ia}, 32'd0, 1'b0);
    @(posedge clk); #1;
    drive_beat(16'd4, {{8{ia[7]}}, ia}, 32'd0, 1'b0);
    // flush to drain result
    @(posedge clk); #1;
    drive_beat(16'd0, 16'd0, 32'd0, 1'b1);

    repeat(3) @(posedge clk);
    // Expected: 2*(1+2+3+4)=20; but OS acc starts loading after flush
    // flush beats: first flush starts acc with beat-1 product
    // so acc = 1*2 + 2*2 + 3*2 + 4*2 = 20
    check_result(acc_out, 32'd20, 2);

    // -----------------------------------------------------------------------
    // Test 3: FP16 WS  -- simple: w=1.0, a=1.0, expected=1.0 (FP16: 0x3C00)
    // -----------------------------------------------------------------------
    $display("\n--- Test 3: FP16 Weight-Stationary ---");
    reset_dut;
    mode      = 1;  // FP16
    stat_mode = 0;  // WS

    // FP16 1.0 = 0x3C00
    drive_beat(16'h3C00, 16'h3C00, 32'd0, 1'b1); // load weight
    @(posedge clk); #1;
    drive_beat(16'h3C00, 16'h3C00, 32'd0, 1'b0); // a=1.0
    repeat(4) @(posedge clk);
    $display("[INFO] FP16 WS result = 0x%08X (expected 0x00003C00 for 1.0 * 1.0)", acc_out);

    // -----------------------------------------------------------------------
    // Test 4: FP16 OS  -- simple smoke test
    // -----------------------------------------------------------------------
    $display("\n--- Test 4: FP16 Output-Stationary ---");
    reset_dut;
    mode      = 1;  // FP16
    stat_mode = 1;  // OS

    drive_beat(16'h3C00, 16'h4000, 32'd0, 1'b1); // 1.0 * 2.0, flush=start
    @(posedge clk); #1;
    drive_beat(16'h3C00, 16'h4000, 32'd0, 1'b0); // 1.0 * 2.0
    @(posedge clk); #1;
    drive_beat(16'h0000, 16'h0000, 32'd0, 1'b1); // flush to get result
    repeat(4) @(posedge clk);
    $display("[INFO] FP16 OS result = 0x%08X", acc_out);

    // -----------------------------------------------------------------------
    en = 0;
    $display("\n=== Summary: PASS=%0d  FAIL=%0d ===", pass_cnt, fail_cnt);
    if (fail_cnt == 0)
        $display("ALL TESTS PASSED");
    else
        $display("SOME TESTS FAILED");
    $finish;
end

// ---------------------------------------------------------------------------
// Timeout watchdog
// ---------------------------------------------------------------------------
initial begin
    #50000;
    $display("[TIMEOUT] Simulation exceeded 50us");
    $finish;
end

endmodule
