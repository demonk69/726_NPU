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
    $dumpfile("../../sim/wave/tb_pe_top.vcd");
    $dumpvars(0, tb_pe_top);

    reset_dut;

    // -----------------------------------------------------------------------
    // Test 1: INT8 WS  -- weight=3, activations=1,2,3,4 accumulated via acc_in
    //
    // Pipeline latency: 3 cycles (Stage-0 latch + Stage-1 mul + Stage-2 output)
    // WS acc_in must be captured AFTER valid_out asserts, then held for 1 cycle
    // before driving the next beat so the correct value is used.
    //
    // Expected: 3*1 + 3*2 + 3*3 + 3*4 = 3 + 6 + 9 + 12 = 30
    // -----------------------------------------------------------------------
    $display("\n--- Test 1: INT8 Weight-Stationary ---");
    mode      = 0;  // INT8
    stat_mode = 0;  // WS
    iw = 3;

    // Beat 1: a=1, acc_in=0
    @(posedge clk); #1; w_in={{8{iw[7]}},iw}; a_in=16'd1; acc_in=32'd0; flush=0; en=1;
    // Wait 3 cycles for result: cyc+3 has valid_out=1 and stable acc_out
    repeat(3) @(posedge clk); #1;
    // Beat 2: a=2, acc_in=acc_out (=3)
    w_in={{8{iw[7]}},iw}; a_in=16'd2; acc_in=acc_out; flush=0; en=1;
    repeat(3) @(posedge clk); #1;
    // Beat 3: a=3, acc_in=acc_out (=9)
    w_in={{8{iw[7]}},iw}; a_in=16'd3; acc_in=acc_out; flush=0; en=1;
    repeat(3) @(posedge clk); #1;
    // Beat 4: a=4, acc_in=acc_out (=18)
    w_in={{8{iw[7]}},iw}; a_in=16'd4; acc_in=acc_out; flush=0; en=1;
    repeat(3) @(posedge clk); #1;
    en=0;
    // Expected: 3*1+3*2+3*3+3*4 = 30
    check_result(acc_out, 32'd30, 1);

    // -----------------------------------------------------------------------
    // Test 2: INT8 OS  -- weight stream 1,2,3,4; fixed activation=2; 4 beats
    //
    // OS accumulates internally; send one-cycle flush after all data passes
    // through pipeline (pipeline depth = 2 cycles for data; flush needs 2 more).
    // Expected: 1*2 + 2*2 + 3*2 + 4*2 = 20
    // -----------------------------------------------------------------------
    $display("\n--- Test 2: INT8 Output-Stationary ---");
    reset_dut;
    mode      = 0;  // INT8
    stat_mode = 1;  // OS
    ia = 2;

    // 4 data beats, flush=0
    @(posedge clk); #1; w_in=16'd1; a_in={{8{ia[7]}},ia}; acc_in=0; flush=0; en=1;
    @(posedge clk); #1; w_in=16'd2; a_in={{8{ia[7]}},ia}; acc_in=0; flush=0; en=1;
    @(posedge clk); #1; w_in=16'd3; a_in={{8{ia[7]}},ia}; acc_in=0; flush=0; en=1;
    @(posedge clk); #1; w_in=16'd4; a_in={{8{ia[7]}},ia}; acc_in=0; flush=0; en=1;
    // Idle 2 cycles: let last data beat drain through Stage-0→Stage-1→Stage-2 (os_acc)
    @(posedge clk); #1; en=0; flush=0;
    @(posedge clk); #1; en=0; flush=0;
    // Single-cycle flush beat
    @(posedge clk); #1; w_in=16'd0; a_in=16'd0; acc_in=0; flush=1; en=1;
    @(posedge clk); #1; flush=0; en=0;
    // Wait for flush to propagate through pipeline (2 cycles)
    repeat(3) @(posedge clk); #1;
    check_result(acc_out, 32'd20, 2);

    // -----------------------------------------------------------------------
    // Test 3: FP16 WS  -- w=2.0, a=1.5  => 2.0*1.5=3.0 (FP16: 0x4200)
    //
    // fp16_mul outputs sign-extended FP16 in low 16 bits of a 32-bit word.
    // WS mode: acc_out = acc_in + sign_extended(fp16_product).
    // Single beat: acc_in=0, product = sign_ext(0x4200) = 0x00004200.
    // So acc_out = 0 + 0x4200 = 0x00004200.
    // -----------------------------------------------------------------------
    $display("\n--- Test 3: FP16 Weight-Stationary ---");
    reset_dut;
    mode      = 1;  // FP16
    stat_mode = 0;  // WS

    // FP16: 2.0 = 0x4000, 1.5 = 0x3E00, product 3.0 = 0x4200
    // Single beat: w=2.0, a=1.5, acc_in=0 → acc_out = 0x00004200
    @(posedge clk); #1;
    w_in = 16'h4000; a_in = 16'h3E00; acc_in = 32'd0; flush = 0; en = 1;
    repeat(3) @(posedge clk); #1;
    en = 0;
    // Check: low 16 bits should be FP16 3.0 = 0x4200, high 16 bits = 0x0000
    check_result(acc_out, 32'h00004200, 3);

    // -----------------------------------------------------------------------
    // Test 4: FP16 OS  -- w=2.0 * a=1.5  twice, accumulated internally.
    //
    // Product of 2.0*1.5 = 3.0 = 0x4200, sign-extended = 0x00004200.
    // OS accumulates two products: os_acc = 0x4200 + 0x4200 = 0x00008400.
    // flush outputs os_acc.
    //
    // Note: this is integer accumulation of sign-extended FP16, NOT true
    // FP16 addition. For a true MAC we would need an FP16 adder.
    // The check here verifies the FP16 *multiplier* is correct and the
    // OS accumulator path works.
    // -----------------------------------------------------------------------
    $display("\n--- Test 4: FP16 Output-Stationary ---");
    reset_dut;
    mode      = 1;  // FP16
    stat_mode = 1;  // OS

    // 2 data beats: 2.0 * 1.5 = 3.0 (0x4200) each, flush=0
    @(posedge clk); #1;
    w_in = 16'h4000; a_in = 16'h3E00; acc_in = 32'd0; flush = 0; en = 1;
    @(posedge clk); #1;
    w_in = 16'h4000; a_in = 16'h3E00; acc_in = 32'd0; flush = 0; en = 1;
    // Idle 2 cycles: let data drain through pipeline
    @(posedge clk); #1; en = 0; flush = 0;
    @(posedge clk); #1; en = 0; flush = 0;
    // Single-cycle flush
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; acc_in = 0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;
    // Wait for flush to propagate
    repeat(3) @(posedge clk); #1;
    // os_acc = 0x4200 + 0x4200 = 0x8400, sign-extended to 32-bit = 0x00008400
    check_result(acc_out, 32'h00008400, 4);

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
