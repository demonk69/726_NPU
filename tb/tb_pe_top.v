// =============================================================================
// Module  : tb_pe_top
// Project : NPU_prj
// Desc    : Testbench for pe_top.
//           Tests: INT8/FP16 x WS/OS, pipeline stalls, edge cases,
//           true FP16 accumulation, true weight-stationary latch.
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
reg               load_w;                // NEW: WS weight latch
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
    .load_w   (load_w),
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
    rst_n = 0; en = 0; flush = 0; load_w = 0;
    w_in = 0; a_in = 0; acc_in = 0;
    mode = 0; stat_mode = 0;
    repeat(4) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
end
endtask

// Drive one PE beat (generic)
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
    load_w = 0;
    en     = 1;
end
endtask

// Drive WS beat — weight is NOT sent (already latched)
// Drives exactly one beat: en=1 for one cycle then drops to 0.
task drive_ws_beat;
    input [DATA_W-1:0] a;
    input [ACC_W-1:0]  acc;
begin
    @(posedge clk);
    #1;
    w_in   = 16'd0;  // ignored — weight_reg is active
    a_in   = a;
    acc_in = acc;
    flush  = 0;
    load_w = 0;
    en     = 1;
    @(posedge clk);
    #1;
    en     = 0;
end
endtask

// ---------------------------------------------------------------------------
// 检查与记录逻辑
// ---------------------------------------------------------------------------
integer pass_cnt = 0;
integer fail_cnt = 0;
integer failed_tests[0:99];
integer fail_idx = 0;

task check_result;
    input [ACC_W-1:0] got;
    input [ACC_W-1:0] exp;
    input integer     test_id;
begin
    $display("Expected Output  : 0x%08X (Dec: %0d)", exp, $signed(exp));
    $display("Actual Output    : 0x%08X (Dec: %0d)", got, $signed(got));

    if (got === exp) begin
        $display("Status           : [PASS]");
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("Status           : [FAIL] <--- !!!");
        fail_cnt = fail_cnt + 1;
        if (fail_idx < 100) begin
            failed_tests[fail_idx] = test_id;
            fail_idx = fail_idx + 1;
        end
    end
    $display("---------------------------------------------------\n");
end
endtask

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------
integer i, k;
reg signed [7:0]  iw, ia;

initial begin
    $dumpfile("tb_pe_top.vcd");
    $dumpvars(0, tb_pe_top);

    reset_dut;

    // =======================================================================
    // Test 1: INT8 WS — internal accumulation with flush
    // Load weight once (w=3), then stream activations {1,2,3,4}, flush to get sum
    // Expected internal: 3*1 + 3*2 + 3*3 + 3*4 = 30
    // =======================================================================
    $display("\n>>> Starting Test 1: INT8 Weight-Stationary (internal acc + flush) <<<");
    mode      = 0;
    stat_mode = 0;
    iw = 3;

    // Load weight into PE (first beat also computes): w=3, a=1
    @(posedge clk); #1;
    w_in = {{8{iw[7]}}, iw}; a_in = 16'd1; acc_in = 32'd0;
    flush = 0; load_w = 1; en = 1;
    @(posedge clk); #1;
    en = 0; load_w = 0;          // drop en after exactly ONE cycle

    // Stream remaining activations (internal accumulation)
    drive_ws_beat(16'd2, 32'd0);      // w=3, a=2 => internal acc += 6
    drive_ws_beat(16'd3, 32'd0);      // w=3, a=3 => internal acc += 9
    drive_ws_beat(16'd4, 32'd0);      // w=3, a=4 => internal acc += 12

    // Flush to get accumulated result (a_in=0: pure flush, no extra product)
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;
    repeat(3) @(posedge clk); #1;

    $display("[TEST 1 RESULT] Info: w=3 loaded once, acts={1,2,3,4}, internal acc, flush");
    check_result(acc_out, 32'd30, 1);

    // =======================================================================
    // Test 2: INT8 OS
    // weights={1,2,3,4}, act=2 => 2+4+6+8 = 20
    // =======================================================================
    $display("\n>>> Starting Test 2: INT8 Output-Stationary <<<");
    reset_dut;
    mode      = 0;
    stat_mode = 1;
    ia = 2;

    @(posedge clk); #1; w_in=16'd1; a_in={{8{ia[7]}},ia}; acc_in=0; flush=0; en=1;
    @(posedge clk); #1; w_in=16'd2; a_in={{8{ia[7]}},ia}; acc_in=0; flush=0; en=1;
    @(posedge clk); #1; w_in=16'd3; a_in={{8{ia[7]}},ia}; acc_in=0; flush=0; en=1;
    @(posedge clk); #1; w_in=16'd4; a_in={{8{ia[7]}},ia}; acc_in=0; flush=0; en=1;
    @(posedge clk); #1; en=0; flush=0;
    @(posedge clk); #1; en=0; flush=0;
    @(posedge clk); #1; w_in=16'd0; a_in=16'd0; acc_in=0; flush=1; en=1;
    @(posedge clk); #1; flush=0; en=0;

    repeat(3) @(posedge clk); #1;
    $display("[TEST 2 RESULT] Info: weights={1,2,3,4}, act=2 internal acc");
    check_result(acc_out, 32'd20, 2);

    // =======================================================================
    // Test 3: FP16 WS — single beat with flush
    // w=2.0 (0x4000) * a=1.5 (0x3E00) = 3.0 (0x40400000 in FP32)
    // =======================================================================
    $display("\n>>> Starting Test 3: FP16 Weight-Stationary (single beat + flush) <<<");
    reset_dut;
    mode      = 1;
    stat_mode = 0;

    // Load weight and compute first product
    @(posedge clk); #1;
    w_in = 16'h4000; a_in = 16'h3E00; acc_in = 32'd0;
    flush = 0; load_w = 1; en = 1;
    @(posedge clk); #1;
    en = 0;
    @(posedge clk); #1;

    // Flush to get result (a_in=0 to avoid extra product)
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;
    repeat(3) @(posedge clk); #1;

    $display("[TEST 3 RESULT] Info: w=2.0 (0x4000), a=1.5 (0x3E00), flush");
    check_result(acc_out, 32'h40400000, 3);

    // =======================================================================
    // Test 4: FP16 OS
    // 2.0*1.5 + 2.0*1.5 = 3.0 + 3.0 = 6.0 (0x4600)
    // =======================================================================
    $display("\n>>> Starting Test 4: FP16 Output-Stationary <<<");
    reset_dut;
    mode      = 1;
    stat_mode = 1;

    @(posedge clk); #1; w_in = 16'h4000; a_in = 16'h3E00; acc_in = 32'd0; flush = 0; en = 1;
    @(posedge clk); #1; w_in = 16'h4000; a_in = 16'h3E00; acc_in = 32'd0; flush = 0; en = 1;
    @(posedge clk); #1; en = 0; flush = 0;
    @(posedge clk); #1; en = 0; flush = 0;
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; acc_in = 0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;

    repeat(3) @(posedge clk); #1;
    $display("[TEST 4 RESULT] Info: 2.0*1.5 + 2.0*1.5");
    check_result(acc_out, 32'h40C00000, 4);

    // =======================================================================
    // Test 5: INT8 OS with Pipeline Stalls
    // 5*2=10, stall 2 cyc, 3*3=9 => total=19
    // =======================================================================
    $display("\n>>> Starting Test 5: INT8 OS with Pipeline Stalls <<<");
    reset_dut;
    mode      = 0;
    stat_mode = 1;

    @(posedge clk); #1; w_in=16'd5; a_in=16'd2; acc_in=0; flush=0; en=1;
    @(posedge clk); #1; en=0;
    @(posedge clk); #1; en=0;
    @(posedge clk); #1; w_in=16'd3; a_in=16'd3; acc_in=0; flush=0; en=1;
    @(posedge clk); #1; w_in=16'd0; a_in=16'd0; acc_in=0; flush=1; en=1;
    @(posedge clk); #1; en=0; flush=0;

    repeat(3) @(posedge clk); #1;
    $display("[TEST 5 RESULT] Info: 5*2, stall 2 cyc, 3*3");
    check_result(acc_out, 32'd19, 5);

    // =======================================================================
    // Test 6: FP16 Special Edge Cases (WS mode with flush)
    // =======================================================================
    $display("\n>>> Starting Test 6: FP16 Edge Cases <<<");
    reset_dut;
    mode      = 1;
    stat_mode = 0;

    // Test 61: Inf * 1.0 = Inf
    @(posedge clk); #1;
    w_in = 16'h7C00; a_in = 16'h3C00; acc_in = 32'd0;
    flush = 0; load_w = 1; en = 1;
    @(posedge clk); #1; en = 0; load_w = 0;
    @(posedge clk); #1;
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;
    repeat(3) @(posedge clk); #1;
    $display("[TEST 61 RESULT] Info: Inf (0x7C00) * 1.0 (0x3C00) -> Inf");
    check_result(acc_out, 32'h7F800000, 61);

    // Test 62: Inf * 0.0 = NaN
    @(posedge clk); #1;
    w_in = 16'h7C00; a_in = 16'h0000; acc_in = 32'd0;
    flush = 0; load_w = 1; en = 1;
    @(posedge clk); #1; en = 0; load_w = 0;
    @(posedge clk); #1;
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;
    repeat(3) @(posedge clk); #1;
    $display("[TEST 62 RESULT] Info: Inf (0x7C00) * 0.0 (0x0000) -> NaN");
    check_result(acc_out, 32'hFFC00000, 62);

    // =======================================================================
    // Test 7: FP16 Sign Toggling & Accumulation (OS)
    // 4.0 + (-2.0) = 2.0 (0x4000)
    // =======================================================================
    $display("\n>>> Starting Test 7: FP16 Sign Toggling & Accumulation <<<");
    reset_dut;
    mode      = 1;
    stat_mode = 1;

    drive_beat(16'h4000, 16'h4000, 32'd0, 1'b0); // 2.0*2.0 = 4.0
    drive_beat(16'hC000, 16'h3C00, 32'd0, 1'b0); // -2.0*1.0 = -2.0
    @(posedge clk); #1; en=0;
    @(posedge clk); #1; en=0;
    drive_beat(16'd0, 16'd0, 32'd0, 1'b1);
    @(posedge clk); #1; en=0; flush=0;

    repeat(3) @(posedge clk); #1;
    $display("[TEST 7 RESULT] Info: FP16 4.0 + (-2.0) using FP32 mixed-precision accumulation");
    check_result(acc_out, 32'h40000000, 7);

    // =======================================================================
    // Test 8: Back-to-Back Flush
    // =======================================================================
    $display("\n>>> Starting Test 8: Back-to-Back Flush <<<");
    reset_dut;
    mode = 0;
    stat_mode = 1;

    drive_beat(16'd4, 16'd5, 32'd0, 1'b0); // 4*5=20
    @(posedge clk); #1; en=0;
    @(posedge clk); #1; en=0;

    drive_beat(16'd0, 16'd0, 32'd0, 1'b1); // Flush 1
    @(posedge clk); #1; en=0; flush=0;
    repeat(3) @(posedge clk); #1;
    $display("[TEST 81 RESULT] Info: Flush 1 (sum of 4*5)");
    check_result(acc_out, 32'd20, 81);

    drive_beat(16'd0, 16'd0, 32'd0, 1'b1); // Flush 2: empty
    @(posedge clk); #1; en=0; flush=0;
    repeat(3) @(posedge clk); #1;
    $display("[TEST 82 RESULT] Info: Flush 2 (back-to-back empty)");
    check_result(acc_out, 32'd0, 82);

    // =======================================================================
    // Test 9: Dynamic Mode Switching (INT8 -> FP16) in WS mode with flush
    // =======================================================================
    $display("\n>>> Starting Test 9: Dynamic Mode Switching (INT8 -> FP16) <<<");
    reset_dut;
    stat_mode = 0;

    // INT8 beat: w=2, a=3 (load_w=1 since WS mode)
    mode = 0;
    @(posedge clk); #1;
    w_in = 16'd2; a_in = 16'd3; acc_in = 32'd0;
    flush = 0; load_w = 1; en = 1;
    @(posedge clk); #1; en = 0; load_w = 0;
    @(posedge clk); #1;
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;
    repeat(3) @(posedge clk); #1;
    $display("[TEST 91 RESULT] Info: INT8 beat (w=2, a=3) with flush");
    check_result(acc_out, 32'd6, 91);

    // FP16 beat: w=1.0, a=2.0 (load_w=1 since WS mode)
    mode = 1;
    @(posedge clk); #1;
    w_in = 16'h3C00; a_in = 16'h4000; acc_in = 32'd0;
    flush = 0; load_w = 1; en = 1;
    @(posedge clk); #1; en = 0; load_w = 0;
    @(posedge clk); #1;
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;
    repeat(3) @(posedge clk); #1;
    $display("[TEST 92 RESULT] Info: FP16 beat (w=1.0, a=2.0) with flush");
    check_result(acc_out, 32'h40000000, 92);

    // =======================================================================
    // Test 10: FP16 WS internal accumulation with flush
    // Load weight w=2.0 once, then multiply with a={1.0, 2.0, 3.0}, flush
    // Internal: 2.0*1.0 + 2.0*2.0 + 2.0*3.0 = 2.0 + 4.0 + 6.0 = 12.0
    // Expected: 12.0 = 0x41400000 (FP32)
    // =======================================================================
    $display("\n>>> Starting Test 10: FP16 WS Internal Accumulation + Flush <<<");
    reset_dut;
    mode      = 1;
    stat_mode = 0;

    // Beat 0: load weight and multiply with a=1.0
    @(posedge clk); #1;
    w_in = 16'h4000; a_in = 16'h3C00; acc_in = 32'd0;  // w=2.0, a=1.0
    flush = 0; load_w = 1; en = 1;
    @(posedge clk); #1; en = 0; load_w = 0;
    // internal ws_acc = 2.0

    // Beat 1: a=2.0 (internal accumulation)
    drive_ws_beat(16'h4000, 32'd0);  // a=2.0
    // internal ws_acc = 2.0 + 4.0 = 6.0

    // Beat 2: a=3.0 (internal accumulation)
    drive_ws_beat(16'h4200, 32'd0);  // a=3.0
    // internal ws_acc = 6.0 + 6.0 = 12.0

    // Flush to get accumulated result (a_in=0: pure flush)
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;
    repeat(3) @(posedge clk); #1;

    $display("[TEST 10 RESULT] Info: w=2.0 loaded once, a={1.0,2.0,3.0}, internal acc, flush");
    check_result(acc_out, 32'h41400000, 10);

    // =======================================================================
    // Test 11: FP16 WS negative internal accumulation with flush
    // Load weight w=-1.5 (0xBE00), a={2.0, 3.0}
    // Internal: -1.5*2.0 + (-1.5)*3.0 = -3.0 + (-4.5) = -7.5
    // Expected: -7.5 = 0xC0F00000 (FP32)
    // =======================================================================
    $display("\n>>> Starting Test 11: FP16 WS Negative Internal Accumulation <<<");
    reset_dut;
    mode      = 1;
    stat_mode = 0;

    // Beat 0: load weight = -1.5, a=2.0
    @(posedge clk); #1;
    w_in = 16'hBE00; a_in = 16'h4000; acc_in = 32'd0;  // w=-1.5, a=2.0
    flush = 0; load_w = 1; en = 1;
    @(posedge clk); #1; en = 0; load_w = 0;
    // internal ws_acc = -3.0

    // Beat 1: a=3.0 (internal accumulation)
    drive_ws_beat(16'h4200, 32'd0);  // a=3.0
    // internal ws_acc = -3.0 + (-4.5) = -7.5

    // Flush to get accumulated result (a_in=0: pure flush)
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;
    repeat(3) @(posedge clk); #1;

    $display("[TEST 11 RESULT] Info: w=-1.5, a={2.0,3.0}, internal acc, flush => -7.5");
    check_result(acc_out, 32'hC0F00000, 11);

    // =======================================================================
    // Test 12: FP16 OS Complex Decimals Accumulation
    // 3.140625 + 1.234375 - 0.875 + 0.109375 = 3.609375 (0x4338)
    // =======================================================================
    $display("\n>>> Starting Test 12: FP16 Complex Decimals Accumulation <<<");
    reset_dut;
    mode      = 1;
    stat_mode = 1;

    drive_beat(16'h4248, 16'h3C00, 32'd0, 1'b0); // + 3.140625
    drive_beat(16'h3CF0, 16'h3C00, 32'd0, 1'b0); // + 1.234375
    drive_beat(16'hBB00, 16'h3C00, 32'd0, 1'b0); // - 0.875
    drive_beat(16'h2F00, 16'h3C00, 32'd0, 1'b0); // + 0.109375

    @(posedge clk); #1; en=0;
    @(posedge clk); #1; en=0;
    drive_beat(16'd0, 16'd0, 32'd0, 1'b1);
    @(posedge clk); #1; en=0; flush=0;

    repeat(3) @(posedge clk); #1;
    $display("[TEST 12 RESULT] Info: 3.140625 + 1.234375 - 0.875 + 0.109375");
    check_result(acc_out, 32'h40670000, 12);

    // =======================================================================
    // Test 13: FP16 OS Big + Tiny Value Alignment
    // 10.0 + 0.015625 = 10.015625 (0x4902)
    // =======================================================================
    $display("\n>>> Starting Test 13: FP16 Big + Tiny Value Alignment <<<");
    reset_dut;
    mode      = 1;
    stat_mode = 1;

    drive_beat(16'h4900, 16'h3C00, 32'd0, 1'b0); // 10.0 * 1.0
    drive_beat(16'h2400, 16'h3C00, 32'd0, 1'b0); // 0.015625 * 1.0

    @(posedge clk); #1; en=0;
    @(posedge clk); #1; en=0;
    drive_beat(16'd0, 16'd0, 32'd0, 1'b1);
    @(posedge clk); #1; en=0; flush=0;

    repeat(3) @(posedge clk); #1;
    $display("[TEST 13 RESULT] Info: 10.0 (0x4900) + 0.015625 (0x2400)");
    check_result(acc_out, 32'h41204000, 13);

    // =======================================================================
    // Test 14: True WS weight latch verification with flush
    // Load w=5 (INT8), send a={10}, flush, verify w=5 is latched.
    // Then load w=7 with load_w, send a={4}, flush, verify w=7.
    // =======================================================================
    $display("\n>>> Starting Test 14: True WS Weight Latch Verification <<<");
    reset_dut;
    mode      = 0;
    stat_mode = 0;

    // Test 141: Load w=5, a=10, flush => 50
    iw = 5;
    @(posedge clk); #1;
    w_in = {{8{iw[7]}}, iw}; a_in = 16'd10; acc_in = 32'd0;
    flush = 0; load_w = 1; en = 1;
    @(posedge clk); #1; en = 0; load_w = 0;
    @(posedge clk); #1;
    // Flush beat: a_in=0 to avoid extra product
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;
    repeat(3) @(posedge clk); #1;
    $display("[TEST 141 RESULT] Info: load w=5, a=10, flush => 50");
    check_result(acc_out, 32'd50, 141);

    // Test 142: w stays 5 (no load_w), a=3, flush => 15
    // drive_ws_beat now internally lowers en after one cycle
    drive_ws_beat(16'd3, 32'd0);
    // en is already 0 after drive_ws_beat completes
    @(posedge clk); #1;
    // Flush beat: a_in=0 to avoid extra product
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;
    repeat(3) @(posedge clk); #1;
    $display("[TEST 142 RESULT] Info: w still 5 (no load_w), a=3, flush => 15");
    check_result(acc_out, 32'd15, 142);

    // Test 143: Load w=7, a=4, flush => 28
    iw = 7;
    @(posedge clk); #1;
    w_in = {{8{iw[7]}}, iw}; a_in = 16'd4; acc_in = 32'd0;
    flush = 0; load_w = 1; en = 1;
    @(posedge clk); #1; en = 0; load_w = 0;
    @(posedge clk); #1;
    // Flush beat: a_in=0 to avoid extra product
    @(posedge clk); #1; w_in = 16'd0; a_in = 16'd0; flush = 1; en = 1;
    @(posedge clk); #1; flush = 0; en = 0;
    repeat(3) @(posedge clk); #1;
    $display("[TEST 143 RESULT] Info: load w=7, a=4, flush => 28");
    check_result(acc_out, 32'd28, 143);

    en = 0;

    // =======================================================================
    // Summary
    // =======================================================================
    en = 0;
    $display("\n===================================================");
    $display("=== Summary: PASS=%0d  FAIL=%0d ===", pass_cnt, fail_cnt);

    if (fail_cnt == 0) begin
        $display("ALL TESTS PASSED SUCCESSFULLY");
    end else begin
        $display("SOME TESTS FAILED. Failed Test IDs:");
        for (k = 0; k < fail_idx; k = k + 1) begin
            $display(" -> Test %0d", failed_tests[k]);
        end
    end
    $display("===================================================\n");

    $finish;
end

// ---------------------------------------------------------------------------
// Timeout watchdog
// ---------------------------------------------------------------------------
initial begin
    #100000;
    $display("[TIMEOUT] Simulation exceeded 100us");
    $finish;
end

endmodule
