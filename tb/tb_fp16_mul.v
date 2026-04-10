// =============================================================================
// Module  : tb_fp16_mul
// Project : NPU_prj
// Desc    : Comprehensive testbench for fp16_mul (IEEE 754 FP16 multiplier).
//           Tests: normal, subnormal, special (Inf/NaN/Zero), negative,
//           rounding edge cases, overflow/underflow, sign combinations.
//
//           All expected values computed from IEEE 754 FP16 arithmetic.
//           Uses RN (round-to-nearest-even) rounding mode.
//
//           Run: iverilog -g2012 -o sim/tb_fp16_mul.vvp rtl/pe/fp16_mul.v tb/tb_fp16_mul.v
//                vvp sim/tb_fp16_mul.vvp
// =============================================================================

`timescale 1ns/1ps

module tb_fp16_mul;
    reg  [15:0] a, b;
    wire [31:0] r;

    fp16_mul #(32) uut (
        .clk(1'b0), .rst_n(1'b1), .en(1'b1),
        .a(a), .b(b), .result(r)
    );

    integer pass_cnt = 0, fail_cnt = 0;

    // Check for exact match
    task check;
        input [15:0] got, exp;
        input integer  id;
        begin
            if (got === exp) begin
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] T%0d: expected=0x%04X  got=0x%04X", id, exp, got);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    // Check for NaN: just verify NaN encoding (exp=31, mant!=0)
    task check_nan;
        input [15:0] got;
        input integer  id;
        reg [4:0] got_exp;
        reg [9:0] got_mant;
        begin
            got_exp  = got[14:10];
            got_mant = got[9:0];
            if (got_exp == 5'h1F && got_mant != 10'd0) begin
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] T%0d: expected NaN  got=0x%04X (exp=%0d mant=%0d)", id, got, got_exp, got_mant);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    initial begin
        $display("====================================================");
        $display("  fp16_mul Comprehensive Test Suite");
        $display("====================================================");

        // ============================================================
        // Group 1: Basic multiplications (normal × normal)
        // ============================================================
        // 1.0 * 1.0 = 1.0
        a = 16'h3C00; b = 16'h3C00; #1;
        check(r[15:0], 16'h3C00, 1);

        // 2.0 * 1.0 = 2.0
        a = 16'h4000; b = 16'h3C00; #1;
        check(r[15:0], 16'h4000, 2);

        // 2.0 * 2.0 = 4.0
        a = 16'h4000; b = 16'h4000; #1;
        check(r[15:0], 16'h4400, 3);

        // 2.0 * 1.5 = 3.0
        a = 16'h4000; b = 16'h3E00; #1;
        check(r[15:0], 16'h4200, 4);

        // 1.5 * 1.5 = 2.25 (lzc=0 case, product >= 2.0)
        a = 16'h3E00; b = 16'h3E00; #1;
        check(r[15:0], 16'h4080, 5);

        // 3.0 * 1.0 = 3.0
        a = 16'h4200; b = 16'h3C00; #1;
        check(r[15:0], 16'h4200, 6);

        // 1.5 * 2.0 = 3.0
        a = 16'h3E00; b = 16'h4000; #1;
        check(r[15:0], 16'h4200, 7);

        // 4.0 * 4.0 = 16.0
        a = 16'h4400; b = 16'h4400; #1;
        check(r[15:0], 16'h4C00, 8);

        // 1.0 * 0.5 = 0.5
        a = 16'h3C00; b = 16'h3800; #1;
        check(r[15:0], 16'h3800, 9);

        // 1.5 * 0.5 = 0.75
        a = 16'h3E00; b = 16'h3800; #1;
        check(r[15:0], 16'h3A00, 10);

        // 0.5 * 0.5 = 0.25
        a = 16'h3800; b = 16'h3800; #1;
        check(r[15:0], 16'h3400, 11);

        // 1.001^2 (needs rounding: 1.0010^2 = 1.002001 -> RN to 1.0020)
        // 1.001 in FP16: exp=15, mant=0000000001 -> 0x3C01
        // Result: 0x3C02
        a = 16'h3C01; b = 16'h3C01; #1;
        check(r[15:0], 16'h3C02, 12);

        // 1.25 * 1.25 = 1.5625 (no rounding needed)
        // 1.25: exp=15, mant=0100000000 = 0x200 -> 0x3D00
        // 1.5625: exp=15, mant=1001000000 = 0x240 -> 0x3E40
        a = 16'h3D00; b = 16'h3D00; #1;
        check(r[15:0], 16'h3E40, 13);

        // ============================================================
        // Group 2: Negative numbers
        // ============================================================
        // -1.0 * 1.0 = -1.0
        a = 16'hBC00; b = 16'h3C00; #1;
        check(r[15:0], 16'hBC00, 20);

        // -1.0 * -1.0 = 1.0
        a = 16'hBC00; b = 16'hBC00; #1;
        check(r[15:0], 16'h3C00, 21);

        // -2.0 * 1.5 = -3.0
        a = 16'hC000; b = 16'h3E00; #1;
        check(r[15:0], 16'hC200, 22);

        // 1.5 * -2.0 = -3.0
        a = 16'h3E00; b = 16'hC000; #1;
        check(r[15:0], 16'hC200, 23);

        // -1.5 * -1.5 = +2.25 (positive!)
        a = 16'hBE00; b = 16'hBE00; #1;
        check(r[15:0], 16'h4080, 24);

        // -0.5 * 0.5 = -0.25
        a = 16'hB800; b = 16'h3800; #1;
        check(r[15:0], 16'hB400, 25);

        // ============================================================
        // Group 3: Special values (Inf, NaN, Zero)
        // ============================================================
        // Inf * 1.0 = Inf
        a = 16'h7C00; b = 16'h3C00; #1;
        check(r[15:0], 16'h7C00, 30);

        // Inf * Inf = Inf
        a = 16'h7C00; b = 16'h7C00; #1;
        check(r[15:0], 16'h7C00, 31);

        // Inf * -1.0 = -Inf
        a = 16'h7C00; b = 16'hBC00; #1;
        check(r[15:0], 16'hFC00, 32);

        // -Inf * -1.0 = Inf
        a = 16'hFC00; b = 16'hBC00; #1;
        check(r[15:0], 16'h7C00, 33);

        // Inf * 0.0 = NaN
        a = 16'h7C00; b = 16'h0000; #1;
        check_nan(r[15:0], 34);

        // 0.0 * Inf = NaN
        a = 16'h0000; b = 16'h7C00; #1;
        check_nan(r[15:0], 35);

        // NaN * 1.0 = NaN (quiet)
        a = 16'h7E00; b = 16'h3C00; #1;
        check_nan(r[15:0], 36);

        // 1.0 * NaN = NaN (quiet)
        a = 16'h3C00; b = 16'h7E00; #1;
        check_nan(r[15:0], 37);

        // NaN * NaN = NaN
        a = 16'h7E00; b = 16'h7E00; #1;
        check_nan(r[15:0], 38);

        // 0.0 * 1.0 = 0.0
        a = 16'h0000; b = 16'h3C00; #1;
        check(r[15:0], 16'h0000, 39);

        // 0.0 * 0.0 = 0.0
        a = 16'h0000; b = 16'h0000; #1;
        check(r[15:0], 16'h0000, 40);

        // -0.0 * 1.0 = -0.0
        a = 16'h8000; b = 16'h3C00; #1;
        check(r[15:0], 16'h8000, 41);

        // -0.0 * -1.0 = +0.0
        a = 16'h8000; b = 16'hBC00; #1;
        check(r[15:0], 16'h0000, 42);

        // ============================================================
        // Group 4: Overflow
        // ============================================================
        // max FP16 (65504) * 1.0 = 65504
        // 65504 = exp=30, mant=1111111111 -> 0x7BFF
        a = 16'h7BFF; b = 16'h3C00; #1;
        check(r[15:0], 16'h7BFF, 50);

        // max FP16 * 2.0 = Inf (overflow)
        a = 16'h7BFF; b = 16'h4000; #1;
        check(r[15:0], 16'h7C00, 51);

        // max FP16 * 1.5 = Inf (65504 * 1.5 = 98256 > 65504)
        a = 16'h7BFF; b = 16'h3E00; #1;
        check(r[15:0], 16'h7C00, 52);

        // ============================================================
        // Group 5: Subnormal multiplications (KEY TEST AREA)
        // Subnormals: exp=0, implicit bit=0, effective exp=1
        // ============================================================
 
        a = 16'h03FF; b = 16'h3C00; #1;
        check(r[15:0], 16'h03FF, 60);

        a = 16'h03FF; b = 16'h4000; #1;
        check(r[15:0], 16'h07FE, 61);

        a = 16'h03FF; b = 16'h3800; #1;
        check(r[15:0], 16'h0200, 62);

        // max subnormal * 2.0 = 0x07FE (normal, exp=1)
        a = 16'h03FF; b = 16'h4000; #1;
        check(r[15:0], 16'h07FE, 63);

        // min subnormal * 1.0 = 0x0001
        a = 16'h0001; b = 16'h3C00; #1;
        check(r[15:0], 16'h0001, 64);

        // min subnormal * 2.0 = 0x0002
        a = 16'h0001; b = 16'h4000; #1;
        check(r[15:0], 16'h0002, 65);

        // min subnormal^2 = 0 (underflow, correct flush-to-zero)
        a = 16'h0001; b = 16'h0001; #1;
        check(r[15:0], 16'h0000, 66);


        a = 16'h0400; b = 16'h3400; #1;
        check(r[15:0], 16'h0100, 66);  // min normal * 0.125 = subnormal

        // sub * sub (underflow to zero)
        a = 16'h0200; b = 16'h0200; #1;
        check(r[15:0], 16'h0000, 67);  // sub * sub = 0

        // ============================================================
        // Summary
        // ============================================================
        $display("");
        $display("====================================================");
        $display("  RESULT: PASS=%0d  FAIL=%0d", pass_cnt, fail_cnt);
        if (fail_cnt == 0)
            $display("  ALL TESTS PASSED!");
        else
            $display("  SOME TESTS FAILED!");
        $display("====================================================");

        $finish;
    end

endmodule
