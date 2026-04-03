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
        // max subnormal (0x03FF) * 1.0 = max subnormal (0x03FF)
        // mant_a = {0, 1111111111} = 0x3FF = 1023
        // mant_b = {1, 0} = 0x400 = 1024
        // prod = 1023 * 1024 = 1047552 = 0xFFC00
        // lzc = 2 (bit[21]=0, bit[20]=0, bit[19]=1)
        // RS = 11 - 2 = 9
        // biased_exp = 1 + 15 - 14 - 2 = 0
        // Expected: exp=0, mant=0x3FF -> 0x03FF
        a = 16'h03FF; b = 16'h3C00; #1;
        check(r[15:0], 16'h03FF, 60);

        // max subnormal * 2.0 = min normal (0x0400)
        // mant_a = 0x3FF, mant_b = 0x400
        // prod = 0xFFC00, lzc = 2, RS = 9
        // biased_exp = 1 + 16 - 14 - 2 = 1
        // mant_norm = 0xFFC00>>9[9:0] = 0x3FE
        // guard = prod[10-2] = prod[8] = 0
        // Result: {0, 00001, 0x3FE} = 0x03FE
        // But 0x03FF * 2.0 = 0x07FE (min normal is 0x0400)
        // Actually: 0x03FF represents 2^-14 * (1023/1024)
        // * 2.0 = 2^-13 * (1023/1024) which is still subnormal
        // = 0x07FE
        a = 16'h03FF; b = 16'h4000; #1;
        check(r[15:0], 16'h07FE, 61);

        // max subnormal * 0.5 = mid subnormal
        // 0x03FF * 0x5 = 0x03FF * 0.5
        // = 2^-14 * (1023/1024) * 0.5 = 2^-15 * (1023/1024)
        // = 2^-14 * (1023/2048) -> subnormal with mant = 0x200
        // = 0x0200
        // mant_a = 0x3FF, mant_b = {1, 1000000000} = 0x600
        // prod = 0x3FF * 0x600 = 1023 * 1536 = 1,571,328 = 0x17F800
        // 0x17F800 in 22-bit: 00_0101_1111_1110_0000_0000
        // lzc = 1 (bit[20]=1), RS = 10
        // 0x17F800 >> 10 = 1536 / ... hmm let me compute
        // 1,571,328 / 1024 = 1534.5 -> int = 1534 = 0x5FE
        // bit[10] of 0x5FE: 1534 > 1024, so bit[10]=1
        // mant_norm = 0x5FE & 0x3FF = 0x1FE
        // guard = prod[10-1] = prod[9] = ?
        // 0x17F800: ...111110_0000_0000, prod[9] = 1, prod[8] = 1
        // guard = prod[9] = 1
        // sticky = |prod[8:0] = 1 (prod[8]=1)
        // do_inc = guard & (sticky | mant_norm[0])
        // mant_norm[0] = 0x1FE & 1 = 0
        // do_inc = 1 & (1 | 0) = 1
        // mant_ext = 0x1FE + 1 = 0x1FF
        // rnd_ovf = 0
        // mant_r = 0x1FF
        // biased_exp = 1 + 14 - 14 - 1 = 0
        // exp_r = 0, mant_r = 0x1FF
        // Result = {0, 00000, 0111111111} = 0x01FF
        // But expected = 0x0200...
        // Wait: 0x03FF * 0.5 = 2^-15 * (1023/1024) = subnormal
        // subnormal value = mant * 2^-24
        // mant * 2^-24 = 1023 * 2^-24 * 0.5 = 511.5 * 2^-24
        // FP16 mant is integer, so round to 512 or 511
        // 512 * 2^-24 = 2^-15 = 0x0200? No.
        // Subnormal: value = mant * 2^-24 (since min subnormal exp = 2^-14 - 10 = 2^-24)
        // Wait, subnormal = 2^(1-bias) * (mant/2^10) = 2^-14 * (mant/2^10) = mant * 2^-24
        // So we need mant * 2^-24 = 511.5 * 2^-24
        // mant = 511.5, RN rounds to 512 (since .5 and LSB of 512 is 0, even -> round up)
        // mant = 512 = 0x200
        // Result = 0x0200 ✓
        //
        // But our calculation gives 0x01FF. Off by 1 again!
        // The problem: when exp=0 (subnormal), we should NOT drop the "implicit 1" bit.
        // For subnormals, there IS no implicit 1. The full shifted value IS the mantissa
        // (shifted one more to the right).
        //
        // Actually I think the issue is different. Let me reconsider.
        // For the case where biased_exp = 0:
        // - The code treats it as a normal number with exp=0
        // - But exp=0 in IEEE 754 means SUBNORMAL (no implicit 1)
        // - The result should be: value = 2^(1-15) × (mant_norm_with_extra_bit / 2^10)
        // - But the code packs {sign, 0, mant_norm} which means value = 2^(-14) × (mant_norm/2^10)
        // - The correct subnormal packing would need the leading bit included
        //
        // For 0x03FF * 0.5 = 0x0200:
        // Product = 0x17F800, after shift RS=10: 0x5FE
        // 0x5FE has bit[10]=1. For normal (exp=1): value = 2^(1-15) × (1 + 0x1FE/1024) = 2^-14 × 1.4990234375
        // For subnormal (exp=0): should be 2^-14 × (0x1FF/1024 + 0x1FE/1024) ... no that's wrong.
        //
        // The CORRECT approach: when biased_exp = 0 and the leading 1 is at bit[10]:
        // The value would be 2^(0-15) × (1 + mant_norm/1024) = 2^-15 × (1 + x)
        // But we want 2^-15 × (1023/1024) = 2^-15 × 0.9990234375
        // 1 + x = 1023/1024 → x = -1/1024, negative! That's impossible.
        //
        // So the code incorrectly sets exp=0 when it should set exp=1.
        // biased_exp = 0, but the product still has a leading 1 at bit[10],
        // meaning the "normalized" value is >= 1.0 (in the FP sense).
        //
        // THE ROOT CAUSE: the code doesn't handle the case where biased_exp is 0
        // but the result is still >= 1.0 in normalized form. This happens because
        // lzc can be large enough to make biased_exp = 0, but the barrel shifter
        // still correctly places the leading 1 at bit[10].
        //
        // Wait, but biased_exp = eff_ea + eff_eb - 14 - lzc.
        // For the correct exponent: E = eff_ea + eff_eb - 14 - lzc.
        // When E = 0, the result is exactly at the boundary between subnormal and normal.
        // If E = 0 and the leading 1 is at bit[10], the result is a normal number
        // with E = 1 (since 1.xxx × 2^(1-15) = 0.xxx × 2^(-14)).
        //
        // But 0.xxx × 2^(-14) is subnormal! So when E = 0 and we have 1.xxx,
        // the value is 1.xxx × 2^(-15), which is subnormal.
        // 1.xxx × 2^(-15) = (1 + mant/1024) × 2^(-15)
        // As subnormal: mant_sub × 2^(-14)
        // So: mant_sub × 2^(-14) = (1 + mant/1024) × 2^(-15)
        //     mant_sub = (1 + mant/1024) / 2 = (1024 + mant) / 2048
        //     = (1024 + 0x1FE) / 2048 = (1024 + 510) / 2048 = 1534 / 2048
        //     mant_sub = 1534/2 = 767 (if exact) or round to 767 or 768
        //     0x300 = 768, 0x2FF = 767. RN: 767.5 -> round to 768 = 0x300.
        //     Result = 0x0300? But expected = 0x0200!
        //
        // Hmm, something doesn't add up. Let me compute numerically:
        // 0x03FF × 0.5: 
        // 0x03FF = 2^(-14) × (1023/1024) = 6.10352e-5 × 0.9990234 = 6.09756e-5
        // × 0.5 = 3.04878e-5
        // As FP16 subnormal: mant = 3.04878e-5 / 2^(-24) = 3.04878e-5 / 5.96e-8 = 511.5
        // RN(511.5) = 512 (since 512 is even)
        // 512 = 0x200
        // Result = 0x0200 ✓
        //
        // OK so the expected result IS 0x0200. But the code gives 0x01FF.
        // The error is 1 ULP. This confirms the subnormal handling bug.
        //
        // Actually, let me reconsider. Maybe the code should produce 0x01FF because
        // of a different rounding path. Let me trace through the code carefully:
        //
        // prod = 0x17F800
        // lzc = 1 (bit[20]=1)
        // RS = 10
        // rs5 = 0x17F800 >> 10 = 1534.5? No, integer shift = 1534 = 0x5FE
        // guard = prod[10-lzc] = prod[9] = 0x17F800 bit 9
        // 0x17F800 = 0001_0111_1111_1000_0000_0000
        // bit 9 = 1
        // sticky_raw: sticky_mask = (1 << (10-1)) - 1 = (1 << 9) - 1 = 0x1FF
        // prod & sticky_mask = 0x17F800 & 0x1FF = 0x180 = 384
        // sticky_raw = |0x180 = 1
        // do_inc = guard & (sticky_raw | mant_norm[0]) = 1 & (1 | 0) = 1
        // mant_ext = {0, 0x1FE} + 1 = 0x1FF
        // rnd_ovf = 0
        // mant_r_raw = 0x1FF
        // biased_exp = 1 + 14 - 14 - 1 = 0
        // exp_after_rnd = 0
        // exp_ovf = !0[6] && (0[5:0] > 30) = 1 && (0 > 30) = 0
        // exp_unf = 0[6] = 0
        // exp_r = 0
        // mant_r = 0x1FF (since neither ovf nor unf)
        // result = {0, 5'd0, 10'd0x1FF} = 0x01FF
        //
        // But correct = 0x0200. Difference of 1 ULP.
        //
        // The issue: when biased_exp = 0, the code packs it as FP16 with exp=0.
        // For IEEE 754, exp=0 means subnormal: value = 2^(1-15) × mant/1024.
        // But our mant_norm was computed assuming NORMAL packing (implicit 1 at bit 10).
        // For normal: value = 2^(exp-15) × (1 + mant/1024) = 2^(0-15) × (1 + 0x1FF/1024)
        //                  = 2^-15 × (1 + 511/1024) = 2^-15 × 1.4990234375
        // For subnormal: value = 2^-14 × (mant/1024)
        // We need: 2^-15 × (1 + 511/1024) = 2^-14 × ((1 + 511/1024)/2) = 2^-14 × (1535/2048)
        // mant_sub = 1535/2 = 767.5 → RN = 768 = 0x300
        // But 768 = 0x300 ≠ 0x200 (the correct answer).
        //
        // WAIT. I think I'm confusing myself. Let me go back to basics.
        // 0x03FF * 0x3800 (0.5):
        // a = 0x03FF: exp=0, mant=1111111111, value = 2^-14 × 1023/1024
        // b = 0x3800: exp=14, mant=0000000000, value = 2^-1 × 1.0 = 0.5
        // Product = 2^-15 × 1023/1024
        //
        // In the multiplier:
        // mant_a = {0, 1111111111} = 0x3FF = 1023
        // mant_b = {1, 0000000000} = 0x400 = 1024
        // prod = 1023 × 1024 = 1,047,552
        //
        // Wait, that's the same as max_sub * 1.0! Of course, because 0.5 = 0x3800 has mantissa 0
        // but exponent 14, giving the same effective mantissa = 1.0 = 0x400.
        //
        // So this IS the same as max_sub * 1.0 (prod = 0xFFC00), not the case I was analyzing.
        //
        // Let me redo for max_sub * 0.5:
        // b = 0x3800 (0.5): exp=14, mant=0x000
        // mant_b = {1, 0x000} = 0x400 = 1024 (same as 1.0!)
        //
        // So max_sub * 0.5 = max_sub * (1.0 / 2) = same product / 2 of the multiplier output
        // But in the multiplier, we compute mant_a * mant_b, not (a * b) directly.
        // The exponent handles the /2 via eff_eb = 14 (not 15).
        //
        // eff_ea = 1 (subnormal), eff_eb = 14 (0x3800 has exp=14)
        // prod = 0x3FF × 0x400 = 1,047,552 = 0xFFC00
        // lzc = 2, RS = 9
        // biased_exp = 1 + 14 - 14 - 2 = -1
        // exp_unf = 1 (negative!) → exp_r = 0, mant_r = 0 (flush to zero)
        //
        // BUT 0x03FF × 0.5 = 0x0200, NOT 0!
        // THE FLUSH-TO-ZERO ON UNDERFLOW IS THE BUG!
        //
        // When biased_exp is negative (underflow), the code flushes to zero.
        // But the correct IEEE 754 behavior is to produce a subnormal result
        // (gradual underflow), not flush to zero.
        //
        // The fix: instead of flushing mant_r to 0 on underflow, we should
        // gradually denormalize by shifting right and keeping the top bits.
        //
        // OK, so the TWO bugs in fp16_mul.v are:
        // 1. When biased_exp is negative, code flushes to zero (should produce subnormal)
        // 2. T8 (0x03FF * 1.0): let me re-check this case...
        //
        // For 0x03FF * 0x3C00 (max_sub * 1.0):
        // eff_ea = 1, eff_eb = 15
        // prod = 0x3FF × 0x400 = 1,047,552 = 0xFFC00
        // lzc = 2, RS = 9
        // biased_exp = 1 + 15 - 14 - 2 = 0
        // exp_unf = 0[6] = 0 → NOT underflow
        // exp_r = biased_exp[4:0] = 0
        // mant_r = mant_r_raw (not flushed)
        // mant_norm = rs5[9:0] = 0x3FE
        // guard = prod[8] = 0, sticky = 0, do_inc = 0
        // mant_r = 0x3FE
        // result = {0, 00000, 0x3FE} = 0x03FE
        //
        // But correct = 0x03FF! Off by 1 ULP.
        //
        // The issue: when exp = 0 (subnormal), the packing should treat mant_norm
        // differently. In normal mode, bit[10] is the implicit 1, and mant[9:0] is stored.
        // In subnormal mode (exp=0), there is no implicit 1. The full value should
        // be stored in mant[9:0].
        //
        // But in our case, after RS=9, rs5 has bit[10]=1 (the "implicit 1" for normal).
        // If we're subnormal, this bit should be included in the mantissa:
        // subnormal_mant = {rs5[10], rs5[9:1]} = {1, 0x1FF} = 0x3FF
        //
        // Wait: rs5 = 0xFFC00 >> 9 = 0x7FE = 11_1111111110
        // {rs5[10], rs5[9:1]} = {1, 111111111} = 0x3FF ✓
        //
        // YES! For subnormal results, we need to include bit[10] in the mantissa
        // and shift right by 1 extra to compensate. This is the standard denormalization
        // step in IEEE 754.
        //
        // So the fix for subnormal output:
        // When biased_exp = 0: subnormal_mant = rs5[10:1], drop guard/sticky from rs5[0]
        // When biased_exp < 0: shift right by (1-biased_exp), keep top 10 bits
        //
        // This is the classic "gradual underflow" handling.
        //
        // Let me verify: for 0x03FF * 1.0:
        // biased_exp = 0, subnormal_mant = rs5[10:1] = 0x3FF
        // Result = {0, 00000, 0x3FF} = 0x03FF ✓✓✓
        //
        // For 0x03FF * 0.5 (max_sub * 0.5):
        // biased_exp = -1, need to shift right by (1-(-1)) = 2
        // rs5 = 0xFFC00 >> 9 = 0x7FE = 11_1111111110
        // Shift right by 2 more: 0x7FE >> 2 = 0x1FF = 1_111111111
        // Take top 10 bits: 0x1FF = 0x200 - 1? No, 0x1FF = 511
        // Subnormal mant = 0x1FF
        // But with rounding: after >> 2, we lose 2 bits
        // The bits shifted out: bit[1] = 1, bit[0] = 0
        // guard = bit[1] = 1, sticky = bit[0] = 0
        // do_inc = guard & (sticky | mant[0]) = 1 & (0 | 1) = 1
        // subnormal_mant = 0x1FF + 1 = 0x200
        // Result = {0, 00000, 0x200} = 0x0200 ✓✓✓
        //
        // For 0x03FF * 2.0 (max_sub * 2.0):
        // eff_ea = 1, eff_eb = 16
        // prod = 0x3FF × 0x400 = 1,047,552 = 0xFFC00 (same product!)
        // lzc = 2, RS = 9
        // biased_exp = 1 + 16 - 14 - 2 = 1
        // exp_r = 1 (normal!), mant_r = 0x3FE
        // guard = 0, sticky = 0, no rounding
        // Result = {0, 00001, 0x3FE} = 0x07FE
        // Check: 0x07FE = exp=1, mant=0x3FE
        // value = 2^(1-15) × (1 + 0x3FE/1024) = 2^-14 × (1 + 1022/1024) = 2^-14 × 2046/1024
        // = 2^-14 × 1.998046875 = 1.220703e-4
        //
        // Correct: 0x03FF * 2.0 = 2 × 6.09756e-5 = 1.21951e-4
        // 0x07FE value = 2^-14 × 1.998047 = 1.220703e-4
        // These are very close but differ by about 1 ULP.
        //
        // The exact answer: 0x03FF × 2.0
        // = 2^-14 × (1023/1024) × 2 = 2^-13 × (1023/1024)
        // = 2^-13 - 2^-23
        // This is a subnormal number (since 2^-13 < 2^-14).
        // Wait, 2^-13 > 2^-14, so this is a NORMAL number!
        // Normal range starts at 2^-14.
        // 2^-13 × (1023/1024) = 2^-13 × 0.999... which is between 2^-14 and 2^-13.
        // Since 2^-14 ≤ value < 2^-13, this is a normal number with exp = -13 + 15 = 2.
        // Wait, FP16 exp = biased exponent. value = 2^(exp-bias) × (1 + mant/1024)
        // For value = 2^-13 × (1023/1024):
        // 2^(exp-15) × (1 + mant/1024) = 2^-13 × 1023/1024
        // We need exp = 2 (so 2^-13) and 1 + mant/1024 = 1023/1024
        // mant = (1023/1024 - 1) × 1024 = 1023 - 1024 = -1
        // Negative mantissa! This means the exact result can't be represented
        // as a normal FP16 number. The closest is:
        // exp=1: 2^-14 × (1 + mant/1024). We need 2^-14 × 1.999 = 2^-14 × (2047/1024)
        // mant = 1023 = 0x3FF. Result = 0x07FF.
        // exp=2: 2^-13 × (1 + mant/1024). We need 2^-13 × 0.999 = 2^-13 × 1023/1024
        // This would need mant = -1. Impossible.
        //
        // So the result with exp=1, mant=0x3FF: value = 2^-14 × (1 + 1023/1024) = 2^-14 × 2.000
        // Wait, 1 + 1023/1024 = 2047/1024 ≈ 1.999
        // value = 2^-14 × 1.999 ≈ 2^-13.001 ≈ very close to 2^-13
        // But the exact answer is 2^-13 × 1023/1024 ≈ 2^-13 × 0.999
        // The nearest FP16 is either:
        // exp=1, mant=0x3FE: value = 2^-14 × (1 + 1022/1024) = 2^-14 × 2046/1024
        // exp=1, mant=0x3FF: value = 2^-14 × (1 + 1023/1024) = 2^-14 × 2047/1024
        // The exact: 2^-13 × 1023/1024 = 2^-14 × 2046/1024
        // = 2^-14 × (1 + 1022/1024) → mant = 0x3FE!
        //
        // So 0x07FE is the CORRECT answer, and 0x07FF would be wrong!
        // Great, the code gives 0x07FE for T51. ✓
        //
        // Now back to the main bugs:
        // BUG 1: Flush-to-zero on underflow (biased_exp < 0)
        //   Should produce subnormal (gradual underflow)
        // BUG 2: When biased_exp = 0, mant packing drops the "implicit 1"
        //   For subnormals (exp=0), there's no implicit 1

        // Now let me also check 0x0001 * 1.0 (min subnormal):
        // mant_a = {0, 0000000001} = 0x001 = 1
        // mant_b = {1, 0} = 0x400 = 1024
        // prod = 1 × 1024 = 1024 = 0x400
        // 0x400 in 22-bit: 00_0000_0000_0100_0000_0000
        // lzc: bit[21..11] = 0, bit[10] = 1
        // lzc = 11 (all bits 21..11 are 0, bit 10 is 1)
        // RS = 11 - 11 = 0
        // rs5 = prod (no shift) = 0x400
        // bit[10] = 1 ✓
        // mant_norm = rs5[9:0] = 0
        // biased_exp = 1 + 15 - 14 - 11 = -9
        // exp_unf = 1 → FLUSH TO ZERO!
        // But 0x0001 × 1.0 = 0x0001 (should stay subnormal)!
        //
        // THIS IS BUG 1! The code flushes to zero when it should produce subnormals.

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

        // min normal * 0.125 = subnormal
        // 0x0400 (min normal) * 0x3400 (0.125):
        // eff_ea = 1, eff_eb = 13
        // mant_a = {1, 0} = 0x400, mant_b = {1, 1000000000} = 0x600
        // prod = 0x400 × 0x600 = 1024 × 1536 = 1,572,864 = 0x180000
        // lzc = 1 (bit[20]=1), RS = 10
        // 0x180000 >> 10 = 1536 = 0x600
        // mant_norm = 0x200
        // guard = prod[10-1] = prod[9] = 0x180000 bit 9 = 0
        // biased_exp = 1 + 13 - 14 - 1 = -1
        // UNDERFLOW → currently flush to zero
        // Correct: gradual underflow → subnormal
        // With biased_exp = -1, shift right by 2 more from rs5:
        // rs5 = 0x600, >> 2 = 0x180, top 10 bits = 0x180 = 0x060
        // Rounding: bits shifted out = rs5[1:0] = 00, guard=0, no inc
        // Result = 0x0060 = 0x0060
        // Wait let me verify: 0x0400 * 0.3400 = min_normal × 0.125 = 2^-14 × 0.125 = 2^-17
        // Subnormal: mant × 2^-24 = 2^-17, mant = 2^-17 / 2^-24 = 2^7 = 128 = 0x80
        // Result = 0x0080
        //
        // Hmm, let me redo. 0x3400 = 0.125 = 2^-3
        // FP16: exp=13, mant=0. 2^(13-15) = 2^-2. But 2^-2 = 0.25, not 0.125.
        // 0.125 = 2^-3. FP16: exp = 13-15+15 = 13? No, FP16 exp field = 12.
        // value = 2^(12-15) × 1.0 = 2^-3 = 0.125. exp=12, mant=0.
        // 0x3400 = 0_01101_0000_0000_000. S=0, E=01101=13, M=0000000000.
        // value = 2^(13-15) × 1.0 = 2^-3 = 0.125 ✓
        //
        // 0x0400 = 2^-14 (min normal). S=0, E=00001=1, M=0.
        // Product = 2^-14 × 2^-3 = 2^-17
        // This is below FP16 minimum normal (2^-14), so it's subnormal.
        // Subnormal: mant × 2^-24 = 2^-17
        // mant = 2^(-17-(-24)) = 2^7 = 128 = 0x80
        // Result = 0x0080
        //
        // Let me verify with the code path:
        // eff_ea = 1, eff_eb = 13
        // mant_a = {0, 0} ... wait, 0x0400 has exp=1, mant=0000000000
        // For exp != 0: mant_a = {1, mant} = {1, 0000000000} = 0x400
        // mant_b = {1, 0000000000} = 0x400 (0x3400 has exp=13, mant=0)
        // prod = 0x400 × 0x400 = 0x100000 (1,048,576)
        // lzc = 1 (bit[20]=1), RS = 10
        // 0x100000 >> 10 = 0x400 = 1024
        // mant_norm = 0, guard = prod[9] = 0
        // biased_exp = 1 + 13 - 14 - 1 = -1
        // UNDERFLOW → currently flushes to zero
        // Correct: gradual underflow
        // biased_exp = -1, extra_shift = 1 - (-1) = 2
        // rs5 = 0x400 = 1_0000_0000_00
        // >> 2 = 0x100 = 1_0000_0000 (in 11 bits: 0_0100_0000_00)
        // Wait, 0x400 >> 2 = 256 = 0x100
        // bit[8] = 1, rest = 0
        // Top 10 bits of the 22-bit shifted value: 0x100[9:0] = 0x100
        // Hmm wait, 0x400 = 1_0000_0000_00 in 22 bits: 00_0001_0000_0000_0000_0000
        // >> 2: 00_0000_0100_0000_0000_0000
        // Top 10 bits: 00_0100_0000 = 0x040
        // Hmm that gives mant = 0x040 = 64
        // value = 64 × 2^-24 = 2^6 × 2^-24 = 2^-18 ≈ 3.8e-6
        // But correct = 2^-17 = 7.6e-6
        //
        // Something off. Let me redo:
        // 0x400 >> 2 as 22-bit: 00_0001_0000_0000_0000_0000 → 00_0000_0100_0000_0000_0000
        // Top 10 bits starting from where?
        // After RS=10, rs5 = 0x400 = 00_0001_0000_0000_0000_0000
        // For normal: mant = rs5[9:0] = 0
        // For subnormal with biased_exp=-1: extra_shift = 1 - (-1) = 2
        // Shifted more: 0x400 >> 2 = 256 = 0x100
        // In 22-bit: 00_0000_0100_0000_0000_0000
        // Top 10 bits: 00_0100_0000 = 0x040
        // Wait, that's bits [9:0] = 0001000000 = 0x040? Let me be precise.
        // 256 in 22-bit binary: 00_0000_0100_0000_0000_0000
        // bits[9:0] = 0100000000 = 0x100
        // OH! bit[8] = 1, and bits [9] = 0.
        // So mant = 0100000000 = 0x100 = 256
        // value = 256 × 2^-24 = 2^-24 × 256 = 2^-24 × 2^8 = 2^-16
        // But correct = 2^-17!
        //
        // Hmm, 2^8 × 2^-24 = 2^-16, but we need 2^-17.
        // The issue: biased_exp = -1, but the extra shift should be 1 - (-1) = 2.
        // After shifting 2 more: the "implicit 1" disappears (shifted past bit 10),
        // and the value decreases by 2^2 = 4. So:
        // Normal value = 2^(0-15) × 1.0 = 2^-15
        // After extra shift by 2: 2^-15 / 4 = 2^-17 ✓
        // mant (subnormal) = bits [9:0] after shifting 2 more = 256 >> (10-2)?
        //
        // OK I think I need to be more careful about how subnormal output is computed.
        // The standard approach:
        // 1. Compute result as if normal (exp E, mantissa M)
        // 2. If E < 1: subnormal. Right-shift M right by (1-E) positions
        //    to get the subnormal mantissa. Round the shifted-out bits.
        //    Result: exp = 0, mant = shifted M (top 10 bits after shift)
        //
        // In our code, E (biased_exp) = -1, M (11-bit with implicit 1) = rs5[10:0] = 0x400.
        // Extra shift = 1 - (-1) = 2.
        // Shift M right by 2: 0x400 >> 2 = 0x100
        // But M has 11 bits. We need the subnormal mantissa (10 bits).
        // After >>2: 0x100 in 11 bits = 0_0100_0000_00 = 256
        // Top 10 bits: 00_1000_0000 = 0x040 = 64
        //
        // Hmm, 256 in binary (11 bits): 0_100_0000_00
        // bits [10:0] = 0_100_0000_00
        // bits [9:0] = 00_100_0000_00? No, bits [9:0] is 10 bits:
        // bit[9] = 0, bit[8] = 1, rest = 0
        // = 0100000000 = 256? No, 0100000000 binary = 256.
        //
        // Wait, 0x100 = 256. In binary: 1_0000_0000. That's only 9 bits.
        // As 11 bits: 0_0100_0000_00
        // bits[9:0] = 0_100_0000_00 = 00_10000000
        //
        // I'm going in circles. Let me just use concrete bit manipulation:
        // rs5 = 0x400 = 1_0000_0000_00 (in 22 bits: 00_0001_0000_0000_0000_0000)
        // For normal: value = 2^(biased_exp - 15) × rs5 / 2^10
        //                = 2^(-1-15) × 1024 / 1024 = 2^-16
        //
        // For subnormal with biased_exp = -1: we need to denormalize.
        // Denormalized value = 2^(0-15) × (M_shifted / 2^10)
        // where M_shifted = M >> (1 - biased_exp) = 0x400 >> 2 = 0x100
        // But 0x100 = 256 in 11 bits? No, 0x100 = 256 as integer, but as
        // a fraction, it's 256/1024 of the "mantissa space".
        //
        // Actually, for subnormal result:
        // value = M_shifted × 2^(-14 - 10) = M_shifted × 2^(-24)
        // where M_shifted is the top 10 bits after denormalization shift
        //
        // The denormalization: we want the value
        // 2^(biased_exp - 15) × (M / 2^10) = 2^(-14) × (M_sub / 2^10)
        // So: M_sub / 2^10 = 2^(biased_exp - 15) × M / 2^10 × 2^(15-14)
        //    M_sub = M × 2^(biased_exp - 14)
        //    For biased_exp = -1: M_sub = M × 2^(-15) = M / 32768
        //    This gives a fraction, not an integer. So we need rounding.
        //
        // Standard approach: shift M right by (1 - biased_exp) = 2 positions,
        // keeping the top 10 bits and rounding.
        //
        // M = 0x400 = 1024 (11-bit). Shift right by 2:
        // 1024 >> 2 = 256 = 0x100
        // But 256 is 9 bits. We need 10 bits for subnormal mantissa.
        // In 10-bit representation: 0100000000 = 256
        //
        // Wait, 256 < 1024, so it fits in 10 bits.
        // 256 = 0x100, but in 10 bits it's just 0100000000.
        //
        // value = 256 × 2^(-24) = 2^8 × 2^(-24) = 2^(-16)
        // But correct = 2^(-17).
        //
        // The issue: we shifted by (1 - biased_exp) = 2, but the "implicit 1"
        // at bit[10] means we're starting from 2^(biased_exp - 15) × (M / 2^10).
        // When we shift M right by 2, we divide M by 4, so the value becomes
        // 2^(biased_exp - 15) × (M/4) / 2^10 = 2^(biased_exp - 15 - 2) × M / 2^10
        // But for subnormal: value = M_sub × 2^(-24)
        // = (M >> 2) × 2^(-24) ... no, that's the shifted M.
        //
        // I think the correct formula is:
        // After normalizing to get M (11-bit with implicit 1 at bit 10) and biased_exp:
        // If biased_exp >= 1: normal, pack {sign, biased_exp[4:0], M[9:0]}
        // If biased_exp == 0: subnormal, need to include implicit 1
        //   sub_mant = {M[10], M[9:1]} >> (1 - 0) = M[10:1] (just include bit 10)
        //   Wait, if biased_exp = 0, the value is 2^(0-15) × (1 + M[9:0]/1024) = 2^-15 × 1.xxx
        //   This IS a subnormal. For subnormal: value = 2^(-14) × mant_sub/1024
        //   2^-15 × (1 + M/1024) = 2^-14 × ((1 + M/1024) / 2) = 2^-14 × ((1024 + M) / 2048)
        //   mant_sub = (1024 + M) / 2 (with rounding)
        //   If M[0] = 0 and bits shifted out during /2 are clean: mant_sub = (1024 + M) >> 1
        //
        // For M = 0x400: mant_sub = (1024 + 1024) >> 1 = 2048 >> 1 = 1024 = 0x400
        // value = 1024 × 2^-24 = 2^-14 = min normal
        // But we expected 0x0080 (2^-17)!
        //
        // I'm confusing the cases. Let me be very specific:
        //
        // Test case: 0x0400 × 0x3400 = 2^-14 × 2^-3 = 2^-17
        // In FP16: subnormal, mant = 2^(-17-(-24)) = 2^7 = 128 = 0x80
        // Result = 0x0080
        //
        // Code computation:
        // eff_ea = 1, eff_eb = 13
        // mant_a = {1, 0} = 0x400, mant_b = {1, 0} = 0x400
        // prod = 0x400 × 0x400 = 1,048,576 = 0x100000
        // lzc = 1, RS = 10
        // rs5 = 0x100000 >> 10 = 0x400
        // mant_norm = 0x400[9:0] = 0
        // biased_exp = 1 + 13 - 14 - 1 = -1
        //
        // Now for denormalization:
        // Current code: biased_exp[6] = 1 → underflow → flush to zero.
        // Correct: denormalize.
        //
        // The value before denormalization:
        // 2^(biased_exp - 15) × (rs5 / 2^10)
        // = 2^(-1-15) × 1024 / 1024 = 2^-16
        //
        // But the TRUE product value is 2^-17.
        // 2^-16 ≠ 2^-17. Off by factor of 2!
        //
        // Wait, this can't be right. Let me verify the formula.
        // Product = 2^(eff_ea - bias) × mant_a × 2^(eff_eb - bias) × mant_b
        // = 2^(1-15) × 1024 × 2^(13-15) × 1024
        // = 2^(-14) × 1024 × 2^(-2) × 1024
        // = 2^(-16) × 1024 × 1024
        // = 2^(-16) × 1048576
        // = 2^(-16) × 2^20
        // = 2^4
        // = 16
        //
        // That's NOT 2^-17! Let me re-examine.
        //
        // 0x0400 = min normal = 2^-14
        // mant_a for 0x0400: exp=1, mant=0. For exp != 0: mant_a = {1, mant} = {1, 0} = 0x400
        // The "value" of mant_a = 1.0 (since implicit 1 at bit 10, mantissa all zeros)
        // So mant_a represents 1.0.
        //
        // 0x3400: exp=13, mant=0. mant_b = {1, 0} = 0x400. Value of mant_b = 1.0.
        //
        // Product of significands = 1.0 × 1.0 = 1.0
        // Product = 2^(1-15) × 2^(13-15) × 1.0 × 1.0
        //        = 2^-14 × 2^-2
        //        = 2^-16
        //
        // But 2^-14 × 2^-3 = 2^-17 ≠ 2^-16!
        //
        // AH! I think the issue is that 0x3400 does NOT represent 2^-3!
        // Let me verify: 0x3400 = 0_01101_0000_0000_000
        // S=0, E=01101=13, M=0000000000
        // value = 2^(13-15) × (1 + 0) = 2^-2 = 0.25
        //
        // 0.25 = 2^-2, NOT 2^-3! So 0x3400 = 0.25.
        // Then 2^-14 × 0.25 = 2^-14 × 2^-2 = 2^-16.
        //
        // And 0x3400 as 2^-2 means 0x0400 × 0x3400 = 2^-14 × 2^-2 = 2^-16.
        // This IS subnormal (since 2^-16 < 2^-14).
        // Subnormal: mant × 2^-24 = 2^-16 → mant = 2^8 = 256 = 0x100
        // Result = 0x0100
        //
        // So the expected result is 0x0100, NOT 0x0080!
        // I made an arithmetic error earlier. Let me fix.

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
