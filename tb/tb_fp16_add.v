`timescale 1ns/1ps
module tb_fp16_add_debug;
reg [15:0] a, b;
wire [15:0] result;

fp16_add uut (.a(a), .b(b), .result(result));

integer pass_cnt = 0, fail_cnt = 0;

task check;
    input [15:0] got, exp;
    input integer id;
    begin
        if (got === exp) begin
            $display("[PASS] Test %0d: a=0x%04X b=0x%04X  expected=0x%04X  got=0x%04X", id, a, b, exp, got);
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("[FAIL] Test %0d: a=0x%04X b=0x%04X  expected=0x%04X  got=0x%04X", id, a, b, exp, got);
            fail_cnt = fail_cnt + 1;
        end
    end
endtask

initial begin
    // Test 1: 1.0 + 1.0 = 2.0 (same exp, same mant, addition)
    a = 16'h3C00; b = 16'h3C00; #10;
    check(result, 16'h4000, 1);

    // Test 2: 2.0 + 3.0 = 5.0 (same exp, diff mant, addition)
    a = 16'h4000; b = 16'h4200; #10;
    check(result, 16'h4500, 2);

    // Test 3: 1.5 + 1.5 = 3.0 (same exp, same mant, addition with carry)
    a = 16'h3E00; b = 16'h3E00; #10;
    check(result, 16'h4200, 3);

    // Test 4: 1.0 + (-1.0) = 0.0 (same exp, subtract to zero)
    a = 16'h3C00; b = 16'hBC00; #10;
    check(result, 16'h0000, 4);

    // Test 5: 2.0 + (-1.0) = 1.0 (same exp, subtract)
    a = 16'h4000; b = 16'hBC00; #10;
    check(result, 16'h3C00, 5);

    // Test 6: -2.0 + 1.0 = -1.0 (same exp, subtract, negative result)
    a = 16'hC000; b = 16'h3C00; #10;
    check(result, 16'hBC00, 6);

    // Test 7: 3.0 + 1.0 = 4.0 (same exp, add, carry)
    a = 16'h4200; b = 16'h3C00; #10;
    check(result, 16'h4400, 7);

    // Test 8: 10.0 + 0.015625 (big exp diff)
    a = 16'h4900; b = 16'h2400; #10;
    check(result, 16'h4902, 8);

    // Test 9: 4.0 + (-2.0) = 2.0 (exp diff=1, subtract)
    a = 16'h4400; b = 16'hC000; #10;
    check(result, 16'h4000, 9);

    // Test 10: 3.140625 + 1.234375 = 4.375
    a = 16'h4248; b = 16'h3CF0; #10;
    check(result, 16'h4460, 10);

    // Test 11: 4.375 + (-0.875) = 3.5
    a = 16'h4460; b = 16'hBB00; #10;
    check(result, 16'h4300, 11);

    // Test 12: 3.5 + 0.109375 = 3.609375
    a = 16'h4300; b = 16'h2F00; #10;
    check(result, 16'h4338, 12);

    // Test 13: Inf + (-Inf) = NaN (quiet, sign is implementation-defined)
    a = 16'h7C00; b = 16'hFC00; #10;
    check(result, 16'hFE00, 13);

    // Test 14: NaN + 1.0 = NaN
    a = 16'h7E00; b = 16'h3C00; #10;
    check(result, 16'h7E00, 14);

    // Test 15: 0.0 + 0.0 = 0.0
    a = 16'h0000; b = 16'h0000; #10;
    check(result, 16'h0000, 15);

    // Test 16: -0.0 + (-0.0) = +0.0
    a = 16'h8000; b = 16'h8000; #10;
    check(result, 16'h0000, 16);

    // Test 17: 1.0 + (-0.0) = 1.0 (not -0)
    a = 16'h3C00; b = 16'h8000; #10;
    check(result, 16'h3C00, 17);

    // Test 18: Subnormal: smallest normal (0x0400 = 2^-14) + 2^-14 = 2^-13 (0x0800)
    a = 16'h0400; b = 16'h0400; #10;
    check(result, 16'h0800, 18);

    // Test 19: 1.0 + 2.0 + (-2.0) sequential test
    // 1.0 + 2.0 = 3.0
    a = 16'h3C00; b = 16'h4000; #10;
    check(result, 16'h4200, 19);

    // Test 20: 2.0*1.0 + 2.0*1.0 = 4.0 (same product added twice)
    // Simulating PE: fp16_mul(2.0,1.0)=0x4000, then 0x4000+0x4000
    a = 16'h4000; b = 16'h4000; #10;
    check(result, 16'h4400, 20);

    $display("\n=== Summary: PASS=%0d  FAIL=%0d ===", pass_cnt, fail_cnt);
    $finish;
end
endmodule
