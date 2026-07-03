// =============================================================================
// Module  : tb_pe_top
// Project : NPU_prj
// Desc    : Contract tests for pe_top.
//
//           Current PE contract:
//             - OS: internal accumulator, valid result on flush.
//             - WS: true psum-chain mode, valid result is acc_in + product.
//             - load_w writes inactive/prefetch weight only.
//             - swap_w selects the active weight register.
//             - swap_w + load_w in one cycle activates the old prefetch weight
//               and writes w_in into the newly inactive register.
// =============================================================================

`timescale 1ns/1ps

module tb_pe_top;
    localparam ACC_W = 32;
    localparam CLK_PERIOD = 10;

    reg clk;

    // 16-bit INT8 PE: scalar compatibility + SIMD2 coverage.
    reg rst16, mode16, stat16, en16, flush16, load16, swap16, init_en16;
    reg [15:0] w16, a16;
    reg [ACC_W-1:0] acc16, init16;
    wire [ACC_W-1:0] out16;
    wire valid16;

    // 32-bit INT8 PE: SIMD4 coverage.
    reg rst32, mode32, stat32, en32, flush32, load32, swap32, init_en32;
    reg [31:0] w32, a32;
    reg [ACC_W-1:0] acc32, init32;
    wire [ACC_W-1:0] out32;
    wire valid32;

    // 64-bit INT8 PE: SIMD8 and full-width scalar compatibility coverage.
    reg rst64, mode64, stat64, en64, flush64, load64, swap64, init_en64;
    reg [63:0] w64, a64;
    reg [ACC_W-1:0] acc64, init64;
    wire [ACC_W-1:0] out64;
    wire valid64;

    // FP16-enabled PE.
    reg rstfp, modefp, statfp, enfp, flushfp, loadfp, swapfp, init_enfp;
    reg [15:0] wfp, afp;
    reg [ACC_W-1:0] accfp, initfp;
    wire [ACC_W-1:0] outfp;
    wire validfp;

    integer pass_cnt;
    integer fail_cnt;

    pe_top #(
        .DATA_W(16),
        .ACC_W(ACC_W),
        .INT8_SIMD_LANES(2),
        .FP16_ENABLE(0)
    ) u_pe16 (
        .clk(clk), .rst_n(rst16), .mode(mode16), .stat_mode(stat16),
        .en(en16), .flush(flush16), .load_w(load16), .swap_w(swap16),
        .acc_init_en(init_en16), .w_in(w16), .a_in(a16), .acc_in(acc16),
        .acc_init(init16), .acc_out(out16), .valid_out(valid16)
    );

    pe_top #(
        .DATA_W(32),
        .ACC_W(ACC_W),
        .INT8_SIMD_LANES(4),
        .FP16_ENABLE(0)
    ) u_pe32 (
        .clk(clk), .rst_n(rst32), .mode(mode32), .stat_mode(stat32),
        .en(en32), .flush(flush32), .load_w(load32), .swap_w(swap32),
        .acc_init_en(init_en32), .w_in(w32), .a_in(a32), .acc_in(acc32),
        .acc_init(init32), .acc_out(out32), .valid_out(valid32)
    );

    pe_top #(
        .DATA_W(64),
        .ACC_W(ACC_W),
        .INT8_SIMD_LANES(8),
        .FP16_ENABLE(0)
    ) u_pe64 (
        .clk(clk), .rst_n(rst64), .mode(mode64), .stat_mode(stat64),
        .en(en64), .flush(flush64), .load_w(load64), .swap_w(swap64),
        .acc_init_en(init_en64), .w_in(w64), .a_in(a64), .acc_in(acc64),
        .acc_init(init64), .acc_out(out64), .valid_out(valid64)
    );

    pe_top #(
        .DATA_W(16),
        .ACC_W(ACC_W),
        .INT8_SIMD_LANES(2),
        .FP16_ENABLE(1)
    ) u_pefp (
        .clk(clk), .rst_n(rstfp), .mode(modefp), .stat_mode(statfp),
        .en(enfp), .flush(flushfp), .load_w(loadfp), .swap_w(swapfp),
        .acc_init_en(init_enfp), .w_in(wfp), .a_in(afp), .acc_in(accfp),
        .acc_init(initfp), .acc_out(outfp), .valid_out(validfp)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    function [15:0] s8_scalar;
        input integer value;
        reg [7:0] v;
        begin
            v = value[7:0];
            s8_scalar = {{8{v[7]}}, v};
        end
    endfunction

    function [15:0] pack_s8x2;
        input integer l1;
        input integer l0;
        begin
            pack_s8x2 = {l1[7:0], l0[7:0]};
        end
    endfunction

    function [31:0] pack_s8x4;
        input integer l3;
        input integer l2;
        input integer l1;
        input integer l0;
        begin
            pack_s8x4 = {l3[7:0], l2[7:0], l1[7:0], l0[7:0]};
        end
    endfunction

    function [63:0] pack_s8x8;
        input integer l7;
        input integer l6;
        input integer l5;
        input integer l4;
        input integer l3;
        input integer l2;
        input integer l1;
        input integer l0;
        begin
            pack_s8x8 = {l7[7:0], l6[7:0], l5[7:0], l4[7:0],
                         l3[7:0], l2[7:0], l1[7:0], l0[7:0]};
        end
    endfunction

    task record_result;
        input match;
        input [ACC_W-1:0] got;
        input [ACC_W-1:0] expected;
        input [511:0] name;
        begin
            if (match) begin
                $display("[PASS] %0s", name);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %0s got=0x%08h exp=0x%08h", name, got, expected);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    task reset_all;
        begin
            rst16 = 1'b0; mode16 = 1'b0; stat16 = 1'b1; en16 = 1'b0; flush16 = 1'b0;
            load16 = 1'b0; swap16 = 1'b0; init_en16 = 1'b0;
            w16 = 16'd0; a16 = 16'd0; acc16 = 32'd0; init16 = 32'd0;

            rst32 = 1'b0; mode32 = 1'b0; stat32 = 1'b1; en32 = 1'b0; flush32 = 1'b0;
            load32 = 1'b0; swap32 = 1'b0; init_en32 = 1'b0;
            w32 = 32'd0; a32 = 32'd0; acc32 = 32'd0; init32 = 32'd0;

            rst64 = 1'b0; mode64 = 1'b0; stat64 = 1'b1; en64 = 1'b0; flush64 = 1'b0;
            load64 = 1'b0; swap64 = 1'b0; init_en64 = 1'b0;
            w64 = 64'd0; a64 = 64'd0; acc64 = 32'd0; init64 = 32'd0;

            rstfp = 1'b0; modefp = 1'b1; statfp = 1'b1; enfp = 1'b0; flushfp = 1'b0;
            loadfp = 1'b0; swapfp = 1'b0; init_enfp = 1'b0;
            wfp = 16'd0; afp = 16'd0; accfp = 32'd0; initfp = 32'd0;

            repeat (4) @(posedge clk);
            rst16 = 1'b1;
            rst32 = 1'b1;
            rst64 = 1'b1;
            rstfp = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task pulse16;
        input stat;
        input e;
        input fl;
        input ld;
        input sw;
        input init_en;
        input [15:0] w;
        input [15:0] a;
        input [ACC_W-1:0] acc;
        input [ACC_W-1:0] init;
        begin
            @(negedge clk);
            stat16 = stat; mode16 = 1'b0; en16 = e; flush16 = fl;
            load16 = ld; swap16 = sw; init_en16 = init_en;
            w16 = w; a16 = a; acc16 = acc; init16 = init;
            @(negedge clk);
            en16 = 1'b0; flush16 = 1'b0; load16 = 1'b0; swap16 = 1'b0; init_en16 = 1'b0;
            w16 = 16'd0; a16 = 16'd0; acc16 = 32'd0; init16 = 32'd0;
        end
    endtask

    task pulse32;
        input stat;
        input e;
        input fl;
        input ld;
        input sw;
        input [31:0] w;
        input [31:0] a;
        input [ACC_W-1:0] acc;
        begin
            @(negedge clk);
            stat32 = stat; mode32 = 1'b0; en32 = e; flush32 = fl;
            load32 = ld; swap32 = sw; init_en32 = 1'b0;
            w32 = w; a32 = a; acc32 = acc; init32 = 32'd0;
            @(negedge clk);
            en32 = 1'b0; flush32 = 1'b0; load32 = 1'b0; swap32 = 1'b0;
            w32 = 32'd0; a32 = 32'd0; acc32 = 32'd0;
        end
    endtask

    task pulse64;
        input stat;
        input e;
        input fl;
        input ld;
        input sw;
        input [63:0] w;
        input [63:0] a;
        input [ACC_W-1:0] acc;
        begin
            @(negedge clk);
            stat64 = stat; mode64 = 1'b0; en64 = e; flush64 = fl;
            load64 = ld; swap64 = sw; init_en64 = 1'b0;
            w64 = w; a64 = a; acc64 = acc; init64 = 32'd0;
            @(negedge clk);
            en64 = 1'b0; flush64 = 1'b0; load64 = 1'b0; swap64 = 1'b0;
            w64 = 64'd0; a64 = 64'd0; acc64 = 32'd0;
        end
    endtask

    task pulsefp;
        input stat;
        input e;
        input fl;
        input ld;
        input sw;
        input init_en;
        input [15:0] w;
        input [15:0] a;
        input [ACC_W-1:0] acc;
        input [ACC_W-1:0] init;
        begin
            @(negedge clk);
            statfp = stat; modefp = 1'b1; enfp = e; flushfp = fl;
            loadfp = ld; swapfp = sw; init_enfp = init_en;
            wfp = w; afp = a; accfp = acc; initfp = init;
            @(negedge clk);
            enfp = 1'b0; flushfp = 1'b0; loadfp = 1'b0; swapfp = 1'b0; init_enfp = 1'b0;
            wfp = 16'd0; afp = 16'd0; accfp = 32'd0; initfp = 32'd0;
        end
    endtask

    task expect16;
        input [ACC_W-1:0] expected;
        input [511:0] name;
        integer guard;
        reg seen;
        begin
            seen = 1'b0;
            for (guard = 0; guard < 10; guard = guard + 1) begin
                @(posedge clk); #1;
                if (valid16 && !seen) begin
                    seen = 1'b1;
                    record_result(out16 === expected, out16, expected, name);
                end
            end
            if (!seen)
                record_result(1'b0, 32'hDEAD_DEAD, expected, name);
        end
    endtask

    task expect32;
        input [ACC_W-1:0] expected;
        input [511:0] name;
        integer guard;
        reg seen;
        begin
            seen = 1'b0;
            for (guard = 0; guard < 10; guard = guard + 1) begin
                @(posedge clk); #1;
                if (valid32 && !seen) begin
                    seen = 1'b1;
                    record_result(out32 === expected, out32, expected, name);
                end
            end
            if (!seen)
                record_result(1'b0, 32'hDEAD_DEAD, expected, name);
        end
    endtask

    task expect64;
        input [ACC_W-1:0] expected;
        input [511:0] name;
        integer guard;
        reg seen;
        begin
            seen = 1'b0;
            for (guard = 0; guard < 10; guard = guard + 1) begin
                @(posedge clk); #1;
                if (valid64 && !seen) begin
                    seen = 1'b1;
                    record_result(out64 === expected, out64, expected, name);
                end
            end
            if (!seen)
                record_result(1'b0, 32'hDEAD_DEAD, expected, name);
        end
    endtask

    task expectfp;
        input [ACC_W-1:0] expected;
        input [511:0] name;
        integer guard;
        reg seen;
        begin
            seen = 1'b0;
            for (guard = 0; guard < 10; guard = guard + 1) begin
                @(posedge clk); #1;
                if (validfp && !seen) begin
                    seen = 1'b1;
                    record_result(outfp === expected, outfp, expected, name);
                end
            end
            if (!seen)
                record_result(1'b0, 32'hDEAD_DEAD, expected, name);
        end
    endtask

    initial begin
        pass_cnt = 0;
        fail_cnt = 0;
        reset_all();

        // INT8 OS: scalar compatibility, stalls, init, SIMD2/4/8.
        pulse16(1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, s8_scalar(3),  s8_scalar(4), 32'd0, 32'd0);
        pulse16(1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, s8_scalar(-2), s8_scalar(5), 32'd0, 32'd0);
        pulse16(1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 16'd0, 16'd0, 32'd0, 32'd0);
        expect16(32'd2, "INT8_OS_SCALAR_ACCUM");

        pulse16(1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 16'd0, 16'd0, 32'd0, 32'd100);
        pulse16(1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, s8_scalar(2), s8_scalar(3), 32'd0, 32'd0);
        pulse16(1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 16'd0, 16'd0, 32'd0, 32'd0);
        expect16(32'd106, "INT8_OS_ACC_INIT");

        pulse16(1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0,
                pack_s8x2(4, 3), pack_s8x2(5, 2), 32'd0, 32'd0);
        pulse16(1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0,
                pack_s8x2(-2, -1), pack_s8x2(7, 6), 32'd0, 32'd0);
        pulse16(1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 16'd0, 16'd0, 32'd0, 32'd0);
        expect16(32'd6, "INT8_OS_SIMD2_ACCUM");

        pulse32(1'b1, 1'b1, 1'b0, 1'b0, 1'b0,
                pack_s8x4(5, -4, 3, 2), pack_s8x4(-1, 6, 4, 7), 32'd0);
        pulse32(1'b1, 1'b1, 1'b0, 1'b0, 1'b0,
                pack_s8x4(1, 2, -3, 4), pack_s8x4(-5, 6, 7, -8), 32'd0);
        pulse32(1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 32'd0, 32'd0, 32'd0);
        expect32(32'hFFFF_FFCF, "INT8_OS_SIMD4_ACCUM");

        pulse64(1'b1, 1'b1, 1'b0, 1'b0, 1'b0,
                pack_s8x8(-8, 7, -6, 5, 4, -3, 2, 1),
                pack_s8x8( 1,-2,  3,-4, 5,  6,-7, 8), 32'd0);
        pulse64(1'b1, 1'b1, 1'b0, 1'b0, 1'b0,
                pack_s8x8( 4,-4,  3,-3, 2, -2, 1,-1),
                pack_s8x8(-4,-3, -2,-1, 1,  2, 3, 4), 32'd0);
        pulse64(1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 64'd0, 64'd0, 32'd0);
        expect64(32'hFFFF_FFB6, "INT8_OS_SIMD8_ACCUM");

        pulse64(1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 64'hFFFF_FFFF_FFFF_FFFF,
                64'hFFFF_FFFF_FFFF_FFFF, 32'd0);
        pulse64(1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 64'd0, 64'd0, 32'd0);
        expect64(32'd1, "INT8_OS_FULLWIDTH_SCALAR_COMPAT");

        // INT8 true WS: prefetch/swap semantics and external psum accumulation.
        pulse16(1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 1'b0, s8_scalar(3), 16'd0, 32'd0, 32'd0);
        pulse16(1'b0, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 16'd0, s8_scalar(5), 32'd7, 32'd0);
        expect16(32'd7, "WS_LOAD_NO_SWAP_ACTIVE_ZERO");

        pulse16(1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 16'd0, 16'd0, 32'd0, 32'd0);
        pulse16(1'b0, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 16'd0, s8_scalar(5), 32'd7, 32'd0);
        expect16(32'd22, "WS_SWAP_MAKES_WEIGHT_ACTIVE");

        pulse16(1'b0, 1'b1, 1'b0, 1'b1, 1'b0, 1'b0, s8_scalar(4), s8_scalar(2), 32'd1, 32'd0);
        expect16(32'd7, "WS_LOAD_DURING_COMPUTE_KEEPS_ACTIVE");

        pulse16(1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 16'd0, 16'd0, 32'd0, 32'd0);
        pulse16(1'b0, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 16'd0, s8_scalar(2), 32'd1, 32'd0);
        expect16(32'd9, "WS_SWAPPED_PREFETCH_WEIGHT_ACTIVE");

        pulse16(1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 1'b0, s8_scalar(5), 16'd0, 32'd0, 32'd0);
        pulse16(1'b0, 1'b0, 1'b0, 1'b1, 1'b1, 1'b0, s8_scalar(6), 16'd0, 32'd0, 32'd0);
        pulse16(1'b0, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 16'd0, s8_scalar(1), 32'd0, 32'd0);
        expect16(32'd5, "WS_SWAP_LOAD_OLD_PREFETCH_ACTIVE");
        pulse16(1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 16'd0, 16'd0, 32'd0, 32'd0);
        pulse16(1'b0, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 16'd0, s8_scalar(1), 32'd0, 32'd0);
        expect16(32'd6, "WS_SWAP_AFTER_SWAP_LOAD_NEW_PREFETCH");

        pulse16(1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 1'b0, pack_s8x2(4, 3), 16'd0, 32'd0, 32'd0);
        pulse16(1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 16'd0, 16'd0, 32'd0, 32'd0);
        pulse16(1'b0, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 16'd0, pack_s8x2(5, 2), 32'd10, 32'd0);
        expect16(32'd36, "WS_SIMD2_EXTERNAL_PSUM");

        pulse32(1'b0, 1'b0, 1'b0, 1'b1, 1'b0, pack_s8x4(4, -3, 2, 1), 32'd0, 32'd0);
        pulse32(1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 32'd0, 32'd0, 32'd0);
        pulse32(1'b0, 1'b1, 1'b0, 1'b0, 1'b0, 32'd0, pack_s8x4(5, 6, -7, 8), 32'd20);
        expect32(32'd16, "WS_SIMD4_EXTERNAL_PSUM");

        // FP16 OS and true WS.  Expected values are FP32 bit patterns.
        pulsefp(1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 16'h4000, 16'h3E00, 32'd0, 32'd0);
        pulsefp(1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 16'h4000, 16'h3E00, 32'd0, 32'd0);
        pulsefp(1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 16'd0, 16'd0, 32'd0, 32'd0);
        expectfp(32'h40C00000, "FP16_OS_ACCUM_6P0");

        pulsefp(1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 16'd0, 16'd0, 32'd0, 32'h3FC00000);
        pulsefp(1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 16'h4000, 16'h4000, 32'd0, 32'd0);
        pulsefp(1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 16'd0, 16'd0, 32'd0, 32'd0);
        expectfp(32'h40B00000, "FP16_OS_ACC_INIT_5P5");

        pulsefp(1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 1'b0, 16'h4000, 16'd0, 32'd0, 32'd0);
        pulsefp(1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 16'd0, 16'd0, 32'd0, 32'd0);
        pulsefp(1'b0, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 16'd0, 16'h3E00, 32'h3F800000, 32'd0);
        expectfp(32'h40800000, "FP16_WS_EXTERNAL_PSUM_4P0");

        $display("[SUMMARY] tb_pe_top PASS=%0d FAIL=%0d", pass_cnt, fail_cnt);
        if (fail_cnt == 0) begin
            $display("[PASS] tb_pe_top");
        end else begin
            $display("[FAIL] tb_pe_top failures=%0d", fail_cnt);
            $fatal;
        end
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 5000);
        $display("[FAIL] tb_pe_top timeout");
        $fatal;
    end
endmodule
