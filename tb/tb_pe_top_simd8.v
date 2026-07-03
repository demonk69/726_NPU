`timescale 1ns/1ps

module tb_pe_top_simd8;
    localparam DATA_W = 64;
    localparam ACC_W  = 32;

    reg clk;
    reg rst_n;
    reg mode;
    reg stat_mode;
    reg en;
    reg flush;
    reg load_w;
    reg swap_w;
    reg acc_init_en;
    reg  [DATA_W-1:0] w_in;
    reg  [DATA_W-1:0] a_in;
    reg  [ACC_W-1:0]  acc_in;
    reg  [ACC_W-1:0]  acc_init;
    wire [ACC_W-1:0]  acc_out;
    wire              valid_out;

    integer errors;

    pe_top #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .INT8_SIMD_LANES(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mode(mode),
        .stat_mode(stat_mode),
        .en(en),
        .flush(flush),
        .load_w(load_w),
        .swap_w(swap_w),
        .acc_init_en(acc_init_en),
        .w_in(w_in),
        .a_in(a_in),
        .acc_in(acc_in),
        .acc_init(acc_init),
        .acc_out(acc_out),
        .valid_out(valid_out)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

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

    task reset_dut;
        begin
            rst_n = 1'b0;
            mode = 1'b0;
            stat_mode = 1'b0;
            en = 1'b0;
            flush = 1'b0;
            load_w = 1'b0;
            swap_w = 1'b0;
            acc_init_en = 1'b0;
            w_in = {DATA_W{1'b0}};
            a_in = {DATA_W{1'b0}};
            acc_in = {ACC_W{1'b0}};
            acc_init = {ACC_W{1'b0}};
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            @(posedge clk);
            #1;
        end
    endtask

    task drive_beat;
        input [63:0] w;
        input [63:0] a;
        input        fl;
        begin
            @(posedge clk);
            #1;
            w_in = w;
            a_in = a;
            flush = fl;
            load_w = 1'b0;
            en = 1'b1;
        end
    endtask

    task check_result;
        input [31:0] expected;
        input [255:0] name;
        begin
            if (acc_out !== expected) begin
                $display("[FAIL] %0s got=0x%08h exp=0x%08h", name, acc_out, expected);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s", name);
            end
        end
    endtask

    initial begin
        errors = 0;

        reset_dut();
        mode = 1'b0;
        stat_mode = 1'b1;

        drive_beat(pack_s8x8(-8, 7, -6, 5, 4, -3, 2, 1),
                   pack_s8x8( 1,-2,  3,-4, 5,  6,-7, 8),
                   1'b0);
        drive_beat(pack_s8x8( 4,-4,  3,-3, 2, -2, 1,-1),
                   pack_s8x8(-4,-3, -2,-1, 1,  2, 3, 4),
                   1'b0);
        @(posedge clk); #1;
        en = 1'b0;
        flush = 1'b0;
        @(posedge clk); #1;
        drive_beat(64'd0, 64'd0, 1'b1);
        @(posedge clk); #1;
        en = 1'b0;
        flush = 1'b0;
        repeat (3) @(posedge clk);
        #1;
        check_result(32'hFFFF_FFB6, "SIMD8_OS_ACCUM");

        reset_dut();
        mode = 1'b0;
        stat_mode = 1'b1;
        drive_beat(64'hFFFF_FFFF_FFFF_FFFF, 64'hFFFF_FFFF_FFFF_FFFF, 1'b0);
        @(posedge clk); #1;
        en = 1'b0;
        flush = 1'b0;
        @(posedge clk); #1;
        drive_beat(64'd0, 64'd0, 1'b1);
        @(posedge clk); #1;
        en = 1'b0;
        flush = 1'b0;
        repeat (3) @(posedge clk);
        #1;
        check_result(32'd1, "SIMD8_SCALAR_COMPAT");

        if (errors == 0) begin
            $display("[PASS] tb_pe_top_simd8");
        end else begin
            $display("[FAIL] tb_pe_top_simd8 errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
