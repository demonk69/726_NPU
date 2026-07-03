`timescale 1ns/1ps

module tb_pingpong_buf_vec64;
    reg clk;
    reg rst_n;
    reg wr_en;
    reg [31:0] wr_data;
    reg rd_en;
    reg rd_vec_en;
    reg [4:0] rd_vec_lanes;
    reg swap;
    reg clear;
    reg packed_int8;

    wire [63:0] rd_data;
    wire [1023:0] rd_vec;
    wire rd_vec_valid;
    wire buf_empty;
    wire buf_full;
    wire buf_ready;
    wire [8:0] rd_fill;
    wire [6:0] wr_fill;

    integer errors;
    integer k;
    integer lane_word;

    pingpong_buf #(
        .DATA_W    (32),
        .DEPTH     (64),
        .OUT_WIDTH (64),
        .THRESHOLD (1),
        .SUBW      (4),
        .VEC_LANES (16)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .wr_en       (wr_en),
        .wr_data     (wr_data),
        .rd_en       (rd_en),
        .rd_data     (rd_data),
        .rd_vec_en   (rd_vec_en),
        .rd_vec_lanes(rd_vec_lanes),
        .rd_vec      (rd_vec),
        .rd_vec_valid(rd_vec_valid),
        .swap        (swap),
        .clear       (clear),
        .packed_int8 (packed_int8),
        .buf_empty   (buf_empty),
        .buf_full    (buf_full),
        .buf_ready   (buf_ready),
        .rd_fill     (rd_fill),
        .wr_fill     (wr_fill)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    function [63:0] exp_lane16;
        input integer lane;
        integer kk;
        reg [7:0] byte_val;
        begin
            exp_lane16 = 64'd0;
            for (kk = 0; kk < 8; kk = kk + 1) begin
                byte_val = (kk * 16 + lane + 1) & 8'hFF;
                exp_lane16[kk*8 +: 8] = byte_val;
            end
        end
    endfunction

    function [31:0] pack_word16;
        input integer kk;
        input integer lw;
        reg [7:0] b0;
        reg [7:0] b1;
        reg [7:0] b2;
        reg [7:0] b3;
        begin
            b0 = (kk * 16 + lw * 4 + 1) & 8'hFF;
            b1 = (kk * 16 + lw * 4 + 2) & 8'hFF;
            b2 = (kk * 16 + lw * 4 + 3) & 8'hFF;
            b3 = (kk * 16 + lw * 4 + 4) & 8'hFF;
            pack_word16 = {b3, b2, b1, b0};
        end
    endfunction

    task tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task write_word;
        input [31:0] data;
        begin
            wr_data = data;
            wr_en = 1'b1;
            tick();
            wr_en = 1'b0;
            wr_data = 32'd0;
            tick();
        end
    endtask

    task pulse_swap;
        begin
            swap = 1'b1;
            tick();
            swap = 1'b0;
            tick();
        end
    endtask

    task pulse_clear;
        begin
            clear = 1'b1;
            tick();
            clear = 1'b0;
            tick();
        end
    endtask

    task consume_vec;
        begin
            rd_vec_en = 1'b1;
            tick();
            rd_vec_en = 1'b0;
            tick();
        end
    endtask

    task expect_lane;
        input integer lane;
        input [63:0] expected;
        input [127:0] name;
        begin
            if (!rd_vec_valid) begin
                $display("[FAIL] %0s rd_vec_valid is low", name);
                errors = errors + 1;
            end
            if (rd_vec[lane*64 +: 64] !== expected) begin
                $display("[FAIL] %0s lane%0d got=0x%016h exp=0x%016h",
                         name, lane, rd_vec[lane*64 +: 64], expected);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s lane%0d", name, lane);
            end
        end
    endtask

    initial begin
        errors = 0;
        rst_n = 1'b0;
        wr_en = 1'b0;
        wr_data = 32'd0;
        rd_en = 1'b0;
        rd_vec_en = 1'b0;
        rd_vec_lanes = 5'd4;
        swap = 1'b0;
        clear = 1'b0;
        packed_int8 = 1'b1;

        repeat (3) tick();
        rst_n = 1'b1;
        tick();

        rd_vec_lanes = 5'd4;
        write_word(32'h04030201);
        write_word(32'h08070605);
        write_word(32'h0C0B0A09);
        write_word(32'h100F0E0D);
        write_word(32'h14131211);
        write_word(32'h18171615);
        write_word(32'h1C1B1A19);
        write_word(32'h201F1E1D);
        pulse_swap();

        expect_lane(0, 64'h1D1915110D090501, "PACKED64_VEC4");
        expect_lane(1, 64'h1E1A16120E0A0602, "PACKED64_VEC4");
        expect_lane(2, 64'h1F1B17130F0B0703, "PACKED64_VEC4");
        expect_lane(3, 64'h201C1814100C0804, "PACKED64_VEC4");
        consume_vec();
        if (!buf_empty) begin
            $display("[FAIL] PACKED64_VEC4 buffer not empty");
            errors = errors + 1;
        end

        pulse_clear();

        rd_vec_lanes = 5'd16;
        for (k = 0; k < 8; k = k + 1) begin
            for (lane_word = 0; lane_word < 4; lane_word = lane_word + 1) begin
                write_word(pack_word16(k, lane_word));
            end
        end
        pulse_swap();

        expect_lane(0,  exp_lane16(0),  "PACKED64_VEC16");
        expect_lane(7,  exp_lane16(7),  "PACKED64_VEC16");
        expect_lane(15, exp_lane16(15), "PACKED64_VEC16");
        consume_vec();
        if (!buf_empty) begin
            $display("[FAIL] PACKED64_VEC16 buffer not empty");
            errors = errors + 1;
        end

        if (errors == 0) begin
            $display("[PASS] tb_pingpong_buf_vec64");
        end else begin
            $display("[FAIL] tb_pingpong_buf_vec64 errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
