`timescale 1ns/1ps

module tb_pingpong_buf_vec32;
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

    wire [31:0] rd_data;
    wire [511:0] rd_vec;
    wire rd_vec_valid;
    wire buf_empty;
    wire buf_full;
    wire buf_ready;
    wire [5:0] rd_fill;
    wire [3:0] wr_fill;

    integer errors;

    pingpong_buf #(
        .DATA_W    (32),
        .DEPTH     (8),
        .OUT_WIDTH (32),
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
        input [31:0] expected;
        begin
            if (rd_vec[lane*32 +: 32] !== expected) begin
                $display("[FAIL] lane%0d got=0x%08h expected=0x%08h",
                         lane, rd_vec[lane*32 +: 32], expected);
                errors = errors + 1;
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

        write_word(32'h04030201); // k0 lanes 0..3
        write_word(32'h08070605); // k1 lanes 0..3
        write_word(32'h0C0B0A09); // k2 lanes 0..3
        write_word(32'h100F0E0D); // k3 lanes 0..3
        pulse_swap();

        if (!rd_vec_valid) begin
            $display("[FAIL] rd_vec_valid is low");
            errors = errors + 1;
        end
        expect_lane(0, 32'h0D09_0501);
        expect_lane(1, 32'h0E0A_0602);
        expect_lane(2, 32'h0F0B_0703);
        expect_lane(3, 32'h100C_0804);
        consume_vec();

        if (!buf_empty) begin
            $display("[FAIL] buffer not empty after packed vec32 consume");
            errors = errors + 1;
        end

        if (errors == 0)
            $display("[PASS] tb_pingpong_buf_vec32");
        else begin
            $display("[FAIL] tb_pingpong_buf_vec32 errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
