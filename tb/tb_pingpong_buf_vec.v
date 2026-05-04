`timescale 1ns/1ps

module tb_pingpong_buf_vec;
    reg clk;
    reg rst_n;
    reg wr_en;
    reg [31:0] wr_data;
    reg rd_en;
    reg rd_vec_en;
    reg [4:0] rd_vec_lanes;
    reg swap;
    reg clear;
    reg fp16_mode;

    wire [15:0] rd_data;
    wire [255:0] rd_vec;
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
        .OUT_WIDTH (16),
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
        .fp16_mode   (fp16_mode),
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

    task expect_vec;
        input [15:0] l0;
        input [15:0] l1;
        input [15:0] l2;
        input [15:0] l3;
        input [127:0] name;
        begin
            if (!rd_vec_valid) begin
                $display("[FAIL] %0s: rd_vec_valid is low", name);
                errors = errors + 1;
            end
            if (rd_vec[15:0] !== l0 || rd_vec[31:16] !== l1 ||
                rd_vec[47:32] !== l2 || rd_vec[63:48] !== l3) begin
                $display("[FAIL] %0s: got %h_%h_%h_%h expected %h_%h_%h_%h",
                         name,
                         rd_vec[63:48], rd_vec[47:32], rd_vec[31:16], rd_vec[15:0],
                         l3, l2, l1, l0);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s", name);
            end
        end
    endtask

    task expect_vec_seq;
        input integer lanes;
        input integer start_value;
        input [127:0] name;
        integer li;
        reg [15:0] exp_lane;
        begin
            if (!rd_vec_valid) begin
                $display("[FAIL] %0s: rd_vec_valid is low", name);
                errors = errors + 1;
            end
            for (li = 0; li < lanes; li = li + 1) begin
                exp_lane = (start_value + li) & 16'hFFFF;
                if (rd_vec[li*16 +: 16] !== exp_lane) begin
                    $display("[FAIL] %0s lane%0d: got 0x%04h expected 0x%04h",
                             name, li, rd_vec[li*16 +: 16], exp_lane);
                    errors = errors + 1;
                end
            end
            if (errors == 0)
                $display("[PASS] %0s", name);
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
        fp16_mode = 1'b0;

        repeat (3) tick();
        rst_n = 1'b1;
        tick();

        // INT8: word = {lane3,lane2,lane1,lane0}, sign-extended to 16 bits.
        write_word(32'h04FD02FF); // lanes: -1, 2, -3, 4
        write_word(32'h807F0100); // lanes: 0, 1, 127, -128
        pulse_swap();

        expect_vec(16'hFFFF, 16'h0002, 16'hFFFD, 16'h0004, "INT8_VEC0");
        consume_vec();
        expect_vec(16'h0000, 16'h0001, 16'h007F, 16'hFF80, "INT8_VEC1");
        consume_vec();
        if (!buf_empty) begin
            $display("[FAIL] INT8 consume: buffer not empty after two vector reads");
            errors = errors + 1;
        end

        pulse_clear();

        // FP16: two 32-bit words produce four 16-bit lanes.
        fp16_mode = 1'b1;
        write_word(32'h40003C00); // lanes: 0x3C00, 0x4000
        write_word(32'h0000C000); // lanes: 0xC000, 0x0000
        pulse_swap();

        expect_vec(16'h3C00, 16'h4000, 16'hC000, 16'h0000, "FP16_VEC0");
        consume_vec();
        if (!buf_empty) begin
            $display("[FAIL] FP16 consume: buffer not empty after one vector read");
            errors = errors + 1;
        end

        pulse_clear();

        // INT8: consume eight lanes across two 32-bit words.
        fp16_mode = 1'b0;
        rd_vec_lanes = 5'd8;
        write_word(32'h04030201);
        write_word(32'h08070605);
        pulse_swap();

        expect_vec_seq(8, 1, "INT8_VEC8");
        consume_vec();
        if (!buf_empty) begin
            $display("[FAIL] INT8_VEC8 consume: buffer not empty");
            errors = errors + 1;
        end

        pulse_clear();

        // INT8: consume sixteen lanes across four 32-bit words.
        fp16_mode = 1'b0;
        rd_vec_lanes = 5'd16;
        write_word(32'h04030201);
        write_word(32'h08070605);
        write_word(32'h0C0B0A09);
        write_word(32'h100F0E0D);
        pulse_swap();

        expect_vec_seq(16, 1, "INT8_VEC16");
        consume_vec();
        if (!buf_empty) begin
            $display("[FAIL] INT8_VEC16 consume: buffer not empty");
            errors = errors + 1;
        end

        pulse_clear();

        // FP16: consume sixteen lanes across eight 32-bit words.
        fp16_mode = 1'b1;
        rd_vec_lanes = 5'd16;
        write_word(32'h20012000);
        write_word(32'h20032002);
        write_word(32'h20052004);
        write_word(32'h20072006);
        write_word(32'h20092008);
        write_word(32'h200B200A);
        write_word(32'h200D200C);
        write_word(32'h200F200E);
        pulse_swap();

        expect_vec_seq(16, 16'h2000, "FP16_VEC16");
        consume_vec();
        if (!buf_empty) begin
            $display("[FAIL] FP16_VEC16 consume: buffer not empty");
            errors = errors + 1;
        end

        if (errors == 0) begin
            $display("[PASS] tb_pingpong_buf_vec");
        end else begin
            $display("[FAIL] tb_pingpong_buf_vec errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
