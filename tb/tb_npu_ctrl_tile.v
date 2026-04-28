`timescale 1ns/1ps

module tb_npu_ctrl_tile;
    reg clk;
    reg rst_n;
    reg [31:0] ctrl_reg;
    reg [31:0] m_dim;
    reg [31:0] n_dim;
    reg [31:0] k_dim;
    reg [31:0] w_addr;
    reg [31:0] a_addr;
    reg [31:0] r_addr;
    reg [7:0]  arr_cfg;
    reg [1:0]  cfg_shape_in;
    reg dma_w_done;
    reg dma_a_done;
    reg dma_r_done;

    wire [1:0] cfg_shape_latched;
    wire tile_mode;
    wire vec_consume;
    wire [31:0] tile_m_base;
    wire [31:0] tile_n_base;
    wire [3:0] tile_row_valid;
    wire [3:0] tile_col_valid;
    wire [2:0] tile_active_rows;
    wire [2:0] tile_active_cols;
    wire [15:0] tile_k_cycle;
    wire busy;
    wire done;
    wire dma_w_start;
    wire [31:0] dma_w_addr;
    wire [15:0] dma_w_len;
    wire dma_a_start;
    wire [31:0] dma_a_addr;
    wire [15:0] dma_a_len;
    wire dma_r_start;
    wire [31:0] dma_r_addr;
    wire [15:0] dma_r_len;
    wire pe_en;
    wire pe_flush;
    wire pe_mode;
    wire pe_stat;
    wire pe_load_w;
    wire pe_swap_w;
    wire w_ppb_swap;
    wire a_ppb_swap;
    wire w_ppb_clear;
    wire a_ppb_clear;
    wire r_fifo_clear;
    wire irq;

    integer errors;
    integer wb_idx;
    integer vec_count;
    reg dma_r_start_d;

    npu_ctrl #(
        .ROWS(16),
        .COLS(16),
        .DATA_W(16),
        .ACC_W(32)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .ctrl_reg(ctrl_reg),
        .m_dim(m_dim),
        .n_dim(n_dim),
        .k_dim(k_dim),
        .w_addr(w_addr),
        .a_addr(a_addr),
        .r_addr(r_addr),
        .arr_cfg(arr_cfg),
        .cfg_shape_in(cfg_shape_in),
        .cfg_shape_latched(cfg_shape_latched),
        .tile_mode(tile_mode),
        .vec_consume(vec_consume),
        .tile_m_base(tile_m_base),
        .tile_n_base(tile_n_base),
        .tile_row_valid(tile_row_valid),
        .tile_col_valid(tile_col_valid),
        .tile_active_rows(tile_active_rows),
        .tile_active_cols(tile_active_cols),
        .tile_k_cycle(tile_k_cycle),
        .busy(busy),
        .done(done),
        .dma_w_start(dma_w_start),
        .dma_w_done(dma_w_done),
        .dma_w_addr(dma_w_addr),
        .dma_w_len(dma_w_len),
        .dma_a_start(dma_a_start),
        .dma_a_done(dma_a_done),
        .dma_a_addr(dma_a_addr),
        .dma_a_len(dma_a_len),
        .dma_r_start(dma_r_start),
        .dma_r_done(dma_r_done),
        .dma_r_addr(dma_r_addr),
        .dma_r_len(dma_r_len),
        .pe_en(pe_en),
        .pe_flush(pe_flush),
        .pe_mode(pe_mode),
        .pe_stat(pe_stat),
        .pe_load_w(pe_load_w),
        .pe_swap_w(pe_swap_w),
        .w_ppb_ready(1'b1),
        .w_ppb_empty(1'b0),
        .a_ppb_ready(1'b1),
        .a_ppb_empty(1'b0),
        .w_ppb_swap(w_ppb_swap),
        .a_ppb_swap(a_ppb_swap),
        .w_ppb_clear(w_ppb_clear),
        .a_ppb_clear(a_ppb_clear),
        .r_fifo_clear(r_fifo_clear),
        .irq(irq)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (!rst_n) begin
            dma_w_done <= 1'b0;
            dma_a_done <= 1'b0;
            dma_r_done <= 1'b0;
        end else begin
            dma_w_done <= dma_w_start;
            dma_a_done <= dma_a_start;
            dma_r_done <= dma_r_start;
        end
    end

    task expect_wb;
        input integer idx;
        input [31:0] exp_m;
        input [31:0] exp_n;
        input [3:0] exp_rmask;
        input [3:0] exp_cmask;
        input [2:0] exp_rows;
        input [2:0] exp_cols;
        input [31:0] exp_r_addr;
        input [15:0] exp_r_len;
        begin
            if (tile_m_base !== exp_m || tile_n_base !== exp_n ||
                tile_row_valid !== exp_rmask || tile_col_valid !== exp_cmask ||
                tile_active_rows !== exp_rows || tile_active_cols !== exp_cols ||
                dma_r_addr !== exp_r_addr || dma_r_len !== exp_r_len) begin
                $display("[FAIL] wb%0d got m=%0d n=%0d rmask=%b cmask=%b rows=%0d cols=%0d r_addr=%08h r_len=%0d",
                         idx, tile_m_base, tile_n_base, tile_row_valid,
                         tile_col_valid, tile_active_rows, tile_active_cols,
                         dma_r_addr, dma_r_len);
                $display("       expected m=%0d n=%0d rmask=%b cmask=%b rows=%0d cols=%0d r_addr=%08h r_len=%0d",
                         exp_m, exp_n, exp_rmask, exp_cmask, exp_rows, exp_cols,
                         exp_r_addr, exp_r_len);
                errors = errors + 1;
            end else begin
                $display("[PASS] wb%0d m=%0d n=%0d r_addr=%08h r_len=%0d",
                         idx, exp_m, exp_n, exp_r_addr, exp_r_len);
            end
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            dma_r_start_d <= 1'b0;
            wb_idx <= 0;
        end else begin
            dma_r_start_d <= dma_r_start;
            if (vec_consume)
                vec_count <= vec_count + 1;
            if (dma_r_start && !dma_r_start_d) begin
                case (wb_idx)
                    0: expect_wb(0, 32'd0, 32'd0, 4'b1111, 4'b1111, 3'd4, 3'd4, 32'h3000, 16'd16);
                    1: expect_wb(1, 32'd0, 32'd0, 4'b1111, 4'b1111, 3'd4, 3'd4, 32'h3018, 16'd16);
                    2: expect_wb(2, 32'd0, 32'd0, 4'b1111, 4'b1111, 3'd4, 3'd4, 32'h3030, 16'd16);
                    3: expect_wb(3, 32'd0, 32'd0, 4'b1111, 4'b1111, 3'd4, 3'd4, 32'h3048, 16'd16);
                    4: expect_wb(4, 32'd0, 32'd4, 4'b1111, 4'b0011, 3'd4, 3'd2, 32'h3010, 16'd8);
                    5: expect_wb(5, 32'd0, 32'd4, 4'b1111, 4'b0011, 3'd4, 3'd2, 32'h3028, 16'd8);
                    6: expect_wb(6, 32'd0, 32'd4, 4'b1111, 4'b0011, 3'd4, 3'd2, 32'h3040, 16'd8);
                    7: expect_wb(7, 32'd0, 32'd4, 4'b1111, 4'b0011, 3'd4, 3'd2, 32'h3058, 16'd8);
                    8: expect_wb(8, 32'd4, 32'd0, 4'b0001, 4'b1111, 3'd1, 3'd4, 32'h3060, 16'd16);
                    9: expect_wb(9, 32'd4, 32'd4, 4'b0001, 4'b0011, 3'd1, 3'd2, 32'h3070, 16'd8);
                    default: begin
                        $display("[FAIL] unexpected writeback tile index %0d", wb_idx);
                        errors = errors + 1;
                    end
                endcase
                if (dma_w_len !== 16'd8 || dma_a_len !== 16'd8) begin
                    $display("[FAIL] vector tile len mismatch w_len=%0d a_len=%0d", dma_w_len, dma_a_len);
                    errors = errors + 1;
                end
                wb_idx <= wb_idx + 1;
            end
        end
    end

    initial begin
        errors = 0;
        wb_idx = 0;
        vec_count = 0;
        dma_r_start_d = 1'b0;
        rst_n = 1'b0;
        ctrl_reg = 32'd0;
        m_dim = 32'd5;
        n_dim = 32'd6;
        k_dim = 32'd2;
        w_addr = 32'h1000;
        a_addr = 32'h2000;
        r_addr = 32'h3000;
        arr_cfg = 8'h80;       // bit7 enables 4x4 tile mode
        cfg_shape_in = 2'b00;  // 4x4 shape
        dma_w_done = 1'b0;
        dma_a_done = 1'b0;
        dma_r_done = 1'b0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        ctrl_reg = 32'h11; // start, INT8, OS
        @(posedge clk);

        wait(done);
        @(posedge clk);

        if (!tile_mode) begin
            $display("[FAIL] tile_mode did not latch");
            errors = errors + 1;
        end
        if (wb_idx !== 10) begin
            $display("[FAIL] expected 10 row writeback bursts, got %0d", wb_idx);
            errors = errors + 1;
        end
        if (vec_count !== 8) begin
            $display("[FAIL] expected 8 vec_consume pulses, got %0d", vec_count);
            errors = errors + 1;
        end

        ctrl_reg = 32'd0;
        @(posedge clk);

        if (errors == 0) begin
            $display("[PASS] tb_npu_ctrl_tile");
        end else begin
            $display("[FAIL] tb_npu_ctrl_tile errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
