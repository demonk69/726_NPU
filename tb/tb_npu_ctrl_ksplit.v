`timescale 1ns/1ps

module tb_npu_ctrl_ksplit;
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
    wire [15:0] tile_row_valid;
    wire [15:0] tile_col_valid;
    wire [4:0] tile_active_rows;
    wire [5:0] tile_active_cols;
    wire [31:0] tile_k_base;
    wire [15:0] tile_k_len;
    wire [31:0] tile_k_index;
    wire [15:0] tile_k_cycle;
    wire busy;
    wire done;
    wire error;
    wire [31:0] err_status;
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
    integer load_idx;
    integer wb_idx;
    integer vec_count;
    integer flush_count;
    reg dma_w_start_d;
    reg dma_r_start_d;

    npu_ctrl #(
        .ROWS(16),
        .COLS(16),
        .DATA_W(16),
        .ACC_W(32),
        .PPB_DEPTH(4)
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
        .tile_k_base(tile_k_base),
        .tile_k_len(tile_k_len),
        .tile_k_index(tile_k_index),
        .tile_k_cycle(tile_k_cycle),
        .busy(busy),
        .done(done),
        .error(error),
        .err_status(err_status),
        .err_clear(1'b0),
        .err_clear_mask(32'd0),
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
        .dma_error_status(32'd0),
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
        .irq(irq),
        .compute_ce(1'b1)
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

    task expect_load;
        input integer idx;
        input [31:0] exp_w_addr;
        input [31:0] exp_a_addr;
        input [15:0] exp_len;
        begin
            if (dma_w_addr !== exp_w_addr || dma_a_addr !== exp_a_addr ||
                dma_w_len !== exp_len || dma_a_len !== exp_len) begin
                $display("[FAIL] load%0d got w=%08h a=%08h w_len=%0d a_len=%0d",
                         idx, dma_w_addr, dma_a_addr, dma_w_len, dma_a_len);
                $display("       expected w=%08h a=%08h len=%0d",
                         exp_w_addr, exp_a_addr, exp_len);
                errors = errors + 1;
            end else begin
                $display("[PASS] load%0d w=%08h a=%08h len=%0d",
                         idx, exp_w_addr, exp_a_addr, exp_len);
            end
        end
    endtask

    task expect_wb;
        input integer idx;
        input [31:0] exp_addr;
        begin
            if (dma_r_addr !== exp_addr || dma_r_len !== 16'd16 ||
                tile_k_index !== 32'd2 || tile_k_base !== 32'd8 || tile_k_len !== 16'd2) begin
                $display("[FAIL] wb%0d got r=%08h len=%0d kidx=%0d kbase=%0d klen=%0d",
                         idx, dma_r_addr, dma_r_len, tile_k_index, tile_k_base, tile_k_len);
                $display("       expected r=%08h len=16 kidx=2 kbase=8 klen=2", exp_addr);
                errors = errors + 1;
            end else begin
                $display("[PASS] wb%0d r=%08h after final k_tile", idx, exp_addr);
            end
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            dma_w_start_d <= 1'b0;
            dma_r_start_d <= 1'b0;
            load_idx <= 0;
            wb_idx <= 0;
            vec_count <= 0;
            flush_count <= 0;
        end else begin
            dma_w_start_d <= dma_w_start;
            dma_r_start_d <= dma_r_start;

            if (vec_consume)
                vec_count <= vec_count + 1;
            if (pe_flush)
                flush_count <= flush_count + 1;

            if (dma_w_start && !dma_w_start_d) begin
                case (load_idx)
                    0: expect_load(0, 32'h1000, 32'h2000, 16'd16); // k=0..3
                    1: expect_load(1, 32'h1010, 32'h2010, 16'd16); // k=4..7
                    2: expect_load(2, 32'h1020, 32'h2020, 16'd16); // k=8..9, padded to SIMD group
                    default: begin
                        $display("[FAIL] unexpected load index %0d", load_idx);
                        errors = errors + 1;
                    end
                endcase
                load_idx <= load_idx + 1;
            end

            if (dma_r_start && !dma_r_start_d) begin
                case (wb_idx)
                    0: expect_wb(0, 32'h3000);
                    1: expect_wb(1, 32'h3010);
                    2: expect_wb(2, 32'h3020);
                    3: expect_wb(3, 32'h3030);
                    default: begin
                        $display("[FAIL] unexpected writeback index %0d", wb_idx);
                        errors = errors + 1;
                    end
                endcase
                wb_idx <= wb_idx + 1;
            end
        end
    end

    initial begin
        errors = 0;
        load_idx = 0;
        wb_idx = 0;
        vec_count = 0;
        flush_count = 0;
        dma_w_start_d = 1'b0;
        dma_r_start_d = 1'b0;
        rst_n = 1'b0;
        ctrl_reg = 32'd0;
        m_dim = 32'd4;
        n_dim = 32'd4;
        k_dim = 32'd10;
        w_addr = 32'h1000;
        a_addr = 32'h2000;
        r_addr = 32'h3000;
        arr_cfg = 8'h80;       // tile mode
        cfg_shape_in = 2'b00;  // 4x4
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

        if (load_idx !== 3) begin
            $display("[FAIL] expected 3 K-split load pairs, got %0d", load_idx);
            errors = errors + 1;
        end
        if (wb_idx !== 4) begin
            $display("[FAIL] expected 4 final row writebacks, got %0d", wb_idx);
            errors = errors + 1;
        end
        if (vec_count !== 3) begin
            $display("[FAIL] expected 3 vec_consume pulses, got %0d", vec_count);
            errors = errors + 1;
        end
        if (flush_count !== 1) begin
            $display("[FAIL] expected exactly 1 flush on final k_tile, got %0d", flush_count);
            errors = errors + 1;
        end

        ctrl_reg = 32'd0;
        @(posedge clk);

        if (errors == 0) begin
            $display("[PASS] tb_npu_ctrl_ksplit");
        end else begin
            $display("[FAIL] tb_npu_ctrl_ksplit errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
