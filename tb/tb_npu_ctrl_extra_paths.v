`timescale 1ns/1ps

module tb_npu_ctrl_extra_paths;
    localparam CLK_T = 10;

    reg clk = 1'b0;
    always #(CLK_T/2) clk = ~clk;

    reg rst_n;
    reg [31:0] ctrl_reg;
    reg [31:0] m_dim;
    reg [31:0] n_dim;
    reg [31:0] k_dim;
    reg [7:0] arr_cfg;
    reg [1:0] cfg_shape_in;
    reg dma_w_done;
    reg dma_a_done;
    reg dma_bias_done;
    reg dma_r_done;
    reg stall_dma_done;

    wire busy;
    wire done;
    wire error;
    wire [31:0] err_status;
    wire dma_w_start;
    wire dma_a_start;
    wire dma_bias_start;
    wire dma_r_start;
    wire pe_half_en;
    wire irq;

    integer errors;
    integer bias_starts;
    integer half_seen;

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
        .w_addr(32'h0000_1000),
        .a_addr(32'h0000_2000),
        .r_addr(32'h0000_3000),
        .bias_addr(32'h0000_4000),
        .quant_cfg(32'h0001_0000),
        .arr_cfg(arr_cfg),
        .desc_base(32'd0),
        .desc_count(32'd0),
        .conv_ifm_shape(32'd0),
        .conv_channels(32'd0),
        .conv_kernel(32'd0),
        .conv_out_shape(32'd0),
        .conv_stride_pad(32'd0),
        .conv_dilation(32'd0),
        .desc_start(),
        .desc_addr(),
        .desc_done(1'b0),
        .desc_words(512'd0),
        .cfg_shape_in(cfg_shape_in),
        .cfg_shape_latched(),
        .post_act_mode(),
        .post_quant_cfg(),
        .bias_en(),
        .tile_mode(),
        .vec_consume(),
        .tile_m_base(),
        .tile_n_base(),
        .tile_row_valid(),
        .tile_col_valid(),
        .tile_active_rows(),
        .tile_active_cols(),
        .tile_k_base(),
        .tile_k_len(),
        .tile_k_index(),
        .tile_k_cycle(),
        .busy(busy),
        .done(done),
        .error(error),
        .err_status(err_status),
        .err_clear(1'b0),
        .err_clear_mask(32'd0),
        .dma_w_start(dma_w_start),
        .dma_w_done(dma_w_done),
        .dma_w_addr(),
        .dma_w_len(),
        .dma_a_start(dma_a_start),
        .dma_a_done(dma_a_done),
        .dma_a_addr(),
        .dma_a_len(),
        .dma_a_ofm_mode(),
        .dma_a_im2col_mode(),
        .dma_a_ofm_stride(),
        .dma_a_ofm_m_base(),
        .dma_a_ofm_k_base(),
        .dma_a_ofm_k_len(),
        .dma_a_ofm_active_rows(),
        .dma_a_ofm_fp16_mode(),
        .dma_a_im2col_m_index(),
        .dma_a_im2col_k_len(),
        .dma_a_im2col_ih(),
        .dma_a_im2col_iw(),
        .dma_a_im2col_cin(),
        .dma_a_im2col_kh(),
        .dma_a_im2col_kw(),
        .dma_a_im2col_oh(),
        .dma_a_im2col_ow(),
        .dma_a_im2col_stride_h(),
        .dma_a_im2col_stride_w(),
        .dma_a_im2col_pad_h(),
        .dma_a_im2col_pad_w(),
        .dma_a_im2col_dilation_h(),
        .dma_a_im2col_dilation_w(),
        .dma_a_im2col_fp16_mode(),
        .dma_bias_start(dma_bias_start),
        .dma_bias_done(dma_bias_done),
        .dma_bias_addr(),
        .dma_r_start(dma_r_start),
        .dma_r_done(dma_r_done),
        .dma_r_addr(),
        .dma_r_len(),
        .dma_error_status(32'd0),
        .pe_en(),
        .pe_flush(),
        .pe_mode(),
        .pe_stat(),
        .pe_load_w(),
        .pe_swap_w(),
        .pe_acc_init_en(),
        .pe_half_en(pe_half_en),
        .w_ppb_ready(1'b1),
        .w_ppb_empty(1'b0),
        .a_ppb_ready(1'b1),
        .a_ppb_empty(1'b0),
        .w_ppb_swap(),
        .a_ppb_swap(),
        .w_ppb_clear(),
        .a_ppb_clear(),
        .r_fifo_clear(),
        .irq(irq),
        .compute_ce(1'b1)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            dma_w_done <= 1'b0;
            dma_a_done <= 1'b0;
            dma_bias_done <= 1'b0;
            dma_r_done <= 1'b0;
            bias_starts <= 0;
            half_seen <= 0;
        end else begin
            dma_w_done <= stall_dma_done ? 1'b0 : dma_w_start;
            dma_a_done <= stall_dma_done ? 1'b0 : dma_a_start;
            dma_bias_done <= dma_bias_start;
            dma_r_done <= dma_r_start;
            if (dma_bias_start)
                bias_starts <= bias_starts + 1;
            if (pe_half_en)
                half_seen <= 1;
        end
    end

    task run_direct;
        input [1:0] shape;
        input [31:0] m;
        input [31:0] n;
        input [31:0] k;
        input bias;
        integer guard;
        begin
            cfg_shape_in = shape;
            m_dim = m;
            n_dim = n;
            k_dim = k;
            arr_cfg = 8'h80;
            @(negedge clk);
            ctrl_reg = bias ? 32'h0000_0211 : 32'h0000_0011;
            @(negedge clk);
            ctrl_reg = 32'd0;
            guard = 0;
            while (!done && !error && guard < 1200) begin
                @(posedge clk);
                guard = guard + 1;
            end
            if (!done || error) begin
                $display("[FAIL] direct shape=%0d bias=%0d did not finish done=%b error=%b err=0x%08h state=%0d",
                         shape, bias, done, error, err_status, dut.state);
                errors = errors + 1;
            end
            repeat (4) @(posedge clk);
        end
    endtask

    task abort_in_fetch_desc;
        integer guard;
        begin
            @(negedge clk);
            ctrl_reg = 32'h0000_0081;
            @(negedge clk);
            ctrl_reg = 32'd0;
            guard = 0;
            while (dut.state != 4'd11 && guard < 40) begin
                @(posedge clk);
                guard = guard + 1;
            end
            @(negedge clk);
            ctrl_reg = 32'h0000_0002;
            @(negedge clk);
            ctrl_reg = 32'd0;
            repeat (4) @(posedge clk);
            if (busy || dut.state != 4'd0) begin
                $display("[FAIL] abort in fetch desc failed busy=%b state=%0d", busy, dut.state);
                errors = errors + 1;
            end
        end
    endtask

    task abort_in_warmup_load;
        integer guard;
        begin
            stall_dma_done = 1'b1;
            cfg_shape_in = 2'b00;
            m_dim = 32'd4;
            n_dim = 32'd4;
            k_dim = 32'd4;
            arr_cfg = 8'h80;
            @(negedge clk);
            ctrl_reg = 32'h0000_0011;
            @(negedge clk);
            ctrl_reg = 32'd0;
            guard = 0;
            while (dut.state != 4'd1 && guard < 40) begin
                @(posedge clk);
                guard = guard + 1;
            end
            @(negedge clk);
            ctrl_reg = 32'h0000_0002;
            @(negedge clk);
            ctrl_reg = 32'd0;
            stall_dma_done = 1'b0;
            repeat (4) @(posedge clk);
            if (busy || dut.state != 4'd0) begin
                $display("[FAIL] abort in warmup failed busy=%b state=%0d", busy, dut.state);
                errors = errors + 1;
            end
        end
    endtask

    task abort_in_overlap_compute;
        integer guard;
        begin
            cfg_shape_in = 2'b00;
            m_dim = 32'd4;
            n_dim = 32'd4;
            k_dim = 32'd16;
            arr_cfg = 8'h80;
            @(negedge clk);
            ctrl_reg = 32'h0000_0011;
            @(negedge clk);
            ctrl_reg = 32'd0;
            guard = 0;
            while (dut.state != 4'd3 && guard < 120) begin
                @(posedge clk);
                guard = guard + 1;
            end
            @(negedge clk);
            ctrl_reg = 32'h0000_0002;
            @(negedge clk);
            ctrl_reg = 32'd0;
            repeat (4) @(posedge clk);
            if (busy || dut.state != 4'd0) begin
                $display("[FAIL] abort in overlap failed busy=%b state=%0d", busy, dut.state);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0;
        rst_n = 1'b0;
        ctrl_reg = 32'd0;
        m_dim = 32'd1;
        n_dim = 32'd1;
        k_dim = 32'd1;
        arr_cfg = 8'h80;
        cfg_shape_in = 2'b00;
        dma_w_done = 1'b0;
        dma_a_done = 1'b0;
        dma_bias_done = 1'b0;
        dma_r_done = 1'b0;
        stall_dma_done = 1'b0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        run_direct(2'b01, 32'd8, 32'd8, 32'd4, 1'b1);
        if (bias_starts == 0) begin
            $display("[FAIL] tile bias path did not start bias DMA");
            errors = errors + 1;
        end

        run_direct(2'b11, 32'd8, 32'd32, 32'd4, 1'b0);
        if (!half_seen) begin
            $display("[FAIL] 8x32 pass-1 half was not observed");
            errors = errors + 1;
        end

        abort_in_fetch_desc();
        abort_in_warmup_load();
        abort_in_overlap_compute();

        if (errors == 0)
            $display("[PASS] tb_npu_ctrl_extra_paths: shape, bias, 8x32, abort paths passed");
        else
            $display("[FAIL] tb_npu_ctrl_extra_paths errors=%0d", errors);
        $finish;
    end

    initial begin
        #(CLK_T * 4000);
        $display("[FAIL] tb_npu_ctrl_extra_paths timeout");
        $finish;
    end
endmodule
