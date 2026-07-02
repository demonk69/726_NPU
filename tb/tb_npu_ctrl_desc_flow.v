`timescale 1ns/1ps

module tb_npu_ctrl_desc_flow;
    localparam CLK_T = 10;

    localparam [31:0] CTRL_START_DESC = 32'h0000_0081;
    localparam [31:0] DESC_CTRL_BASE =
        (32'h1 << 28) | // VERSION=1
        (32'h1 << 18) | // FIRST_K
        (32'h1 << 16) | // TILE_PACKED
        (32'h1 <<  8) | // DATAFLOW=OS
        32'h1;          // OP=GEMM_TILEPACK

    reg clk = 1'b0;
    always #(CLK_T/2) clk = ~clk;

    reg rst_n;
    reg [31:0] ctrl_reg;
    reg [31:0] desc_base;
    reg [31:0] desc_count;
    wire desc_start;
    wire [31:0] desc_addr;
    reg desc_done;
    reg [511:0] desc_words;

    wire busy;
    wire done;
    wire error;
    wire [31:0] err_status;
    wire dma_w_start;
    reg dma_w_done;
    wire [31:0] dma_w_addr;
    wire [15:0] dma_w_len;
    wire dma_a_start;
    reg dma_a_done;
    wire [31:0] dma_a_addr;
    wire [15:0] dma_a_len;
    wire dma_a_ofm_mode;
    wire [31:0] dma_a_ofm_stride;
    wire [31:0] dma_a_ofm_m_base;
    wire [31:0] dma_a_ofm_k_base;
    wire [15:0] dma_a_ofm_k_len;
    wire [4:0] dma_a_ofm_active_rows;
    wire dma_r_start;
    reg dma_r_done;
    wire [31:0] dma_r_addr;
    wire [15:0] dma_r_len;
    wire irq;

    integer errors;
    integer desc_fetches;
    integer prev_ofm_loads;
    reg [31:0] last_r_addr;

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
        .m_dim(32'd1),
        .n_dim(32'd1),
        .k_dim(32'd1),
        .w_addr(32'h0000_1000),
        .a_addr(32'h0000_2000),
        .r_addr(32'h0000_3000),
        .bias_addr(32'd0),
        .quant_cfg(32'h0001_0000),
        .arr_cfg(8'h80),
        .desc_base(desc_base),
        .desc_count(desc_count),
        .conv_ifm_shape(32'd0),
        .conv_channels(32'd0),
        .conv_kernel(32'd0),
        .conv_out_shape(32'd0),
        .conv_stride_pad(32'd0),
        .conv_dilation(32'd0),
        .desc_start(desc_start),
        .desc_addr(desc_addr),
        .desc_done(desc_done),
        .desc_words(desc_words),
        .cfg_shape_in(2'b00),
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
        .dma_w_addr(dma_w_addr),
        .dma_w_len(dma_w_len),
        .dma_a_start(dma_a_start),
        .dma_a_done(dma_a_done),
        .dma_a_addr(dma_a_addr),
        .dma_a_len(dma_a_len),
        .dma_a_ofm_mode(dma_a_ofm_mode),
        .dma_a_im2col_mode(),
        .dma_a_ofm_stride(dma_a_ofm_stride),
        .dma_a_ofm_m_base(dma_a_ofm_m_base),
        .dma_a_ofm_k_base(dma_a_ofm_k_base),
        .dma_a_ofm_k_len(dma_a_ofm_k_len),
        .dma_a_ofm_active_rows(dma_a_ofm_active_rows),
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
        .dma_bias_start(),
        .dma_bias_done(1'b0),
        .dma_bias_addr(),
        .dma_r_start(dma_r_start),
        .dma_r_done(dma_r_done),
        .dma_r_addr(dma_r_addr),
        .dma_r_len(dma_r_len),
        .dma_error_status(32'd0),
        .pe_en(),
        .pe_flush(),
        .pe_mode(),
        .pe_stat(),
        .pe_load_w(),
        .pe_swap_w(),
        .pe_acc_init_en(),
        .pe_half_en(),
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

    task make_desc;
        input [31:0] ctrl;
        input [31:0] m;
        input [31:0] n;
        input [31:0] k;
        input [31:0] ifm;
        input [31:0] wgt;
        input [31:0] ofm;
        input [31:0] next_desc;
        begin
            desc_words = 512'd0;
            desc_words[0*32 +: 32]  = ctrl;
            desc_words[1*32 +: 32]  = m;
            desc_words[2*32 +: 32]  = n;
            desc_words[3*32 +: 32]  = k;
            desc_words[4*32 +: 32]  = ifm;
            desc_words[5*32 +: 32]  = wgt;
            desc_words[8*32 +: 32]  = ofm;
            desc_words[15*32 +: 32] = next_desc;
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            desc_done <= 1'b0;
            dma_w_done <= 1'b0;
            dma_a_done <= 1'b0;
            dma_r_done <= 1'b0;
            desc_fetches <= 0;
            prev_ofm_loads <= 0;
            last_r_addr <= 32'd0;
        end else begin
            desc_done <= desc_start;
            dma_w_done <= dma_w_start;
            dma_a_done <= dma_a_start;
            dma_r_done <= dma_r_start;

            if (desc_start) begin
                desc_fetches <= desc_fetches + 1;
                if (desc_addr == 32'h0000_5000)
                    make_desc(DESC_CTRL_BASE | (32'h1 << 19) | (32'h1 << 20) | (32'h1 << 23),
                              32'd4, 32'd4, 32'd4,
                              32'hDEAD_0000, 32'h0000_1200, 32'h0000_4000, 32'd0);
                else
                    make_desc(DESC_CTRL_BASE, 32'd4, 32'd4, 32'd4,
                              32'h0000_2000, 32'h0000_1000, 32'h0000_3000, 32'h0000_5000);
            end

            if (dma_a_start && dma_a_ofm_mode) begin
                prev_ofm_loads <= prev_ofm_loads + 1;
                if (dma_a_addr !== 32'h0000_3000 ||
                    dma_a_ofm_stride !== 32'd4 ||
                    dma_a_ofm_k_base !== 32'd0 ||
                    dma_a_ofm_k_len !== 16'd4 ||
                    dma_a_ofm_active_rows !== 5'd4) begin
                    $display("[FAIL] prev-OFM DMA mismatch addr=%08h stride=%0d k_base=%0d k_len=%0d rows=%0d",
                             dma_a_addr, dma_a_ofm_stride, dma_a_ofm_k_base,
                             dma_a_ofm_k_len, dma_a_ofm_active_rows);
                    errors = errors + 1;
                end
            end

            if (dma_r_start)
                last_r_addr <= dma_r_addr;
        end
    end

    integer guard;
    initial begin
        errors = 0;
        rst_n = 1'b0;
        ctrl_reg = 32'd0;
        desc_base = 32'h0000_4000;
        desc_count = 32'd2;
        desc_done = 1'b0;
        desc_words = 512'd0;
        dma_w_done = 1'b0;
        dma_a_done = 1'b0;
        dma_r_done = 1'b0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        @(negedge clk);
        ctrl_reg = CTRL_START_DESC;
        @(negedge clk);
        ctrl_reg = 32'd0;

        guard = 0;
        while (!done && !error && guard < 1000) begin
            @(posedge clk);
            guard = guard + 1;
        end

        if (!done || error || !irq) begin
            $display("[FAIL] descriptor flow did not finish done=%b error=%b irq=%b err=0x%08h state=%0d",
                     done, error, irq, err_status, dut.state);
            errors = errors + 1;
        end
        if (desc_fetches != 2) begin
            $display("[FAIL] expected two descriptor fetches, got %0d", desc_fetches);
            errors = errors + 1;
        end
        if (prev_ofm_loads == 0) begin
            $display("[FAIL] second descriptor did not use previous OFM");
            errors = errors + 1;
        end
        if (last_r_addr < 32'h0000_4000) begin
            $display("[FAIL] final result address did not reach second OFM, last=%08h", last_r_addr);
            errors = errors + 1;
        end

        @(negedge clk);
        ctrl_reg = 32'h0000_0040;
        @(negedge clk);
        ctrl_reg = 32'd0;
        repeat (3) @(posedge clk);
        if (irq !== 1'b0) begin
            $display("[FAIL] irq clear did not deassert irq");
            errors = errors + 1;
        end

        if (errors == 0)
            $display("[PASS] tb_npu_ctrl_desc_flow: descriptor chain and prev-OFM passed");
        else
            $display("[FAIL] tb_npu_ctrl_desc_flow errors=%0d", errors);
        $finish;
    end

    initial begin
        #(CLK_T * 2000);
        $display("[FAIL] tb_npu_ctrl_desc_flow timeout");
        $finish;
    end
endmodule
