`timescale 1ns/1ps

module tb_npu_ctrl_error_status;
    localparam CLK_T = 10;

    localparam [31:0] CTRL_START_DIRECT = 32'h0000_0011;
    localparam [31:0] CTRL_START_DESC = 32'h0000_0081;
    localparam [31:0] DESC_CTRL_SUPPORTED =
        (32'h1 << 28) | // VERSION=1
        (32'h1 << 18) | // LAST_K
        (32'h1 << 17) | // FIRST_K
        (32'h1 << 16) | // TILE_PACKED
        (32'h1 <<  8) | // DATAFLOW=OS
        32'h1;          // OP=GEMM_TILEPACK
    localparam [31:0] DESC_CTRL_UNSUPPORTED =
        (32'h1 << 28) |
        (32'h1 << 16) |
        (32'h1 <<  8) |
        32'h2;          // OP=GEMM_ROWMAJOR, not implemented by T5.5

    localparam [31:0] ERR_DESC_COUNT_ZERO      = 32'h0000_0001;
    localparam [31:0] ERR_DESC_UNSUPPORTED     = 32'h0000_0002;
    localparam [31:0] ERR_DESC_COUNT_EXHAUSTED = 32'h0000_0004;
    localparam [31:0] ERR_DMA_RRESP            = 32'h0000_0010;
    localparam [31:0] ERR_DIRECT_INVALID_DIM   = 32'h0000_0100;

    reg clk = 1'b0;
    always #(CLK_T/2) clk = ~clk;

    reg rst_n;
    reg [31:0] ctrl_reg;
    reg [31:0] m_dim_cfg;
    reg [31:0] n_dim_cfg;
    reg [31:0] k_dim_cfg;
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
    reg err_clear;
    reg [31:0] err_clear_mask;

    wire dma_w_start;
    reg  dma_w_done;
    wire [31:0] dma_w_addr;
    wire [15:0] dma_w_len;
    wire dma_a_start;
    reg  dma_a_done;
    wire [31:0] dma_a_addr;
    wire [15:0] dma_a_len;
    wire dma_r_start;
    reg  dma_r_done;
    wire [31:0] dma_r_addr;
    wire [15:0] dma_r_len;
    reg [31:0] dma_error_status;
    wire irq;

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
        .m_dim(m_dim_cfg),
        .n_dim(n_dim_cfg),
        .k_dim(k_dim_cfg),
        .w_addr(32'h0000_1000),
        .a_addr(32'h0000_2000),
        .r_addr(32'h0000_3000),
        .arr_cfg(8'h80),
        .desc_base(desc_base),
        .desc_count(desc_count),
        .desc_start(desc_start),
        .desc_addr(desc_addr),
        .desc_done(desc_done),
        .desc_words(desc_words),
        .cfg_shape_in(2'b00),
        .cfg_shape_latched(),
        .tile_mode(),
        .router_enable(),
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
        .err_clear(err_clear),
        .err_clear_mask(err_clear_mask),
        .dma_w_start(dma_w_start),
        .dma_w_done(dma_w_done),
        .dma_w_addr(dma_w_addr),
        .dma_w_len(dma_w_len),
        .dma_a_start(dma_a_start),
        .dma_a_done(dma_a_done),
        .dma_a_addr(dma_a_addr),
        .dma_a_len(dma_a_len),
        .dma_a_ofm_mode(),
        .dma_a_ofm_stride(),
        .dma_a_ofm_m_base(),
        .dma_a_ofm_k_base(),
        .dma_a_ofm_k_len(),
        .dma_a_ofm_active_rows(),
        .dma_a_ofm_fp16_mode(),
        .dma_r_start(dma_r_start),
        .dma_r_done(dma_r_done),
        .dma_r_addr(dma_r_addr),
        .dma_r_len(dma_r_len),
        .dma_error_status(dma_error_status),
        .pe_en(),
        .pe_flush(),
        .pe_mode(),
        .pe_stat(),
        .pe_load_w(),
        .pe_swap_w(),
        .pe_array_ready(1'b1),
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
            desc_done  <= 1'b0;
            dma_w_done <= 1'b0;
            dma_a_done <= 1'b0;
            dma_r_done <= 1'b0;
        end else begin
            desc_done  <= desc_start;
            dma_w_done <= dma_w_start;
            dma_a_done <= dma_a_start;
            dma_r_done <= dma_r_start;
        end
    end

    task start_desc;
        input [31:0] count;
        begin
            desc_count = count;
            @(negedge clk);
            ctrl_reg = CTRL_START_DESC;
            @(negedge clk);
            ctrl_reg = 32'd0;
        end
    endtask

    task start_direct;
        begin
            @(negedge clk);
            ctrl_reg = CTRL_START_DIRECT;
            @(negedge clk);
            ctrl_reg = 32'd0;
        end
    endtask

    task clear_errors;
        begin
            @(negedge clk);
            err_clear = 1'b1;
            err_clear_mask = 32'hFFFF_FFFF;
            @(negedge clk);
            err_clear = 1'b0;
            err_clear_mask = 32'd0;
            repeat (2) @(posedge clk);
            if (err_status !== 32'd0 || error !== 1'b0) begin
                $display("[FAIL] error clear failed status=0x%08h error=%b", err_status, error);
                $fatal;
            end
        end
    endtask

    task wait_for_error;
        input [31:0] exp_mask;
        integer guard;
        begin
            guard = 0;
            while (((err_status & exp_mask) != exp_mask) && guard < 200) begin
                @(posedge clk);
                guard = guard + 1;
            end
            if ((err_status & exp_mask) != exp_mask) begin
                $display("[FAIL] timeout waiting error mask=0x%08h got=0x%08h state=%0d",
                         exp_mask, err_status, dut.state);
                $fatal;
            end
            if (busy !== 1'b0 || done !== 1'b0 || irq !== 1'b0 || error !== 1'b1) begin
                $display("[FAIL] bad error status busy=%b done=%b irq=%b error=%b status=0x%08h",
                         busy, done, irq, error, err_status);
                $fatal;
            end
            $display("[PASS] observed error mask 0x%08h", exp_mask);
        end
    endtask

    task write_desc_words;
        input [31:0] ctrl;
        input [31:0] next_desc;
        begin
            desc_words = 512'd0;
            desc_words[0*32 +: 32]  = ctrl;
            desc_words[1*32 +: 32]  = 32'd1;          // M
            desc_words[2*32 +: 32]  = 32'd1;          // N
            desc_words[3*32 +: 32]  = 32'd1;          // K
            desc_words[4*32 +: 32]  = 32'h0000_2000;  // A/IFM
            desc_words[5*32 +: 32]  = 32'h0000_1000;  // W
            desc_words[8*32 +: 32]  = 32'h0000_3000;  // R/OFM
            desc_words[15*32 +: 32] = next_desc;
        end
    endtask

    initial begin
        rst_n = 1'b0;
        ctrl_reg = 32'd0;
        m_dim_cfg = 32'd1;
        n_dim_cfg = 32'd1;
        k_dim_cfg = 32'd1;
        desc_base = 32'h0000_4000;
        desc_count = 32'd0;
        desc_done = 1'b0;
        desc_words = 512'd0;
        dma_w_done = 1'b0;
        dma_a_done = 1'b0;
        dma_r_done = 1'b0;
        dma_error_status = 32'd0;
        err_clear = 1'b0;
        err_clear_mask = 32'd0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        m_dim_cfg = 32'd0;
        start_direct();
        wait_for_error(ERR_DIRECT_INVALID_DIM);
        clear_errors();
        m_dim_cfg = 32'd1;

        @(negedge clk);
        dma_error_status = ERR_DMA_RRESP;
        @(negedge clk);
        dma_error_status = 32'd0;
        wait_for_error(ERR_DMA_RRESP);
        clear_errors();

        start_desc(32'd0);
        wait_for_error(ERR_DESC_COUNT_ZERO);
        clear_errors();

        write_desc_words(DESC_CTRL_UNSUPPORTED, 32'd0);
        start_desc(32'd1);
        wait_for_error(ERR_DESC_UNSUPPORTED);
        clear_errors();

        write_desc_words(DESC_CTRL_SUPPORTED, 32'h0000_5000);
        start_desc(32'd1);
        wait_for_error(ERR_DESC_COUNT_EXHAUSTED);
        clear_errors();

        $display("[PASS] tb_npu_ctrl_error_status: controller errors passed");
        $finish;
    end

    initial begin
        #(CLK_T * 2000);
        $display("[FAIL] tb_npu_ctrl_error_status timeout");
        $finish;
    end
endmodule
