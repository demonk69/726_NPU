`timescale 1ns/1ps

module tb_npu_ctrl_dataflow_modes;
    localparam CLK_T = 10;

    localparam [31:0] CTRL_START_WS_INT8 = 32'h0000_0001;
    localparam [31:0] CTRL_START_OS_INT8 = 32'h0000_0011;
    localparam [31:0] M_DIM = 32'd1;
    localparam [31:0] N_DIM = 32'd1;
    localparam [31:0] K_DIM = 32'd4;

    reg clk = 1'b0;
    always #(CLK_T/2) clk = ~clk;

    reg rst_n;
    reg [31:0] ctrl_reg;
    reg dma_w_done;
    reg dma_a_done;
    reg dma_r_done;
    reg w_ppb_empty;
    reg a_ppb_empty;

    wire [1:0] cfg_shape_latched;
    wire tile_mode;
    wire vec_consume;
    wire [31:0] tile_m_base;
    wire [31:0] tile_n_base;
    wire [3:0] tile_row_valid;
    wire [3:0] tile_col_valid;
    wire [2:0] tile_active_rows;
    wire [2:0] tile_active_cols;
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

    npu_ctrl #(
        .ROWS(1),
        .COLS(1),
        .DATA_W(16),
        .ACC_W(32),
        .PPB_DEPTH(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .ctrl_reg(ctrl_reg),
        .m_dim(M_DIM),
        .n_dim(N_DIM),
        .k_dim(K_DIM),
        .w_addr(32'h0000_1000),
        .a_addr(32'h0000_2000),
        .r_addr(32'h0000_3000),
        .arr_cfg(8'd0),
        .desc_base(32'd0),
        .desc_count(32'd0),
        .desc_start(),
        .desc_addr(),
        .desc_done(1'b0),
        .desc_words(512'd0),
        .cfg_shape_in(2'b00),
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
        .pe_en(pe_en),
        .pe_flush(pe_flush),
        .pe_mode(pe_mode),
        .pe_stat(pe_stat),
        .pe_load_w(pe_load_w),
        .pe_swap_w(pe_swap_w),
        .w_ppb_ready(1'b1),
        .w_ppb_empty(w_ppb_empty),
        .a_ppb_ready(1'b1),
        .a_ppb_empty(a_ppb_empty),
        .w_ppb_swap(w_ppb_swap),
        .a_ppb_swap(a_ppb_swap),
        .w_ppb_clear(w_ppb_clear),
        .a_ppb_clear(a_ppb_clear),
        .r_fifo_clear(r_fifo_clear),
        .irq(irq)
    );

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

    task run_mode;
        input is_os;
        input [31:0] ctrl_value;
        integer guard;
        integer pe_en_count;
        integer pe_load_w_count;
        integer stat_mismatch;
        integer saw_writeback;
        begin
            ctrl_reg = 32'd0;
            w_ppb_empty = 1'b0;
            a_ppb_empty = 1'b0;
            repeat (4) @(posedge clk);

            @(negedge clk);
            ctrl_reg = ctrl_value;
            @(negedge clk);

            guard = 0;
            while (dut.state != 4'd3 && guard < 100) begin
                @(posedge clk);
                guard = guard + 1;
            end
            if (dut.state != 4'd3) begin
                $display("[FAIL] %0s did not enter compute state, state=%0d",
                         is_os ? "OS" : "WS", dut.state);
                $fatal;
            end

            pe_en_count = 0;
            pe_load_w_count = 0;
            stat_mismatch = 0;
            saw_writeback = 0;
            guard = 0;
            while (!done && guard < 200) begin
                @(posedge clk);
                guard = guard + 1;

                if (dut.state == 4'd3) begin
                    if (pe_stat !== is_os) stat_mismatch = stat_mismatch + 1;
                    if (pe_en) pe_en_count = pe_en_count + 1;
                    if (pe_load_w) pe_load_w_count = pe_load_w_count + 1;

                    if (is_os && pe_en_count >= K_DIM) begin
                        w_ppb_empty = 1'b1;
                        a_ppb_empty = 1'b1;
                    end
                end

                if (dma_r_start) saw_writeback = 1;
            end

            if (!done || error) begin
                $display("[FAIL] %0s did not finish cleanly done=%b error=%b err=0x%08h state=%0d",
                         is_os ? "OS" : "WS", done, error, err_status, dut.state);
                $fatal;
            end
            if (stat_mismatch != 0) begin
                $display("[FAIL] %0s pe_stat mismatch count=%0d",
                         is_os ? "OS" : "WS", stat_mismatch);
                $fatal;
            end
            if (saw_writeback == 0) begin
                $display("[FAIL] %0s did not issue result writeback",
                         is_os ? "OS" : "WS");
                $fatal;
            end
            if (is_os && pe_load_w_count != 0) begin
                $display("[FAIL] OS unexpectedly asserted pe_load_w %0d cycles", pe_load_w_count);
                $fatal;
            end
            if (!is_os && pe_load_w_count != K_DIM) begin
                $display("[FAIL] WS pe_load_w cycles got=%0d exp=%0d", pe_load_w_count, K_DIM);
                $fatal;
            end
            if (pe_en_count == 0) begin
                $display("[FAIL] %0s did not assert pe_en during compute", is_os ? "OS" : "WS");
                $fatal;
            end

            $display("[PASS] %0s direct dataflow pe_en=%0d pe_load_w=%0d",
                     is_os ? "OS" : "WS", pe_en_count, pe_load_w_count);

            @(negedge clk);
            ctrl_reg = 32'd0;
            w_ppb_empty = 1'b0;
            a_ppb_empty = 1'b0;
            repeat (6) @(posedge clk);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        ctrl_reg = 32'd0;
        dma_w_done = 1'b0;
        dma_a_done = 1'b0;
        dma_r_done = 1'b0;
        w_ppb_empty = 1'b0;
        a_ppb_empty = 1'b0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        run_mode(1'b1, CTRL_START_OS_INT8);
        run_mode(1'b0, CTRL_START_WS_INT8);

        $display("[PASS] tb_npu_ctrl_dataflow_modes: OS and WS control branches passed");
        $finish;
    end

    initial begin
        #(CLK_T * 2000);
        $display("[FAIL] tb_npu_ctrl_dataflow_modes timeout");
        $finish;
    end
endmodule
