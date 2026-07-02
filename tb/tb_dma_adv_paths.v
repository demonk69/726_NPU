`timescale 1ns/1ps

module tb_dma_adv_paths;

localparam DATA_W = 32;
localparam CLK_T = 10;
localparam MEM_WORDS = 4096;

reg clk = 1'b0;
always #(CLK_T/2) clk = ~clk;

reg rst_n = 1'b0;
initial begin
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
end

reg w_start, a_start, desc_start, bias_start, r_start;
reg [31:0] w_base_addr, a_base_addr, desc_base_addr, bias_addr, r_base_addr;
reg [15:0] w_len_bytes, a_len_bytes, r_len_bytes;
wire w_done, a_done, desc_done, bias_done, r_done;
wire w_ppb_wr_en, a_ppb_wr_en;
wire [31:0] w_ppb_wr_data, a_ppb_wr_data;
reg w_ppb_full, a_ppb_full;
reg r_fifo_clear, r_fifo_wr_en;
reg [31:0] r_fifo_din;
wire r_fifo_full;
wire [511:0] desc_words;
wire [31:0] bias_data, dma_err_status;

reg a_ofm_mode, a_im2col_mode, a_ofm_fp16_mode, a_im2col_fp16_mode;
reg [31:0] a_ofm_stride, a_ofm_m_base, a_ofm_k_base, a_im2col_m_index;
reg [15:0] a_ofm_k_len, a_im2col_k_len;
reg [4:0] a_ofm_active_rows;
reg [15:0] a_im2col_ih, a_im2col_iw, a_im2col_cin, a_im2col_kh, a_im2col_kw, a_im2col_oh, a_im2col_ow;
reg [7:0] a_im2col_stride_h, a_im2col_stride_w, a_im2col_pad_h, a_im2col_pad_w;
reg [7:0] a_im2col_dilation_h, a_im2col_dilation_w;

wire [31:0] m_axi_awaddr, m_axi_araddr, m_axi_wdata;
wire [7:0] m_axi_awlen, m_axi_arlen;
wire [2:0] m_axi_awsize, m_axi_arsize;
wire [1:0] m_axi_awburst, m_axi_arburst;
wire m_axi_awvalid, m_axi_wvalid, m_axi_wlast, m_axi_bready, m_axi_arvalid, m_axi_rready;
wire [3:0] m_axi_wstrb;
reg m_axi_awready, m_axi_wready, m_axi_bvalid, m_axi_arready, m_axi_rvalid, m_axi_rlast;
reg [1:0] m_axi_bresp, m_axi_rresp;
reg [31:0] m_axi_rdata;

npu_dma #(
    .DATA_W(DATA_W),
    .PE_DATA_W(16),
    .BURST_MAX(16),
    .PPB_DEPTH(128),
    .R_FIFO_DEPTH(128)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .w_start(w_start),
    .w_base_addr(w_base_addr),
    .w_len_bytes(w_len_bytes),
    .w_done(w_done),
    .w_ppb_wr_en(w_ppb_wr_en),
    .w_ppb_wr_data(w_ppb_wr_data),
    .w_ppb_full(w_ppb_full),
    .w_ppb_buf_ready(1'b0),
    .w_ppb_buf_empty(1'b1),
    .w_ppb_drain_done(1'b1),
    .a_start(a_start),
    .a_base_addr(a_base_addr),
    .a_len_bytes(a_len_bytes),
    .a_done(a_done),
    .a_ppb_wr_en(a_ppb_wr_en),
    .a_ppb_wr_data(a_ppb_wr_data),
    .a_ppb_full(a_ppb_full),
    .a_ppb_buf_ready(1'b0),
    .a_ppb_buf_empty(1'b1),
    .a_ppb_drain_done(1'b1),
    .a_ofm_mode(a_ofm_mode),
    .a_im2col_mode(a_im2col_mode),
    .a_ofm_stride(a_ofm_stride),
    .a_ofm_m_base(a_ofm_m_base),
    .a_ofm_k_base(a_ofm_k_base),
    .a_ofm_k_len(a_ofm_k_len),
    .a_ofm_active_rows(a_ofm_active_rows),
    .a_ofm_fp16_mode(a_ofm_fp16_mode),
    .a_im2col_m_index(a_im2col_m_index),
    .a_im2col_k_len(a_im2col_k_len),
    .a_im2col_ih(a_im2col_ih),
    .a_im2col_iw(a_im2col_iw),
    .a_im2col_cin(a_im2col_cin),
    .a_im2col_kh(a_im2col_kh),
    .a_im2col_kw(a_im2col_kw),
    .a_im2col_oh(a_im2col_oh),
    .a_im2col_ow(a_im2col_ow),
    .a_im2col_stride_h(a_im2col_stride_h),
    .a_im2col_stride_w(a_im2col_stride_w),
    .a_im2col_pad_h(a_im2col_pad_h),
    .a_im2col_pad_w(a_im2col_pad_w),
    .a_im2col_dilation_h(a_im2col_dilation_h),
    .a_im2col_dilation_w(a_im2col_dilation_w),
    .a_im2col_fp16_mode(a_im2col_fp16_mode),
    .desc_start(desc_start),
    .desc_base_addr(desc_base_addr),
    .desc_done(desc_done),
    .desc_words(desc_words),
    .bias_start(bias_start),
    .bias_addr(bias_addr),
    .bias_done(bias_done),
    .bias_data(bias_data),
    .r_start(r_start),
    .r_base_addr(r_base_addr),
    .r_len_bytes(r_len_bytes),
    .r_done(r_done),
    .r_fifo_clear(r_fifo_clear),
    .r_fifo_wr_en(r_fifo_wr_en),
    .r_fifo_din(r_fifo_din),
    .r_fifo_full(r_fifo_full),
    .dma_err_status(dma_err_status),
    .m_axi_awaddr(m_axi_awaddr),
    .m_axi_awlen(m_axi_awlen),
    .m_axi_awsize(m_axi_awsize),
    .m_axi_awburst(m_axi_awburst),
    .m_axi_awvalid(m_axi_awvalid),
    .m_axi_awready(m_axi_awready),
    .m_axi_wdata(m_axi_wdata),
    .m_axi_wstrb(m_axi_wstrb),
    .m_axi_wlast(m_axi_wlast),
    .m_axi_wvalid(m_axi_wvalid),
    .m_axi_wready(m_axi_wready),
    .m_axi_bresp(m_axi_bresp),
    .m_axi_bvalid(m_axi_bvalid),
    .m_axi_bready(m_axi_bready),
    .m_axi_araddr(m_axi_araddr),
    .m_axi_arlen(m_axi_arlen),
    .m_axi_arsize(m_axi_arsize),
    .m_axi_arburst(m_axi_arburst),
    .m_axi_arvalid(m_axi_arvalid),
    .m_axi_arready(m_axi_arready),
    .m_axi_rdata(m_axi_rdata),
    .m_axi_rresp(m_axi_rresp),
    .m_axi_rvalid(m_axi_rvalid),
    .m_axi_rready(m_axi_rready),
    .m_axi_rlast(m_axi_rlast)
);

reg [31:0] mem [0:MEM_WORDS-1];
integer errors;
integer a_wr_count;
reg saw_w_done;
reg saw_a_done;
reg saw_dma_rd_align;
reg saw_dma_wr_align;
reg rd_active;
reg [31:0] rd_addr;
reg [7:0] rd_len;
reg [7:0] rd_cnt;

function [31:0] word_at;
    input [31:0] addr;
    begin
        word_at = mem[addr >> 2];
    end
endfunction

always @(posedge clk) begin
    if (!rst_n) begin
        m_axi_arready <= 1'b0;
        m_axi_rvalid <= 1'b0;
        m_axi_rlast <= 1'b0;
        m_axi_rdata <= 32'd0;
        m_axi_rresp <= 2'b00;
        rd_active <= 1'b0;
        rd_addr <= 32'd0;
        rd_len <= 8'd0;
        rd_cnt <= 8'd0;
    end else begin
        m_axi_arready <= 1'b1;
        if (m_axi_arvalid && m_axi_arready) begin
            rd_active <= 1'b1;
            rd_addr <= m_axi_araddr;
            rd_len <= m_axi_arlen;
            rd_cnt <= 8'd0;
        end

        if (m_axi_rvalid && m_axi_rready) begin
            if (m_axi_rlast) begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast <= 1'b0;
                rd_active <= 1'b0;
            end else begin
                rd_cnt <= rd_cnt + 1'b1;
                m_axi_rdata <= word_at(rd_addr + ((rd_cnt + 1'b1) << 2));
                m_axi_rlast <= ((rd_cnt + 1'b1) >= rd_len);
            end
        end else if (rd_active && !m_axi_rvalid) begin
            m_axi_rvalid <= 1'b1;
            m_axi_rdata <= word_at(rd_addr);
            m_axi_rlast <= (rd_len == 8'd0);
        end
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        m_axi_awready <= 1'b0;
        m_axi_wready <= 1'b0;
        m_axi_bvalid <= 1'b0;
        m_axi_bresp <= 2'b00;
    end else begin
        m_axi_awready <= 1'b1;
        m_axi_wready <= 1'b1;
        if (m_axi_wvalid && m_axi_wready && m_axi_wlast)
            m_axi_bvalid <= 1'b1;
        else if (m_axi_bvalid && m_axi_bready)
            m_axi_bvalid <= 1'b0;
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        a_wr_count <= 0;
        saw_w_done <= 1'b0;
        saw_a_done <= 1'b0;
        saw_dma_rd_align <= 1'b0;
        saw_dma_wr_align <= 1'b0;
    end else begin
        if (a_ppb_wr_en) a_wr_count <= a_wr_count + 1;
        if (w_done) saw_w_done <= 1'b1;
        if (a_done) saw_a_done <= 1'b1;
        if (dma_err_status & 32'h0000_0040) saw_dma_rd_align <= 1'b1;
        if (dma_err_status & 32'h0000_0080) saw_dma_wr_align <= 1'b1;
    end
end

task clear_starts;
    begin
        w_start = 0; a_start = 0; desc_start = 0; bias_start = 0; r_start = 0;
    end
endtask

task pulse_desc;
    input [31:0] addr;
    begin
        @(negedge clk);
        desc_base_addr = addr;
        desc_start = 1'b1;
        @(negedge clk);
        desc_start = 1'b0;
    end
endtask

task pulse_bias;
    input [31:0] addr;
    begin
        @(negedge clk);
        bias_addr = addr;
        bias_start = 1'b1;
        @(negedge clk);
        bias_start = 1'b0;
    end
endtask

task pulse_a_ofm_zero;
    begin
        @(negedge clk);
        a_base_addr = 32'h0000_0300;
        a_ofm_k_len = 16'd0;
        a_ofm_active_rows = 5'd4;
        a_ofm_fp16_mode = 1'b0;
        a_ofm_mode = 1'b1;
        a_im2col_mode = 1'b0;
        a_start = 1'b1;
        @(negedge clk);
        a_start = 1'b0;
    end
endtask

task pulse_a_im2col_zero;
    begin
        @(negedge clk);
        a_base_addr = 32'h0000_0800;
        a_im2col_k_len = 16'd0;
        a_im2col_mode = 1'b1;
        a_ofm_mode = 1'b0;
        a_start = 1'b1;
        @(negedge clk);
        a_start = 1'b0;
    end
endtask

task pulse_bad_w_align;
    begin
        @(negedge clk);
        w_base_addr = 32'h0000_0102;
        w_len_bytes = 16'd8;
        a_start = 1'b0;
        bias_start = 1'b0;
        w_start = 1'b1;
        @(negedge clk);
        w_start = 1'b0;
    end
endtask

task pulse_bad_a_align;
    input ofm_mode;
    input im2col_mode;
    begin
        @(negedge clk);
        a_base_addr = 32'h0000_0802;
        a_len_bytes = 16'd8;
        a_ofm_mode = ofm_mode;
        a_im2col_mode = im2col_mode;
        a_ofm_k_len = 16'd4;
        a_im2col_k_len = 16'd4;
        a_start = 1'b1;
        @(negedge clk);
        a_start = 1'b0;
    end
endtask

task pulse_bad_bias_align;
    begin
        @(negedge clk);
        bias_addr = 32'h0000_0202;
        bias_start = 1'b1;
        @(negedge clk);
        bias_start = 1'b0;
    end
endtask

task pulse_a_direct_bias;
    begin
        @(negedge clk);
        a_base_addr = 32'h0000_0900;
        a_len_bytes = 16'd8;
        a_ofm_mode = 1'b0;
        a_im2col_mode = 1'b0;
        bias_addr = 32'h0000_0200;
        a_start = 1'b1;
        bias_start = 1'b1;
        @(negedge clk);
        a_start = 1'b0;
        bias_start = 1'b0;
    end
endtask

task pulse_a_ofm;
    input fp16_mode;
    input [4:0] rows;
    begin
        @(negedge clk);
        a_base_addr = 32'h0000_0300;
        a_ofm_stride = 32'd8;
        a_ofm_m_base = 32'd0;
        a_ofm_k_base = 32'd0;
        a_ofm_k_len = 16'd2;
        a_ofm_active_rows = rows;
        a_ofm_fp16_mode = fp16_mode;
        a_ofm_mode = 1'b1;
        a_im2col_mode = 1'b0;
        a_start = 1'b1;
        @(negedge clk);
        a_start = 1'b0;
    end
endtask

task pulse_a_im2col;
    input fp16_mode;
    input [7:0] pad;
    input [15:0] k_len;
    begin
        @(negedge clk);
        a_base_addr = 32'h0000_0800;
        a_im2col_m_index = 32'd0;
        a_im2col_k_len = k_len;
        a_im2col_ih = 16'd2;
        a_im2col_iw = 16'd2;
        a_im2col_cin = 16'd1;
        a_im2col_kh = 16'd2;
        a_im2col_kw = 16'd2;
        a_im2col_oh = 16'd2;
        a_im2col_ow = 16'd2;
        a_im2col_stride_h = 8'd1;
        a_im2col_stride_w = 8'd1;
        a_im2col_pad_h = pad;
        a_im2col_pad_w = pad;
        a_im2col_dilation_h = 8'd1;
        a_im2col_dilation_w = 8'd1;
        a_im2col_fp16_mode = fp16_mode;
        a_im2col_mode = 1'b1;
        a_ofm_mode = 1'b0;
        a_start = 1'b1;
        @(negedge clk);
        a_start = 1'b0;
    end
endtask

task pulse_w_then_ofm_bias;
    begin
        @(negedge clk);
        w_base_addr = 32'h0000_0100;
        w_len_bytes = 16'd8;
        a_base_addr = 32'h0000_0300;
        a_ofm_stride = 32'd8;
        a_ofm_m_base = 32'd0;
        a_ofm_k_base = 32'd0;
        a_ofm_k_len = 16'd2;
        a_ofm_active_rows = 5'd2;
        a_ofm_fp16_mode = 1'b0;
        a_ofm_mode = 1'b1;
        a_im2col_mode = 1'b0;
        bias_addr = 32'h0000_0200;
        w_start = 1'b1;
        a_start = 1'b1;
        bias_start = 1'b1;
        @(negedge clk);
        w_start = 1'b0;
        a_start = 1'b0;
        bias_start = 1'b0;
    end
endtask

task pulse_w_then_im2col_bias;
    begin
        @(negedge clk);
        w_base_addr = 32'h0000_0100;
        w_len_bytes = 16'd8;
        a_base_addr = 32'h0000_0800;
        a_im2col_m_index = 32'd1;
        a_im2col_k_len = 16'd3;
        a_im2col_ih = 16'd2;
        a_im2col_iw = 16'd2;
        a_im2col_cin = 16'd1;
        a_im2col_kh = 16'd2;
        a_im2col_kw = 16'd2;
        a_im2col_oh = 16'd2;
        a_im2col_ow = 16'd2;
        a_im2col_stride_h = 8'd1;
        a_im2col_stride_w = 8'd1;
        a_im2col_pad_h = 8'd0;
        a_im2col_pad_w = 8'd0;
        a_im2col_dilation_h = 8'd1;
        a_im2col_dilation_w = 8'd1;
        a_im2col_fp16_mode = 1'b0;
        a_im2col_mode = 1'b1;
        a_ofm_mode = 1'b0;
        bias_addr = 32'h0000_0200;
        w_start = 1'b1;
        a_start = 1'b1;
        bias_start = 1'b1;
        @(negedge clk);
        w_start = 1'b0;
        a_start = 1'b0;
        bias_start = 1'b0;
    end
endtask

task wait_pulse;
    input [1023:0] name;
    input integer max_cycles;
    input integer kind;
    integer guard;
    reg seen;
    begin
        guard = 0;
        seen = 1'b0;
        while (!seen && guard < max_cycles) begin
            @(posedge clk);
            if (kind == 0 && desc_done) seen = 1'b1;
            if (kind == 1 && bias_done) seen = 1'b1;
            if (kind == 2 && a_done) seen = 1'b1;
            if (kind == 3 && w_done) seen = 1'b1;
            if (kind == 4 && r_done) seen = 1'b1;
            guard = guard + 1;
        end
        if (!seen) begin
            $display("[FAIL] timeout waiting %0s", name);
            errors = errors + 1;
        end
    end
endtask

integer i;
initial begin
    clear_starts();
    w_base_addr = 0; a_base_addr = 0; desc_base_addr = 0; bias_addr = 0; r_base_addr = 0;
    w_len_bytes = 0; a_len_bytes = 0; r_len_bytes = 0;
    w_ppb_full = 0; a_ppb_full = 0; r_fifo_clear = 0; r_fifo_wr_en = 0; r_fifo_din = 0;
    a_ofm_mode = 0; a_im2col_mode = 0; a_ofm_fp16_mode = 0; a_im2col_fp16_mode = 0;
    a_ofm_stride = 0; a_ofm_m_base = 0; a_ofm_k_base = 0; a_ofm_k_len = 0; a_ofm_active_rows = 0;
    a_im2col_m_index = 0; a_im2col_k_len = 0; a_im2col_ih = 0; a_im2col_iw = 0; a_im2col_cin = 0;
    a_im2col_kh = 0; a_im2col_kw = 0; a_im2col_oh = 0; a_im2col_ow = 0;
    a_im2col_stride_h = 1; a_im2col_stride_w = 1; a_im2col_pad_h = 0; a_im2col_pad_w = 0;
    a_im2col_dilation_h = 1; a_im2col_dilation_w = 1;
    errors = 0;
    saw_w_done = 1'b0;
    saw_a_done = 1'b0;
    saw_dma_rd_align = 1'b0;
    saw_dma_wr_align = 1'b0;

    for (i = 0; i < MEM_WORDS; i = i + 1)
        mem[i] = 32'hA000_0000 + i;
    for (i = 0; i < 16; i = i + 1)
        mem[(32'h0000_0040 >> 2) + i] = 32'hD000_0000 + i;
    mem[32'h0000_0200 >> 2] = 32'hB1A5_1234;
    mem[32'h0000_0300 >> 2] = 32'h4433_2211;
    mem[32'h0000_0320 >> 2] = 32'h8877_6655;
    mem[32'h0000_0340 >> 2] = 32'hCCBB_AA99;
    mem[32'h0000_0360 >> 2] = 32'h00FF_EEDD;
    mem[32'h0000_0800 >> 2] = 32'h0403_0201;

    @(posedge rst_n);
    repeat (3) @(posedge clk);

    @(negedge clk);
    w_start = 1'b1;
    a_start = 1'b1;
    a_ofm_mode = 1'b0;
    a_im2col_mode = 1'b0;
    a_len_bytes = 16'd0;
    w_len_bytes = 16'd0;
    @(negedge clk);
    w_start = 1'b0;
    a_start = 1'b0;
    repeat (4) @(posedge clk);
    if (!saw_w_done || !saw_a_done) begin
        $display("[FAIL] zero-length done missing w=%0b a=%0b", saw_w_done, saw_a_done);
        errors = errors + 1;
    end
    saw_w_done = 1'b0;
    saw_a_done = 1'b0;

    pulse_a_ofm_zero();
    repeat (4) @(posedge clk);
    if (!saw_a_done) begin
        $display("[FAIL] zero-length OFM done missing");
        errors = errors + 1;
    end
    saw_a_done = 1'b0;

    pulse_a_im2col_zero();
    repeat (4) @(posedge clk);
    if (!saw_a_done) begin
        $display("[FAIL] zero-length im2col done missing");
        errors = errors + 1;
    end
    saw_a_done = 1'b0;

    pulse_desc(32'h0000_0040);
    wait_pulse("descriptor", 80, 0);
    if (desc_words[31:0] !== 32'hD000_0000 || desc_words[511:480] !== 32'hD000_000F) begin
        $display("[FAIL] descriptor data mismatch first=0x%08h last=0x%08h",
                 desc_words[31:0], desc_words[511:480]);
        errors = errors + 1;
    end

    pulse_bias(32'h0000_0200);
    wait_pulse("bias", 40, 1);
    if (bias_data !== 32'hB1A5_1234) begin
        $display("[FAIL] bias got=0x%08h", bias_data);
        errors = errors + 1;
    end

    saw_dma_rd_align = 1'b0;
    pulse_bad_w_align();
    repeat (8) @(posedge clk);
    if (!saw_dma_rd_align) begin
        $display("[FAIL] bad W alignment error not observed");
        errors = errors + 1;
    end

    saw_dma_rd_align = 1'b0;
    pulse_bad_a_align(1'b0, 1'b0);
    repeat (8) @(posedge clk);
    if (!saw_dma_rd_align) begin
        $display("[FAIL] bad A alignment error not observed");
        errors = errors + 1;
    end

    saw_dma_rd_align = 1'b0;
    pulse_bad_a_align(1'b1, 1'b0);
    repeat (8) @(posedge clk);
    if (!saw_dma_rd_align) begin
        $display("[FAIL] bad OFM alignment error not observed");
        errors = errors + 1;
    end

    saw_dma_rd_align = 1'b0;
    pulse_bad_a_align(1'b0, 1'b1);
    repeat (8) @(posedge clk);
    if (!saw_dma_rd_align) begin
        $display("[FAIL] bad im2col alignment error not observed");
        errors = errors + 1;
    end

    saw_dma_rd_align = 1'b0;
    pulse_bad_bias_align();
    repeat (8) @(posedge clk);
    if (!saw_dma_rd_align) begin
        $display("[FAIL] bad bias alignment error not observed");
        errors = errors + 1;
    end

    pulse_a_direct_bias();
    wait_pulse("direct a", 80, 2);
    wait_pulse("direct bias", 60, 1);

    pulse_a_ofm(1'b0, 5'd3);
    wait_pulse("ofm int8", 120, 2);

    pulse_a_ofm(1'b0, 5'd0);
    wait_pulse("ofm int8 rows0", 120, 2);

    pulse_a_ofm(1'b1, 5'd1);
    wait_pulse("ofm fp16 rows1", 160, 2);

    pulse_a_ofm(1'b1, 5'd4);
    wait_pulse("ofm fp16", 160, 2);

    pulse_a_im2col(1'b0, 8'd1, 16'd4);
    wait_pulse("im2col pad", 120, 2);

    pulse_a_im2col(1'b1, 8'd0, 16'd3);
    wait_pulse("im2col fp16", 140, 2);

    pulse_w_then_ofm_bias();
    wait_pulse("w chained ofm", 120, 3);
    wait_pulse("ofm chained", 160, 2);
    wait_pulse("bias chained ofm", 80, 1);

    pulse_w_then_im2col_bias();
    wait_pulse("w chained", 120, 3);
    wait_pulse("im2col chained", 160, 2);
    wait_pulse("bias chained", 80, 1);

    @(negedge clk);
    r_base_addr = 32'h0000_0A00;
    r_len_bytes = 16'd0;
    r_start = 1'b1;
    @(negedge clk);
    r_start = 1'b0;
    wait_pulse("writeback zero", 40, 4);
    repeat (4) @(posedge clk);

    @(negedge clk);
    saw_dma_wr_align = 1'b0;
    r_base_addr = 32'h0000_0A02;
    r_len_bytes = 16'd8;
    r_start = 1'b1;
    @(negedge clk);
    r_start = 1'b0;
    repeat (8) @(posedge clk);
    if (!saw_dma_wr_align) begin
        $display("[FAIL] writeback alignment error not observed");
        errors = errors + 1;
    end

    @(negedge clk);
    r_fifo_wr_en = 1'b1;
    r_fifo_din = 32'hCAFE_0001;
    @(negedge clk);
    r_fifo_din = 32'hCAFE_0002;
    @(negedge clk);
    r_fifo_wr_en = 1'b0;
    r_base_addr = 32'h0000_0A00;
    r_len_bytes = 16'd8;
    r_start = 1'b1;
    @(negedge clk);
    r_start = 1'b0;
    wait_pulse("writeback normal", 120, 4);

    pulse_desc(32'h0000_0042);
    repeat (8) @(posedge clk);
    if (!saw_dma_rd_align) begin
        $display("[FAIL] descriptor alignment error not observed");
        errors = errors + 1;
    end

    if (a_wr_count < 5) begin
        $display("[FAIL] expected several A writes, got %0d", a_wr_count);
        errors = errors + 1;
    end

    if (errors == 0)
        $display("[PASS] tb_dma_adv_paths: descriptor, OFM, im2col, bias, zero-length paths passed");
    else
        $display("[FAIL] tb_dma_adv_paths errors=%0d", errors);
    $finish;
end

initial begin
    #(CLK_T * 5000);
    $display("[FAIL] tb_dma_adv_paths timeout");
    $finish;
end

endmodule
