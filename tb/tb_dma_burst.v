`timescale 1ns/1ps

module tb_dma_burst;

localparam DATA_W    = 32;
localparam CLK_T     = 10;
localparam MEM_WORDS = 8192;

reg clk = 1'b0;
always #(CLK_T/2) clk = ~clk;

reg rst_n = 1'b0;
initial begin
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
end

reg         w_start;
reg [31:0]  w_base_addr;
reg [15:0]  w_len_bytes;
wire        w_done;
wire        w_ppb_wr_en;
wire [31:0] w_ppb_wr_data;
reg         w_ppb_full;

reg         a_start;
reg [31:0]  a_base_addr;
reg [15:0]  a_len_bytes;
wire        a_done;
wire        a_ppb_wr_en;
wire [31:0] a_ppb_wr_data;
reg         a_ppb_full;

reg         r_start;
reg [31:0]  r_base_addr;
reg [15:0]  r_len_bytes;
wire        r_done;
reg         r_fifo_clear;
reg         r_fifo_wr_en;
reg  [31:0] r_fifo_din;
wire        r_fifo_full;

wire [31:0] m_axi_awaddr;
wire [7:0]  m_axi_awlen;
wire [2:0]  m_axi_awsize;
wire [1:0]  m_axi_awburst;
wire        m_axi_awvalid;
reg         m_axi_awready;
wire [31:0] m_axi_wdata;
wire [3:0]  m_axi_wstrb;
wire        m_axi_wlast;
wire        m_axi_wvalid;
reg         m_axi_wready;
reg  [1:0]  m_axi_bresp;
reg         m_axi_bvalid;
wire        m_axi_bready;

wire [31:0] m_axi_araddr;
wire [7:0]  m_axi_arlen;
wire [2:0]  m_axi_arsize;
wire [1:0]  m_axi_arburst;
wire        m_axi_arvalid;
reg         m_axi_arready;
reg  [31:0] m_axi_rdata;
reg  [1:0]  m_axi_rresp;
reg         m_axi_rvalid;
wire        m_axi_rready;
reg         m_axi_rlast;

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
    .a_ofm_mode(1'b0),
    .a_ofm_stride(32'd0),
    .a_ofm_m_base(32'd0),
    .a_ofm_k_base(32'd0),
    .a_ofm_k_len(16'd0),
    .a_ofm_active_rows(3'd0),
    .a_ofm_fp16_mode(1'b0),
    .a_im2col_mode(1'b0),
    .a_im2col_m_index(32'd0),
    .a_im2col_k_len(16'd0),
    .a_im2col_ih(16'd0),
    .a_im2col_iw(16'd0),
    .a_im2col_cin(16'd0),
    .a_im2col_kh(16'd0),
    .a_im2col_kw(16'd0),
    .a_im2col_oh(16'd0),
    .a_im2col_ow(16'd0),
    .a_im2col_stride_h(8'd1),
    .a_im2col_stride_w(8'd1),
    .a_im2col_pad_h(8'd0),
    .a_im2col_pad_w(8'd0),
    .a_im2col_dilation_h(8'd1),
    .a_im2col_dilation_w(8'd1),
    .a_im2col_fp16_mode(1'b0),
    .desc_start(1'b0),
    .desc_base_addr(32'd0),
    .desc_done(),
    .desc_words(),
    .r_start(r_start),
    .r_base_addr(r_base_addr),
    .r_len_bytes(r_len_bytes),
    .r_done(r_done),
    .r_fifo_clear(r_fifo_clear),
    .r_fifo_wr_en(r_fifo_wr_en),
    .r_fifo_din(r_fifo_din),
    .r_fifo_full(r_fifo_full),
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

reg [31:0] exp_araddr [0:7];
reg [7:0]  exp_arlen  [0:7];
reg [31:0] exp_awaddr [0:7];
reg [7:0]  exp_awlen  [0:7];
integer exp_ar_total;
integer exp_aw_total;
integer ar_seen;
integer aw_seen;
integer write_word_seen;
integer errors;

reg        rd_active;
reg [31:0] rd_addr;
reg [7:0]  rd_len;
reg [7:0]  rd_cnt;

reg [31:0] wr_base;
reg [7:0]  wr_len;
reg [7:0]  wr_cnt;
reg        wr_active;
reg        b_pending;

integer w_wr_count;
integer a_wr_count;
reg [31:0] exp_write_base;

function [31:0] read_pattern;
    input [31:0] addr;
    begin
        read_pattern = 32'hC000_0000 + (addr >> 2);
    end
endfunction

always @(posedge clk) begin
    if (!rst_n) begin
        m_axi_arready <= 1'b0;
        m_axi_rvalid  <= 1'b0;
        m_axi_rlast   <= 1'b0;
        rd_active     <= 1'b0;
        rd_cnt        <= 8'd0;
        ar_seen       <= 0;
    end else begin
        m_axi_arready <= 1'b1;

        if (m_axi_arvalid && m_axi_arready) begin
            if (ar_seen >= exp_ar_total) begin
                $display("[FAIL] unexpected AR addr=0x%08h len=%0d", m_axi_araddr, m_axi_arlen);
                errors = errors + 1;
            end else if (m_axi_araddr !== exp_araddr[ar_seen] ||
                         m_axi_arlen  !== exp_arlen[ar_seen]  ||
                         m_axi_arsize !== 3'd2 ||
                         m_axi_arburst !== 2'b01) begin
                $display("[FAIL] AR[%0d] addr=0x%08h len=%0d size=%0d burst=%0d, expected addr=0x%08h len=%0d",
                         ar_seen, m_axi_araddr, m_axi_arlen, m_axi_arsize, m_axi_arburst,
                         exp_araddr[ar_seen], exp_arlen[ar_seen]);
                errors = errors + 1;
            end
            rd_active <= 1'b1;
            rd_addr   <= m_axi_araddr;
            rd_len    <= m_axi_arlen;
            rd_cnt    <= 8'd0;
            ar_seen   <= ar_seen + 1;
        end

        if (m_axi_rvalid && m_axi_rready) begin
            if (m_axi_rlast) begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast  <= 1'b0;
                rd_active    <= 1'b0;
            end else begin
                rd_cnt      <= rd_cnt + 1'b1;
                m_axi_rdata <= read_pattern(rd_addr + ((rd_cnt + 1'b1) << 2));
                m_axi_rlast <= ((rd_cnt + 1'b1) >= rd_len);
            end
        end else if (rd_active && !m_axi_rvalid) begin
            m_axi_rvalid <= 1'b1;
            m_axi_rdata  <= read_pattern(rd_addr);
            m_axi_rlast  <= (rd_len == 8'd0);
            rd_cnt       <= 8'd0;
        end
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        m_axi_awready <= 1'b0;
        m_axi_wready  <= 1'b0;
        m_axi_bvalid  <= 1'b0;
        wr_active     <= 1'b0;
        b_pending     <= 1'b0;
        wr_cnt        <= 8'd0;
        aw_seen       <= 0;
        write_word_seen <= 0;
    end else begin
        m_axi_awready <= 1'b1;

        if (m_axi_awvalid && m_axi_awready) begin
            if (aw_seen >= exp_aw_total) begin
                $display("[FAIL] unexpected AW addr=0x%08h len=%0d", m_axi_awaddr, m_axi_awlen);
                errors = errors + 1;
            end else if (m_axi_awaddr !== exp_awaddr[aw_seen] ||
                         m_axi_awlen  !== exp_awlen[aw_seen]  ||
                         m_axi_awsize !== 3'd2 ||
                         m_axi_awburst !== 2'b01) begin
                $display("[FAIL] AW[%0d] addr=0x%08h len=%0d size=%0d burst=%0d, expected addr=0x%08h len=%0d",
                         aw_seen, m_axi_awaddr, m_axi_awlen, m_axi_awsize, m_axi_awburst,
                         exp_awaddr[aw_seen], exp_awlen[aw_seen]);
                errors = errors + 1;
            end
            wr_active <= 1'b1;
            wr_base   <= m_axi_awaddr;
            wr_len    <= m_axi_awlen;
            wr_cnt    <= 8'd0;
            m_axi_wready <= 1'b1;
            aw_seen <= aw_seen + 1;
        end

        if (wr_active && m_axi_wvalid && m_axi_wready) begin
            mem[(wr_base >> 2) + wr_cnt] <= m_axi_wdata;
            if (m_axi_wdata !== (exp_write_base + write_word_seen)) begin
                $display("[FAIL] WDATA[%0d] got=0x%08h exp=0x%08h",
                         write_word_seen, m_axi_wdata, exp_write_base + write_word_seen);
                errors = errors + 1;
            end
            if (m_axi_wstrb !== 4'hF) begin
                $display("[FAIL] WSTRB got=0x%0h", m_axi_wstrb);
                errors = errors + 1;
            end
            if (m_axi_wlast !== (wr_cnt == wr_len)) begin
                $display("[FAIL] WLAST at beat %0d len=%0d got=%0b", wr_cnt, wr_len, m_axi_wlast);
                errors = errors + 1;
            end
            write_word_seen <= write_word_seen + 1;

            if (m_axi_wlast) begin
                wr_active <= 1'b0;
                m_axi_wready <= 1'b0;
                b_pending <= 1'b1;
            end else begin
                wr_cnt <= wr_cnt + 1'b1;
            end
        end

        if (b_pending && !m_axi_bvalid) begin
            m_axi_bvalid <= 1'b1;
            b_pending <= 1'b0;
        end else if (m_axi_bvalid && m_axi_bready) begin
            m_axi_bvalid <= 1'b0;
        end
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        w_wr_count <= 0;
        a_wr_count <= 0;
    end else begin
        if (w_ppb_wr_en) begin
            if (w_ppb_wr_data !== read_pattern(w_base_addr + (w_wr_count << 2))) begin
                $display("[FAIL] W read data[%0d] got=0x%08h exp=0x%08h",
                         w_wr_count, w_ppb_wr_data, read_pattern(w_base_addr + (w_wr_count << 2)));
                errors = errors + 1;
            end
            w_wr_count <= w_wr_count + 1;
        end
        if (a_ppb_wr_en) begin
            if (a_ppb_wr_data !== read_pattern(a_base_addr + (a_wr_count << 2))) begin
                $display("[FAIL] A read data[%0d] got=0x%08h exp=0x%08h",
                         a_wr_count, a_ppb_wr_data, read_pattern(a_base_addr + (a_wr_count << 2)));
                errors = errors + 1;
            end
            a_wr_count <= a_wr_count + 1;
        end
    end
end

task push_result_words;
    input integer count;
    input [31:0] base;
    integer i;
    begin
        for (i = 0; i < count; i = i + 1) begin
            @(negedge clk);
            if (r_fifo_full) begin
                $display("[FAIL] result FIFO full while pushing word %0d", i);
                errors = errors + 1;
            end
            r_fifo_din   = base + i;
            r_fifo_wr_en = 1'b1;
        end
        @(negedge clk);
        r_fifo_wr_en = 1'b0;
        r_fifo_din   = 32'd0;
    end
endtask

task pulse_mixed_start;
    begin
        @(negedge clk);
        w_start = 1'b1;
        a_start = 1'b1;
        r_start = 1'b1;
        @(negedge clk);
        w_start = 1'b0;
        a_start = 1'b0;
        r_start = 1'b0;
    end
endtask

task wait_all_done;
    integer guard;
    reg saw_w;
    reg saw_a;
    reg saw_r;
    begin
        guard = 0;
        saw_w = 1'b0;
        saw_a = 1'b0;
        saw_r = 1'b0;
        while (!(saw_w && saw_a && saw_r) && guard < 800) begin
            @(posedge clk);
            if (w_done) saw_w = 1'b1;
            if (a_done) saw_a = 1'b1;
            if (r_done) saw_r = 1'b1;
            guard = guard + 1;
        end
        if (!(saw_w && saw_a && saw_r)) begin
            $display("[FAIL] timeout waiting done w=%0b a=%0b r=%0b", saw_w, saw_a, saw_r);
            errors = errors + 1;
        end
    end
endtask

task expect_mem;
    input [31:0] addr;
    input [31:0] exp;
    begin
        if (mem[addr >> 2] !== exp) begin
            $display("[FAIL] mem[0x%08h] got=0x%08h exp=0x%08h", addr, mem[addr >> 2], exp);
            errors = errors + 1;
        end
    end
endtask

integer i;

initial begin
    w_start = 0; w_base_addr = 0; w_len_bytes = 0; w_ppb_full = 0;
    a_start = 0; a_base_addr = 0; a_len_bytes = 0; a_ppb_full = 0;
    r_start = 0; r_base_addr = 0; r_len_bytes = 0;
    r_fifo_clear = 0; r_fifo_wr_en = 0; r_fifo_din = 0;
    m_axi_awready = 0; m_axi_wready = 0;
    m_axi_bresp = 0; m_axi_bvalid = 0;
    m_axi_arready = 0; m_axi_rdata = 0; m_axi_rresp = 0; m_axi_rvalid = 0; m_axi_rlast = 0;
    exp_ar_total = 0; exp_aw_total = 0;
    ar_seen = 0; aw_seen = 0; write_word_seen = 0;
    w_wr_count = 0; a_wr_count = 0;
    exp_write_base = 0;
    errors = 0;

    for (i = 0; i < MEM_WORDS; i = i + 1)
        mem[i] = 32'd0;
    for (i = 0; i < 8; i = i + 1) begin
        exp_araddr[i] = 32'd0;
        exp_arlen[i]  = 8'd0;
        exp_awaddr[i] = 32'd0;
        exp_awlen[i]  = 8'd0;
    end

    @(posedge rst_n);
    repeat (2) @(posedge clk);

    // Mixed transfer:
    //   W read: 16 beats
    //   A read: 8 beats
    //   R write: 24 beats -> 16-beat burst + 8-beat burst
    exp_araddr[0] = 32'h0000_1000; exp_arlen[0] = 8'd15;
    exp_araddr[1] = 32'h0000_2000; exp_arlen[1] = 8'd7;
    exp_ar_total = 2;

    exp_awaddr[0] = 32'h0000_3000; exp_awlen[0] = 8'd15;
    exp_awaddr[1] = 32'h0000_3040; exp_awlen[1] = 8'd7;
    exp_aw_total = 2;

    w_base_addr = 32'h0000_1000; w_len_bytes = 16'd64;
    a_base_addr = 32'h0000_2000; a_len_bytes = 16'd32;
    r_base_addr = 32'h0000_3000; r_len_bytes = 16'd96;
    exp_write_base = 32'hA400_0000;

    push_result_words(24, exp_write_base);
    pulse_mixed_start();
    wait_all_done();
    repeat (4) @(posedge clk);

    if (ar_seen !== exp_ar_total || aw_seen !== exp_aw_total ||
        w_wr_count !== 16 || a_wr_count !== 8 || write_word_seen !== 24) begin
        $display("[FAIL] counts ar=%0d/%0d aw=%0d/%0d w=%0d a=%0d wr=%0d",
                 ar_seen, exp_ar_total, aw_seen, exp_aw_total,
                 w_wr_count, a_wr_count, write_word_seen);
        errors = errors + 1;
    end

    expect_mem(32'h0000_3000, 32'hA400_0000);
    expect_mem(32'h0000_303C, 32'hA400_000F);
    expect_mem(32'h0000_3040, 32'hA400_0010);
    expect_mem(32'h0000_305C, 32'hA400_0017);

    if (errors == 0)
        $display("[PASS] tb_dma_burst: mixed 8/16-beat read/write burst data passed");
    else
        $display("[FAIL] tb_dma_burst errors=%0d", errors);
    $finish;
end

endmodule
