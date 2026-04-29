`timescale 1ns/1ps

module tb_dma_read_burst;

localparam DATA_W = 32;
localparam CLK_T  = 10;

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

wire        r_done;
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
    .PPB_DEPTH(64),
    .R_FIFO_DEPTH(64)
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
    .desc_start(1'b0),
    .desc_base_addr(32'd0),
    .desc_done(),
    .desc_words(),
    .r_start(1'b0),
    .r_base_addr(32'd0),
    .r_len_bytes(16'd0),
    .r_done(r_done),
    .r_fifo_clear(1'b0),
    .r_fifo_wr_en(1'b0),
    .r_fifo_din(32'd0),
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

reg [31:0] exp_araddr [0:7];
reg [7:0]  exp_arlen  [0:7];
integer exp_ar_total;
integer ar_seen;
integer errors;

reg        rd_active;
reg [31:0] rd_addr;
reg [7:0]  rd_len;
reg [7:0]  rd_cnt;

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
            end else begin
                if (m_axi_araddr !== exp_araddr[ar_seen] ||
                    m_axi_arlen  !== exp_arlen[ar_seen]  ||
                    m_axi_arsize !== 3'd2 ||
                    m_axi_arburst !== 2'b01) begin
                    $display("[FAIL] AR[%0d] addr=0x%08h len=%0d size=%0d burst=%0d, expected addr=0x%08h len=%0d",
                             ar_seen, m_axi_araddr, m_axi_arlen, m_axi_arsize, m_axi_arburst,
                             exp_araddr[ar_seen], exp_arlen[ar_seen]);
                    errors = errors + 1;
                end
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
                rd_cnt       <= rd_cnt + 1'b1;
                m_axi_rdata  <= rd_addr + ((rd_cnt + 1'b1) << 2);
                m_axi_rlast  <= ((rd_cnt + 1'b1) >= rd_len);
            end
        end else if (rd_active && !m_axi_rvalid) begin
            m_axi_rvalid <= 1'b1;
            m_axi_rdata  <= rd_addr;
            m_axi_rlast  <= (rd_len == 8'd0);
            rd_cnt       <= 8'd0;
        end
    end
end

integer w_wr_count;
integer a_wr_count;
reg [31:0] w_data_base;
reg [31:0] a_data_base;

always @(posedge clk) begin
    if (!rst_n) begin
        w_wr_count <= 0;
        a_wr_count <= 0;
    end else begin
        if (w_ppb_wr_en) begin
            if (w_ppb_wr_data !== (w_data_base + (w_wr_count << 2))) begin
                $display("[FAIL] W data[%0d] got=0x%08h exp=0x%08h",
                         w_wr_count, w_ppb_wr_data, w_data_base + (w_wr_count << 2));
                errors = errors + 1;
            end
            w_wr_count <= w_wr_count + 1;
        end
        if (a_ppb_wr_en) begin
            if (a_ppb_wr_data !== (a_data_base + (a_wr_count << 2))) begin
                $display("[FAIL] A data[%0d] got=0x%08h exp=0x%08h",
                         a_wr_count, a_ppb_wr_data, a_data_base + (a_wr_count << 2));
                errors = errors + 1;
            end
            a_wr_count <= a_wr_count + 1;
        end
    end
end

task wait_for_done;
    input wait_w;
    input wait_a;
    integer guard;
    reg saw_w;
    reg saw_a;
    begin
        saw_w = !wait_w;
        saw_a = !wait_a;
        guard = 0;
        while ((!(saw_w && saw_a)) && guard < 200) begin
            @(posedge clk);
            if (w_done) saw_w = 1'b1;
            if (a_done) saw_a = 1'b1;
            guard = guard + 1;
        end
        if (guard >= 200) begin
            $display("[FAIL] timeout waiting for DMA done");
            errors = errors + 1;
        end
    end
endtask

integer i;

initial begin
    w_start = 0; a_start = 0;
    w_base_addr = 0; a_base_addr = 0;
    w_len_bytes = 0; a_len_bytes = 0;
    w_ppb_full = 0; a_ppb_full = 0;
    m_axi_awready = 1; m_axi_wready = 1;
    m_axi_bresp = 0; m_axi_bvalid = 0;
    m_axi_rresp = 0;
    exp_ar_total = 0;
    errors = 0;
    w_data_base = 0;
    a_data_base = 0;
    for (i = 0; i < 8; i = i + 1) begin
        exp_araddr[i] = 32'd0;
        exp_arlen[i]  = 8'd0;
    end

    @(posedge rst_n);
    repeat (2) @(posedge clk);

    // Case 1: simultaneous W/A request. W is 20 beats -> 16-beat + 4-beat.
    exp_araddr[0] = 32'h0000_1000; exp_arlen[0] = 8'd15;
    exp_araddr[1] = 32'h0000_1040; exp_arlen[1] = 8'd3;
    exp_araddr[2] = 32'h0000_2000; exp_arlen[2] = 8'd7;
    exp_ar_total = 3;
    ar_seen = 0;
    w_wr_count = 0;
    a_wr_count = 0;
    w_data_base = 32'h0000_1000;
    a_data_base = 32'h0000_2000;
    w_base_addr = 32'h0000_1000; w_len_bytes = 16'd80;
    a_base_addr = 32'h0000_2000; a_len_bytes = 16'd32;
    @(negedge clk);
    w_start = 1'b1; a_start = 1'b1;
    @(negedge clk);
    w_start = 1'b0; a_start = 1'b0;
    wait_for_done(1, 1);
    repeat (2) @(posedge clk);
    if (ar_seen !== exp_ar_total || w_wr_count !== 20 || a_wr_count !== 8) begin
        $display("[FAIL] case1 counts ar=%0d/%0d w=%0d a=%0d",
                 ar_seen, exp_ar_total, w_wr_count, a_wr_count);
        errors = errors + 1;
    end

    // Case 2: read burst split at 4KB boundary.
    exp_araddr[0] = 32'h0000_0ff0; exp_arlen[0] = 8'd3;
    exp_araddr[1] = 32'h0000_1000; exp_arlen[1] = 8'd3;
    exp_ar_total = 2;
    ar_seen = 0;
    w_wr_count = 0;
    a_wr_count = 0;
    w_data_base = 32'h0000_0ff0;
    a_data_base = 32'h0000_0000;
    w_base_addr = 32'h0000_0ff0; w_len_bytes = 16'd32;
    a_base_addr = 32'h0000_0000; a_len_bytes = 16'd0;
    @(negedge clk);
    w_start = 1'b1;
    @(negedge clk);
    w_start = 1'b0;
    wait_for_done(1, 0);
    repeat (2) @(posedge clk);
    if (ar_seen !== exp_ar_total || w_wr_count !== 8 || a_wr_count !== 0) begin
        $display("[FAIL] case2 counts ar=%0d/%0d w=%0d a=%0d",
                 ar_seen, exp_ar_total, w_wr_count, a_wr_count);
        errors = errors + 1;
    end

    if (errors == 0)
        $display("[PASS] tb_dma_read_burst: INCR read bursts and 4KB split passed");
    else
        $display("[FAIL] tb_dma_read_burst errors=%0d", errors);
    $finish;
end

endmodule
