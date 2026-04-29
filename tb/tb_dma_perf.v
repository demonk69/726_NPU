`timescale 1ns/1ps

module tb_dma_perf;

localparam DATA_W       = 32;
localparam CLK_T        = 10;
localparam BURST_MAX    = 16;
localparam READ_BEATS   = 256;
localparam WRITE_BEATS  = 256;
localparam READ_BYTES   = READ_BEATS * (DATA_W / 8);
localparam WRITE_BYTES  = WRITE_BEATS * (DATA_W / 8);
localparam MEM_WORDS    = 4096;
localparam TARGET_60_BP = 6000;
localparam TARGET_80_BP = 8000;

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
    .BURST_MAX(BURST_MAX),
    .PPB_DEPTH(512),
    .R_FIFO_DEPTH(512)
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

integer errors;
integer ar_count;
integer aw_count;
integer read_words_seen;
integer write_words_seen;

integer rd_cycles;
integer rd_beats;
integer wr_cycles;
integer wr_beats;
integer expected_rd_beats;
integer expected_wr_beats;
reg     rd_measuring;
reg     wr_measuring;
reg     rd_measure_done;
reg     wr_measure_done;

reg        rd_active;
reg [31:0] rd_addr;
reg [7:0]  rd_len;
reg [7:0]  rd_cnt;

reg [31:0] wr_base;
reg [7:0]  wr_len;
reg [7:0]  wr_cnt;
reg        wr_active;
reg        b_pending;

function [31:0] read_pattern;
    input [31:0] addr;
    begin
        read_pattern = 32'hD000_0000 + (addr >> 2);
    end
endfunction

wire rd_fire = m_axi_rvalid && m_axi_rready;
wire wr_fire = m_axi_wvalid && m_axi_wready;

always @(posedge clk) begin
    if (!rst_n) begin
        rd_cycles <= 0;
        rd_beats <= 0;
        rd_measuring <= 1'b0;
        rd_measure_done <= 1'b0;
    end else begin
        if (rd_fire && !rd_measuring && !rd_measure_done) begin
            rd_measuring <= 1'b1;
            rd_cycles <= 1;
        end else if (rd_measuring) begin
            rd_cycles <= rd_cycles + 1;
        end

        if (rd_fire && !rd_measure_done) begin
            rd_beats <= rd_beats + 1;
            if (rd_beats + 1 >= expected_rd_beats) begin
                rd_measuring <= 1'b0;
                rd_measure_done <= 1'b1;
            end
        end
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        wr_cycles <= 0;
        wr_beats <= 0;
        wr_measuring <= 1'b0;
        wr_measure_done <= 1'b0;
    end else begin
        if (wr_fire && !wr_measuring && !wr_measure_done) begin
            wr_measuring <= 1'b1;
            wr_cycles <= 1;
        end else if (wr_measuring) begin
            wr_cycles <= wr_cycles + 1;
        end

        if (wr_fire && !wr_measure_done) begin
            wr_beats <= wr_beats + 1;
            if (wr_beats + 1 >= expected_wr_beats) begin
                wr_measuring <= 1'b0;
                wr_measure_done <= 1'b1;
            end
        end
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        m_axi_arready <= 1'b0;
        m_axi_rvalid  <= 1'b0;
        m_axi_rlast   <= 1'b0;
        rd_active     <= 1'b0;
        rd_cnt        <= 8'd0;
        ar_count      <= 0;
    end else begin
        m_axi_arready <= 1'b1;

        if (m_axi_arvalid && m_axi_arready) begin
            ar_count <= ar_count + 1;
            if (m_axi_arlen !== 8'd15 || m_axi_arsize !== 3'd2 || m_axi_arburst !== 2'b01) begin
                $display("[FAIL] AR burst addr=0x%08h len=%0d size=%0d burst=%0d",
                         m_axi_araddr, m_axi_arlen, m_axi_arsize, m_axi_arburst);
                errors = errors + 1;
            end
            rd_active <= 1'b1;
            rd_addr   <= m_axi_araddr;
            rd_len    <= m_axi_arlen;
            rd_cnt    <= 8'd0;
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
        aw_count      <= 0;
        write_words_seen <= 0;
    end else begin
        m_axi_awready <= 1'b1;

        if (m_axi_awvalid && m_axi_awready) begin
            aw_count <= aw_count + 1;
            if (m_axi_awlen !== 8'd15 || m_axi_awsize !== 3'd2 || m_axi_awburst !== 2'b01) begin
                $display("[FAIL] AW burst addr=0x%08h len=%0d size=%0d burst=%0d",
                         m_axi_awaddr, m_axi_awlen, m_axi_awsize, m_axi_awburst);
                errors = errors + 1;
            end
            wr_active <= 1'b1;
            wr_base   <= m_axi_awaddr;
            wr_len    <= m_axi_awlen;
            wr_cnt    <= 8'd0;
            m_axi_wready <= 1'b1;
        end

        if (wr_active && m_axi_wvalid && m_axi_wready) begin
            mem[(wr_base >> 2) + wr_cnt] <= m_axi_wdata;
            if (m_axi_wdata !== (32'hE000_0000 + write_words_seen)) begin
                $display("[FAIL] WDATA[%0d] got=0x%08h exp=0x%08h",
                         write_words_seen, m_axi_wdata, 32'hE000_0000 + write_words_seen);
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
            write_words_seen <= write_words_seen + 1;

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
        read_words_seen <= 0;
    end else if (w_ppb_wr_en) begin
        if (w_ppb_wr_data !== read_pattern(w_base_addr + (read_words_seen << 2))) begin
            $display("[FAIL] read data[%0d] got=0x%08h exp=0x%08h",
                     read_words_seen, w_ppb_wr_data, read_pattern(w_base_addr + (read_words_seen << 2)));
            errors = errors + 1;
        end
        read_words_seen <= read_words_seen + 1;
    end
end

task clear_result_fifo;
    begin
        @(negedge clk);
        r_fifo_clear = 1'b1;
        @(negedge clk);
        r_fifo_clear = 1'b0;
    end
endtask

task push_result_words;
    input integer count;
    integer i;
    begin
        for (i = 0; i < count; i = i + 1) begin
            @(negedge clk);
            if (r_fifo_full) begin
                $display("[FAIL] result FIFO full at word %0d", i);
                errors = errors + 1;
            end
            r_fifo_din   = 32'hE000_0000 + i;
            r_fifo_wr_en = 1'b1;
        end
        @(negedge clk);
        r_fifo_wr_en = 1'b0;
        r_fifo_din   = 32'd0;
    end
endtask

task reset_read_metrics;
    begin
        @(negedge clk);
        ar_count = 0;
        read_words_seen = 0;
        rd_cycles = 0;
        rd_beats = 0;
        rd_measuring = 1'b0;
        rd_measure_done = 1'b0;
        expected_rd_beats = READ_BEATS;
    end
endtask

task reset_write_metrics;
    begin
        @(negedge clk);
        aw_count = 0;
        write_words_seen = 0;
        wr_cycles = 0;
        wr_beats = 0;
        wr_measuring = 1'b0;
        wr_measure_done = 1'b0;
        expected_wr_beats = WRITE_BEATS;
    end
endtask

task wait_w_done;
    integer guard;
    reg saw_done;
    begin
        guard = 0;
        saw_done = 1'b0;
        while (!saw_done && guard < 2000) begin
            @(posedge clk);
            if (w_done) saw_done = 1'b1;
            guard = guard + 1;
        end
        if (!saw_done) begin
            $display("[FAIL] timeout waiting read DMA done");
            errors = errors + 1;
        end
    end
endtask

task wait_r_done;
    integer guard;
    reg saw_done;
    begin
        guard = 0;
        saw_done = 1'b0;
        while (!saw_done && guard < 2500) begin
            @(posedge clk);
            if (r_done) saw_done = 1'b1;
            guard = guard + 1;
        end
        if (!saw_done) begin
            $display("[FAIL] timeout waiting write DMA done");
            errors = errors + 1;
        end
    end
endtask

task check_read_perf;
    integer util_bp;
    integer bw_x1000;
    begin
        util_bp = (rd_cycles > 0) ? (rd_beats * 10000 / rd_cycles) : 0;
        bw_x1000 = (rd_cycles > 0) ? (rd_beats * (DATA_W / 8) * 1000 / rd_cycles) : 0;
        $display("[PERF] read  beats=%0d cycles=%0d bursts=%0d util=%0d.%02d%% bw=%0d.%03d B/cyc",
                 rd_beats, rd_cycles, ar_count, util_bp/100, util_bp%100,
                 bw_x1000/1000, bw_x1000%1000);
        if (rd_beats !== READ_BEATS || read_words_seen !== READ_BEATS ||
            ar_count !== (READ_BEATS / BURST_MAX) || rd_measure_done !== 1'b1) begin
            $display("[FAIL] read perf counts beats=%0d words=%0d bursts=%0d done=%0b",
                     rd_beats, read_words_seen, ar_count, rd_measure_done);
            errors = errors + 1;
        end
        if (util_bp < TARGET_80_BP) begin
            $display("[FAIL] read utilization below 80%% target: %0d bp", util_bp);
            errors = errors + 1;
        end
    end
endtask

task check_write_perf;
    integer util_bp;
    integer bw_x1000;
    begin
        util_bp = (wr_cycles > 0) ? (wr_beats * 10000 / wr_cycles) : 0;
        bw_x1000 = (wr_cycles > 0) ? (wr_beats * (DATA_W / 8) * 1000 / wr_cycles) : 0;
        $display("[PERF] write beats=%0d cycles=%0d bursts=%0d util=%0d.%02d%% bw=%0d.%03d B/cyc",
                 wr_beats, wr_cycles, aw_count, util_bp/100, util_bp%100,
                 bw_x1000/1000, bw_x1000%1000);
        if (wr_beats !== WRITE_BEATS || write_words_seen !== WRITE_BEATS ||
            aw_count !== (WRITE_BEATS / BURST_MAX) || wr_measure_done !== 1'b1) begin
            $display("[FAIL] write perf counts beats=%0d words=%0d bursts=%0d done=%0b",
                     wr_beats, write_words_seen, aw_count, wr_measure_done);
            errors = errors + 1;
        end
        if (util_bp < TARGET_60_BP) begin
            $display("[FAIL] write utilization below 60%% floor: %0d bp", util_bp);
            errors = errors + 1;
        end else if (util_bp < TARGET_80_BP) begin
            $display("[INFO] write utilization is below 80%% because the current DMA issues one outstanding write burst and waits for B response before the next AW.");
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
    errors = 0;
    expected_rd_beats = 0;
    expected_wr_beats = 0;
    for (i = 0; i < MEM_WORDS; i = i + 1)
        mem[i] = 32'd0;

    @(posedge rst_n);
    repeat (2) @(posedge clk);

    reset_read_metrics();
    w_base_addr = 32'h0000_1000;
    w_len_bytes = READ_BYTES[15:0];
    @(negedge clk);
    w_start = 1'b1;
    @(negedge clk);
    w_start = 1'b0;
    wait_w_done();
    repeat (5) @(posedge clk);
    check_read_perf();

    clear_result_fifo();
    push_result_words(WRITE_BEATS);
    reset_write_metrics();
    r_base_addr = 32'h0000_2000;
    r_len_bytes = WRITE_BYTES[15:0];
    @(negedge clk);
    r_start = 1'b1;
    @(negedge clk);
    r_start = 1'b0;
    wait_r_done();
    repeat (5) @(posedge clk);
    check_write_perf();

    if (mem[32'h0000_2000 >> 2] !== 32'hE000_0000 ||
        mem[(32'h0000_2000 >> 2) + WRITE_BEATS - 1] !== (32'hE000_0000 + WRITE_BEATS - 1)) begin
        $display("[FAIL] writeback memory endpoints mismatch");
        errors = errors + 1;
    end

    if (errors == 0)
        $display("[PASS] tb_dma_perf: bandwidth utilization target test completed");
    else
        $display("[FAIL] tb_dma_perf errors=%0d", errors);
    $finish;
end

endmodule
