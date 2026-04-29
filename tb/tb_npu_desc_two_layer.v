`timescale 1ns/1ps

module tb_npu_desc_two_layer;

localparam DATA_W  = 16;
localparam ACC_W   = 32;
localparam CLK_T   = 10;
localparam DRAM_SZ = 2048;

localparam REG_CTRL       = 32'h00;
localparam REG_STATUS     = 32'h04;
localparam REG_DESC_BASE  = 32'h40;
localparam REG_DESC_COUNT = 32'h44;

localparam CTRL_START_DESC = 32'h81;

localparam M_DIM = 4;
localparam N_DIM = 4;
localparam K_DIM = 4;

localparam DESC0_ADDR = 32'h00000040;
localparam DESC1_ADDR = 32'h00000080;
localparam W0_ADDR    = 32'h00000100;
localparam A0_ADDR    = 32'h00000140;
localparam R0_ADDR    = 32'h00000200;
localparam W1_ADDR    = 32'h00000300;
localparam A1_ADDR    = 32'h00000340;
localparam R1_ADDR    = 32'h00000400;

localparam DESC_CTRL_COMMON = (32'h1 << 28) | // VERSION=1
                              (32'h1 << 18) | // LAST_K
                              (32'h1 << 17) | // FIRST_K
                              (32'h1 << 16) | // TILE_PACKED
                              (32'h1 <<  8) | // DATAFLOW=OS
                              32'h1;          // OP=GEMM_TILEPACK
localparam DESC_CTRL_LAST = DESC_CTRL_COMMON |
                            (32'h1 << 20) |   // IRQ_EN
                            (32'h1 << 19);    // LAST_LAYER

reg clk = 1'b0;
always #(CLK_T/2) clk = ~clk;

reg rst_n = 1'b0;
initial begin
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
end

reg  [31:0] s_awaddr, s_wdata;
reg  [3:0]  s_wstrb;
reg         s_awvalid, s_wvalid, s_bready;
wire        s_awready, s_wready, s_bvalid;
wire [1:0]  s_bresp;

reg  [31:0] s_araddr;
reg         s_arvalid, s_rready;
wire        s_arready, s_rvalid;
wire [31:0] s_rdata;
wire [1:0]  s_rresp;

wire [31:0] m_awaddr;
wire [7:0]  m_awlen;
wire [2:0]  m_awsize;
wire [1:0]  m_awburst;
wire        m_awvalid;
reg         m_awready;
wire [31:0] m_wdata;
wire [3:0]  m_wstrb;
wire        m_wlast;
wire        m_wvalid;
reg         m_wready;
wire [1:0]  m_bresp;
reg         m_bvalid;
wire        m_bready;

wire [31:0] m_araddr;
wire [7:0]  m_arlen;
wire [2:0]  m_arsize;
wire [1:0]  m_arburst;
wire        m_arvalid;
reg         m_arready;
reg  [31:0] m_rdata;
wire [1:0]  m_rresp;
reg         m_rvalid;
wire        m_rready;
reg         m_rlast;

wire npu_irq;

reg [31:0] dram [0:DRAM_SZ-1];
reg [31:0] expected0 [0:M_DIM*N_DIM-1];
reg [31:0] expected1 [0:M_DIM*N_DIM-1];

integer pass_cnt;
integer fail_cnt;
integer ar_count;
integer aw_count;
reg [31:0] ar_addr_seen [0:15];
reg [7:0]  ar_len_seen  [0:15];
reg [31:0] aw_addr_seen [0:15];
reg [7:0]  aw_len_seen  [0:15];

npu_top #(
    .DATA_W(DATA_W),
    .ACC_W (ACC_W)
) u_npu (
    .sys_clk       (clk),
    .sys_rst_n     (rst_n),
    .s_axi_awaddr  (s_awaddr),
    .s_axi_awvalid (s_awvalid),
    .s_axi_awready (s_awready),
    .s_axi_wdata   (s_wdata),
    .s_axi_wstrb   (s_wstrb),
    .s_axi_wvalid  (s_wvalid),
    .s_axi_wready  (s_wready),
    .s_axi_bresp   (s_bresp),
    .s_axi_bvalid  (s_bvalid),
    .s_axi_bready  (s_bready),
    .s_axi_araddr  (s_araddr),
    .s_axi_arvalid (s_arvalid),
    .s_axi_arready (s_arready),
    .s_axi_rdata   (s_rdata),
    .s_axi_rresp   (s_rresp),
    .s_axi_rvalid  (s_rvalid),
    .s_axi_rready  (s_rready),
    .m_axi_awaddr  (m_awaddr),
    .m_axi_awlen   (m_awlen),
    .m_axi_awsize  (m_awsize),
    .m_axi_awburst (m_awburst),
    .m_axi_awvalid (m_awvalid),
    .m_axi_awready (m_awready),
    .m_axi_wdata   (m_wdata),
    .m_axi_wstrb   (m_wstrb),
    .m_axi_wlast   (m_wlast),
    .m_axi_wvalid  (m_wvalid),
    .m_axi_wready  (m_wready),
    .m_axi_bresp   (m_bresp),
    .m_axi_bvalid  (m_bvalid),
    .m_axi_bready  (m_bready),
    .m_axi_araddr  (m_araddr),
    .m_axi_arlen   (m_arlen),
    .m_axi_arsize  (m_arsize),
    .m_axi_arburst (m_arburst),
    .m_axi_arvalid (m_arvalid),
    .m_axi_arready (m_arready),
    .m_axi_rdata   (m_rdata),
    .m_axi_rresp   (m_rresp),
    .m_axi_rvalid  (m_rvalid),
    .m_axi_rready  (m_rready),
    .m_axi_rlast   (m_rlast),
    .npu_irq       (npu_irq)
);

assign m_bresp = 2'b00;
assign m_rresp = 2'b00;

function integer a_val;
    input integer layer;
    input integer r;
    input integer k;
    begin
        a_val = (((r + 1) * (k + 2) + layer) % 9) - 4;
    end
endfunction

function integer b_val;
    input integer layer;
    input integer k;
    input integer c;
    begin
        b_val = (((k + 1) * (c + 3) + (2 * layer)) % 11) - 5;
    end
endfunction

function [31:0] pack4_int8;
    input integer v0;
    input integer v1;
    input integer v2;
    input integer v3;
    begin
        pack4_int8 = ((v0 & 8'hff)      ) |
                     ((v1 & 8'hff) <<  8) |
                     ((v2 & 8'hff) << 16) |
                     ((v3 & 8'hff) << 24);
    end
endfunction

task write_desc;
    input [31:0] base;
    input [31:0] ctrl;
    input [31:0] a_addr;
    input [31:0] w_addr;
    input [31:0] r_addr;
    input [31:0] next_addr;
    integer b;
    begin
        b = base >> 2;
        dram[b +  0] = ctrl;
        dram[b +  1] = M_DIM;
        dram[b +  2] = N_DIM;
        dram[b +  3] = K_DIM;
        dram[b +  4] = a_addr;
        dram[b +  5] = w_addr;
        dram[b +  6] = 32'd0;
        dram[b +  7] = 32'd0;
        dram[b +  8] = r_addr;
        dram[b +  9] = 32'd0;
        dram[b + 10] = 32'd0;
        dram[b + 11] = 32'd0;
        dram[b + 12] = 32'd0;
        dram[b + 13] = 32'd0;
        dram[b + 14] = 32'd0;
        dram[b + 15] = next_addr;
    end
endtask

task fill_layer;
    input integer layer;
    input [31:0] a_addr;
    input [31:0] w_addr;
    input [31:0] r_addr;
    integer r;
    integer c;
    integer k;
    integer acc;
    begin
        for (k = 0; k < K_DIM; k = k + 1) begin
            dram[(w_addr >> 2) + k] = pack4_int8(
                b_val(layer, k, 0), b_val(layer, k, 1),
                b_val(layer, k, 2), b_val(layer, k, 3));
            dram[(a_addr >> 2) + k] = pack4_int8(
                a_val(layer, 0, k), a_val(layer, 1, k),
                a_val(layer, 2, k), a_val(layer, 3, k));
        end

        for (r = 0; r < M_DIM; r = r + 1) begin
            for (c = 0; c < N_DIM; c = c + 1) begin
                acc = 0;
                for (k = 0; k < K_DIM; k = k + 1)
                    acc = acc + a_val(layer, r, k) * b_val(layer, k, c);
                if (layer == 0)
                    expected0[r*N_DIM + c] = acc;
                else
                    expected1[r*N_DIM + c] = acc;
                dram[(r_addr >> 2) + r*N_DIM + c] = 32'hdead_0000 + r*N_DIM + c;
            end
        end
    end
endtask

task axi_write;
    input [31:0] addr;
    input [31:0] data;
    integer guard;
    begin
        s_awaddr  <= addr;
        s_awvalid <= 1'b1;
        s_wdata   <= data;
        s_wstrb   <= 4'hF;
        s_wvalid  <= 1'b1;
        s_bready  <= 1'b1;
        @(posedge clk);
        while (!s_awready) @(posedge clk);
        @(posedge clk);
        s_awvalid <= 1'b0;
        s_wvalid  <= 1'b0;
        guard = 0;
        while (!s_bvalid && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        s_bready <= 1'b0;
        if (guard >= 100) begin
            $display("[FAIL] AXI-Lite write timeout at 0x%08h", addr);
            $fatal;
        end
    end
endtask

task axi_read;
    input  [31:0] addr;
    output [31:0] data;
    integer guard;
    begin
        s_araddr  <= addr;
        s_arvalid <= 1'b1;
        s_rready  <= 1'b1;
        @(posedge clk);
        while (!s_arready) @(posedge clk);
        @(posedge clk);
        s_arvalid <= 1'b0;
        guard = 0;
        while (!s_rvalid && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        data = s_rdata;
        s_rready <= 1'b0;
        if (guard >= 100) begin
            $display("[FAIL] AXI-Lite read timeout at 0x%08h", addr);
            $fatal;
        end
    end
endtask

task wait_done;
    input [31:0] timeout;
    integer guard;
    reg [31:0] status;
    begin
        status = 32'd0;
        guard = 0;
        while (!status[1] && guard < timeout) begin
            axi_read(REG_STATUS, status);
            @(posedge clk);
            guard = guard + 1;
        end
        if (!status[1]) begin
            $display("[FAIL] descriptor run timeout state=%0d dma=%0d",
                     u_npu.u_ctrl.state, u_npu.u_dma.dma_state);
            $fatal;
        end
        axi_write(REG_CTRL, 32'h0);
    end
endtask

task expect_read;
    input integer idx;
    input [31:0] exp_addr;
    input [7:0]  exp_len;
    begin
        if (ar_addr_seen[idx] !== exp_addr || ar_len_seen[idx] !== exp_len) begin
            $display("[FAIL] read%0d addr=%08h len=%0d, expected addr=%08h len=%0d",
                     idx, ar_addr_seen[idx], ar_len_seen[idx], exp_addr, exp_len);
            fail_cnt = fail_cnt + 1;
        end else begin
            pass_cnt = pass_cnt + 1;
        end
    end
endtask

task expect_write;
    input integer idx;
    input [31:0] exp_addr;
    begin
        if (aw_addr_seen[idx] !== exp_addr || aw_len_seen[idx] !== 8'd3) begin
            $display("[FAIL] write%0d addr=%08h len=%0d, expected addr=%08h len=3",
                     idx, aw_addr_seen[idx], aw_len_seen[idx], exp_addr);
            fail_cnt = fail_cnt + 1;
        end else begin
            pass_cnt = pass_cnt + 1;
        end
    end
endtask

task check_result;
    input integer layer;
    input integer idx;
    reg [31:0] got;
    reg [31:0] exp;
    reg [31:0] base;
    begin
        base = (layer == 0) ? R0_ADDR : R1_ADDR;
        got = dram[(base >> 2) + idx];
        exp = (layer == 0) ? expected0[idx] : expected1[idx];
        if (got !== exp) begin
            $display("[FAIL] layer%0d C[%0d][%0d] got=%0d (0x%08h), expected=%0d (0x%08h)",
                     layer, idx / N_DIM, idx % N_DIM,
                     $signed(got), got, $signed(exp), exp);
            fail_cnt = fail_cnt + 1;
        end else begin
            pass_cnt = pass_cnt + 1;
        end
    end
endtask

reg [31:0] rd_base;
reg [7:0]  rd_len;
reg [7:0]  rd_cnt;
reg        rd_active;

always @(posedge clk) begin
    if (!rst_n) begin
        m_arready <= 1'b0;
        m_rvalid  <= 1'b0;
        m_rlast   <= 1'b0;
        rd_active <= 1'b0;
        rd_cnt    <= 8'd0;
        ar_count  <= 0;
    end else begin
        m_arready <= 1'b1;
        if (m_arvalid && m_arready && ar_count < 16) begin
            ar_addr_seen[ar_count] <= m_araddr;
            ar_len_seen[ar_count]  <= m_arlen;
            ar_count <= ar_count + 1;
        end

        if (m_rvalid && m_rready && m_rlast) begin
            m_rvalid <= 1'b0;
            m_rlast  <= 1'b0;
        end else if (!rd_active && !m_rvalid && m_arvalid && m_arready) begin
            rd_active <= 1'b1;
            rd_base   <= m_araddr;
            rd_len    <= m_arlen;
            rd_cnt    <= 8'd0;
            m_rvalid  <= 1'b0;
            m_rlast   <= 1'b0;
        end else if (rd_active && (!m_rvalid || (m_rvalid && m_rready))) begin
            m_rdata  <= dram[((rd_base >> 2) + rd_cnt) % DRAM_SZ];
            m_rvalid <= 1'b1;
            m_rlast  <= (rd_cnt >= rd_len);
            if (rd_cnt >= rd_len)
                rd_active <= 1'b0;
            else
                rd_cnt <= rd_cnt + 1'b1;
        end
    end
end

reg [31:0] wr_base;
reg [7:0]  wr_cnt;
reg        wr_phase;
reg        b_pending;

always @(posedge clk) begin
    if (!rst_n) begin
        m_awready <= 1'b0;
        m_wready  <= 1'b0;
        m_bvalid  <= 1'b0;
        wr_phase  <= 1'b0;
        b_pending <= 1'b0;
        wr_cnt    <= 8'd0;
        aw_count  <= 0;
    end else begin
        m_awready <= 1'b1;
        if (m_awvalid && m_awready && !wr_phase) begin
            wr_phase <= 1'b1;
            wr_base  <= m_awaddr;
            wr_cnt   <= 8'd0;
            m_wready <= 1'b1;
            if (aw_count < 16) begin
                aw_addr_seen[aw_count] <= m_awaddr;
                aw_len_seen[aw_count]  <= m_awlen;
            end
            aw_count <= aw_count + 1;
        end

        if (wr_phase && m_wvalid && m_wready) begin
            dram[((wr_base >> 2) + wr_cnt) % DRAM_SZ] <= m_wdata;
            wr_cnt <= wr_cnt + 1'b1;
            if (m_wlast) begin
                wr_phase  <= 1'b0;
                m_wready  <= 1'b0;
                b_pending <= 1'b1;
            end
        end

        if (b_pending && !m_bvalid) begin
            m_bvalid  <= 1'b1;
            b_pending <= 1'b0;
        end else if (m_bvalid && m_bready) begin
            m_bvalid <= 1'b0;
        end
    end
end

integer i;

initial begin
    for (i = 0; i < DRAM_SZ; i = i + 1)
        dram[i] = 32'd0;
    for (i = 0; i < 16; i = i + 1) begin
        ar_addr_seen[i] = 32'd0;
        ar_len_seen[i]  = 8'd0;
        aw_addr_seen[i] = 32'd0;
        aw_len_seen[i]  = 8'd0;
    end

    write_desc(DESC0_ADDR, DESC_CTRL_COMMON, A0_ADDR, W0_ADDR, R0_ADDR, DESC1_ADDR);
    write_desc(DESC1_ADDR, DESC_CTRL_LAST,   A1_ADDR, W1_ADDR, R1_ADDR, 32'd0);
    fill_layer(0, A0_ADDR, W0_ADDR, R0_ADDR);
    fill_layer(1, A1_ADDR, W1_ADDR, R1_ADDR);

    s_awaddr = 0; s_wdata = 0; s_wstrb = 0;
    s_awvalid = 0; s_wvalid = 0; s_bready = 0;
    s_araddr = 0; s_arvalid = 0; s_rready = 0;
    pass_cnt = 0;
    fail_cnt = 0;

    @(posedge rst_n);
    repeat (4) @(posedge clk);

    axi_write(REG_DESC_BASE, DESC0_ADDR);
    axi_write(REG_DESC_COUNT, 32'd2);
    axi_write(REG_CTRL, CTRL_START_DESC);

    wait_done(10000);

    if (ar_count !== 6) begin
        $display("[FAIL] expected 6 descriptor/W/A read bursts, got %0d", ar_count);
        fail_cnt = fail_cnt + 1;
    end else begin
        pass_cnt = pass_cnt + 1;
    end

    expect_read(0, DESC0_ADDR, 8'd15);
    expect_read(1, W0_ADDR,    8'd3);
    expect_read(2, A0_ADDR,    8'd3);
    expect_read(3, DESC1_ADDR, 8'd15);
    expect_read(4, W1_ADDR,    8'd3);
    expect_read(5, A1_ADDR,    8'd3);

    if (aw_count !== 8) begin
        $display("[FAIL] expected 8 row write bursts, got %0d", aw_count);
        fail_cnt = fail_cnt + 1;
    end else begin
        pass_cnt = pass_cnt + 1;
    end

    expect_write(0, R0_ADDR + 32'h00);
    expect_write(1, R0_ADDR + 32'h10);
    expect_write(2, R0_ADDR + 32'h20);
    expect_write(3, R0_ADDR + 32'h30);
    expect_write(4, R1_ADDR + 32'h00);
    expect_write(5, R1_ADDR + 32'h10);
    expect_write(6, R1_ADDR + 32'h20);
    expect_write(7, R1_ADDR + 32'h30);

    for (i = 0; i < M_DIM*N_DIM; i = i + 1) begin
        check_result(0, i);
        check_result(1, i);
    end

    if (fail_cnt == 0) begin
        $display("[PASS] tb_npu_desc_two_layer: ALL %0d CHECKS PASSED", pass_cnt);
    end else begin
        $display("[FAIL] tb_npu_desc_two_layer: %0d passed, %0d failed",
                 pass_cnt, fail_cnt);
        $fatal;
    end

    $finish;
end

initial begin
    #(CLK_T * 200000);
    $display("[FAIL] tb_npu_desc_two_layer global timeout");
    $fatal;
end

endmodule
