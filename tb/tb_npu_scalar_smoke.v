// =============================================================================
// Module  : tb_npu_scalar_smoke
// Desc    : Minimal NPU top-level smoke test.
//           Verifies the current scalar compatibility path:
//             DRAM -> DMA -> PPBuf -> scalar PE -> result FIFO -> DMA -> DRAM.
// =============================================================================

`timescale 1ns/1ps

module tb_npu_scalar_smoke;

localparam DATA_W  = 16;
localparam ACC_W   = 32;
localparam CLK_T   = 10;
localparam DRAM_SZ = 1024;

localparam REG_CTRL      = 32'h00;
localparam REG_STATUS    = 32'h04;
localparam REG_M_DIM     = 32'h10;
localparam REG_N_DIM     = 32'h14;
localparam REG_K_DIM     = 32'h18;
localparam REG_W_ADDR    = 32'h20;
localparam REG_A_ADDR    = 32'h24;
localparam REG_R_ADDR    = 32'h28;
localparam REG_CFG_SHAPE = 32'h3C;

localparam CTRL_START    = 32'h01;
localparam CTRL_OS       = 32'h10;

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

`ifdef TRACE_SMOKE
always @(posedge clk) begin
    if (u_npu.scalar_pe_en || u_npu.scalar_valid) begin
        $display("t=%0t state=%0d rd_en=%b w=%0d a=%0d empty=%b/%b flush=%b s_valid=%b s_res=%0d",
                 $time, u_npu.u_ctrl.state, u_npu.scalar_pe_en,
                 $signed(u_npu.pe_w_data[7:0]), $signed(u_npu.pe_a_data[7:0]),
                 u_npu.w_ppb_buf_empty_int, u_npu.a_ppb_buf_empty_int,
                 u_npu.pe_flush, u_npu.scalar_valid, $signed(u_npu.scalar_result));
    end
end
`endif

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

reg [31:0] rd_base;
reg [7:0]  rd_len, rd_cnt;
reg        rd_active;

always @(posedge clk) begin
    if (!rst_n) begin
        m_arready <= 1'b0;
        m_rvalid  <= 1'b0;
        m_rlast   <= 1'b0;
        rd_active <= 1'b0;
        rd_cnt    <= 8'd0;
    end else begin
        m_arready <= 1'b1;
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
            if (rd_cnt >= rd_len) begin
                rd_active <= 1'b0;
            end else begin
                rd_cnt <= rd_cnt + 1'b1;
            end
        end
    end
end

reg [31:0] wr_base;
reg [7:0]  wr_cnt;
reg        wr_phase, b_pending;

always @(posedge clk) begin
    if (!rst_n) begin
        m_awready <= 1'b0;
        m_wready  <= 1'b0;
        m_bvalid  <= 1'b0;
        wr_phase  <= 1'b0;
        b_pending <= 1'b0;
        wr_cnt    <= 8'd0;
    end else begin
        m_awready <= 1'b1;
        if (m_awvalid && m_awready && !wr_phase) begin
            wr_phase <= 1'b1;
            wr_base  <= m_awaddr;
            wr_cnt   <= 8'd0;
            m_wready <= 1'b1;
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
            $finish;
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
            $finish;
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
            $display("[FAIL] NPU timeout state=%0d dma=%0d",
                     u_npu.u_ctrl.state, u_npu.u_dma.dma_state);
            $finish;
        end
        axi_write(REG_CTRL, 32'h0);
    end
endtask

integer i;
reg [31:0] got;

initial begin
    for (i = 0; i < DRAM_SZ; i = i + 1)
        dram[i] = 32'd0;

    s_awaddr = 0; s_wdata = 0; s_wstrb = 0;
    s_awvalid = 0; s_wvalid = 0; s_bready = 0;
    s_araddr = 0; s_arvalid = 0; s_rready = 0;

    dram[32'h100 >> 2] = 32'h04030201; // W = [1,2,3,4]
    dram[32'h120 >> 2] = 32'h281E140A; // A = [10,20,30,40]

    @(posedge rst_n);
    repeat (4) @(posedge clk);

    axi_write(REG_M_DIM, 32'd1);
    axi_write(REG_N_DIM, 32'd1);
    axi_write(REG_K_DIM, 32'd4);
    axi_write(REG_W_ADDR, 32'h100);
    axi_write(REG_A_ADDR, 32'h120);
    axi_write(REG_R_ADDR, 32'h140);
    axi_write(REG_CFG_SHAPE, 32'h0);
    axi_write(REG_CTRL, CTRL_START | CTRL_OS);

    repeat (3) @(posedge clk);
    axi_write(REG_CFG_SHAPE, 32'h3);

    wait_done(1000);

    got = dram[32'h140 >> 2];
    if (got !== 32'd300) begin
        $display("[FAIL] scalar INT8 OS got %0d (0x%08h), expected 300", $signed(got), got);
        $finish;
    end

    if (u_npu.ctrl_cfg_shape !== 2'b00) begin
        $display("[FAIL] cfg_shape was not latched for current run: got %0d", u_npu.ctrl_cfg_shape);
        $finish;
    end

    $display("[PASS] tb_npu_scalar_smoke: scalar INT8 OS result=300 and cfg_shape latched");
    $finish;
end

endmodule
