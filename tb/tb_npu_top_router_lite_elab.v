`timescale 1ns/1ps

`ifndef DUT_USE_ROUTER_MESH
`define DUT_USE_ROUTER_MESH 1
`endif
`ifndef DUT_PHY_ROWS
`define DUT_PHY_ROWS 4
`endif
`ifndef DUT_PHY_COLS
`define DUT_PHY_COLS 4
`endif
`ifndef DUT_IDLE_CYCLES
`define DUT_IDLE_CYCLES 20
`endif
`ifndef DUT_ELAB_ONLY
`define DUT_ELAB_ONLY 0
`endif

module tb_npu_top_router_lite_elab;
    localparam PHY_ROWS = `DUT_PHY_ROWS;
    localparam PHY_COLS = `DUT_PHY_COLS;
    localparam DATA_W = 64;
    localparam ACC_W  = 32;

    reg clk;
    reg rst_n;

    reg  [31:0] s_axi_awaddr;
    reg         s_axi_awvalid;
    wire        s_axi_awready;
    reg  [31:0] s_axi_wdata;
    reg  [3:0]  s_axi_wstrb;
    reg         s_axi_wvalid;
    wire        s_axi_wready;
    wire [1:0]  s_axi_bresp;
    wire        s_axi_bvalid;
    reg         s_axi_bready;
    reg  [31:0] s_axi_araddr;
    reg         s_axi_arvalid;
    wire        s_axi_arready;
    wire [31:0] s_axi_rdata;
    wire [1:0]  s_axi_rresp;
    wire        s_axi_rvalid;
    reg         s_axi_rready;

    wire [31:0] m_axi_awaddr;
    wire [7:0]  m_axi_awlen;
    wire [2:0]  m_axi_awsize;
    wire [1:0]  m_axi_awburst;
    wire        m_axi_awvalid;
    reg         m_axi_awready;
    wire [ACC_W-1:0] m_axi_wdata;
    wire [ACC_W/8-1:0] m_axi_wstrb;
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
    reg  [ACC_W-1:0] m_axi_rdata;
    reg  [1:0]  m_axi_rresp;
    reg         m_axi_rvalid;
    wire        m_axi_rready;
    reg         m_axi_rlast;
    wire        npu_irq;

    npu_top #(
        .PHY_ROWS(PHY_ROWS),
        .PHY_COLS(PHY_COLS),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .PPB_DEPTH(16),
        .PPB_THRESH(4),
        .INT8_SIMD_LANES(8),
        .FP16_ENABLE(0),
        .USE_ROUTER_MESH(`DUT_USE_ROUTER_MESH)
    ) dut (
        .sys_clk(clk),
        .sys_rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
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
        .m_axi_rlast(m_axi_rlast),
        .npu_irq(npu_irq)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 1'b0;
        s_axi_awaddr = 32'd0;
        s_axi_awvalid = 1'b0;
        s_axi_wdata = 32'd0;
        s_axi_wstrb = 4'd0;
        s_axi_wvalid = 1'b0;
        s_axi_bready = 1'b1;
        s_axi_araddr = 32'd0;
        s_axi_arvalid = 1'b0;
        s_axi_rready = 1'b1;
        m_axi_awready = 1'b1;
        m_axi_wready = 1'b1;
        m_axi_bresp = 2'b00;
        m_axi_bvalid = 1'b0;
        m_axi_arready = 1'b1;
        m_axi_rdata = {ACC_W{1'b0}};
        m_axi_rresp = 2'b00;
        m_axi_rvalid = 1'b0;
        m_axi_rlast = 1'b0;

        if (`DUT_ELAB_ONLY != 0) begin
            #1;
            $display("[PASS] tb_npu_top_router_lite_elab rows=%0d cols=%0d USE_ROUTER_MESH=%0d elab_only=1",
                     PHY_ROWS, PHY_COLS, `DUT_USE_ROUTER_MESH);
            $finish;
        end

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        if (`DUT_USE_ROUTER_MESH != 0)
            force dut.ctrl_router_enable = 1'b1;
        repeat (`DUT_IDLE_CYCLES) @(posedge clk);

        if (npu_irq !== 1'b0) begin
            $display("[FAIL] npu_irq asserted during idle router elab smoke");
            $fatal;
        end

        $display("[PASS] tb_npu_top_router_lite_elab rows=%0d cols=%0d USE_ROUTER_MESH=%0d",
                 PHY_ROWS, PHY_COLS, `DUT_USE_ROUTER_MESH);
        $finish;
    end
endmodule
