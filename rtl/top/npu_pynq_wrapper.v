// =============================================================================
// Module  : npu_pynq_wrapper
// Project : NPU_prj
// Desc    : Thin PYNQ/Zynq wrapper around npu_top.
//
//           npu_top keeps a board-independent minimal AXI signal set. This
//           wrapper exposes common Xilinx AXI sideband signals for Vivado IP
//           Integrator and strips the AXI-Lite base address down to a local
//           register offset.
// =============================================================================

`timescale 1ns/1ps

module npu_pynq_wrapper #(
    parameter PHY_ROWS     = 16,
    parameter PHY_COLS     = 16,
    parameter DATA_W       = 32,
    parameter ACC_W        = 32,
    parameter PPB_DEPTH    = 64,
    parameter PPB_THRESH   = 16,
    parameter INT8_SIMD_LANES = 4,
    parameter PERF_ENABLE_DERIVED = 0,
    parameter S_AXI_OFFSET_BITS = 16,
    parameter M_AXI_ID_WIDTH = 1
)(
    input  wire                         aclk,
    input  wire                         aresetn,

    // AXI4-Lite slave: PS GP master -> NPU registers.
    input  wire [31:0]                  s_axi_awaddr,
    input  wire [2:0]                   s_axi_awprot,
    input  wire                         s_axi_awvalid,
    output wire                         s_axi_awready,
    input  wire [31:0]                  s_axi_wdata,
    input  wire [3:0]                   s_axi_wstrb,
    input  wire                         s_axi_wvalid,
    output wire                         s_axi_wready,
    output wire [1:0]                   s_axi_bresp,
    output wire                         s_axi_bvalid,
    input  wire                         s_axi_bready,
    input  wire [31:0]                  s_axi_araddr,
    input  wire [2:0]                   s_axi_arprot,
    input  wire                         s_axi_arvalid,
    output wire                         s_axi_arready,
    output wire [31:0]                  s_axi_rdata,
    output wire [1:0]                   s_axi_rresp,
    output wire                         s_axi_rvalid,
    input  wire                         s_axi_rready,

    // AXI4 master: NPU DMA -> PS DDR through HP/SmartConnect.
    output wire [M_AXI_ID_WIDTH-1:0]    m_axi_awid,
    output wire [31:0]                  m_axi_awaddr,
    output wire [7:0]                   m_axi_awlen,
    output wire [2:0]                   m_axi_awsize,
    output wire [1:0]                   m_axi_awburst,
    output wire                         m_axi_awlock,
    output wire [3:0]                   m_axi_awcache,
    output wire [2:0]                   m_axi_awprot,
    output wire [3:0]                   m_axi_awqos,
    output wire                         m_axi_awvalid,
    input  wire                         m_axi_awready,
    output wire [ACC_W-1:0]             m_axi_wdata,
    output wire [ACC_W/8-1:0]           m_axi_wstrb,
    output wire                         m_axi_wlast,
    output wire                         m_axi_wvalid,
    input  wire                         m_axi_wready,
    input  wire [M_AXI_ID_WIDTH-1:0]    m_axi_bid,
    input  wire [1:0]                   m_axi_bresp,
    input  wire                         m_axi_bvalid,
    output wire                         m_axi_bready,
    output wire [M_AXI_ID_WIDTH-1:0]    m_axi_arid,
    output wire [31:0]                  m_axi_araddr,
    output wire [7:0]                   m_axi_arlen,
    output wire [2:0]                   m_axi_arsize,
    output wire [1:0]                   m_axi_arburst,
    output wire                         m_axi_arlock,
    output wire [3:0]                   m_axi_arcache,
    output wire [2:0]                   m_axi_arprot,
    output wire [3:0]                   m_axi_arqos,
    output wire                         m_axi_arvalid,
    input  wire                         m_axi_arready,
    input  wire [M_AXI_ID_WIDTH-1:0]    m_axi_rid,
    input  wire [ACC_W-1:0]             m_axi_rdata,
    input  wire [1:0]                   m_axi_rresp,
    input  wire                         m_axi_rvalid,
    output wire                         m_axi_rready,
    input  wire                         m_axi_rlast,

    output wire                         npu_irq
);

wire [31:0] s_axi_awaddr_off = {{(32-S_AXI_OFFSET_BITS){1'b0}},
                                s_axi_awaddr[S_AXI_OFFSET_BITS-1:0]};
wire [31:0] s_axi_araddr_off = {{(32-S_AXI_OFFSET_BITS){1'b0}},
                                s_axi_araddr[S_AXI_OFFSET_BITS-1:0]};

assign m_axi_awid    = {M_AXI_ID_WIDTH{1'b0}};
assign m_axi_arid    = {M_AXI_ID_WIDTH{1'b0}};
assign m_axi_awlock  = 1'b0;
assign m_axi_arlock  = 1'b0;
assign m_axi_awcache = 4'b0011;
assign m_axi_arcache = 4'b0011;
assign m_axi_awprot  = 3'b000;
assign m_axi_arprot  = 3'b000;
assign m_axi_awqos   = 4'b0000;
assign m_axi_arqos   = 4'b0000;

// Sideband inputs and high AXI-Lite address bits are intentionally unused;
// npu_top only needs local offsets inside the assigned 64KB register window.
wire unused_axi_sideband = |s_axi_awprot | |s_axi_arprot |
                           |s_axi_awaddr[31:S_AXI_OFFSET_BITS] |
                           |s_axi_araddr[31:S_AXI_OFFSET_BITS] |
                           |m_axi_bid | |m_axi_rid;

npu_top #(
    .ROWS(PHY_ROWS),
    .COLS(PHY_COLS),
    .PHY_ROWS(PHY_ROWS),
    .PHY_COLS(PHY_COLS),
    .DATA_W(DATA_W),
    .ACC_W(ACC_W),
    .PPB_DEPTH(PPB_DEPTH),
    .PPB_THRESH(PPB_THRESH),
    .INT8_SIMD_LANES(INT8_SIMD_LANES),
    .PERF_ENABLE_DERIVED(PERF_ENABLE_DERIVED)
) u_npu_top (
    .sys_clk(aclk),
    .sys_rst_n(aresetn),
    .s_axi_awaddr(s_axi_awaddr_off),
    .s_axi_awvalid(s_axi_awvalid),
    .s_axi_awready(s_axi_awready),
    .s_axi_wdata(s_axi_wdata),
    .s_axi_wstrb(s_axi_wstrb),
    .s_axi_wvalid(s_axi_wvalid),
    .s_axi_wready(s_axi_wready),
    .s_axi_bresp(s_axi_bresp),
    .s_axi_bvalid(s_axi_bvalid),
    .s_axi_bready(s_axi_bready),
    .s_axi_araddr(s_axi_araddr_off),
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

endmodule
