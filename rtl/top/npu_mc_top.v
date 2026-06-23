// =============================================================================
// Module  : npu_mc_top
// Project : NPU_prj
// Desc    : Multi-core NPU wrapper. Replicates NUM_CORES npu_top instances
//           with flattened AXI-Lite slave, AXI master, and interrupt buses.
//
//           Each core is a complete independent npu_top. The multi-core logic
//           (address decode, scheduling) is outside this module.
// =============================================================================

`timescale 1ns/1ps

module npu_mc_top #(
    parameter NUM_CORES       = 2,
    parameter PHY_ROWS        = 16,
    parameter PHY_COLS        = 16,
    parameter DATA_W          = 32,
    parameter ACC_W           = 32,
    parameter PPB_DEPTH       = 64,
    parameter PPB_THRESH      = 16,
    parameter INT8_SIMD_LANES = 4,
    parameter PERF_ENABLE_DERIVED = 0,
    parameter FP16_ENABLE     = 0,
    parameter PPB_SCALAR_READ_ENABLE = 1
)(
    input  wire                          sys_clk,
    input  wire                          sys_rst_n,

    // ---- AXI4-Lite slaves (flattened, one per core) ----
    input  wire [NUM_CORES*32-1:0]       s_axi_awaddr,
    input  wire [NUM_CORES-1:0]          s_axi_awvalid,
    output wire [NUM_CORES-1:0]          s_axi_awready,

    input  wire [NUM_CORES*32-1:0]       s_axi_wdata,
    input  wire [NUM_CORES*4-1:0]        s_axi_wstrb,
    input  wire [NUM_CORES-1:0]          s_axi_wvalid,
    output wire [NUM_CORES-1:0]          s_axi_wready,

    output wire [NUM_CORES*2-1:0]        s_axi_bresp,
    output wire [NUM_CORES-1:0]          s_axi_bvalid,
    input  wire [NUM_CORES-1:0]          s_axi_bready,

    input  wire [NUM_CORES*32-1:0]       s_axi_araddr,
    input  wire [NUM_CORES-1:0]          s_axi_arvalid,
    output wire [NUM_CORES-1:0]          s_axi_arready,

    output wire [NUM_CORES*32-1:0]       s_axi_rdata,
    output wire [NUM_CORES*2-1:0]        s_axi_rresp,
    output wire [NUM_CORES-1:0]          s_axi_rvalid,
    input  wire [NUM_CORES-1:0]          s_axi_rready,

    // ---- AXI4 Masters (flattened, one per core) ----
    output wire [NUM_CORES*32-1:0]       m_axi_awaddr,
    output wire [NUM_CORES*8-1:0]        m_axi_awlen,
    output wire [NUM_CORES*3-1:0]        m_axi_awsize,
    output wire [NUM_CORES*2-1:0]        m_axi_awburst,
    output wire [NUM_CORES-1:0]          m_axi_awvalid,
    input  wire [NUM_CORES-1:0]          m_axi_awready,

    output wire [NUM_CORES*ACC_W-1:0]    m_axi_wdata,
    output wire [NUM_CORES*(ACC_W/8)-1:0] m_axi_wstrb,
    output wire [NUM_CORES-1:0]          m_axi_wlast,
    output wire [NUM_CORES-1:0]          m_axi_wvalid,
    input  wire [NUM_CORES-1:0]          m_axi_wready,

    input  wire [NUM_CORES*2-1:0]        m_axi_bresp,
    input  wire [NUM_CORES-1:0]          m_axi_bvalid,
    output wire [NUM_CORES-1:0]          m_axi_bready,

    output wire [NUM_CORES*32-1:0]       m_axi_araddr,
    output wire [NUM_CORES*8-1:0]        m_axi_arlen,
    output wire [NUM_CORES*3-1:0]        m_axi_arsize,
    output wire [NUM_CORES*2-1:0]        m_axi_arburst,
    output wire [NUM_CORES-1:0]          m_axi_arvalid,
    input  wire [NUM_CORES-1:0]          m_axi_arready,

    input  wire [NUM_CORES*ACC_W-1:0]    m_axi_rdata,
    input  wire [NUM_CORES*2-1:0]        m_axi_rresp,
    input  wire [NUM_CORES-1:0]          m_axi_rvalid,
    output wire [NUM_CORES-1:0]          m_axi_rready,
    input  wire [NUM_CORES-1:0]          m_axi_rlast,

    // ---- Interrupts (one per core) ----
    output wire [NUM_CORES-1:0]          npu_irq
);

// ---------------------------------------------------------------------------
// Replicated NPU cores
// ---------------------------------------------------------------------------
genvar core_idx;
generate
    for (core_idx = 0; core_idx < NUM_CORES; core_idx = core_idx + 1) begin : gen_cores

        npu_top #(
            .PHY_ROWS             (PHY_ROWS),
            .PHY_COLS             (PHY_COLS),
            .DATA_W               (DATA_W),
            .ACC_W                (ACC_W),
            .PPB_DEPTH            (PPB_DEPTH),
            .PPB_THRESH           (PPB_THRESH),
            .INT8_SIMD_LANES      (INT8_SIMD_LANES),
            .PERF_ENABLE_DERIVED  (PERF_ENABLE_DERIVED),
            .FP16_ENABLE          (FP16_ENABLE),
            .PPB_SCALAR_READ_ENABLE(PPB_SCALAR_READ_ENABLE)
        ) u_npu_core (
            .sys_clk              (sys_clk),
            .sys_rst_n            (sys_rst_n),

            .s_axi_awaddr         (s_axi_awaddr[core_idx*32 +: 32]),
            .s_axi_awvalid        (s_axi_awvalid[core_idx]),
            .s_axi_awready        (s_axi_awready[core_idx]),
            .s_axi_wdata          (s_axi_wdata[core_idx*32 +: 32]),
            .s_axi_wstrb          (s_axi_wstrb[core_idx*4 +: 4]),
            .s_axi_wvalid         (s_axi_wvalid[core_idx]),
            .s_axi_wready         (s_axi_wready[core_idx]),
            .s_axi_bresp          (s_axi_bresp[core_idx*2 +: 2]),
            .s_axi_bvalid         (s_axi_bvalid[core_idx]),
            .s_axi_bready         (s_axi_bready[core_idx]),
            .s_axi_araddr         (s_axi_araddr[core_idx*32 +: 32]),
            .s_axi_arvalid        (s_axi_arvalid[core_idx]),
            .s_axi_arready        (s_axi_arready[core_idx]),
            .s_axi_rdata          (s_axi_rdata[core_idx*32 +: 32]),
            .s_axi_rresp          (s_axi_rresp[core_idx*2 +: 2]),
            .s_axi_rvalid         (s_axi_rvalid[core_idx]),
            .s_axi_rready         (s_axi_rready[core_idx]),

            .m_axi_awaddr         (m_axi_awaddr[core_idx*32 +: 32]),
            .m_axi_awlen          (m_axi_awlen[core_idx*8 +: 8]),
            .m_axi_awsize         (m_axi_awsize[core_idx*3 +: 3]),
            .m_axi_awburst        (m_axi_awburst[core_idx*2 +: 2]),
            .m_axi_awvalid        (m_axi_awvalid[core_idx]),
            .m_axi_awready        (m_axi_awready[core_idx]),
            .m_axi_wdata          (m_axi_wdata[core_idx*ACC_W +: ACC_W]),
            .m_axi_wstrb          (m_axi_wstrb[core_idx*(ACC_W/8) +: (ACC_W/8)]),
            .m_axi_wlast          (m_axi_wlast[core_idx]),
            .m_axi_wvalid         (m_axi_wvalid[core_idx]),
            .m_axi_wready         (m_axi_wready[core_idx]),
            .m_axi_bresp          (m_axi_bresp[core_idx*2 +: 2]),
            .m_axi_bvalid         (m_axi_bvalid[core_idx]),
            .m_axi_bready         (m_axi_bready[core_idx]),
            .m_axi_araddr         (m_axi_araddr[core_idx*32 +: 32]),
            .m_axi_arlen          (m_axi_arlen[core_idx*8 +: 8]),
            .m_axi_arsize         (m_axi_arsize[core_idx*3 +: 3]),
            .m_axi_arburst        (m_axi_arburst[core_idx*2 +: 2]),
            .m_axi_arvalid        (m_axi_arvalid[core_idx]),
            .m_axi_arready        (m_axi_arready[core_idx]),
            .m_axi_rdata          (m_axi_rdata[core_idx*ACC_W +: ACC_W]),
            .m_axi_rresp          (m_axi_rresp[core_idx*2 +: 2]),
            .m_axi_rvalid         (m_axi_rvalid[core_idx]),
            .m_axi_rready         (m_axi_rready[core_idx]),
            .m_axi_rlast          (m_axi_rlast[core_idx]),

            .npu_irq              (npu_irq[core_idx])
        );

    end
endgenerate

endmodule
