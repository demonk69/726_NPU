// =============================================================================
// Module  : soc_mc_top
// Project : NPU_prj
// Desc    : Multi-core NPU SoC simulation top-level.
//           Integrates PicoRV32 CPU, SRAM, shared multi-port DRAM model,
//           multi-core AXI-Lite bridge, and npu_mc_top.
//
//           NUM_CORES=1 must behave identically to soc_top.
// =============================================================================

`timescale 1ns/1ps

`ifndef PICORV32_REGS
  `define PICORV32_REGS picorv32_regs
`endif

module soc_mc_top #(
    parameter MEM_WORDS      = 1024,
    parameter DRAM_WORDS     = 15360,
    parameter NUM_CORES      = 2,
    parameter NPU_ROWS       = 4,
    parameter NPU_COLS       = 4,
    parameter NPU_DATA_W     = 32,
    parameter NPU_ACC_W      = 32,
    parameter NPU_PPB_DEPTH  = 64,
    parameter NPU_PPB_THRESH = 16,
    parameter NPU_INT8_SIMD_LANES = 4
)(
    input  wire        clk,
    input  wire        rst_n
);

// =========================================================================
// Address constants
// =========================================================================
localparam [31:0] NPU_BASE_ADDR   = 32'h0200_0000;
localparam [31:0] SRAM_SIZE_BYTES = 4 * MEM_WORDS;
localparam [31:0] DRAM_BASE_ADDR  = SRAM_SIZE_BYTES;

// =========================================================================
// PicoRV32 memory interface signals
// =========================================================================
wire        mem_valid;
wire        mem_instr;
wire        mem_ready;
wire [31:0] mem_addr;
wire [31:0] mem_wdata;
wire [3:0]  mem_wstrb;
wire [31:0] mem_rdata;

// =========================================================================
// Address decode
// =========================================================================
wire addr_is_ram  = mem_valid && (mem_addr < SRAM_SIZE_BYTES);
wire addr_is_dram = mem_valid && (mem_addr >= DRAM_BASE_ADDR) && (mem_addr < NPU_BASE_ADDR);
wire addr_is_npu  = mem_valid && (mem_addr >= NPU_BASE_ADDR);

// =========================================================================
// SRAM signals
// =========================================================================
wire        ram_ready;
wire [31:0] ram_rdata;

// =========================================================================
// DRAM signals (CPU simple interface)
// =========================================================================
wire        dram_cpu_valid;
wire        dram_cpu_ready;
wire        dram_cpu_ready_raw;
wire        dram_cpu_we;
wire [3:0]  dram_cpu_wstrb;
wire [31:0] dram_cpu_addr;
wire [31:0] dram_cpu_wdata;
wire [31:0] dram_cpu_rdata;

// =========================================================================
// Multi-core AXI-Lite bridge signals
// =========================================================================
wire        npu_iomem_valid = addr_is_npu;
wire        npu_iomem_ready;
wire [3:0]  npu_iomem_wstrb = mem_wstrb;
wire [31:0] npu_iomem_addr  = mem_addr;
wire [31:0] npu_iomem_wdata = mem_wdata;
wire [31:0] npu_iomem_rdata;

// AXI-Lite bridge -> NPU cores (flattened)
wire [NUM_CORES*32-1:0] npu_awaddr;
wire [NUM_CORES-1:0]    npu_awvalid;
wire [NUM_CORES-1:0]    npu_awready;
wire [NUM_CORES*32-1:0] npu_wdata;
wire [NUM_CORES*4-1:0]  npu_wstrb;
wire [NUM_CORES-1:0]    npu_wvalid;
wire [NUM_CORES-1:0]    npu_wready;
wire [NUM_CORES*2-1:0]  npu_bresp;
wire [NUM_CORES-1:0]    npu_bvalid;
wire [NUM_CORES-1:0]    npu_bready;
wire [NUM_CORES*32-1:0] npu_araddr;
wire [NUM_CORES-1:0]    npu_arvalid;
wire [NUM_CORES-1:0]    npu_arready;
wire [NUM_CORES*32-1:0] npu_rdata;
wire [NUM_CORES*2-1:0]  npu_rresp;
wire [NUM_CORES-1:0]    npu_rvalid;
wire [NUM_CORES-1:0]    npu_rready;

// =========================================================================
// NPU multi-core signals
// =========================================================================
wire [NUM_CORES-1:0] npu_irq;

// NPU DMA -> DRAM (flattened AXI4 Master)
wire [NUM_CORES*32-1:0]              npu_m_awaddr;
wire [NUM_CORES*8-1:0]               npu_m_awlen;
wire [NUM_CORES*3-1:0]               npu_m_awsize;
wire [NUM_CORES*2-1:0]               npu_m_awburst;
wire [NUM_CORES-1:0]                 npu_m_awvalid;
wire [NUM_CORES-1:0]                 npu_m_awready;
wire [NUM_CORES*NPU_ACC_W-1:0]       npu_m_wdata;
wire [NUM_CORES*(NPU_ACC_W/8)-1:0]    npu_m_wstrb;
wire [NUM_CORES-1:0]                 npu_m_wlast;
wire [NUM_CORES-1:0]                 npu_m_wvalid;
wire [NUM_CORES-1:0]                 npu_m_wready;
wire [NUM_CORES*2-1:0]               npu_m_bresp;
wire [NUM_CORES-1:0]                 npu_m_bvalid;
wire [NUM_CORES-1:0]                 npu_m_bready;
wire [NUM_CORES*32-1:0]              npu_m_araddr;
wire [NUM_CORES*8-1:0]               npu_m_arlen;
wire [NUM_CORES*3-1:0]               npu_m_arsize;
wire [NUM_CORES*2-1:0]               npu_m_arburst;
wire [NUM_CORES-1:0]                 npu_m_arvalid;
wire [NUM_CORES-1:0]                 npu_m_arready;
wire [NUM_CORES*NPU_ACC_W-1:0]       npu_m_rdata;
wire [NUM_CORES*2-1:0]               npu_m_rresp;
wire [NUM_CORES-1:0]                 npu_m_rvalid;
wire [NUM_CORES-1:0]                 npu_m_rready;
wire [NUM_CORES-1:0]                 npu_m_rlast;

// =========================================================================
// Memory ready mux
// =========================================================================
assign mem_ready = ram_ready || dram_cpu_ready || npu_iomem_ready;

assign mem_rdata = addr_is_ram  ? ram_rdata :
                   addr_is_dram ? dram_cpu_rdata :
                   addr_is_npu  ? npu_iomem_rdata : 32'h0;

// =========================================================================
// SRAM (CPU instruction + data)
// =========================================================================
assign ram_ready = addr_is_ram;

soc_mem #(
    .WORDS(MEM_WORDS)
) u_sram (
    .clk   (clk),
    .wen   (addr_is_ram ? mem_wstrb : 4'b0),
    .addr  (mem_addr[23:2]),
    .wdata (mem_wdata),
    .rdata (ram_rdata)
);

// =========================================================================
// CPU -> DRAM interface (simple port)
// =========================================================================
assign dram_cpu_valid = addr_is_dram;
assign dram_cpu_we    = |mem_wstrb;
assign dram_cpu_wstrb = mem_wstrb;
assign dram_cpu_addr  = mem_addr;
assign dram_cpu_wdata = mem_wdata;
assign dram_cpu_ready = addr_is_dram && dram_cpu_ready_raw;

// =========================================================================
// AXI-Lite Multi-Core Bridge (iomem -> per-core AXI4-Lite)
// =========================================================================
axi_lite_mc_bridge #(
    .NUM_CORES       (NUM_CORES),
    .NPU_CORE_STRIDE (256)
) u_axi_mc_bridge (
    .clk           (clk),
    .rst_n         (rst_n),
    .iomem_valid   (npu_iomem_valid),
    .iomem_ready   (npu_iomem_ready),
    .iomem_wstrb   (npu_iomem_wstrb),
    .iomem_addr    (npu_iomem_addr),
    .iomem_wdata   (npu_iomem_wdata),
    .iomem_rdata   (npu_iomem_rdata),
    .m_axi_awaddr  (npu_awaddr),
    .m_axi_awvalid (npu_awvalid),
    .m_axi_awready (npu_awready),
    .m_axi_wdata   (npu_wdata),
    .m_axi_wstrb   (npu_wstrb),
    .m_axi_wvalid  (npu_wvalid),
    .m_axi_wready  (npu_wready),
    .m_axi_bresp   (npu_bresp),
    .m_axi_bvalid  (npu_bvalid),
    .m_axi_bready  (npu_bready),
    .m_axi_araddr  (npu_araddr),
    .m_axi_arvalid (npu_arvalid),
    .m_axi_arready (npu_arready),
    .m_axi_rdata   (npu_rdata),
    .m_axi_rresp   (npu_rresp),
    .m_axi_rvalid  (npu_rvalid),
    .m_axi_rready  (npu_rready),
    .npu_base_addr (NPU_BASE_ADDR)
);

// =========================================================================
// DRAM Multi-Port Model (CPU + all NPU AXI masters)
// =========================================================================
dram_multi_port #(
    .WORDS     (DRAM_WORDS),
    .DATA_W    (NPU_ACC_W),
    .NUM_CORES (NUM_CORES)
) u_dram (
    .clk          (clk),
    .rst_n        (rst_n),
    // CPU side
    .cpu_valid    (dram_cpu_valid),
    .cpu_ready    (dram_cpu_ready_raw),
    .cpu_we       (dram_cpu_we),
    .cpu_wstrb    (dram_cpu_wstrb),
    .cpu_addr     (dram_cpu_addr),
    .cpu_wdata    (dram_cpu_wdata),
    .cpu_rdata    (dram_cpu_rdata),
    // NPU DMA side (AXI4, flattened)
    .axi_awaddr   (npu_m_awaddr),
    .axi_awvalid  (npu_m_awvalid),
    .axi_awready  (npu_m_awready),
    .axi_wdata    (npu_m_wdata),
    .axi_wstrb    (npu_m_wstrb),
    .axi_wlast    (npu_m_wlast),
    .axi_wvalid   (npu_m_wvalid),
    .axi_wready   (npu_m_wready),
    .axi_bresp    (npu_m_bresp),
    .axi_bvalid   (npu_m_bvalid),
    .axi_bready   (npu_m_bready),
    .axi_araddr   (npu_m_araddr),
    .axi_arlen    (npu_m_arlen),
    .axi_arvalid  (npu_m_arvalid),
    .axi_arready  (npu_m_arready),
    .axi_rdata    (npu_m_rdata),
    .axi_rresp    (npu_m_rresp),
    .axi_rvalid   (npu_m_rvalid),
    .axi_rlast    (npu_m_rlast),
    .axi_rready   (npu_m_rready)
);

// =========================================================================
// Multi-Core NPU
// =========================================================================
npu_mc_top #(
    .NUM_CORES       (NUM_CORES),
    .PHY_ROWS        (16),
    .PHY_COLS        (16),
    .DATA_W          (NPU_DATA_W),
    .ACC_W           (NPU_ACC_W),
    .PPB_DEPTH       (NPU_PPB_DEPTH),
    .PPB_THRESH      (NPU_PPB_THRESH),
    .INT8_SIMD_LANES (NPU_INT8_SIMD_LANES),
    .FP16_ENABLE     (0),
    .PPB_SCALAR_READ_ENABLE(1)
) u_npu_mc (
    .sys_clk        (clk),
    .sys_rst_n      (rst_n),
    // AXI4-Lite slaves
    .s_axi_awaddr   (npu_awaddr),
    .s_axi_awvalid  (npu_awvalid),
    .s_axi_awready  (npu_awready),
    .s_axi_wdata    (npu_wdata),
    .s_axi_wstrb    (npu_wstrb),
    .s_axi_wvalid   (npu_wvalid),
    .s_axi_wready   (npu_wready),
    .s_axi_bresp    (npu_bresp),
    .s_axi_bvalid   (npu_bvalid),
    .s_axi_bready   (npu_bready),
    .s_axi_araddr   (npu_araddr),
    .s_axi_arvalid  (npu_arvalid),
    .s_axi_arready  (npu_arready),
    .s_axi_rdata    (npu_rdata),
    .s_axi_rresp    (npu_rresp),
    .s_axi_rvalid   (npu_rvalid),
    .s_axi_rready   (npu_rready),
    // AXI4 Masters
    .m_axi_awaddr   (npu_m_awaddr),
    .m_axi_awlen    (npu_m_awlen),
    .m_axi_awsize   (npu_m_awsize),
    .m_axi_awburst  (npu_m_awburst),
    .m_axi_awvalid  (npu_m_awvalid),
    .m_axi_awready  (npu_m_awready),
    .m_axi_wdata    (npu_m_wdata),
    .m_axi_wstrb    (npu_m_wstrb),
    .m_axi_wlast    (npu_m_wlast),
    .m_axi_wvalid   (npu_m_wvalid),
    .m_axi_wready   (npu_m_wready),
    .m_axi_bresp    (npu_m_bresp),
    .m_axi_bvalid   (npu_m_bvalid),
    .m_axi_bready   (npu_m_bready),
    .m_axi_araddr   (npu_m_araddr),
    .m_axi_arlen    (npu_m_arlen),
    .m_axi_arsize   (npu_m_arsize),
    .m_axi_arburst  (npu_m_arburst),
    .m_axi_arvalid  (npu_m_arvalid),
    .m_axi_arready  (npu_m_arready),
    .m_axi_rdata    (npu_m_rdata),
    .m_axi_rresp    (npu_m_rresp),
    .m_axi_rvalid   (npu_m_rvalid),
    .m_axi_rready   (npu_m_rready),
    .m_axi_rlast    (npu_m_rlast),
    .npu_irq        (npu_irq)
);

// =========================================================================
// CPU IRQ vector — single assignment only
// =========================================================================
wire [31:0] cpu_irq;
genvar irq_g;
generate
    for (irq_g = 0; irq_g < NUM_CORES; irq_g = irq_g + 1) begin : gen_cpu_irq
        assign cpu_irq[7 + irq_g] = npu_irq[irq_g];
    end
    if (NUM_CORES <= 8) begin
        assign cpu_irq[31 : 7 + NUM_CORES] = {(25 - NUM_CORES){1'b0}};
    end else begin
        assign cpu_irq[31:15] = 17'd0;
    end
endgenerate

// =========================================================================
// PicoRV32 CPU
// =========================================================================
`PICORV32_REGS picosoc_regs_inst (
    .clk    (clk),
    .wen    (),
    .waddr  (),
    .raddr1 (),
    .raddr2 (),
    .wdata  (),
    .rdata1 (),
    .rdata2 ()
);

picorv32 #(
    .STACKADDR      (SRAM_SIZE_BYTES),
    .PROGADDR_RESET (32'h0000_0000),
    .PROGADDR_IRQ   (32'h0000_0000),
    .BARREL_SHIFTER (1),
    .COMPRESSED_ISA (1),
    .ENABLE_COUNTERS(1),
    .ENABLE_MUL     (1),
    .ENABLE_DIV     (1),
    .ENABLE_IRQ     (1),
    .ENABLE_IRQ_QREGS(0)
) u_cpu (
    .clk         (clk),
    .resetn      (rst_n),
    .mem_valid   (mem_valid),
    .mem_instr   (mem_instr),
    .mem_ready   (mem_ready),
    .mem_addr    (mem_addr),
    .mem_wdata   (mem_wdata),
    .mem_wstrb   (mem_wstrb),
    .mem_rdata   (mem_rdata),
    .irq         (cpu_irq),
    .trap        (),
    .mem_la_read (),
    .mem_la_write(),
    .mem_la_wstrb(),
    .pcpi_valid  (),
    .pcpi_insn   (),
    .pcpi_rs1    (),
    .pcpi_rs2    (),
    .pcpi_wr     (1'b0),
    .pcpi_rd     (32'h0),
    .pcpi_wait   (1'b0),
    .pcpi_ready  (1'b0),
    .trace_data  (),
    .trace_valid ()
);

endmodule
