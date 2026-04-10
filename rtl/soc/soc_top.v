`ifndef PICORV32_REGS
  `define PICORV32_REGS picorv32_regs
`endif

// =============================================================================
// Module  : soc_top
// Project : NPU_prj
// Desc    : SoC top-level: PicoRV32 CPU + NPU accelerator + SRAM + DRAM.
//
//   Architecture (inspired by PicoRV32 picosoc):
//
//     PicoRV32 (mem_if)
//         │
//         ├── addr < 4*MEM_WORDS ──► SRAM (soc_mem)
//         │                               CPU instruction + data storage
//         │
//         ├── 4*MEM_WORDS ≤ addr < 0x02000000 ──► DRAM (dram_model)
//         │                               NPU data area (weights, activations, results)
//         │                               Accessible by both CPU and NPU DMA
//         │
//         └── addr ≥ 0x02000000 ──► NPU Registers (via axi_lite_bridge)
//                                         CPU configures NPU parameters
//
//   NPU DMA accesses DRAM directly via AXI4 Master port (separate from CPU).
//
//   Address Map:
//     0x0000_0000 - 0x0000_0FFF  SRAM (4KB)     CPU only
//     0x0000_0100 - 0x0000_FFFF  DRAM (~60KB)    CPU + NPU DMA
//     0x0200_0000 - 0x0200_003F  NPU Registers   CPU only
//
//   Interrupt:
//     NPU done → CPU irq[7] (via PicoRV32 irq vector bit 7)
// =============================================================================

`timescale 1ns/1ps

module soc_top #(
    parameter MEM_WORDS   = 1024,       // SRAM: 4KB (1024 × 4 bytes)
    parameter DRAM_WORDS  = 15360,      // DRAM: ~60KB
    parameter NPU_ROWS    = 4,
    parameter NPU_COLS    = 4,
    parameter NPU_DATA_W  = 16,
    parameter NPU_ACC_W   = 32,
    parameter NPU_PPB_DEPTH  = 32,
    parameter NPU_PPB_THRESH = 16
)(
    input  wire        clk,
    input  wire        rst_n
);

// =========================================================================
// Address constants
// =========================================================================
localparam [31:0] NPU_BASE_ADDR  = 32'h0200_0000;
localparam [31:0] NPU_ADDR_MASK  = 32'hFFFF_FFFF;
localparam [31:0] SRAM_SIZE_BYTES = 4 * MEM_WORDS;   // 0x1000 for 1024 words
localparam [31:0] DRAM_BASE_ADDR = SRAM_SIZE_BYTES;  // 0x0000_1000

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
// DRAM signals (CPU side - simple interface)
// =========================================================================
wire        dram_cpu_valid;
wire        dram_cpu_ready;
wire        dram_cpu_we;
wire [3:0]  dram_cpu_wstrb;
wire [31:0] dram_cpu_addr;
wire [31:0] dram_cpu_wdata;
wire [31:0] dram_cpu_rdata;

// =========================================================================
// NPU bridge signals (iomem → AXI-Lite)
// =========================================================================
wire        npu_iomem_valid = addr_is_npu;
wire        npu_iomem_ready;
wire [3:0]  npu_iomem_wstrb = mem_wstrb;
wire [31:0] npu_iomem_addr  = mem_addr;
wire [31:0] npu_iomem_wdata = mem_wdata;
wire [31:0] npu_iomem_rdata;

// AXI-Lite bridge → NPU
wire [31:0] npu_awaddr, npu_awaddr_out;
wire        npu_awvalid, npu_awready;
wire [31:0] npu_wdata, npu_wdata_out;
wire [3:0]  npu_wstrb, npu_wstrb_out;
wire        npu_wvalid, npu_wready;
wire [1:0]  npu_bresp;
wire        npu_bvalid, npu_bready;
wire [31:0] npu_araddr, npu_araddr_out;
wire        npu_arvalid, npu_arready;
wire [31:0] npu_rdata;
wire [1:0]  npu_rresp;
wire        npu_rvalid, npu_rready;

// =========================================================================
// NPU signals
// =========================================================================
wire        npu_irq;

// NPU DMA → DRAM (AXI4 Master side)
wire [31:0] npu_m_awaddr;
wire [7:0]  npu_m_awlen;
wire [2:0]  npu_m_awsize;
wire [1:0]  npu_m_awburst;
wire        npu_m_awvalid;
wire        npu_m_awready;
wire [NPU_ACC_W-1:0] npu_m_wdata;
wire [NPU_ACC_W/8-1:0] npu_m_wstrb;
wire        npu_m_wlast;
wire        npu_m_wvalid;
wire        npu_m_wready;
wire [1:0]  npu_m_bresp;
wire        npu_m_bvalid;
wire        npu_m_bready;
wire [31:0] npu_m_araddr;
wire [7:0]  npu_m_arlen;
wire [2:0]  npu_m_arsize;
wire [1:0]  npu_m_arburst;
wire        npu_m_arvalid;
wire        npu_m_arready;
wire [NPU_ACC_W-1:0] npu_m_rdata;
wire [1:0]  npu_m_rresp;
wire        npu_m_rvalid;
wire        npu_m_rready;
wire        npu_m_rlast;

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
    .addr  (mem_addr[21:2]),
    .wdata (mem_wdata),
    .rdata (ram_rdata)
);

// =========================================================================
// DRAM (CPU side interface)
// =========================================================================
assign dram_cpu_valid = addr_is_dram;
assign dram_cpu_we    = |mem_wstrb;
assign dram_cpu_wstrb = mem_wstrb;
assign dram_cpu_addr  = mem_addr;
assign dram_cpu_wdata = mem_wdata;
assign dram_cpu_ready = addr_is_dram;  // always ready in simulation

// =========================================================================
// AXI-Lite Bridge (iomem → NPU AXI4-Lite)
// =========================================================================
axi_lite_bridge u_axi_bridge (
    .clk           (clk),
    .rst_n         (rst_n),
    .iomem_valid   (npu_iomem_valid),
    .iomem_ready   (npu_iomem_ready),
    .iomem_wstrb   (npu_iomem_wstrb),
    .iomem_addr    (npu_iomem_addr),
    .iomem_wdata   (npu_iomem_wdata),
    .iomem_rdata   (npu_iomem_rdata),
    .m_axi_awaddr  (npu_awaddr_out),
    .m_axi_awvalid (npu_awvalid),
    .m_axi_awready (npu_awready),
    .m_axi_wdata   (npu_wdata_out),
    .m_axi_wstrb   (npu_wstrb_out),
    .m_axi_wvalid  (npu_wvalid),
    .m_axi_wready  (npu_wready),
    .m_axi_bresp   (npu_bresp),
    .m_axi_bvalid  (npu_bvalid),
    .m_axi_bready  (npu_bready),
    .m_axi_araddr  (npu_araddr_out),
    .m_axi_arvalid (npu_arvalid),
    .m_axi_arready (npu_arready),
    .m_axi_rdata   (npu_rdata),
    .m_axi_rresp   (npu_rresp),
    .m_axi_rvalid  (npu_rvalid),
    .m_axi_rready  (npu_rready),
    .npu_base_addr (NPU_BASE_ADDR)
);

// =========================================================================
// DRAM Model (dual-port: CPU + NPU DMA)
// =========================================================================
dram_model #(
    .WORDS (DRAM_WORDS),
    .DATA_W(NPU_ACC_W)
) u_dram (
    .clk          (clk),
    .rst_n        (rst_n),
    // CPU side
    .cpu_valid    (dram_cpu_valid),
    .cpu_ready    (dram_cpu_ready),
    .cpu_we       (dram_cpu_we),
    .cpu_wstrb    (dram_cpu_wstrb),
    .cpu_addr     (dram_cpu_addr),
    .cpu_wdata    (dram_cpu_wdata),
    .cpu_rdata    (dram_cpu_rdata),
    // NPU DMA side (AXI4)
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
    .axi_arvalid  (npu_m_arvalid),
    .axi_arready  (npu_m_arready),
    .axi_rdata    (npu_m_rdata),
    .axi_rresp    (npu_m_rresp),
    .axi_rvalid   (npu_m_rvalid),
    .axi_rlast    (npu_m_rlast),
    .axi_rready   (npu_m_rready)
);

// =========================================================================
// NPU
// =========================================================================
npu_top #(
    .ROWS      (NPU_ROWS),
    .COLS      (NPU_COLS),
    .DATA_W    (NPU_DATA_W),
    .ACC_W     (NPU_ACC_W),
    .PPB_DEPTH (NPU_PPB_DEPTH),
    .PPB_THRESH(NPU_PPB_THRESH)
) u_npu (
    .sys_clk       (clk),
    .sys_rst_n     (rst_n),
    // AXI4-Lite slave (CPU config)
    .s_axi_awaddr  (npu_awaddr_out),
    .s_axi_awvalid (npu_awvalid),
    .s_axi_awready (npu_awready),
    .s_axi_wdata   (npu_wdata_out),
    .s_axi_wstrb   (npu_wstrb_out),
    .s_axi_wvalid  (npu_wvalid),
    .s_axi_wready  (npu_wready),
    .s_axi_bresp   (npu_bresp),
    .s_axi_bvalid  (npu_bvalid),
    .s_axi_bready  (npu_bready),
    .s_axi_araddr  (npu_araddr_out),
    .s_axi_arvalid (npu_arvalid),
    .s_axi_arready (npu_arready),
    .s_axi_rdata   (npu_rdata),
    .s_axi_rresp   (npu_rresp),
    .s_axi_rvalid  (npu_rvalid),
    .s_axi_rready  (npu_rready),
    // AXI4 Master (DMA → DRAM)
    .m_axi_awaddr  (npu_m_awaddr),
    .m_axi_awlen   (npu_m_awlen),
    .m_axi_awsize  (npu_m_awsize),
    .m_axi_awburst (npu_m_awburst),
    .m_axi_awvalid (npu_m_awvalid),
    .m_axi_awready (npu_m_awready),
    .m_axi_wdata   (npu_m_wdata),
    .m_axi_wstrb   (npu_m_wstrb),
    .m_axi_wlast   (npu_m_wlast),
    .m_axi_wvalid  (npu_m_wvalid),
    .m_axi_wready  (npu_m_wready),
    .m_axi_bresp   (npu_m_bresp),
    .m_axi_bvalid  (npu_m_bvalid),
    .m_axi_bready  (npu_m_bready),
    .m_axi_araddr  (npu_m_araddr),
    .m_axi_arlen   (npu_m_arlen),
    .m_axi_arsize  (npu_m_arsize),
    .m_axi_arburst (npu_m_arburst),
    .m_axi_arvalid (npu_m_arvalid),
    .m_axi_arready (npu_m_arready),
    .m_axi_rdata   (npu_m_rdata),
    .m_axi_rresp   (npu_m_rresp),
    .m_axi_rvalid  (npu_m_rvalid),
    .m_axi_rready  (npu_m_rready),
    .m_axi_rlast   (npu_m_rlast),
    // Interrupt
    .npu_irq       (npu_irq)
);

// =========================================================================
// PicoRV32 CPU
// =========================================================================
wire [31:0] cpu_irq;

assign cpu_irq = 32'h0;
assign cpu_irq[7] = npu_irq;

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
    .STACKADDR      (SRAM_SIZE_BYTES),         // Stack at end of SRAM
    .PROGADDR_RESET (32'h0000_0000),           // Start at SRAM base
    .PROGADDR_IRQ   (32'h0000_0000),           // IRQ handler at address 0
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
    .pcpi_ready  (),
    .pcpi_wait   (),
    .pcpi_ir     (),
    .pcpi_rd     (),
    .pcpi_wr     (),
    .pcpi_rs1    (),
    .pcpi_alu    (),
    .pcpi_alu_o  (),
    .pcpi_mul_wait(),
    .pcpi_mul_ir (),
    .pcpi_mul_rs1(),
    .pcpi_mul_rs2(),
    .pcpi_div_wait(),
    .pcpi_div_ir (),
    .pcpi_div_rs1(),
    .pcpi_div_rs2(),
    .pcpi_fast_mul_done(),
    .pcpi_fast_mul_shift(),
    .pcpi_fast_mul_sum (),
    .trace_data  (),
    .trace_valid ()
);

endmodule
