// =============================================================================
// Module  : npu_top
// Project : NPU_prj
// Desc    : NPU top-level integration.
//           Connects: AXI-Lite registers, NPU controller, DMA, PE array,
//                     power management, result buffer.
// =============================================================================

`timescale 1ns/1ps

module npu_top #(
    parameter ROWS   = 4,
    parameter COLS   = 4,
    parameter DATA_W = 16,
    parameter ACC_W  = 32
)(
    // System
    input  wire              sys_clk,
    input  wire              sys_rst_n,
    // AXI4-Lite slave (CPU config port)
    // AW
    input  wire [31:0]       s_axi_awaddr,
    input  wire              s_axi_awvalid,
    output wire              s_axi_awready,
    // W
    input  wire [31:0]       s_axi_wdata,
    input  wire [3:0]        s_axi_wstrb,
    input  wire              s_axi_wvalid,
    output wire              s_axi_wready,
    // B
    output wire [1:0]        s_axi_bresp,
    output wire              s_axi_bvalid,
    input  wire              s_axi_bready,
    // AR
    input  wire [31:0]       s_axi_araddr,
    input  wire              s_axi_arvalid,
    output wire              s_axi_arready,
    // R
    output wire [31:0]       s_axi_rdata,
    output wire [1:0]        s_axi_rresp,
    output wire              s_axi_rvalid,
    input  wire              s_axi_rready,
    // AXI4 Master (DMA port to DRAM)
    output wire [31:0]       m_axi_awaddr,
    output wire [7:0]        m_axi_awlen,
    output wire [2:0]        m_axi_awsize,
    output wire [1:0]        m_axi_awburst,
    output wire              m_axi_awvalid,
    input  wire              m_axi_awready,
    output wire [ACC_W-1:0]  m_axi_wdata,
    output wire [ACC_W/8-1:0] m_axi_wstrb,
    output wire              m_axi_wlast,
    output wire              m_axi_wvalid,
    input  wire              m_axi_wready,
    input  wire [1:0]        m_axi_bresp,
    input  wire              m_axi_bvalid,
    output wire              m_axi_bready,
    output wire [31:0]       m_axi_araddr,
    output wire [7:0]        m_axi_arlen,
    output wire [2:0]        m_axi_arsize,
    output wire [1:0]        m_axi_arburst,
    output wire              m_axi_arvalid,
    input  wire              m_axi_arready,
    input  wire [ACC_W-1:0]  m_axi_rdata,
    input  wire [1:0]        m_axi_rresp,
    input  wire              m_axi_rvalid,
    output wire              m_axi_rready,
    input  wire              m_axi_rlast,
    // Interrupt
    output wire              npu_irq
);

// ---------------------------------------------------------------------------
// Wires: register file → controller
// ---------------------------------------------------------------------------
wire [31:0] ctrl_reg, m_dim_r, n_dim_r, k_dim_r;
wire [31:0] w_addr_r, a_addr_r, r_addr_r;
wire [7:0]  arr_cfg_r;
wire [2:0]  clk_div_r;
wire        cg_en_r;
wire        status_busy, status_done, irq_flag;

// ---------------------------------------------------------------------------
// AXI4-Lite Register File
// ---------------------------------------------------------------------------
npu_axi_lite u_axi_lite (
    .aclk       (sys_clk),
    .aresetn    (sys_rst_n),
    .awaddr     (s_axi_awaddr),
    .awvalid    (s_axi_awvalid),
    .awready    (s_axi_awready),
    .wdata      (s_axi_wdata),
    .wstrb      (s_axi_wstrb),
    .wvalid     (s_axi_wvalid),
    .wready     (s_axi_wready),
    .bresp      (s_axi_bresp),
    .bvalid     (s_axi_bvalid),
    .bready     (s_axi_bready),
    .araddr     (s_axi_araddr),
    .arvalid    (s_axi_arvalid),
    .arready    (s_axi_arready),
    .rdata      (s_axi_rdata),
    .rresp      (s_axi_rresp),
    .rvalid     (s_axi_rvalid),
    .rready     (s_axi_rready),
    .ctrl_reg   (ctrl_reg),
    .m_dim      (m_dim_r),
    .n_dim      (n_dim_r),
    .k_dim      (k_dim_r),
    .w_addr     (w_addr_r),
    .a_addr     (a_addr_r),
    .r_addr     (r_addr_r),
    .arr_cfg    (arr_cfg_r),
    .clk_div    (clk_div_r),
    .cg_en      (cg_en_r),
    .status_busy(status_busy),
    .status_done(status_done),
    .irq_flag   (irq_flag),
    .npu_irq    (npu_irq)
);

// ---------------------------------------------------------------------------
// NPU Controller
// ---------------------------------------------------------------------------
wire dma_w_start, dma_a_start, dma_r_start;
wire dma_w_done,  dma_a_done,  dma_r_done;
wire [31:0] dma_w_addr, dma_a_addr, dma_r_addr;
wire [15:0] dma_w_len,  dma_a_len,  dma_r_len;
wire pe_en, pe_flush, pe_mode, pe_stat;

npu_ctrl #(
    .ROWS  (ROWS),
    .COLS  (COLS),
    .DATA_W(DATA_W),
    .ACC_W (ACC_W)
) u_ctrl (
    .clk        (sys_clk),
    .rst_n      (sys_rst_n),
    .ctrl_reg   (ctrl_reg),
    .m_dim      (m_dim_r),
    .n_dim      (n_dim_r),
    .k_dim      (k_dim_r),
    .w_addr     (w_addr_r),
    .a_addr     (a_addr_r),
    .r_addr     (r_addr_r),
    .arr_cfg    (arr_cfg_r),
    .busy       (status_busy),
    .done       (status_done),
    .dma_w_start(dma_w_start),
    .dma_w_done (dma_w_done),
    .dma_w_addr (dma_w_addr),
    .dma_w_len  (dma_w_len),
    .dma_a_start(dma_a_start),
    .dma_a_done (dma_a_done),
    .dma_a_addr (dma_a_addr),
    .dma_a_len  (dma_a_len),
    .dma_r_start(dma_r_start),
    .dma_r_done (dma_r_done),
    .dma_r_addr (dma_r_addr),
    .dma_r_len  (dma_r_len),
    .pe_en      (pe_en),
    .pe_flush   (pe_flush),
    .pe_mode    (pe_mode),
    .pe_stat    (pe_stat),
    .irq        (irq_flag)
);

// ---------------------------------------------------------------------------
// DMA (simplified: uses ACC_W=32 for data bus)
// ---------------------------------------------------------------------------
wire w_fifo_empty, a_fifo_empty, r_fifo_full;
wire [DATA_W-1:0] w_fifo_dout, a_fifo_dout;
wire r_fifo_rd_en;
wire [ACC_W-1:0] r_fifo_din;

// Map PE result → DMA result FIFO
assign r_fifo_din = pe_array_result;
assign r_fifo_wr_en = pe_array_valid[0]; // simplified

npu_dma #(
    .DATA_W   (ACC_W),
    .FIFO_DEPTH(64)
) u_dma (
    .clk         (sys_clk),
    .rst_n       (sys_rst_n),
    .w_start     (dma_w_start),
    .w_base_addr (dma_w_addr),
    .w_len_bytes (dma_w_len),
    .w_done      (dma_w_done),
    .w_fifo_empty(w_fifo_empty),
    .w_fifo_rd_en(w_fifo_rd_en_r),
    .w_fifo_dout (),
    .a_start     (dma_a_start),
    .a_base_addr (dma_a_addr),
    .a_len_bytes (dma_a_len),
    .a_done      (dma_a_done),
    .a_fifo_empty(a_fifo_empty),
    .a_fifo_rd_en(a_fifo_rd_en_r),
    .a_fifo_dout (),
    .r_start     (dma_r_start),
    .r_base_addr (dma_r_addr),
    .r_len_bytes (dma_r_len),
    .r_done      (dma_r_done),
    .r_fifo_full (r_fifo_full),
    .r_fifo_wr_en(r_fifo_wr_en),
    .r_fifo_din  (r_fifo_din),
    .m_axi_awaddr (m_axi_awaddr),
    .m_axi_awlen  (m_axi_awlen),
    .m_axi_awsize (m_axi_awsize),
    .m_axi_awburst(m_axi_awburst),
    .m_axi_awvalid(m_axi_awvalid),
    .m_axi_awready(m_axi_awready),
    .m_axi_wdata  (m_axi_wdata),
    .m_axi_wstrb  (m_axi_wstrb),
    .m_axi_wlast  (m_axi_wlast),
    .m_axi_wvalid (m_axi_wvalid),
    .m_axi_wready (m_axi_wready),
    .m_axi_bresp  (m_axi_bresp),
    .m_axi_bvalid (m_axi_bvalid),
    .m_axi_bready (m_axi_bready),
    .m_axi_araddr (m_axi_araddr),
    .m_axi_arlen  (m_axi_arlen),
    .m_axi_arsize (m_axi_arsize),
    .m_axi_arburst(m_axi_arburst),
    .m_axi_arvalid(m_axi_arvalid),
    .m_axi_arready(m_axi_arready),
    .m_axi_rdata  (m_axi_rdata),
    .m_axi_rresp  (m_axi_rresp),
    .m_axi_rvalid (m_axi_rvalid),
    .m_axi_rready (m_axi_rready),
    .m_axi_rlast  (m_axi_rlast)
);

// Simplified DMA FIFO read enables (tie to pe_en when not empty)
wire w_fifo_rd_en_r = pe_en && !w_fifo_empty;
wire a_fifo_rd_en_r = pe_en && !a_fifo_empty;

// ---------------------------------------------------------------------------
// PE Array
// ---------------------------------------------------------------------------
wire [COLS*ACC_W-1:0] pe_array_result;
wire [COLS-1:0]       pe_array_valid;
wire [COLS*DATA_W-1:0] pe_w_in, pe_a_in;
wire [COLS*ACC_W-1:0]  pe_acc_in;

// Simplified: tie DMA FIFO outputs to array inputs
assign pe_w_in   = {(COLS){w_fifo_dout}};  // broadcast weight
assign pe_a_in   = {(ROWS){a_fifo_dout}};  // broadcast activation
assign pe_acc_in = 0;

pe_array #(
    .ROWS  (ROWS),
    .COLS  (COLS),
    .DATA_W(DATA_W),
    .ACC_W (ACC_W)
) u_pe_array (
    .clk      (sys_clk),
    .rst_n    (sys_rst_n),
    .mode     (pe_mode),
    .stat_mode(pe_stat),
    .en       (pe_en),
    .flush    (pe_flush),
    .w_in     (pe_w_in),
    .act_in   (pe_a_in),
    .acc_in   (pe_acc_in),
    .acc_out  (pe_array_result),
    .valid_out(pe_array_valid)
);

// ---------------------------------------------------------------------------
// Power Management
// ---------------------------------------------------------------------------
wire [ROWS-1:0] row_cg = {ROWS{~pe_en}};  // gate when idle
wire [COLS-1:0] col_cg = {COLS{~pe_en}};

npu_power #(
    .ROWS(ROWS),
    .COLS(COLS)
) u_power (
    .clk         (sys_clk),
    .rst_n       (sys_rst_n),
    .div_sel     (clk_div_r),
    .row_cg_en   (row_cg),
    .col_cg_en   (col_cg),
    .npu_clk     (),
    .row_clk_gated(),
    .col_clk_gated()
);

endmodule
