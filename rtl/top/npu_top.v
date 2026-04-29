// =============================================================================
// Module  : npu_top
// Project : NPU_prj
// Desc    : NPU top-level integration with Reconfigurable 16×16 PE Array.
//
//           Physical PE array: always 16×16, reconfigured via cfg_shape:
//             2'b00 → 4×4   (clock-gate rest)
//             2'b01 → 8×8   (clock-gate rest)
//             2'b10 → 16×16 (full active)
//             2'b11 → 8×32  (folded: top-half + bottom-half stitched)
//
//           Data flow:
//             DRAM --DMA--> PPBuf_W --PE(16×16)--> Result FIFO --DMA--> DRAM
//             DRAM --DMA--> PPBuf_A --PE(16×16)--^
//
//           Each PE has dual weight registers for prefetch hiding.
//           swap_w signal atomically swaps active/prefetch weight regs.
// =============================================================================

`timescale 1ns/1ps

module npu_top #(
    // ROWS/COLS are kept only as legacy aliases for older testbenches.
    // The reconfigurable array remains 16x16 unless PHY_* is explicitly set.
    parameter ROWS         = 16,
    parameter COLS         = 16,
    parameter PHY_ROWS     = 16,       // physical rows
    parameter PHY_COLS     = 16,       // physical cols
    parameter DATA_W       = 16,
    parameter ACC_W        = 32,
    parameter PPB_DEPTH    = 64,
    parameter PPB_THRESH   = 16
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
    // AW
    output wire [31:0]       m_axi_awaddr,
    output wire [7:0]        m_axi_awlen,
    output wire [2:0]        m_axi_awsize,
    output wire [1:0]        m_axi_awburst,
    output wire              m_axi_awvalid,
    input  wire              m_axi_awready,
    // W
    output wire [ACC_W-1:0]  m_axi_wdata,
    output wire [ACC_W/8-1:0] m_axi_wstrb,
    output wire              m_axi_wlast,
    output wire              m_axi_wvalid,
    input  wire              m_axi_wready,
    // B
    input  wire [1:0]        m_axi_bresp,
    input  wire              m_axi_bvalid,
    output wire              m_axi_bready,
    // AR
    output wire [31:0]       m_axi_araddr,
    output wire [7:0]        m_axi_arlen,
    output wire [2:0]        m_axi_arsize,
    output wire [1:0]        m_axi_arburst,
    output wire              m_axi_arvalid,
    input  wire              m_axi_arready,
    // R
    input  wire [ACC_W-1:0]  m_axi_rdata,
    input  wire [1:0]        m_axi_rresp,
    input  wire              m_axi_rvalid,
    output wire              m_axi_rready,
    input  wire              m_axi_rlast,
    // Interrupt
    output wire              npu_irq
);

localparam TILE_LANES = 4;

// ---------------------------------------------------------------------------
// Wires: register file → controller
// ---------------------------------------------------------------------------
wire [31:0] ctrl_reg, m_dim_r, n_dim_r, k_dim_r;
wire [31:0] w_addr_r, a_addr_r, r_addr_r;
wire [31:0] desc_base_r, desc_count_r;
wire [7:0]  arr_cfg_r;
wire [2:0]  clk_div_r;
wire [1:0]  cfg_shape_r;
wire        cg_en_r;
wire        status_busy, status_done, status_error, irq_flag;
wire [31:0] err_status;
wire        err_clear;
wire [31:0] err_clear_mask;
wire [31:0] perf_s_axi_wr_cnt, perf_s_axi_rd_cnt;
wire [31:0] perf_s_axi_wr_beats, perf_s_axi_rd_beats;
wire [31:0] perf_s_axi_wr_lat, perf_s_axi_rd_lat;
wire [31:0] perf_m_axi_wr_cnt, perf_m_axi_rd_cnt;
wire [31:0] perf_m_axi_wr_bytes, perf_m_axi_rd_bytes;
wire [31:0] perf_m_axi_wr_beats, perf_m_axi_rd_beats;
wire [31:0] perf_m_axi_wr_lat, perf_m_axi_rd_lat;
wire [31:0] perf_total_cycles;
wire [31:0] perf_m_axi_rd_bw, perf_m_axi_wr_bw;
wire [31:0] perf_m_axi_rd_util, perf_m_axi_wr_util;

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
    .cfg_shape  (cfg_shape_r),
    .desc_base  (desc_base_r),
    .desc_count (desc_count_r),
    .status_busy(status_busy),
    .status_done(status_done),
    .status_error(status_error),
    .err_status (err_status),
    .err_clear  (err_clear),
    .err_clear_mask(err_clear_mask),
    .irq_flag   (irq_flag),
    .perf_cycles(perf_total_cycles),
    .perf_m_axi_rd_beats(perf_m_axi_rd_beats),
    .perf_m_axi_wr_beats(perf_m_axi_wr_beats),
    .perf_m_axi_rd_bytes(perf_m_axi_rd_bytes),
    .perf_m_axi_wr_bytes(perf_m_axi_wr_bytes),
    .perf_m_axi_rd_bw(perf_m_axi_rd_bw),
    .perf_m_axi_wr_bw(perf_m_axi_wr_bw),
    .perf_m_axi_rd_util(perf_m_axi_rd_util),
    .perf_m_axi_wr_util(perf_m_axi_wr_util),
    .perf_m_axi_rd_bursts(perf_m_axi_rd_cnt),
    .perf_m_axi_wr_bursts(perf_m_axi_wr_cnt),
    .npu_irq    (npu_irq)
);

// ---------------------------------------------------------------------------
// NPU Controller
// ---------------------------------------------------------------------------
wire dma_w_start, dma_a_start, dma_r_start;
wire dma_w_done,  dma_a_done,  dma_r_done;
wire [31:0] dma_w_addr, dma_a_addr, dma_r_addr;
wire [15:0] dma_w_len,  dma_a_len,  dma_r_len;
wire dma_a_ofm_mode;
wire [31:0] dma_a_ofm_stride;
wire [31:0] dma_a_ofm_m_base;
wire [31:0] dma_a_ofm_k_base;
wire [15:0] dma_a_ofm_k_len;
wire [2:0]  dma_a_ofm_active_rows;
wire dma_a_ofm_fp16_mode;
wire desc_fetch_start, desc_fetch_done;
wire [31:0] desc_fetch_addr;
wire [511:0] desc_fetch_words;
wire pe_en, pe_flush, pe_mode, pe_stat;
wire pe_load_w, pe_swap_w;   // WS mode weight control
wire ctrl_w_ppb_swap, ctrl_a_ppb_swap, ctrl_w_ppb_clear, ctrl_a_ppb_clear;
wire ctrl_r_fifo_clear;
wire [1:0] ctrl_cfg_shape;
wire ctrl_tile_mode, ctrl_vec_consume;
wire [31:0] ctrl_tile_m_base, ctrl_tile_n_base; // global C tile origin: m0/n0
wire [3:0] ctrl_tile_row_valid, ctrl_tile_col_valid; // valid r/c lanes for edge tiles
wire [2:0] ctrl_tile_active_rows, ctrl_tile_active_cols; // row/col count to serialize
wire [31:0] ctrl_tile_k_base, ctrl_tile_k_index;
wire [15:0] ctrl_tile_k_len;
wire [15:0] ctrl_tile_k_cycle; // OS feeder cycle, includes row-skew drain cycles

npu_ctrl #(
    .ROWS  (PHY_ROWS),       // controller sees physical dimensions
    .COLS  (PHY_COLS),
    .DATA_W(DATA_W),
    .ACC_W (ACC_W),
    .PPB_DEPTH(PPB_DEPTH)
) u_ctrl (
    .clk          (sys_clk),
    .rst_n        (sys_rst_n),
    .ctrl_reg     (ctrl_reg),
    .m_dim        (m_dim_r),
    .n_dim        (n_dim_r),
    .k_dim        (k_dim_r),
    .w_addr       (w_addr_r),
    .a_addr       (a_addr_r),
    .r_addr       (r_addr_r),
    .arr_cfg      (arr_cfg_r),
    .desc_base    (desc_base_r),
    .desc_count   (desc_count_r),
    .desc_start   (desc_fetch_start),
    .desc_addr    (desc_fetch_addr),
    .desc_done    (desc_fetch_done),
    .desc_words   (desc_fetch_words),
    .cfg_shape_in (cfg_shape_r),
    .cfg_shape_latched(ctrl_cfg_shape),
    .tile_mode    (ctrl_tile_mode),
    .vec_consume  (ctrl_vec_consume),
    .tile_m_base  (ctrl_tile_m_base),
    .tile_n_base  (ctrl_tile_n_base),
    .tile_row_valid(ctrl_tile_row_valid),
    .tile_col_valid(ctrl_tile_col_valid),
    .tile_active_rows(ctrl_tile_active_rows),
    .tile_active_cols(ctrl_tile_active_cols),
    .tile_k_base  (ctrl_tile_k_base),
    .tile_k_len   (ctrl_tile_k_len),
    .tile_k_index (ctrl_tile_k_index),
    .tile_k_cycle (ctrl_tile_k_cycle),
    .busy         (status_busy),
    .done         (status_done),
    .error        (status_error),
    .err_status   (err_status),
    .err_clear    (err_clear),
    .err_clear_mask(err_clear_mask),
    .dma_w_start  (dma_w_start),
    .dma_w_done   (dma_w_done),
    .dma_w_addr   (dma_w_addr),
    .dma_w_len    (dma_w_len),
    .dma_a_start  (dma_a_start),
    .dma_a_done   (dma_a_done),
    .dma_a_addr   (dma_a_addr),
    .dma_a_len    (dma_a_len),
    .dma_a_ofm_mode(dma_a_ofm_mode),
    .dma_a_ofm_stride(dma_a_ofm_stride),
    .dma_a_ofm_m_base(dma_a_ofm_m_base),
    .dma_a_ofm_k_base(dma_a_ofm_k_base),
    .dma_a_ofm_k_len(dma_a_ofm_k_len),
    .dma_a_ofm_active_rows(dma_a_ofm_active_rows),
    .dma_a_ofm_fp16_mode(dma_a_ofm_fp16_mode),
    .dma_r_start  (dma_r_start),
    .dma_r_done   (dma_r_done),
    .dma_r_addr   (dma_r_addr),
    .dma_r_len    (dma_r_len),
    .pe_en        (pe_en),
    .pe_flush     (pe_flush),
    .pe_mode      (pe_mode),
    .pe_stat      (pe_stat),
    .pe_load_w    (pe_load_w),
    .pe_swap_w    (pe_swap_w),
    .w_ppb_ready  (u_w_ppb.buf_ready),
    .w_ppb_empty  (u_w_ppb.buf_empty),
    .a_ppb_ready  (u_a_ppb.buf_ready),
    .a_ppb_empty  (u_a_ppb.buf_empty),
    .w_ppb_swap   (ctrl_w_ppb_swap),
    .a_ppb_swap   (ctrl_a_ppb_swap),
    .w_ppb_clear  (ctrl_w_ppb_clear),
    .a_ppb_clear  (ctrl_a_ppb_clear),
    .r_fifo_clear  (ctrl_r_fifo_clear),
    .irq          (irq_flag)
);

// ---------------------------------------------------------------------------
// Ping-Pong Buffers: Weight and Activation
// ---------------------------------------------------------------------------
wire        w_ppb_wr_en,  a_ppb_wr_en;
wire [ACC_W-1:0] w_ppb_wr_data, a_ppb_wr_data;
wire        w_ppb_full,   a_ppb_full;
wire        w_ppb_rd_en,  a_ppb_rd_en;
wire [DATA_W-1:0] w_ppb_rd_data, a_ppb_rd_data;
wire        w_ppb_rd_vec_en, a_ppb_rd_vec_en;
wire [TILE_LANES*DATA_W-1:0] w_ppb_rd_vec, a_ppb_rd_vec;
wire        w_ppb_rd_vec_valid, a_ppb_rd_vec_valid;

wire w_ppb_buf_empty_int, a_ppb_buf_empty_int;
wire w_ppb_buf_ready_int, a_ppb_buf_ready_int;

// Weight Ping-Pong Buffer
pingpong_buf #(
    .DATA_W    (ACC_W),
    .DEPTH     (PPB_DEPTH),
    .OUT_WIDTH (DATA_W),
    .THRESHOLD (PPB_THRESH),
    .SUBW      (4)
) u_w_ppb (
    .clk       (sys_clk),
    .rst_n     (sys_rst_n),
    .wr_en     (w_ppb_wr_en),
    .wr_data   (w_ppb_wr_data),
    .rd_en     (w_ppb_rd_en),
    .rd_data   (w_ppb_rd_data),
    .rd_vec_en  (w_ppb_rd_vec_en),
    .rd_vec     (w_ppb_rd_vec),
    .rd_vec_valid(w_ppb_rd_vec_valid),
    .swap      (ctrl_w_ppb_swap),
    .clear     (ctrl_w_ppb_clear),
    .fp16_mode (pe_mode),
    .buf_empty (w_ppb_buf_empty_int),
    .buf_full  (w_ppb_full),
    .buf_ready (w_ppb_buf_ready_int),
    .rd_fill   (),
    .wr_fill   ()
);

// Activation Ping-Pong Buffer
pingpong_buf #(
    .DATA_W    (ACC_W),
    .DEPTH     (PPB_DEPTH),
    .OUT_WIDTH (DATA_W),
    .THRESHOLD (PPB_THRESH),
    .SUBW      (4)
) u_a_ppb (
    .clk       (sys_clk),
    .rst_n     (sys_rst_n),
    .wr_en     (a_ppb_wr_en),
    .wr_data   (a_ppb_wr_data),
    .rd_en     (a_ppb_rd_en),
    .rd_data   (a_ppb_rd_data),
    .rd_vec_en  (a_ppb_rd_vec_en),
    .rd_vec     (a_ppb_rd_vec),
    .rd_vec_valid(a_ppb_rd_vec_valid),
    .swap      (ctrl_a_ppb_swap),
    .clear     (ctrl_a_ppb_clear),
    .fp16_mode (pe_mode),
    .buf_empty (a_ppb_buf_empty_int),
    .buf_full  (a_ppb_full),
    .buf_ready (a_ppb_buf_ready_int),
    .rd_fill   (),
    .wr_fill   ()
);

// ---------------------------------------------------------------------------
// DMA
// ---------------------------------------------------------------------------
wire r_fifo_full;
wire r_fifo_wr_en;
wire [ACC_W-1:0] r_fifo_din;

npu_dma #(
    .DATA_W      (ACC_W),
    .PE_DATA_W   (DATA_W),
    .BURST_MAX   (16),
    .PPB_DEPTH   (PPB_DEPTH),
    .PPB_THRESH  (PPB_THRESH),
    .R_FIFO_DEPTH(64)
) u_dma (
    .clk            (sys_clk),
    .rst_n          (sys_rst_n),
    // Weight channel
    .w_start        (dma_w_start),
    .w_base_addr    (dma_w_addr),
    .w_len_bytes    (dma_w_len),
    .w_done         (dma_w_done),
    .w_ppb_wr_en    (w_ppb_wr_en),
    .w_ppb_wr_data  (w_ppb_wr_data),
    .w_ppb_full     (w_ppb_full),
    .w_ppb_buf_ready(w_ppb_buf_ready_int),
    .w_ppb_buf_empty(w_ppb_buf_empty_int),
    .w_ppb_drain_done(1'b1),
    // Activation channel
    .a_start        (dma_a_start),
    .a_base_addr    (dma_a_addr),
    .a_len_bytes    (dma_a_len),
    .a_done         (dma_a_done),
    .a_ppb_wr_en    (a_ppb_wr_en),
    .a_ppb_wr_data  (a_ppb_wr_data),
    .a_ppb_full     (a_ppb_full),
    .a_ppb_buf_ready(a_ppb_buf_ready_int),
    .a_ppb_buf_empty(a_ppb_buf_empty_int),
    .a_ppb_drain_done(1'b1),
    .a_ofm_mode    (dma_a_ofm_mode),
    .a_ofm_stride  (dma_a_ofm_stride),
    .a_ofm_m_base  (dma_a_ofm_m_base),
    .a_ofm_k_base  (dma_a_ofm_k_base),
    .a_ofm_k_len   (dma_a_ofm_k_len),
    .a_ofm_active_rows(dma_a_ofm_active_rows),
    .a_ofm_fp16_mode(dma_a_ofm_fp16_mode),
    // Descriptor fetch
    .desc_start     (desc_fetch_start),
    .desc_base_addr (desc_fetch_addr),
    .desc_done      (desc_fetch_done),
    .desc_words     (desc_fetch_words),
    // Result channel
    .r_start        (dma_r_start),
    .r_base_addr    (dma_r_addr),
    .r_len_bytes    (dma_r_len),
    .r_done         (dma_r_done),
    .r_fifo_clear   (ctrl_r_fifo_clear),
    .r_fifo_wr_en   (r_fifo_wr_en),
    .r_fifo_din     (r_fifo_din),
    .r_fifo_full    (r_fifo_full),
    // AXI4 Master
    .m_axi_awaddr  (m_axi_awaddr),
    .m_axi_awlen   (m_axi_awlen),
    .m_axi_awsize  (m_axi_awsize),
    .m_axi_awburst (m_axi_awburst),
    .m_axi_awvalid (m_axi_awvalid),
    .m_axi_awready (m_axi_awready),
    .m_axi_wdata   (m_axi_wdata),
    .m_axi_wstrb   (m_axi_wstrb),
    .m_axi_wlast   (m_axi_wlast),
    .m_axi_wvalid  (m_axi_wvalid),
    .m_axi_wready  (m_axi_wready),
    .m_axi_bresp   (m_axi_bresp),
    .m_axi_bvalid  (m_axi_bvalid),
    .m_axi_bready  (m_axi_bready),
    .m_axi_araddr  (m_axi_araddr),
    .m_axi_arlen   (m_axi_arlen),
    .m_axi_arsize  (m_axi_arsize),
    .m_axi_arburst (m_axi_arburst),
    .m_axi_arvalid (m_axi_arvalid),
    .m_axi_arready (m_axi_arready),
    .m_axi_rdata   (m_axi_rdata),
    .m_axi_rresp   (m_axi_rresp),
    .m_axi_rvalid  (m_axi_rvalid),
    .m_axi_rready  (m_axi_rready),
    .m_axi_rlast   (m_axi_rlast)
);

// ---------------------------------------------------------------------------
// AXI Performance Monitor
// ---------------------------------------------------------------------------
axi_monitor #(
    .ACC_W(ACC_W)
) u_axi_monitor (
    .clk(sys_clk),
    .rst_n(sys_rst_n),
    .s_awvalid(s_axi_awvalid),
    .s_awready(s_axi_awready),
    .s_wvalid(s_axi_wvalid),
    .s_wready(s_axi_wready),
    .s_bvalid(s_axi_bvalid),
    .s_bready(s_axi_bready),
    .s_arvalid(s_axi_arvalid),
    .s_arready(s_axi_arready),
    .s_rvalid(s_axi_rvalid),
    .s_rready(s_axi_rready),
    .m_awlen(m_axi_awlen),
    .m_awvalid(m_axi_awvalid),
    .m_awready(m_axi_awready),
    .m_wlast(m_axi_wlast),
    .m_wvalid(m_axi_wvalid),
    .m_wready(m_axi_wready),
    .m_bvalid(m_axi_bvalid),
    .m_bready(m_axi_bready),
    .m_arlen(m_axi_arlen),
    .m_arvalid(m_axi_arvalid),
    .m_arready(m_axi_arready),
    .m_rlast(m_axi_rlast),
    .m_rvalid(m_axi_rvalid),
    .m_rready(m_axi_rready),
    .s_axi_wr_cnt(perf_s_axi_wr_cnt),
    .s_axi_rd_cnt(perf_s_axi_rd_cnt),
    .s_axi_wr_beats(perf_s_axi_wr_beats),
    .s_axi_rd_beats(perf_s_axi_rd_beats),
    .s_axi_wr_lat(perf_s_axi_wr_lat),
    .s_axi_rd_lat(perf_s_axi_rd_lat),
    .m_axi_wr_cnt(perf_m_axi_wr_cnt),
    .m_axi_rd_cnt(perf_m_axi_rd_cnt),
    .m_axi_wr_bytes(perf_m_axi_wr_bytes),
    .m_axi_rd_bytes(perf_m_axi_rd_bytes),
    .m_axi_wr_beats(perf_m_axi_wr_beats),
    .m_axi_rd_beats(perf_m_axi_rd_beats),
    .m_axi_wr_lat(perf_m_axi_wr_lat),
    .m_axi_rd_lat(perf_m_axi_rd_lat),
    .total_cycles(perf_total_cycles),
    .m_axi_rd_bw(perf_m_axi_rd_bw),
    .m_axi_wr_bw(perf_m_axi_wr_bw),
    .m_axi_rd_util(perf_m_axi_rd_util),
    .m_axi_wr_util(perf_m_axi_wr_util)
);

// ---------------------------------------------------------------------------
// FP16 Packer / Consumer Logic
// ---------------------------------------------------------------------------

wire pe_data_ready = !w_ppb_buf_empty_int && !a_ppb_buf_empty_int;
wire pe_consume = pe_en && (pe_data_ready || pe_flush);
// One packed A vector and one packed W vector are consumed only when both PPBufs
// have a valid 4-lane preview for the controller's current k cycle.
wire tile_vec_fire = ctrl_vec_consume &&
                     w_ppb_rd_vec_valid &&
                     a_ppb_rd_vec_valid;
wire tile_feed_step = ctrl_tile_mode && pe_en && pe_stat && !pe_flush;

assign w_ppb_rd_en = ctrl_tile_mode ? 1'b0 : pe_consume;
assign a_ppb_rd_en = ctrl_tile_mode ? 1'b0 : pe_consume;
assign w_ppb_rd_vec_en = tile_vec_fire;
assign a_ppb_rd_vec_en = tile_vec_fire;

// ---------------------------------------------------------------------------
// Reconfigurable PE Array (16×16 physical)
// ---------------------------------------------------------------------------

wire [PHY_COLS*DATA_W-1:0] pe_w_in;
wire [PHY_ROWS*DATA_W-1:0] pe_a_in;
wire [PHY_COLS*ACC_W-1:0]  pe_acc_in;
wire [PHY_ROWS*PHY_COLS*ACC_W-1:0] pe_acc_init;
wire [PHY_ROWS*PHY_COLS-1:0] pe_acc_init_mask;
wire [32*ACC_W-1:0]         pe_array_result;    // max 32 output columns (8×32 mode)
wire [31:0]                 pe_array_valid;
wire [PHY_ROWS*PHY_COLS-1:0] pe_active_dbg;

// Data source from PPBuf
wire [DATA_W-1:0] pe_w_data = w_ppb_rd_data;
wire [DATA_W-1:0] pe_a_data = a_ppb_rd_data;
wire [DATA_W-1:0] w_vec_lane0 = w_ppb_rd_vec[0*DATA_W +: DATA_W];
wire [DATA_W-1:0] w_vec_lane1 = w_ppb_rd_vec[1*DATA_W +: DATA_W];
wire [DATA_W-1:0] w_vec_lane2 = w_ppb_rd_vec[2*DATA_W +: DATA_W];
wire [DATA_W-1:0] w_vec_lane3 = w_ppb_rd_vec[3*DATA_W +: DATA_W];
wire [DATA_W-1:0] a_vec_lane0 = a_ppb_rd_vec[0*DATA_W +: DATA_W];
wire [DATA_W-1:0] a_vec_lane1 = a_ppb_rd_vec[1*DATA_W +: DATA_W];
wire [DATA_W-1:0] a_vec_lane2 = a_ppb_rd_vec[2*DATA_W +: DATA_W];
wire [DATA_W-1:0] a_vec_lane3 = a_ppb_rd_vec[3*DATA_W +: DATA_W];

reg [DATA_W-1:0] a_lane1_d0;
reg [DATA_W-1:0] a_lane2_d0, a_lane2_d1;
reg [DATA_W-1:0] a_lane3_d0, a_lane3_d1, a_lane3_d2;

// OS row-skew feeder:
//   row0 gets A[m0+0,k] immediately,
//   row1 gets A[m0+1,k] after 1 cycle,
//   row2 gets A[m0+2,k] after 2 cycles,
//   row3 gets A[m0+3,k] after 3 cycles.
// The first r cycles of row r are zero bubbles; this is required so A[k] meets
// the matching W[k] after W has shifted down r rows.
// This aligns each row with the vertically shifted W[k,n0+c] stream.
always @(posedge sys_clk) begin
    if (!sys_rst_n || !ctrl_tile_mode) begin
        a_lane1_d0 <= {DATA_W{1'b0}};
        a_lane2_d0 <= {DATA_W{1'b0}};
        a_lane2_d1 <= {DATA_W{1'b0}};
        a_lane3_d0 <= {DATA_W{1'b0}};
        a_lane3_d1 <= {DATA_W{1'b0}};
        a_lane3_d2 <= {DATA_W{1'b0}};
    end else if (tile_feed_step) begin
        a_lane1_d0 <= tile_vec_fire ? a_vec_lane1 : {DATA_W{1'b0}};
        a_lane2_d0 <= tile_vec_fire ? a_vec_lane2 : {DATA_W{1'b0}};
        a_lane2_d1 <= a_lane2_d0;
        a_lane3_d0 <= tile_vec_fire ? a_vec_lane3 : {DATA_W{1'b0}};
        a_lane3_d1 <= a_lane3_d0;
        a_lane3_d2 <= a_lane3_d1;
    end
end

wire [DATA_W-1:0] pe_w_lane0 = ctrl_tile_mode
    ? (tile_vec_fire ? w_vec_lane0 : {DATA_W{1'b0}})
    : pe_w_data;
wire [DATA_W-1:0] pe_w_lane1 = ctrl_tile_mode && tile_vec_fire ? w_vec_lane1 : {DATA_W{1'b0}};
wire [DATA_W-1:0] pe_w_lane2 = ctrl_tile_mode && tile_vec_fire ? w_vec_lane2 : {DATA_W{1'b0}};
wire [DATA_W-1:0] pe_w_lane3 = ctrl_tile_mode && tile_vec_fire ? w_vec_lane3 : {DATA_W{1'b0}};

wire [DATA_W-1:0] pe_a_lane0 = ctrl_tile_mode
    ? (tile_vec_fire ? a_vec_lane0 : {DATA_W{1'b0}})
    : pe_a_data;
wire [DATA_W-1:0] pe_a_lane1 = ctrl_tile_mode ? a_lane1_d0 : {DATA_W{1'b0}};
wire [DATA_W-1:0] pe_a_lane2 = ctrl_tile_mode ? a_lane2_d1 : {DATA_W{1'b0}};
wire [DATA_W-1:0] pe_a_lane3 = ctrl_tile_mode ? a_lane3_d2 : {DATA_W{1'b0}};

// ── Weight input to PE array ──
// T2.2: lower four columns receive w_vec[0..3]. Upper columns stay zero until
// wider 8x8/16x16 feeding is implemented.
generate
    if (PHY_COLS >= 4) begin : gen_pe_w_in_4plus
        assign pe_w_in = {{(PHY_COLS-TILE_LANES)*DATA_W{1'b0}},
                          pe_w_lane3, pe_w_lane2, pe_w_lane1, pe_w_lane0};
    end else if (PHY_COLS == 3) begin : gen_pe_w_in_3
        assign pe_w_in = {pe_w_lane2, pe_w_lane1, pe_w_lane0};
    end else if (PHY_COLS == 2) begin : gen_pe_w_in_2
        assign pe_w_in = {pe_w_lane1, pe_w_lane0};
    end else begin : gen_pe_w_in_1
        assign pe_w_in = pe_w_lane0;
    end
endgenerate

// ── Activation input to PE array ──
// T2.2: lower four rows receive a_vec[0..3]. OS row-skew scheduling is a T2.3
// controller/feeder responsibility; this stage only exposes the 4-lane path.
generate
    if (PHY_ROWS >= 4) begin : gen_pe_a_in_4plus
        assign pe_a_in = {{(PHY_ROWS-TILE_LANES)*DATA_W{1'b0}},
                          pe_a_lane3, pe_a_lane2, pe_a_lane1, pe_a_lane0};
    end else if (PHY_ROWS == 3) begin : gen_pe_a_in_3
        assign pe_a_in = {pe_a_lane2, pe_a_lane1, pe_a_lane0};
    end else if (PHY_ROWS == 2) begin : gen_pe_a_in_2
        assign pe_a_in = {pe_a_lane1, pe_a_lane0};
    end else begin : gen_pe_a_in_1
        assign pe_a_in = pe_a_lane0;
    end
endgenerate

assign pe_acc_in = {PHY_COLS*ACC_W{1'b0}};
assign pe_acc_init = {PHY_ROWS*PHY_COLS*ACC_W{1'b0}};
assign pe_acc_init_mask = {PHY_ROWS*PHY_COLS{1'b0}};

// ---------------------------------------------------------------------------
// Scalar compatibility PE
// ---------------------------------------------------------------------------
//
// Keep the original single-output path for non-tile mode. 4x4 tile mode writes
// back through the array serializer below.
wire scalar_pe_en = (!ctrl_tile_mode) && pe_consume;
wire [ACC_W-1:0] scalar_result;
wire             scalar_valid;

pe_top #(
    .DATA_W(DATA_W),
    .ACC_W (ACC_W)
) u_scalar_pe (
    .clk      (sys_clk),
    .rst_n    (sys_rst_n),
    .mode     (pe_mode),
    .stat_mode(pe_stat),
    .en       (scalar_pe_en),
    .flush    (pe_flush),
    .load_w   (pe_load_w),
    .swap_w   (pe_swap_w),
    .acc_init_en(1'b0),
    .w_in     (pe_w_data),
    .a_in     (pe_a_data),
    .acc_in   ({ACC_W{1'b0}}),
    .acc_init ({ACC_W{1'b0}}),
    .acc_out  (scalar_result),
    .valid_out(scalar_valid)
);

// WS load row indicator (for debug/status)
wire [3:0] ws_load_row_status;

reconfig_pe_array #(
    .PHY_ROWS(PHY_ROWS),
    .PHY_COLS(PHY_COLS),
    .DATA_W  (DATA_W),
    .ACC_W   (ACC_W)
) u_pe_array (
    .clk            (sys_clk),
    .rst_n          (sys_rst_n),
    .cfg_shape      (ctrl_cfg_shape),
    .mode           (pe_mode),
    .stat_mode      (pe_stat),
    .en             (pe_en),
    .flush          (pe_flush),
    .load_w         (pe_load_w),
    .swap_w         (pe_swap_w),
    .acc_init_en    (1'b0),
    .w_in           (pe_w_in),
    .act_in         (pe_a_in),
    .acc_in         (pe_acc_in),
    .acc_init       (pe_acc_init),
    .acc_init_mask  (pe_acc_init_mask),
    .acc_out        (pe_array_result),
    .valid_out      (pe_array_valid),
    .ws_load_row_out(ws_load_row_status),
    .pe_active      (pe_active_dbg)
);

// ---------------------------------------------------------------------------
// Result FIFO interface
// ---------------------------------------------------------------------------
reg [ACC_W-1:0] tile_result_buf [0:15]; // captured C tile, index = r*4+c
reg             tile_ser_busy;
reg [2:0]       tile_ser_active_rows;   // valid rows in current edge/full tile
reg [2:0]       tile_ser_active_cols;   // valid cols in current edge/full tile
reg [1:0]       tile_ser_row;           // serializer row r
reg [1:0]       tile_ser_col;           // serializer col c

wire [3:0] tile_ser_idx = {tile_ser_row, tile_ser_col};
wire       tile_ser_fire = tile_ser_busy && !r_fifo_full;
wire       tile_ser_last_col = ({1'b0, tile_ser_col} + 3'd1) >= tile_ser_active_cols;
wire       tile_ser_last_row = ({1'b0, tile_ser_row} + 3'd1) >= tile_ser_active_rows;
wire       tile_ser_last     = tile_ser_last_col && tile_ser_last_row;
wire       tile_result_capture = ctrl_tile_mode &&
                                 pe_stat &&
                                 pe_array_valid[0] &&
                                 !tile_ser_busy;

integer tile_ser_i;
always @(posedge sys_clk) begin
    if (!sys_rst_n || !ctrl_tile_mode) begin
        tile_ser_busy       <= 1'b0;
        tile_ser_active_rows <= 3'd0;
        tile_ser_active_cols <= 3'd0;
        tile_ser_row        <= 2'd0;
        tile_ser_col        <= 2'd0;
        for (tile_ser_i = 0; tile_ser_i < 16; tile_ser_i = tile_ser_i + 1)
            tile_result_buf[tile_ser_i] <= {ACC_W{1'b0}};
    end else if (tile_result_capture) begin
        // Capture all 16 PE outputs at once, then push only active rows/cols
        // into the result FIFO in row-major order for DMA row-wise writeback.
        for (tile_ser_i = 0; tile_ser_i < 16; tile_ser_i = tile_ser_i + 1)
            tile_result_buf[tile_ser_i] <= pe_array_result[tile_ser_i*ACC_W +: ACC_W];
        tile_ser_active_rows <= ctrl_tile_active_rows;
        tile_ser_active_cols <= ctrl_tile_active_cols;
        tile_ser_row         <= 2'd0;
        tile_ser_col         <= 2'd0;
        tile_ser_busy        <= (ctrl_tile_active_rows != 3'd0) &&
                                (ctrl_tile_active_cols != 3'd0);
    end else if (tile_ser_fire) begin
        if (tile_ser_last) begin
            tile_ser_busy <= 1'b0;
        end else if (tile_ser_last_col) begin
            tile_ser_row <= tile_ser_row + 1'b1;
            tile_ser_col <= 2'd0;
        end else begin
            tile_ser_col <= tile_ser_col + 1'b1;
        end
    end
end

assign r_fifo_din   = ctrl_tile_mode ? tile_result_buf[tile_ser_idx] : scalar_result;
assign r_fifo_wr_en = ctrl_tile_mode ? tile_ser_fire
                                     : (scalar_valid && !r_fifo_full);

// ---------------------------------------------------------------------------
// Power Management
// ---------------------------------------------------------------------------
wire [PHY_ROWS-1:0] row_cg = {PHY_ROWS{~pe_en}};
wire [PHY_COLS-1:0] col_cg = {PHY_COLS{~pe_en}};

npu_power #(
    .ROWS(PHY_ROWS),
    .COLS(PHY_COLS)
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
