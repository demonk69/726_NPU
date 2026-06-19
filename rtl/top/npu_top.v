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
    parameter DATA_W       = 32,
    parameter ACC_W        = 32,
    parameter PPB_DEPTH    = 64,
    parameter PPB_THRESH   = 16,
    parameter INT8_SIMD_LANES = 4,
    parameter PERF_ENABLE_DERIVED = 0,
    parameter FP16_ENABLE = 0,
    parameter PPB_SCALAR_READ_ENABLE = 1
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

localparam MAX_TILE_LANES = 16;
localparam TILE_BIAS_DEPTH = 32;   // max tile columns (8x32)
localparam MAX_TILE_RESULTS = 256;  // 16x16 or 8x32 grid
localparam [3:0] PERF_SIMD_LANES = INT8_SIMD_LANES;

function [4:0] shape_tile_lanes;
    input [1:0] shape;
    begin
        case (shape)
            2'b00: shape_tile_lanes = 5'd4;
            2'b01: shape_tile_lanes = 5'd8;
            default: shape_tile_lanes = 5'd16;
        endcase
    end
endfunction

function [PHY_ROWS-1:0] shape_row_ce_mask;
    input [1:0] shape;
    integer i;
    integer active_rows;
    begin
        active_rows = (shape == 2'b00) ? 4 :
                      (shape == 2'b01) ? 8 :
                                          PHY_ROWS;
        for (i = 0; i < PHY_ROWS; i = i + 1)
            shape_row_ce_mask[i] = (i < active_rows);
    end
endfunction

function [PHY_COLS-1:0] shape_col_ce_mask;
    input [1:0] shape;
    integer i;
    integer active_cols;
    begin
        active_cols = (shape == 2'b00) ? 4 :
                      (shape == 2'b01) ? 8 :
                                          PHY_COLS;
        for (i = 0; i < PHY_COLS; i = i + 1)
            shape_col_ce_mask[i] = (i < active_cols);
    end
endfunction

// ---------------------------------------------------------------------------
// Wires: register file → controller
// ---------------------------------------------------------------------------
wire [31:0] ctrl_reg, m_dim_r, n_dim_r, k_dim_r;
wire [31:0] w_addr_r, a_addr_r, r_addr_r, bias_addr_r, quant_cfg_r;
wire [31:0] desc_base_r, desc_count_r;
wire [31:0] conv_ifm_shape_r, conv_channels_r, conv_kernel_r;
wire [31:0] conv_out_shape_r, conv_stride_pad_r, conv_dilation_r;
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
wire [63:0] perf_mac_ops, perf_ops;
wire [31:0] perf_busy_cycles, perf_compute_cycles, perf_dma_cycles;
wire [31:0] perf_tops_x1e6, perf_compute_util_bp, perf_e2e_util_bp;
wire [31:0] perf_peak_ops_per_cycle;
wire        perf_clear;

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
    .bias_addr  (bias_addr_r),
    .quant_cfg  (quant_cfg_r),
    .arr_cfg    (arr_cfg_r),
    .clk_div    (clk_div_r),
    .cg_en      (cg_en_r),
    .cfg_shape  (cfg_shape_r),
    .desc_base  (desc_base_r),
    .desc_count (desc_count_r),
    .conv_ifm_shape(conv_ifm_shape_r),
    .conv_channels(conv_channels_r),
    .conv_kernel(conv_kernel_r),
    .conv_out_shape(conv_out_shape_r),
    .conv_stride_pad(conv_stride_pad_r),
    .conv_dilation(conv_dilation_r),
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
    .perf_mac_ops(perf_mac_ops),
    .perf_ops(perf_ops),
    .perf_busy_cycles(perf_busy_cycles),
    .perf_compute_cycles(perf_compute_cycles),
    .perf_dma_cycles(perf_dma_cycles),
    .perf_tops_x1e6(perf_tops_x1e6),
    .perf_compute_util_bp(perf_compute_util_bp),
    .perf_e2e_util_bp(perf_e2e_util_bp),
    .perf_peak_ops_per_cycle(perf_peak_ops_per_cycle),
    .perf_clear (perf_clear),
    .npu_irq    (npu_irq)
);

// ---------------------------------------------------------------------------
// NPU Controller
// ---------------------------------------------------------------------------
wire dma_w_start, dma_a_start, dma_r_start;
wire dma_w_done,  dma_a_done,  dma_bias_done, dma_r_done;
wire [31:0] dma_w_addr, dma_a_addr, dma_r_addr;
wire [15:0] dma_w_len,  dma_a_len,  dma_r_len;
wire dma_bias_start;
wire [31:0] dma_bias_addr;
wire [31:0] dma_bias_data;
wire dma_a_ofm_mode;
wire dma_a_im2col_mode;
wire [31:0] dma_a_ofm_stride;
wire [31:0] dma_a_ofm_m_base;
wire [31:0] dma_a_ofm_k_base;
wire [15:0] dma_a_ofm_k_len;
wire [4:0]  dma_a_ofm_active_rows;
wire dma_a_ofm_fp16_mode;
wire [31:0] dma_a_im2col_m_index;
wire [15:0] dma_a_im2col_k_len;
wire [15:0] dma_a_im2col_ih, dma_a_im2col_iw, dma_a_im2col_cin;
wire [15:0] dma_a_im2col_kh, dma_a_im2col_kw, dma_a_im2col_oh, dma_a_im2col_ow;
wire [7:0]  dma_a_im2col_stride_h, dma_a_im2col_stride_w;
wire [7:0]  dma_a_im2col_pad_h, dma_a_im2col_pad_w;
wire [7:0]  dma_a_im2col_dilation_h, dma_a_im2col_dilation_w;
wire        dma_a_im2col_fp16_mode;
wire desc_fetch_start, desc_fetch_done;
wire [31:0] desc_fetch_addr;
wire [511:0] desc_fetch_words;
wire [31:0] dma_error_status;
wire pe_en, pe_flush, pe_mode, pe_stat;
wire pe_mode_hw = (FP16_ENABLE != 0) && pe_mode;
wire pe_load_w, pe_swap_w, pe_acc_init_en;   // WS mode weight control and accumulator init
wire pe_half_en;           // 8x32: half-array enable (0=top, 1=bottom)
wire ctrl_w_ppb_swap, ctrl_a_ppb_swap, ctrl_w_ppb_clear, ctrl_a_ppb_clear;
wire ctrl_r_fifo_clear;
wire [1:0] ctrl_cfg_shape;
wire [1:0] ctrl_post_act_mode;
wire [31:0] ctrl_post_quant_cfg;
wire       ctrl_bias_en;
wire ctrl_tile_mode, ctrl_vec_consume;
wire ppb_packed_int8 = ctrl_tile_mode && !pe_mode_hw && (INT8_SIMD_LANES > 1);
wire tile_ws_direct = ctrl_tile_mode && !pe_mode_hw && !pe_stat;
wire [31:0] ctrl_tile_m_base, ctrl_tile_n_base; // global C tile origin: m0/n0
wire [15:0] ctrl_tile_row_valid, ctrl_tile_col_valid; // valid r/c lanes for edge tiles
wire [4:0]  ctrl_tile_active_rows;
wire [5:0]  ctrl_tile_active_cols; // row/col count to serialize
wire [31:0] ctrl_tile_k_base, ctrl_tile_k_index;
wire [15:0] ctrl_tile_k_len;
wire [15:0] ctrl_tile_k_cycle; // OS feeder cycle, includes row-skew drain cycles
wire [4:0]  tile_lane_count = shape_tile_lanes(ctrl_cfg_shape);

npu_ctrl #(
    .ROWS  (PHY_ROWS),       // controller sees physical dimensions
    .COLS  (PHY_COLS),
    .DATA_W(DATA_W),
    .ACC_W (ACC_W),
    .PPB_DEPTH(PPB_DEPTH),
    .INT8_SIMD_LANES(INT8_SIMD_LANES),
    .FP16_ENABLE(FP16_ENABLE)
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
    .bias_addr    (bias_addr_r),
    .quant_cfg    (quant_cfg_r),
    .arr_cfg      (arr_cfg_r),
    .desc_base    (desc_base_r),
    .desc_count   (desc_count_r),
    .conv_ifm_shape(conv_ifm_shape_r),
    .conv_channels(conv_channels_r),
    .conv_kernel  (conv_kernel_r),
    .conv_out_shape(conv_out_shape_r),
    .conv_stride_pad(conv_stride_pad_r),
    .conv_dilation(conv_dilation_r),
    .desc_start   (desc_fetch_start),
    .desc_addr    (desc_fetch_addr),
    .desc_done    (desc_fetch_done),
    .desc_words   (desc_fetch_words),
    .cfg_shape_in (cfg_shape_r),
    .cfg_shape_latched(ctrl_cfg_shape),
    .post_act_mode(ctrl_post_act_mode),
    .post_quant_cfg(ctrl_post_quant_cfg),
    .bias_en    (ctrl_bias_en),
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
    .dma_a_im2col_mode(dma_a_im2col_mode),
    .dma_a_ofm_stride(dma_a_ofm_stride),
    .dma_a_ofm_m_base(dma_a_ofm_m_base),
    .dma_a_ofm_k_base(dma_a_ofm_k_base),
    .dma_a_ofm_k_len(dma_a_ofm_k_len),
    .dma_a_ofm_active_rows(dma_a_ofm_active_rows),
    .dma_a_ofm_fp16_mode(dma_a_ofm_fp16_mode),
    .dma_a_im2col_m_index(dma_a_im2col_m_index),
    .dma_a_im2col_k_len(dma_a_im2col_k_len),
    .dma_a_im2col_ih(dma_a_im2col_ih),
    .dma_a_im2col_iw(dma_a_im2col_iw),
    .dma_a_im2col_cin(dma_a_im2col_cin),
    .dma_a_im2col_kh(dma_a_im2col_kh),
    .dma_a_im2col_kw(dma_a_im2col_kw),
    .dma_a_im2col_oh(dma_a_im2col_oh),
    .dma_a_im2col_ow(dma_a_im2col_ow),
    .dma_a_im2col_stride_h(dma_a_im2col_stride_h),
    .dma_a_im2col_stride_w(dma_a_im2col_stride_w),
    .dma_a_im2col_pad_h(dma_a_im2col_pad_h),
    .dma_a_im2col_pad_w(dma_a_im2col_pad_w),
    .dma_a_im2col_dilation_h(dma_a_im2col_dilation_h),
    .dma_a_im2col_dilation_w(dma_a_im2col_dilation_w),
    .dma_a_im2col_fp16_mode(dma_a_im2col_fp16_mode),
    .dma_bias_start(dma_bias_start),
    .dma_bias_done (dma_bias_done),
    .dma_bias_addr (dma_bias_addr),
    .dma_r_start  (dma_r_start),
    .dma_r_done   (dma_r_done),
    .dma_r_addr   (dma_r_addr),
    .dma_r_len    (dma_r_len),
    .dma_error_status(dma_error_status),
    .pe_en        (pe_en),
    .pe_flush     (pe_flush),
    .pe_mode      (pe_mode),
    .pe_stat      (pe_stat),
    .pe_load_w    (pe_load_w),
    .pe_swap_w    (pe_swap_w),
    .pe_acc_init_en(pe_acc_init_en),
    .pe_half_en   (pe_half_en),
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
wire [MAX_TILE_LANES*DATA_W-1:0] w_ppb_rd_vec, a_ppb_rd_vec;
wire        w_ppb_rd_vec_valid, a_ppb_rd_vec_valid;

wire w_ppb_buf_empty_int, a_ppb_buf_empty_int;
wire w_ppb_buf_ready_int, a_ppb_buf_ready_int;

// Weight Ping-Pong Buffer
pingpong_buf #(
    .DATA_W    (ACC_W),
    .DEPTH     (PPB_DEPTH),
    .OUT_WIDTH (DATA_W),
    .THRESHOLD (PPB_THRESH),
    .SUBW      (4),
    .VEC_LANES (MAX_TILE_LANES),
    .SCALAR_READ_ENABLE(PPB_SCALAR_READ_ENABLE)
) u_w_ppb (
    .clk       (sys_clk),
    .rst_n     (sys_rst_n),
    .wr_en     (w_ppb_wr_en),
    .wr_data   (w_ppb_wr_data),
    .rd_en     (w_ppb_rd_en),
    .rd_data   (w_ppb_rd_data),
    .rd_vec_en  (w_ppb_rd_vec_en),
    .rd_vec_lanes(tile_lane_count),
    .rd_vec     (w_ppb_rd_vec),
    .rd_vec_valid(w_ppb_rd_vec_valid),
    .swap      (ctrl_w_ppb_swap),
    .clear     (ctrl_w_ppb_clear),
    .packed_int8(ppb_packed_int8),
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
    .SUBW      (4),
    .VEC_LANES (MAX_TILE_LANES),
    .SCALAR_READ_ENABLE(PPB_SCALAR_READ_ENABLE)
) u_a_ppb (
    .clk       (sys_clk),
    .rst_n     (sys_rst_n),
    .wr_en     (a_ppb_wr_en),
    .wr_data   (a_ppb_wr_data),
    .rd_en     (a_ppb_rd_en),
    .rd_data   (a_ppb_rd_data),
    .rd_vec_en  (a_ppb_rd_vec_en),
    .rd_vec_lanes(tile_lane_count),
    .rd_vec     (a_ppb_rd_vec),
    .rd_vec_valid(a_ppb_rd_vec_valid),
    .swap      (ctrl_a_ppb_swap),
    .clear     (ctrl_a_ppb_clear),
    .packed_int8(ppb_packed_int8),
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
    .R_FIFO_DEPTH(256)
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
    .a_im2col_mode (dma_a_im2col_mode),
    .a_ofm_stride  (dma_a_ofm_stride),
    .a_ofm_m_base  (dma_a_ofm_m_base),
    .a_ofm_k_base  (dma_a_ofm_k_base),
    .a_ofm_k_len   (dma_a_ofm_k_len),
    .a_ofm_active_rows(dma_a_ofm_active_rows),
    .a_ofm_fp16_mode(dma_a_ofm_fp16_mode),
    .a_im2col_m_index(dma_a_im2col_m_index),
    .a_im2col_k_len(dma_a_im2col_k_len),
    .a_im2col_ih(dma_a_im2col_ih),
    .a_im2col_iw(dma_a_im2col_iw),
    .a_im2col_cin(dma_a_im2col_cin),
    .a_im2col_kh(dma_a_im2col_kh),
    .a_im2col_kw(dma_a_im2col_kw),
    .a_im2col_oh(dma_a_im2col_oh),
    .a_im2col_ow(dma_a_im2col_ow),
    .a_im2col_stride_h(dma_a_im2col_stride_h),
    .a_im2col_stride_w(dma_a_im2col_stride_w),
    .a_im2col_pad_h(dma_a_im2col_pad_h),
    .a_im2col_pad_w(dma_a_im2col_pad_w),
    .a_im2col_dilation_h(dma_a_im2col_dilation_h),
    .a_im2col_dilation_w(dma_a_im2col_dilation_w),
    .a_im2col_fp16_mode(dma_a_im2col_fp16_mode),
    // Descriptor fetch
    .desc_start     (desc_fetch_start),
    .desc_base_addr (desc_fetch_addr),
    .desc_done      (desc_fetch_done),
    .desc_words     (desc_fetch_words),
    // Bias fetch
    .bias_start     (dma_bias_start),
    .bias_addr      (dma_bias_addr),
    .bias_done      (dma_bias_done),
    .bias_data      (dma_bias_data),
    // Result channel
    .r_start        (dma_r_start),
    .r_base_addr    (dma_r_addr),
    .r_len_bytes    (dma_r_len),
    .r_done         (dma_r_done),
    .r_fifo_clear   (ctrl_r_fifo_clear),
    .r_fifo_wr_en   (r_fifo_wr_en),
    .r_fifo_din     (r_fifo_din),
    .r_fifo_full    (r_fifo_full),
    .dma_err_status (dma_error_status),
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
    .ACC_W(ACC_W),
    .ENABLE_DERIVED(PERF_ENABLE_DERIVED)
) u_axi_monitor (
    .clk(sys_clk),
    .rst_n(sys_rst_n),
    .clear(perf_clear),
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
// have enough lanes for the controller's current shape and k cycle.
wire tile_vec_fire = ctrl_vec_consume &&
                     w_ppb_rd_vec_valid &&
                     a_ppb_rd_vec_valid;
wire tile_feed_step = ctrl_tile_mode && pe_en && pe_stat && !pe_flush;

assign w_ppb_rd_en = ctrl_tile_mode ? 1'b0 : pe_consume;
assign a_ppb_rd_en = ctrl_tile_mode ? 1'b0 : pe_consume;
assign w_ppb_rd_vec_en = tile_vec_fire;
assign a_ppb_rd_vec_en = tile_vec_fire;

`ifdef DIAG_FIRE_K5
always @(posedge sys_clk) begin
    if (tile_vec_fire) begin
        $display("[DIAG_FIRE] rd_fill_a=%0d rd_fill_b=%0d rd_ptr_a=%0d rd_ptr_b=%0d lane0=0x%08h",
                 u_w_ppb.rd_fill_a, u_w_ppb.rd_fill_b,
                 u_w_ppb.rd_ptr_a, u_w_ppb.rd_ptr_b,
                 w_ppb_rd_vec[0*DATA_W +: DATA_W]);
    end
end
`endif

`ifdef VERILATOR_TRACE
always @(posedge sys_clk) begin
    if (ctrl_vec_consume)
        $display("[V_TRACE] fire: w_lane0=0x%08h a_lane0=0x%08h",
                 w_ppb_rd_vec[0*DATA_W +: DATA_W],
                 a_ppb_rd_vec[0*DATA_W +: DATA_W]);
end
`endif

// ---------------------------------------------------------------------------
// Reconfigurable PE Array (16×16 physical)
// ---------------------------------------------------------------------------

wire [PHY_COLS*DATA_W-1:0] pe_w_in;
wire [PHY_ROWS*DATA_W-1:0] pe_a_in;
wire [PHY_COLS*ACC_W-1:0]  pe_acc_in;
wire [PHY_ROWS*PHY_COLS*ACC_W-1:0] pe_acc_init;
wire [PHY_ROWS*PHY_COLS-1:0] pe_acc_init_mask;
wire [MAX_TILE_RESULTS*ACC_W-1:0] pe_array_result;
wire [MAX_TILE_RESULTS-1:0]       pe_array_valid;
wire [PHY_ROWS*PHY_COLS-1:0] pe_active_dbg;
wire pe_array_global_ce;
wire [PHY_ROWS-1:0] pe_array_row_ce;
wire [PHY_COLS-1:0] pe_array_col_ce;

wire [PHY_ROWS-1:0] power_row_cg = cg_en_r ? ~shape_row_ce_mask(ctrl_cfg_shape)
                                           : {PHY_ROWS{1'b0}};
wire [PHY_COLS-1:0] power_col_cg = cg_en_r ? ~shape_col_ce_mask(ctrl_cfg_shape)
                                           : {PHY_COLS{1'b0}};

// CLK_DIV is intentionally held at 1x for the compute CE path until the
// controller/DMA scheduler is made CE-aware. Row/column CE is safe today.
npu_power #(
    .ROWS(PHY_ROWS),
    .COLS(PHY_COLS)
) u_power (
    .clk         (sys_clk),
    .rst_n       (sys_rst_n),
    .div_sel     (3'b000),
    .row_cg_en   (power_row_cg),
    .col_cg_en   (power_col_cg),
    .global_ce   (pe_array_global_ce),
    .row_ce      (pe_array_row_ce),
    .col_ce      (pe_array_col_ce),
    .npu_clk     (),
    .row_clk_gated(),
    .col_clk_gated()
);

// Data source from PPBuf
wire [DATA_W-1:0] pe_w_data = w_ppb_rd_data;
wire [DATA_W-1:0] pe_a_data = a_ppb_rd_data;
wire [MAX_TILE_LANES*DATA_W-1:0] a_skew_vec;
wire [MAX_TILE_LANES*DATA_W-1:0] tile_a_feed_vec = tile_ws_direct
    ? (tile_vec_fire ? a_ppb_rd_vec : {MAX_TILE_LANES*DATA_W{1'b0}})
    : a_skew_vec;

// OS row-skew feeder. Row r receives A lane r after r compute cycles, so the
// A wavefront stays aligned with the W stream as it shifts down the array.
genvar skew_lane;
generate
    for (skew_lane = 0; skew_lane < MAX_TILE_LANES; skew_lane = skew_lane + 1) begin : gen_a_skew
        localparam integer LANE = skew_lane;
        if (LANE == 0) begin : gen_lane0
            assign a_skew_vec[LANE*DATA_W +: DATA_W] =
                (ctrl_tile_mode && tile_vec_fire) ? a_ppb_rd_vec[LANE*DATA_W +: DATA_W]
                                                   : {DATA_W{1'b0}};
        end else if (LANE == 1) begin : gen_lane1_reg
            // Single-cycle registered delay: captures on vec_fire, holds during
            // drain so the value is not lost when the pipe input is later cleared.
            reg [DATA_W-1:0] l1_reg;
            always @(posedge sys_clk) begin
                if (!sys_rst_n || !ctrl_tile_mode)
                    l1_reg <= {DATA_W{1'b0}};
                else if (tile_vec_fire)
                    l1_reg <= a_ppb_rd_vec[LANE*DATA_W +: DATA_W];
            end
            assign a_skew_vec[LANE*DATA_W +: DATA_W] = l1_reg;
        end else begin : gen_lane_pipe
            reg [DATA_W-1:0] pipe [0:LANE-1];
            integer stage_i;
            always @(posedge sys_clk) begin
                if (!sys_rst_n || !ctrl_tile_mode) begin
                    for (stage_i = 0; stage_i < LANE; stage_i = stage_i + 1)
                        pipe[stage_i] <= {DATA_W{1'b0}};
                end else if (tile_feed_step) begin
                    pipe[0] <= tile_vec_fire ? a_ppb_rd_vec[LANE*DATA_W +: DATA_W]
                                             : {DATA_W{1'b0}};
                    for (stage_i = 1; stage_i < LANE; stage_i = stage_i + 1)
                        pipe[stage_i] <= pipe[stage_i-1];
                end
            end
            assign a_skew_vec[LANE*DATA_W +: DATA_W] = pipe[LANE-1];
        end
    end
endgenerate

// ---------------------------------------------------------------------------
// Diagnostic: log A-skew outputs and weight inputs every cycle after fire
// ---------------------------------------------------------------------------
`ifdef DIAG_SKEW
reg [15:0] diag_cyc;
reg        diag_active;
reg [511:0] diag_a_fire, diag_w_fire;
integer    diag_r;

always @(posedge sys_clk) begin
    if (!sys_rst_n || !ctrl_tile_mode) begin
        diag_cyc    <= 16'd0;
        diag_active <= 1'b0;
    end else if (tile_vec_fire && !diag_active) begin
        diag_active  <= 1'b1;
        diag_cyc     <= 16'd0;
        diag_a_fire  <= a_ppb_rd_vec;
        diag_w_fire  <= w_ppb_rd_vec;
    end else if (diag_active && diag_cyc < 20) begin
        diag_cyc <= diag_cyc + 16'd1;
    end
end

always @(posedge sys_clk) begin
    if (diag_active && diag_cyc < 20) begin
        for (diag_r = 0; diag_r < 16; diag_r = diag_r + 1) begin
            if (a_skew_vec[diag_r*DATA_W +: DATA_W] != 0)
                $display("[DIAG] cyc=%0d askew[%0d]=0x%08h (fire_A[%0d]=0x%08h)",
                         diag_cyc, diag_r,
                         a_skew_vec[diag_r*DATA_W +: DATA_W],
                         diag_r,
                         diag_a_fire[diag_r*DATA_W +: DATA_W]);
        end
    end
end
`endif

genvar feed_col;
generate
    for (feed_col = 0; feed_col < PHY_COLS; feed_col = feed_col + 1) begin : gen_pe_w_in
        localparam [4:0] COL_IDX = feed_col;
        if (feed_col < MAX_TILE_LANES) begin : gen_tile_w_lane
            assign pe_w_in[feed_col*DATA_W +: DATA_W] =
                ctrl_tile_mode
                    ? ((tile_vec_fire && (COL_IDX < tile_lane_count))
                        ? w_ppb_rd_vec[feed_col*DATA_W +: DATA_W]
                        : {DATA_W{1'b0}})
                    : ((feed_col == 0) ? pe_w_data : {DATA_W{1'b0}});
        end else begin : gen_zero_w_lane
            assign pe_w_in[feed_col*DATA_W +: DATA_W] = {DATA_W{1'b0}};
        end
    end
endgenerate

genvar feed_row;
generate
    for (feed_row = 0; feed_row < PHY_ROWS; feed_row = feed_row + 1) begin : gen_pe_a_in
        localparam [4:0] ROW_IDX = feed_row;
        if (feed_row < MAX_TILE_LANES) begin : gen_tile_a_lane
            assign pe_a_in[feed_row*DATA_W +: DATA_W] =
                ctrl_tile_mode
                    ? ((ROW_IDX < tile_lane_count)
                        ? tile_a_feed_vec[feed_row*DATA_W +: DATA_W]
                        : {DATA_W{1'b0}})
                    : ((feed_row == 0) ? pe_a_data : {DATA_W{1'b0}});
        end else begin : gen_zero_a_lane
            assign pe_a_in[feed_row*DATA_W +: DATA_W] = {DATA_W{1'b0}};
        end
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
wire [ACC_W-1:0] scalar_act_result;
wire [ACC_W-1:0] scalar_post_result;
wire             scalar_valid;

wire [4:0] perf_active_rows = ctrl_tile_mode
    ? ((ctrl_cfg_shape == 2'b11) ? 5'd8 : tile_lane_count)
    : 5'd1;
wire [5:0] perf_active_cols = ctrl_tile_mode
    ? ((ctrl_cfg_shape == 2'b11) ? 6'd32 : {1'b0, tile_lane_count})
    : 6'd1;
wire       tile_ws_perf_compute = tile_ws_direct && pe_en && !pe_flush &&
                                  (ctrl_vec_consume || (ctrl_tile_k_cycle != 16'd0));
wire       perf_compute_valid = ctrl_tile_mode ? (tile_feed_step || tile_ws_perf_compute)
                                               : scalar_pe_en;

op_counter #(
    .ROWS(PHY_ROWS),
    .COLS(MAX_TILE_RESULTS),
    .FREQ_MHZ(500),
    .ENABLE_DERIVED(PERF_ENABLE_DERIVED)
) u_op_counter (
    .clk(sys_clk),
    .rst_n(sys_rst_n),
    .clear(perf_clear),
    .pe_en(pe_en),
    .pe_flush(pe_flush),
    .ctrl_busy(status_busy),
    .ctrl_done(status_done),
    .dma_w_done(dma_w_done),
    .dma_a_done(dma_a_done),
    .dma_r_done(dma_r_done),
    .pe_valid(pe_array_valid),
    .m_dim(m_dim_r),
    .n_dim(n_dim_r),
    .k_dim(k_dim_r),
    .compute_valid(perf_compute_valid),
    .active_rows(perf_active_rows),
    .active_cols(perf_active_cols),
    .simd_lanes(PERF_SIMD_LANES),
    .total_mac_ops(perf_mac_ops),
    .total_busy_cycles(perf_busy_cycles),
    .total_compute_cycles(perf_compute_cycles),
    .total_dma_cycles(perf_dma_cycles),
    .total_ops(perf_ops),
    .peak_ops_per_cycle(perf_peak_ops_per_cycle),
    .tops_x1e6(perf_tops_x1e6),
    .compute_util_bp(perf_compute_util_bp),
    .e2e_util_bp(perf_e2e_util_bp)
);

pe_top #(
    .DATA_W(DATA_W),
    .ACC_W (ACC_W),
    .INT8_SIMD_LANES(INT8_SIMD_LANES),
    .FP16_ENABLE(FP16_ENABLE)
) u_scalar_pe (
    .clk      (sys_clk),
    .rst_n    (sys_rst_n),
    .mode     (pe_mode_hw),
    .stat_mode(pe_stat),
    .en       (scalar_pe_en),
    .flush    (pe_flush),
    .load_w   (pe_load_w),
    .swap_w   (pe_swap_w),
    .acc_init_en(pe_acc_init_en && !ctrl_tile_mode),
    .w_in     (pe_w_data),
    .a_in     (pe_a_data),
    .acc_in   ({ACC_W{1'b0}}),
    .acc_init (dma_bias_data),
    .acc_out  (scalar_result),
    .valid_out(scalar_valid)
);

function [31:0] apply_scalar_activation;
    input [31:0] value;
    input        is_fp16;
    input [1:0]  act_mode;
    reg signed [31:0] value_s;
    begin
        value_s = value;
        if (act_mode == 2'b01) begin
            if (is_fp16)
                apply_scalar_activation = value[31] ? 32'd0 : value;
            else
                apply_scalar_activation = (value_s < 0) ? 32'd0 : value;
        end else if (act_mode == 2'b10) begin
            if (is_fp16) begin
                if (value[31])
                    apply_scalar_activation = 32'd0;
                else if (value > 32'h40c0_0000)
                    apply_scalar_activation = 32'h40c0_0000; // 6.0f
                else
                    apply_scalar_activation = value;
            end else begin
                if (value_s < 0)
                    apply_scalar_activation = 32'd0;
                else if (value_s > 32'sd6)
                    apply_scalar_activation = 32'd6;
                else
                    apply_scalar_activation = value;
            end
        end else begin
            apply_scalar_activation = value;
        end
    end
endfunction

function [31:0] apply_scalar_quant;
    input [31:0] value;
    input        is_fp16;
    input [31:0] quant_cfg;
    reg          quant_en;
    reg          round_en;
    reg [7:0]    shift_u;
    reg signed [15:0] scale_s;
    reg signed [63:0] value_s;
    reg signed [63:0] scale_ext_s;
    reg signed [63:0] prod_s;
    reg signed [63:0] rounded_s;
    reg signed [63:0] shifted_s;
    reg signed [63:0] round_offset_s;
    reg signed [7:0]  q8_s;
    begin
        quant_en = quant_cfg[0];
        round_en = quant_cfg[1];
        shift_u = quant_cfg[15:8];
        scale_s = quant_cfg[31:16];

        if (!quant_en || is_fp16) begin
            apply_scalar_quant = value;
        end else begin
            value_s = {{32{value[31]}}, value};
            scale_ext_s = {{48{scale_s[15]}}, scale_s};
            prod_s = value_s * scale_ext_s;
            rounded_s = prod_s;

            if (round_en && (shift_u != 8'd0) && (shift_u < 8'd63)) begin
                round_offset_s = 64'sd1 <<< (shift_u - 8'd1);
                rounded_s = prod_s[63] ? (prod_s + round_offset_s - 64'sd1)
                                       : (prod_s + round_offset_s);
            end

            if (shift_u >= 8'd63)
                shifted_s = rounded_s[63] ? -64'sd1 : 64'sd0;
            else
                shifted_s = rounded_s >>> shift_u[5:0];

            if (shifted_s > 64'sd127)
                q8_s = 8'sd127;
            else if (shifted_s < -64'sd128)
                q8_s = -8'sd128;
            else
                q8_s = shifted_s[7:0];

            apply_scalar_quant = {{24{q8_s[7]}}, q8_s[7:0]};
        end
    end
endfunction

assign scalar_act_result = apply_scalar_activation(scalar_result,
                                                   pe_mode_hw,
                                                   ctrl_post_act_mode);
assign scalar_post_result = apply_scalar_quant(scalar_act_result,
                                               pe_mode_hw,
                                               ctrl_post_quant_cfg);

// Tile-mode post-processing: bias -> activation -> quant.
// Order: accumulator -> bias -> ReLU/ReLU6 -> quant/saturate.
wire [31:0] tile_bias_val = ctrl_bias_en ? tile_bias_buf[tile_ser_col[4:0]] : 32'd0;
wire [ACC_W-1:0] tile_with_bias = tile_result_buf[tile_ser_idx] + tile_bias_val;
wire [ACC_W-1:0] tile_act_result;
wire [ACC_W-1:0] tile_post_result;
assign tile_act_result = apply_scalar_activation(tile_with_bias,
                                                   pe_mode_hw,
                                                   ctrl_post_act_mode);
assign tile_post_result = apply_scalar_quant(tile_act_result,
                                             pe_mode_hw,
                                             ctrl_post_quant_cfg);

// WS load row indicator (for debug/status)
wire [3:0] ws_load_row_status;

reconfig_pe_array #(
    .PHY_ROWS(PHY_ROWS),
    .PHY_COLS(PHY_COLS),
    .DATA_W  (DATA_W),
    .ACC_W   (ACC_W),
    .MAX_TILE_RESULTS(MAX_TILE_RESULTS),
    .INT8_SIMD_LANES(INT8_SIMD_LANES),
    .FP16_ENABLE(FP16_ENABLE)
) u_pe_array (
    .clk            (sys_clk),
    .rst_n          (sys_rst_n),
    .cfg_shape      (ctrl_cfg_shape),
    .mode           (pe_mode_hw),
    .stat_mode      (pe_stat),
    .en             (pe_en),
    .flush          (pe_flush),
    .load_w         (pe_load_w),
    .swap_w         (pe_swap_w),
    .ws_direct      (tile_ws_direct),
    .acc_init_en    (1'b0),
    .half_en        (pe_half_en),
    .array_ce       (pe_array_global_ce),
    .row_ce         (pe_array_row_ce),
    .col_ce         (pe_array_col_ce),
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
// Tile-mode bias buffer.  Captures per-column bias words as the controller
// fetches them sequentially via dma_bias_start / dma_bias_data.
// ---------------------------------------------------------------------------
reg [31:0] tile_bias_buf [0:TILE_BIAS_DEPTH-1];
reg [5:0]  tile_bias_idx;
reg        tile_bias_fetch_active;
reg        tile_bias_ctrl_start_d1;
wire       tile_bias_ctrl_start_rise = ctrl_reg[0] && !tile_bias_ctrl_start_d1;

always @(posedge sys_clk) begin
    if (!sys_rst_n)
        tile_bias_ctrl_start_d1 <= 1'b0;
    else
        tile_bias_ctrl_start_d1 <= ctrl_reg[0];
end

always @(posedge sys_clk) begin
    if (!sys_rst_n || !ctrl_tile_mode || !ctrl_bias_en || tile_bias_ctrl_start_rise) begin
        tile_bias_idx <= 6'd0;
        tile_bias_fetch_active <= 1'b0;
    end else if (dma_bias_start && !tile_bias_fetch_active) begin
        tile_bias_idx <= 6'd0;
        tile_bias_fetch_active <= 1'b1;
    end else if (dma_bias_done && ctrl_bias_en) begin
        tile_bias_buf[tile_bias_idx[4:0]] <= dma_bias_data;
        if (tile_bias_idx + 6'd1 >= {1'b0, ctrl_tile_active_cols}) begin
            tile_bias_idx <= 6'd0;
            tile_bias_fetch_active <= 1'b0;
        end else begin
            tile_bias_idx <= tile_bias_idx + 1'b1;
        end
    end
end

// ---------------------------------------------------------------------------
// Result FIFO interface — tile-mode serializer.
//
//   Captures the full PE grid output (up to 256 results), then pushes
//   only the active rows/cols in row-major order into the result FIFO
//   for DMA row-wise writeback.
//
//   grid_rows = (8x32 ? 8 : tile_lane_count)
//   grid_cols = (8x32 ? 32 : tile_lane_count)
//   capture   = grid_rows * grid_cols
// ---------------------------------------------------------------------------
wire [4:0] tile_grid_rows = (ctrl_cfg_shape == 2'b11) ? 5'd8  : tile_lane_count;
wire [5:0] tile_grid_cols = (ctrl_cfg_shape == 2'b11) ? 6'd32 : {1'b0, tile_lane_count};
wire [8:0] tile_capture_cnt = {3'd0, tile_grid_rows} * {2'd0, tile_grid_cols}; // max 256

reg [ACC_W-1:0] tile_result_buf [0:255]; // captured C tile, index = row*grid_cols+col
reg             tile_ser_busy;
reg [4:0]       tile_ser_active_rows;   // valid rows in current edge/full tile
reg [5:0]       tile_ser_active_cols;   // valid cols (6-bit for 8x32)
reg [4:0]       tile_ser_row;           // serializer row r
reg [5:0]       tile_ser_col;           // serializer col c

wire [7:0] tile_ser_idx = tile_ser_row * tile_grid_cols + {2'd0, tile_ser_col};
wire       tile_ser_fire = tile_ser_busy && !r_fifo_full;
wire       tile_ser_last_col = (tile_ser_col + 6'd1 >= tile_ser_active_cols);
wire       tile_ser_last_row = (tile_ser_row + 5'd1 >= tile_ser_active_rows);
wire       tile_ser_last     = tile_ser_last_col && tile_ser_last_row;
wire tile_result_capture_pending = ctrl_tile_mode &&
                              pe_array_valid[0] &&
                              !tile_ser_busy;

`ifdef DIAG_8X32
always @(posedge sys_clk) begin
    if (tile_result_capture_pending)
        $display("[DIAG_CAP] capture pending: tile_ser_busy=%0d pe_flush=%0d",
                 tile_ser_busy, pe_flush);
end
`endif

// Delay capture by 1 cycle so acc_out NB from flush has taken effect
reg tile_cap_d1;
always @(posedge sys_clk) begin
    if (!sys_rst_n || !ctrl_tile_mode)
        tile_cap_d1 <= 1'b0;
    else
        tile_cap_d1 <= tile_result_capture_pending;
end
wire tile_result_capture = tile_cap_d1;

integer tile_ser_i;
always @(posedge sys_clk) begin
    if (!sys_rst_n || !ctrl_tile_mode) begin
        tile_ser_busy       <= 1'b0;
        tile_ser_active_rows <= 5'd0;
        tile_ser_active_cols <= 6'd0;
        tile_ser_row        <= 5'd0;
        tile_ser_col        <= 6'd0;
        for (tile_ser_i = 0; tile_ser_i < MAX_TILE_RESULTS; tile_ser_i = tile_ser_i + 1)
            tile_result_buf[tile_ser_i] <= {ACC_W{1'b0}};
    end else if (tile_result_capture) begin
        // Fixed loop bound keeps synthesis tools from treating tile_capture_cnt
        // as a dynamic generate-style limit.
        for (tile_ser_i = 0; tile_ser_i < MAX_TILE_RESULTS; tile_ser_i = tile_ser_i + 1) begin
            if (tile_ser_i < tile_capture_cnt)
                tile_result_buf[tile_ser_i] <= pe_array_result[tile_ser_i*ACC_W +: ACC_W];
        end
        tile_ser_active_rows <= ctrl_tile_active_rows;
        tile_ser_active_cols <= ctrl_tile_active_cols;
        tile_ser_row         <= 5'd0;
        tile_ser_col         <= 6'd0;
        tile_ser_busy        <= (ctrl_tile_active_rows != 5'd0) &&
                                (ctrl_tile_active_cols != 6'd0);
        `ifdef DIAG_8X32
        $display("[DIAG_CAP] capture done: buf[0]=0x%08h buf[16]=0x%08h",
                 pe_array_result[0*ACC_W +: ACC_W],
                 pe_array_result[16*ACC_W +: ACC_W]);
        `endif
    end else if (tile_ser_fire) begin
        if (tile_ser_last) begin
            tile_ser_busy <= 1'b0;
        end else if (tile_ser_last_col) begin
            tile_ser_row <= tile_ser_row + 5'd1;
            tile_ser_col <= 6'd0;
        end else begin
            tile_ser_col <= tile_ser_col + 6'd1;
        end
        `ifdef DIAG_8X32
        if (tile_ser_idx == 16 || tile_ser_idx == 0)
            $display("[DIAG_SER] serializer fire idx=%0d r=%0d c=%0d din=0x%08h",
                     tile_ser_idx, tile_ser_row, tile_ser_col, r_fifo_din);
        `endif
    end
end

assign r_fifo_din   = ctrl_tile_mode ? tile_post_result : scalar_post_result;
assign r_fifo_wr_en = ctrl_tile_mode ? tile_ser_fire
                                     : (scalar_valid && !r_fifo_full);

// ---------------------------------------------------------------------------
// Diagnostic: log serializer output to FIFO
// ---------------------------------------------------------------------------
`ifdef DIAG_FIFO
reg [8:0] diag_fifo_idx;
always @(posedge sys_clk) begin
    if (!sys_rst_n || !ctrl_tile_mode)
        diag_fifo_idx <= 9'd0;
    else if (r_fifo_wr_en) begin
        $display("[DIAG_FIFO] seq=%0d (r=%0d c=%0d) data=0x%08h",
                 diag_fifo_idx, tile_ser_row, tile_ser_col, r_fifo_din);
        diag_fifo_idx <= diag_fifo_idx + 9'd1;
    end
end
`endif

endmodule
