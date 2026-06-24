// =============================================================================
// Module  : npu_ctrl
// Project : NPU_prj
// Desc    : NPU top-level controller FSM — True Ping-Pong Overlap Edition
//
// ── Architecture ──────────────────────────────────────────────────────────
//
//   This controller implements a TRUE PIPELINE between DMA Load and PE Compute,
//   eliminating the serial "Load → Compute → WB" stall.
//
//   Tile execution pipeline (steady-state):
//
//     Cycle:  T0           T1           T2           T3
//             LOAD(0,0)    COMPUTE(0,0) COMPUTE(0,0) ...
//                          LOAD(0,1)    LOAD(0,1)    COMPUTE(0,1)
//                                                    LOAD(0,2)
//
//   Three execution phases:
//     Phase 0 – Warm-up  : Load tile(0,0) only. PE idle.
//     Phase 1 – Overlap  : PE computes tile(i,j) from Ping bank WHILE
//                          DMA loads tile(next_i, next_j) into Pong bank.
//     Phase 2 – Drain    : Final tile: compute + WB, no new load.
//
// ── FSM States ────────────────────────────────────────────────────────────
//
//   S_IDLE            - Wait for CPU start; latch all config registers.
//   S_WARMUP_LOAD     - Phase 0: wait for tile(0,0) DMA load to finish.
//   S_WARMUP_WAIT     - 1-cycle PPBuf swap propagation.
//   S_OVERLAP_COMPUTE - Compute tile(i,j) while DMA prefetches next tile.
//   S_DRAIN           - Assert pe_flush 1 cycle.
//   S_DRAIN2          - Drain pipeline 2nd cycle.
//   S_WRITE_BACK      - Initiate DMA write-back.
//   S_WB_WAIT         - Wait DMA WB done; swap banks; S_PRELOAD launches next prefetch.
//   S_DONE            - Assert IRQ, reset counters, enter S_IDLE.
//
// ── IRQ Clear ─────────────────────────────────────────────────────────────
//
//   CPU clears IRQ by writing 1 to ctrl_reg[6] (IRQ_CLR bit).
//   npu_ctrl samples and immediately de-asserts irq. Bit is self-clearing.
//
// ── Address Latch ─────────────────────────────────────────────────────────
//
//   All config registers (m/n/k_dim, w/a/r_addr, mode, stat) are latched
//   on cfg_start_rise into shadow regs. FSM uses shadows only.
//
// =============================================================================

`timescale 1ns/1ps

module npu_ctrl #(
    parameter ROWS   = 4,
    parameter COLS   = 4,
    parameter DATA_W = 16,
    parameter ACC_W  = 32,
    parameter PPB_DEPTH = 64,
    parameter [3:0] INT8_SIMD_LANES = 4,
    parameter FP16_ENABLE = 0
)(
    input  wire              clk,
    input  wire              rst_n,
    // Config from AXI-Lite register file (live, may change during run)
    input  wire [31:0]       ctrl_reg,
    input  wire [31:0]       m_dim,
    input  wire [31:0]       n_dim,
    input  wire [31:0]       k_dim,
    input  wire [31:0]       w_addr,
    input  wire [31:0]       a_addr,
    input  wire [31:0]       r_addr,
    input  wire [31:0]       bias_addr,
    input  wire [31:0]       quant_cfg,
    input  wire [7:0]        arr_cfg,
    input  wire [31:0]       desc_base,
    input  wire [31:0]       desc_count,
    input  wire [31:0]       conv_ifm_shape,
    input  wire [31:0]       conv_channels,
    input  wire [31:0]       conv_kernel,
    input  wire [31:0]       conv_out_shape,
    input  wire [31:0]       conv_stride_pad,
    input  wire [31:0]       conv_dilation,
    output reg               desc_start,
    output reg  [31:0]       desc_addr,
    input  wire              desc_done,
    input  wire [511:0]      desc_words,
    // Shape configuration for reconfigurable array
    input  wire [1:0]        cfg_shape_in,
    output wire [1:0]        cfg_shape_latched,
    output wire [1:0]        post_act_mode,
    output wire [31:0]       post_quant_cfg,
    output wire              bias_en,
    // 4x4 tile planner outputs (T2.3)
    output wire              tile_mode,
    output wire              vec_consume,
    output wire [31:0]       tile_m_base,
    output wire [31:0]       tile_n_base,
    output wire [15:0]        tile_row_valid,
    output wire [15:0]        tile_col_valid,
    output wire [4:0]         tile_active_rows,
    output wire [5:0]         tile_active_cols,
    output wire [31:0]       tile_k_base,
    output wire [15:0]       tile_k_len,
    output wire [31:0]       tile_k_index,
    output reg  [15:0]       tile_k_cycle,
    // Status outputs
    output reg               busy,
    output reg               done,
    output wire              error,
    output reg [31:0]        err_status,
    input  wire              err_clear,
    input  wire [31:0]       err_clear_mask,
    // DMA interface
    output reg               dma_w_start,
    input  wire              dma_w_done,
    output reg [31:0]        dma_w_addr,
    output reg [15:0]        dma_w_len,
    output reg               dma_a_start,
    input  wire              dma_a_done,
    output reg [31:0]        dma_a_addr,
    output reg [15:0]        dma_a_len,
    output reg               dma_a_ofm_mode,
    output reg               dma_a_im2col_mode,
    output reg [31:0]        dma_a_ofm_stride,
    output reg [31:0]        dma_a_ofm_m_base,
    output reg [31:0]        dma_a_ofm_k_base,
    output reg [15:0]        dma_a_ofm_k_len,
    output reg [4:0]         dma_a_ofm_active_rows,
    output reg               dma_a_ofm_fp16_mode,
    output reg [31:0]        dma_a_im2col_m_index,
    output reg [15:0]        dma_a_im2col_k_len,
    output reg [15:0]        dma_a_im2col_ih,
    output reg [15:0]        dma_a_im2col_iw,
    output reg [15:0]        dma_a_im2col_cin,
    output reg [15:0]        dma_a_im2col_kh,
    output reg [15:0]        dma_a_im2col_kw,
    output reg [15:0]        dma_a_im2col_oh,
    output reg [15:0]        dma_a_im2col_ow,
    output reg [7:0]         dma_a_im2col_stride_h,
    output reg [7:0]         dma_a_im2col_stride_w,
    output reg [7:0]         dma_a_im2col_pad_h,
    output reg [7:0]         dma_a_im2col_pad_w,
    output reg [7:0]         dma_a_im2col_dilation_h,
    output reg [7:0]         dma_a_im2col_dilation_w,
    output reg               dma_a_im2col_fp16_mode,
    output reg               dma_bias_start,
    input  wire              dma_bias_done,
    output reg [31:0]        dma_bias_addr,
    output reg               dma_r_start,
    input  wire              dma_r_done,
    output reg [31:0]        dma_r_addr,
    output reg [15:0]        dma_r_len,
    input  wire [31:0]       dma_error_status,
    // PE array control
    output reg               pe_en,
    output reg               pe_flush,
    output reg               pe_mode,     // 0=INT8, 1=FP16
    output reg               pe_stat,     // 0=WS,  1=OS
    output reg               pe_load_w,   // WS mode: latch weight into PE
    output reg               pe_swap_w,   // WS mode: swap dual weight regs
    output reg               pe_acc_init_en,
    output wire              pe_half_en,  // 8x32: 0=top half, 1=bottom half
    // Ping-Pong Buffer status
    input  wire              w_ppb_ready,
    input  wire              w_ppb_empty,
    input  wire              a_ppb_ready,
    input  wire              a_ppb_empty,
    // Ping-Pong Buffer control
    output reg               w_ppb_swap,
    output reg               a_ppb_swap,
    output reg               w_ppb_clear,
    output reg               a_ppb_clear,
    // Result FIFO clear
    output reg               r_fifo_clear,
    // Interrupt
    output reg               irq,
    // DFS clock-enable: controller advances compute progress only when high
    input  wire              compute_ce
);

// ---------------------------------------------------------------------------
// FSM state encoding
// ---------------------------------------------------------------------------
localparam S_IDLE            = 4'd0;
localparam S_WARMUP_LOAD     = 4'd1;
localparam S_WARMUP_WAIT     = 4'd2;
localparam S_PRELOAD         = 4'd9;   // 1-cycle swap propagation before compute
localparam S_OVERLAP_COMPUTE = 4'd3;
localparam S_DRAIN           = 4'd4;
localparam S_DRAIN2          = 4'd5;
localparam S_WRITE_BACK      = 4'd6;
localparam S_WB_WAIT         = 4'd7;
localparam S_WAIT_PREFETCH   = 4'd10;  // Wait for prefetch DMA before swap+compute
localparam S_FETCH_DESC      = 4'd11;
localparam S_DECODE_DESC     = 4'd12;
localparam S_DESC_LAUNCH     = 4'd13;
localparam S_DONE            = 4'd8;

reg [3:0] state;
localparam [31:0] PPB_DEPTH_WORDS = PPB_DEPTH;
localparam [31:0] INT8_SIMD_LANES_32 = {28'd0, INT8_SIMD_LANES};

function [4:0] shape_tile_lanes;
    input [1:0] shape;
    begin
        case (shape)
            2'b00: shape_tile_lanes = 5'd4;
            2'b01: shape_tile_lanes = 5'd8;
            2'b10: shape_tile_lanes = 5'd16;
            2'b11: shape_tile_lanes = 5'd8;
        endcase
    end
endfunction

function [4:0] shape_tile_a_lanes;
    input [1:0] shape;
    begin
        case (shape)
            2'b00: shape_tile_a_lanes = 5'd4;
            2'b01: shape_tile_a_lanes = 5'd8;
            default: shape_tile_a_lanes = 5'd16;
        endcase
    end
endfunction

function [5:0] shape_tile_n_lanes;
    input [1:0] shape;
    begin
        case (shape)
            2'b00: shape_tile_n_lanes = 6'd4;
            2'b01: shape_tile_n_lanes = 6'd8;
            2'b10: shape_tile_n_lanes = 6'd16;
            2'b11: shape_tile_n_lanes = 6'd32;
        endcase
    end
endfunction

localparam [31:0] ERR_DESC_COUNT_ZERO      = 32'h0000_0001;
localparam [31:0] ERR_DESC_UNSUPPORTED     = 32'h0000_0002;
localparam [31:0] ERR_DESC_COUNT_EXHAUSTED = 32'h0000_0004;
localparam [31:0] ERR_IFM_PREV_MISSING     = 32'h0000_0008;
localparam [31:0] ERR_DIRECT_INVALID_DIM   = 32'h0000_0100;
localparam [31:0] ERR_DIRECT_INVALID_CONV  = 32'h0000_0200;
localparam [31:0] ERR_FP16_DISABLED        = 32'h0000_0400;

// ---------------------------------------------------------------------------
// ctrl_reg bit decode  (live, from AXI-Lite)
//   Address 0x00 CTRL register bit layout:
//     bit[0]   = start
//     bit[1]   = abort
//     bit[3:2] = data_mode (00=INT8, 10=FP16)
//     bit[5:4] = stat_mode (00=WS,   01=OS)
//     bit[6]   = irq_clr   (CPU writes 1 to acknowledge/clear IRQ; self-clearing)
//     bit[7]   = desc_mode
//     bit[8]   = direct scalar Conv2D on-the-fly im2col
//     bit[9]   = direct scalar bias enable
//     bit[11:10] = direct scalar activation (00=none, 01=ReLU, 10=ReLU6)
// ---------------------------------------------------------------------------
wire        cfg_start   = ctrl_reg[0];
wire        cfg_abort   = ctrl_reg[1];
wire [1:0]  cfg_mode    = ctrl_reg[3:2]; // 00=INT8  10=FP16
wire [1:0]  cfg_stat    = ctrl_reg[5:4]; // 00=WS    01=OS
wire        cfg_irq_clr = ctrl_reg[6];   // CPU writes 1 → clear IRQ
wire        cfg_desc_mode = ctrl_reg[7];
wire        cfg_conv_im2col = ctrl_reg[8];
wire        cfg_bias_en = ctrl_reg[9];
wire [1:0]  cfg_post_act_mode = ctrl_reg[11:10]; // 00=none, 01=ReLU, 10=ReLU6
wire        cfg_non_int8_mode = (cfg_mode != 2'b00);
wire        cfg_fp16_disabled = (FP16_ENABLE == 0) && cfg_non_int8_mode;
wire        cfg_dma_fp16_mode = (FP16_ENABLE != 0) && (cfg_mode == 2'b10);
wire [1:0]  cfg_data_bytes = ((FP16_ENABLE != 0) && cfg_non_int8_mode) ? 2'd2 : 2'd1;
wire [4:0]  cfg_shape_lanes = shape_tile_a_lanes(cfg_shape_in);
wire [15:0] cfg_scalar_elem_bytes = {14'b0, cfg_data_bytes};
wire [15:0] cfg_vector_elem_bytes_max = (cfg_half_bytes_w_8x32 > cfg_vector_elem_bytes_a)
    ? cfg_half_bytes_w_8x32 : cfg_vector_elem_bytes_a;
wire [31:0] cfg_bytes_per_k = {16'd0, cfg_vector_elem_bytes_max};
wire [31:0] cfg_k_tile_elems_raw = arr_cfg[7]
    ? ((PPB_DEPTH_WORDS << 2) / cfg_bytes_per_k)
    : k_dim;
wire [31:0] cfg_k_tile_elems = (cfg_k_tile_elems_raw == 32'd0) ? 32'd1
                                                               : cfg_k_tile_elems_raw;
wire [31:0] cfg_start_k_len_32 = (k_dim <= cfg_k_tile_elems) ? k_dim
                                                             : cfg_k_tile_elems;
// Warm-up uses live config because shadow registers latch on the same start edge.
// In tile mode W and A may have different lane counts (e.g. 8x32).
wire [5:0]  cfg_shape_n_lanes = shape_tile_n_lanes(cfg_shape_in);
wire [15:0] cfg_vector_elem_bytes_w = cfg_scalar_elem_bytes * {11'd0, cfg_shape_n_lanes};
wire [15:0] cfg_vector_elem_bytes_a = cfg_scalar_elem_bytes * {11'd0, cfg_shape_lanes};
wire [15:0] cfg_half_bytes_w_8x32 = (cfg_shape_in == 2'b11) ?
    (cfg_scalar_elem_bytes * 16'd16) : cfg_vector_elem_bytes_w;
// cfg_start uses live arr_cfg (not latched lk_arr_cfg) because config is
// latched in the same cycle the initial DMA is launched.  Using tile_mode
// (which reads lk_arr_cfg) would see stale 0 and drop the packed pad.
wire [3:0]  cfg_packed_k_rem = cfg_start_k_len_32 % {28'd0, INT8_SIMD_LANES};
wire [15:0] cfg_packed_pad   = arr_cfg[7] && (INT8_SIMD_LANES > 1) && (cfg_packed_k_rem != 4'd0)
    ? (cfg_half_bytes_w_8x32 * ({4'd0, INT8_SIMD_LANES} - {1'b0, cfg_packed_k_rem}))
    : 16'd0;
wire [15:0] cfg_packed_pad_a = arr_cfg[7] && (INT8_SIMD_LANES > 1) && (cfg_packed_k_rem != 4'd0)
    ? (cfg_vector_elem_bytes_a * ({4'd0, INT8_SIMD_LANES} - {1'b0, cfg_packed_k_rem}))
    : 16'd0;
wire [15:0] cfg_start_tile_len_raw_w = cfg_start_k_len_32[15:0] *
    (arr_cfg[7] ? cfg_half_bytes_w_8x32 : cfg_scalar_elem_bytes)
    + (arr_cfg[7] ? cfg_packed_pad : 16'd0);
wire [15:0] cfg_start_tile_len_raw_a = cfg_start_k_len_32[15:0] *
    (arr_cfg[7] ? cfg_vector_elem_bytes_a : cfg_scalar_elem_bytes)
    + (arr_cfg[7] ? cfg_packed_pad_a : 16'd0);
// Tile mode already moves whole words per k and is naturally aligned.
wire [15:0] cfg_start_tile_len_w = arr_cfg[7] ? cfg_start_tile_len_raw_w
    : ((cfg_start_tile_len_raw_w + 16'd3) & 16'hfffc);
wire [15:0] cfg_start_tile_len_a = arr_cfg[7] ? cfg_start_tile_len_raw_a
    : ((cfg_start_tile_len_raw_a + 16'd3) & 16'hfffc);

// Rising-edge detect for start
reg cfg_start_d1;
always @(posedge clk) begin
    if (!rst_n) cfg_start_d1 <= 1'b0;
    else        cfg_start_d1 <= cfg_start;
end
wire cfg_start_rise = cfg_start && !cfg_start_d1;
wire direct_start_rise = cfg_start_rise && !cfg_desc_mode;
wire desc_mode_start_rise = cfg_start_rise && cfg_desc_mode;

wire direct_dim_invalid = (m_dim == 32'd0) || (n_dim == 32'd0) || (k_dim == 32'd0);
wire direct_conv_invalid = cfg_conv_im2col &&
    ((conv_ifm_shape[15:0] == 16'd0) || (conv_ifm_shape[31:16] == 16'd0) ||
     (conv_channels[15:0] == 16'd0) ||
     (conv_kernel[15:0] == 16'd0) || (conv_kernel[31:16] == 16'd0) ||
     (conv_out_shape[15:0] == 16'd0) || (conv_out_shape[31:16] == 16'd0) ||
     (conv_stride_pad[7:0] == 8'd0) || (conv_stride_pad[15:8] == 8'd0) ||
     (conv_dilation[7:0] == 8'd0) || (conv_dilation[15:8] == 8'd0));
wire [31:0] direct_start_err_mask =
    (direct_dim_invalid ? ERR_DIRECT_INVALID_DIM : 32'd0) |
    (direct_conv_invalid ? ERR_DIRECT_INVALID_CONV : 32'd0) |
    (cfg_fp16_disabled ? ERR_FP16_DISABLED : 32'd0);
wire direct_start_invalid = (direct_start_err_mask != 32'd0);
wire direct_start_valid_rise = direct_start_rise && !direct_start_invalid;

reg        prev_ofm_valid;
reg [31:0] prev_ofm_addr;

// ---------------------------------------------------------------------------
// Descriptor v1 decode helpers
// ---------------------------------------------------------------------------
localparam integer DESC_W_CTRL        = 0;
localparam integer DESC_W_M           = 1;
localparam integer DESC_W_N           = 2;
localparam integer DESC_W_K           = 3;
localparam integer DESC_W_IFM_ADDR    = 4;
localparam integer DESC_W_WEIGHT_ADDR = 5;
localparam integer DESC_W_OFM_ADDR    = 8;
localparam integer DESC_W_NEXT_DESC   = 15;

wire [31:0] desc_ctrl_word = desc_words[DESC_W_CTRL*32 +: 32];
wire [31:0] desc_m_word    = desc_words[DESC_W_M*32 +: 32];
wire [31:0] desc_n_word    = desc_words[DESC_W_N*32 +: 32];
wire [31:0] desc_k_word    = desc_words[DESC_W_K*32 +: 32];
wire [31:0] desc_ifm_word  = desc_words[DESC_W_IFM_ADDR*32 +: 32];
wire [31:0] desc_wgt_word  = desc_words[DESC_W_WEIGHT_ADDR*32 +: 32];
wire [31:0] desc_ofm_word  = desc_words[DESC_W_OFM_ADDR*32 +: 32];
wire [31:0] desc_next_word = desc_words[DESC_W_NEXT_DESC*32 +: 32];

wire [3:0] desc_op_field      = desc_ctrl_word[3:0];
wire [3:0] desc_dtype_field   = desc_ctrl_word[7:4];
wire [3:0] desc_flow_field    = desc_ctrl_word[11:8];
wire [3:0] desc_shape_field   = desc_ctrl_word[15:12];
wire [3:0] desc_reserved_bits = desc_ctrl_word[27:24];
wire [3:0] desc_version_field = desc_ctrl_word[31:28];
wire [1:0] desc_mode_bits     = desc_dtype_field[1:0];
wire [1:0] desc_flow_bits     = desc_flow_field[1:0];
wire [1:0] desc_shape_bits    = desc_shape_field[1:0];
wire       desc_tile_packed   = desc_ctrl_word[16];
wire       desc_last_layer_bit = desc_ctrl_word[19];
wire       desc_irq_en_bit    = desc_ctrl_word[20];
wire       desc_use_bias      = desc_ctrl_word[21];
wire       desc_use_psum      = desc_ctrl_word[22];
wire       desc_ifm_from_prev_ofm = desc_ctrl_word[23];
wire       desc_ifm_prev_missing = desc_ifm_from_prev_ofm && !prev_ofm_valid;
wire       desc_fp16_disabled = (FP16_ENABLE == 0) && (desc_dtype_field == 4'd2);
wire       desc_supported_base =
    (desc_version_field == 4'd1) &&
    (desc_op_field == 4'd1) &&                    // GEMM_TILEPACK
    ((desc_dtype_field == 4'd0) || (desc_dtype_field == 4'd2)) &&
    (desc_flow_field == 4'd1) &&                  // OS
    (desc_shape_field <= 4'd3) &&                  // 4x4, 8x8, 16x16, 8x32
    desc_tile_packed &&
    !desc_use_bias &&
    !desc_use_psum &&
    (desc_reserved_bits == 4'd0);
wire [31:0] desc_decode_err_mask =
    (desc_supported_base ? 32'd0 : ERR_DESC_UNSUPPORTED) |
    (desc_ifm_prev_missing ? ERR_IFM_PREV_MISSING : 32'd0) |
    (desc_fp16_disabled ? ERR_FP16_DISABLED : 32'd0);
wire       desc_decode_error = (desc_decode_err_mask != 32'd0);

// ---------------------------------------------------------------------------
// Shadow (latched) configuration registers
// Latched once on cfg_start_rise; FSM exclusively uses these.
// ---------------------------------------------------------------------------
reg [31:0] lk_m_dim, lk_n_dim, lk_k_dim;
reg [31:0] lk_w_addr, lk_a_addr, lk_r_addr;
reg [31:0] lk_bias_addr;
reg [1:0]  lk_mode, lk_stat;
reg [1:0]  lk_shape;   // latched cfg_shape
reg [7:0]  lk_arr_cfg;
reg        lk_a_from_prev_ofm;
reg        lk_conv_im2col;
reg        lk_bias_en;
reg [1:0]  lk_post_act_mode;
reg [31:0] lk_quant_cfg;
reg [15:0] lk_conv_ih, lk_conv_iw, lk_conv_cin;
reg [15:0] lk_conv_kh, lk_conv_kw, lk_conv_oh, lk_conv_ow;
reg [7:0]  lk_conv_stride_h, lk_conv_stride_w;
reg [7:0]  lk_conv_pad_h, lk_conv_pad_w;
reg [7:0]  lk_conv_dilation_h, lk_conv_dilation_w;
reg        lk_desc_irq_en;

assign cfg_shape_latched = lk_shape;
assign post_act_mode = lk_post_act_mode;
assign post_quant_cfg = lk_quant_cfg;
assign bias_en = lk_bias_en;
assign tile_mode = lk_arr_cfg[7];

wire lk_non_int8_mode = (lk_mode != 2'b00);
wire lk_fp16_mode = (FP16_ENABLE != 0) && lk_non_int8_mode;
wire lk_dma_fp16_mode = (FP16_ENABLE != 0) && (lk_mode == 2'b10);

always @(posedge clk) begin
    if (!rst_n) begin
        lk_m_dim  <= 32'd1; lk_n_dim  <= 32'd1; lk_k_dim  <= 32'd1;
        lk_w_addr <= 32'd0; lk_a_addr <= 32'd0; lk_r_addr <= 32'd0;
        lk_bias_addr <= 32'd0;
        lk_mode   <= (FP16_ENABLE != 0) ? 2'b10 : 2'b00;
        lk_stat   <= 2'b01;                      // default OS
        lk_shape  <= 2'b10;                      // default 16x16
        lk_arr_cfg <= 8'd0;
        lk_a_from_prev_ofm <= 1'b0;
        lk_conv_im2col <= 1'b0;
        lk_bias_en <= 1'b0;
        lk_post_act_mode <= 2'b00;
        lk_quant_cfg <= 32'h0001_0000;
        lk_conv_ih <= 16'd0; lk_conv_iw <= 16'd0; lk_conv_cin <= 16'd0;
        lk_conv_kh <= 16'd0; lk_conv_kw <= 16'd0; lk_conv_oh <= 16'd0; lk_conv_ow <= 16'd0;
        lk_conv_stride_h <= 8'd1; lk_conv_stride_w <= 8'd1;
        lk_conv_pad_h <= 8'd0; lk_conv_pad_w <= 8'd0;
        lk_conv_dilation_h <= 8'd1; lk_conv_dilation_w <= 8'd1;
        lk_desc_irq_en <= 1'b0;
    end else if (direct_start_valid_rise) begin
        lk_m_dim  <= m_dim;
        lk_n_dim  <= n_dim;
        lk_k_dim  <= k_dim;
        lk_w_addr <= w_addr;
        lk_a_addr <= a_addr;
        lk_r_addr <= r_addr;
        lk_bias_addr <= bias_addr;
        lk_mode   <= cfg_mode;
        lk_stat   <= cfg_stat;
        lk_shape  <= cfg_shape_in;
        lk_arr_cfg <= arr_cfg;
        lk_a_from_prev_ofm <= 1'b0;
        lk_conv_im2col <= cfg_conv_im2col;
        lk_bias_en <= cfg_bias_en;
        lk_post_act_mode <= cfg_post_act_mode;
        lk_quant_cfg <= (cfg_mode == 2'b00) ? quant_cfg
                                             : 32'h0001_0000;
        lk_conv_ih <= conv_ifm_shape[15:0];
        lk_conv_iw <= conv_ifm_shape[31:16];
        lk_conv_cin <= conv_channels[15:0];
        lk_conv_kh <= conv_kernel[15:0];
        lk_conv_kw <= conv_kernel[31:16];
        lk_conv_oh <= conv_out_shape[15:0];
        lk_conv_ow <= conv_out_shape[31:16];
        lk_conv_stride_h <= conv_stride_pad[7:0];
        lk_conv_stride_w <= conv_stride_pad[15:8];
        lk_conv_pad_h <= conv_stride_pad[23:16];
        lk_conv_pad_w <= conv_stride_pad[31:24];
        lk_conv_dilation_h <= conv_dilation[7:0];
        lk_conv_dilation_w <= conv_dilation[15:8];
        lk_desc_irq_en <= 1'b1;
    end else if (state == S_DECODE_DESC && !desc_decode_error) begin
        lk_m_dim  <= desc_m_word;
        lk_n_dim  <= desc_n_word;
        lk_k_dim  <= desc_k_word;
        lk_w_addr <= desc_wgt_word;
        lk_a_addr <= (desc_ifm_from_prev_ofm && prev_ofm_valid) ? prev_ofm_addr
                                                                : desc_ifm_word;
        lk_r_addr <= desc_ofm_word;
        lk_bias_addr <= 32'd0;
        lk_mode   <= desc_mode_bits;
        lk_stat   <= desc_flow_bits;
        lk_shape  <= desc_shape_bits;
        lk_arr_cfg <= {desc_tile_packed, 7'd0};
        lk_a_from_prev_ofm <= desc_ifm_from_prev_ofm && prev_ofm_valid;
        lk_conv_im2col <= 1'b0;
        lk_bias_en <= 1'b0;
        lk_post_act_mode <= 2'b00;
        lk_quant_cfg <= 32'h0001_0000;
        lk_conv_ih <= 16'd0; lk_conv_iw <= 16'd0; lk_conv_cin <= 16'd0;
        lk_conv_kh <= 16'd0; lk_conv_kw <= 16'd0; lk_conv_oh <= 16'd0; lk_conv_ow <= 16'd0;
        lk_conv_stride_h <= 8'd1; lk_conv_stride_w <= 8'd1;
        lk_conv_pad_h <= 8'd0; lk_conv_pad_w <= 8'd0;
        lk_conv_dilation_h <= 8'd1; lk_conv_dilation_w <= 8'd1;
        lk_desc_irq_en <= desc_irq_en_bit;
    end
end

// ---------------------------------------------------------------------------
// Mode decode (combinational, uses shadow regs)
// ---------------------------------------------------------------------------
always @(*) begin
    if (FP16_ENABLE == 0) begin
        pe_mode = 1'b0;
    end else begin
        case (lk_mode)
            2'b00:   pe_mode = 1'b0;   // INT8
            2'b10:   pe_mode = 1'b1;   // FP16
            default: pe_mode = 1'b1;
        endcase
    end
end

// Bytes per element and active tile lane count (from shadow)
wire [1:0] data_bytes = lk_fp16_mode ? 2'd2 : 2'd1;
wire [4:0] tile_shape_lanes = shape_tile_lanes(lk_shape);
wire [4:0] tile_shape_a_lanes = shape_tile_a_lanes(lk_shape);
wire [5:0] tile_shape_n_lanes = shape_tile_n_lanes(lk_shape);
wire [31:0] tile_shape_lanes_32 = {27'd0, tile_shape_lanes};
wire [31:0] tile_shape_n_lanes_32 = {26'd0, tile_shape_n_lanes};
wire [15:0] scalar_elem_bytes = {14'b0, data_bytes};
wire [15:0] vector_elem_bytes_a = scalar_elem_bytes * {11'd0, tile_shape_a_lanes};
wire [15:0] vector_elem_bytes_w = scalar_elem_bytes * {11'd0, tile_shape_n_lanes};
wire [15:0] bytes_per_k_w = tile_mode ? vector_elem_bytes_w : scalar_elem_bytes;
wire [15:0] bytes_per_k_a = tile_mode ? vector_elem_bytes_a : scalar_elem_bytes;
wire [15:0] half_vector_elem_bytes_w = scalar_elem_bytes * 16'd16;  // 16 cols for 8x32 pass
wire [15:0] bytes_per_k_w_8x32 = is_8x32 ? half_vector_elem_bytes_w : bytes_per_k_w;
wire [31:0] bytes_per_k_w_32 = {16'd0, bytes_per_k_w_8x32};
wire [31:0] bytes_per_k_a_32 = {16'd0, bytes_per_k_a};
// Packed tile streams pad each N/M tile's K dimension to a full SIMD group.
// Internal tile-to-tile address strides must include that pad; k_tile offsets
// still use tile_k_base because full k_tile_elems is SIMD-aligned.
wire [31:0] k_dim_padded = (INT8_SIMD_LANES_32 <= 32'd1)
    ? lk_k_dim
    : (((lk_k_dim + INT8_SIMD_LANES_32 - 32'd1) / INT8_SIMD_LANES_32) * INT8_SIMD_LANES_32);
wire [31:0] w_addr_pass1_offset = is_8x32 ? (k_dim_padded * {16'd0, half_vector_elem_bytes_w}) : 32'd0;
wire [31:0] w_tile_stride_n = is_8x32
    ? (k_dim_padded * {16'd0, half_vector_elem_bytes_w} * 32'd2)
    : (k_dim_padded * bytes_per_k_w_32);
wire [31:0] a_tile_stride_m = k_dim_padded * bytes_per_k_a_32;

// ---------------------------------------------------------------------------
// Tile-loop counters:
//   scalar mode: tile_i/tile_j are i/j for one C[i,j]
//   tile mode:   tile_i/tile_j are m_tile/n_tile for one 4x4 C tile
// ---------------------------------------------------------------------------
reg [31:0] tile_i;
reg [31:0] tile_j;
reg [31:0] k_tile_idx;
reg [4:0]  wb_row;  // tile-mode writeback row r within the current C tile
reg [5:0]  bias_col; // tile-mode bias fetch column counter
reg         bias_pending; // set when bias_start issued, cleared when done received
reg         pass_idx; // 0 or 1 for 8x32 two-pass weight scheduling
wire        is_8x32 = tile_mode && (lk_shape == 2'b11);
assign pe_half_en = is_8x32 && pass_idx;
// Scalar mode iterates individual C[i,j]; tile mode iterates one shape-sized C tile.
wire [31:0] tile_m_tiles = tile_mode ? ((lk_m_dim + tile_shape_lanes_32 - 32'd1) / tile_shape_lanes_32) : lk_m_dim;
wire [31:0] tile_n_tiles = tile_mode ? ((lk_n_dim + tile_shape_n_lanes_32 - 32'd1) / tile_shape_n_lanes_32) : lk_n_dim;
wire [31:0] tile_iter_m_count = (tile_m_tiles == 32'd0) ? 32'd1 : tile_m_tiles;
wire [31:0] tile_iter_n_count = (tile_n_tiles == 32'd0) ? 32'd1 : tile_n_tiles;
wire [31:0] bytes_per_k_max = (bytes_per_k_w_32 > bytes_per_k_a_32)
    ? bytes_per_k_w_32 : bytes_per_k_a_32;
wire [31:0] k_tile_elems_raw = tile_mode
    ? ((PPB_DEPTH_WORDS << 2) / bytes_per_k_max)
    : lk_k_dim;
wire [31:0] k_tile_elems = (k_tile_elems_raw == 32'd0) ? 32'd1 : k_tile_elems_raw;
wire [31:0] k_tile_count_raw = (lk_k_dim + k_tile_elems - 32'd1) / k_tile_elems;
wire [31:0] k_tile_count = (k_tile_count_raw == 32'd0) ? 32'd1 : k_tile_count_raw;

assign tile_m_base = tile_mode ? (tile_i * tile_shape_lanes_32) : tile_i; // global M row of tile row 0
assign tile_n_base = tile_mode ? (tile_j * tile_shape_n_lanes_32) : tile_j; // global N col of tile col 0
assign tile_k_index = k_tile_idx;
assign tile_k_base = tile_mode ? (k_tile_idx * k_tile_elems) : 32'd0;
wire [31:0] tile_k_rem = (lk_k_dim > tile_k_base) ? (lk_k_dim - tile_k_base) : 32'd0;
wire [31:0] tile_k_len_32 = tile_mode ?
                            ((tile_k_rem > k_tile_elems) ? k_tile_elems : tile_k_rem) :
                            lk_k_dim;
assign tile_k_len = (tile_k_len_32 == 32'd0) ? 16'd1 : tile_k_len_32[15:0];

wire [31:0] tile_row_rem = (lk_m_dim > tile_m_base) ? (lk_m_dim - tile_m_base) : 32'd0;
wire [31:0] tile_col_rem = (lk_n_dim > tile_n_base) ? (lk_n_dim - tile_n_base) : 32'd0;
wire [31:0] tile_active_rows_32 = !tile_mode ? 32'd1 :
                                  (tile_row_rem >= tile_shape_lanes_32) ? tile_shape_lanes_32 :
                                  tile_row_rem;
wire [31:0] tile_active_cols_32 = !tile_mode ? 32'd1 :
                                  (tile_col_rem >= tile_shape_n_lanes_32) ? tile_shape_n_lanes_32 :
                                  tile_col_rem;

assign tile_active_rows = !tile_mode ? 5'd1 : tile_active_rows_32[4:0];
assign tile_active_cols = !tile_mode ? 6'd1 : tile_active_cols_32[5:0];

// Edge tiles may have fewer valid rows/cols when M or N is not a multiple
// of the shape lanes. These masks prevent inactive lanes from being written back.
genvar vg;
generate
    for (vg = 0; vg < 16; vg = vg + 1) begin : gen_tile_valid
        assign tile_row_valid[vg] = ({4'd0, tile_active_rows} > vg);
        assign tile_col_valid[vg] = ({4'd0, tile_active_cols} > vg);
    end
endgenerate

// One-tile DMA byte length — W and A may differ for non-square shapes (8x32)
// Packed INT8 SIMD: pad so the last lane block has a full SIMD group.
// General: pad = vector_bytes * (SIMD_LANES - k_len mod SIMD_LANES) when
// remainder non-zero.  For K=4/LANES=2 this gives 0; K=3/LANES=4 gives 1.
wire [3:0] packed_k_rem = tile_mode ? (tile_k_len % {12'd0, INT8_SIMD_LANES}) : 4'd0;
wire [15:0] packed_pad = tile_mode && (INT8_SIMD_LANES > 1) && (packed_k_rem != 4'd0)
    ? (bytes_per_k_w_8x32 * ({4'd0, INT8_SIMD_LANES} - {1'b0, packed_k_rem}))
    : 16'd0;
wire [15:0] packed_pad_a = tile_mode && (INT8_SIMD_LANES > 1) && (packed_k_rem != 4'd0)
    ? (vector_elem_bytes_a * ({4'd0, INT8_SIMD_LANES} - {1'b0, packed_k_rem}))
    : 16'd0;
wire [15:0] tile_len_raw_w = tile_k_len * (tile_mode ? bytes_per_k_w_8x32 : scalar_elem_bytes)
                               + (tile_mode ? packed_pad : 16'd0);
wire [15:0] tile_len_raw_a = tile_k_len * (tile_mode ? vector_elem_bytes_a : scalar_elem_bytes)
                              + (tile_mode ? packed_pad_a : 16'd0);
wire [15:0] tile_len_w = tile_mode ? tile_len_raw_w
                                   : ((tile_len_raw_w + 16'd3) & 16'hfffc);
wire [15:0] tile_len_a = tile_mode ? tile_len_raw_a
                                   : ((tile_len_raw_a + 16'd3) & 16'hfffc);

// Current-tile addresses (used for write-back address calc)
wire [31:0] comp_r_addr = lk_r_addr +
                          (tile_m_base * lk_n_dim + tile_n_base) * (ACC_W/8);
// Row-wise tile writeback address:
//   C row = tile_m_base + wb_row, C col starts at tile_n_base.
wire [31:0] comp_row_r_addr = lk_r_addr +
                              (((tile_m_base + {29'd0, wb_row}) * lk_n_dim) +
                               tile_n_base) * (ACC_W/8);
// One row burst writes active_cols 32-bit accumulation words.
wire [15:0] tile_row_r_len = {13'd0, tile_active_cols} << 2;

// ── Last-tile flag ──
wire is_last_tile = (tile_i == tile_iter_m_count - 1) &&
                    (tile_j == tile_iter_n_count - 1);

// Result DMA: scalar mode writes one word; tile mode writes one row burst.
localparam [15:0] TILE_R_LEN = 16'd4;

wire [15:0] tile_k_cycles = tile_mode
    ? ((tile_k_len + {12'd0, INT8_SIMD_LANES} - 16'd1) / {12'd0, INT8_SIMD_LANES})
    : tile_k_len;
wire tile_direct_ws = tile_mode && !lk_fp16_mode && !lk_stat[0];
// Direct WS has no array row-skew drain, but the PE's input/multiply stages
// still need two enabled cycles before a pass transition or flush.
wire [15:0] tile_compute_drain_cycles = tile_direct_ws ? 16'd2
                                                        : (tile_active_rows_32[15:0] - 16'd1);
wire [15:0] tile_compute_cycles = tile_k_cycles + tile_compute_drain_cycles;

// Packed INT8 tile WS uses a direct feeder: W and A vectors are sent to the PE
// grid in lockstep and each PE accumulates internally.  Do not promote WS to OS.
wire tile_force_os = 1'b0;
wire lk_stat0_eff = lk_stat[0];

// vec_consume advances one packed A vector and one packed W vector for each
// logical k. Extra cycles after K drain row-skewed activations through the array.
assign vec_consume = tile_mode &&
                      pe_en &&
                      (state == S_OVERLAP_COMPUTE) &&
                      ((lk_stat0_eff == 1'b1) || tile_direct_ws) &&
                       (tile_k_cycle < tile_k_cycles);

// ---------------------------------------------------------------------------
// DMA completion latches
// ---------------------------------------------------------------------------
reg dma_w_done_r, dma_a_done_r, dma_bias_done_r, dma_r_done_r;
wire dma_load_done = dma_w_done_r && dma_a_done_r;

// ---------------------------------------------------------------------------
// WS consume counter
// ---------------------------------------------------------------------------
reg [15:0] ws_consume_cnt;

// ---------------------------------------------------------------------------
// Descriptor sequencing state
// desc_left is the remaining fetch budget after the current descriptor fetch.
// ---------------------------------------------------------------------------
reg        desc_mode_run;
reg [31:0] desc_left;
reg [31:0] desc_next_addr;
reg        desc_last_layer;
wire       desc_normal_stop = desc_last_layer || (desc_next_addr == 32'd0);
wire       desc_can_fetch_next = desc_mode_run && !desc_normal_stop && (desc_left != 32'd0);
wire       desc_count_exhausted = desc_mode_run && !desc_normal_stop && (desc_left == 32'd0);

reg [31:0] err_set_mask;
wire [31:0] err_status_after_clear = err_clear ? (err_status & ~err_clear_mask) : err_status;

assign error = (err_status != 32'd0);

always @(posedge clk) begin
    if (!rst_n) begin
        err_status <= 32'd0;
    end else if (err_set_mask != 32'd0) begin
        err_status <= err_status_after_clear | err_set_mask;
    end else begin
        err_status <= err_status_after_clear;
    end
end

// ---------------------------------------------------------------------------
// Combinational: next sequence coordinates.
// T4.4 makes K the innermost loop:
//   for m_tile:
//     for n_tile:
//       for k_tile:
//         load/compute current K slice
// ---------------------------------------------------------------------------
wire k_tile_last = !tile_mode || (k_tile_idx + 32'd1 >= k_tile_count);
wire cur_has_next_k = tile_mode && !k_tile_last;
wire [31:0] next_mn_j = (tile_j + 1 < tile_iter_n_count) ? (tile_j + 1) : 32'd0;
wire [31:0] next_mn_i = (tile_j + 1 < tile_iter_n_count) ? tile_i       : (tile_i + 1);

wire [31:0] seq1_k = cur_has_next_k ? (k_tile_idx + 32'd1) : 32'd0;
wire [31:0] seq1_j = cur_has_next_k ? tile_j               : next_mn_j;
wire [31:0] seq1_i = cur_has_next_k ? tile_i               : next_mn_i;

wire is_last_seq = is_last_tile && k_tile_last;
wire can_prefetch_next = !is_last_seq && (!is_8x32 || pass_idx);

wire [31:0] seq1_k_base = tile_mode ? (seq1_k * k_tile_elems) : 32'd0;
wire [31:0] seq1_k_rem = (lk_k_dim > seq1_k_base) ? (lk_k_dim - seq1_k_base) : 32'd0;
wire [31:0] seq1_k_len_32 = tile_mode ?
                            ((seq1_k_rem > k_tile_elems) ? k_tile_elems : seq1_k_rem) :
                            lk_k_dim;
wire [3:0] seq1_packed_k_rem = tile_mode ? (seq1_k_len_32 % {28'd0, INT8_SIMD_LANES}) : 4'd0;
wire [15:0] seq1_packed_pad = tile_mode && (INT8_SIMD_LANES > 1) && (seq1_packed_k_rem != 4'd0)
    ? (bytes_per_k_w_8x32 * ({4'd0, INT8_SIMD_LANES} - {1'b0, seq1_packed_k_rem}))
    : 16'd0;
wire [15:0] seq1_packed_pad_a = tile_mode && (INT8_SIMD_LANES > 1) && (seq1_packed_k_rem != 4'd0)
    ? (vector_elem_bytes_a * ({4'd0, INT8_SIMD_LANES} - {1'b0, seq1_packed_k_rem}))
    : 16'd0;
wire [15:0] seq1_len_bytes_raw_w = tile_mode
    ? (seq1_k_len_32[15:0] * bytes_per_k_w_8x32 + seq1_packed_pad)
    : (seq1_k_len_32[15:0] * bytes_per_k_w_8x32);
wire [15:0] seq1_len_bytes_raw_a = tile_mode
    ? (seq1_k_len_32[15:0] * bytes_per_k_a + seq1_packed_pad_a)
    : (seq1_k_len_32[15:0] * bytes_per_k_a);
wire [15:0] seq1_len_bytes_w = tile_mode ? seq1_len_bytes_raw_w
                                       : ((seq1_len_bytes_raw_w + 16'd3) & 16'hfffc);
wire [15:0] seq1_len_bytes_a = tile_mode ? seq1_len_bytes_raw_a
                                       : ((seq1_len_bytes_raw_a + 16'd3) & 16'hfffc);
wire [31:0] seq1_m_base = tile_mode ? (seq1_i * tile_shape_lanes_32) : seq1_i;
wire [31:0] seq1_row_rem = (lk_m_dim > seq1_m_base) ? (lk_m_dim - seq1_m_base) : 32'd0;
wire [31:0] seq1_active_rows_32 = !tile_mode ? 32'd1 :
                                  (seq1_row_rem >= tile_shape_lanes_32) ? tile_shape_lanes_32 :
                                  seq1_row_rem;
wire [4:0] seq1_active_rows = !tile_mode ? 5'd1 : seq1_active_rows_32[4:0];

wire [31:0] pfetch_w_addr = tile_mode ?
    (lk_w_addr + (seq1_j * w_tile_stride_n) + (seq1_k_base * bytes_per_k_w_32)) :
    (lk_w_addr + seq1_j * ({16'b0, tile_len_w}));
wire [31:0] pfetch_a_addr = tile_mode ?
    (lk_a_addr + (seq1_i * a_tile_stride_m) + (seq1_k_base * bytes_per_k_a_32)) :
    (lk_a_addr + seq1_i * ({16'b0, tile_len_a}));

// ---------------------------------------------------------------------------
// Main FSM
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        state          <= S_IDLE;
        busy           <= 1'b0;
        done           <= 1'b0;
        irq            <= 1'b0;
        pe_en          <= 1'b0;
        pe_flush       <= 1'b0;
        pe_stat        <= 1'b1;
        pe_load_w      <= 1'b0;
        pe_swap_w      <= 1'b0;
        pe_acc_init_en <= 1'b0;
        dma_w_start    <= 1'b0;
        dma_a_start    <= 1'b0;
        dma_bias_start <= 1'b0;
        dma_r_start    <= 1'b0;
        desc_start     <= 1'b0;
        desc_addr      <= 32'd0;
        dma_w_addr     <= 32'd0;
        dma_a_addr     <= 32'd0;
        dma_r_addr     <= 32'd0;
        dma_w_len      <= 16'd0;
        dma_a_len      <= 16'd0;
        dma_r_len      <= 16'd0;
        dma_a_ofm_mode <= 1'b0;
        dma_a_im2col_mode <= 1'b0;
        dma_a_ofm_stride <= 32'd0;
        dma_a_ofm_m_base <= 32'd0;
        dma_a_ofm_k_base <= 32'd0;
        dma_a_ofm_k_len <= 16'd0;
        dma_a_ofm_active_rows <= 5'd0;
        dma_a_ofm_fp16_mode <= 1'b0;
        dma_a_im2col_m_index <= 32'd0;
        dma_a_im2col_k_len <= 16'd0;
        dma_a_im2col_ih <= 16'd0;
        dma_a_im2col_iw <= 16'd0;
        dma_a_im2col_cin <= 16'd0;
        dma_a_im2col_kh <= 16'd0;
        dma_a_im2col_kw <= 16'd0;
        dma_a_im2col_oh <= 16'd0;
        dma_a_im2col_ow <= 16'd0;
        dma_a_im2col_stride_h <= 8'd1;
        dma_a_im2col_stride_w <= 8'd1;
        dma_a_im2col_pad_h <= 8'd0;
        dma_a_im2col_pad_w <= 8'd0;
        dma_a_im2col_dilation_h <= 8'd1;
        dma_a_im2col_dilation_w <= 8'd1;
        dma_a_im2col_fp16_mode <= 1'b0;
        dma_bias_addr  <= 32'd0;
        dma_w_done_r   <= 1'b0;
        dma_a_done_r   <= 1'b0;
        dma_bias_done_r <= 1'b0;
        dma_r_done_r   <= 1'b0;
        w_ppb_swap     <= 1'b0;
        a_ppb_swap     <= 1'b0;
        w_ppb_clear    <= 1'b0;
        a_ppb_clear    <= 1'b0;
        r_fifo_clear   <= 1'b0;
        ws_consume_cnt <= 16'd0;
        tile_k_cycle   <= 16'd0;
        tile_i         <= 32'd0;
        pass_idx   <= 1'b0;
        tile_j         <= 32'd0;
        k_tile_idx     <= 32'd0;
        pass_idx   <= 1'b0;
        wb_row         <= 3'd0;
        pass_idx   <= 1'b0;
        desc_mode_run  <= 1'b0;
        desc_left      <= 32'd0;
        desc_next_addr <= 32'd0;
        desc_last_layer <= 1'b0;
        prev_ofm_valid <= 1'b0;
        prev_ofm_addr  <= 32'd0;
        err_set_mask   <= 32'd0;
    end else begin
        // ── Default: de-assert all one-cycle pulse signals ──
        dma_w_start  <= 1'b0;
        dma_a_start  <= 1'b0;
        dma_bias_start <= 1'b0;
        dma_r_start  <= 1'b0;
        desc_start   <= 1'b0;
        w_ppb_swap   <= 1'b0;
        a_ppb_swap   <= 1'b0;
        w_ppb_clear  <= 1'b0;
        a_ppb_clear  <= 1'b0;
        r_fifo_clear <= 1'b0;
        pe_swap_w    <= 1'b0;
        pe_acc_init_en <= 1'b0;
        err_set_mask <= 32'd0;

        // ── Latch DMA done pulses into registered flags ──
        if (dma_w_done)   dma_w_done_r   <= 1'b1;
        if (dma_a_done)   dma_a_done_r   <= 1'b1;
        if (dma_bias_done) dma_bias_done_r <= 1'b1;
        if (dma_r_done)   dma_r_done_r   <= 1'b1;

        // ── IRQ Clear: CPU writes ctrl_reg[6] = 1 to acknowledge IRQ ──
        if (cfg_irq_clr) irq <= 1'b0;

        // ── done auto-clear when CPU de-asserts start ──
        if (!cfg_start) done <= 1'b0;

        if (dma_error_status != 32'd0) begin
            busy           <= 1'b0;
            done           <= 1'b0;
            irq            <= 1'b0;
            pe_en          <= 1'b0;
            pe_flush       <= 1'b0;
            pe_load_w      <= 1'b0;
            dma_w_start    <= 1'b0;
            dma_a_start    <= 1'b0;
            dma_bias_start <= 1'b0;
            dma_r_start    <= 1'b0;
            desc_start     <= 1'b0;
            desc_mode_run  <= 1'b0;
            w_ppb_clear    <= 1'b1;
            a_ppb_clear    <= 1'b1;
            r_fifo_clear   <= 1'b1;
            err_set_mask   <= dma_error_status;
            state          <= S_IDLE;
        end else begin
        case (state)

            // =================================================================
            // S_IDLE
            // =================================================================
            S_IDLE: begin
                pe_en    <= 1'b0;
                pe_flush <= 1'b0;
                pe_load_w <= 1'b0;
                dma_w_done_r <= 1'b0;
                dma_a_done_r <= 1'b0;
                dma_bias_done_r <= 1'b0;
                dma_r_done_r <= 1'b0;

                if (desc_mode_start_rise) begin
                    done            <= 1'b0;
                    irq             <= 1'b0;
                    desc_mode_run   <= 1'b0;
                    desc_left       <= 32'd0;
                    desc_next_addr  <= 32'd0;
                    desc_last_layer <= 1'b0;
                    prev_ofm_valid  <= 1'b0;
                    prev_ofm_addr   <= 32'd0;
                    tile_i          <= 32'd0;
        pass_idx   <= 1'b0;
                    tile_j          <= 32'd0;
                    k_tile_idx      <= 32'd0;
        pass_idx   <= 1'b0;
                    wb_row          <= 3'd0;
        pass_idx   <= 1'b0;
                    ws_consume_cnt  <= 16'd0;
                    tile_k_cycle    <= 16'd0;
                    w_ppb_clear     <= 1'b1;
                    a_ppb_clear     <= 1'b1;
                    r_fifo_clear    <= 1'b1;
                    if (desc_count == 32'd0) begin
                        busy         <= 1'b0;
                        err_set_mask <= ERR_DESC_COUNT_ZERO;
                        state        <= S_IDLE;
                    end else begin
                        busy          <= 1'b1;
                        desc_mode_run <= 1'b1;
                        desc_left     <= desc_count - 32'd1;
                        desc_addr     <= desc_base;
                        desc_start    <= 1'b1;
                        state         <= S_FETCH_DESC;
                    end
                end else if (direct_start_rise && direct_start_invalid) begin
                    busy           <= 1'b0;
                    done           <= 1'b0;
                    irq            <= 1'b0;
                    desc_mode_run  <= 1'b0;
                    w_ppb_clear    <= 1'b1;
                    a_ppb_clear    <= 1'b1;
                    r_fifo_clear   <= 1'b1;
                    err_set_mask   <= direct_start_err_mask;
                    state          <= S_IDLE;
                end else if (direct_start_valid_rise) begin
                    busy           <= 1'b1;
                    desc_mode_run  <= 1'b0;
                    prev_ofm_valid <= 1'b0;
                    prev_ofm_addr  <= 32'd0;
                    tile_i         <= 32'd0;
        pass_idx   <= 1'b0;
                    tile_j         <= 32'd0;
                    ws_consume_cnt <= 16'd0;
                    tile_k_cycle   <= 16'd0;
                    k_tile_idx     <= 32'd0;
        pass_idx   <= 1'b0;
        wb_row         <= 3'd0;
        pass_idx   <= 1'b0;
        bias_col       <= 6'd0;
        bias_pending   <= 1'b0;

                    // Clear all buffers for fresh layer start
                    w_ppb_clear  <= 1'b1;
                    a_ppb_clear  <= 1'b1;
                    r_fifo_clear <= 1'b1;

                    // Phase-0: launch warm-up load of tile(0,0)
                    // Use live cfg (shadow latched this same cycle)
                    dma_w_addr  <= w_addr;
                    dma_w_len   <= cfg_start_tile_len_w;
                    dma_a_addr  <= a_addr;
                    dma_a_len   <= cfg_start_tile_len_a;
                    dma_a_ofm_mode <= 1'b0;
                    dma_a_im2col_mode <= cfg_conv_im2col;
                    dma_a_ofm_stride <= 32'd0;
                    dma_a_ofm_m_base <= 32'd0;
                    dma_a_ofm_k_base <= 32'd0;
                    dma_a_ofm_k_len <= 16'd0;
                    dma_a_ofm_active_rows <= 5'd0;
                    dma_a_ofm_fp16_mode <= 1'b0;
                    dma_a_im2col_m_index <= 32'd0;
                    dma_a_im2col_k_len <= k_dim[15:0];
                    dma_a_im2col_ih <= conv_ifm_shape[15:0];
                    dma_a_im2col_iw <= conv_ifm_shape[31:16];
                    dma_a_im2col_cin <= conv_channels[15:0];
                    dma_a_im2col_kh <= conv_kernel[15:0];
                    dma_a_im2col_kw <= conv_kernel[31:16];
                    dma_a_im2col_oh <= conv_out_shape[15:0];
                    dma_a_im2col_ow <= conv_out_shape[31:16];
                    dma_a_im2col_stride_h <= conv_stride_pad[7:0];
                    dma_a_im2col_stride_w <= conv_stride_pad[15:8];
                    dma_a_im2col_pad_h <= conv_stride_pad[23:16];
                    dma_a_im2col_pad_w <= conv_stride_pad[31:24];
                    dma_a_im2col_dilation_h <= conv_dilation[7:0];
                    dma_a_im2col_dilation_w <= conv_dilation[15:8];
                    dma_a_im2col_fp16_mode <= cfg_dma_fp16_mode;
                    dma_bias_addr <= bias_addr;
                    dma_w_start <= 1'b1;
                    dma_a_start <= 1'b1;
                    dma_bias_start <= cfg_bias_en && !arr_cfg[7];

                    state <= S_WARMUP_LOAD;
                end
            end

            // =================================================================
            // S_WARMUP_LOAD – wait for tile(0,0) DMA load done.
            // =================================================================
            S_FETCH_DESC: begin
                pe_en    <= 1'b0;
                pe_flush <= 1'b0;
                if (cfg_abort) begin
                    state <= S_IDLE;
                    busy  <= 1'b0;
                    desc_mode_run <= 1'b0;
                end else if (desc_done) begin
                    state <= S_DECODE_DESC;
                end
            end

            S_DECODE_DESC: begin
                pe_en          <= 1'b0;
                pe_flush       <= 1'b0;
                tile_i         <= 32'd0;
        pass_idx   <= 1'b0;
                tile_j         <= 32'd0;
                k_tile_idx     <= 32'd0;
        pass_idx   <= 1'b0;
                wb_row         <= 3'd0;
        pass_idx   <= 1'b0;
                ws_consume_cnt <= 16'd0;
                tile_k_cycle   <= 16'd0;
                if (desc_decode_error) begin
                    busy          <= 1'b0;
                    done          <= 1'b0;
                    irq           <= 1'b0;
                    desc_mode_run <= 1'b0;
                    err_set_mask  <= desc_decode_err_mask;
                    state         <= S_IDLE;
                end else begin
                    desc_next_addr  <= desc_next_word;
                    desc_last_layer <= desc_last_layer_bit;
                    state           <= S_DESC_LAUNCH;
                end
            end

            S_DESC_LAUNCH: begin
                pe_en    <= 1'b0;
                pe_flush <= 1'b0;
                w_ppb_clear  <= 1'b1;
                a_ppb_clear  <= 1'b1;
                r_fifo_clear <= 1'b1;
                dma_w_addr   <= lk_w_addr;
                dma_w_len    <= tile_len_w;
                dma_a_addr   <= lk_a_addr;
                dma_a_len    <= tile_len_a;
                dma_a_ofm_mode <= lk_a_from_prev_ofm && tile_mode;
                dma_a_im2col_mode <= lk_conv_im2col;
                dma_a_ofm_stride <= lk_k_dim;
                dma_a_ofm_m_base <= tile_m_base;
                dma_a_ofm_k_base <= tile_k_base;
                dma_a_ofm_k_len <= tile_k_len;
                dma_a_ofm_active_rows <= tile_active_rows;
                dma_a_ofm_fp16_mode <= lk_dma_fp16_mode;
                dma_a_im2col_m_index <= 32'd0;
                dma_a_im2col_k_len <= 16'd0;
                dma_w_start  <= 1'b1;
                dma_a_start  <= 1'b1;
                dma_bias_start <= 1'b0;
                dma_w_done_r <= 1'b0;
                dma_a_done_r <= 1'b0;
                dma_bias_done_r <= 1'b0;
                dma_r_done_r <= 1'b0;
                state        <= S_WARMUP_LOAD;
            end

            S_WARMUP_LOAD: begin
                `ifdef DIAG_8X32
                if (pass_idx) $display("[DIAG_CTRL] S_WARMUP_LOAD pass 1: dma_load_done=%0d", dma_load_done);
                `endif
                pe_en    <= 1'b0;
                pe_flush <= 1'b0;

                if (cfg_abort) begin
                    state <= S_IDLE; busy <= 1'b0;
                end else if (dma_load_done) begin
                    // Swap PPBuf Pong→Ping so PE can read tile(0,0)
                    w_ppb_swap   <= 1'b1;
                    a_ppb_swap   <= 1'b1;
                    dma_w_done_r <= 1'b0;
                    dma_a_done_r <= 1'b0;
                    dma_bias_done_r <= 1'b0;
                    state        <= S_WARMUP_WAIT;
                end
            end

            // =================================================================
            // S_WARMUP_WAIT - 1-cycle swap propagation.
            // Launch of the next prefetch is delayed to S_PRELOAD after swap.
            // =================================================================
            S_WARMUP_WAIT: begin
                pe_stat        <= lk_stat[0];
                pe_load_w      <= ((lk_stat[0] == 1'b0) && (!tile_mode || tile_direct_ws)) ? 1'b1 : 1'b0;
                ws_consume_cnt <= 16'd0;
                tile_k_cycle   <= 16'd0;

                state <= S_PRELOAD;
            end

            // =================================================================
            // S_PRELOAD - 1-cycle wait for PPBuf swap to propagate.
            // After this cycle, rd_sel/wr_sel/rd_fill are stable for PE.
            // Also handles tile-mode bias sequential fetch.
            // =================================================================
            S_PRELOAD: begin
                pe_en    <= 1'b0;
                pe_flush <= 1'b0;
                pe_acc_init_en <= lk_bias_en && !tile_mode;
                // pe_stat and pe_load_w are already set by previous state
                // Launch the next prefetch one cycle after PPBuf swap so DMA
                // samples the freshly reset writer bank rather than stale full.
                if (can_prefetch_next) begin
                    dma_w_addr  <= pfetch_w_addr;
                    dma_w_len   <= seq1_len_bytes_w;
                    dma_a_addr  <= (lk_a_from_prev_ofm && tile_mode) ? lk_a_addr :
                                   (lk_conv_im2col) ? lk_a_addr :
                                   pfetch_a_addr;
                    dma_a_len   <= seq1_len_bytes_a;
                    dma_a_ofm_mode <= lk_a_from_prev_ofm && tile_mode;
                                dma_a_im2col_mode <= lk_conv_im2col && !lk_a_from_prev_ofm;
                    dma_a_ofm_stride <= lk_k_dim;
                    dma_a_ofm_m_base <= seq1_m_base;
                    dma_a_ofm_k_base <= seq1_k_base;
                    dma_a_ofm_k_len <= seq1_k_len_32[15:0];
                    dma_a_ofm_active_rows <= seq1_active_rows;
                    dma_a_ofm_fp16_mode <= lk_dma_fp16_mode;
                    dma_a_im2col_m_index <= seq1_m_base;
                    dma_a_im2col_k_len <= lk_k_dim[15:0];
                    dma_a_im2col_ih <= lk_conv_ih;
                    dma_a_im2col_iw <= lk_conv_iw;
                    dma_a_im2col_cin <= lk_conv_cin;
                    dma_a_im2col_kh <= lk_conv_kh;
                    dma_a_im2col_kw <= lk_conv_kw;
                    dma_a_im2col_oh <= lk_conv_oh;
                    dma_a_im2col_ow <= lk_conv_ow;
                    dma_a_im2col_stride_h <= lk_conv_stride_h;
                    dma_a_im2col_stride_w <= lk_conv_stride_w;
                    dma_a_im2col_pad_h <= lk_conv_pad_h;
                    dma_a_im2col_pad_w <= lk_conv_pad_w;
                    dma_a_im2col_dilation_h <= lk_conv_dilation_h;
                    dma_a_im2col_dilation_w <= lk_conv_dilation_w;
                    dma_a_im2col_fp16_mode <= lk_dma_fp16_mode;
                    dma_bias_addr <= lk_bias_addr + (seq1_j << 2);
                end

                // Tile-mode bias: sequential per-column fetch
                if (lk_bias_en && tile_mode && (bias_col < tile_active_cols)) begin
                    if (!bias_pending) begin
                        `ifdef DIAG_BIAS
                        $display("[DIAG_BIAS] fetch col=%0d addr=0x%08h", bias_col,
                                 lk_bias_addr + ((tile_n_base + {27'd0, bias_col}) << 2));
                        `endif
                        dma_bias_start <= 1'b1;
                        dma_bias_addr <= lk_bias_addr + ((tile_n_base + {27'd0, bias_col}) << 2);
                        bias_pending  <= 1'b1;
                    end else if (dma_bias_done_r) begin
                        dma_bias_done_r <= 1'b0;
                        bias_col       <= bias_col + 1'b1;
                        bias_pending   <= 1'b0;
                    end
                end else begin
                    bias_col    <= 6'd0;
                    bias_pending <= 1'b0;
                end

                // Stay in S_PRELOAD while bias fetch is in progress
                if (!(lk_bias_en && tile_mode && (bias_col < tile_active_cols)))
                    state <= S_OVERLAP_COMPUTE;
            end

            // =================================================================
            // S_OVERLAP_COMPUTE
            //
            //   Ping bank   → PE is computing tile(tile_i, tile_j)
            //   Pong bank   → DMA is loading next tile (already launched)
            //
            //   Exit when all Ping data consumed (OS) or K+2 beats (WS).
            //   Then go to S_DRAIN.
            // =================================================================
            S_OVERLAP_COMPUTE: begin
                pe_flush <= 1'b0;

                if (cfg_abort) begin
                    state <= S_IDLE; busy <= 1'b0;
                    pe_en <= 1'b0; pe_load_w <= 1'b0;

                end else if (lk_stat[0] == 1'b0 && !tile_mode) begin
                    // ─── Legacy scalar WS mode ───
                    pe_load_w <= (ws_consume_cnt < lk_k_dim[15:0]) ? 1'b1 : 1'b0;
                    if (ws_consume_cnt < lk_k_dim[15:0] + 16'd2) begin
                        pe_en <= 1'b1;
                        if (compute_ce) ws_consume_cnt <= ws_consume_cnt + 1;
                    end else begin
                        pe_en     <= 1'b0;
                        pe_load_w <= 1'b0;
                        state     <= S_DRAIN;
                    end

                end else begin
                    // ─── Tile OS or packed direct-WS mode ───
                    pe_load_w <= tile_direct_ws && (tile_k_cycle < tile_k_cycles);
                    if (tile_mode) begin
                        pe_en <= 1'b1;
                        if (!pe_en) begin
                            // ── Deferred prefetch launch ──
                            // Was in S_PRELOAD but moved here to avoid racing bias_start
                            // with w_start/a_start on the shared DMA channel.
                            // Launched once per k_tile compute entry.
                            if (can_prefetch_next) begin
                                dma_w_addr  <= pfetch_w_addr;
                                dma_w_len   <= seq1_len_bytes_w;
                                dma_a_addr  <= (lk_a_from_prev_ofm && tile_mode) ? lk_a_addr :
                                               (lk_conv_im2col) ? lk_a_addr :
                                               pfetch_a_addr;
                                dma_a_len   <= seq1_len_bytes_a;
                                dma_a_ofm_mode <= lk_a_from_prev_ofm && tile_mode;
                    dma_a_im2col_mode <= lk_conv_im2col && !lk_a_from_prev_ofm;
                                dma_a_ofm_stride <= lk_k_dim;
                                dma_a_ofm_m_base <= seq1_m_base;
                                dma_a_ofm_k_base <= seq1_k_base;
                                dma_a_ofm_k_len <= seq1_k_len_32[15:0];
                                dma_a_ofm_active_rows <= seq1_active_rows;
                                dma_a_ofm_fp16_mode <= lk_dma_fp16_mode;
                                dma_a_im2col_m_index <= seq1_m_base;
                                dma_a_im2col_k_len <= lk_k_dim[15:0];
                                dma_a_im2col_ih <= lk_conv_ih;
                                dma_a_im2col_iw <= lk_conv_iw;
                                dma_a_im2col_cin <= lk_conv_cin;
                                dma_a_im2col_kh <= lk_conv_kh;
                                dma_a_im2col_kw <= lk_conv_kw;
                                dma_a_im2col_oh <= lk_conv_oh;
                                dma_a_im2col_ow <= lk_conv_ow;
                                dma_a_im2col_stride_h <= lk_conv_stride_h;
                                dma_a_im2col_stride_w <= lk_conv_stride_w;
                                dma_a_im2col_pad_h <= lk_conv_pad_h;
                                dma_a_im2col_pad_w <= lk_conv_pad_w;
                                dma_a_im2col_dilation_h <= lk_conv_dilation_h;
                                dma_a_im2col_dilation_w <= lk_conv_dilation_w;
                                dma_a_im2col_fp16_mode <= lk_dma_fp16_mode;
                                dma_w_start <= 1'b1;
                                dma_a_start <= 1'b1;
                                dma_w_done_r <= 1'b0;
                                dma_a_done_r <= 1'b0;
                            end
                            tile_k_cycle <= 16'd0;
                        end else if (compute_ce && (tile_k_cycle + 16'd1 >= tile_compute_cycles)) begin
                    // Keep pe_en=1 so the PE pipeline drains completely
                    // before S_DRAIN asserts flush on the next cycle.
                    tile_k_cycle <= 16'd0;
                    pe_load_w <= 1'b0;
                    // 8x32 two-pass: pass 0 done, load W for cols 16-31
                    if (is_8x32 && !pass_idx) begin
                        `ifdef DIAG_8X32
                        $display("[DIAG_CTRL] 8x32 pass transition: dma_w_addr=0x%08h tile_len_w=%0d",
                                 lk_w_addr + w_addr_pass1_offset +
                                 ((tile_j * lk_k_dim + tile_k_base) * bytes_per_k_w_32),
                                 tile_len_w);
                        `endif
                        pass_idx <= 1'b1;
                        dma_w_addr   <= lk_w_addr + (tile_j * w_tile_stride_n) +
                            w_addr_pass1_offset + (tile_k_base * bytes_per_k_w_32);
                        dma_w_len    <= tile_len_w;
                        dma_w_start  <= 1'b1;
                        dma_w_done_r <= 1'b0;
                        // Also re-load A for pass 1 (A PPBuf was consumed in pass 0)
                        dma_a_addr   <= lk_a_addr + (tile_i * a_tile_stride_m) +
                            (tile_k_base * bytes_per_k_a_32);
                        dma_a_len    <= tile_len_a;
                        dma_a_start  <= 1'b1;
                        dma_a_done_r <= 1'b0;
                        state <= S_WARMUP_LOAD;  // wait for both DMA, then swap and re-compute
                    end else if (!k_tile_last) begin
                        tile_i     <= seq1_i;
                        tile_j     <= seq1_j;
                        k_tile_idx <= seq1_k;
                        pass_idx   <= 1'b0;
                        if (dma_load_done) begin
                            w_ppb_swap   <= 1'b1;
                            a_ppb_swap   <= 1'b1;
                            dma_w_done_r <= 1'b0;
                            dma_a_done_r <= 1'b0;
                            dma_bias_done_r <= 1'b0;
                            state <= S_PRELOAD;
                        end else begin
                            state <= S_WAIT_PREFETCH;
                        end
                    end else begin
                        state <= S_DRAIN;
                    end
                        end else if (compute_ce) begin
                            tile_k_cycle <= tile_k_cycle + 16'd1;
                        end
                    end else begin
                        pe_en <= 1'b1;
                        if (w_ppb_empty && a_ppb_empty) begin
                            state <= S_DRAIN;
                        end
                    end
                end
            end

            // =================================================================
            // S_DRAIN – Assert pe_flush for 1 cycle.
            // =================================================================
            S_DRAIN: begin
                pe_en    <= 1'b1;
                pe_flush <= 1'b1;
                pe_load_w <= 1'b0;
                if (compute_ce) state <= S_DRAIN2;
            end

            // =================================================================
            // S_DRAIN2 – Flush pipeline 2nd cycle.
            // =================================================================
            S_DRAIN2: begin
                pe_en    <= 1'b1;
                pe_flush <= 1'b0;
                if (compute_ce) state <= S_WRITE_BACK;
            end

            // =================================================================
            // S_WRITE_BACK – Initiate DMA result write-back.
            // =================================================================
            S_WRITE_BACK: begin
                pe_en        <= 1'b1;
                pe_flush     <= 1'b0;
                // Scalar mode writes one C[i,j]. Tile mode writes one valid C row
                // per DMA transaction so global row-major gaps between rows are kept.
                dma_r_addr   <= tile_mode ? comp_row_r_addr : comp_r_addr;
                dma_r_len    <= tile_mode ? tile_row_r_len  : TILE_R_LEN;
                dma_r_start  <= 1'b1;
                dma_r_done_r <= 1'b0;
                state        <= S_WB_WAIT;
            end

            // =================================================================
            // S_WB_WAIT – Wait DMA WB done; then check prefetch status.
            //
            //   After WB completes:
            //     - If IS last tile → go to S_DONE.
            //     - If NOT last tile:
            //         If prefetch DMA ALREADY done → swap + S_PRELOAD → compute.
            //         Else → go to S_WAIT_PREFETCH to wait for it.
            // =================================================================
            S_WB_WAIT: begin
                pe_en <= 1'b1;
                dma_r_start <= 1'b0;
                if (dma_r_done_r) begin
                    dma_r_start  <= 1'b0;
                    dma_r_done_r <= 1'b0;
                    pe_en        <= 1'b0;

                    if (tile_mode && (wb_row + 5'd1 < tile_active_rows)) begin
                        wb_row <= wb_row + 5'd1;
                        state  <= S_WRITE_BACK;
                    end else if (is_last_seq) begin
                        wb_row <= 5'd0;
                        if (desc_mode_run) begin
                            prev_ofm_valid <= 1'b1;
                            prev_ofm_addr  <= lk_r_addr;
                        end
                        if (desc_count_exhausted) begin
                            busy          <= 1'b0;
                            done          <= 1'b0;
                            irq           <= 1'b0;
                            desc_mode_run <= 1'b0;
                            err_set_mask  <= ERR_DESC_COUNT_EXHAUSTED;
                            w_ppb_clear   <= 1'b1;
                            a_ppb_clear   <= 1'b1;
                            r_fifo_clear  <= 1'b1;
                            state         <= S_IDLE;
                        end else if (desc_can_fetch_next) begin
                            desc_addr       <= desc_next_addr;
                            desc_start      <= 1'b1;
                            desc_left       <= desc_left - 32'd1;
                            desc_next_addr  <= 32'd0;
                            desc_last_layer <= 1'b0;
                            w_ppb_clear     <= 1'b1;
                            a_ppb_clear     <= 1'b1;
                            r_fifo_clear    <= 1'b1;
                            state           <= S_FETCH_DESC;
                        end else begin
                            state <= S_DONE;
                        end
                    end else begin
                        wb_row <= 5'd0;
                        // Advance to the next m/n/k sequence.
                        tile_i     <= seq1_i;
                        tile_j     <= seq1_j;
                        k_tile_idx <= seq1_k;
                        pass_idx   <= 1'b0;

                        // Decide what to do based on prefetch status
                        if (dma_load_done) begin
                            // Prefetch already finished — swap banks
                            w_ppb_swap   <= 1'b1;
                            a_ppb_swap   <= 1'b1;
                            r_fifo_clear <= 1'b1;
                            pe_stat        <= lk_stat[0];
                            pe_load_w      <= ((lk_stat[0] == 1'b0) && (!tile_mode || tile_direct_ws)) ? 1'b1 : 1'b0;
                            ws_consume_cnt <= 16'd0;
                            tile_k_cycle   <= 16'd0;
                            dma_w_done_r   <= 1'b0;
                            dma_a_done_r   <= 1'b0;
                            dma_bias_done_r <= 1'b0;
                            state <= S_PRELOAD;  // 1-cycle swap propagation
                        end else begin
                            // Prefetch not yet done — wait in S_WAIT_PREFETCH
                            state <= S_WAIT_PREFETCH;
                        end
                    end
                end
            end

            // =================================================================
            // S_WAIT_PREFETCH - Wait for the in-flight prefetch DMA to finish,
            // then swap banks, wait S_PRELOAD, and enter compute.
            // =================================================================
            S_WAIT_PREFETCH: begin
                pe_en <= 1'b0;
                if (cfg_abort) begin
                    state <= S_IDLE; busy <= 1'b0;
                end else if (dma_load_done) begin
                    `ifdef DIAG_8X32
                    $display("[DIAG_8x32] pass-1 W/A DMA done, swapping PPBufs, entering S_PRELOAD");
                    `endif
                    // Prefetch finished — swap and start compute
                    w_ppb_swap   <= 1'b1;
                    a_ppb_swap   <= 1'b1;
                    if (!is_8x32)
                        r_fifo_clear <= 1'b1;
                    pe_stat        <= lk_stat[0];
                    pe_load_w      <= ((lk_stat[0] == 1'b0) && (!tile_mode || tile_direct_ws)) ? 1'b1 : 1'b0;
                    ws_consume_cnt <= 16'd0;
                    tile_k_cycle   <= 16'd0;
                    dma_w_done_r   <= 1'b0;
                    dma_a_done_r   <= 1'b0;
                    dma_bias_done_r <= 1'b0;
                    state <= S_PRELOAD;  // 1-cycle swap propagation
                end
            end

            // =================================================================
            // S_DONE – Layer complete. Assert IRQ. Return to S_IDLE.
            // =================================================================
            S_DONE: begin
                pe_en  <= 1'b0;
                busy   <= 1'b0;
                done   <= 1'b1;
                irq    <= (!desc_mode_run) || lk_desc_irq_en;
                pass_idx <= 1'b0;
                // Reset counters; clear buffers for next layer
                ws_consume_cnt <= 16'd0;
                tile_k_cycle   <= 16'd0;
                tile_i         <= 32'd0;
        pass_idx   <= 1'b0;
                tile_j         <= 32'd0;
                k_tile_idx     <= 32'd0;
        pass_idx   <= 1'b0;
                wb_row         <= 3'd0;
        pass_idx   <= 1'b0;
                desc_mode_run  <= 1'b0;
                desc_left      <= 32'd0;
                w_ppb_clear    <= 1'b1;
                a_ppb_clear    <= 1'b1;
                r_fifo_clear   <= 1'b1;
                state          <= S_IDLE;
            end

            default: state <= S_IDLE;

        endcase
        end
    end
end

endmodule
