// =============================================================================
// Module  : npu_dma
// Project : NPU_prj
// Desc    : AXI4 Master DMA with **Dual-FSM architecture** (Strategy C).
//
//           Two fully independent state machines share one AXI4 master port:
//             1. Load-FSM : Weight+Activation reads (AR/R channels)
//             2. WB-FSM   : Result write-back (AW/W/B channels)
//
//           AXI4 read/write channels are architecturally independent,
//           so both FSMs operate simultaneously with ZERO conflict.
//
//           Data flow:
//             DRAM --[Load-FSM]--> pingpong_buf --> PE --> FIFO --[WB-FSM]--> DRAM
// =============================================================================

`timescale 1ns/1ps

module npu_dma #(
    parameter DATA_W    = 32,
    parameter PE_DATA_W = 16,
    parameter BURST_MAX = 16,
    parameter PPB_DEPTH = 32,
    parameter PPB_THRESH= 16,
    parameter R_FIFO_DEPTH = 256
)(
    input  wire        clk,
    input  wire        rst_n,
    // ---- Channel 0: Weight ----
    input  wire        w_start,
    input  wire [31:0] w_base_addr,
    input  wire [15:0] w_len_bytes,
    output reg         w_done,
    output wire        w_ppb_wr_en,
    output wire [DATA_W-1:0] w_ppb_wr_data,
    input  wire        w_ppb_full,
    input  wire        w_ppb_buf_ready,
    input  wire        w_ppb_buf_empty,
    input  wire        w_ppb_drain_done,

    // ---- Channel 1: Activation ----
    input  wire        a_start,
    input  wire [31:0] a_base_addr,
    input  wire [15:0] a_len_bytes,
    output reg         a_done,
    output wire        a_ppb_wr_en,
    output wire [DATA_W-1:0] a_ppb_wr_data,
    input  wire        a_ppb_full,
    input  wire        a_ppb_buf_ready,
    input  wire        a_ppb_buf_empty,
    input  wire        a_ppb_drain_done,
    input  wire        a_ofm_mode,
    input  wire        a_im2col_mode,
    input  wire [31:0] a_ofm_stride,
    input  wire [31:0] a_ofm_m_base,
    input  wire [31:0] a_ofm_k_base,
    input  wire [15:0] a_ofm_k_len,
    input  wire [4:0]  a_ofm_active_rows,
    input  wire        a_ofm_fp16_mode,
    input  wire [31:0] a_im2col_m_index,
    input  wire [15:0] a_im2col_k_len,
    input  wire [15:0] a_im2col_ih,
    input  wire [15:0] a_im2col_iw,
    input  wire [15:0] a_im2col_cin,
    input  wire [15:0] a_im2col_kh,
    input  wire [15:0] a_im2col_kw,
    input  wire [15:0] a_im2col_oh,
    input  wire [15:0] a_im2col_ow,
    input  wire [7:0]  a_im2col_stride_h,
    input  wire [7:0]  a_im2col_stride_w,
    input  wire [7:0]  a_im2col_pad_h,
    input  wire [7:0]  a_im2col_pad_w,
    input  wire [7:0]  a_im2col_dilation_h,
    input  wire [7:0]  a_im2col_dilation_w,
    input  wire        a_im2col_fp16_mode,

    // ---- Descriptor fetch ----
    input  wire        desc_start,
    input  wire [31:0] desc_base_addr,
    output reg         desc_done,
    output reg  [511:0] desc_words,

    // ---- Bias fetch ----
    input  wire        bias_start,
    input  wire [31:0] bias_addr,
    output reg         bias_done,
    output reg  [31:0] bias_data,

    // ---- Channel 2: Result ----
    input  wire        r_start,
    input  wire [31:0] r_base_addr,
    input  wire [15:0] r_len_bytes,
    output reg         r_done,
    input  wire        r_fifo_clear,      // synchronous clear for result FIFO
    input  wire        r_fifo_wr_en,
    input  wire [DATA_W-1:0] r_fifo_din,
    output wire        r_fifo_full,

    // ---- AXI4 Master (shared) ----
    // AW channel
    output wire [31:0] m_axi_awaddr,
    output wire [7:0]  m_axi_awlen,
    output wire [2:0]  m_axi_awsize,
    output wire [1:0]  m_axi_awburst,
    output reg         m_axi_awvalid,
    input  wire        m_axi_awready,
    // W channel
    output wire [DATA_W-1:0] m_axi_wdata,
    output reg  [DATA_W/8-1:0] m_axi_wstrb,
    output wire        m_axi_wlast,
    output reg         m_axi_wvalid,
    input  wire        m_axi_wready,
    // B channel
    input  wire [1:0]  m_axi_bresp,
    input  wire        m_axi_bvalid,
    output reg         m_axi_bready,
    // AR channel
    output wire [31:0] m_axi_araddr,
    output wire [7:0]  m_axi_arlen,
    output wire [2:0]  m_axi_arsize,
    output wire [1:0]  m_axi_arburst,
    output reg         m_axi_arvalid,
    input  wire        m_axi_arready,
    // R channel
    input  wire [DATA_W-1:0] m_axi_rdata,
    input  wire [1:0]  m_axi_rresp,
    input  wire        m_axi_rvalid,
    output reg         m_axi_rready,
    input  wire        m_axi_rlast
);

// ===========================================================================
// Result sync FIFO
// ===========================================================================
wire [$clog2(R_FIFO_DEPTH):0] r_fill;
wire r_fifo_rd_wire = (wb_state == WB_ACTIVE) && m_axi_wvalid && m_axi_wready;

sync_fifo #(.DATA_W(DATA_W), .DEPTH(R_FIFO_DEPTH), .ALMOST_FULL(4), .ALMOST_EMPTY(2))
    u_r_fifo (
        .clk(clk), .rst_n(rst_n),
        .clear(r_fifo_clear),
        .wr_en(r_fifo_wr_en), .wr_data(r_fifo_din),
        .full(r_fifo_full), .almost_full(),
        .rd_en(r_fifo_rd_wire), .rd_data(r_fifo_dout_int),
        .empty(r_fifo_empty_n), .almost_empty(), .fill_count(r_fill)
    );

wire [DATA_W-1:0] r_fifo_dout_int;
wire r_fifo_empty_n;

// ===========================================================================
// PPBuf write enables
// ===========================================================================
assign w_ppb_wr_en = (load_state == L_WREAD || (load_state == L_WA_READ && load_reading_w))
                      && m_axi_rvalid && m_axi_rready && !w_ppb_full;
assign w_ppb_wr_data = m_axi_rdata;

reg        a_ofm_wr_en;
reg [DATA_W-1:0] a_ofm_wr_data;
reg        a_im2col_wr_en;
reg [DATA_W-1:0] a_im2col_wr_data;

wire a_read_ppb_wr_en = (load_state == L_AREAD || (load_state == L_WA_READ && !load_reading_w))
                         && m_axi_rvalid && m_axi_rready && !a_ppb_full;

assign a_ppb_wr_en = a_read_ppb_wr_en ||
                     (a_ofm_wr_en && !a_ppb_full) ||
                     (a_im2col_wr_en && !a_ppb_full);
assign a_ppb_wr_data = a_ofm_wr_en ? a_ofm_wr_data :
                       a_im2col_wr_en ? a_im2col_wr_data :
                       m_axi_rdata;

// ===========================================================================
// Load-FSM states
// ===========================================================================
localparam [3:0] L_IDLE   = 4'd0;
localparam [3:0] L_WREAD  = 4'd1;
localparam [3:0] L_AREAD  = 4'd2;
localparam [3:0] L_WA_READ= 4'd3;
localparam [3:0] L_DESC   = 4'd4;
localparam [3:0] L_A_OFM  = 4'd5;
localparam [3:0] L_A_OFM_DONE = 4'd6;
localparam [3:0] L_A_IM2COL = 4'd7;
localparam [3:0] L_A_IM2COL_DONE = 4'd8;
localparam [3:0] L_BIAS = 4'd9;

reg [3:0] load_state;
reg [31:0] load_addr_cnt;
reg [15:0] load_byte_cnt;
reg [15:0] w_bytes_done, a_bytes_done;
reg [15:0] desc_bytes_done;
reg [4:0]  desc_word_idx;
reg        load_reading_w;
reg [7:0]  load_arlen;
reg        load_r_active;
reg        load_a_after_w;
reg        load_a_after_w_ofm;
reg        load_a_after_w_im2col;
reg        load_bias_after_data;
reg [31:0] bias_addr_latch;

reg [31:0] ofm_stride_latch;
reg [31:0] ofm_base_latch;
reg [31:0] ofm_m_base_latch;
reg [31:0] ofm_k_base_latch;
reg [15:0] ofm_k_len_latch;
reg [4:0]  ofm_active_rows_latch;
reg        ofm_fp16_latch;
reg [15:0] ofm_k_pos;
reg [4:0]  ofm_lane_pos;
reg [1:0]  ofm_emit_phase;
reg [31:0] ofm_pack0;
reg [31:0] ofm_pack1;

reg [31:0] im2col_base_latch;
reg [31:0] im2col_m_latch;
reg [15:0] im2col_k_len_latch;
reg [15:0] im2col_ih_latch, im2col_iw_latch, im2col_cin_latch;
reg [15:0] im2col_kh_latch, im2col_kw_latch, im2col_oh_latch, im2col_ow_latch;
reg [7:0]  im2col_stride_h_latch, im2col_stride_w_latch;
reg [7:0]  im2col_pad_h_latch, im2col_pad_w_latch;
reg [7:0]  im2col_dilation_h_latch, im2col_dilation_w_latch;
reg        im2col_fp16_latch;
reg [15:0] im2col_k_pos;
reg [1:0]  im2col_lane_pos;
reg [31:0] im2col_pack_word;

localparam integer AXI_DATA_BYTES = DATA_W / 8;
localparam integer AXI_BYTE_SHIFT = $clog2(DATA_W / 8);
localparam [15:0] READ_BURST_MAX_BEATS =
    (BURST_MAX > 256) ? 16'd256 : BURST_MAX;
localparam [15:0] DESC_BYTES = 16'd64;

wire [15:0] load_target_len =
    (load_state == L_DESC)  ? DESC_BYTES  :
    (load_state == L_WREAD) ? w_len_bytes :
    (load_state == L_AREAD) ? a_len_bytes :
    load_reading_w          ? w_len_bytes : a_len_bytes;

wire [15:0] load_target_done =
    (load_state == L_DESC)  ? desc_bytes_done :
    (load_state == L_WREAD) ? w_bytes_done    :
    (load_state == L_AREAD) ? a_bytes_done    :
    load_reading_w          ? w_bytes_done    : a_bytes_done;

wire [31:0] ofm_cur_addr =
    ofm_base_latch +
    ((((ofm_m_base_latch + {29'd0, ofm_lane_pos}) * ofm_stride_latch) +
      (ofm_k_base_latch + {16'd0, ofm_k_pos})) << AXI_BYTE_SHIFT);

function [31:0] pack_im2col_elem;
    input [31:0] cur_word;
    input [1:0]  lane;
    input        fp16_mode;
    input [15:0] elem;
    begin
        pack_im2col_elem = cur_word;
        if (fp16_mode) begin
            if (lane[0] == 1'b0)
                pack_im2col_elem[15:0] = elem;
            else
                pack_im2col_elem[31:16] = elem;
        end else begin
            case (lane)
                2'd0: pack_im2col_elem[7:0]   = elem[7:0];
                2'd1: pack_im2col_elem[15:8]  = elem[7:0];
                2'd2: pack_im2col_elem[23:16] = elem[7:0];
                default: pack_im2col_elem[31:24] = elem[7:0];
            endcase
        end
    end
endfunction

wire [15:0] im2col_lanes_per_word = im2col_fp16_latch ? 16'd2 : 16'd4;
wire [15:0] im2col_total_lanes =
    im2col_fp16_latch ? ((im2col_k_len_latch + 16'd1) & 16'hfffe)
                      : ((im2col_k_len_latch + 16'd3) & 16'hfffc);
wire        im2col_lane_last =
    im2col_fp16_latch ? (im2col_lane_pos[0] == 1'b1)
                      : (im2col_lane_pos == 2'd3);
wire        im2col_elem_last = (im2col_k_pos + 16'd1 >= im2col_total_lanes);
wire        im2col_is_padding = (im2col_k_pos >= im2col_k_len_latch);

wire [31:0] im2col_ohow = im2col_oh_latch * im2col_ow_latch;
wire [31:0] im2col_spatial = (im2col_ohow == 32'd0) ? 32'd0 :
                              (im2col_m_latch % im2col_ohow);
wire [31:0] im2col_b_idx = (im2col_ohow == 32'd0) ? 32'd0 :
                            (im2col_m_latch / im2col_ohow);
wire [31:0] im2col_out_h = (im2col_ow_latch == 16'd0) ? 32'd0 :
                            (im2col_spatial / im2col_ow_latch);
wire [31:0] im2col_out_w = (im2col_ow_latch == 16'd0) ? 32'd0 :
                            (im2col_spatial % im2col_ow_latch);
wire [31:0] im2col_kernel_area = im2col_kh_latch * im2col_kw_latch;
wire [31:0] im2col_c_idx = (im2col_kernel_area == 32'd0) ? 32'd0 :
                            (im2col_k_pos / im2col_kernel_area);
wire [31:0] im2col_kernel_rem = (im2col_kernel_area == 32'd0) ? 32'd0 :
                                 (im2col_k_pos % im2col_kernel_area);
wire [31:0] im2col_kh_idx = (im2col_kw_latch == 16'd0) ? 32'd0 :
                             (im2col_kernel_rem / im2col_kw_latch);
wire [31:0] im2col_kw_idx = (im2col_kw_latch == 16'd0) ? 32'd0 :
                             (im2col_kernel_rem % im2col_kw_latch);

wire signed [31:0] im2col_in_h_s =
    $signed({1'b0, im2col_out_h * {24'd0, im2col_stride_h_latch}}) +
    $signed({1'b0, im2col_kh_idx * {24'd0, im2col_dilation_h_latch}}) -
    $signed({24'd0, im2col_pad_h_latch});
wire signed [31:0] im2col_in_w_s =
    $signed({1'b0, im2col_out_w * {24'd0, im2col_stride_w_latch}}) +
    $signed({1'b0, im2col_kw_idx * {24'd0, im2col_dilation_w_latch}}) -
    $signed({24'd0, im2col_pad_w_latch});

wire im2col_in_bounds =
    !im2col_is_padding &&
    (im2col_in_h_s >= 0) && (im2col_in_w_s >= 0) &&
    (im2col_in_h_s < $signed({16'd0, im2col_ih_latch})) &&
    (im2col_in_w_s < $signed({16'd0, im2col_iw_latch}));

wire [31:0] im2col_elem_index =
    ((((im2col_b_idx * im2col_cin_latch) + im2col_c_idx) * im2col_ih_latch +
      im2col_in_h_s[31:0]) * im2col_iw_latch) + im2col_in_w_s[31:0];
wire [31:0] im2col_elem_addr =
    im2col_base_latch + (im2col_fp16_latch ? (im2col_elem_index << 1)
                                           : im2col_elem_index);
wire [31:0] im2col_aligned_addr = {im2col_elem_addr[31:2], 2'b00};
wire [15:0] im2col_r_elem =
    im2col_fp16_latch ? (im2col_elem_addr[1] ? m_axi_rdata[31:16] : m_axi_rdata[15:0]) :
                        {8'd0,
                         (im2col_elem_addr[1:0] == 2'd0) ? m_axi_rdata[7:0] :
                         (im2col_elem_addr[1:0] == 2'd1) ? m_axi_rdata[15:8] :
                         (im2col_elem_addr[1:0] == 2'd2) ? m_axi_rdata[23:16] :
                                                           m_axi_rdata[31:24]};

wire [15:0] load_remaining_bytes =
    (load_target_len > load_target_done) ? (load_target_len - load_target_done) : 16'd0;

wire [15:0] load_remaining_beats =
    (load_remaining_bytes == 16'd0) ? 16'd0 :
    ((load_remaining_bytes + AXI_DATA_BYTES - 1) >> AXI_BYTE_SHIFT);

// AXI INCR bursts must not cross a 4KB boundary.
wire [15:0] load_bytes_to_4k =
    16'd4096 - {4'd0, load_addr_cnt[11:0]};
wire [15:0] load_beats_to_4k =
    (load_bytes_to_4k >> AXI_BYTE_SHIFT);

wire [15:0] load_burst_beats_cap0 =
    (load_remaining_beats > READ_BURST_MAX_BEATS) ? READ_BURST_MAX_BEATS :
                                                    load_remaining_beats;
wire [15:0] load_burst_beats =
    (load_burst_beats_cap0 > load_beats_to_4k) ? load_beats_to_4k :
                                                 load_burst_beats_cap0;

wire [7:0] load_next_arlen =
    ((load_burst_beats == 16'd0) ? 16'd1 : load_burst_beats) - 1'b1;

always @(posedge clk) begin
    if (!rst_n) begin
        load_state <= L_IDLE; load_addr_cnt <= 0; load_byte_cnt <= 0;
        w_done <= 0; a_done <= 0;
        w_bytes_done <= 0; a_bytes_done <= 0;
        desc_done <= 1'b0;
        desc_words <= 512'd0;
        bias_done <= 1'b0;
        bias_data <= 32'd0;
        desc_bytes_done <= 16'd0;
        desc_word_idx <= 5'd0;
        load_reading_w <= 1;
        load_arlen <= 8'd0;
        load_r_active <= 1'b0;
        load_a_after_w <= 1'b0;
        load_a_after_w_ofm <= 1'b0;
        load_a_after_w_im2col <= 1'b0;
        load_bias_after_data <= 1'b0;
        bias_addr_latch <= 32'd0;
        a_ofm_wr_en <= 1'b0;
        a_ofm_wr_data <= {DATA_W{1'b0}};
        a_im2col_wr_en <= 1'b0;
        a_im2col_wr_data <= {DATA_W{1'b0}};
        ofm_stride_latch <= 32'd0;
        ofm_base_latch <= 32'd0;
        ofm_m_base_latch <= 32'd0;
        ofm_k_base_latch <= 32'd0;
        ofm_k_len_latch <= 16'd0;
        ofm_active_rows_latch <= 3'd0;
        ofm_fp16_latch <= 1'b0;
        ofm_k_pos <= 16'd0;
        ofm_lane_pos <= 3'd0;
        ofm_emit_phase <= 2'd0;
        ofm_pack0 <= 32'd0;
        ofm_pack1 <= 32'd0;
        im2col_base_latch <= 32'd0;
        im2col_m_latch <= 32'd0;
        im2col_k_len_latch <= 16'd0;
        im2col_ih_latch <= 16'd0; im2col_iw_latch <= 16'd0; im2col_cin_latch <= 16'd0;
        im2col_kh_latch <= 16'd0; im2col_kw_latch <= 16'd0; im2col_oh_latch <= 16'd0; im2col_ow_latch <= 16'd0;
        im2col_stride_h_latch <= 8'd1; im2col_stride_w_latch <= 8'd1;
        im2col_pad_h_latch <= 8'd0; im2col_pad_w_latch <= 8'd0;
        im2col_dilation_h_latch <= 8'd1; im2col_dilation_w_latch <= 8'd1;
        im2col_fp16_latch <= 1'b0;
        im2col_k_pos <= 16'd0;
        im2col_lane_pos <= 2'd0;
        im2col_pack_word <= 32'd0;
        m_axi_arvalid <= 0; m_axi_rready <= 0;
    end else begin
        w_done <= 1'b0;
        a_done <= 1'b0;
        desc_done <= 1'b0;
        bias_done <= 1'b0;
        a_ofm_wr_en <= 1'b0;
        a_im2col_wr_en <= 1'b0;
        case (load_state)
            L_IDLE: begin
                w_bytes_done <= 0; a_bytes_done <= 0;
                desc_bytes_done <= 16'd0;
                desc_word_idx <= 5'd0;
                m_axi_arvalid <= 0; m_axi_rready <= 0;
                load_r_active <= 1'b0;
                load_reading_w <= 1;
                load_byte_cnt <= 16'd0;
                load_a_after_w <= 1'b0;
                load_a_after_w_ofm <= 1'b0;
                load_a_after_w_im2col <= 1'b0;
                load_bias_after_data <= 1'b0;
                ofm_emit_phase <= 2'd0;
                ofm_lane_pos <= 3'd0;
                ofm_k_pos <= 16'd0;
                ofm_pack0 <= 32'd0;
                ofm_pack1 <= 32'd0;
                im2col_k_pos <= 16'd0;
                im2col_lane_pos <= 2'd0;
                im2col_pack_word <= 32'd0;

                if (w_start && (w_len_bytes == 16'd0))
                    w_done <= 1'b1;
                if (a_start && !a_ofm_mode && !a_im2col_mode && (a_len_bytes == 16'd0))
                    a_done <= 1'b1;
                if (a_start && a_ofm_mode && (a_ofm_k_len == 16'd0))
                    a_done <= 1'b1;
                if (a_start && a_im2col_mode && (a_im2col_k_len == 16'd0))
                    a_done <= 1'b1;

                // Result WB uses independent AW/W/B channels; read-side DMA can
                // issue W/A bursts while writeback is active.
                if (desc_start) begin
                    load_state       <= L_DESC;
                    load_addr_cnt    <= desc_base_addr;
                    load_byte_cnt    <= 16'd0;
                    desc_bytes_done  <= 16'd0;
                    desc_word_idx    <= 5'd0;
                end else if (w_start && (w_len_bytes != 16'd0) && !w_ppb_full) begin
                    `ifdef DIAG_8X32
                    $display("[DIAG_DMA] W DMA start: addr=0x%08h len=%0d", w_base_addr, w_len_bytes);
                    `endif
                    load_state    <= L_WREAD;
                    load_addr_cnt <= w_base_addr;
                    load_byte_cnt <= 16'd0;
                    w_bytes_done  <= 16'd0;
                    a_bytes_done  <= 16'd0;
                    load_a_after_w <= a_start && ((a_ofm_mode && (a_ofm_k_len != 16'd0)) ||
                                                  (a_im2col_mode && (a_im2col_k_len != 16'd0)) ||
                                                  (!a_ofm_mode && !a_im2col_mode && (a_len_bytes != 16'd0)));
                    load_a_after_w_ofm <= a_ofm_mode;
                    load_a_after_w_im2col <= a_im2col_mode;
                    load_bias_after_data <= bias_start;
                    bias_addr_latch <= bias_addr;
                    if (a_start && a_ofm_mode) begin
                        ofm_stride_latch <= a_ofm_stride;
                        ofm_base_latch <= a_base_addr;
                        ofm_m_base_latch <= a_ofm_m_base;
                        ofm_k_base_latch <= a_ofm_k_base;
                        ofm_k_len_latch <= a_ofm_k_len;
                        ofm_active_rows_latch <= a_ofm_active_rows;
                        ofm_fp16_latch <= a_ofm_fp16_mode;
                    end else if (a_start && a_im2col_mode) begin
                        im2col_base_latch <= a_base_addr;
                        im2col_m_latch <= a_im2col_m_index;
                        im2col_k_len_latch <= a_im2col_k_len;
                        im2col_ih_latch <= a_im2col_ih;
                        im2col_iw_latch <= a_im2col_iw;
                        im2col_cin_latch <= a_im2col_cin;
                        im2col_kh_latch <= a_im2col_kh;
                        im2col_kw_latch <= a_im2col_kw;
                        im2col_oh_latch <= a_im2col_oh;
                        im2col_ow_latch <= a_im2col_ow;
                        im2col_stride_h_latch <= a_im2col_stride_h;
                        im2col_stride_w_latch <= a_im2col_stride_w;
                        im2col_pad_h_latch <= a_im2col_pad_h;
                        im2col_pad_w_latch <= a_im2col_pad_w;
                        im2col_dilation_h_latch <= a_im2col_dilation_h;
                        im2col_dilation_w_latch <= a_im2col_dilation_w;
                        im2col_fp16_latch <= a_im2col_fp16_mode;
                        im2col_k_pos <= 16'd0;
                        im2col_lane_pos <= 2'd0;
                        im2col_pack_word <= 32'd0;
                    end
                end else if (a_start && a_ofm_mode && (a_ofm_k_len != 16'd0) && !a_ppb_full) begin
                    load_state       <= L_A_OFM;
                    load_addr_cnt    <= a_base_addr;
                    load_byte_cnt    <= 16'd0;
                    a_bytes_done     <= 16'd0;
                    ofm_stride_latch <= a_ofm_stride;
                    ofm_base_latch <= a_base_addr;
                    ofm_m_base_latch <= a_ofm_m_base;
                    ofm_k_base_latch <= a_ofm_k_base;
                    ofm_k_len_latch <= a_ofm_k_len;
                    ofm_active_rows_latch <= a_ofm_active_rows;
                    ofm_fp16_latch <= a_ofm_fp16_mode;
                    ofm_k_pos <= 16'd0;
                    ofm_lane_pos <= 3'd0;
                    ofm_emit_phase <= 2'd0;
                    ofm_pack0 <= 32'd0;
                    ofm_pack1 <= 32'd0;
                    load_bias_after_data <= bias_start;
                    bias_addr_latch <= bias_addr;
                end else if (a_start && a_im2col_mode && (a_im2col_k_len != 16'd0) && !a_ppb_full) begin
                    load_state <= L_A_IM2COL;
                    load_addr_cnt <= a_base_addr;
                    load_byte_cnt <= 16'd0;
                    a_bytes_done <= 16'd0;
                    im2col_base_latch <= a_base_addr;
                    im2col_m_latch <= a_im2col_m_index;
                    im2col_k_len_latch <= a_im2col_k_len;
                    im2col_ih_latch <= a_im2col_ih;
                    im2col_iw_latch <= a_im2col_iw;
                    im2col_cin_latch <= a_im2col_cin;
                    im2col_kh_latch <= a_im2col_kh;
                    im2col_kw_latch <= a_im2col_kw;
                    im2col_oh_latch <= a_im2col_oh;
                    im2col_ow_latch <= a_im2col_ow;
                    im2col_stride_h_latch <= a_im2col_stride_h;
                    im2col_stride_w_latch <= a_im2col_stride_w;
                    im2col_pad_h_latch <= a_im2col_pad_h;
                    im2col_pad_w_latch <= a_im2col_pad_w;
                    im2col_dilation_h_latch <= a_im2col_dilation_h;
                    im2col_dilation_w_latch <= a_im2col_dilation_w;
                    im2col_fp16_latch <= a_im2col_fp16_mode;
                    im2col_k_pos <= 16'd0;
                    im2col_lane_pos <= 2'd0;
                    im2col_pack_word <= 32'd0;
                    load_bias_after_data <= bias_start;
                    bias_addr_latch <= bias_addr;
                end else if (a_start && (a_len_bytes != 16'd0) && !a_ppb_full) begin
                    load_state    <= L_AREAD;
                    load_addr_cnt <= a_base_addr;
                    load_byte_cnt <= 16'd0;
                    a_bytes_done  <= 16'd0;
                    load_bias_after_data <= bias_start;
                    bias_addr_latch <= bias_addr;
                end else if (bias_start) begin
                    load_state <= L_BIAS;
                    load_addr_cnt <= bias_addr;
                    load_byte_cnt <= 16'd0;
                end
            end

            // Descriptor fetch. Reads one descriptor v1 record: 16 x 32-bit words.
            L_DESC: begin
                if (!m_axi_arvalid && !load_r_active) begin
                    m_axi_arvalid <= 1'b1;
                    load_arlen    <= load_next_arlen;
                end

                if (m_axi_arvalid && m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;
                    load_r_active <= 1'b1;
                    m_axi_rready  <= 1'b1;
                    load_byte_cnt <= 16'd0;
                end else if (load_r_active) begin
                    m_axi_rready <= 1'b1;
                    if (m_axi_rvalid && m_axi_rready) begin
                        desc_words[(desc_word_idx * DATA_W) +: DATA_W] <= m_axi_rdata;
                        desc_word_idx   <= desc_word_idx + 1'b1;
                        load_byte_cnt   <= load_byte_cnt + AXI_DATA_BYTES;
                        desc_bytes_done <= desc_bytes_done + AXI_DATA_BYTES;
                        if (m_axi_rlast) begin
                            load_r_active <= 1'b0;
                            m_axi_rready  <= 1'b0;
                            load_addr_cnt <= load_addr_cnt +
                                             (({8'd0, load_arlen} + 16'd1) << AXI_BYTE_SHIFT);
                            load_byte_cnt <= 16'd0;
                            if (desc_bytes_done + AXI_DATA_BYTES >= DESC_BYTES) begin
                                desc_done  <= 1'b1;
                                load_state <= L_IDLE;
                            end
                        end
                    end
                end
            end

            // Weight read. T3.1: one outstanding INCR burst at a time.
            L_WREAD: begin
                if (!m_axi_arvalid && !load_r_active && !w_ppb_full) begin
                    m_axi_arvalid <= 1'b1;
                    load_arlen    <= load_next_arlen;
                end

                if (m_axi_arvalid && m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;
                    load_r_active <= 1'b1;
                    m_axi_rready  <= !w_ppb_full;
                    load_byte_cnt <= 16'd0;
                end else if (load_r_active) begin
                    m_axi_rready <= !w_ppb_full;
                    if (m_axi_rvalid && m_axi_rready) begin
                        load_byte_cnt <= load_byte_cnt + AXI_DATA_BYTES;
                        w_bytes_done  <= w_bytes_done + AXI_DATA_BYTES;
                        if (m_axi_rlast) begin
                            load_r_active <= 1'b0;
                            m_axi_rready  <= 1'b0;
                            load_addr_cnt <= load_addr_cnt +
                                             (({8'd0, load_arlen} + 16'd1) << AXI_BYTE_SHIFT);
                            load_byte_cnt <= 16'd0;
                            if (w_bytes_done + AXI_DATA_BYTES >= w_len_bytes) begin
                                w_done <= 1'b1;
                                if (load_a_after_w) begin
                                    load_state    <= load_a_after_w_ofm ? L_A_OFM :
                                                     load_a_after_w_im2col ? L_A_IM2COL :
                                                     L_AREAD;
                                    load_addr_cnt <= a_base_addr;
                                    a_bytes_done  <= 16'd0;
                                    ofm_k_pos     <= 16'd0;
                                    ofm_lane_pos  <= 3'd0;
                                    ofm_emit_phase <= 2'd0;
                                    ofm_pack0     <= 32'd0;
                                    ofm_pack1     <= 32'd0;
                                    im2col_k_pos <= 16'd0;
                                    im2col_lane_pos <= 2'd0;
                                    im2col_pack_word <= 32'd0;
                                    load_a_after_w <= 1'b0;
                                    load_a_after_w_ofm <= 1'b0;
                                    load_a_after_w_im2col <= 1'b0;
                                end else if (load_bias_after_data) begin
                                    load_state <= L_BIAS;
                                    load_addr_cnt <= bias_addr_latch;
                                end else begin
                                    load_state <= L_IDLE;
                                    load_a_after_w <= 1'b0;
                                    load_a_after_w_ofm <= 1'b0;
                                    load_a_after_w_im2col <= 1'b0;
                                    load_bias_after_data <= 1'b0;
                                end
                            end
                        end
                    end
                end
            end

            // Activation read. Same burst engine as WREAD, targeting A PPBuf.
            L_AREAD: begin
                if (!m_axi_arvalid && !load_r_active && !a_ppb_full) begin
                    m_axi_arvalid <= 1'b1;
                    load_arlen    <= load_next_arlen;
                end

                if (m_axi_arvalid && m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;
                    load_r_active <= 1'b1;
                    m_axi_rready  <= !a_ppb_full;
                    load_byte_cnt <= 16'd0;
                end else if (load_r_active) begin
                    m_axi_rready <= !a_ppb_full;
                    if (m_axi_rvalid && m_axi_rready) begin
                        load_byte_cnt <= load_byte_cnt + AXI_DATA_BYTES;
                        a_bytes_done  <= a_bytes_done + AXI_DATA_BYTES;
                        if (m_axi_rlast) begin
                            load_r_active <= 1'b0;
                            m_axi_rready  <= 1'b0;
                            load_addr_cnt <= load_addr_cnt +
                                             (({8'd0, load_arlen} + 16'd1) << AXI_BYTE_SHIFT);
                            load_byte_cnt <= 16'd0;
                            if (a_bytes_done + AXI_DATA_BYTES >= a_len_bytes) begin
                                a_done     <= 1'b1;
                                if (load_bias_after_data) begin
                                    load_state <= L_BIAS;
                                    load_addr_cnt <= bias_addr_latch;
                                end else begin
                                    load_state <= L_IDLE;
                                end
                            end
                        end
                    end
                end
            end

            // Activation source is the previous layer's 32-bit row-major OFM.
            // Gather A[m0+r,k] for r=0..3 and repack into the current 4-lane
            // tile stream consumed by pingpong_buf.
            L_A_OFM: begin
                if (ofm_emit_phase != 2'd0) begin
                    if (!a_ppb_full) begin
                        a_ofm_wr_en <= 1'b1;
                        a_ofm_wr_data <= (ofm_emit_phase == 2'd1) ? ofm_pack0 : ofm_pack1;
                        if (!ofm_fp16_latch || ofm_emit_phase == 2'd2) begin
                            if (ofm_k_pos + 16'd1 >= ofm_k_len_latch) begin
                                load_state <= L_A_OFM_DONE;
                                ofm_emit_phase <= 2'd0;
                            end else begin
                                ofm_k_pos <= ofm_k_pos + 16'd1;
                                ofm_lane_pos <= 3'd0;
                                ofm_emit_phase <= 2'd0;
                                ofm_pack0 <= 32'd0;
                                ofm_pack1 <= 32'd0;
                            end
                        end else begin
                            ofm_emit_phase <= 2'd2;
                        end
                    end
                end else if (ofm_lane_pos >= 5'd4) begin
                    ofm_emit_phase <= 2'd1;
                end else if (ofm_lane_pos >= ofm_active_rows_latch) begin
                    case (ofm_lane_pos)
                        3'd0: ofm_pack0[ 7: 0] <= 8'd0;
                        3'd1: begin
                            if (ofm_fp16_latch) ofm_pack0[31:16] <= 16'd0;
                            else                ofm_pack0[15: 8] <= 8'd0;
                        end
                        3'd2: begin
                            if (ofm_fp16_latch) ofm_pack1[15: 0] <= 16'd0;
                            else                ofm_pack0[23:16] <= 8'd0;
                        end
                        default: begin
                            if (ofm_fp16_latch) ofm_pack1[31:16] <= 16'd0;
                            else                ofm_pack0[31:24] <= 8'd0;
                        end
                    endcase
                    ofm_lane_pos <= ofm_lane_pos + 1'b1;
                end else if (!m_axi_arvalid && !load_r_active && !a_ppb_full) begin
                    load_addr_cnt <= ofm_cur_addr;
                    load_arlen    <= 8'd0;
                    m_axi_arvalid <= 1'b1;
                end else if (m_axi_arvalid && m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;
                    load_r_active <= 1'b1;
                    m_axi_rready  <= 1'b1;
                end else if (load_r_active) begin
                    m_axi_rready <= 1'b1;
                    if (m_axi_rvalid && m_axi_rready) begin
                        case (ofm_lane_pos)
                            3'd0: begin
                                if (ofm_fp16_latch) ofm_pack0[15: 0] <= m_axi_rdata[15:0];
                                else                ofm_pack0[ 7: 0] <= m_axi_rdata[7:0];
                            end
                            3'd1: begin
                                if (ofm_fp16_latch) ofm_pack0[31:16] <= m_axi_rdata[15:0];
                                else                ofm_pack0[15: 8] <= m_axi_rdata[7:0];
                            end
                            3'd2: begin
                                if (ofm_fp16_latch) ofm_pack1[15: 0] <= m_axi_rdata[15:0];
                                else                ofm_pack0[23:16] <= m_axi_rdata[7:0];
                            end
                            default: begin
                                if (ofm_fp16_latch) ofm_pack1[31:16] <= m_axi_rdata[15:0];
                                else                ofm_pack0[31:24] <= m_axi_rdata[7:0];
                            end
                        endcase
                        load_r_active <= 1'b0;
                        m_axi_rready  <= 1'b0;
                        ofm_lane_pos  <= ofm_lane_pos + 1'b1;
                    end
                end
            end

            // One-cycle delay after the final packed OFM word so the A PPBuf
            // sees the last a_ofm_wr_en before the controller swaps banks.
            L_A_OFM_DONE: begin
                a_done <= 1'b1;
                if (load_bias_after_data) begin
                    load_state <= L_BIAS;
                    load_addr_cnt <= bias_addr_latch;
                end else begin
                    load_state <= L_IDLE;
                end
            end

            // T6.2: on-the-fly Conv2D im2col gather for direct scalar mode.
            // The controller supplies the output row m; this FSM walks k over
            // Cin/KH/KW, reads IFM[b,cin,ih,iw] when in bounds, emits zero for
            // padding, and packs the generated A row into 32-bit PPBuf words.
            L_A_IM2COL: begin
                if (im2col_k_pos == 16'd0) $display("[DMA_IM2COL] START k_len=%0d total=%0d ppb_full=%0d",
                    im2col_k_len_latch, im2col_total_lanes, a_ppb_full);
                if (im2col_k_pos >= im2col_total_lanes) begin
                    $display("[DMA_IM2COL] DONE at cycle %0t, k_pos=%0d", $time, im2col_k_pos);
                    load_state <= L_A_IM2COL_DONE;
                end else if (im2col_is_padding || !im2col_in_bounds) begin
                    if (!im2col_lane_last || !a_ppb_full) begin
                        if (im2col_k_pos < 16'd5) $display("[DMA_IM2COL] pad k=%0d", im2col_k_pos);
                        if (im2col_lane_last) begin
                            a_im2col_wr_en <= 1'b1;
                            a_im2col_wr_data <= pack_im2col_elem(im2col_pack_word,
                                                                  im2col_lane_pos,
                                                                  im2col_fp16_latch,
                                                                  16'd0);
                            im2col_pack_word <= 32'd0;
                            im2col_lane_pos <= 2'd0;
                        end else begin
                            im2col_pack_word <= pack_im2col_elem(im2col_pack_word,
                                                                  im2col_lane_pos,
                                                                  im2col_fp16_latch,
                                                                  16'd0);
                            im2col_lane_pos <= im2col_lane_pos + 1'b1;
                        end
                        im2col_k_pos <= im2col_k_pos + 1'b1;
                        if (im2col_elem_last)
                            load_state <= L_A_IM2COL_DONE;
                    end
                end else if (!m_axi_arvalid && !load_r_active && (!im2col_lane_last || !a_ppb_full)) begin
                    if (im2col_k_pos < 16'd5) $display("[DMA_IM2COL] read k=%0d addr=0x%08h", im2col_k_pos, im2col_aligned_addr);
                    load_addr_cnt <= im2col_aligned_addr;
                    load_arlen    <= 8'd0;
                    m_axi_arvalid <= 1'b1;
                end else if (m_axi_arvalid && m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;
                    load_r_active <= 1'b1;
                    m_axi_rready  <= 1'b1;
                end else if (load_r_active) begin
                    m_axi_rready <= 1'b1;
                    if (m_axi_rvalid && m_axi_rready && (!im2col_lane_last || !a_ppb_full)) begin
                        load_r_active <= 1'b0;
                        m_axi_rready  <= 1'b0;
                        if (im2col_lane_last) begin
                            a_im2col_wr_en <= 1'b1;
                            a_im2col_wr_data <= pack_im2col_elem(im2col_pack_word,
                                                                  im2col_lane_pos,
                                                                  im2col_fp16_latch,
                                                                  im2col_r_elem);
                            im2col_pack_word <= 32'd0;
                            im2col_lane_pos <= 2'd0;
                        end else begin
                            im2col_pack_word <= pack_im2col_elem(im2col_pack_word,
                                                                  im2col_lane_pos,
                                                                  im2col_fp16_latch,
                                                                  im2col_r_elem);
                            im2col_lane_pos <= im2col_lane_pos + 1'b1;
                        end
                        im2col_k_pos <= im2col_k_pos + 1'b1;
                        if (im2col_elem_last)
                            load_state <= L_A_IM2COL_DONE;
                    end
                end
            end

            L_A_IM2COL_DONE: begin
                a_done <= 1'b1;
                if (load_bias_after_data) begin
                    load_state <= L_BIAS;
                    load_addr_cnt <= bias_addr_latch;
                end else begin
                    load_state <= L_IDLE;
                end
            end

            // T6.3: one-beat 32-bit bias fetch. The controller issues this
            // alongside the current direct-scalar W/A load, and waits for
            // bias_done before allowing compute to start.
            L_BIAS: begin
                if (!m_axi_arvalid && !load_r_active) begin
                    m_axi_arvalid <= 1'b1;
                    load_arlen    <= 8'd0;
                end

                if (m_axi_arvalid && m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;
                    load_r_active <= 1'b1;
                    m_axi_rready  <= 1'b1;
                end else if (load_r_active) begin
                    m_axi_rready <= 1'b1;
                    if (m_axi_rvalid && m_axi_rready) begin
                        bias_data <= m_axi_rdata;
                        bias_done <= 1'b1;
                        load_r_active <= 1'b0;
                        m_axi_rready  <= 1'b0;
                        load_bias_after_data <= 1'b0;
                        load_state <= L_IDLE;
                    end
                end
            end

            // Reserved for a future interleaved read scheduler. Current T3.1 uses
            // sequential W then A bursts to preserve the existing controller contract.
            L_WA_READ: begin
                load_state <= L_IDLE;
                m_axi_arvalid <= 1'b0;
                m_axi_rready  <= 1'b0;
                load_r_active <= 1'b0;
            end

            default: load_state <= L_IDLE;
        endcase
    end
end

// ===========================================================================
// WB-FSM states
// ===========================================================================
localparam [1:0] WB_IDLE   = 2'd0;
localparam [1:0] WB_ACTIVE = 2'd1;

reg [1:0] wb_state;
reg [31:0] wb_addr_cnt;
reg [15:0] wb_byte_cnt;
reg [15:0] wb_bytes_done;
reg [7:0]  wb_burst_len;
reg        wb_aw_sent;
reg        wb_wait_b;
reg        wb_req_done_after_b;

reg r_pending;
reg [31:0] r_base_latch;
reg [15:0] r_len_latch;
reg r_pending_clr;

localparam [15:0] WRITE_BURST_MAX_BEATS =
    (BURST_MAX > 256) ? 16'd256 : BURST_MAX;

wire [15:0] wb_remaining_bytes =
    (r_len_latch > wb_bytes_done) ? (r_len_latch - wb_bytes_done) : 16'd0;

wire [15:0] wb_remaining_beats =
    (wb_remaining_bytes == 16'd0) ? 16'd0 :
    ((wb_remaining_bytes + AXI_DATA_BYTES - 1) >> AXI_BYTE_SHIFT);

wire [15:0] wb_bytes_to_4k =
    16'd4096 - {4'd0, wb_addr_cnt[11:0]};
wire [15:0] wb_beats_to_4k =
    (wb_bytes_to_4k >> AXI_BYTE_SHIFT);

wire [15:0] wb_burst_beats_cap0 =
    (wb_remaining_beats > WRITE_BURST_MAX_BEATS) ? WRITE_BURST_MAX_BEATS :
                                                   wb_remaining_beats;
wire [15:0] wb_burst_beats =
    (wb_burst_beats_cap0 > wb_beats_to_4k) ? wb_beats_to_4k :
                                             wb_burst_beats_cap0;

wire [7:0] wb_next_awlen =
    ((wb_burst_beats == 16'd0) ? 16'd1 : wb_burst_beats) - 1'b1;
wire [15:0] wb_cur_burst_bytes =
    ({8'd0, wb_burst_len} + 16'd1) << AXI_BYTE_SHIFT;
wire        wb_cur_burst_last_beat =
    (wb_byte_cnt + AXI_DATA_BYTES >= wb_cur_burst_bytes);
wire        wb_request_last_beat =
    (wb_bytes_done + AXI_DATA_BYTES >= r_len_latch);
wire        r_fifo_has_next_word =
    (r_fill > 1) || (r_fifo_wr_en && !r_fifo_full);

always @(posedge clk) begin
    if (!rst_n) begin
        r_pending <= 0; r_base_latch <= 0; r_len_latch <= 0;
    end else begin
        if (r_pending_clr) r_pending <= 0;
        // Allow r_start to (re-)arm at any time when WB-FSM is idle.
        // This also clears stale r_pending from previous run and updates
        // r_base_latch/r_len_latch to the current request's parameters.
        else if (r_start && wb_state == WB_IDLE) begin
            r_pending <= 1; r_base_latch <= r_base_addr; r_len_latch <= r_len_bytes;
        end
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        wb_state <= WB_IDLE; wb_addr_cnt <= 0; wb_byte_cnt <= 0;
        wb_bytes_done <= 0;
        r_done <= 0; wb_burst_len <= 0; wb_aw_sent <= 0;
        wb_wait_b <= 1'b0; wb_req_done_after_b <= 1'b0;
        r_pending_clr <= 0;
        m_axi_awvalid <= 0; m_axi_wvalid <= 0;
        m_axi_bready <= 0; m_axi_wstrb <= 0;
    end else begin
        r_pending_clr <= 0;
        r_done <= 0;   // default: single-cycle pulse (cleared every cycle)
        case (wb_state)
            WB_IDLE: begin
                m_axi_awvalid <= 0; m_axi_wvalid <= 0;
                m_axi_bready  <= 0; m_axi_wstrb   <= 0; wb_aw_sent <= 0;
                wb_wait_b <= 1'b0; wb_req_done_after_b <= 1'b0;
                if (r_pending && (r_len_latch == 16'd0)) begin
                    r_done        <= 1'b1;
                    r_pending_clr <= 1'b1;
                end else if (r_pending && !r_fifo_empty_n) begin
                    wb_state      <= WB_ACTIVE;
                    wb_addr_cnt   <= r_base_latch;
                    wb_byte_cnt   <= 16'd0;
                    wb_bytes_done <= 16'd0;
                    wb_burst_len  <= 8'd0;
                    r_pending_clr <= 1'b1;
                end
            end
            WB_ACTIVE: begin
                m_axi_wstrb <= {(DATA_W/8){1'b1}};

                if (wb_wait_b) begin
                    m_axi_awvalid <= 1'b0;
                    m_axi_wvalid  <= 1'b0;
                    m_axi_bready  <= 1'b1;
                    if (m_axi_bvalid && m_axi_bready) begin
                        m_axi_bready <= 1'b0;
                        wb_wait_b    <= 1'b0;
                        wb_aw_sent   <= 1'b0;
                        wb_byte_cnt  <= 16'd0;
                        wb_req_done_after_b <= 1'b0;
                        if (wb_req_done_after_b) begin
                            r_done   <= 1'b1;
                            wb_state <= WB_IDLE;
                        end
                    end
                end else if (!wb_aw_sent) begin
                    m_axi_bready <= 1'b0;
                    m_axi_wvalid <= 1'b0;
                    if (!m_axi_awvalid) begin
                        wb_burst_len  <= wb_next_awlen;
                        wb_byte_cnt   <= 16'd0;
                        m_axi_awvalid <= 1'b1;
                    end else if (m_axi_awready) begin
                        m_axi_awvalid <= 1'b0;
                        wb_aw_sent    <= 1'b1;
                    end
                end else begin
                    m_axi_awvalid <= 1'b0;
                    m_axi_bready  <= 1'b0;
                    if (!m_axi_wvalid && !r_fifo_empty_n)
                        m_axi_wvalid <= 1'b1;

                    if (m_axi_wvalid && m_axi_wready) begin
                        wb_byte_cnt   <= wb_byte_cnt + AXI_DATA_BYTES;
                        wb_bytes_done <= wb_bytes_done + AXI_DATA_BYTES;
                        if (wb_cur_burst_last_beat) begin
                            wb_addr_cnt <= wb_addr_cnt + wb_cur_burst_bytes;
                            wb_wait_b   <= 1'b1;
                            wb_req_done_after_b <= wb_request_last_beat;
                            m_axi_wvalid <= 1'b0;
                            m_axi_bready <= 1'b1;
                        end else begin
                            m_axi_wvalid <= r_fifo_has_next_word;
                        end
                    end
                end
            end
            default: wb_state <= WB_IDLE;
        endcase
    end
end

// ===========================================================================
// AXI outputs (combinational)
// ===========================================================================
assign m_axi_araddr  = load_addr_cnt;
assign m_axi_arlen   = load_arlen;
assign m_axi_arsize  = $clog2(DATA_W/8);
assign m_axi_arburst = 2'b01;

assign m_axi_awaddr  = wb_addr_cnt;
assign m_axi_awlen   = wb_burst_len;
assign m_axi_awsize  = $clog2(DATA_W/8);
assign m_axi_awburst = 2'b01;

assign m_axi_wdata   = r_fifo_dout_int;
assign m_axi_wlast   = (wb_state == WB_ACTIVE) && m_axi_wvalid && wb_cur_burst_last_beat;

// Debug: combined state for testbench compatibility
wire [5:0] dma_state = {wb_state, load_state};

endmodule
