// =============================================================================
// Module  : npu_axi_lite
// Project : NPU_prj
// Desc    : AXI4-Lite slave with register file for NPU configuration.
//           Address map:
//             0x00  CTRL      - bit0=start, bit1=abort,
//                               [3:2]=data_mode (00=INT8, 10=FP16),
//                               [5:4]=stat_mode (00=WS,  01=OS),
//                               bit6=irq_clr (W1C: write 1 to clear IRQ),
//                               bit7=desc_mode, bit8=conv_im2col,
//                               bit9=bias_en, [11:10]=activation
//             0x04  STATUS    - bit0=busy, bit1=done, bit2=error
//             0x08  INT_EN    - bit0=done IRQ enable, bit1=error IRQ enable
//             0x0C  INT_CLR   - interrupt clear (write-1-to-clear, alt path)
//             0x10  M_DIM     - matrix M dimension
//             0x14  N_DIM     - matrix N dimension
//             0x18  K_DIM     - matrix K dimension
//             0x20  W_ADDR    - weight base address in DRAM
//             0x24  A_ADDR    - activation base address in DRAM
//             0x28  R_ADDR    - result base address in DRAM
//             0x30  ARR_CFG   - bit7=4x4 tile mode enable
//             0x34  CLK_DIV   - [2:0]=div_sel
//             0x38  CG_EN     - clock gating enable
//             0x3C  CFG_SHAPE - [1:0]=shape (00=4x4, 01=8x8, 10=16x16, 11=8x32)
//             0x40  DESC_BASE - descriptor list base address
//             0x44  DESC_COUNT- descriptor count / fetch limit
//             0x48  PERF_CYCLES      - snapshotted monitor cycles since clear
//             0x4C  PERF_RD_BEATS    - snapshotted AXI master read data beats
//             0x50  PERF_WR_BEATS    - snapshotted AXI master write data beats
//             0x54  PERF_RD_BYTES    - snapshotted AXI master read bytes
//             0x58  PERF_WR_BYTES    - snapshotted AXI master write bytes
//             0x5C  PERF_RD_BW       - derived read bytes/cycle x1000, optional
//             0x60  PERF_WR_BW       - derived write bytes/cycle x1000, optional
//             0x64  PERF_RD_UTIL     - derived read data-channel utilization, optional
//             0x68  PERF_WR_UTIL     - derived write data-channel utilization, optional
//             0x6C  PERF_RD_BURSTS   - snapshotted AXI master read burst count
//             0x70  PERF_WR_BURSTS   - snapshotted AXI master write burst count
//             0x74  ERR_STATUS        - W1C error status from controller
//             0x78  PERF_CTRL         - W1P bit0=clear counters, bit1=snapshot
//             0x80  CONV_IFM_SHAPE    - [15:0]=IH, [31:16]=IW
//             0x84  CONV_CHANNELS     - [15:0]=Cin, [31:16]=Batch
//             0x88  CONV_KERNEL       - [15:0]=KH, [31:16]=KW
//             0x8C  CONV_OUT_SHAPE    - [15:0]=OH, [31:16]=OW
//             0x90  CONV_STRIDE_PAD   - [7:0]=stride_h, [15:8]=stride_w,
//                                      - [23:16]=pad_h, [31:24]=pad_w
//             0x94  CONV_DILATION     - [7:0]=dilation_h, [15:8]=dilation_w
//             0x98  BIAS_ADDR         - 32-bit bias vector base address
//             0x9C  QUANT_CFG         - bit0=enable, bit1=round,
//                                      - [15:8]=right shift,
//                                      - [31:16]=signed scale
//             0xA0  PERF_MAC_OPS_LO   - snapshotted useful MAC operations[31:0]
//             0xA4  PERF_MAC_OPS_HI   - snapshotted useful MAC operations[63:32]
//             0xA8  PERF_OPS_LO       - snapshotted useful operations[31:0]
//             0xAC  PERF_OPS_HI       - snapshotted useful operations[63:32]
//             0xB0  PERF_BUSY_CYCLES  - snapshotted NPU busy cycles
//             0xB4  PERF_COMPUTE_CYCLES - snapshotted compute-active cycles
//             0xB8  PERF_DMA_CYCLES   - snapshotted busy cycles not in compute
//             0xBC  PERF_TOPS_X1E6    - derived TOPS * 1,000,000, optional
//             0xC0  PERF_COMPUTE_UTIL - derived compute utilization, optional
//             0xC4  PERF_E2E_UTIL     - derived end-to-end utilization, optional
//             0xC8  PERF_PEAK_OPS_CYC - snapshotted peak operations per cycle
//
//           IRQ Clear dual path:
//             Path A: write 0x0C (INT_CLR) bit0 = 1  → clears int_pending
//             Path B: write 0x00 (CTRL)    bit6 = 1  → also clears int_pending
//                     (bit6 is W1C; subsequent reads return 0)
//             CTRL bit7 is descriptor mode and is readable/writable.
// =============================================================================

`timescale 1ns/1ps

module npu_axi_lite (
    // AXI4-Lite slave
    input  wire                  aclk,
    input  wire                  aresetn,
    // AW channel
    input  wire [31:0]           awaddr,
    input  wire                  awvalid,
    output wire                  awready,
    // W channel
    input  wire [31:0]           wdata,
    input  wire [3:0]            wstrb,
    input  wire                  wvalid,
    output wire                  wready,
    // B channel
    output wire [1:0]            bresp,
    output wire                  bvalid,
    input  wire                  bready,
    // AR channel
    input  wire [31:0]           araddr,
    input  wire                  arvalid,
    output wire                  arready,
    // R channel
    output wire [31:0]           rdata,
    output wire [1:0]            rresp,
    output wire                  rvalid,
    input  wire                  rready,
    // Control interface to NPU controller
    output reg  [31:0]           ctrl_reg,
    output reg  [31:0]           m_dim,
    output reg  [31:0]           n_dim,
    output reg  [31:0]           k_dim,
    output reg  [31:0]           w_addr,
    output reg  [31:0]           a_addr,
    output reg  [31:0]           r_addr,
    output reg  [7:0]            arr_cfg,    // bit7=4x4 tile mode enable
    output reg  [2:0]            clk_div,
    output reg                   cg_en,
    output reg  [1:0]            cfg_shape, // shape select for reconfig array
    output reg  [31:0]           desc_base,
    output reg  [31:0]           desc_count,
    output reg  [31:0]           conv_ifm_shape,
    output reg  [31:0]           conv_channels,
    output reg  [31:0]           conv_kernel,
    output reg  [31:0]           conv_out_shape,
    output reg  [31:0]           conv_stride_pad,
    output reg  [31:0]           conv_dilation,
    output reg  [31:0]           bias_addr,
    output reg  [31:0]           quant_cfg,
    // Status from NPU controller
    input  wire                  status_busy,
    input  wire                  status_done,
    input  wire                  status_error,
    input  wire [31:0]           err_status,
    output reg                   err_clear,
    output reg  [31:0]           err_clear_mask,
    input  wire                  irq_flag,
    // Performance counters from axi_monitor
    input  wire [31:0]           perf_cycles,
    input  wire [31:0]           perf_m_axi_rd_beats,
    input  wire [31:0]           perf_m_axi_wr_beats,
    input  wire [31:0]           perf_m_axi_rd_bytes,
    input  wire [31:0]           perf_m_axi_wr_bytes,
    input  wire [31:0]           perf_m_axi_rd_bw,
    input  wire [31:0]           perf_m_axi_wr_bw,
    input  wire [31:0]           perf_m_axi_rd_util,
    input  wire [31:0]           perf_m_axi_wr_util,
    input  wire [31:0]           perf_m_axi_rd_bursts,
    input  wire [31:0]           perf_m_axi_wr_bursts,
    input  wire [63:0]           perf_mac_ops,
    input  wire [63:0]           perf_ops,
    input  wire [31:0]           perf_busy_cycles,
    input  wire [31:0]           perf_compute_cycles,
    input  wire [31:0]           perf_dma_cycles,
    input  wire [31:0]           perf_tops_x1e6,
    input  wire [31:0]           perf_compute_util_bp,
    input  wire [31:0]           perf_e2e_util_bp,
    input  wire [31:0]           perf_peak_ops_per_cycle,
    output reg                   perf_clear,
    // Interrupt output
    output wire                  npu_irq
);

// ---------------------------------------------------------------------------
// Internal registers
// ---------------------------------------------------------------------------
reg [31:0] int_en_reg;
reg [31:0] int_clr_reg;
reg        int_pending;
reg        irq_flag_d;
reg        status_error_d;
reg [31:0] perf_cycles_snap;
reg [31:0] perf_m_axi_rd_beats_snap;
reg [31:0] perf_m_axi_wr_beats_snap;
reg [31:0] perf_m_axi_rd_bytes_snap;
reg [31:0] perf_m_axi_wr_bytes_snap;
reg [31:0] perf_m_axi_rd_bw_snap;
reg [31:0] perf_m_axi_wr_bw_snap;
reg [31:0] perf_m_axi_rd_util_snap;
reg [31:0] perf_m_axi_wr_util_snap;
reg [31:0] perf_m_axi_rd_bursts_snap;
reg [31:0] perf_m_axi_wr_bursts_snap;
reg [63:0] perf_mac_ops_snap;
reg [63:0] perf_ops_snap;
reg [31:0] perf_busy_cycles_snap;
reg [31:0] perf_compute_cycles_snap;
reg [31:0] perf_dma_cycles_snap;
reg [31:0] perf_tops_x1e6_snap;
reg [31:0] perf_compute_util_bp_snap;
reg [31:0] perf_e2e_util_bp_snap;
reg [31:0] perf_peak_ops_per_cycle_snap;

// ---------------------------------------------------------------------------
// Write FSM
// ---------------------------------------------------------------------------
reg        aw_q, ar_q;
reg        bvalid_r;
reg [31:0] awaddr_q, araddr_q;
wire       wr_en = aw_q && wvalid && wready;
wire [3:0] w_strb = wstrb;

assign awready = !aw_q && !bvalid_r;
assign wready  = aw_q && !bvalid_r;
assign bvalid  = bvalid_r;
assign bresp   = 2'b00; // OKAY

always @(posedge aclk) begin
    if (!aresetn) aw_q <= 0;
    else if (wr_en) aw_q <= 0;
    else if (awvalid && awready) aw_q <= 1;
end

always @(posedge aclk) begin
    if (awvalid && awready) awaddr_q <= awaddr;
end

always @(posedge aclk) begin
    if (!aresetn)
        bvalid_r <= 1'b0;
    else if (wr_en)
        bvalid_r <= 1'b1;
    else if (bvalid_r && bready)
        bvalid_r <= 1'b0;
end

// Write data to register file
always @(posedge aclk) begin
    if (!aresetn) begin
        ctrl_reg  <= 0;
        int_en_reg <= 0;
        int_clr_reg <= 0;
        m_dim     <= 0;
        n_dim     <= 0;
        k_dim     <= 0;
        w_addr    <= 0;
        a_addr    <= 0;
        r_addr    <= 0;
        arr_cfg   <= 0;
        clk_div   <= 0;
        cg_en     <= 0;
        cfg_shape <= 2'b10;  // default: 16x16 mode
        desc_base <= 32'd0;
        desc_count <= 32'd0;
        conv_ifm_shape <= 32'd0;
        conv_channels <= 32'd0;
        conv_kernel <= 32'd0;
        conv_out_shape <= 32'd0;
        conv_stride_pad <= 32'd0;
        conv_dilation <= 32'd0;
        bias_addr <= 32'd0;
        quant_cfg <= 32'h0001_0000;
        err_clear <= 1'b0;
        err_clear_mask <= 32'd0;
        perf_clear <= 1'b0;
        perf_cycles_snap <= 32'd0;
        perf_m_axi_rd_beats_snap <= 32'd0;
        perf_m_axi_wr_beats_snap <= 32'd0;
        perf_m_axi_rd_bytes_snap <= 32'd0;
        perf_m_axi_wr_bytes_snap <= 32'd0;
        perf_m_axi_rd_bw_snap <= 32'd0;
        perf_m_axi_wr_bw_snap <= 32'd0;
        perf_m_axi_rd_util_snap <= 32'd0;
        perf_m_axi_wr_util_snap <= 32'd0;
        perf_m_axi_rd_bursts_snap <= 32'd0;
        perf_m_axi_wr_bursts_snap <= 32'd0;
        perf_mac_ops_snap <= 64'd0;
        perf_ops_snap <= 64'd0;
        perf_busy_cycles_snap <= 32'd0;
        perf_compute_cycles_snap <= 32'd0;
        perf_dma_cycles_snap <= 32'd0;
        perf_tops_x1e6_snap <= 32'd0;
        perf_compute_util_bp_snap <= 32'd0;
        perf_e2e_util_bp_snap <= 32'd0;
        perf_peak_ops_per_cycle_snap <= 32'd0;
    end else if (wr_en) begin
        err_clear <= 1'b0;
        err_clear_mask <= 32'd0;
        perf_clear <= 1'b0;
        case (awaddr_q)
            32'h00: begin
                if (w_strb[0]) begin
                    // bit6 (irq_clr) is W1C: always reads back 0
                    ctrl_reg <= {wdata[31:7], 1'b0, wdata[5:0]};
                end
            end
            32'h08: if (w_strb[0]) int_en_reg <= wdata;
            32'h0C: int_clr_reg  <= wdata;  // W1C handled below
            32'h10: if (w_strb[0]) m_dim   <= wdata;
            32'h14: if (w_strb[0]) n_dim   <= wdata;
            32'h18: if (w_strb[0]) k_dim   <= wdata;
            32'h20: if (w_strb[0]) w_addr  <= wdata;
            32'h24: if (w_strb[0]) a_addr  <= wdata;
            32'h28: if (w_strb[0]) r_addr  <= wdata;
            32'h30: if (w_strb[0]) arr_cfg <= wdata[7:0];
            32'h34: if (w_strb[0]) clk_div <= wdata[2:0];
            32'h38: if (w_strb[0]) cg_en   <= wdata[0];
            32'h3C: if (w_strb[0]) cfg_shape <= wdata[1:0];
            32'h40: if (w_strb[0]) desc_base <= wdata;
            32'h44: if (w_strb[0]) desc_count <= wdata;
            32'h74: begin
                err_clear      <= 1'b1;
                err_clear_mask <= wdata;
            end
            32'h78: begin
                if (w_strb[0]) begin
                    perf_clear <= wdata[0];
                    if (wdata[1]) begin
                        perf_cycles_snap <= perf_cycles;
                        perf_m_axi_rd_beats_snap <= perf_m_axi_rd_beats;
                        perf_m_axi_wr_beats_snap <= perf_m_axi_wr_beats;
                        perf_m_axi_rd_bytes_snap <= perf_m_axi_rd_bytes;
                        perf_m_axi_wr_bytes_snap <= perf_m_axi_wr_bytes;
                        perf_m_axi_rd_bw_snap <= perf_m_axi_rd_bw;
                        perf_m_axi_wr_bw_snap <= perf_m_axi_wr_bw;
                        perf_m_axi_rd_util_snap <= perf_m_axi_rd_util;
                        perf_m_axi_wr_util_snap <= perf_m_axi_wr_util;
                        perf_m_axi_rd_bursts_snap <= perf_m_axi_rd_bursts;
                        perf_m_axi_wr_bursts_snap <= perf_m_axi_wr_bursts;
                        perf_mac_ops_snap <= perf_mac_ops;
                        perf_ops_snap <= perf_ops;
                        perf_busy_cycles_snap <= perf_busy_cycles;
                        perf_compute_cycles_snap <= perf_compute_cycles;
                        perf_dma_cycles_snap <= perf_dma_cycles;
                        perf_tops_x1e6_snap <= perf_tops_x1e6;
                        perf_compute_util_bp_snap <= perf_compute_util_bp;
                        perf_e2e_util_bp_snap <= perf_e2e_util_bp;
                        perf_peak_ops_per_cycle_snap <= perf_peak_ops_per_cycle;
                    end
                end
            end
            32'h80: if (w_strb[0]) conv_ifm_shape <= wdata;
            32'h84: if (w_strb[0]) conv_channels <= wdata;
            32'h88: if (w_strb[0]) conv_kernel <= wdata;
            32'h8C: if (w_strb[0]) conv_out_shape <= wdata;
            32'h90: if (w_strb[0]) conv_stride_pad <= wdata;
            32'h94: if (w_strb[0]) conv_dilation <= wdata;
            32'h98: if (w_strb[0]) bias_addr <= wdata;
            32'h9C: if (w_strb[0]) quant_cfg <= wdata;
            default: ;
        endcase
    end else begin
        err_clear <= 1'b0;
        err_clear_mask <= 32'd0;
        perf_clear <= 1'b0;
    end
end

// ---------------------------------------------------------------------------
// Read FSM
// ---------------------------------------------------------------------------
assign arready = !ar_q;
assign rvalid  = ar_q;
assign rresp   = 2'b00;

always @(posedge aclk) begin
    if (!aresetn) ar_q <= 0;
    else if (arvalid && !ar_q) ar_q <= 1;
    else if (rvalid && rready) ar_q <= 0;
end

always @(posedge aclk) begin
    if (arvalid && !ar_q) araddr_q <= araddr;
end

reg [31:0] rdata_r;
always @(*) begin
    case (araddr_q)
        32'h00: rdata_r = ctrl_reg;
        32'h04: rdata_r = {29'b0, status_error, status_done, status_busy};
        32'h08: rdata_r = int_en_reg;
        32'h0C: rdata_r = int_pending;
        32'h10: rdata_r = m_dim;
        32'h14: rdata_r = n_dim;
        32'h18: rdata_r = k_dim;
        32'h20: rdata_r = w_addr;
        32'h24: rdata_r = a_addr;
        32'h28: rdata_r = r_addr;
        32'h30: rdata_r = {24'b0, arr_cfg};
        32'h34: rdata_r = {29'b0, clk_div};
        32'h38: rdata_r = {31'b0, cg_en};
        32'h3C: rdata_r = {30'b0, cfg_shape};
        32'h40: rdata_r = desc_base;
        32'h44: rdata_r = desc_count;
        32'h48: rdata_r = perf_cycles_snap;
        32'h4C: rdata_r = perf_m_axi_rd_beats_snap;
        32'h50: rdata_r = perf_m_axi_wr_beats_snap;
        32'h54: rdata_r = perf_m_axi_rd_bytes_snap;
        32'h58: rdata_r = perf_m_axi_wr_bytes_snap;
        32'h5C: rdata_r = perf_m_axi_rd_bw_snap;
        32'h60: rdata_r = perf_m_axi_wr_bw_snap;
        32'h64: rdata_r = perf_m_axi_rd_util_snap;
        32'h68: rdata_r = perf_m_axi_wr_util_snap;
        32'h6C: rdata_r = perf_m_axi_rd_bursts_snap;
        32'h70: rdata_r = perf_m_axi_wr_bursts_snap;
        32'h74: rdata_r = err_status;
        32'h78: rdata_r = 32'd0;
        32'h80: rdata_r = conv_ifm_shape;
        32'h84: rdata_r = conv_channels;
        32'h88: rdata_r = conv_kernel;
        32'h8C: rdata_r = conv_out_shape;
        32'h90: rdata_r = conv_stride_pad;
        32'h94: rdata_r = conv_dilation;
        32'h98: rdata_r = bias_addr;
        32'h9C: rdata_r = quant_cfg;
        32'hA0: rdata_r = perf_mac_ops_snap[31:0];
        32'hA4: rdata_r = perf_mac_ops_snap[63:32];
        32'hA8: rdata_r = perf_ops_snap[31:0];
        32'hAC: rdata_r = perf_ops_snap[63:32];
        32'hB0: rdata_r = perf_busy_cycles_snap;
        32'hB4: rdata_r = perf_compute_cycles_snap;
        32'hB8: rdata_r = perf_dma_cycles_snap;
        32'hBC: rdata_r = perf_tops_x1e6_snap;
        32'hC0: rdata_r = perf_compute_util_bp_snap;
        32'hC4: rdata_r = perf_e2e_util_bp_snap;
        32'hC8: rdata_r = perf_peak_ops_per_cycle_snap;
        default: rdata_r = 32'hDEADBEEF;
    endcase
end
assign rdata = rdata_r;

// ---------------------------------------------------------------------------
// Interrupt logic
// ---------------------------------------------------------------------------
// IRQ is set when irq_flag arrives and INT_EN[0] is asserted.
// IRQ can be cleared via TWO paths:
//   Path A: write INT_CLR (0x0C) bit0 = 1
//   Path B: write CTRL    (0x00) bit6 = 1  (W1C bit, see above)
always @(posedge aclk) begin
    if (!aresetn) begin
        int_pending <= 0;
        irq_flag_d <= 1'b0;
        status_error_d <= 1'b0;
    end else begin
        irq_flag_d <= irq_flag;
        status_error_d <= status_error;
        if (irq_flag && !irq_flag_d && int_en_reg[0])
            int_pending <= 1'b1;
        if (status_error && !status_error_d && int_en_reg[1])
            int_pending <= 1'b1;
        // Path A: INT_CLR register
        if (wr_en && awaddr_q == 32'h0C && wdata[0])
            int_pending <= 1'b0;
        // Path B: CTRL bit6 (irq_clr)
        if (wr_en && awaddr_q == 32'h00 && wdata[6])
            int_pending <= 1'b0;
    end
end
assign npu_irq = int_pending;

endmodule
