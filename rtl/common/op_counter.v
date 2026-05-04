// =============================================================================
// Module  : op_counter
// Project : NPU_prj
// Desc    : NPU operation counter and fixed-point performance profiler.
//
// Units:
//   - total_mac_ops counts useful GEMM MACs accumulated on ctrl_done.
//   - total_ops counts one MAC as two operations.
//   - tops_x1e6 is TOPS * 1,000,000. At 0.512 TOPS this reports 512000.
//   - *_util_bp values are basis points: 10000 = 100.00%.
// =============================================================================

`timescale 1ns/1ps

module op_counter #(
    parameter ROWS = 4,
    parameter COLS = 4,
    parameter FREQ_MHZ = 500
)(
    input  wire              clk,
    input  wire              rst_n,
    // Legacy controller/PE signals
    input  wire              pe_en,
    input  wire              pe_flush,
    input  wire              ctrl_busy,
    input  wire              ctrl_done,
    input  wire              dma_w_done,
    input  wire              dma_a_done,
    input  wire              dma_r_done,
    input  wire [COLS-1:0]   pe_valid,
    input  wire [31:0]       m_dim,
    input  wire [31:0]       n_dim,
    input  wire [31:0]       k_dim,
    // T7.5 performance context
    input  wire              compute_valid,
    input  wire [4:0]        active_rows,
    input  wire [5:0]        active_cols,
    input  wire [3:0]        simd_lanes,
    // Legacy statistics outputs
    output wire [63:0]       total_mac_ops,
    output wire [31:0]       total_pe_cycles,
    output wire [31:0]       total_busy_cycles,
    output wire [31:0]       total_compute_cycles,
    output wire [31:0]       total_dma_cycles,
    output wire [31:0]       active_pe_cnt,
    output wire [31:0]       peak_active_pe,
    output wire [31:0]       fsm_transitions,
    output wire [31:0]       utilization_pct,
    output wire [31:0]       mac_per_cycle,
    output wire [31:0]       efficiency_pct,
    // T7.5 derived outputs
    output wire [63:0]       total_ops,
    output wire [31:0]       peak_ops_per_cycle,
    output wire [31:0]       tops_x1e6,
    output wire [31:0]       compute_util_bp,
    output wire [31:0]       e2e_util_bp
);

function [31:0] sat32_from_96;
    input [95:0] value;
    begin
        sat32_from_96 = (|value[95:32]) ? 32'hFFFF_FFFF : value[31:0];
    end
endfunction

function [31:0] sat32_from_64;
    input [63:0] value;
    begin
        sat32_from_64 = (|value[63:32]) ? 32'hFFFF_FFFF : value[31:0];
    end
endfunction

function [7:0] popcount_valid;
    input [COLS-1:0] value;
    integer i;
    begin
        popcount_valid = 8'd0;
        for (i = 0; i < COLS; i = i + 1)
            popcount_valid = popcount_valid + value[i];
    end
endfunction

wire done_rise;
reg  ctrl_done_d;
reg  ctrl_busy_d;

always @(posedge clk) begin
    if (!rst_n) begin
        ctrl_done_d <= 1'b0;
        ctrl_busy_d <= 1'b0;
    end else begin
        ctrl_done_d <= ctrl_done;
        ctrl_busy_d <= ctrl_busy;
    end
end

assign done_rise = ctrl_done && !ctrl_done_d;
wire busy_rise = ctrl_busy && !ctrl_busy_d;

localparam [4:0] ROWS_DEFAULT = ROWS;
localparam [5:0] COLS_DEFAULT = COLS;

wire [4:0] rows_eff = (active_rows == 5'd0) ? ROWS_DEFAULT : active_rows;
wire [5:0] cols_eff = (active_cols == 6'd0) ? COLS_DEFAULT : active_cols;
wire [3:0] simd_eff = (simd_lanes == 4'd0) ? 4'd1 : simd_lanes;

wire [31:0] peak_ops_cfg =
    {27'd0, rows_eff} * {26'd0, cols_eff} * {28'd0, simd_eff} * 32'd2;

wire [127:0] m_ext = {96'd0, m_dim};
wire [127:0] n_ext = {96'd0, n_dim};
wire [127:0] k_ext = {96'd0, k_dim};
wire [127:0] mn_wide = m_ext * n_ext;
wire [127:0] mac_wide = mn_wide * k_ext;
wire [63:0] task_mac_ops = (|mac_wide[127:64]) ? 64'hFFFF_FFFF_FFFF_FFFF
                                                : mac_wide[63:0];

reg [63:0] mac_ops_r;
reg [31:0] pe_cycles_r;
reg [31:0] busy_cycles_r;
reg [31:0] compute_cycles_r;
reg [31:0] dma_cycles_r;
reg [31:0] fsm_trans_r;
reg [31:0] peak_ops_per_cycle_r;
reg [7:0]  peak_pe_r;

wire [7:0] active_pe_now = popcount_valid(pe_valid);

always @(posedge clk) begin
    if (!rst_n) begin
        mac_ops_r            <= 64'd0;
        pe_cycles_r          <= 32'd0;
        busy_cycles_r        <= 32'd0;
        compute_cycles_r     <= 32'd0;
        dma_cycles_r         <= 32'd0;
        fsm_trans_r          <= 32'd0;
        peak_ops_per_cycle_r <= ROWS * COLS * 2;
        peak_pe_r            <= 8'd0;
    end else begin
        if (done_rise)
            mac_ops_r <= mac_ops_r + task_mac_ops;

        if (pe_en && !pe_flush)
            pe_cycles_r <= pe_cycles_r + 32'd1;

        if (ctrl_busy)
            busy_cycles_r <= busy_cycles_r + 32'd1;

        if (compute_valid)
            compute_cycles_r <= compute_cycles_r + 32'd1;

        if (ctrl_busy && !compute_valid)
            dma_cycles_r <= dma_cycles_r + 32'd1;

        if (busy_rise) begin
            fsm_trans_r <= fsm_trans_r + 32'd1;
            peak_ops_per_cycle_r <= (peak_ops_cfg == 32'd0) ? 32'd2
                                                             : peak_ops_cfg;
        end

        if (active_pe_now > peak_pe_r)
            peak_pe_r <= active_pe_now;
    end
end

wire [63:0] ops_total_w = mac_ops_r << 1;
wire [95:0] tops_num = {32'd0, ops_total_w} * FREQ_MHZ;
wire [95:0] util_num = {32'd0, ops_total_w} * 32'd10000;
wire [63:0] compute_peak_ops =
    {32'd0, compute_cycles_r} * {32'd0, peak_ops_per_cycle_r};
wire [63:0] busy_peak_ops =
    {32'd0, busy_cycles_r} * {32'd0, peak_ops_per_cycle_r};
wire [95:0] tops_w = (busy_cycles_r != 32'd0) ? (tops_num / busy_cycles_r)
                                              : 96'd0;
wire [95:0] compute_util_w = (compute_peak_ops != 64'd0) ? (util_num / compute_peak_ops)
                                                         : 96'd0;
wire [95:0] e2e_util_w = (busy_peak_ops != 64'd0) ? (util_num / busy_peak_ops)
                                                 : 96'd0;
wire [63:0] mac_per_cycle_w =
    (busy_cycles_r != 32'd0) ? (mac_ops_r * 64'd100 / {32'd0, busy_cycles_r})
                             : 64'd0;

assign total_mac_ops       = mac_ops_r;
assign total_ops           = ops_total_w;
assign total_pe_cycles     = pe_cycles_r;
assign total_busy_cycles   = busy_cycles_r;
assign total_compute_cycles = compute_cycles_r;
assign total_dma_cycles    = dma_cycles_r;
assign active_pe_cnt       = {24'd0, active_pe_now};
assign peak_active_pe      = {24'd0, peak_pe_r};
assign fsm_transitions     = fsm_trans_r;
assign peak_ops_per_cycle  = peak_ops_per_cycle_r;
assign tops_x1e6           = sat32_from_96(tops_w);
assign compute_util_bp     = sat32_from_96(compute_util_w);
assign e2e_util_bp         = sat32_from_96(e2e_util_w);

// Legacy aliases retain the historical fixed-point names.
assign utilization_pct = compute_util_bp;
assign mac_per_cycle   = sat32_from_64(mac_per_cycle_w);
assign efficiency_pct  = e2e_util_bp;

endmodule
