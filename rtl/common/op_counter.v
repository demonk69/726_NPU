// =============================================================================
// Module  : op_counter
// Project : NPU_prj
// Desc    : NPU operation counter & performance profiler.
//           Tracks MAC operations, PE utilization, FSM state timing,
//           and compute efficiency metrics.
// =============================================================================

`timescale 1ns/1ps

module op_counter #(
    parameter ROWS = 4,
    parameter COLS = 4
)(
    input  wire        clk,
    input  wire        rst_n,
    // NPU controller signals
    input  wire        pe_en,
    input  wire        pe_flush,
    input  wire        ctrl_busy,
    input  wire        ctrl_done,
    input  wire        dma_w_done,
    input  wire        dma_a_done,
    input  wire        dma_r_done,
    // PE array valid outputs
    input  wire [COLS-1:0] pe_valid,
    // Configuration
    input  wire [31:0] m_dim,
    input  wire [31:0] n_dim,
    input  wire [31:0] k_dim,
    // Statistics outputs
    output wire [63:0] total_mac_ops,     // total MAC operations executed
    output wire [31:0] total_pe_cycles,   // cycles pe_en was active
    output wire [31:0] total_busy_cycles, // cycles NPU was busy (total)
    output wire [31:0] total_compute_cycles, // cycles in compute state
    output wire [31:0] total_dma_cycles,  // cycles spent in DMA
    output wire [31:0] active_pe_cnt,     // number of PEs producing valid output
    output wire [31:0] peak_active_pe,    // peak PE utilization
    output wire [31:0] fsm_transitions,   // number of FSM state transitions
    // Performance metrics
    output wire [31:0] utilization_pct,   // PE utilization % (×100)
    output wire [31:0] mac_per_cycle,     // average MACs per cycle (×100)
    output wire [31:0] efficiency_pct     // overall efficiency % (×100)
);

// ---------------------------------------------------------------------------
// MAC operation counter
// Each PE producing valid output contributes 1 MAC per cycle
// (A MAC = 1 multiply + 1 accumulate)
// ---------------------------------------------------------------------------
reg [63:0] mac_ops_r;
reg [31:0] pe_cycles_r;

// Count active PEs (valid outputs)
reg [7:0] active_pe_r;
integer i;
always @(*) begin
    active_pe_r = 0;
    for (i = 0; i < COLS; i = i+1) begin
        active_pe_r = active_pe_r + pe_valid[i];
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        mac_ops_r   <= 0;
        pe_cycles_r <= 0;
    end else begin
        if (pe_en) begin
            pe_cycles_r <= pe_cycles_r + 1;
            mac_ops_r   <= mac_ops_r + {56'b0, active_pe_r};
        end
    end
end

// ---------------------------------------------------------------------------
// Busy / DMA cycle counters
// ---------------------------------------------------------------------------
reg [31:0] busy_cycles_r, compute_cycles_r, dma_cycles_r;
reg        busy_d1, busy_d2;
reg [3:0]  fsm_state_d1;

// Detect FSM transitions (simplified via busy edge detection)
reg        prev_busy;
always @(posedge clk) begin
    if (!rst_n) prev_busy <= 0;
    else        prev_busy <= ctrl_busy;
end
wire busy_rising = ctrl_busy && !prev_busy;

reg [31:0] fsm_trans_r;
always @(posedge clk) begin
    if (!rst_n) fsm_trans_r <= 0;
    else if (busy_rising) fsm_trans_r <= fsm_trans_r + 1;
end

always @(posedge clk) begin
    if (!rst_n) begin
        busy_cycles_r     <= 0;
        compute_cycles_r  <= 0;
        dma_cycles_r      <= 0;
    end else begin
        if (ctrl_busy)
            busy_cycles_r <= busy_cycles_r + 1;
        if (pe_en && !pe_flush)
            compute_cycles_r <= compute_cycles_r + 1;
        // DMA cycles: approximate as busy but not computing
        if (ctrl_busy && !pe_en)
            dma_cycles_r <= dma_cycles_r + 1;
    end
end

// ---------------------------------------------------------------------------
// Peak PE utilization tracking
// ---------------------------------------------------------------------------
reg [7:0]  peak_pe_r;
always @(posedge clk) begin
    if (!rst_n)
        peak_pe_r <= 0;
    else if (active_pe_r > peak_pe_r)
        peak_pe_r <= active_pe_r;
end

// ---------------------------------------------------------------------------
// Performance metrics (×100 for fixed-point display)
// ---------------------------------------------------------------------------
wire [31:0] max_possible_mac = ROWS * COLS; // per cycle

assign utilization_pct = (pe_cycles_r > 0) ?
    (active_pe_r * 10000 / (pe_cycles_r * max_possible_mac)) : 0;
    // Note: this gives average over all PE-enable cycles
    // Recompute as running average:

reg [31:0] util_accum;
always @(posedge clk) begin
    if (!rst_n)
        util_accum <= 0;
    else if (pe_en)
        util_accum <= util_accum + (active_pe_r * 10000 / max_possible_mac);
end
assign utilization_pct = (pe_cycles_r > 0) ? (util_accum / pe_cycles_r) : 0;

assign mac_per_cycle = (pe_cycles_r > 0) ?
    ((mac_ops_r[31:0] * 100) / pe_cycles_r) : 0;

// Overall efficiency: useful compute time / total busy time
assign efficiency_pct = (busy_cycles_r > 0) ?
    ((compute_cycles_r * 10000) / busy_cycles_r) : 0;

// ---------------------------------------------------------------------------
// Output assignments
// ---------------------------------------------------------------------------
assign total_mac_ops      = mac_ops_r;
assign total_pe_cycles    = pe_cycles_r;
assign total_busy_cycles  = busy_cycles_r;
assign total_compute_cycles = compute_cycles_r;
assign total_dma_cycles   = dma_cycles_r;
assign active_pe_cnt      = {24'b0, active_pe_r};
assign peak_active_pe     = {24'b0, peak_pe_r};
assign fsm_transitions    = fsm_trans_r;

endmodule
