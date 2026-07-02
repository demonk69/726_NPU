// =============================================================================
// Module  : pe_top
// Project : NPU_prj  -- Scalable PE Array
// Desc    : Processing Element (PE) with dual weight register for prefetch hiding.
//
//           Architecture:
//           - Dual Weight Registers: w_reg[0] (active) / w_reg[1] (prefetch)
//             swap_w signal atomically switches which register is used for compute.
//             While computing with w_reg[0], new weights can be loaded into w_reg[1],
//             then swapped in one cycle — hides the reload latency.
//           - 3-stage pipeline: Stage0=Input-Reg, Stage1=MUL, Stage2=ACC/Output
//
//           WS mode: load_w=1 latches w_in into the PREFETCH weight register.
//                    swap_w atomically swaps active/prefetch registers.
//           OS mode: weight streams through like activation.
//           INT8 mode supports packed 2-lane SIMD with DATA_W=16 and
//           packed 4-lane SIMD with DATA_W=32.
//           Legacy sign-extended scalar INT8 inputs are treated as 1-lane.
//           acc_init_en loads the internal accumulator from PSUM/OUT_BUF before
//           a K-split continuation tile starts.
// =============================================================================

`timescale 1ns/1ps

module pe_top #(
    parameter DATA_W = 16,   // max packed INT8 data width
    parameter ACC_W  = 32,   // accumulator width
    parameter INT8_SIMD_LANES = (DATA_W >= 32) ? 4 : 2,
    parameter FP16_ENABLE = 0,
    parameter INT8_SCALAR_SIGNEXT_COMPAT = 1
)(
    input  wire              clk,
    input  wire              rst_n,
    // mode control
    input  wire              mode,      // reserved; PE compute is INT8-only
    input  wire              stat_mode, // 0=Weight-Stationary, 1=Output-Stationary
    input  wire              en,        // pipeline enable
    input  wire              flush,     // flush accumulator
    input  wire              load_w,    // WS: latch w_in into PREFETCH weight reg this cycle
    input  wire              swap_w,    // WS: atomic swap of active↔prefetch weight regs
    input  wire              acc_init_en, // load internal accumulator from acc_init
    // data
    input  wire [DATA_W-1:0] w_in,      // weight input
    input  wire [DATA_W-1:0] a_in,      // activation input
    input  wire [ACC_W-1:0]  acc_in,    // incoming partial sum (from above PE)
    input  wire [ACC_W-1:0]  acc_init,  // initial accumulator value for K-split
    output reg  [ACC_W-1:0]  acc_out,   // accumulated result output
    output reg               valid_out
);

// ---------------------------------------------------------------------------
// Dual Weight Register Bank for Prefetch Hiding
//   w_reg[0]: ACTIVE   — currently used by the datapath for multiply
//   w_reg[1]: PREFETCH — being loaded; will become active on next swap_w
//   w_sel: selects which register is currently active (0 or 1)
// ---------------------------------------------------------------------------

reg [DATA_W-1:0] w_reg [0:1];       // dual weight registers
reg              w_sel;              // which register is active

always @(posedge clk) begin
    if (!rst_n) begin
        w_reg[0] <= {DATA_W{1'b0}};
        w_reg[1] <= {DATA_W{1'b0}};
        w_sel   <= 1'b0;
    end else begin
        // Atomic swap on swap_w pulse
        if (swap_w)
            w_sel <= ~w_sel;
        // Load only into the INACTIVE (prefetch) register. The active weight
        // remains stable for ongoing compute until swap_w selects the new bank.
        if (load_w)
            w_reg[~w_sel] <= w_in;
    end
end

// The active weight value presented to Stage-0
wire [DATA_W-1:0] active_weight = w_reg[w_sel];
wire unused_acc_in = |acc_in;
wire unused_fp16_cfg = mode | (FP16_ENABLE != 0);

// ---------------------------------------------------------------------------
// Stage-0 : Input register
// ---------------------------------------------------------------------------
reg  [DATA_W-1:0] s0_w, s0_a;
reg               s0_valid;
reg               s0_flush;
reg               s0_stat;

always @(posedge clk) begin
    if (!rst_n) begin
        s0_w     <= 0;
        s0_a     <= 0;
        s0_valid <= 0;
        s0_flush <= 0;
        s0_stat  <= 0;
    end else if (acc_init_en) begin
        s0_w     <= 0;
        s0_a     <= 0;
        s0_valid <= 0;
        s0_flush <= 0;
        s0_stat  <= stat_mode;
    end else if (en) begin
        s0_stat <= stat_mode;
        s0_flush <= flush;
        s0_valid <= 1'b1;

        // WS mode: use the active weight from the dual-reg bank. During a
        // load/swap compute beat, w_in is the just-prefetched value; subsequent
        // beats use the bank selected by swap_w.
        if (stat_mode == 1'b0) begin
            s0_w <= load_w ? w_in : active_weight;
            s0_a <= a_in;
        end
        // OS mode: weight and activation both stream in
        else begin
            s0_w <= w_in;
            s0_a <= a_in;
        end
    end else begin
        s0_valid <= 1'b0;
        s0_flush <= flush;
    end
end

// ---------------------------------------------------------------------------
// Stage-1: Multiply  (INT8)
// ---------------------------------------------------------------------------
reg  [ACC_W-1:0]  s1_mul;
reg               s1_valid;
reg               s1_flush;
reg               s1_stat;

// INT8: DATA_W carries up to four signed INT8 lanes in packed SIMD form.
// Some legacy scalar feeders sign-extend one INT8 value across DATA_W; keep
// that compatibility only where the caller explicitly leaves it enabled.
wire signed [7:0] int8_w0 = s0_w[7:0];
wire signed [7:0] int8_a0 = s0_a[7:0];
wire signed [7:0] int8_w1 = s0_w[15:8];
wire signed [7:0] int8_a1 = s0_a[15:8];
wire signed [7:0] int8_w2;
wire signed [7:0] int8_a2;
wire signed [7:0] int8_w3;
wire signed [7:0] int8_a3;

generate
    if (DATA_W >= 24) begin : gen_int8_lane2
        assign int8_w2 = s0_w[23:16];
        assign int8_a2 = s0_a[23:16];
    end else begin : gen_int8_lane2_zero
        assign int8_w2 = 8'sd0;
        assign int8_a2 = 8'sd0;
    end

    if (DATA_W >= 32) begin : gen_int8_lane3
        assign int8_w3 = s0_w[31:24];
        assign int8_a3 = s0_a[31:24];
    end else begin : gen_int8_lane3_zero
        assign int8_w3 = 8'sd0;
        assign int8_a3 = 8'sd0;
    end
endgenerate

(* use_dsp = "yes" *) wire signed [15:0] int8_mul0_16;
(* use_dsp = "yes" *) wire signed [15:0] int8_mul1_16;
(* use_dsp = "yes" *) wire signed [15:0] int8_mul2_16;
(* use_dsp = "yes" *) wire signed [15:0] int8_mul3_16;

assign int8_mul0_16 = $signed(int8_w0) * $signed(int8_a0);
assign int8_mul1_16 = $signed(int8_w1) * $signed(int8_a1);
assign int8_mul2_16 = $signed(int8_w2) * $signed(int8_a2);
assign int8_mul3_16 = $signed(int8_w3) * $signed(int8_a3);

wire int8_lane1_cfg_en = (INT8_SIMD_LANES >= 2) && (DATA_W >= 16);
wire int8_lane2_cfg_en = (INT8_SIMD_LANES >= 3) && (DATA_W >= 24);
wire int8_lane3_cfg_en = (INT8_SIMD_LANES >= 4) && (DATA_W >= 32);
wire int8_w_scalar_se  = (s0_w[DATA_W-1:8] == {(DATA_W-8){s0_w[7]}});
wire int8_a_scalar_se  = (s0_a[DATA_W-1:8] == {(DATA_W-8){s0_a[7]}});
wire int8_scalar_se    = int8_w_scalar_se && int8_a_scalar_se;
wire int8_scalar_compat = (INT8_SCALAR_SIGNEXT_COMPAT != 0) && int8_scalar_se;
wire int8_lane1_en     = int8_lane1_cfg_en && !int8_scalar_compat;
wire int8_lane2_en     = int8_lane2_cfg_en && !int8_scalar_compat;
wire int8_lane3_en     = int8_lane3_cfg_en && !int8_scalar_compat;

wire signed [17:0] int8_mul0_ext = {{2{int8_mul0_16[15]}}, int8_mul0_16};
wire signed [17:0] int8_mul1_ext = {{2{int8_mul1_16[15]}}, int8_mul1_16};
wire signed [17:0] int8_mul2_ext = {{2{int8_mul2_16[15]}}, int8_mul2_16};
wire signed [17:0] int8_mul3_ext = {{2{int8_mul3_16[15]}}, int8_mul3_16};
(* use_dsp = "yes" *) wire signed [17:0] int8_sum_18;

assign int8_sum_18 = int8_mul0_ext +
                     (int8_lane1_en ? int8_mul1_ext : 18'sd0) +
                     (int8_lane2_en ? int8_mul2_ext : 18'sd0) +
                     (int8_lane3_en ? int8_mul3_ext : 18'sd0);
wire [ACC_W-1:0] int8_prod = {{(ACC_W-18){int8_sum_18[17]}}, int8_sum_18};

// Register INT8 product
always @(posedge clk) begin
    if (!rst_n) begin
        s1_mul    <= 0;
        s1_valid  <= 0;
        s1_flush  <= 0;
        s1_stat   <= 0;
    end else if (acc_init_en) begin
        s1_mul    <= 0;
        s1_valid  <= 0;
        s1_flush  <= 0;
        s1_stat   <= stat_mode;
    end else if (s0_valid) begin
        s1_valid  <= s0_valid;
        s1_flush  <= s0_flush;
        s1_stat   <= s0_stat;
        s1_mul    <= int8_prod;
    end else begin
        s1_valid <= 0;
    end
end

// ---------------------------------------------------------------------------
// Stage-2: Accumulate / output
//
// WS (Weight-Stationary):
//   Internal ws_acc accumulates products across K beats.
//   flush=1 outputs the accumulated sum and clears ws_acc.
//
// OS (Output-Stationary):
//   Internal os_acc accumulates products. flush=1 outputs and clears.
// ---------------------------------------------------------------------------

reg [ACC_W-1:0]  os_acc;      // Output-Stationary accumulator
reg [ACC_W-1:0]  ws_acc;      // Weight-Stationary accumulator

always @(posedge clk) begin
    if (!rst_n) begin
        os_acc    <= 0;
        ws_acc    <= 0;
        acc_out   <= 0;
        valid_out <= 0;
    end else begin
        valid_out <= 0;   // default

        if (acc_init_en) begin
            if (stat_mode)
                os_acc <= acc_init;
            else
                ws_acc <= acc_init;
        end else if (s1_valid) begin
            if (s1_stat == 1'b0) begin
                // ----- Weight-Stationary -----
                if (s1_flush) begin
                    acc_out <= ws_acc;
                    ws_acc  <= 32'd0;
                    valid_out <= 1'b1;
                end else begin
                    ws_acc <= $signed(ws_acc) + $signed(s1_mul);
                    valid_out <= 1'b0;
                end
            end else begin
                // ----- Output-Stationary -----
                if (s1_flush) begin
                    acc_out <= $signed(os_acc) + $signed(s1_mul);
                    os_acc    <= 32'd0;
                    valid_out <= 1'b1;
                end else begin
                    os_acc <= $signed(os_acc) + $signed(s1_mul);
                end
            end
        end
    end
end

endmodule
