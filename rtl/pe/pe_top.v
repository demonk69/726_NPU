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
//           Optional FP16/INT8 support with mixed-precision FP32 accumulation.
//           INT8 mode supports packed 2/4/8-lane SIMD with DATA_W=16/32/64.
//           Legacy sign-extended scalar INT8 inputs are treated as 1-lane.
//           acc_init_en loads the OS internal accumulator from PSUM/OUT_BUF
//           before a K-split continuation tile starts.
// =============================================================================

`timescale 1ns/1ps

module pe_top #(
    parameter DATA_W = 16,   // max data width (FP16)
    parameter ACC_W  = 32,   // accumulator width
    parameter INT8_SIMD_LANES = (DATA_W >= 64) ? 8 : ((DATA_W >= 32) ? 4 : 2),
    parameter FP16_ENABLE = 0,
    parameter INT8_SCALAR_SIGNEXT_COMPAT = 1
)(
    input  wire              clk,
    input  wire              rst_n,
    // mode control
    input  wire              mode,      // 0=INT8, 1=FP16
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
        // Atomic swap on swap_w pulse.
        if (swap_w)
            w_sel <= ~w_sel;
        // Load only into the register that is inactive after this edge.  When
        // load_w and swap_w coincide, the old active register becomes prefetch.
        if (load_w)
            w_reg[swap_w ? w_sel : ~w_sel] <= w_in;
    end
end

// The active weight value presented to Stage-0
wire [DATA_W-1:0] active_weight = w_reg[swap_w ? ~w_sel : w_sel];
wire pe_fp16_mode = (FP16_ENABLE != 0) && mode;

// ---------------------------------------------------------------------------
// Stage-0 : Input register
// ---------------------------------------------------------------------------
reg  [DATA_W-1:0] s0_w, s0_a;
reg  [ACC_W-1:0]  s0_acc;
reg               s0_valid;
reg               s0_flush;
reg               s0_mode;
reg               s0_stat;

always @(posedge clk) begin
    if (!rst_n) begin
        s0_w     <= 0;
        s0_a     <= 0;
        s0_acc   <= 0;
        s0_valid <= 0;
        s0_flush <= 0;
        s0_mode  <= 0;
        s0_stat  <= 0;
    end else if (acc_init_en) begin
        s0_w     <= 0;
        s0_a     <= 0;
        s0_acc   <= 0;
        s0_valid <= 0;
        s0_flush <= 0;
        s0_mode  <= pe_fp16_mode;
        s0_stat  <= stat_mode;
    end else if (en) begin
        s0_mode  <= pe_fp16_mode;
        s0_stat <= stat_mode;
        s0_flush <= flush;
        s0_valid <= 1'b1;
        s0_acc   <= acc_in;

        // WS mode: use only the active weight selected by swap_w.
        if (stat_mode == 1'b0) begin
            s0_w <= active_weight;
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
// Stage-1: Multiply  (INT8 or FP16)
// ---------------------------------------------------------------------------
reg  [ACC_W-1:0]  s1_mul;
reg  [ACC_W-1:0]  s1_acc;
reg               s1_valid;
reg               s1_flush;
reg               s1_stat;
reg               s1_mode;

// INT8: DATA_W carries up to eight signed INT8 lanes in packed SIMD form.
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
wire signed [7:0] int8_w4;
wire signed [7:0] int8_a4;
wire signed [7:0] int8_w5;
wire signed [7:0] int8_a5;
wire signed [7:0] int8_w6;
wire signed [7:0] int8_a6;
wire signed [7:0] int8_w7;
wire signed [7:0] int8_a7;

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

    if (DATA_W >= 40) begin : gen_int8_lane4
        assign int8_w4 = s0_w[39:32];
        assign int8_a4 = s0_a[39:32];
    end else begin : gen_int8_lane4_zero
        assign int8_w4 = 8'sd0;
        assign int8_a4 = 8'sd0;
    end

    if (DATA_W >= 48) begin : gen_int8_lane5
        assign int8_w5 = s0_w[47:40];
        assign int8_a5 = s0_a[47:40];
    end else begin : gen_int8_lane5_zero
        assign int8_w5 = 8'sd0;
        assign int8_a5 = 8'sd0;
    end

    if (DATA_W >= 56) begin : gen_int8_lane6
        assign int8_w6 = s0_w[55:48];
        assign int8_a6 = s0_a[55:48];
    end else begin : gen_int8_lane6_zero
        assign int8_w6 = 8'sd0;
        assign int8_a6 = 8'sd0;
    end

    if (DATA_W >= 64) begin : gen_int8_lane7
        assign int8_w7 = s0_w[63:56];
        assign int8_a7 = s0_a[63:56];
    end else begin : gen_int8_lane7_zero
        assign int8_w7 = 8'sd0;
        assign int8_a7 = 8'sd0;
    end
endgenerate

(* use_dsp = "yes" *) wire signed [15:0] int8_mul0_16;
(* use_dsp = "yes" *) wire signed [15:0] int8_mul1_16;
(* use_dsp = "yes" *) wire signed [15:0] int8_mul2_16;
(* use_dsp = "yes" *) wire signed [15:0] int8_mul3_16;
(* use_dsp = "yes" *) wire signed [15:0] int8_mul4_16;
(* use_dsp = "yes" *) wire signed [15:0] int8_mul5_16;
(* use_dsp = "yes" *) wire signed [15:0] int8_mul6_16;
(* use_dsp = "yes" *) wire signed [15:0] int8_mul7_16;

assign int8_mul0_16 = $signed(int8_w0) * $signed(int8_a0);
assign int8_mul1_16 = $signed(int8_w1) * $signed(int8_a1);
assign int8_mul2_16 = $signed(int8_w2) * $signed(int8_a2);
assign int8_mul3_16 = $signed(int8_w3) * $signed(int8_a3);
assign int8_mul4_16 = $signed(int8_w4) * $signed(int8_a4);
assign int8_mul5_16 = $signed(int8_w5) * $signed(int8_a5);
assign int8_mul6_16 = $signed(int8_w6) * $signed(int8_a6);
assign int8_mul7_16 = $signed(int8_w7) * $signed(int8_a7);

wire int8_lane1_cfg_en = (INT8_SIMD_LANES >= 2) && (DATA_W >= 16);
wire int8_lane2_cfg_en = (INT8_SIMD_LANES >= 3) && (DATA_W >= 24);
wire int8_lane3_cfg_en = (INT8_SIMD_LANES >= 4) && (DATA_W >= 32);
wire int8_lane4_cfg_en = (INT8_SIMD_LANES >= 5) && (DATA_W >= 40);
wire int8_lane5_cfg_en = (INT8_SIMD_LANES >= 6) && (DATA_W >= 48);
wire int8_lane6_cfg_en = (INT8_SIMD_LANES >= 7) && (DATA_W >= 56);
wire int8_lane7_cfg_en = (INT8_SIMD_LANES >= 8) && (DATA_W >= 64);
wire int8_w_scalar_se  = (s0_w[DATA_W-1:8] == {(DATA_W-8){s0_w[7]}});
wire int8_a_scalar_se  = (s0_a[DATA_W-1:8] == {(DATA_W-8){s0_a[7]}});
wire int8_scalar_se    = int8_w_scalar_se && int8_a_scalar_se;
wire int8_scalar_compat = (INT8_SCALAR_SIGNEXT_COMPAT != 0) && int8_scalar_se;
wire int8_lane1_en     = int8_lane1_cfg_en && !int8_scalar_compat;
wire int8_lane2_en     = int8_lane2_cfg_en && !int8_scalar_compat;
wire int8_lane3_en     = int8_lane3_cfg_en && !int8_scalar_compat;
wire int8_lane4_en     = int8_lane4_cfg_en && !int8_scalar_compat;
wire int8_lane5_en     = int8_lane5_cfg_en && !int8_scalar_compat;
wire int8_lane6_en     = int8_lane6_cfg_en && !int8_scalar_compat;
wire int8_lane7_en     = int8_lane7_cfg_en && !int8_scalar_compat;

wire signed [18:0] int8_mul0_ext = {{3{int8_mul0_16[15]}}, int8_mul0_16};
wire signed [18:0] int8_mul1_ext = {{3{int8_mul1_16[15]}}, int8_mul1_16};
wire signed [18:0] int8_mul2_ext = {{3{int8_mul2_16[15]}}, int8_mul2_16};
wire signed [18:0] int8_mul3_ext = {{3{int8_mul3_16[15]}}, int8_mul3_16};
wire signed [18:0] int8_mul4_ext = {{3{int8_mul4_16[15]}}, int8_mul4_16};
wire signed [18:0] int8_mul5_ext = {{3{int8_mul5_16[15]}}, int8_mul5_16};
wire signed [18:0] int8_mul6_ext = {{3{int8_mul6_16[15]}}, int8_mul6_16};
wire signed [18:0] int8_mul7_ext = {{3{int8_mul7_16[15]}}, int8_mul7_16};
(* use_dsp = "yes" *) wire signed [18:0] int8_sum_19;

assign int8_sum_19 = int8_mul0_ext +
                     (int8_lane1_en ? int8_mul1_ext : 19'sd0) +
                     (int8_lane2_en ? int8_mul2_ext : 19'sd0) +
                     (int8_lane3_en ? int8_mul3_ext : 19'sd0) +
                     (int8_lane4_en ? int8_mul4_ext : 19'sd0) +
                     (int8_lane5_en ? int8_mul5_ext : 19'sd0) +
                     (int8_lane6_en ? int8_mul6_ext : 19'sd0) +
                     (int8_lane7_en ? int8_mul7_ext : 19'sd0);
wire [ACC_W-1:0] int8_prod = {{(ACC_W-19){int8_sum_19[18]}}, int8_sum_19};

// FP16: instantiated only when enabled so INT8-only builds can drop it.
wire [ACC_W-1:0] fp16_mul_out;
generate
    if (FP16_ENABLE != 0) begin : gen_fp16_mul
        fp16_mul u_fp16_mul (
            .clk     (clk),
            .rst_n   (rst_n),
            .en      (en),
            .a       (s0_w[15:0]),
            .b       (s0_a[15:0]),
            .result  (fp16_mul_out)
        );
    end else begin : gen_no_fp16_mul
        assign fp16_mul_out = {ACC_W{1'b0}};
    end
endgenerate

// Mux INT8 / FP16 + register
always @(posedge clk) begin
    if (!rst_n) begin
        s1_mul    <= 0;
        s1_acc    <= 0;
        s1_valid  <= 0;
        s1_flush  <= 0;
        s1_stat   <= 0;
        s1_mode   <= 0;
    end else if (acc_init_en) begin
        s1_mul    <= 0;
        s1_acc    <= 0;
        s1_valid  <= 0;
        s1_flush  <= 0;
        s1_stat   <= stat_mode;
        s1_mode   <= pe_fp16_mode;
    end else if (s0_valid) begin
        s1_valid  <= s0_valid;
        s1_flush  <= s0_flush;
        s1_stat   <= s0_stat;
        s1_mode   <= s0_mode;
        s1_acc    <= s0_acc;
        s1_mul    <= s0_mode ? fp16_mul_out : int8_prod;
    end else begin
        s1_valid <= 0;
    end
end

// ---------------------------------------------------------------------------
// Stage-2: Accumulate / output
//
// WS (Weight-Stationary):
//   Weight is held locally.  Each valid beat emits acc_in + product, allowing
//   partial sums to move through an external chain.
//
// OS (Output-Stationary):
//   Internal os_acc accumulates products. flush=1 outputs and clears.
// ---------------------------------------------------------------------------

reg [ACC_W-1:0]  os_acc;      // Output-Stationary accumulator

// --- FP16 to FP32 conversion ---
function [31:0] fp16_to_fp32;
    input [15:0] fp16;
    reg [4:0] exp16;
    reg is_zero, is_inf, is_nan;
    begin
        exp16  = fp16[14:10];
        is_zero = (exp16 == 5'd0) && (fp16[9:0] == 10'd0);
        is_inf  = (exp16 == 5'h1F) && (fp16[9:0] == 10'd0);
        is_nan  = (exp16 == 5'h1F) && (fp16[9:0] != 10'd0);
        if (is_zero)
            fp16_to_fp32 = {fp16[15], 31'd0};
        else if (is_inf)
            fp16_to_fp32 = {fp16[15], 8'hFF, 23'd0};
        else if (is_nan)
            fp16_to_fp32 = {1'b1, 8'hFF, 23'h400000};
        else if (exp16 == 5'd0)
            fp16_to_fp32 = {fp16[15], 8'd113, fp16[9:0], 13'd0};
        else
            fp16_to_fp32 = {fp16[15],
                            8'd127 + {3'b0, exp16} - 8'd15,
                            fp16[9:0], 13'd0};
    end
endfunction

// FP32 adder for FP16 accumulation. Removed from INT8-only elaboration.
wire [31:0] fp32_sum;
generate
    if (FP16_ENABLE != 0) begin : gen_fp16_accum
        wire [31:0] fp32_a = s1_stat ? os_acc : s1_acc;
        wire [31:0] fp32_b = fp16_to_fp32(s1_mul[15:0]);

        fp32_add u_fp32_add (
            .a      (fp32_a),
            .b      (fp32_b),
            .result (fp32_sum)
        );
    end else begin : gen_no_fp16_accum
        assign fp32_sum = 32'd0;
    end
endgenerate

always @(posedge clk) begin
    if (!rst_n) begin
        os_acc    <= 0;
        acc_out   <= 0;
        valid_out <= 0;
    end else begin
        valid_out <= 0;   // default

        if (acc_init_en) begin
            if (stat_mode)
                os_acc <= acc_init;
        end else if (s1_valid) begin
            if (s1_stat == 1'b0) begin
                // ----- Weight-Stationary -----
                if (s1_mode) begin
                    acc_out <= fp32_sum;
                end else begin
                    acc_out <= $signed(s1_acc) + $signed(s1_mul);
                end
                valid_out <= 1'b1;
            end else begin
                // ----- Output-Stationary -----
                if (s1_flush) begin
                    if (s1_mode) begin
                        acc_out <= fp32_sum;
                    end else begin
                        acc_out <= $signed(os_acc) + $signed(s1_mul);
                    end
                    os_acc    <= 32'd0;
                    valid_out <= 1'b1;
                end else begin
                    if (s1_mode) begin
                        os_acc <= fp32_sum;
                    end else begin
                        os_acc <= $signed(os_acc) + $signed(s1_mul);
                    end
                end
            end
        end
    end
end

endmodule
