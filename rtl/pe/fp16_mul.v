// =============================================================================
// Module  : fp16_mul
// Project : NPU_prj
// Desc    : IEEE 754 half-precision (FP16) multiplier, 1-cycle registered output.
//           FP16: 1 sign + 5 exponent + 10 mantissa  (bias=15)
//           Output is expanded to ACC_W=32 bit fixed-point representation
//           (sign-extended FP32-lite result) for easy accumulation.
//
//           Pipeline: Combinational multiply in Stage-1; result registered here.
// =============================================================================

`timescale 1ns/1ps

module fp16_mul #(
    parameter ACC_W = 32
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire [15:0] a,       // FP16 operand A
    input  wire [15:0] b,       // FP16 operand B
    output reg  [ACC_W-1:0] result  // product (FP32-format, sign-extended)
);

// ---------------------------------------------------------------------------
// Unpack FP16
// ---------------------------------------------------------------------------
wire        sa = a[15];
wire [4:0]  ea = a[14:10];
wire [9:0]  ma = a[9:0];

wire        sb = b[15];
wire [4:0]  eb = b[14:10];
wire [9:0]  mb = b[9:0];

// ---------------------------------------------------------------------------
// Special case flags
// ---------------------------------------------------------------------------
wire a_zero    = (ea == 5'd0)  && (ma == 10'd0);
wire b_zero    = (eb == 5'd0)  && (mb == 10'd0);
wire a_inf     = (ea == 5'h1F) && (ma == 10'd0);
wire b_inf     = (eb == 5'h1F) && (mb == 10'd0);
wire a_nan     = (ea == 5'h1F) && (ma != 10'd0);
wire b_nan     = (eb == 5'h1F) && (mb != 10'd0);
wire is_nan    = a_nan | b_nan | (a_zero & b_inf) | (b_zero & a_inf);
wire is_inf    = (a_inf | b_inf) & !is_nan;
wire is_zero   = (a_zero | b_zero) & !is_nan;

// ---------------------------------------------------------------------------
// Normal multiply
// ---------------------------------------------------------------------------
wire        sign_r  = sa ^ sb;
wire [5:0]  exp_sum = {1'b0, ea} + {1'b0, eb}; // raw sum (max 62)
// Biased exponent: subtract FP16 bias-15, leave in FP32 domain (bias-127)
// For now we keep FP16 style and expand to a 16-bit fixed-point result.
// Simplified: treat output as Q1.15 scaled integer for accumulation.

// Leading 1 + mantissa
wire [10:0] mant_a = {1'b1, ma};
wire [10:0] mant_b = {1'b1, mb};
wire [21:0] mant_prod = mant_a * mant_b; // 11x11 = 22 bit

// Normalise: if bit[21] set, shift right and increment exp
wire        norm_shift = mant_prod[21];
wire [10:0] mant_norm  = norm_shift ? mant_prod[21:11] : mant_prod[20:10];
wire [5:0]  exp_raw    = exp_sum - 6'd15 + {5'b0, norm_shift};

// Clamp exponent
wire        exp_ovf    = (exp_raw[5] == 0) && (exp_raw[4:0] > 5'd30);
wire        exp_unf    = exp_raw[5];  // negative after bias removal
wire [4:0]  exp_r      = exp_ovf ? 5'h1E : (exp_unf ? 5'h0 : exp_raw[4:0]);
wire [9:0]  mant_r     = exp_unf ? 10'd0 : mant_norm[9:0];

// Pack FP16 result
wire [15:0] fp16_result = is_nan  ? {sign_r, 5'h1F, 10'h001} :
                          is_inf  ? {sign_r, 5'h1F, 10'h000} :
                          is_zero ? {sign_r, 15'd0}           :
                                    {sign_r, exp_r, mant_r};

// ---------------------------------------------------------------------------
// Sign-extend to ACC_W for downstream accumulation
// ---------------------------------------------------------------------------
wire [ACC_W-1:0] fp16_ext = {{(ACC_W-16){fp16_result[15]}}, fp16_result};

// ---------------------------------------------------------------------------
// Combinational output — registration happens in pe_top Stage-1 always block.
// Removing the extra register here aligns FP16 latency with INT8 (both = 0
// combinational delay through the multiply, registered once in pe_top).
// ---------------------------------------------------------------------------
assign result = fp16_ext;

// Suppress unused-port warnings for clk/rst_n/en (kept for interface compat)
// synthesis translate_off
wire _unused = clk & rst_n & en;
// synthesis translate_on

endmodule
