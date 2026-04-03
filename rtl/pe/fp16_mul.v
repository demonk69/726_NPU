// =============================================================================
// Module  : fp16_mul
// Project : NPU_prj
// Desc    : IEEE 754 half-precision (FP16) multiplier, combinational output.
//           FP16: 1 sign + 5 exponent + 10 mantissa  (bias=15)
//           Output is zero-extended to ACC_W=32 bits.
//           NEVER sign-extend IEEE 754 bit patterns!
//
//           Features:
//           - Subnormal support (dynamic implicit bit)
//           - Full leading-zero count (LZC) for normalization
//           - Gradual underflow (subnormal output when biased_exp <= 0)
//           - NaN / Inf / Zero special cases (IEEE 754 compliant)
//           - Round-to-nearest-even (RN)
//
//           Pipeline: Combinational; result registered in pe_top Stage-1.
//
//           Algorithm:
//           1. Unpack FP16, detect specials
//           2. Subnormals: implicit=0, eff_exp=1; Normals: implicit=1, eff_exp=stored
//           3. Multiply 11-bit mantissas -> 22-bit product
//           4. LZC on 22-bit product (0..21 for normal, 22=all-zero)
//           5. Normalization: right-shift product by (11-lzc) to get 11-bit mantissa
//              at bit position [10:0] (implicit 1 at bit 10, fraction at [9:0])
//           6. Guard = prod[10-lzc], Sticky = |prod[9-lzc:0] (when lzc<=10)
//           7. Exponent: biased_exp = eff_ea + eff_eb - 16 + lzc
//           8a. Normal (biased_exp > 0): RN rounding, overflow clamping
//           8b. Subnormal (biased_exp <= 0): denormalize via extra right-shift,
//               RN rounding on shifted-out bits; flush-to-zero if too small
//               (biased_exp < -10); round-up to min normal if mantissa overflows
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
    output wire [ACC_W-1:0] result  // product (zero-extended FP16)
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
// Sign
// ---------------------------------------------------------------------------
wire sign_r = sa ^ sb;

// ---------------------------------------------------------------------------
// Subnormal support: dynamic implicit bit
// Subnormal: exp=0, mant!=0 => implicit=0, effective exp = 1
// Normal:    exp!=0 => implicit=1, effective exp = stored exp
// ---------------------------------------------------------------------------
wire [10:0] mant_a = {(ea != 5'd0), ma};
wire [10:0] mant_b = {(eb != 5'd0), mb};

wire [5:0] eff_ea = (ea == 5'd0) ? 6'd1 : {1'b0, ea};
wire [5:0] eff_eb = (eb == 5'd0) ? 6'd1 : {1'b0, eb};

// ---------------------------------------------------------------------------
// Multiply: 11-bit mantissa * 11-bit mantissa -> 22-bit product
// Use 22-bit wide operands to avoid Verilog context-determined width truncation.
// Two normal numbers: leading 1 at bit 21 or 20 of the 22-bit product.
// With subnormals: leading 1 can be at any bit 20 down to 0.
// ---------------------------------------------------------------------------
wire [21:0] wa = {11'd0, mant_a};
wire [21:0] wb = {11'd0, mant_b};
wire [43:0] prod_full = wa * wb;
wire [21:0] prod = prod_full[21:0]; // 22-bit product (sufficient for 11x11)

// ---------------------------------------------------------------------------
// Leading Zero Count (LZC) on 22-bit product
// lzc ranges from 0 (bit 21 set) to 21 (bit 0 set) or 22 (all zeros)
// ---------------------------------------------------------------------------
wire lz0  = !prod[21];
wire lz1  = lz0 & !prod[20];
wire lz2  = lz1 & !prod[19];
wire lz3  = lz2 & !prod[18];
wire lz4  = lz3 & !prod[17];
wire lz5  = lz4 & !prod[16];
wire lz6  = lz5 & !prod[15];
wire lz7  = lz6 & !prod[14];
wire lz8  = lz7 & !prod[13];
wire lz9  = lz8 & !prod[12];
wire lz10 = lz9 & !prod[11];
wire lz11 = lz10 & !prod[10];
wire lz12 = lz11 & !prod[9];
wire lz13 = lz12 & !prod[8];
wire lz14 = lz13 & !prod[7];
wire lz15 = lz14 & !prod[6];
wire lz16 = lz15 & !prod[5];
wire lz17 = lz16 & !prod[4];
wire lz18 = lz17 & !prod[3];
wire lz19 = lz18 & !prod[2];
wire lz20 = lz19 & !prod[1];
wire lz21 = lz20 & !prod[0];

wire [4:0] lzc;
assign lzc = lz21 ? 5'd22 :
              lz20 ? 5'd21 :
              lz19 ? 5'd20 :
              lz18 ? 5'd19 :
              lz17 ? 5'd18 :
              lz16 ? 5'd17 :
              lz15 ? 5'd16 :
              lz14 ? 5'd15 :
              lz13 ? 5'd14 :
              lz12 ? 5'd13 :
              lz11 ? 5'd12 :
              lz10 ? 5'd11 :
              lz9  ? 5'd10 :
              lz8  ? 5'd9  :
              lz7  ? 5'd8  :
              lz6  ? 5'd7  :
              lz5  ? 5'd6  :
              lz4  ? 5'd5  :
              lz3  ? 5'd4  :
              lz2  ? 5'd3  :
              lz1  ? 5'd2  :
              lz0  ? 5'd1  :
              5'd0;

// ---------------------------------------------------------------------------
// Normalize: right-shift product by (11 - lzc)
// After shift: bit 10 = implicit 1, bits [9:0] = mantissa fraction
// Right-shift amount rs = 11 - lzc (can be 0..11)
//
// We use a barrel RIGHT-shifter. Since rs = 11 - lzc and lzc is 0..22,
// rs ranges from -11..11, but for lzc > 11, the product is too small to have
// an implicit 1 at bit 10, meaning the result is a subnormal or zero.
//
// For lzc <= 11: rs = 11 - lzc (range 0..11), valid normalization
// For lzc > 11: product has leading 1 at bit (21-lzc) < bit 10
//   This means even after left-shifting by lzc, we can't get bit 10 set.
//   The result underflows to subnormal or zero.
// ---------------------------------------------------------------------------
// Barrel RIGHT shifter on 22-bit prod, shift amount = rs = 11 - lzc
// We compute rs = 11 - lzc using 5-bit subtraction.
wire [4:0] rs = 5'd11 - lzc;  // right-shift amount

// For barrel right-shift, we shift by rs[0], rs[1], rs[2], rs[3], rs[4] bits
wire [21:0] rs1  = rs[0] ? (prod >> 1)  : prod;
wire [21:0] rs2  = rs[1] ? (rs1  >> 2)  : rs1;
wire [21:0] rs3  = rs[2] ? (rs2  >> 4)  : rs2;
wire [21:0] rs4  = rs[3] ? (rs3  >> 8)  : rs3;
wire [21:0] rs5  = rs[4] ? (rs4  >> 16) : rs4;

// After right-shift: rs5[10] = 1 (implicit 1), rs5[9:0] = mantissa
wire [9:0] mant_norm = rs5[9:0];

// ---------------------------------------------------------------------------
// Exponent calculation
// biased_exp = eff_ea + eff_eb - 16 + lzc = eff_ea + eff_eb - 14 - lzc
// For normal * normal: lzc is 0 or 1
// For subnormals: lzc can be larger, compensated by larger lzc term
// ---------------------------------------------------------------------------
wire signed [6:0] biased_exp_s =
    $signed({1'b0, eff_ea}) + $signed({1'b0, eff_eb}) - 7'sd14 - $signed({2'b0, lzc});

// ---------------------------------------------------------------------------
// Subnormal output detection
// When biased_exp <= 0: the result is subnormal or zero.
//   extra_shift = 1 - biased_exp (how many more bits to shift right)
//   Max meaningful shift: 10 (to keep at least 1 bit in mantissa)
//   If biased_exp <= -10: result is too small, flush to zero.
//
// For subnormal result: take rs5[10:0] (11-bit normalized mantissa),
// shift right by extra_shift, keep top 10 bits as subnormal mantissa.
// Apply guard/sticky rounding on the shifted-out bits.
// ---------------------------------------------------------------------------
wire        is_sub = biased_exp_s <= 7'sd0;  // biased_exp <= 0
wire [4:0]  extra_shift = 5'd1 - biased_exp_s[4:0];  // 1..31
wire        too_small   = (biased_exp_s < -7'sd10);    // can't represent, flush to 0

// Barrel right-shift the 11-bit normalized mantissa rs5[10:0]
// by extra_shift to produce the subnormal mantissa (10 bits).
// We pack M = rs5[10:0] into bits [21:11] of a 22-bit word (low 11 bits = 0).
// After >> extra_shift, extract:
//   mantissa = bits [20:11] (top 10 bits of shifted M)
//   guard    = bit  [10]
//   sticky   = |bits [9:0]
wire [21:0] sub_in = {rs5[10:0], 11'd0};  // M at [21:11], zeros at [10:0]
wire [21:0] sub_s1 = extra_shift[0] ? (sub_in >> 1)  : sub_in;
wire [21:0] sub_s2 = extra_shift[1] ? (sub_s1 >> 2)  : sub_s1;
wire [21:0] sub_s3 = extra_shift[2] ? (sub_s2 >> 4)  : sub_s2;
wire [21:0] sub_s4 = extra_shift[3] ? (sub_s3 >> 8)  : sub_s3;
wire [21:0] sub_s5 = extra_shift[4] ? (sub_s4 >> 16) : sub_s4;
wire [9:0]  sub_mant_raw = sub_s5[20:11];

// Guard and sticky for subnormal:
wire        sub_guard  = sub_s5[10];
wire        sub_sticky = |sub_s5[9:0];

// Round-to-nearest-even for subnormal
wire        sub_do_inc = sub_guard & (sub_sticky | sub_mant_raw[0]);
wire [10:0] sub_mant_ext = {1'b0, sub_mant_raw} + {10'd0, sub_do_inc};
wire [9:0]  sub_mant_r = sub_mant_ext[9:0];
// If rounding causes mantissa to reach 0x400, the subnormal result
// actually rounds up to the minimum normal number (exp=1, mant=0).
wire        sub_rounds_to_normal = sub_mant_ext[10]; // overflow from 10-bit mantissa

// ---------------------------------------------------------------------------
// Normal path: guard/sticky/rounding (unchanged for biased_exp > 0)
// Guard is the first bit shifted out = prod[10-lzc] when lzc <= 10
// Sticky is OR of all bits below guard = |prod[9-lzc:0] when lzc <= 9
// When lzc >= 11: no guard or sticky (product is exactly at bit 10 after shift)
// ---------------------------------------------------------------------------
wire guard_bit  = (lzc <= 5'd10) ? prod[10-lzc] : 1'b0;
wire [21:0] sticky_mask = (lzc <= 5'd10) ? ((22'd1 << (10 - lzc)) - 22'd1) : 22'd0;
wire sticky_raw = |(prod & sticky_mask);

// Round to nearest even (RN) for normal path
wire do_inc = guard_bit & (sticky_raw | mant_norm[0]);

wire [10:0] mant_ext = {1'b0, mant_norm} + {10'd0, do_inc};
wire rnd_ovf = mant_ext[10]; // 1.1111111111 + 1 = 10.0000000000
wire [9:0]  mant_r_raw = rnd_ovf ? 10'b0100000000 : mant_ext[9:0];
wire signed [6:0] exp_after_rnd = rnd_ovf ? (biased_exp_s + 7'sd1) : biased_exp_s;

// ---------------------------------------------------------------------------
// Clamp: overflow -> Inf, underflow -> subnormal or zero
// ---------------------------------------------------------------------------
wire exp_ovf = !exp_after_rnd[6] && (exp_after_rnd[5:0] > 7'd30);
wire exp_unf = exp_after_rnd[6]; // negative -> underflow

// ---------------------------------------------------------------------------
// Pack FP16 result
// ---------------------------------------------------------------------------
wire [15:0] fp16_result =
    is_nan  ? {sign_r, 5'h1F, 10'h200} :
    is_inf  ? {sign_r, 5'h1F, 10'h000} :
    is_zero ? {sign_r, 15'd0}           :
    // Overflow: clamp to Inf
    exp_ovf ? {sign_r, 5'h1F, 10'd0}    :
    // Subnormal result (biased_exp <= 0)
    is_sub  ? (too_small ? {sign_r, 15'd0} :                        // flush to zero
               sub_rounds_to_normal ? {sign_r, 5'd1, 10'd0} :      // rounds up to min normal
               {sign_r, 5'd0, sub_mant_r})                         // subnormal
    :
    // Normal result
    {sign_r, exp_after_rnd[4:0], mant_r_raw};

// ---------------------------------------------------------------------------
// Zero-extend to ACC_W — NEVER sign-extend IEEE 754 bit patterns!
// ---------------------------------------------------------------------------
wire [ACC_W-1:0] fp16_ext = {{(ACC_W-16){1'b0}}, fp16_result};

// ---------------------------------------------------------------------------
// Combinational output — registration happens in pe_top Stage-1 always block.
// ---------------------------------------------------------------------------
assign result = fp16_ext;

// Suppress unused-port warnings for clk/rst_n/en (kept for interface compat)
// synthesis translate_off
wire _unused = clk & rst_n & en;
// synthesis translate_on

endmodule
