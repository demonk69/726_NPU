// =============================================================================
// Module  : fp16_add
// Project : NPU_prj
// Desc    : IEEE 754 half-precision (FP16) adder, pure combinational output.
//           Supports normal, subnormal, zero, infinity, NaN.
//           Rounding: round-to-nearest-even (RN).
//           Verilog-2001 compatible.
// =============================================================================

`timescale 1ns/1ps

module fp16_add (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] result
);

// -----------------------------------------------------------------------
// Unpack
// -----------------------------------------------------------------------
wire        sa = a[15], sb = b[15];
wire [4:0]  ea = a[14:10], eb = b[14:10];
wire [9:0]  ma = a[9:0],   mb = b[9:0];

// -----------------------------------------------------------------------
// Special case flags
// -----------------------------------------------------------------------
wire a_zero = (ea == 5'd0) && (ma == 10'd0);
wire b_zero = (eb == 5'd0) && (mb == 10'd0);
wire a_inf  = (ea == 5'h1F) && (ma == 10'd0);
wire b_inf  = (eb == 5'h1F) && (mb == 10'd0);
wire a_nan  = (ea == 5'h1F) && (ma != 10'd0);
wire b_nan  = (eb == 5'h1F) && (mb != 10'd0);

wire any_nan   = a_nan | b_nan;
wire any_inf   = (a_inf | b_inf) & !any_nan;
wire both_zero = a_zero & b_zero;
wire one_zero  = a_zero ^ b_zero;  // exactly one is zero (not both)

// NaN: propagate payload (prefer a). Sign: if both NaN, use a's sign; quiet NaN has sign=1
wire [15:0] nan_res = a_nan ? a : b;
// Inf: same-sign -> that inf; opposite -> NaN (quiet, sign=1)
wire inf_same = (sa == sb);
wire [15:0] inf_res = inf_same ? {sa, 5'h1F, 10'd0} : {1'b1, 5'h1F, 10'h200};
// Zero: always +0
wire [15:0] zero_res = 16'd0;

// -----------------------------------------------------------------------
// Effective mantissa (11-bit with implicit 1) and exponent (6-bit)
// Subnormals treated as exp=1, mant={1, ma}
// -----------------------------------------------------------------------
wire [10:0] m_a = {1'b1, ma};
wire [10:0] m_b = {1'b1, mb};
wire [5:0]  e_a = (ea == 5'd0) ? 6'd1 : {1'b0, ea};
wire [5:0]  e_b = (eb == 5'd0) ? 6'd1 : {1'b0, eb};

// -----------------------------------------------------------------------
// Swap so |A| >= |B|
// -----------------------------------------------------------------------
wire a_bigger = (e_a > e_b) || ((e_a == e_b) && (m_a >= m_b));

wire [10:0] m_big = a_bigger ? m_a : m_b;
wire [10:0] m_sml = a_bigger ? m_b : m_a;
wire [5:0]  e_big = a_bigger ? e_a : e_b;
wire [5:0]  e_sml = a_bigger ? e_b : e_a;
wire        s_big = a_bigger ? sa  : sb;
wire        s_sml = a_bigger ? sb  : sa;

// -----------------------------------------------------------------------
// Add vs subtract
// -----------------------------------------------------------------------
wire do_sub = (s_big != s_sml);

// -----------------------------------------------------------------------
// Align mantissas
// Use 13-bit fractional representation:
//   Bit 12: implicit 1 (or 0 for zero result)
//   Bits 11:2: FP16 mantissa (10 bits)
//   Bit 1: guard
//   Bit 0: sticky
// Total: 13 bits. Max value 1.1111111111_1_1 = 8191
// Addition can produce up to 14 bits (carry out of bit 12).
// -----------------------------------------------------------------------
wire [4:0] shift_amt = e_big - e_sml;

// Extend to 13 bits: {implicit1, mantissa[9:0], guard=0, sticky=0}
wire [12:0] m_big_ext = {m_big, 2'b00};  // 11 + 2 = 13 bits
wire [12:0] m_sml_ext = {m_sml, 2'b00};

// Right shift small mantissa. If shift >= 13, everything shifts out.
wire [12:0] m_sml_sh = (shift_amt >= 5'd13) ? 13'd0 : (m_sml_ext >> shift_amt);

// Sticky bit: OR of all bits that were shifted out
wire sticky;
assign sticky = (shift_amt >= 5'd13) ? (|m_sml) :          // all shifted out
                (shift_amt == 5'd0)  ? 1'b0 :             // no shift
                (|m_sml_ext << (6'd13 - {1'b0, shift_amt}));  // bits above 12

// Merge sticky into bit 0 of shifted mantissa
wire [12:0] m_sml_final = {m_sml_sh[12:1], m_sml_sh[0] | sticky};

// -----------------------------------------------------------------------
// Add or subtract (use 14-bit to accommodate carry)
// -----------------------------------------------------------------------
wire [13:0] sum_raw = do_sub ? ({1'b0, m_big_ext} - {1'b0, m_sml_final}) :
                               ({1'b0, m_big_ext} + {1'b0, m_sml_final});

// -----------------------------------------------------------------------
// Addition path: check for carry in bit 13
// -----------------------------------------------------------------------
wire add_carry = sum_raw[13] & !do_sub;

// -----------------------------------------------------------------------
// Subtraction path: leading zero count
// For subtraction, the leading 1 is at bit 12 (since 14-bit is {0, 13-bit}).
// So we count leading zeros starting from bit 12.
// -----------------------------------------------------------------------
wire lz0  = !sum_raw[12];
wire lz1  = lz0 & !sum_raw[11];
wire lz2  = lz1 & !sum_raw[10];
wire lz3  = lz2 & !sum_raw[9];
wire lz4  = lz3 & !sum_raw[8];
wire lz5  = lz4 & !sum_raw[7];
wire lz6  = lz5 & !sum_raw[6];
wire lz7  = lz6 & !sum_raw[5];
wire lz8  = lz7 & !sum_raw[4];
wire lz9  = lz8 & !sum_raw[3];
wire lz10 = lz9 & !sum_raw[2];
wire lz11 = lz10 & !sum_raw[1];
wire lz12 = lz11 & !sum_raw[0];

wire [3:0] lz_full;
assign lz_full = lz12 ? 4'd13 :
                 lz11 ? 4'd12 :
                 lz10 ? 4'd11 :
                 lz9  ? 4'd10 :
                 lz8  ? 4'd9  :
                 lz7  ? 4'd8  :
                 lz6  ? 4'd7  :
                 lz5  ? 4'd6  :
                 lz4  ? 4'd5  :
                 lz3  ? 4'd4  :
                 lz2  ? 4'd3  :
                 lz1  ? 4'd2  :
                 lz0  ? 4'd1  :
                 4'd0;

// Barrel left-shift to normalize (subtract path)
wire [13:0] sh1  = sum_raw << (lz_full >= 4'd1  ? 1 : 0);
wire [13:0] sh2  = sh1    << (lz_full >= 4'd2  ? 1 : 0);
wire [13:0] sh3  = sh2    << (lz_full >= 4'd3  ? 1 : 0);
wire [13:0] sh4  = sh3    << (lz_full >= 4'd4  ? 1 : 0);
wire [13:0] sh5  = sh4    << (lz_full >= 4'd5  ? 1 : 0);
wire [13:0] sh6  = sh5    << (lz_full >= 4'd6  ? 1 : 0);
wire [13:0] sh7  = sh6    << (lz_full >= 4'd7  ? 1 : 0);
wire [13:0] sh8  = sh7    << (lz_full >= 4'd8  ? 1 : 0);
wire [13:0] sh9  = sh8    << (lz_full >= 4'd9  ? 1 : 0);
wire [13:0] sh10 = sh9    << (lz_full >= 4'd10 ? 1 : 0);
wire [13:0] sh11 = sh10   << (lz_full >= 4'd11 ? 1 : 0);
wire [13:0] sh12 = sh11   << (lz_full >= 4'd12 ? 1 : 0);
wire [13:0] sh13 = sh12   << (lz_full >= 4'd13 ? 1 : 0);

// -----------------------------------------------------------------------
// Select normalized mantissa and exponent
// Layout after normalization: bit 12 = implicit 1, bits 11:2 = mantissa, [1]=guard, [0]=sticky
// -----------------------------------------------------------------------
wire [4:0]  norm_exp;
wire [12:0] norm_mant;  // 13-bit: [12]=implicit1, [11:2]=fraction, [1:0]=guard+sticky

// Addition: if carry, shift right 1 and increment exp
wire [12:0] add_mant = add_carry ? sum_raw[13:1] : sum_raw[12:0];
wire [4:0]  add_exp  = add_carry ? (e_big + 5'd1) : e_big;

// Subtraction: shift left by lz, decrement exp
wire [12:0] sub_mant = sh13[12:0];
wire [4:0]  sub_exp  = (sum_raw[13:0] == 14'd0) ? 5'd0 :
                       (lz_full > 4'd13) ? 5'd0 :
                       e_big - {1'b0, lz_full[3:0]};

assign norm_mant = do_sub ? sub_mant : add_mant;
assign norm_exp  = do_sub ? sub_exp  : add_exp;

// -----------------------------------------------------------------------
// Round to nearest even
// FP16 mantissa = norm_mant[11:2] (10 bits)
// Guard = norm_mant[1], Round|Sticky = norm_mant[0]
// -----------------------------------------------------------------------
wire [9:0] mant_pre   = norm_mant[11:2];
wire        guard_bit  = norm_mant[1];
wire        rnd_stk    = norm_mant[0];  // sticky already merged into bit 0

wire do_inc = guard_bit & (rnd_stk | mant_pre[0]);

// Increment mantissa (11-bit to detect overflow)
wire [10:0] mant_ext = {1'b0, mant_pre} + {10'd0, do_inc};

// Handle rounding overflow: 1.1111111111 + 1 = 10.0000000000
wire rnd_ovf = mant_ext[10];  // overflow into bit 10
wire [9:0]  final_mant = rnd_ovf ? 10'b0100000000 : mant_ext[9:0];
wire [4:0]  final_exp  = rnd_ovf ? (norm_exp + 5'd1) : norm_exp;

// -----------------------------------------------------------------------
// Sign determination
// -----------------------------------------------------------------------
wire norm_sign = do_sub ? ((sum_raw[13:0] == 14'd0) ? 1'b0 : s_big) : s_big;

// -----------------------------------------------------------------------
// Pack result
// Overflow clamp to FP16 max (65504 = 0x7BFF, exp=30, mant=1111111111)
// -----------------------------------------------------------------------
wire [4:0]  pack_exp  = (final_exp >= 5'h1E) ? 5'h1E : final_exp;
wire [9:0]  pack_mant = final_mant;

wire [15:0] normal_res = {norm_sign, pack_exp, pack_mant};

// -----------------------------------------------------------------------
// Final result
// -----------------------------------------------------------------------
// When one operand is zero, return the other (IEEE 754: x + 0 = x)
wire [15:0] one_zero_res = a_zero ? b : a;

assign result = any_nan   ? nan_res      :
                any_inf   ? inf_res      :
                both_zero ? zero_res     :
                one_zero  ? one_zero_res :
                normal_res;

endmodule
