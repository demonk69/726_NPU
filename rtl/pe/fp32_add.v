// =============================================================================
// Module  : fp32_add
// Project : NPU_prj
// Desc    : IEEE 754 single-precision (FP32) adder, pure combinational output.
//           Supports normal, subnormal, zero, infinity, NaN.
//           Rounding: round-to-nearest-even (RN).
//           Verilog-2001 compatible.
//
//           FP32: 1 sign + 8 exponent + 23 mantissa  (bias=127)
// =============================================================================

`timescale 1ns/1ps

module fp32_add (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result
);

// -----------------------------------------------------------------------
// Unpack
// -----------------------------------------------------------------------
wire        sa = a[31], sb = b[31];
wire [7:0]  ea = a[30:23], eb = b[30:23];
wire [22:0] ma = a[22:0],  mb = b[22:0];

// -----------------------------------------------------------------------
// Special case flags
// -----------------------------------------------------------------------
wire a_zero = (ea == 8'd0) && (ma == 23'd0);
wire b_zero = (eb == 8'd0) && (mb == 23'd0);
wire a_inf  = (ea == 8'hFF) && (ma == 23'd0);
wire b_inf  = (eb == 8'hFF) && (mb == 23'd0);
wire a_nan  = (ea == 8'hFF) && (ma != 23'd0);
wire b_nan  = (eb == 8'hFF) && (mb != 23'd0);

wire any_nan   = a_nan | b_nan;
wire any_inf   = (a_inf | b_inf) & !any_nan;
wire both_zero = a_zero & b_zero;
wire one_zero  = a_zero ^ b_zero;

wire [31:0] nan_res = a_nan ? a : b;
wire inf_same = (sa == sb);
wire [31:0] inf_res = inf_same ? {sa, 8'hFF, 23'd0} : {1'b1, 8'hFF, 23'h400000};
wire [31:0] zero_res = 32'd0;

// -----------------------------------------------------------------------
// Effective mantissa (24-bit with implicit 1) and exponent (9-bit)
// Subnormals: exp=0 -> treated as exp=1, mant={1, ma}
// -----------------------------------------------------------------------
wire [23:0] m_a = {1'b1, ma};
wire [23:0] m_b = {1'b1, mb};
wire [8:0]  e_a = (ea == 8'd0) ? 9'd1 : {1'b0, ea};
wire [8:0]  e_b = (eb == 8'd0) ? 9'd1 : {1'b0, eb};

// -----------------------------------------------------------------------
// Swap so |A| >= |B|
// -----------------------------------------------------------------------
wire a_bigger = (e_a > e_b) || ((e_a == e_b) && (m_a >= m_b));

wire [23:0] m_big = a_bigger ? m_a : m_b;
wire [23:0] m_sml = a_bigger ? m_b : m_a;
wire [8:0]  e_big = a_bigger ? e_a : e_b;
wire [8:0]  e_sml = a_bigger ? e_b : e_a;
wire        s_big = a_bigger ? sa  : sb;
wire        s_sml = a_bigger ? sb  : sa;

wire do_sub = (s_big != s_sml);

// -----------------------------------------------------------------------
// Align mantissas
// 26-bit fractional representation:
//   Bit 25: implicit 1
//   Bits 24:2: FP32 mantissa (23 bits)
//   Bit 1: guard
//   Bit 0: sticky
// -----------------------------------------------------------------------
wire [7:0] shift_amt = e_big - e_sml;

wire [25:0] m_big_ext = {m_big, 2'b00};
wire [25:0] m_sml_ext = {m_sml, 2'b00};

wire [25:0] m_sml_sh = (shift_amt >= 8'd26) ? 26'd0 : (m_sml_ext >> shift_amt);

// Sticky bit
wire sticky;
assign sticky = (shift_amt >= 8'd26) ? (|m_sml) :
                (shift_amt == 8'd0)  ? 1'b0 :
                (|m_sml_ext << (8'd26 - {1'b0, shift_amt}));

wire [25:0] m_sml_final = {m_sml_sh[25:1], m_sml_sh[0] | sticky};

// -----------------------------------------------------------------------
// Add or subtract (27-bit for carry)
// -----------------------------------------------------------------------
wire [26:0] sum_raw = do_sub ? ({1'b0, m_big_ext} - {1'b0, m_sml_final}) :
                               ({1'b0, m_big_ext} + {1'b0, m_sml_final});

// -----------------------------------------------------------------------
// Addition: check carry in bit 26
// -----------------------------------------------------------------------
wire add_carry = sum_raw[26] & !do_sub;

// -----------------------------------------------------------------------
// Subtraction: leading zero count on 27-bit (bit 26 is always 0 for sub)
// Count from bit 25 down to bit 0.
// -----------------------------------------------------------------------
wire lz0  = !sum_raw[25];
wire lz1  = lz0 & !sum_raw[24];
wire lz2  = lz1 & !sum_raw[23];
wire lz3  = lz2 & !sum_raw[22];
wire lz4  = lz3 & !sum_raw[21];
wire lz5  = lz4 & !sum_raw[20];
wire lz6  = lz5 & !sum_raw[19];
wire lz7  = lz6 & !sum_raw[18];
wire lz8  = lz7 & !sum_raw[17];
wire lz9  = lz8 & !sum_raw[16];
wire lz10 = lz9 & !sum_raw[15];
wire lz11 = lz10 & !sum_raw[14];
wire lz12 = lz11 & !sum_raw[13];
wire lz13 = lz12 & !sum_raw[12];
wire lz14 = lz13 & !sum_raw[11];
wire lz15 = lz14 & !sum_raw[10];
wire lz16 = lz15 & !sum_raw[9];
wire lz17 = lz16 & !sum_raw[8];
wire lz18 = lz17 & !sum_raw[7];
wire lz19 = lz18 & !sum_raw[6];
wire lz20 = lz19 & !sum_raw[5];
wire lz21 = lz20 & !sum_raw[4];
wire lz22 = lz21 & !sum_raw[3];
wire lz23 = lz22 & !sum_raw[2];
wire lz24 = lz23 & !sum_raw[1];
wire lz25 = lz24 & !sum_raw[0];

wire [4:0] lz_full;
assign lz_full = lz25 ? 5'd26 :
                 lz24 ? 5'd25 :
                 lz23 ? 5'd24 :
                 lz22 ? 5'd23 :
                 lz21 ? 5'd22 :
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

// Barrel left-shift (subtract path)
wire [26:0] sh1  = sum_raw << (lz_full >= 5'd1  ? 1 : 0);
wire [26:0] sh2  = sh1    << (lz_full >= 5'd2  ? 1 : 0);
wire [26:0] sh3  = sh2    << (lz_full >= 5'd3  ? 1 : 0);
wire [26:0] sh4  = sh3    << (lz_full >= 5'd4  ? 1 : 0);
wire [26:0] sh5  = sh4    << (lz_full >= 5'd5  ? 1 : 0);
wire [26:0] sh6  = sh5    << (lz_full >= 5'd6  ? 1 : 0);
wire [26:0] sh7  = sh6    << (lz_full >= 5'd7  ? 1 : 0);
wire [26:0] sh8  = sh7    << (lz_full >= 5'd8  ? 1 : 0);
wire [26:0] sh9  = sh8    << (lz_full >= 5'd9  ? 1 : 0);
wire [26:0] sh10 = sh9    << (lz_full >= 5'd10 ? 1 : 0);
wire [26:0] sh11 = sh10   << (lz_full >= 5'd11 ? 1 : 0);
wire [26:0] sh12 = sh11   << (lz_full >= 5'd12 ? 1 : 0);
wire [26:0] sh13 = sh12   << (lz_full >= 5'd13 ? 1 : 0);
wire [26:0] sh14 = sh13   << (lz_full >= 5'd14 ? 1 : 0);
wire [26:0] sh15 = sh14   << (lz_full >= 5'd15 ? 1 : 0);
wire [26:0] sh16 = sh15   << (lz_full >= 5'd16 ? 1 : 0);
wire [26:0] sh17 = sh16   << (lz_full >= 5'd17 ? 1 : 0);
wire [26:0] sh18 = sh17   << (lz_full >= 5'd18 ? 1 : 0);
wire [26:0] sh19 = sh18   << (lz_full >= 5'd19 ? 1 : 0);
wire [26:0] sh20 = sh19   << (lz_full >= 5'd20 ? 1 : 0);
wire [26:0] sh21 = sh20   << (lz_full >= 5'd21 ? 1 : 0);
wire [26:0] sh22 = sh21   << (lz_full >= 5'd22 ? 1 : 0);
wire [26:0] sh23 = sh22   << (lz_full >= 5'd23 ? 1 : 0);
wire [26:0] sh24 = sh23   << (lz_full >= 5'd24 ? 1 : 0);
wire [26:0] sh25 = sh24   << (lz_full >= 5'd25 ? 1 : 0);
wire [26:0] sh26 = sh25   << (lz_full >= 5'd26 ? 1 : 0);

// -----------------------------------------------------------------------
// Select normalized mantissa and exponent
// Layout: [25]=implicit1, [24:2]=fraction(23 bits), [1]=guard, [0]=sticky
// -----------------------------------------------------------------------
wire [8:0]  norm_exp;
wire [25:0] norm_mant;

wire [25:0] add_mant = add_carry ? sum_raw[26:1] : sum_raw[25:0];
wire [8:0]  add_exp  = add_carry ? (e_big + 8'd1) : e_big;

wire [25:0] sub_mant = sh26[25:0];
wire [8:0]  sub_exp  = (sum_raw[26:0] == 27'd0) ? 8'd0 :
                       (lz_full > 5'd25) ? 8'd0 :
                       e_big - {1'b0, lz_full[4:0]};

assign norm_mant = do_sub ? sub_mant : add_mant;
assign norm_exp  = do_sub ? sub_exp  : add_exp;

// -----------------------------------------------------------------------
// Round to nearest even
// -----------------------------------------------------------------------
wire [22:0] mant_pre   = norm_mant[24:2];
wire        guard_bit  = norm_mant[1];
wire        rnd_stk    = norm_mant[0];

wire do_inc = guard_bit & (rnd_stk | mant_pre[0]);

wire [23:0] mant_ext = {1'b0, mant_pre} + {23'd0, do_inc};

wire rnd_ovf = mant_ext[23];
wire [22:0]  final_mant = rnd_ovf ? 23'b01000000000000000000000 : mant_ext[22:0];
wire [8:0]   final_exp  = rnd_ovf ? (norm_exp + 8'd1) : norm_exp;

// -----------------------------------------------------------------------
// Sign determination
// -----------------------------------------------------------------------
wire norm_sign = do_sub ? ((sum_raw[26:0] == 27'd0) ? 1'b0 : s_big) : s_big;

// -----------------------------------------------------------------------
// Pack result
// Overflow clamp to FP32 max
// -----------------------------------------------------------------------
wire [8:0]  pack_exp  = (final_exp >= 9'hFE) ? 9'hFE : final_exp;
wire [22:0] pack_mant = final_mant;

wire [31:0] normal_res = {norm_sign, pack_exp[7:0], pack_mant};

// -----------------------------------------------------------------------
// Final result
// -----------------------------------------------------------------------
wire [31:0] one_zero_res = a_zero ? b : a;

assign result = any_nan   ? nan_res      :
                any_inf   ? inf_res      :
                both_zero ? zero_res     :
                one_zero  ? one_zero_res :
                normal_res;

endmodule
