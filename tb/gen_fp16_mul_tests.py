import struct, math

def float_to_fp16_bits(f):
    f32 = struct.unpack('<I', struct.pack('<f', f))[0]
    sign = (f32 >> 31) & 1
    exp32 = (f32 >> 23) & 0xFF
    mant32 = f32 & 0x7FFFFF
    if exp32 == 255:
        exp16 = 31
        mant16 = (mant32 >> 13) | 0x200
    elif exp32 == 0:
        exp16 = 0
        mant16 = mant32 >> 13
    else:
        exp16 = exp32 - 127 + 15
        if exp16 <= 0:
            if exp16 < -10:
                return sign << 15
            mant32 = (mant32 | 0x800000) >> (-exp16 + 1)
            exp16 = 0
            mant16 = mant32 >> 13
            halfway = 1 << 12
            if mant32 & halfway:
                mant16 += 1
                if mant16 >= 0x400:
                    exp16 = 1
                    mant16 = 0
        else:
            mant16 = (mant32 + (1 << 12)) >> 13
            if mant16 >= 0x400:
                exp16 += 1
                mant16 = 0
            mant16 &= 0x3FF
    return (sign << 15) | (exp16 << 10) | mant16

def fp16_to_float(bits):
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 31:
        return float('inf') if sign == 0 else float('-inf')
    elif exp == 0:
        if mant == 0:
            return 0.0
        return (-1)**sign * (mant / 1024.0) * (2**(-14))
    else:
        return (-1)**sign * (1 + mant / 1024.0) * (2**(exp - 15))

def fp16_mul_bits(a_bits, b_bits):
    a = fp16_to_float(a_bits)
    b = fp16_to_float(b_bits)
    if math.isnan(a) or math.isnan(b):
        return 0x7E00
    a_inf = ((a_bits >> 10) & 0x1F) == 31 and (a_bits & 0x3FF) == 0
    b_inf = ((b_bits >> 10) & 0x1F) == 31 and (b_bits & 0x3FF) == 0
    a_zero = ((a_bits >> 10) & 0x1F) == 0 and (a_bits & 0x3FF) == 0
    b_zero = ((b_bits >> 10) & 0x1F) == 0 and (b_bits & 0x3FF) == 0
    if (a_inf and b_zero) or (a_zero and b_inf):
        return 0x7E00
    result = a * b
    return float_to_fp16_bits(result)

tests = []
def add_test(a_hex, b_hex, desc):
    a = int(a_hex, 16)
    b = int(b_hex, 16)
    expected = fp16_mul_bits(a, b)
    tests.append((a_hex, b_hex, format(expected, '04X'), desc))

# Basic
add_test('3C00', '3C00', '1.0 * 1.0 = 1.0')
add_test('4000', '3C00', '2.0 * 1.0 = 2.0')
add_test('4000', '4000', '2.0 * 2.0 = 4.0')
add_test('4000', '3E00', '2.0 * 1.5 = 3.0')
add_test('3E00', '3E00', '1.5 * 1.5 = 2.25')
add_test('4200', '3C00', '3.0 * 1.0 = 3.0')
add_test('3E00', '4000', '1.5 * 2.0 = 3.0')
add_test('4400', '4400', '4.0 * 4.0 = 16.0')
add_test('4B00', '4B00', '20.0 * 20.0 = 400.0')

# Negative
add_test('BC00', '3C00', '-1.0 * 1.0 = -1.0')
add_test('BC00', 'BC00', '-1.0 * -1.0 = 1.0')
add_test('C000', '3E00', '-2.0 * 1.5 = -3.0')
add_test('3E00', 'C000', '1.5 * -2.0 = -3.0')

# Special
add_test('7C00', '3C00', 'Inf * 1.0 = Inf')
add_test('7C00', '7C00', 'Inf * Inf = Inf')
add_test('7C00', 'BC00', 'Inf * -1.0 = -Inf')
add_test('7C00', '0000', 'Inf * 0.0 = NaN')
add_test('0000', '7C00', '0.0 * Inf = NaN')
add_test('7E00', '3C00', 'NaN * 1.0 = NaN')
add_test('3C00', '7E00', '1.0 * NaN = NaN')

# Zero
add_test('0000', '3C00', '0.0 * 1.0 = 0.0')
add_test('0000', '0000', '0.0 * 0.0 = 0.0')
add_test('8000', '3C00', '-0.0 * 1.0 = -0.0')
add_test('8000', 'BC00', '-0.0 * -1.0 = +0.0')

# Large
add_test('7BFF', '3C00', 'max FP16 * 1.0 = max')
add_test('7BFF', '4000', 'max FP16 * 2.0 = Inf')

# Subnormal multiplications
add_test('03FF', '3C00', 'max sub * 1.0')
add_test('03FF', '4000', 'max sub * 2.0 = min normal')
add_test('03FF', '3800', 'max sub * 0.5')
add_test('0001', '3C00', 'min sub * 1.0')
add_test('0001', '0001', 'min sub^2 = 0')
add_test('0001', '4000', 'min sub * 2.0')

# Products needing rounding
add_test('3C01', '3C01', '1.001^2 rounding')
add_test('42F6', '4158', '3.14 * 2.72')
add_test('3E00', '3800', '1.5 * 0.5 = 0.75')
add_test('0400', '3400', 'min normal * 0.125 = sub')

# Negative subnormal
add_test('83FF', '3C00', '-max sub * 1.0')
add_test('8001', '3C00', '-min sub * 1.0')

# Subnormal * subnormal
add_test('0200', '0200', 'sub * sub rounding')
add_test('0300', '0200', 'sub * sub 2')

# Generate Verilog
lines = []
lines.append("// FP16 multiply test vectors (Python-generated, IEEE 754)")
lines.append("")
lines.append("  a = 16'h3C00; b = 16'h3C00; #1;")
lines.append("  check(r[15:0], 16'h3C00, 1);  // 1.0 * 1.0 = 1.0")
lines.append("")
lines.append("  a = 16'h4000; b = 16'h3C00; #1;")
lines.append("  check(r[15:0], 16'h4000, 2);  // 2.0 * 1.0 = 2.0")
lines.append("")
lines.append("  a = 16'h4000; b = 16'h4000; #1;")
lines.append("  check(r[15:0], 16'h4400, 3);  // 2.0 * 2.0 = 4.0")
lines.append("")
lines.append("  a = 16'h4000; b = 16'h3E00; #1;")
lines.append("  check(r[15:0], 16'h4200, 4);  // 2.0 * 1.5 = 3.0")
lines.append("")
lines.append("  a = 16'h3E00; b = 16'h3E00; #1;")
lines.append("  check(r[15:0], 16'h4080, 5);  // 1.5 * 1.5 = 2.25")
lines.append("")
lines.append("  a = 16'h4200; b = 16'h3C00; #1;")
lines.append("  check(r[15:0], 16'h4200, 6);  // 3.0 * 1.0 = 3.0")
lines.append("")
lines.append("  a = 16'h3E00; b = 16'h4000; #1;")
lines.append("  check(r[15:0], 16'h4200, 7);  // 1.5 * 2.0 = 3.0")
lines.append("")
lines.append("  a = 16'h4400; b = 16'h4400; #1;")
lines.append("  check(r[15:0], 16'h4C00, 8);  // 4.0 * 4.0 = 16.0")
lines.append("")
lines.append("  a = 16'h4B00; b = 16'h4B00; #1;")
lines.append("  check(r[15:0], 16'h5C00, 9);  // 20.0 * 20.0 = 400.0")
lines.append("")

# Negative numbers
neg_tests = [
    ('BC00', '3C00', '10', '-1.0 * 1.0'),
    ('BC00', 'BC00', '11', '-1.0 * -1.0'),
    ('C000', '3E00', '12', '-2.0 * 1.5'),
    ('3E00', 'C000', '13', '1.5 * -2.0'),
]
for a_hex, b_hex, tid, desc in neg_tests:
    expected = fp16_mul_bits(int(a_hex, 16), int(b_hex, 16))
    lines.append(f"  a = 16'h{a_hex}; b = 16'h{b_hex}; #1;")
    lines.append(f"  check(r[15:0], 16'h{format(expected, '04X')}, {tid});  // {desc}")

lines.append("")

# Special values
lines.append("  a = 16'h7C00; b = 16'h3C00; #1;")
lines.append("  check(r[15:0], 16'h7C00, 20);  // Inf * 1.0 = Inf")
lines.append("")
lines.append("  a = 16'h7C00; b = 16'h7C00; #1;")
lines.append("  check(r[15:0], 16'h7C00, 21);  // Inf * Inf = Inf")
lines.append("")
lines.append("  a = 16'h7C00; b = 16'hBC00; #1;")
lines.append("  check(r[15:0], 16'hFC00, 22);  // Inf * -1.0 = -Inf")
lines.append("")
lines.append("  a = 16'h7C00; b = 16'h0000; #1;")
lines.append("  check_nan(r[15:0], 23);  // Inf * 0.0 = NaN")
lines.append("")
lines.append("  a = 16'h0000; b = 16'h7C00; #1;")
lines.append("  check_nan(r[15:0], 24);  // 0.0 * Inf = NaN")
lines.append("")
lines.append("  a = 16'h7E00; b = 16'h3C00; #1;")
lines.append("  check_nan(r[15:0], 25);  // NaN * 1.0 = NaN")
lines.append("")
lines.append("  a = 16'h3C00; b = 16'h7E00; #1;")
lines.append("  check_nan(r[15:0], 26);  // 1.0 * NaN = NaN")
lines.append("")

# Zero
lines.append("  a = 16'h0000; b = 16'h3C00; #1;")
lines.append("  check(r[15:0], 16'h0000, 30);  // 0.0 * 1.0 = 0.0")
lines.append("")
lines.append("  a = 16'h0000; b = 16'h0000; #1;")
lines.append("  check(r[15:0], 16'h0000, 31);  // 0.0 * 0.0 = 0.0")
lines.append("")
lines.append("  a = 16'h8000; b = 16'h3C00; #1;")
lines.append("  check(r[15:0], 16'h8000, 32);  // -0.0 * 1.0 = -0.0")
lines.append("")
lines.append("  a = 16'h8000; b = 16'hBC00; #1;")
lines.append("  check(r[15:0], 16'h0000, 33);  // -0.0 * -1.0 = +0.0")
lines.append("")

# Large / Overflow
lines.append("  a = 16'h7BFF; b = 16'h3C00; #1;")
lines.append("  check(r[15:0], 16'h7BFF, 40);  // max FP16 * 1.0")
lines.append("")
lines.append("  a = 16'h7BFF; b = 16'h4000; #1;")
lines.append("  check(r[15:0], 16'h7C00, 41);  // max * 2.0 = Inf")
lines.append("")

# Subnormal tests
sub_tests = [
    ('03FF', '3C00', 50, 'max sub * 1.0'),
    ('03FF', '4000', 51, 'max sub * 2.0'),
    ('03FF', '3800', 52, 'max sub * 0.5'),
    ('0001', '3C00', 53, 'min sub * 1.0'),
    ('0001', '0001', 54, 'min sub^2'),
    ('0001', '4000', 55, 'min sub * 2.0'),
    ('0400', '3400', 56, 'min normal * 0.125'),
    ('0200', '0200', 57, 'sub * sub'),
    ('0300', '0200', 58, 'sub * sub 2'),
]
for a_hex, b_hex, tid, desc in sub_tests:
    expected = fp16_mul_bits(int(a_hex, 16), int(b_hex, 16))
    lines.append(f"  a = 16'h{a_hex}; b = 16'h{b_hex}; #1;")
    if expected == 0:
        lines.append(f"  check(r[15:0], 16'h0000, {tid});  // {desc} -> 0")
    else:
        lines.append(f"  check(r[15:0], 16'h{format(expected, '04X')}, {tid});  // {desc}")
    lines.append("")

# Negative subnormal
neg_sub = [
    ('83FF', '3C00', 60, '-max sub * 1.0'),
    ('8001', '3C00', 61, '-min sub * 1.0'),
]
for a_hex, b_hex, tid, desc in neg_sub:
    expected = fp16_mul_bits(int(a_hex, 16), int(b_hex, 16))
    lines.append(f"  a = 16'h{a_hex}; b = 16'h{b_hex}; #1;")
    lines.append(f"  check(r[15:0], 16'h{format(expected, '04X')}, {tid});  // {desc}")
    lines.append("")

# Rounding
rnd_tests = [
    ('3C01', '3C01', 70, '1.001^2'),
    ('42F6', '4158', 71, '3.14 * 2.72'),
    ('3E00', '3800', 72, '1.5 * 0.5'),
]
for a_hex, b_hex, tid, desc in rnd_tests:
    expected = fp16_mul_bits(int(a_hex, 16), int(b_hex, 16))
    lines.append(f"  a = 16'h{a_hex}; b = 16'h{b_hex}; #1;")
    lines.append(f"  check(r[15:0], 16'h{format(expected, '04X')}, {tid});  // {desc}")
    lines.append("")

# Also print expected values for reference
print("// ============ EXPECTED VALUES ============")
for a_hex, b_hex, exp_hex, desc in tests:
    exp_val = int(exp_hex, 16)
    is_nan = ((exp_val >> 10) & 0x1F) == 31 and (exp_val & 0x3FF) != 0
    val_str = fp16_to_float(exp_val) if not is_nan else "NaN"
    print(f"// T: a={a_hex} b={b_hex} -> exp={exp_hex} ({desc}) = {val_str}")

print("\n// ============ VERILOG TEST CODE ============")
print()
for line in lines:
    print(line)
