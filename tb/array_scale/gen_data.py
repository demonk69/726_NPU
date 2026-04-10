# =============================================================================
# gen_data.py - Generate test data for K-depth scale verification
#
# Design: COLS=1, ROWS=1, K varies (K=N for each N ∈ {4, 8, 16, 32}).
#
# PPBuf SUBW=4 (current RTL):
#   INT8  : packed as 4 × INT8 per 32-bit word
#   FP16  : packed as 2 × FP16 per 32-bit word
#
# Tests per K:
#   T0  : INT8  WS K=K   (internal ws_acc accumulation, flush writes 1 dot-product)
#   T1  : INT8  OS K=K   (internal os_acc accumulation, result = dot product)
#   T2  : FP16  WS K=K   (internal ws_acc accumulation in FP32, flush writes 1 dot-product)
#   T3  : FP16  OS K=K   (FP32 mixed-precision dot product)
#
# Note on current WS semantics:
#   In the current RTL, WS mode also accumulates across K inside pe_top.ws_acc
#   and flushes exactly one result word through DMA (r_len_bytes = 4 bytes).
#   Therefore WS expected values must be the full K-element dot product, not
#   the first beat or the last beat product.
#
# Output per K value (directory NK/):
#   dram_init.hex   - DRAM initialisation
#   expected.hex    - Expected results (one 32-bit word per test)
#   test_params.vh  - Verilog `define parameters
# =============================================================================


import struct, os, math, random

BASE_ADDR = 0x10000
ADDR_GAP  = 0x100   # 256 bytes = 64 words; plenty for K<=32

# ---------------------------------------------------------------------------
# Packing helpers
#
# PPBuf is now configured as OUT_WIDTH=8, SUBW=4:
#   INT8  : 4 elements per 32-bit word (bytes [0..3] = elements [0..3])
#   FP16  : 2 elements per 32-bit word (packed as 16-bit halves)
#   (FP16 goes through byte packer in npu_top.v which reassembles 2 bytes
#    into one 16-bit FP16 value.)
# ---------------------------------------------------------------------------
def int8_pack_subw4(vals):
    """Pack INT8 values 4-per-word (SUBW=4).
    Each 32-bit word: byte[0]=val[4i], byte[1]=val[4i+1],
                      byte[2]=val[4i+2], byte[3]=val[4i+3].
    This matches PPBuf OUT_WIDTH=8, SUBW=4: each beat outputs one byte.
    """
    words = []
    for i in range(0, len(vals), 4):
        w = (vals[i] & 0xFF)
        w |= ((vals[i+1] & 0xFF) if i+1 < len(vals) else 0) << 8
        w |= ((vals[i+2] & 0xFF) if i+2 < len(vals) else 0) << 16
        w |= ((vals[i+3] & 0xFF) if i+3 < len(vals) else 0) << 24
        words.append(w)
    return words

def fp16_pack(vals):
    """Pack FP16 values 2-per-word (standard FP16 packing).
    Each 32-bit word: bits[15:0]=val[2i], bits[31:16]=val[2i+1].
    PPBuf SUBW=4 gives 4 bytes per word; byte packer reassembles
    bytes 0-1 → FP16[0], bytes 2-3 → FP16[1].
    """
    words = []
    for i in range(0, len(vals), 2):
        lo = vals[i] & 0xFFFF
        hi = (vals[i+1] & 0xFFFF) if i+1 < len(vals) else 0
        words.append(lo | (hi << 16))
    return words

# ---------------------------------------------------------------------------
# FP16 helpers
# ---------------------------------------------------------------------------
def fp16_to_float(h):
    s = (h >> 15) & 1; e = (h >> 10) & 0x1F; f = h & 0x3FF
    if e == 0 and f == 0: return -0.0 if s else 0.0
    if e == 0: return ((-1)**s) * (f / 1024.0) * 2**(-14)
    if e == 31: return float('-inf') if s else float('inf')
    return ((-1)**s) * (1 + f/1024.0) * 2**(e-15)

def float_to_fp16(f):
    if math.isnan(f): return 0x7E00
    if math.isinf(f): return 0xFC00 if f < 0 else 0x7C00
    if f == 0: return 0x8000 if math.copysign(1, f) < 0 else 0x0000
    s = 1 if f < 0 else 0; f = abs(f)
    e = int(math.floor(math.log2(f))); be = e + 15
    if be <= 0:
        af = round(f / 2**(-24))
        return (s<<15) | min(af, 0x3FF)
    if be >= 31: return 0xFC00 if s else 0x7C00
    fr = f / 2**e - 1.0; m = round(fr * 1024)
    if m >= 1024: m = 0; be += 1
    if be >= 31: return 0xFC00 if s else 0x7C00
    return (s<<15) | (be<<10) | m

def float_to_fp32_word(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]

def write_hex(path, words):
    with open(path, 'w') as f:
        for w in words:
            f.write(f'{w:08x}\n')

# ---------------------------------------------------------------------------
# ctrl register encoding
# ---------------------------------------------------------------------------
def make_ctrl(dtype, mode):
    # bit0=start, [3:2]=dtype(00=INT8,10=FP16), [5:4]=stat(00=WS,01=OS)
    dtype_bits = 2 if dtype == 'fp16' else 0
    stat_bits  = 1 if mode == 'OS' else 0
    return (stat_bits << 4) | (dtype_bits << 2) | 1

# ---------------------------------------------------------------------------
# Test generation
# ---------------------------------------------------------------------------
def gen_test_int8_os(K, seed):
    """INT8 OS: K-element dot product. Expected = sum(w[i]*a[i])."""
    rng = random.Random(seed)
    ws = [rng.randint(-64, 63) for _ in range(K)]
    ac = [rng.randint(-64, 63) for _ in range(K)]
    exp = sum(ws[i]*ac[i] for i in range(K)) & 0xFFFFFFFF
    return ws, ac, exp

def gen_test_int8_ws(K, seed):
    """INT8 WS: full K-element dot product.
    Current RTL accumulates in pe_top.ws_acc and writes a single flush result,
    so the golden value must match the complete dot product.
    """
    rng = random.Random(seed)
    ws = [rng.randint(-64, 63) for _ in range(K)]
    ac = [rng.randint(-64, 63) for _ in range(K)]
    exp = sum(ws[i] * ac[i] for i in range(K)) & 0xFFFFFFFF
    return ws, ac, exp


def gen_test_fp16_os(K, seed):
    """FP16 OS: K-element dot product (FP32 mixed-precision accumulation)."""
    rng = random.Random(seed)
    choices = [0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 0.25, -0.25]
    ws = [float_to_fp16(rng.choice(choices)) for _ in range(K)]
    ac = [float_to_fp16(rng.choice(choices)) for _ in range(K)]
    exp_f = sum(fp16_to_float(ws[i]) * fp16_to_float(ac[i]) for i in range(K))
    exp_u32 = float_to_fp32_word(exp_f)
    return ws, ac, exp_u32

def gen_test_fp16_ws(K, seed):
    """FP16 WS: full K-element dot product accumulated in FP32.
    Current RTL uses pe_top.ws_acc and flushes a single accumulated FP32 word.
    """
    rng = random.Random(seed)
    choices = [0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 0.25, -0.25]
    ws = [float_to_fp16(rng.choice(choices)) for _ in range(K)]
    ac = [float_to_fp16(rng.choice(choices)) for _ in range(K)]
    exp_f = sum(fp16_to_float(ws[i]) * fp16_to_float(ac[i]) for i in range(K))
    exp_u32 = float_to_fp32_word(float(exp_f))
    return ws, ac, exp_u32


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def generate_for_N(N, out_dir):
    K = N   # K = array width (N)
    sub_dir = os.path.join(out_dir, f'N{N}')
    os.makedirs(sub_dir, exist_ok=True)

    tests = []
    addr  = BASE_ADDR
    seed0 = N * 1000

    # ---- Test 0: INT8 WS ----
    ws, ac, exp = gen_test_int8_ws(K, seed0)
    w_words = int8_pack_subw4(ws)
    a_words = int8_pack_subw4(ac)
    tests.append({
        'id': f'int8_WS_K{K}', 'dtype': 'int8', 'mode': 'WS', 'K': K,
        'ctrl': make_ctrl('int8', 'WS'), 'exp': exp,
        'w_words': w_words, 'a_words': a_words,
        'w_addr': addr, 'a_addr': addr + ADDR_GAP, 'r_addr': addr + 2*ADDR_GAP,
    })

    seed0 += 1
    addr += 3 * ADDR_GAP

    # ---- Test 1: INT8 OS ----
    ws, ac, exp = gen_test_int8_os(K, seed0)
    w_words = int8_pack_subw4(ws)
    a_words = int8_pack_subw4(ac)
    tests.append({
        'id': f'int8_OS_K{K}', 'dtype': 'int8', 'mode': 'OS', 'K': K,
        'ctrl': make_ctrl('int8', 'OS'), 'exp': exp,
        'w_words': w_words, 'a_words': a_words,
        'w_addr': addr, 'a_addr': addr + ADDR_GAP, 'r_addr': addr + 2*ADDR_GAP,
    })
    seed0 += 1
    addr += 3 * ADDR_GAP

    # ---- Test 2: FP16 WS ----
    ws, ac, exp = gen_test_fp16_ws(K, seed0)
    w_words = fp16_pack(ws)
    a_words = fp16_pack(ac)
    tests.append({

        'id': f'fp16_WS_K{K}', 'dtype': 'fp16', 'mode': 'WS', 'K': K,
        'ctrl': make_ctrl('fp16', 'WS'), 'exp': exp,
        'w_words': w_words, 'a_words': a_words,
        'w_addr': addr, 'a_addr': addr + ADDR_GAP, 'r_addr': addr + 2*ADDR_GAP,
    })
    seed0 += 1
    addr += 3 * ADDR_GAP

    # ---- Test 3: FP16 OS ----
    ws, ac, exp = gen_test_fp16_os(K, seed0)
    w_words = fp16_pack(ws)
    a_words = fp16_pack(ac)
    tests.append({
        'id': f'fp16_OS_K{K}', 'dtype': 'fp16', 'mode': 'OS', 'K': K,
        'ctrl': make_ctrl('fp16', 'OS'), 'exp': exp,
        'w_words': w_words, 'a_words': a_words,
        'w_addr': addr, 'a_addr': addr + ADDR_GAP, 'r_addr': addr + 2*ADDR_GAP,
    })
    seed0 += 1
    addr += 3 * ADDR_GAP

    num_tests = len(tests)

    # ---- Build DRAM image ----
    dram = {}
    for t in tests:
        for i, w in enumerate(t['w_words']):
            dram[(t['w_addr'] + i*4) >> 2] = w
        for i, w in enumerate(t['a_words']):
            dram[(t['a_addr'] + i*4) >> 2] = w

    # DRAM size must cover result addresses too
    max_wa = max(dram.keys()) if dram else 0
    max_r  = max((t['r_addr'] >> 2) + K for t in tests)  # WS writes K words
    max_a  = max(max_wa, max_r)
    dram_arr = [0] * (max_a + 1)
    for a, v in dram.items():
        dram_arr[a] = v

    write_hex(os.path.join(sub_dir, 'dram_init.hex'), dram_arr)

    # ---- Expected values ----
    exp_words = [t['exp'] for t in tests]
    write_hex(os.path.join(sub_dir, 'expected.hex'), exp_words)

    # ---- Verilog parameters ----
    vh = os.path.join(sub_dir, 'test_params.vh')
    with open(vh, 'w') as f:
        f.write(f'// Auto-generated for N={N} (COLS=1, K={K}), {num_tests} tests\n')
        f.write(f'`define NUM_TESTS {num_tests}\n')
        f.write(f'`define DRAM_SIZE {max(max_a + 1, 8192)}\n')
        for i, t in enumerate(tests):
            f.write(f'\n// Test {i}: {t["id"]}\n')
            f.write(f'`define TEST_{i}\n')
            f.write(f'`define T{i}_W_ADDR   32\'h{t["w_addr"]:08x}\n')
            f.write(f'`define T{i}_A_ADDR   32\'h{t["a_addr"]:08x}\n')
            f.write(f'`define T{i}_R_ADDR   32\'h{t["r_addr"]:08x}\n')
            f.write(f'`define T{i}_M_DIM    {1}\n')
            f.write(f'`define T{i}_N_DIM    {1}\n')
            f.write(f'`define T{i}_K_DIM    {t["K"]}\n')
            f.write(f'`define T{i}_CTRL     32\'h{t["ctrl"]:02x}\n')
            f.write(f'`define T{i}_EXPECTED 32\'h{t["exp"]:08x}\n')
            f.write(f'`define T{i}_IS_FP16  {1 if t["dtype"]=="fp16" else 0}\n')
            f.write(f'`define T{i}_IS_OS    {1 if t["mode"]=="OS" else 0}\n')

    print(f"  N={N:2d}: {num_tests} tests, DRAM={max_a+1} words")
    for t in tests:
        exp_str = f"0x{t['exp']:08x}"
        if t['dtype'] == 'int8':
            sv = t['exp'] if t['exp'] < 0x80000000 else t['exp'] - 0x100000000
            exp_str += f" ({sv})"
        else:
            fv = struct.unpack('<f', struct.pack('<I', t['exp']))[0]
            exp_str += f" ({fv:.4f})"
        print(f"     {t['id']:25s} K={t['K']:2d} exp={exp_str}")

    return num_tests

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Generating test data (COLS=1, SUBW=4 packing) in:\n  {out_dir}\n")
    total = 0
    for N in [4, 8, 16, 32]:
        total += generate_for_N(N, out_dir)
    print(f"\nTotal: {total} tests across 4 K-depth configurations")

if __name__ == '__main__':
    main()
