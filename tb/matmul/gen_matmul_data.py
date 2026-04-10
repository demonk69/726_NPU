# =============================================================================
# gen_matmul_data.py - Generate matrix multiplication test data
#
# Generates A[M×K] × B[K×N] = C[M×N] test data for the NPU tile-loop
# architecture.
#
# DRAM layout:
#   W_ADDR:  B matrix, column-major (col 0, col 1, ..., col N-1)
#            Each column K elements. Column j starts at W_ADDR + j*K*elem_bytes.
#   A_ADDR:  A matrix, row-major (row 0, row 1, ..., row M-1)
#            Each row K elements. Row i starts at A_ADDR + i*K*elem_bytes.
#   R_ADDR:  C matrix, row-major (row 0, row 1, ..., row M-1)
#            Each row N × 32-bit words. C[i][j] at R_ADDR + (i*N+j)*4.
#
# Packing:
#   INT8: 4 elements per 32-bit word (SUBW=4, OUT_WIDTH=8)
#   FP16: 2 elements per 32-bit word (byte packer reassembles)
#
# Output:
#   <test_dir>/dram_init.hex  - DRAM initialization
#   <test_dir>/expected.hex   - Expected C matrix (M×N words, row-major)
#   <test_dir>/test_params.vh  - Verilog parameters
# =============================================================================

import struct, os, math, random, sys

# ---------------------------------------------------------------------------
# Packing helpers
# ---------------------------------------------------------------------------
def int8_pack(vals):
    """Pack INT8 values 4-per-word (SUBW=4)."""
    words = []
    for i in range(0, len(vals), 4):
        w = 0
        for j in range(4):
            if i+j < len(vals):
                w |= (vals[i+j] & 0xFF) << (j*8)
        words.append(w)
    return words

def fp16_pack(vals):
    """Pack FP16 values 2-per-word."""
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
    if e == 31 and f == 0: return float('-inf') if s else float('inf')
    if e == 31: return float('nan')
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
    """bit0=start, [3:2]=dtype(00=INT8,10=FP16), [5:4]=stat(00=WS,01=OS)"""
    dtype_bits = 2 if dtype == 'fp16' else 0
    stat_bits  = 1 if mode == 'OS' else 0
    return (stat_bits << 4) | (dtype_bits << 2) | 1

# ---------------------------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------------------------
def gen_int8_matmul(M, K, N, seed):
    """Generate INT8 matrices A[M×K] and B[K×N], compute C=A×B."""
    rng = random.Random(seed)
    # A: M rows × K cols, values in [-64, 63]
    A = [[rng.randint(-64, 63) for _ in range(K)] for _ in range(M)]
    # B: K rows × N cols, column-major storage means we store B^T effectively
    B = [[rng.randint(-64, 63) for _ in range(N)] for _ in range(K)]
    # C = A × B
    C = []
    for i in range(M):
        row = []
        for j in range(N):
            val = sum(A[i][k] * B[k][j] for k in range(K))
            row.append(val & 0xFFFFFFFF)  # 32-bit unsigned
        C.append(row)
    return A, B, C

def gen_fp16_matmul(M, K, N, seed):
    """Generate FP16 matrices A[M×K] and B[K×N], compute C=A×B (FP32 accum)."""
    rng = random.Random(seed)
    choices = [0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 0.25, -0.25]
    # A: FP16 values
    A = [[float_to_fp16(rng.choice(choices)) for _ in range(K)] for _ in range(M)]
    # B: FP16 values
    B = [[float_to_fp16(rng.choice(choices)) for _ in range(N)] for _ in range(K)]
    # C = A × B with FP32 accumulation
    C = []
    for i in range(M):
        row = []
        for j in range(N):
            # FP32 accumulation: sum of FP16 products
            acc = 0.0  # Python float = FP64, good enough
            for k in range(K):
                af = fp16_to_float(A[i][k])
                bf = fp16_to_float(B[k][j])
                acc += af * bf
            row.append(float_to_fp32_word(acc))
        C.append(row)
    return A, B, C

# ---------------------------------------------------------------------------
# Build DRAM image for matrix multiplication
# ---------------------------------------------------------------------------
def build_dram(A, B, C, M, K, N, dtype, w_base, a_base, r_base, b_col_stride=0, a_row_stride=0):
    """Build DRAM image with A, B matrices and space for C results.
    b_col_stride: byte offset between B columns (0 = auto-compute K*elem_bytes).
    a_row_stride: byte offset between A rows (0 = auto-compute K*elem_bytes).
    """
    elem_bytes = 1 if dtype == 'int8' else 2
    pack_fn = int8_pack if dtype == 'int8' else fp16_pack

    if b_col_stride == 0: b_col_stride = K * elem_bytes
    if a_row_stride == 0: a_row_stride = K * elem_bytes

    dram = {}  # word_addr -> word_value

    # B matrix: column-major
    # Column j: B[0][j], B[1][j], ..., B[K-1][j]
    # Stored at w_base + j * b_col_stride
    for j in range(N):
        col_data = [B[k][j] for k in range(K)]
        packed = pack_fn(col_data)
        col_start = w_base + j * b_col_stride
        for i, w in enumerate(packed):
            dram[(col_start + i*4) >> 2] = w

    # A matrix: row-major
    # Row i: A[i][0], A[i][1], ..., A[i][K-1]
    # Stored at a_base + i * a_row_stride
    for i in range(M):
        row_data = A[i]
        packed = pack_fn(row_data)
        row_start = a_base + i * a_row_stride
        for idx, w in enumerate(packed):
            dram[(row_start + idx*4) >> 2] = w

    # C matrix: row-major (will be written by NPU)
    # C[i][j] at r_base + (i*N + j) * 4
    # Initialize to 0 for debugging
    for i in range(M):
        for j in range(N):
            addr = r_base + (i*N + j) * 4
            dram[addr >> 2] = 0

    return dram

# ---------------------------------------------------------------------------
# Generate test parameters
# ---------------------------------------------------------------------------
def generate(M, K, N, dtype, mode, seed, out_dir, test_id):
    """Generate one matrix multiplication test."""
    elem_bytes = 1 if dtype == 'int8' else 2

    # Address layout (word-aligned: 4-byte boundary)
    w_base = 0x10000
    # B column stride = DMA bytes for K elements (word-aligned)
    # INT8: ceil(K/4)*4; FP16: ceil(K/2)*4
    b_col_bytes = ((K * elem_bytes + 3) >> 2) << 2
    a_base = w_base + b_col_bytes * N + 0x100  # word-aligned
    # A row stride = same formula
    a_row_bytes = ((K * elem_bytes + 3) >> 2) << 2
    r_base = a_base + a_row_bytes * M + 0x100  # word-aligned

    # Generate matrices
    if dtype == 'int8':
        A, B, C = gen_int8_matmul(M, K, N, seed)
    else:
        A, B, C = gen_fp16_matmul(M, K, N, seed)

    # Word-aligned stride for B columns and A rows
    b_col_stride = b_col_bytes   # bytes between B columns (word-aligned DMA len)
    a_row_stride = a_row_bytes   # bytes between A rows (word-aligned DMA len)

    # Build DRAM (with strides)
    dram = build_dram(A, B, C, M, K, N, dtype, w_base, a_base, r_base, b_col_stride, a_row_stride)

    # Flatten C for expected output
    expected = []
    for i in range(M):
        for j in range(N):
            expected.append(C[i][j])

    # Compute DRAM array
    max_addr = max(dram.keys()) if dram else 0
    dram_arr = [0] * (max_addr + 1)
    for a, v in dram.items():
        dram_arr[a] = v

    # Create output directory
    sub_dir = os.path.join(out_dir, test_id)
    os.makedirs(sub_dir, exist_ok=True)

    # Write files
    write_hex(os.path.join(sub_dir, 'dram_init.hex'), dram_arr)
    write_hex(os.path.join(sub_dir, 'expected.hex'), expected)

    ctrl = make_ctrl(dtype, mode)
    num_results = M * N

    # Write Verilog parameters
    vh = os.path.join(sub_dir, 'test_params.vh')
    with open(vh, 'w') as f:
        f.write(f'// Auto-generated: {test_id} ({dtype} {mode})\n')
        f.write(f'// C = A[{M}x{K}] x B[{K}x{N}] = C[{M}x{N}]\n')
        f.write(f'`define NUM_RESULTS {num_results}\n')
        f.write(f'`define M_DIM {M}\n')
        f.write(f'`define N_DIM {N}\n')
        f.write(f'`define K_DIM {K}\n')
        f.write(f'`define W_ADDR 32\'h{w_base:08x}\n')
        f.write(f'`define A_ADDR 32\'h{a_base:08x}\n')
        f.write(f'`define R_ADDR 32\'h{r_base:08x}\n')
        f.write(f'`define CTRL   32\'h{ctrl:02x}\n')
        f.write(f'`define DRAM_SIZE {max(max_addr + 1, 8192)}\n')
        f.write(f'`define IS_FP16 {1 if dtype == "fp16" else 0}\n')
        f.write(f'`define IS_OS   {1 if mode == "OS" else 0}\n')

    # Print summary
    print(f"  {test_id:30s} ({dtype:4s} {mode:2s}) M={M} K={K} N={N}")
    print(f"    W_BASE=0x{w_base:08x}  A_BASE=0x{a_base:08x}  R_BASE=0x{r_base:08x}")
    print(f"    DRAM={max_addr+1} words, {num_results} expected results")
    # Print a few expected values
    for i in range(min(M, 3)):
        for j in range(min(N, 4)):
            idx = i * N + j
            val = expected[idx]
            if dtype == 'int8':
                sv = val if val < 0x80000000 else val - 0x100000000
                print(f"    C[{i}][{j}] = {sv}", end="")
            else:
                fv = struct.unpack('<f', struct.pack('<I', val))[0]
                print(f"    C[{i}][{j}] = {fv:.4f}", end="")
        print()

    return sub_dir

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate NPU matmul test data')
    parser.add_argument('--square', action='store_true', help='Generate square matrix tests only')
    parser.add_argument('--ws', action='store_true', help='Generate WS mode tests only')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    seed_base = 20260407

    print(f"Generating matrix multiplication test data in:\n  {out_dir}\n")

    if args.ws:
        # WS mode square matrix tests
        print("=== WS Square Matrix Tests ===")
        print("\n--- WS INT8 Square ---")
        generate(4, 4, 4, 'int8', 'WS', seed_base+3000, out_dir, 'ws_sq_int8_4x4')
        generate(8, 8, 8, 'int8', 'WS', seed_base+3001, out_dir, 'ws_sq_int8_8x8')
        generate(16,16,16, 'int8', 'WS', seed_base+3002, out_dir, 'ws_sq_int8_16x16')
        print("\n--- WS FP16 Square ---")
        generate(4, 4, 4, 'fp16', 'WS', seed_base+4000, out_dir, 'ws_sq_fp16_4x4')
        generate(8, 8, 8, 'fp16', 'WS', seed_base+4001, out_dir, 'ws_sq_fp16_8x8')
    elif args.square:
        # Square matrix tests: INT8 4x4, 8x8, 16x16 and FP16 4x4, 8x8
        print("=== Square Matrix Tests ===")
        print("\n--- INT8 Square ---")
        generate(4, 4, 4, 'int8', 'OS', seed_base+1000, out_dir, 'sq_int8_4x4')
        generate(8, 8, 8, 'int8', 'OS', seed_base+1001, out_dir, 'sq_int8_8x8')
        generate(16,16,16, 'int8', 'OS', seed_base+1002, out_dir, 'sq_int8_16x16')
        print("\n--- FP16 Square ---")
        generate(4, 4, 4, 'fp16', 'OS', seed_base+2000, out_dir, 'sq_fp16_4x4')
        generate(8, 8, 8, 'fp16', 'OS', seed_base+2001, out_dir, 'sq_fp16_8x8')
    else:
        # OS mode tests (verify first)
        print("=== OS Mode Tests ===")
        generate(2, 3, 2, 'int8', 'OS', seed_base,     out_dir, 'os_int8_2x3x2')
        generate(3, 4, 3, 'int8', 'OS', seed_base+100,  out_dir, 'os_int8_3x4x3')
        generate(2, 4, 3, 'int8', 'OS', seed_base+200,  out_dir, 'os_int8_2x4x3')
        generate(2, 3, 2, 'fp16', 'OS', seed_base+300,  out_dir, 'os_fp16_2x3x2')
        generate(3, 4, 3, 'fp16', 'OS', seed_base+400,  out_dir, 'os_fp16_3x4x3')

        print()
        print("=== WS Mode Tests ===")
        generate(2, 3, 2, 'int8', 'WS', seed_base+500,  out_dir, 'ws_int8_2x3x2')
        generate(3, 4, 3, 'int8', 'WS', seed_base+600,  out_dir, 'ws_int8_3x4x3')
        generate(2, 3, 2, 'fp16', 'WS', seed_base+700,  out_dir, 'ws_fp16_2x3x2')

    print(f"\nDone. Tests generated in: {out_dir}")

if __name__ == '__main__':
    main()
