#!/usr/bin/env python3
# =============================================================================
# gen_multi_shape_data.py — Generate NPU tile-mode test data for 8x8/16x16/8x32.
#
# Usage:
#   python gen_multi_shape_data.py --shape 8x8 --M 8 --K 4 --N 8 --out-dir out
#   python gen_multi_shape_data.py --shape 16x16 --M 16 --K 2 --N 16
#   python gen_multi_shape_data.py --shape 8x32 --M 8 --K 2 --N 32
#   python gen_multi_shape_data.py --shape 8x8 --M 9 --K 4 --N 5  # boundary
# =============================================================================

import argparse
import struct
import os
from pathlib import Path

SHAPE_CONFIGS = {
    "4x4":  {"cfg_shape": 0, "grid_rows": 4, "grid_cols": 4},
    "8x8":  {"cfg_shape": 1, "grid_rows": 8, "grid_cols": 8},
    "16x16":{"cfg_shape": 2, "grid_rows": 16, "grid_cols": 16},
    "8x32": {"cfg_shape": 3, "grid_rows": 8, "grid_cols": 32},
}

PPB_DEPTH = 64       # PPBuf word depth
SIMD_LANES = 4

def align_up(v, align):
    return ((v + align - 1) // align) * align

def estimate_timeout_cycles(M, K, N, grid_rows, grid_cols, kt_elems):
    num_tiles_m = (M + grid_rows - 1) // grid_rows
    num_tiles_n = (N + grid_cols - 1) // grid_cols
    num_tiles_k = (K + kt_elems - 1) // kt_elems
    tile_work = num_tiles_m * num_tiles_n * num_tiles_k
    per_tile_budget = kt_elems + grid_rows + grid_cols + 256
    return max(1_000_000, tile_work * per_tile_budget + M * N * 4 + 100_000)

def compute_k_tile_elems(shape_cfg, is_8x32_half=False):
    """Max K elements per k_tile matching controller logic.
       For 8x32 half-pass, W uses only 16 columns."""
    rows = shape_cfg["grid_rows"]
    cols = shape_cfg["grid_cols"]
    bytes_a = rows * 1  # INT8
    bytes_w = (16 if is_8x32_half else cols) * 1
    max_bytes = max(bytes_a, bytes_w)
    return max(1, (PPB_DEPTH * 4) // max_bytes)

def packed_pad_words(elem_bytes_per_k, k_len):
    """Zero-word pad count when k_len not multiple of SIMD_LANES."""
    k_rem = k_len % SIMD_LANES
    if k_rem == 0:
        return 0
    pad_bytes = elem_bytes_per_k * (SIMD_LANES - k_rem)
    return (pad_bytes + 3) // 4

def pack_tile_a_ksplit(A, M, K, grid_rows, kt_elems, eff_lanes=0):
    """Pack A into tile-stream with K-split support.
       eff_lanes: PPBuf output lane count (shape_tile_lanes), used to pad words_per_k."""
    words_per_k = max((grid_rows + 3) // 4, (eff_lanes + 3) // 4)
    packed = []
    num_m = (M + grid_rows - 1) // grid_rows
    for mt in range(num_m):
        m0 = mt * grid_rows
        kpos = 0
        while kpos < K:
            k_len = min(K - kpos, kt_elems)
            for k in range(kpos, kpos + k_len):
                for w in range(words_per_k):
                    word = 0
                    for r in range(4):
                        lane = w * 4 + r
                        if lane < grid_rows:
                            mr = m0 + lane
                            val = A[mr][k] if mr < M else 0
                            word |= (val & 0xFF) << (r * 8)
                    packed.append(word)
            padded_k = ((k_len + SIMD_LANES - 1) // SIMD_LANES) * SIMD_LANES
            for _ in range((padded_k - k_len) * words_per_k):
                packed.append(0)
            kpos += k_len
    return packed

def pack_tile_w_ksplit(W, K, N, grid_cols, kt_elems):
    """Pack W into tile-stream with K-split support."""
    words_per_k = (grid_cols + 3) // 4
    packed = []
    num_n = (N + grid_cols - 1) // grid_cols
    for nt in range(num_n):
        n0 = nt * grid_cols
        kpos = 0
        while kpos < K:
            k_len = min(K - kpos, kt_elems)
            for k in range(kpos, kpos + k_len):
                for w in range(words_per_k):
                    word = 0
                    for c in range(4):
                        lane = w * 4 + c
                        if lane < grid_cols:
                            nc = n0 + lane
                            val = W[k][nc] if nc < N else 0
                            word |= (val & 0xFF) << (c * 8)
                    packed.append(word)
            pw = packed_pad_words(grid_cols * 1, k_len)
            for _ in range(pw):
                packed.append(0)
            kpos += k_len
    return packed

def gen_golden(A, W, M, K, N):
    """A[M,K] * W[K,N] = C[M,N], signed INT8 x INT8 -> int32."""
    C = [[0]*N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            s = 0
            for k in range(K):
                s += A[m][k] * W[k][n]
            C[m][n] = s & 0xFFFFFFFF  # 32-bit two's complement
    return C

def pack_tile_a(A, M, K, grid_rows, min_words_per_k=1):
    """Pack A into tile-stream: A_TILE[m_tile][k][r]
       Pad to at least min_words_per_k per K so DMA load length matches PPBuf expectation.
       Also pad with zero words at end of each tile when K is not multiple of SIMD_LANES=4."""
    words_per_k = max(min_words_per_k, (grid_rows + 3) // 4)
    # packed_pad bytes per tile when K % SIMD_LANES != 0
    k_rem = K % SIMD_LANES
    packed_pad_bytes = (grid_rows * 1) * (SIMD_LANES - k_rem) if k_rem != 0 else 0
    packed_pad_words = (packed_pad_bytes + 3) // 4
    packed = []
    for m_tile in range((M + grid_rows - 1) // grid_rows):
        m0 = m_tile * grid_rows
        for k in range(K):
            for w in range(words_per_k):
                word = 0
                for r in range(4):
                    lane = w * 4 + r
                    if lane < grid_rows:
                        mr = m0 + lane
                        val = A[mr][k] if mr < M else 0
                        val_byte = val & 0xFF
                        word |= val_byte << (r * 8)
                packed.append(word)
        for _ in range(packed_pad_words):
            packed.append(0)
    return packed

def pack_tile_w(W, K, N, grid_cols):
    """Pack W into tile-stream: W_TILE[n_tile][k][c]
       Full tile data (zero-padded for edge tiles) to match DMA load size.
       Also pad with zero words at end of each tile when K is not multiple of SIMD_LANES=4."""
    words_per_k = (grid_cols + 3) // 4
    k_rem = K % SIMD_LANES
    packed_pad_bytes = (grid_cols * 1) * (SIMD_LANES - k_rem) if k_rem != 0 else 0
    packed_pad_words = (packed_pad_bytes + 3) // 4
    packed = []
    for n_tile in range((N + grid_cols - 1) // grid_cols):
        n0 = n_tile * grid_cols
        for k in range(K):
            for w in range(words_per_k):
                word = 0
                for c in range(4):
                    lane = w * 4 + c
                    if lane < grid_cols:
                        nc = n0 + lane
                        val = W[k][nc] if nc < N else 0
                        val_byte = val & 0xFF
                        word |= val_byte << (c * 8)
                packed.append(word)
        for _ in range(packed_pad_words):
            packed.append(0)
    return packed

def pack_tile_w_8x32_pass(W, K, N, pass_idx, kt_elems):
    """Pack W for 8x32 two-pass: pass 0 = logical cols 0-15, pass 1 = cols 16-31.
       Each pass produces 4 words per K (16 cols / 4 lanes per word), contiguous.
       Supports K-split with packed SIMD pad per k_tile."""
    packed = []
    num_tiles_n = (N + 31) // 32
    half_cols = 16
    words_per_k = 4
    for n_tile in range(num_tiles_n):
        n0 = n_tile * 32
        kpos = 0
        while kpos < K:
            k_len = min(K - kpos, kt_elems)
            for k in range(kpos, kpos + k_len):
                for w in range(words_per_k):
                    word = 0
                    for c in range(4):
                        lane = w * 4 + c
                        logical_c = n0 + pass_idx * half_cols + lane
                        val = W[k][logical_c] if logical_c < N else 0
                        val_byte = val & 0xFF
                        word |= val_byte << (c * 8)
                    packed.append(word)
            pw = packed_pad_words(half_cols * 1, k_len)
            for _ in range(pw):
                packed.append(0)
            kpos += k_len
    return packed

def pack_tile_w_8x32(W, K, N, kt_elems):
    """Pack W for 8x32 as per-N-tile pass0+pass1 blocks.

       This matches npu_ctrl's folded scheduling: each logical 8x32 N tile
       computes cols 0..15, then cols 16..31 before advancing to the next
       N tile.
    """
    packed = []
    num_tiles_n = (N + 31) // 32
    half_cols = 16
    words_per_k = 4
    for n_tile in range(num_tiles_n):
        n0 = n_tile * 32
        for pass_idx in (0, 1):
            kpos = 0
            while kpos < K:
                k_len = min(K - kpos, kt_elems)
                for k in range(kpos, kpos + k_len):
                    for w in range(words_per_k):
                        word = 0
                        for c in range(4):
                            lane = w * 4 + c
                            logical_c = n0 + pass_idx * half_cols + lane
                            val = W[k][logical_c] if logical_c < N else 0
                            word |= (val & 0xFF) << (c * 8)
                        packed.append(word)
                pw = packed_pad_words(half_cols * 1, k_len)
                for _ in range(pw):
                    packed.append(0)
                kpos += k_len
    return packed

def gen_int32_bias(N, seed=None):
    """Generate random signed int32 bias vector."""
    import random as _r
    rng = _r.Random(seed) if seed is not None else _r.Random()
    return [rng.randint(-2**20, 2**20) for _ in range(N)]  # reasonable range

def apply_bias(C, bias, M, N):
    """Apply per-column bias: C_bias[m][n] = C[m][n] + bias[n]."""
    out = [[0]*N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            out[m][n] = (C[m][n] + bias[n]) & 0xFFFFFFFF
    return out

def apply_relu(C, M, N):
    """ReLU: max(0, int32 value)."""
    out = [[0]*N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            v = signed32(C[m][n])
            out[m][n] = (0 if v < 0 else v) & 0xFFFFFFFF
    return out

def apply_relu6(C, M, N):
    """ReLU6: clamp to [0, 6] for int32."""
    out = [[0]*N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            v = signed32(C[m][n])
            if v < 0: v = 0
            elif v > 6: v = 6
            out[m][n] = v & 0xFFFFFFFF
    return out

def apply_quant(C, M, N, scale, shift, rounding):
    """INT8 quantize: (value * scale) >> shift, clamp [-128, 127]."""
    out = [[0]*N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            v = signed32(C[m][n])
            prod = v * scale
            if rounding and shift > 0:
                offset = (1 << (shift - 1))
                prod = prod + offset if prod >= 0 else prod + offset - 1
            shifted = prod >> shift if shift < 31 else (0 if prod >= 0 else -1)
            shifted = max(-128, min(127, shifted))
            out[m][n] = (shifted & 0xFF) | ((0xFFFFFF if shifted < 0 else 0) << 8)
    return out

def signed32(v):
    return v if v < 0x80000000 else v - 0x100000000

def gen_expected(C, M, N):
    """Flatten C to row-major 32-bit words (expected order)."""
    flat = []
    for m in range(M):
        for n in range(N):
            flat.append(C[m][n])
    return flat

def main():
    global SIMD_LANES
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True, choices=["4x4", "8x8", "16x16", "8x32"])
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--lanes", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None)
    parser.add_argument("--flow", choices=["os", "ws"], default="os")
    parser.add_argument("--bias", action="store_true", help="Add per-column int32 bias")
    parser.add_argument("--activation", choices=["none","relu","relu6"], default="none")
    parser.add_argument("--quant", action="store_true", help="Apply INT8 quantize/saturate")
    parser.add_argument("--quant-scale", type=int, default=1)
    parser.add_argument("--quant-shift", type=int, default=3)
    parser.add_argument("--quant-round", action="store_true")
    parser.add_argument("--perf-only", action="store_true", help="Skip golden C generation/checking for cycle-only runs")
    parser.add_argument("--strict-aw", action="store_true", help="Fail if observed result AW bursts differ from the model")
    parser.add_argument("--timeout-cycles", type=int, default=None, help="Override testbench global timeout cycles")
    parser.add_argument("--wait-timeout", type=int, default=None, help="Override AXI-Lite status poll timeout")
    args = parser.parse_args()
    SIMD_LANES = args.lanes

    cfg = SHAPE_CONFIGS[args.shape]
    grid_rows = cfg["grid_rows"]
    grid_cols = cfg["grid_cols"]
    kt_elems = compute_k_tile_elems(cfg, is_8x32_half=(args.shape == "8x32"))
    M = args.M
    K = args.K
    N = args.N
    data_w = 32 if SIMD_LANES == 4 else 16

    case_name = args.name or f"{args.shape}_M{M}_K{K}_N{N}"
    out_dir = Path(args.out_dir) / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    import random
    rng = random.Random(args.seed)

    # Generate random signed INT8 data
    def rng_s8():
        return rng.randint(-128, 127)
    A = [[rng_s8() for _ in range(K)] for _ in range(M)]
    W = [[rng_s8() for _ in range(N)] for _ in range(K)]

    C = None if args.perf_only else gen_golden(A, W, M, K, N)

    # ── Bias ──
    BIAS_BASE = 0x3000
    bias = None
    bias_words = []
    if args.bias:
        bias = gen_int32_bias(N, args.seed + 1000)
        bias_words = [v & 0xFFFFFFFF for v in bias]
        if C is not None:
            C = apply_bias(C, bias, M, N)
    # ── Activation ──
    ctrl_val = 0x11 if args.flow == "os" else 0x01  # INT8 + OS/WS + start
    if args.bias:
        ctrl_val |= (1 << 9)  # CTRL[9] = bias enable
    if args.activation == 'relu':
        if C is not None:
            C = apply_relu(C, M, N)
        ctrl_val |= (1 << 10)  # CTRL[11:10] = 01
    elif args.activation == 'relu6':
        if C is not None:
            C = apply_relu6(C, M, N)
        ctrl_val |= (2 << 10)  # CTRL[11:10] = 10
    # ── Quant ──
    quant_cfg = 0x00010000  # default: disabled (bit0=0)
    if args.quant:
        scale = args.quant_scale & 0xFFFF
        shift = args.quant_shift & 0xFF
        round_bit = 1 if args.quant_round else 0
        quant_cfg = (scale << 16) | (shift << 8) | (round_bit << 1) | 1  # bit0=enable
        if C is not None:
            C = apply_quant(C, M, N, args.quant_scale, args.quant_shift, args.quant_round)

    # Effective tile lanes for A PPBuf = shape_tile_lanes(shape)
    # 8x32: 16, 16x16: 16, 8x8: 8, 4x4: 4
    a_eff_lanes = 16 if args.shape == "8x32" else grid_rows
    a_tile = pack_tile_a_ksplit(A, M, K, grid_rows, kt_elems, a_eff_lanes)
    if args.shape == "8x32":
        w_tile = pack_tile_w_8x32(W, K, N, kt_elems)
    else:
        w_tile = pack_tile_w_ksplit(W, K, N, grid_cols, kt_elems)

    # DRAM layout is packed dynamically so large perf-only cases do not
    # overlap W/A/bias/result regions. Result storage is not needed in
    # perf-only because the testbench sinks write data instead of checking C.
    W_BASE_WORD = 0
    A_BASE_WORD = align_up(W_BASE_WORD + len(w_tile), 16)
    BIAS_BASE_WORD = align_up(A_BASE_WORD + len(a_tile), 16)
    R_BASE_WORD = align_up(BIAS_BASE_WORD + len(bias_words), 16)
    result_words = 0 if args.perf_only else M * N
    DRAM_SIZE = max(1, align_up(R_BASE_WORD + result_words, 16))

    W_BASE = W_BASE_WORD << 2
    A_BASE = A_BASE_WORD << 2
    BIAS_BASE = BIAS_BASE_WORD << 2
    R_BASE = R_BASE_WORD << 2
    if args.shape == "8x32":
        W_ADDR2 = W_BASE + ((K + SIMD_LANES - 1) // SIMD_LANES) * SIMD_LANES * 16
    else:
        W_ADDR2 = 0  # unused

    expected = [] if args.perf_only else gen_expected(C, M, N)

    dram = [0] * DRAM_SIZE
    for i, v in enumerate(w_tile):
        dram[(W_BASE >> 2) + i] = v
    for i, v in enumerate(a_tile):
        dram[(A_BASE >> 2) + i] = v
    if bias_words:
        for j, word in enumerate(bias_words):
            dram[(BIAS_BASE >> 2) + j] = word & 0xFFFFFFFF

    num_tiles_m = (M + grid_rows - 1) // grid_rows
    num_tiles_n = (N + grid_cols - 1) // grid_cols
    timeout_cycles = args.timeout_cycles
    if timeout_cycles is None:
        timeout_cycles = estimate_timeout_cycles(M, K, N, grid_rows, grid_cols, kt_elems)
    wait_timeout = args.wait_timeout
    if wait_timeout is None:
        wait_timeout = max(50_000, min((timeout_cycles + 7) // 8, 4_000_000_000))

    # Write test_params.vh
    vh_path = out_dir / "test_params.vh"
    with open(vh_path, "w") as f:
        f.write(f"// Auto-generated multi-shape test case: {case_name}\n")
        f.write(f"`define TEST_NAME \"{case_name}\"\n")
        f.write(f"`define DRAM_HEX \"{out_dir.as_posix()}/dram.hex\"\n")
        f.write(f"`define EXPECTED_HEX \"{out_dir.as_posix()}/expected.hex\"\n")
        f.write(f"`define DRAM_SIZE {DRAM_SIZE}\n")
        if args.perf_only:
            f.write("`define PERF_ONLY 1\n")
        if args.strict_aw:
            f.write("`define STRICT_AW_EXPECT 1\n")
        f.write(f"`define WAIT_TIMEOUT_VAL 32'd{wait_timeout}\n")
        f.write(f"`define TB_TIMEOUT_CYCLES_VAL 64'd{timeout_cycles}\n")
        f.write(f"`define DATA_W_VAL {data_w}\n")
        f.write(f"`define INT8_SIMD_LANES_VAL {SIMD_LANES}\n")
        f.write(f"`define M_DIM {M}\n")
        f.write(f"`define N_DIM {N}\n")
        f.write(f"`define K_DIM {K}\n")
        f.write(f"`define W_ADDR 32'h{W_BASE:08X}\n")
        if args.shape == "8x32":
            f.write(f"`define W_ADDR2 32'h{W_ADDR2:08X}\n")
        f.write(f"`define A_ADDR 32'h{A_BASE:08X}\n")
        f.write(f"`define R_ADDR 32'h{R_BASE:08X}\n")
        f.write(f"`define ARR_CFG 32'h{0x80:02X}\n")
        f.write(f"`define CFG_SHAPE_VAL {cfg['cfg_shape']}\n")
        f.write(f"`define GRID_COLS_VAL {grid_cols}\n")
        # Compute exact AW count: each tile row writes one or more bursts,
        # limited by BURST_MAX=16 (64 bytes). 32-bit words, 4 bytes each.
        aw_expect = 0
        BURST_MAX_BYTES = 64  # 16 beats * 4 bytes
        for mt in range(num_tiles_m):
            mr = min(M - mt * grid_rows, grid_rows)
            for nt in range(num_tiles_n):
                nc = min(N - nt * grid_cols, grid_cols)
                row_bytes = nc * 4
                bursts_per_row = (row_bytes + BURST_MAX_BYTES - 1) // BURST_MAX_BYTES
                aw_expect += mr * bursts_per_row
        f.write(f"`define AW_EXPECT_VAL {aw_expect}\n")
        f.write(f"`define GRID_ROWS {grid_rows}\n")
        f.write(f"`define GRID_COLS {grid_cols}\n")
        f.write(f"`define NUM_RESULTS {M * N}\n")
        f.write(f"`define NUM_TILES_M {num_tiles_m}\n")
        f.write(f"`define NUM_TILES_N {num_tiles_n}\n")
        f.write(f"`define OUTPUT_HEX \"{out_dir.as_posix()}/npu_output.hex\"\n")
        f.write(f"`define IS_FP16 0\n")
        if args.bias:
            f.write(f"`define BIAS_ADDR_VAL 32'h{BIAS_BASE:08X}\n")
        f.write(f"`define CTRL_VAL 32'h{ctrl_val:08X}\n")
        if args.quant:
            f.write(f"`define QUANT_CFG_VAL 32'h{quant_cfg:08X}\n")

    # Write dram.hex
    dram_path = out_dir / "dram.hex"
    with open(dram_path, "w") as f:
        for i, v in enumerate(dram):
            f.write(f"{v:08X}\n")

    # Write expected.hex (row-major C)
    exp_path = out_dir / "expected.hex"
    with open(exp_path, "w") as f:
        if expected:
            for v in expected:
                f.write(f"{v & 0xFFFFFFFF:08X}\n")
        else:
            f.write("00000000\n")

    print(f"Generated {case_name}: M={M} K={K} N={N} grid={args.shape} lanes={SIMD_LANES} data_w={data_w} perf_only={args.perf_only}")
    print(f"  num_results = {M * N}")
    print(f"  dram_words = {DRAM_SIZE}")
    print(f"  tiles M x N = {num_tiles_m} x {num_tiles_n}")
    print(f"  dram.hex, expected.hex, test_params.vh -> {out_dir}")

if __name__ == "__main__":
    main()
