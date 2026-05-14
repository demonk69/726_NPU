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
            pw = packed_pad_words(grid_rows * 1, k_len)
            for _ in range(pw):
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
    SIMD_LANES = 4
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
    SIMD_LANES = 4
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

def gen_expected(C, M, N):
    """Flatten C to row-major 32-bit words (expected order)."""
    flat = []
    for m in range(M):
        for n in range(N):
            flat.append(C[m][n])
    return flat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True, choices=["4x4", "8x8", "16x16", "8x32"])
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    cfg = SHAPE_CONFIGS[args.shape]
    grid_rows = cfg["grid_rows"]
    grid_cols = cfg["grid_cols"]
    kt_elems = compute_k_tile_elems(cfg, is_8x32_half=(args.shape == "8x32"))
    M = args.M
    K = args.K
    N = args.N

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

    C = gen_golden(A, W, M, K, N)

    # DRAM layout:
    #   0x0000: packed W tile stream (split into two halves for 8x32)
    #   0x1000: packed A tile stream
    #   0x2000: result area (row-major C)
    W_BASE = 0x0000
    A_BASE = 0x1000
    R_BASE = 0x2000
    DRAM_SIZE = 0x4000 // 4  # 16K words

    # Effective tile lanes for A PPBuf = shape_tile_lanes(shape)
    # 8x32: 16, 16x16: 16, 8x8: 8, 4x4: 4
    a_eff_lanes = 16 if args.shape == "8x32" else grid_rows
    a_tile = pack_tile_a_ksplit(A, M, K, grid_rows, kt_elems, a_eff_lanes)
    if args.shape == "8x32":
        w_tile0 = pack_tile_w_8x32_pass(W, K, N, 0, kt_elems)  # cols 0-15
        w_tile1 = pack_tile_w_8x32_pass(W, K, N, 1, kt_elems)  # cols 16-31
        w_tile = w_tile0 + w_tile1
        W_ADDR2 = W_BASE + len(w_tile0) * 4  # second pass starts after first
    else:
        w_tile = pack_tile_w_ksplit(W, K, N, grid_cols, kt_elems)
        W_ADDR2 = 0  # unused
    expected = gen_expected(C, M, N)

    dram = [0] * DRAM_SIZE
    for i, v in enumerate(w_tile):
        dram[(W_BASE >> 2) + i] = v
    for i, v in enumerate(a_tile):
        dram[(A_BASE >> 2) + i] = v

    num_tiles_m = (M + grid_rows - 1) // grid_rows
    num_tiles_n = (N + grid_cols - 1) // grid_cols

    # Write test_params.vh
    vh_path = out_dir / "test_params.vh"
    with open(vh_path, "w") as f:
        f.write(f"// Auto-generated multi-shape test case: {case_name}\n")
        f.write(f"`define TEST_NAME \"{case_name}\"\n")
        f.write(f"`define DRAM_HEX \"{out_dir.as_posix()}/dram.hex\"\n")
        f.write(f"`define EXPECTED_HEX \"{out_dir.as_posix()}/expected.hex\"\n")
        f.write(f"`define DRAM_SIZE {DRAM_SIZE}\n")
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

    # Write dram.hex
    dram_path = out_dir / "dram.hex"
    with open(dram_path, "w") as f:
        for i, v in enumerate(dram[:16384]):
            f.write(f"{v:08X}\n")

    # Write expected.hex (row-major C)
    exp_path = out_dir / "expected.hex"
    with open(exp_path, "w") as f:
        for v in expected:
            f.write(f"{v & 0xFFFFFFFF:08X}\n")

    print(f"Generated {case_name}: M={M} K={K} N={N} grid={args.shape}")
    print(f"  num_results = {M * N}")
    print(f"  tiles M x N = {num_tiles_m} x {num_tiles_n}")
    print(f"  dram.hex, expected.hex, test_params.vh -> {out_dir}")

if __name__ == "__main__":
    main()
