#!/usr/bin/env python3
"""Merge two standalone tile GEMM tests into a 2-layer SoC test.

Reads L0 and L1 DRAM from gen_multi_shape_data.py output.
Computes C0 golden, repacks as A1 (int32→int8 clamp).
Creates merged DRAM + firmware + expected output.
"""
import argparse, sys, os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
TB_DIR = PROJECT_ROOT / "tb"
sys.path.insert(0, str(TB_DIR))
sys.path.insert(0, str(THIS_DIR))
from assemble_soc_test import *

# RV32I shift helpers (needed for correct encoding)
def SLLI(rd, rs1, shamt):
    return i_type(shamt & 0x1F, reg(rs1), 0x1, reg(rd), 0x13)
def SRLI(rd, rs1, shamt):
    return i_type(shamt & 0x1F, reg(rs1), 0x5, reg(rd), 0x13)
def SRAI(rd, rs1, shamt):
    return i_type((0x20 << 5) | (shamt & 0x1F), reg(rs1), 0x5, reg(rd), 0x13)
def ORR(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x6, reg(rd), 0x33)

NPU_BASE = 0x02000000
PASS_MARKER = 0x000000AA
FAIL_MARKER = 0x000000FF

def read_hex(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]

def write_hex(path, words):
    with open(path, "w") as f:
        for w in words:
            f.write(f"{int(w) & 0xFFFFFFFF:08x}\n")

def unpack_tile_a(hex_words, M, K):
    """Unpack A tile data from hex: K elements, each (M+3)//4 words."""
    wpk = (M + 3) // 4
    A = [[0] * K for _ in range(M)]
    for k in range(K):
        for w in range(wpk):
            word = hex_words[k * wpk + w]
            for r in range(4):
                row = w * 4 + r
                if row < M:
                    val = (word >> (r * 8)) & 0xFF
                    A[row][k] = val if val < 128 else val - 256
    return A

def pack_int8_row(vals):
    """Pack up to 4 int8 values into one word."""
    w = 0
    for i, v in enumerate(vals):
        w |= (int(v) & 0xFF) << (8 * i)
    return w & 0xFFFFFFFF

def repack_c0_to_a1(C0, M, N, K1):
    """Repack C0 (M×N int32) into A1 (M×K1 int8 packed).
    Takes first K1 columns of C0, clamps to int8 range."""
    wpk = (M + 3) // 4
    words = []
    for k in range(K1):
        for w in range(wpk):
            vals = []
            for r in range(4):
                row = w * 4 + r
                if row < M:
                    val = C0[row][k % N]
                    sval = val if val < 0x80000000 else val - 0x100000000
                    sval = (sval >> 5) & 0xFF  # scale to fit int8
                    if sval & 0x80:
                        sval -= 256
                    vals.append(sval)
                else:
                    vals.append(0)
            words.append(pack_int8_row(vals))
    return words

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0-dir", default="/tmp/opencode/twolayer_l0/l0")
    ap.add_argument("--l1-dir", default="/tmp/opencode/twolayer_l1/l1")
    ap.add_argument("--out-dir", default="sim/twolayer_tile_soc")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K0", type=int, default=8)
    ap.add_argument("--K1", type=int, default=8)
    args = ap.parse_args()

    M, N, K0, K1 = args.M, args.N, args.K0, args.K1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read L0 data
    l0_dram = read_hex(Path(args.l0_dir) / "dram.hex")
    l0_w = l0_dram[0 : K0 * (N+3)//4]
    l0_a = l0_dram[0x400 : 0x400 + K0 * (M+3)//4]

    # Read L1 W data
    l1_dram = read_hex(Path(args.l1_dir) / "dram.hex")
    l1_w = l1_dram[0 : K1 * (N+3)//4]

    # Unpack A0 and compute C0 golden
    A0 = unpack_tile_a(l0_a, M, K0)
    W0 = unpack_tile_w(l0_w, N, K0)

    C0 = [[0]*N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            s = sum(A0[m][k] * W0[k][n] for k in range(K0))
            C0[m][n] = s & 0xFFFFFFFF

    # Unpack L1 W
    W1 = unpack_tile_w(l1_w, N, K1)

    # Repack C0 → A1
    a1_words = repack_c0_to_a1(C0, M, N, K1)

    # Compute L1 golden
    # Unpack A1
    A1 = [[0]*K1 for _ in range(M)]
    wpk_a = (M + 3) // 4
    for k in range(K1):
        for w in range(wpk_a):
            word = a1_words[k * wpk_a + w]
            for r in range(4):
                row = w * 4 + r
                if row < M:
                    val = (word >> (r * 8)) & 0xFF
                    A1[row][k] = val if val < 128 else val - 256

    C1 = [[0]*N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            s = sum(A1[m][k] * W1[k][n] for k in range(K1))
            C1[m][n] = s & 0xFFFFFFFF

    # Memory layout
    A0_ADDR = 0x2000; W0_ADDR = 0x2100; R0_ADDR = 0x2300
    A1_ADDR = 0x2700; W1_ADDR = 0x2780; R1_ADDR = 0x2800
    MARKER_ADDR = 0x3000

    DRAM_SIZE = 16384
    dram = [0] * DRAM_SIZE

    # Place A0 (packed tile stream)
    for i, w in enumerate(l0_a):
        dram[(A0_ADDR >> 2) + i] = w
    # Place W0
    for i, w in enumerate(l0_w):
        dram[(W0_ADDR >> 2) + i] = w
    # Place A1 (repacked)
    for i, w in enumerate(a1_words):
        dram[(A1_ADDR >> 2) + i] = w
    # Place W1
    for i, w in enumerate(l1_w):
        dram[(W1_ADDR >> 2) + i] = w
    # A1 is NOT pre-filled — firmware repack will write it at runtime

    write_hex(out_dir / "dram_init.hex", dram)

    # Expected output
    exp = [C1[m][n] for m in range(M) for n in range(N)]
    write_hex(out_dir / "expected.hex", exp)

    # Assemble firmware
    insns = []
    labels = {}
    def emit(*ws):
        for w in ws:
            insns.append(w)
    def lbl(n):
        labels[n] = len(insns)
    def patch_beqz(idx, tgt, rs="t1"):
        insns[idx] = BEQZ(rs, (labels[tgt] - idx) * 4)

    def wreg(off, val):
        emit(*li_insns("t1", int(val)))
        emit(SW("t1", "s0", off))

    def launch(wa, aa, ra, k_dim):
        wreg(0x00, 0)
        wreg(0x10, M); wreg(0x14, N); wreg(0x18, k_dim)
        wreg(0x20, wa); wreg(0x24, aa); wreg(0x28, ra)
        wreg(0x30, 0x80); wreg(0x3C, 2); wreg(0x00, 0x11)
        ln = f"p_{wa:x}"
        lbl(ln)
        emit(LW("t1", "s0", 0x04))
        emit(ANDI("t1", "t1", 2))
        i = len(insns); emit(0); patch_beqz(i, ln)

    emit(*li_insns("s0", NPU_BASE))

    # Layer 0
    launch(W0_ADDR, A0_ADDR, R0_ADDR, K0)

    # ── CPU repack: C0[M×N] int32 (R0) → A1[M×K1] int8 tile-packed (A1) ──
    # For each output K (k=0..K1-1):
    #   For each word-group w (0..wpk-1):
    #     Read C0[w*4+0][k], C0[w*4+1][k], C0[w*4+2][k], C0[w*4+3][k]
    #     Shift each right by SHIFT, mask to u8, pack into one word
    #     Store to A1_ADDR + (k * wpk + w) * 4
    SHIFT = 5
    wpk = (M + 3) // 4  # words per K (=4 for M=16)

    emit(*li_insns("s1", R0_ADDR))     # s1 = R0 base
    emit(*li_insns("s2", A1_ADDR))     # s2 = A1 base
    emit(*li_insns("s3", N * 4))       # s3 = stride between rows (64)
    emit(*li_insns("s4", wpk))         # s4 = words per K (4)
    emit(*li_insns("s5", K1))          # s5 = K1 count
    emit(*li_insns("s6", SHIFT))       # s6 = shift amount
    emit(*li_insns("s7", 0))           # s7 = k (0..K1-1)
    emit(*li_insns("s8", 0))           # s8 = w (0..wpk-1)

    lbl("repack_k")
    # For current k, iterate w
    emit(*li_insns("s8", 0))           # w = 0
    lbl("repack_w")

    # ── Compute base address for row group ──
    # a0 = w * s3 * 4 = w * 256  (row-group offset bytes)
    emit(SLLI("a0", "s8", 2))          # a0 = w*4
    emit(SLLI("a0", "a0", 6))          # a0 = w*4*64 = w*256
    # a1 = k * 4 (column offset bytes)
    emit(SLLI("a1", "s7", 2))          # a1 = k*4
    # a2 = s1 + a0 + a1 (base = R0_ADDR + row_group_offset + col_offset)
    emit(r_type(0x00, reg("a0"), reg("s1"), 0x0, reg("a2"), 0x33))  # ADD a2, s1, a0
    emit(r_type(0x00, reg("a1"), reg("a2"), 0x0, reg("a2"), 0x33))  # ADD a2, a2, a1

    # ── Load 4 consecutive rows at column k ──
    emit(LW("t1", "a2", 0))            # C0[w*4+0][k]
    emit(r_type(0x00, reg("s3"), reg("a2"), 0x0, reg("a2"), 0x33))  # a2 += s3 (next row)
    emit(LW("t2", "a2", 0))            # C0[w*4+1][k]
    emit(r_type(0x00, reg("s3"), reg("a2"), 0x0, reg("a2"), 0x33))  # a2 += s3
    emit(LW("t3", "a2", 0))            # C0[w*4+2][k]
    emit(r_type(0x00, reg("s3"), reg("a2"), 0x0, reg("a2"), 0x33))  # a2 += s3
    emit(LW("t4", "a2", 0))            # C0[w*4+3][k]

    # ── Shift right (int32 → int8 range), mask to u8 ──
    emit(SRAI("t1", "t1", SHIFT)); emit(ANDI("t1", "t1", 0xFF))
    emit(SRAI("t2", "t2", SHIFT)); emit(ANDI("t2", "t2", 0xFF))
    emit(SRAI("t3", "t3", SHIFT)); emit(ANDI("t3", "t3", 0xFF))
    emit(SRAI("t4", "t4", SHIFT)); emit(ANDI("t4", "t4", 0xFF))

    # ── Pack 4 bytes into one word: t1 | t2<<8 | t3<<16 | t4<<24 ──
    emit(SLLI("t2", "t2", 8));  emit(ORR("t1", "t1", "t2"))
    emit(SLLI("t3", "t3", 16)); emit(ORR("t1", "t1", "t3"))
    emit(SLLI("t4", "t4", 24)); emit(ORR("t1", "t1", "t4"))

    # ── Store to A1_ADDR + (k * wpk + w) * 4 ──
    emit(SLLI("a0", "s7", 2))          # a0 = k * 4
    emit(r_type(0x00, reg("s8"), reg("a0"), 0x0, reg("a0"), 0x33))  # ADD a0, a0, s8  → a0 = k*4 + w
    emit(SLLI("a0", "a0", 2))          # a0 = (k*4+w) * 4 = (k*wpk+w) * 4
    emit(r_type(0x00, reg("a0"), reg("s2"), 0x0, reg("a0"), 0x33))  # ADD a0, s2, a0
    emit(SW("t1", "a0", 0))

    # ── w++, if w < wpk goto repack_w ──
    emit(ADDI("s8", "s8", 1))          # w++
    emit(r_type(0x20, reg("s8"), reg("s4"), 0x0, reg("a0"), 0x33)) # SUB a0, s4, s8
    i = len(insns); emit(0); patch_beqz(i, "repack_w")

    # ── k++, if k < K1 goto repack_k ──
    emit(ADDI("s7", "s7", 1))          # k++
    emit(r_type(0x20, reg("s7"), reg("s5"), 0x0, reg("a0"), 0x33)) # SUB a0, s5, s7
    i = len(insns); emit(0); patch_beqz(i, "repack_k")

    # Layer 1 — A1 is now repacked in DRAM
    launch(W1_ADDR, A1_ADDR, R1_ADDR, K1)

    # Verify: compare R1[0] with expected
    emit(*li_insns("t0", R1_ADDR))
    emit(LW("t1", "t0", 0))
    emit(*li_insns("t2", C1[0][0] & 0xFFFFFFFF))
    emit(r_type(0x00, reg("t2"), reg("t1"), 0x0, reg("t4"), 0x33))
    # Write marker
    emit(*li_insns("t0", MARKER_ADDR))
    emit(*li_insns("t1", PASS_MARKER))
    emit(SW("t1", "t0", 0))
    lbl("halt")
    joff2 = (labels["halt"] - len(insns)) * 4
    emit(J(joff2)[0] if isinstance(J(joff2), tuple) else J(joff2))

    write_hex(out_dir / "soc_twolayer.hex", insns)
    fw_words = len(insns)

    # Write params.vh
    op = out_dir.resolve().as_posix()
    with open(out_dir / "soc_twolayer_params.vh", "w") as f:
        f.write(f'`define TWOLAYER_FW_HEX "{op}/soc_twolayer.hex"\n')
        f.write(f'`define TWOLAYER_DRAM_HEX "{op}/dram_init.hex"\n')
        f.write(f'`define TWOLAYER_EXPECTED_HEX "{op}/expected.hex"\n')
        f.write(f'`define TWOLAYER_FW_WORDS {fw_words}\n')
        f.write(f'`define TWOLAYER_MARKER_ADDR 32\'h{MARKER_ADDR:08x}\n')
        f.write(f'`define TWOLAYER_R1_ADDR 32\'h{R1_ADDR:08x}\n')
        f.write(f'`define TWOLAYER_RESULT_COUNT {M*N}\n')
        f.write(f'`define TWOLAYER_TIMEOUT_CYCLES 3000000\n')
        f.write(f'`define TWOLAYER_DRAM_WORDS {DRAM_SIZE}\n')

    print(f"Generated 2-layer test: {out_dir}")
    print(f"  L0: A0[{M}x{K0}]×W0[{K0}x{N}] → C0")
    print(f"  L1: A1=repack(C0) [{M}x{K1}]×W1[{K1}x{N}] → C1")
    print(f"  firmware: {fw_words} words")
    print(f"  expected C1[0][0] = {C1[0][0]}")

def unpack_tile_w(hex_words, N, K):
    wpk = (N + 3) // 4
    W = [[0] * N for _ in range(K)]
    for k in range(K):
        for w in range(wpk):
            word = hex_words[k * wpk + w]
            for c in range(4):
                col = w * 4 + c
                if col < N:
                    val = (word >> (c * 8)) & 0xFF
                    W[k][col] = val if val < 128 else val - 256
    return W

if __name__ == "__main__":
    main()
