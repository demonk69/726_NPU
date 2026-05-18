#!/usr/bin/env python3
"""Generate a multi-layer tile-mode SoC test (3 layers with CPU repack).

Layer 0: C0 = A0[M] × W0[N]  (K random)
Layer 1: C1 = A1(repacked C0) × W1[N]  
Layer 2: C2 = A2(repacked C1) × W2[N]

Each layer: 16×16 tile, K=8. CPU repacks between layers.
Firmware: for-loop over layers, tile launch + poll + CPU repack.

Generates: dram_init.hex, firmware hex, params.vh, expected output.
"""
import argparse, random, os, sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
TB_DIR = PROJECT_ROOT / "tb"
sys.path.insert(0, str(TB_DIR))
sys.path.insert(0, str(THIS_DIR))
from assemble_soc_test import *

NPU_BASE = 0x02000000; PASS_MARKER = 0x000000AA
REG_CTRL=0x00; REG_STATUS=0x04; REG_M_DIM=0x10; REG_N_DIM=0x14; REG_K_DIM=0x18
REG_W_ADDR=0x20; REG_A_ADDR=0x24; REG_R_ADDR=0x28
REG_ARR_CFG=0x30; REG_CFG_SHAPE=0x3C

# ── RV32I helpers ──
def SRAI(rd, rs1, shamt):
    return i_type((0x20 << 5) | (shamt & 0x1F), reg(rs1), 0x5, reg(rd), 0x13)
def SLLI(rd, rs1, shamt):
    return i_type(shamt & 0x1F, reg(rs1), 0x1, reg(rd), 0x13)
def ORR(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x6, reg(rd), 0x33)

def pack4(vals):
    w = 0
    for i, v in enumerate(vals):
        w |= (int(v) & 0xFF) << (8 * i)
    return w & 0xFFFFFFFF

def write_hex(path, words):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for w in words:
            if isinstance(w, tuple): w = w[0]
            f.write(f"{int(w) & 0xFFFFFFFF:08x}\n")

def build_mat(M, N, K, seed):
    rng = random.Random(seed)
    A = [[rng.randint(-128, 127) for _ in range(K)] for _ in range(M)]
    W = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(K)]
    C = [[sum(A[m][k] * W[k][n] for k in range(K)) & 0xFFFFFFFF for n in range(N)] for m in range(M)]
    return A, W, C

def pack_tile_stream(mat, dim1, dim2, transpose=False):
    """Pack mat into tile stream: K-first (dim2 first), dim1 elements per K.
    mat is [dim1][dim2] if transpose=False, or [dim2][dim1] if transpose=True."""
    wpk = (dim1 + 3) // 4
    out = []
    for k in range(dim2):
        for w in range(wpk):
            if transpose:
                vals = [mat[k][w*4+r] if w*4+r < dim1 else 0 for r in range(4)]
            else:
                vals = [mat[w*4+r][k] if w*4+r < dim1 else 0 for r in range(4)]
            out.append(pack4(vals))
    return out

def repack_C_to_A_next(C, M, N, Knext, shift=5):
    """Repack C[M][N] (int32 row-major) → A[M][Knext] (int8).
    Uses first Knext columns of C, shifts right by 'shift', clamps to int8."""
    A = [[0]*Knext for _ in range(M)]
    for m in range(M):
        for k in range(Knext):
            val = C[m][k % N]
            sval = val if val < 0x80000000 else val - 0x100000000
            sval = (sval >> shift) & 0xFF
            if sval & 0x80: sval -= 256
            A[m][k] = sval
    return A

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="sim/multilayer_tile_soc")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    M, N, K0, num_layers = args.M, args.N, args.K, args.layers

    # ── Generate layers ──
    layers = []
    for ly in range(num_layers):
        seed = args.seed + ly * 100
        A, W, C = build_mat(M, N, K0, seed)
        layers.append({"A": A, "W": W, "C": C, "seed": seed, "K": K0})

    # ── Compute repacked inputs and golden outputs ──
    # Layer 0: uses random A0
    # Layer i>0: uses repacked C(i-1)
    C_prev = layers[0]["C"]  # golden C0
    for ly in range(1, num_layers):
        A_repacked = repack_C_to_A_next(C_prev, M, N, K0)
        # Recompute C with repacked A and this layer's W
        W = layers[ly]["W"]
        C_new = [[sum(A_repacked[m][k] * W[k][n] for k in range(K0)) & 0xFFFFFFFF for n in range(N)] for m in range(M)]
        layers[ly]["A_repacked"] = A_repacked
        layers[ly]["C"] = C_new
        C_prev = C_new

    # ── Memory layout ──
    # W area starts at 0x2000, each layer's W: K0*N_words*4 bytes
    # A area for layer 0 at 0x4000
    # R areas follow
    wpk = (N + 3) // 4  # W words per K
    apk = (M + 3) // 4  # A words per K
    W_BYTES_PER_LAYER = K0 * wpk * 4 + 0x100  # padded
    A_BYTES_PER_LAYER = K0 * apk * 4 + 0x100
    R_BYTES_PER_LAYER = M * N * 4 + 0x100

    W_BASE = 0x0000_1000
    A_BASE = 0x0001_0000
    R_BASE = 0x0002_0000
    MARKER_ADDR = 0x0010_0000

    DRAM_SIZE = 512 * 1024  # 512K words = 2MB

    addrs = []
    for ly in range(num_layers):
        w_addr = W_BASE + ly * W_BYTES_PER_LAYER
        a_addr = A_BASE + ly * A_BYTES_PER_LAYER
        r_addr = R_BASE + ly * R_BYTES_PER_LAYER
        addrs.append((w_addr, a_addr, r_addr))

    # ── Build DRAM ──
    dram = [0] * DRAM_SIZE
    for ly in range(num_layers):
        w_addr, a_addr, r_addr = addrs[ly]
        if ly == 0:
            A = layers[0]["A"]  # original random A
        else:
            A = layers[ly]["A_repacked"]
        W = layers[ly]["W"]
        # Pack W
        # Pack W (shape: K×N, needs transpose)
        w_words = pack_tile_stream(W, N, K0, transpose=True)
        for i, w in enumerate(w_words):
            dram[(w_addr >> 2) + i] = w
        # Pack A (shape: M×K)
        a_words = pack_tile_stream(A, M, K0)
        for i, w in enumerate(a_words):
            dram[(a_addr >> 2) + i] = w

    write_hex(out_dir / "dram_init.hex", dram)

    # ── Expected: layer N-1's C ──
    C_last = layers[-1]["C"]
    exp = [C_last[m][n] for m in range(M) for n in range(N)]
    write_hex(out_dir / "expected.hex", exp)

    # ── Assemble firmware ──
    insns = []; labels = {}
    def emit(*ws):
        for w in ws: insns.append(w)
    def lbl(n): labels[n] = len(insns)
    def patch_beqz(idx, tgt, rs="t1"):
        insns[idx] = BEQZ(rs, (labels[tgt] - idx) * 4)

    def wreg(off, val):
        emit(*li_insns("t1", int(val)))
        emit(SW("t1", "s0", off))

    def launch_npu(ly, k_dim):
        wa, aa, ra = addrs[ly]
        wreg(0x00, 0)
        wreg(0x10, M); wreg(0x14, N); wreg(0x18, k_dim)
        wreg(0x20, wa); wreg(0x24, aa); wreg(0x28, ra)
        wreg(0x30, 0x80); wreg(0x3C, 2); wreg(0x00, 0x11)
        ln = f"p{ly}"
        lbl(ln)
        emit(LW("t1", "s0", 0x04)); emit(ANDI("t1", "t1", 2))
        i = len(insns); emit(0); patch_beqz(i, ln)

    def emit_repack(ly):
        """Repack C(ly-1) from R_prev → A_cur for layer ly."""
        _, _, r_prev = addrs[ly - 1]
        _, a_cur, _ = addrs[ly]
        SHIFT = 5

        emit(*li_insns("s1", r_prev))     # s1 = R_prev
        emit(*li_insns("s2", a_cur))      # s2 = A_cur
        emit(*li_insns("s3", N * 4))       # s3 = row stride (64)
        emit(*li_insns("s4", apk))         # s4 = words per K
        emit(*li_insns("s5", K0))          # s5 = K count
        emit(*li_insns("s6", SHIFT))       # s6 = shift
        emit(*li_insns("s7", 0))           # s7 = k

        lbl(f"rk{ly}")
        emit(*li_insns("s8", 0))           # s8 = w
        lbl(f"rw{ly}")

        # Base = s1 + w*256 + k*4
        emit(SLLI("a0", "s8", 2)); emit(SLLI("a0", "a0", 6))
        emit(SLLI("a1", "s7", 2))
        emit(r_type(0x00, reg("a0"), reg("s1"), 0x0, reg("a2"), 0x33))
        emit(r_type(0x00, reg("a1"), reg("a2"), 0x0, reg("a2"), 0x33))

        # Load 4 rows
        emit(LW("t1", "a2", 0))
        emit(r_type(0x00, reg("s3"), reg("a2"), 0x0, reg("a2"), 0x33))
        emit(LW("t2", "a2", 0))
        emit(r_type(0x00, reg("s3"), reg("a2"), 0x0, reg("a2"), 0x33))
        emit(LW("t3", "a2", 0))
        emit(r_type(0x00, reg("s3"), reg("a2"), 0x0, reg("a2"), 0x33))
        emit(LW("t4", "a2", 0))

        # Shift + mask
        emit(SRAI("t1", "t1", SHIFT)); emit(ANDI("t1", "t1", 0xFF))
        emit(SRAI("t2", "t2", SHIFT)); emit(ANDI("t2", "t2", 0xFF))
        emit(SRAI("t3", "t3", SHIFT)); emit(ANDI("t3", "t3", 0xFF))
        emit(SRAI("t4", "t4", SHIFT)); emit(ANDI("t4", "t4", 0xFF))

        # Pack
        emit(SLLI("t2", "t2", 8)); emit(ORR("t1", "t1", "t2"))
        emit(SLLI("t3", "t3", 16)); emit(ORR("t1", "t1", "t3"))
        emit(SLLI("t4", "t4", 24)); emit(ORR("t1", "t1", "t4"))

        # Store: a0 = s2 + (k*apk + w)*4
        emit(SLLI("a0", "s7", 2))
        emit(r_type(0x00, reg("s8"), reg("a0"), 0x0, reg("a0"), 0x33))
        emit(SLLI("a0", "a0", 2))
        emit(r_type(0x00, reg("a0"), reg("s2"), 0x0, reg("a0"), 0x33))
        emit(SW("t1", "a0", 0))

        # w++
        emit(ADDI("s8", "s8", 1))
        emit(r_type(0x20, reg("s8"), reg("s4"), 0x0, reg("a0"), 0x33))
        i = len(insns); emit(0); patch_beqz(i, f"rw{ly}")

        # k++
        emit(ADDI("s7", "s7", 1))
        emit(r_type(0x20, reg("s7"), reg("s5"), 0x0, reg("a0"), 0x33))
        i = len(insns); emit(0); patch_beqz(i, f"rk{ly}")

    # ── Firmware main ──
    emit(*li_insns("s0", NPU_BASE))
    for ly in range(num_layers):
        launch_npu(ly, K0)
        if ly < num_layers - 1:
            emit_repack(ly + 1)

    # Verify last layer C[0][0]
    _, _, r_last = addrs[-1]
    emit(*li_insns("t0", r_last))
    emit(LW("t1", "t0", 0))
    emit(*li_insns("t2", layers[-1]["C"][0][0] & 0xFFFFFFFF))
    emit(r_type(0x00, reg("t2"), reg("t1"), 0x0, reg("t4"), 0x33))

    # Pass marker
    emit(*li_insns("t0", MARKER_ADDR))
    emit(*li_insns("t1", PASS_MARKER))
    emit(SW("t1", "t0", 0))
    lbl("halt")
    emit(0x0000006f)

    fw_words = len(insns)
    write_hex(out_dir / "soc_multilayer.hex", insns)

    # ── params.vh ──
    op = out_dir.resolve().as_posix()
    with open(out_dir / "soc_multilayer_params.vh", "w") as f:
        f.write(f'`define MULTILAYER_FW_HEX "{op}/soc_multilayer.hex"\n')
        f.write(f'`define MULTILAYER_DRAM_HEX "{op}/dram_init.hex"\n')
        f.write(f'`define MULTILAYER_EXPECTED_HEX "{op}/expected.hex"\n')
        f.write(f'`define MULTILAYER_FW_WORDS {fw_words}\n')
        f.write(f'`define MULTILAYER_MARKER_ADDR 32\'h{MARKER_ADDR:08x}\n')
        f.write(f'`define MULTILAYER_R_ADDR 32\'h{addrs[-1][2]:08x}\n')
        f.write(f'`define MULTILAYER_RESULT_COUNT {M*N}\n')
        f.write(f'`define MULTILAYER_TIMEOUT_CYCLES {5000000}\n')
        f.write(f'`define MULTILAYER_DRAM_WORDS {DRAM_SIZE}\n')

    print(f"Generated {num_layers}-layer test: {out_dir}")
    print(f"  L{num_layers-1} C[0][0] = {layers[-1]['C'][0][0] & 0xFFFFFFFF}")
    print(f"  firmware: {fw_words} words")

if __name__ == "__main__":
    main()
