#!/usr/bin/env python3
"""Generate firmware and DRAM hex for multi-core SoC tests."""
import sys
from pathlib import Path

TB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TB_DIR))
from assemble_soc_test import ADDI, ANDI, BEQ, BNE, J, JAL, LW, SW, b_type, i_type, li_insns, r_type, reg, s_type

def ADD(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)

def SUB(rd, rs1, rs2):
    return r_type(0x20, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)

def AND(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x7, reg(rd), 0x33)

def OR(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x6, reg(rd), 0x33)

def SLLI(rd, rs1, shamt):
    return i_type(shamt & 0x1F, reg(rs1), 0x1, reg(rd), 0x13)

def SRLI(rd, rs1, shamt):
    return i_type(shamt & 0x1F, reg(rs1), 0x5, reg(rd), 0x13)

def SLTI(rd, rs1, imm):
    return i_type(imm, reg(rs1), 0x2, reg(rd), 0x13)


def write_hex(path, words):
    with open(path, "w") as f:
        for w in words:
            f.write(f"{int(w) & 0xFFFFFFFF:08x}\n")


# ====================================================================
# MMIO test firmware
# ====================================================================
def gen_mmio_fw():
    """PicoRV32 firmware: write/read core0 and core1 M_DIM, verify."""
    a = []
    NPU_BASE = 0x02000000; MARKER_ADDR = 0x1000
    FAIL = 0xFF; PASS = 0xAA
    def emit(*ws): a.extend(int(w) & 0xFFFFFFFF for w in ws)

    emit(*li_insns("s0", NPU_BASE))
    emit(*li_insns("s1", MARKER_ADDR))
    emit(*li_insns("s2", FAIL))

    def test_write_read(core_offset, expect, next_test_offs):
        emit(*li_insns("t0", expect)); emit(SW("t0", "s0", core_offset))
        emit(LW("t1", "s0", core_offset)); emit(*li_insns("t2", expect))
        emit(r_type(0x20, reg("t2"), reg("t1"), 0x0, reg("t0"), 0x33))
        emit(BNE("t0", "zero", 20)); emit(J(12))
        emit(SW("s2", "s1", 0)); emit(J(0))

    # Test 1: core0 M_DIM
    test_write_read(0x10, 0x1111, 0)
    # Test 2: core1 M_DIM
    test_write_read(0x110, 0x2222, 0)
    # Test 3: invalid window read
    emit(LW("t1", "s0", 0x400)); emit(*li_insns("t2", 0xDEADBEEF))
    emit(r_type(0x20, reg("t2"), reg("t1"), 0x0, reg("t0"), 0x33))
    emit(BNE("t0", "zero", 20)); emit(J(12))
    emit(SW("s2", "s1", 0)); emit(J(0))
    # Test 4: core isolation
    emit(*li_insns("t0", 0xAAAA)); emit(SW("t0", "s0", 0x10))
    emit(*li_insns("t0", 0xBBBB)); emit(SW("t0", "s0", 0x110))
    test_write_read(0x10, 0xAAAA, 0)
    test_write_read(0x110, 0xBBBB, 0)
    # PASS
    emit(*li_insns("t0", PASS)); emit(SW("t0", "s1", 0)); emit(J(0))
    return a


# ====================================================================
# Shared-A test firmware + DRAM
# ====================================================================
def gen_shared_a():
    """Generate firmware and DRAM for shared-A test.

    M=4, N=8, K=4. Tile shape 4x4.
    Core0: N[0:3], Core1: N[4:7].
    Both cores read same A_WORK_SHARED.
    Each core reads its own W tile, writes its own R buffer.
    """
    NPU_BASE = 0x02000000
    PASS = 0xAA
    FAIL = 0xFF
    MARKER_ADDR = 0x1000
    A_WORK_SHARED = 0x0000_0100
    W_CORE0 = 0x0000_0200
    W_CORE1 = 0x0000_0300
    R_CORE0 = 0x0000_0400
    R_CORE1 = 0x0000_0500

    # DRAM: initialize shared A and per-core W
    # A[4x4], each word contains one row of 4 INT8 bytes
    # Packed INT8 OS tile stream: A[0][0:3], A[1][0:3], ... padded to SIMD
    # K=4, SIMD=4 => 4 packed words for W (one per K step, N lanes)
    # and 4 packed words for A (one per K step, M lanes)
    dram = [0] * 4096

    # W_CORE0: tile for N channels [0:3], K=4
    # W packed: for each k, 4 INT8 values for N[0:3]
    # W value = N_channel + 1 (makes each column distinct)
    # W[0:3,0] = {1,2,3,4}
    dram[W_CORE0 >> 2] = 0x0403_0201  # k=0, N[0:3]
    dram[(W_CORE0 >> 2) + 1] = 0x0000_0000  # SIMD pad

    # W_CORE1: tile for N channels [4:7], K=4
    # W[4:7,0] = {5,6,7,8}
    dram[W_CORE1 >> 2] = 0x0807_0605  # k=0, N[4:7]
    dram[(W_CORE1 >> 2) + 1] = 0x0000_0000  # SIMD pad

    # A[4x4], K=4, single K step has 4 INT8 lanes
    # A[0:3, 0] = {1, 2, 3, 4} (same value for simplicity)
    dram[A_WORK_SHARED >> 2] = 0x0403_0201  # k=0, M[0:3]
    dram[(A_WORK_SHARED >> 2) + 1] = 0x0000_0000  # SIMD pad

    # Expected results (C = A * W^T, INT32):
    # C[i,j] = sum_k A[i,k] * W[k,j] = sum_k A[i,k] * W[k,j]
    # Since K=1 (K=4 but all the data is in k=0, remaining k are 0 due to padding),
    # C[i,j] = A[i,0] * W[0,j]
    #
    # Core0 (N[0:3]): W_CORE0 = {1,2,3,4} for N[0:3]
    # C[0,0]=1*1=1, C[0,1]=1*2=2, C[0,2]=1*3=3, C[0,3]=1*4=4
    # C[1,0]=2*1=2, C[1,1]=2*2=4, C[1,2]=2*3=6, C[1,3]=2*4=8
    # C[2,0]=3*1=3, C[2,1]=3*2=6, C[2,2]=3*3=9, C[2,3]=3*4=12
    # C[3,0]=4*1=4, C[3,1]=4*2=8, C[3,2]=4*3=12,C[3,3]=4*4=16
    #
    # Core1 (N[4:7]): W_CORE1 = {5,6,7,8}
    # C[0,0]=1*5=5, C[0,1]=1*6=6, C[0,2]=1*7=7, C[0,3]=1*8=8
    # ... etc

    expected_c0 = [
        1, 2, 3, 4,
        2, 4, 6, 8,
        3, 6, 9, 12,
        4, 8, 12, 16,
    ]
    expected_c1 = [
        5, 6, 7, 8,
        10, 12, 14, 16,
        15, 18, 21, 24,
        20, 24, 28, 32,
    ]

    # Place expected values at known addresses
    EXP_C0 = 0x0000_0800
    EXP_C1 = 0x0000_0840
    for i, v in enumerate(expected_c0):
        dram[(EXP_C0 >> 2) + i] = v & 0xFFFFFFFF
    for i, v in enumerate(expected_c1):
        dram[(EXP_C1 >> 2) + i] = v & 0xFFFFFFFF

    # Firmware:
    # s0 = NPU_BASE (Core0)
    # Program and launch both cores
    fw = []
    def emit(*ws):
        fw.extend(int(w) & 0xFFFFFFFF for w in ws)

    # Constants
    REG_CTRL = 0x00
    REG_STATUS = 0x04
    REG_M_DIM = 0x10
    REG_N_DIM = 0x14
    REG_K_DIM = 0x18
    REG_W_ADDR = 0x20
    REG_A_ADDR = 0x24
    REG_R_ADDR = 0x28
    REG_ARR_CFG = 0x30
    REG_CFG_SHAPE = 0x3C
    REG_BIAS_ADDR = 0x98
    REG_QUANT_CFG = 0x9C
    CORE_STRIDE = 0x100

    # Helper: store value to core's register
    # t0 = value, t1 = core's offset (core*256 + reg_offset)
    def store_core_reg(rd_val, rs_core_base, reg_offs):
        emit(*li_insns("t0", reg_offs))
        emit(ADD("t0", "t0", rs_core_base))
        emit(*li_insns("t2", rd_val))
        emit(SW("t2", "t0", 0))

    # s0 = NPU_BASE (core0)
    # s1 = core1 base = NPU_BASE + 0x100
    emit(*li_insns("s0", NPU_BASE))
    emit(*li_insns("s1", NPU_BASE + CORE_STRIDE))

    # ---- Program core0 ----
    store_core_reg(4, "s0", REG_M_DIM)
    store_core_reg(4, "s0", REG_N_DIM)
    store_core_reg(4, "s0", REG_K_DIM)
    store_core_reg(W_CORE0, "s0", REG_W_ADDR)
    store_core_reg(A_WORK_SHARED, "s0", REG_A_ADDR)
    store_core_reg(R_CORE0, "s0", REG_R_ADDR)
    store_core_reg(0x80, "s0", REG_ARR_CFG)
    store_core_reg(0, "s0", REG_CFG_SHAPE)
    store_core_reg(0x0001_0000, "s0", REG_QUANT_CFG)

    # ---- Program core1 ----
    store_core_reg(4, "s1", REG_M_DIM)
    store_core_reg(4, "s1", REG_N_DIM)
    store_core_reg(4, "s1", REG_K_DIM)
    store_core_reg(W_CORE1, "s1", REG_W_ADDR)
    store_core_reg(A_WORK_SHARED, "s1", REG_A_ADDR)
    store_core_reg(R_CORE1, "s1", REG_R_ADDR)
    store_core_reg(0x80, "s1", REG_ARR_CFG)
    store_core_reg(0, "s1", REG_CFG_SHAPE)
    store_core_reg(0x0001_0000, "s1", REG_QUANT_CFG)

    # ---- Start both cores ----
    emit(*li_insns("t0", 0x211))           # CTRL_BIAS_TILE_PACK
    emit(SW("t0", "s0", REG_CTRL))         # start core0
    emit(SW("t0", "s1", REG_CTRL))         # start core1

    # ---- Poll both ----
    # s2 = done_mask, s3 = launched_mask (both cores launched)
    emit(*li_insns("s2", 0))
    emit(*li_insns("s3", 3))               # bits 0 and 1

    emit(*li_insns("t2", 2))               # status.done bit

    # poll_loop:
    # Read core0 STATUS
    emit(LW("t0", "s0", REG_STATUS))
    emit(AND("t0", "t0", "t2"))
    # if done and not tracked, mark
    emit(ADDI("t3", "s2", 0))
    emit(ANDI("t3", "t3", 1))
    emit(BNE("t3", "zero", 4 * 3))         # skip if already marked
    emit(BNE("t0", "zero", 4 * 4))         # skip if not done
    emit(*li_insns("t0", 1))
    emit(OR("s2", "s2", "t0"))

    # Read core1 STATUS
    emit(LW("t0", "s1", REG_STATUS))
    emit(AND("t0", "t0", "t2"))
    emit(ADDI("t3", "s2", 0))
    emit(ANDI("t3", "t3", 2))
    emit(BNE("t3", "zero", 4 * 3))
    emit(BNE("t0", "zero", 4 * 4))
    emit(*li_insns("t0", 2))
    emit(OR("s2", "s2", "t0"))

    # check if all launched done
    emit(ADDI("t0", "s2", 0))
    emit(AND("t0", "t0", "s3"))
    emit(SUB("t0", "t0", "s3"))
    emit(BNE("t0", "zero", 4 * -20))

    # All done. Compare results.
    # s3 = R_CORE0 base, s4 = R_CORE1 base
    # s5 = loop counter (16 results per core)
    # s6 = EXP_C0 base, s7 = EXP_C1 base
    emit(*li_insns("s3", R_CORE0))
    emit(*li_insns("s4", R_CORE1))
    emit(*li_insns("s6", EXP_C0))
    emit(*li_insns("s7", EXP_C1))

    emit(*li_insns("s5", 16))
    # compare_loop:
    emit(LW("t0", "s3", 0))
    emit(LW("t1", "s6", 0))
    emit(SUB("t0", "t0", "t1"))
    emit(*li_insns("t3", FAIL))
    emit(BNE("t0", "zero", 4 * 8))
    emit(LW("t0", "s4", 0))
    emit(LW("t1", "s7", 0))
    emit(SUB("t0", "t0", "t1"))
    emit(BNE("t0", "zero", 4 * 4))

    emit(ADDI("s3", "s3", 4))
    emit(ADDI("s4", "s4", 4))
    emit(ADDI("s6", "s6", 4))
    emit(ADDI("s7", "s7", 4))
    emit(ADDI("s5", "s5", -1))
    emit(BNE("s5", "zero", 4 * -11))

    # PASS
    emit(*li_insns("t0", MARKER_ADDR))
    emit(*li_insns("t1", 0xAA))
    emit(SW("t1", "t0", 0))
    emit(JAL("zero", 4 * -1))

    # fail_label: 
    emit(SW("t3", "t0", 0))
    emit(JAL("zero", 4 * -1))

    # Halt loop
    emit(JAL("zero", 4 * -1))

    return fw, dram


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "sim/mc_tests"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fw_mmio = gen_mmio_fw()
    write_hex(out / "mc_mmio_fw.hex", fw_mmio)

    fw_sa, dram_sa = gen_shared_a()
    write_hex(out / "mc_shared_a_fw.hex", fw_sa)
    write_hex(out / "mc_shared_a_dram.hex", dram_sa)

    print(f"Generated test assets to {out}")
    print(f"  mc_mmio_fw.hex: {len(fw_mmio)} words")
    print(f"  mc_shared_a_fw.hex: {len(fw_sa)} words")
    print(f"  mc_shared_a_dram.hex: {len(dram_sa)} words")


if __name__ == "__main__":
    main()
