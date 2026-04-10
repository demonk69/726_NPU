#!/usr/bin/env python3
"""
assemble_soc_test.py - Manual RV32I assembler for soc_test firmware.

Assembles the NPU SoC integration test firmware into a hex file
compatible with Verilog $readmemh.

Instruction set used: RV32I (no compressed, no M/C)
Mode: .option norvc (no compressed instructions)
"""

import struct
import sys

# ============================================================
# RV32I Instruction Encoder
# ============================================================

def reg(name):
    """Return register number for ABI name."""
    table = {
        'zero':0, 'ra':1, 'sp':2, 'gp':3, 'tp':4,
        't0':5, 't1':6, 't2':7,
        's0':8, 'fp':8, 's1':9,
        'a0':10, 'a1':11, 'a2':12, 'a3':13,
        'a4':14, 'a5':15, 'a6':16, 'a7':17,
        's2':18, 's3':19, 's4':20, 's5':21,
        's6':22, 's7':23, 's8':24, 's9':25,
        's10':26, 's11':27,
        't3':28, 't4':29, 't5':30, 't6':31,
    }
    return table[name]

def r_type(funct7, rs2, rs1, funct3, rd, opcode):
    return ((funct7 & 0x7F) << 25 | (rs2 & 0x1F) << 20 |
            (rs1 & 0x1F) << 15 | (funct3 & 0x7) << 12 |
            (rd & 0x1F) << 7 | (opcode & 0x7F))

def i_type(imm, rs1, funct3, rd, opcode):
    imm12 = imm & 0xFFF
    return (imm12 << 20 | (rs1 & 0x1F) << 15 | (funct3 & 0x7) << 12 |
            (rd & 0x1F) << 7 | (opcode & 0x7F))

def s_type(imm, rs2, rs1, funct3, opcode):
    imm &= 0xFFF
    imm11_5 = (imm >> 5) & 0x7F
    imm4_0  = imm & 0x1F
    return (imm11_5 << 25 | (rs2 & 0x1F) << 20 | (rs1 & 0x1F) << 15 |
            (funct3 & 0x7) << 12 | imm4_0 << 7 | (opcode & 0x7F))

def b_type(imm, rs2, rs1, funct3, opcode):
    """Branch immediate is in multiples of 2 (byte offset)."""
    imm &= 0x1FFE   # 13-bit, bit0 always 0
    b12    = (imm >> 12) & 1
    b11    = (imm >> 11) & 1
    b10_5  = (imm >> 5)  & 0x3F
    b4_1   = (imm >> 1)  & 0xF
    return (b12 << 31 | b10_5 << 25 | (rs2 & 0x1F) << 20 |
            (rs1 & 0x1F) << 15 | (funct3 & 0x7) << 12 |
            b4_1 << 8 | b11 << 7 | (opcode & 0x7F))

def u_type(imm, rd, opcode):
    """Upper immediate: imm is the 20-bit upper value (bits [31:12])."""
    return ((imm & 0xFFFFF) << 12 | (rd & 0x1F) << 7 | (opcode & 0x7F))

def j_type(imm, rd, opcode):
    """JAL: 21-bit byte offset."""
    imm &= 0x1FFFFE  # 21-bit, bit0 always 0
    b20     = (imm >> 20) & 1
    b10_1   = (imm >> 1)  & 0x3FF
    b11     = (imm >> 11) & 1
    b19_12  = (imm >> 12) & 0xFF
    return (b20 << 31 | b10_1 << 21 | b11 << 20 | b19_12 << 12 |
            (rd & 0x1F) << 7 | (opcode & 0x7F))

# ============================================================
# Instruction helpers
# ============================================================

def LUI(rd, imm):
    """LUI rd, imm  (imm = upper 20 bits value, i.e. the number to shift left by 12)"""
    return u_type(imm >> 12, reg(rd), 0x37)

def ADDI(rd, rs1, imm):
    return i_type(imm, reg(rs1), 0x0, reg(rd), 0x13)

def SW(rs2, rs1, imm):
    return s_type(imm, reg(rs2), reg(rs1), 0x2, 0x23)

def LW(rd, rs1, imm):
    return i_type(imm, reg(rs1), 0x2, reg(rd), 0x03)

def ANDI(rd, rs1, imm):
    return i_type(imm, reg(rs1), 0x7, reg(rd), 0x13)

def BEQ(rs1, rs2, byte_offset):
    return b_type(byte_offset, reg(rs2), reg(rs1), 0x0, 0x63)

def BNE(rs1, rs2, byte_offset):
    return b_type(byte_offset, reg(rs2), reg(rs1), 0x1, 0x63)

def BEQZ(rs1, byte_offset):
    return BEQ(rs1, 'zero', byte_offset)

def BNEZ(rs1, byte_offset):
    return BNE(rs1, 'zero', byte_offset)

def JAL(rd, byte_offset):
    return j_type(byte_offset, reg(rd), 0x6F)

def J(byte_offset):
    """Pseudo: JAL zero, offset"""
    return JAL('zero', byte_offset)

def MV(rd, rs1):
    """Pseudo: ADDI rd, rs1, 0"""
    return ADDI(rd, rs1, 0)

def NOP():
    return ADDI('zero', 'zero', 0)

# ============================================================
# LI macro: load 32-bit immediate using LUI + ADDI
# Returns 1 or 2 instructions
# ============================================================
def li_insns(rd, imm):
    """Encode 'li rd, imm' returning list of 32-bit words."""
    imm32 = imm & 0xFFFFFFFF
    # sign-extend to Python int
    if imm32 >= 0x80000000:
        imm32 -= 0x100000000
    
    lo12 = imm32 & 0xFFF
    # sign-extend lo12
    if lo12 >= 0x800:
        lo12 -= 0x1000
    hi20 = (imm32 - lo12) >> 12
    
    if hi20 == 0:
        # Only ADDI
        return [ADDI(rd, 'zero', lo12 & 0xFFF)]
    else:
        insns = [LUI(rd, (hi20 << 12) & 0xFFFFFFFF)]
        if lo12 != 0:
            insns.append(ADDI(rd, rd, lo12))
        return insns

# ============================================================
# Assemble the firmware
# ============================================================

def assemble():
    """
    Firmware layout (all .option norvc = 32-bit instructions only):
    
    _start:
        li   sp, 0x00001000           # stack = SRAM top
        li   t0, 0x02000000           # NPU base
        
        li   t1, 2
        sw   t1, 0x10(t0)             # M_DIM = 2
        sw   t1, 0x14(t0)             # N_DIM = 2
        sw   t1, 0x18(t0)             # K_DIM = 2
        
        li   t1, 0x00001000
        sw   t1, 0x20(t0)             # W_ADDR
        
        li   t1, 0x00001010
        sw   t1, 0x24(t0)             # A_ADDR
        
        li   t1, 0x00001020
        sw   t1, 0x28(t0)             # R_ADDR
        
        li   t1, 0x11
        sw   t1, 0x00(t0)             # CTRL: start=1, OS mode, INT8
        
    poll_loop:
        lw   t1, 0x04(t0)             # read STATUS
        andi t1, t1, 2                # check done bit
        beqz t1, poll_loop            # loop if not done
        
        # verify C[0][0] == 19
        li   t2, 0x00001020
        lw   t1, 0(t2)
        li   t3, 19
        bne  t1, t3, verify_fail
        # verify C[0][1] == 22
        lw   t1, 4(t2)
        li   t3, 22
        bne  t1, t3, verify_fail
        # verify C[1][0] == 43
        lw   t1, 8(t2)
        li   t3, 43
        bne  t1, t3, verify_fail
        # verify C[1][1] == 50
        lw   t1, 12(t2)
        li   t3, 50
        bne  t1, t3, verify_fail
        # pass
        j    write_pass
        
    verify_fail:
        li   t0, 0x00000F00
        li   t1, 0xFF
        sw   t1, 0(t0)
        j    end
        
    write_pass:
        li   t0, 0x00000F00
        li   t1, 0xAA
        sw   t1, 0(t0)
        
    end:
        j    end                      # infinite loop
    """
    
    insns = []  # list of (label_name_if_any, instruction_word)
    labels = {}  # label -> instruction index
    
    def emit(*words):
        for w in words:
            insns.append(w)
    
    def label(name):
        labels[name] = len(insns)
    
    # ---- _start ----
    label('_start')
    emit(*li_insns('sp', 0x00001000))   # li sp, 0x1000
    emit(*li_insns('t0', 0x02000000))   # li t0, NPU_BASE

    # M_DIM, N_DIM, K_DIM = 2
    emit(*li_insns('t1', 2))
    emit(SW('t1', 't0', 0x10))          # M_DIM
    emit(SW('t1', 't0', 0x14))          # N_DIM
    emit(SW('t1', 't0', 0x18))          # K_DIM

    # W_ADDR = 0x1000
    emit(*li_insns('t1', 0x00001000))
    emit(SW('t1', 't0', 0x20))

    # A_ADDR = 0x1010
    emit(*li_insns('t1', 0x00001010))
    emit(SW('t1', 't0', 0x24))

    # R_ADDR = 0x1020
    emit(*li_insns('t1', 0x00001020))
    emit(SW('t1', 't0', 0x28))

    # CTRL = 0x11 (start=1, INT8, OS)
    emit(*li_insns('t1', 0x11))
    emit(SW('t1', 't0', 0x00))

    # ---- poll_loop ----
    label('poll_loop')
    # LW t1, 4(t0)
    emit(LW('t1', 't0', 0x04))
    # ANDI t1, t1, 2
    emit(ANDI('t1', 't1', 2))
    # BEQZ t1, poll_loop  (branch offset calculated after all insns are placed)
    poll_loop_beqz_idx = len(insns)
    emit(0)  # placeholder

    # ---- verify ----
    emit(*li_insns('t2', 0x00001020))   # t2 = R_ADDR

    # C[0][0] == 19
    emit(LW('t1', 't2', 0))
    emit(*li_insns('t3', 19))
    verify_fail_bne_0 = len(insns); emit(0)  # BNE placeholder

    # C[0][1] == 22
    emit(LW('t1', 't2', 4))
    emit(*li_insns('t3', 22))
    verify_fail_bne_1 = len(insns); emit(0)

    # C[1][0] == 43
    emit(LW('t1', 't2', 8))
    emit(*li_insns('t3', 43))
    verify_fail_bne_2 = len(insns); emit(0)

    # C[1][1] == 50
    emit(LW('t1', 't2', 12))
    emit(*li_insns('t3', 50))
    verify_fail_bne_3 = len(insns); emit(0)

    # j write_pass
    write_pass_j_idx = len(insns); emit(0)

    # ---- verify_fail ----
    label('verify_fail')
    emit(*li_insns('t0', 0x00002000))   # marker address in DRAM: 0x2000
    emit(*li_insns('t1', 0xFF))
    emit(SW('t1', 't0', 0))
    end_j_fail = len(insns); emit(0)  # j end

    # ---- write_pass ----
    label('write_pass')
    emit(*li_insns('t0', 0x00002000))   # marker address in DRAM: 0x2000
    emit(*li_insns('t1', 0xAA))
    emit(SW('t1', 't0', 0))

    # ---- end ----
    label('end')
    end_j_end = len(insns); emit(0)  # j end

    # ---- Patch branch/jump placeholders ----
    def patch_beqz(idx, target_label, rs='t1'):
        offset = (labels[target_label] - idx) * 4
        insns[idx] = BEQZ(rs, offset)

    def patch_bne(idx, target_label, rs1='t1', rs2='t3'):
        offset = (labels[target_label] - idx) * 4
        insns[idx] = BNE(rs1, rs2, offset)

    def patch_j(idx, target_label):
        offset = (labels[target_label] - idx) * 4
        insns[idx] = J(offset)

    patch_beqz(poll_loop_beqz_idx, 'poll_loop')
    patch_bne(verify_fail_bne_0, 'verify_fail')
    patch_bne(verify_fail_bne_1, 'verify_fail')
    patch_bne(verify_fail_bne_2, 'verify_fail')
    patch_bne(verify_fail_bne_3, 'verify_fail')
    patch_j(write_pass_j_idx, 'write_pass')
    patch_j(end_j_fail, 'end')
    patch_j(end_j_end, 'end')

    return insns, labels

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    insns, labels = assemble()
    
    print(f"Total instructions: {len(insns)}")
    print(f"Labels: {', '.join(f'{k}@{v*4:#x}' for k,v in labels.items())}")
    print()

    # Disassembly-like output for verification
    for i, w in enumerate(insns):
        addr = i * 4
        print(f"  {addr:04x}: {w:08x}")
    print()

    # Write hex file
    out_path = 'soc_test.hex'
    if len(sys.argv) > 1:
        out_path = sys.argv[1]

    with open(out_path, 'w') as f:
        for w in insns:
            f.write(f'{w & 0xFFFFFFFF:08x}\n')

    print(f"Written {len(insns)} words to {out_path}")
