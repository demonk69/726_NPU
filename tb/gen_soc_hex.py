#!/usr/bin/env python3
"""
Generate soc_test.hex - RISC-V machine code firmware for NPU SoC integration test.

Test: C = A * B (INT8, M=2, N=2, K=2)
  A = [[1,2],[3,4]]  (row-major)
  B = [[5,6],[7,8]]  (column-major storage for NPU)
  C = [[19,22],[43,50]]

DRAM layout (from 0x1000):
  W_ADDR 0x1000: B column-major: col0=[5,7] col1=[6,8]
                 Word0: 0x00000705  Word1: 0x00000806
  A_ADDR 0x1010: A row-major: row0=[1,2] row1=[3,4]
                 Word0: 0x00000201  Word1: 0x00000403
  R_ADDR 0x1020: Result: 4 x 32-bit words (C row-major)
                 [19, 22, 43, 50]
  MARKER 0x0F00: 0xAA (PASS) or 0xFF (FAIL)

Register addresses (NPU base = 0x02000000):
  0x00 CTRL, 0x04 STATUS, 0x10 M_DIM, 0x14 N_DIM, 0x18 K_DIM
  0x20 W_ADDR, 0x24 A_ADDR, 0x28 R_ADDR

NPU CTRL: bit0=start, bit4=OS_mode => 0x11 to start in OS mode
"""

import struct
import sys

# RISC-V instruction helpers (RV32I)
def r_type(funct7, rs2, rs1, funct3, rd, opcode):
    return ((funct7 & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F)

def i_type(imm, rs1, funct3, rd, opcode):
    return ((imm & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F)

def s_type(imm, rs1, funct3, rs2, opcode):
    return (((imm >> 5) & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) | ((imm & 0x1F) << 7) | (opcode & 0x7F)

def b_type(imm, rs1, rs2, funct3, opcode):
    # imm[12|10:5] | rs2 | rs1 | funct3 | imm[4:1|11] | opcode
    bit12 = (imm >> 12) & 1
    bit11 = (imm >> 11) & 1
    bits10_5 = (imm >> 5) & 0x3F
    bits4_1 = (imm >> 1) & 0xF
    return (bit12 << 31) | (bits10_5 << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) | (bits4_1 << 8) | (bit11 << 7) | (opcode & 0x7F)

def u_type(imm, rd, opcode):
    return (imm & 0xFFFFF000) | ((rd & 0x1F) << 7) | (opcode & 0x7F)

def lui(rd, imm):
    return u_type(imm, rd, 0x37)

def addi(rd, rs1, imm):
    return i_type(imm, rs1, 0x0, rd, 0x13)

def sw(rs1, offset, rs2):
    return s_type(offset, rs1, 0x2, rs2, 0x23)

def lw(rd, offset, rs1):
    return i_type(offset, rs1, 0x2, rd, 0x03)

def beq(rs1, rs2, offset):
    return b_type(offset, rs1, rs2, 0x0, 0x63)

def bne(rs1, rs2, offset):
    return b_type(offset, rs1, rs2, 0x1, 0x63)

def jal(rd, offset):
    # J-type: imm[20|10:1|11|19:12] | rd | opcode
    bit20 = (offset >> 20) & 1
    bit11 = (offset >> 11) & 1
    bits10_1 = (offset >> 1) & 0x3FF
    bits19_12 = (offset >> 12) & 0xFF
    return (bit20 << 31) | (bits10_1 << 21) | (bit11 << 20) | (bits19_12 << 12) | ((rd & 0x1F) << 7) | 0x6F

def andi(rd, rs1, imm):
    return i_type(imm, rs1, 0x7, rd, 0x13)

# Register names
ZERO = 0
RA = 1
SP = 2
T0 = 5
T1 = 6
T2 = 7
T3 = 28  # x28 = t3
T4 = 29  # x29 = t4
T5 = 30  # x30 = t5
T6 = 31  # x31 = t6

# Actually let's use ABI names
# t0=x5, t1=x6, t2=x7, t3=x28, t4=x29, t5=x30, t6=x31
# s0=x8, s1=x9

code = []

# ========== Firmware ==========
# t0 = 0x02000000 (NPU base)

# li t0, 0x02000000
# lui t0, 0x02000
code.append(lui(T0, 0x02000000))

# ==== Configure NPU registers ====

# M_DIM = 2
code.append(addi(T1, ZERO, 2))
code.append(sw(T0, 0x10, T1))

# N_DIM = 2
code.append(addi(T1, ZERO, 2))
code.append(sw(T0, 0x14, T1))

# K_DIM = 2
code.append(addi(T1, ZERO, 2))
code.append(sw(T0, 0x18, T1))

# W_ADDR = 0x1000
code.append(lui(T1, 0x1000))  # li t1, 0x1000
code.append(sw(T0, 0x20, T1))

# A_ADDR = 0x1010
code.append(lui(T1, 0x1000))  # li t1, 0x1000
code.append(addi(T1, T1, 0x10))  # t1 = 0x1010
code.append(sw(T0, 0x24, T1))

# R_ADDR = 0x1020
code.append(lui(T1, 0x1000))  # li t1, 0x1000
code.append(addi(T1, T1, 0x20))  # t1 = 0x1020
code.append(sw(T0, 0x28, T1))

# ==== Start NPU (CTRL = 0x11: start + OS mode) ====
code.append(addi(T1, ZERO, 0x11))
code.append(sw(T0, 0x00, T1))

# ==== Poll STATUS for done ====
# poll_loop:
#   lw t1, 0x04(t0)
#   andi t1, t1, 0x02
#   beqz t1, poll_loop
POLL_LOOP = len(code)

code.append(lw(T1, 0x04, T0))       # lw t1, 4(t0)
code.append(andi(T1, T1, 0x02))     # andi t1, t1, 2
# beq t1, zero, poll_loop -> offset = -3 instructions = -12 bytes
code.append(beq(T1, ZERO, -12))

# ==== Verify result ====
# Expected: C = [[19,22],[43,50]]
# Result at R_ADDR = 0x1020, 4 words
# t2 = result base address pointer, t3 = expected base

# li t2, 0x1020
code.append(lui(T2, 0x1000))
code.append(addi(T2, T2, 0x20))

# Check C[0][0] = 19
code.append(lw(T1, 0, T2))           # lw t1, 0(t2)
code.append(addi(T3, ZERO, 19))      # li t3, 19
code.append(bne(T1, T3, 8))          # bne t1, t3, fail (skip 8 instructions forward)

# Check C[0][1] = 22
code.append(lw(T1, 4, T2))           # lw t1, 4(t2)
code.append(addi(T3, ZERO, 22))      # li t3, 22
code.append(bne(T1, T3, 5))          # bne t1, t3, fail (skip 5 instructions)

# Check C[1][0] = 43
code.append(lw(T1, 8, T2))           # lw t1, 8(t2)
code.append(addi(T3, ZERO, 43))      # li t3, 43
code.append(bne(T1, T3, 2))          # bne t1, t3, fail (skip 2 instructions)

# Check C[1][1] = 50
code.append(lw(T1, 12, T2))          # lw t1, 12(t2)
code.append(addi(T3, ZERO, 50))      # li t3, 50
code.append(bne(T1, T3, -1))         # bne t1, t3, fail (but we need to calc)

# Actually let me be more careful with branch offsets
# Let me rewrite the verification section with calculated offsets

# Recalculate from scratch:
verify_start = len(code)

# Reset code from verify_start
code = code[:verify_start]

# li t2, 0x1020  (2 instructions)
code.append(lui(T2, 0x1000))
code.append(addi(T2, T2, 0x20))

# Verify 4 elements:
# Load each, compare, branch to fail_label on mismatch
# fail_label will be at the end

# Check C[0][0] = 19
code.append(lw(T1, 0, T2))           # idx 2
code.append(addi(T3, ZERO, 19))      # idx 3
code.append(bne(T1, T3, 0))          # idx 4, placeholder offset -> fixup

check00_bne_idx = len(code) - 1

# Check C[0][1] = 22
code.append(lw(T1, 4, T2))           # idx 5
code.append(addi(T3, ZERO, 22))      # idx 6
code.append(bne(T1, T3, 0))          # idx 7, placeholder

check01_bne_idx = len(code) - 1

# Check C[1][0] = 43
code.append(lw(T1, 8, T2))           # idx 8
code.append(addi(T3, ZERO, 43))      # idx 9
code.append(bne(T1, T3, 0))          # idx 10, placeholder

check10_bne_idx = len(code) - 1

# Check C[1][1] = 50
code.append(lw(T1, 12, T2))          # idx 11
code.append(addi(T3, ZERO, 50))      # idx 12
code.append(bne(T1, T3, 0))          # idx 13, placeholder

check11_bne_idx = len(code) - 1

# PASS: write 0xAA to marker
# li t0, 0x0F00  (but t0 is NPU base... need to reuse)
# Use t2 as temp: already set. Use different register.
# Actually we can just overwrite t0 since NPU is done.
pass_label = len(code)

code.append(addi(T0, ZERO, 0))       # t0 = 0
code.append(lui(T0, 0x00000))        # Hmm this doesn't work for 0xF00
# li t0, 0x0F00 = lui t0, 0x1  + addi t0, t0, -0x700
# Actually: 0x0F00 = 0x10000 - 0xF100... no
# 0x0F00 in 20-bit: 0x0F00 >> 12 = 0, so lui t0, 0; addi t0, t0, 0x0F00
# But 0x0F00 = 3840, which fits in 12-bit signed? max is 2047. No.
# lui t0, 0x1  => t0 = 0x1000; addi t0, t0, -0x100 => t0 = 0xF00. No that gives 0.
# Actually: lui t0, 0x1 gives t0=0x10000. addi t0, t0, -0x100 gives 0xFF00. Wrong.
# Need: 0x0F00. lui rd, 0x1 => 0x10000. addi rd, rd, -0xF100? No.
# The correct way: 0x0F00 = 0x01000 - wait...
# lui t0, 0x00001 => t0 = 0x01000. addi t0, t0, -256 => 0x01000 - 0x100 = 0x0F00. Yes!

# Wait, I was using t0 for NPU base. Let me just use a different register for marker.
# Since we already verified, we can use any register.
code.append(lui(T0, 0x00001))        # t0 = 0x1000
code.append(addi(T0, T0, -256))      # t0 = 0xF00 = 0x0F00? No: 0x1000 - 0x100 = 0x0F00. 
# Actually in 32-bit: 0x00001000 - 0x00000100 = 0x00000F00. Yes correct!
code.append(addi(T1, ZERO, 0xAA))    # t1 = PASS marker
code.append(sw(T0, 0, T1))           # store marker

# Done loop: j self
done_label = len(code)
code.append(jal(ZERO, 0))            # j . (infinite loop, offset will be 0)

# FAIL label
fail_label = len(code)

# Write 0xFF to marker
code.append(lui(T0, 0x00001))
code.append(addi(T0, T0, -256))      # t0 = 0x0F00
code.append(addi(T1, ZERO, 0xFF))    # t1 = FAIL marker
code.append(sw(T0, 0, T1))

# Done loop: j self
code.append(jal(ZERO, 0))

# Now fixup all BNE branch offsets
# BNE at check00_bne_idx -> fail_label
# offset = (fail_label - check00_bne_idx) * 4
for idx in [check00_bne_idx, check01_bne_idx, check10_bne_idx, check11_bne_idx]:
    offset = (fail_label - idx) * 4
    code[idx] = bne(T1, T3, offset)

# Fixup poll_loop BNE (backwards branch)
# The beq for polling is at index POLL_LOOP + 2 (3rd instruction in poll block)
# Actually let me recalculate...
# Poll loop starts at POLL_LOOP
#   POLL_LOOP + 0: lw t1, 4(t0)
#   POLL_LOOP + 1: andi t1, t1, 2
#   POLL_LOOP + 2: beq t1, zero, -12
#   Target = POLL_LOOP (current PC = POLL_LOOP+2, need to go back 3 instructions)
#   Offset = (POLL_LOOP - (POLL_LOOP + 2)) * 4 = -8
# Wait, PC-relative branch in RISC-V: offset is relative to PC of the branch instruction
# PC of branch = (POLL_LOOP + 2) * 4
# Target = POLL_LOOP * 4
# Offset = Target - PC = POLL_LOOP*4 - (POLL_LOOP+2)*4 = -8
code[POLL_LOOP + 2] = beq(T1, ZERO, -8)

# Fixup the jal (j .) instructions
# jal ZERO, 0 at done_label: offset = 0 (jump to self)
code[done_label] = jal(ZERO, 0)
code[len(code) - 1] = jal(ZERO, 0)  # last instruction, fail done loop

# ========== Output ==========
# Pad to reasonable SRAM size
SRAM_WORDS = 1024
while len(code) < SRAM_WORDS:
    code.append(0)

# Write hex file
hex_path = sys.argv[1] if len(sys.argv) > 1 else "soc_test.hex"
with open(hex_path, "w") as f:
    for w in code[:SRAM_WORDS]:
        f.write(f"{w & 0xFFFFFFFF:08x}\n")

print(f"Generated {hex_path}: {len(code)} words ({len(code)*4} bytes)")
print(f"Verify section starts at word {verify_start} (addr 0x{verify_start*4:04x})")
print(f"Pass label at word {pass_label} (addr 0x{pass_label*4:04x})")
print(f"Fail label at word {fail_label} (addr 0x{fail_label*4:04x})")

# Verify poll loop
print(f"\nPoll loop at word {POLL_LOOP} (addr 0x{POLL_LOOP*4:04x})")
print(f"  lw     t1, 4(t0)   = 0x{code[POLL_LOOP]:08x}")
print(f"  andi   t1, t1, 2   = 0x{code[POLL_LOOP+1]:08x}")
print(f"  beq    t1, x0, -8  = 0x{code[POLL_LOOP+2]:08x}")

# Disassemble check section
print(f"\nVerify section:")
for i in range(verify_start, fail_label):
    print(f"  [{i:3d}] 0x{code[i]:08x}")
