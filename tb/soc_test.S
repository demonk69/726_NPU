# =============================================================================
# soc_test.S - RISC-V test firmware for NPU SoC integration test
#
# This firmware:
#   1. Configures NPU registers (matrix dims, DMA addresses, PE config)
#   2. Starts NPU computation (4×4 INT8 matrix multiply)
#   3. Polls NPU status register for completion
#   4. Verifies result from DRAM
#   5. Writes PASS (0xAA) or FAIL (0xFF) marker to DRAM[0x0F00]
#
# Memory Map:
#   0x0000_0000 - 0x0000_0FFF  SRAM (code + data)
#   0x0000_1000 - 0x0000_FFFF  DRAM (NPU data area)
#   0x0200_0000 - 0x0200_003F  NPU Registers
#
# NPU Register Map (offset from 0x0200_0000):
#   0x00  CTRL      bit0=start, bit1=abort, [3:2]=mode, [5:4]=stat
#   0x04  STATUS    bit0=busy, bit1=done
#   0x08  INT_EN    interrupt enable
#   0x10  M_DIM     matrix M dimension
#   0x14  N_DIM     matrix N dimension
#   0x18  K_DIM     matrix K dimension
#   0x20  W_ADDR    weight base address in DRAM
#   0x24  A_ADDR    activation base address in DRAM
#   0x28  R_ADDR    result base address in DRAM
#   0x30  ARR_CFG   [3:0]=rows, [7:4]=cols
#
# DRAM Data Layout (pre-loaded by testbench):
#   Weight:  @ 0x0000_1000  (4×4 INT8 identity matrix)
#   Activ:   @ 0x0000_1200  (4×4 INT8, [[1..4],[5..8],[9..12],[13..16]])
#   Result:  @ 0x0000_1400  (expected: same as activation for identity W)
#   Marker:  @ 0x0000_0F00  (firmware writes PASS/FAIL here)
# =============================================================================

    .section .text.init, "ax", @progbits
    .global _start
    .option norvc

_start:
    # ---- Save stack pointer ----
    addi    sp, sp, -16
    sw      ra, 12(sp)

    # ---- Load NPU base address ----
    li      t0, 0x02000000          # NPU register base

    # ==== Configure NPU registers ====

    # M_DIM = 4
    li      t1, 4
    sw      t1, 0x10(t0)

    # N_DIM = 4
    li      t1, 4
    sw      t1, 0x14(t0)

    # K_DIM = 4
    li      t1, 4
    sw      t1, 0x18(t0)

    # W_ADDR = 0x0000_1000 (weight in DRAM)
    li      t1, 0x00001000
    sw      t1, 0x20(t0)

    # A_ADDR = 0x0000_1200 (activation in DRAM)
    li      t1, 0x00001200
    sw      t1, 0x24(t0)

    # R_ADDR = 0x0000_1400 (result in DRAM)
    li      t1, 0x00001400
    sw      t1, 0x28(t0)

    # ARR_CFG = 0x44 (4 rows × 4 cols)
    li      t1, 0x44
    sw      t1, 0x30(t0)

    # Enable interrupt (optional, we'll poll instead)
    li      t1, 1
    sw      t1, 0x08(t0)

    # ==== Start NPU ====
    # CTRL = 0x01 (start=1, mode=INT8, stat=WS)
    li      t1, 0x01
    sw      t1, 0x00(t0)

    # ==== Poll for completion ====
poll_loop:
    lw      t1, 0x04(t0)           # Read STATUS register
    andi    t1, t1, 0x02           # Check bit1 (done)
    beqz    t1, poll_loop

    # ==== Clear interrupt flag ====
    li      t1, 0x01
    sw      t1, 0x0C(t0)           # Write INT_CLR

    # ==== Verify result ====
    # Result is at DRAM[0x1400..0x141F] (4×4 INT8 = 16 bytes = 4 words)
    # Expected: C = A × I = A = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    # Stored as 32-bit words: each word = 4 INT8 values

    li      t2, 0x00001400         # Result base address
    li      t3, 0                  # row counter
    li      t5, 4                  # max rows
    li      t6, 0                  # error flag

verify_row:
    beq     t3, t5, verify_done

    lw      t1, 0(t2)             # Load result word
    # Compute expected value
    # Row 0: 1,2,3,4 -> 0x04030201
    # Row 1: 5,6,7,8 -> 0x08070605
    # Row 2: 9,10,11,12 -> 0x0C0B0A09
    # Row 3: 13,14,15,16 -> 0x100F0E0D
    mv      t4, t3
    slli    t4, t4, 2              # t4 = row * 4
    add     t4, t4, t3             # t4 = row * 5... no, we need byte value

    # Simpler approach: compute expected packed word
    # expected_byte = row*4 + col, for col in [0..3]
    li      t4, 1
    slli    t7, t3, 2              # t7 = row * 4
    add     t4, t7, t4             # t4 = row*4 + 1 (first element of this row)
    # Pack 4 bytes: t4, t4+1, t4+2, t4+3 into expected
    li      t7, 0
    slli    t7, t4, 24
    addi    t8, t4, 1
    slli    t8, t8, 16
    or      t7, t7, t8
    addi    t8, t4, 2
    slli    t8, t8, 8
    or      t7, t7, t8
    or      t7, t7, t4             # t7 = expected packed word (little-endian: b3b2b1b0)

    bne     t1, t7, verify_fail

    addi    t2, t2, 4
    addi    t3, t3, 1
    j       verify_row

verify_fail:
    li      t6, 1                  # set error flag

verify_done:
    # ==== Write PASS/FAIL marker ====
    li      t0, 0x00000F00         # marker address in DRAM
    beqz    t6, write_pass

write_fail:
    li      t1, 0x000000FF         # FAIL magic
    sw      t1, 0(t0)
    j       end

write_pass:
    li      t1, 0x000000AA         # PASS magic
    sw      t1, 0(t0)

end:
    # Infinite loop (testbench will detect the marker and $finish)
    j       end

    .option rvc
