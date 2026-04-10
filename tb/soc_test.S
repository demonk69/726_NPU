# =============================================================================
# soc_test.S - RISC-V firmware for NPU SoC integration test
#
# Test: 2x2 INT8 matrix multiply in OS tile-loop mode
#   C = A * B   where:
#     A = [[1, 2],    B = [[5, 6],
#          [3, 4]]         [7, 8]]
#     C = [[19, 22],
#          [43, 50]]
#
# DRAM Layout (pre-loaded by tb_soc.v):
#   W_ADDR 0x1000: B column-major, stride = K*1 bytes rounded to 4
#     B[:,0] = [5, 7] -> word at 0x1000: 0x00000705
#     B[:,1] = [6, 8] -> word at 0x1004: 0x00000806
#   A_ADDR 0x1010: A row-major
#     A[0,:] = [1, 2] -> word at 0x1010: 0x00000201
#     A[1,:] = [3, 4] -> word at 0x1014: 0x00000403
#   R_ADDR 0x1020: 4 x 32-bit result words (written by NPU DMA)
#     C[0][0]=19, C[0][1]=22, C[1][0]=43, C[1][1]=50
#   MARKER 0x2000: 0xAA (PASS) or 0xFF (FAIL)  -- DRAM space (>= 0x1000)
#
# NPU Register Map (NPU_BASE = 0x0200_0000):
#   Offset 0x00: CTRL    bit0=start, bit1=abort, [3:2]=dtype, [5:4]=stat_mode
#                         stat_mode: 00=WS, 01=OS
#                         dtype:     00=INT8, 01=FP16
#   Offset 0x04: STATUS  bit0=busy, bit1=done
#   Offset 0x08: INT_EN
#   Offset 0x0C: INT_CLR
#   Offset 0x10: M_DIM
#   Offset 0x14: N_DIM
#   Offset 0x18: K_DIM
#   Offset 0x20: W_ADDR
#   Offset 0x24: A_ADDR
#   Offset 0x28: R_ADDR
#
# OS mode CTRL value:
#   bit0=1 (start), bit1=0, [3:2]=00 (INT8), [5:4]=01 (OS)
#   CTRL = 0x11
# =============================================================================

    .section .text.init, "ax", @progbits
    .global _start
    .option norvc

_start:
    # ---- Setup stack ----
    li      sp, 0x00001000       # SRAM top (4KB)

    # ---- Load NPU base address ----
    li      t0, 0x02000000       # NPU register base

    # ==== Configure NPU dimensions ====
    # M_DIM = 2 (rows of A, rows of C)
    li      t1, 2
    sw      t1, 0x10(t0)

    # N_DIM = 2 (cols of B, cols of C)
    li      t1, 2
    sw      t1, 0x14(t0)

    # K_DIM = 2 (cols of A = rows of B)
    li      t1, 2
    sw      t1, 0x18(t0)

    # ==== Configure DMA addresses ====
    # W_ADDR = 0x1000 (B matrix column-major)
    li      t1, 0x00001000
    sw      t1, 0x20(t0)

    # A_ADDR = 0x1010 (A matrix row-major)
    li      t1, 0x00001010
    sw      t1, 0x24(t0)

    # R_ADDR = 0x1020 (result area)
    li      t1, 0x00001020
    sw      t1, 0x28(t0)

    # ==== Start NPU (OS mode, INT8) ====
    # CTRL = 0x11: start=1, dtype=INT8 (bits[3:2]=00), stat_mode=OS (bits[5:4]=01)
    li      t1, 0x11
    sw      t1, 0x00(t0)

    # ==== Poll STATUS until done ====
poll_loop:
    lw      t1, 0x04(t0)         # Read STATUS
    andi    t1, t1, 0x02         # Check bit1 (done)
    beqz    t1, poll_loop

    # ==== Verify results ====
    # R_ADDR = 0x1020
    # C[0][0] = 19 = 0x13
    # C[0][1] = 22 = 0x16
    # C[1][0] = 43 = 0x2B
    # C[1][1] = 50 = 0x32
    li      t2, 0x00001020       # result base address
    li      t6, 0                # error flag

    # Check C[0][0] == 19
    lw      t1, 0(t2)
    li      t3, 19
    bne     t1, t3, verify_fail

    # Check C[0][1] == 22
    lw      t1, 4(t2)
    li      t3, 22
    bne     t1, t3, verify_fail

    # Check C[1][0] == 43
    lw      t1, 8(t2)
    li      t3, 43
    bne     t1, t3, verify_fail

    # Check C[1][1] == 50
    lw      t1, 12(t2)
    li      t3, 50
    bne     t1, t3, verify_fail

    # All checks passed
    j       write_pass

verify_fail:
    li      t6, 1

write_marker:
    li      t0, 0x00002000       # marker DRAM address
    beqz    t6, write_pass

write_fail:
    li      t1, 0x000000FF       # FAIL magic
    sw      t1, 0(t0)
    j       end

write_pass:
    li      t0, 0x00002000
    li      t1, 0x000000AA       # PASS magic
    sw      t1, 0(t0)

end:
    j       end                  # infinite loop, testbench detects marker

    .option rvc
