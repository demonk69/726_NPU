#!/usr/bin/env python3
# =============================================================================
# gen_repopt_two_layer_soc_case.py - Generate a two-layer RepOpt SoC case.
#
# Flow:
#   stage1_0_conv full layer -> CPU bias/ReLU/requant + NCHW repack
#   -> stage1_1_conv selected tile window -> CPU bias/ReLU/requant
#   -> optional CPU NCHW repack + MaxPool stage1 strip
#
# The first layer always runs full because the second 3x3 same-pad conv needs
# the complete repacked first-layer tensor as its input feature map.
# =============================================================================

import argparse
import json
import sys
import warnings
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
TB_DIR = PROJECT_ROOT / "tb"
if str(TB_DIR) not in sys.path:
    sys.path.insert(0, str(TB_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from assemble_soc_test import ADDI, ANDI, BEQZ, BNE, J, LW, MV, SW, i_type, li_insns, r_type, reg, s_type  # noqa: E402
from gen_repopt_tile_case import (  # noqa: E402
    build_tile_matrices,
    load_cifar_sample,
    load_json,
    pack4_int8,
    quantize_qint8,
    signed32,
    tensor_scalar,
    unwrap_state_dict,
)
from run_repopt_vgg_host import conv2d_acc_npu, load_int32_hex, maxpool2d_cpu, requant_qint8  # noqa: E402


NPU_BASE = 0x02000000
PASS_MARKER = 0x000000AA
FAIL_MARKER = 0x000000FF
MIN_DRAM_WORDS = 163840

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

CTRL_TILE_OS_INT8 = 0x00000011
ARR_TILE4 = 0x00000080
CFG_4X4 = 0x00000000

BASE_ADDR = 0x00002000
REQUANT_SHIFT = 24


def ADD(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)


def MUL(rd, rs1, rs2):
    return r_type(0x01, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)


def OR(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x6, reg(rd), 0x33)


def SRLI(rd, rs1, shamt):
    return i_type(shamt & 0x1F, reg(rs1), 0x5, reg(rd), 0x13)


def SLLI(rd, rs1, shamt):
    return i_type((shamt & 0x1F) | (0x00 << 5), reg(rs1), 0x1, reg(rd), 0x13)


def SLTI(rd, rs1, imm):
    return i_type(imm, reg(rs1), 0x2, reg(rd), 0x13)


def LB(rd, rs1, imm):
    return i_type(imm, reg(rs1), 0x0, reg(rd), 0x03)


def BLT(rs1, rs2, byte_offset):
    imm = byte_offset & 0x1FFE
    b12 = (imm >> 12) & 1
    b11 = (imm >> 11) & 1
    b10_5 = (imm >> 5) & 0x3F
    b4_1 = (imm >> 1) & 0xF
    return (
        (b12 << 31)
        | (b10_5 << 25)
        | (reg(rs2) << 20)
        | (reg(rs1) << 15)
        | (0x4 << 12)
        | (b4_1 << 8)
        | (b11 << 7)
        | 0x63
    )


def BGE(rs1, rs2, byte_offset):
    imm = byte_offset & 0x1FFE
    b12 = (imm >> 12) & 1
    b11 = (imm >> 11) & 1
    b10_5 = (imm >> 5) & 0x3F
    b4_1 = (imm >> 1) & 0xF
    return (
        (b12 << 31)
        | (b10_5 << 25)
        | (reg(rs2) << 20)
        | (reg(rs1) << 15)
        | (0x5 << 12)
        | (b4_1 << 8)
        | (b11 << 7)
        | 0x63
    )


def SB(rs2, rs1, imm):
    return s_type(imm, reg(rs2), reg(rs1), 0x0, 0x23)


def require_torch():
    try:
        import torch  # noqa: WPS433
    except Exception as exc:
        raise SystemExit(
            "PyTorch is required to generate the RepOpt two-layer SoC case. "
            "Install CPU PyTorch with the command in tools/pth/README.md.\n"
            f"Import error: {exc}"
        ) from exc
    return torch


def align(value, alignment):
    return (value + alignment - 1) & ~(alignment - 1)


def write_hex(path, words):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for word in words:
            f.write(f"{int(word) & 0xFFFFFFFF:08x}\n")


def pack_int8_words(values):
    words = []
    for idx in range(0, len(values), 4):
        words.append(pack4_int8(values[idx:idx + 4]))
    return words


def requant_fixed(acc, multiplier_q):
    q = (int(acc) * int(multiplier_q) + (1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT
    if q < -128:
        return -128
    if q > 127:
        return 127
    return q


def collect_q_tile(q_tensor, m_base, n_base):
    out = []
    ow = int(q_tensor.shape[3])
    for row in range(4):
        global_m = m_base + row
        out_h = global_m // ow
        out_w = global_m % ow
        for col in range(4):
            out.append(int(q_tensor[0, n_base + col, out_h, out_w].item()))
    return out


def collect_pool_strip(pool_tensor, channel_base, channel_count, row_base, row_count):
    out = []
    for ch in range(channel_base, channel_base + channel_count):
        for row in range(row_base, row_base + row_count):
            for col in range(int(pool_tensor.shape[3])):
                out.append(int(pool_tensor[0, ch, row, col].item()))
    return out


def collect_pool_window(pool_tensor, channel_base, channel_count, row_base, row_count, col_base, col_count):
    out = []
    for ch in range(channel_base, channel_base + channel_count):
        for row in range(row_base, row_base + row_count):
            for col in range(col_base, col_base + col_count):
                out.append(int(pool_tensor[0, ch, row, col].item()))
    return out


def stage1_required_stage0_m_bases(layer1, stage1_m_bases):
    regs = layer1["registers"]
    stride_pad = int(regs["CONV_STRIDE_PAD"])
    dilation_word = int(regs["CONV_DILATION"])
    ow = int(layer1["input_shape"][3])
    ow_out = int(layer1["output_shape"][3])
    kh = int(layer1["weight_shape"][2])
    kw = int(layer1["weight_shape"][3])
    stride_h = stride_pad & 0xFF
    stride_w = (stride_pad >> 8) & 0xFF
    pad_h = (stride_pad >> 16) & 0xFF
    pad_w = (stride_pad >> 24) & 0xFF
    dilation_h = dilation_word & 0xFF
    dilation_w = (dilation_word >> 8) & 0xFF

    needed_m = set()
    for m_base in stage1_m_bases:
        for global_m in range(m_base, m_base + 4):
            out_h = global_m // ow_out
            out_w = global_m % ow_out
            for ker_h in range(kh):
                for ker_w in range(kw):
                    in_h = out_h * stride_h + ker_h * dilation_h - pad_h
                    in_w = out_w * stride_w + ker_w * dilation_w - pad_w
                    if 0 <= in_h < layer1["input_shape"][2] and 0 <= in_w < layer1["input_shape"][3]:
                        needed_m.add(in_h * ow + in_w)
    return sorted({(value // 4) * 4 for value in needed_m})


def stage1_pool_required_stage1_m_bases(row_base, row_count, col_base, col_count):
    needed_m = set()
    for out_row in range(row_base * 2, (row_base + row_count) * 2):
        for out_col in range(col_base * 2, (col_base + col_count) * 2):
            needed_m.add(out_row * 32 + out_col)
    return sorted({(value // 4) * 4 for value in needed_m})


def build_case(plan, plan_dir, state_dict, args):
    layer0 = next(
        (item for item in plan["layers"] if item.get("name") == "stage1_0_conv" and item.get("op") == "conv2d"),
        None,
    )
    layer1 = next(
        (item for item in plan["layers"] if item.get("name") == "stage1_1_conv" and item.get("op") == "conv2d"),
        None,
    )
    pool_layer = next(
        (item for item in plan["layers"] if item.get("name") == "stage1_pool" and item.get("op") == "maxpool2d"),
        None,
    )
    if layer0 is None or layer1 is None:
        raise ValueError("stage1_0_conv or stage1_1_conv not found in plan")
    if args.with_pool and pool_layer is None:
        raise ValueError("stage1_pool not found in plan")
    if list(layer0["input_shape"]) != list(plan["input"]["shape"]):
        raise ValueError("this generator expects stage1_0_conv to be the first layer")

    x_float, label = load_cifar_sample(args.data_root, args.index)
    in_scale = float(tensor_scalar(state_dict[plan["input"]["scale_key"]]))
    in_zp = int(tensor_scalar(state_dict[plan["input"]["zero_point_key"]]))
    x_q = quantize_qint8(x_float, in_scale, in_zp)

    qweight0 = state_dict[layer0["weight_key"]]
    qweight1 = state_dict[layer1["weight_key"]]
    bias0 = load_int32_hex(plan_dir / layer0["assets"]["bias_int32_hex"])
    bias1 = load_int32_hex(plan_dir / layer1["assets"]["bias_int32_hex"])

    current1 = requant_qint8(
        conv2d_acc_npu(x_q, qweight0, bias0, layer0),
        layer0["cpu_requant_after_npu"]["multipliers"],
        layer0["cpu_requant_after_npu"]["output_zero_point"],
    )
    current2 = requant_qint8(
        conv2d_acc_npu(current1, qweight1, bias1, layer1),
        layer1["cpu_requant_after_npu"]["multipliers"],
        layer1["cpu_requant_after_npu"]["output_zero_point"],
    )
    current_pool = maxpool2d_cpu(current2, pool_layer) if args.with_pool else None

    m_dim0 = int(layer0["registers"]["M_DIM"])
    n_dim0 = int(layer0["registers"]["N_DIM"])
    k_dim0 = int(layer0["registers"]["K_DIM"])
    stage1_m_bases = list(range(args.m_base2, args.m_base2 + args.m_tiles2 * 4, 4))
    if args.with_pool:
        stage1_m_bases = stage1_pool_required_stage1_m_bases(
            args.pool_row_base, args.pool_rows, args.pool_col_base, args.pool_cols
        )
    stage0_m_bases = stage1_required_stage0_m_bases(layer1, stage1_m_bases)
    m_tiles0 = len(stage0_m_bases)
    n_tiles0 = n_dim0 // 4
    tile_count0 = m_tiles0 * n_tiles0

    m_dim1 = int(layer1["registers"]["M_DIM"])
    n_dim1 = int(layer1["registers"]["N_DIM"])
    if args.m_base2 + args.m_tiles2 * 4 > m_dim1:
        raise ValueError(f"requested stage1_1 M window exceeds M={m_dim1}")
    if args.n_base2 + args.n_tiles2 * 4 > n_dim1:
        raise ValueError(f"requested stage1_1 N window exceeds N={n_dim1}")
    k_dim1 = int(layer1["registers"]["K_DIM"])
    tile_count1 = len(stage1_m_bases) * args.n_tiles2

    a_tiles0 = []
    for m_base in stage0_m_bases:
        a_tile, _ = build_tile_matrices(x_q, qweight0, layer0, m_base, 0)
        a_tiles0.append(a_tile)

    w_tiles0 = []
    for nt in range(n_tiles0):
        _, w_tile = build_tile_matrices(x_q, qweight0, layer0, 0, nt * 4)
        w_tiles0.append(w_tile)

    w_tiles1 = []
    for nt in range(args.n_tiles2):
        _, w_tile = build_tile_matrices(current1, qweight1, layer1, stage1_m_bases[0], args.n_base2 + nt * 4)
        w_tiles1.append(w_tile)

    multiplier_q0 = [
        int(round(value * (1 << REQUANT_SHIFT))) for value in layer0["cpu_requant_after_npu"]["multipliers"]
    ]
    multiplier_q1 = [
        int(round(value * (1 << REQUANT_SHIFT)))
        for value in layer1["cpu_requant_after_npu"]["multipliers"][args.n_base2 : args.n_base2 + args.n_tiles2 * 4]
    ]
    bias_window1 = [int(v) for v in bias1[args.n_base2 : args.n_base2 + args.n_tiles2 * 4]]

    expected_q2 = []
    for m_base in stage1_m_bases:
        for nt in range(args.n_tiles2):
            expected_q2.extend(collect_q_tile(current2, m_base, args.n_base2 + nt * 4))
    expected_pool_values = (
        collect_pool_window(
            current_pool,
            args.n_base2,
            args.n_tiles2 * 4,
            args.pool_row_base,
            args.pool_rows,
            args.pool_col_base,
            args.pool_cols,
        )
        if args.with_pool
        else []
    )

    repack0_values = []
    for ch in range(current1.shape[1]):
        for out_h in range(current1.shape[2]):
            for out_w in range(current1.shape[3]):
                repack0_values.append(int(current1[0, ch, out_h, out_w].item()))

    a0_base = BASE_ADDR
    w0_base = align(a0_base + m_tiles0 * k_dim0 * 4, 0x100)
    bias0_base = align(w0_base + n_tiles0 * k_dim0 * 4, 0x100)
    mult0_base = align(bias0_base + n_dim0 * 4, 0x100)
    r0_base = align(mult0_base + n_dim0 * 4, 0x100)
    q0_base = align(r0_base + tile_count0 * 16 * 4, 0x100)
    repack0_base = align(q0_base + tile_count0 * 16 * 4, 0x100)
    repack0_words = align(len(repack0_values), 4) // 4
    w1_base = align(repack0_base + repack0_words * 4, 0x100)
    bias1_base = align(w1_base + args.n_tiles2 * k_dim1 * 4, 0x100)
    mult1_base = align(bias1_base + len(bias_window1) * 4, 0x100)
    a1_base = align(mult1_base + len(multiplier_q1) * 4, 0x100)
    r1_base = align(a1_base + k_dim1 * 4, 0x100)
    q1_base = align(r1_base + 16 * 4, 0x100)
    repack1_base = align(q1_base + tile_count1 * 16 * 4, 0x100)
    repack1_words = align(int(layer1["output_shape"][1]) * int(layer1["output_shape"][2]) * int(layer1["output_shape"][3]), 4) // 4
    pool_base = align(repack1_base + repack1_words * 4, 0x100)
    pool_words = align(len(expected_pool_values), 4) // 4
    result_end = pool_base + pool_words * 4 if args.with_pool else q1_base + tile_count1 * 16 * 4
    marker_addr = align(result_end, 0x100)
    dram_words = max(MIN_DRAM_WORDS, align(marker_addr + 4, 0x1000) // 4)

    layout = {
        "a0_base": a0_base,
        "w0_base": w0_base,
        "bias0_base": bias0_base,
        "mult0_base": mult0_base,
        "r0_base": r0_base,
        "q0_base": q0_base,
        "repack0_base": repack0_base,
        "repack0_words": repack0_words,
        "w1_base": w1_base,
        "bias1_base": bias1_base,
        "mult1_base": mult1_base,
        "a1_base": a1_base,
        "r1_base": r1_base,
        "q1_base": q1_base,
        "repack1_base": repack1_base,
        "repack1_words": repack1_words,
        "pool_base": pool_base,
        "pool_words": pool_words,
        "marker_addr": marker_addr,
        "dram_words": dram_words,
        "m_dim0": m_dim0,
        "n_dim0": n_dim0,
        "k_dim0": k_dim0,
        "stage0_m_bases": stage0_m_bases,
        "m_tiles0": m_tiles0,
        "n_tiles0": n_tiles0,
        "tile_count0": tile_count0,
        "m_dim1": m_dim1,
        "n_dim1": n_dim1,
        "k_dim1": k_dim1,
        "stage1_m_bases": stage1_m_bases,
        "tile_count1": tile_count1,
        "a_tiles0": a_tiles0,
        "w_tiles0": w_tiles0,
        "w_tiles1": w_tiles1,
        "bias0_all": [int(v) for v in bias0],
        "mult0_all": multiplier_q0,
        "bias1_window": bias_window1,
        "mult1_window": multiplier_q1,
        "repack0_values": repack0_values,
        "with_pool": args.with_pool,
        "pool_row_base": args.pool_row_base,
        "pool_rows": args.pool_rows,
        "pool_col_base": args.pool_col_base,
        "pool_cols": args.pool_cols,
        "expected_pool_values": expected_pool_values,
    }
    return layer0, layer1, pool_layer, label, in_scale, in_zp, expected_q2, layout


def write_dram_init(out_dir, layout):
    dram = [0 for _ in range(layout["dram_words"])]

    for mt, a_tile in enumerate(layout["a_tiles0"]):
        base = (layout["a0_base"] >> 2) + mt * layout["k_dim0"]
        for k_index, lanes in enumerate(a_tile):
            dram[base + k_index] = pack4_int8(lanes)

    for nt, w_tile in enumerate(layout["w_tiles0"]):
        base = (layout["w0_base"] >> 2) + nt * layout["k_dim0"]
        for k_index, lanes in enumerate(w_tile):
            dram[base + k_index] = pack4_int8(lanes)

    for idx, value in enumerate(layout["bias0_all"]):
        dram[(layout["bias0_base"] >> 2) + idx] = value & 0xFFFFFFFF

    for idx, value in enumerate(layout["mult0_all"]):
        dram[(layout["mult0_base"] >> 2) + idx] = value & 0xFFFFFFFF

    for idx in range(layout["tile_count0"] * 16):
        dram[(layout["r0_base"] >> 2) + idx] = 0
        dram[(layout["q0_base"] >> 2) + idx] = 0

    for idx in range(layout["repack0_words"]):
        dram[(layout["repack0_base"] >> 2) + idx] = 0

    for nt, w_tile in enumerate(layout["w_tiles1"]):
        base = (layout["w1_base"] >> 2) + nt * layout["k_dim1"]
        for k_index, lanes in enumerate(w_tile):
            dram[base + k_index] = pack4_int8(lanes)

    for idx, value in enumerate(layout["bias1_window"]):
        dram[(layout["bias1_base"] >> 2) + idx] = value & 0xFFFFFFFF

    for idx, value in enumerate(layout["mult1_window"]):
        dram[(layout["mult1_base"] >> 2) + idx] = value & 0xFFFFFFFF

    for idx in range(layout["k_dim1"]):
        dram[(layout["a1_base"] >> 2) + idx] = 0
    for idx in range(16):
        dram[(layout["r1_base"] >> 2) + idx] = 0
    for idx in range(layout["tile_count1"] * 16):
        dram[(layout["q1_base"] >> 2) + idx] = 0
    for idx in range(layout["repack1_words"]):
        dram[(layout["repack1_base"] >> 2) + idx] = 0
    for idx in range(layout["pool_words"]):
        dram[(layout["pool_base"] >> 2) + idx] = 0

    dram[layout["marker_addr"] >> 2] = 0
    write_hex(out_dir / "dram_init.hex", dram)
    write_hex(out_dir / "expected_stage1_repack.hex", pack_int8_words(layout["repack0_values"]))


def assemble_firmware(layout, args):
    insns = []
    labels = {}

    def emit(*words):
        for word in words:
            insns.append(word)

    def label(name):
        labels[name] = len(insns)

    def patch_beqz(idx, target_label, rs="t1"):
        offset = (labels[target_label] - idx) * 4
        insns[idx] = BEQZ(rs, offset)

    def patch_bne(idx, target_label, rs1="t1", rs2="t3"):
        offset = (labels[target_label] - idx) * 4
        insns[idx] = BNE(rs1, rs2, offset)

    def patch_j(idx, target_label):
        offset = (labels[target_label] - idx) * 4
        insns[idx] = J(offset)

    def patch_blt(idx, target_label, rs1, rs2):
        offset = (labels[target_label] - idx) * 4
        insns[idx] = BLT(rs1, rs2, offset)

    def patch_bge(idx, target_label, rs1, rs2):
        offset = (labels[target_label] - idx) * 4
        insns[idx] = BGE(rs1, rs2, offset)

    def write_reg_imm(offset, value):
        emit(*li_insns("t1", int(value)))
        emit(SW("t1", "s0", offset))

    def write_reg_reg(offset, rs):
        emit(SW(rs, "s0", offset))

    def run_current_tile(k_dim):
        write_reg_imm(REG_CTRL, 0)
        write_reg_imm(REG_M_DIM, 4)
        write_reg_imm(REG_N_DIM, 4)
        write_reg_imm(REG_K_DIM, k_dim)
        write_reg_reg(REG_W_ADDR, "s4")
        write_reg_reg(REG_A_ADDR, "s3")
        write_reg_reg(REG_R_ADDR, "s5")
        write_reg_imm(REG_ARR_CFG, ARR_TILE4)
        write_reg_imm(REG_CFG_SHAPE, CFG_4X4)
        write_reg_imm(REG_CTRL, CTRL_TILE_OS_INT8)

        poll_label = f"poll_{len(insns)}"
        label(poll_label)
        emit(LW("t1", "s0", REG_STATUS))
        emit(ANDI("t1", "t1", 2))
        beq_idx = len(insns)
        emit(0)
        patch_beqz(beq_idx, poll_label)
        write_reg_imm(REG_CTRL, 0)

    def emit_postprocess_tile_with_repack():
        emit(MV("t2", "s5"))
        emit(*li_insns("t6", 0))

        emit(MV("a2", "s2"))
        emit(MUL("a2", "a2", "a7"))
        emit(ADD("a2", "a2", "a6"))
        emit(MV("a3", "t0"))
        emit(ADD("a2", "a2", "a3"))

        label("post0_row_loop")
        emit(*li_insns("t5", 0))
        emit(MV("a0", "a2"))
        emit(MV("a4", "s10"))
        emit(MV("a5", "s11"))

        label("post0_col_loop")
        emit(LW("t1", "t2", 0))
        emit(LW("t3", "a4", 0))
        emit(ADD("t1", "t1", "t3"))

        emit(SRLI("t4", "t1", 31))
        beq_idx = len(insns)
        emit(0)
        emit(*li_insns("t1", 0))
        label("post0_relu_ok")
        patch_beqz(beq_idx, "post0_relu_ok", "t4")

        emit(LW("t3", "a5", 0))
        emit(MUL("t1", "t1", "t3"))
        emit(*li_insns("t3", 1 << (REQUANT_SHIFT - 1)))
        emit(ADD("t1", "t1", "t3"))
        emit(SRLI("t1", "t1", REQUANT_SHIFT))

        emit(SLTI("t4", "t1", 128))
        clamp_idx = len(insns)
        emit(0)
        store_j_idx = len(insns)
        emit(0)
        label("post0_clamp")
        emit(*li_insns("t1", 127))
        label("post0_store")
        patch_beqz(clamp_idx, "post0_clamp", "t4")
        patch_j(store_j_idx, "post0_store")

        emit(SW("t1", "s6", 0))
        emit(SB("t1", "a0", 0))
        emit(ADDI("t2", "t2", 4))
        emit(ADDI("s6", "s6", 4))
        emit(ADD("a0", "a0", "s9"))
        emit(ADDI("a4", "a4", 4))
        emit(ADDI("a5", "a5", 4))
        emit(ADDI("t5", "t5", 1))
        emit(*li_insns("t3", 4))
        bne_col_idx = len(insns)
        emit(0)
        patch_bne(bne_col_idx, "post0_col_loop", "t5", "t3")

        emit(ADDI("a2", "a2", 1))
        emit(ADDI("t6", "t6", 1))
        emit(*li_insns("t3", 4))
        bne_row_idx = len(insns)
        emit(0)
        patch_bne(bne_row_idx, "post0_row_loop", "t6", "t3")

    def emit_postprocess_tile_q_only():
        emit(MV("t2", "s5"))
        emit(*li_insns("t6", 0))

        label("post1_row_loop")
        emit(*li_insns("t5", 0))
        emit(MV("a4", "s10"))
        emit(MV("a5", "s11"))

        label("post1_col_loop")
        emit(LW("t1", "t2", 0))
        emit(LW("t3", "a4", 0))
        emit(ADD("t1", "t1", "t3"))

        emit(SRLI("t4", "t1", 31))
        beq_idx = len(insns)
        emit(0)
        emit(*li_insns("t1", 0))
        label("post1_relu_ok")
        patch_beqz(beq_idx, "post1_relu_ok", "t4")

        emit(LW("t3", "a5", 0))
        emit(MUL("t1", "t1", "t3"))
        emit(*li_insns("t3", 1 << (REQUANT_SHIFT - 1)))
        emit(ADD("t1", "t1", "t3"))
        emit(SRLI("t1", "t1", REQUANT_SHIFT))

        emit(SLTI("t4", "t1", 128))
        clamp_idx = len(insns)
        emit(0)
        store_j_idx = len(insns)
        emit(0)
        label("post1_clamp")
        emit(*li_insns("t1", 127))
        label("post1_store")
        patch_beqz(clamp_idx, "post1_clamp", "t4")
        patch_j(store_j_idx, "post1_store")

        emit(SW("t1", "s6", 0))
        emit(ADDI("t2", "t2", 4))
        emit(ADDI("s6", "s6", 4))
        emit(ADDI("a4", "a4", 4))
        emit(ADDI("a5", "a5", 4))
        emit(ADDI("t5", "t5", 1))
        emit(*li_insns("t3", 4))
        bne_col_idx = len(insns)
        emit(0)
        patch_bne(bne_col_idx, "post1_col_loop", "t5", "t3")

        emit(ADDI("t6", "t6", 1))
        emit(*li_insns("t3", 4))
        bne_row_idx = len(insns)
        emit(0)
        patch_bne(bne_row_idx, "post1_row_loop", "t6", "t3")

    def emit_postprocess_tile_with_repack1():
        emit(MV("t2", "s5"))
        emit(*li_insns("t6", 0))
        emit(SLLI("a2", "s2", 12))  # n_tile * (4 * 1024)
        emit(ADD("a2", "a2", "a6"))
        emit(ADD("a2", "a2", "t0"))
        emit(*li_insns("a1", 1024))

        label("post1r_row_loop")
        emit(*li_insns("t5", 0))
        emit(MV("a0", "a2"))
        emit(MV("a4", "s10"))
        emit(MV("a5", "s11"))

        label("post1r_col_loop")
        emit(LW("t1", "t2", 0))
        emit(LW("t3", "a4", 0))
        emit(ADD("t1", "t1", "t3"))

        emit(SRLI("t4", "t1", 31))
        beq_idx = len(insns)
        emit(0)
        emit(*li_insns("t1", 0))
        label("post1r_relu_ok")
        patch_beqz(beq_idx, "post1r_relu_ok", "t4")

        emit(LW("t3", "a5", 0))
        emit(MUL("t1", "t1", "t3"))
        emit(*li_insns("t3", 1 << (REQUANT_SHIFT - 1)))
        emit(ADD("t1", "t1", "t3"))
        emit(SRLI("t1", "t1", REQUANT_SHIFT))

        emit(SLTI("t4", "t1", 128))
        clamp_idx = len(insns)
        emit(0)
        store_j_idx = len(insns)
        emit(0)
        label("post1r_clamp")
        emit(*li_insns("t1", 127))
        label("post1r_store")
        patch_beqz(clamp_idx, "post1r_clamp", "t4")
        patch_j(store_j_idx, "post1r_store")

        emit(SW("t1", "s6", 0))
        emit(SB("t1", "a0", 0))
        emit(ADDI("t2", "t2", 4))
        emit(ADDI("s6", "s6", 4))
        emit(ADD("a0", "a0", "a1"))
        emit(ADDI("a4", "a4", 4))
        emit(ADDI("a5", "a5", 4))
        emit(ADDI("t5", "t5", 1))
        emit(*li_insns("t3", 4))
        bne_col_idx = len(insns)
        emit(0)
        patch_bne(bne_col_idx, "post1r_col_loop", "t5", "t3")

        emit(ADDI("a2", "a2", 1))
        emit(ADDI("t6", "t6", 1))
        emit(*li_insns("t3", 4))
        bne_row_idx = len(insns)
        emit(0)
        patch_bne(bne_row_idx, "post1r_row_loop", "t6", "t3")

    def emit_stage1_pool():
        emit(*li_insns("s6", layout["pool_base"]))
        emit(*li_insns("s7", args.n_tiles2 * 4))
        emit(*li_insns("s8", args.pool_rows))
        emit(*li_insns("s1", 0))
        emit(*li_insns("a6", layout["repack1_base"] + args.n_base2 * 1024 + args.pool_row_base * 64 + args.pool_col_base * 2))

        label("pool_ch_loop")
        emit(SLLI("t0", "s1", 10))
        emit(ADD("s3", "a6", "t0"))
        emit(*li_insns("s2", 0))

        label("pool_row_loop")
        emit(SLLI("t0", "s2", 6))
        emit(ADD("t2", "s3", "t0"))
        emit(ADDI("t3", "t2", 32))
        emit(*li_insns("t4", 0))

        label("pool_col_loop")
        emit(SLLI("t5", "t4", 1))
        emit(ADD("a0", "t2", "t5"))
        emit(ADD("a1", "t3", "t5"))
        emit(LB("t1", "a0", 0))
        emit(LB("a2", "a0", 1))
        keep0 = f"pool_keep0_{len(insns)}"
        cmp0 = len(insns)
        emit(0)
        emit(MV("t1", "a2"))
        label(keep0)
        patch_bge(cmp0, keep0, "t1", "a2")
        emit(LB("a2", "a1", 0))
        keep1 = f"pool_keep1_{len(insns)}"
        cmp1 = len(insns)
        emit(0)
        emit(MV("t1", "a2"))
        label(keep1)
        patch_bge(cmp1, keep1, "t1", "a2")
        emit(LB("a2", "a1", 1))
        keep2 = f"pool_keep2_{len(insns)}"
        cmp2 = len(insns)
        emit(0)
        emit(MV("t1", "a2"))
        label(keep2)
        patch_bge(cmp2, keep2, "t1", "a2")
        emit(SB("t1", "s6", 0))
        emit(ADDI("s6", "s6", 1))
        emit(ADDI("t4", "t4", 1))
        emit(*li_insns("a3", args.pool_cols))
        col_bne_idx = len(insns)
        emit(0)
        patch_bne(col_bne_idx, "pool_col_loop", "t4", "a3")

        emit(ADDI("s2", "s2", 1))
        row_bne_idx = len(insns)
        emit(0)
        patch_bne(row_bne_idx, "pool_row_loop", "s2", "s8")

        emit(ADDI("s1", "s1", 1))
        ch_bne_idx = len(insns)
        emit(0)
        patch_bne(ch_bne_idx, "pool_ch_loop", "s1", "s7")

    def emit_build_stage1_input_tile():
        emit(MV("t2", "s3"))
        emit(*li_insns("t3", 0))

        label("l1a_chan_loop")
        emit(MV("t4", "t0"))
        emit(SLLI("t5", "t3", 10))
        emit(*li_insns("t6", 0))

        label("l1a_kh_loop")
        emit(*li_insns("a0", 0))

        label("l1a_kw_loop")
        emit(MV("a1", "t4"))
        emit(*li_insns("a2", 0))

        label("l1a_row_loop")
        emit(*li_insns("t1", 0))
        emit(SRLI("a3", "a1", 5))
        emit(ANDI("a4", "a1", 31))
        emit(ADD("a3", "a3", "t6"))
        emit(ADDI("a3", "a3", -1))
        emit(ADD("a4", "a4", "a0"))
        emit(ADDI("a4", "a4", -1))

        br_h_neg = len(insns)
        emit(0)
        br_h_hi = len(insns)
        emit(0)
        br_w_neg = len(insns)
        emit(0)
        br_w_hi = len(insns)
        emit(0)

        emit(SLLI("a5", "a3", 5))
        emit(ADD("a5", "a5", "a4"))
        emit(ADD("a5", "a5", "t5"))
        emit(ADD("a5", "a5", "a6"))
        emit(LB("t1", "a5", 0))

        label("l1a_store_byte")
        emit(SB("t1", "t2", 0))
        emit(ADDI("t2", "t2", 1))
        emit(ADDI("a1", "a1", 1))
        emit(ADDI("a2", "a2", 1))
        emit(*li_insns("a5", 4))
        row_bne_idx = len(insns)
        emit(0)

        emit(ADDI("a0", "a0", 1))
        emit(*li_insns("a5", 3))
        kw_bne_idx = len(insns)
        emit(0)

        emit(ADDI("t6", "t6", 1))
        emit(*li_insns("a5", 3))
        kh_bne_idx = len(insns)
        emit(0)

        emit(ADDI("t3", "t3", 1))
        emit(*li_insns("a5", 64))
        chan_bne_idx = len(insns)
        emit(0)

        patch_blt(br_h_neg, "l1a_store_byte", "a3", "zero")
        patch_bge(br_h_hi, "l1a_store_byte", "a3", "s9")
        patch_blt(br_w_neg, "l1a_store_byte", "a4", "zero")
        patch_bge(br_w_hi, "l1a_store_byte", "a4", "s9")
        patch_bne(row_bne_idx, "l1a_row_loop", "a2", "a5")
        patch_bne(kw_bne_idx, "l1a_kw_loop", "a0", "a5")
        patch_bne(kh_bne_idx, "l1a_kh_loop", "t6", "a5")
        patch_bne(chan_bne_idx, "l1a_chan_loop", "t3", "a5")

    label("_start")
    emit(*li_insns("sp", 0x00002000))
    emit(*li_insns("s0", NPU_BASE))

    emit(*li_insns("s7", layout["n_tiles0"]))
    emit(*li_insns("s8", layout["m_tiles0"]))
    emit(*li_insns("s9", layout["m_dim0"]))
    emit(*li_insns("s1", 0))
    emit(*li_insns("s3", layout["a0_base"]))
    emit(*li_insns("s5", layout["r0_base"]))
    emit(*li_insns("s6", layout["q0_base"]))
    emit(*li_insns("a6", layout["repack0_base"]))
    emit(*li_insns("a7", layout["m_dim0"] * 4))
    emit(*li_insns("t0", layout["stage0_m_bases"][0]))

    label("m0_loop")
    emit(*li_insns("s2", 0))
    emit(*li_insns("s4", layout["w0_base"]))
    emit(*li_insns("s10", layout["bias0_base"]))
    emit(*li_insns("s11", layout["mult0_base"]))

    label("n0_loop")
    run_current_tile(layout["k_dim0"])
    emit_postprocess_tile_with_repack()

    emit(ADDI("s2", "s2", 1))
    emit(ADDI("s4", "s4", layout["k_dim0"] * 4))
    emit(ADDI("s5", "s5", 16 * 4))
    emit(ADDI("s10", "s10", 4 * 4))
    emit(ADDI("s11", "s11", 4 * 4))
    n0_bne_idx = len(insns)
    emit(0)
    patch_bne(n0_bne_idx, "n0_loop", "s2", "s7")

    emit(ADDI("s1", "s1", 1))
    emit(ADDI("s3", "s3", layout["k_dim0"] * 4))
    if layout["m_tiles0"] > 1:
        for idx, stage0_m_base in enumerate(layout["stage0_m_bases"][1:], start=1):
            emit(*li_insns("t3", idx))
            check_idx = len(insns)
            emit(0)
            emit(*li_insns("a0", stage0_m_base))
            emit(MV("t0", "a0"))
            label(f"m0_base_set_{idx}")
            patch_bne(check_idx, f"m0_base_set_{idx}", "s1", "t3")
    m0_bne_idx = len(insns)
    emit(0)
    patch_bne(m0_bne_idx, "m0_loop", "s1", "s8")

    emit(*li_insns("s7", args.n_tiles2))
    emit(*li_insns("s8", len(layout["stage1_m_bases"])))
    emit(*li_insns("s9", 32))
    emit(*li_insns("s1", 0))
    emit(*li_insns("s3", layout["a1_base"]))
    emit(*li_insns("s5", layout["r1_base"]))
    emit(*li_insns("s6", layout["q1_base"]))
    emit(*li_insns("a6", layout["repack0_base"]))
    emit(*li_insns("a7", layout["k_dim1"] * 4))
    emit(*li_insns("t0", layout["stage1_m_bases"][0]))

    label("m1_loop")
    emit(*li_insns("a6", layout["repack0_base"]))
    emit_build_stage1_input_tile()
    emit(*li_insns("a6", layout["repack1_base"] + args.n_base2 * 1024))
    emit(*li_insns("s2", 0))
    emit(*li_insns("s4", layout["w1_base"]))
    emit(*li_insns("s10", layout["bias1_base"]))
    emit(*li_insns("s11", layout["mult1_base"]))

    label("n1_loop")
    run_current_tile(layout["k_dim1"])
    emit_postprocess_tile_with_repack1()

    emit(ADDI("s2", "s2", 1))
    emit(ADD("s4", "s4", "a7"))
    emit(ADDI("s10", "s10", 4 * 4))
    emit(ADDI("s11", "s11", 4 * 4))
    n1_bne_idx = len(insns)
    emit(0)
    patch_bne(n1_bne_idx, "n1_loop", "s2", "s7")

    emit(ADDI("s1", "s1", 1))
    if len(layout["stage1_m_bases"]) > 1:
        for idx, stage1_m_base in enumerate(layout["stage1_m_bases"][1:], start=1):
            emit(*li_insns("t3", idx))
            check_idx = len(insns)
            emit(0)
            emit(*li_insns("a0", stage1_m_base))
            emit(MV("t0", "a0"))
            label(f"m1_base_set_{idx}")
            patch_bne(check_idx, f"m1_base_set_{idx}", "s1", "t3")
    m1_bne_idx = len(insns)
    emit(0)
    patch_bne(m1_bne_idx, "m1_loop", "s1", "s8")

    if args.with_pool:
        emit_stage1_pool()

    write_pass_j_idx = len(insns)
    emit(0)

    label("verify_fail")
    emit(*li_insns("t0", layout["marker_addr"]))
    emit(*li_insns("t1", FAIL_MARKER))
    emit(SW("t1", "t0", 0))
    end_j_fail = len(insns)
    emit(0)

    label("write_pass")
    emit(*li_insns("t0", layout["marker_addr"]))
    emit(*li_insns("t1", PASS_MARKER))
    emit(SW("t1", "t0", 0))

    label("end")
    end_j_end = len(insns)
    emit(0)

    patch_j(write_pass_j_idx, "write_pass")
    patch_j(end_j_fail, "end")
    patch_j(end_j_end, "end")
    return insns


def write_expected_files(out_dir, expected_q2, layout):
    write_hex(out_dir / "expected_q2.hex", expected_q2)
    if layout["with_pool"]:
        write_hex(out_dir / "expected_pool_words.hex", pack_int8_words(layout["expected_pool_values"]))


def write_debug_expected_a1(out_dir, layout, args):
    import json as _json

    debug = {
        "stage0_m_bases": layout["stage0_m_bases"],
        "stage1_window": {
            "m_base": args.m_base2,
            "n_base": args.n_base2,
            "m_tiles": args.m_tiles2,
            "n_tiles": args.n_tiles2,
            "m_bases": layout["stage1_m_bases"],
        },
        "repack0_base": f"0x{layout['repack0_base']:08x}",
        "a1_base": f"0x{layout['a1_base']:08x}",
    }
    with open(out_dir / "debug_layout.json", "w", encoding="utf-8", newline="\n") as f:
        _json.dump(debug, f, indent=2)


def write_params(out_dir, fw_words, args, label, in_scale, in_zp, expected_q2, layout):
    timeout_cycles = max(12000000, layout["tile_count0"] * 2500 + layout["tile_count1"] * 80000)
    result_hex = (
        f"../sim/{out_dir.name}/expected_pool_words.hex"
        if layout["with_pool"]
        else f"../sim/{out_dir.name}/expected_q2.hex"
    )
    result_base = layout["pool_base"] if layout["with_pool"] else layout["q1_base"]
    result_words = layout["pool_words"] if layout["with_pool"] else len(expected_q2)
    lines = [
        f"`define REP_TWO_SOC_FW_HEX \"../sim/{out_dir.name}/soc_repopt_two_layer.hex\"",
        f"`define REP_TWO_SOC_DRAM_HEX \"../sim/{out_dir.name}/dram_init.hex\"",
        f"`define REP_TWO_SOC_EXPECTED_Q2_HEX \"../sim/{out_dir.name}/expected_q2.hex\"",
        f"`define REP_TWO_SOC_EXPECTED_RESULT_HEX \"{result_hex}\"",
        f"`define REP_TWO_SOC_FW_WORDS {fw_words}",
        f"`define REP_TWO_SOC_DRAM_WORDS {layout['dram_words']}",
        f"`define REP_TWO_SOC_TIMEOUT_CYCLES {timeout_cycles}",
        f"`define REP_TWO_SOC_MARKER_ADDR 32'h{layout['marker_addr']:08x}",
        f"`define REP_TWO_SOC_REPACK0_BASE 32'h{layout['repack0_base']:08x}",
        f"`define REP_TWO_SOC_Q2_BASE 32'h{layout['q1_base']:08x}",
        f"`define REP_TWO_SOC_Q2_COUNT {len(expected_q2)}",
        f"`define REP_TWO_SOC_RESULT_BASE 32'h{result_base:08x}",
        f"`define REP_TWO_SOC_RESULT_WORDS {result_words}",
        f"`define REP_TWO_SOC_WITH_POOL {1 if layout['with_pool'] else 0}",
        f"`define REP_TWO_SOC_STAGE0_TILE_COUNT {layout['tile_count0']}",
        f"`define REP_TWO_SOC_STAGE1_TILE_COUNT {layout['tile_count1']}",
        f"`define REP_TWO_SOC_STAGE1_M_BASE {args.m_base2}",
        f"`define REP_TWO_SOC_STAGE1_N_BASE {args.n_base2}",
        f"`define REP_TWO_SOC_STAGE1_M_TILES {len(layout['stage1_m_bases'])}",
        f"`define REP_TWO_SOC_STAGE1_N_TILES {args.n_tiles2}",
        f"`define REP_TWO_SOC_CIFAR_LABEL {label}",
    ]
    with open(out_dir / "soc_repopt_two_layer_params.vh", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
        f.write("\n")

    meta = {
        "schema": "repopt_two_layer_soc_case_v1",
        "sample_index": args.index,
        "cifar_label": label,
        "input_quant_scale": in_scale,
        "input_quant_zero_point": in_zp,
        "stage1_0_full_tiles": layout["tile_count0"],
        "stage1_1_window": {
            "m_base": args.m_base2,
            "n_base": args.n_base2,
            "m_tiles": len(layout["stage1_m_bases"]),
            "n_tiles": args.n_tiles2,
            "m_bases": layout["stage1_m_bases"],
        },
        "layout": {
            "repack0_base": f"0x{layout['repack0_base']:08x}",
            "w1_base": f"0x{layout['w1_base']:08x}",
            "a1_base": f"0x{layout['a1_base']:08x}",
            "q1_base": f"0x{layout['q1_base']:08x}",
            "repack1_base": f"0x{layout['repack1_base']:08x}",
            "pool_base": f"0x{layout['pool_base']:08x}",
            "marker_addr": f"0x{layout['marker_addr']:08x}",
            "dram_words": layout["dram_words"],
        },
        "expected_q2_count": len(expected_q2),
        "with_pool": layout["with_pool"],
        "pool_row_base": args.pool_row_base,
        "pool_rows": args.pool_rows,
        "pool_col_base": args.pool_col_base,
        "pool_cols": args.pool_cols,
        "note": (
            "First layer runs full and repacks to NCHW int8. Second layer consumes that repacked IFM through "
            "CPU-built tile scratch data. Optional mode repacks stage1_1 output and runs CPU MaxPool strips."
        ),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate a two-layer RepOpt SoC MMIO scheduling case")
    parser.add_argument(
        "--pth",
        default=".06_RepOpt_VGG/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth",
        help="Path to qat_int8_quantized.pth",
    )
    parser.add_argument("--plan", default="sim/pth_repopt_probe/model_plan.json", help="Path to model_plan.json")
    parser.add_argument("--data-root", default=".06_RepOpt_VGG/06_RepOpt_VGG/data", help="CIFAR-10 data root")
    parser.add_argument("--index", type=int, default=0, help="CIFAR-10 test sample index")
    parser.add_argument("--m-base2", type=int, default=0, help="Stage1_1 global GEMM M base")
    parser.add_argument("--n-base2", type=int, default=0, help="Stage1_1 global GEMM N base")
    parser.add_argument("--m-tiles2", type=int, default=2, help="Stage1_1 number of 4-row tiles")
    parser.add_argument("--n-tiles2", type=int, default=2, help="Stage1_1 number of 4-col tiles")
    parser.add_argument("--with-pool", action="store_true", help="Run CPU MaxPool on the stage1_1 strip after repack")
    parser.add_argument("--pool-row-base", type=int, default=0, help="Pooled output row base when --with-pool is set")
    parser.add_argument("--pool-rows", type=int, default=1, help="Number of pooled output rows when --with-pool is set")
    parser.add_argument("--pool-col-base", type=int, default=0, help="Pooled output column base when --with-pool is set")
    parser.add_argument("--pool-cols", type=int, default=4, help="Number of pooled output columns when --with-pool is set")
    parser.add_argument("--out-dir", default="sim/repopt_two_layer_soc", help="Output case directory")
    args = parser.parse_args()

    torch = require_torch()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_path = Path(args.plan).resolve()
    plan = load_json(plan_path)
    if args.with_pool:
        args.m_base2 = 0
        args.m_tiles2 = 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
        checkpoint = torch.load(args.pth, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)

    layer0, layer1, pool_layer, label, in_scale, in_zp, expected_q2, layout = build_case(plan, plan_path.parent, state_dict, args)
    write_dram_init(out_dir, layout)
    write_expected_files(out_dir, expected_q2, layout)
    write_debug_expected_a1(out_dir, layout, args)
    fw_words = assemble_firmware(layout, args)
    write_hex(out_dir / "soc_repopt_two_layer.hex", fw_words)
    write_params(out_dir, len(fw_words), args, label, in_scale, in_zp, expected_q2, layout)

    print(f"generated RepOpt two-layer SoC case: {out_dir}")
    print(f"firmware words: {len(fw_words)}")
    print(f"stage1_0 tiles: {layout['tile_count0']} (full layer)")
    print(
        "stage1_1 tiles: {0} window M[{1}:{2}) N[{3}:{4})".format(
            layout["tile_count1"],
            layout["stage1_m_bases"][0],
            layout["stage1_m_bases"][-1] + 4,
            args.n_base2,
            args.n_base2 + args.n_tiles2 * 4,
        )
    )
    if args.with_pool:
        print(
            "stage1_pool window: rows[{0}:{1}) cols[{2}:{3}) channels[{4}:{5})".format(
                args.pool_row_base,
                args.pool_row_base + args.pool_rows,
                args.pool_col_base,
                args.pool_col_base + args.pool_cols,
                args.n_base2,
                args.n_base2 + args.n_tiles2 * 4,
            )
        )
    print(f"repack0_base: 0x{layout['repack0_base']:08x}, q2_base: 0x{layout['q1_base']:08x}")
    print(f"marker: 0x{layout['marker_addr']:08x}, dram_words: {layout['dram_words']}")


if __name__ == "__main__":
    main()
