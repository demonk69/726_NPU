#!/usr/bin/env python3
# =============================================================================
# gen_repopt_tile_soc_case.py - Generate a RepOpt tile-window SoC case.
#
# Host generation only prepares input DRAM and firmware. During simulation the
# reference CPU schedules multiple NPU ARR_CFG[7] 4x4 tile-mode GEMMs through
# MMIO, then runs bias/ReLU/requant postprocess in firmware.
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
    expected_tile,
    load_cifar_sample,
    load_json,
    pack4_int8,
    quantize_qint8,
    signed32,
    tensor_scalar,
    unwrap_state_dict,
)


NPU_BASE = 0x02000000
PASS_MARKER = 0x000000AA
FAIL_MARKER = 0x000000FF
MIN_DRAM_WORDS = 262144  # increased for 8x8 full-layer support

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
ARR_TILE = 0x00000080
CFG_SHAPE_DEFAULT = 0x00000000  # 4x4; 1=8x8, 2=16x16, 3=8x32

BASE_ADDR = 0x00002000
REQUANT_SHIFT = 24
PPB_DEPTH = 64  # must match RTL npu_top PPB_DEPTH parameter


def ADD(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)


def MUL(rd, rs1, rs2):
    return r_type(0x01, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)


def SRLI(rd, rs1, shamt):
    return i_type(shamt & 0x1F, reg(rs1), 0x5, reg(rd), 0x13)


def SLTI(rd, rs1, imm):
    return i_type(imm, reg(rs1), 0x2, reg(rd), 0x13)


def SB(rs2, rs1, imm):
    return s_type(imm, reg(rs2), reg(rs1), 0x0, 0x23)


def require_torch():
    try:
        import torch  # noqa: WPS433
    except Exception as exc:
        raise SystemExit(
            "PyTorch is required to generate the RepOpt tile SoC case. "
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


def load_hex_words(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                words.append(int(line, 16) & 0xFFFFFFFF)
    return words


def load_int32_hex(path):
    return [signed32(word) for word in load_hex_words(path)]


def requant_fixed(acc, multiplier_q):
    q = (int(acc) * int(multiplier_q) + (1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT
    if q < -128:
        return -128
    if q > 127:
        return 127
    return q


def build_case_tiles(plan, plan_dir, state_dict, args):
    tile_rows = args.tile_rows
    tile_cols = args.tile_cols
    layer = next(
        (item for item in plan["layers"] if item.get("name") == args.layer_name and item.get("op") == "conv2d"),
        None,
    )
    if layer is None:
        raise ValueError(f"conv layer not found in plan: {args.layer_name}")
    if list(layer["input_shape"]) != list(plan["input"]["shape"]):
        raise ValueError("this SoC tile generator currently supports the first RepOpt Conv layer only")

    x_float, label = load_cifar_sample(args.data_root, args.index)
    in_scale = float(tensor_scalar(state_dict[plan["input"]["scale_key"]]))
    in_zp = int(tensor_scalar(state_dict[plan["input"]["zero_point_key"]]))
    x_q = quantize_qint8(x_float, in_scale, in_zp)
    qweight = state_dict[layer["weight_key"]]
    bias_int32 = load_int32_hex(plan_dir / layer["assets"]["bias_int32_hex"])
    multipliers = layer["cpu_requant_after_npu"]["multipliers"]
    multiplier_q = [int(round(value * (1 << REQUANT_SHIFT))) for value in multipliers]

    m_dim = int(layer["registers"]["M_DIM"])
    n_dim = int(layer["registers"]["N_DIM"])
    if args.m_base + args.m_tiles * tile_rows > m_dim:
        raise ValueError(f"requested M window exceeds M={m_dim}")
    if args.n_base + args.n_tiles * tile_cols > n_dim:
        raise ValueError(f"requested N window exceeds N={n_dim}")

    k_dim = int(layer["registers"]["K_DIM"])
    tile_count = args.m_tiles * args.n_tiles

    a_tiles = []
    for mt in range(args.m_tiles):
        m_base = args.m_base + mt * tile_rows
        a_tile, _ = build_tile_matrices(x_q, qweight, layer, m_base, args.n_base, tile_rows, tile_cols)
        a_tiles.append(a_tile)

    w_tiles = []
    for nt in range(args.n_tiles):
        n_base = args.n_base + nt * tile_cols
        _, w_tile = build_tile_matrices(x_q, qweight, layer, args.m_base, n_base, tile_rows, tile_cols)
        w_tiles.append(w_tile)

    bias_window = [bias_int32[args.n_base + idx] for idx in range(args.n_tiles * tile_cols)]
    multiplier_window = [multiplier_q[args.n_base + idx] for idx in range(args.n_tiles * tile_cols)]

    results_per_tile = tile_rows * tile_cols
    words_per_k_a = (tile_rows + 3) // 4
    words_per_k_w = (tile_cols + 3) // 4

    # Compute padded per-tile stream size (must match controller DMA with SIMD padding)
    kt_elems = ((PPB_DEPTH << 2) // max(tile_rows, tile_cols))
    SIMD = 4
    kpos = 0
    a_padded_words_total = 0
    w_padded_words_total = 0
    while kpos < k_dim:
        k_len = min(k_dim - kpos, kt_elems)
        a_padded_words_total += ((k_len + SIMD - 1) // SIMD) * SIMD * words_per_k_a
        w_padded_words_total += ((k_len + SIMD - 1) // SIMD) * SIMD * words_per_k_w
        kpos += k_len
    a_total_bytes = args.m_tiles * a_padded_words_total * 4
    w_total_bytes = args.n_tiles * w_padded_words_total * 4

    a_base = BASE_ADDR
    w_base = align(a_base + a_total_bytes, 0x100)
    bias_base = align(w_base + w_total_bytes, 0x100)
    multiplier_base = align(bias_base + len(bias_window) * 4, 0x100)
    r_base = align(multiplier_base + len(multiplier_window) * 4, 0x100)
    q_base = align(r_base + tile_count * results_per_tile * 4, 0x100)
    repack_base = align(q_base + tile_count * 16 * 4, 0x100)
    repack_bytes = m_dim * n_dim
    repack_words = align(repack_bytes, 4) // 4
    marker_addr = align(repack_base + repack_words * 4, 0x100)
    dram_words = max(MIN_DRAM_WORDS, align(marker_addr + 4, 0x1000) // 4)

    tiles = []
    for mt in range(args.m_tiles):
        for nt in range(args.n_tiles):
            m_base = args.m_base + mt * 4
            n_base = args.n_base + nt * 4
            tile_idx = mt * args.n_tiles + nt
            m_base = args.m_base + mt * tile_rows
            n_base = args.n_base + nt * tile_cols
            a_tile = a_tiles[mt]
            w_tile = w_tiles[nt]
            expected = expected_tile(a_tile, w_tile, tile_rows, tile_cols)
            post = []
            for idx, raw_word in enumerate(expected):
                out_c = n_base + (idx % tile_cols)
                acc = signed32(raw_word) + bias_int32[out_c]
                if layer.get("activation") == "relu" and acc < 0:
                    acc = 0
                elif layer.get("activation") == "relu6":
                    if acc < 0:
                        acc = 0
                    if acc > 6:
                        acc = 6
                q = requant_fixed(acc, multiplier_q[out_c])
                post.append(
                    {
                        "bias": bias_int32[out_c],
                        "multiplier_q": multiplier_q[out_c],
                        "post_acc": acc,
                        "q": q,
                    }
                )

            # Compute K-padded per-tile stream size
            kt_elems = ((PPB_DEPTH << 2) // max(tile_rows, tile_cols))
            kpos = 0
            padded_k_strides = 0
            while kpos < k_dim:
                k_len = min(k_dim - kpos, kt_elems)
                padded_k_strides += ((k_len + 3) // 4) * 4
                kpos += k_len
            padded_w_bytes = padded_k_strides * words_per_k_w * 4
            padded_a_bytes = padded_k_strides * words_per_k_a * 4

            tiles.append(
                {
                    "m_base": m_base,
                    "n_base": n_base,
                    "w_addr": w_base + nt * padded_w_bytes,
                    "a_addr": a_base + mt * padded_a_bytes,
                    "r_addr": r_base + tile_idx * results_per_tile * 4,
                    "q_addr": q_base + tile_idx * results_per_tile * 4,
                    "k_dim": k_dim,
                    "a_tile": a_tile,
                    "w_tile": w_tile,
                    "expected": expected,
                    "post": post,
                }
            )

    layout = {
        "a_base": a_base,
        "w_base": w_base,
        "bias_base": bias_base,
        "multiplier_base": multiplier_base,
        "r_base": r_base,
        "q_base": q_base,
        "repack_base": repack_base,
        "repack_words": repack_words,
        "marker_addr": marker_addr,
        "dram_words": dram_words,
        "a_tiles": a_tiles,
        "w_tiles": w_tiles,
        "bias_window": bias_window,
        "multiplier_window": multiplier_window,
        "k_dim": k_dim,
        "m_dim": m_dim,
        "n_dim": n_dim,
        "tile_count": tile_count,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "results_per_tile": results_per_tile,
        "words_per_k_a": words_per_k_a,
        "words_per_k_w": words_per_k_w,
        "kt_elems": ((PPB_DEPTH << 2) // max(tile_rows, tile_cols)),
    }
    return layer, label, in_scale, in_zp, tiles, layout


def write_dram_init(out_dir, tiles, layout):
    dram = [0 for _ in range(layout["dram_words"])]

    words_per_k_a = (layout["tile_rows"] + 3) // 4
    words_per_k_w = (layout["tile_cols"] + 3) // 4

    # SIMD padding: round up K count to SIMD_LANES (4) per k_tile,
    # then compute total padded words per tile stream.
    SIMD = 4
    k_dim = layout["k_dim"]
    kt_elems = layout.get("kt_elems", 16)
    # Padded A stream size (words): for each k_tile, pad to SIMD boundary
    a_padded_words = 0
    w_padded_words = 0
    kpos = 0
    while kpos < k_dim:
        k_len = min(k_dim - kpos, kt_elems)
        a_padded_words += ((k_len + SIMD - 1) // SIMD) * SIMD * words_per_k_a
        w_padded_words += ((k_len + SIMD - 1) // SIMD) * SIMD * words_per_k_w
        kpos += k_len

    for mt, a_tile in enumerate(layout["a_tiles"]):
        base = (layout["a_base"] >> 2) + mt * a_padded_words
        k_index_offset = 0
        kpos = 0
        while kpos < k_dim:
            k_len = min(k_dim - kpos, kt_elems)
            for k_rel in range(k_len):
                k_index = kpos + k_rel
                lanes = a_tile[k_index]
                for w in range(words_per_k_a):
                    dram[base + k_index_offset + k_rel * words_per_k_a + w] = pack4_int8(lanes[w*4:w*4+4])
            # pad to SIMD boundary
            pad_k = ((k_len + SIMD - 1) // SIMD) * SIMD
            pad_words = pad_k * words_per_k_a
            for p in range(k_len * words_per_k_a, pad_words):
                dram[base + k_index_offset + p] = 0
            k_index_offset += pad_words
            kpos += k_len

    for nt, w_tile in enumerate(layout["w_tiles"]):
        base = (layout["w_base"] >> 2) + nt * w_padded_words
        k_index_offset = 0
        kpos = 0
        while kpos < k_dim:
            k_len = min(k_dim - kpos, kt_elems)
            for k_rel in range(k_len):
                k_index = kpos + k_rel
                lanes = w_tile[k_index]
                for w in range(words_per_k_w):
                    dram[base + k_index_offset + k_rel * words_per_k_w + w] = pack4_int8(lanes[w*4:w*4+4])
            pad_k = ((k_len + SIMD - 1) // SIMD) * SIMD
            pad_words = pad_k * words_per_k_w
            for p in range(k_len * words_per_k_w, pad_words):
                dram[base + k_index_offset + p] = 0
            k_index_offset += pad_words
            kpos += k_len

    for idx, value in enumerate(layout["bias_window"]):
        dram[(layout["bias_base"] >> 2) + idx] = value & 0xFFFFFFFF

    for idx, value in enumerate(layout["multiplier_window"]):
        dram[(layout["multiplier_base"] >> 2) + idx] = value & 0xFFFFFFFF

    for idx in range(len(tiles) * layout["results_per_tile"]):
        dram[(layout["r_base"] >> 2) + idx] = 0
        dram[(layout["q_base"] >> 2) + idx] = 0
    dram[layout["marker_addr"] >> 2] = 0
    write_hex(out_dir / "dram_init.hex", dram)
    return dram

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

    def write_reg_imm(offset, value):
        emit(*li_insns("t1", int(value)))
        emit(SW("t1", "s0", offset))

    def write_reg_reg(offset, rs):
        emit(SW(rs, "s0", offset))

    def run_current_tile():
        write_reg_imm(REG_CTRL, 0)
        write_reg_imm(REG_M_DIM, args.tile_rows)
        write_reg_imm(REG_N_DIM, args.tile_cols)
        write_reg_imm(REG_K_DIM, layout["k_dim"])
        write_reg_reg(REG_W_ADDR, "s4")
        write_reg_reg(REG_A_ADDR, "s3")
        write_reg_reg(REG_R_ADDR, "s5")
        write_reg_imm(REG_ARR_CFG, ARR_TILE)
        write_reg_imm(REG_CFG_SHAPE, args.cfg_shape)
        write_reg_imm(REG_CTRL, CTRL_TILE_OS_INT8)

        poll_label = "poll_tile"
        label(poll_label)
        emit(LW("t1", "s0", REG_STATUS))
        emit(ANDI("t1", "t1", 2))
        beq_idx = len(insns)
        emit(0)
        patch_beqz(beq_idx, poll_label)
        write_reg_imm(REG_CTRL, 0)

    def emit_postprocess_tile():
        emit(MV("t2", "s5"))     # raw pointer for this tile
        emit(*li_insns("t6", 0))  # row index

        label("post_row_loop")
        emit(*li_insns("t5", 0))  # col index
        emit(MV("a4", "s10"))    # bias pointer for current N tile
        emit(MV("a5", "s11"))    # requant multiplier pointer for current N tile

        label("post_col_loop")
        emit(LW("t1", "t2", 0))
        emit(LW("t3", "a4", 0))
        emit(ADD("t1", "t1", "t3"))

        emit(SRLI("t4", "t1", 31))
        beq_idx = len(insns)
        emit(0)
        emit(*li_insns("t1", 0))
        label("post_relu_ok")
        patch_beqz(beq_idx, "post_relu_ok", "t4")

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
        label("post_clamp")
        emit(*li_insns("t1", 127))
        label("post_store")
        patch_beqz(clamp_idx, "post_clamp", "t4")
        patch_j(store_j_idx, "post_store")

        emit(SW("t1", "s6", 0))
        emit(ADDI("t2", "t2", 4))
        emit(ADDI("s6", "s6", 4))
        emit(ADDI("a4", "a4", 4))
        emit(ADDI("a5", "a5", 4))
        emit(ADDI("t5", "t5", 1))
        emit(*li_insns("t3", args.tile_cols))
        bne_col_idx = len(insns)
        emit(0)
        patch_bne(bne_col_idx, "post_col_loop", "t5", "t3")

        emit(ADDI("t6", "t6", 1))
        emit(*li_insns("t3", args.tile_rows))
        bne_row_idx = len(insns)
        emit(0)
        patch_bne(bne_row_idx, "post_row_loop", "t6", "t3")

    label("_start")
    emit(*li_insns("sp", 0x00002000))
    emit(*li_insns("s0", NPU_BASE))
    emit(*li_insns("s7", args.n_tiles))
    emit(*li_insns("s8", args.m_tiles))
    emit(*li_insns("s9", layout["k_dim"]))
    emit(*li_insns("s1", 0))
    emit(*li_insns("s3", layout["a_base"]))
    emit(*li_insns("s5", layout["r_base"]))
    emit(*li_insns("s6", layout["q_base"]))

    label("m_loop")
    emit(*li_insns("s2", 0))
    emit(*li_insns("s4", layout["w_base"]))
    emit(*li_insns("s10", layout["bias_base"]))
    emit(*li_insns("s11", layout["multiplier_base"]))

    label("n_loop")
    run_current_tile()
    emit_postprocess_tile()

    emit(ADDI("s2", "s2", 1))
    emit(ADDI("s4", "s4", layout["k_dim"] * layout["words_per_k_w"] * 4))
    emit(ADDI("s5", "s5", args.tile_rows * args.tile_cols * 4))
    emit(ADDI("s10", "s10", args.tile_cols * 4))
    emit(ADDI("s11", "s11", args.tile_cols * 4))
    n_bne_idx = len(insns)
    emit(0)
    patch_bne(n_bne_idx, "n_loop", "s2", "s7")

    emit(ADDI("s1", "s1", 1))
    emit(ADDI("s3", "s3", layout["k_dim"] * layout["words_per_k_a"] * 4))
    m_bne_idx = len(insns)
    emit(0)
    patch_bne(m_bne_idx, "m_loop", "s1", "s8")

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


def write_expected_files(out_dir, tiles):
    raw_addr = []
    expected_raw = []
    expected_q = []
    for tile in tiles:
        for idx, value in enumerate(tile["expected"]):
            raw_addr.append(tile["r_addr"] + idx * 4)
            expected_raw.append(value)
            expected_q.append(tile["post"][idx]["q"] & 0xFFFFFFFF)
    write_hex(out_dir / "expected_raw_addr.hex", raw_addr)
    write_hex(out_dir / "expected_raw.hex", expected_raw)
    write_hex(out_dir / "expected_q.hex", expected_q)
    return raw_addr, expected_raw, expected_q


def write_params(out_dir, fw_words, args, layer, label, in_scale, in_zp, tiles, layout, expected_q):
    timeout_cycles = max(1200000, len(tiles) * 10000)
    op = out_dir.as_posix()
    lines = [
        f"`define REP_TILE_SOC_FW_HEX \"{op}/soc_repopt_tile_window.hex\"",
        f"`define REP_TILE_SOC_DRAM_HEX \"{op}/dram_init.hex\"",
        f"`define REP_TILE_SOC_RAW_ADDR_HEX \"{op}/expected_raw_addr.hex\"",
        f"`define REP_TILE_SOC_EXPECTED_RAW_HEX \"{op}/expected_raw.hex\"",
        f"`define REP_TILE_SOC_EXPECTED_Q_HEX \"{op}/expected_q.hex\"",
        f"`define REP_TILE_SOC_FW_WORDS {fw_words}",
        f"`define REP_TILE_SOC_DRAM_WORDS {layout['dram_words']}",
        f"`define REP_TILE_SOC_TIMEOUT_CYCLES {timeout_cycles}",
        f"`define REP_TILE_SOC_MARKER_ADDR 32'h{layout['marker_addr']:08x}",
        f"`define REP_TILE_SOC_A_BASE 32'h{layout['a_base']:08x}",
        f"`define REP_TILE_SOC_W_BASE 32'h{layout['w_base']:08x}",
        f"`define REP_TILE_SOC_R_BASE 32'h{layout['r_base']:08x}",
        f"`define REP_TILE_SOC_Q_BASE 32'h{layout['q_base']:08x}",
        f"`define REP_TILE_SOC_RAW_COUNT {len(tiles) * layout['results_per_tile']}",
        f"`define REP_TILE_SOC_Q_COUNT {len(expected_q)}",
        f"`define REP_TILE_SOC_REQUANT_SHIFT {REQUANT_SHIFT}",
        f"`define REP_TILE_SOC_TILE_COUNT {len(tiles)}",
        f"`define REP_TILE_SOC_M_BASE {args.m_base}",
        f"`define REP_TILE_SOC_N_BASE {args.n_base}",
        f"`define REP_TILE_SOC_M_TILES {args.m_tiles}",
        f"`define REP_TILE_SOC_N_TILES {args.n_tiles}",
        f"`define REP_TILE_SOC_CIFAR_LABEL {label}",
        f"`define REP_TILE_SOC_R_ADDR_0 32'h{layout['r_base']:08x}",
    ]
    with open(out_dir / "soc_repopt_tile_window_params.vh", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
        f.write("\n")

    meta = {
        "schema": "repopt_tile_soc_case_v1",
        "layer": args.layer_name,
        "sample_index": args.index,
        "cifar_label": label,
        "input_quant_scale": in_scale,
        "input_quant_zero_point": in_zp,
        "output_shape": layer["output_shape"],
        "window": {
            "m_base": args.m_base,
            "n_base": args.n_base,
            "m_tiles": args.m_tiles,
            "n_tiles": args.n_tiles,
        },
        "tiles": [
            {
                "m_base": tile["m_base"],
                "n_base": tile["n_base"],
                "w_addr": f"0x{tile['w_addr']:08x}",
                "a_addr": f"0x{tile['a_addr']:08x}",
                "r_addr": f"0x{tile['r_addr']:08x}",
                "expected_signed": [signed32(word) for word in tile["expected"]],
                "q_expected": [post["q"] for post in tile["post"]],
            }
            for tile in tiles
        ],
        "layout": {
            "a_base": f"0x{layout['a_base']:08x}",
            "w_base": f"0x{layout['w_base']:08x}",
            "bias_base": f"0x{layout['bias_base']:08x}",
            "multiplier_base": f"0x{layout['multiplier_base']:08x}",
            "r_base": f"0x{layout['r_base']:08x}",
            "q_base": f"0x{layout['q_base']:08x}",
            "marker_addr": f"0x{layout['marker_addr']:08x}",
            "dram_words": layout["dram_words"],
        },
        "requant_shift": REQUANT_SHIFT,
        "note": "Reference CPU firmware schedules NPU tiles through MMIO and runs CPU bias/ReLU/fixed-point requant.",
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate a RepOpt tile-window SoC MMIO scheduling case")
    parser.add_argument(
        "--pth",
        default=".06_RepOpt_VGG/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth",
        help="Path to qat_int8_quantized.pth",
    )
    parser.add_argument("--plan", default="sim/pth_repopt_probe/model_plan.json", help="Path to model_plan.json")
    parser.add_argument("--data-root", default=".06_RepOpt_VGG/06_RepOpt_VGG/data", help="CIFAR-10 data root")
    parser.add_argument("--layer-name", default="stage1_0_conv", help="First Conv layer name")
    parser.add_argument("--index", type=int, default=0, help="CIFAR-10 test sample index")
    parser.add_argument("--m-base", type=int, default=0, help="Global GEMM M base")
    parser.add_argument("--n-base", type=int, default=0, help="Global GEMM N base")
    parser.add_argument("--m-tiles", type=int, default=2, help="Number of tile rows")
    parser.add_argument("--n-tiles", type=int, default=2, help="Number of tile cols")
    parser.add_argument("--tile-rows", type=int, default=4, help="Tile rows (4/8/16)")
    parser.add_argument("--tile-cols", type=int, default=4, help="Tile cols (4/8/16/32)")
    parser.add_argument("--cfg-shape", type=int, default=0, help="CFG_SHAPE (0=4x4,1=8x8,2=16x16,3=8x32)")
    parser.add_argument("--full-layer", action="store_true", help="Run the complete first layer window")
    parser.add_argument("--out-dir", default="sim/repopt_tile_soc", help="Output case directory")
    args = parser.parse_args()

    torch = require_torch()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_path = Path(args.plan).resolve()
    plan = load_json(plan_path)
    if args.full_layer:
        layer = next(
            (item for item in plan["layers"] if item.get("name") == args.layer_name and item.get("op") == "conv2d"),
            None,
        )
        if layer is None:
            raise ValueError(f"conv layer not found in plan: {args.layer_name}")
        args.m_base = 0
        args.n_base = 0
        args.m_tiles = int(layer["registers"]["M_DIM"]) // args.tile_rows
        args.n_tiles = int(layer["registers"]["N_DIM"]) // args.tile_cols
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
        checkpoint = torch.load(args.pth, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)
    layer, label, in_scale, in_zp, tiles, layout = build_case_tiles(plan, plan_path.parent, state_dict, args)
    write_dram_init(out_dir, tiles, layout)
    raw_addr, expected_raw, expected_q = write_expected_files(out_dir, tiles)
    fw_words = assemble_firmware(layout, args)
    write_hex(out_dir / "soc_repopt_tile_window.hex", fw_words)
    write_params(out_dir, len(fw_words), args, layer, label, in_scale, in_zp, tiles, layout, expected_q)

    print(f"generated RepOpt tile SoC case: {out_dir}")
    print(f"firmware words: {len(fw_words)}")
    print(f"tiles: {len(tiles)} window M[{args.m_base}:{args.m_base + args.m_tiles * args.tile_rows}) N[{args.n_base}:{args.n_base + args.n_tiles * args.tile_cols})")
    print(f"q_base: 0x{layout['q_base']:08x}, q_count: {len(expected_q)}")
    print(f"marker: 0x{layout['marker_addr']:08x}, dram_words: {layout['dram_words']}")


if __name__ == "__main__":
    main()
