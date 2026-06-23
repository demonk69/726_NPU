#!/usr/bin/env python3
"""Generate a runtime closed-loop RepOpt VGG SoC case.

Python emits only static assets: input image, packed weights, bias, per-channel
requant multipliers, classifier params and metadata. PicoRV32 firmware packs A
tiles from dense runtime activations, schedules the NPU, postprocesses raw MAC
tiles, performs maxpool/avgpool/classifier, and writes the prediction marker.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
TB_DIR = PROJECT_ROOT / "tb"
sys.path.insert(0, str(TB_DIR))
sys.path.insert(0, str(THIS_DIR))

from assemble_soc_test import (  # noqa: E402
    ADDI, ANDI, BEQ, BNE, J, JAL, LW, SW, b_type, i_type, li_insns,
    r_type, reg, s_type,
)
from run_repopt_vgg_host import (  # noqa: E402
    adaptive_avgpool2d_cpu, conv2d_acc_npu, load_cifar_sample,
    load_image_input, load_int32_hex, load_json, maxpool2d_cpu,
    quantize_qint8, requant_qint8, tensor_scalar, unwrap_state_dict,
)


PPB_DEPTH = 8192
SIMD = 4
WORD_INT8_LANES = 4
SHAPE_CONFIGS = {
    "4x4": {"cfg_shape": 0x0, "tile_rows": 4, "tile_cols": 4},
    "8x8": {"cfg_shape": 0x1, "tile_rows": 8, "tile_cols": 8},
    "16x16": {"cfg_shape": 0x2, "tile_rows": 16, "tile_cols": 16},
    "8x32": {"cfg_shape": 0x3, "tile_rows": 8, "tile_cols": 32},
}
DEFAULT_SHAPE = "16x16"


def shape_kt_elems(tile_rows, tile_cols):
    # 8x32 consumes the 32 logical columns as two 16-column physical passes.
    w_lanes = 16 if (tile_rows, tile_cols) == (8, 32) else tile_cols
    return (PPB_DEPTH * 4) // max(tile_rows, w_lanes)


TR = SHAPE_CONFIGS[DEFAULT_SHAPE]["tile_rows"]
TC = SHAPE_CONFIGS[DEFAULT_SHAPE]["tile_cols"]
CFG_SHAPE = SHAPE_CONFIGS[DEFAULT_SHAPE]["cfg_shape"]
KT_ELEMS = shape_kt_elems(TR, TC)
A_PACK_LANES = 16 if (TR, TC) == (8, 32) else TR
DRAM_WORDS_MAX = 2 * 1024 * 1024
MEM_WORDS = 8192

NPU_BASE = 0x02000000
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

CTRL_BIAS_TILE_OS = 0x211  # start | INT8 | OS | bias, raw INT32 result
CTRL_BIAS_TILE_WS = 0x201  # start | INT8 | WS | bias, raw INT32 result
CTRL_BIAS_TILE = CTRL_BIAS_TILE_OS
ARR_TILE = 0x080
QUANT_DISABLED = 0x00010000

ACT_A = 0x00010000
ACT_B = 0x00030000
A_WORK_SHARED = 0x00050000
R_WORK_BASE = [0x00070000, 0x00074000, 0x00078000, 0x0007C000]
R_WORK_STRIDE = 0x00004000
FEAT_BASE = 0x00080000
SCORE_BASE = 0x00081000
MARKER_ADDR = 0x00082000
DESC_BASE = 0x00083000
STATIC_BASE = 0x00090000

REQUANT_SHIFT = 24
REQUANT_ROUND = 1 << (REQUANT_SHIFT - 1)
PASS_BASE = 0x100
FAIL_MARKER = 0xFF

NUM_CORES = 1
NPU_CORE_STRIDE = 0x100


def align(value, alignment):
    return (value + alignment - 1) & ~(alignment - 1)


def signed32(value):
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def pack4(values):
    word = 0
    for lane, value in enumerate(values):
        word |= (int(value) & 0xFF) << (lane * 8)
    return word & 0xFFFFFFFF


def write_hex(path, words):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for word in words:
            f.write(f"{int(word) & 0xFFFFFFFF:08x}\n")


def write_sparse_hex(path, words, limit_words):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        idx = 0
        while idx < limit_words:
            if (int(words[idx]) & 0xFFFFFFFF) == 0:
                idx += 1
                continue
            f.write(f"@{idx:x}\n")
            while idx < limit_words and (int(words[idx]) & 0xFFFFFFFF) != 0:
                f.write(f"{int(words[idx]) & 0xFFFFFFFF:08x}\n")
                idx += 1


def store_word(dram, addr, value):
    dram[addr >> 2] = int(value) & 0xFFFFFFFF


def store_byte(dram, addr, value):
    idx = addr >> 2
    shift = (addr & 3) * 8
    mask = 0xFF << shift
    dram[idx] = (dram[idx] & ~mask) | ((int(value) & 0xFF) << shift)


def pack_weight_tile_8x32(qweight, n_base, n_count, k_dim):
    w_int = qweight.int_repr().cpu()
    cout, _cin, kh, kw = [int(x) for x in w_int.shape]
    out = []
    half_cols = 16
    words_per_k = half_cols // WORD_INT8_LANES
    for pass_base in (0, half_cols):
        kpos = 0
        while kpos < k_dim:
            k_len = min(k_dim - kpos, KT_ELEMS)
            for k_rel in range(k_len):
                k_index = kpos + k_rel
                chan = k_index // (kh * kw)
                rem = k_index % (kh * kw)
                ker_h = rem // kw
                ker_w = rem % kw
                for group in range(words_per_k):
                    vals = []
                    for lane in range(WORD_INT8_LANES):
                        local_col = pass_base + group * WORD_INT8_LANES + lane
                        out_c = n_base + local_col
                        if local_col < n_count and out_c < cout:
                            vals.append(int(w_int[out_c, chan, ker_h, ker_w].item()))
                        else:
                            vals.append(0)
                    out.append(pack4(vals))
            padded_k = ((k_len + SIMD - 1) // SIMD) * SIMD
            for _ in range((padded_k - k_len) * words_per_k):
                out.append(0)
            kpos += k_len
    return out


def pack_weight_tile(qweight, n_base, n_count, k_dim):
    if (TR, TC) == (8, 32):
        return pack_weight_tile_8x32(qweight, n_base, n_count, k_dim)

    w_int = qweight.int_repr().cpu()
    _cout, _cin, kh, kw = [int(x) for x in w_int.shape]
    out = []
    kpos = 0
    while kpos < k_dim:
        k_len = min(k_dim - kpos, KT_ELEMS)
        for k_rel in range(k_len):
            k_index = kpos + k_rel
            chan = k_index // (kh * kw)
            rem = k_index % (kh * kw)
            ker_h = rem // kw
            ker_w = rem % kw
            for group in range((n_count + WORD_INT8_LANES - 1) // WORD_INT8_LANES):
                vals = []
                for lane in range(WORD_INT8_LANES):
                    out_c = n_base + group * WORD_INT8_LANES + lane
                    if group * WORD_INT8_LANES + lane < n_count:
                        vals.append(int(w_int[out_c, chan, ker_h, ker_w].item()))
                    else:
                        vals.append(0)
                out.append(pack4(vals))
        padded_k = ((k_len + SIMD - 1) // SIMD) * SIMD
        words_per_k = (n_count + WORD_INT8_LANES - 1) // WORD_INT8_LANES
        for _ in range((padded_k - k_len) * words_per_k):
            out.append(0)
        kpos += k_len
    return out


def fixed_requant(acc, multipliers):
    qmult = torch.tensor(
        [int(round(float(x) * (1 << REQUANT_SHIFT))) for x in multipliers],
        dtype=torch.int64,
        device=acc.device,
    ).view(1, -1, 1, 1)
    prod = torch.clamp(acc.to(torch.int64), min=0) * qmult
    q = (prod + REQUANT_ROUND) >> REQUANT_SHIFT
    return torch.clamp(q, 0, 127).to(torch.float64)


def run_fixed_host(plan, plan_dir, state_dict, x_q):
    current = x_q
    for layer in plan["layers"]:
        if layer["op"] == "conv2d":
            qweight = state_dict[layer["weight_key"]]
            bias = load_int32_hex(plan_dir / layer["assets"]["bias_int32_hex"])
            acc = conv2d_acc_npu(current, qweight, bias, layer)
            current = fixed_requant(acc, layer["cpu_requant_after_npu"]["multipliers"])
        elif layer["op"] == "maxpool2d":
            current = maxpool2d_cpu(current, layer)
        elif layer["op"] == "adaptive_avgpool2d":
            current = adaptive_avgpool2d_cpu(current, layer)
        elif layer["op"] == "flatten":
            current = current.reshape(current.shape[0], -1)
        elif layer["op"] == "linear":
            break
    features = torch.clamp(current.reshape(-1).to(torch.int64), 0, 127)
    packed = state_dict["model.classifier._packed_params._packed_params"]
    cls_w = packed[0].int_repr().to(torch.int64)
    cls_b = packed[1]
    scores = []
    for cls in range(10):
        score = int(cls_b[cls].item())
        for feat in range(512):
            score += int(features[feat].item()) * int(cls_w[cls, feat].item())
        scores.append(score)
    return scores.index(max(scores)), scores, features


def rv_blt(rs1, rs2, offset):
    return b_type(offset, reg(rs2), reg(rs1), 0x4, 0x63)


def rv_bgeu(rs1, rs2, offset):
    return b_type(offset, reg(rs2), reg(rs1), 0x7, 0x63)


def ADD(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)


def SUB(rd, rs1, rs2):
    return r_type(0x20, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)


def OR(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x6, reg(rd), 0x33)


def AND(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x7, reg(rd), 0x33)


def SLL(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x1, reg(rd), 0x33)


def SRL(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x5, reg(rd), 0x33)


def MUL(rd, rs1, rs2):
    return r_type(0x01, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)


def MULHU(rd, rs1, rs2):
    return r_type(0x01, reg(rs2), reg(rs1), 0x3, reg(rd), 0x33)


def SLT(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x2, reg(rd), 0x33)


def SLTU(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x3, reg(rd), 0x33)


def SLLI(rd, rs1, shamt):
    return i_type(shamt & 0x1F, reg(rs1), 0x1, reg(rd), 0x13)


def SRLI(rd, rs1, shamt):
    return i_type(shamt & 0x1F, reg(rs1), 0x5, reg(rd), 0x13)


def SLTI(rd, rs1, imm):
    return i_type(imm, reg(rs1), 0x2, reg(rd), 0x13)


def LBU(rd, rs1, imm):
    return i_type(imm, reg(rs1), 0x4, reg(rd), 0x03)


def SB(rs2, rs1, imm):
    return s_type(imm, reg(rs2), reg(rs1), 0x0, 0x23)


def JALR(rd, rs1, imm=0):
    return i_type(imm, reg(rs1), 0x0, reg(rd), 0x67)


class Asm:
    def __init__(self):
        self.insns = []
        self.labels = {}
        self.patches = []
        self.uid = 0

    def unique(self, prefix):
        self.uid += 1
        return f"{prefix}_{self.uid}"

    def emit(self, *words):
        self.insns.extend(int(w) & 0xFFFFFFFF for w in words)

    def li(self, rd, imm):
        self.emit(*li_insns(rd, int(imm)))

    def label(self, name):
        self.labels[name] = len(self.insns)

    def branch(self, kind, rs1, rs2, target):
        idx = len(self.insns)
        self.emit(0)
        self.patches.append((kind, idx, rs1, rs2, target))

    def jump(self, target):
        idx = len(self.insns)
        self.emit(0)
        self.patches.append(("j", idx, None, None, target))

    def call(self, target):
        idx = len(self.insns)
        self.emit(0)
        self.patches.append(("jal", idx, "ra", None, target))

    def ret(self):
        self.emit(JALR("zero", "ra", 0))

    def finalize(self):
        for kind, idx, rs1, rs2, target in self.patches:
            if target not in self.labels:
                raise KeyError(f"unknown label: {target}")
            offset = (self.labels[target] - idx) * 4
            if kind == "beq":
                self.insns[idx] = BEQ(rs1, rs2, offset)
            elif kind == "bne":
                self.insns[idx] = BNE(rs1, rs2, offset)
            elif kind == "blt":
                self.insns[idx] = rv_blt(rs1, rs2, offset)
            elif kind == "bgeu":
                self.insns[idx] = rv_bgeu(rs1, rs2, offset)
            elif kind == "j":
                self.insns[idx] = J(offset)
            elif kind == "jal":
                self.insns[idx] = JAL(rs1, offset)
            else:
                raise ValueError(kind)
        return self.insns


def emit_write_reg_imm(a, offset, value):
    a.li("t1", value)
    a.emit(SW("t1", "s0", offset))


def emit_write_reg_reg(a, offset, rs):
    a.emit(SW(rs, "s0", offset))


def emit_load_u32_unaligned(a):
    done = a.unique("load4_done")
    a.emit(ANDI("t5", "t2", 3))
    a.emit(ANDI("t4", "t2", -4))
    a.emit(LW("t0", "t4", 0))
    a.branch("beq", "t5", "zero", done)
    a.emit(LW("t1", "t4", 4))
    a.emit(SLLI("t5", "t5", 3))
    a.emit(SRL("t0", "t0", "t5"))
    a.li("t4", 32)
    a.emit(SUB("t4", "t4", "t5"))
    a.emit(SLL("t1", "t1", "t4"))
    a.emit(OR("t0", "t0", "t1"))
    a.label(done)


def emit_pack_group_padded(a):
    a.emit(SLLI("t1", "a5", 2))       # group * 4 INT8 bytes per 32-bit word
    a.emit(ADD("t1", "t1", "s10"))    # m = m_base + group*4
    a.emit(SRL("t2", "t1", "a4"))     # oh = m >> ow_shift
    a.emit(ADD("t4", "t1", "zero"))
    a.emit(AND("t4", "t4", "t3"))     # ow = m & ow_mask
    a.emit(ADD("t2", "t2", "a7"))     # padded row = oh + kh
    a.emit(MUL("t2", "t2", "tp"))     # row byte offset
    a.emit(ADD("t4", "t4", "t6"))     # padded col = ow + kw
    a.emit(ADD("t2", "t2", "t4"))
    a.emit(ADD("t2", "t2", "a2"))     # c_base + row*padded_w + col
    emit_load_u32_unaligned(a)


def emit_requant_nonnegative(a):
    relu_ok = a.unique("relu_ok")
    clamp = a.unique("clamp")
    store = a.unique("store")
    a.emit(SRLI("t4", "t1", 31))
    a.branch("beq", "t4", "zero", relu_ok)
    a.li("t1", 0)
    a.label(relu_ok)
    a.emit(MUL("t5", "t1", "a7"))
    a.emit(MULHU("t6", "t1", "a7"))
    a.li("t4", REQUANT_ROUND)
    a.emit(ADD("t5", "t5", "t4"))
    a.emit(SLTU("t4", "t5", "t4"))
    a.emit(ADD("t6", "t6", "t4"))
    a.emit(SRLI("t5", "t5", REQUANT_SHIFT))
    a.emit(SLLI("t6", "t6", 32 - REQUANT_SHIFT))
    a.emit(OR("t1", "t5", "t6"))
    a.emit(SLTI("t4", "t1", 128))
    a.branch("beq", "t4", "zero", clamp)
    a.jump(store)
    a.label(clamp)
    a.li("t1", 127)
    a.label(store)


def emit_run_conv_layer(a):
    a.label("run_conv_layer")
    a.emit(LW("s2", "s1", 0))     # ifm base
    a.emit(LW("s3", "s1", 4))     # ofm base
    a.emit(LW("s4", "s1", 8))     # w base
    a.emit(LW("s5", "s1", 12))    # bias base
    a.emit(LW("s6", "s1", 16))    # per-channel requant multiplier base
    a.emit(LW("s7", "s1", 20))    # M
    a.emit(LW("s8", "s1", 24))    # N
    a.emit(LW("s9", "s1", 28))    # K

    a.emit(LW("t0", "s1", 72))    # output padded spatial bytes/channel
    a.emit(MUL("t0", "t0", "s8"))
    a.emit(ADDI("t0", "t0", 3))
    a.emit(SRLI("t0", "t0", 2))
    a.emit(ADD("t1", "s3", "zero"))
    a.label("conv_clear_ofm")
    a.emit(SW("zero", "t1", 0))
    a.emit(ADDI("t1", "t1", 4))
    a.emit(ADDI("t0", "t0", -1))
    a.branch("bne", "t0", "zero", "conv_clear_ofm")

    a.li("s10", 0)                 # m_base

    a.label("conv_m_loop")
    a.emit(LW("s2", "s1", 0))      # ifm base
    a.li("gp", A_WORK_SHARED)       # A pack dst
    a.emit(LW("s11", "s1", 60))    # input padded spatial bytes/channel
    a.emit(LW("tp", "s1", 68))     # input padded width
    a.emit(LW("a4", "s1", 52))     # ow shift
    a.emit(LW("t3", "s1", 56))     # ow mask
    a.emit(LW("a6", "s1", 32))     # Cin remaining
    a.emit(ADD("a2", "s2", "zero"))

    a.label("pack_c_loop")
    a.li("a7", 0)                  # kh
    a.label("pack_kh_loop")
    a.li("t6", 0)                  # kw
    a.label("pack_kw_loop")
    a.li("a5", 0)                  # lane group
    a.label("pack_group_loop")
    emit_pack_group_padded(a)
    a.emit(SW("t0", "gp", 0))
    a.emit(ADDI("gp", "gp", 4))
    a.emit(ADDI("a5", "a5", 1))
    a.li("t1", TR // WORD_INT8_LANES)
    a.branch("bne", "a5", "t1", "pack_group_loop")
    for _ in range((A_PACK_LANES - TR) // WORD_INT8_LANES):
        a.emit(SW("zero", "gp", 0))
        a.emit(ADDI("gp", "gp", 4))
    a.emit(ADDI("t6", "t6", 1))
    a.li("t1", 3)
    a.branch("bne", "t6", "t1", "pack_kw_loop")
    a.emit(ADDI("a7", "a7", 1))
    a.li("t1", 3)
    a.branch("bne", "a7", "t1", "pack_kh_loop")
    a.emit(ADD("a2", "a2", "s11"))
    a.emit(ADDI("a6", "a6", -1))
    a.branch("bne", "a6", "zero", "pack_c_loop")

    a.li("s11", 0)                 # n_base
    a.emit(LW("s4", "s1", 8))      # reset W ptr
    a.label("conv_n_loop")
    emit_write_reg_imm(a, REG_CTRL, 0)
    emit_write_reg_imm(a, REG_M_DIM, TR)
    emit_write_reg_imm(a, REG_N_DIM, TC)
    emit_write_reg_reg(a, REG_K_DIM, "s9")
    emit_write_reg_reg(a, REG_W_ADDR, "s4")
    a.li("t1", A_WORK_SHARED)
    a.emit(SW("t1", "s0", REG_A_ADDR))
    a.li("t1", R_WORK_BASE[0])
    a.emit(SW("t1", "s0", REG_R_ADDR))
    a.emit(SLLI("t1", "s11", 2))
    a.emit(ADD("t1", "s5", "t1"))
    emit_write_reg_reg(a, REG_BIAS_ADDR, "t1")
    emit_write_reg_imm(a, REG_QUANT_CFG, QUANT_DISABLED)
    emit_write_reg_imm(a, REG_ARR_CFG, ARR_TILE)
    emit_write_reg_imm(a, REG_CFG_SHAPE, CFG_SHAPE)
    emit_write_reg_imm(a, REG_CTRL, CTRL_BIAS_TILE)

    a.label("poll_npu")
    a.emit(LW("t1", "s0", REG_STATUS))
    a.emit(ANDI("t1", "t1", 2))
    a.branch("beq", "t1", "zero", "poll_npu")
    emit_write_reg_imm(a, REG_CTRL, 0)

    a.emit(LW("s2", "s1", 52))      # ow shift (s2 no longer needed as ifm base)
    a.emit(LW("t3", "s1", 56))      # ow mask
    a.emit(LW("gp", "s1", 72))      # output padded spatial bytes/channel
    a.emit(LW("tp", "s1", 76))      # output padded width
    a.li("a2", 0)                   # col
    post_row_full = a.unique("post_row_full")
    post_no_wrap = a.unique("post_no_wrap")
    a.label("post_col_loop")
    a.li("a3", R_WORK_BASE[0])
    a.emit(SLLI("t1", "a2", 2))
    a.emit(ADD("a3", "a3", "t1"))
    a.emit(ADD("t1", "s11", "a2"))
    a.emit(SLLI("t1", "t1", 2))
    a.emit(ADD("t1", "s6", "t1"))
    a.emit(LW("a7", "t1", 0))      # Q24 multiplier for this output channel
    a.emit(LW("t3", "s1", 56))      # ow mask
    a.emit(ADD("a4", "s11", "a2"))
    a.emit(MUL("a4", "a4", "gp"))
    a.emit(ADD("a4", "a4", "s3"))
    a.emit(SRL("t0", "s10", "s2"))
    a.emit(MUL("t0", "t0", "tp"))
    a.emit(ADD("t2", "s10", "zero"))
    a.emit(AND("t2", "t2", "t3"))
    a.emit(ADD("t0", "t0", "t2"))
    a.emit(ADD("t0", "t0", "tp"))
    a.emit(ADDI("t0", "t0", 1))
    a.emit(ADD("a4", "a4", "t0"))
    a.li("t3", TR)
    a.emit(LW("t2", "s1", 48))      # OW
    a.emit(SLTI("t0", "t2", TR))
    a.branch("beq", "t0", "zero", post_row_full)
    a.emit(ADD("t3", "t2", "zero"))
    a.label(post_row_full)
    a.emit(ADD("t0", "t3", "zero")) # remaining values before padded row gap
    a.li("a5", TR)
    a.label("post_row_loop")
    a.emit(LW("t1", "a3", 0))
    emit_requant_nonnegative(a)
    a.emit(SB("t1", "a4", 0))
    a.emit(ADDI("a3", "a3", TC * 4))
    a.emit(ADDI("a4", "a4", 1))
    a.emit(ADDI("t0", "t0", -1))
    a.branch("bne", "t0", "zero", post_no_wrap)
    a.emit(ADDI("a4", "a4", 2))
    a.emit(ADD("t0", "t3", "zero"))
    a.label(post_no_wrap)
    a.emit(ADDI("a5", "a5", -1))
    a.branch("bne", "a5", "zero", "post_row_loop")
    a.emit(ADDI("a2", "a2", 1))
    a.li("t1", TC)
    a.branch("bne", "a2", "t1", "post_col_loop")

    a.emit(ADDI("s11", "s11", TC))
    a.emit(LW("t1", "s1", 64))      # W stride
    a.emit(ADD("s4", "s4", "t1"))
    a.branch("bne", "s11", "s8", "conv_n_loop")
    a.emit(ADDI("s10", "s10", TR))
    a.branch("bne", "s10", "s7", "conv_m_loop")
    a.ret()


def emit_run_conv_layer_mc(a):
    """Multi-core conv layer: shared A_WORK_SHARED, per-core R_WORK."""
    nc = NUM_CORES
    a.label("run_conv_layer_mc")
    a.emit(LW("s2", "s1", 0))     # ifm base
    a.emit(LW("s3", "s1", 4))     # ofm base
    a.emit(LW("s4", "s1", 8))     # w base
    a.emit(LW("s5", "s1", 12))    # bias base
    a.emit(LW("s6", "s1", 16))    # Q24 multiplier base
    a.emit(LW("s7", "s1", 20))    # M
    a.emit(LW("s8", "s1", 24))    # N
    a.emit(LW("s9", "s1", 28))    # K

    # clear OFM
    a.emit(LW("t0", "s1", 72))
    a.emit(MUL("t0", "t0", "s8"))
    a.emit(ADDI("t0", "t0", 3))
    a.emit(SRLI("t0", "t0", 2))
    a.emit(ADD("t1", "s3", "zero"))
    a.label("mc_clear_ofm")
    a.emit(SW("zero", "t1", 0))
    a.emit(ADDI("t1", "t1", 4))
    a.emit(ADDI("t0", "t0", -1))
    a.branch("bne", "t0", "zero", "mc_clear_ofm")

    a.li("s10", 0)  # m_base

    a.label("mc_m_loop")

    # --- Pack A_WORK_SHARED once per M tile ---
    a.emit(LW("s2", "s1", 0))
    a.li("gp", A_WORK_SHARED)
    a.emit(LW("s11", "s1", 60))
    a.emit(LW("tp", "s1", 68))
    a.emit(LW("a4", "s1", 52))
    a.emit(LW("t3", "s1", 56))
    a.emit(LW("a6", "s1", 32))
    a.emit(ADD("a2", "s2", "zero"))

    a.label("mc_pack_c_loop")
    a.li("a7", 0)
    a.label("mc_pack_kh_loop")
    a.li("t6", 0)
    a.label("mc_pack_kw_loop")
    a.li("a5", 0)
    a.label("mc_pack_group_loop")
    emit_pack_group_padded(a)
    a.emit(SW("t0", "gp", 0))
    a.emit(ADDI("gp", "gp", 4))
    a.emit(ADDI("a5", "a5", 1))
    a.li("t1", TR // WORD_INT8_LANES)
    a.branch("bne", "a5", "t1", "mc_pack_group_loop")
    for _ in range((A_PACK_LANES - TR) // WORD_INT8_LANES):
        a.emit(SW("zero", "gp", 0))
        a.emit(ADDI("gp", "gp", 4))
    a.emit(ADDI("t6", "t6", 1))
    a.li("t1", 3)
    a.branch("bne", "t6", "t1", "mc_pack_kw_loop")
    a.emit(ADDI("a7", "a7", 1))
    a.li("t1", 3)
    a.branch("bne", "a7", "t1", "mc_pack_kh_loop")
    a.emit(ADD("a2", "a2", "s11"))
    a.emit(ADDI("a6", "a6", -1))
    a.branch("bne", "a6", "zero", "mc_pack_c_loop")

    a.emit(LW("s4", "s1", 8))
    a.li("s11", 0)  # n_tile index

    a.label("mc_n_round")
    a.li("a6", 0)   # launched_mask
    a.li("a5", 0)   # core index for launch

    a.label("mc_launch_loop")
    a.li("t1", nc)
    a.emit(SLT("t0", "a5", "t1"))     # core < nc?
    a.emit(LW("t2", "s1", 80))        # t2 = n_tiles (pre-computed)
    a.emit(SLT("t2", "s11", "t2"))    # n_tile < n_tiles?
    a.emit(AND("t0", "t0", "t2"))
    a.branch("beq", "t0", "zero", "mc_launch_done")

    a.li("t0", NPU_CORE_STRIDE)
    a.emit(MUL("t0", "t0", "a5"))
    a.emit(ADD("t0", "s0", "t0"))     # core_base = NPU_BASE + core*256

    a.emit(SW("zero", "t0", REG_CTRL))
    a.li("t1", TR);         a.emit(SW("t1", "t0", REG_M_DIM))
    a.li("t1", TC);         a.emit(SW("t1", "t0", REG_N_DIM))
    a.emit(SW("s9", "t0", REG_K_DIM))
    a.li("t1", A_WORK_SHARED); a.emit(SW("t1", "t0", REG_A_ADDR))
    a.li("t1", QUANT_DISABLED); a.emit(SW("t1", "t0", REG_QUANT_CFG))
    a.li("t1", ARR_TILE);  a.emit(SW("t1", "t0", REG_ARR_CFG))
    a.li("t1", CFG_SHAPE); a.emit(SW("t1", "t0", REG_CFG_SHAPE))

    a.emit(LW("t1", "s1", 64)); a.emit(MUL("t1", "t1", "s11"))
    a.emit(ADD("t1", "s4", "t1"));   a.emit(SW("t1", "t0", REG_W_ADDR))

    a.li("t1", R_WORK_STRIDE); a.emit(MUL("t1", "t1", "a5"))
    a.li("t2", R_WORK_BASE[0]); a.emit(ADD("t1", "t1", "t2"))
    a.emit(SW("t1", "t0", REG_R_ADDR))

    a.li("t1", TC * 4); a.emit(MUL("t1", "t1", "s11"))
    a.emit(ADD("t1", "s5", "t1"));   a.emit(SW("t1", "t0", REG_BIAS_ADDR))

    a.li("t1", CTRL_BIAS_TILE); a.emit(SW("t1", "t0", REG_CTRL))

    a.li("t1", 1); a.emit(SLL("t1", "t1", "a5"))
    a.emit(OR("a6", "a6", "t1"))
    a.emit(ADDI("s11", "s11", 1))
    a.emit(ADDI("a5", "a5", 1))
    a.jump("mc_launch_loop")

    a.label("mc_launch_done")
    a.branch("beq", "a6", "zero", "mc_m_advance")

    # Poll launched cores
    a.li("a7", 0)  # done_mask
    a.label("mc_poll_loop")
    a.li("a5", 0)
    a.label("mc_poll_core")
    a.li("t1", 1); a.emit(SLL("t1", "t1", "a5"))
    a.emit(AND("t0", "t1", "a6"))
    a.branch("beq", "t0", "zero", "mc_poll_next")
    a.emit(AND("t0", "t1", "a7"))
    a.branch("bne", "t0", "zero", "mc_poll_next")
    a.li("t0", NPU_CORE_STRIDE); a.emit(MUL("t0", "t0", "a5"))
    a.emit(ADD("t0", "s0", "t0"))
    a.emit(LW("t2", "t0", REG_STATUS))
    a.emit(ANDI("t0", "t2", 4))
    err_seen = a.unique("mc_err")
    a.branch("bne", "t0", "zero", err_seen)
    a.emit(ANDI("t0", "t2", 2))
    a.branch("beq", "t0", "zero", "mc_poll_next")
    # Core done
    a.li("t0", NPU_CORE_STRIDE); a.emit(MUL("t0", "t0", "a5"))
    a.emit(ADD("t0", "s0", "t0"))
    a.emit(SW("zero", "t0", REG_CTRL))
    a.emit(OR("a7", "a7", "t1"))
    a.label("mc_poll_next")
    a.emit(ADDI("a5", "a5", 1))
    a.li("t1", nc)
    a.branch("bne", "a5", "t1", "mc_poll_core")
    a.emit(ADDI("t0", "a7", 0))
    a.emit(ADDI("t1", "a6", 0))
    a.branch("bne", "t0", "t1", "mc_poll_loop")

    # Postprocess
    a.emit(LW("s2", "s1", 52))   # ow_shift
    a.emit(LW("t3", "s1", 56))   # ow_mask
    a.emit(LW("gp", "s1", 72))   # out padded spatial
    a.emit(LW("tp", "s1", 76))   # out padded width

    a.li("a3", 0)  # n_tile tracker
    a.li("a5", 0)  # core index
    a.label("mc_post_core")
    a.li("t1", 1); a.emit(SLL("t1", "t1", "a5"))
    a.emit(AND("t0", "t1", "a6"))
    a.branch("beq", "t0", "zero", "mc_post_skip")
    a.li("t1", TC); a.emit(MUL("t1", "t1", "a3"))
    a.emit(ADD("a2", "t1", "zero"))  # a2 = global n_base

    a.li("t1", R_WORK_STRIDE); a.emit(MUL("t1", "t1", "a5"))
    a.li("t2", R_WORK_BASE[0]); a.emit(ADD("t1", "t1", "t2"))  # R_WORK_BASE[core]

    a.li("a4", 0)
    a.label("mc_post_row")
    a.li("a7", 0)  # col
    a.label("mc_post_col")
    a.li("t0", TC * 4); a.emit(MUL("t0", "t0", "a4"))
    a.emit(SLLI("t2", "a7", 2)); a.emit(ADD("t0", "t0", "t2"))
    a.emit(ADD("t0", "t1", "t0")); a.emit(LW("t4", "t0", 0))
    a.emit(ADD("t0", "a2", "a7")); a.emit(SLLI("t0", "t0", 2))
    a.emit(ADD("t0", "t6", "t0")); a.emit(LW("t5", "t0", 0))
    a.emit(ADD("t1", "t4", "zero")); emit_requant_nonnegative(a)
    a.emit(ADD("t0", "s10", "a4"))
    a.emit(SRL("t2", "t0", "s2")); a.emit(MUL("t2", "t2", "tp"))
    a.emit(AND("t4", "t0", "t3")); a.emit(ADD("t2", "t2", "t4"))
    a.emit(ADD("t2", "t2", "tp")); a.emit(ADDI("t2", "t2", 1))
    a.emit(ADD("t0", "a2", "a7")); a.emit(MUL("t0", "t0", "gp"))
    a.emit(ADD("t0", "t0", "s3")); a.emit(ADD("t0", "t0", "t2"))
    a.emit(SB("t1", "t0", 0))
    a.emit(ADDI("a7", "a7", 1)); a.li("t0", TC)
    a.branch("bne", "a7", "t0", "mc_post_col")
    a.emit(ADDI("a4", "a4", 1)); a.li("t0", TR)
    a.branch("bne", "a4", "t0", "mc_post_row")
    a.emit(ADDI("a3", "a3", 1))
    a.label("mc_post_skip")
    a.emit(ADDI("a5", "a5", 1)); a.li("t1", nc)
    a.branch("bne", "a5", "t1", "mc_post_core")
    a.jump("mc_n_round")

    a.label(err_seen)
    a.li("t0", MARKER_ADDR); a.li("t1", FAIL_MARKER)
    a.emit(SW("t1", "t0", 0))
    a.jump("mc_halt_inner")
    a.label("mc_halt_inner")
    a.jump("mc_halt_inner")

    a.label("mc_m_advance")
    a.emit(ADDI("s10", "s10", TR))
    a.branch("bne", "s10", "s7", "mc_m_loop")
    a.ret()


def emit_maxpool(a):
    a.label("maxpool2x2")
    a.emit(LW("s2", "s1", 0))      # input base
    a.emit(LW("s3", "s1", 4))      # output base
    a.emit(LW("s4", "s1", 8))      # C
    a.emit(LW("s5", "s1", 12))     # H
    a.emit(LW("s6", "s1", 16))     # W
    a.emit(LW("s7", "s1", 20))     # OH
    a.emit(LW("s8", "s1", 24))     # OW
    a.emit(LW("gp", "s1", 28))     # input padded width
    a.emit(LW("s9", "s1", 32))     # input padded spatial bytes/channel
    a.emit(LW("tp", "s1", 36))     # output padded width
    a.emit(LW("s10", "s1", 40))    # output padded spatial bytes/channel

    a.emit(MUL("t0", "s4", "s10"))
    a.emit(ADDI("t0", "t0", 3))
    a.emit(SRLI("t0", "t0", 2))
    a.emit(ADD("t1", "s3", "zero"))
    a.label("pool_clear_ofm")
    a.emit(SW("zero", "t1", 0))
    a.emit(ADDI("t1", "t1", 4))
    a.emit(ADDI("t0", "t0", -1))
    a.branch("bne", "t0", "zero", "pool_clear_ofm")

    a.li("s11", 0)                 # channel
    a.label("pool_c_loop")
    a.emit(MUL("t0", "s11", "s9"))
    a.emit(ADD("t0", "s2", "t0"))
    a.emit(MUL("t1", "s11", "s10"))
    a.emit(ADD("t1", "s3", "t1"))
    a.li("a0", 0)                  # oh
    a.label("pool_oh_loop")
    a.li("a1", 0)                  # ow
    a.label("pool_ow_loop")
    a.emit(SLLI("a2", "a0", 1))
    a.emit(ADDI("a2", "a2", 1))
    a.emit(MUL("a2", "a2", "gp"))
    a.emit(SLLI("a3", "a1", 1))
    a.emit(ADDI("a3", "a3", 1))
    a.emit(ADD("a2", "a2", "a3"))
    a.emit(ADD("a2", "a2", "t0"))
    a.emit(LBU("t2", "a2", 0))
    a.emit(LBU("t3", "a2", 1))
    a.emit(SLT("t4", "t2", "t3"))
    skip0 = a.unique("pool_skip")
    a.branch("beq", "t4", "zero", skip0)
    a.emit(ADD("t2", "t3", "zero"))
    a.label(skip0)
    a.emit(ADD("a4", "a2", "gp"))
    a.emit(LBU("t3", "a4", 0))
    a.emit(SLT("t4", "t2", "t3"))
    skip1 = a.unique("pool_skip")
    a.branch("beq", "t4", "zero", skip1)
    a.emit(ADD("t2", "t3", "zero"))
    a.label(skip1)
    a.emit(LBU("t3", "a4", 1))
    a.emit(SLT("t4", "t2", "t3"))
    skip2 = a.unique("pool_skip")
    a.branch("beq", "t4", "zero", skip2)
    a.emit(ADD("t2", "t3", "zero"))
    a.label(skip2)
    a.emit(ADDI("a5", "a0", 1))
    a.emit(MUL("a5", "a5", "tp"))
    a.emit(ADDI("t4", "a1", 1))
    a.emit(ADD("a5", "a5", "t4"))
    a.emit(ADD("a5", "a5", "t1"))
    a.emit(SB("t2", "a5", 0))
    a.emit(ADDI("a1", "a1", 1))
    a.branch("bne", "a1", "s8", "pool_ow_loop")
    a.emit(ADDI("a0", "a0", 1))
    a.branch("bne", "a0", "s7", "pool_oh_loop")
    a.emit(ADDI("s11", "s11", 1))
    a.branch("bne", "s11", "s4", "pool_c_loop")
    a.ret()


def emit_classifier(a, cls_w_base, cls_b_base, expected_label):
    a.label("avgpool_classifier")
    a.li("s2", ACT_A)       # final IFM base
    a.li("s3", FEAT_BASE)
    a.li("s4", 512)
    a.li("s5", 0)           # channel
    a.li("t6", 36)          # final padded 4x4 feature map uses 6x6 bytes/channel
    a.label("avg_c_loop")
    a.emit(MUL("t0", "s5", "t6"))
    a.emit(ADD("t0", "s2", "t0"))
    a.li("t1", 0)           # sum
    for off in (7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28):
        a.emit(LBU("t2", "t0", off))
        a.emit(ADD("t1", "t1", "t2"))
    a.emit(ADDI("t1", "t1", 8))
    a.emit(SRLI("t1", "t1", 4))
    a.emit(SLLI("t2", "s5", 2))
    a.emit(ADD("t2", "s3", "t2"))
    a.emit(SW("t1", "t2", 0))
    a.emit(ADDI("s5", "s5", 1))
    a.branch("bne", "s5", "s4", "avg_c_loop")

    a.li("s2", cls_w_base)
    a.li("s3", cls_b_base)
    a.li("s4", FEAT_BASE)
    a.li("s5", 0)           # class
    a.li("s6", 0)           # best class
    a.li("s7", -0x7FFFFFFF) # best score
    a.label("cls_loop")
    a.emit(SLLI("t0", "s5", 2))
    a.emit(ADD("t0", "s3", "t0"))
    a.emit(LW("s8", "t0", 0))
    a.li("s9", 0)           # feature index
    a.label("cls_feat_loop")
    a.emit(SLLI("t1", "s9", 2))
    a.emit(ADD("t2", "s4", "t1"))
    a.emit(LW("t3", "t2", 0))
    a.emit(MUL("t4", "s5", "s4"))  # overwritten below; s4 value is not count
    a.li("t4", 512)
    a.emit(MUL("t4", "s5", "t4"))
    a.emit(ADD("t4", "t4", "s9"))
    a.emit(SLLI("t4", "t4", 2))
    a.emit(ADD("t4", "s2", "t4"))
    a.emit(LW("t5", "t4", 0))
    a.emit(MUL("t6", "t3", "t5"))
    a.emit(ADD("s8", "s8", "t6"))
    a.emit(ADDI("s9", "s9", 1))
    a.li("t0", 512)
    a.branch("bne", "s9", "t0", "cls_feat_loop")
    a.emit(SLT("t0", "s7", "s8"))
    keep = a.unique("keep_best")
    a.branch("beq", "t0", "zero", keep)
    a.emit(ADD("s7", "s8", "zero"))
    a.emit(ADD("s6", "s5", "zero"))
    a.label(keep)
    a.emit(ADDI("s5", "s5", 1))
    a.li("t0", 10)
    a.branch("bne", "s5", "t0", "cls_loop")
    a.li("t0", SCORE_BASE)
    a.emit(SW("s7", "t0", 0))
    a.li("t0", MARKER_ADDR)
    a.li("t1", PASS_BASE)
    a.emit(ADD("t1", "t1", "s6"))
    a.emit(SW("t1", "t0", 0))
    a.ret()


def emit_top(a, conv_descs, pool_descs, cls_w_base, cls_b_base, expected_label):
    def mark(value):
        a.li("t0", MARKER_ADDR)
        a.li("t1", value)
        a.emit(SW("t1", "t0", 0))

    a.label("_start")
    a.li("sp", MEM_WORDS * 4)
    a.li("s0", NPU_BASE)
    mark(0x200)
    sequence = [
        ("conv", 0), ("conv", 1), ("pool", 0),
        ("conv", 2), ("conv", 3), ("pool", 1),
        ("conv", 4), ("conv", 5), ("conv", 6), ("pool", 2),
        ("conv", 7), ("conv", 8),
    ]
    for step, (kind, idx) in enumerate(sequence, start=1):
        if kind == "conv":
            a.li("s1", conv_descs[idx])
            if NUM_CORES > 1:
                a.call("run_conv_layer_mc")
            else:
                a.call("run_conv_layer")
        else:
            a.li("s1", pool_descs[idx])
            a.call("maxpool2x2")
        mark(0x200 + step)
    a.call("avgpool_classifier")
    a.label("end")
    a.jump("end")
    if NUM_CORES > 1:
        emit_run_conv_layer_mc(a)
    else:
        emit_run_conv_layer(a)
    emit_maxpool(a)
    emit_classifier(a, cls_w_base, cls_b_base, expected_label)


def emit_firmware(conv_descs, pool_descs, cls_w_base, cls_b_base, expected_label):
    a = Asm()
    emit_top(a, conv_descs, pool_descs, cls_w_base, cls_b_base, expected_label)
    return a.finalize()


def generate(args):
    global TR, TC, CFG_SHAPE, KT_ELEMS, A_PACK_LANES, CTRL_BIAS_TILE, SIMD, NUM_CORES

    NUM_CORES = args.num_cores

    shape = SHAPE_CONFIGS[args.shape]
    SIMD = args.lanes
    TR = shape["tile_rows"]
    TC = shape["tile_cols"]
    CFG_SHAPE = shape["cfg_shape"]
    KT_ELEMS = shape_kt_elems(TR, TC)
    A_PACK_LANES = 16 if (TR, TC) == (8, 32) else TR
    CTRL_BIAS_TILE = CTRL_BIAS_TILE_WS if args.flow == "ws" else CTRL_BIAS_TILE_OS

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan_path = Path(args.plan).resolve()
    plan = load_json(plan_path)
    spec = load_json(args.spec)
    spec_by_name = {item["name"]: item for item in spec["layers"]}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
        ckpt = torch.load(args.pth, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(ckpt)

    if args.image:
        x_f = load_image_input(args.image, args.image_size)
        true_label = None
    else:
        x_f, true_label = load_cifar_sample(args.data_root, args.img_idx)
    in_sc = float(tensor_scalar(state_dict[plan["input"]["scale_key"]]))
    in_zp = int(tensor_scalar(state_dict[plan["input"]["zero_point_key"]]))
    x_q = quantize_qint8(x_f, in_sc, in_zp)

    fixed_pred, fixed_scores, fixed_features = run_fixed_host(plan, plan_path.parent, state_dict, x_q)

    exact = x_q
    for layer in plan["layers"]:
        if layer["op"] == "conv2d":
            exact = requant_qint8(
                conv2d_acc_npu(
                    exact,
                    state_dict[layer["weight_key"]],
                    load_int32_hex(plan_path.parent / layer["assets"]["bias_int32_hex"]),
                    layer,
                ),
                layer["cpu_requant_after_npu"]["multipliers"],
                layer["cpu_requant_after_npu"]["output_zero_point"],
            )
        elif layer["op"] == "maxpool2d":
            exact = maxpool2d_cpu(exact, layer)
        elif layer["op"] == "adaptive_avgpool2d":
            exact = adaptive_avgpool2d_cpu(exact, layer)
        elif layer["op"] == "flatten":
            exact = exact.reshape(exact.shape[0], -1)
        elif layer["op"] == "linear":
            break
    packed = state_dict["model.classifier._packed_params._packed_params"]
    cls_w = packed[0].int_repr()
    cls_b = packed[1]
    exact_scores = [int(cls_b[c].item()) for c in range(10)]
    exact_feat = torch.clamp(exact.reshape(-1).to(torch.int64), -128, 127)
    for c in range(10):
        for f in range(512):
            exact_scores[c] += int(exact_feat[f].item()) * int(cls_w[c, f].item())
    exact_pred = exact_scores.index(max(exact_scores))

    dram = [0] * DRAM_WORDS_MAX
    x_int = x_q.reshape(3, 32, 32).to(torch.int64)
    input_padded_w = 34
    input_padded_spatial = input_padded_w * input_padded_w
    for c in range(3):
        for h in range(32):
            for w in range(32):
                addr = ACT_A + c * input_padded_spatial + (h + 1) * input_padded_w + (w + 1)
                store_byte(dram, addr, int(x_int[c, h, w].item()))

    next_static = STATIC_BASE
    conv_layers = []
    for layer in plan["layers"]:
        if layer["op"] != "conv2d":
            continue
        merged = dict(layer)
        spec_layer = spec_by_name.get(layer["name"], {})
        for key in ("stride", "padding", "dilation"):
            merged[key] = spec_layer.get(key, merged.get(key, [1, 1]))
        qweight = state_dict[layer["weight_key"]]
        n_dim = int(layer["registers"]["N_DIM"])
        m_dim = int(layer["registers"]["M_DIM"])
        k_dim = int(layer["registers"]["K_DIM"])
        if n_dim % TC != 0:
            raise RuntimeError(f"{layer['name']} N_DIM={n_dim} is not divisible by tile cols {TC}")
        if m_dim % TR != 0:
            raise RuntimeError(f"{layer['name']} M_DIM={m_dim} is not divisible by tile rows {TR}")
        n_tiles = n_dim // TC
        w_base = align(next_static, 0x100)
        first_tile_words = None
        for nt in range(n_tiles):
            n_base = nt * TC
            words = pack_weight_tile(qweight, n_base, TC, k_dim)
            if first_tile_words is None:
                first_tile_words = len(words)
            tile_addr = w_base + nt * align(len(words) * 4, 0x100)
            for i, word in enumerate(words):
                store_word(dram, tile_addr + i * 4, word)
        w_stride = align(first_tile_words * 4, 0x100)
        next_static = w_base + n_tiles * w_stride

        bias_base = align(next_static, 0x100)
        bias = load_int32_hex(plan_path.parent / layer["assets"]["bias_int32_hex"])
        for i, value in enumerate(bias):
            store_word(dram, bias_base + i * 4, int(value))
        next_static = bias_base + len(bias) * 4

        qcfg_base = align(next_static, 0x100)
        multipliers = layer["cpu_requant_after_npu"]["multipliers"]
        for i, multiplier in enumerate(multipliers):
            qmult = int(round(float(multiplier) * (1 << REQUANT_SHIFT)))
            store_word(dram, qcfg_base + i * 4, qmult)
        next_static = qcfg_base + n_dim * 4

        conv_layers.append({
            "layer": merged,
            "w_base": w_base,
            "w_stride": w_stride,
            "bias_base": bias_base,
            "qcfg_base": qcfg_base,
        })

    cls_w_base = align(next_static, 0x100)
    for c in range(10):
        for f in range(512):
            store_word(dram, cls_w_base + (c * 512 + f) * 4, int(cls_w[c, f].item()))
    next_static = cls_w_base + 10 * 512 * 4
    cls_b_base = align(next_static, 0x100)
    for c in range(10):
        store_word(dram, cls_b_base + c * 4, int(cls_b[c].item()))
    next_static = cls_b_base + 10 * 4

    conv_ifm_ofm = [
        (ACT_A, ACT_B),
        (ACT_B, ACT_A),
        (ACT_B, ACT_A),
        (ACT_A, ACT_B),
        (ACT_A, ACT_B),
        (ACT_B, ACT_A),
        (ACT_A, ACT_B),
        (ACT_A, ACT_B),
        (ACT_B, ACT_A),
    ]
    conv_desc_addrs = []
    desc_ptr = DESC_BASE
    for idx, item in enumerate(conv_layers):
        layer = item["layer"]
        _, cin, ih, iw = [int(x) for x in layer["input_shape"]]
        _, _cout, oh, ow = [int(x) for x in layer["output_shape"]]
        m_dim = int(layer["registers"]["M_DIM"])
        n_dim = int(layer["registers"]["N_DIM"])
        k_dim = int(layer["registers"]["K_DIM"])
        ifm_base, ofm_base = conv_ifm_ofm[idx]
        ow_shift = {4: 2, 8: 3, 16: 4, 32: 5}[ow]
        in_padded_w = iw + 2
        in_padded_spatial = in_padded_w * (ih + 2)
        out_padded_w = ow + 2
        out_padded_spatial = out_padded_w * (oh + 2)
        fields = [
            ifm_base, ofm_base, item["w_base"], item["bias_base"], item["qcfg_base"],
            m_dim, n_dim, k_dim, cin, ih, iw, oh, ow, ow_shift, ow - 1,
            in_padded_spatial, item["w_stride"], in_padded_w,
            out_padded_spatial, out_padded_w,
            n_dim // TC,  # n_tiles at offset 80
        ]
        conv_desc_addrs.append(desc_ptr)
        for i, value in enumerate(fields):
            store_word(dram, desc_ptr + i * 4, value)
        desc_ptr += align(len(fields) * 4, 0x40)

    pool_specs = [
        (ACT_A, ACT_B, 64, 32, 32, 16, 16, 34, 34 * 34, 18, 18 * 18),
        (ACT_B, ACT_A, 128, 16, 16, 8, 8, 18, 18 * 18, 10, 10 * 10),
        (ACT_B, ACT_A, 256, 8, 8, 4, 4, 10, 10 * 10, 6, 6 * 6),
    ]
    pool_desc_addrs = []
    for fields in pool_specs:
        pool_desc_addrs.append(desc_ptr)
        for i, value in enumerate(fields):
            store_word(dram, desc_ptr + i * 4, value)
        desc_ptr += align(len(fields) * 4, 0x40)

    fw = emit_firmware(conv_desc_addrs, pool_desc_addrs, cls_w_base, cls_b_base, exact_pred)
    if len(fw) > MEM_WORDS:
        raise RuntimeError(f"firmware too large: {len(fw)} words > MEM_WORDS={MEM_WORDS}")

    max_addr = max(next_static, desc_ptr, MARKER_ADDR + 4)
    if max_addr >= DRAM_WORDS_MAX * 4:
        raise RuntimeError(f"DRAM image too small: max addr 0x{max_addr:x}")
    used_dram_words = align(max_addr + 0x10000, 0x10000) // 4

    write_sparse_hex(out / "dram_init.hex", dram, used_dram_words)
    write_hex(out / "soc_vgg_closed_loop.hex", fw)
    write_hex(out / "expected_features.hex", [int(x.item()) for x in fixed_features])

    with open(out / "soc_vgg_closed_loop_params.vh", "w", encoding="utf-8", newline="\n") as f:
        op = out.as_posix()
        f.write(f'`define VGG_CLOSED_FW_HEX "{op}/soc_vgg_closed_loop.hex"\n')
        f.write(f'`define VGG_CLOSED_DRAM_HEX "{op}/dram_init.hex"\n')
        f.write(f'`define VGG_CLOSED_FEAT_HEX "{op}/expected_features.hex"\n')
        f.write(f'`define VGG_CLOSED_FW_WORDS {len(fw)}\n')
        f.write(f'`define VGG_CLOSED_MEM_WORDS {MEM_WORDS}\n')
        f.write(f'`define VGG_CLOSED_DRAM_WORDS {used_dram_words}\n')
        f.write(f'`define VGG_CLOSED_TIMEOUT_CYCLES {args.timeout_cycles}\n')
        f.write(f"`define VGG_CLOSED_MARKER_ADDR 32'h{MARKER_ADDR:08x}\n")
        f.write(f"`define VGG_CLOSED_FEAT_BASE 32'h{FEAT_BASE:08x}\n")
        f.write(f'`define VGG_CLOSED_LABEL {exact_pred}\n')
        f.write(f'`define VGG_CLOSED_EXACT_LABEL {exact_pred}\n')
        f.write(f'`define VGG_CLOSED_FIXED_LABEL {fixed_pred}\n')
        f.write(f'`define VGG_CLOSED_SHAPE "{args.shape}"\n')
        f.write(f'`define VGG_CLOSED_FLOW "{args.flow}"\n')
        f.write(f'`define VGG_CLOSED_INT8_SIMD_LANES {args.lanes}\n')
        f.write(f'`define VGG_CLOSED_NPU_DATA_W {32 if args.lanes == 4 else 16}\n')
        f.write(f'`define VGG_CLOSED_NUM_CORES {NUM_CORES}\n')

    meta = {
        "schema": "vgg_closed_loop_v1",
        "img_idx": args.img_idx,
        "true_label": true_label,
        "fixed_runtime_pred": fixed_pred,
        "exact_python_pred": exact_pred,
        "fixed_scores": fixed_scores,
        "exact_scores": exact_scores,
        "shape": args.shape,
        "flow": args.flow,
        "int8_simd_lanes": args.lanes,
        "npu_data_w": 32 if args.lanes == 4 else 16,
        "tile_rows": TR,
        "tile_cols": TC,
        "a_pack_lanes": A_PACK_LANES,
        "kt_elems": KT_ELEMS,
        "firmware_words": len(fw),
        "max_addr": f"0x{max_addr:08x}",
        "dram_words": used_dram_words,
        "note": "Python does not pre-generate Conv A tiles; firmware packs A_WORK at runtime from dense activation buffers. fixed_runtime_pred models the CPU Q24 per-channel requant path used by firmware; exact_python_pred is the validation target.",
    }
    with open(out / "metadata.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)

    print(f"Generated closed-loop VGG case: {out}")
    print(f"  shape: {args.shape} (TR={TR}, TC={TC}, A_PACK_LANES={A_PACK_LANES}, KT_ELEMS={KT_ELEMS}), flow={args.flow}, lanes={args.lanes}")
    print(f"  firmware words: {len(fw)} / {MEM_WORDS}")
    print(f"  static end: 0x{next_static:08x}, desc end: 0x{desc_ptr:08x}, max: 0x{max_addr:08x}")
    print(f"  sparse dram words: {used_dram_words}")
    print(f"  fixed-runtime pred={fixed_pred}, exact-python pred={exact_pred}, true_label={true_label}")


def main():
    parser = argparse.ArgumentParser(description="Generate runtime closed-loop RepOpt VGG SoC case")
    parser.add_argument("--pth", default="RepOpt/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth")
    parser.add_argument("--plan", default="sim/pth_repopt_probe/model_plan.json")
    parser.add_argument("--spec", default="tools/pth/examples/repopt_vgg_int8_spec.json")
    parser.add_argument("--data-root", default="RepOpt/06_RepOpt_VGG/data")
    parser.add_argument("--out-dir", default="sim/vgg_closed_loop")
    parser.add_argument("--img-idx", type=int, default=0)
    parser.add_argument("--image", default="")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--shape", choices=sorted(SHAPE_CONFIGS), default=DEFAULT_SHAPE,
                        help="NPU tile shape: 4x4, 8x8, 16x16, or 8x32")
    parser.add_argument("--flow", choices=("os", "ws"), default="os",
                        help="Tile dataflow mode for NPU conv GEMMs")
    parser.add_argument("--lanes", type=int, choices=(1, 2, 4), default=4,
                        help="INT8 SIMD lanes per PE: 1, 2, or 4")
    parser.add_argument("--timeout-cycles", type=int, default=500000000)
    parser.add_argument("--num-cores", type=int, default=1, choices=(1, 2, 4),
                        help="Number of NPU cores: 1, 2, or 4")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
