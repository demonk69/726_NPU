#!/usr/bin/env python3
# =============================================================================
# gen_tiny_multilayer_soc_case.py - Build a small 3-layer .pth-driven SoC case.
#
# The generated RV32I firmware runs on the reference CPU and schedules:
#   Conv3x3(1->1)+ReLU -> CPU repack -> Conv1x1(1->2)+ReLU
#   -> CPU repack -> Conv1x1(2->1)+ReLU.
#
# This is intentionally small enough for the current ~60KB DRAM.
# =============================================================================

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
TB_DIR = PROJECT_ROOT / "tb"
if str(TB_DIR) not in sys.path:
    sys.path.insert(0, str(TB_DIR))

from assemble_soc_test import ANDI, BEQZ, BNE, J, LW, SW, i_type, li_insns, r_type, reg  # noqa: E402
from pth_to_npu_assets import convert as convert_pth  # noqa: E402


NPU_BASE = 0x02000000
A0_ADDR = 0x00001000
R0_ADDR = 0x00001100
A1_ADDR = 0x00001200
R1_ADDR = 0x00001300
A2_ADDR = 0x00001400
R2_ADDR = 0x00001500
W_BASE = 0x00002000
MARKER_ADDR = 0x00003000
PASS_MARKER = 0x000000AA
FAIL_MARKER = 0x000000FF
DRAM_WORDS = 15360

REG_CTRL = 0x00
REG_STATUS = 0x04
REG_M_DIM = 0x10
REG_N_DIM = 0x14
REG_K_DIM = 0x18
REG_W_ADDR = 0x20
REG_A_ADDR = 0x24
REG_R_ADDR = 0x28
REG_CONV_IFM_SHAPE = 0x80
REG_CONV_CHANNELS = 0x84
REG_CONV_KERNEL = 0x88
REG_CONV_OUT_SHAPE = 0x8C
REG_CONV_STRIDE_PAD = 0x90
REG_CONV_DILATION = 0x94
REG_BIAS_ADDR = 0x98
REG_QUANT_CFG = 0x9C


def OR(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x6, reg(rd), 0x33)


def SLLI(rd, rs1, shamt):
    return i_type(shamt & 0x1F, reg(rs1), 0x1, reg(rd), 0x13)


def require_torch():
    try:
        import torch  # noqa: WPS433
    except Exception as exc:
        raise SystemExit(
            "PyTorch is required to generate the tiny multilayer .pth case. "
            "Install CPU PyTorch with the command in tools/pth/README.md.\n"
            f"Import error: {exc}"
        ) from exc
    return torch


def int8_pack(values):
    words = []
    for idx in range(0, len(values), 4):
        word = 0
        for lane in range(4):
            if idx + lane < len(values):
                word |= (int(values[idx + lane]) & 0xFF) << (8 * lane)
        words.append(word & 0xFFFFFFFF)
    return words


def write_hex(path, words):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for word in words:
            f.write(f"{word & 0xFFFFFFFF:08x}\n")


def load_hex_words(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                words.append(int(line, 16))
    return words


def conv2d_nchw(ifm, weights, bias, relu=True):
    batch = len(ifm)
    cin = len(ifm[0])
    ih = len(ifm[0][0])
    iw = len(ifm[0][0][0])
    cout = len(weights)
    kh = len(weights[0][0])
    kw = len(weights[0][0][0])
    oh = ih - kh + 1
    ow = iw - kw + 1
    out = [[[[0 for _ in range(ow)] for _ in range(oh)] for _ in range(cout)] for _ in range(batch)]
    for n in range(batch):
        for oc in range(cout):
            for y in range(oh):
                for x in range(ow):
                    acc = bias[oc]
                    for ic in range(cin):
                        for ky in range(kh):
                            for kx in range(kw):
                                acc += ifm[n][ic][y + ky][x + kx] * weights[oc][ic][ky][kx]
                    if relu and acc < 0:
                        acc = 0
                    out[n][oc][y][x] = acc
    return out


def flatten_nchw(tensor):
    vals = []
    for n in range(len(tensor)):
        for c in range(len(tensor[n])):
            for y in range(len(tensor[n][c])):
                for x in range(len(tensor[n][c][y])):
                    vals.append(tensor[n][c][y][x])
    return vals


def flatten_nhwc_words(tensor):
    vals = []
    batch = len(tensor)
    channels = len(tensor[0])
    height = len(tensor[0][0])
    width = len(tensor[0][0][0])
    for n in range(batch):
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    vals.append(tensor[n][c][y][x] & 0xFFFFFFFF)
    return vals


def qweight(torch, nested):
    tensor = torch.tensor(nested, dtype=torch.float32)
    cout = tensor.shape[0]
    scales = torch.ones(cout, dtype=torch.float64)
    zero_points = torch.zeros(cout, dtype=torch.int64)
    return torch.quantize_per_channel(tensor, scales, zero_points, axis=0, dtype=torch.qint8)


def make_checkpoint(out_dir):
    torch = require_torch()

    ifm = [[[
        [1, 2, 0, -1],
        [3, 1, -2, 2],
        [0, 1, 4, 1],
        [-1, 2, 1, 3],
    ]]]

    w0 = [[[
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]]]
    b0 = [0]

    w1 = [
        [[[1]]],
        [[[2]]],
    ]
    b1 = [1, -3]

    w2 = [
        [
            [[1]],
            [[1]],
        ]
    ]
    b2 = [0]

    y0 = conv2d_nchw(ifm, w0, b0)
    y1 = conv2d_nchw(y0, w1, b1)
    y2 = conv2d_nchw(y1, w2, b2)

    state = {
        "quant.scale": torch.tensor(1.0, dtype=torch.float32),
        "quant.zero_point": torch.tensor(0, dtype=torch.int64),
        "conv0.weight": qweight(torch, w0),
        "conv0.bias": torch.tensor(b0, dtype=torch.float32),
        "conv0.scale": torch.tensor(1.0, dtype=torch.float32),
        "conv0.zero_point": torch.tensor(0, dtype=torch.int64),
        "conv1.weight": qweight(torch, w1),
        "conv1.bias": torch.tensor(b1, dtype=torch.float32),
        "conv1.scale": torch.tensor(1.0, dtype=torch.float32),
        "conv1.zero_point": torch.tensor(0, dtype=torch.int64),
        "conv2.weight": qweight(torch, w2),
        "conv2.bias": torch.tensor(b2, dtype=torch.float32),
        "conv2.scale": torch.tensor(1.0, dtype=torch.float32),
        "conv2.zero_point": torch.tensor(0, dtype=torch.int64),
    }
    pth_path = out_dir / "tiny_multilayer_int8.pth"
    torch.save({"state_dict": state}, pth_path)

    spec = {
        "name": "tiny_multilayer_conv_relu_soc_int8",
        "input": {
            "shape": [1, 1, 4, 4],
            "dtype": "int8",
            "layout": "NCHW",
            "scale_key": "quant.scale",
            "zero_point_key": "quant.zero_point",
            "source_preprocess": "fixed tiny NCHW int8 tensor generated by gen_tiny_multilayer_soc_case.py",
        },
        "layers": [
            {
                "name": "conv0",
                "op": "conv2d",
                "weight": "conv0.weight",
                "bias": "conv0.bias",
                "stride": [1, 1],
                "padding": [0, 0],
                "dilation": [1, 1],
                "activation": "relu",
            },
            {
                "name": "conv1",
                "op": "conv2d",
                "weight": "conv1.weight",
                "bias": "conv1.bias",
                "stride": [1, 1],
                "padding": [0, 0],
                "dilation": [1, 1],
                "activation": "relu",
            },
            {
                "name": "conv2",
                "op": "conv2d",
                "weight": "conv2.weight",
                "bias": "conv2.bias",
                "stride": [1, 1],
                "padding": [0, 0],
                "dilation": [1, 1],
                "activation": "relu",
            },
        ],
    }
    spec_path = out_dir / "tiny_multilayer_int8_spec.json"
    with open(spec_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(spec, f, indent=2)

    write_hex(out_dir / "input_nchw_int8.hex", int8_pack(flatten_nchw(ifm)))
    write_hex(out_dir / "layer0_expected_acc_int32.hex", flatten_nhwc_words(y0))
    write_hex(out_dir / "layer1_expected_acc_int32.hex", flatten_nhwc_words(y1))
    write_hex(out_dir / "expected_acc_int32.hex", flatten_nhwc_words(y2))
    with open(out_dir / "expected_acc_int32.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            {
                "shape": [1, 1, 2, 2],
                "layout": "NHWC row-major words",
                "layer0_nhwc": flatten_nhwc_words(y0),
                "layer1_nhwc": flatten_nhwc_words(y1),
                "final_nhwc": flatten_nhwc_words(y2),
            },
            f,
            indent=2,
        )

    return pth_path, spec_path, flatten_nhwc_words(y2)


def write_dram_init(out_dir, plan, input_words):
    dram = [0 for _ in range(DRAM_WORDS)]
    for idx, word in enumerate(input_words):
        dram[(A0_ADDR >> 2) + idx] = word

    for layer in plan["layers"]:
        assets = layer["assets"]
        for key, addr_key in (("w_col_hex", "w_addr"), ("bias_int32_hex", "bias_addr")):
            words = load_hex_words(out_dir / assets[key])
            base = int(assets[addr_key], 0) >> 2
            for idx, word in enumerate(words):
                dram[base + idx] = word

    write_hex(out_dir / "dram_init.hex", dram)


def assemble_firmware(plan, expected):
    insns = []
    labels = {}

    def emit(*words):
        for word in words:
            insns.append(word)

    def label(name):
        labels[name] = len(insns)

    def write_reg(offset, value):
        emit(*li_insns("t1", int(value)))
        emit(SW("t1", "t0", offset))

    def patch_beqz(idx, target_label, rs="t1"):
        offset = (labels[target_label] - idx) * 4
        insns[idx] = BEQZ(rs, offset)

    def patch_bne(idx, target_label, rs1="t1", rs2="t3"):
        offset = (labels[target_label] - idx) * 4
        insns[idx] = BNE(rs1, rs2, offset)

    def patch_j(idx, target_label):
        offset = (labels[target_label] - idx) * 4
        insns[idx] = J(offset)

    def run_layer(layer_idx, ifm_addr, ofm_addr):
        layer = plan["layers"][layer_idx]
        regs = layer["registers"]
        write_reg(REG_CTRL, 0)
        write_reg(REG_M_DIM, regs["M_DIM"])
        write_reg(REG_N_DIM, regs["N_DIM"])
        write_reg(REG_K_DIM, regs["K_DIM"])
        write_reg(REG_W_ADDR, int(regs["W_ADDR"]))
        write_reg(REG_A_ADDR, ifm_addr)
        write_reg(REG_R_ADDR, ofm_addr)
        write_reg(REG_CONV_IFM_SHAPE, regs["CONV_IFM_SHAPE"])
        write_reg(REG_CONV_CHANNELS, regs["CONV_CHANNELS"])
        write_reg(REG_CONV_KERNEL, regs["CONV_KERNEL"])
        write_reg(REG_CONV_OUT_SHAPE, regs["CONV_OUT_SHAPE"])
        write_reg(REG_CONV_STRIDE_PAD, regs["CONV_STRIDE_PAD"])
        write_reg(REG_CONV_DILATION, regs["CONV_DILATION"])
        write_reg(REG_BIAS_ADDR, int(regs["BIAS_ADDR"]))
        write_reg(REG_QUANT_CFG, regs["QUANT_CFG"])
        write_reg(REG_CTRL, regs["CTRL"])

        poll_label = f"poll_layer{layer_idx}"
        label(poll_label)
        emit(LW("t1", "t0", REG_STATUS))
        emit(ANDI("t1", "t1", 2))
        beq_idx = len(insns)
        emit(0)
        patch_beqz(beq_idx, poll_label)
        write_reg(REG_CTRL, 0)

    def emit_pack_word(src_base, src_word_offsets, dst_addr):
        emit(*li_insns("t2", src_base))
        emit(*li_insns("t3", dst_addr))
        emit(*li_insns("t4", 0))
        for lane, word_offset in enumerate(src_word_offsets):
            emit(LW("t1", "t2", word_offset * 4))
            emit(ANDI("t1", "t1", 0xFF))
            if lane != 0:
                emit(SLLI("t1", "t1", lane * 8))
            emit(OR("t4", "t4", "t1"))
        emit(SW("t4", "t3", 0))

    label("_start")
    emit(*li_insns("sp", 0x00001000))
    emit(*li_insns("t0", NPU_BASE))

    run_layer(0, A0_ADDR, R0_ADDR)
    emit_pack_word(R0_ADDR, [0, 1, 2, 3], A1_ADDR)

    run_layer(1, A1_ADDR, R1_ADDR)
    emit_pack_word(R1_ADDR, [0, 2, 4, 6], A2_ADDR)
    emit_pack_word(R1_ADDR, [1, 3, 5, 7], A2_ADDR + 4)

    run_layer(2, A2_ADDR, R2_ADDR)

    emit(*li_insns("t2", R2_ADDR))
    fail_branches = []
    for idx, value in enumerate(expected):
        emit(LW("t1", "t2", idx * 4))
        emit(*li_insns("t3", value))
        fail_branches.append(len(insns))
        emit(0)

    write_pass_j_idx = len(insns)
    emit(0)

    label("verify_fail")
    emit(*li_insns("t0", MARKER_ADDR))
    emit(*li_insns("t1", FAIL_MARKER))
    emit(SW("t1", "t0", 0))
    end_j_fail = len(insns)
    emit(0)

    label("write_pass")
    emit(*li_insns("t0", MARKER_ADDR))
    emit(*li_insns("t1", PASS_MARKER))
    emit(SW("t1", "t0", 0))

    label("end")
    end_j_end = len(insns)
    emit(0)

    for branch_idx in fail_branches:
        patch_bne(branch_idx, "verify_fail")
    patch_j(write_pass_j_idx, "write_pass")
    patch_j(end_j_fail, "end")
    patch_j(end_j_end, "end")
    return insns


def write_params(out_dir, fw_words, expected):
    lines = [
        f"`define PTH_MULTI_FW_HEX \"../sim/{out_dir.name}/soc_pth_multilayer.hex\"",
        f"`define PTH_MULTI_DRAM_HEX \"../sim/{out_dir.name}/dram_init.hex\"",
        f"`define PTH_MULTI_FW_WORDS {fw_words}",
        f"`define PTH_MULTI_RESULT_BASE 32'h{R2_ADDR:08x}",
        f"`define PTH_MULTI_MARKER_ADDR 32'h{MARKER_ADDR:08x}",
        f"`define PTH_MULTI_RESULT_COUNT {len(expected)}",
    ]
    for idx, value in enumerate(expected):
        lines.append(f"`define PTH_MULTI_EXPECTED_{idx} 32'sd{value}")
    with open(out_dir / "soc_pth_multilayer_params.vh", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Generate a 3-layer .pth-driven SoC Conv/ReLU case")
    parser.add_argument("--out-dir", default="sim/pth_multilayer_conv", help="Output case directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pth_path, spec_path, expected = make_checkpoint(out_dir)
    convert_pth(
        SimpleNamespace(
            pth=str(pth_path),
            spec=str(spec_path),
            out_dir=str(out_dir),
            mode="OS",
            base_addr=f"0x{W_BASE:08x}",
        )
    )

    with open(out_dir / "model_plan.json", "r", encoding="utf-8") as f:
        plan = json.load(f)
    write_dram_init(out_dir, plan, load_hex_words(out_dir / "input_nchw_int8.hex"))

    insns = assemble_firmware(plan, expected)
    write_hex(out_dir / "soc_pth_multilayer.hex", insns)
    write_params(out_dir, len(insns), expected)

    print(f"generated 3-layer SoC pth case: {out_dir}")
    print(f"firmware words: {len(insns)}")
    print(f"expected final: {expected}")


if __name__ == "__main__":
    main()
