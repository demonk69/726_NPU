#!/usr/bin/env python3
# =============================================================================
# gen_repopt_layer_case.py - Generate a staged RepOpt VGG Conv2D RTL case.
#
# The first supported stage is a real checkpoint Conv/ReLU layer driven by a
# real CIFAR-10 test sample. It creates the same files consumed by
# tb/matmul/tb_matmul_os.v: test_params.vh, dram_init.hex, expected.hex.
# =============================================================================

import argparse
import json
import pickle
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in obj:
                return obj[key]
    return obj


def tensor_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def load_cifar_sample(data_root, index):
    batch_path = Path(data_root) / "cifar-10-batches-py" / "test_batch"
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    raw = torch.tensor(batch["data"][index], dtype=torch.float64).reshape(3, 32, 32) / 255.0
    label = int(batch["labels"][index])
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float64).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float64).view(3, 1, 1)
    return ((raw - mean) / std).unsqueeze(0), label


def quantize_qint8(x, scale, zero_point):
    q = torch.round(x / float(scale)) + int(zero_point)
    return torch.clamp(q, -128, 127).to(torch.int16)


def load_hex_words(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                words.append(int(line, 16) & 0xFFFFFFFF)
    return words


def signed32(word):
    word &= 0xFFFFFFFF
    return word - 0x100000000 if word & 0x80000000 else word


def load_int32_hex(path):
    return torch.tensor([signed32(word) for word in load_hex_words(path)], dtype=torch.float64)


def pack_int8(values):
    vals = [int(v) for v in values]
    words = []
    for idx in range(0, len(vals), 4):
        word = 0
        for lane in range(4):
            if idx + lane < len(vals):
                word |= (vals[idx + lane] & 0xFF) << (8 * lane)
        words.append(word & 0xFFFFFFFF)
    return words


def write_hex(path, words):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for word in words:
            f.write(f"{int(word) & 0xFFFFFFFF:08x}\n")


def activation_mode_id(name):
    return {"none": 0, "relu": 1, "relu6": 2}.get(name, 0)


def first_conv_input_q(plan, state_dict, data_root, index):
    x_float, label = load_cifar_sample(data_root, index)
    in_scale = float(tensor_scalar(state_dict[plan["input"]["scale_key"]]))
    in_zp = int(tensor_scalar(state_dict[plan["input"]["zero_point_key"]]))
    return quantize_qint8(x_float, in_scale, in_zp), label, in_scale, in_zp


def conv2d_acc_expected(x_q, qweight, bias_acc, layer):
    regs = layer["registers"]
    stride_pad = int(regs["CONV_STRIDE_PAD"])
    dilation_word = int(regs["CONV_DILATION"])
    stride = (stride_pad & 0xFF, (stride_pad >> 8) & 0xFF)
    padding = ((stride_pad >> 16) & 0xFF, (stride_pad >> 24) & 0xFF)
    dilation = (dilation_word & 0xFF, (dilation_word >> 8) & 0xFF)

    w = qweight.int_repr().to(torch.float64)
    acc = F.conv2d(
        x_q.to(torch.float64),
        w,
        bias=bias_acc.to(torch.float64),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if layer.get("activation") == "relu":
        acc = torch.clamp(acc, min=0)
    elif layer.get("activation") == "relu6":
        acc = torch.clamp(acc, min=0, max=6)
    return acc.to(torch.int64)


def flatten_nchw(values):
    flat = []
    n, c, h, w = values.shape
    for batch in range(n):
        for ch in range(c):
            for row in range(h):
                for col in range(w):
                    flat.append(int(values[batch, ch, row, col]))
    return flat


def flatten_nhwc(values):
    flat = []
    n, c, h, w = values.shape
    for batch in range(n):
        for row in range(h):
            for col in range(w):
                for ch in range(c):
                    flat.append(int(values[batch, ch, row, col]))
    return flat


def build_dram(w_words, ifm_words, bias_words, m_dim, n_dim, w_base, a_base, r_base, bias_base):
    dram = {}
    for idx, word in enumerate(w_words):
        dram[(w_base >> 2) + idx] = word
    for idx, word in enumerate(ifm_words):
        dram[(a_base >> 2) + idx] = word
    for idx, word in enumerate(bias_words):
        dram[(bias_base >> 2) + idx] = word
    for idx in range(m_dim * n_dim):
        dram[(r_base >> 2) + idx] = 0
    return dram


def write_test_params(path, test_id, layer, label, addresses, dram_size):
    in_shape = layer["input_shape"]
    out_shape = layer["output_shape"]
    regs = layer["registers"]
    activation = layer.get("activation", "none")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"// Auto-generated: {test_id}\n")
        f.write("// RepOpt VGG staged RTL case using real checkpoint weights and CIFAR input.\n")
        f.write(f"// CIFAR label index: {label}\n")
        f.write(
            f"// Conv2D: B={in_shape[0]} IFM={in_shape[2]}x{in_shape[3]} Cin={in_shape[1]} "
            f"Cout={out_shape[1]} OFM={out_shape[2]}x{out_shape[3]}\n"
        )
        f.write(f"`define NUM_RESULTS {int(regs['M_DIM']) * int(regs['N_DIM'])}\n")
        f.write(f"`define M_DIM {int(regs['M_DIM'])}\n")
        f.write(f"`define N_DIM {int(regs['N_DIM'])}\n")
        f.write(f"`define K_DIM {int(regs['K_DIM'])}\n")
        f.write(f"`define W_ADDR 32'h{addresses['w_base']:08x}\n")
        f.write(f"`define A_ADDR 32'h{addresses['a_base']:08x}\n")
        f.write(f"`define R_ADDR 32'h{addresses['r_base']:08x}\n")
        f.write(f"`define BIAS_ADDR 32'h{addresses['bias_base']:08x}\n")
        f.write("`define BIAS_EN 1\n")
        f.write(f"`define ACT_MODE {activation_mode_id(activation)}\n")
        f.write("`define QUANT_EN 0\n")
        f.write("`define QUANT_CFG 32'h00010000\n")
        f.write("`define QUANT_SCALE 1\n")
        f.write("`define QUANT_SHIFT 0\n")
        f.write("`define QUANT_ROUND 0\n")
        f.write(f"`define CTRL   32'h{int(regs['CTRL']):02x}\n")
        f.write(f"`define DRAM_SIZE {dram_size}\n")
        f.write("`define IS_FP16 0\n")
        f.write("`define IS_OS   1\n")
        f.write(f"`define CONV_BATCH {in_shape[0]}\n")
        f.write(f"`define CONV_IH {in_shape[2]}\n")
        f.write(f"`define CONV_IW {in_shape[3]}\n")
        f.write(f"`define CONV_CIN {in_shape[1]}\n")
        f.write(f"`define CONV_COUT {out_shape[1]}\n")
        f.write(f"`define CONV_KH {layer['weight_shape'][2]}\n")
        f.write(f"`define CONV_KW {layer['weight_shape'][3]}\n")
        f.write(f"`define CONV_OH {out_shape[2]}\n")
        f.write(f"`define CONV_OW {out_shape[3]}\n")
        f.write("`define CONV_IM2COL 1\n")
        f.write(f"`define CONV_IFM_SHAPE 32'h{int(regs['CONV_IFM_SHAPE']):08x}\n")
        f.write(f"`define CONV_CHANNELS 32'h{int(regs['CONV_CHANNELS']):08x}\n")
        f.write(f"`define CONV_KERNEL 32'h{int(regs['CONV_KERNEL']):08x}\n")
        f.write(f"`define CONV_OUT_SHAPE 32'h{int(regs['CONV_OUT_SHAPE']):08x}\n")
        f.write(f"`define CONV_STRIDE_PAD 32'h{int(regs['CONV_STRIDE_PAD']):08x}\n")
        f.write(f"`define CONV_DILATION 32'h{int(regs['CONV_DILATION']):08x}\n")


def generate(args):
    plan_path = Path(args.plan).resolve()
    plan = load_json(plan_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
        checkpoint = torch.load(args.pth, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)

    layer = next(
        (item for item in plan["layers"] if item.get("name") == args.layer_name and item.get("op") == "conv2d"),
        None,
    )
    if layer is None:
        raise ValueError(f"conv layer not found in plan: {args.layer_name}")
    if list(layer["input_shape"]) != list(plan["input"]["shape"]):
        raise ValueError("this staged generator currently supports the first RepOpt Conv layer only")

    x_q, label, in_scale, in_zp = first_conv_input_q(plan, state_dict, args.data_root, args.index)
    qweight = state_dict[layer["weight_key"]]
    plan_dir = plan_path.parent
    w_words = load_hex_words(plan_dir / layer["assets"]["w_col_hex"])
    bias_words = load_hex_words(plan_dir / layer["assets"]["bias_int32_hex"])
    bias_acc = load_int32_hex(plan_dir / layer["assets"]["bias_int32_hex"])
    full_expected_acc = conv2d_acc_expected(x_q, qweight, bias_acc, layer)
    full_out_shape = layer["output_shape"]
    case_oh = int(args.tile_oh) if int(args.tile_oh) > 0 else int(full_out_shape[2])
    case_ow = int(args.tile_ow) if int(args.tile_ow) > 0 else int(full_out_shape[3])
    if case_oh > int(full_out_shape[2]) or case_ow > int(full_out_shape[3]):
        raise ValueError(f"tile {case_oh}x{case_ow} exceeds layer output {full_out_shape[2]}x{full_out_shape[3]}")
    expected_acc = full_expected_acc[:, :, :case_oh, :case_ow].contiguous()

    case_layer = dict(layer)
    case_layer["output_shape"] = [full_out_shape[0], full_out_shape[1], case_oh, case_ow]
    case_regs = dict(layer["registers"])
    case_regs["M_DIM"] = int(full_out_shape[0]) * case_oh * case_ow
    case_regs["CONV_OUT_SHAPE"] = (case_ow << 16) | case_oh
    case_layer["registers"] = case_regs

    m_dim = int(case_layer["registers"]["M_DIM"])
    n_dim = int(case_layer["registers"]["N_DIM"])
    k_dim = int(case_layer["registers"]["K_DIM"])
    elem_bytes = 1
    b_col_stride = ((k_dim * elem_bytes + 3) >> 2) << 2
    expected_w_words = (b_col_stride // 4) * n_dim
    if len(w_words) != expected_w_words:
        raise ValueError(f"weight words {len(w_words)} != expected {expected_w_words}")

    ifm_words = pack_int8(flatten_nchw(x_q))
    w_base = int(args.w_base, 0)
    a_base = w_base + len(w_words) * 4 + 0x100
    r_base = a_base + len(ifm_words) * 4 + 0x100
    bias_base = r_base + m_dim * n_dim * 4 + 0x100

    dram = build_dram(w_words, ifm_words, bias_words, m_dim, n_dim, w_base, a_base, r_base, bias_base)
    max_word_addr = max(dram.keys()) if dram else 0
    dram_arr = [0] * (max(max_word_addr + 1, 8192))
    for addr, word in dram.items():
        dram_arr[addr] = word

    tile_suffix = "" if (case_oh == int(full_out_shape[2]) and case_ow == int(full_out_shape[3])) else f"_tile{case_oh}x{case_ow}"
    test_id = args.test_id or f"repopt_{args.layer_name}_idx{args.index}{tile_suffix}"
    out_dir = Path(args.out_root).resolve() / test_id
    out_dir.mkdir(parents=True, exist_ok=True)

    write_hex(out_dir / "dram_init.hex", dram_arr)
    write_hex(out_dir / "expected.hex", [value & 0xFFFFFFFF for value in flatten_nhwc(expected_acc)])
    write_hex(out_dir / "expected_acc_nchw.hex", [value & 0xFFFFFFFF for value in flatten_nchw(expected_acc)])
    write_hex(out_dir / "ifm_q_nchw.hex", ifm_words)
    write_test_params(
        out_dir / "test_params.vh",
        test_id,
        case_layer,
        label,
        {"w_base": w_base, "a_base": a_base, "r_base": r_base, "bias_base": bias_base},
        len(dram_arr),
    )

    with open(out_dir / "metadata.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"test_id={test_id}\n")
        f.write(f"source_pth={Path(args.pth).resolve()}\n")
        f.write(f"plan={plan_path}\n")
        f.write(f"layer={args.layer_name}\n")
        f.write(f"sample_index={args.index}\n")
        f.write(f"cifar_label={label}\n")
        f.write(f"input_quant_scale={in_scale}\n")
        f.write(f"input_quant_zero_point={in_zp}\n")
        f.write(f"full_ofm={full_out_shape[2]}x{full_out_shape[3]}\n")
        f.write(f"case_ofm={case_oh}x{case_ow}\n")
        f.write(f"gemm={m_dim}x{k_dim}x{n_dim}\n")
        f.write(f"w_addr=0x{w_base:08x}\n")
        f.write(f"a_addr=0x{a_base:08x}\n")
        f.write(f"r_addr=0x{r_base:08x}\n")
        f.write(f"bias_addr=0x{bias_base:08x}\n")
        f.write(f"dram_words={len(dram_arr)}\n")
        f.write("npu_output=int32_accumulator_after_bias_relu\n")
        f.write("cpu_requant=not_in_this_case\n")

    print(f"Generated RepOpt layer RTL case: {out_dir}")
    print(f"  layer={args.layer_name} sample_index={args.index} cifar_label={label}")
    print(f"  output_window={case_oh}x{case_ow} of full {full_out_shape[2]}x{full_out_shape[3]}")
    print(f"  GEMM M={m_dim} K={k_dim} N={n_dim} results={m_dim * n_dim}")
    print(f"  W=0x{w_base:08x} A=0x{a_base:08x} R=0x{r_base:08x} BIAS=0x{bias_base:08x}")
    print(f"  expected_acc_min={int(expected_acc.min().item())} expected_acc_max={int(expected_acc.max().item())}")


def main():
    parser = argparse.ArgumentParser(description="Generate a RepOpt VGG staged Conv2D RTL case")
    parser.add_argument(
        "--pth",
        default=".06_RepOpt_VGG/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth",
        help="Path to qat_int8_quantized.pth",
    )
    parser.add_argument("--plan", default="sim/pth_repopt_probe/model_plan.json", help="Path to model_plan.json")
    parser.add_argument("--data-root", default=".06_RepOpt_VGG/06_RepOpt_VGG/data", help="CIFAR-10 data root")
    parser.add_argument("--layer-name", default="stage1_0_conv", help="Conv layer name in model_plan.json")
    parser.add_argument("--index", type=int, default=0, help="CIFAR-10 test sample index")
    parser.add_argument("--out-root", default="sim/pth_repopt_layer_cases", help="Generated case root")
    parser.add_argument("--test-id", default="", help="Optional case directory name")
    parser.add_argument("--w-base", default="0x00010000", help="Weight base address in test DRAM")
    parser.add_argument("--tile-oh", type=int, default=0, help="Optional top-left output tile height; 0 means full layer")
    parser.add_argument("--tile-ow", type=int, default=0, help="Optional top-left output tile width; 0 means full layer")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
