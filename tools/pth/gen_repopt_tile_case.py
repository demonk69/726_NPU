#!/usr/bin/env python3
# =============================================================================
# gen_repopt_tile_case.py - Generate a RepOpt VGG 4x4 tile-mode GEMM case.
#
# This validates the existing ARR_CFG[7] tile datapath with real RepOpt Conv2D
# data. It pre-packs one local GEMM tile:
#   A_TILE[k][r] = im2col(global_m_base + r, k)
#   W_TILE[k][c] = weight(k, global_n_base + c)
#
# Tile mode currently verifies raw int32 MAC accumulators. Direct-scalar
# bias/ReLU/per-channel requant are separate stages.
# =============================================================================

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import torch


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


def pack4_int8(values):
    word = 0
    for lane, value in enumerate(values):
        word |= (int(value) & 0xFF) << (lane * 8)
    return word & 0xFFFFFFFF


def write_hex(path, words):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for word in words:
            f.write(f"{int(word) & 0xFFFFFFFF:08x}\n")


def signed8(value):
    value = int(value) & 0xFF
    return value - 0x100 if value & 0x80 else value


def signed32(value):
    value = int(value) & 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def conv_tile_a_value(x_q, layer, global_m, k_index):
    in_shape = layer["input_shape"]
    out_shape = layer["output_shape"]
    weight_shape = layer["weight_shape"]
    _batch, cin, ih, iw = in_shape
    _out_batch, _cout, oh, ow = out_shape
    _w_cout, _w_cin, kh, kw = weight_shape
    regs = layer["registers"]
    stride_pad = int(regs["CONV_STRIDE_PAD"])
    dilation_word = int(regs["CONV_DILATION"])
    stride_h = stride_pad & 0xFF
    stride_w = (stride_pad >> 8) & 0xFF
    pad_h = (stride_pad >> 16) & 0xFF
    pad_w = (stride_pad >> 24) & 0xFF
    dilation_h = dilation_word & 0xFF
    dilation_w = (dilation_word >> 8) & 0xFF

    batch = global_m // (oh * ow)
    rem = global_m % (oh * ow)
    out_h = rem // ow
    out_w = rem % ow
    chan = k_index // (kh * kw)
    rem_k = k_index % (kh * kw)
    ker_h = rem_k // kw
    ker_w = rem_k % kw

    in_h = out_h * stride_h + ker_h * dilation_h - pad_h
    in_w = out_w * stride_w + ker_w * dilation_w - pad_w
    if batch < 0 or batch >= in_shape[0] or chan < 0 or chan >= cin:
        return 0
    if in_h < 0 or in_h >= ih or in_w < 0 or in_w >= iw:
        return 0
    return int(x_q[batch, chan, in_h, in_w].item())


def build_tile_matrices(x_q, qweight, layer, m_base, n_base, tile_rows=4, tile_cols=4):
    k_dim = int(layer["registers"]["K_DIM"])
    cout = int(layer["weight_shape"][0])
    total_m = int(layer["registers"]["M_DIM"])
    if m_base < 0 or m_base + tile_rows > total_m:
        raise ValueError(f"m-base {m_base} does not fit one {tile_rows}-row tile in M={total_m}")
    if n_base < 0 or n_base + tile_cols > cout:
        raise ValueError(f"n-base {n_base} does not fit one {tile_cols}-col tile in Cout={cout}")

    weight_int = qweight.int_repr().cpu()
    _cout, cin, kh, kw = [int(x) for x in weight_int.shape]
    a_tile = []
    w_tile = []
    for k_index in range(k_dim):
        a_tile.append([conv_tile_a_value(x_q, layer, m_base + r, k_index) for r in range(tile_rows)])
        chan = k_index // (kh * kw)
        rem_k = k_index % (kh * kw)
        ker_h = rem_k // kw
        ker_w = rem_k % kw
        w_tile.append([int(weight_int[n_base + c, chan, ker_h, ker_w].item()) for c in range(tile_cols)])
    return a_tile, w_tile


def expected_tile(a_tile, w_tile, tile_rows=4, tile_cols=4):
    out = []
    for r in range(tile_rows):
        for c in range(tile_cols):
            acc = 0
            for k_index in range(len(a_tile)):
                acc += signed8(a_tile[k_index][r]) * signed8(w_tile[k_index][c])
            out.append(acc & 0xFFFFFFFF)
    return out


def build_dram(a_tile, w_tile, w_addr, a_addr, r_addr, dram_words):
    dram = [0] * dram_words
    for k_index in range(len(a_tile)):
        dram[(w_addr >> 2) + k_index] = pack4_int8(w_tile[k_index])
        dram[(a_addr >> 2) + k_index] = pack4_int8(a_tile[k_index])
    for idx in range(16):
        dram[(r_addr >> 2) + idx] = 0
    return dram


def rel_path_for_readmemh(path, project_root):
    rel = os.path.relpath(path, project_root)
    return Path(rel).as_posix()


def generate(args):
    project_root = Path(args.project_root).resolve()
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
        raise ValueError("this tile generator currently supports the first RepOpt Conv layer only")

    x_float, label = load_cifar_sample(args.data_root, args.index)
    in_scale = float(tensor_scalar(state_dict[plan["input"]["scale_key"]]))
    in_zp = int(tensor_scalar(state_dict[plan["input"]["zero_point_key"]]))
    x_q = quantize_qint8(x_float, in_scale, in_zp)
    qweight = state_dict[layer["weight_key"]]

    a_tile, w_tile = build_tile_matrices(x_q, qweight, layer, args.m_base, args.n_base)
    expected = expected_tile(a_tile, w_tile)

    test_id = args.test_id or f"repopt_{args.layer_name}_tile4_m{args.m_base}_n{args.n_base}_idx{args.index}"
    out_dir = Path(args.out_root).resolve() / test_id
    out_dir.mkdir(parents=True, exist_ok=True)

    w_addr = int(args.w_addr, 0)
    a_addr = int(args.a_addr, 0)
    r_addr = int(args.r_addr, 0)
    dram_words = max(int(args.dram_words), (r_addr >> 2) + 16, 1024)
    dram = build_dram(a_tile, w_tile, w_addr, a_addr, r_addr, dram_words)

    dram_hex = out_dir / "dram_init.hex"
    expected_hex = out_dir / "expected.hex"
    output_hex = out_dir / "npu_output.hex"
    write_hex(dram_hex, dram)
    write_hex(expected_hex, expected)

    with open(out_dir / "test_params.vh", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"// Auto-generated: {test_id}\n")
        f.write("// RepOpt VGG 4x4 tile-mode GEMM case using real checkpoint data.\n")
        f.write("// Expected results are raw int32 MAC accumulators without bias/ReLU.\n")
        f.write(f'`define TEST_NAME "{test_id}"\n')
        f.write("`define M_DIM 4\n")
        f.write("`define N_DIM 4\n")
        f.write(f"`define K_DIM {int(layer['registers']['K_DIM'])}\n")
        f.write("`define NUM_RESULTS 16\n")
        f.write(f"`define DRAM_SIZE {dram_words}\n")
        f.write(f"`define W_ADDR 32'h{w_addr:08x}\n")
        f.write(f"`define A_ADDR 32'h{a_addr:08x}\n")
        f.write(f"`define R_ADDR 32'h{r_addr:08x}\n")
        f.write("`define CTRL 32'h00000011\n")
        f.write("`define IS_FP16 0\n")
        f.write(f'`define DRAM_HEX "{rel_path_for_readmemh(dram_hex, project_root)}"\n')
        f.write(f'`define EXPECTED_HEX "{rel_path_for_readmemh(expected_hex, project_root)}"\n')
        f.write(f'`define OUTPUT_HEX "{rel_path_for_readmemh(output_hex, project_root)}"\n')

    with open(out_dir / "metadata.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"test_id={test_id}\n")
        f.write(f"source_pth={Path(args.pth).resolve()}\n")
        f.write(f"plan={plan_path}\n")
        f.write(f"layer={args.layer_name}\n")
        f.write(f"sample_index={args.index}\n")
        f.write(f"cifar_label={label}\n")
        f.write(f"input_quant_scale={in_scale}\n")
        f.write(f"input_quant_zero_point={in_zp}\n")
        f.write(f"global_m_base={args.m_base}\n")
        f.write(f"global_n_base={args.n_base}\n")
        f.write(f"local_gemm=4x{int(layer['registers']['K_DIM'])}x4\n")
        f.write("npu_output=raw_int32_mac_accumulator\n")
        f.write("bias_relu=not_in_tile_mode_case\n")

    print(f"Generated RepOpt tile case: {out_dir}")
    print(f"  layer={args.layer_name} sample_index={args.index} cifar_label={label}")
    print(f"  global tile base: m={args.m_base} n={args.n_base}")
    print(f"  GEMM M=4 K={int(layer['registers']['K_DIM'])} N=4 results=16")
    signed_expected = [signed32(x) for x in expected]
    print(f"  expected_min={min(signed_expected)}")
    print(f"  expected_max={max(signed_expected)}")


def main():
    parser = argparse.ArgumentParser(description="Generate a RepOpt VGG 4x4 tile-mode GEMM case")
    parser.add_argument(
        "--pth",
        default=".06_RepOpt_VGG/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth",
        help="Path to qat_int8_quantized.pth",
    )
    parser.add_argument("--plan", default="sim/pth_repopt_probe/model_plan.json", help="Path to model_plan.json")
    parser.add_argument("--data-root", default=".06_RepOpt_VGG/06_RepOpt_VGG/data", help="CIFAR-10 data root")
    parser.add_argument("--layer-name", default="stage1_0_conv", help="Conv layer name in model_plan.json")
    parser.add_argument("--index", type=int, default=0, help="CIFAR-10 test sample index")
    parser.add_argument("--m-base", type=int, default=0, help="Global GEMM M base for the 4-row tile")
    parser.add_argument("--n-base", type=int, default=0, help="Global GEMM N base for the 4-column tile")
    parser.add_argument("--out-root", default="sim/pth_repopt_tile_cases", help="Generated tile case root")
    parser.add_argument("--test-id", default="", help="Optional case directory name")
    parser.add_argument("--project-root", default=".", help="Project root for readmemh relative paths")
    parser.add_argument("--w-addr", default="0x00000100", help="Tile W stream base address")
    parser.add_argument("--a-addr", default="0x00000200", help="Tile A stream base address")
    parser.add_argument("--r-addr", default="0x00000300", help="Tile result base address")
    parser.add_argument("--dram-words", type=int, default=1024, help="DRAM words in generated testbench image")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
