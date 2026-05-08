#!/usr/bin/env python3
# =============================================================================
# gen_cpu_runtime.py - Generate reference-CPU layer descriptors from an
# npu_pth_plan_v1 model_plan.json file.
#
# The generated C is a bridge, not a full deployment stack: it gives firmware a
# typed layer table, fixed-point requant constants, and a small header-only
# runtime for direct-register NPU Conv2D plus CPU-side pooling/repack/linear.
# =============================================================================

import argparse
import json
import re
import shutil
from pathlib import Path


OP_ENUM = {
    "conv2d": "NPU_PTH_OP_CONV2D",
    "maxpool2d": "NPU_PTH_OP_MAXPOOL2D",
    "adaptive_avgpool2d": "NPU_PTH_OP_ADAPTIVE_AVGPOOL2D",
    "flatten": "NPU_PTH_OP_FLATTEN",
    "linear": "NPU_PTH_OP_LINEAR",
}

EXEC_ENUM = {
    "cpu": "NPU_PTH_EXEC_CPU",
    "npu_direct": "NPU_PTH_EXEC_NPU_DIRECT",
}


def parse_int(value, default=0):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return int(str(value), 0)


def safe_ident(name):
    ident = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not ident:
        ident = "layer"
    if ident[0].isdigit():
        ident = f"l_{ident}"
    return ident


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def product(values):
    out = 1
    for value in values:
        out *= int(value)
    return out


def shape_expr(shape):
    if len(shape) == 4:
        return "NPU_PTH_SHAPE4({0}u, {1}u, {2}u, {3}u)".format(*[int(x) for x in shape])
    if len(shape) == 2:
        return "NPU_PTH_SHAPE2({0}u, {1}u)".format(*[int(x) for x in shape])
    raise ValueError(f"unsupported generated C shape rank: {shape}")


def fixed_point_multiplier(real_multiplier):
    real_multiplier = float(real_multiplier)
    if real_multiplier <= 0.0:
        return 0, 0

    shift = 30
    scaled = int(round(real_multiplier * (1 << shift)))
    while scaled > 0x7FFFFFFF and shift > 0:
        shift -= 1
        scaled = int(round(real_multiplier * (1 << shift)))
    while scaled != 0 and scaled < (1 << 23) and shift < 62:
        shift += 1
        scaled = int(round(real_multiplier * (1 << shift)))
    if scaled > 0x7FFFFFFF:
        scaled = 0x7FFFFFFF
    return scaled, shift


def format_array(values, c_type, name, per_line=8):
    if not values:
        return ""
    lines = [f"static const {c_type} {name}[] = {{"]
    for idx in range(0, len(values), per_line):
        chunk = values[idx : idx + per_line]
        suffix = "," if idx + per_line < len(values) else ""
        lines.append("    " + ", ".join(str(v) for v in chunk) + suffix)
    lines.append("};")
    return "\n".join(lines) + "\n\n"


def layer_requant_info(layer):
    req = layer.get("cpu_requant_after_npu") or layer.get("cpu_requant") or {}
    multipliers = req.get("multipliers", [])
    if not multipliers:
        return None
    fixed = [fixed_point_multiplier(value) for value in multipliers]
    return {
        "multipliers": [item[0] for item in fixed],
        "shifts": [item[1] for item in fixed],
        "count": len(fixed),
        "zero_point": int(req.get("output_zero_point", layer.get("output_zero_point", 0))),
    }


def unpack_conv_params(registers):
    kernel = parse_int(registers.get("CONV_KERNEL"))
    stride_pad = parse_int(registers.get("CONV_STRIDE_PAD"))
    dilation = parse_int(registers.get("CONV_DILATION"))
    return {
        "kernel_h": kernel & 0xFFFF,
        "kernel_w": (kernel >> 16) & 0xFFFF,
        "stride_h": stride_pad & 0xFF,
        "stride_w": (stride_pad >> 8) & 0xFF,
        "pad_h": (stride_pad >> 16) & 0xFF,
        "pad_w": (stride_pad >> 24) & 0xFF,
        "dilation_h": dilation & 0xFF,
        "dilation_w": (dilation >> 8) & 0xFF,
    }


def cpu_params(layer):
    op = layer["op"]
    if op == "maxpool2d":
        kernel_h, kernel_w = layer.get("kernel", [1, 1])
        stride_h, stride_w = layer.get("stride", layer.get("kernel", [1, 1]))
        pad_h, pad_w = layer.get("padding", [0, 0])
        dilation_h, dilation_w = layer.get("dilation", [1, 1])
        return {
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "dilation_h": dilation_h,
            "dilation_w": dilation_w,
        }
    return {
        "kernel_h": 0,
        "kernel_w": 0,
        "stride_h": 0,
        "stride_w": 0,
        "pad_h": 0,
        "pad_w": 0,
        "dilation_h": 0,
        "dilation_w": 0,
    }


def layer_init(layer, requant_names):
    name = layer["name"]
    op = layer["op"]
    exec_mode = layer["exec"]
    registers = layer.get("registers", {})
    assets = layer.get("assets", {})
    params = unpack_conv_params(registers) if op == "conv2d" else cpu_params(layer)
    rq = requant_names.get(name)

    if op == "linear":
        w_addr = parse_int(assets.get("weight_addr"))
        w_bytes = parse_int(assets.get("weight_bytes"))
    else:
        w_addr = parse_int(assets.get("w_addr"))
        w_bytes = parse_int(assets.get("w_bytes"))

    bias_addr = parse_int(assets.get("bias_addr"))
    bias_bytes = parse_int(assets.get("bias_bytes"))
    rq_mult = rq["mult_name"] if rq else "0"
    rq_shift = rq["shift_name"] if rq else "0"
    rq_count = rq["count"] if rq else 0
    out_zp = rq["zero_point"] if rq else int(layer.get("output_zero_point", 0))

    return f"""    {{
        .name = \"{name}\",
        .op = {OP_ENUM[op]},
        .exec = {EXEC_ENUM[exec_mode]},
        .input = {shape_expr(layer["input_shape"])},
        .output = {shape_expr(layer["output_shape"])},
        .kernel_h = {int(params["kernel_h"])}u,
        .kernel_w = {int(params["kernel_w"])}u,
        .stride_h = {int(params["stride_h"])}u,
        .stride_w = {int(params["stride_w"])}u,
        .pad_h = {int(params["pad_h"])}u,
        .pad_w = {int(params["pad_w"])}u,
        .dilation_h = {int(params["dilation_h"])}u,
        .dilation_w = {int(params["dilation_w"])}u,
        .m_dim = {parse_int(registers.get("M_DIM"))}u,
        .n_dim = {parse_int(registers.get("N_DIM"))}u,
        .k_dim = {parse_int(registers.get("K_DIM"))}u,
        .w_addr = 0x{w_addr:08x}u,
        .bias_addr = 0x{bias_addr:08x}u,
        .w_bytes = {w_bytes}u,
        .bias_bytes = {bias_bytes}u,
        .ctrl = 0x{parse_int(registers.get("CTRL")):08x}u,
        .conv_ifm_shape = 0x{parse_int(registers.get("CONV_IFM_SHAPE")):08x}u,
        .conv_channels = 0x{parse_int(registers.get("CONV_CHANNELS")):08x}u,
        .conv_kernel = 0x{parse_int(registers.get("CONV_KERNEL")):08x}u,
        .conv_out_shape = 0x{parse_int(registers.get("CONV_OUT_SHAPE")):08x}u,
        .conv_stride_pad = 0x{parse_int(registers.get("CONV_STRIDE_PAD")):08x}u,
        .conv_dilation = 0x{parse_int(registers.get("CONV_DILATION")):08x}u,
        .quant_cfg = 0x{parse_int(registers.get("QUANT_CFG")):08x}u,
        .requant_multiplier = {rq_mult},
        .requant_shift = {rq_shift},
        .requant_count = {rq_count}u,
        .output_zero_point = {out_zp},
    }}"""


def generate_header(plan, plan_path):
    array_text = []
    requant_names = {}
    for layer in plan["layers"]:
        rq = layer_requant_info(layer)
        if rq is None:
            continue
        ident = safe_ident(layer["name"])
        mult_name = f"g_{ident}_requant_multiplier"
        shift_name = f"g_{ident}_requant_shift"
        array_text.append(format_array(rq["multipliers"], "int32_t", mult_name, per_line=6))
        array_text.append(format_array(rq["shifts"], "uint8_t", shift_name, per_line=16))
        requant_names[layer["name"]] = {
            "mult_name": mult_name,
            "shift_name": shift_name,
            "count": rq["count"],
            "zero_point": rq["zero_point"],
        }

    asset_base = parse_int(plan.get("asset_base_addr"))
    asset_end = parse_int(plan.get("asset_end_addr"))
    conv_count = sum(1 for layer in plan["layers"] if layer["op"] == "conv2d")
    layer_inits = ",\n".join(layer_init(layer, requant_names) for layer in plan["layers"])

    return f"""#ifndef MODEL_PLAN_GENERATED_H
#define MODEL_PLAN_GENERATED_H

/* Generated from {plan_path.as_posix()}. Do not edit by hand. */

#include \"npu_pth_runtime.h\"

#define NPU_MODEL_NAME \"{plan["name"]}\"
#define NPU_MODEL_LAYER_COUNT {len(plan["layers"])}u
#define NPU_MODEL_CONV_COUNT {conv_count}u
#define NPU_MODEL_ASSET_BASE 0x{asset_base:08x}u
#define NPU_MODEL_ASSET_END 0x{asset_end:08x}u
#define NPU_MODEL_ASSET_BYTES {asset_end - asset_base}u
#define NPU_MODEL_INPUT_SHAPE {shape_expr(plan["input"]["shape"])}
#define NPU_MODEL_OUTPUT_SHAPE {shape_expr(plan["output_shape"])}

{''.join(array_text)}static const npu_pth_layer_t g_npu_model_layers[NPU_MODEL_LAYER_COUNT] = {{
{layer_inits}
}};

#endif
"""


def memory_report(plan, dram_base, dram_words):
    asset_base = parse_int(plan.get("asset_base_addr"))
    asset_end = parse_int(plan.get("asset_end_addr"))
    dram_bytes = dram_words * 4
    dram_end = dram_base + dram_bytes

    rank4_shapes = [layer["output_shape"] for layer in plan["layers"] if len(layer["output_shape"]) == 4]
    max_ifm_bytes = max([product(shape) for shape in rank4_shapes] + [product(plan["input"]["shape"])])
    max_acc_bytes = max(
        [product(layer["output_shape"]) * 4 for layer in plan["layers"] if layer["op"] == "conv2d"]
        + [0]
    )
    scratch_estimate = max_ifm_bytes * 2 + max_acc_bytes

    return {
        "asset_base": f"0x{asset_base:08x}",
        "asset_end": f"0x{asset_end:08x}",
        "asset_bytes": asset_end - asset_base,
        "current_dram_base": f"0x{dram_base:08x}",
        "current_dram_end": f"0x{dram_end:08x}",
        "current_dram_bytes": dram_bytes,
        "assets_fit_current_dram": asset_base >= dram_base and asset_end <= dram_end,
        "max_int8_feature_bytes": max_ifm_bytes,
        "max_int32_accumulator_bytes": max_acc_bytes,
        "rough_double_buffer_plus_acc_bytes": scratch_estimate,
        "rough_total_with_assets_bytes": (asset_end - asset_base) + scratch_estimate,
    }


def generate_readme(plan, report):
    fit_text = "yes" if report["assets_fit_current_dram"] else "no"
    return f"""# Generated CPU/NPU Runtime Assets

Model: `{plan["name"]}`

Generated files:

- `model_plan_generated.h`: layer descriptors and fixed-point requant constants.
- `npu_pth_runtime.h`: header-only reference CPU runtime helpers.
- `runtime_smoke.c`: include/compile smoke file.
- `runtime_summary.json`: memory fit and scratch estimates.

Current memory fit:

```text
assets bytes             : {report["asset_bytes"]}
current DRAM bytes        : {report["current_dram_bytes"]}
assets fit current DRAM   : {fit_text}
max int8 feature bytes    : {report["max_int8_feature_bytes"]}
max int32 accumulator bytes: {report["max_int32_accumulator_bytes"]}
rough total bytes         : {report["rough_total_with_assets_bytes"]}
```

For the RepOpt VGG probe this is expected not to fit the current `tb_soc.v`
default DRAM. The generated C still defines the CPU/NPU schedule that a larger
memory map or a smaller model can use.
"""


def generate_smoke():
    return """#include \"model_plan_generated.h\"

int main(void)
{
    return (NPU_MODEL_LAYER_COUNT == 0u) ? 1 : 0;
}
"""


def convert(args):
    plan_path = Path(args.plan).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = load_json(plan_path)
    if plan.get("schema") != "npu_pth_plan_v1":
        raise ValueError(f"unsupported plan schema: {plan.get('schema')}")

    runtime_src = Path(__file__).resolve().parent / "runtime" / "npu_pth_runtime.h"
    shutil.copyfile(runtime_src, out_dir / "npu_pth_runtime.h")

    report = memory_report(plan, int(args.dram_base, 0), int(args.dram_words, 0))
    write_text(out_dir / "model_plan_generated.h", generate_header(plan, plan_path))
    write_text(out_dir / "runtime_smoke.c", generate_smoke())
    write_text(out_dir / "README.md", generate_readme(plan, report))
    with open(out_dir / "runtime_summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)

    print(f"generated: {out_dir}")
    print(f"layers: {len(plan['layers'])}")
    print(f"assets_fit_current_dram: {report['assets_fit_current_dram']}")
    print(f"rough_total_with_assets_bytes: {report['rough_total_with_assets_bytes']}")


def main():
    parser = argparse.ArgumentParser(description="Generate reference CPU runtime descriptors from model_plan.json")
    parser.add_argument("--plan", required=True, help="Path to model_plan.json")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated C files")
    parser.add_argument("--dram-base", default="0x1000", help="Current SoC DRAM CPU-visible base address")
    parser.add_argument("--dram-words", default="15360", help="Current tb_soc/soc_top DRAM_WORDS")
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
