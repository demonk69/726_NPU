#!/usr/bin/env python3
# =============================================================================
# pth_to_npu_assets.py - Convert a constrained PyTorch state_dict checkpoint
# into NPU-facing Conv2D assets and a CPU-scheduled inference plan.
#
# This is a host-side tool. It is not intended to run on the reference CPU.
# =============================================================================

import argparse
import json
import math
import re
import sys
import warnings
from collections import OrderedDict
from pathlib import Path


def require_torch():
    try:
        import torch  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - depends on host env
        raise SystemExit(
            "PyTorch is required on the host to read .pth files.\n"
            "Install CPU PyTorch with:\n"
            "  python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu\n"
            f"Import error: {exc}"
        ) from exc
    return torch


def align(value, alignment):
    return (value + alignment - 1) & ~(alignment - 1)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def int8_pack(vals):
    words = []
    for i in range(0, len(vals), 4):
        word = 0
        for lane in range(4):
            if i + lane < len(vals):
                word |= (int(vals[i + lane]) & 0xFF) << (8 * lane)
        words.append(word & 0xFFFFFFFF)
    return words


def write_hex(path, words):
    with open(path, "w", encoding="utf-8") as f:
        for word in words:
            f.write(f"{word & 0xFFFFFFFF:08x}\n")


def output_dim(input_size, kernel, stride, pad, dilation):
    numerator = input_size + 2 * pad - dilation * (kernel - 1) - 1
    if numerator < 0:
        raise ValueError("invalid Conv2D shape")
    return numerator // stride + 1


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in obj:
                return obj[key], list(obj.keys())
    return obj, list(obj.keys()) if isinstance(obj, dict) else []


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tensor_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def tensor_summary(torch, value):
    if not torch.is_tensor(value):
        if isinstance(value, (tuple, list)):
            return {
                "type": type(value).__name__,
                "len": len(value),
                "elem_types": [type(item).__name__ for item in value[:4]],
            }
        return {"type": type(value).__name__, "repr": repr(value)[:120]}

    item = {
        "type": "Tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "quantized": bool(value.is_quantized),
    }
    if value.is_quantized:
        item["qscheme"] = str(value.qscheme())
        if value.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
            item["q_axis"] = int(value.q_per_channel_axis())
            item["q_scales_shape"] = list(value.q_per_channel_scales().shape)
            item["q_zero_points_shape"] = list(value.q_per_channel_zero_points().shape)
        else:
            item["q_scale"] = float(value.q_scale())
            item["q_zero_point"] = int(value.q_zero_point())
    return item


def get_state_value(state_dict, key, required=True):
    if key in state_dict:
        return state_dict[key]
    if required:
        raise KeyError(f"missing checkpoint key: {key}")
    return None


def get_scale_value(state_dict, key, default=None):
    value = get_state_value(state_dict, key, required=default is None)
    if value is None:
        return default
    return float(tensor_scalar(value))


def get_zero_point_value(state_dict, key, default=0):
    value = get_state_value(state_dict, key, required=False)
    if value is None:
        return default
    return int(tensor_scalar(value))


def qweight_to_int8_and_scales(torch, qweight):
    if not torch.is_tensor(qweight):
        raise TypeError("weight is not a tensor")
    if not qweight.is_quantized:
        raise TypeError("V1 converter expects quantized qint8 weights")
    if qweight.qscheme() not in (torch.per_channel_affine, torch.per_channel_symmetric):
        raise TypeError(f"V1 converter expects per-channel quantized weights, got {qweight.qscheme()}")
    if qweight.q_per_channel_axis() != 0:
        raise TypeError("per-channel quantization axis must be output-channel/axis 0")

    int_tensor = qweight.int_repr().cpu()
    scales = [float(x) for x in qweight.q_per_channel_scales().cpu().tolist()]
    zero_points = [int(x) for x in qweight.q_per_channel_zero_points().cpu().tolist()]
    return int_tensor, scales, zero_points


def conv_w_col_words(weight_int):
    shape = tuple(weight_int.shape)
    if len(shape) != 4:
        raise ValueError(f"Conv2d weight must have rank 4, got {shape}")
    cout, cin, kh, kw = shape
    arr = weight_int.cpu().numpy().astype("int16", copy=False)
    words_per_channel = (cin * kh * kw + 3) // 4
    all_words = []
    for out_c in range(cout):
        flat = arr[out_c].reshape(-1)
        padded_len = words_per_channel * 4
        if flat.size < padded_len:
            import numpy as np

            flat = np.pad(flat, (0, padded_len - flat.size), mode="constant")
        lanes = (flat.astype("uint8", copy=False).reshape(-1, 4).astype("uint32", copy=False))
        words = lanes[:, 0] | (lanes[:, 1] << 8) | (lanes[:, 2] << 16) | (lanes[:, 3] << 24)
        all_words.extend(int(word) for word in words.tolist())
    return all_words


def linear_weight_words(weight_int):
    shape = tuple(weight_int.shape)
    if len(shape) != 2:
        raise ValueError(f"Linear weight must have rank 2, got {shape}")
    flat = weight_int.cpu().reshape(-1).tolist()
    return int8_pack(flat)


def linear_prefix_from_packed_key(weight_key):
    suffix = "._packed_params._packed_params"
    if weight_key.endswith(suffix):
        return weight_key[: -len(suffix)]
    return weight_key.rsplit(".", 1)[0]


def bias_to_int32(torch, bias_value, input_scale, weight_scales, cout):
    if bias_value is None:
        return [0 for _ in range(cout)]
    if not torch.is_tensor(bias_value):
        raise TypeError("bias is not a tensor")
    vals = [float(x) for x in bias_value.detach().cpu().reshape(-1).tolist()]
    if len(vals) != cout:
        raise ValueError(f"bias length {len(vals)} does not match Cout {cout}")
    out = []
    for idx, value in enumerate(vals):
        denom = input_scale * weight_scales[idx]
        if denom == 0.0:
            raise ValueError("zero input/weight scale while converting bias")
        out.append(int(round(value / denom)))
    return out


def quant_fit_summary(requant_multipliers, tolerance=1e-6):
    if not requant_multipliers:
        return {"single_layer_quant_exact": False, "reason": "empty multipliers"}
    first = requant_multipliers[0]
    max_delta = max(abs(x - first) for x in requant_multipliers)
    return {
        "single_layer_quant_exact": max_delta <= tolerance,
        "first_multiplier": first,
        "min_multiplier": min(requant_multipliers),
        "max_multiplier": max(requant_multipliers),
        "max_delta_from_first": max_delta,
        "reason": "current NPU QUANT_CFG is one scale/shift per layer; this checkpoint uses per-channel weight scales",
    }


def make_ctrl(mode, activation, bias):
    ctrl = 0x1
    if mode == "OS":
        ctrl |= 0x10
    ctrl |= 0x100  # direct Conv2D on-the-fly im2col
    if bias:
        ctrl |= 0x200
    if activation == "relu":
        ctrl |= 0x400
    elif activation == "relu6":
        ctrl |= 0x800
    elif activation != "none":
        raise ValueError(f"unsupported activation: {activation}")
    return ctrl


def conv_registers(in_shape, out_shape, layer, ctrl, addresses):
    batch, cin, ih, iw = in_shape
    _, cout, oh, ow = out_shape
    kh, kw = layer["kernel"]
    stride_h, stride_w = layer.get("stride", [1, 1])
    pad_h, pad_w = layer.get("padding", [0, 0])
    dilation_h, dilation_w = layer.get("dilation", [1, 1])
    return {
        "M_DIM": batch * oh * ow,
        "N_DIM": cout,
        "K_DIM": cin * kh * kw,
        "W_ADDR": addresses["w_addr"],
        "A_ADDR": "runtime_current_ifm",
        "R_ADDR": "runtime_int32_ofm",
        "BIAS_ADDR": addresses["bias_addr"],
        "CTRL": ctrl,
        "CONV_IFM_SHAPE": (iw << 16) | ih,
        "CONV_CHANNELS": (batch << 16) | cin,
        "CONV_KERNEL": (kw << 16) | kh,
        "CONV_OUT_SHAPE": (ow << 16) | oh,
        "CONV_STRIDE_PAD": (pad_w << 24) | (pad_h << 16) | (stride_w << 8) | stride_h,
        "CONV_DILATION": (dilation_w << 8) | dilation_h,
        "QUANT_CFG": 0x00010000,
    }


def process_conv(torch, state_dict, layer, in_shape, out_dir, asset_dir, address_cursor, mode):
    weight_key = layer["weight"]
    qweight = get_state_value(state_dict, weight_key)
    weight_int, weight_scales, weight_zps = qweight_to_int8_and_scales(torch, qweight)
    cout, cin, kh, kw = [int(x) for x in weight_int.shape]
    batch, in_c, ih, iw = in_shape
    if in_c != cin:
        raise ValueError(f"{layer['name']}: spec Cin {in_c} does not match weight Cin {cin}")

    stride_h, stride_w = layer.get("stride", [1, 1])
    pad_h, pad_w = layer.get("padding", [0, 0])
    dilation_h, dilation_w = layer.get("dilation", [1, 1])
    oh = output_dim(ih, kh, stride_h, pad_h, dilation_h)
    ow = output_dim(iw, kw, stride_w, pad_w, dilation_w)
    out_shape = [batch, cout, oh, ow]

    input_scale = float(layer["input_scale"])
    input_zero_point = int(layer.get("input_zero_point", 0))
    output_scale = get_scale_value(state_dict, layer.get("output_scale_key", f"{weight_key[:-7]}.scale"))
    output_zero_point = get_zero_point_value(state_dict, layer.get("output_zero_point_key", f"{weight_key[:-7]}.zero_point"))

    bias_key = layer.get("bias")
    bias_value = get_state_value(state_dict, bias_key, required=False) if bias_key else None
    bias_int32 = bias_to_int32(torch, bias_value, input_scale, weight_scales, cout)

    all_words = conv_w_col_words(weight_int)
    layer_slug = safe_name(layer["name"])
    w_path = asset_dir / f"{layer_slug}_w_col.hex"
    bias_path = asset_dir / f"{layer_slug}_bias_int32.hex"
    write_hex(w_path, all_words)
    write_hex(bias_path, [x & 0xFFFFFFFF for x in bias_int32])

    w_bytes = len(all_words) * 4
    bias_bytes = len(bias_int32) * 4
    w_addr = address_cursor
    bias_addr = align(w_addr + w_bytes, 0x100)
    next_addr = align(bias_addr + bias_bytes, 0x100)

    requant_multipliers = [
        (input_scale * weight_scales[out_c]) / output_scale
        for out_c in range(cout)
    ]
    quant_fit = quant_fit_summary(requant_multipliers)
    activation = layer.get("activation", "none")
    ctrl = make_ctrl(mode, activation, bias=bool(bias_key))
    registers = conv_registers(
        in_shape,
        out_shape,
        {"kernel": [kh, kw], **layer},
        ctrl,
        {"w_addr": w_addr, "bias_addr": bias_addr},
    )

    plan_layer = {
        "name": layer["name"],
        "op": "conv2d",
        "exec": "npu_direct",
        "input_shape": list(in_shape),
        "output_shape": out_shape,
        "weight_key": weight_key,
        "bias_key": bias_key,
        "weight_shape": [cout, cin, kh, kw],
        "weight_qscheme": str(qweight.qscheme()),
        "weight_scale_mode": "per_output_channel",
        "weight_zero_points_unique": sorted(set(weight_zps)),
        "activation": activation,
        "npu_post_quant": {
            "enabled": False,
            "reason": "exact PyTorch int8 requant is per-output-channel; current NPU QUANT_CFG is per-layer",
        },
        "cpu_requant_after_npu": {
            "enabled": True,
            "input_scale": input_scale,
            "input_zero_point": input_zero_point,
            "weight_scales": weight_scales,
            "output_scale": output_scale,
            "output_zero_point": output_zero_point,
            "multipliers": requant_multipliers,
            "fit_to_current_npu_quant": quant_fit,
        },
        "assets": {
            "w_col_hex": str(w_path.relative_to(out_dir)).replace("\\", "/"),
            "bias_int32_hex": str(bias_path.relative_to(out_dir)).replace("\\", "/"),
            "w_addr": f"0x{w_addr:08x}",
            "bias_addr": f"0x{bias_addr:08x}",
            "w_bytes": w_bytes,
            "bias_bytes": bias_bytes,
        },
        "registers": registers,
    }
    return plan_layer, out_shape, output_scale, output_zero_point, next_addr


def process_cpu_layer(torch, state_dict, layer, in_shape, out_dir, asset_dir, address_cursor, input_scale, input_zero_point):
    op = layer["op"]
    plan_layer = {
        "name": layer["name"],
        "op": op,
        "exec": "cpu",
        "input_shape": list(in_shape),
        "input_scale": input_scale,
        "input_zero_point": input_zero_point,
        "note": layer.get("note", "CPU runtime must implement this op for end-to-end inference"),
    }

    if op == "maxpool2d":
        batch, channels, ih, iw = in_shape
        kh, kw = layer["kernel"]
        stride_h, stride_w = layer.get("stride", layer["kernel"])
        pad_h, pad_w = layer.get("padding", [0, 0])
        dilation_h, dilation_w = layer.get("dilation", [1, 1])
        oh = output_dim(ih, kh, stride_h, pad_h, dilation_h)
        ow = output_dim(iw, kw, stride_w, pad_w, dilation_w)
        out_shape = [batch, channels, oh, ow]
        plan_layer["kernel"] = [kh, kw]
        plan_layer["stride"] = [stride_h, stride_w]
        plan_layer["padding"] = [pad_h, pad_w]
        plan_layer["dilation"] = [dilation_h, dilation_w]
        output_scale = input_scale
        output_zero_point = input_zero_point
    elif op == "adaptive_avgpool2d":
        batch, channels, _ih, _iw = in_shape
        oh, ow = layer["output_size"]
        out_shape = [batch, channels, oh, ow]
        plan_layer["output_size"] = [oh, ow]
        output_scale = input_scale
        output_zero_point = input_zero_point
    elif op == "flatten":
        batch, channels, ih, iw = in_shape
        start_dim = int(layer.get("start_dim", 1))
        if start_dim != 1:
            raise ValueError("V1 only supports flatten start_dim=1")
        out_shape = [batch, channels * ih * iw]
        plan_layer["start_dim"] = start_dim
        output_scale = input_scale
        output_zero_point = input_zero_point
    elif op == "linear":
        batch, features = in_shape
        weight_key = layer["weight"]
        packed = get_state_value(state_dict, weight_key)
        if not isinstance(packed, tuple) or len(packed) < 2:
            raise TypeError(f"{layer['name']}: expected packed quantized Linear tuple at {weight_key}")
        qweight, bias_value = packed[0], packed[1]
        weight_int, weight_scales, weight_zps = qweight_to_int8_and_scales(torch, qweight)
        out_features, in_features = [int(x) for x in weight_int.shape]
        if features != in_features:
            raise ValueError(f"{layer['name']}: input features {features} do not match Linear weight {in_features}")
        if int(layer.get("out_features", out_features)) != out_features:
            raise ValueError(f"{layer['name']}: spec out_features does not match Linear weight")

        prefix = linear_prefix_from_packed_key(weight_key)
        output_scale = get_scale_value(state_dict, layer.get("output_scale_key", f"{prefix}.scale"))
        output_zero_point = get_zero_point_value(state_dict, layer.get("output_zero_point_key", f"{prefix}.zero_point"))
        bias_int32 = bias_to_int32(torch, bias_value, input_scale, weight_scales, out_features)
        requant_multipliers = [
            (input_scale * weight_scales[out_c]) / output_scale
            for out_c in range(out_features)
        ]

        layer_slug = safe_name(layer["name"])
        w_path = asset_dir / f"{layer_slug}_linear_w_int8.hex"
        bias_path = asset_dir / f"{layer_slug}_linear_bias_int32.hex"
        weight_words = linear_weight_words(weight_int)
        write_hex(w_path, weight_words)
        write_hex(bias_path, [x & 0xFFFFFFFF for x in bias_int32])

        w_bytes = len(weight_words) * 4
        bias_bytes = len(bias_int32) * 4
        w_addr = address_cursor
        bias_addr = align(w_addr + w_bytes, 0x100)
        address_cursor = align(bias_addr + bias_bytes, 0x100)
        out_shape = [batch, out_features]
        plan_layer.update(
            {
                "weight_key": weight_key,
                "weight_shape": [out_features, in_features],
                "weight_qscheme": str(qweight.qscheme()),
                "weight_scale_mode": "per_output_channel",
                "weight_zero_points_unique": sorted(set(weight_zps)),
                "assets": {
                    "weight_int8_hex": str(w_path.relative_to(out_dir)).replace("\\", "/"),
                    "bias_int32_hex": str(bias_path.relative_to(out_dir)).replace("\\", "/"),
                    "weight_addr": f"0x{w_addr:08x}",
                    "bias_addr": f"0x{bias_addr:08x}",
                    "weight_bytes": w_bytes,
                    "bias_bytes": bias_bytes,
                },
                "cpu_requant": {
                    "enabled": True,
                    "input_scale": input_scale,
                    "input_zero_point": input_zero_point,
                    "weight_scales": weight_scales,
                    "output_scale": output_scale,
                    "output_zero_point": output_zero_point,
                    "multipliers": requant_multipliers,
                },
            }
        )
    else:
        raise ValueError(f"unsupported CPU op in V1: {op}")

    plan_layer["output_shape"] = out_shape
    plan_layer["output_scale"] = output_scale
    plan_layer["output_zero_point"] = output_zero_point
    return plan_layer, out_shape, output_scale, output_zero_point, address_cursor


def convert(args):
    torch = require_torch()
    spec = load_json(args.spec)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
        checkpoint = torch.load(args.pth, map_location="cpu", weights_only=False)
    state_dict, top_keys = unwrap_state_dict(checkpoint)
    if not hasattr(state_dict, "items"):
        raise TypeError("checkpoint does not contain a state_dict-like mapping")

    out_dir = Path(args.out_dir).resolve()
    asset_dir = out_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    inventory = OrderedDict()
    for key, value in state_dict.items():
        inventory[key] = tensor_summary(torch, value)

    input_info = spec["input"]
    current_shape = list(input_info["shape"])
    current_scale = float(input_info.get("scale", get_scale_value(state_dict, input_info.get("scale_key", "quant.scale"))))
    current_zero_point = int(input_info.get("zero_point", get_zero_point_value(state_dict, input_info.get("zero_point_key", "quant.zero_point"))))

    address_cursor = int(args.base_addr, 0)
    plan_layers = []
    plan_warnings = []
    for layer in spec["layers"]:
        op = layer["op"]
        if op == "conv2d":
            layer = dict(layer)
            layer["input_scale"] = current_scale
            layer["input_zero_point"] = current_zero_point
            plan_layer, current_shape, current_scale, current_zero_point, address_cursor = process_conv(
                torch,
                state_dict,
                layer,
                current_shape,
                out_dir,
                asset_dir,
                address_cursor,
                args.mode,
            )
            if not plan_layer["cpu_requant_after_npu"]["fit_to_current_npu_quant"]["single_layer_quant_exact"]:
                plan_warnings.append(
                    f"{layer['name']}: current NPU per-layer QUANT_CFG cannot exactly replace "
                    "PyTorch per-channel requant; CPU requant/repack is required after NPU Conv/ReLU."
                )
        else:
            plan_layer, current_shape, current_scale, current_zero_point, address_cursor = process_cpu_layer(
                torch,
                state_dict,
                layer,
                current_shape,
                out_dir,
                asset_dir,
                address_cursor,
                current_scale,
                current_zero_point,
            )
            # MaxPool preserves quant scale; AvgPool/Linear are CPU-defined in V1.
            if op in ("adaptive_avgpool2d", "linear"):
                if op == "linear" and "assets" in plan_layer:
                    plan_warnings.append(f"{layer['name']}: {op} assets exported, but execution remains CPU-side in V1.")
                else:
                    plan_warnings.append(f"{layer['name']}: {op} remains CPU-side in V1.")
        plan_layers.append(plan_layer)

    plan = {
        "schema": "npu_pth_plan_v1",
        "name": spec.get("name", Path(args.pth).stem),
        "source_pth": str(Path(args.pth).resolve()),
        "top_checkpoint_keys": top_keys,
        "mode": args.mode,
        "input": input_info,
        "output_shape": current_shape,
        "asset_base_addr": f"0x{int(args.base_addr, 0):08x}",
        "asset_end_addr": f"0x{address_cursor:08x}",
        "layers": plan_layers,
        "warnings": plan_warnings,
        "current_hardware_boundary": {
            "can_run_directly": False,
            "reason": (
                "This plan still requires CPU-side layer scheduling, pooling/linear execution, "
                "and per-channel requant/repack between Conv layers."
            ),
            "npu_roles": ["Conv2D on-the-fly im2col", "bias add", "ReLU/ReLU6"],
            "cpu_roles": ["MMIO scheduling", "per-channel requant", "NCHW repack", "MaxPool/AvgPool/Linear"],
        },
    }

    with open(out_dir / "model_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    with open(out_dir / "checkpoint_inventory.json", "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"name={plan['name']}\n")
        f.write(f"layers={len(plan_layers)}\n")
        f.write(f"conv_layers={sum(1 for layer in plan_layers if layer['op'] == 'conv2d')}\n")
        f.write(f"warnings={len(plan_warnings)}\n")
        for warning in plan_warnings:
            f.write(f"warning={warning}\n")

    print(f"Converted checkpoint: {args.pth}")
    print(f"  output       : {out_dir}")
    print(f"  layers       : {len(plan_layers)}")
    print(f"  conv layers  : {sum(1 for layer in plan_layers if layer['op'] == 'conv2d')}")
    print(f"  warnings     : {len(plan_warnings)}")
    for warning in plan_warnings[:8]:
        print(f"  warning: {warning}")
    if len(plan_warnings) > 8:
        print(f"  ... {len(plan_warnings) - 8} more warnings")


def main():
    parser = argparse.ArgumentParser(description="Convert constrained .pth checkpoints to NPU assets")
    parser.add_argument("--pth", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--spec", required=True, help="Path to model_spec.json")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["OS", "WS"], default="OS")
    parser.add_argument("--base-addr", default="0x00100000", help="Base byte address for generated weights")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
