#!/usr/bin/env python3
# =============================================================================
# run_repopt_vgg_host.py - Host-side CPU/NPU-split interpreter for RepOpt VGG.
#
# This does not use PyTorch quantized conv kernels because the checkpoint uses
# qint8 activations while the stock CPU quantized conv backend expects quint8.
# Instead, it interprets model_plan.json with the same V1 split used by the
# reference CPU path:
#   NPU role: Conv2D int8*int8 -> int32 accumulator, bias, ReLU
#   CPU role: per-channel requant, NCHW repack, MaxPool/AvgPool/Flatten/Linear
# =============================================================================

import argparse
import json
import pickle
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F


CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tensor_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in obj:
                return obj[key]
    return obj


def load_hex_words(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                words.append(int(line, 16))
    return words


def signed32(word):
    word &= 0xFFFFFFFF
    return word - 0x100000000 if word & 0x80000000 else word


def int8_from_word(word):
    value = word & 0xFF
    return value - 0x100 if value & 0x80 else value


def load_int32_hex(path):
    return torch.tensor([signed32(word) for word in load_hex_words(path)], dtype=torch.float64)


def load_int8_hex(path, count):
    vals = []
    for word in load_hex_words(path):
        for shift in (0, 8, 16, 24):
            vals.append(int8_from_word(word >> shift))
            if len(vals) == count:
                return torch.tensor(vals, dtype=torch.float64)
    return torch.tensor(vals, dtype=torch.float64)


def load_cifar_sample(data_root, index):
    batch_path = Path(data_root) / "cifar-10-batches-py" / "test_batch"
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    raw = torch.tensor(batch["data"][index], dtype=torch.float64).reshape(3, 32, 32) / 255.0
    label = int(batch["labels"][index])
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float64).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float64).view(3, 1, 1)
    return ((raw - mean) / std).unsqueeze(0), label


def normalize_cifar10_image_tensor(raw_chw):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float64).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float64).view(3, 1, 1)
    return ((raw_chw.to(torch.float64) - mean) / std).unsqueeze(0)


def load_image_input(image_path, image_size):
    try:
        from PIL import Image
    except Exception as exc:
        raise SystemExit(
            "Pillow is required for --image input.\n"
            "Install it with:\n"
            "  python -m pip install pillow\n"
            f"Import error: {exc}"
        ) from exc

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.BILINEAR)
        pixels = torch.tensor(list(img.getdata()), dtype=torch.float64).view(image_size, image_size, 3)
    raw_chw = pixels.permute(2, 0, 1).contiguous() / 255.0
    return normalize_cifar10_image_tensor(raw_chw)


def quantize_qint8(x, scale, zero_point):
    q = torch.round(x / float(scale)) + int(zero_point)
    return torch.clamp(q, -128, 127).to(torch.float64)


def requant_qint8(acc, multipliers, output_zero_point):
    mult = torch.tensor(multipliers, dtype=torch.float64).view(1, -1, 1, 1)
    q = torch.round(acc * mult) + int(output_zero_point)
    return torch.clamp(q, -128, 127).to(torch.float64)


def conv2d_acc_npu(x_q, qweight, bias_acc, layer):
    regs = layer["registers"]
    stride_pad = int(regs["CONV_STRIDE_PAD"])
    dilation_word = int(regs["CONV_DILATION"])
    stride = (stride_pad & 0xFF, (stride_pad >> 8) & 0xFF)
    padding = ((stride_pad >> 16) & 0xFF, (stride_pad >> 24) & 0xFF)
    dilation = (dilation_word & 0xFF, (dilation_word >> 8) & 0xFF)

    w = qweight.int_repr().to(torch.float64)
    bias = bias_acc.to(torch.float64)
    acc = F.conv2d(x_q.to(torch.float64), w, bias=bias, stride=stride, padding=padding, dilation=dilation)
    if layer.get("activation") == "relu":
        acc = torch.clamp(acc, min=0)
    elif layer.get("activation") == "relu6":
        acc = torch.clamp(acc, min=0, max=6)
    return acc


def conv2d_raw_tile4_cpu_scheduler(x_q, qweight, layer):
    regs = layer["registers"]
    stride_pad = int(regs["CONV_STRIDE_PAD"])
    dilation_word = int(regs["CONV_DILATION"])
    stride = (stride_pad & 0xFF, (stride_pad >> 8) & 0xFF)
    padding = ((stride_pad >> 16) & 0xFF, (stride_pad >> 24) & 0xFF)
    dilation = (dilation_word & 0xFF, (dilation_word >> 8) & 0xFF)
    kh, kw = layer["weight_shape"][2], layer["weight_shape"][3]
    batch, cout, oh, ow = layer["output_shape"]

    unfolded = F.unfold(
        x_q.to(torch.float64),
        kernel_size=(kh, kw),
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    a_mat = unfolded.transpose(1, 2).reshape(batch * oh * ow, -1).contiguous()
    w_mat = qweight.int_repr().to(torch.float64).reshape(cout, -1).t().contiguous()

    m_dim, k_dim = a_mat.shape
    n_dim = w_mat.shape[1]
    acc_mat = torch.zeros((m_dim, n_dim), dtype=torch.float64)
    tile_count = 0
    for m_base in range(0, m_dim, 4):
        m_end = min(m_base + 4, m_dim)
        a_tile = a_mat[m_base:m_end, :]
        for n_base in range(0, n_dim, 4):
            n_end = min(n_base + 4, n_dim)
            acc_mat[m_base:m_end, n_base:n_end] = a_tile @ w_mat[:, n_base:n_end]
            tile_count += 1

    acc = acc_mat.reshape(batch, oh, ow, cout).permute(0, 3, 1, 2).contiguous()
    return acc, {"tile_m": 4, "tile_n": 4, "tile_count": tile_count, "k_dim": int(k_dim)}


def apply_cpu_conv_postprocess(acc, bias_acc, layer):
    acc = acc + bias_acc.to(torch.float64).view(1, -1, 1, 1)
    if layer.get("activation") == "relu":
        acc = torch.clamp(acc, min=0)
    elif layer.get("activation") == "relu6":
        acc = torch.clamp(acc, min=0, max=6)
    return acc


def maxpool2d_cpu(x_q, layer):
    kh, kw = layer["kernel"]
    sh, sw = layer["stride"]
    ph, pw = layer["padding"]
    return F.max_pool2d(x_q.to(torch.float32), kernel_size=(kh, kw), stride=(sh, sw), padding=(ph, pw)).to(torch.float64)


def adaptive_avgpool2d_cpu(x_q, layer):
    out_h, out_w = layer["output_size"]
    pooled = F.adaptive_avg_pool2d(x_q.to(torch.float64), output_size=(out_h, out_w))
    return torch.clamp(torch.round(pooled), -128, 127).to(torch.float64)


def linear_cpu(x_q, state_dict, layer, out_dir):
    packed = state_dict[layer["weight_key"]]
    qweight, _bias_float = packed[0], packed[1]
    out_features, in_features = qweight.int_repr().shape
    x_flat = x_q.reshape(x_q.shape[0], -1).to(torch.float64)
    if x_flat.shape[1] != in_features:
        raise ValueError(f"linear input features {x_flat.shape[1]} != {in_features}")

    w = qweight.int_repr().to(torch.float64)
    bias = load_int32_hex(out_dir / layer["assets"]["bias_int32_hex"]).to(torch.float64)
    acc = x_flat @ w.t()
    acc = acc + bias.view(1, out_features)

    req = layer["cpu_requant"]
    multipliers = torch.tensor(req["multipliers"], dtype=torch.float64).view(1, out_features)
    q = torch.round(acc * multipliers) + int(req["output_zero_point"])
    return torch.clamp(q, -128, 127).to(torch.float64)


def run(args):
    plan_path = Path(args.plan).resolve()
    out_dir = plan_path.parent
    plan = load_json(plan_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
        checkpoint = torch.load(args.pth, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)

    if args.image:
        x_float = load_image_input(args.image, args.image_size)
        true_label = None
        input_source = {
            "kind": "image",
            "path": str(Path(args.image).resolve()),
            "preprocess": f"RGB resize to {args.image_size}x{args.image_size}, ToTensor, CIFAR-10 Normalize",
        }
    else:
        x_float, true_label = load_cifar_sample(args.data_root, args.index)
        input_source = {
            "kind": "cifar10_test_batch",
            "data_root": str(Path(args.data_root).resolve()),
            "sample_index": args.index,
        }
    in_scale = float(tensor_scalar(state_dict[plan["input"]["scale_key"]]))
    in_zp = int(tensor_scalar(state_dict[plan["input"]["zero_point_key"]]))
    x_q = quantize_qint8(x_float, in_scale, in_zp)

    trace = []
    current = x_q
    for layer in plan["layers"]:
        if layer["op"] == "conv2d":
            bias_acc = load_int32_hex(out_dir / layer["assets"]["bias_int32_hex"])
            qweight = state_dict[layer["weight_key"]]
            tile_info = None
            if args.conv_backend == "direct":
                acc = conv2d_acc_npu(current, qweight, bias_acc, layer)
                conv_exec = "npu_acc_then_cpu_requant"
            elif args.conv_backend == "tile4":
                raw_acc, tile_info = conv2d_raw_tile4_cpu_scheduler(current, qweight, layer)
                acc = apply_cpu_conv_postprocess(raw_acc, bias_acc, layer)
                conv_exec = "tile4_raw_acc_then_cpu_bias_activation_requant"
            else:
                raise ValueError(f"unsupported conv backend: {args.conv_backend}")
            req = layer["cpu_requant_after_npu"]
            current = requant_qint8(acc, req["multipliers"], req["output_zero_point"])
            trace_item = {
                "name": layer["name"],
                "op": "conv2d",
                "exec": conv_exec,
                "acc_shape": list(acc.shape),
                "out_shape": list(current.shape),
                "acc_min": float(acc.min().item()),
                "acc_max": float(acc.max().item()),
                "q_min": int(current.min().item()),
                "q_max": int(current.max().item()),
            }
            if tile_info:
                trace_item["tile_scheduler"] = tile_info
            trace.append(trace_item)
        elif layer["op"] == "maxpool2d":
            current = maxpool2d_cpu(current, layer)
            trace.append({"name": layer["name"], "op": "maxpool2d", "exec": "cpu", "out_shape": list(current.shape)})
        elif layer["op"] == "adaptive_avgpool2d":
            current = adaptive_avgpool2d_cpu(current, layer)
            trace.append(
                {"name": layer["name"], "op": "adaptive_avgpool2d", "exec": "cpu", "out_shape": list(current.shape)}
            )
        elif layer["op"] == "flatten":
            current = current.reshape(current.shape[0], -1)
            trace.append({"name": layer["name"], "op": "flatten", "exec": "cpu", "out_shape": list(current.shape)})
        elif layer["op"] == "linear":
            current = linear_cpu(current, state_dict, layer, out_dir)
            trace.append(
                {
                    "name": layer["name"],
                    "op": "linear",
                    "exec": "cpu",
                    "out_shape": list(current.shape),
                    "q_min": int(current.min().item()),
                    "q_max": int(current.max().item()),
                }
            )
        else:
            raise ValueError(f"unsupported op: {layer['op']}")

    logits_int8 = [int(v) for v in current.reshape(-1).tolist()]
    classifier_layer = plan["layers"][-1]
    out_scale = float(classifier_layer["cpu_requant"]["output_scale"])
    out_zp = int(classifier_layer["cpu_requant"]["output_zero_point"])
    logits_dequant = [(v - out_zp) * out_scale for v in logits_int8]
    pred = max(range(len(logits_dequant)), key=lambda idx: logits_dequant[idx])

    result = {
        "source_pth": str(Path(args.pth).resolve()),
        "plan": str(plan_path),
        "input_source": input_source,
        "conv_backend": args.conv_backend,
        "true_label": true_label,
        "true_class": CLASSES[true_label] if true_label is not None else None,
        "pred_label": pred,
        "pred_class": CLASSES[pred],
        "logits_int8": logits_int8,
        "logits_dequant": logits_dequant,
        "input_quant": {
            "scale": in_scale,
            "zero_point": in_zp,
            "q_min": int(x_q.min().item()),
            "q_max": int(x_q.max().item()),
        },
        "trace": trace,
        "note": (
            "Host-side interpreter for the planned CPU/NPU split. It is the golden "
            "for staged RTL/SoC validation, not a stock PyTorch quantized backend run."
        ),
    }

    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)

    print(f"RepOpt host CPU/NPU-split run written to: {out_path}")
    if true_label is None:
        print(f"input_image={Path(args.image).resolve()} pred={pred}:{CLASSES[pred]} backend={args.conv_backend}")
    else:
        print(f"sample_index={args.index} true={true_label}:{CLASSES[true_label]} pred={pred}:{CLASSES[pred]} backend={args.conv_backend}")
    print("logits_int8=" + " ".join(str(v) for v in logits_int8))
    print("logits_dequant=" + " ".join(f"{v:.6f}" for v in logits_dequant))


def main():
    parser = argparse.ArgumentParser(description="Run RepOpt VGG with the host CPU/NPU-split interpreter")
    parser.add_argument(
        "--pth",
        default=".06_RepOpt_VGG/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth",
        help="Path to qat_int8_quantized.pth",
    )
    parser.add_argument("--plan", default="sim/pth_repopt_probe/model_plan.json", help="Path to model_plan.json")
    parser.add_argument("--data-root", default=".06_RepOpt_VGG/06_RepOpt_VGG/data", help="CIFAR-10 data root")
    parser.add_argument("--index", type=int, default=0, help="CIFAR-10 test sample index")
    parser.add_argument("--image", default="", help="Optional RGB image path. If set, --index is ignored.")
    parser.add_argument("--image-size", type=int, default=32, help="Image resize size; RepOpt CIFAR-10 expects 32")
    parser.add_argument(
        "--conv-backend",
        choices=["direct", "tile4"],
        default="direct",
        help="Conv execution model: direct uses full F.conv2d accumulator; tile4 uses a 4x4 Conv-as-GEMM tile scheduler plus CPU postprocess.",
    )
    parser.add_argument("--out-json", default="sim/pth_repopt_host_run/host_run.json", help="Output JSON report")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
