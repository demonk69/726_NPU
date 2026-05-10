#!/usr/bin/env python3
# =============================================================================
# run_repopt_layer_tile_rtl.py - Schedule a small RepOpt Conv tile window on RTL.
#
# This is the first bridge from the host tile scheduler to actual RTL NPU
# execution. It invokes scripts/run_repopt_tile_case.ps1 per 4x4 tile, collects
# npu_output.hex, stitches a raw accumulator window, and runs CPU postprocess
# (bias/ReLU/per-channel requant) on that window.
# =============================================================================

import argparse
import json
import subprocess
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
    return [signed32(word) for word in load_hex_words(path)]


def find_layer(plan, name):
    for layer in plan["layers"]:
        if layer.get("name") == name and layer.get("op") == "conv2d":
            return layer
    raise ValueError(f"conv layer not found: {name}")


def run_tile(project_root, args, m_base, n_base):
    case_name = f"repopt_{args.layer_name}_rtlwin_m{m_base}_n{n_base}_idx{args.index}"
    script = project_root / "scripts" / "run_repopt_tile_case.ps1"
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Layer",
        args.layer_name,
        "-Index",
        str(args.index),
        "-MBase",
        str(m_base),
        "-NBase",
        str(n_base),
        "-Name",
        case_name,
        "-Pth",
        str(Path(args.pth)),
        "-Plan",
        str(Path(args.plan)),
        "-DataRoot",
        str(Path(args.data_root)),
        "-DumpResult",
    ]
    proc = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "RTL tile run failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    case_dir = project_root / "sim" / "pth_repopt_tile_cases" / case_name
    output_hex = case_dir / "npu_output.hex"
    if not output_hex.exists():
        raise FileNotFoundError(f"missing RTL output: {output_hex}")
    words = load_hex_words(output_hex)
    if len(words) != 16:
        raise ValueError(f"{output_hex} contains {len(words)} words, expected 16")
    return {
        "name": case_name,
        "m_base": m_base,
        "n_base": n_base,
        "case_dir": str(case_dir),
        "output_hex": str(output_hex),
        "stdout_summary": [line for line in proc.stdout.splitlines() if "PASS" in line or "Output:" in line],
        "raw_acc": [signed32(word) for word in words],
    }


def postprocess_window(raw_acc, layer, bias_int32, m_base, n_base):
    rows = len(raw_acc)
    cols = len(raw_acc[0]) if rows else 0
    multipliers = layer["cpu_requant_after_npu"]["multipliers"]
    out_zp = int(layer["cpu_requant_after_npu"]["output_zero_point"])
    activation = layer.get("activation", "none")

    post_acc = torch.tensor(raw_acc, dtype=torch.float64)
    for col in range(cols):
        global_col = n_base + col
        post_acc[:, col] += float(bias_int32[global_col])
    if activation == "relu":
        post_acc = torch.clamp(post_acc, min=0)
    elif activation == "relu6":
        post_acc = torch.clamp(post_acc, min=0, max=6)

    mult = torch.tensor([multipliers[n_base + col] for col in range(cols)], dtype=torch.float64).view(1, cols)
    q = torch.round(post_acc * mult) + out_zp
    q = torch.clamp(q, -128, 127).to(torch.int16)
    return post_acc.to(torch.int64).tolist(), q.tolist()


def run(args):
    project_root = Path(args.project_root).resolve()
    plan_path = (project_root / args.plan).resolve() if not Path(args.plan).is_absolute() else Path(args.plan).resolve()
    plan = load_json(plan_path)
    layer = find_layer(plan, args.layer_name)
    out_shape = layer["output_shape"]
    m_dim = int(layer["registers"]["M_DIM"])
    n_dim = int(layer["registers"]["N_DIM"])

    if args.m_base < 0 or args.n_base < 0:
        raise ValueError("m-base and n-base must be non-negative")
    if args.m_base + args.m_tiles * 4 > m_dim:
        raise ValueError(f"requested M window exceeds layer M={m_dim}")
    if args.n_base + args.n_tiles * 4 > n_dim:
        raise ValueError(f"requested N window exceeds layer N={n_dim}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
        checkpoint = torch.load(args.pth, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)
    if layer["weight_key"] not in state_dict:
        raise KeyError(f"missing weight in checkpoint: {layer['weight_key']}")

    rows = args.m_tiles * 4
    cols = args.n_tiles * 4
    raw_window = [[0 for _ in range(cols)] for _ in range(rows)]
    tiles = []
    for mt in range(args.m_tiles):
        for nt in range(args.n_tiles):
            m_base = args.m_base + mt * 4
            n_base = args.n_base + nt * 4
            tile = run_tile(project_root, args, m_base, n_base)
            tiles.append(tile)
            for r in range(4):
                for c in range(4):
                    raw_window[mt * 4 + r][nt * 4 + c] = tile["raw_acc"][r * 4 + c]

    bias_path = plan_path.parent / layer["assets"]["bias_int32_hex"]
    bias_int32 = load_int32_hex(bias_path)
    post_acc, q_window = postprocess_window(raw_window, layer, bias_int32, args.m_base, args.n_base)

    result = {
        "schema": "repopt_layer_tile_rtl_window_v1",
        "source_pth": str(Path(args.pth).resolve()),
        "plan": str(plan_path),
        "layer": args.layer_name,
        "sample_index": args.index,
        "output_shape": out_shape,
        "window": {
            "m_base": args.m_base,
            "n_base": args.n_base,
            "m_tiles": args.m_tiles,
            "n_tiles": args.n_tiles,
            "rows": rows,
            "cols": cols,
        },
        "tiles": tiles,
        "raw_acc_window": raw_window,
        "cpu_post_acc_window": post_acc,
        "cpu_requant_qint8_window": q_window,
        "note": (
            "Each 4x4 tile was computed by RTL npu_top tile mode. CPU postprocess "
            "applies bias/ReLU/per-channel requant on the stitched window."
        ),
    }

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)

    print(f"RTL tile window written to: {out_json}")
    print(
        f"layer={args.layer_name} sample_index={args.index} "
        f"window=M[{args.m_base}:{args.m_base + rows}) N[{args.n_base}:{args.n_base + cols})"
    )
    print(f"tiles_run={len(tiles)} raw_min={min(min(row) for row in raw_window)} raw_max={max(max(row) for row in raw_window)}")
    print(f"q_min={min(min(row) for row in q_window)} q_max={max(max(row) for row in q_window)}")


def main():
    parser = argparse.ArgumentParser(description="Run a small RepOpt Conv tile window through RTL NPU tile mode")
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
    parser.add_argument("--m-tiles", type=int, default=2, help="Number of 4-row tiles to run")
    parser.add_argument("--n-tiles", type=int, default=2, help="Number of 4-col tiles to run")
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument("--out-json", default="sim/pth_repopt_tile_rtl/layer_tile_window.json", help="Output JSON report")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
