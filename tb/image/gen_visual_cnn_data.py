#!/usr/bin/env python3
# =============================================================================
# gen_visual_cnn_data.py - Generate and render a visual multi-channel Conv2D
# feature-bank verification case.
#
# The generated case reuses the existing direct scalar Conv2D on-the-fly im2col
# testbench path. A real image is converted to signed INT8 grayscale, then six
# 3x3 filters are executed as Cout=6 Conv2D. Bias, activation and INT8 quant are
# enabled by default so the visual case covers the T6.2-T6.5 feature surface.
# =============================================================================

import argparse
import html
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
MATMUL_DIR = PROJECT_ROOT / "tb" / "matmul"
if str(MATMUL_DIR) not in sys.path:
    sys.path.insert(0, str(MATMUL_DIR))

from gen_matmul_data import (  # noqa: E402
    activation_ctrl_bits,
    activation_mode_id,
    apply_activation,
    apply_bias,
    apply_quant,
    int8_pack,
    make_ctrl,
    quant_cfg_word,
    signed32,
    write_hex,
)


FEATURE_BANK = [
    {
        "name": "sobel_x",
        "kernel": [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        "bias": 0,
    },
    {
        "name": "sobel_y",
        "kernel": [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        "bias": 0,
    },
    {
        "name": "laplacian",
        "kernel": [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ],
        "bias": -8,
    },
    {
        "name": "sharpen",
        "kernel": [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        "bias": 8,
    },
    {
        "name": "outline",
        "kernel": [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ],
        "bias": -16,
    },
    {
        "name": "emboss",
        "kernel": [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2],
        ],
        "bias": 0,
    },
]

FUSION = {
    "r": ["sobel_x", "sharpen"],
    "g": ["sobel_y", "emboss"],
    "b": ["laplacian", "outline"],
}


def align4(value):
    return (value + 3) & ~3


def word_to_signed32(value):
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def gray_to_int8(value):
    signed = int(value) - 128
    return min(max(signed, -128), 127)


def flatten_kernel(kernel):
    vals = []
    for row in kernel:
        vals.extend(row)
    return vals


def conv2d_feature_bank(ifm, feature_bank):
    height = len(ifm)
    width = len(ifm[0])
    rows = []
    for y in range(height):
        for x in range(width):
            out_row = []
            for feature in feature_bank:
                kernel = feature["kernel"]
                acc = 0
                for ky in range(3):
                    for kx in range(3):
                        iy = y + ky - 1
                        ix = x + kx - 1
                        pix = ifm[iy][ix] if 0 <= iy < height and 0 <= ix < width else 0
                        acc += pix * kernel[ky][kx]
                out_row.append(acc & 0xFFFFFFFF)
            rows.append(out_row)
    return rows


def write_matrix_txt(path, matrix):
    with open(path, "w", encoding="utf-8") as f:
        for row in matrix:
            f.write(" ".join(str(v) for v in row))
            f.write("\n")


def write_vector_txt(path, values):
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(str(v) for v in values))
        f.write("\n")


def read_hex_words(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                words.append(int(line, 16))
    return words


def words_to_rows(words, width, height, channels):
    if len(words) != width * height * channels:
        raise ValueError(
            f"word count mismatch: got {len(words)}, expected {width * height * channels}"
        )
    rows = []
    for idx in range(width * height):
        base = idx * channels
        rows.append([word_to_signed32(words[base + ch]) for ch in range(channels)])
    return rows


def scale_positive(values):
    vmax = max([abs(v) for v in values] + [1])
    return [min(255, max(0, int(round(abs(v) * 255.0 / vmax)))) for v in values]


def channel_image(rows, width, height, channel):
    values = [row[channel] for row in rows]
    pixels = scale_positive(values)
    return Image.frombytes("L", (width, height), bytes(pixels))


def diff_channel_image(golden_rows, npu_rows, width, height, channel):
    diffs = [abs(g[channel] - n[channel]) for g, n in zip(golden_rows, npu_rows)]
    pixels = scale_positive(diffs)
    return Image.frombytes("L", (width, height), bytes(pixels))


def pseudo_rgb_image(rows, width, height, names):
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    def fused(row, channel_name):
        vals = [max(0, row[name_to_idx[name]]) for name in FUSION[channel_name]]
        return max(vals) if vals else 0

    rgb = []
    for row in rows:
        r = min(255, fused(row, "r") * 2)
        g = min(255, fused(row, "g") * 2)
        b = min(255, fused(row, "b") * 2)
        rgb.extend([r, g, b])
    return Image.frombytes("RGB", (width, height), bytes(rgb))


def diff_heatmap(golden_rows, npu_rows, width, height):
    diffs = []
    for g, n in zip(golden_rows, npu_rows):
        diffs.append(max(abs(gv - nv) for gv, nv in zip(g, n)))
    scale = max(diffs) if diffs else 1
    if scale <= 0:
        scale = 1
    rgb = []
    for diff in diffs:
        hot = min(255, int(round(diff * 255.0 / scale)))
        rgb.extend([hot, 0, 0])
    return Image.frombytes("RGB", (width, height), bytes(rgb))


def write_grid(path, panels, columns=4):
    label_h = 18
    gap = 8
    rows = (len(panels) + columns - 1) // columns
    cell_w = max(img.width for _, img in panels)
    cell_h = max(img.height for _, img in panels) + label_h
    canvas = Image.new(
        "RGB",
        (columns * cell_w + (columns - 1) * gap, rows * cell_h + (rows - 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(panels):
        row = idx // columns
        col = idx % columns
        x = col * (cell_w + gap)
        y = row * (cell_h + gap)
        draw.text((x, y + 2), label, fill=(0, 0, 0))
        canvas.paste(img.convert("RGB"), (x, y + label_h))
    canvas.save(path)


def build_dram(ifm, feature_bank, width, height, expected_words, bias_words):
    k_dim = 9
    n_dim = len(feature_bank)
    elem_bytes = 1
    w_base = 0x00010000
    w_col_stride = align4(k_dim * elem_bytes)
    a_base = w_base + w_col_stride * n_dim + 0x100
    ifm_bytes = align4(width * height * elem_bytes)
    r_base = a_base + ifm_bytes + 0x100
    result_bytes = width * height * n_dim * 4
    bias_base = r_base + result_bytes + 0x100

    dram = {}
    for out_c, feature in enumerate(feature_bank):
        col = flatten_kernel(feature["kernel"])
        for idx, word in enumerate(int8_pack(col)):
            dram[(w_base + out_c * w_col_stride + idx * 4) >> 2] = word

    ifm_flat = [pix for row in ifm for pix in row]
    for idx, word in enumerate(int8_pack(ifm_flat)):
        dram[(a_base + idx * 4) >> 2] = word

    for idx in range(len(expected_words)):
        dram[(r_base + idx * 4) >> 2] = 0

    for idx, word in enumerate(bias_words):
        dram[(bias_base + idx * 4) >> 2] = word & 0xFFFFFFFF

    max_addr = max(dram.keys()) if dram else 0
    dram_arr = [0] * (max_addr + 1)
    for addr, value in dram.items():
        dram_arr[addr] = value & 0xFFFFFFFF

    return dram_arr, w_base, a_base, r_base, bias_base, w_col_stride


def post_process(c_raw, bias, activation, quant_scale, quant_shift, quant_round):
    return apply_quant(
        apply_activation(apply_bias(c_raw, bias, "int8"), "int8", activation),
        "int8",
        True,
        quant_scale,
        quant_shift,
        quant_round,
    )


def channel_stats(rows, feature_names):
    stats = []
    for ch, name in enumerate(feature_names):
        vals = [row[ch] for row in rows]
        mean = sum(vals) / len(vals) if vals else 0.0
        stats.append(
            {
                "name": name,
                "min": min(vals) if vals else 0,
                "max": max(vals) if vals else 0,
                "mean": mean,
            }
        )
    return stats


def write_report(case_dir, metadata, summary):
    feature_rows = []
    for item in summary["channel_stats"]:
        feature_rows.append(
            "<tr>"
            f"<td>{html.escape(item['name'])}</td>"
            f"<td>{item['min']}</td>"
            f"<td>{item['max']}</td>"
            f"<td>{item['mean']:.2f}</td>"
            f"<td><img src=\"golden_{html.escape(item['name'])}.png\"></td>"
            f"<td><img src=\"npu_{html.escape(item['name'])}.png\"></td>"
            f"<td><img src=\"diff_{html.escape(item['name'])}.png\"></td>"
            "</tr>"
        )

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(metadata['test_id'])}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #20242a; }}
    img {{ image-rendering: pixelated; max-width: 192px; border: 1px solid #d0d7de; }}
    table {{ border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; }}
    .hero img {{ max-width: 320px; margin-right: 12px; vertical-align: top; }}
    .ok {{ color: #096a2e; font-weight: 700; }}
    .fail {{ color: #b42318; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>{html.escape(metadata['test_id'])}</h1>
  <p>
    Mode={html.escape(metadata['mode'])},
    size={metadata['width']}x{metadata['height']},
    M={metadata['m_dim']}, K={metadata['k_dim']}, N={metadata['n_dim']},
    activation={html.escape(metadata['activation'])},
    quant=scale {metadata['quant_scale']} shift {metadata['quant_shift']} round {metadata['quant_round']}.
  </p>
  <p class="{ 'ok' if summary['mismatches'] == 0 else 'fail' }">
    mismatches={summary['mismatches']}, max_abs_diff={summary['max_abs_diff']}
  </p>
  <div class="hero">
    <img src="input_gray.png" alt="input">
    <img src="golden_rgb.png" alt="golden rgb">
    <img src="npu_rgb.png" alt="npu rgb">
    <img src="diff_heatmap.png" alt="diff heatmap">
  </div>
  <table>
    <tr><th>channel</th><th>min</th><th>max</th><th>mean</th><th>golden</th><th>npu</th><th>diff</th></tr>
    {''.join(feature_rows)}
  </table>
  <p><a href="comparison_grid.png">comparison_grid.png</a></p>
</body>
</html>
"""
    with open(case_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(html_doc)


def generate_case(args):
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    test_id = args.test_id or f"visual_cnn_{image_path.stem}_{args.mode.lower()}"
    out_root = Path(args.output_root).resolve()
    case_dir = out_root / test_id
    case_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("L")
    if args.resize:
        image = image.resize((args.resize, args.resize), Image.Resampling.BILINEAR)
    width, height = image.size
    gray_rows = [[image.getpixel((x, y)) for x in range(width)] for y in range(height)]
    ifm = [[gray_to_int8(v) for v in row] for row in gray_rows]

    feature_names = [feature["name"] for feature in FEATURE_BANK]
    bias = [feature["bias"] for feature in FEATURE_BANK]
    c_raw = conv2d_feature_bank(ifm, FEATURE_BANK)
    c_expected = post_process(c_raw, bias, args.activation, args.quant_scale, args.quant_shift, args.quant_round)
    expected_words = []
    for row in c_expected:
        expected_words.extend([value & 0xFFFFFFFF for value in row])

    bias_words = [value & 0xFFFFFFFF for value in bias]
    dram_arr, w_base, a_base, r_base, bias_base, w_col_stride = build_dram(
        ifm, FEATURE_BANK, width, height, expected_words, bias_words
    )

    write_hex(case_dir / "dram_init.hex", dram_arr)
    write_hex(case_dir / "expected.hex", expected_words)
    write_hex(case_dir / "conv_expected_nhwc.hex", expected_words)
    write_matrix_txt(case_dir / "ifm_nchw.txt", ifm)
    write_matrix_txt(case_dir / "w_col.txt", [flatten_kernel(feature["kernel"]) for feature in FEATURE_BANK])
    write_vector_txt(case_dir / "bias.txt", bias)
    image.save(case_dir / "input_gray.png")

    ctrl = make_ctrl("int8", args.mode) | 0x100 | 0x200 | activation_ctrl_bits(args.activation)
    quant_cfg = quant_cfg_word(True, args.quant_scale, args.quant_shift, args.quant_round)
    conv_ifm_shape = (width << 16) | height
    conv_channels = (1 << 16) | 1
    conv_kernel = (3 << 16) | 3
    conv_out_shape = (width << 16) | height
    conv_stride_pad = (1 << 24) | (1 << 16) | (1 << 8) | 1
    conv_dilation = (1 << 8) | 1
    num_results = width * height * len(FEATURE_BANK)
    max_addr = max(len(dram_arr), 8192)

    with open(case_dir / "test_params.vh", "w", encoding="utf-8") as f:
        f.write(f"// Auto-generated visual CNN case: {test_id}\n")
        f.write(f"// image={image_path.as_posix()} size={width}x{height}\n")
        f.write("// Six 3x3 feature filters run as one Cout=6 Conv2D OTF layer.\n")
        f.write(f"`define NUM_RESULTS {num_results}\n")
        f.write(f"`define M_DIM {width * height}\n")
        f.write(f"`define N_DIM {len(FEATURE_BANK)}\n")
        f.write("`define K_DIM 9\n")
        f.write(f"`define W_ADDR 32'h{w_base:08x}\n")
        f.write(f"`define A_ADDR 32'h{a_base:08x}\n")
        f.write(f"`define R_ADDR 32'h{r_base:08x}\n")
        f.write(f"`define BIAS_ADDR 32'h{bias_base:08x}\n")
        f.write("`define BIAS_EN 1\n")
        f.write(f"`define ACT_MODE {activation_mode_id(args.activation)}\n")
        f.write("`define QUANT_EN 1\n")
        f.write(f"`define QUANT_CFG 32'h{quant_cfg:08x}\n")
        f.write(f"`define QUANT_SCALE {args.quant_scale}\n")
        f.write(f"`define QUANT_SHIFT {args.quant_shift}\n")
        f.write(f"`define QUANT_ROUND {1 if args.quant_round else 0}\n")
        f.write(f"`define CTRL 32'h{ctrl:08x}\n")
        f.write(f"`define DRAM_SIZE {max_addr}\n")
        f.write("`define IS_FP16 0\n")
        f.write(f"`define IS_OS {1 if args.mode == 'OS' else 0}\n")
        f.write("`define CONV_BATCH 1\n")
        f.write(f"`define CONV_IH {height}\n")
        f.write(f"`define CONV_IW {width}\n")
        f.write("`define CONV_CIN 1\n")
        f.write(f"`define CONV_COUT {len(FEATURE_BANK)}\n")
        f.write("`define CONV_KH 3\n")
        f.write("`define CONV_KW 3\n")
        f.write(f"`define CONV_OH {height}\n")
        f.write(f"`define CONV_OW {width}\n")
        f.write("`define CONV_IM2COL 1\n")
        f.write(f"`define CONV_IFM_SHAPE 32'h{conv_ifm_shape:08x}\n")
        f.write(f"`define CONV_CHANNELS 32'h{conv_channels:08x}\n")
        f.write(f"`define CONV_KERNEL 32'h{conv_kernel:08x}\n")
        f.write(f"`define CONV_OUT_SHAPE 32'h{conv_out_shape:08x}\n")
        f.write(f"`define CONV_STRIDE_PAD 32'h{conv_stride_pad:08x}\n")
        f.write(f"`define CONV_DILATION 32'h{conv_dilation:08x}\n")

    metadata = {
        "test_id": test_id,
        "image": image_path.as_posix(),
        "width": width,
        "height": height,
        "mode": args.mode,
        "feature_names": feature_names,
        "feature_bank": FEATURE_BANK,
        "fusion": FUSION,
        "m_dim": width * height,
        "k_dim": 9,
        "n_dim": len(FEATURE_BANK),
        "num_results": num_results,
        "w_addr": f"0x{w_base:08x}",
        "w_col_stride": w_col_stride,
        "a_addr": f"0x{a_base:08x}",
        "r_addr": f"0x{r_base:08x}",
        "bias_addr": f"0x{bias_base:08x}",
        "bias": bias,
        "activation": args.activation,
        "quant_scale": args.quant_scale,
        "quant_shift": args.quant_shift,
        "quant_round": bool(args.quant_round),
        "input_mapping": "signed_int8 = grayscale - 128",
    }
    with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(case_dir / "metadata.txt", "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}={value}\n")

    render_golden_only(case_dir)
    print(f"Generated visual CNN case: {case_dir}")
    print(f"  image={image_path} size={width}x{height} mode={args.mode}")
    print(f"  GEMM M={width * height} K=9 N={len(FEATURE_BANK)}")
    print(f"  features={', '.join(feature_names)}")
    print(f"  golden_rgb={case_dir / 'golden_rgb.png'}")
    return case_dir


def render_outputs(case_dir, require_npu):
    case_dir = Path(case_dir).resolve()
    with open(case_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    width = int(metadata["width"])
    height = int(metadata["height"])
    names = metadata["feature_names"]
    channels = len(names)

    golden_words = read_hex_words(case_dir / "expected.hex")
    golden_rows = words_to_rows(golden_words, width, height, channels)
    npu_path = case_dir / "npu_output.hex"
    if npu_path.exists():
        npu_words = read_hex_words(npu_path)
        npu_rows = words_to_rows(npu_words, width, height, channels)
    elif require_npu:
        raise FileNotFoundError(f"missing NPU output: {npu_path}")
    else:
        npu_words = golden_words
        npu_rows = golden_rows

    mismatches = 0
    max_abs_diff = 0
    for g, n in zip(golden_rows, npu_rows):
        for gv, nv in zip(g, n):
            diff = abs(gv - nv)
            if diff != 0:
                mismatches += 1
            if diff > max_abs_diff:
                max_abs_diff = diff

    input_img = Image.open(case_dir / "input_gray.png").convert("L")
    panels = [("input", input_img)]
    for ch, name in enumerate(names):
        g_img = channel_image(golden_rows, width, height, ch)
        n_img = channel_image(npu_rows, width, height, ch)
        d_img = diff_channel_image(golden_rows, npu_rows, width, height, ch)
        g_img.save(case_dir / f"golden_{name}.png")
        n_img.save(case_dir / f"npu_{name}.png")
        d_img.save(case_dir / f"diff_{name}.png")
        panels.append((f"g_{name}", g_img))
        panels.append((f"n_{name}", n_img))
        panels.append((f"d_{name}", d_img))

    golden_rgb = pseudo_rgb_image(golden_rows, width, height, names)
    npu_rgb = pseudo_rgb_image(npu_rows, width, height, names)
    diff_img = diff_heatmap(golden_rows, npu_rows, width, height)
    golden_rgb.save(case_dir / "golden_rgb.png")
    npu_rgb.save(case_dir / "npu_rgb.png")
    diff_img.save(case_dir / "diff_heatmap.png")
    panels[1:1] = [("golden_rgb", golden_rgb), ("npu_rgb", npu_rgb), ("diff_heatmap", diff_img)]
    write_grid(case_dir / "comparison_grid.png", panels, columns=4)

    summary = {
        "case_dir": str(case_dir),
        "width": width,
        "height": height,
        "channels": channels,
        "mismatches": mismatches,
        "max_abs_diff": max_abs_diff,
        "channel_stats": channel_stats(npu_rows, names),
        "images": {
            "input": "input_gray.png",
            "golden_rgb": "golden_rgb.png",
            "npu_rgb": "npu_rgb.png",
            "diff_heatmap": "diff_heatmap.png",
            "comparison_grid": "comparison_grid.png",
            "report": "report.html",
        },
    }
    with open(case_dir / "visual_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(case_dir / "visual_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"case_dir={case_dir}\n")
        f.write(f"width={width}\n")
        f.write(f"height={height}\n")
        f.write(f"channels={channels}\n")
        f.write(f"mismatches={mismatches}\n")
        f.write(f"max_abs_diff={max_abs_diff}\n")
        f.write("input=input_gray.png\n")
        f.write("golden_rgb=golden_rgb.png\n")
        f.write("npu_rgb=npu_rgb.png\n")
        f.write("diff_heatmap=diff_heatmap.png\n")
        f.write("comparison_grid=comparison_grid.png\n")
        f.write("report=report.html\n")

    write_report(case_dir, metadata, summary)
    print(f"Rendered visual CNN outputs in: {case_dir}")
    print(f"  mismatches={mismatches} max_abs_diff={max_abs_diff}")
    print(f"  input       ={case_dir / 'input_gray.png'}")
    print(f"  golden_rgb  ={case_dir / 'golden_rgb.png'}")
    print(f"  npu_rgb     ={case_dir / 'npu_rgb.png'}")
    print(f"  diff_heatmap={case_dir / 'diff_heatmap.png'}")
    print(f"  grid        ={case_dir / 'comparison_grid.png'}")
    print(f"  report      ={case_dir / 'report.html'}")


def render_golden_only(case_dir):
    render_outputs(case_dir, require_npu=False)


def render_case(case_dir):
    render_outputs(case_dir, require_npu=True)


def main():
    parser = argparse.ArgumentParser(description="Generate/render visual CNN feature-bank cases")
    parser.add_argument("--image", default=str(PROJECT_ROOT / "pic" / "test2_128.png"))
    parser.add_argument("--mode", choices=["OS", "WS"], default="OS")
    parser.add_argument("--test-id", default="")
    parser.add_argument("--output-root", default=str(THIS_DIR))
    parser.add_argument("--resize", type=int, default=64, help="Resize input to NxN before generating data")
    parser.add_argument("--activation", choices=["none", "relu", "relu6"], default="relu")
    parser.add_argument("--quant-scale", type=int, default=1)
    parser.add_argument("--quant-shift", type=int, default=3)
    parser.add_argument("--quant-round", action="store_true", default=True)
    parser.add_argument("--no-quant-round", action="store_false", dest="quant_round")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--case-dir", default="")
    args = parser.parse_args()

    if args.render_only:
        if not args.case_dir:
            parser.error("--render-only requires --case-dir")
        render_case(args.case_dir)
    else:
        generate_case(args)


if __name__ == "__main__":
    main()
