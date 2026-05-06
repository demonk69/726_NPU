#!/usr/bin/env python3
# =============================================================================
# gen_image_conv_data.py - Generate and render an image Conv2D visualization case.
#
# The generated case reuses the existing direct scalar Conv2D on-the-fly im2col
# testbench path. Input image pixels are converted to signed INT8 by gray-128.
# =============================================================================

import argparse
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

from gen_matmul_data import int8_pack, make_ctrl, write_hex  # noqa: E402


KERNELS = {
    "laplacian": [
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ],
    "sobel_x": [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    "sobel_y": [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],
    "sharpen": [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
}


def align4(value):
    return (value + 3) & ~3


def signed32_to_word(value):
    return value & 0xFFFFFFFF


def word_to_signed32(value):
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def gray_to_int8(value):
    signed = int(value) - 128
    if signed < -128:
        return -128
    if signed > 127:
        return 127
    return signed


def flatten_kernel(kernel):
    vals = []
    for row in kernel:
        vals.extend(row)
    return vals


def conv2d_same(ifm, kernel):
    height = len(ifm)
    width = len(ifm[0])
    out = []
    for y in range(height):
        for x in range(width):
            acc = 0
            for ky in range(3):
                for kx in range(3):
                    iy = y + ky - 1
                    ix = x + kx - 1
                    pix = ifm[iy][ix] if 0 <= iy < height and 0 <= ix < width else 0
                    acc += pix * kernel[ky][kx]
            out.append(acc)
    return out


def write_matrix_txt(path, matrix):
    with open(path, "w", encoding="utf-8") as f:
        for row in matrix:
            f.write(" ".join(str(v) for v in row))
            f.write("\n")


def read_hex_words(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                words.append(int(line, 16))
    return words


def edge_image(values, width, height, scale_abs=None):
    signed = [word_to_signed32(v) if v >= 0 else v for v in values]
    mags = [abs(v) for v in signed]
    scale = scale_abs if scale_abs is not None else max(mags) if mags else 1
    if scale <= 0:
        scale = 1
    pixels = [min(255, int(round(v * 255.0 / scale))) for v in mags]
    return Image.frombytes("L", (width, height), bytes(pixels))


def diff_image(golden_words, npu_words, width, height):
    diffs = [abs(word_to_signed32(g) - word_to_signed32(n)) for g, n in zip(golden_words, npu_words)]
    scale = max(diffs) if diffs else 1
    if scale <= 0:
        scale = 1
    pixels = [min(255, int(round(v * 255.0 / scale))) for v in diffs]
    return Image.frombytes("L", (width, height), bytes(pixels))


def write_comparison(case_dir, panels):
    label_h = 18
    gap = 8
    width = sum(img.width for _, img in panels) + gap * (len(panels) - 1)
    height = max(img.height for _, img in panels) + label_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, img in panels:
        rgb = img.convert("RGB")
        canvas.paste(rgb, (x, label_h))
        draw.text((x, 2), label, fill=(0, 0, 0))
        x += img.width + gap
    canvas.save(case_dir / "comparison.png")


def build_dram(ifm, kernel_vals, expected_words, width, height):
    k_dim = 9
    n_dim = 1
    elem_bytes = 1
    w_base = 0x00010000
    b_col_bytes = align4(k_dim * elem_bytes)
    a_base = w_base + b_col_bytes * n_dim + 0x100
    ifm_bytes = align4(width * height * elem_bytes)
    r_base = a_base + ifm_bytes + 0x100

    dram = {}
    for idx, word in enumerate(int8_pack(kernel_vals)):
        dram[(w_base + idx * 4) >> 2] = word

    ifm_flat = [pix for row in ifm for pix in row]
    for idx, word in enumerate(int8_pack(ifm_flat)):
        dram[(a_base + idx * 4) >> 2] = word

    for idx in range(width * height):
        dram[(r_base + idx * 4) >> 2] = 0

    max_addr = max(dram.keys()) if dram else 0
    dram_arr = [0] * (max_addr + 1)
    for addr, value in dram.items():
        dram_arr[addr] = value

    return dram_arr, w_base, a_base, r_base


def generate_case(args):
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    kernel = KERNELS[args.kernel]
    kernel_vals = flatten_kernel(kernel)
    test_id = args.test_id or f"image_{args.kernel}_{image_path.stem}"
    out_root = Path(args.output_root).resolve()
    case_dir = out_root / test_id
    case_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("L")
    if args.resize:
        image = image.resize((args.resize, args.resize), Image.Resampling.BILINEAR)
    width, height = image.size
    gray_rows = [[image.getpixel((x, y)) for x in range(width)] for y in range(height)]
    ifm = [[gray_to_int8(v) for v in row] for row in gray_rows]
    expected_signed = conv2d_same(ifm, kernel)
    expected_words = [signed32_to_word(v) for v in expected_signed]
    dram_arr, w_base, a_base, r_base = build_dram(ifm, kernel_vals, expected_words, width, height)

    write_hex(case_dir / "dram_init.hex", dram_arr)
    write_hex(case_dir / "expected.hex", expected_words)
    write_hex(case_dir / "conv_expected_nhwc.hex", expected_words)
    write_matrix_txt(case_dir / "ifm_nchw.txt", ifm)
    write_matrix_txt(case_dir / "w_col.txt", [[v] for v in kernel_vals])

    input_copy = case_dir / "input_gray.png"
    image.save(input_copy)
    max_abs = max(abs(v) for v in expected_signed) if expected_signed else 1
    edge_image(expected_words, width, height, max_abs).save(case_dir / "golden.png")

    ctrl = make_ctrl("int8", args.mode) | 0x100
    conv_ifm_shape = (width << 16) | height
    conv_channels = (1 << 16) | 1
    conv_kernel = (3 << 16) | 3
    conv_out_shape = (width << 16) | height
    conv_stride_pad = (1 << 24) | (1 << 16) | (1 << 8) | 1
    conv_dilation = (1 << 8) | 1
    num_results = width * height
    max_addr = max(len(dram_arr), 8192)

    with open(case_dir / "test_params.vh", "w", encoding="utf-8") as f:
        f.write(f"// Auto-generated image Conv2D case: {test_id}\n")
        f.write(f"// image={image_path.as_posix()} kernel={args.kernel} size={width}x{height}\n")
        f.write("// Input is grayscale signed INT8: gray - 128. Output is raw INT32 response.\n")
        f.write(f"`define NUM_RESULTS {num_results}\n")
        f.write(f"`define M_DIM {num_results}\n")
        f.write("`define N_DIM 1\n")
        f.write("`define K_DIM 9\n")
        f.write(f"`define W_ADDR 32'h{w_base:08x}\n")
        f.write(f"`define A_ADDR 32'h{a_base:08x}\n")
        f.write(f"`define R_ADDR 32'h{r_base:08x}\n")
        f.write("`define BIAS_ADDR 32'h00000000\n")
        f.write("`define BIAS_EN 0\n")
        f.write("`define ACT_MODE 0\n")
        f.write("`define QUANT_EN 0\n")
        f.write("`define QUANT_CFG 32'h00010000\n")
        f.write("`define QUANT_SCALE 1\n")
        f.write("`define QUANT_SHIFT 0\n")
        f.write("`define QUANT_ROUND 0\n")
        f.write(f"`define CTRL 32'h{ctrl:08x}\n")
        f.write(f"`define DRAM_SIZE {max_addr}\n")
        f.write("`define IS_FP16 0\n")
        f.write(f"`define IS_OS {1 if args.mode == 'OS' else 0}\n")
        f.write("`define CONV_BATCH 1\n")
        f.write(f"`define CONV_IH {height}\n")
        f.write(f"`define CONV_IW {width}\n")
        f.write("`define CONV_CIN 1\n")
        f.write("`define CONV_COUT 1\n")
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
        "kernel": args.kernel,
        "kernel_values": kernel,
        "width": width,
        "height": height,
        "mode": args.mode,
        "m_dim": num_results,
        "n_dim": 1,
        "k_dim": 9,
        "w_addr": f"0x{w_base:08x}",
        "a_addr": f"0x{a_base:08x}",
        "r_addr": f"0x{r_base:08x}",
        "max_abs_golden": max_abs,
        "input_mapping": "signed_int8 = grayscale - 128",
    }
    with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(case_dir / "metadata.txt", "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}={value}\n")

    print(f"Generated image Conv2D case: {case_dir}")
    print(f"  image={image_path} size={width}x{height} kernel={args.kernel}")
    print(f"  GEMM M={num_results} K=9 N=1")
    print(f"  golden={case_dir / 'golden.png'}")
    return case_dir


def render_case(case_dir):
    case_dir = Path(case_dir).resolve()
    with open(case_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    width = int(metadata["width"])
    height = int(metadata["height"])
    scale = int(metadata.get("max_abs_golden", 1)) or 1

    golden_words = read_hex_words(case_dir / "expected.hex")
    npu_path = case_dir / "npu_output.hex"
    if not npu_path.exists():
        raise FileNotFoundError(f"missing NPU output: {npu_path}")
    npu_words = read_hex_words(npu_path)
    if len(npu_words) != len(golden_words):
        raise ValueError(f"npu words={len(npu_words)} expected={len(golden_words)}")

    mismatches = 0
    max_abs_diff = 0
    for g, n in zip(golden_words, npu_words):
        diff = abs(word_to_signed32(g) - word_to_signed32(n))
        if diff != 0:
            mismatches += 1
        if diff > max_abs_diff:
            max_abs_diff = diff

    input_img = Image.open(case_dir / "input_gray.png").convert("L")
    golden_img = edge_image(golden_words, width, height, scale)
    npu_img = edge_image(npu_words, width, height, scale)
    diff_img = diff_image(golden_words, npu_words, width, height)
    golden_img.save(case_dir / "golden.png")
    npu_img.save(case_dir / "npu.png")
    diff_img.save(case_dir / "diff.png")
    write_comparison(
        case_dir,
        [("input", input_img), ("golden", golden_img), ("npu", npu_img), ("diff", diff_img)],
    )
    with open(case_dir / "visual_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"case_dir={case_dir}\n")
        f.write(f"width={width}\n")
        f.write(f"height={height}\n")
        f.write(f"mismatches={mismatches}\n")
        f.write(f"max_abs_diff={max_abs_diff}\n")
        f.write("input=input_gray.png\n")
        f.write("golden=golden.png\n")
        f.write("npu=npu.png\n")
        f.write("diff=diff.png\n")
        f.write("comparison=comparison.png\n")

    print(f"Rendered image Conv2D outputs in: {case_dir}")
    print(f"  mismatches={mismatches} max_abs_diff={max_abs_diff}")
    print(f"  input ={case_dir / 'input_gray.png'}")
    print(f"  golden={case_dir / 'golden.png'}")
    print(f"  npu   ={case_dir / 'npu.png'}")
    print(f"  diff  ={case_dir / 'diff.png'}")
    print(f"  panel ={case_dir / 'comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate/render image Conv2D visualization cases")
    parser.add_argument("--image", default=str(PROJECT_ROOT / "pic" / "test1_128.png"))
    parser.add_argument("--kernel", choices=sorted(KERNELS.keys()), default="laplacian")
    parser.add_argument("--mode", choices=["OS", "WS"], default="OS")
    parser.add_argument("--test-id", default="")
    parser.add_argument("--output-root", default=str(THIS_DIR))
    parser.add_argument("--resize", type=int, default=0, help="Optionally resize input to NxN before generating data")
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
