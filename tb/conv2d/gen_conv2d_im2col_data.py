# =============================================================================
# gen_conv2d_im2col_data.py - Generate Conv2D-as-GEMM T6.x test data
#
# T6.1 keeps im2col in DRAM/test software. This script expands a dense Conv2D
# layer into A_im2col[M,K] and W_col[K,N], writes them using the existing
# direct-matmul DRAM layout, and stores the Conv2D golden output in expected.hex.
# =============================================================================

import argparse
import os
import random
import struct
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_TB_DIR = os.path.dirname(THIS_DIR)
MATMUL_DIR = os.path.join(PROJECT_TB_DIR, "matmul")
if MATMUL_DIR not in sys.path:
    sys.path.insert(0, MATMUL_DIR)

from gen_matmul_data import (  # noqa: E402
    activation_ctrl_bits,
    activation_mode_id,
    apply_activation,
    apply_bias,
    apply_quant,
    bias_to_words,
    build_dram,
    float_to_fp16,
    fp16_to_float,
    float_to_fp32_word,
    fp32_word_to_float,
    gen_fp32_bias,
    gen_int32_bias,
    fp16_pack,
    int8_pack,
    make_ctrl,
    quant_cfg_word,
    write_hex,
)


def output_dim(input_size, kernel, stride, pad, dilation):
    numerator = input_size + 2 * pad - dilation * (kernel - 1) - 1
    if numerator < 0:
        raise ValueError("invalid Conv2D shape: kernel window is larger than padded input")
    return numerator // stride + 1


def signed8(x):
    x &= 0xFF
    return x - 256 if x & 0x80 else x


def make_int8_tensors(batch, cin, ih, iw, cout, kh, kw, seed):
    rng = random.Random(seed)
    ifm = [
        [
            [[rng.randint(-5, 5) for _ in range(iw)] for _ in range(ih)]
            for _ in range(cin)
        ]
        for _ in range(batch)
    ]
    weight = [
        [
            [[rng.randint(-4, 4) for _ in range(kw)] for _ in range(kh)]
            for _ in range(cin)
        ]
        for _ in range(cout)
    ]
    return ifm, weight


def make_fp16_tensors(batch, cin, ih, iw, cout, kh, kw, seed):
    rng = random.Random(seed)
    choices = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0]
    ifm = [
        [
            [[float_to_fp16(rng.choice(choices)) for _ in range(iw)] for _ in range(ih)]
            for _ in range(cin)
        ]
        for _ in range(batch)
    ]
    weight = [
        [
            [[float_to_fp16(rng.choice(choices)) for _ in range(kw)] for _ in range(kh)]
            for _ in range(cin)
        ]
        for _ in range(cout)
    ]
    return ifm, weight


def conv_to_im2col(ifm, weight, cfg):
    batch = cfg["batch"]
    cin = cfg["cin"]
    ih = cfg["ih"]
    iw = cfg["iw"]
    cout = cfg["cout"]
    kh = cfg["kh"]
    kw = cfg["kw"]
    oh = cfg["oh"]
    ow = cfg["ow"]
    stride_h = cfg["stride_h"]
    stride_w = cfg["stride_w"]
    pad_h = cfg["pad_h"]
    pad_w = cfg["pad_w"]
    dilation_h = cfg["dilation_h"]
    dilation_w = cfg["dilation_w"]
    dtype = cfg["dtype"]

    zero = 0
    a_im2col = []
    for b in range(batch):
        for out_h in range(oh):
            for out_w in range(ow):
                row = []
                for c in range(cin):
                    for ker_h in range(kh):
                        for ker_w in range(kw):
                            in_h = out_h * stride_h + ker_h * dilation_h - pad_h
                            in_w = out_w * stride_w + ker_w * dilation_w - pad_w
                            if 0 <= in_h < ih and 0 <= in_w < iw:
                                row.append(ifm[b][c][in_h][in_w])
                            else:
                                row.append(zero)
                a_im2col.append(row)

    w_col = []
    for c in range(cin):
        for ker_h in range(kh):
            for ker_w in range(kw):
                row = []
                for out_c in range(cout):
                    row.append(weight[out_c][c][ker_h][ker_w])
                w_col.append(row)

    c_mat = []
    for m in range(len(a_im2col)):
        out_row = []
        for n in range(cout):
            if dtype == "int8":
                acc = 0
                for k in range(len(w_col)):
                    acc += signed8(a_im2col[m][k]) * signed8(w_col[k][n])
                out_row.append(acc & 0xFFFFFFFF)
            else:
                acc = 0.0
                for k in range(len(w_col)):
                    acc += fp16_to_float(a_im2col[m][k]) * fp16_to_float(w_col[k][n])
                out_row.append(float_to_fp32_word(acc))
        c_mat.append(out_row)

    return a_im2col, w_col, c_mat


def write_matrix_txt(path, matrix, dtype):
    with open(path, "w") as f:
        for row in matrix:
            if dtype == "int8":
                f.write(" ".join(str(signed8(v)) for v in row))
            else:
                f.write(" ".join(f"{fp16_to_float(v):.6g}" for v in row))
            f.write("\n")


def write_vector_txt(path, values, dtype):
    with open(path, "w") as f:
        if dtype == "int8":
            f.write(" ".join(str(signed8(v)) for v in values))
        else:
            f.write(" ".join(f"{fp16_to_float(v):.6g}" for v in values))
        f.write("\n")


def write_expected_nhwc(path, c_mat, batch, oh, ow, cout):
    words = []
    for b in range(batch):
        for out_h in range(oh):
            for out_w in range(ow):
                m = (b * oh + out_h) * ow + out_w
                for out_c in range(cout):
                    words.append(c_mat[m][out_c])
    write_hex(path, words)


def flatten_ifm_nchw(ifm, batch, cin, ih, iw):
    vals = []
    for b in range(batch):
        for c in range(cin):
            for in_h in range(ih):
                for in_w in range(iw):
                    vals.append(ifm[b][c][in_h][in_w])
    return vals


def build_dram_otf(ifm, w_col, c_mat, cfg, w_base, a_base, r_base, bias_words=None, bias_base=0):
    dtype = cfg["dtype"]
    k_dim = cfg["cin"] * cfg["kh"] * cfg["kw"]
    n_dim = cfg["cout"]
    m_dim = cfg["batch"] * cfg["oh"] * cfg["ow"]
    elem_bytes = 1 if dtype == "int8" else 2
    pack_fn = int8_pack if dtype == "int8" else fp16_pack
    b_col_stride = ((k_dim * elem_bytes + 3) >> 2) << 2

    dram = {}

    for j in range(n_dim):
        col_data = [w_col[k][j] for k in range(k_dim)]
        for idx, word in enumerate(pack_fn(col_data)):
            dram[(w_base + j * b_col_stride + idx * 4) >> 2] = word

    ifm_data = flatten_ifm_nchw(ifm, cfg["batch"], cfg["cin"], cfg["ih"], cfg["iw"])
    for idx, word in enumerate(pack_fn(ifm_data)):
        dram[(a_base + idx * 4) >> 2] = word

    for i in range(m_dim):
        for j in range(n_dim):
            dram[(r_base + (i * n_dim + j) * 4) >> 2] = 0

    if bias_words:
        for j, word in enumerate(bias_words):
            dram[(bias_base + j * 4) >> 2] = word & 0xFFFFFFFF

    return dram


def generate(args):
    if args.quant and args.dtype != "int8":
        raise ValueError("INT8 quant/saturate requires dtype=int8")

    oh = output_dim(args.ih, args.kh, args.stride_h, args.pad_h, args.dilation_h)
    ow = output_dim(args.iw, args.kw, args.stride_w, args.pad_w, args.dilation_w)
    cfg = {
        "batch": args.batch,
        "ih": args.ih,
        "iw": args.iw,
        "cin": args.cin,
        "cout": args.cout,
        "kh": args.kh,
        "kw": args.kw,
        "stride_h": args.stride_h,
        "stride_w": args.stride_w,
        "pad_h": args.pad_h,
        "pad_w": args.pad_w,
        "dilation_h": args.dilation_h,
        "dilation_w": args.dilation_w,
        "oh": oh,
        "ow": ow,
        "dtype": args.dtype,
    }

    if args.dtype == "int8":
        ifm, weight = make_int8_tensors(
            args.batch, args.cin, args.ih, args.iw, args.cout, args.kh, args.kw, args.seed
        )
    else:
        ifm, weight = make_fp16_tensors(
            args.batch, args.cin, args.ih, args.iw, args.cout, args.kh, args.kw, args.seed
        )

    a_im2col, w_col, c_mat = conv_to_im2col(ifm, weight, cfg)
    m_dim = args.batch * oh * ow
    k_dim = args.cin * args.kh * args.kw
    n_dim = args.cout
    elem_bytes = 1 if args.dtype == "int8" else 2
    bias = gen_int32_bias(n_dim, args.seed + 17) if args.bias and args.dtype == "int8" else []
    if args.bias and args.dtype == "fp16":
        bias = gen_fp32_bias(n_dim, args.seed + 17)
    c_expected = apply_quant(
        apply_activation(apply_bias(c_mat, bias, args.dtype), args.dtype, args.activation),
        args.dtype,
        args.quant,
        args.quant_scale,
        args.quant_shift,
        args.quant_round,
    )
    bias_words = bias_to_words(bias, args.dtype) if args.bias else []

    w_base = 0x10000
    b_col_bytes = ((k_dim * elem_bytes + 3) >> 2) << 2
    a_base = w_base + b_col_bytes * n_dim + 0x100
    a_row_bytes = ((k_dim * elem_bytes + 3) >> 2) << 2
    ifm_bytes = ((args.batch * args.cin * args.ih * args.iw * elem_bytes + 3) >> 2) << 2
    r_base = a_base + (ifm_bytes if args.on_the_fly else a_row_bytes * m_dim) + 0x100
    bias_base = r_base + (m_dim * n_dim * 4) + 0x100 if args.bias else 0

    if args.on_the_fly:
        dram = build_dram_otf(ifm, w_col, c_mat, cfg, w_base, a_base, r_base, bias_words, bias_base)
    else:
        dram = build_dram(
            a_im2col,
            w_col,
            c_mat,
            m_dim,
            k_dim,
            n_dim,
            args.dtype,
            w_base,
            a_base,
            r_base,
            b_col_bytes,
            a_row_bytes,
            bias_words,
            bias_base,
        )

    expected = []
    for m in range(m_dim):
        for n in range(n_dim):
            expected.append(c_expected[m][n])

    max_addr = max(dram.keys()) if dram else 0
    dram_arr = [0] * (max_addr + 1)
    for addr, value in dram.items():
        dram_arr[addr] = value

    test_id = args.test_id
    if not test_id:
        prefix = "conv2d_otf" if args.on_the_fly else "conv2d_im2col"
        test_id = (
            f"{prefix}_{args.dtype}_{args.mode.lower()}_"
            f"b{args.batch}_{args.ih}x{args.iw}_c{args.cin}_"
            f"k{args.kh}x{args.kw}_co{args.cout}_p{args.pad_h}x{args.pad_w}"
        )

    out_dir = os.path.join(THIS_DIR, test_id)
    os.makedirs(out_dir, exist_ok=True)

    write_hex(os.path.join(out_dir, "dram_init.hex"), dram_arr)
    write_hex(os.path.join(out_dir, "expected.hex"), expected)
    write_expected_nhwc(os.path.join(out_dir, "conv_expected_nhwc.hex"), c_expected, args.batch, oh, ow, args.cout)
    if args.on_the_fly:
        write_vector_txt(os.path.join(out_dir, "ifm_nchw.txt"), flatten_ifm_nchw(ifm, args.batch, args.cin, args.ih, args.iw), args.dtype)
    else:
        write_matrix_txt(os.path.join(out_dir, "a_im2col.txt"), a_im2col, args.dtype)
    write_matrix_txt(os.path.join(out_dir, "w_col.txt"), w_col, args.dtype)
    if args.bias:
        with open(os.path.join(out_dir, "bias.txt"), "w") as f:
            if args.dtype == "int8":
                f.write(" ".join(str(v) for v in bias))
            else:
                f.write(" ".join(f"{v:.6g}" for v in bias))
            f.write("\n")

    ctrl = make_ctrl(args.dtype, args.mode)
    if args.on_the_fly:
        ctrl |= 0x100
    if args.bias:
        ctrl |= 0x200
    ctrl |= activation_ctrl_bits(args.activation)
    quant_cfg = quant_cfg_word(args.quant, args.quant_scale, args.quant_shift, args.quant_round)
    num_results = m_dim * n_dim
    conv_ifm_shape = (args.iw << 16) | args.ih
    conv_channels = (args.batch << 16) | args.cin
    conv_kernel = (args.kw << 16) | args.kh
    conv_out_shape = (ow << 16) | oh
    conv_stride_pad = (args.pad_w << 24) | (args.pad_h << 16) | (args.stride_w << 8) | args.stride_h
    conv_dilation = (args.dilation_w << 8) | args.dilation_h
    with open(os.path.join(out_dir, "test_params.vh"), "w") as f:
        f.write(f"// Auto-generated: {test_id} ({args.dtype} {args.mode})\n")
        if args.on_the_fly:
            f.write("// T6.2 Conv2D uses on-the-fly hardware im2col gather from IFM.\n")
        else:
            f.write("// T6.1 Conv2D uses DRAM pre-expanded im2col.\n")
        f.write(
            f"// Conv2D: B={args.batch} IFM={args.ih}x{args.iw} Cin={args.cin} "
            f"KHxKW={args.kh}x{args.kw} Cout={args.cout} OHxOW={oh}x{ow}\n"
        )
        f.write(f"// GEMM: A_im2col[{m_dim}x{k_dim}] x W_col[{k_dim}x{n_dim}]\n")
        f.write(f"`define NUM_RESULTS {num_results}\n")
        f.write(f"`define M_DIM {m_dim}\n")
        f.write(f"`define N_DIM {n_dim}\n")
        f.write(f"`define K_DIM {k_dim}\n")
        f.write(f"`define W_ADDR 32'h{w_base:08x}\n")
        f.write(f"`define A_ADDR 32'h{a_base:08x}\n")
        f.write(f"`define R_ADDR 32'h{r_base:08x}\n")
        f.write(f"`define BIAS_ADDR 32'h{bias_base:08x}\n")
        f.write(f"`define BIAS_EN {1 if args.bias else 0}\n")
        f.write(f"`define ACT_MODE {activation_mode_id(args.activation)}\n")
        f.write(f"`define QUANT_EN {1 if args.quant else 0}\n")
        f.write(f"`define QUANT_CFG 32'h{quant_cfg:08x}\n")
        f.write(f"`define QUANT_SCALE {args.quant_scale}\n")
        f.write(f"`define QUANT_SHIFT {args.quant_shift}\n")
        f.write(f"`define QUANT_ROUND {1 if args.quant_round else 0}\n")
        f.write(f"`define CTRL   32'h{ctrl:02x}\n")
        f.write(f"`define DRAM_SIZE {max(max_addr + 1, 8192)}\n")
        f.write(f"`define IS_FP16 {1 if args.dtype == 'fp16' else 0}\n")
        f.write(f"`define IS_OS   {1 if args.mode == 'OS' else 0}\n")
        f.write(f"`define CONV_BATCH {args.batch}\n")
        f.write(f"`define CONV_IH {args.ih}\n")
        f.write(f"`define CONV_IW {args.iw}\n")
        f.write(f"`define CONV_CIN {args.cin}\n")
        f.write(f"`define CONV_COUT {args.cout}\n")
        f.write(f"`define CONV_KH {args.kh}\n")
        f.write(f"`define CONV_KW {args.kw}\n")
        f.write(f"`define CONV_OH {oh}\n")
        f.write(f"`define CONV_OW {ow}\n")
        f.write(f"`define CONV_IM2COL {1 if args.on_the_fly else 0}\n")
        f.write(f"`define CONV_IFM_SHAPE 32'h{conv_ifm_shape:08x}\n")
        f.write(f"`define CONV_CHANNELS 32'h{conv_channels:08x}\n")
        f.write(f"`define CONV_KERNEL 32'h{conv_kernel:08x}\n")
        f.write(f"`define CONV_OUT_SHAPE 32'h{conv_out_shape:08x}\n")
        f.write(f"`define CONV_STRIDE_PAD 32'h{conv_stride_pad:08x}\n")
        f.write(f"`define CONV_DILATION 32'h{conv_dilation:08x}\n")

    with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
        f.write(f"test_id={test_id}\n")
        f.write(f"dtype={args.dtype}\n")
        f.write(f"mode={args.mode}\n")
        f.write(f"on_the_fly={1 if args.on_the_fly else 0}\n")
        f.write(f"batch={args.batch}\n")
        f.write(f"ifm={args.ih}x{args.iw}x{args.cin}\n")
        f.write(f"weight={args.cout}x{args.cin}x{args.kh}x{args.kw}\n")
        f.write(f"stride={args.stride_h}x{args.stride_w}\n")
        f.write(f"pad={args.pad_h}x{args.pad_w}\n")
        f.write(f"dilation={args.dilation_h}x{args.dilation_w}\n")
        f.write(f"ofm={oh}x{ow}x{args.cout}\n")
        f.write(f"gemm={m_dim}x{k_dim}x{n_dim}\n")
        f.write(f"w_addr=0x{w_base:08x}\n")
        f.write(f"a_addr=0x{a_base:08x}\n")
        f.write(f"r_addr=0x{r_base:08x}\n")
        if args.bias:
            f.write(f"bias_addr=0x{bias_base:08x}\n")
        if args.quant:
            f.write(f"quant_cfg=0x{quant_cfg:08x}\n")
            f.write(f"quant_scale={args.quant_scale}\n")
            f.write(f"quant_shift={args.quant_shift}\n")
            f.write(f"quant_round={1 if args.quant_round else 0}\n")

    print(f"  {test_id:46s} ({args.dtype:4s} {args.mode:2s})")
    print(
        f"    Conv2D B={args.batch} IFM={args.ih}x{args.iw} Cin={args.cin} "
        f"KHxKW={args.kh}x{args.kw} Cout={args.cout} -> OHxOW={oh}x{ow}"
    )
    print(f"    GEMM M={m_dim} K={k_dim} N={n_dim}, expected results={num_results}")
    print(f"    A source={'IFM NCHW on-the-fly' if args.on_the_fly else 'DRAM A_im2col'}")
    bias_part = f" BIAS_BASE=0x{bias_base:08x}" if args.bias else ""
    print(f"    W_BASE=0x{w_base:08x} A_BASE=0x{a_base:08x} R_BASE=0x{r_base:08x}{bias_part}")
    if args.bias:
        shown_bias = []
        for value in bias[:min(n_dim, 4)]:
            shown_bias.append(str(value) if args.dtype == "int8" else f"{value:.4f}")
        print(f"    Bias[0:{len(shown_bias)}] = {', '.join(shown_bias)}")
    if args.activation != "none":
        print(f"    Activation = {args.activation}")
    if args.quant:
        print(
            f"    Quant = scale {args.quant_scale}, shift {args.quant_shift}, "
            f"round {1 if args.quant_round else 0}, saturate int8"
        )
    for m in range(min(m_dim, 2)):
        for n in range(min(n_dim, 4)):
            value = c_expected[m][n]
            if args.dtype == "int8":
                shown = value if value < 0x80000000 else value - 0x100000000
                print(f"    C[{m}][{n}] = {shown}", end="")
            else:
                shown = fp32_word_to_float(value)
                print(f"    C[{m}][{n}] = {shown:.4f}", end="")
        print()

    return out_dir


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return ivalue


def nonnegative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return ivalue


def main():
    parser = argparse.ArgumentParser(description="Generate T6.1-T6.5 Conv2D test data")
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--ih", type=positive_int, default=5)
    parser.add_argument("--iw", type=positive_int, default=5)
    parser.add_argument("--cin", type=positive_int, default=2)
    parser.add_argument("--cout", type=positive_int, default=3)
    parser.add_argument("--kh", type=positive_int, default=3)
    parser.add_argument("--kw", type=positive_int, default=3)
    parser.add_argument("--stride-h", type=positive_int, default=1)
    parser.add_argument("--stride-w", type=positive_int, default=1)
    parser.add_argument("--pad-h", type=nonnegative_int, default=1)
    parser.add_argument("--pad-w", type=nonnegative_int, default=1)
    parser.add_argument("--dilation-h", type=positive_int, default=1)
    parser.add_argument("--dilation-w", type=positive_int, default=1)
    parser.add_argument("--dtype", choices=["int8", "fp16"], default="int8")
    parser.add_argument("--mode", choices=["OS", "WS"], default="OS")
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--test-id", default="")
    parser.add_argument("--on-the-fly", action="store_true", help="Store raw IFM and enable T6.2 hardware im2col")
    parser.add_argument("--bias", action="store_true", help="Enable T6.3 32-bit bias addition")
    parser.add_argument("--activation", choices=["none", "relu", "relu6"], default="none",
                        help="Enable T6.4 direct-scalar activation")
    parser.add_argument("--quant", action="store_true", help="Enable T6.5 INT8 quant/saturate")
    parser.add_argument("--quant-scale", type=int, default=1, help="T6.5 signed 16-bit quant scale")
    parser.add_argument("--quant-shift", type=nonnegative_int, default=0, help="T6.5 arithmetic right shift [0,31]")
    parser.add_argument("--quant-round", action="store_true", help="T6.5 round before right shift")
    args = parser.parse_args()

    if args.quant_shift > 31:
        parser.error("--quant-shift must be in [0,31]")

    label = "T6.2 Conv2D on-the-fly im2col" if args.on_the_fly else "T6.1 Conv2D im2col"
    if args.bias:
        label += " + T6.3 bias"
    if args.activation != "none":
        label += f" + T6.4 {args.activation}"
    if args.quant:
        label += " + T6.5 quant"
    print(f"Generating {label} test data in:\n  {THIS_DIR}\n")
    generate(args)
    print(f"\nDone. Tests generated in: {THIS_DIR}")


if __name__ == "__main__":
    main()
