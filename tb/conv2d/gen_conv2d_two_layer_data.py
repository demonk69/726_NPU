# =============================================================================
# gen_conv2d_two_layer_data.py - Generate T6.6 two-layer Conv2D E2E data
#
# Layer 0:
#   raw NCHW IFM -> on-the-fly im2col Conv2D -> bias -> ReLU -> INT8 quant
#   Output is written as one sign-extended int8 value per 32-bit word.
#
# Layer 1:
#   1x1 Conv2D/GEMM consumes layer0 R_ADDR directly as its A matrix. This works
#   because K=1, so each A row is one 32-bit word and the direct INT8 scalar path
#   consumes the low byte.
# =============================================================================

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_TB_DIR = os.path.dirname(THIS_DIR)
MATMUL_DIR = os.path.join(PROJECT_TB_DIR, "matmul")
if MATMUL_DIR not in sys.path:
    sys.path.insert(0, MATMUL_DIR)

from gen_conv2d_im2col_data import conv_to_im2col, output_dim  # noqa: E402
from gen_matmul_data import (  # noqa: E402
    activation_ctrl_bits,
    apply_activation,
    apply_bias,
    apply_quant,
    int8_pack,
    make_ctrl,
    quant_cfg_word,
    signed32,
    write_hex,
)


def make_layer0_tensors():
    batch, cin, ih, iw = 1, 1, 4, 4
    cout, kh, kw = 1, 3, 3

    ifm = [[[[0 for _ in range(iw)] for _ in range(ih)] for _ in range(cin)] for _ in range(batch)]
    for h in range(ih):
        for w in range(iw):
            ifm[0][0][h][w] = ((h * 3 + w * 2) % 9) - 4

    weight_vals = [
        [1, -2, 1],
        [0, 1, -1],
        [2, 0, -1],
    ]
    weight = [[[[weight_vals[kh_i][kw_i] for kw_i in range(kw)] for kh_i in range(kh)] for _ in range(cin)] for _ in range(cout)]
    return ifm, weight


def flatten_ifm_nchw(ifm):
    vals = []
    for b in range(len(ifm)):
        for c in range(len(ifm[b])):
            for h in range(len(ifm[b][c])):
                for w in range(len(ifm[b][c][h])):
                    vals.append(ifm[b][c][h][w])
    return vals


def place_words(dram, base, words):
    for idx, word in enumerate(words):
        dram[(base + idx * 4) >> 2] = word & 0xFFFFFFFF


def generate(test_id):
    cfg0 = {
        "batch": 1,
        "ih": 4,
        "iw": 4,
        "cin": 1,
        "cout": 1,
        "kh": 3,
        "kw": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "dilation_h": 1,
        "dilation_w": 1,
        "dtype": "int8",
    }
    cfg0["oh"] = output_dim(cfg0["ih"], cfg0["kh"], cfg0["stride_h"], cfg0["pad_h"], cfg0["dilation_h"])
    cfg0["ow"] = output_dim(cfg0["iw"], cfg0["kw"], cfg0["stride_w"], cfg0["pad_w"], cfg0["dilation_w"])

    ifm0, weight0 = make_layer0_tensors()
    _a0_im2col, w0_col, c0_raw = conv_to_im2col(ifm0, weight0, cfg0)
    m0 = cfg0["batch"] * cfg0["oh"] * cfg0["ow"]
    k0 = cfg0["cin"] * cfg0["kh"] * cfg0["kw"]
    n0 = cfg0["cout"]

    bias0 = [-3]
    l0_quant_scale, l0_quant_shift, l0_quant_round = 1, 2, True
    c0_post = apply_quant(
        apply_activation(apply_bias(c0_raw, bias0, "int8"), "int8", "relu"),
        "int8",
        True,
        l0_quant_scale,
        l0_quant_shift,
        l0_quant_round,
    )

    # Layer1 is a 1x1 Conv2D over the single layer0 output channel.
    m1, k1, n1 = m0, 1, 2
    w1 = [[-3, 2]]
    bias1 = [4, -5]
    c1_raw = []
    for m in range(m1):
        a_val = signed32(c0_post[m][0])
        row = []
        for n in range(n1):
            row.append((a_val * w1[0][n]) & 0xFFFFFFFF)
        c1_raw.append(row)

    l1_quant_scale, l1_quant_shift, l1_quant_round = 1, 1, True
    c1_post = apply_quant(
        apply_activation(apply_bias(c1_raw, bias1, "int8"), "int8", "relu"),
        "int8",
        True,
        l1_quant_scale,
        l1_quant_shift,
        l1_quant_round,
    )

    w0_base = 0x10000
    w0_col_bytes = ((k0 + 3) >> 2) << 2
    a0_base = w0_base + w0_col_bytes * n0 + 0x100
    ifm0_bytes = ((len(flatten_ifm_nchw(ifm0)) + 3) >> 2) << 2
    r0_base = a0_base + ifm0_bytes + 0x100
    bias0_base = r0_base + m0 * n0 * 4 + 0x100
    w1_base = bias0_base + n0 * 4 + 0x100
    r1_base = w1_base + n1 * 4 + 0x100
    bias1_base = r1_base + m1 * n1 * 4 + 0x100

    dram = {}

    for n in range(n0):
        col = [w0_col[k][n] for k in range(k0)]
        place_words(dram, w0_base + n * w0_col_bytes, int8_pack(col))

    place_words(dram, a0_base, int8_pack(flatten_ifm_nchw(ifm0)))

    for idx in range(m0 * n0):
        dram[(r0_base + idx * 4) >> 2] = 0

    place_words(dram, bias0_base, [v & 0xFFFFFFFF for v in bias0])

    for n in range(n1):
        place_words(dram, w1_base + n * 4, int8_pack([w1[0][n]]))

    for idx in range(m1 * n1):
        dram[(r1_base + idx * 4) >> 2] = 0

    place_words(dram, bias1_base, [v & 0xFFFFFFFF for v in bias1])

    max_addr = max(dram.keys()) if dram else 0
    dram_arr = [0] * (max_addr + 1)
    for addr, value in dram.items():
        dram_arr[addr] = value & 0xFFFFFFFF

    out_dir = os.path.join(THIS_DIR, test_id)
    os.makedirs(out_dir, exist_ok=True)

    expected0 = [c0_post[m][0] & 0xFFFFFFFF for m in range(m0)]
    expected1 = []
    for m in range(m1):
        for n in range(n1):
            expected1.append(c1_post[m][n] & 0xFFFFFFFF)

    write_hex(os.path.join(out_dir, "dram_init.hex"), dram_arr)
    write_hex(os.path.join(out_dir, "layer0_expected.hex"), expected0)
    write_hex(os.path.join(out_dir, "expected.hex"), expected1)

    conv_ifm_shape = (cfg0["iw"] << 16) | cfg0["ih"]
    conv_channels = (cfg0["batch"] << 16) | cfg0["cin"]
    conv_kernel = (cfg0["kw"] << 16) | cfg0["kh"]
    conv_out_shape = (cfg0["ow"] << 16) | cfg0["oh"]
    conv_stride_pad = (
        (cfg0["pad_w"] << 24)
        | (cfg0["pad_h"] << 16)
        | (cfg0["stride_w"] << 8)
        | cfg0["stride_h"]
    )
    conv_dilation = (cfg0["dilation_w"] << 8) | cfg0["dilation_h"]

    l0_quant_cfg = quant_cfg_word(True, l0_quant_scale, l0_quant_shift, l0_quant_round)
    l1_quant_cfg = quant_cfg_word(True, l1_quant_scale, l1_quant_shift, l1_quant_round)
    l0_ctrl = make_ctrl("int8", "OS") | 0x100 | 0x200 | activation_ctrl_bits("relu")
    l1_ctrl = make_ctrl("int8", "OS") | 0x200 | activation_ctrl_bits("relu")

    with open(os.path.join(out_dir, "test_params.vh"), "w") as f:
        f.write(f"// Auto-generated: {test_id} (T6.6 two-layer Conv2D E2E)\n")
        f.write(f"`define DRAM_SIZE {max(max_addr + 1, 8192)}\n")
        f.write(f"`define L0_NUM_RESULTS {m0 * n0}\n")
        f.write(f"`define L1_NUM_RESULTS {m1 * n1}\n")
        f.write(f"`define L0_M_DIM {m0}\n")
        f.write(f"`define L0_N_DIM {n0}\n")
        f.write(f"`define L0_K_DIM {k0}\n")
        f.write(f"`define L0_W_ADDR 32'h{w0_base:08x}\n")
        f.write(f"`define L0_A_ADDR 32'h{a0_base:08x}\n")
        f.write(f"`define L0_R_ADDR 32'h{r0_base:08x}\n")
        f.write(f"`define L0_BIAS_ADDR 32'h{bias0_base:08x}\n")
        f.write(f"`define L0_CTRL 32'h{l0_ctrl:08x}\n")
        f.write(f"`define L0_QUANT_CFG 32'h{l0_quant_cfg:08x}\n")
        f.write(f"`define L0_CONV_IFM_SHAPE 32'h{conv_ifm_shape:08x}\n")
        f.write(f"`define L0_CONV_CHANNELS 32'h{conv_channels:08x}\n")
        f.write(f"`define L0_CONV_KERNEL 32'h{conv_kernel:08x}\n")
        f.write(f"`define L0_CONV_OUT_SHAPE 32'h{conv_out_shape:08x}\n")
        f.write(f"`define L0_CONV_STRIDE_PAD 32'h{conv_stride_pad:08x}\n")
        f.write(f"`define L0_CONV_DILATION 32'h{conv_dilation:08x}\n")
        f.write(f"`define L1_M_DIM {m1}\n")
        f.write(f"`define L1_N_DIM {n1}\n")
        f.write(f"`define L1_K_DIM {k1}\n")
        f.write(f"`define L1_W_ADDR 32'h{w1_base:08x}\n")
        f.write(f"`define L1_A_ADDR 32'h{r0_base:08x}\n")
        f.write(f"`define L1_R_ADDR 32'h{r1_base:08x}\n")
        f.write(f"`define L1_BIAS_ADDR 32'h{bias1_base:08x}\n")
        f.write(f"`define L1_CTRL 32'h{l1_ctrl:08x}\n")
        f.write(f"`define L1_QUANT_CFG 32'h{l1_quant_cfg:08x}\n")

    with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
        f.write(f"test_id={test_id}\n")
        f.write("layer0=conv2d_otf_int8_os_bias_relu_quant\n")
        f.write("layer1=conv2d_1x1_direct_int8_os_bias_relu_quant\n")
        f.write(f"layer0_r_addr=0x{r0_base:08x}\n")
        f.write(f"layer1_a_addr=0x{r0_base:08x}\n")
        f.write(f"layer1_r_addr=0x{r1_base:08x}\n")

    print(f"  {test_id:34s} T6.6 two-layer Conv2D E2E")
    print(f"    L0: IFM 4x4x1, KHxKW=3x3, Cout=1 -> {m0}x1 quantized OFM")
    print(f"    L1: 1x1 Conv consumes L0 R_ADDR=0x{r0_base:08x} as A_ADDR")
    print(f"    Final expected results={len(expected1)}")
    print("    First layer0 outputs:", " ".join(str(signed32(v)) for v in expected0[:8]))
    print("    First final outputs:", " ".join(str(signed32(v)) for v in expected1[:8]))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate T6.6 two-layer Conv2D E2E data")
    parser.add_argument("--test-id", default="conv2d_two_layer_int8_os_default")
    args = parser.parse_args()

    print(f"Generating T6.6 two-layer Conv2D E2E data in:\n  {THIS_DIR}\n")
    generate(args.test_id)
    print(f"\nDone. Tests generated in: {THIS_DIR}")


if __name__ == "__main__":
    main()
