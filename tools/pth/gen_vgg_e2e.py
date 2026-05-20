#!/usr/bin/env python3
"""RepOpt VGG e2e: 9-layer 1024-tile NPU chain + avgpool + classifier.

NPU: 1024 tiles (all spatial × channel), hw bias+ReLU+requant.
CPU: avgpool (stage4_1 → 512 features) → classifier → argmax → prediction.
Features flow: NPU R results → avgpool → FEAT_BASE → classifier → prediction.
"""
import argparse, json, os, sys, warnings, torch
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
TB_DIR = PROJECT_ROOT / "tb"
sys.path.insert(0, str(TB_DIR)); sys.path.insert(0, str(THIS_DIR))
from assemble_soc_test import *
from gen_repopt_tile_case import *
from run_repopt_vgg_host import (
    load_json, tensor_scalar, unwrap_state_dict,
    load_int32_hex, quantize_qint8, requant_qint8,
    conv2d_acc_npu, maxpool2d_cpu, adaptive_avgpool2d_cpu,
    load_cifar_sample, load_image_input,
)

SIMD=4; TR=16; TC=16; PPB=64; DRAM=2*1024*1024; NPU=0x02000000
QUANT_SHIFT=16
kt=(PPB*4)//max(TR,TC)

def quant_cfg_for(mults, start, count):
    avg = sum(mults[start:start+count]) / count
    scale = int(round(avg * (1 << QUANT_SHIFT)))
    scale = max(-32768, min(32767, scale))
    return (scale << 16) | (QUANT_SHIFT << 8) | 3

def pack4(vals):
    w = 0
    for i, v in enumerate(vals): w |= (int(v) & 0xFF) << (8 * i)
    return w & 0xFFFFFFFF

def pack_tile_stream(mat, dim1, dim2, kt_elems):
    wpk = (dim1 + SIMD - 1) // SIMD
    out = []; kp = 0
    while kp < dim2:
        kl = min(dim2 - kp, kt_elems)
        for kr in range(kl):
            k = kp + kr
            for w in range(wpk):
                vals = [mat[k][w*SIMD+r] if w*SIMD+r < dim1 else 0 for r in range(SIMD)]
                out.append(pack4(vals))
        pk = ((kl + SIMD - 1) // SIMD) * SIMD
        for p in range(kl * wpk, pk * wpk): out.append(0)
        kp += kl
    return out

def write_hex(path, words):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for w in words: f.write(f"{int(w) & 0xFFFFFFFF:08x}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", default="RepOpt/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth")
    ap.add_argument("--plan", default="sim/pth_repopt_probe/model_plan.json")
    ap.add_argument("--spec", default="tools/pth/examples/repopt_vgg_int8_spec.json")
    ap.add_argument("--data-root", default="RepOpt/06_RepOpt_VGG/data")
    ap.add_argument("--out-dir", default="sim/vgg_e2e")
    ap.add_argument("--img-idx", type=int, default=0)
    ap.add_argument("--image", default="", help="Arbitrary RGB image path (overrides --img-idx)")
    ap.add_argument("--image-size", type=int, default=32)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    plan = load_json(args.plan); spec = load_json(args.spec)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
        ckpt = torch.load(args.pth, map_location="cpu", weights_only=False)
    sd = unwrap_state_dict(ckpt)

    if args.image:
        x_f, label = load_image_input(args.image, args.image_size), None
    else:
        x_f, label = load_cifar_sample(args.data_root, args.img_idx)
    in_sc = float(tensor_scalar(sd[plan["input"]["scale_key"]]))
    in_zp = int(tensor_scalar(sd[plan["input"]["zero_point_key"]]))
    x_q = quantize_qint8(x_f, in_sc, in_zp)

    spec_by_name = {l["name"]: l for l in spec["layers"]}
    conv_layers = [l for l in plan["layers"] if l["op"] == "conv2d"]
    bias_dir = Path(args.plan).resolve().parent

    def merge_spec(cl_p):
        nm = cl_p["name"]; cl_s = spec_by_name.get(nm, {})
        for k in ("stride", "padding", "dilation"): cl_p[k] = cl_s.get(k, cl_p.get(k, 1))
        return cl_p

    # DRAM layout
    W_BASE = 0x1000; A_BASE = 0x40000; R_BASE = 0x80000; B_BASE = 0x2000
    FEAT_BASE = 0x600000; CLS_W_BASE = 0x602000; CLS_B_BASE = 0x610000
    SCORE_BASE = 0x611000; MARKER = 0x612000; LABEL_ADDR = 0x613000
    TILE_TABLE_BASE = 0x3000
    dram = [0] * DRAM

    tile_table = []; next_w = W_BASE; next_a = A_BASE; next_r = R_BASE; next_b = B_BASE
    current = x_q

    for li, cl_r in enumerate(conv_layers):
        cl_p = merge_spec(cl_r.copy())
        M = int(cl_p["registers"]["M_DIM"]); N = int(cl_p["registers"]["N_DIM"])
        K = int(cl_p["registers"]["K_DIM"]); qw = sd[cl_p["weight_key"]]
        bk = cl_p.get("bias_key", ""); bias = [0] * N
        if bk and bk in sd:
            for i in range(min(N, sd[bk].shape[0])): bias[i] = int(sd[bk][i].item())
        n_tiles = (N + TC - 1) // TC; m_tiles = (M + TR - 1) // TR
        mults = cl_p["cpu_requant_after_npu"]["multipliers"]

        w_addrs = []; qcfgs = []
        for ni in range(n_tiles):
            nb_act = min(TC, N - ni * TC)
            _, wt = build_tile_matrices(current, qw, cl_p, 0, ni * TC, TR, nb_act)
            wp = pack_tile_stream(wt, nb_act, K, kt)
            for i, w in enumerate(wp): dram[(next_w >> 2) + i] = w
            w_addrs.append(next_w); next_w += len(wp) * 4 + 0x100
            qcfgs.append(quant_cfg_for(mults, ni * TC, TC))

        bias_addr = next_b
        for i, b in enumerate(bias): dram[(next_b >> 2) + i] = b & 0xFFFFFFFF
        next_b += len(bias) * 4 + 0x100

        for ni in range(n_tiles):
            nb_act = min(TC, N - ni * TC)
            for mi in range(m_tiles):
                mb_act = min(TR, M - mi * TR)
                at, _ = build_tile_matrices(current, qw, cl_p, mi * TR, ni * TC, mb_act, nb_act)
                ap = pack_tile_stream(at, mb_act, K, kt)
                for i, w in enumerate(ap): dram[(next_a >> 2) + i] = w
                tile_table.append([next_a, next_r, w_addrs[ni], bias_addr,
                                   mb_act, nb_act, K, qcfgs[ni], 0x80])
                next_a += len(ap) * 4 + 0x100; next_r += mb_act * nb_act * 4

        current = requant_qint8(
            conv2d_acc_npu(current, qw, load_int32_hex(bias_dir / cl_p["assets"]["bias_int32_hex"]), cl_p),
            cl_p["cpu_requant_after_npu"]["multipliers"],
            cl_p["cpu_requant_after_npu"]["output_zero_point"])

        if cl_p["name"] in ("stage1_1_conv", "stage2_1_conv", "stage3_2_conv"):
            pool_layer = [l for l in plan["layers"] if l["op"] == "maxpool2d" and
                         l["name"].startswith(cl_p["name"].split("_")[0])][0]
            current = maxpool2d_cpu(current, pool_layer)
        next_a = A_BASE + 0x1000

    total_tiles = len(tile_table)
    for i, e in enumerate(tile_table):
        for j, v in enumerate(e):
            dram[(TILE_TABLE_BASE >> 2) + i * 9 + j] = v & 0xFFFFFFFF

    # Stage4_1 config for avgpool firmware
    S4_NT = 32
    S4_INFO_ADDR = TILE_TABLE_BASE + total_tiles * 9 * 4
    S4_R_BASE = tile_table[total_tiles - S4_NT][1]
    dram[S4_INFO_ADDR >> 2] = S4_R_BASE & 0xFFFFFFFF
    dram[(S4_INFO_ADDR + 4) >> 2] = S4_NT
    dram[(S4_INFO_ADDR + 8) >> 2] = FEAT_BASE & 0xFFFFFFFF

    # Classifier weights and biases in DRAM
    cp = sd["model.classifier._packed_params._packed_params"]
    cls_w = cp[0].int_repr(); cls_b = cp[1]
    cls_bias = [int(cls_b[i].item()) for i in range(10)]

    # Python golden features and prediction
    cur2 = x_q
    for layer in plan["layers"]:
        if layer["op"] == "conv2d":
            cur2 = requant_qint8(conv2d_acc_npu(cur2, sd[layer["weight_key"]],
                load_int32_hex(bias_dir / layer["assets"]["bias_int32_hex"]), layer),
                layer["cpu_requant_after_npu"]["multipliers"],
                layer["cpu_requant_after_npu"]["output_zero_point"])
        elif layer["op"] == "maxpool2d": cur2 = maxpool2d_cpu(cur2, layer)
        elif layer["op"] == "adaptive_avgpool2d": cur2 = adaptive_avgpool2d_cpu(cur2, layer)
        elif layer["op"] == "flatten": cur2 = cur2.reshape(cur2.shape[0], -1); break
        elif layer["op"] == "linear": break
    feat_py = torch.clamp(cur2.reshape(-1).to(torch.float64), -128, 127)

    scores = [cls_bias[c] for c in range(10)]
    for c in range(10):
        for f in range(512): scores[c] += int(feat_py[f].item()) * int(cls_w[c, f].item())
    pred_class = scores.index(max(scores))

    print(f"NPU: {total_tiles} tiles (9 layers)")
    print(f"Classifier: 512 features, Python pred={pred_class}, "
          f"true_label={label}, scores[0]={scores[0]}, scores[pred]={scores[pred_class]}")

    # Store features as 0 (will be filled by avgpool firmware)
    for f in range(512): dram[(FEAT_BASE >> 2) + f] = 0
    for c in range(10):
        for f in range(512): dram[(CLS_W_BASE >> 2) + c * 512 + f] = int(cls_w[c, f].item()) & 0xFFFFFFFF
    for c in range(10): dram[(CLS_B_BASE >> 2) + c] = cls_bias[c] & 0xFFFFFFFF
    dram[LABEL_ADDR >> 2] = pred_class & 0xFFFFFFFF

    lr = tile_table[-1][1]; lc = tile_table[-1][4] * tile_table[-1][5]
    expected_last = [dram[(lr >> 2) + i] for i in range(lc)]

    write_hex(out / "dram_init.hex", dram)
    write_hex(out / "expected.hex", expected_last)

    # Use proven firmware template
    fw_path = THIS_DIR / "vgg_fw_template.hex"
    with open(fw_path) as f:
        ins = [int(l.strip(), 16) for l in f if l.strip()]
    fw = len(ins)
    write_hex(out / "soc_vgg.hex", ins)

    op = out.resolve().as_posix()
    with open(out / "soc_vgg_params.vh", "w") as f:
        f.write(f'`define VGG_FW_HEX "{op}/soc_vgg.hex"\n')
        f.write(f'`define VGG_DRAM_HEX "{op}/dram_init.hex"\n')
        f.write(f'`define VGG_EXPECTED_HEX "{op}/expected.hex"\n')
        f.write(f'`define VGG_FW_WORDS {fw}\n')
        f.write(f'`define VGG_MARKER_ADDR 32\'h{MARKER:08x}\n')
        f.write(f'`define VGG_R_ADDR 32\'h{lr:08x}\n')
        f.write(f'`define VGG_RESULT_COUNT {lc}\n')
        f.write(f'`define VGG_LABEL_ADDR 32\'h{LABEL_ADDR:08x}\n')
        f.write(f'`define VGG_LABEL {pred_class}\n')
        f.write(f'`define VGG_TIMEOUT_CYCLES 50000000\n')
        f.write(f'`define VGG_DRAM_WORDS {DRAM}\n')

    print(f"\nGenerated: {out}, {fw} words")
    print(f"  Full pipeline: 1024 tiles + avgpool + classifier + argmax")

if __name__ == "__main__": main()
