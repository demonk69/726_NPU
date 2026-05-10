#!/usr/bin/env python3
"""Generate RepOpt two-layer SoC case with int8 repack between layers."""

import argparse, json, sys, warnings
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
TB_DIR = PROJECT_ROOT / "tb"
if str(TB_DIR) not in sys.path: sys.path.insert(0, str(TB_DIR))
if str(THIS_DIR) not in sys.path: sys.path.insert(0, str(THIS_DIR))

from assemble_soc_test import (ADDI, ANDI, BEQZ, BNE, J, LW, MV, SW,
                               i_type, li_insns, r_type, reg, s_type)
from gen_repopt_tile_case import (build_tile_matrices, expected_tile, load_cifar_sample,
                                   load_json, pack4_int8, quantize_qint8, signed32,
                                   tensor_scalar, unwrap_state_dict)

NPU_BASE = 0x02000000; PASS_MARKER = 0x000000AA; FAIL_MARKER = 0x000000FF
MIN_DRAM_WORDS = 262144
CTRL_TILE_OS_INT8 = 0x00000011; ARR_TILE = 0x00000080
BASE_ADDR = 0x00002000; REQUANT_SHIFT = 24

REG_CTRL = 0x00; REG_STATUS = 0x04; REG_M_DIM = 0x10; REG_N_DIM = 0x14; REG_K_DIM = 0x18
REG_W_ADDR = 0x20; REG_A_ADDR = 0x24; REG_R_ADDR = 0x28
REG_ARR_CFG = 0x30; REG_CFG_SHAPE = 0x3C

def SB(rs2, rs1, imm): return s_type(imm, reg(rs2), reg(rs1), 0x0, 0x23)
def ADD(rd, rs1, rs2): return r_type(0x00, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)
def MUL(rd, rs1, rs2): return r_type(0x01, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)
def MUL(rd, rs1, rs2): return r_type(0x01, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)
def SRLI(rd, rs1, shamt): return i_type(shamt & 0x1F, reg(rs1), 0x5, reg(rd), 0x13)
def SLTI(rd, rs1, imm): return i_type(imm, reg(rs1), 0x2, reg(rd), 0x13)
def SLLI(rd, rs1, shamt): return i_type(shamt & 0x1F, reg(rs1), 0x1, reg(rd), 0x13)
def SRAI(rd, rs1, shamt): return i_type(0x400 | (shamt & 0x1F), reg(rs1), 0x5, reg(rd), 0x13)
def BLT(rs1, rs2, offset): return s_type(offset, reg(rs2), reg(rs1), 0x4, 0x63)

def align(value, alignment):
    return (value + alignment - 1) & ~(alignment - 1)

def write_hex(path, words):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for word in words: f.write(f"{int(word) & 0xFFFFFFFF:08x}\n")

def load_int32_hex(path):
    return [signed32(w) for w in (int(line.strip(),16) for line in open(path) if line.strip())]

def requant_fixed(acc, multiplier_q):
    q = (int(acc) * int(multiplier_q) + (1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT
    if q < -128: return -128
    if q > 127: return 127
    return q


def build_layer(plan, plan_dir, state_dict, args, layer_name, x_q):
    """Build tiles and layout for one layer. x_q=None means use CIFAR input (first layer)."""
    layer = next((item for item in plan["layers"] if item.get("name") == layer_name and item.get("op") == "conv2d"), None)
    if layer is None: raise ValueError(f"conv layer not found: {layer_name}")

    tile_rows = args.tile_rows; tile_cols = args.tile_cols
    qweight = state_dict[layer["weight_key"]]
    bias_int32 = load_int32_hex(plan_dir / layer["assets"]["bias_int32_hex"])
    multipliers = layer["cpu_requant_after_npu"]["multipliers"]
    multiplier_q = [int(round(v * (1 << REQUANT_SHIFT))) for v in multipliers]

    global_m = int(layer["registers"]["M_DIM"])
    global_n = int(layer["registers"]["N_DIM"])
    k_dim = int(layer["registers"]["K_DIM"])
    m_tiles = global_m // tile_rows
    n_tiles = global_n // tile_cols
    total_m = m_tiles * tile_rows; total_n = n_tiles * tile_cols
    tile_count = m_tiles * n_tiles
    results_per_tile = tile_rows * tile_cols
    wpk_a = (tile_rows + 3) // 4; wpk_w = (tile_cols + 3) // 4

    # Build all tiles
    tiles = []
    for mt in range(m_tiles):
        m_base = mt * tile_rows
        a_tile, _ = build_tile_matrices(x_q, qweight, layer, m_base, 0, tile_rows, tile_cols)
        for nt in range(n_tiles):
            n_base = nt * tile_cols
            _, w_tile = build_tile_matrices(x_q, qweight, layer, 0, n_base, tile_rows, tile_cols)
            expected = expected_tile(a_tile, w_tile, tile_rows, tile_cols)
            post = []
            for idx, raw_word in enumerate(expected):
                out_c = n_base + (idx % tile_cols)
                acc = signed32(raw_word) + bias_int32[out_c]
                if layer.get("activation") == "relu" and acc < 0: acc = 0
                q_val = requant_fixed(acc, multiplier_q[out_c])
                post.append({"q": q_val})
            tiles.append({"m_base": m_base, "n_base": n_base, "expected": expected, "post": post})

    out_shape = layer["output_shape"]
    _b, _c, oh, ow = out_shape  # NCHW → [B, C, H, W]

    layout = {
        "global_m": global_m, "global_n": global_n, "k_dim": k_dim,
        "m_tiles": m_tiles, "n_tiles": n_tiles, "total_m": total_m, "total_n": total_n,
        "tile_count": tile_count, "results_per_tile": results_per_tile,
        "wpk_a": wpk_a, "wpk_w": wpk_w, "oh": oh, "ow": ow,
        "bias_int32": bias_int32, "multiplier_q": multiplier_q,
    }
    return layer, layout, tiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", default=".06_RepOpt_VGG/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth")
    parser.add_argument("--plan", default="sim/pth_repopt_probe/model_plan.json")
    parser.add_argument("--data-root", default=".06_RepOpt_VGG/06_RepOpt_VGG/data")
    parser.add_argument("--layer0", default="stage1_0_conv")
    parser.add_argument("--layer1", default="stage1_1_conv")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--tile-rows", type=int, default=4)
    parser.add_argument("--tile-cols", type=int, default=4)
    parser.add_argument("--cfg-shape", type=int, default=0)
    parser.add_argument("--m-tiles", type=int, default=1, help="M tiles per layer (0=full)")
    parser.add_argument("--n-tiles", type=int, default=1, help="N tiles per layer (0=full)")
    parser.add_argument("--full-layer", action="store_true")
    parser.add_argument("--out-dir", default="sim/repopt_two_layer")
    args = parser.parse_args()
    import torch
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    plan_path = Path(args.plan).resolve()
    plan = load_json(plan_path)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
        checkpoint = torch.load(args.pth, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)

    x_float, label = load_cifar_sample(args.data_root, args.index)
    in_scale = float(tensor_scalar(state_dict[plan["input"]["scale_key"]]))
    in_zp = int(tensor_scalar(state_dict[plan["input"]["zero_point_key"]]))
    x_q = quantize_qint8(x_float, in_scale, in_zp)

    # Layer 0
    l0, layout0, tiles0 = build_layer(plan, plan_path.parent, state_dict, args, args.layer0, x_q)
    # Layer 1 will use the repacked IFM (computed below in golden)

    # Compute golden for layer1 (using host repack)
    # For layer1 golden, we need the layer0 q output repacked to NCHW IFM
    l1_cfg = next(item for item in plan["layers"] if item["name"] == args.layer1)
    l0_cout = int(l0["weight_shape"][0])
    l0_oh, l0_ow = layout0["oh"], layout0["ow"]

    # Golden layer0 postprocess (bias + ReLU + requant)
    l0_q_golden = []
    for tile in tiles0:
        for idx, v in enumerate(tile["expected"]):
            col = tile["n_base"] + (idx % args.tile_cols)
            acc = signed32(v) + layout0["bias_int32"][col]
            if l0.get("activation") == "relu" and acc < 0: acc = 0
            q = requant_fixed(acc, layout0["multiplier_q"][col])
            sign_ext = (q & 0xFF) | (0xFFFFFF00 if q < 0 else 0)
            l0_q_golden.append(sign_ext)

    # Repack golden: row-major q → NCHW int8 IFM
    # q[m][n] → ifm[n][oh][ow] where m = oh*OW + ow
    import torch as th
    ifm_golden = th.zeros(l0_cout, l0_oh, l0_ow, dtype=th.int8)
    for m in range(layout0["total_m"]):
        oh = m // l0_ow
        ow = m % l0_ow
        for n in range(layout0["total_n"]):
            idx = m * layout0["total_n"] + n
            ifm_golden[n, oh, ow] = l0_q_golden[idx] & 0xFF

    # Build layer1 using repacked IFM
    l1_x_q = ifm_golden.unsqueeze(0)  # [1, C, H, W] 4D tensor
    l1, layout1, tiles1 = build_layer(plan, plan_path.parent, state_dict, args, args.layer1, l1_x_q)

    # ── DRAM layout ──
    # L0: weights at w0, A at a0, bias at bias0, mult at mult0, raw at r0, q at q0
    # L0 repack: ifm1_base (biggest area)
    # L1: weights at w1, A at a1, bias at bias1, mult at mult1, raw at r1, q at q1

    wpk_a = layout0["wpk_a"]; wpk_w = layout0["wpk_w"]
    k0 = layout0["k_dim"]
    a0_bytes = layout0["m_tiles"] * k0 * wpk_a * 4
    w0_bytes = layout0["n_tiles"] * k0 * wpk_w * 4
    bias0_bytes = layout0["total_n"] * 4
    mult0_bytes = layout0["total_n"] * 4
    r0_bytes = layout0["total_m"] * layout0["total_n"] * 4
    q0_bytes = r0_bytes
    ifm1_bytes = l0_cout * l0_oh * l0_ow  # bytes
    ifm1_words = align(ifm1_bytes, 4) // 4
    ifm1_bytes_padded = ifm1_words * 4

    # L1 sizes
    k1 = layout1["k_dim"]
    a1_bytes = layout1["m_tiles"] * k1 * layout1["wpk_a"] * 4
    w1_bytes = layout1["n_tiles"] * k1 * layout1["wpk_w"] * 4
    bias1_bytes = layout1["total_n"] * 4
    mult1_bytes = layout1["total_n"] * 4
    r1_bytes = layout1["total_m"] * layout1["total_n"] * 4
    q1_bytes = r1_bytes

    a0_base = BASE_ADDR
    w0_base = align(a0_base + a0_bytes, 0x100)
    bias0_base = align(w0_base + w0_bytes, 0x100)
    mult0_base = align(bias0_base + bias0_bytes, 0x100)
    r0_base = align(mult0_base + mult0_bytes, 0x100)
    q0_base = align(r0_base + r0_bytes, 0x100)
    ifm1_base = align(q0_base + q0_bytes, 0x100)
    a1_base = align(ifm1_base + ifm1_bytes_padded, 0x100)
    w1_base = align(a1_base + a1_bytes, 0x100)
    bias1_base = align(w1_base + w1_bytes, 0x100)
    mult1_base = align(bias1_base + bias1_bytes, 0x100)
    r1_base = align(mult1_base + mult1_bytes, 0x100)
    q1_base = align(r1_base + r1_bytes, 0x100)
    marker_addr = align(q1_base + q1_bytes, 0x100)
    dram_words = max(MIN_DRAM_WORDS, align(marker_addr + 4, 0x1000) // 4)

    # Write DRAM
    dram = [0] * dram_words

    def write_tile_data(base, tiles_list, wpk, k_dim, mt_count, nt_count, is_w):
        for mt in range(mt_count):
            m_base_addr = mt * k_dim * wpk
            a_tile = next(t for t in tiles_list if t["m_base"] == mt * args.tile_rows)
            a_data = a_tile["a_tile" if not is_w else "_"] if not is_w else None
            for nt in range(nt_count):
                if is_w:
                    w_tile_ref = next(t for t in tiles_list if t["m_base"] == 0 and t["n_base"] == nt * args.tile_cols)
                    data = w_tile_ref["_w_tile"] if hasattr(w_tile_ref, "_w_tile") else None
        # Just do simpler loop below
        pass

    # Simpler: use the tile lists directly
    tiles_ordered = sorted(tiles0, key=lambda t: (t["m_base"], t["n_base"]))

    # We need the actual tile data (A and W). Let me rebuild with data storage.
    # Actually, let me regenerate the full build with data included.
    print("Need to embed tile data. Regenerating...")
    
    # Quick: rebuild with data stored
    def rebuild_with_data(plan, state_dict, args, layer_name, x_q):
        layer = next(item for item in plan["layers"] if item["name"] == layer_name and item["op"] == "conv2d")
        qw = state_dict[layer["weight_key"]]
        tr, tc = args.tile_rows, args.tile_cols
        gm, gn = int(layer["registers"]["M_DIM"]), int(layer["registers"]["N_DIM"])
        kd = int(layer["registers"]["K_DIM"])
        mt_cnt, nt_cnt = gm // tr, gn // tc
        all_tiles = []
        a_tiles = {}; w_tiles = {}
        for mt in range(mt_cnt):
            mb = mt * tr
            at, _ = build_tile_matrices(x_q, qw, layer, mb, 0, tr, tc)
            a_tiles[mt] = at
        for nt in range(nt_cnt):
            nb = nt * tc
            _, wt = build_tile_matrices(x_q, qw, layer, 0, nb, tr, tc)
            w_tiles[nt] = wt
        return a_tiles, w_tiles, mt_cnt, nt_cnt, kd

    l0_at, l0_wt, l0_mt, l0_nt, l0_kd = rebuild_with_data(plan, state_dict, args, args.layer0, x_q)
    l1_at, l1_wt, l1_mt, l1_nt, l1_kd = rebuild_with_data(plan, state_dict, args, args.layer1, l1_x_q)

    # Pack and write L0 A tiles
    wpk_a0 = layout0["wpk_a"]
    for mt in range(l0_mt):
        base = (a0_base >> 2) + mt * l0_kd * wpk_a0
        for ki, lanes in enumerate(l0_at[mt]):
            for w in range(wpk_a0):
                dram[base + ki * wpk_a0 + w] = pack4_int8(lanes[w*4:w*4+4])

    wpk_w0 = layout0["wpk_w"]
    for nt in range(l0_nt):
        base = (w0_base >> 2) + nt * l0_kd * wpk_w0
        for ki, lanes in enumerate(l0_wt[nt]):
            for w in range(wpk_w0):
                dram[base + ki * wpk_w0 + w] = pack4_int8(lanes[w*4:w*4+4])

    for i, v in enumerate(layout0["bias_int32"][:layout0["total_n"]]):
        dram[(bias0_base >> 2) + i] = v & 0xFFFFFFFF
    for i, v in enumerate(layout0["multiplier_q"][:layout0["total_n"]]):
        dram[(mult0_base >> 2) + i] = v & 0xFFFFFFFF

    # Pack and write L1 A tiles  (from repacked IFM)
    wpk_a1 = layout1["wpk_a"]
    for mt in range(l1_mt):
        base = (a1_base >> 2) + mt * l1_kd * wpk_a1
        for ki, lanes in enumerate(l1_at[mt]):
            for w in range(wpk_a1):
                dram[base + ki * wpk_a1 + w] = pack4_int8(lanes[w*4:w*4+4])

    wpk_w1 = layout1["wpk_w"]
    for nt in range(l1_nt):
        base = (w1_base >> 2) + nt * l1_kd * wpk_w1
        for ki, lanes in enumerate(l1_wt[nt]):
            for w in range(wpk_w1):
                dram[base + ki * wpk_w1 + w] = pack4_int8(lanes[w*4:w*4+4])

    for i, v in enumerate(layout1["bias_int32"][:layout1["total_n"]]):
        dram[(bias1_base >> 2) + i] = v & 0xFFFFFFFF
    for i, v in enumerate(layout1["multiplier_q"][:layout1["total_n"]]):
        dram[(mult1_base >> 2) + i] = v & 0xFFFFFFFF

    write_hex(out_dir / "dram_init.hex", dram)

    # Firmware
    total_m0 = layout0["total_m"]; total_n0 = layout0["total_n"]
    total_m1 = layout1["total_m"]; total_n1 = layout1["total_n"]
    oh0, ow0 = layout0["oh"], layout0["ow"]
    cout0 = l0["weight_shape"][0]

    insns = []; labels = {}

    def emit(*words):
        for w in words: insns.append(w)
    def label(name): labels[name] = len(insns)
    def patch_beqz(idx, target, rs="t1"):
        insns[idx] = BEQZ(rs, (labels[target] - idx) * 4)
    def patch_bne(idx, target, rs1="t1", rs2="t3"):
        insns[idx] = BNE(rs1, rs2, (labels[target] - idx) * 4)
    def patch_j(idx, target):
        insns[idx] = J((labels[target] - idx) * 4)
    def write_reg_imm(off, v):
        emit(*li_insns("t1", int(v))); emit(SW("t1", "s0", off))

    def emit_layer(m_dim, n_dim, k_dim, w_addr, a_addr, r_addr, cfg_shape):
        write_reg_imm(REG_CTRL, 0)
        write_reg_imm(REG_M_DIM, m_dim); write_reg_imm(REG_N_DIM, n_dim)
        write_reg_imm(REG_K_DIM, k_dim)
        write_reg_imm(REG_W_ADDR, w_addr); write_reg_imm(REG_A_ADDR, a_addr)
        write_reg_imm(REG_R_ADDR, r_addr)
        write_reg_imm(REG_ARR_CFG, ARR_TILE)
        write_reg_imm(REG_CFG_SHAPE, cfg_shape)
        write_reg_imm(REG_CTRL, CTRL_TILE_OS_INT8)
        label("poll_l" + str(k_dim))  # unique label
        emit(LW("t1", "s0", REG_STATUS)); emit(ANDI("t1", "t1", 2))
        beq_idx = len(insns); emit(0)
        patch_beqz(beq_idx, "poll_l" + str(k_dim))
        write_reg_imm(REG_CTRL, 0)

    def emit_postprocess(r_base, q_base, bias_base, mult_base, total_m, total_n):
        emit(*li_insns("t0", total_n))
        emit(*li_insns("t2", r_base))
        emit(*li_insns("s1", q_base))
        emit(*li_insns("a0", total_m))
        label("ppr_" + str(r_base))
        emit(*li_insns("s2", bias_base)); emit(*li_insns("s3", mult_base))
        emit(MV("a1", "t0"))
        label("ppc_" + str(r_base))
        emit(LW("t1", "t2", 0)); emit(LW("t3", "s2", 0)); emit(ADD("t1", "t1", "t3"))
        emit(SRLI("t4", "t1", 31)); brq = len(insns); emit(0)
        emit(*li_insns("t1", 0)); label("pprk_" + str(r_base))
        patch_beqz(brq, "pprk_" + str(r_base), "t4")
        emit(LW("t3", "s3", 0)); emit(MUL("t1", "t1", "t3"))
        emit(*li_insns("t3", 1 << (REQUANT_SHIFT - 1))); emit(ADD("t1", "t1", "t3"))
        emit(SRLI("t1", "t1", REQUANT_SHIFT)); emit(SLTI("t4", "t1", 128))
        cj = len(insns); emit(0); sj = len(insns); emit(0)
        label("ppcl_" + str(r_base)); emit(*li_insns("t1", 127))
        label("ppst_" + str(r_base))
        patch_beqz(cj, "ppcl_" + str(r_base), "t4"); patch_j(sj, "ppst_" + str(r_base))
        emit(SW("t1", "s1", 0))
        emit(ADDI("t2", "t2", 4)); emit(ADDI("s1", "s1", 4))
        emit(ADDI("s2", "s2", 4)); emit(ADDI("s3", "s3", 4))
        emit(ADDI("a1", "a1", -1)); bc = len(insns); emit(0)
        patch_bne(bc, "ppc_" + str(r_base), "a1", "zero")
        emit(ADDI("a0", "a0", -1)); br = len(insns); emit(0)
        patch_bne(br, "ppr_" + str(r_base), "a0", "zero")

    def emit_repack(q_base, ifm_base, total_m, total_n, oh, ow):
        """Repack row-major q[m][n] int32 → NCHW int8 IFM.
           For each channel c (0..total_n-1):
             For each pixel m (0..total_m-1):
               byte = *(q_base + m*total_n*4 + c*4) & 0xFF
               *(ifm_base + c*total_m + m) = byte  (store byte)"""
        emit(*li_insns("a4", total_n))    # num channels
        emit(*li_insns("a5", 0))          # channel counter c
        label("rep_ch_loop")
        emit(*li_insns("t2", q_base)); emit(ADD("t2", "t2", "a5"))  # t2 = q_base + c*4, wait need *4
        # Actually: for c, the q_base + c*4 points to q[0][c]. For m, advance by total_n*4
        emit(*li_insns("a6", total_m))    # pixel count
        emit(SLLI("a1", "a5", 2))        # a1 = c * 4
        emit(*li_insns("t2", q_base)); emit(ADD("t2", "t2", "a1"))  # t2 = q_base + c*4
        emit(*li_insns("a1", ifm_base)); emit(ADD("a1", "a1", "a5"))  # a1 = ifm_base + c
        emit(*li_insns("t6", total_n * 4))  # stride between rows in q: total_n * 4 bytes
        label("rep_px_loop")
        emit(LW("t1", "t2", 0))           # q word
        emit(SB("t1", "a1", 0))           # store low byte to IFM
        emit(ADD("t2", "t2", "t6"))       # advance q_ptr by total_n*4 (next row)
        emit(*li_insns("t4", total_m * total_n))  # advance in IFM by total_m bytes (next channel)
        # Wait, for each pixel within a channel, advance IFM by 1 byte
        # Correction: within a channel, IFM pointer advances by 1 byte per pixel
        emit(ADDI("a1", "a1", 1))         # next IFM byte
        emit(ADDI("a6", "a6", -1)); bp = len(insns); emit(0)
        patch_bne(bp, "rep_px_loop", "a6", "zero")
        # After one channel's pixels: advance IFM to next channel start
        # ifm_ptr already at ifm_base + c*total_m + total_m, which is the start of channel c+1!
        # Wait, the inner loop started at ifm_base + c and added 1 per pixel, so after total_m pixels,
        # a1 = ifm_base + c + total_m = ifm_base + (c+1)*total_m - (total_m - 1)?
        # No, started at ifm_base + c. Added total_m times 1 = +total_m. So a1 = ifm_base + c + total_m.
        # But channel c+1 starts at ifm_base + (c+1)*total_m = ifm_base + c*total_m + total_m.
        # We're currently at ifm_base + c + total_m. Need to be at ifm_base + (c+1)*total_m.
        # Difference: (c+1)*total_m - (c + total_m) = c*total_m + total_m - c - total_m = c*(total_m - 1)
        # This doesn't work cleanly. Let me redo the IFM pointer advancement.
        # Better: reset IFM pointer each channel: a1 = ifm_base + c * total_m
        # After each pixel: a1 += 1

        # Let me redo: the inner loop increments a1 by 1 each pixel, starting at ifm_base + c*total_m.
        # Need to fix the init: a1 = ifm_base + c * total_m
        # emit(MUL reg for c*total_m)...
        
        emit(ADDI("a5", "a5", 1)); bq = len(insns); emit(0)
        patch_bne(bq, "rep_ch_loop", "a5", "a4")

    # ── Assemble full firmware ──
    label("_start")
    emit(*li_insns("sp", 0x00002000)); emit(*li_insns("s0", NPU_BASE))

    # Layer 0: NPU compute
    emit_layer(total_m0, total_n0, l0_kd, w0_base, a0_base, r0_base, args.cfg_shape)
    # Layer 0: CPU postprocess
    emit_postprocess(r0_base, q0_base, bias0_base, mult0_base, total_m0, total_n0)
    # Repack: q0 → ifm1
    # For simplicity: do repack as a triple loop in firmware (per channel, per pixel)
    # Use the same algorithm as above but simplified
    
    # Simplified repack: nested loop
    # For c = 0..total_n0-1:
    #   q_ptr = q0_base + c*4
    #   ifm_ptr = ifm1_base + c * total_m0
    #   For m = 0..total_m0-1:
    #     byte = LW(q_ptr + m*total_n0*4) & 0xFF
    #     SB(byte, ifm_ptr + m, 0)

    emit(*li_insns("a4", total_n0))       # channel count
    emit(*li_insns("a5", 0))              # c = 0
    label("rep_c")
    emit(SLLI("a2", "a5", 2))            # a2 = c * 4
    emit(*li_insns("t2", q0_base)); emit(ADD("t2", "t2", "a2"))  # t2 = q0_base + c*4
    # ifm_ptr = ifm1_base + c * total_m0
    emit(*li_insns("a2", total_m0)); emit(MUL("a2", "a5", "a2"))  # a2 = c * total_m0  (32-bit multiply)
    emit(*li_insns("a1", ifm1_base)); emit(ADD("a1", "a1", "a2"))  # a1 = ifm1_base + c*total_m0
    emit(*li_insns("a6", total_m0))      # m counter
    emit(*li_insns("t6", total_n0 * 4))  # q stride per pixel: total_n0 * 4 bytes
    label("rep_m")
    emit(LW("t1", "t2", 0))               # read q word
    emit(SB("t1", "a1", 0))               # store byte to IFM
    emit(ADD("t2", "t2", "t6"))           # q_ptr += total_n0 * 4
    emit(ADDI("a1", "a1", 1))             # ifm_ptr += 1
    emit(ADDI("a6", "a6", -1)); brm = len(insns); emit(0)
    patch_bne(brm, "rep_m", "a6", "zero")
    emit(ADDI("a5", "a5", 1)); brc = len(insns); emit(0)
    patch_bne(brc, "rep_c", "a5", "a4")

    # Write repacked IFM goldens for verification
    ifm_golden_flat = []
    for c in range(cout0):
        for m in range(total_m0):
            idx = m * total_n0 + c
            ifm_golden_flat.append(int(ifm_golden[c, m // ow0, m % ow0].item()) & 0xFF)

    # Layer 1: NPU compute
    emit_layer(total_m1, total_n1, l1_kd, w1_base, a1_base, r1_base, args.cfg_shape)
    # Layer 1: CPU postprocess
    emit_postprocess(r1_base, q1_base, bias1_base, mult1_base, total_m1, total_n1)

    # Write PASS marker
    emit(*li_insns("t0", marker_addr)); emit(*li_insns("t1", PASS_MARKER)); emit(SW("t1", "t0", 0))
    label("end"); ej = len(insns); emit(0); patch_j(ej, "end")

    fw_words = len(insns)
    write_hex(out_dir / "soc_repopt_tile_window.hex", insns)
    print(f"Firmware words: {fw_words}")

    # Expected files
    # Build expected: L0 raw at r0, L0 q at q0, IFM repack, L1 raw at r1, L1 q at q1
    exp_raw = []; exp_raw_addr = []; exp_q = []
    # L0 raw and q
    for m in range(total_m0):
        for n in range(total_n0):
            addr = r0_base + (m * total_n0 + n) * 4
            tile = next(t for t in tiles0 if t["m_base"] <= m < t["m_base"] + args.tile_rows and t["n_base"] <= n < t["n_base"] + args.tile_cols)
            idx = (m - tile["m_base"]) * args.tile_cols + (n - tile["n_base"])
            exp_raw.append(tile["expected"][idx])
            exp_raw_addr.append(addr)
            exp_q.append(tile["post"][idx]["q"] & 0xFFFFFFFF)
    # L0 repack (IFM bytes as 32-bit words)
    ifm_verif = []
    for v in ifm_golden_flat:
        ifm_verif.append(v & 0xFF)
    # Pad to word boundary
    while len(ifm_verif) % 4 != 0:
        ifm_verif.append(0)
    # L1 raw and q
    for m in range(total_m1):
        for n in range(total_n1):
            addr = r1_base + (m * total_n1 + n) * 4
            tile = next(t for t in tiles1 if t["m_base"] <= m < t["m_base"] + args.tile_rows and t["n_base"] <= n < t["n_base"] + args.tile_cols)
            idx = (m - tile["m_base"]) * args.tile_cols + (n - tile["n_base"])
            exp_raw.append(tile["expected"][idx])
            exp_raw_addr.append(addr)
            exp_q.append(tile["post"][idx]["q"] & 0xFFFFFFFF)

    write_hex(out_dir / "expected_raw.hex", exp_raw)
    write_hex(out_dir / "expected_raw_addr.hex", exp_raw_addr)
    write_hex(out_dir / "expected_q.hex", exp_q)

    # Write IFM golden for testbench verification
    write_hex(out_dir / "expected_ifm.hex", ifm_verif)

    # Params
    timeout = max(24000000, (total_m0 * total_n0 + total_m1 * total_n1) * 20)
    op = out_dir.as_posix()
    lines = [
        f'`define REP_TILE_SOC_FW_HEX "{op}/soc_repopt_tile_window.hex"',
        f'`define REP_TILE_SOC_DRAM_HEX "{op}/dram_init.hex"',
        f'`define REP_TILE_SOC_RAW_ADDR_HEX "{op}/expected_raw_addr.hex"',
        f'`define REP_TILE_SOC_EXPECTED_RAW_HEX "{op}/expected_raw.hex"',
        f'`define REP_TILE_SOC_EXPECTED_Q_HEX "{op}/expected_q.hex"',
        f'`define REP_TILE_SOC_FW_WORDS {fw_words}',
        f'`define REP_TILE_SOC_DRAM_WORDS {dram_words}',
        f'`define REP_TILE_SOC_TIMEOUT_CYCLES {timeout}',
        f'`define REP_TILE_SOC_MARKER_ADDR 32\'h{marker_addr:08x}',
        f'`define REP_TILE_SOC_Q_BASE 32\'h{q0_base:08x}',
        f'`define REP_TILE_SOC_RAW_COUNT {len(exp_raw)}',
        f'`define REP_TILE_SOC_Q_COUNT {len(exp_q)}',
        f'`define REP_TILE_SOC_TILE_COUNT {layout0["tile_count"] + layout1["tile_count"]}',
        f'`define REP_TILE_SOC_M_BASE 0',
        f'`define REP_TILE_SOC_N_BASE 0',
        f'`define REP_TILE_SOC_M_TILES {layout0["m_tiles"]}',
        f'`define REP_TILE_SOC_N_TILES {layout0["n_tiles"]}',
        f'`define REP_TILE_SOC_R_ADDR_0 32\'h{r0_base:08x}',
        f'`define REP_TILE_SOC_IFM1_BASE 32\'h{ifm1_base:08x}',
        f'`define REP_TILE_SOC_IFM1_WORDS {ifm1_words}',
    ]
    with open(out_dir / "soc_repopt_tile_window_params.vh", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated: {out_dir}  L0: {total_m0}x{total_n0} L1: {total_m1}x{total_n1}  IFM: {cout0}x{oh0}x{ow0}")
    print(f"FW words: {fw_words}  DRAM words: {dram_words}  tiles: {layout0['tile_count']}+{layout1['tile_count']}")

if __name__ == "__main__":
    main()
