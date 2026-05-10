#!/usr/bin/env python3
"""Generate RepOpt tile SoC case with global N_DIM/M_DIM (auto-tile-iteration)."""

import argparse, json, sys, warnings
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
TB_DIR = PROJECT_ROOT / "tb"
if str(TB_DIR) not in sys.path: sys.path.insert(0, str(TB_DIR))
if str(THIS_DIR) not in sys.path: sys.path.insert(0, str(THIS_DIR))

from assemble_soc_test import (ADDI, ANDI, BEQZ, BNE, J, LW, MV, SW, i_type, li_insns, r_type, reg, s_type)
from gen_repopt_tile_case import (build_tile_matrices, expected_tile, load_cifar_sample,
                                   load_json, pack4_int8, quantize_qint8, signed32,
                                   tensor_scalar, unwrap_state_dict)

NPU_BASE = 0x02000000
PASS_MARKER = 0x000000AA; FAIL_MARKER = 0x000000FF
MIN_DRAM_WORDS = 262144

REG_CTRL = 0x00; REG_STATUS = 0x04
REG_M_DIM = 0x10; REG_N_DIM = 0x14; REG_K_DIM = 0x18
REG_W_ADDR = 0x20; REG_A_ADDR = 0x24; REG_R_ADDR = 0x28
REG_ARR_CFG = 0x30; REG_CFG_SHAPE = 0x3C

CTRL_TILE_OS_INT8 = 0x00000011
ARR_TILE = 0x00000080
BASE_ADDR = 0x00002000
REQUANT_SHIFT = 24


def ADD(rd, rs1, rs2): return r_type(0x00, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)
def MUL(rd, rs1, rs2): return r_type(0x01, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)
def SRLI(rd, rs1, shamt): return i_type(shamt & 0x1F, reg(rs1), 0x5, reg(rd), 0x13)
def SLTI(rd, rs1, imm): return i_type(imm, reg(rs1), 0x2, reg(rd), 0x13)
def SB(rs2, rs1, imm): return s_type(imm, reg(rs2), reg(rs1), 0x0, 0x23)


def align(value, alignment):
    return (value + alignment - 1) & ~(alignment - 1)

def write_hex(path, words):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for word in words: f.write(f"{int(word) & 0xFFFFFFFF:08x}\n")

def load_hex_words(path):
    words = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: words.append(int(line, 16) & 0xFFFFFFFF)
    return words

def load_int32_hex(path):
    return [signed32(w) for w in load_hex_words(path)]

def requant_fixed(acc, multiplier_q):
    q = (int(acc) * int(multiplier_q) + (1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT
    if q < -128: return -128
    if q > 127: return 127
    return q


def build_case(plan, plan_dir, state_dict, args):
    tile_rows = args.tile_rows; tile_cols = args.tile_cols
    layer = next((item for item in plan["layers"] if item.get("name") == args.layer_name and item.get("op") == "conv2d"), None)
    if layer is None: raise ValueError(f"conv layer not found: {args.layer_name}")

    x_float, label = load_cifar_sample(args.data_root, args.index)
    in_scale = float(tensor_scalar(state_dict[plan["input"]["scale_key"]]))
    in_zp = int(tensor_scalar(state_dict[plan["input"]["zero_point_key"]]))
    x_q = quantize_qint8(x_float, in_scale, in_zp)
    qweight = state_dict[layer["weight_key"]]
    bias_int32 = load_int32_hex(plan_dir / layer["assets"]["bias_int32_hex"])
    multipliers = layer["cpu_requant_after_npu"]["multipliers"]
    multiplier_q = [int(round(v * (1 << REQUANT_SHIFT))) for v in multipliers]

    global_m = int(layer["registers"]["M_DIM"])
    global_n = int(layer["registers"]["N_DIM"])
    k_dim = int(layer["registers"]["K_DIM"])

    if args.full_layer:
        args.m_base = 0; args.n_base = 0
        args.m_tiles = global_m // tile_rows
        args.n_tiles = global_n // tile_cols

    total_m = args.m_tiles * tile_rows
    total_n = args.n_tiles * tile_cols
    tile_count = args.m_tiles * args.n_tiles
    results_per_tile = tile_rows * tile_cols
    words_per_k_a = (tile_rows + 3) // 4
    words_per_k_w = (tile_cols + 3) // 4

    a_total_bytes = args.m_tiles * k_dim * words_per_k_a * 4
    w_total_bytes = args.n_tiles * k_dim * words_per_k_w * 4
    a_base = BASE_ADDR
    w_base = align(a_base + a_total_bytes, 0x100)
    bias_base = align(w_base + w_total_bytes, 0x100)
    multiplier_base = align(bias_base + total_n * 4, 0x100)
    r_base = align(multiplier_base + total_n * 4, 0x100)
    q_base = align(r_base + total_m * total_n * 4, 0x100)
    repack_base = align(q_base + total_m * total_n * 4, 0x100)
    repack_bytes = total_m * total_n
    repack_words = align(repack_bytes, 4) // 4
    marker_addr = align(repack_base + repack_words * 4, 0x100)
    dram_words = max(MIN_DRAM_WORDS, align(marker_addr + 4, 0x1000) // 4)

    # Build all tiles
    tiles = []
    for mt in range(args.m_tiles):
        m_base = args.m_base + mt * tile_rows
        a_tile, _ = build_tile_matrices(x_q, qweight, layer, m_base, args.n_base, tile_rows, tile_cols)
        for nt in range(args.n_tiles):
            n_base = args.n_base + nt * tile_cols
            _, w_tile = build_tile_matrices(x_q, qweight, layer, args.m_base, n_base, tile_rows, tile_cols)
            expected = expected_tile(a_tile, w_tile, tile_rows, tile_cols)
            post = []
            for idx, raw_word in enumerate(expected):
                out_c = n_base + (idx % tile_cols)
                acc = signed32(raw_word) + bias_int32[out_c]
                if layer.get("activation") == "relu" and acc < 0: acc = 0
                q = requant_fixed(acc, multiplier_q[out_c])
                post.append({"out_c": out_c, "acc": acc, "q": q})
            tiles.append({"m_base": m_base, "n_base": n_base, "a_tile": a_tile, "w_tile": w_tile,
                          "expected": expected, "post": post})

    # DRAM layout
    layout = {"a_base": a_base, "w_base": w_base, "bias_base": bias_base,
              "multiplier_base": multiplier_base, "r_base": r_base, "q_base": q_base,
              "repack_base": repack_base, "repack_words": repack_words,
              "marker_addr": marker_addr, "dram_words": dram_words,
              "k_dim": k_dim, "m_dim": global_m, "n_dim": global_n,
              "tile_count": tile_count, "tile_rows": tile_rows, "tile_cols": tile_cols,
              "results_per_tile": results_per_tile, "words_per_k_a": words_per_k_a,
              "words_per_k_w": words_per_k_w, "total_m": total_m, "total_n": total_n,
              "bias_all": bias_int32[:total_n], "mult_all": multiplier_q[:total_n]}
    return layer, label, in_scale, in_zp, tiles, layout


def write_dram_init(out_dir, tiles, layout):
    dram = [0] * layout["dram_words"]
    # Pack A tiles
    wpk_a = layout["words_per_k_a"]
    for mt in range(len({t["m_base"] for t in tiles})):  # unique M tiles
        a_tile_ref = next(t for t in tiles if t["m_base"] == layout["tile_rows"] * mt)
        a_tile = a_tile_ref["a_tile"]
        base = (layout["a_base"] >> 2) + mt * layout["k_dim"] * wpk_a
        for ki, lanes in enumerate(a_tile):
            for w in range(wpk_a):
                dram[base + ki * wpk_a + w] = pack4_int8(lanes[w*4:w*4+4])
    # Pack W tiles
    wpk_w = layout["words_per_k_w"]
    for nt, w_tile_data in enumerate({(t["n_base"] // layout["tile_cols"]): t for t in tiles}.values()):
        w_tile = w_tile_data["w_tile"]
        base = (layout["w_base"] >> 2) + nt * layout["k_dim"] * wpk_w
        for ki, lanes in enumerate(w_tile):
            for w in range(wpk_w):
                dram[base + ki * wpk_w + w] = pack4_int8(lanes[w*4:w*4+4])
    # Bias and multipliers
    for idx, val in enumerate(layout["bias_all"]):
        dram[(layout["bias_base"] >> 2) + idx] = val & 0xFFFFFFFF
    for idx, val in enumerate(layout["mult_all"]):
        dram[(layout["multiplier_base"] >> 2) + idx] = val & 0xFFFFFFFF
    write_hex(out_dir / "dram_init.hex", dram)


def write_expected_files(out_dir, tiles, layout):
    total_m = layout["total_m"]; total_n = layout["total_n"]
    raw_addr, exp_raw, exp_q = [], [], []
    # Build a lookup: (m, n) → (raw, q)
    grid = {}
    for tile in tiles:
        for idx, value in enumerate(tile["expected"]):
            r = idx // layout["tile_cols"]
            c = idx % layout["tile_cols"]
            m = tile["m_base"] + r
            n = tile["n_base"] + c
            grid[(m, n)] = (value, tile["post"][idx]["q"])
    # Output in global row-major order
    for m in range(total_m):
        for n in range(total_n):
            val, q = grid.get((m, n), (0, 0))
            addr = layout["r_base"] + (m * total_n + n) * 4
            raw_addr.append(addr)
            exp_raw.append(val)
            exp_q.append(q & 0xFFFFFFFF)
    write_hex(out_dir / "expected_raw_addr.hex", raw_addr)
    write_hex(out_dir / "expected_raw.hex", exp_raw)
    write_hex(out_dir / "expected_q.hex", exp_q)
    return raw_addr, exp_raw, exp_q


def assemble_firmware_global(layout, args):
    """Firmware that sets global M_DIM/N_DIM and lets the controller iterate all tiles.
       Postprocessing reads from the global row-major layout."""
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

    total_n = layout["total_n"]
    total_m = layout["total_m"]
    bias_base = layout["bias_base"]
    mult_base = layout["multiplier_base"]
    q_base = layout["q_base"]
    r_base = layout["r_base"]
    k_dim = layout["k_dim"]

    label("_start")
    emit(*li_insns("sp", 0x00002000))
    emit(*li_insns("s0", NPU_BASE))

    # Write dimensions: global M, N, K
    write_reg_imm(REG_M_DIM, total_m)
    write_reg_imm(REG_N_DIM, total_n)
    write_reg_imm(REG_K_DIM, k_dim)
    write_reg_imm(REG_W_ADDR, layout["w_base"])
    write_reg_imm(REG_A_ADDR, layout["a_base"])
    write_reg_imm(REG_R_ADDR, r_base)
    write_reg_imm(REG_ARR_CFG, ARR_TILE)
    write_reg_imm(REG_CFG_SHAPE, args.cfg_shape)
    write_reg_imm(REG_CTRL, CTRL_TILE_OS_INT8)

    # Poll for done
    label("poll")
    emit(LW("t1", "s0", REG_STATUS))
    emit(ANDI("t1", "t1", 2))
    beq_idx = len(insns); emit(0)
    patch_beqz(beq_idx, "poll")
    write_reg_imm(REG_CTRL, 0)

    # Postprocess: iterate rows × cols, reset bias/multiplier per row
    emit(*li_insns("t0", total_n))      # N_DIM (cols per row), used for row loop
    emit(*li_insns("t2", r_base))       # raw read pointer
    emit(*li_insns("s1", q_base))       # q write pointer
    emit(*li_insns("a0", total_m))      # remaining rows counter

    label("pp_row_loop")
    emit(*li_insns("s2", bias_base))    # reset bias ptr
    emit(*li_insns("s3", mult_base))    # reset multiplier ptr
    emit(MV("a1", "t0"))               # remaining cols in this row = total_n

    label("pp_col_loop")
    emit(LW("t1", "t2", 0))            # raw
    emit(LW("t3", "s2", 0))            # bias
    emit(ADD("t1", "t1", "t3"))        # +bias
    emit(SRLI("t4", "t1", 31))
    beq_relu = len(insns); emit(0)
    emit(*li_insns("t1", 0))
    label("pp_relu_done")
    patch_beqz(beq_relu, "pp_relu_done", "t4")
    emit(LW("t3", "s3", 0))            # multiplier
    emit(MUL("t1", "t1", "t3"))
    emit(*li_insns("t3", 1 << (REQUANT_SHIFT - 1)))
    emit(ADD("t1", "t1", "t3"))
    emit(SRLI("t1", "t1", REQUANT_SHIFT))
    emit(SLTI("t4", "t1", 128))
    clamp_j = len(insns); emit(0)
    store_j = len(insns); emit(0)
    label("pp_clamp")
    emit(*li_insns("t1", 127))
    label("pp_store")
    patch_beqz(clamp_j, "pp_clamp", "t4")
    patch_j(store_j, "pp_store")
    emit(SW("t1", "s1", 0))

    emit(ADDI("t2", "t2", 4))          # next raw
    emit(ADDI("s1", "s1", 4))          # next q
    emit(ADDI("s2", "s2", 4))          # next bias
    emit(ADDI("s3", "s3", 4))          # next multiplier
    emit(ADDI("a1", "a1", -1))
    bne_col = len(insns); emit(0)
    patch_bne(bne_col, "pp_col_loop", "a1", "zero")

    emit(ADDI("a0", "a0", -1))
    bne_row = len(insns); emit(0)
    patch_bne(bne_row, "pp_row_loop", "a0", "zero")

    # Write PASS marker
    emit(*li_insns("t0", layout["marker_addr"]))
    emit(*li_insns("t1", PASS_MARKER))
    emit(SW("t1", "t0", 0))

    label("end")
    end_j = len(insns); emit(0)
    patch_j(end_j, "end")
    return insns


def write_params_global(out_dir, fw_words, args, layer, label, in_scale, in_zp, tiles, layout, exp_q):
    total_m = layout["total_m"]; total_n = layout["total_n"]
    timeout = max(1200000, layout["tile_count"] * 10000)
    op = out_dir.as_posix()
    lines = [
        f'`define REP_TILE_SOC_FW_HEX "{op}/soc_repopt_tile_window.hex"',
        f'`define REP_TILE_SOC_DRAM_HEX "{op}/dram_init.hex"',
        f'`define REP_TILE_SOC_RAW_ADDR_HEX "{op}/expected_raw_addr.hex"',
        f'`define REP_TILE_SOC_EXPECTED_RAW_HEX "{op}/expected_raw.hex"',
        f'`define REP_TILE_SOC_EXPECTED_Q_HEX "{op}/expected_q.hex"',
        f"`define REP_TILE_SOC_FW_WORDS {fw_words}",
        f"`define REP_TILE_SOC_DRAM_WORDS {layout['dram_words']}",
        f"`define REP_TILE_SOC_TIMEOUT_CYCLES {timeout}",
        f"`define REP_TILE_SOC_MARKER_ADDR 32'h{layout['marker_addr']:08x}",
        f"`define REP_TILE_SOC_Q_BASE 32'h{layout['q_base']:08x}",
        f"`define REP_TILE_SOC_RAW_COUNT {total_m * total_n}",
        f"`define REP_TILE_SOC_Q_COUNT {total_m * total_n}",
        f"`define REP_TILE_SOC_REQUANT_SHIFT {REQUANT_SHIFT}",
        f"`define REP_TILE_SOC_TILE_COUNT {layout['tile_count']}",
        f"`define REP_TILE_SOC_M_BASE {args.m_base}",
        f"`define REP_TILE_SOC_N_BASE {args.n_base}",
        f"`define REP_TILE_SOC_M_TILES {args.m_tiles}",
        f"`define REP_TILE_SOC_N_TILES {args.n_tiles}",
        f"`define REP_TILE_SOC_CIFAR_LABEL {label}",
        f"`define REP_TILE_SOC_R_ADDR_0 32'h{layout['r_base']:08x}",
    ]
    with open(out_dir / "soc_repopt_tile_window_params.vh", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate RepOpt tile SoC case (global N_DIM)")
    parser.add_argument("--pth", default=".06_RepOpt_VGG/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth")
    parser.add_argument("--plan", default="sim/pth_repopt_probe/model_plan.json")
    parser.add_argument("--data-root", default=".06_RepOpt_VGG/06_RepOpt_VGG/data")
    parser.add_argument("--layer-name", default="stage1_0_conv")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--m-base", type=int, default=0)
    parser.add_argument("--n-base", type=int, default=0)
    parser.add_argument("--m-tiles", type=int, default=2)
    parser.add_argument("--n-tiles", type=int, default=2)
    parser.add_argument("--tile-rows", type=int, default=4)
    parser.add_argument("--tile-cols", type=int, default=4)
    parser.add_argument("--cfg-shape", type=int, default=0)
    parser.add_argument("--full-layer", action="store_true")
    parser.add_argument("--out-dir", default="sim/repopt_tile_soc", help="Output directory")
    args = parser.parse_args()
    import torch
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    plan_path = Path(args.plan).resolve()
    plan = load_json(plan_path)
    if args.full_layer:
        layer = next((item for item in plan["layers"] if item.get("name") == args.layer_name and item.get("op") == "conv2d"), None)
        if layer is None: raise ValueError("conv layer not found")
        args.m_base = 0; args.n_base = 0
        args.m_tiles = int(layer["registers"]["M_DIM"]) // args.tile_rows
        args.n_tiles = int(layer["registers"]["N_DIM"]) // args.tile_cols

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
        checkpoint = torch.load(args.pth, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)
    layer, label, in_scale, in_zp, tiles, layout = build_case(plan, plan_path.parent, state_dict, args)
    write_dram_init(out_dir, tiles, layout)
    raw_addr, exp_raw, exp_q = write_expected_files(out_dir, tiles, layout)
    fw_words = assemble_firmware_global(layout, args)
    write_hex(out_dir / "soc_repopt_tile_window.hex", fw_words)
    write_params_global(out_dir, len(fw_words), args, layer, label, in_scale, in_zp, tiles, layout, exp_q)
    print(f"generated: {out_dir}")
    print(f"firmware words: {len(fw_words)}, tiles: {layout['tile_count']}, total M×N: {layout['total_m']}×{layout['total_n']}")

if __name__ == "__main__":
    main()
