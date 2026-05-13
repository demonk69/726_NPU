#!/usr/bin/env python3
"""Generate tile-mode edge-detection Conv2D data for P2.

The first P2 generator deliberately keeps im2col/tile-pack on the CPU/testbench
side. It emits tile-packed A/W streams plus int32 golden output so later RTL
tests can focus on tile-mode PE-array behavior rather than raw-IFM gather.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]

SHAPES: Dict[str, Tuple[int, int]] = {
    "4x4": (4, 4),
    "8x8": (8, 8),
    "16x16": (16, 16),
    "8x32": (8, 32),
}

CFG_SHAPE = {
    "4x4": 0,
    "8x8": 1,
    "16x16": 2,
    "8x32": 3,
}

DATAFLOW_CTRL = {
    "ws": 0x00000001,
    "os": 0x00000011,
}


BASE_FILTERS: List[Tuple[str, List[List[int]]]] = [
    ("sobel_x", [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    ("sobel_y", [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    ("laplacian_4", [[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    ("laplacian_8", [[1, 1, 1], [1, -8, 1], [1, 1, 1]]),
    ("scharr_x", [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]),
    ("scharr_y", [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]),
    ("prewitt_x", [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    ("prewitt_y", [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
]


def signed8(value: int) -> int:
    value &= 0xFF
    return value - 256 if value & 0x80 else value


def word32(value: int) -> int:
    return value & 0xFFFFFFFF


def align_words(byte_addr: int) -> int:
    return byte_addr >> 2


def write_hex(path: Path, words: Iterable[int]) -> None:
    with path.open("w", encoding="ascii") as f:
        for word in words:
            f.write(f"{word32(word):08x}\n")


def write_matrix_txt(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="ascii") as f:
        for row in matrix:
            f.write(" ".join(str(v) for v in row))
            f.write("\n")


def make_ifm(batch: int, cin: int, ih: int, iw: int) -> List[List[List[List[int]]]]:
    ifm: List[List[List[List[int]]]] = []
    for b in range(batch):
        batch_data = []
        for c in range(cin):
            chan = []
            for h in range(ih):
                row = []
                for w in range(iw):
                    # Deterministic nonzero pattern with sign changes. Avoid a
                    # padding-only or all-zero row being mistaken for PE work.
                    val = ((b * 17 + c * 11 + h * 5 + w * 3 + 1) % 23) - 11
                    if val == 0:
                        val = 7
                    row.append(val)
                chan.append(row)
            batch_data.append(chan)
        ifm.append(batch_data)
    return ifm


def expand_filters(cout: int, cin: int) -> Tuple[List[str], List[List[List[List[int]]]]]:
    names: List[str] = []
    filters: List[List[List[List[int]]]] = []
    for out_c in range(cout):
        base_name, base = BASE_FILTERS[out_c % len(BASE_FILTERS)]
        repeat = out_c // len(BASE_FILTERS)
        # Use sign/scale variants when Cout exceeds the base bank, so columns
        # are not all identical while staying in signed int8 range.
        sign = -1 if (repeat & 1) else 1
        scale = 1 + (repeat // 2)
        scale = min(scale, 3)
        filt_cin = []
        for in_c in range(cin):
            channel_scale = 1 if in_c == 0 else ((in_c % 3) + 1)
            filt = []
            for row in base:
                filt.append([max(-128, min(127, sign * scale * channel_scale * v)) for v in row])
            filt_cin.append(filt)
        suffix = "" if repeat == 0 else f"_v{repeat}"
        names.append(f"{base_name}{suffix}")
        filters.append(filt_cin)
    return names, filters


def output_dim(input_size: int, kernel: int, stride: int, pad: int, dilation: int) -> int:
    numerator = input_size + 2 * pad - dilation * (kernel - 1) - 1
    if numerator < 0:
        raise ValueError("invalid Conv2D shape")
    return numerator // stride + 1


def conv_to_im2col(
    ifm: Sequence[Sequence[Sequence[Sequence[int]]]],
    filters: Sequence[Sequence[Sequence[Sequence[int]]]],
    batch: int,
    cin: int,
    ih: int,
    iw: int,
    cout: int,
    oh: int,
    ow: int,
    pad: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    kh = kw = 3
    a_rows: List[List[int]] = []
    for b in range(batch):
        for out_h in range(oh):
            for out_w in range(ow):
                row = []
                for in_c in range(cin):
                    for ker_h in range(kh):
                        for ker_w in range(kw):
                            in_h = out_h + ker_h - pad
                            in_w = out_w + ker_w - pad
                            if 0 <= in_h < ih and 0 <= in_w < iw:
                                row.append(int(ifm[b][in_c][in_h][in_w]))
                            else:
                                row.append(0)
                a_rows.append(row)

    w_col: List[List[int]] = []
    for in_c in range(cin):
        for ker_h in range(kh):
            for ker_w in range(kw):
                row = []
                for out_c in range(cout):
                    row.append(int(filters[out_c][in_c][ker_h][ker_w]))
                w_col.append(row)

    c_mat: List[List[int]] = []
    for m, a_row in enumerate(a_rows):
        out = []
        for n in range(cout):
            acc = 0
            for k, a_val in enumerate(a_row):
                acc += signed8(a_val) * signed8(w_col[k][n])
            out.append(acc)
        c_mat.append(out)

    return a_rows, w_col, c_mat


def pack_int8(values: Sequence[int]) -> int:
    word = 0
    for lane, value in enumerate(values):
        word |= (value & 0xFF) << (lane * 8)
    return word32(word)


def pack_vector(values: Sequence[int], lanes_per_word: int = 4) -> List[int]:
    words = []
    for start in range(0, len(values), lanes_per_word):
        chunk = list(values[start : start + lanes_per_word])
        while len(chunk) < lanes_per_word:
            chunk.append(0)
        words.append(pack_int8(chunk))
    return words


def stream_elem_bits(lanes: int) -> int:
    if lanes == 1:
        return 8
    if lanes == 2:
        return 16
    if lanes == 4:
        return 32
    raise ValueError("lanes must be 1, 2, or 4")


def pack_k_lanes(values: Sequence[int], lanes: int) -> int:
    """Pack one PE input element.

    lanes=1 keeps the existing tile stream: one signed INT8 value per spatial
    PE lane. lanes=2/4 packs multiple GEMM-K values into one PE word.
    """
    if len(values) != lanes:
        raise ValueError("packed K-lane count mismatch")
    word = 0
    for lane, value in enumerate(values):
        word |= (value & 0xFF) << (lane * 8)
    return word32(word)


def pack_stream_elems(elems: Sequence[int], elem_bits: int) -> List[int]:
    if elem_bits not in (8, 16, 32):
        raise ValueError("stream element width must be 8, 16, or 32 bits")
    elems_per_word = 32 // elem_bits
    mask = (1 << elem_bits) - 1
    words: List[int] = []
    for start in range(0, len(elems), elems_per_word):
        word = 0
        for lane in range(elems_per_word):
            idx = start + lane
            value = elems[idx] if idx < len(elems) else 0
            word |= (value & mask) << (lane * elem_bits)
        words.append(word32(word))
    return words


def pack_tile_stream(
    a_im2col: Sequence[Sequence[int]],
    w_col: Sequence[Sequence[int]],
    tile_m: int,
    tile_n: int,
    k_dim: int,
    m_base: int,
    n_base: int,
    lanes: int,
) -> Tuple[List[int], List[int]]:
    if lanes not in (1, 2, 4):
        raise ValueError("lanes must be 1, 2, or 4")

    elem_bits = stream_elem_bits(lanes)
    groups = (k_dim + lanes - 1) // lanes
    a_words: List[int] = []
    w_words: List[int] = []

    for group in range(groups):
        a_elems = []
        for r in range(tile_m):
            packed = []
            for sub in range(lanes):
                k = group * lanes + sub
                packed.append(a_im2col[m_base + r][k] if k < k_dim else 0)
            a_elems.append(pack_k_lanes(packed, lanes))

        w_elems = []
        for c in range(tile_n):
            packed = []
            for sub in range(lanes):
                k = group * lanes + sub
                packed.append(w_col[k][n_base + c] if k < k_dim else 0)
            w_elems.append(pack_k_lanes(packed, lanes))

        a_words.extend(pack_stream_elems(a_elems, elem_bits))
        w_words.extend(pack_stream_elems(w_elems, elem_bits))

    return a_words, w_words


def high_lane_contributes(
    a_im2col: Sequence[Sequence[int]],
    w_col: Sequence[Sequence[int]],
    tile_m: int,
    tile_n: int,
    k_dim: int,
    m_base: int,
    n_base: int,
    lanes: int,
) -> bool:
    if lanes == 1:
        return False
    for group in range((k_dim + lanes - 1) // lanes):
        for sub in range(1, lanes):
            k = group * lanes + sub
            if k >= k_dim:
                continue
            for r in range(tile_m):
                a_val = signed8(a_im2col[m_base + r][k])
                if a_val == 0:
                    continue
                for c in range(tile_n):
                    w_val = signed8(w_col[k][n_base + c])
                    if w_val != 0:
                        return True
    return False


def build_dram(
    a_words: Sequence[int],
    w_words: Sequence[int],
    result_words: int,
    w_addr: int,
    a_addr: int,
    r_addr: int,
    dram_words: int,
) -> List[int]:
    dram = [0] * dram_words
    w_base = align_words(w_addr)
    a_base = align_words(a_addr)
    r_base = align_words(r_addr)
    for idx, word in enumerate(w_words):
        dram[w_base + idx] = word32(word)
    for idx, word in enumerate(a_words):
        dram[a_base + idx] = word32(word)
    for idx in range(result_words):
        dram[r_base + idx] = 0
    return dram


def flatten_nchw(ifm: Sequence[Sequence[Sequence[Sequence[int]]]]) -> List[int]:
    out = []
    for batch in ifm:
        for channel in batch:
            for row in channel:
                out.extend(row)
    return out


def flatten_filters(filters: Sequence[Sequence[Sequence[Sequence[int]]]]) -> List[int]:
    out = []
    for out_c in filters:
        for in_c in out_c:
            for row in in_c:
                out.extend(row)
    return out


def emit_case(args: argparse.Namespace) -> None:
    tile_m, tile_n = SHAPES[args.shape]
    cout = args.cout if args.cout is not None else tile_n
    if cout < tile_n:
        raise ValueError("cout must be >= tile_n to keep all tile columns active")

    batch = args.batch
    cin = args.cin
    ih = args.ih
    iw = args.iw
    oh = output_dim(ih, 3, 1, 1, 1)
    ow = output_dim(iw, 3, 1, 1, 1)
    m_dim = batch * oh * ow
    k_dim = cin * 3 * 3
    n_dim = cout

    if m_dim < tile_m:
        raise ValueError("batch*OH*OW must be >= tile_m for a full-tile P2.1.1 case")
    if args.m_base + tile_m > m_dim:
        raise ValueError("m_base + tile_m exceeds M")
    if args.n_base + tile_n > n_dim:
        raise ValueError("n_base + tile_n exceeds N")

    out_dir = ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    ifm = make_ifm(batch, cin, ih, iw)
    filter_names, filters = expand_filters(cout, cin)
    a_im2col, w_col, c_mat = conv_to_im2col(ifm, filters, batch, cin, ih, iw, cout, oh, ow, 1)

    a_words, w_words = pack_tile_stream(
        a_im2col,
        w_col,
        tile_m,
        tile_n,
        k_dim,
        args.m_base,
        args.n_base,
        args.lanes,
    )
    packed_k_groups = (k_dim + args.lanes - 1) // args.lanes
    elem_bits = stream_elem_bits(args.lanes)
    high_lane_seen = high_lane_contributes(
        a_im2col,
        w_col,
        tile_m,
        tile_n,
        k_dim,
        args.m_base,
        args.n_base,
        args.lanes,
    )
    if args.lanes > 1 and not high_lane_seen:
        raise ValueError("multi-lane case has no observable high packed-K lane contribution")

    expected_tile = []
    for r in range(tile_m):
        for c in range(tile_n):
            expected_tile.append(word32(c_mat[args.m_base + r][args.n_base + c]))

    w_addr = args.w_addr
    a_addr = args.a_addr
    r_addr = args.r_addr
    words_needed = max(
        align_words(w_addr) + len(w_words),
        align_words(a_addr) + len(a_words),
        align_words(r_addr) + len(expected_tile),
        1024,
    )
    dram_words = 1
    while dram_words < words_needed + 64:
        dram_words *= 2

    dram = build_dram(a_words, w_words, len(expected_tile), w_addr, a_addr, r_addr, dram_words)

    write_hex(out_dir / "dram_init.hex", dram)
    write_hex(out_dir / "a_tile.hex", a_words)
    write_hex(out_dir / "w_tile.hex", w_words)
    write_hex(out_dir / "expected.hex", expected_tile)
    write_hex(out_dir / "ifm_nchw.hex", pack_vector(flatten_nchw(ifm)))
    write_hex(out_dir / "filters_nchw.hex", pack_vector(flatten_filters(filters)))
    write_matrix_txt(out_dir / "a_im2col.txt", a_im2col)
    write_matrix_txt(out_dir / "w_col.txt", w_col)
    write_matrix_txt(out_dir / "expected_c_matrix.txt", c_mat)

    meta = {
        "name": args.name,
        "shape": args.shape,
        "dataflow": args.dataflow,
        "lanes": args.lanes,
        "batch": batch,
        "cin": cin,
        "cout": cout,
        "ih": ih,
        "iw": iw,
        "oh": oh,
        "ow": ow,
        "m_dim": m_dim,
        "n_dim": n_dim,
        "k_dim": k_dim,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "m_base": args.m_base,
        "n_base": args.n_base,
        "filter_names": filter_names,
        "a_words": len(a_words),
        "w_words": len(w_words),
        "expected_words": len(expected_tile),
        "stream_elem_bits": elem_bits,
        "packed_k_groups": packed_k_groups,
        "packed_high_lane_contributes": high_lane_seen,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="ascii")

    rel_dir = out_dir.relative_to(REPO_ROOT).as_posix()
    ctrl = DATAFLOW_CTRL[args.dataflow]
    with (out_dir / "test_params.vh").open("w", encoding="ascii") as f:
        f.write(f"// Auto-generated by tb/edge/gen_tile_edge_data.py: {args.name}\n")
        f.write("// P2 tile edge case. A/W are CPU/testbench tile-packed streams.\n")
        f.write("// M_DIM/N_DIM are local tile dimensions; GLOBAL_* records the Conv2D GEMM shape.\n")
        f.write(f'`define TEST_NAME "{args.name}"\n')
        f.write(f'`define EDGE_SHAPE "{args.shape}"\n')
        f.write(f'`define EDGE_DATAFLOW "{args.dataflow}"\n')
        f.write(f"`define EDGE_LANES {args.lanes}\n")
        f.write(f"`define EDGE_STREAM_ELEM_BITS {elem_bits}\n")
        f.write(f"`define EDGE_PACKED_K_GROUPS {packed_k_groups}\n")
        f.write(f"`define EDGE_PACKED_HIGH_LANE_CONTRIB {1 if high_lane_seen else 0}\n")
        f.write(f"`define EDGE_PE_ACTIVE_EXPECT {tile_m * tile_n}\n")
        f.write(f"`define EDGE_PE_VALID_EXPECT {tile_m * tile_n}\n")
        f.write(f"`define TILE_M {tile_m}\n")
        f.write(f"`define TILE_N {tile_n}\n")
        f.write(f"`define M_DIM {tile_m}\n")
        f.write(f"`define N_DIM {tile_n}\n")
        f.write(f"`define GLOBAL_M_DIM {m_dim}\n")
        f.write(f"`define GLOBAL_N_DIM {n_dim}\n")
        f.write(f"`define K_DIM {k_dim}\n")
        f.write(f"`define M_BASE {args.m_base}\n")
        f.write(f"`define N_BASE {args.n_base}\n")
        f.write(f"`define BATCH {batch}\n")
        f.write(f"`define CIN {cin}\n")
        f.write(f"`define COUT {cout}\n")
        f.write(f"`define IH {ih}\n")
        f.write(f"`define IW {iw}\n")
        f.write(f"`define OH {oh}\n")
        f.write(f"`define OW {ow}\n")
        f.write(f"`define NUM_RESULTS {len(expected_tile)}\n")
        f.write(f"`define A_WORDS {len(a_words)}\n")
        f.write(f"`define W_WORDS {len(w_words)}\n")
        f.write(f"`define DRAM_SIZE {dram_words}\n")
        f.write(f"`define W_ADDR 32'h{w_addr:08x}\n")
        f.write(f"`define A_ADDR 32'h{a_addr:08x}\n")
        f.write(f"`define R_ADDR 32'h{r_addr:08x}\n")
        f.write(f"`define CTRL 32'h{ctrl:08x}\n")
        f.write(f"`define CFG_SHAPE 32'h{CFG_SHAPE[args.shape]:08x}\n")
        f.write("`define IS_FP16 0\n")
        f.write(f'`define DRAM_HEX "{rel_dir}/dram_init.hex"\n')
        f.write(f'`define EXPECTED_HEX "{rel_dir}/expected.hex"\n')

    print(f"generated {args.name}: {out_dir}")
    print(f"  shape={args.shape} dataflow={args.dataflow} lanes={args.lanes}")
    print(f"  M/N/K={m_dim}/{n_dim}/{k_dim}, tile={tile_m}x{tile_n}, A_words={len(a_words)}, W_words={len(w_words)}")


def emit_matrix(args: argparse.Namespace) -> None:
    shapes = list(SHAPES.keys()) if args.shape == "all" else [args.shape]
    dataflows = list(DATAFLOW_CTRL.keys()) if args.dataflow == "all" else [args.dataflow]
    lanes_list = [1, 2, 4] if args.lanes == 0 else [args.lanes]
    prefix = args.name if args.name else "edge"

    total = 0
    for shape in shapes:
        tile_m, tile_n = SHAPES[shape]
        for dataflow in dataflows:
            for lanes in lanes_list:
                case_args = argparse.Namespace(**vars(args))
                case_args.shape = shape
                case_args.dataflow = dataflow
                case_args.lanes = lanes
                case_args.name = f"{prefix}_{shape}_{dataflow}_l{lanes}"
                case_args.cout = args.cout if args.cout is not None else tile_n
                if case_args.m_base + tile_m > case_args.batch * output_dim(case_args.ih, 3, 1, 1, 1) * output_dim(case_args.iw, 3, 1, 1, 1):
                    case_args.m_base = 0
                emit_case(case_args)
                total += 1
    print(f"generated {total} tile edge case(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="")
    parser.add_argument("--shape", choices=sorted(SHAPES) + ["all"], default="4x4")
    parser.add_argument("--dataflow", choices=sorted(DATAFLOW_CTRL) + ["all"], default="os")
    parser.add_argument("--lanes", type=int, choices=(0, 1, 2, 4), default=1, help="0 emits lanes=1,2,4 in --matrix mode")
    parser.add_argument("--matrix", action="store_true", help="emit a shape/dataflow/lane case matrix")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cin", type=int, default=1)
    parser.add_argument("--cout", type=int, default=None)
    parser.add_argument("--ih", type=int, default=8)
    parser.add_argument("--iw", type=int, default=8)
    parser.add_argument("--m-base", type=int, default=9)
    parser.add_argument("--n-base", type=int, default=0)
    parser.add_argument("--w-addr", type=lambda x: int(x, 0), default=0x1000)
    parser.add_argument("--a-addr", type=lambda x: int(x, 0), default=0x3000)
    parser.add_argument("--r-addr", type=lambda x: int(x, 0), default=0x5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.matrix:
        emit_matrix(args)
    else:
        if args.shape == "all" or args.dataflow == "all" or args.lanes == 0:
            raise ValueError("--shape all, --dataflow all, and --lanes 0 require --matrix")
        if not args.name:
            args.name = f"edge_{args.shape}_{args.dataflow}_l{args.lanes}_smoke"
        emit_case(args)


if __name__ == "__main__":
    main()
