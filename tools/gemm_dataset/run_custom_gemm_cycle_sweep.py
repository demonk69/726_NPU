#!/usr/bin/env python3
"""Run a custom MNK GEMM cycle sweep over flows, PE shapes, and SIMD lanes."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path


SHAPES = ("4x4", "8x8", "16x16", "8x32")
FLOWS = ("os", "ws")
LANES = ("1", "2", "4")
SHAPE_DIMS = {
    "4x4": (4, 4),
    "8x8": (8, 8),
    "16x16": (16, 16),
    "8x32": (8, 32),
}

RESULT_FIELDS = [
    "test", "cfg_shape", "data_w", "run_cycles", "perf_cycles", "busy_cycles",
    "compute_cycles", "dma_cycles", "rd_beats", "wr_beats", "rd_bytes", "wr_bytes",
    "rd_bursts", "wr_bursts", "mac_ops_lo", "mac_ops_hi", "ops_lo", "ops_hi",
    "peak_ops_cycle", "aw_count", "err_status",
]

SUMMARY_FIELDS = [
    "case_id", "git_commit", "source", "sim", "flow", "shape", "lanes", "M", "K", "N",
    "timeout_sec",
    *RESULT_FIELDS, "status", "returncode", "reason", "log",
]

MD_FIELDS = [
    "case_id", "status", "run_cycles", "perf_cycles", "busy_cycles", "compute_cycles",
    "dma_cycles", "rd_bytes", "wr_bytes", "peak_ops_cycle", "log",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
        ).strip()
    except subprocess.SubprocessError:
        return "unknown"


def repo_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_csv_list(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise SystemExit(f"empty {name} list")
    bad = [item for item in items if item not in allowed]
    if bad:
        raise SystemExit(f"invalid {name}: {','.join(bad)}")
    return items


def parse_mnk(value: str) -> tuple[int, int, int]:
    parts = [part for part in re.split(r"[xX,]", value.strip()) if part]
    if len(parts) != 3:
        raise SystemExit(f"invalid --mnk '{value}', expected M,K,N or MxKxN")
    try:
        m, k, n = (int(part) for part in parts)
    except ValueError as exc:
        raise SystemExit(f"invalid --mnk '{value}', dimensions must be integers") from exc
    if min(m, k, n) <= 0:
        raise SystemExit(f"invalid --mnk '{value}', dimensions must be positive")
    return m, k, n


def load_points(path: Path, limit: int | None) -> list[tuple[int, int, int]]:
    points: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"M", "K", "N"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SystemExit(f"{path} must have M, K, N columns")
        for row in reader:
            try:
                m, k, n = int(row["M"]), int(row["K"]), int(row["N"])
            except ValueError as exc:
                raise SystemExit(f"invalid MNK row in {path}: {row}") from exc
            if min(m, k, n) <= 0:
                raise SystemExit(f"invalid non-positive MNK row in {path}: {row}")
            points.append((m, k, n))
            if limit is not None and len(points) >= limit:
                break
    return points


def collect_points(root: Path, args: argparse.Namespace) -> list[tuple[int, int, int]]:
    points: list[tuple[int, int, int]] = []
    for value in args.mnk or []:
        points.append(parse_mnk(value))
    if args.M is not None or args.K is not None or args.N is not None:
        if args.M is None or args.K is None or args.N is None:
            raise SystemExit("--M, --K, and --N must be specified together")
        if min(args.M, args.K, args.N) <= 0:
            raise SystemExit("--M, --K, and --N must be positive")
        points.append((args.M, args.K, args.N))
    if args.points:
        points.extend(load_points(repo_path(root, args.points), args.limit))
    if not points:
        raise SystemExit("specify at least one --mnk M,K,N, --M/--K/--N, or --points file")
    return points


def parse_result_line(output: str) -> dict[str, str] | None:
    for line in output.splitlines():
        if not line.startswith("RESULT\t"):
            continue
        fields: dict[str, str] = {}
        for part in line.split("\t")[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key] = value
        return fields
    return None


def estimate_tb_timeout_cycles(m: int, k: int, n: int, shape: str) -> int:
    rows, cols = SHAPE_DIMS[shape]
    w_cols = 16 if shape == "8x32" else cols
    kt_elems = max(1, (64 * 4) // max(rows, w_cols))
    tiles_m = (m + rows - 1) // rows
    tiles_n = (n + cols - 1) // cols
    tiles_k = (k + kt_elems - 1) // kt_elems
    tile_work = tiles_m * tiles_n * tiles_k
    per_tile_budget = kt_elems + rows + cols + 256
    return max(1_000_000, tile_work * per_tile_budget + m * n * 4 + 100_000)


def estimate_shell_timeout_sec(m: int, k: int, n: int, shape: str, sim: str) -> int:
    cycles = estimate_tb_timeout_cycles(m, k, n, shape)
    if sim == "icarus":
        return min(21_600, max(600, (cycles + 1_499) // 1_500 + 300))
    return min(7_200, max(300, (cycles + 19_999) // 20_000 + 120))


def write_delimited(path: Path, rows: list[dict[str, str]], delimiter: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def md_escape(value: str) -> str:
    return value.replace("|", "\\|")


def write_markdown(path: Path, rows: list[dict[str, str]], title: str) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(MD_FIELDS) + " |")
    lines.append("| " + " | ".join(["---"] * len(MD_FIELDS)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(md_escape(row.get(key, "")) for key in MD_FIELDS) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_case(
    root: Path,
    runner: Path,
    out_dir: Path,
    work_dir: Path,
    args: argparse.Namespace,
    commit: str,
    index: int,
    total: int,
    m: int,
    k: int,
    n: int,
    flow: str,
    shape: str,
    lanes: str,
) -> dict[str, str]:
    source = "perf_only" if args.perf_only else "full_check"
    case_id = f"M{m}_K{k}_N{n}_{flow}_{shape}_L{lanes}_{args.sim}"
    if args.perf_only:
        case_id += "_perf"
    cmd = [
        str(runner), f"--{args.sim}", "--shape", shape,
        "--M", str(m), "--K", str(k), "--N", str(n),
        "--flow", flow, "--lanes", lanes,
    ]
    if args.perf_only:
        cmd.append("--perf-only")
    if args.strict_aw:
        cmd.append("--strict-aw")
    timeout_sec = args.timeout_sec
    if timeout_sec is None:
        timeout_sec = estimate_shell_timeout_sec(m, k, n, shape, args.sim)
    cmd.extend(["--timeout-sec", str(timeout_sec)])

    env = os.environ.copy()
    env["WORK_DIR"] = str(work_dir)
    if args.cleanup_work:
        env["CLEANUP_WORK"] = "1"

    print(f"[{index}/{total}] {case_id}", flush=True)
    proc = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = parse_result_line(proc.stdout)

    log_path = out_dir / "logs" / f"{case_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    log_rel = log_path.relative_to(out_dir).as_posix()

    row: dict[str, str] = {
        "case_id": case_id,
        "git_commit": commit,
        "source": source,
        "sim": args.sim,
        "flow": flow,
        "shape": shape,
        "lanes": lanes,
        "M": str(m),
        "K": str(k),
        "N": str(n),
        "timeout_sec": str(timeout_sec),
        "returncode": str(proc.returncode),
        "log": log_rel,
    }

    if result is not None:
        row.update(result)
        row["flow"] = flow
        row["shape"] = shape
        row["lanes"] = lanes
        row["K"] = str(k)
    if proc.returncode == 124:
        row["status"] = "FAIL"
        row["reason"] = "shell timeout"
    elif "global timeout" in proc.stdout:
        row["status"] = "FAIL"
        row["reason"] = "testbench timeout"
    elif proc.returncode != 0:
        row["status"] = "FAIL"
        row["reason"] = "runner failed"
    elif result is None:
        row["status"] = "FAIL"
        row["reason"] = "missing RESULT"
    elif result.get("status") != "PASS":
        row["status"] = result.get("status", "FAIL")
        row["reason"] = "non-PASS RESULT"
    else:
        row["status"] = "PASS"
        row["reason"] = ""
    return row


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnk", action="append", help="matrix point as M,K,N or MxKxN; repeatable")
    parser.add_argument("--M", type=int, default=None, help="single-point M dimension")
    parser.add_argument("--K", type=int, default=None, help="single-point K dimension")
    parser.add_argument("--N", type=int, default=None, help="single-point N dimension")
    parser.add_argument("--points", default=None, help="optional TSV with M, K, N columns")
    parser.add_argument("--limit", type=int, default=None, help="limit rows loaded from --points")
    parser.add_argument("--flows", default="os,ws")
    parser.add_argument("--shapes", default="4x4,8x8,16x16,8x32")
    parser.add_argument("--lanes", default="1,2,4")
    parser.add_argument("--sim", choices=["icarus", "verilator"], default="icarus")
    parser.add_argument("--runner", default="tb/tile4/run_verilator.sh")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--perf-only", action="store_true", help="skip golden checks and collect counters only")
    parser.add_argument("--strict-aw", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=None, help="per-case shell timeout; default is estimated from MNK/shape/sim")
    parser.add_argument("--continue-on-fail", action="store_true", help="run remaining cases after a failure")
    parser.add_argument("--stop-on-fail", dest="continue_on_fail", action="store_false", help="stop at first failure (default)")
    parser.add_argument("--cleanup-work", action="store_true", help="remove per-case generated runner work directories")
    args = parser.parse_args()

    points = collect_points(root, args)
    flows = parse_csv_list(args.flows, FLOWS, "flow")
    shapes = parse_csv_list(args.shapes, SHAPES, "shape")
    lanes_list = parse_csv_list(args.lanes, LANES, "lanes")
    runner = repo_path(root, args.runner)
    if not runner.exists():
        raise SystemExit(f"runner not found: {runner}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_path(root, args.out_dir or f"data/gemm_cycles/custom_sweep_{timestamp}")
    work_dir = repo_path(root, args.work_dir) if args.work_dir else out_dir / "work"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    summary_tsv = out_dir / "summary.tsv"
    summary_md = out_dir / "summary.md"
    commit = git_commit(root)
    total = len(points) * len(flows) * len(shapes) * len(lanes_list)
    rows: list[dict[str, str]] = []

    idx = 0
    for m, k, n in points:
        for flow in flows:
            for shape in shapes:
                for lanes in lanes_list:
                    idx += 1
                    row = run_case(root, runner, out_dir, work_dir, args, commit, idx, total, m, k, n, flow, shape, lanes)
                    rows.append(row)
                    write_delimited(summary_csv, rows, ",")
                    write_delimited(summary_tsv, rows, "\t")
                    write_markdown(summary_md, rows, "Custom GEMM Cycle Sweep")
                    if row.get("status") != "PASS" and not args.continue_on_fail:
                        print(f"failed: {row['case_id']} reason={row.get('reason', '')} log={out_dir / row['log']}")
                        raise SystemExit(1)

    failures = sum(1 for row in rows if row.get("status") != "PASS")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_tsv}")
    print(f"wrote {summary_md}")
    print(f"cases={len(rows)} pass={len(rows) - failures} fail={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
