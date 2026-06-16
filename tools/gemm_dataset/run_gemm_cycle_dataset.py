#!/usr/bin/env python3
"""Collect GEMM cycle rows by running the tile GEMM RTL test entry."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


SHAPES = ("4x4", "8x8", "16x16", "8x32")
LANES = ("1", "2", "4")
HEADER = [
    "case_id", "split", "source", "sim", "shape", "cfg_shape", "lanes", "data_w",
    "M", "N", "K", "run_cycles", "perf_cycles", "busy_cycles", "compute_cycles",
    "dma_cycles", "rd_beats", "wr_beats", "rd_bytes", "wr_bytes", "rd_bursts",
    "wr_bursts", "mac_ops_lo", "mac_ops_hi", "ops_lo", "ops_hi", "peak_ops_cycle",
    "aw_count", "err_status", "status", "test", "git_commit",
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


def parse_csv_list(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    bad = [item for item in items if item not in allowed]
    if bad:
        raise SystemExit(f"invalid {name}: {','.join(bad)}")
    return items


def load_points(path: Path, limit: int | None) -> list[dict[str, str]]:
    points: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"M", "N", "K"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SystemExit(f"{path} must have at least M, N, K columns")
        for row in reader:
            row.setdefault("split", "train")
            points.append(row)
            if limit is not None and len(points) >= limit:
                break
    return points


def existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "case_id" not in reader.fieldnames:
            return set()
        return {row["case_id"] for row in reader if row.get("case_id")}


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


def write_header_if_needed(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()


def append_row(path: Path, row: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        writer.writerow({key: row.get(key, "") for key in HEADER})


def append_failure(path: Path, row: dict[str, str]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        fields = ["case_id", "split", "sim", "shape", "lanes", "M", "N", "K", "returncode", "reason"]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", default="data/gemm_cycles/mnk_points.tsv")
    parser.add_argument("--out", default="data/gemm_cycles/gemm_cycle_dataset.txt")
    parser.add_argument("--failures", default=None)
    parser.add_argument("--shapes", default="4x4,8x8,16x16,8x32")
    parser.add_argument("--lanes", default="1,2,4")
    parser.add_argument("--sim", choices=["icarus", "verilator"], default="icarus")
    parser.add_argument("--limit", type=int, default=None, help="limit number of MNK points")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--runner", default="tb/tile4/run_verilator.sh")
    args = parser.parse_args()

    points_path = root / args.points
    out_path = root / args.out
    failures_path = root / args.failures if args.failures else out_path.with_suffix(".failures.tsv")
    runner = root / args.runner
    shapes = parse_csv_list(args.shapes, SHAPES, "shape")
    lanes_list = parse_csv_list(args.lanes, LANES, "lanes")
    commit = git_commit(root)

    points = load_points(points_path, args.limit)
    write_header_if_needed(out_path)
    seen = set() if args.no_resume else existing_case_ids(out_path)

    total = len(points) * len(shapes) * len(lanes_list)
    done = 0
    skipped = 0
    failed = 0

    for point in points:
        m = point["M"]
        n = point["N"]
        k = point["K"]
        split = point.get("split") or "train"
        for shape in shapes:
            for lanes in lanes_list:
                case_id = f"M{m}_N{n}_K{k}_{shape}_L{lanes}_{args.sim}"
                if case_id in seen:
                    skipped += 1
                    continue
                cmd = [
                    str(runner), f"--{args.sim}", "--shape", shape,
                    "--M", m, "--K", k, "--N", n, "--lanes", lanes,
                ]
                print(f"[{done + skipped + failed + 1}/{total}] {case_id}", flush=True)
                proc = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                result = parse_result_line(proc.stdout)
                if proc.returncode != 0 or result is None:
                    failed += 1
                    append_failure(failures_path, {
                        "case_id": case_id,
                        "split": split,
                        "sim": args.sim,
                        "shape": shape,
                        "lanes": lanes,
                        "M": m,
                        "N": n,
                        "K": k,
                        "returncode": str(proc.returncode),
                        "reason": "missing RESULT" if result is None else "runner failed",
                    })
                    sys.stdout.write(proc.stdout)
                    sys.stdout.flush()
                    continue
                row = {
                    "case_id": case_id,
                    "split": split,
                    "source": "full_check",
                    "sim": args.sim,
                    "shape": shape,
                    "git_commit": commit,
                    **result,
                }
                append_row(out_path, row)
                seen.add(case_id)
                done += 1

    print(f"wrote={done} skipped={skipped} failed={failed} out={out_path}")
    if failed:
        print(f"failures={failures_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
