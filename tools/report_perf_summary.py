#!/usr/bin/env python3
"""Summarize NPU performance lines emitted by VGG closed-loop testbenches."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def fmt_percent(bp: int) -> str:
    return f"{bp // 100}.{bp % 100:02d}%"


def fmt_tops(tops_x1e6: int) -> str:
    return f"{tops_x1e6 // 1_000_000}.{tops_x1e6 % 1_000_000:06d}"


def ratio_bp(num: int, den: int) -> int:
    return 0 if den == 0 else (num * 10_000) // den


def parse_perf_lines(text: str) -> list[dict[str, int]]:
    """Parse [PERF] pipe-table blocks from log text.

    Each block begins with '[PERF]' followed by lines like:
        | key           | value |
    The core number is extracted from the 'core' row.
    """
    rows: list[dict[str, int]] = []
    current: dict[str, int] | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[PERF]":
            if current is not None and "core" in current:
                rows.append(current)
            current = {}
            continue
        if current is not None and stripped.startswith("|") and stripped.endswith("|"):
            inner = stripped[1:-1].strip()
            parts = inner.split("|", 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            raw_val = parts[1].strip()
            val_str = raw_val.split("%")[0].strip()
            try:
                val = int(val_str, 0)
            except ValueError:
                continue
            current[key] = val
        elif current is not None and not stripped.startswith("|"):
            if "core" in current:
                rows.append(current)
            current = None

    if current is not None and "core" in current:
        rows.append(current)
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: report_perf_summary.py <run.log>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        return 0

    rows = parse_perf_lines(path.read_text(encoding="utf-8", errors="replace"))
    if not rows:
        return 0

    totals = {
        "peak_ops_per_cycle": sum(r.get("peak_ops_per_cycle", 0) for r in rows),
        "peak_tops_x1e6": sum(r.get("peak_tops_x1e6", 0) for r in rows),
        "mac_ops": sum(r.get("mac_ops", 0) for r in rows),
        "ops": sum(r.get("ops", 0) for r in rows),
        "busy_cycles": sum(r.get("busy_cycles", 0) for r in rows),
        "compute_cycles": sum(r.get("compute_cycles", 0) for r in rows),
        "dma_cycles": sum(r.get("dma_cycles", 0) for r in rows),
        "rd_beats": sum(r.get("rd_beats", 0) for r in rows),
        "rd_bursts": sum(r.get("rd_bursts", 0) for r in rows),
        "rd_burst_cycles": sum(r.get("rd_burst_cycles", 0) for r in rows),
        "wr_beats": sum(r.get("wr_beats", 0) for r in rows),
        "wr_bursts": sum(r.get("wr_bursts", 0) for r in rows),
        "wr_burst_cycles": sum(r.get("wr_burst_cycles", 0) for r in rows),
    }
    rd_util_bp = ratio_bp(totals["rd_beats"], totals["rd_burst_cycles"])
    wr_util_bp = ratio_bp(totals["wr_beats"], totals["wr_burst_cycles"])

    print("[PERF_SUMMARY]")
    print(f"| cores            | {len(rows):-10d} |")
    print(f"| peak_tops        | {fmt_tops(totals['peak_tops_x1e6']):>10s} |")
    print(f"| mac_ops          | {totals['mac_ops']:-10d} |")
    print(f"| ops              | {totals['ops']:-10d} |")
    print(f"| busy_cycles_sum  | {totals['busy_cycles']:-10d} |")
    print(f"| compute_cycles_sum | {totals['compute_cycles']:-8d} |")
    print(f"| dma_cycles_sum   | {totals['dma_cycles']:-10d} |")
    print(f"| rd_beats         | {totals['rd_beats']:-10d} |")
    print(f"| rd_bursts        | {totals['rd_bursts']:-10d} |")
    print(f"| rd_burst_cycles  | {totals['rd_burst_cycles']:-10d} |")
    print(f"| rd_burst_util    | {fmt_percent(rd_util_bp):>10s} |")
    print(f"| wr_beats         | {totals['wr_beats']:-10d} |")
    print(f"| wr_bursts        | {totals['wr_bursts']:-10d} |")
    print(f"| wr_burst_cycles  | {totals['wr_burst_cycles']:-10d} |")
    print(f"| wr_burst_util    | {fmt_percent(wr_util_bp):>10s} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
