#!/usr/bin/env python3
"""Generate MNK sample points for GEMM cycle dataset collection."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def add_point(points: set[tuple[int, int, int]], m: int, n: int, k: int, lo: int, hi: int) -> None:
    points.add((clamp(m, lo, hi), clamp(n, lo, hi), clamp(k, lo, hi)))


def log_uniform_int(rng: random.Random, lo: int, hi: int) -> int:
    if lo <= 1:
        return int(round(math.exp(rng.uniform(math.log(1), math.log(hi)))))
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


def assign_split(idx: int) -> str:
    bucket = idx % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8000)
    parser.add_argument("--min", dest="min_dim", type=int, default=1)
    parser.add_argument("--max", dest="max_dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=726)
    parser.add_argument("--out", default="data/gemm_cycles/mnk_points.tsv")
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.min_dim <= 0 or args.max_dim < args.min_dim:
        raise SystemExit("invalid --min/--max range")

    rng = random.Random(args.seed)
    lo = args.min_dim
    hi = args.max_dim
    points: set[tuple[int, int, int]] = set()

    anchors = [
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17,
        31, 32, 33, 63, 64, 65, 127, 128, 129,
        255, 256, 257, 511, 512, 513, 999, 1000,
        1001, 1023, 1024, 1025, 1536, 1999, 2000, 2048,
    ]
    anchors = [v for v in anchors if lo <= v <= hi]

    for v in anchors:
        add_point(points, v, v, v, lo, hi)
        add_point(points, v, 16, 16, lo, hi)
        add_point(points, 16, v, 16, lo, hi)
        add_point(points, 16, 16, v, lo, hi)

    while len(points) < args.count:
        r = rng.random()
        if r < 0.35:
            m = log_uniform_int(rng, lo, hi)
            n = log_uniform_int(rng, lo, hi)
            k = log_uniform_int(rng, lo, hi)
        elif r < 0.55:
            m = rng.randint(lo, hi)
            n = rng.randint(lo, hi)
            k = rng.randint(lo, hi)
        elif r < 0.75 and anchors:
            base = rng.choice(anchors)
            jitter = rng.choice([-2, -1, 0, 1, 2])
            m = base + jitter
            n = rng.choice([rng.randint(lo, hi), log_uniform_int(rng, lo, hi), rng.choice(anchors)])
            k = rng.choice([rng.randint(lo, hi), log_uniform_int(rng, lo, hi), rng.choice(anchors)])
        elif r < 0.90:
            small = rng.choice([1, 2, 3, 4, 8, 16, 32])
            large = rng.choice([512, 768, 1000, 1024, 1536, 2000, 2048])
            pattern = rng.randrange(6)
            if pattern == 0:
                m, n, k = small, large, log_uniform_int(rng, lo, hi)
            elif pattern == 1:
                m, n, k = large, small, log_uniform_int(rng, lo, hi)
            elif pattern == 2:
                m, n, k = large, log_uniform_int(rng, lo, hi), small
            elif pattern == 3:
                m, n, k = log_uniform_int(rng, lo, hi), large, small
            elif pattern == 4:
                m, n, k = small, log_uniform_int(rng, lo, hi), large
            else:
                m, n, k = log_uniform_int(rng, lo, hi), small, large
        else:
            m = rng.choice(anchors) if anchors else rng.randint(lo, hi)
            n = rng.choice(anchors) if anchors else rng.randint(lo, hi)
            k = rng.choice(anchors) if anchors else rng.randint(lo, hi)
        add_point(points, m, n, k, lo, hi)

    ordered = sorted(points)
    rng.shuffle(ordered)
    ordered = ordered[: args.count]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("M\tN\tK\tsplit\n")
        for idx, (m, n, k) in enumerate(ordered):
            f.write(f"{m}\t{n}\t{k}\t{assign_split(idx)}\n")
    print(f"wrote {len(ordered)} points to {out}")


if __name__ == "__main__":
    main()
