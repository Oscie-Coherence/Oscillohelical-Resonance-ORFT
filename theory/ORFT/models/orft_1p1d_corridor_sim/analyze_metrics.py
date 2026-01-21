#!/usr/bin/env python3
"""
Analyze ORFT sim metrics.csv 
-----------------------------------------
Computes:
- % steps where global coherence is within [0.59, 0.62]
- basic coherence stats
- top early-warning spikes (coherence gradient RMS)

Usage:
python analyze_metrics.py --metrics out/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from typing import List, Dict


def read_rows(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def stdev(xs: List[float]) -> float:
    m = mean(xs)
    return math.sqrt(mean([(x - m) ** 2 for x in xs]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True, help="Path to metrics.csv")
    ap.add_argument("--top", type=int, default=8, help="Top N early-warning spikes to show")
    args = ap.parse_args()

    rows = read_rows(args.metrics)
    if not rows:
        raise SystemExit("No rows found.")

    Cg = [row["C_global"] for row in rows]
    ew = [row["early_warning_coherence_grad_rms"] for row in rows]
    hits = [row["corridor_hit_global_059_062"] for row in rows]

    hit_rate = 100.0 * (sum(hits) / len(hits))

    print("\nORFT SIM — METRICS SUMMARY")
    print("--------------------------")
    print(f"Rows: {len(rows)}")
    print(f"Global coherence mean: {mean(Cg):.4f}  stdev: {stdev(Cg):.4f}")
    print(f"Global coherence min/max: {min(Cg):.4f} / {max(Cg):.4f}")
    print(f"Corridor hit-rate (Cg in [0.59, 0.62]): {hit_rate:.2f}%")

    # Top early-warning spikes
    ranked = sorted(enumerate(ew), key=lambda x: x[1], reverse=True)[: args.top]
    print("\nTop early-warning spikes (coherence-grad RMS):")
    print(" step | t | ew_rms | C_global")
    for idx, val in ranked:
        step = int(rows[idx]["step"])
        t = rows[idx]["t"]
        print(f"{step:5d} | {t:6.2f} | {val:7.5f} | {Cg[idx]:7.4f}")

    print("")


if __name__ == "__main__":
    main()
