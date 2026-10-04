#!/usr/bin/env python3
"""Tổng hợp eval_full.csv: trung bình ± độ lệch chuẩn qua seed cho mỗi điểm lưới.

    python scripts/summarize_eval.py --csv results/t3/h1_dgppo_orig_full.csv
    python scripts/summarize_eval.py --csv ... --by label env N n_obs sigma_w sigma_v online_eta \\
        --metrics safety_rate task_cost clearance_cvar5

Dòng trùng (cùng run, step, eval_seed, điểm lưới, tham số online) chỉ tính một lần.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

DEFAULT_BY = ["label", "env", "N", "n_obs", "sigma_w", "sigma_v", "online_eta"]
DEFAULT_METRICS = ["safety_rate", "safety_rate_swarm", "task_cost", "clearance_cvar5"]


def mean_std(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--by", nargs="+", default=DEFAULT_BY)
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--eval-seed", default=None, help="chỉ lấy dòng có eval_seed này")
    args = ap.parse_args()

    groups, seen = defaultdict(list), set()
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if args.eval_seed and r.get("eval_seed") != args.eval_seed:
                continue
            dup = (r["run"], r["step"], r.get("eval_seed"), r["sigma_w"], r["sigma_v"], r["N"], r["n_obs"],
                   r["online_eta"], r["online_alpha"], r["online_window"], r["online_per_agent"],
                   r.get("online_delta0"))
            if dup in seen:
                continue
            seen.add(dup)
            groups[tuple(r[k] for k in args.by)].append(r)

    def sort_key(k):
        return tuple(float(x) if x.replace(".", "", 1).replace("-", "", 1).isdigit() else x for x in k)

    head = args.by + ["n"] + args.metrics
    rows = []
    for k in sorted(groups, key=sort_key):
        rs = groups[k]
        cells = list(k) + [str(len(rs))]
        for m in args.metrics:
            mu, sd = mean_std([float(r[m]) for r in rs])
            cells.append(f"{mu:.3f}±{sd:.3f}")
        rows.append(cells)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    print("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


if __name__ == "__main__":
    main()
