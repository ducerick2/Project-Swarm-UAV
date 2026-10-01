#!/usr/bin/env python3
"""Chọn delta* cho baseline biên cố định từ eval_full.csv (luật chốt TRƯỚC khi chạy).

Luật: trong các delta có safety_rate trung bình (qua seed) >= 1 - alpha tại điểm eval
σ_train (validation), chọn delta có task_cost trung bình thấp nhất. Nếu không delta nào đạt,
chọn delta có safety_rate cao nhất và báo rõ.

    python scripts/select_delta_star.py --csv results/t3/eval_val.csv --alpha 0.05

Nên eval quét delta bằng `--eval-seed` KHÁC tập test chính (vd. 20000) để chọn delta*
không nhìn thấy dữ liệu test.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--env", default=None)
    args = ap.parse_args()

    groups = defaultdict(list)
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if r["method"] != "fixed_margin" or r["online_eta"] not in ("0", "0.0"):
                continue
            if args.env and r["env"] != args.env:
                continue
            at_train = (float(r["sigma_w"]) == float(r["train_sigma_w"]) and float(r["sigma_v"]) == float(r["train_sigma_v"])
                        and r["N"] == r["train_N"] and r["n_obs"] == r["train_n_obs"])
            if at_train:
                groups[(r["env"], float(r["fixed_delta"]))].append(r)

    if not groups:
        raise SystemExit("Không có dòng fixed_margin nào tại σ_train trong CSV.")
    by_env = defaultdict(list)
    for (env, d), rows in sorted(groups.items()):
        safety = sum(float(r["safety_rate"]) for r in rows) / len(rows)
        cost = sum(float(r["task_cost"]) for r in rows) / len(rows)
        by_env[env].append((d, safety, cost, len(rows)))

    for env, items in by_env.items():
        print(f"== {env}")
        for d, s, c, k in items:
            print(f"  delta={d:<8g} safety={s:.4f} cost={c:.4f} (n_seed={k})")
        ok = [it for it in items if it[1] >= 1 - args.alpha]
        if ok:
            best = min(ok, key=lambda it: it[2])
            print(f"  -> delta* = {best[0]:g} (cost thấp nhất trong các delta đạt safety >= {1 - args.alpha:g})")
        else:
            best = max(items, key=lambda it: it[1])
            print(f"  -> KHÔNG delta nào đạt safety >= {1 - args.alpha:g}; tạm chọn delta = {best[0]:g} "
                  f"(safety cao nhất). Cần mở rộng lưới delta.")


if __name__ == "__main__":
    main()
