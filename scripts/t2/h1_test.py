#!/usr/bin/env python3
"""T2 — kiểm định H1 trên CSV từng episode (tiêu chí: docs/T2_calibration.md mục 3).

    python scripts/t2/h1_test.py <sweep_dir>/episodes.csv [--levels configs/t2/sigma_grid.yaml]
        [--seeds 0 1 2] [--out <file.md>]

Với mỗi (method, env, trục σ, mức low/mid/high): ghép cặp từng (seed, episode) giữa σ = 0 và
mức đó, d = safe_agent_frac(σ=0) − safe_agent_frac(σ). Báo:
    drop_pp        trung bình d × 100 (điểm phần trăm suy giảm)
    CI95           khoảng tin cậy bootstrap 95% của drop_pp (10000 lần, ghép cặp)
    p_one_sided    kiểm định hoán vị đổi dấu một phía, H0: E[d] <= 0
    H1 (ngưỡng)    drop_pp >= ngưỡng VÀ p < 0.05
Tiêu chí chính: mức mid, ngưỡng 5 pp, đúng trên cả hai env. Độ nhạy: 2 / 10 pp, mức low/high.
Chỉ cần numpy + pyyaml.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
THRESHOLDS = (2.0, 5.0, 10.0)


def paired_test(d: np.ndarray, rng: np.random.Generator, n: int = 10000):
    """(drop_pp, ci_lo, ci_hi, p một phía) cho hiệu ghép cặp d."""
    mean = d.mean()
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    perm = (signs * d).mean(axis=1)
    p = (np.sum(perm >= mean) + 1) / (n + 1)
    return mean * 100, np.percentile(boots, 2.5) * 100, np.percentile(boots, 97.5) * 100, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--levels", default=str(REPO / "configs" / "t2" / "sigma_grid.yaml"))
    ap.add_argument("--seeds", nargs="+", default=None, help="chỉ dùng các seed này (vd 1 2)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    levels = yaml.safe_load(open(args.levels))["levels"]
    rows = list(csv.DictReader(open(args.csv)))
    if args.seeds:
        rows = [r for r in rows if r["seed"] in args.seeds]
    # (method, env, mode, sigma_w, sigma_v) -> {(seed, episode): safe_agent_frac}
    data = defaultdict(dict)
    for r in rows:
        key = (r["method"], r["env"], r["policy_mode"], float(r["sigma_w"]), float(r["sigma_v"]))
        data[key][(r["seed"], int(r["episode"]))] = float(r["safe_agent_frac"])

    rng = np.random.default_rng(args.rng_seed)
    lines = [f"Nguồn: `{args.csv}`" + (f" (seed {' '.join(args.seeds)})" if args.seeds else ""), "",
             "| method | env | mode | trục | mức | σ | cặp | drop_pp [CI95] | p một phía | H1@2pp | H1@5pp | H1@10pp |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verdict = defaultdict(dict)   # (method, mode, axis, level) -> {env: bool@5pp}
    for (method, env, mode, sw, sv) in sorted(k for k in data if k[3] == 0 and k[4] == 0):
        base = data[(method, env, mode, 0.0, 0.0)]
        for axis in ("sigma_w", "sigma_v"):
            for lvl in ("low", "mid", "high"):
                s = float(levels[lvl][axis])
                key = (method, env, mode, s, 0.0) if axis == "sigma_w" else (method, env, mode, 0.0, s)
                if key not in data:
                    continue
                common = sorted(set(base) & set(data[key]))
                d = np.array([base[c] - data[key][c] for c in common])
                drop, lo, hi, p = paired_test(d, rng)
                ok = {t: bool(drop >= t and p < 0.05) for t in THRESHOLDS}
                verdict[(method, mode, axis, lvl)][env] = ok[5.0]
                lines.append(f"| {method} | {env} | {mode} | {axis} | {lvl} | {s:g} | {len(common)} "
                             f"| {drop:+.2f} [{lo:+.2f}, {hi:+.2f}] | {p:.4f} | "
                             + " | ".join("✔" if ok[t] else "✘" for t in THRESHOLDS) + " |")
    lines += ["", "**Tiêu chí chính (mức mid, ngưỡng 5 pp, đúng trên cả hai env):**", ""]
    for (method, mode, axis, lvl), per_env in sorted(verdict.items()):
        if lvl != "mid":
            continue
        res = all(per_env.values()) and len(per_env) >= 2
        lines.append(f"- {method} ({mode}), {axis}: **{'H1 ĐÚNG' if res else 'H1 KHÔNG ĐẠT'}** "
                     f"({', '.join(f'{e}: {v}' for e, v in sorted(per_env.items()))})")
    out = "\n".join(lines)
    print(out)
    if args.out:
        Path(args.out).write_text(out + "\n")


if __name__ == "__main__":
    main()
