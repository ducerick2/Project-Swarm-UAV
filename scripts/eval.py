#!/usr/bin/env python3
"""Điểm vào đánh giá dùng chung.

    python scripts/eval.py --config configs/example.yaml --ckpt <path>

Chạy đánh giá trên nhiều mức nhiễu / seed, tính chỉ số (analysis/metrics.py)
và ghi CSV (analysis/logging_csv.py). Là KHUNG — mỗi mạch nối phần eval của mình.
"""
from __future__ import annotations

import argparse

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"[eval] method={cfg['method']} env={cfg['env']} ckpt={args.ckpt}")
    # TODO: rollout, tính safety_rate/violation_frequency/task_cost, append_row(...)


if __name__ == "__main__":
    main()
