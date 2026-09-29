#!/usr/bin/env python3
"""Điểm vào huấn luyện dùng chung.

    python scripts/train.py --config configs/example.yaml

Đọc config theo lược đồ, dựng env (bọc NoiseWrapper), chọn phương pháp,
huấn luyện và ghi kết quả CSV theo analysis/logging_csv.py.
Đây là KHUNG — mỗi mạch nối phần train tương ứng của mình.
"""
from __future__ import annotations

import argparse

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    print(f"[train] method={cfg['method']} env={cfg['env']} N={cfg['N']} seed={cfg['seed']}")
    # TODO(T1): dựng env DGPPO + 3 baseline
    # TODO(T2): bọc NoiseWrapper(env, **cfg['noise'])
    # TODO(T3): nếu method == cm_dgppo -> MarginUpdater(MarginConfig(**cfg['margin']))
    # TODO: train loop; ghi kết quả bằng analysis.logging_csv.append_row(...)


if __name__ == "__main__":
    main()
