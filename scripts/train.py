#!/usr/bin/env python3
"""Điểm vào huấn luyện dùng chung — KHUNG (skeleton), CHƯA hiện thực.

    python scripts/train.py --config configs/example.yaml

Đọc config theo lược đồ, dựng env (bọc NoiseWrapper), chọn phương pháp,
huấn luyện và ghi kết quả CSV theo analysis/logging_csv.py.

LƯU Ý: đây CHỈ là khung cho T2/T3 (NoiseWrapper, CM-DGPPO) về sau.
- Tái hiện baseline (T1) KHÔNG dùng file này: train thật là code gốc trong
  submodule third_party/dgppo/train.py, gọi qua scripts/t1/run_baseline.py.
- Mỗi mạch (T2/T3) sẽ nối phần train tương ứng vào các TODO bên dưới.
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
