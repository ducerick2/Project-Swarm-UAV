"""Định dạng kết quả CSV dùng chung (Giai đoạn 0).

Tập cột CỐ ĐỊNH — mọi mạch ghi kết quả theo đúng thứ tự này để T4 tổng hợp được.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict

COLUMNS = [
    "method",        # dgppo | informarl | informarl_lag | cm_dgppo | fixed_margin | domain_rand
    "env",           # LidarSpread | LidarLine | MPESpread | CEFC-lite
    "N",             # số agent: 3 | 5 | 7
    "sigma",         # mức nhiễu (sigma_w=sigma_v hoặc mã cấu hình nhiễu)
    "seed",
    "task_cost",     # chi phí nhiệm vụ (định nghĩa DGPPO)
    "safety_rate",   # tỉ lệ quỹ đạo không vi phạm lần nào
    "violation_freq",# tần suất vi phạm mỗi bước
    "runtime_s",     # thời gian chạy (giây)
    "git_commit",    # commit tái hiện
]


def append_row(path: str, row: Dict[str, Any]) -> None:
    """Ghi 1 dòng kết quả vào CSV, tự tạo header nếu file chưa tồn tại."""
    missing = [c for c in COLUMNS if c not in row]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({c: row[c] for c in COLUMNS})
