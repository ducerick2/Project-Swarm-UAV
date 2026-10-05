"""Định dạng CSV kết quả của mạch T2 — MỖI EPISODE MỘT DÒNG.

Tách riêng khỏi analysis/logging_csv.py (lược đồ dùng chung Giai đoạn 0 chỉ có một cột
`sigma`). Đề xuất gộp vào lược đồ chung qua PR khi cả nhóm thống nhất.
Ý nghĩa chỉ số: xem analysis/robust_metrics.py.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable

COLUMNS = [
    # --- định danh cấu hình ---
    "method",          # dgppo | informarl | informarl_lagr (tên theo checkpoint DGPPO)
    "env",             # LidarSpread | LidarLine
    "N_train",
    "N_test",
    "obs_train",
    "obs_test",
    "seed",            # seed train của checkpoint
    "test_seed",
    "sigma_w",
    "sigma_v",
    "shift_type",      # none | n_agents | n_obs | n_agents+n_obs
    "shift_level",
    "episode",
    "policy_mode",     # det | stoch
    # --- an toàn ---
    "safe_agent_frac",
    "safe_traj",
    "viol_freq",
    "min_dist",
    "max_h",
    "t_first_viol",    # -1 nếu không vi phạm
    "n_viol_agent",
    "n_viol_obs",
    # --- nhiệm vụ ---
    "task_cost",       # = -tổng reward
    "reach_rate",
    "mean_dist2goal",
    # --- tái lập ---
    "git_commit",
]


def append_rows(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """Ghi nhiều dòng vào CSV, tự tạo header nếu file chưa tồn tại."""
    rows = list(rows)
    for row in rows:
        missing = [c for c in COLUMNS if c not in row]
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc: {missing}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        for row in rows:
            w.writerow({c: row[c] for c in COLUMNS})
