"""Công cụ phân tích thống kê (mạch T4).

Đọc CSV theo lược đồ analysis/logging_csv.py, tính khoảng tin cậy bootstrap,
kiểm định giả thuyết và chuẩn bị dữ liệu trực quan hóa.
TODO(T4): triển khai bootstrap CI + kiểm định (paired test giữa DGPPO vs CM-DGPPO).
"""
from __future__ import annotations

from typing import Sequence, Tuple


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Trả (mean, lo, hi) — khoảng tin cậy bootstrap percentile.

    Placeholder: T4 hoàn thiện bằng numpy. Ở đây chỉ trả mean cho lo/hi để pipeline chạy.
    """
    if not values:
        return (0.0, 0.0, 0.0)
    m = sum(values) / len(values)
    return (m, m, m)  # TODO(T4): resample n_boot lần, lấy percentile alpha/2 và 1-alpha/2
