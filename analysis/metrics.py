"""Chỉ số đánh giá dùng chung, theo định nghĩa DGPPO (Giai đoạn 0).

TODO(T1/T4): nối với định nghĩa chính xác trong repo DGPPO khi đã pin commit.
"""
from __future__ import annotations

from typing import Sequence


def safety_rate(traj_violations: Sequence[bool]) -> float:
    """Tỉ lệ quỹ đạo KHÔNG vi phạm lần nào (mỗi phần tử = 1 quỹ đạo có vi phạm hay không)."""
    if not traj_violations:
        return 0.0
    clean = sum(1 for v in traj_violations if not v)
    return clean / len(traj_violations)


def violation_frequency(step_violations: Sequence[bool]) -> float:
    """Tần suất vi phạm trung bình mỗi bước."""
    if not step_violations:
        return 0.0
    return sum(1 for v in step_violations if v) / len(step_violations)


def task_cost(costs: Sequence[float]) -> float:
    """Chi phí nhiệm vụ trung bình (định nghĩa cụ thể theo DGPPO)."""
    if not costs:
        return 0.0
    return sum(costs) / len(costs)
