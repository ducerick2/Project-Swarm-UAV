"""Bộ cập nhật biên an toàn thích nghi delta cho CM-DGPPO (mạch T3).

Ràng buộc siết:  h~_i(o) = h_i(o) + delta_t
Cập nhật biên (dạng Adaptive Conformal Inference):
    e_t      = 1[ có vi phạm thật h_i(o_i^k) > 0 trong cửa sổ t ]
    delta_{t+1} = clip( delta_t + eta * (e_t - alpha), 0, delta_max )

Biến thể ablation (a)-(e):
    (a) biên chỉ khi train        -> dùng update() lúc train, bỏ lúc eval
    (b) biên chỉ khi triển khai   -> freeze train, update() online lúc eval
    (c) cả hai
    (d) delta riêng từng agent     -> per_agent=True (delta là vector theo N)
    (e) cửa sổ theo episode vs T_w -> window_steps
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MarginConfig:
    alpha: float = 0.05        # mức vi phạm mục tiêu
    eta: float = 0.05          # bước cập nhật
    delta_max: float = 1.0     # biên tối đa
    per_agent: bool = False    # (d) delta riêng từng agent
    window_steps: int = 0      # (e) 0 = theo episode; >0 = cửa sổ T_w bước
    n_agents: int = 1


class MarginUpdater:
    def __init__(self, cfg: MarginConfig):
        self.cfg = cfg
        n = cfg.n_agents if cfg.per_agent else 1
        self.delta: List[float] = [0.0] * n

    def current(self) -> List[float]:
        """Trả biên delta_t hiện tại (scalar-list, hoặc per-agent nếu (d))."""
        return list(self.delta)

    def tighten(self, h):
        """Trả h~ = h + delta_t. `h` là mảng ràng buộc tránh của các agent."""
        d = self.delta
        if len(d) == 1:
            return [hi + d[0] for hi in h]
        return [hi + d[i] for i, hi in enumerate(h)]

    def update(self, violated) -> List[float]:
        """Cập nhật delta theo tín hiệu vi phạm e_t.

        `violated`: bool (delta chung) hoặc list[bool] theo agent (khi per_agent).
        """
        a, eta, dmax = self.cfg.alpha, self.cfg.eta, self.cfg.delta_max
        if isinstance(violated, bool):
            violated = [violated] * len(self.delta)
        for i, e in enumerate(violated):
            self.delta[i] = min(max(self.delta[i] + eta * (float(e) - a), 0.0), dmax)
        return list(self.delta)
