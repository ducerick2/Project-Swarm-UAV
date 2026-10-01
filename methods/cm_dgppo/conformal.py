"""Bộ phân vị conformal online (ACI trên mức phân vị) cho CM-DGPPO (mạch T3).

Mỗi luồng (stream) theo dõi một loại điểm số bất tương hợp s (vd. sai số h do nhiễu
cảm biến của ràng buộc agent-agent). Cập nhật theo Gibbs & Candès (2021):

    err_t     = tỉ lệ điểm số trong lô t vượt q_t
    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    q_{t+1}   = Quantile_{1 - alpha_{t+1}}(cửa sổ điểm số gần nhất)
                (= q_max nếu alpha_{t+1} <= 0, = q_min nếu alpha_{t+1} >= 1)

Cận |(1/T) sum err_t - alpha| <= (max(alpha_1, 1 - alpha_1) + gamma) / (gamma T)
đúng với mọi chuỗi điểm số, MIỄN LÀ q không bị kẹp ở q_min/q_max. Số lần bị kẹp
được đếm trong `n_clipped` để báo cáo.

Thuần numpy, chạy trên host, không phụ thuộc JAX.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalConfig:
    alpha: float = 0.05        # mức miscoverage mục tiêu của luồng này
    gamma: float = 0.01        # bước cập nhật ACI
    q_min: float = -np.inf     # chặn dưới q (chỉ để an toàn số học)
    q_max: float = np.inf      # chặn trên q, thay cho +inf khi alpha_t <= 0
    window: int = 50_000       # số điểm số giữ lại để tính phân vị
    samples: int = 4_096       # số điểm số lấy mẫu con mỗi lô đưa vào cửa sổ
    q_init: float = 0.0        # q trước khi có dữ liệu (0 => giống DGPPO)


class OnlineConformal:

    def __init__(self, cfg: ConformalConfig, seed: int = 0):
        self.cfg = cfg
        self.alpha_t = cfg.alpha
        self.q = cfg.q_init
        self.n_clipped = 0
        self.n_updates = 0
        self._buf = np.empty((0,), dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def update(self, scores) -> dict:
        """Nhận điểm số của lô t (đã quan sát SAU khi dùng q_t), cập nhật q_{t+1}."""
        s = np.asarray(scores, dtype=np.float64).ravel()
        s = s[np.isfinite(s)]
        if s.size == 0:
            return self.state_info(err=np.nan)
        err = float(np.mean(s > self.q))
        self.alpha_t += self.cfg.gamma * (self.cfg.alpha - err)

        if s.size > self.cfg.samples:
            s = self._rng.choice(s, self.cfg.samples, replace=False)
        self._buf = np.concatenate([self._buf, s])[-self.cfg.window:]

        if self.alpha_t <= 0.0:
            q = np.inf
        elif self.alpha_t >= 1.0:
            q = -np.inf
        else:
            q = float(np.quantile(self._buf, 1.0 - self.alpha_t, method="higher"))
        q_clipped = float(np.clip(q, self.cfg.q_min, self.cfg.q_max))
        self.n_clipped += int(q_clipped != q)
        self.q = q_clipped
        self.n_updates += 1
        return self.state_info(err=err)

    def state_info(self, err: float) -> dict:
        return {"err": err, "alpha_t": self.alpha_t, "q": self.q, "n_clipped": self.n_clipped}

    def state_dict(self) -> dict:
        return {
            "alpha_t": self.alpha_t, "q": self.q, "n_clipped": self.n_clipped,
            "n_updates": self.n_updates, "buf": self._buf.copy(),
        }

    def load_state_dict(self, d: dict) -> None:
        self.alpha_t = float(d["alpha_t"])
        self.q = float(d["q"])
        self.n_clipped = int(d["n_clipped"])
        self.n_updates = int(d["n_updates"])
        self._buf = np.asarray(d["buf"], dtype=np.float64)
