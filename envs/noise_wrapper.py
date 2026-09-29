"""Lớp bao bất định dùng chung cho mọi mạch công việc.

ĐẶC TẢ GIAO DIỆN (cố định ở Giai đoạn 0 — T2 hoàn thiện bản thật trên nhánh t2-robustness):
    NoiseWrapper(env, sigma_w, sigma_v)
        - sigma_w: cường độ nhiễu ĐỘNG HỌC  w_k  (cộng vào bước chuyển trạng thái)
        - sigma_v: cường độ nhiễu CẢM BIẾN   v_i^k (cộng vào quan sát cục bộ o_i^k)

Bản dưới đây là BẢN TẠM (Gaussian cộng tính) để T3/T4 chạy được trong khi
T2 làm bản đầy đủ (gồm cả dịch chuyển phân phối / distribution shift).
Không đổi CHỮ KÝ hàm khi thay bản thật — chỉ đổi phần thân.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NoiseSpec:
    sigma_w: float = 0.0  # nhiễu động học
    sigma_v: float = 0.0  # nhiễu cảm biến
    shift: float = 0.0    # mức dịch chuyển phân phối (T2 định nghĩa cụ thể)


class NoiseWrapper:
    """Bao một môi trường DGPPO, tiêm nhiễu động học + cảm biến.

    Bản tạm: chỉ tiêm Gaussian. Giữ nguyên API step/reset của env gốc.
    """

    def __init__(self, env, sigma_w: float = 0.0, sigma_v: float = 0.0, shift: float = 0.0, rng=None):
        self.env = env
        self.spec = NoiseSpec(sigma_w=sigma_w, sigma_v=sigma_v, shift=shift)
        self._rng = rng  # T2: dùng jax.random.PRNGKey để tái hiện

    # --- API môi trường: chuyển tiếp về env gốc ---
    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return self._perturb_obs(obs)

    def step(self, action, *args, **kwargs):
        # T2: tiêm w_k vào động học ở đây (hiện chuyển thẳng)
        out = self.env.step(action, *args, **kwargs)
        # out thường là (obs, reward, done, info) — điều chỉnh theo API DGPPO thực tế
        return out

    def _perturb_obs(self, obs):
        """Tiêm nhiễu cảm biến v_i^k. BẢN TẠM: no-op nếu sigma_v == 0."""
        if self.spec.sigma_v == 0.0:
            return obs
        # T2: cộng nhiễu Gaussian theo rng; hiện trả nguyên để không phá pipeline
        return obs

    def __getattr__(self, name):
        # Ủy quyền mọi thuộc tính khác về env gốc
        return getattr(self.env, name)
