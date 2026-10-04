"""CM-DGPPO: DGPPO với ngưỡng nới ràng buộc an toàn để tăng reward (mạch T3).

DGPPO yêu cầu h(o) <= 0 ở mọi bước (an toàn ~100%) nên thận trọng thừa, reward thấp.
CM-DGPPO nới ràng buộc một khoảng eps >= 0 (đơn vị khoảng cách):

    h~ = h(o) - eps          <=>  khoảng cách tới UAV khác >= 2r - eps, tới vật cản >= r - eps

Hai biến thể (`cm_variant`):

- "relax" (đề xuất, adaptive): eps tự điều chỉnh theo tỉ lệ va chạm thật,
      eps_{t+1} = clip(eps_t + eta * (alpha - e_t), 0, eps_max)
      e_t = tỉ lệ cặp (UAV, episode) của det_rollout có va chạm THẬT (h > 0).
  Bước cập nhật (config): eta = eps_max / (alpha * tau), tau = số bước DGPPO hội tụ về an toàn
  (đo từ T1, scripts/calib_fixed_eps.py --tau-only) => eps lên trần trong đúng tau bước.
  Nếu config không đặt cm_eta: dự phòng eta = gamma_ACI * eps_max (Gibbs & Candès 2021).
  Lý giải: methods/cm_dgppo/PARAMS.md.
  Va chạm < alpha => nới thêm (bay sát hơn, reward cao hơn); > alpha => siết lại.
  eps = 0 đúng là DGPPO gốc.
- "fixed" (baseline): h~ = h(o) + delta với delta cố định (`cm_fixed_delta`; âm = nới).

Cả hai chỉ thay `rollout.costs` / `det_rollout.costs` (cost thô, trước bước dịch +-0.5 của
DGPPO) rồi gọi `DGPPO.update_inner` gốc. Đánh giá luôn dùng h thật, không có ngưỡng.
"""
from __future__ import annotations

import os
import pickle

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dgppo.algo.dgppo import DGPPO
from dgppo.trainer.data import Rollout

from .costs import raw_cost_obs, raw_cost_true, shape_cost

VARIANTS = ("relax", "fixed")
ACI_GAMMA = 0.005  # bước ACI của Gibbs & Candès (2021), cho đại lượng trên thang [0, 1]


class CMDGPPO(DGPPO):

    def __init__(
            self,
            env,
            *args,
            cm_variant: str = "relax",
            cm_alpha: float = 0.05,
            cm_eta: float | None = None,
            cm_eps_max: float = 0.05,
            cm_fixed_delta: float = 0.0,
            **kwargs
    ):
        super().__init__(env, *args, **kwargs)
        assert cm_variant in VARIANTS, f"cm_variant phải thuộc {VARIANTS}"
        self.cm_variant = cm_variant
        self.cm_alpha = cm_alpha
        self.cm_eps_max = cm_eps_max
        # eta = gamma_ACI * eps_max nếu không đặt tay (đổi thang [0, 1] của ACI sang [0, eps_max])
        self.cm_eta = float(cm_eta) if cm_eta is not None else ACI_GAMMA * cm_eps_max
        self.cm_fixed_delta = cm_fixed_delta
        self.eps = 0.0  # "relax": khoảng nới hiện tại

        def h_diag_single(graph):
            return raw_cost_true(env, graph.env_states), raw_cost_obs(env, graph)

        self._h_diag = jax.jit(jax.vmap(jax.vmap(h_diag_single)))

    @property
    def config(self) -> dict:
        return super().config | {
            "cm_variant": self.cm_variant, "cm_alpha": self.cm_alpha, "cm_eta": self.cm_eta,
            "cm_eps_max": self.cm_eps_max, "cm_fixed_delta": self.cm_fixed_delta,
        }

    # ------------------------------------------------------------------ helpers
    def margin(self) -> float:
        """Lượng cộng vào h thô: -eps ("relax") hoặc cm_fixed_delta ("fixed")."""
        return -self.eps if self.cm_variant == "relax" else self.cm_fixed_delta

    @staticmethod
    def _agent_violation_rate(h_true: np.ndarray) -> float:
        """h_true: (b, T, a, n_cost) -> tỉ lệ cặp (episode, UAV) có ít nhất một bước va chạm thật."""
        return float((h_true > 0).any(axis=(1, 3)).mean())

    def _update_eps(self, e: float) -> float:
        """"relax": eps <- clip(eps + eta (alpha - e), 0, eps_max)."""
        self.eps = float(np.clip(self.eps + self.cm_eta * (self.cm_alpha - e), 0.0, self.cm_eps_max))
        return self.eps

    # ------------------------------------------------------------------ update
    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)
        b_key = jr.split(key, rollout.dones.shape[0])
        det_rollout = self.det_rollout_fn(self.params, b_key)

        # thay cost bằng h~ = h(o~) + margin; margin là giá trị có TRƯỚC lô này
        m = self.margin()
        _, h_obs = self._h_diag(rollout.graph)
        rollout = rollout._replace(costs=shape_cost(h_obs + m))
        h_true_d, h_obs_d = self._h_diag(det_rollout.graph)
        det_rollout = det_rollout._replace(costs=shape_cost(h_obs_d + m))
        e = self._agent_violation_rate(np.asarray(h_true_d))
        cm_info = {"cm/margin": m, "cm/viol_rate": e}
        if self.cm_variant == "relax":
            cm_info["cm/eps"] = self.eps
            self._update_eps(e)

        # phần còn lại giống DGPPO.update (commit 51b3b11)
        graph_clean = rollout.graph._replace(env_states=None)
        next_graph_clean = rollout.next_graph._replace(env_states=None)
        rollout = rollout._replace(graph=graph_clean, next_graph=next_graph_clean)
        graph_clean = det_rollout.graph._replace(env_states=None)
        next_graph_clean = det_rollout.next_graph._replace(env_states=None)
        det_rollout = det_rollout._replace(graph=graph_clean, next_graph=next_graph_clean)

        update_info = {}
        assert rollout.dones.shape[0] * rollout.dones.shape[1] >= self.batch_size
        for i_epoch in range(self.epoch_ppo):
            idx = np.arange(rollout.dones.shape[0])
            np.random.shuffle(idx)
            rnn_chunk_ids = jnp.arange(rollout.dones.shape[1])
            rnn_chunk_ids = jnp.array(jnp.array_split(rnn_chunk_ids, rollout.dones.shape[1] // self.rnn_step))
            batch_idx = jnp.array(jnp.array_split(idx, idx.shape[0] // (self.batch_size // rollout.dones.shape[1])))
            Vl_train_state, Vh_train_state, policy_train_state, update_info = self.update_inner(
                self.Vl_train_state, self.Vh_train_state, self.policy_train_state,
                rollout, det_rollout, batch_idx, rnn_chunk_ids, jnp.array(step),
            )
            self.Vl_train_state = Vl_train_state
            self.Vh_train_state = Vh_train_state
            self.policy_train_state = policy_train_state
        return update_info | cm_info

    # ------------------------------------------------------------------ io
    def save(self, save_dir: str, step: int):
        super().save(save_dir, step)
        with open(os.path.join(save_dir, str(step), "margin.pkl"), "wb") as f:
            pickle.dump({"variant": self.cm_variant, "eps": self.eps, "margin": self.margin()}, f)

    def load(self, load_dir: str, step: int):
        super().load(load_dir, step)
        path = os.path.join(load_dir, str(step), "margin.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.eps = float(pickle.load(f)["eps"])
