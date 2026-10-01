"""CM-DGPPO: DGPPO + biên conformal thích nghi (mạch T3).

Ba biến thể (`variant`):

- "state": biên theo trạng thái trên h (bù nhiễu CẢM BIẾN).
      h~_{i,m} = h_m(o~_i) + q^h_m * sigma_m(o~_i)
      điểm số   s = (h_m(x) - h_m(o~)) / sigma_m(o~)        (h thật trừ h quan sát)
  GCBF Vh được học trên h~ thay cho h.

- "cbf": biên trong điều kiện suy giảm CBF rời rạc (bù nhiễu ĐỘNG HỌC + cảm biến bước kế).
      DGPPO:     (Vh(o'_{k+1})            - Vh(o_k)) / dt + alpha Vh(o_k) <= 0
      CM "cbf":  (Vh(o^nom_{k+1}) + q^cbf_m - Vh(o_k)) / dt + alpha Vh(o_k) <= 0
      điểm số   s = Vh(o'_{k+1}) - Vh(o^nom_{k+1})
  với o^nom là quan sát kế tiếp nếu không có nhiễu (bước đối chứng trong mô phỏng), nên
  Vh(o^nom) + q^cbf là cận trên conformal của Vh thực tế ở bước kế.

- "full": cả hai.

Hai biến thể baseline/ablation (biên KHÔNG cấu trúc, cùng một hằng cho mọi trạng thái):

- "fixed": h~ = h(o~) + delta, delta cố định (`cm_fixed_delta`, đơn vị khoảng cách).
  Quét delta rồi chọn delta* bằng scripts/select_delta_star.py.
- "scalar": bản ACI vô hướng của docs/scripts.tex: h~ = h(o~) + delta_t,
  delta_{t+1} = clip(delta_t + eta (e_t - alpha), 0, delta_max), e_t = tỉ lệ cửa sổ của
  det_rollout có vi phạm THẬT (`MarginUpdater`, methods/cm_dgppo/margin.py).

Mỗi (loại biên, loại ràng buộc m) là một luồng ACI riêng (`OnlineConformal`); ngân sách
alpha chia đều cho các luồng (union bound => tổng miscoverage <= alpha).
Điểm số luôn lấy từ det_rollout (chính sách deterministic, khớp lúc triển khai);
q dùng để siết lô t là q_t có TRƯỚC khi thấy lô t.

Với sigma_w = sigma_v = 0 mọi điểm số bằng 0 nên q giữ 0 và CM-DGPPO trùng DGPPO
(xem tests/test_cm_regression.py).
"""
from __future__ import annotations

import functools as ft
import os
import pickle
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from flax.training.train_state import TrainState
from jax import lax

from dgppo.algo.dgppo import DGPPO
from dgppo.algo.utils import compute_dec_ocp_gae
from dgppo.trainer.data import Rollout
from dgppo.utils.typing import Array, Params
from dgppo.utils.utils import jax_vmap, tree_index

from .conformal import ConformalConfig, OnlineConformal
from .margin import MarginConfig, MarginUpdater
from .costs import raw_cost_obs, raw_cost_true, shape_cost, sigma_scale

VARIANTS = ("state", "cbf", "full", "scalar", "fixed")


class CMDGPPO(DGPPO):

    def __init__(
            self,
            env,
            *args,
            cm_variant: str = "full",
            cm_alpha: float = 0.05,
            cm_gamma: float = 0.01,
            cm_level: str = "step",
            sigma_base: float = 0.05,
            sigma_kappa: float = 5.0,
            q_max_h: float = 5.0,
            q_max_cbf: float = 0.1,
            cm_window: int = 50_000,
            cm_samples: int = 4_096,
            cm_fixed_delta: float = 0.02,
            cm_eta: float = 1e-3,
            cm_delta_max: float = 0.05,
            cm_per_agent: bool = False,
            cm_window_steps: int = 0,
            **kwargs
    ):
        super().__init__(env, *args, **kwargs)
        assert cm_variant in VARIANTS, f"cm_variant phải thuộc {VARIANTS}"
        assert cm_level in ("step", "episode")
        for attr in ("nominal_next_graph", "sigma_v"):
            assert hasattr(env, attr), "CM-DGPPO cần env từ methods.cm_dgppo.noisy_env.make_noisy_env"

        self.cm_variant = cm_variant
        self.use_h = cm_variant in ("state", "full")
        self.use_cbf = cm_variant in ("cbf", "full")
        self.use_delta = cm_variant in ("scalar", "fixed")
        self.cm_fixed_delta = cm_fixed_delta
        self.cm_eta = cm_eta
        self.cm_delta_max = cm_delta_max
        self.cm_per_agent = cm_per_agent
        self.cm_window_steps = cm_window_steps
        self.cm_alpha = cm_alpha
        self.cm_gamma = cm_gamma
        self.cm_level = cm_level
        self.sigma_base = sigma_base
        self.sigma_kappa = sigma_kappa
        self.q_max_h = q_max_h
        self.q_max_cbf = q_max_cbf

        n_cost = env.n_cost
        n_streams = n_cost * (int(self.use_h) + int(self.use_cbf))
        alpha_stream = cm_alpha / max(n_streams, 1)
        seed = kwargs.get("seed", 0)

        def streams(q_max, offset):
            return [OnlineConformal(ConformalConfig(
                alpha=alpha_stream, gamma=cm_gamma, q_min=-q_max, q_max=q_max,
                window=cm_window, samples=cm_samples), seed=seed * 100 + offset + m) for m in range(n_cost)]

        self.conf_h = streams(q_max_h, 0) if self.use_h else []
        self.conf_cbf = streams(q_max_cbf, 50) if self.use_cbf else []
        self.delta_updater = None
        if cm_variant == "scalar":
            self.delta_updater = MarginUpdater(MarginConfig(
                alpha=cm_alpha, eta=cm_eta, delta_max=cm_delta_max, per_agent=cm_per_agent,
                window_steps=cm_window_steps, n_agents=self.n_agents))

        # --- các hàm jit phụ trợ (đóng gói env, không đổi theo bước) ---
        def h_diag_single(graph):
            return (raw_cost_true(env, graph.env_states),
                    raw_cost_obs(env, graph),
                    sigma_scale(env, graph, sigma_base, sigma_kappa))

        self._h_diag = jax.jit(jax.vmap(jax.vmap(h_diag_single)))
        self._nominal = jax.jit(jax.vmap(jax.vmap(env.nominal_next_graph)))
        self._cbf_scores = jax.jit(self._cbf_scores_fn)

    @property
    def config(self) -> dict:
        return super().config | {
            "cm_variant": self.cm_variant, "cm_alpha": self.cm_alpha, "cm_gamma": self.cm_gamma,
            "cm_level": self.cm_level, "sigma_base": self.sigma_base, "sigma_kappa": self.sigma_kappa,
            "q_max_h": self.q_max_h, "q_max_cbf": self.q_max_cbf, "cm_fixed_delta": self.cm_fixed_delta,
            "cm_eta": self.cm_eta, "cm_delta_max": self.cm_delta_max, "cm_per_agent": self.cm_per_agent,
            "cm_window_steps": self.cm_window_steps,
        }

    # ------------------------------------------------------------------ helpers
    def _Vh_next(self, Vh_params: Params, policy_params: Params, bT_graph, rollout: Rollout) -> Array:
        """Vh(bT_graph[k], rnn_{k+1}) với rnn_{k+1} lấy đúng như DGPPO (bTp1ah_Vh[:, 1:])."""

        def single(graph_T, next_graph_T, rnn_T):
            _, final_rnn = self.act(tree_index(next_graph_T, -1), rnn_T[-1], {"policy": policy_params})
            rnn_next = jnp.concatenate([rnn_T[1:], final_rnn[None]], axis=0)
            return jax.vmap(ft.partial(self.get_Vh, params={"Vh": Vh_params}))(graph_T, rnn_next)

        return jax.vmap(single)(bT_graph, rollout.next_graph, rollout.rnn_states)

    def _cbf_scores_fn(self, Vh_params, policy_params, rollout: Rollout, nom_graph) -> Array:
        Vh_act = self._Vh_next(Vh_params, policy_params, rollout.next_graph, rollout)
        Vh_nom = self._Vh_next(Vh_params, policy_params, nom_graph, rollout)
        return Vh_act - Vh_nom  # (b, T, a, n_cost)

    def _reduce(self, s: np.ndarray) -> np.ndarray:
        """s: (b, T, a). 'step' => mọi (k, i); 'episode' => max theo (k, i) mỗi episode."""
        return s.max(axis=(1, 2)) if self.cm_level == "episode" else s

    def margins(self) -> Tuple[np.ndarray, np.ndarray]:
        n_cost = self._env.n_cost
        q_h = np.array([c.q for c in self.conf_h], dtype=np.float32) if self.use_h else np.zeros(n_cost, np.float32)
        q_cbf = np.array([c.q for c in self.conf_cbf], dtype=np.float32) if self.use_cbf else np.zeros(n_cost, np.float32)
        return q_h, q_cbf

    def delta(self) -> np.ndarray:
        """Biên vô hướng (n_agent,) của "fixed"/"scalar"; 0 cho các biến thể khác."""
        if self.cm_variant == "fixed":
            return np.full(self.n_agents, self.cm_fixed_delta, np.float32)
        if self.delta_updater is not None:
            return np.broadcast_to(np.asarray(self.delta_updater.current(), np.float32), (self.n_agents,)).copy()
        return np.zeros(self.n_agents, np.float32)

    def _window_violation(self, h_true: np.ndarray):
        """h_true: (b, T, a, n_cost) -> e_t: tỉ lệ cửa sổ có vi phạm (float hoặc list theo agent)."""
        viol = (h_true > 0).any(axis=-1)                              # (b, T, a)
        w = self.cm_window_steps
        if w > 0:
            n_w = viol.shape[1] // w
            viol = viol[:, :n_w * w].reshape(viol.shape[0], n_w, w, -1).any(axis=2)  # (b, n_w, a)
        else:
            viol = viol.any(axis=1, keepdims=True)                    # (b, 1, a)
        if self.cm_per_agent:
            return [float(x) for x in viol.mean(axis=(0, 1))]
        return float(viol.any(axis=-1).mean())

    # ------------------------------------------------------------------ update
    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)
        b_key = jr.split(key, rollout.dones.shape[0])
        det_rollout = self.det_rollout_fn(self.params, b_key)

        q_h, q_cbf = self.margins()
        cm_info = {}

        # CM "state": siết h bằng biên theo trạng thái, cập nhật luồng ACI của h
        if self.use_h:
            _, h_obs, sig = self._h_diag(rollout.graph)
            rollout = rollout._replace(costs=shape_cost(h_obs + q_h * sig))
            h_true_d, h_obs_d, sig_d = self._h_diag(det_rollout.graph)
            det_rollout = det_rollout._replace(costs=shape_cost(h_obs_d + q_h * sig_d))
            scores = np.asarray((h_true_d - h_obs_d) / sig_d)
            cm_info["cm/margin_h_mean"] = float(np.mean(q_h * np.asarray(sig_d)))
            for m, c in enumerate(self.conf_h):
                for k, v in c.update(self._reduce(scores[..., m])).items():
                    cm_info[f"cm/h{m}_{k}"] = v

        # CM "fixed"/"scalar": biên hằng delta (đơn vị khoảng cách) trên h quan sát
        if self.use_delta:
            d = self.delta()[:, None]                                  # (a, 1)
            _, h_obs, _ = self._h_diag(rollout.graph)
            rollout = rollout._replace(costs=shape_cost(h_obs + d))
            h_true_d, h_obs_d, _ = self._h_diag(det_rollout.graph)
            det_rollout = det_rollout._replace(costs=shape_cost(h_obs_d + d))
            cm_info["cm/delta_mean"] = float(d.mean())
            if self.delta_updater is not None:
                e = self._window_violation(np.asarray(h_true_d))
                self.delta_updater.update(e)
                cm_info["cm/delta_err"] = float(np.mean(e))

        # CM "cbf": bước đối chứng không nhiễu, cập nhật luồng ACI của điều kiện CBF
        nom_graph = None
        if self.use_cbf:
            nom_graph = self._nominal(rollout.graph, rollout.actions)
            nom_det = self._nominal(det_rollout.graph, det_rollout.actions)
            scores = np.asarray(self._cbf_scores(
                self.Vh_train_state.params, self.policy_train_state.params, det_rollout, nom_det))
            for m, c in enumerate(self.conf_cbf):
                for k, v in c.update(self._reduce(scores[..., m])).items():
                    cm_info[f"cm/cbf{m}_{k}"] = v

        # phần còn lại giống DGPPO.update
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
            Vl_train_state, Vh_train_state, policy_train_state, update_info = self.update_inner_cm(
                self.Vl_train_state,
                self.Vh_train_state,
                self.policy_train_state,
                rollout,
                det_rollout,
                batch_idx,
                rnn_chunk_ids,
                jnp.array(step),
                nom_graph,
                jnp.asarray(q_cbf),
            )
            self.Vl_train_state = Vl_train_state
            self.Vh_train_state = Vh_train_state
            self.policy_train_state = policy_train_state
        return update_info | cm_info

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner_cm(
            self,
            Vl_train_state: TrainState,
            Vh_train_state: TrainState,
            policy_train_state: TrainState,
            rollout: Rollout,
            det_rollout: Rollout,
            batch_idx: Array,
            rnn_chunk_ids: Array,
            step: Array,
            nom_graph,
            q_cbf: Array,
    ) -> Tuple[TrainState, TrainState, TrainState, dict]:
        """Sao chép DGPPO.update_inner (commit 51b3b11); chỗ khác đánh dấu `# CM:`."""
        b, T, a, _ = rollout.actions.shape

        # calculate Vl
        bT_Vl, bT_Vl_rnn_states, final_Vl_rnn_states = jax.vmap(
            ft.partial(self.scan_Vl,
                       init_Vl_rnn_state=self.init_Vl_rnn_state,
                       Vl_params=Vl_train_state.params)
        )(rollout)

        def final_Vl_fn_(graph, rnn_state):
            Vl, _ = self.Vl.get_value(Vl_train_state.params, tree_index(graph, -1), rnn_state)
            return Vl.squeeze(0).squeeze(0)

        b_final_Vl = jax_vmap(final_Vl_fn_)(rollout.next_graph, final_Vl_rnn_states)
        bTp1_Vl = jnp.concatenate([bT_Vl, b_final_Vl[:, None]], axis=1)
        assert bTp1_Vl.shape[:2] == (b, T + 1)

        # calculate Vh
        bTah_Vh = jax.vmap(jax.vmap(ft.partial(
            self.get_Vh, params={'Vh': Vh_train_state.params})))(rollout.graph, rollout.rnn_states)

        def final_Vh_fn_(graph, rnn_state):
            _, final_rnn_state = self.act(tree_index(graph, -1), rnn_state[-1], {'policy': policy_train_state.params})
            return self.get_Vh(tree_index(graph, -1), final_rnn_state, {'Vh': Vh_train_state.params})

        final_Vh = jax.vmap(final_Vh_fn_)(rollout.next_graph, rollout.rnn_states)

        bTp1ah_Vh = jnp.concatenate([bTah_Vh, final_Vh[:, None]], axis=1)
        assert bTp1ah_Vh.shape[:4] == (b, T + 1, a, self._env.n_cost)

        # calculate Dec-EFOCP GAE
        bTah_Qh, bT_Ql = jax.vmap(
            ft.partial(compute_dec_ocp_gae, disc_gamma=self.gamma, gae_lambda=self.gae_lambda)
        )(Tah_hs=rollout.costs,
          T_l=-rollout.rewards,
          Tp1ah_Vh=bTp1ah_Vh,
          Tp1_Vl=bTp1_Vl)

        # calculate advantages and normalize
        # cost advantage
        bT_Al = bT_Ql - bT_Vl
        bT_Al = (bT_Al - bT_Al.mean(axis=1, keepdims=True)) / (bT_Al.std(axis=1, keepdims=True) + 1e-8)
        bTa_Al = bT_Al[:, :, None].repeat(self.n_agents, axis=-1)

        # safety advantage
        # CM: biến thể "cbf" thay Vh bước kế thực tế bằng cận trên conformal Vh(o^nom) + q^cbf
        if self.use_cbf:
            bTah_Vh_next = self._Vh_next(Vh_train_state.params, policy_train_state.params, nom_graph, rollout)
            bTah_Vh_next = bTah_Vh_next + q_cbf
        else:
            bTah_Vh_next = bTp1ah_Vh[:, 1:]
        bTah_cbf_deriv = (bTah_Vh_next - bTah_Vh) / self._env.dt + self.alpha * bTah_Vh
        bTah_Acbf = jnp.maximum(bTah_cbf_deriv + self.cbf_eps, 0)

        # merge advantage
        bTa_is_safe = (bTah_cbf_deriv <= 0).min(axis=-1)
        safe_data = bTa_is_safe.mean()
        bTa_A = jnp.where(bTa_is_safe, bTa_Al, jnp.zeros_like(bTa_Al))
        if self.cbf_schedule:
            bTa_A += bTah_Acbf.max(axis=-1) * self.cbf_schedule_fn(step)
        else:
            bTa_A += bTah_Acbf.max(axis=-1) * self.cbf_weight

        # reverse advantage
        bTa_A = -bTa_A

        # calculate Vh for deterministic policy
        bTah_Vh_det = jax.vmap(jax.vmap(ft.partial(
            self.get_Vh, params={'Vh': Vh_train_state.params})))(det_rollout.graph, det_rollout.rnn_states)
        final_Vh_det = jax.vmap(final_Vh_fn_)(det_rollout.next_graph, det_rollout.rnn_states)
        bTp1ah_Vh_det = jnp.concatenate([bTah_Vh_det, final_Vh_det[:, None]], axis=1)

        # calculate Qh for deterministic policy
        bTah_Qh_det, _ = jax.vmap(
            ft.partial(compute_dec_ocp_gae, disc_gamma=self.gamma, gae_lambda=self.gae_lambda)
        )(Tah_hs=det_rollout.costs,
          T_l=-det_rollout.rewards,
          Tp1ah_Vh=bTp1ah_Vh_det,
          Tp1_Vl=bTp1_Vl)

        # ppo update
        def update_fn(carry, idx):
            Vl_model, Vh_model, policy_model = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            det_rollout_batch = jtu.tree_map(lambda x: x[idx], det_rollout)
            Vl_model, Vl_info = self.update_Vl(
                Vl_model, rollout_batch, bT_Ql[idx], bT_Vl_rnn_states[idx], rnn_chunk_ids)
            Vh_model, Vh_info = self.update_Vh(
                Vh_model, det_rollout_batch, bTah_Qh_det[idx], rollout.rnn_states[idx], rnn_chunk_ids)
            policy_model, policy_info = self.update_policy(policy_model, rollout_batch, bTa_A[idx], rnn_chunk_ids)
            return (Vl_model, Vh_model, policy_model), (Vl_info | Vh_info | policy_info)

        (Vl_train_state, Vh_train_state, policy_train_state), info = lax.scan(
            update_fn, (Vl_train_state, Vh_train_state, policy_train_state), batch_idx
        )

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], info) | {'eval/safe_data': safe_data}

        return Vl_train_state, Vh_train_state, policy_train_state, info

    # ------------------------------------------------------------------ io
    def save(self, save_dir: str, step: int):
        super().save(save_dir, step)
        state = {"h": [c.state_dict() for c in self.conf_h], "cbf": [c.state_dict() for c in self.conf_cbf],
                 "delta": self.delta().tolist()}
        with open(os.path.join(save_dir, str(step), "conformal.pkl"), "wb") as f:
            pickle.dump(state, f)

    def load(self, load_dir: str, step: int):
        super().load(load_dir, step)
        path = os.path.join(load_dir, str(step), "conformal.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                state = pickle.load(f)
            for c, d in zip(self.conf_h, state["h"]):
                c.load_state_dict(d)
            for c, d in zip(self.conf_cbf, state["cbf"]):
                c.load_state_dict(d)
            if self.delta_updater is not None and "delta" in state:
                n = len(self.delta_updater.delta)
                self.delta_updater.delta = [float(x) for x in state["delta"][:n]]
