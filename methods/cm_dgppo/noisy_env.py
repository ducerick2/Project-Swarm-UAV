"""Môi trường LiDAR có nhiễu TẠM THỜI cho mạch T3 (cho tới khi T2 merge NoiseWrapper thật).

DGPPO dùng API thuần hàm `env.step(graph, action)` không có PRNG key, nên key được mang
trong `env_states`. Quy ước:

- `graph.env_states.agent` là trạng thái THẬT; node/edge của graph là QUAN SÁT có nhiễu.
- Nhiễu động học w_k ~ N(0, sigma_w^2): cộng vào vị trí sau bước Euler.
- Nhiễu cảm biến v ~ N(0, sigma_v^2): cộng vào vị trí agent tự định vị và vào vị trí
  tương đối của từng điểm LiDAR.
- `get_cost` tính trên trạng thái THẬT, nên rollout.costs, eval của Trainer và
  `test.py` đều đo vi phạm thật. Reward cũng tính trên trạng thái thật.
- Phình quan sát lúc triển khai (`env_states.delta`, mặc định 0): agent i thấy láng giềng
  và điểm LiDAR gần hơn một đoạn m_i = delta_i * sigma_m(o~_i). Chỉ sửa ĐẶC TRƯNG CẠNH
  (vị trí tương đối), không sửa node; delta = 0 cho graph trùng từng bit.

- (sigma_w, sigma_v) nằm trong `env_states.noise` (số JAX, không phải hằng Python), nên
  MỘT hàm đã biên dịch chạy được mọi mức nhiễu: `set_params(graph, noise=..., delta=...)`.
  `sigma_w`/`sigma_v` truyền vào make_noisy_env chỉ là giá trị mặc định khi reset.

Với sigma_w = sigma_v = 0 và delta = 0, động học, quan sát và cost trùng env gốc.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp
import jax.random as jr

from dgppo.env import ENV, make_env
from dgppo.env.lidar_env.base import LidarEnv, LidarEnvState
from dgppo.utils.typing import Action, Array, State

from .costs import raw_cost_true, shape_cost, sigma_from_obs


class NoisyLidarEnvState(NamedTuple):
    agent: State      # trạng thái thật
    goal: State
    obstacle: object
    key: Array        # PRNG key cho nhiễu bước kế tiếp
    delta: Array      # (n_agent,) hệ số phình quan sát lúc triển khai; 0 khi train
    noise: Array      # (2,) = (sigma_w, sigma_v)


class _InflatedObs(NamedTuple):
    """Trạng thái quan sát đưa vào get_graph/edge_blocks, kèm biên phình (n_agent, 2)."""
    agent: State
    goal: State
    obstacle: object
    margin: Array


def _shrink(rel: Array, m: Array) -> Array:
    """Kéo vector tương đối (..., >=2) lại gần gốc một đoạn m (...,); m = 0 giữ nguyên từng bit."""
    pos = rel[..., :2]
    d = jnp.linalg.norm(pos, axis=-1, keepdims=True)
    unit = pos / jnp.maximum(d, 1e-6)
    shrink = jnp.minimum(m[..., None], d - 1e-3)  # không kéo qua chính agent
    new_pos = jnp.where(m[..., None] == 0, pos, pos - unit * shrink)
    return rel.at[..., :2].set(new_pos)


class NoisyLidarMixin:

    def __init__(self, *args, sigma_w: float = 0.0, sigma_v: float = 0.0,
                 sigma_base: float = 0.05, sigma_kappa: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_w = float(sigma_w)
        self.sigma_v = float(sigma_v)
        self.sigma_base = float(sigma_base)    # thang sigma_m(o) dùng cho phình quan sát
        self.sigma_kappa = float(sigma_kappa)

    def reset(self, key: Array):
        # cùng key => cùng trạng thái đầu như env gốc (và như mọi mức sigma): so sánh cặp
        s = super().reset(key).env_states
        noise_key = jr.fold_in(key, 0x5EED)
        delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        noise = jnp.array([self.sigma_w, self.sigma_v], dtype=jnp.float32)
        return self._observe(NoisyLidarEnvState(s.agent, s.goal, s.obstacle, noise_key, delta, noise))

    def set_params(self, graph, noise: Optional[Array] = None, delta: Optional[Array] = None):
        """Đặt mức nhiễu (sigma_w, sigma_v) và/hoặc hệ số phình, rồi quan sát lại.

        Dùng ở đầu episode lúc eval/triển khai; noise/delta có thể là giá trị traced trong jit.
        """
        s = graph.env_states
        if noise is not None:
            s = s._replace(noise=jnp.asarray(noise, jnp.float32))
        if delta is not None:
            s = s._replace(delta=jnp.asarray(delta, jnp.float32))
        return self._observe(s)

    def set_delta(self, graph, delta: Array):
        """Đặt hệ số phình và quan sát lại (dùng khi bắt đầu episode lúc triển khai)."""
        return self.set_params(graph, delta=delta)

    def edge_blocks(self, state, lidar_data=None):
        blocks = super().edge_blocks(state, lidar_data)
        if not isinstance(state, _InflatedObs):
            return blocks
        # agent-agent: feats[i, j] = feat_i - feat_j, i là agent nhận. Node của một agent được
        # mọi láng giềng dùng chung nên chỉ phình được CẠNH (node vẫn mang vị trí chưa phình).
        # Điểm LiDAR đã được dời ngay trong _observe (node và cạnh nhất quán).
        m = state.margin
        aa = blocks[0]
        blocks[0] = aa._replace(edge_feats=_shrink(aa.edge_feats, jnp.broadcast_to(m[:, None, 0], aa.edge_feats.shape[:2])))
        return blocks

    def _observe(self, state: NoisyLidarEnvState, noisy: bool = True):
        lidar_true = self.get_lidar_data(state.agent, state.obstacle)
        agent_obs, lidar_obs, key = state.agent, lidar_true, state.key
        if noisy:
            sv = state.noise[1]
            k_pos, k_lidar, key = jr.split(state.key, 3)
            agent_obs = state.agent.at[:, :2].add(sv * jr.normal(k_pos, (self.num_agents, 2)))
            if lidar_true is not None:
                rel_true = state.agent[:, None, :2] - lidar_true
                rel_obs = rel_true + sv * jr.normal(k_lidar, rel_true.shape)
                # sigma_v = 0: giữ nguyên lidar_true từng bit (a - (a - l) có thể lệch float)
                lidar_obs = jnp.where(sv > 0, agent_obs[:, None, :2] - rel_obs, lidar_true)
        sigma = sigma_from_obs(self, agent_obs, lidar_obs, self.sigma_base, self.sigma_kappa)
        margin = state.delta[:, None] * sigma
        if lidar_obs is not None:
            # mỗi điểm hit thuộc đúng một agent: dời chính NODE hit lại gần agent một đoạn m_i
            m_obs = jnp.broadcast_to(margin[:, 1:2], lidar_obs.shape[:2])
            rel = agent_obs[:, None, :2] - lidar_obs
            moved = agent_obs[:, None, :2] - _shrink(rel, m_obs)
            lidar_obs = jnp.where(m_obs[..., None] == 0, lidar_obs, moved)  # m = 0: giữ từng bit
        graph = self.get_graph(_InflatedObs(agent_obs, state.goal, state.obstacle, margin), lidar_obs)
        return graph._replace(env_states=state._replace(key=key))

    def _true_states_graph(self, graph):
        """Graph có `states` thật (để tính reward thật); node features và hàng pad giữ nguyên."""
        s = graph.env_states
        states = jnp.concatenate([s.agent, s.goal], axis=0)
        if self.params["n_obs"] > 0:
            lidar = self.get_lidar_data(s.agent, s.obstacle).reshape(-1, 2)
            lidar = jnp.concatenate([lidar, jnp.zeros((lidar.shape[0], self.state_dim - 2))], axis=1)
            states = jnp.concatenate([states, lidar], axis=0)
        return graph._replace(states=graph.states.at[:states.shape[0]].set(states))

    def step(self, graph, action: Action, get_eval_info: bool = False):
        s = graph.env_states
        action = self.clip_action(action)
        k_w, key = jr.split(s.key)
        next_agent = self.agent_step_euler(s.agent, action)  # đã clip_state; clip lại là idempotent
        next_agent = self.clip_state(next_agent.at[:, :2].add(s.noise[0] * jr.normal(k_w, (self.num_agents, 2))))

        reward = self.get_reward(self._true_states_graph(graph), action)
        cost = self.get_cost(graph)
        next_graph = self._observe(NoisyLidarEnvState(next_agent, s.goal, s.obstacle, key, s.delta, s.noise))
        return next_graph, reward, cost, jnp.array(False), {}

    def get_cost(self, graph):
        return shape_cost(raw_cost_true(self, graph.env_states))


_NOISY_CLS = {}


def make_noisy_env(
        env_id: str,
        num_agents: int,
        sigma_w: float = 0.0,
        sigma_v: float = 0.0,
        num_obs: Optional[int] = None,
        n_rays: Optional[int] = None,
        max_step: Optional[int] = None,
        full_observation: bool = False,
        sigma_base: float = 0.05,
        sigma_kappa: float = 5.0,
):
    base_cls = ENV[env_id]
    if not issubclass(base_cls, LidarEnv):
        raise NotImplementedError(f"NoisyLidar chỉ hỗ trợ LidarEnv, nhận {env_id}")
    if base_cls not in _NOISY_CLS:
        _NOISY_CLS[base_cls] = type(f"Noisy{base_cls.__name__}", (NoisyLidarMixin, base_cls), {})
    # make_env xử lý num_obs/n_rays/full_observation; lấy lại params và dựng bản có nhiễu
    base = make_env(env_id, num_agents, max_step=max_step, full_observation=full_observation,
                    num_obs=num_obs, n_rays=n_rays)
    return _NOISY_CLS[base_cls](
        num_agents=num_agents, area_size=None, max_step=base.max_episode_steps, dt=base.dt,
        params=dict(base.params), sigma_w=sigma_w, sigma_v=sigma_v,
        sigma_base=sigma_base, sigma_kappa=sigma_kappa,
    )
