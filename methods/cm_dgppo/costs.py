"""Hàm ràng buộc h cho CM-DGPPO trên môi trường LiDAR của DGPPO (mạch T3).

Tách h THẬT (từ trạng thái thật trong env_states) và h QUAN SÁT (từ node của graph,
tức cái agent nhìn thấy, có nhiễu cảm biến). Mọi hàm ở đây trả cost THÔ (đơn vị khoảng
cách, chưa dịch +-0.5 / clip); `shape_cost` làm bước dịch + clip giống hệt
`LidarEnv.get_cost` của DGPPO.

Biên chỉ được cộng vào cost thô: `shape_cost(raw + margin)`.
"""
from __future__ import annotations

import jax.numpy as jnp

from dgppo.utils.graph import GraphsTuple

COST_EPS = 0.5  # khớp hằng `eps` trong LidarEnv.get_cost


def shape_cost(raw: jnp.ndarray) -> jnp.ndarray:
    """Dịch +-0.5 rồi clip [-1, 1], đúng như LidarEnv.get_cost."""
    cost = jnp.where(raw <= 0.0, raw - COST_EPS, raw + COST_EPS)
    return jnp.clip(cost, a_min=-1.0, a_max=1.0)


def _raw_cost(env, agent_pos: jnp.ndarray, hits: jnp.ndarray | None) -> jnp.ndarray:
    """agent_pos: (n_agent, 2); hits: (n_agent, top_k, 2) hoặc None. Trả (n_agent, 2)."""
    r = env.params["car_radius"]
    dist = jnp.linalg.norm(agent_pos[:, None, :] - agent_pos[None, :, :], axis=-1)
    dist += jnp.eye(env.num_agents) * 1e6
    agent_cost = r * 2 - jnp.min(dist, axis=1)
    if hits is None:
        obs_cost = jnp.zeros((env.num_agents,), dtype=jnp.float32)
    else:
        obs_cost = r - jnp.linalg.norm(hits - agent_pos[:, None, :], axis=-1).min(axis=1)
    return jnp.stack([agent_cost, obs_cost], axis=1)


def _graph_obs(env, graph: GraphsTuple):
    """Vị trí/vận tốc agent và điểm LiDAR như agent quan sát (node của graph)."""
    agent_states = graph.type_states(type_idx=0, n_type=env.num_agents)
    hits = None
    if env.params["n_obs"] > 0:
        k = env.params["top_k_rays"]
        hits = graph.type_states(type_idx=2, n_type=k * env.num_agents)[:, :2]
        hits = jnp.reshape(hits, (env.num_agents, k, 2))
    return agent_states, hits


def raw_cost_obs(env, graph: GraphsTuple) -> jnp.ndarray:
    """h(o~): cost thô tính trên quan sát (có nhiễu). Trùng logic LidarEnv.get_cost."""
    agent_states, hits = _graph_obs(env, graph)
    return _raw_cost(env, agent_states[:, :2], hits)


def raw_cost_true(env, env_state) -> jnp.ndarray:
    """h(x): cost thô tính trên trạng thái thật (env_states của NoisyLidar)."""
    hits = None
    if env.params["n_obs"] > 0:
        hits = env.get_lidar_data(env_state.agent, env_state.obstacle)
    return _raw_cost(env, env_state.agent[:, :2], hits)


def sigma_scale(env, graph: GraphsTuple, base: float, kappa: float) -> jnp.ndarray:
    """Thang bất định cục bộ sigma_m(o~) >= base, shape (n_agent, 2), đơn vị khoảng cách.

    sigma = base + kappa * dt * tốc_độ_tiếp_cận, với tốc độ tiếp cận tới láng giềng gần
    nhất (cột 0) và tới điểm LiDAR gần nhất (cột 1), tính trên quan sát.
    """
    agent_states, hits = _graph_obs(env, graph)
    return sigma_from_obs(env, agent_states, hits, base, kappa)


def sigma_from_obs(env, agent_states: jnp.ndarray, hits: jnp.ndarray | None, base: float, kappa: float):
    """Như `sigma_scale` nhưng nhận mảng quan sát: agent_states (n, 4), hits (n, top_k, 2)."""
    pos, vel = agent_states[:, :2], agent_states[:, 2:4]
    n = env.num_agents

    diff = pos[:, None, :] - pos[None, :, :]                       # i - j
    dist = jnp.linalg.norm(diff, axis=-1) + jnp.eye(n) * 1e6
    j = jnp.argmin(dist, axis=1)
    rel_v = vel - vel[j]
    d_ij = diff[jnp.arange(n), j]
    closing_agent = jnp.maximum(-(d_ij * rel_v).sum(-1) / dist[jnp.arange(n), j], 0.0)

    if hits is None:
        closing_obs = jnp.zeros((n,))
    else:
        d_h = pos[:, None, :] - hits                                 # (n, k, 2)
        dist_h = jnp.linalg.norm(d_h, axis=-1)
        k = jnp.argmin(dist_h, axis=1)
        d_ik = d_h[jnp.arange(n), k]
        closing_obs = jnp.maximum(-(d_ik * vel).sum(-1) / jnp.maximum(dist_h[jnp.arange(n), k], 1e-6), 0.0)

    closing = jnp.stack([closing_agent, closing_obs], axis=1)
    return base + kappa * env.dt * closing
