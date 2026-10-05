"""Chỉ số an toàn / nhiệm vụ theo từng episode cho mạch T2 (độ bền vững).

Mọi chỉ số tính trên GRAPH THẬT của rollout (xem envs/noise_wrapper.py), khớp định nghĩa
của third_party/dgppo/test.py:
    - unsafe(t, i, c) = cost_c(graph_t)_i >= 0, với cost đã dịch biên ±0.5 của env LiDAR
      (tương đương h > 0), xét t = 0..T-1 (rollout.graph, không xét next_graph cuối).
    - safe_rate của test.py là THEO AGENT: 1 - mean_i max_t unsafe -> `safe_agent_frac`.
      `safe_traj` = 1 nếu cả episode không agent nào vi phạm.
    - "cost" in ra của test.py là max h -> `max_h`. Chi phí nhiệm vụ = -tổng reward.
Thành phần cost: c = 0 va chạm agent-agent, c = 1 va chạm vật cản (đo bằng 8 tia LiDAR gần
nhất của trạng thái thật — xấp xỉ của env, chưa phải hình học vật cản).
"""
from __future__ import annotations

import math

import jax.numpy as jnp


def episode_metrics(env, rollout) -> dict:
    """Chỉ số của 1 rollout (không batch). Trả dict mảng JAX; vmap được theo episode."""
    costs = rollout.costs                         # (T, n, n_cost)
    unsafe = costs >= 0.0                         # (T, n, n_cost)
    unsafe_tn = unsafe.any(axis=-1)               # (T, n)
    agent_unsafe = unsafe_tn.any(axis=0)          # (n,)
    step_unsafe = unsafe_tn.any(axis=1)           # (T,)

    # khoảng cách agent-agent nhỏ nhất trong cả episode (trạng thái thật)
    pos = rollout.graph.env_states.agent[..., :2]                     # (T, n, 2)
    dist = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)
    dist = dist + jnp.eye(env.num_agents) * 1e6
    min_dist = dist.min()

    # phủ goal ở trạng thái cuối (LidarLine: goal suy từ 2 landmark)
    final_pos = rollout.next_graph.env_states.agent[-1, :, :2]
    goals = rollout.next_graph.env_states.goal[-1, :, :2]
    if hasattr(env, "landmark2goal"):
        goals = env.landmark2goal(goals)
    dist2goal = jnp.linalg.norm(goals[:, None, :] - final_pos[None, :, :], axis=-1).min(axis=1)

    return {
        "agent_unsafe": agent_unsafe,
        "safe_agent_frac": 1.0 - agent_unsafe.mean(),
        "safe_traj": 1.0 - agent_unsafe.any().astype(jnp.float32),
        "viol_freq": unsafe_tn.mean(),            # tỉ lệ (bước, agent) vi phạm
        "min_dist": min_dist,
        "max_h": costs.max(),
        "t_first_viol": jnp.where(step_unsafe.any(), jnp.argmax(step_unsafe), -1),
        "n_viol_agent": unsafe[..., 0].any(axis=0).sum(),   # số agent từng va agent khác
        "n_viol_obs": unsafe[..., 1].any(axis=0).sum(),     # số agent từng va vật cản
        "reward": rollout.rewards.sum(),
        "task_cost": -rollout.rewards.sum(),
        "reach_rate": (dist2goal <= env.params["dist2goal"]).mean(),
        "mean_dist2goal": dist2goal.mean(),
    }


def wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson cho tỉ lệ successes/n (mặc định 95%)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (center - half, center + half)
