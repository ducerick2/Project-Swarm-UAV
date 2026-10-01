"""Rollout lúc triển khai, có cập nhật biên ONLINE (mạch T3).

Lúc triển khai không có trạng thái thật, nên điểm số conformal lúc train
(h(x) - h(o~), Vh(o') - Vh(o^nom)) không tính được nữa. Tín hiệu duy nhất là VI PHẠM
quan sát được (va chạm). Cơ chế:

    biên phình quan sát của agent i:  m_i(o~) = Delta_i * sigma_m(o~_i)
    cuối mỗi cửa sổ T_w bước:          e_i = 1[agent i vi phạm trong cửa sổ]
                                       Delta_i <- clip(Delta_i + eta (e_i - alpha_w), lo, hi)

Chính sách không đổi; agent thấy láng giềng/vật cản gần hơn m_i (noisy_env.edge_blocks).
Delta_0 = 0 (chính sách gốc); Delta < 0 cho phép BỚT thận trọng khi môi trường dễ hơn
lúc train. Delta được mang qua các episode nối tiếp của cùng một "luồng triển khai".
eta = 0 cho đúng chính sách gốc (dùng làm eval offline).

`per_agent=True`: mỗi UAV tự cập nhật từ vi phạm của chính mình (phân tán).
`per_agent=False`: dùng chung một Delta, e = có agent nào vi phạm.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr

from .costs import raw_cost_true


@dataclass(frozen=True)
class OnlineConfig:
    eta: float = 0.0        # 0 = tắt cập nhật online
    alpha: float = 0.05     # mức vi phạm mục tiêu cho mỗi cửa sổ
    window: int = 16        # T_w (bước)
    lo: float = -1.0        # chặn dưới Delta (âm = cho phép bớt thận trọng)
    hi: float = 3.0         # chặn trên Delta (đơn vị sigma)
    per_agent: bool = True
    delta0: float = 0.0


def make_deploy_fn(env, algo, n_episodes: int, cfg: OnlineConfig):
    """Trả hàm jit `fn(params, keys) -> dict` với keys: (n_streams, 2).

    Mỗi luồng chạy n_episodes episode nối tiếp; kết quả có shape (n_streams, n_episodes, T, ...).
    """
    T = env.max_episode_steps
    n = env.num_agents

    def episode(params, delta, key):
        graph = env.set_delta(env.reset(key), delta)

        def body(carry, k):
            graph, rnn, delta, acc = carry
            action, rnn = algo.act(graph, rnn, params)
            next_graph, reward, _, _, _ = env.step(graph, action)
            raw = raw_cost_true(env, graph.env_states)          # (n, n_cost), vi phạm thật bước k
            acc = acc | (raw > 0).any(axis=-1)
            end = (k + 1) % cfg.window == 0
            e = acc if cfg.per_agent else jnp.broadcast_to(acc.any(), acc.shape)
            new_delta = jnp.where(
                end, jnp.clip(delta + cfg.eta * (e.astype(jnp.float32) - cfg.alpha), cfg.lo, cfg.hi), delta)
            acc = jnp.where(end, jnp.zeros_like(acc), acc)
            next_graph = next_graph._replace(env_states=next_graph.env_states._replace(delta=new_delta))
            out = dict(reward=reward, raw=raw, delta=delta, window_end=end, window_err=e)
            return (next_graph, rnn, new_delta, acc), out

        init = (graph, algo.init_rnn_state, delta, jnp.zeros((n,), dtype=bool))
        (_, _, delta, _), out = jax.lax.scan(body, init, jnp.arange(T))
        return delta, out

    def stream(params, key):
        def ep(delta, k):
            return episode(params, delta, k)

        delta0 = jnp.full((n,), cfg.delta0, jnp.float32)
        _, out = jax.lax.scan(ep, delta0, jr.split(key, n_episodes))
        return out

    return jax.jit(jax.vmap(stream, in_axes=(None, 0)))
