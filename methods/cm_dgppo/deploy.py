"""Rollout lúc triển khai, có cập nhật biên ONLINE (mạch T3).

Dùng cho MỌI lần eval (eta = 0: chính sách cố định). Với eta > 0: biên phình quan sát
cập nhật online từ VI PHẠM quan sát được (va chạm), không cần trạng thái thật. Cơ chế:

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
    alpha: float = 0.05     # mức vi phạm mục tiêu cho MỖI CỬA SỔ (eval_cm.py quy đổi từ mức mỗi episode)
    window: int = 16        # T_w (bước)
    lo: float = -1.0        # chặn dưới Delta (âm = cho phép bớt thận trọng)
    hi: float = 3.0         # chặn trên Delta (đơn vị sigma)
    per_agent: bool = True
    delta0: float = 0.0


def make_deploy_fn(env, algo, n_episodes: int, cfg: OnlineConfig):
    """Trả hàm `fn(params, keys, noise=None) -> dict`, keys: (n_streams, 2), noise: (sigma_w, sigma_v).

    Mỗi luồng chạy n_episodes episode nối tiếp; kết quả có shape (n_streams, n_episodes, T, ...).
    `noise` là đối số động: chỉ biên dịch một lần rồi chạy được cả lưới nhiễu (mặc định:
    mức nhiễu lúc dựng env). Đổi N hoặc n_obs vẫn cần env/hàm mới (đổi kích thước mảng).
    """
    T = env.max_episode_steps
    n = env.num_agents
    if T % cfg.window != 0:
        raise ValueError(f"T={T} phải chia hết cho window={cfg.window} (nếu không, cửa sổ cuối bị bỏ qua)")

    def episode(params, noise, delta, key):
        graph = env.set_params(env.reset(key), noise=noise, delta=delta)

        def body(carry, k):
            graph, rnn, delta, acc = carry                       # graph đã quan sát với `delta`
            action, rnn = algo.act(graph, rnn, params)
            raw = raw_cost_true(env, graph.env_states)          # (n, n_cost), vi phạm thật bước k
            acc = acc | (raw > 0).any(axis=-1)
            end = (k + 1) % cfg.window == 0
            e = acc if cfg.per_agent else jnp.broadcast_to(acc.any(), acc.shape)
            new_delta = jnp.where(
                end, jnp.clip(delta + cfg.eta * (e.astype(jnp.float32) - cfg.alpha), cfg.lo, cfg.hi), delta)
            acc = jnp.where(end, jnp.zeros_like(acc), acc)
            # đặt Delta mới TRƯỚC env.step để quan sát k+1 dùng ngay Delta mới (không trễ một bước)
            graph = graph._replace(env_states=graph.env_states._replace(delta=new_delta))
            next_graph, reward, _, _, _ = env.step(graph, action)
            out = dict(reward=reward, raw=raw, delta=delta, delta_next=new_delta, window_end=end, window_err=e)
            return (next_graph, rnn, new_delta, acc), out

        init = (graph, algo.init_rnn_state, delta, jnp.zeros((n,), dtype=bool))
        (_, _, delta, _), out = jax.lax.scan(body, init, jnp.arange(T))
        return delta, out

    def stream(params, noise, key):
        def ep(delta, k):
            return episode(params, noise, delta, k)

        delta0 = jnp.full((n,), cfg.delta0, jnp.float32)
        _, out = jax.lax.scan(ep, delta0, jr.split(key, n_episodes))
        return out

    run = jax.jit(jax.vmap(stream, in_axes=(None, None, 0)))
    default_noise = jnp.array([env.sigma_w, env.sigma_v], jnp.float32)

    def fn(params, keys, noise=None):
        return run(params, default_noise if noise is None else jnp.asarray(noise, jnp.float32), keys)

    return fn
