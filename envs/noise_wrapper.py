"""Lớp bao bất định dùng chung — bản thật của mạch T2 (hàm thuần JAX).

ĐẶC TẢ GIAO DIỆN (Giai đoạn 0 — giữ nguyên ba tham số đầu):
    NoiseWrapper(env, sigma_w, sigma_v)
        - sigma_w: std nhiễu ĐỘNG HỌC, cộng vào VẬN TỐC agent sau mỗi bước rồi clip theo
                   giới hạn env (env LiDAR: |v| <= 0.5). Vị trí lệch theo qua tích phân.
        - sigma_v: std nhiễu CẢM BIẾN, cộng vào VỊ TRÍ ước lượng của agent và các điểm hit
                   LiDAR trong quan sát đưa cho policy (đơn vị độ dài; bán kính agent r = 0.05).
                   Vận tốc quan sát giữ sạch.
    Dịch chuyển phân phối kiểu số agent / số vật cản (shift_type = n_agents | n_obs) áp lúc
    dựng env (make_env(num_agents=..., num_obs=...)), không nằm trong wrapper.

Env DGPPO là hàm thuần JAX: reset(key), step(graph, action) trả 5 giá trị, quan sát là
GraphsTuple. Không bọc kiểu gym stateful được, nên module này cung cấp `noisy_test_rollout`
— bản sao `dgppo.trainer.utils.test_rollout` có tiêm nhiễu — chạy được trong jit/scan/vmap.

Nguyên tắc: tách GRAPH THẬT (đưa vào env.step, tính reward/cost/vi phạm) và GRAPH QUAN SÁT
(có nhiễu, chỉ đưa cho actor). Policy chỉ đọc graph.nodes/edges, còn env đọc graph.states /
graph.env_states, nên thước đo vi phạm không bị nhiễu cảm biến làm sai.

Hạn chế của bản này:
    - Nhiễu cảm biến TƯƠNG QUAN: mọi agent nhìn cùng một bản nhiễu vị trí của agent khác
      (kể cả vị trí của chính nó), chưa có nhiễu độc lập theo từng cặp quan sát.
    - Chỉ hỗ trợ env LiDAR (LidarSpread, LidarLine, ...) vì dựa vào LidarEnvState.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from dgppo.env.lidar_env.base import LidarEnvState
from dgppo.trainer.data import Rollout


@dataclass(frozen=True)
class NoiseSpec:
    sigma_w: float = 0.0       # nhiễu động học (vận tốc)
    sigma_v: float = 0.0       # nhiễu cảm biến (vị trí + điểm LiDAR)
    shift_type: str = "none"   # none | n_agents | n_obs — áp lúc make_env
    shift_level: int = 0       # N_test hoặc số vật cản lúc test


def _lidar_hits(env, graph):
    """Điểm hit LiDAR (n_agent, top_k_rays, 2) lấy từ graph; None nếu env không có vật cản."""
    if env.params["n_obs"] == 0:
        return None
    k = env.params["top_k_rays"]
    hits = graph.type_states(type_idx=env.OBS, n_type=k * env.num_agents)[:, :2]
    return hits.reshape(env.num_agents, k, 2)


def perturb_dynamics(env, graph, key, sigma_w):
    """Cộng nhiễu w_k ~ N(0, sigma_w^2) vào vận tốc agent của graph THẬT, clip, dựng lại graph.

    Gọi ngay sau env.step để nhiễu tác động lên trạng thái của bước kế tiếp.
    """
    s = graph.env_states
    noise = sigma_w * jr.normal(key, (env.num_agents, 2))
    agent = env.clip_state(s.agent.at[:, 2:].add(noise))
    obstacle = s.obstacle if env.params["n_obs"] > 0 else None
    lidar = env.get_lidar_data(agent, obstacle)
    return env.get_graph(LidarEnvState(agent, s.goal, obstacle), lidar)


def observe(env, graph, key, sigma_v):
    """Trả graph QUAN SÁT: vị trí agent + điểm hit LiDAR cộng nhiễu v ~ N(0, sigma_v^2).

    Graph thật không bị sửa. Mask cạnh (bán kính liên lạc, tia LiDAR còn hiệu lực) tự tính
    lại theo giá trị nhiễu, như cảm biến thật báo sai vị trí.
    """
    s = graph.env_states
    k_agent, k_lidar = jr.split(key)
    agent = s.agent.at[:, :2].add(sigma_v * jr.normal(k_agent, (env.num_agents, 2)))
    lidar = _lidar_hits(env, graph)
    if lidar is not None:
        lidar = lidar + sigma_v * jr.normal(k_lidar, lidar.shape)
    return env.get_graph(LidarEnvState(agent, s.goal, s.obstacle), lidar)


def noisy_test_rollout(
        env,
        actor: Callable,
        init_rnn_state,
        key,
        sigma_w=0.0,
        sigma_v=0.0,
        stochastic: bool = False,
) -> Rollout:
    """Bản sao `test_rollout` (DGPPO @51b3b11) có tiêm nhiễu động học + cảm biến.

    Giữ nguyên cách tách khóa của `test_rollout`: actor (khi stochastic) nhận đúng `key_`
    của bước, khóa nhiễu lấy bằng `jr.fold_in`.

    sigma là số Python/NumPy -> quyết định TĨNH: sigma = 0 bỏ hẳn nhánh nhiễu, chương trình
    giống hệt `test_rollout`. sigma là mảng JAX (traced) -> nhánh nhiễu
    bọc trong `lax.cond`, quét nhiều mức sigma không phải biên dịch lại; khi đó ở sigma = 0
    kết quả có thể lệch `test_rollout` ở mức làm tròn float vì XLA tối ưu chương trình khác đi.

    Rollout.graph / next_graph là graph THẬT; rewards/costs tính trên graph thật.
    """
    def noise_step(sigma, fn):
        """Trả hàm graph -> graph: bỏ qua (sigma tĩnh = 0), luôn tiêm (sigma tĩnh > 0) hoặc lax.cond."""
        if not isinstance(sigma, jax.Array):
            sigma = float(sigma)
            if sigma == 0.0:
                return lambda g, k: g
            return lambda g, k: fn(g, k, sigma)
        sigma = sigma.astype(jnp.float32)
        return lambda g, k: jax.lax.cond(sigma > 0, lambda g_: fn(g_, k, sigma), lambda g_: g_, g)

    obs_step = noise_step(sigma_v, lambda g, k, s: observe(env, g, k, s))
    dyn_step = noise_step(sigma_w, lambda g, k, s: perturb_dynamics(env, g, k, s))

    key_x0, key = jr.split(key)
    init_graph = env.reset(key_x0)

    def body_(data, key_):
        graph, rnn_state = data
        obs = obs_step(graph, jr.fold_in(key_, 1))
        if not stochastic:
            action, rnn_state = actor(obs, rnn_state)
        else:
            action, rnn_state = actor(obs, rnn_state, key_)
        next_graph, reward, cost, done, info = env.step(graph, action)
        next_graph = dyn_step(next_graph, jr.fold_in(key_, 2))
        return (next_graph, rnn_state), (graph, action, rnn_state, reward, cost, done, None, next_graph)

    keys = jr.split(key, env.max_episode_steps)
    _, (graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs) = jax.lax.scan(
        body_, (init_graph, init_rnn_state), keys, length=env.max_episode_steps)
    return Rollout(graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs)


class NoiseWrapper:
    """Bao một env LiDAR của DGPPO kèm mức nhiễu; gom các hàm thuần JAX ở trên.

    Ủy quyền mọi thuộc tính khác về env gốc.
    """

    def __init__(self, env, sigma_w: float = 0.0, sigma_v: float = 0.0,
                 shift_type: str = "none", shift_level: int = 0):
        self.env = env
        self.spec = NoiseSpec(sigma_w=sigma_w, sigma_v=sigma_v,
                              shift_type=shift_type, shift_level=shift_level)

    def observe(self, graph, key):
        return observe(self.env, graph, key, self.spec.sigma_v)

    def perturb_dynamics(self, graph, key):
        return perturb_dynamics(self.env, graph, key, self.spec.sigma_w)

    def rollout(self, actor: Callable, init_rnn_state, key, stochastic: bool = False) -> Rollout:
        return noisy_test_rollout(self.env, actor, init_rnn_state, key,
                                  self.spec.sigma_w, self.spec.sigma_v, stochastic)

    def __getattr__(self, name):
        return getattr(self.env, name)
