"""Test bắt buộc của NoiseWrapper (mạch T2) — cần JAX + `pip install -e third_party/dgppo`.

    PYTHONPATH=scripts/compat pytest tests/ -q

Mặc định chạy trên CPU; T2_TEST_PLATFORM=gpu để chạy trên GPU. Không cần checkpoint — dùng
InforMARL khởi tạo ngẫu nhiên.

So σ = 0 với test_rollout: XLA (JAX 0.6.2) không tất định trên cả CPU lẫn GPU — cùng chương
trình, cùng đầu vào chạy hai lần lệch ở mức làm tròn (đã kiểm: trạng thái đầu, khóa, rnn ban
đầu giống hệt, nên không phải do khởi tạo ngẫu nhiên). Sai lệch nhỏ đó có thể nhảy vọt khi
chạm ngưỡng rời rạc (chọn 8 tia LiDAR gần nhất, mask bán kính liên lạc) rồi khuếch đại theo
thời gian. Vì vậy: vài bước đầu so CHẶT (logic phải giống), cả rollout chỉ so LỎNG.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ON_CPU = os.environ.get("T2_TEST_PLATFORM", "cpu") == "cpu"
if ON_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"
if (REPO / "scripts" / "compat" / "sitecustomize.py").exists():
    runpy.run_path(str(REPO / "scripts" / "compat" / "sitecustomize.py"))   # shim JAX 0.6
sys.path.insert(0, str(REPO))

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("dgppo")

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.utils import test_rollout as dgppo_test_rollout

from analysis.robust_metrics import episode_metrics
from envs.noise_wrapper import NoiseWrapper, _lidar_hits, noisy_test_rollout, observe, perturb_dynamics

T = 32  # rút ngắn episode cho nhanh (mặc định env là 128)


@pytest.fixture(scope="module", params=["LidarSpread", "LidarLine"])
def setup(request):
    env = make_env(env_id=request.param, num_agents=3, num_obs=3, max_step=T)
    algo = make_algo(
        algo="informarl", env=env, node_dim=env.node_dim, edge_dim=env.edge_dim,
        state_dim=env.state_dim, action_dim=env.action_dim, n_agents=env.num_agents,
        cost_weight=0.0, actor_gnn_layers=2, Vl_gnn_layers=2, Vh_gnn_layers=1,
        lr_actor=3e-4, lr_Vl=1e-3, max_grad_norm=2.0, seed=0,
        use_rnn=True, rnn_layers=1, use_lstm=False)

    def stoch_actor(graph, rnn_state, key):
        action, _, rnn_state = algo.step(graph, rnn_state, key)
        return action, rnn_state

    return env, algo, {False: algo.act, True: stoch_actor}


K_EARLY = 3   # số bước đầu so chặt


def _assert_rollouts_match(ref, out):
    """Vài bước đầu gần bằng chặt; cả rollout gần bằng lỏng (xem docstring module)."""
    tight = dict(rtol=1e-4, atol=1e-5) if ON_CPU else dict(rtol=1e-3, atol=1e-4)
    for name in ("actions", "rewards", "costs", "rnn_states"):
        a, b = np.asarray(getattr(ref, name)), np.asarray(getattr(out, name))
        np.testing.assert_allclose(a[:K_EARLY], b[:K_EARLY], **tight, err_msg=f"{name}[:{K_EARLY}]")
    for x, y in zip(jax.tree_util.tree_leaves(ref.graph), jax.tree_util.tree_leaves(out.graph)):
        np.testing.assert_allclose(np.asarray(x)[:K_EARLY], np.asarray(y)[:K_EARLY], **tight, err_msg="graph")
    np.testing.assert_allclose(np.asarray(ref.actions), np.asarray(out.actions), atol=5e-2, err_msg="actions")


@pytest.mark.parametrize("stochastic", [False, True])
def test_sigma0_matches_test_rollout(setup, stochastic):
    env, algo, actors = setup
    actor = actors[stochastic]
    key = jr.PRNGKey(1234)
    ref = jax.jit(lambda k: dgppo_test_rollout(env, actor, algo.init_rnn_state, k, stochastic=stochastic))(key)
    out = jax.jit(lambda k: noisy_test_rollout(
        env, actor, algo.init_rnn_state, k, 0.0, 0.0, stochastic=stochastic))(key)
    _assert_rollouts_match(ref, out)


def test_traced_sigma0_close_to_test_rollout(setup):
    """σ traced (lax.cond) = 0: cùng logic, chỉ cho phép sai số làm tròn float."""
    env, algo, actors = setup
    key = jr.PRNGKey(1234)
    ref = jax.jit(lambda k: dgppo_test_rollout(env, actors[False], algo.init_rnn_state, k))(key)
    out = jax.jit(lambda k, sw, sv: noisy_test_rollout(env, actors[False], algo.init_rnn_state, k, sw, sv))(
        key, jnp.float32(0.0), jnp.float32(0.0))
    _assert_rollouts_match(ref, out)


def test_observe_noise_std_and_scope(setup):
    env, _, _ = setup
    graph = env.reset(jr.PRNGKey(0))
    sigma_v = 0.05
    keys = jr.split(jr.PRNGKey(1), 2000)
    obs = jax.jit(jax.vmap(lambda k: observe(env, graph, k, sigma_v)))(keys)

    true_agent = np.asarray(graph.env_states.agent)
    obs_agent = np.asarray(obs.env_states.agent)
    d_pos = obs_agent[..., :2] - true_agent[None, :, :2]
    assert abs(d_pos.std() / sigma_v - 1) < 0.05
    assert abs(d_pos.mean()) < 0.1 * sigma_v
    np.testing.assert_array_equal(obs_agent[..., 2:], np.broadcast_to(true_agent[:, 2:], obs_agent[..., 2:].shape))
    np.testing.assert_array_equal(np.asarray(obs.env_states.goal),
                                  np.broadcast_to(graph.env_states.goal, obs.env_states.goal.shape))

    # điểm hit LiDAR thật (bỏ tia không trúng gì — nằm ở ~5e5, mất độ chính xác float32)
    true_hits = np.asarray(_lidar_hits(env, graph))                       # (n, k, 2)
    obs_hits = np.asarray(jax.vmap(lambda g: _lidar_hits(env, g))(obs))   # (S, n, k, 2)
    real = np.linalg.norm(true_hits - true_agent[:, None, :2], axis=-1) < 1.0
    if real.any():
        d_hit = (obs_hits - true_hits[None])[:, real]
        assert abs(d_hit.std() / sigma_v - 1) < 0.05


def test_perturb_dynamics_noise_std(setup):
    env, _, _ = setup
    graph = env.reset(jr.PRNGKey(0))        # vận tốc ban đầu = 0 -> sigma nhỏ không bị clip
    sigma_w = 0.01
    keys = jr.split(jr.PRNGKey(2), 2000)
    out = jax.jit(jax.vmap(lambda k: perturb_dynamics(env, graph, k, sigma_w)))(keys)
    true_agent = np.asarray(graph.env_states.agent)
    out_agent = np.asarray(out.env_states.agent)
    d_vel = out_agent[..., 2:] - true_agent[None, :, 2:]
    assert abs(d_vel.std() / sigma_w - 1) < 0.05
    np.testing.assert_array_equal(out_agent[..., :2], np.broadcast_to(true_agent[:, :2], out_agent[..., :2].shape))
    # graph dựng lại nhất quán: states trong graph khớp env_states
    # (agent là các node đầu tiên; out đã vmap nên states có thêm trục mẫu)
    np.testing.assert_array_equal(np.asarray(out.states[:, :env.num_agents]), out_agent)


def test_costs_measured_on_true_graph(setup):
    env, algo, actors = setup
    ro = jax.jit(lambda k: noisy_test_rollout(
        env, actors[False], algo.init_rnn_state, k, 0.02, 0.1))(jr.PRNGKey(3))
    # cost lưu trong rollout = cost tính lại trên graph thật từng bước
    np.testing.assert_array_equal(np.asarray(ro.costs), np.asarray(jax.vmap(env.get_cost)(ro.graph)))
    # chuỗi graph thật liền mạch: graph[t+1] == next_graph[t]
    np.testing.assert_array_equal(np.asarray(ro.graph.states[1:]), np.asarray(ro.next_graph.states[:-1]))
    # nhiễu thật sự có tác dụng: action khác bản sigma = 0
    ro0 = jax.jit(lambda k: noisy_test_rollout(
        env, actors[False], algo.init_rnn_state, k, 0.0, 0.0))(jr.PRNGKey(3))
    assert not np.array_equal(np.asarray(ro.actions), np.asarray(ro0.actions))


def test_jit_vmap_traced_sigma_and_metrics(setup):
    env, algo, actors = setup

    def run(key, sw, sv):
        ro = noisy_test_rollout(env, actors[True], algo.init_rnn_state, key, sw, sv, stochastic=True)
        return episode_metrics(env, ro)

    fn = jax.jit(jax.vmap(run, in_axes=(0, None, None)))
    keys = jr.split(jr.PRNGKey(4), 4)
    for sw, sv in [(0.0, 0.0), (0.02, 0.05)]:   # cùng hàm đã biên dịch, sigma traced
        m = jax.device_get(fn(keys, jnp.float32(sw), jnp.float32(sv)))
        assert m["safe_agent_frac"].shape == (4,)
        assert m["agent_unsafe"].shape == (4, env.num_agents)
        assert np.all((m["safe_agent_frac"] >= 0) & (m["safe_agent_frac"] <= 1))
        assert np.all((m["reach_rate"] >= 0) & (m["reach_rate"] <= 1))


def test_noise_wrapper_class_delegates(setup):
    env, algo, actors = setup
    w = NoiseWrapper(env, sigma_w=0.0, sigma_v=0.0)
    assert w.num_agents == env.num_agents          # ủy quyền thuộc tính về env gốc
    key = jr.PRNGKey(5)
    a = jax.jit(lambda k: w.rollout(actors[False], algo.init_rnn_state, k))(key)
    b = jax.jit(lambda k: dgppo_test_rollout(env, actors[False], algo.init_rnn_state, k))(key)
    _assert_rollouts_match(b, a)
