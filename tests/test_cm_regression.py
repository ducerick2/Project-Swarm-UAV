"""Hồi quy: không nhiễu => mọi điểm số = 0 => q = 0 => CM-DGPPO trùng DGPPO từng bit (gần đúng float)."""
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

from dgppo.algo import DGPPO
from methods.cm_dgppo.algo import CMDGPPO
from methods.cm_dgppo.noisy_env import make_noisy_env

N_ENV, T, N_AGENT = 4, 32, 2

# scatter-add trên GPU không tất định: DGPPO so với chính nó đã lệch ~3e-4 sau 2 update.
# So sánh từng bit chỉ có nghĩa trên CPU: JAX_PLATFORMS=cpu pytest tests/test_cm_regression.py
requires_cpu = pytest.mark.skipif(jax.default_backend() != "cpu", reason="so sánh từng bit cần CPU (tất định)")


def small_kwargs(env):
    return dict(
        env=env, node_dim=env.node_dim, edge_dim=env.edge_dim, state_dim=env.state_dim,
        action_dim=env.action_dim, n_agents=env.num_agents, batch_size=N_ENV * T, rnn_step=16,
        seed=0, train_steps=10,
    )


def run_updates(algo, n_updates=2):
    infos = []
    for step in range(n_updates):
        np.random.seed(step)  # DGPPO.update xáo trộn bằng np.random toàn cục
        rollouts = algo.collect(algo.params, jr.split(jr.PRNGKey(100 + step), N_ENV))
        infos.append(algo.update(rollouts, step))
    return infos


def max_param_diff(a, b):
    diffs = jtu.tree_map(lambda x, y: float(jnp.abs(x - y).max()), a.params, b.params)
    return max(jtu.tree_leaves(diffs))


@requires_cpu
@pytest.mark.parametrize("variant", ["state", "cbf", "full"])
def test_noise_free_cm_equals_dgppo(variant):
    env = make_noisy_env("LidarSpread", N_AGENT, num_obs=2, max_step=T)
    ref = DGPPO(**small_kwargs(env))
    cm = CMDGPPO(**small_kwargs(env), cm_variant=variant)
    assert max_param_diff(ref, cm) == 0.0  # cùng khởi tạo

    run_updates(ref)
    infos = run_updates(cm)
    assert max_param_diff(ref, cm) < 1e-5
    q_h, q_cbf = cm.margins()
    assert np.all(q_h == 0) and np.all(q_cbf == 0)
    assert any(k.startswith("cm/") for k in infos[-1])


def test_noisy_margins_become_positive():
    env = make_noisy_env("LidarSpread", N_AGENT, sigma_w=0.01, sigma_v=0.01, num_obs=2, max_step=T)
    cm = CMDGPPO(**small_kwargs(env), cm_variant="full", cm_gamma=0.05)
    infos = run_updates(cm, n_updates=3)
    q_h, q_cbf = cm.margins()
    assert np.all(q_h > 0), q_h
    assert np.all(np.isfinite(q_cbf))
    for k, v in infos[-1].items():
        assert np.all(np.isfinite(np.asarray(v))), k


@pytest.mark.parametrize("variant", ["fixed", "scalar"])
def test_unstructured_margin_variants(variant):
    env = make_noisy_env("LidarSpread", N_AGENT, sigma_v=0.01, num_obs=2, max_step=T)
    cm = CMDGPPO(**small_kwargs(env), cm_variant=variant, cm_fixed_delta=0.02, cm_eta=0.01, cm_delta_max=0.05)
    infos = run_updates(cm)
    d = cm.delta()
    if variant == "fixed":
        np.testing.assert_allclose(d, 0.02)
    else:
        assert np.all((0 <= d) & (d <= 0.05))
        assert "cm/delta_err" in infos[-1]
    for k, v in infos[-1].items():
        assert np.all(np.isfinite(np.asarray(v))), k
