"""Hồi quy CM-DGPPO: ngưỡng 0 => trùng DGPPO từng bit (CPU); relax/fixed chạy và cập nhật đúng."""
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
def test_fixed_zero_equals_dgppo():
    # fixed với delta = 0 => không đổi cost => trùng DGPPO từng bit
    env = make_noisy_env("LidarSpread", N_AGENT, num_obs=2, max_step=T)
    ref = DGPPO(**small_kwargs(env))
    cm = CMDGPPO(**small_kwargs(env), cm_variant="fixed", cm_fixed_delta=0.0)
    assert max_param_diff(ref, cm) == 0.0  # cùng khởi tạo
    run_updates(ref)
    run_updates(cm)
    assert max_param_diff(ref, cm) < 1e-5


def test_fixed_relax_runs():
    env = make_noisy_env("LidarSpread", N_AGENT, num_obs=2, max_step=T)
    cm = CMDGPPO(**small_kwargs(env), cm_variant="fixed", cm_fixed_delta=-0.02)
    infos = run_updates(cm)
    assert infos[-1]["cm/margin"] == -0.02
    for k, v in infos[-1].items():
        assert np.all(np.isfinite(np.asarray(v))), k


@requires_cpu
def test_relax_eta0_equals_dgppo():
    # eta = 0 => eps giữ 0 => "relax" trùng DGPPO từng bit
    env = make_noisy_env("LidarSpread", N_AGENT, num_obs=2, max_step=T)
    ref = DGPPO(**small_kwargs(env))
    cm = CMDGPPO(**small_kwargs(env), cm_variant="relax", cm_eta=0.0)
    run_updates(ref)
    infos = run_updates(cm)
    assert max_param_diff(ref, cm) < 1e-5
    assert cm.eps == 0.0 and "cm/viol_rate" in infos[-1]


def test_relax_eps_direction_and_bounds():
    env = make_noisy_env("LidarSpread", N_AGENT, num_obs=2, max_step=T)
    cm = CMDGPPO(**small_kwargs(env), cm_variant="relax", cm_alpha=0.05, cm_eta=0.01, cm_eps_max=0.05)
    assert cm._update_eps(0.0) > 0                      # ít va chạm hơn alpha => nới
    for _ in range(1000):
        cm._update_eps(0.0)
    assert cm.eps == 0.05                               # kẹp ở eps_max
    for _ in range(1000):
        cm._update_eps(1.0)                             # va chạm nhiều => siết
    assert cm.eps == 0.0                                # kẹp ở 0 (= DGPPO)
    h = np.zeros((2, 4, 3, 2)) - 1.0
    h[0, 1, 2, 0] = 0.01                                # episode 0, UAV 2 va chạm
    assert cm._agent_violation_rate(h) == 1 / 6         # 1 trên 6 cặp (episode, UAV)


def test_relax_runs_and_loosens_cost():
    env = make_noisy_env("LidarSpread", N_AGENT, num_obs=2, max_step=T)
    cm = CMDGPPO(**small_kwargs(env), cm_variant="relax", cm_eta=0.01)
    cm.eps = 0.03
    infos = run_updates(cm, n_updates=1)
    assert infos[-1]["cm/eps"] == 0.03 and infos[-1]["cm/margin"] == -0.03
    for k, v in infos[-1].items():
        assert np.all(np.isfinite(np.asarray(v))), k


def test_relax_eta_from_aci_gamma():
    # eta = gamma_ACI * eps_max = 0.005 * 0.05; cm_eta đặt tay thì ưu tiên
    env = make_noisy_env("LidarSpread", N_AGENT, num_obs=2, max_step=T)
    cm = CMDGPPO(**small_kwargs(env), cm_variant="relax", cm_eps_max=0.05)
    assert abs(cm.cm_eta - 0.00025) < 1e-12
    assert CMDGPPO(**small_kwargs(env), cm_variant="relax", cm_eta=0.002).cm_eta == 0.002


def test_margin_sign_relaxes_cost():
    # nới (eps > 0): cost h - eps không bao giờ lớn hơn cost gốc, và có chỗ nhỏ hẳn
    from methods.cm_dgppo.costs import shape_cost
    h = jnp.linspace(-0.2, 0.2, 401)
    relaxed, orig = shape_cost(h - 0.03), shape_cost(h)
    assert bool(jnp.all(relaxed <= orig + 1e-7)) and bool(jnp.any(relaxed < orig - 1e-6))
    # h trong (0, eps]: va chạm thật nhưng khi train được coi là an toàn (cost < 0)
    assert float(shape_cost(jnp.array(0.02) - 0.03)) < 0 < float(shape_cost(jnp.array(0.02)))


def test_nonzero_margin_changes_vh_training():
    # ngưỡng khác 0 phải đổi việc học GCBF Vh (so với DGPPO), không chỉ đổi số log
    env = make_noisy_env("LidarSpread", N_AGENT, num_obs=2, max_step=T)
    ref = DGPPO(**small_kwargs(env))
    cm = CMDGPPO(**small_kwargs(env), cm_variant="fixed", cm_fixed_delta=-0.03)
    run_updates(ref, n_updates=1)
    run_updates(cm, n_updates=1)
    diff = jtu.tree_map(lambda x, y: float(jnp.abs(x - y).max()), ref.Vh_train_state.params, cm.Vh_train_state.params)
    assert max(jtu.tree_leaves(diff)) > 1e-6


def test_unknown_cm_key_rejected():
    from methods.cm_dgppo.build import build_algo, build_env
    cfg = {"method": "cm_dgppo_relax", "env": "LidarSpread", "N": N_AGENT, "n_obs": 2,
           "noise": {"sigma_w": 0.0, "sigma_v": 0.0}, "cm": {"cm_epsmax": 0.03}}
    with pytest.raises(ValueError, match="cm_epsmax"):
        build_algo(cfg, build_env(cfg), seed=0, steps=10, batch_size=N_ENV * T)
