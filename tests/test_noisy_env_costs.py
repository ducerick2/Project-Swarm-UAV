import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from dgppo.env import make_env
from methods.cm_dgppo.costs import raw_cost_obs, raw_cost_true, shape_cost, sigma_scale
from methods.cm_dgppo.noisy_env import make_noisy_env

ENVS = ["LidarSpread", "LidarLine"]


def rollout_states(env, key, T=20):
    g = env.reset(key)
    graphs = [g]
    for t in range(T):
        a = jr.uniform(jr.fold_in(key, t), (env.num_agents, 2), minval=-1, maxval=1)
        g, *_ = env.step(g, a)
        graphs.append(g)
    return graphs


@pytest.mark.parametrize("env_id", ENVS)
def test_noise_free_matches_base(env_id):
    base = make_env(env_id, 3, num_obs=3)
    env = make_noisy_env(env_id, 3, num_obs=3)
    for g in rollout_states(env, jr.PRNGKey(0)):
        # cost của env có nhiễu (tính trên trạng thái thật) == cost gốc của DGPPO trên graph
        np.testing.assert_allclose(env.get_cost(g), base.get_cost(g), atol=1e-6)
        np.testing.assert_allclose(shape_cost(raw_cost_obs(env, g)), base.get_cost(g), atol=1e-6)
        np.testing.assert_allclose(raw_cost_true(env, g.env_states), raw_cost_obs(env, g), atol=1e-6)


@pytest.mark.parametrize("env_id", ENVS)
def test_noise_free_step_matches_base(env_id):
    base = make_env(env_id, 3, num_obs=3)
    env = make_noisy_env(env_id, 3, num_obs=3)
    g = env.reset(jr.PRNGKey(1))
    a = jnp.full((3, 2), 0.3)
    g_noisy, r1, c1, _, _ = env.step(g, a)
    g_base, r2, c2, _, _ = base.step(g, a)
    np.testing.assert_allclose(g_noisy.nodes, g_base.nodes, atol=1e-6)
    np.testing.assert_allclose(r1, r2, atol=1e-7)
    np.testing.assert_allclose(c1, c2, atol=1e-6)
    np.testing.assert_allclose(env.nominal_next_graph(g, a).nodes, g_base.nodes, atol=1e-6)


def test_sensor_noise_only_touches_observation():
    env = make_noisy_env("LidarSpread", 3, sigma_v=0.02, num_obs=3)
    clean = make_noisy_env("LidarSpread", 3, num_obs=3)
    key = jr.PRNGKey(2)
    g, g0 = env.reset(key), clean.reset(key)
    np.testing.assert_allclose(g.env_states.agent, g0.env_states.agent)  # trạng thái thật không đổi
    assert not np.allclose(g.nodes, g0.nodes)                              # quan sát bị nhiễu
    resid = raw_cost_true(env, g.env_states) - raw_cost_obs(env, g)
    assert np.abs(resid).max() > 0


def test_dynamics_noise_changes_true_state():
    env = make_noisy_env("LidarSpread", 3, sigma_w=0.01, num_obs=3)
    g = env.reset(jr.PRNGKey(3))
    a = jnp.zeros((3, 2))
    g1, *_ = env.step(g, a)
    nom = env.nominal_next_graph(g, a)
    assert not np.allclose(g1.env_states.agent, nom.env_states.agent)


def test_sigma_scale_lower_bound_and_jit():
    env = make_noisy_env("LidarSpread", 3, num_obs=3)
    g = env.reset(jr.PRNGKey(4))
    s = jax.jit(lambda g: sigma_scale(env, g, 0.05, 5.0))(g)
    assert s.shape == (3, 2) and np.all(np.asarray(s) >= 0.05 - 1e-7)


@pytest.mark.parametrize("env_id", ENVS)
def test_same_key_same_episode_as_base(env_id):
    # common random numbers: cùng key => cùng trạng thái đầu và cùng quỹ đạo như env gốc khi sigma = 0
    base = make_env(env_id, 3, num_obs=3)
    env = make_noisy_env(env_id, 3, num_obs=3)
    noisy = make_noisy_env(env_id, 3, sigma_w=0.01, sigma_v=0.01, num_obs=3)
    key = jr.PRNGKey(7)
    gb, ge, gn = base.reset(key), env.reset(key), noisy.reset(key)
    np.testing.assert_array_equal(np.asarray(gb.env_states.agent), np.asarray(ge.env_states.agent))
    np.testing.assert_array_equal(np.asarray(gb.env_states.agent), np.asarray(gn.env_states.agent))
    for t in range(10):
        a = jr.uniform(jr.fold_in(key, t), (3, 2), minval=-1, maxval=1)
        gb, rb, cb, _, _ = base.step(gb, a)
        ge, re, ce, _, _ = env.step(ge, a)
        np.testing.assert_allclose(re, rb, atol=1e-7)
        np.testing.assert_allclose(ce, cb, atol=1e-6)
