import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dgppo.algo import DGPPO
from methods.cm_dgppo.costs import sigma_scale
from methods.cm_dgppo.deploy import OnlineConfig, make_deploy_fn
from methods.cm_dgppo.noisy_env import _InflatedObs, make_noisy_env

T = 32


def small_algo(env):
    return DGPPO(env=env, node_dim=env.node_dim, edge_dim=env.edge_dim, state_dim=env.state_dim,
                 action_dim=env.action_dim, n_agents=env.num_agents, batch_size=64, seed=0)


def test_inflation_geometry():
    env = make_noisy_env("LidarSpread", 3, num_obs=3)
    g = env.reset(jr.PRNGKey(0))
    g0 = env.set_delta(g, jnp.zeros(3))
    np.testing.assert_array_equal(np.asarray(g0.edges), np.asarray(g.edges))  # delta = 0: trùng từng bit
    g1 = env.set_delta(g, jnp.array([1.0, 0.0, 0.0]))
    assert not np.allclose(np.asarray(g1.edges), np.asarray(g.edges))

    sig = np.asarray(sigma_scale(env, g, env.sigma_base, env.sigma_kappa))   # (3, 2)
    s = g.env_states
    margin = jnp.array([[sig[0, 0], sig[0, 1]], [0, 0], [0, 0]])
    b0 = env.edge_blocks(s, None)
    b1 = env.edge_blocks(_InflatedObs(s.agent, s.goal, s.obstacle, margin), None)
    f0, f1 = np.asarray(b0[0].edge_feats), np.asarray(b1[0].edge_feats)
    d0 = np.linalg.norm(f0[0, 1:, :2], axis=-1)
    d1 = np.linalg.norm(f1[0, 1:, :2], axis=-1)
    np.testing.assert_allclose(d0 - d1, np.minimum(sig[0, 0], d0 - 1e-3), atol=1e-5)  # agent 0 thấy gần hơn m
    np.testing.assert_array_equal(f0[1:], f1[1:])                                       # agent khác không đổi


def test_eta_zero_matches_plain_rollout():
    env = make_noisy_env("LidarSpread", 2, num_obs=2, max_step=T)
    algo = small_algo(env)
    fn = make_deploy_fn(env, algo, n_episodes=1, cfg=OnlineConfig(eta=0.0))
    key = jr.PRNGKey(5)
    out = jax.device_get(fn(algo.params, key[None]))

    g, rnn, rewards = env.reset(jr.split(key, 1)[0]), algo.init_rnn_state, []
    for _ in range(T):
        a, rnn = algo.act(g, rnn, algo.params)
        g, r, *_ = env.step(g, a)
        rewards.append(float(r))
    np.testing.assert_allclose(out["reward"][0, 0], rewards, rtol=1e-5, atol=1e-6)
    assert np.all(out["delta"] == 0)


def test_online_delta_moves_and_carries_over():
    env = make_noisy_env("LidarSpread", 2, num_obs=2, max_step=T)
    algo = small_algo(env)
    # alpha = 1: e - alpha <= 0 mọi cửa sổ => Delta giảm đều về lo, và được mang qua episode
    fn = make_deploy_fn(env, algo, n_episodes=3, cfg=OnlineConfig(eta=0.1, alpha=1.0, window=8, lo=-0.5, hi=2.0))
    out = jax.device_get(fn(algo.params, jr.split(jr.PRNGKey(1), 2)))
    d = out["delta"]                                  # (S, E, T, n)
    assert np.all(np.diff(d.reshape(2, -1, 2), axis=1) <= 1e-7)
    assert d[:, 1, 0].max() < 0                       # episode 2 bắt đầu từ Delta của episode 1
    np.testing.assert_allclose(d[:, -1, -1], -0.5, atol=1e-6)
