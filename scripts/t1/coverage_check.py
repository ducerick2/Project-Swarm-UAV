#!/usr/bin/env python3
"""Chẩn đoán coverage 1 episode của 1 checkpoint (tái lập ĐÚNG episode như test.py/video).

    python scripts/t1/coverage_check.py --path <ckpt_dir> --epi 6 [--test-seed 1234]

In: vị trí goal, hộp bao goal (xem có dồn góc không), vị trí agent đầu/cuối,
và khoảng cách mỗi goal -> agent gần nhất ở bước cuối (coi là "phủ" nếu <= ngưỡng dist2goal).
Dùng cùng cách sinh key với test.py nên episode khớp với video cùng chỉ số.
"""
from __future__ import annotations
import argparse, os, yaml
import functools as ft
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr

from dgppo.env import make_env
from dgppo.algo import make_algo
from dgppo.trainer.utils import test_rollout
from dgppo.utils.utils import tree_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--epi", type=int, default=0)
    ap.add_argument("--test-seed", type=int, default=1234)
    ap.add_argument("--n-epi", type=int, default=32, help="phải >= epi+1, khớp --epi của lần render")
    args = ap.parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    with open(os.path.join(args.path, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    env = make_env(env_id=config.env, num_agents=config.num_agents, num_obs=config.obs)
    model_path = os.path.join(args.path, "models")
    step = max(int(m) for m in os.listdir(model_path) if m.isdigit())
    algo = make_algo(
        algo=config.algo, env=env, node_dim=env.node_dim, edge_dim=env.edge_dim,
        state_dim=env.state_dim, action_dim=env.action_dim, n_agents=env.num_agents,
        cost_weight=config.cost_weight, actor_gnn_layers=config.actor_gnn_layers,
        Vl_gnn_layers=config.Vl_gnn_layers,
        Vh_gnn_layers=getattr(config, "Vh_gnn_layers", 1),
        lr_actor=config.lr_actor, lr_Vl=config.lr_Vl, max_grad_norm=2.0,
        seed=config.seed, use_rnn=config.use_rnn, rnn_layers=config.rnn_layers, use_lstm=config.use_lstm)
    algo.load(model_path, step)
    act_fn = jax.jit(algo.act)

    # tái lập key episode y như test.py
    test_keys = jr.split(jr.PRNGKey(args.test_seed), 1000)[: args.n_epi]
    key_x0, _ = jr.split(test_keys[args.epi], 2)
    rollout_fn = jax.jit(ft.partial(test_rollout, env, act_fn, algo.init_rnn_state, stochastic=False))
    rollout = rollout_fn(key_x0)

    nA, nG = env.num_agents, env.num_goals
    thr = env._params.get("dist2goal", 0.01)

    g0 = tree_index(rollout.graph, 0)              # bước đầu
    gT = tree_index(rollout.next_graph, env.max_episode_steps - 1)  # bước cuối
    goals = np.array(g0.type_states(1, nG))[:, :2]
    a0 = np.array(g0.type_states(0, nA))[:, :2]
    aT = np.array(gT.type_states(0, nA))[:, :2]

    def cover(agents):
        d = np.linalg.norm(goals[:, None, :] - agents[None, :, :], axis=-1).min(axis=1)
        return d
    dT = cover(aT)

    area = env.area_size
    print(f"env={config.env} N_agent={nA} N_goal={nG} area_size={area} dist2goal_thr={thr}")
    print(f"reward(sum)={float(rollout.rewards.sum()):.3f}  cost(max)={float(rollout.costs.max()):.3f}")
    print(f"\nGoal positions:            {np.round(goals,3).tolist()}")
    print(f"  goal bbox x:[{goals[:,0].min():.2f},{goals[:,0].max():.2f}] y:[{goals[:,1].min():.2f},{goals[:,1].max():.2f}]  (area 0..{area})")
    print(f"Agent START positions:     {np.round(a0,3).tolist()}")
    print(f"Agent FINAL positions:     {np.round(aT,3).tolist()}")
    print(f"\nPer-goal dist->nearest agent (FINAL):")
    for i,(gp,d) in enumerate(zip(goals,dT)):
        print(f"  goal{i} @({gp[0]:.2f},{gp[1]:.2f}): dist={d:.3f}  -> {'PHỦ' if d<=thr else ('gần' if d<0.1 else 'CHƯA PHỦ')}")
    print(f"\n=> covered(final, <= {thr}): {int((dT<=thr).sum())}/{nG} | <=0.1: {int((dT<=0.1).sum())}/{nG} | mean dist={dT.mean():.3f}")


if __name__ == "__main__":
    main()
