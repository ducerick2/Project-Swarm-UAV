#!/usr/bin/env python3
"""Đánh giá checkpoint (train_cm.py) trên lưới nhiễu / dịch chuyển phân phối, ghi CSV.

    # offline: chính sách cố định, 256 episode mỗi điểm lưới
    python scripts/eval_cm.py --run logs/LidarSpread/cm_dgppo_full/seed0_XXXX \\
        --sigma-w 0 0.005 0.01 --sigma-v 0 0.01 0.02 --n-obs 3 6 -n 3 5

    # online: cập nhật biên lúc triển khai (methods/cm_dgppo/deploy.py)
    python scripts/eval_cm.py --run ... --online-eta 0.05 --streams 32 --episodes 16

Ghi 2 file:
- `--csv` (mặc định results/t3/eval.csv): đúng cột chung của analysis/logging_csv.py.
  Cột `sigma` mã hóa điểm lưới: "w<sigma_w>_v<sigma_v>_obs<n_obs>".
- `--csv-full` (mặc định results/t3/eval_full.csv): mọi chỉ số + tham số của run.

Key eval cố định (`--eval-seed`, mặc định 10000) cho MỌI phương pháp và seed: cùng tập
episode để so sánh cặp; tách khỏi key train (seed 0..19) và key hiệu chỉnh.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "third_party", "dgppo")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import yaml  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def latest_step(model_dir: str) -> int:
    return max(int(d) for d in os.listdir(model_dir) if d.isdigit())


def cvar_low(x: np.ndarray, q: float = 0.05) -> float:
    """Trung bình của q phần thấp nhất (đuôi xấu của khoảng hở an toàn)."""
    x = np.sort(np.asarray(x).ravel())
    k = max(1, int(np.ceil(q * x.size)))
    return float(x[:k].mean())


def compute_metrics(out: dict) -> dict:
    """out: mảng (S, E, T, ...) từ deploy.make_deploy_fn."""
    raw = np.asarray(out["raw"])                         # (S, E, T, n, n_cost) cost thô thật
    viol = raw.max(axis=-1) > 0                          # (S, E, T, n)
    agent_ep_unsafe = viol.any(axis=2)                   # (S, E, n)
    E = raw.shape[1]
    clearance = -raw.max(axis=(2, 3, 4))                 # (S, E) khoảng hở nhỏ nhất mỗi episode
    ends = np.asarray(out["window_end"])                 # (S, E, T)
    werr = np.asarray(out["window_err"])                 # (S, E, T, n)
    delta = np.asarray(out["delta"])                     # (S, E, T, n)
    return {
        "task_cost": float((-np.asarray(out["reward"]).sum(axis=2)).mean()),
        "safety_rate": float(1 - agent_ep_unsafe.mean()),             # định nghĩa DGPPO (agent-episode)
        "safety_rate_swarm": float(1 - agent_ep_unsafe.any(axis=-1).mean()),
        "safety_rate_late": float(1 - agent_ep_unsafe[:, E // 2:].mean()),  # nửa sau (sau khi thích nghi)
        "violation_freq": float(viol.mean()),
        "clearance_mean": float(clearance.mean()),
        "clearance_cvar5": cvar_low(clearance, 0.05),
        "online_window_err": float(werr[ends].mean()) if ends.any() else float("nan"),
        "online_delta_mean": float(delta.mean()),
        "online_delta_final": float(delta[:, -1, -1].mean()),
    }


def append_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="thư mục log của train_cm.py")
    ap.add_argument("--step", type=int, default=None, help="mặc định: checkpoint mới nhất")
    ap.add_argument("--sigma-w", type=float, nargs="*", default=None)
    ap.add_argument("--sigma-v", type=float, nargs="*", default=None)
    ap.add_argument("--n-obs", type=int, nargs="*", default=None)
    ap.add_argument("-n", "--num-agents", type=int, nargs="*", default=None)
    ap.add_argument("--streams", type=int, default=256, help="số luồng triển khai song song")
    ap.add_argument("--episodes", type=int, default=1, help="số episode nối tiếp mỗi luồng")
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--online-eta", type=float, default=0.0, help="0 = eval offline")
    ap.add_argument("--online-alpha", type=float, default=0.05)
    ap.add_argument("--online-window", type=int, default=16)
    ap.add_argument("--online-lo", type=float, default=-1.0)
    ap.add_argument("--online-hi", type=float, default=3.0)
    ap.add_argument("--online-shared", action="store_true", help="một Delta chung cho cả bầy")
    ap.add_argument("--label", default=None, help="tên phương pháp trong CSV (mặc định: method[+online])")
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "t3", "eval.csv"))
    ap.add_argument("--csv-full", default=os.path.join(ROOT, "results", "t3", "eval_full.csv"))
    args = ap.parse_args()

    import jax
    import jax.random as jr

    from analysis.logging_csv import append_row
    from methods.cm_dgppo.build import build_algo, build_env
    from methods.cm_dgppo.deploy import OnlineConfig, make_deploy_fn

    with open(os.path.join(args.run, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_dir = os.path.join(args.run, "models")
    step = latest_step(model_dir) if args.step is None else args.step
    with open(os.path.join(model_dir, str(step), "actor.pkl"), "rb") as f:
        actor_params = pickle.load(f)
    conformal = {}
    cpath = os.path.join(model_dir, str(step), "conformal.pkl")
    if os.path.exists(cpath):
        with open(cpath, "rb") as f:
            conformal = pickle.load(f)

    noise = cfg.get("noise", {})
    grid_w = args.sigma_w if args.sigma_w else [noise.get("sigma_w", 0.0)]
    grid_v = args.sigma_v if args.sigma_v else [noise.get("sigma_v", 0.0)]
    grid_obs = args.n_obs if args.n_obs else [cfg.get("n_obs", 3)]
    grid_n = args.num_agents if args.num_agents else [cfg["N"]]
    online = OnlineConfig(eta=args.online_eta, alpha=args.online_alpha, window=args.online_window,
                          lo=args.online_lo, hi=args.online_hi, per_agent=not args.online_shared)
    label = args.label or (cfg["method"] + ("+online" if args.online_eta > 0 else ""))
    commit = git_commit()
    keys = jr.split(jr.PRNGKey(args.eval_seed), args.streams)
    q_h = [c["q"] for c in conformal.get("h", [])]
    q_cbf = [c["q"] for c in conformal.get("cbf", [])]

    for n_agents in grid_n:
        for n_obs in grid_obs:
            for sw in grid_w:
                for sv in grid_v:
                    env = build_env(cfg, sigma_w=sw, sigma_v=sv, n_obs=n_obs, num_agents=n_agents)
                    algo = build_algo(cfg, env, seed=cfg["seed"], steps=1, batch_size=1, policy_only=True)
                    algo.policy_train_state = algo.policy_train_state.replace(params=actor_params)
                    params = algo.params
                    fn = make_deploy_fn(env, algo, args.episodes, online)

                    jax.block_until_ready(fn(params, keys[:1]))  # biên dịch
                    t0 = time.time()
                    out = jax.device_get(fn(params, keys))
                    runtime = time.time() - t0
                    m = compute_metrics(out)

                    act = jax.jit(algo.act)
                    g = env.reset(jr.PRNGKey(0))
                    jax.block_until_ready(act(g, algo.init_rnn_state, params))
                    t1 = time.time()
                    for _ in range(100):
                        a, _ = act(g, algo.init_rnn_state, params)
                    jax.block_until_ready(a)
                    infer_ms = (time.time() - t1) * 10.0

                    sigma_code = f"w{sw:g}_v{sv:g}_obs{n_obs}"
                    append_row(args.csv, {
                        "method": label, "env": cfg["env"], "N": n_agents, "sigma": sigma_code,
                        "seed": cfg["seed"], "task_cost": m["task_cost"], "safety_rate": m["safety_rate"],
                        "violation_freq": m["violation_freq"], "runtime_s": runtime, "git_commit": commit,
                    })
                    append_rows(args.csv_full, [{
                        "label": label, "method": cfg["method"], "tag": cfg.get("tag", ""),
                        "env": cfg["env"], "seed": cfg["seed"], "step": step, "run": args.run,
                        "train_sigma_w": noise.get("sigma_w", 0.0), "train_sigma_v": noise.get("sigma_v", 0.0),
                        "train_N": cfg["N"], "train_n_obs": cfg.get("n_obs", 3),
                        "sigma_w": sw, "sigma_v": sv, "N": n_agents, "n_obs": n_obs,
                        "fixed_delta": cfg.get("cm", {}).get("cm_fixed_delta", "") if cfg["method"] == "fixed_margin" else "",
                        "q_h": ";".join(f"{x:.4g}" for x in q_h), "q_cbf": ";".join(f"{x:.4g}" for x in q_cbf),
                        "online_eta": online.eta, "online_alpha": online.alpha, "online_window": online.window,
                        "online_per_agent": online.per_agent, "streams": args.streams, "episodes": args.episodes,
                        **m, "runtime_s": runtime, "infer_ms": infer_ms, "git_commit": commit,
                    }])
                    print(f"[eval] {label} N={n_agents} obs={n_obs} {sigma_code}: "
                          f"safety={m['safety_rate']:.3f} cost={m['task_cost']:.3f} "
                          f"viol={m['violation_freq']:.4f} clr_cvar5={m['clearance_cvar5']:.4f}"
                          + (f" Δ={m['online_delta_final']:.3f} werr={m['online_window_err']:.3f}"
                             if online.eta > 0 else ""), flush=True)


if __name__ == "__main__":
    main()
