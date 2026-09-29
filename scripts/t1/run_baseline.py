#!/usr/bin/env python3
"""T1 — chạy 1 cấu hình baseline (train + test) và ghi kết quả CSV.

    python scripts/t1/run_baseline.py --algo dgppo --env LidarSpread -n 3 \
        --seed 0 --steps 200000 --gpu 1 \
        --outdir /data/ducbm3/dgppo_runs --csv results/t1_baseline.csv

Yêu cầu: chạy trong venv dgppo_env, có shim JAX0.6 trong PYTHONPATH.
Ghi CSV (T1 chi tiết): method,env,N,obs,seed,steps,reward,cost,safety_rate,
train_runtime_s,git_commit,ckpt
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path

DGPPO = Path(__file__).resolve().parents[2] / "third_party" / "dgppo"
COMPAT = Path(__file__).resolve().parents[1] / "compat"

CSV_COLS = [
    "method", "env", "N", "obs", "seed", "steps",
    "reward", "cost", "safety_rate", "train_runtime_s", "git_commit", "ckpt",
]


def run(cmd, env, cwd):
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)


def parse_test_output(text: str) -> dict:
    """Lấy dòng tổng hợp cuối: reward, cost, safe_rate."""
    reward = cost = safe = None
    for line in text.splitlines():
        m = re.match(r"\s*reward:\s*([-\d.]+).*cost:\s*([-\d.]+).*safe_rate:\s*([\d.]+)%", line)
        if m:
            reward, cost, safe = float(m.group(1)), float(m.group(2)), float(m.group(3)) / 100.0
    return {"reward": reward, "cost": cost, "safety_rate": safe}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("-n", "--num-agents", type=int, required=True)
    ap.add_argument("--obs", type=int, default=3)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--test-epi", type=int, default=32)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["WANDB_MODE"] = "offline"
    env["WANDB_SILENT"] = "true"
    env["PYTHONPATH"] = f"{COMPAT}:{env.get('PYTHONPATH','')}"

    outdir = os.path.join(args.outdir, f"n{args.num_agents}")
    os.makedirs(outdir, exist_ok=True)

    git_commit = subprocess.run(
        ["git", "-C", str(DGPPO), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()

    tag = f"{args.algo}/{args.env}/n{args.num_agents}/seed{args.seed}"
    print(f"[T1] TRAIN {tag} steps={args.steps} gpu={args.gpu}", flush=True)
    t0 = time.time()
    tr = run(["python", "train.py", "--env", args.env, "--algo", args.algo,
              "-n", str(args.num_agents), "--obs", str(args.obs),
              "--seed", str(args.seed), "--steps", str(args.steps),
              "--log-dir", outdir], env, str(DGPPO))
    train_s = round(time.time() - t0, 1)
    if tr.returncode != 0:
        print(f"[T1] TRAIN FAILED {tag}\n{tr.stdout[-2000:]}\n{tr.stderr[-2000:]}", flush=True)
        return

    # tìm checkpoint mới nhất khớp seed
    base = Path(outdir) / args.env / args.algo
    cands = sorted(base.glob(f"seed{args.seed}_*"), key=os.path.getmtime)
    if not cands:
        print(f"[T1] NO CKPT for {tag}", flush=True)
        return
    ckpt = cands[-1]

    print(f"[T1] TEST  {tag} epi={args.test_epi}", flush=True)
    te = run(["python", "test.py", "--path", str(ckpt), "--no-video",
              "--epi", str(args.test_epi)], env, str(DGPPO))
    metrics = parse_test_output(te.stdout)
    if metrics["safety_rate"] is None:
        print(f"[T1] TEST PARSE FAIL {tag}\n{te.stdout[-1500:]}\n{te.stderr[-800:]}", flush=True)

    row = {
        "method": args.algo, "env": args.env, "N": args.num_agents, "obs": args.obs,
        "seed": args.seed, "steps": args.steps,
        "reward": metrics["reward"], "cost": metrics["cost"],
        "safety_rate": metrics["safety_rate"], "train_runtime_s": train_s,
        "git_commit": git_commit, "ckpt": str(ckpt),
    }
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[T1] DONE  {tag} -> reward={metrics['reward']} cost={metrics['cost']} "
          f"safe={metrics['safety_rate']} train={train_s}s", flush=True)


if __name__ == "__main__":
    main()
