#!/usr/bin/env python3
"""Huấn luyện CM-DGPPO (mạch T3) hoặc baseline DGPPO trên env LiDAR có nhiễu.

    python scripts/train_cm.py --config configs/t3/cm_dgppo_relax.yaml [--seed 3] [--steps 50 --debug]

Đọc YAML theo lược đồ chung (configs/schema.md) cộng khối `cm:`; dựng env có nhiễu
(methods/cm_dgppo/noisy_env.py) cho cả train và eval; dùng Trainer gốc của DGPPO.
Method: cm_dgppo_relax (adaptive) hoặc fixed_margin (ngưỡng cố định); xem configs/t3/.
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import tempfile

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n-env-train", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    ap.add_argument("--debug", action="store_true", help="không lưu log, tắt W&B")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="ghi đè config, vd. --set cm.cm_fixed_delta=0.01 noise.sigma_v=0.02")
    args = ap.parse_args()

    from methods.cm_dgppo.build import apply_overrides, build_algo, build_env

    with open(args.config) as f:
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    seed = cfg["seed"] if args.seed is None else args.seed
    train_cfg = cfg.get("train", {})
    steps = args.steps or train_cfg.get("steps", 200_000)
    n_env_train = args.n_env_train or train_cfg.get("n_env_train", 128)
    batch_size = args.batch_size or train_cfg.get("batch_size", 16384)
    # ghi lại giá trị THỰC dùng (kể cả ghi đè từ dòng lệnh) vào config.yaml của run
    cfg["train"] = train_cfg | {"steps": steps, "n_env_train": n_env_train, "batch_size": batch_size}
    if cfg["method"] == "fixed_margin" and "tag" not in cfg:
        cfg["tag"] = f"d{cfg.get('cm', {}).get('cm_fixed_delta', 0.0):g}"  # khớp mặc định của CMDGPPO

    if args.debug or cfg.get("logging", {}).get("backend", "wandb") == "none":
        os.environ["WANDB_MODE"] = "disabled"
    np.random.seed(seed)

    from dgppo.trainer.trainer import Trainer

    env, env_test = build_env(cfg), build_env(cfg)
    algo = build_algo(cfg, env, seed=seed, steps=steps, batch_size=batch_size)
    method = cfg["method"]

    stamp = datetime.datetime.now().strftime("%m%d%H%M%S")
    tag = cfg.get("tag")  # vd. tag: d0.02 để phân biệt các run quét delta
    run = f"seed{seed}_{stamp}" if not tag else f"{tag}_seed{seed}_{stamp}"
    log_dir = os.path.join(args.log_dir, cfg["env"], method, run)
    if args.debug:  # W&B vẫn tạo thư mục log_dir, nên trỏ vào thư mục tạm
        log_dir = tempfile.mkdtemp(prefix="cm_debug_")
    else:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg | {"seed": seed, "git_commit": git_commit(), "algo_config": {
                k: v for k, v in algo.config.items() if isinstance(v, (int, float, str, bool))}}, f)

    trainer = Trainer(
        env=env, env_test=env_test, algo=algo, gamma=0.99, log_dir=log_dir, n_env_train=n_env_train,
        n_env_test=train_cfg.get("n_env_test", 32), seed=seed,
        params={"run_name": f"{method}_seed{seed:03}_{stamp}", "training_steps": steps,
                "eval_interval": train_cfg.get("eval_interval", 50), "eval_epi": 1,
                "save_interval": train_cfg.get("save_interval", 50)},
        save_log=not args.debug,
    )
    trainer.train()


if __name__ == "__main__":
    main()
