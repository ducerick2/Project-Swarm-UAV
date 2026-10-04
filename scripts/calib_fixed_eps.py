#!/usr/bin/env python3
"""Hiệu chỉnh tham số CM-DGPPO từ dữ liệu DGPPO của T1 (không nhiễu), RIÊNG cho từng môi trường.

(1) Bước cập nhật của `relax` (không cần GPU):
    tau = trung vị (qua các seed) của số bước để DGPPO hội tụ về an toàn: bước đầu tiên t mà
          trung bình trượt 10 lần eval (10k bước) của eval/unsafe_frac <= alpha TẠI t và ở dưới
          alpha >= 90% thời gian còn lại. Đọc eval/unsafe_frac độ chính xác đầy đủ từ file W&B
          offline của T1 (log console chỉ in 2 chữ số thập phân).
    eta = eps_max / (alpha * tau)        (eps đi từ 0 lên eps_max trong đúng tau bước)

(2) Ngưỡng nới CỐ ĐỊNH cho baseline `fixed_relax` (cần GPU/CPU để rollout):
    c_1 <= ... <= c_n : khoảng hở nhỏ nhất của mỗi cặp (UAV, episode) dưới DGPPO
                        (lấy loại sát hơn trong UAV–UAV và UAV–vật cản; c < 0 là va chạm)
    eps_fixed = c_k,   k = floor(alpha * (n + 1))      (phân vị alpha có hiệu chỉnh mẫu hữu hạn)
    Dùng tập VALIDATION (--eval-seed 20000), tách khỏi tập test 10000 dùng cho bảng kết quả.

    python scripts/calib_fixed_eps.py --tau-only                 # chỉ (1)
    python scripts/calib_fixed_eps.py [--alpha 0.05] [--envs LidarSpread LidarLine]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "third_party", "dgppo"), os.path.join(ROOT, "scripts")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def wandb_history(path: str, key: str):
    """Đọc (step, value) của `key` từ file .wandb offline (định dạng log khối 32 KB kiểu LevelDB)."""
    from wandb.proto import wandb_internal_pb2 as pb

    data = open(path, "rb").read()
    off, buf, out = 7, b"", []  # bỏ header file 7 byte
    while off + 7 <= len(data):
        left = 32768 - (off % 32768)
        if left < 7:
            off += left
            continue
        _, ln, typ = struct.unpack("<IHB", data[off:off + 7])
        off += 7
        if typ == 0 and ln == 0:
            off += left - 7
            continue
        chunk = data[off:off + ln]
        off += ln
        if typ == 2:
            buf = chunk
            continue
        if typ == 3:
            buf += chunk
            continue
        rec_bytes = chunk if typ == 1 else buf + chunk
        buf = b""
        r = pb.Record()
        try:
            r.ParseFromString(rec_bytes)
        except Exception:
            continue
        if r.WhichOneof("record_type") != "history":
            continue
        kv = {(i.key or "/".join(i.nested_key)): i.value_json for i in r.history.item}
        if key in kv:
            out.append((int(kv["_step"]), float(kv[key])))
    return out


def tau_and_eta(args) -> None:
    import numpy as np

    for env_id in args.envs:
        taus = []
        for run in sorted(glob.glob(f"{args.t1_dir}/{env_id}/dgppo/seed*")):
            files = glob.glob(f"{run}/wandb/offline-run-*/run-*.wandb")
            if not files:
                continue
            pts = wandb_history(files[0], "eval/unsafe_frac")
            if len(pts) < 20:
                continue
            u = np.array([v for _, v in pts])
            s = np.array([t for t, _ in pts])  # _step của W&B = bước train (Trainer log với step=update_steps)
            ma = np.convolve(u, np.ones(10) / 10, mode="valid")
            ms = s[9:]
            ok = ma <= args.alpha
            tau = next((ms[i] for i in range(len(ok)) if ok[i] and ok[i:].mean() >= 0.9), None)
            print(f"  {env_id:12s} {os.path.basename(run)[:5]}: hội tụ an toàn ở bước {tau}")
            if tau is not None:
                taus.append(int(tau))
        if not taus:
            print(f"{env_id}: không đọc được W&B của T1 để tính tau")
            continue
        tau = float(np.median(taus))
        print(f"{env_id}: tau = trung vị {len(taus)} seed = {tau:.0f} bước;  eta = eps_max / (alpha tau) = "
              f"{args.eps_max} / ({args.alpha} × {tau:.0f}) = {args.eps_max / (args.alpha * tau):.4g}")


def eps_fixed(args) -> None:
    import jax
    import jax.random as jr
    import numpy as np

    from eval_cm import load_run_config
    from methods.cm_dgppo.build import build_algo, build_env
    from methods.cm_dgppo.deploy import OnlineConfig, make_deploy_fn

    for env_id in args.envs:
        cs = []
        for run in sorted(glob.glob(f"{args.t1_dir}/{env_id}/dgppo/seed*")):
            cfg = load_run_config(run)
            env = build_env(cfg, sigma_w=0.0, sigma_v=0.0)
            algo = build_algo(cfg, env, seed=0, steps=1, batch_size=1, policy_only=True)
            with open(f"{run}/models/{args.step}/actor.pkl", "rb") as f:
                algo.policy_train_state = algo.policy_train_state.replace(params=pickle.load(f))
            fn = make_deploy_fn(env, algo, 1, OnlineConfig())
            out = jax.device_get(fn(algo.params, jr.split(jr.PRNGKey(args.eval_seed), args.episodes), (0.0, 0.0)))
            raw = np.asarray(out["raw"])[:, 0]                 # (episode, T, UAV, 2): h thô
            cs.append((-raw.max(axis=(1, 3))).ravel())          # khoảng hở nhỏ nhất mỗi (UAV, episode)
        c = np.sort(np.concatenate(cs))
        n = c.size
        k = math.floor(args.alpha * (n + 1))
        print(f"{env_id}: eval-seed {args.eval_seed}, n = {n} (từ {len(cs)} seed), k = floor({args.alpha}·{n + 1}) = {k}, "
              f"eps_fixed = c_{k} = {c[k - 1]:.6f}   (va chạm DGPPO {np.mean(c < 0):.4f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--eps-max", type=float, default=0.05)
    ap.add_argument("--envs", nargs="+", default=["LidarSpread", "LidarLine"])
    ap.add_argument("--t1-dir", default="/data/ducbm3/dgppo_runs/n3")
    ap.add_argument("--step", type=int, default=200000)
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--eval-seed", type=int, default=20000, help="tập VALIDATION (test dùng 10000)")
    ap.add_argument("--tau-only", action="store_true", help="chỉ tính tau và eta (không cần GPU)")
    args = ap.parse_args()

    tau_and_eta(args)
    if not args.tau_only:
        eps_fixed(args)


if __name__ == "__main__":
    main()
