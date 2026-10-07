#!/usr/bin/env python3
"""T2 — đánh giá độ bền vững 1 checkpoint dưới dịch chuyển (N, số vật cản) và nhiễu (σ_w, σ_v).

    # lưới σ tự chọn (tích Descartes sigma_w × sigma_v)
    python scripts/t2/eval_robust.py --path <ckpt_dir> --sigma-w 0 0.01 --sigma-v 0 \
        --epi 256 --csv /data/.../t2_noise.csv
    # lưới σ dùng chung của nhóm (quét từng yếu tố)
    python scripts/t2/eval_robust.py --path <ckpt_dir> --grid configs/t2/sigma_grid.yaml --csv ...
    # dịch chuyển số agent / số vật cản (σ = 0)
    python scripts/t2/eval_robust.py --path <ckpt_dir> -n 5 --obs 3 --csv ...
    # đọc khối noise/eval từ config theo lược đồ chung
    python scripts/t2/eval_robust.py --config configs/example.yaml --csv ...

Một process = một cấu hình env (ckpt, N_test, obs_test), vì make_env của DGPPO sửa TẠI CHỖ
dict PARAMS của class env. Trong process lặp trên danh sách cặp (σ_w, σ_v); σ là scalar JAX
nên không biên dịch lại.

Khóa episode sinh y như test.py: jr.split(PRNGKey(test_seed), 1000)[:epi], rồi tách
key_x0 -> 32 episode đầu trùng với 32 episode T1 đã báo. Với σ = 0 và --batch 1 phải khớp
test.py (batch > 1 dùng vmap, có thể lệch số thực rất nhỏ).
Không dùng `test.py --stochastic` (lỗi upstream: actor 4 tham số bị gọi với 3) — chế độ
stochastic ở đây tự bọc algo.step.

Ghi CSV mỗi episode một dòng (analysis/robust_csv.py) + in dòng tổng hợp cùng format
test.py (parse được bằng scripts/t1/run_baseline.py::parse_test_output).

Mỗi lần chạy tạo một thư mục hồ sơ trong --run-root (mặc định <thư mục của --csv>/runs) chứa
ĐỦ config + kết quả: run.yaml, ckpt_config.yaml, inputs/, code/, episodes.csv, summary.md,
summary.csv, console.log — xem analysis/run_record.py.

Env dựng giống train.py: env_id, num_agents, num_obs, n_rays, full_observation lấy từ
config.yaml của checkpoint (chỉ -n / --obs ghi đè khi thử dịch chuyển); max_step mặc định
128 như lúc train và test.py.

Yêu cầu: venv có `pip install -e third_party/dgppo`; shim JAX 0.6 (scripts/compat) — script
tự nạp shim, nhưng vẫn nên `export PYTHONPATH=<repo>/scripts/compat:$PYTHONPATH`.
"""
from __future__ import annotations

import argparse
import itertools
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
COMPAT = REPO / "scripts" / "compat"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="thư mục checkpoint (có config.yaml, models/)")
    ap.add_argument("--config", default=None, help="YAML theo configs/schema.md (khối noise, eval)")
    ap.add_argument("-n", "--num-agents", type=int, default=None, help="N lúc test (mặc định = N train)")
    ap.add_argument("--obs", type=int, default=None, help="số vật cản lúc test (mặc định = lúc train)")
    ap.add_argument("--sigma-w", type=float, nargs="+", default=None)
    ap.add_argument("--sigma-v", type=float, nargs="+", default=None)
    ap.add_argument("--grid", default=None, help="YAML lưới σ (vd configs/t2/sigma_grid.yaml)")
    ap.add_argument("--epi", type=int, default=None, help="số episode (mặc định 256)")
    ap.add_argument("--seed", type=int, default=None, help="test seed (mặc định 1234, như test.py)")
    ap.add_argument("--stochastic", action="store_true", default=None)
    ap.add_argument("--step", type=int, default=None, help="step checkpoint (mặc định lớn nhất)")
    ap.add_argument("--batch", type=int, default=32, help="số episode chạy song song (vmap); 1 = như test.py")
    ap.add_argument("--csv", default=str(REPO / "results" / "t2" / "robust.csv"),
                    help="CSV gộp (nối thêm); mỗi run còn có episodes.csv riêng trong thư mục hồ sơ")
    ap.add_argument("--run-root", default=None, help="nơi tạo thư mục hồ sơ run (mặc định <dir csv>/runs)")
    ap.add_argument("--gpu", default=None, help="đặt CUDA_VISIBLE_DEVICES")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


def resolve(args):
    """Gộp tham số CLI với file config (CLI ưu tiên). Trả dict cấu hình đã chốt."""
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    noise = cfg.get("noise", {}) or {}
    ev = cfg.get("eval", {}) or {}

    num_agents, obs = args.num_agents, args.obs
    shift_type, shift_level = noise.get("shift_type", "none"), noise.get("shift_level", 0)
    if num_agents is None and shift_type == "n_agents":
        num_agents = int(shift_level)
    if obs is None and shift_type == "n_obs":
        obs = int(shift_level)

    if args.grid:
        with open(args.grid) as f:
            grid = yaml.safe_load(f)
        sw_list, sv_list = grid["sigma_w"], grid["sigma_v"]
        mode = grid.get("mode", "one_at_a_time")
    else:
        sw_list = args.sigma_w if args.sigma_w is not None else [noise.get("sigma_w", 0.0)]
        sv_list = args.sigma_v if args.sigma_v is not None else [noise.get("sigma_v", 0.0)]
        mode = "product"
    if mode == "one_at_a_time":
        # quét từng yếu tố: (σ_w, 0) rồi (0, σ_v); điểm (0, 0) chỉ chạy một lần
        pairs = [(float(w), 0.0) for w in sw_list]
        pairs += [(0.0, float(v)) for v in sv_list if (0.0, float(v)) not in pairs]
    elif mode == "product":
        pairs = [(float(w), float(v)) for w, v in itertools.product(sw_list, sv_list)]
    else:
        raise ValueError(f"mode lưới không hợp lệ: {mode}")

    path = args.path or ev.get("ckpt")
    if path is None:
        raise SystemExit("Cần --path hoặc eval.ckpt trong config")
    return {
        "path": path,
        "num_agents": num_agents,
        "obs": obs,
        "pairs": pairs,
        "epi": args.epi if args.epi is not None else int(ev.get("n_episodes", 256)),
        "test_seed": args.seed if args.seed is not None else int(ev.get("test_seed", 1234)),
        "stochastic": bool(args.stochastic) if args.stochastic is not None else bool(ev.get("stochastic", False)),
    }


def git_commit() -> str:
    out = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                         capture_output=True, text=True)
    return out.stdout.strip()


def shift_of(n_train, n_test, obs_train, obs_test):
    """(shift_type, shift_level) suy từ cấu hình test so với lúc train."""
    if n_test != n_train and obs_test != obs_train:
        return "n_agents+n_obs", f"{n_test}/{obs_test}"
    if n_test != n_train:
        return "n_agents", n_test
    if obs_test != obs_train:
        return "n_obs", obs_test
    return "none", 0


def main() -> None:
    args = parse_args()
    run = resolve(args)

    # biến môi trường phải đặt TRƯỚC khi import jax
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = os.environ["JAX_PLATFORM_NAME"] = "cpu"
    # KHÔNG bật --xla_gpu_deterministic_ops: với JAX 0.6.2 cờ này làm rollout policy sai hẳn
    # (DGPPO LidarSpread seed0 σ=0: safe 47.9% thay vì 100%; env đơn lẻ thì vẫn đúng). Hệ quả:
    # chạy lại không ra y hệt. XLA (JAX 0.6.2) không tất định trên cả GPU (scatter-add atomic;
    # ~0.4% episode đổi kết quả an toàn ở σ=0) lẫn CPU (lệch ~1e-8..1e-6, kể cả khi ép 1 luồng)
    # -> số liệu chỉ tái lập ở mức thống kê.
    if "xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", ""):
        raise SystemExit("XLA_FLAGS có --xla_gpu_deterministic_ops=true: cờ này làm sai rollout (xem chú thích)")
    if (COMPAT / "sitecustomize.py").exists():
        runpy.run_path(str(COMPAT / "sitecustomize.py"))   # shim JAX 0.6
    sys.path.insert(0, str(REPO))

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from dgppo.algo import make_algo
    from dgppo.env import make_env

    from analysis.robust_csv import append_rows
    from analysis.robust_metrics import episode_metrics, wilson_ci
    from analysis.robust_summary import load, summarize, to_markdown, write_summary_csv
    from analysis.run_record import RunRecord, sha256
    from envs.noise_wrapper import noisy_test_rollout

    path = run["path"]
    with open(os.path.join(path, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    n_test = config.num_agents if run["num_agents"] is None else run["num_agents"]
    obs_test = config.obs if run["obs"] is None else run["obs"]
    mode = "stoch" if run["stochastic"] else "det"

    run_root = args.run_root or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "runs")
    rec = RunRecord(run_root, f"{config.algo}_{config.env}_seed{config.seed}_N{n_test}_obs{obs_test}_{mode}", REPO)
    try:
        _evaluate(args, run, rec, config, n_test, obs_test, mode, jax, jnp, jr, make_algo, make_env,
                  append_rows, episode_metrics, wilson_ci, load, summarize, to_markdown,
                  write_summary_csv, sha256, noisy_test_rollout)
    except BaseException as e:
        rec.set("error", repr(e))
        rec.finish("failed")
        raise
    rec.finish("done")
    print(f"[T2] hồ sơ run: {rec.dir}", flush=True)


def _evaluate(args, run, rec, config, n_test, obs_test, mode, jax, jnp, jr, make_algo, make_env,
              append_rows, episode_metrics, wilson_ci, load, summarize, to_markdown,
              write_summary_csv, sha256, noisy_test_rollout):
    path = run["path"]
    shutil.copy2(os.path.join(path, "config.yaml"), rec.dir / "ckpt_config.yaml")
    rec.copy_input(args.config, "config")
    rec.copy_input(args.grid, "grid")

    # giống train.py: n_rays / full_observation lấy từ config checkpoint
    env_kwargs = dict(env_id=config.env, num_agents=n_test, num_obs=obs_test,
                      n_rays=getattr(config, "n_rays", None),
                      full_observation=getattr(config, "full_observation", False))
    env = make_env(**env_kwargs)

    model_path = os.path.join(path, "models")
    step = args.step if args.step is not None else max(
        int(m) for m in os.listdir(model_path) if m.isdigit())
    # giống hệt test.py (DGPPO @51b3b11)
    algo = make_algo(
        algo=config.algo, env=env, node_dim=env.node_dim, edge_dim=env.edge_dim,
        state_dim=env.state_dim, action_dim=env.action_dim, n_agents=env.num_agents,
        cost_weight=config.cost_weight, actor_gnn_layers=config.actor_gnn_layers,
        Vl_gnn_layers=config.Vl_gnn_layers,
        Vh_gnn_layers=getattr(config, "Vh_gnn_layers", 1),
        lr_actor=config.lr_actor, lr_Vl=config.lr_Vl, max_grad_norm=2.0,
        seed=config.seed, use_rnn=config.use_rnn, rnn_layers=config.rnn_layers,
        use_lstm=config.use_lstm)
    algo.load(model_path, step)
    model_dir = Path(model_path) / str(step)

    rec.set("settings", {
        **{k: v for k, v in run.items() if k != "pairs"},
        "pairs": [list(p) for p in run["pairs"]],
        "batch": max(1, args.batch), "csv": os.path.abspath(args.csv), "policy_mode": mode,
        "episode_keys": "jr.split(jr.PRNGKey(test_seed), 1000)[:epi]; key_x0 = jr.split(key, 2)[0] (như test.py)",
    })
    rec.set("checkpoint", {
        "path": os.path.abspath(path), "step": step,
        "files": {f.name: sha256(f) for f in sorted(model_dir.glob("*.pkl"))},
        "config": vars(config),
    })
    rec.set("env", {
        "make_env_kwargs": env_kwargs,
        "class": type(env).__name__, "num_agents": env.num_agents,
        "max_episode_steps": env.max_episode_steps, "dt": env.dt, "area_size": env.area_size,
        "params": dict(env.params),
        "differs_from_train": {
            "num_agents": [config.num_agents, n_test] if n_test != config.num_agents else None,
            "n_obs": [config.obs, obs_test] if obs_test != config.obs else None,
        },
    })
    rec.set("noise_definition", {
        "sigma_w": "std nhiễu cộng vào vận tốc agent sau mỗi bước, rồi clip_state (|v| <= 0.5)",
        "sigma_v": "std nhiễu cộng vào vị trí quan sát của agent + điểm hit LiDAR; vận tốc quan sát sạch",
        "violation": "cost (đã dịch biên ±0.5) >= 0 trên graph THẬT, t = 0..T-1 (như test.py)",
    })
    rec.set("device", {"devices": [str(d) for d in jax.devices()],
                       "device_kind": jax.devices()[0].device_kind})
    episodes_csv = str(rec.dir / "episodes.csv")

    stochastic = run["stochastic"]
    if stochastic:
        def actor(graph, rnn_state, key):
            action, _, rnn_state = algo.step(graph, rnn_state, key)
            return action, rnn_state
    else:
        actor = algo.act

    def run_one(key, sigma_w, sigma_v, use_w, use_v):
        key_x0, _ = jr.split(key, 2)   # như test.py
        # trục σ = 0 truyền 0.0 TĨNH -> bỏ hẳn nhánh nhiễu (σ = 0 giống hệt test_rollout)
        rollout = noisy_test_rollout(env, actor, algo.init_rnn_state, key_x0,
                                     sigma_w if use_w else 0.0, sigma_v if use_v else 0.0,
                                     stochastic=stochastic)
        return episode_metrics(env, rollout)

    batch = max(1, args.batch)
    # biên dịch tối đa 4 bản (có/không σ_w × có/không σ_v); trong mỗi bản σ là traced
    compiled = {}

    def get_run_fn(use_w, use_v):
        if (use_w, use_v) not in compiled:
            def fn(key, sigma_w, sigma_v):   # closure: use_w/use_v là bool Python (tĩnh)
                return run_one(key, sigma_w, sigma_v, use_w, use_v)
            compiled[use_w, use_v] = jax.jit(fn if batch == 1 else jax.vmap(fn, in_axes=(0, None, None)))
        return compiled[use_w, use_v]

    epi = run["epi"]
    test_keys = jr.split(jr.PRNGKey(run["test_seed"]), 1_000)[:epi]
    shift_type, shift_level = shift_of(config.num_agents, n_test, config.obs, obs_test)
    commit = git_commit()
    tag = f"{config.algo}/{config.env}/seed{config.seed} N={config.num_agents}->{n_test} obs={config.obs}->{obs_test} {mode}"

    for sigma_w, sigma_v in run["pairs"]:
        sw, sv = jnp.asarray(sigma_w, jnp.float32), jnp.asarray(sigma_v, jnp.float32)
        run_fn = get_run_fn(sigma_w > 0, sigma_v > 0)
        chunks = []
        for i in range(0, epi, batch):
            keys = test_keys[i:i + batch]
            if batch == 1:
                out = jax.device_get(run_fn(keys[0], sw, sv))
                chunks.append({k: np.asarray(v)[None] for k, v in out.items()})
            else:
                n_real = keys.shape[0]
                if n_real < batch:   # đệm cho đủ batch để không biên dịch lại
                    keys = jnp.concatenate([keys, jnp.repeat(keys[-1:], batch - n_real, axis=0)])
                out = jax.device_get(run_fn(keys, sw, sv))
                chunks.append({k: np.asarray(v)[:n_real] for k, v in out.items()})
        m = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}

        rows = []
        for e in range(epi):
            rows.append({
                "method": config.algo, "env": config.env,
                "N_train": config.num_agents, "N_test": n_test,
                "obs_train": config.obs, "obs_test": obs_test,
                "seed": config.seed, "test_seed": run["test_seed"],
                "sigma_w": sigma_w, "sigma_v": sigma_v,
                "shift_type": shift_type, "shift_level": shift_level,
                "episode": e, "policy_mode": mode,
                "safe_agent_frac": round(float(m["safe_agent_frac"][e]), 6),
                "safe_traj": int(m["safe_traj"][e]),
                "viol_freq": round(float(m["viol_freq"][e]), 6),
                "min_dist": round(float(m["min_dist"][e]), 6),
                "max_h": round(float(m["max_h"][e]), 6),
                "t_first_viol": int(m["t_first_viol"][e]),
                "n_viol_agent": int(m["n_viol_agent"][e]),
                "n_viol_obs": int(m["n_viol_obs"][e]),
                "task_cost": round(float(m["task_cost"][e]), 6),
                "reach_rate": round(float(m["reach_rate"][e]), 6),
                "mean_dist2goal": round(float(m["mean_dist2goal"][e]), 6),
                "git_commit": commit,
            })
        append_rows(args.csv, rows)
        append_rows(episodes_csv, rows)

        agent_safe = 1 - m["agent_unsafe"]                     # (epi, n)
        lo, hi = wilson_ci(float(agent_safe.sum()), agent_safe.size)
        tlo, thi = wilson_ci(float(m["safe_traj"].sum()), epi)
        print(f"[T2] {tag} sigma_w={sigma_w} sigma_v={sigma_v} epi={epi}", flush=True)
        print(f"reward: {m['reward'].mean():.3f}, min/max reward: {m['reward'].min():.3f}/{m['reward'].max():.3f}, "
              f"cost: {m['max_h'].mean():.3f}, min/max cost: {m['max_h'].min():.3f}/{m['max_h'].max():.3f}, "
              f"safe_rate: {agent_safe.mean() * 100:.3f}%", flush=True)
        print(f"     safe_rate Wilson95 [{lo * 100:.1f}, {hi * 100:.1f}]%, "
              f"safe_traj: {m['safe_traj'].mean() * 100:.1f}% [{tlo * 100:.1f}, {thi * 100:.1f}]%, "
              f"reach: {m['reach_rate'].mean():.3f}, viol agent/obs (agent): "
              f"{int(m['n_viol_agent'].sum())}/{int(m['n_viol_obs'].sum())}", flush=True)
    table = summarize(load([episodes_csv]))
    md = to_markdown(table)
    (rec.dir / "summary.md").write_text(md + "\n")
    write_summary_csv(table, str(rec.dir / "summary.csv"))
    rec.set("results", {
        "episodes_csv": episodes_csv, "summary_md": str(rec.dir / "summary.md"),
        "summary": [{"sigma_w": t["key"][5], "sigma_v": t["key"][6], "n_epi": t["n_epi"],
                     "safe_agent": round(t["safe_agent"], 6),
                     "safe_agent_wilson95": [round(x, 6) for x in t["safe_agent_ci"]],
                     "delta_pp": None if t["delta_pp"] is None else round(t["delta_pp"], 4),
                     "safe_traj": round(t["safe_traj"], 6),
                     "viol_agent": t["viol_agent"], "viol_obs": t["viol_obs"],
                     "dist2goal": round(t["dist2goal"], 6), "task_cost": round(t["task_cost"], 6),
                     "max_h": round(t["max_h"], 6)}
                    for t in table],
    })
    print(f"[T2] CSV: {args.csv}", flush=True)


if __name__ == "__main__":
    main()
