"""Tổng hợp CSV từng episode (analysis/robust_csv.py) thành bảng theo cấu hình × σ (mạch T2).

Nhóm theo (method, env, N_test, obs_test, policy_mode, sigma_w, sigma_v), gộp mọi seed.
Cột:
    safe_agent   tỉ lệ (episode, agent) không vi phạm = safe_rate của test.py, kèm Wilson 95%
    Δ vs σ=0     chênh safe_agent so với điểm σ_w = σ_v = 0 cùng nhóm (điểm phần trăm)
    safe_traj    tỉ lệ episode không agent nào vi phạm, kèm Wilson 95%
    viol a/o     tổng số agent từng va agent khác / va vật cản
    dist2goal    khoảng cách goal–agent gần nhất ở bước cuối (trung bình)
Chỉ đọc csv — không cần JAX. CLI: scripts/t2/summarize.py.
"""
from __future__ import annotations

import csv
from collections import defaultdict

from analysis.robust_metrics import wilson_ci

GROUP = ("method", "env", "N_test", "obs_test", "policy_mode")
SUMMARY_COLUMNS = [
    "method", "env", "N_test", "obs_test", "policy_mode", "sigma_w", "sigma_v", "n_epi", "seeds",
    "safe_agent", "safe_agent_lo", "safe_agent_hi", "delta_pp", "safe_traj", "safe_traj_lo",
    "safe_traj_hi", "viol_agent", "viol_obs", "dist2goal", "task_cost",
]


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in GROUP) + (float(r["sigma_w"]), float(r["sigma_v"]))
        groups[key].append(r)

    table = []
    for key in sorted(groups):
        rs = groups[key]
        n_epi = len(rs)
        n_agent = int(rs[0]["N_test"])
        safe_agents = sum(float(r["safe_agent_frac"]) * n_agent for r in rs)
        trials = n_epi * n_agent
        safe_traj = sum(int(r["safe_traj"]) for r in rs)
        table.append({
            "key": key,
            "n_epi": n_epi,
            "seeds": sorted({r["seed"] for r in rs}),
            "safe_agent": safe_agents / trials,
            "safe_agent_ci": wilson_ci(safe_agents, trials),
            "safe_traj": safe_traj / n_epi,
            "safe_traj_ci": wilson_ci(safe_traj, n_epi),
            "viol_agent": sum(int(r["n_viol_agent"]) for r in rs),
            "viol_obs": sum(int(r["n_viol_obs"]) for r in rs),
            "dist2goal": sum(float(r["mean_dist2goal"]) for r in rs) / n_epi,
            "task_cost": sum(float(r["task_cost"]) for r in rs) / n_epi,
        })

    base = {t["key"][:len(GROUP)]: t["safe_agent"] for t in table if t["key"][-2:] == (0.0, 0.0)}
    for t in table:
        b = base.get(t["key"][:len(GROUP)])
        t["delta_pp"] = None if b is None else (t["safe_agent"] - b) * 100
    return table


def to_markdown(table) -> str:
    lines = [
        "| method | env | N | obs | mode | σ_w | σ_v | epi | safe_agent % [Wilson95] | Δ vs σ=0 (pp) "
        "| safe_traj % [Wilson95] | viol a/o | dist2goal | task_cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in table:
        method, env, n, obs, mode, sw, sv = t["key"]
        lo, hi = t["safe_agent_ci"]
        tlo, thi = t["safe_traj_ci"]
        delta = "—" if t["delta_pp"] is None else f"{t['delta_pp']:+.1f}"
        lines.append(
            f"| {method} | {env} | {n} | {obs} | {mode} | {sw:g} | {sv:g} | {t['n_epi']} "
            f"| {t['safe_agent'] * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}] | {delta} "
            f"| {t['safe_traj'] * 100:.1f} [{tlo * 100:.1f}, {thi * 100:.1f}] "
            f"| {t['viol_agent']}/{t['viol_obs']} | {t['dist2goal']:.3f} | {t['task_cost']:.3f} |")
    return "\n".join(lines)


def write_summary_csv(table, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for t in table:
            method, env, n, obs, mode, sw, sv = t["key"]
            w.writerow({
                "method": method, "env": env, "N_test": n, "obs_test": obs, "policy_mode": mode,
                "sigma_w": sw, "sigma_v": sv, "n_epi": t["n_epi"], "seeds": " ".join(t["seeds"]),
                "safe_agent": round(t["safe_agent"], 6),
                "safe_agent_lo": round(t["safe_agent_ci"][0], 6), "safe_agent_hi": round(t["safe_agent_ci"][1], 6),
                "delta_pp": "" if t["delta_pp"] is None else round(t["delta_pp"], 4),
                "safe_traj": round(t["safe_traj"], 6),
                "safe_traj_lo": round(t["safe_traj_ci"][0], 6), "safe_traj_hi": round(t["safe_traj_ci"][1], 6),
                "viol_agent": t["viol_agent"], "viol_obs": t["viol_obs"],
                "dist2goal": round(t["dist2goal"], 6), "task_cost": round(t["task_cost"], 6),
            })
