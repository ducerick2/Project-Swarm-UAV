#!/usr/bin/env python3
"""T2 — điền các bảng số liệu vào docs/T2_RESULTS.md từ các lượt quét có hồ sơ.

    python scripts/t2/fill_results_tables.py --runs /home/mantd/DGPPO/t2_runs/sweeps [--doc docs/T2_RESULTS.md]

Trong tài liệu, mỗi bảng nằm giữa hai dấu
    <!-- BEGIN:<tên> -->  ...  <!-- END:<tên> -->
Script thay nội dung giữa hai dấu bằng bảng sinh từ episodes.csv, nên chạy lại bao nhiêu lần
cũng được và số liệu luôn khớp dữ liệu gốc. Lượt quét được tìm theo hậu tố tên thư mục
(`*_check0`, `*_calib`, `*_calib_fine`, `*_noise_h1`, `*_shift`; lấy thư mục mới nhất).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analysis.robust_metrics import wilson_ci  # noqa: E402
from analysis.robust_summary import load, summarize  # noqa: E402

T1_PILOT = {  # scripts/t1/pilot_results.csv (nhánh t1-reproduce), DGPPO seed0, 32 episode
    "LidarSpread": {"safe": 1.0, "reward": -0.885},
    "LidarLine": {"safe": 0.98958, "reward": -0.28},
}
R = 0.05  # car_radius


def sweep(root: Path, name: str) -> Path:
    cands = sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(rf"\d{{8}}-\d{{6}}_{name}", p.name))
    if not cands:
        raise SystemExit(f"không thấy lượt quét *_{name} trong {root}")
    return cands[-1]


def fmt_sigma(sw: float, sv: float) -> tuple[str, str]:
    v = f"{sv:g}" + (f" ({sv / R:g}r)" if sv else "")
    return f"{sw:g}", v


def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def ci(c) -> str:
    return f"[{c[0] * 100:.1f}, {c[1] * 100:.1f}]"


# ---------------------------------------------------------------- bảng
def tbl_summary(rows, with_shift: bool = False) -> str:
    table = summarize(rows)
    head = ["env"] + (["N", "obs", "mode"] if with_shift else []) + \
           ["σ_w", "σ_v", "epi", "safe_agent % [Wilson95]", "Δ vs σ=0 (pp)", "safe_traj % [Wilson95]",
            "viol agent/obs", "dist2goal", "task_cost"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t in table:
        method, env, n, obs, mode, sw, sv = t["key"]
        s_w, s_v = fmt_sigma(sw, sv)
        delta = "—" if t["delta_pp"] is None else f"{t['delta_pp']:+.1f}"
        cells = [env] + ([n, obs, mode] if with_shift else []) + [
            s_w, s_v, str(t["n_epi"]), f"{pct(t['safe_agent'])} {ci(t['safe_agent_ci'])}", delta,
            f"{pct(t['safe_traj'])} {ci(t['safe_traj_ci'])}", f"{t['viol_agent']}/{t['viol_obs']}",
            f"{t['dist2goal']:.3f}", f"{t['task_cost']:.3f}"]
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(out)


def _pivot(rows, keyf):
    """{key: {seed: (safe_agents, trials, safe_traj, n_epi)}} + gộp."""
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0, 0]))
    for r in rows:
        k = keyf(r)
        n = int(r["N_test"])
        for s in (r["seed"], "all"):
            a = acc[k][s]
            a[0] += float(r["safe_agent_frac"]) * n
            a[1] += n
            a[2] += int(r["safe_traj"])
            a[3] += 1
    return acc


def tbl_per_seed_noise(rows) -> str:
    acc = _pivot(rows, lambda r: (r["env"], float(r["sigma_w"]), float(r["sigma_v"])))
    seeds = sorted({r["seed"] for r in rows})
    head = ["env", "σ_w", "σ_v"] + [f"seed {s}" for s in seeds] + ["gộp [Wilson95]", "safe_traj gộp"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (env, sw, sv) in sorted(acc):
        a = acc[(env, sw, sv)]
        s_w, s_v = fmt_sigma(sw, sv)
        cells = [env, s_w, s_v] + [pct(a[s][0] / a[s][1]) for s in seeds]
        al = a["all"]
        cells += [f"{pct(al[0] / al[1])} {ci(wilson_ci(al[0], al[1]))}", pct(al[2] / al[3])]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def tbl_per_seed_shift(rows) -> str:
    acc = _pivot(rows, lambda r: (r["env"], r["policy_mode"], int(r["N_test"]), int(r["obs_test"])))
    seeds = sorted({r["seed"] for r in rows})
    head = ["env", "mode", "N", "obs"] + [f"seed {s}" for s in seeds] + ["gộp [Wilson95]", "safe_traj gộp"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k in sorted(acc):
        env, mode, n, obs = k
        a = acc[k]
        al = a["all"]
        cells = [env, mode, str(n), str(obs)] + [pct(a[s][0] / a[s][1]) for s in seeds]
        cells += [f"{pct(al[0] / al[1])} {ci(wilson_ci(al[0], al[1]))}", pct(al[2] / al[3])]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def tbl_check0(rows) -> str:
    out = ["| env | epi | safe_agent T2 | safe_agent T1 | reward T2 | reward T1 |", "|---|---|---|---|---|---|"]
    for t in summarize(rows):
        env = t["key"][1]
        ref = T1_PILOT[env]
        out.append(f"| {env} | {t['n_epi']} | {t['safe_agent'] * 100:.3f}% | {ref['safe'] * 100:.3f}% "
                   f"| {-t['task_cost']:.3f} | {ref['reward']:.3f} |")
    return "\n".join(out)


def tbl_viol_breakdown(rows) -> str:
    """Tỉ lệ va vật cản trong tổng vi phạm + thời điểm vi phạm đầu tiên (trung vị)."""
    acc = defaultdict(lambda: [0, 0, []])
    for r in rows:
        k = (r["env"], float(r["sigma_w"]), float(r["sigma_v"]))
        acc[k][0] += int(r["n_viol_agent"])
        acc[k][1] += int(r["n_viol_obs"])
        if int(r["t_first_viol"]) >= 0:
            acc[k][2].append(int(r["t_first_viol"]))
    head = ["env", "σ_w", "σ_v", "viol agent", "viol obs", "% vật cản", "episode có vi phạm", "t vi phạm đầu (trung vị)"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k in sorted(acc):
        a, o, ts = acc[k]
        s_w, s_v = fmt_sigma(k[1], k[2])
        share = f"{o / (a + o) * 100:.0f}" if a + o else "—"
        med = f"{sorted(ts)[len(ts) // 2]}" if ts else "—"
        out.append(f"| {k[0]} | {s_w} | {s_v} | {a} | {o} | {share} | {len(ts)} | {med} |")
    return "\n".join(out)


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def build(root: Path) -> dict[str, str]:
    d = {n: sweep(root, n) for n in ("check0", "calib", "calib_fine", "noise_h1", "shift")}
    rows = {n: load([str(p / "episodes.csv")]) for n, p in d.items()}
    # dòng "Nguồn:" trong hồ sơ ghi đường dẫn lúc chạy -> hiển thị đường dẫn hiện tại (tương đối repo)
    src = f"Nguồn: `{_rel(d['noise_h1'] / 'episodes.csv')}`"
    h1 = re.sub(r"^Nguồn: `[^`]*`", src, (d["noise_h1"] / "h1_test.md").read_text().strip())
    h1_12 = re.sub(r"^Nguồn: `[^`]*`", src, (d["noise_h1"] / "h1_test_seed12.md").read_text().strip())
    tables = {
        "sources": "\n".join(f"- `{n}`: `{_rel(p)}`" for n, p in d.items()),
        "check0": tbl_check0(rows["check0"]),
        "calib": tbl_summary(rows["calib"]),
        "calib_fine": tbl_summary(rows["calib_fine"]),
        "noise_pooled": tbl_summary(rows["noise_h1"]),
        "noise_per_seed": tbl_per_seed_noise(rows["noise_h1"]),
        "noise_viol": tbl_viol_breakdown(rows["noise_h1"]),
        "h1_all": h1,
        "h1_seed12": h1_12,
        "shift_pooled": tbl_summary(rows["shift"], with_shift=True),
        "shift_per_seed": tbl_per_seed_shift(rows["shift"]),
    }
    return tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="thư mục chứa các lượt quét ($OUTDIR/sweeps)")
    ap.add_argument("--doc", default=str(REPO / "docs" / "T2_RESULTS.md"))
    args = ap.parse_args()
    tables = build(Path(args.runs))
    doc = Path(args.doc).read_text()
    for name, body in tables.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(<!-- END:{name} -->)", re.S)
        if not pat.search(doc):
            print(f"!! tài liệu thiếu dấu BEGIN/END cho bảng '{name}'")
            continue
        doc = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), doc)
    Path(args.doc).write_text(doc)
    print(f"đã điền {len(tables)} bảng vào {args.doc}")


if __name__ == "__main__":
    main()
