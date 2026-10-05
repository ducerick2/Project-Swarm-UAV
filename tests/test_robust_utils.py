"""Test phần không cần JAX của mạch T2: Wilson CI, CSV, gộp tham số eval_robust.py.

    pytest tests/test_robust_utils.py -q
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.robust_csv import COLUMNS, append_rows
from analysis.robust_metrics import wilson_ci


def _load_eval_robust():
    spec = importlib.util.spec_from_file_location("eval_robust", REPO / "scripts" / "t2" / "eval_robust.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**kw):
    base = dict(path="ckpt", config=None, num_agents=None, obs=None, sigma_w=None, sigma_v=None,
                grid=None, epi=None, seed=None, stochastic=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(95, 96)
    assert lo < 95 / 96 < hi <= 1.0
    assert lo == pytest.approx(0.9433, abs=1e-3)
    lo, hi = wilson_ci(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0 < hi < 0.35
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_append_rows_header_once_and_missing(tmp_path):
    path = tmp_path / "sub" / "r.csv"
    row = {c: 0 for c in COLUMNS}
    append_rows(str(path), [row, row])
    append_rows(str(path), [row])
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == COLUMNS and len(rows) == 4
    bad = dict(row)
    bad.pop("sigma_v")
    with pytest.raises(ValueError):
        append_rows(str(path), [bad])


def test_resolve_grid_one_at_a_time(tmp_path):
    er = _load_eval_robust()
    grid = tmp_path / "g.yaml"
    grid.write_text("mode: one_at_a_time\nsigma_w: [0.0, 0.01]\nsigma_v: [0.0, 0.05]\n")
    run = er.resolve(_args(grid=str(grid)))
    assert run["pairs"] == [(0.0, 0.0), (0.01, 0.0), (0.0, 0.05)]
    assert run["epi"] == 256 and run["test_seed"] == 1234 and run["stochastic"] is False


def test_resolve_product_and_config(tmp_path):
    er = _load_eval_robust()
    run = er.resolve(_args(sigma_w=[0.0, 0.01], sigma_v=[0.0, 0.05]))
    assert len(run["pairs"]) == 4

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "noise: {sigma_w: 0.02, sigma_v: 0.0, shift_type: n_agents, shift_level: 5}\n"
        "eval: {ckpt: some/ckpt, n_episodes: 64, test_seed: 7, stochastic: true}\n")
    run = er.resolve(_args(path=None, config=str(cfg)))
    assert run["path"] == "some/ckpt" and run["num_agents"] == 5 and run["obs"] is None
    assert run["pairs"] == [(0.02, 0.0)]
    assert (run["epi"], run["test_seed"], run["stochastic"]) == (64, 7, True)
    # CLI ưu tiên hơn config
    run = er.resolve(_args(path="cli", config=str(cfg), num_agents=7, epi=8))
    assert (run["path"], run["num_agents"], run["epi"]) == ("cli", 7, 8)


def test_shift_of():
    er = _load_eval_robust()
    assert er.shift_of(3, 3, 3, 3) == ("none", 0)
    assert er.shift_of(3, 5, 3, 3) == ("n_agents", 5)
    assert er.shift_of(3, 3, 3, 8) == ("n_obs", 8)
    assert er.shift_of(3, 5, 3, 8) == ("n_agents+n_obs", "5/8")
