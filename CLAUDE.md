# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project and this checkout

Safe multi-agent RL for UAV swarms under uncertainty (Safe MARL / CM-DGPPO), built on **DGPPO** (ICLR 2025, JAX, https://github.com/MIT-REALM/dgppo). Five-person team, one workstream branch each. The README, docstrings, comments and commit messages are in Vietnamese. Keep new ones in Vietnamese.

This checkout is the **`t2-robustness`** branch (member B). It evaluates whether A's clean-trained checkpoints stay safe under test-time shift: agent count, obstacle count, dynamics noise σ_w and sensor noise σ_v. It tests hypothesis **H1**: the safety rate of DGPPO drops clearly as σ grows. Scope is DGPPO only; InforMARL and InforMARL-Lag belong to T1. `docs/T2_HANDOFF.md` describes the work: what T1 did, its results and data caveats, then T2's goals, method and a task list with status (done vs remaining). `docs/T2_RESULTS.md` is the formal report.

## Setup and commands

- `third_party/dgppo` is a submodule pinned at `51b3b11`. Run `git submodule update --init --recursive` after cloning. Never edit code inside it; compatibility fixes go in `scripts/compat/sitecustomize.py`.
- `scripts/compat/sitecustomize.py` was copied from `t1-reproduce`. It is a JAX 0.6 shim that restores the `jax.tree_*` aliases. The venv recipe is in `scripts/t1/README.md` on `t1-reproduce`: `jax[cuda12]` ≥0.6, DGPPO requirements minus the jax pin, plus `jax_dataclasses imageio-ffmpeg`, then `pip install -e third_party/dgppo`. The tests also need `pytest`.
- Checkpoints are **not** on this branch. They live on `t1-reproduce` under `results/checkpoints/<env>/<method>/seed<N>/`:
  - envs `LidarSpread` and `LidarLine`; methods `dgppo`, `informarl` and `informarl_lagr`; N=3, obs=3, seeds 0–2;
  - only `dgppo` and `informarl_lagr` have `Vh.pkl`;
  - `scripts/t2/_env.sh` defaults `CKPT_ROOT` to the sibling worktree `../Project-Swarm-UAV-t1-reproduce`.

```bash
export PYTHONPATH=$PWD/scripts/compat:$PYTHONPATH
pytest tests/ -q                                   # JAX tests run on CPU and need no checkpoint
pytest tests/test_noise_wrapper.py::test_sigma0_bit_identical_to_test_rollout -q
python scripts/t2/eval_robust.py --path <ckpt> --epi 32 --batch 1 --sigma-w 0 --sigma-v 0   # must match test.py
python scripts/t2/eval_robust.py --path <ckpt> --grid configs/t2/sigma_grid.yaml --csv <out.csv>
python scripts/t2/eval_robust.py --path <ckpt> -n 5 --obs 3 --stochastic --csv <out.csv>
bash scripts/t2/run_shift_grid.sh                  # N∈{3,5,7}, obs∈{3,5,8}, det+stoch
bash scripts/t2/run_noise_grid.sh                  # every ckpt × sigma_grid.yaml
```

Both sweep scripts take overrides through env vars defined in `scripts/t2/_env.sh` (`CKPT_ROOT OUTDIR GPU EPI ENVS METHODS SEEDS`). The user wants anything that runs to be launched in a tmux session, one window per job, with output teed to a log.

On this machine (RTX 3080, so `GPU=0`, not the `_env.sh` default of 1):
- the venv is `/home/mantd/DGPPO/dgppo_env`;
- run records (config + results) go in `/home/mantd/DGPPO/t2_runs/` (set `OUTDIR` to it; outside the repo, not committed, because they exceed 1000 files);
- `python scripts/t2/summarize.py <csv>...` turns per-episode CSVs into a table (Wilson CIs, Δ vs σ=0).

σ levels are calibrated in `configs/t2/sigma_grid.yaml` (`levels`). `docs/T2_calibration.md` has the pre-registered selection rule, the H1 criterion and the calibration results.

GPU caveats:
- **Runs are not bit-reproducible on either GPU or CPU** with JAX 0.6.2 here.
  - GPU: scatter-add in `type_states`/`segment_sum` uses atomics; about 0.4% of episodes flip their safety outcome between reruns.
  - CPU: runs drift by about 1e-8 to 1e-6, even single-threaded.
  - Rounding differences compound over 128 steps, so results reproduce statistically, not exactly. Tests compare σ=0 against `test_rollout` with tolerances.
- **Never set `--xla_gpu_deterministic_ops=true`.** With JAX 0.6.2 it silently breaks policy rollouts (47.9% vs 100% safe), and `eval_robust.py` refuses to run with it.

`scripts/train.py` and `scripts/eval.py` are still TODO skeletons. T2 evaluation goes through `scripts/t2/eval_robust.py`.

## Architecture that spans several files

- **DGPPO envs are pure functional JAX, not gym.**
  - `reset(key)`; `step(graph, action)` returns a 5-tuple.
  - Reward and cost are computed on the **pre-step** graph.
  - Dynamics are a double integrator (`x_dot = [v, 10·a]`, dt = 0.03), clipped to pos ∈ [0, 1.5] and |v| ≤ 0.5. T = 128 and r = 0.05.
  - The policy GNN reads only `graph.nodes`/`graph.edges`, while env cost and reward read `graph.states`/`graph.env_states`.
- **`envs/noise_wrapper.py`** relies on that split: `noisy_test_rollout` copies upstream `test_rollout` but keeps a **true graph** (fed to `env.step`, so costs are measured on it) and an **observed graph** (fed to the actor).
  - `observe()` adds σ_v to agent positions and LiDAR hit points, then rebuilds the graph with `env.get_graph`. Observed velocities stay clean. The noise is correlated: every agent sees the same noisy copy.
  - `perturb_dynamics()` adds σ_w to velocity after each step, clips, then recomputes LiDAR.
  - A σ given as a Python float equal to 0 drops its noise branch at trace time, so σ=0 runs the same program as `test_rollout`.
  - A traced σ goes through `lax.cond`. `eval_robust.py` compiles at most 4 variants (σ_w on/off × σ_v on/off) and sweeps σ inside each one without recompiling.
  - The actor gets the step key unchanged; the noise keys come from `jr.fold_in`.
  - `NoiseWrapper(env, sigma_w, sigma_v, ...)` keeps the Phase 0 signature and just wraps these functions.
  - Agent-count and obstacle-count shift happen at `make_env` time, not in the wrapper.
- **Metrics** (`analysis/robust_metrics.py`, matching upstream `test.py`):
  - an agent is unsafe if the margin-shifted cost is ≥ 0 at any t in 0..T-1;
  - `test.py`'s safe_rate is **per agent** (`safe_agent_frac`); `safe_traj` is per episode;
  - the "cost" that `test.py` prints is max h (`max_h`); task cost is −Σreward.
  - Obstacle cost still comes from the 8 nearest LiDAR rays of the *true* state, which approximates the real geometry. A geometric `true_violation` is not written yet.
  - `reach_rate` uses the env's strict `dist2goal = 0.01` and is about 0 even at σ=0, so use `mean_dist2goal` instead.
- **`scripts/t2/eval_robust.py`** runs one env config per process, because `make_env` mutates the env class's `PARAMS` dict in place. It loops over (σ_w, σ_v) pairs.
  - Episode keys are `jr.split(PRNGKey(test_seed), 1000)[:epi]`, the same as `test.py`, so the first 32 episodes are A's episodes.
  - Episodes run vmapped in batches; `--batch 1` reproduces `test.py` exactly.
  - It writes per-episode CSV rows with the columns in `analysis/robust_csv.py`, deliberately separate from the shared `analysis/logging_csv.py`.
- **Upstream `test.py` bugs to avoid** (don't patch the submodule):
  - `--stochastic` crashes because a 4-argument actor is called with 3 arguments. `eval_robust.py` wraps `algo.step` itself.
  - `--offset ≠ 0` raises IndexError.
  - There is no `--area-size` flag.
- **Config:**
  - `noise.{sigma_w, sigma_v, shift_type, shift_level}` and `eval.{ckpt, n_episodes, test_seed, stochastic}` are defined in `configs/schema.md`.
  - `configs/t2/sigma_grid.yaml` holds candidate σ levels that are **not calibrated yet**. Its `levels` block gets filled in after calibration.
- **Shared Phase 0 interfaces**: `configs/schema.md`, `analysis/logging_csv.py`, `analysis/seeds.py` and `methods/cm_dgppo/margin.py` (owned by T3; don't touch the `margin:` block). Changes to these need a team PR.
- **Naming mismatch:** the schema and CSV say `informarl_lag`; DGPPO code and checkpoints say `informarl_lagr`.

## Conventions

- Don't retrain or change the DGPPO baselines. T2 only changes **test-time** conditions on A's checkpoints.
- Fix the H1 numeric pass criterion before looking at results.
- Before design choices with several valid options, especially σ levels or how a shift is defined, ask the user instead of deciding.
- Don't commit raw results (CSV, logs, wandb, videos). Record the seed, git commit and config for every run. Never hand-edit reported numbers.
- One task per branch, PR into `main` with a reviewer, `git pull --rebase origin main` daily. Never push directly to `main`.
