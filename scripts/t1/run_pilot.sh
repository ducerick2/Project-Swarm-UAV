#!/usr/bin/env bash
# T1 pilot: 3 algo x 2 env x seed0 x N=3 x 200k steps, tuần tự trên 1 GPU.
# Chạy trong tmux:  tmux new -d -s t1pilot 'bash scripts/t1/run_pilot.sh'
set -u

REPO=/data/ducbm3/Master/TKPTTT/Project-Swarm-UAV
OUTDIR=/data/ducbm3/dgppo_runs
CSV=$OUTDIR/t1_baseline_pilot.csv
GPU=${GPU:-1}
STEPS=${STEPS:-200000}
SEED=${SEED:-0}
N=${N:-3}

source /data/ducbm3/dgppo_env/bin/activate
mkdir -p "$OUTDIR"

echo "=== T1 PILOT start $(date) | gpu=$GPU steps=$STEPS seed=$SEED N=$N ==="

for ALGO in dgppo informarl informarl_lagr; do
  for ENV in LidarSpread LidarLine; do
    echo "--- $(date '+%F %T') RUN $ALGO / $ENV ---"
    python "$REPO/scripts/t1/run_baseline.py" \
      --algo "$ALGO" --env "$ENV" -n "$N" --seed "$SEED" \
      --steps "$STEPS" --gpu "$GPU" --test-epi 32 \
      --outdir "$OUTDIR" --csv "$CSV"
  done
done

echo "=== T1 PILOT done $(date) ==="
echo "CSV: $CSV"
