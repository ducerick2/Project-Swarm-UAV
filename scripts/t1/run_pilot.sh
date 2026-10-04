#!/usr/bin/env bash
# T1 pilot: 3 algo × 2 env × 1 seed × N=3 × 200k steps, tuần tự trên 1 GPU.
#
# Chạy trong tmux:
#   tmux new -d -s t1pilot 'bash scripts/t1/run_pilot.sh'
#   tail -f "$OUTDIR/pilot_console.log"   # nếu bạn tự tee ra file
#
# Knob (env-var, xem scripts/t1/_env.sh):
#   GPU=1 STEPS=200000 SEED=0 N=3 OUTDIR=... CSV=... VENV=...
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

GPU="${GPU:-1}"; STEPS="${STEPS:-200000}"; SEED="${SEED:-0}"; N="${N:-3}"

t1_activate
mkdir -p "$OUTDIR"

echo "=== T1 PILOT start $(date) | repo=$REPO gpu=$GPU steps=$STEPS seed=$SEED N=$N ==="
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
