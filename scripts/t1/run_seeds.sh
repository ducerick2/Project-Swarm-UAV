#!/usr/bin/env bash
# T1 mở rộng seed: với mỗi seed chạy 3 algo x 2 env (N=3, 200k), tuần tự trên 1 GPU.
# Append vào CÙNG CSV với pilot (seed 0). Chạy trong tmux:
#   tmux send-keys -t uav:1 'bash scripts/t1/run_seeds.sh' Enter
set -u
REPO=/data/ducbm3/Master/TKPTTT/Project-Swarm-UAV
OUTDIR=/data/ducbm3/Master/TKPTTT/dgppo_runs
CSV=$OUTDIR/t1_baseline_pilot.csv
GPU=${GPU:-1}
STEPS=${STEPS:-200000}
N=${N:-3}
SEEDS=${SEEDS:-"1 2 3 4"}    # seed 0 đã có từ pilot -> tổng 5 seed

source /data/ducbm3/Master/TKPTTT/dgppo_env/bin/activate

echo "=== T1 SEED EXPANSION start $(date) | seeds=[$SEEDS] gpu=$GPU steps=$STEPS N=$N ==="
for SEED in $SEEDS; do
  echo "--- $(date '+%F %T') BẮT ĐẦU SEED $SEED ---"
  for ALGO in dgppo informarl informarl_lagr; do
    for ENV in LidarSpread LidarLine; do
      echo "--- $(date '+%F %T') RUN seed$SEED $ALGO / $ENV ---"
      python "$REPO/scripts/t1/run_baseline.py" \
        --algo "$ALGO" --env "$ENV" -n "$N" --seed "$SEED" \
        --steps "$STEPS" --gpu "$GPU" --test-epi 32 \
        --outdir "$OUTDIR" --csv "$CSV"
    done
  done
  echo "=== SEED $SEED done $(date) ==="
done
echo "=== T1 ALL SEEDS done $(date) ==="
echo "CSV: $CSV"
