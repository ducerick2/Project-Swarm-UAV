#!/usr/bin/env bash
# T2 — quét H1: mọi checkpoint × lưới σ trong GRID (mặc định quét từng yếu tố:
# (σ_w, 0) rồi (0, σ_v)), cùng tập khóa episode để so ghép cặp giữa các mức σ.
# N và số vật cản giữ như lúc train.
#
#   tmux new -d -s t2noise 'bash scripts/t2/run_noise_grid.sh'
#   # hiệu chỉnh σ (DGPPO seed0, lưới rộng):
#   SEEDS=0 GRID=configs/t2/calib_grid.yaml SWEEP_NAME=calib bash scripts/t2/run_noise_grid.sh
#
# Knob (env-var, xem scripts/t2/_env.sh): CKPT_ROOT OUTDIR GPU EPI ENVS METHODS SEEDS
#   + GRID MODES SWEEP_NAME (CSV mặc định: $SWEEP_DIR/episodes.csv)
# Mọi kết quả + cấu hình của lượt quét: $OUTDIR/sweeps/<thời_gian>_<tên>/ (xem t2_sweep_init).
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

GRID="${GRID:-$REPO/configs/t2/sigma_grid.yaml}"
MODES="${MODES:-det}"

t2_activate
t2_sweep_init "${SWEEP_NAME:-noise}" "$GRID"

echo "=== T2 NOISE start $(date) | grid=$GRID modes=[$MODES] epi=$EPI gpu=$GPU ==="
for ENV in $ENVS; do
  for METHOD in $METHODS; do
    for SEED in $SEEDS; do
      CKPT="$CKPT_ROOT/$ENV/$METHOD/seed$SEED"
      [ -d "$CKPT" ] || { echo "!! thiếu checkpoint $CKPT"; continue; }
      for MODE in $MODES; do
        FLAG=""; [ "$MODE" = "stoch" ] && FLAG="--stochastic"
        echo "--- $(date '+%F %T') $METHOD/$ENV/seed$SEED $MODE ---"
        t2_eval --path "$CKPT" --grid "$GRID" $FLAG --csv "$CSV"
      done
    done
  done
done
echo "=== T2 NOISE done $(date) ==="
echo "CSV: $CSV"
t2_sweep_finish
