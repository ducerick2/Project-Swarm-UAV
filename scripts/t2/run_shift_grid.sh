#!/usr/bin/env bash
# T2 — dịch chuyển cấu hình KHÔNG cần nhiễu: checkpoint train ở N=3, obs=3 chạy với
#   - N lúc test ∈ N_LIST (obs giữ như lúc train),
#   - số vật cản lúc test ∈ OBS_LIST (N giữ như lúc train),
# σ = 0, cả policy tất định lẫn stochastic. Mỗi episode một dòng vào CSV.
#
#   tmux new -d -s t2shift 'bash scripts/t2/run_shift_grid.sh'
#
# Knob (env-var, xem scripts/t2/_env.sh): CKPT_ROOT OUTDIR GPU EPI ENVS METHODS SEEDS
#   + N_LIST OBS_LIST MODES SWEEP_NAME (CSV mặc định: $SWEEP_DIR/episodes.csv)
# Mọi kết quả + cấu hình của lượt quét: $OUTDIR/sweeps/<thời_gian>_<tên>/ (xem t2_sweep_init).
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

N_LIST="${N_LIST:-3 5 7}"
OBS_LIST="${OBS_LIST:-3 5 8}"
MODES="${MODES:-det stoch}"

t2_activate
t2_sweep_init "${SWEEP_NAME:-shift}"

echo "=== T2 SHIFT start $(date) | N=[$N_LIST] obs=[$OBS_LIST] modes=[$MODES] epi=$EPI gpu=$GPU ==="
for ENV in $ENVS; do
  for METHOD in $METHODS; do
    for SEED in $SEEDS; do
      CKPT="$CKPT_ROOT/$ENV/$METHOD/seed$SEED"
      [ -d "$CKPT" ] || { echo "!! thiếu checkpoint $CKPT"; continue; }
      N_TRAIN=$(t2_ckpt_key "$CKPT" num_agents)
      OBS_TRAIN=$(t2_ckpt_key "$CKPT" obs)
      for MODE in $MODES; do
        FLAG=""; [ "$MODE" = "stoch" ] && FLAG="--stochastic"
        for N in $N_LIST; do
          echo "--- $(date '+%F %T') $METHOD/$ENV/seed$SEED $MODE N=$N obs=$OBS_TRAIN ---"
          t2_eval --path "$CKPT" -n "$N" --obs "$OBS_TRAIN" $FLAG --csv "$CSV"
        done
        for OBS in $OBS_LIST; do
          [ "$OBS" = "$OBS_TRAIN" ] && continue   # điểm gốc đã chạy ở vòng N
          echo "--- $(date '+%F %T') $METHOD/$ENV/seed$SEED $MODE N=$N_TRAIN obs=$OBS ---"
          t2_eval --path "$CKPT" -n "$N_TRAIN" --obs "$OBS" $FLAG --csv "$CSV"
        done
      done
    done
  done
done
echo "=== T2 SHIFT done $(date) ==="
echo "CSV: $CSV"
t2_sweep_finish
