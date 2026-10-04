#!/usr/bin/env bash
# Eval mạch T3: các run CM-DGPPO (relax / fixed) + baseline DGPPO của T1, rồi in bảng so sánh.
#
#   bash scripts/eval_t3.sh runs        # eval mọi run đã train xong trong logs/<ENV_ID>/
#   bash scripts/eval_t3.sh baseline    # eval checkpoint DGPPO của T1 (cùng cấu hình, không nhiễu)
#   bash scripts/eval_t3.sh summary     # bảng: safety, task cost, khoảng hở
#
#   ENV_ID=LidarLine bash scripts/eval_t3.sh runs     # môi trường khác
#   SIGMA_W_GRID="0 0.005 0.01" bash scripts/eval_t3.sh runs   # thêm mức nhiễu (mặc định chỉ 0)
#
# Mỗi run ghi CSV riêng trong results/t3/runs/ (tránh nhiều tiến trình ghi chung một file);
# `summary` gộp lại. Key eval cố định (EVAL_SEED) cho MỌI phương pháp => cùng tập episode.
# Run chưa train xong (checkpoint mới nhất < train.steps) được bỏ qua.

set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-2}
PARALLEL=${PARALLEL:-3}          # số eval song song (mỗi eval ~2-3 GB)
MEM_GB=${MEM_GB:-3}
ENV_ID=${ENV_ID:-LidarSpread}
PY=${PY:-.venv-jax/bin/python}
EVAL_SEED=${EVAL_SEED:-10000}
SIGMA_W_GRID=${SIGMA_W_GRID:-"0"}
SIGMA_V=${SIGMA_V:-0}
T1_DIR=${T1_DIR:-/data/ducbm3/dgppo_runs/n3}   # checkpoint baseline của T1
T1_STEP=${T1_STEP:-200000}

log() { echo "[$(date +%H:%M:%S)] $*"; }

setup_gpu() {
  "$PY" -c "import jax; assert jax.__version__.startswith('0.5.')" 2>/dev/null \
    || { echo "Thiếu JAX 0.5.x ở $PY (xem AGENTS.md, mục Môi trường chạy)"; exit 1; }
  local total; total=$(nvidia-smi --id="$GPU" --query-gpu=memory.total --format=csv,noheader,nounits)
  MEM_FRAC=$(python3 -c "print(f'{$MEM_GB*1024/$total:.4f}')")
  mkdir -p results/t3/runs logs/launch
}

eval_one() {  # eval_one <run_dir> [tham số thêm cho eval_cm.py...]
  local run=$1; shift
  local tag; tag=$(echo "$run" | tr '/' '_' | sed 's/^_*//')
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRAC \
    "$PY" scripts/eval_cm.py --run "$run" --eval-seed "$EVAL_SEED" \
    --sigma-w $SIGMA_W_GRID --sigma-v "$SIGMA_V" "$@" \
    --csv "results/t3/runs/test_${tag}.csv" --csv-full "results/t3/runs/test_full_${tag}.csv" \
    > "logs/launch/eval_${tag}.log" 2>&1
}

runs() {
  setup_gpu
  local run last want n=0
  for run in logs/"$ENV_ID"/*/*/; do
    run=${run%/}
    [[ -d "$run/models" ]] || continue
    last=$(ls "$run/models" | grep -E '^[0-9]+$' | sort -n | tail -1)
    want=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['train']['steps'])" "$run/config.yaml" 2>/dev/null || echo 0)
    if (( ${last:-0} < want )); then log "bỏ qua run chưa xong: $run (checkpoint $last/$want)"; continue; fi
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 5; done
    log "eval $run"; eval_one "$run" & n=$((n + 1))
  done
  wait; log "xong $n run (log: logs/launch/eval_*.log)"
}

baseline() {
  setup_gpu
  local run
  for run in "$T1_DIR/$ENV_ID"/dgppo/seed*; do
    [[ -d "$run/models/$T1_STEP" ]] || { log "thiếu checkpoint $T1_STEP: $run"; continue; }
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 5; done
    log "eval baseline $run (bước $T1_STEP)"; eval_one "$run" --step "$T1_STEP" &
  done
  wait; log "xong baseline"
}

summary() {
  local out=results/t3/test_full_${ENV_ID}.csv first=1 f
  : > "$out"
  for f in results/t3/runs/test_full_*.csv; do
    [[ -f "$f" ]] || continue
    grep -q ",$ENV_ID," "$f" || continue
    if (( first )); then cat "$f" >> "$out"; first=0; else tail -n +2 "$f" >> "$out"; fi
  done
  [[ -s "$out" ]] || { echo "Chưa có kết quả eval cho $ENV_ID"; exit 1; }
  python3 scripts/summarize_eval.py --csv "$out" --eval-seed "$EVAL_SEED" --by label sigma_w \
    --metrics safety_rate safety_rate_swarm task_cost clearance_mean clearance_cvar5
}

case "${1:-}" in
  runs) runs ;;
  baseline) baseline ;;
  summary) summary ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
