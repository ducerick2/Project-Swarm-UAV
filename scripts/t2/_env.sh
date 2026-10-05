#!/usr/bin/env bash
# scripts/t2/_env.sh — Cấu hình dùng chung cho mọi script T2.
#
# Source từ script khác:  source "$(dirname "$0")/_env.sh"
# MỌI biến đều override được bằng env-var khi gọi, ví dụ:
#   CKPT_ROOT=~/ckpts GPU=0 EPI=32 bash scripts/t2/run_shift_grid.sh
#
# File này KHÔNG bật `set -u` và KHÔNG tự activate venv — để caller tự quyết.

_T2_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="${REPO:-$(cd "$_T2_ENV_DIR/../.." && pwd)}"

# venv JAX (cách dựng: scripts/t1/README.md §1 trên nhánh t1-reproduce)
VENV="${VENV:-/data/ducbm3/dgppo_env}"

# Checkpoint baseline của T1: <CKPT_ROOT>/<env>/<method>/seed<N>/{config.yaml,models/}
# Mặc định trỏ sang worktree t1-reproduce cạnh repo này.
CKPT_ROOT="${CKPT_ROOT:-$REPO/../Project-Swarm-UAV-t1-reproduce/results/checkpoints}"

# Nơi ghi hồ sơ chạy (config + kết quả), NGOÀI repo — không commit (hơn 1000 file).
# Cấu trúc: $OUTDIR/sweeps/<thời_gian>_<tên>/. Máy RTX 3080: OUTDIR=/home/mantd/DGPPO/t2_runs
OUTDIR="${OUTDIR:-/data/ducbm3/dgppo_runs/t2}"

GPU="${GPU:-1}"
EPI="${EPI:-256}"
ENVS="${ENVS:-LidarSpread LidarLine}"
METHODS="${METHODS:-dgppo}"        # T2 chỉ đánh giá DGPPO (InforMARL* thuộc T1)
SEEDS="${SEEDS:-0 1 2}"

t2_activate() { source "$VENV/bin/activate"; }

# Mở một lượt quét: tạo $SWEEP_DIR, ghi lại toàn bộ cấu hình + lệnh, chuyển console vào
# $SWEEP_DIR/console.log. Dùng: t2_sweep_init <tên> [file_lưới]
t2_sweep_init() {
  local name="$1" grid="${2:-}"
  SWEEP_DIR="${SWEEP_DIR:-$OUTDIR/sweeps/$(date +%Y%m%d-%H%M%S)_$name}"
  RUN_ROOT="$SWEEP_DIR/runs"
  CSV="${CSV:-$SWEEP_DIR/episodes.csv}"
  mkdir -p "$RUN_ROOT"
  [ -n "$grid" ] && cp "$grid" "$SWEEP_DIR/"
  {
    echo "# lượt quét T2: $name — $(date -Iseconds) — host $(hostname)"
    echo "SCRIPT=\"$0\""
    for v in REPO VENV CKPT_ROOT OUTDIR GPU EPI ENVS METHODS SEEDS GRID MODES N_LIST OBS_LIST CSV SWEEP_DIR; do
      echo "$v=\"${!v:-}\""
    done
    echo "GIT_DESCRIBE=\"$(git -C "$REPO" describe --always --dirty)\""
  } > "$SWEEP_DIR/sweep.env"
  git -C "$REPO" diff HEAD > "$SWEEP_DIR/git_diff.patch"
  exec > >(tee -a "$SWEEP_DIR/console.log") 2>&1
  echo "SWEEP_DIR: $SWEEP_DIR"
}

# Đóng lượt quét: bảng tổng hợp toàn lượt.
t2_sweep_finish() {
  python "$REPO/scripts/t2/summarize.py" "$CSV" --md "$SWEEP_DIR/summary.md" \
    --out-csv "$SWEEP_DIR/summary.csv" > /dev/null && echo "Tổng hợp: $SWEEP_DIR/summary.md"
}

# Gọi eval_robust.py với shim JAX 0.6 + GPU + số episode chung; thêm cờ riêng qua "$@".
t2_eval() {
  PYTHONPATH="$REPO/scripts/compat:${PYTHONPATH:-}" WANDB_MODE=offline \
    python "$REPO/scripts/t2/eval_robust.py" --epi "$EPI" --gpu "$GPU" \
    ${RUN_ROOT:+--run-root "$RUN_ROOT"} "$@"
}

# Đọc một khóa từ config.yaml của checkpoint (vd: t2_ckpt_key <ckpt> obs)
t2_ckpt_key() { awk -v k="$2:" '$1 == k {print $2; exit}' "$1/config.yaml"; }
