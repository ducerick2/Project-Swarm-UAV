#!/usr/bin/env bash
# scripts/t1/_env.sh — Cấu hình dùng chung cho mọi script T1.
#
# Source từ script khác:  source "$(dirname "$0")/_env.sh"
# MỌI biến đều override được bằng env-var khi gọi, ví dụ:
#   OUTDIR=/tmp/runs GPU=0 VENV=~/myenv bash scripts/t1/run_pilot.sh
#
# File này KHÔNG bật `set -u` và KHÔNG tự activate venv — để caller tự quyết.

# Thư mục gốc repo: suy ra từ vị trí file này (…/scripts/t1/_env.sh -> lên 2 cấp)
_T1_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="${REPO:-$(cd "$_T1_ENV_DIR/../.." && pwd)}"

# Nơi lưu checkpoint/log/CSV THÔ (NGOÀI repo, không commit — xem .gitignore)
OUTDIR="${OUTDIR:-/data/ducbm3/Master/TKPTTT/dgppo_runs}"

# venv JAX cho Blackwell/5090 (cách dựng: xem scripts/t1/README.md)
VENV="${VENV:-/data/ducbm3/Master/TKPTTT/dgppo_env}"

# CSV kết quả T1. Giữ tên lịch sử "_pilot" vì file này đang chứa CẢ seed-sweep
# (pilot seed0 + seed1..4); có thể đổi tên sau khi xong toàn bộ 30 run.
CSV="${CSV:-$OUTDIR/t1_baseline_pilot.csv}"

# Log console của seed-sweep (run_seeds.sh ghi/tee vào đây) — monitor đọc file này
LOG="${LOG:-$OUTDIR/seeds_console.log}"

# Tổng số run kỳ vọng (5 seed × 6 config) — monitor dùng để báo finished
TOTAL="${TOTAL:-30}"

# Tiện ích: activate venv. Gọi `t1_activate` trong script nào cần chạy python.
t1_activate() { source "$VENV/bin/activate"; }
