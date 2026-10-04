#!/usr/bin/env bash
# T1 — gom "best checkpoint" (final step, mặc định 200000) + config.yaml của MỌI
# run đã train ĐỦ, vào trong repo để commit/chia sẻ:
#   results/checkpoints/<env>/<method>/seed<N>/
#       ├─ models/<FINAL>/{actor,Vl,Vh}.pkl
#       └─ config.yaml
#
# Chỉ gom run đã đạt đúng step FINAL (bỏ qua run đang train dở). Idempotent
# (copy đè, chạy lại nhiều lần an toàn). Load lại: trỏ test.py --path vào thư mục
# seed<N>/ (algo.load sẽ đọc models/<FINAL>).
#
#   bash scripts/t1/collect_best_ckpts.sh
#
# Knob (env-var): OUTDIR (nguồn runs, xem _env.sh), FINAL (step cần gom)
set -eu
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

FINAL="${FINAL:-200000}"
DEST="$REPO/results/checkpoints"

n=0; skip=0
for d in "$OUTDIR"/n*/*/*/seed*_*/; do
  [ -d "${d}models/$FINAL" ] || { skip=$((skip+1)); continue; }
  METHOD=$(basename "$(dirname "${d%/}")")
  ENV=$(basename "$(dirname "$(dirname "${d%/}")")")
  SEED=$(basename "${d%/}" | sed -E 's/^seed([0-9]+)_.*/\1/')
  out="$DEST/$ENV/$METHOD/seed$SEED"
  mkdir -p "$out/models/$FINAL"
  cp -f "${d}models/$FINAL/"*.pkl "$out/models/$FINAL/"
  [ -f "${d}config.yaml" ] && cp -f "${d}config.yaml" "$out/config.yaml"
  echo "gom: $ENV/$METHOD/seed$SEED  (step $FINAL)"
  n=$((n+1))
done
echo "=== Xong: gom $n run vào $DEST ; bỏ qua $skip run chưa đủ step $FINAL ==="
echo "Dung lượng: $(du -sh "$DEST" 2>/dev/null | awk '{print $1}')"
