#!/usr/bin/env bash
# T1 monitor: in trạng thái tiến độ + đánh dấu sự kiện MỚI kể từ lần check trước.
# Dùng cho vòng theo dõi định kỳ. Chỉ ghi 2 file state, không sửa gì khác.
#
#   bash scripts/t1/monitor_check.sh
#
# Knob (env-var, xem scripts/t1/_env.sh): OUTDIR CSV LOG TOTAL
# Lưu ý: ĐỪNG pipe output qua `head` (SIGPIPE có thể bỏ qua bước ghi state ở cuối).
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

RUNDIR="$OUTDIR"
STATE=$RUNDIR/.monitor_state          # số run đã báo lần trước
ERRSTATE=$RUNDIR/.monitor_errcount    # số dòng lỗi đã báo lần trước
ERRPAT="TRAIN FAILED|TEST PARSE FAIL|Traceback|CUDA_ERROR|OutOfMemory|NO CKPT"

# số run đã xong = số dòng CSV - 1 (header)
done_n=0
[ -f "$CSV" ] && done_n=$(($(wc -l < "$CSV") - 1))
[ "$done_n" -lt 0 ] && done_n=0
prev=$(cat "$STATE" 2>/dev/null || echo 0)
new=$((done_n - prev))

# số dòng lỗi trong log
err_n=0
[ -f "$LOG" ] && err_n=$(grep -cE "$ERRPAT" "$LOG" 2>/dev/null)
perr=$(cat "$ERRSTATE" 2>/dev/null || echo 0)
new_err=$((err_n - perr))

# trạng thái tmux (session uav có window "train") + GPU
tmux_ok=no
tmux has-session -t uav 2>/dev/null && tmux list-windows -t uav 2>/dev/null | grep -q "train" && tmux_ok=yes
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 2>/dev/null)
finished=no; [ "$done_n" -ge "$TOTAL" ] && finished=yes

echo "STATUS done=$done_n/$TOTAL new=$new errors=$err_n new_err=$new_err tmux=$tmux_ok gpu1_util=${gpu_util}% finished=$finished"

if [ "$new" -gt 0 ]; then
  echo "--- RUN MỚI XONG ($new) ---"
  { head -1 "$CSV"; tail -n "$new" "$CSV"; }
fi
if [ "$new_err" -gt 0 ]; then
  echo "--- LỖI MỚI ($new_err) ---"
  grep -nE "$ERRPAT" "$LOG" | tail -n "$new_err"
fi

# cập nhật state
echo "$done_n" > "$STATE"
echo "$err_n" > "$ERRSTATE"
