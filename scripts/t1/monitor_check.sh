#!/usr/bin/env bash
# T1 pilot monitor: in ra trạng thái + đánh dấu sự kiện MỚI kể từ lần check trước.
# Dùng cho vòng theo dõi định kỳ. Không tự sửa gì ngoài file state.
set -u
RUNDIR=/data/ducbm3/dgppo_runs
CSV=$RUNDIR/t1_baseline_pilot.csv
LOG=${LOG:-$RUNDIR/seeds_console.log}
STATE=$RUNDIR/.monitor_state          # số run đã báo
ERRSTATE=$RUNDIR/.monitor_errcount    # số dòng lỗi đã báo
TOTAL=${TOTAL:-30}                    # 5 seed x 6 config

done_n=0
[ -f "$CSV" ] && done_n=$(($(wc -l < "$CSV") - 1))
[ "$done_n" -lt 0 ] && done_n=0
prev=$(cat "$STATE" 2>/dev/null || echo 0)
new=$((done_n - prev))

err_n=0
[ -f "$LOG" ] && err_n=$(grep -cE "TRAIN FAILED|TEST PARSE FAIL|Traceback|CUDA_ERROR|OutOfMemory|NO CKPT" "$LOG" 2>/dev/null)
perr=$(cat "$ERRSTATE" 2>/dev/null || echo 0)
new_err=$((err_n - perr))

# trạng thái tmux + GPU
tmux_ok=no; tmux has-session -t uav 2>/dev/null && tmux list-windows -t uav 2>/dev/null | grep -q "train" && tmux_ok=yes
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 2>/dev/null)
finished=no; [ "$done_n" -ge "$TOTAL" ] && finished=yes

echo "STATUS done=$done_n/$TOTAL new=$new errors=$err_n new_err=$new_err tmux=$tmux_ok gpu1_util=${gpu_util}% finished=$finished"

if [ "$new" -gt 0 ]; then
  echo "--- RUN MỚI XONG ($new) ---"
  { head -1 "$CSV"; tail -n "$new" "$CSV"; }
fi
if [ "$new_err" -gt 0 ]; then
  echo "--- LỖI MỚI ($new_err) ---"
  grep -nE "TRAIN FAILED|TEST PARSE FAIL|Traceback|CUDA_ERROR|OutOfMemory|NO CKPT" "$LOG" | tail -n "$new_err"
fi

# cập nhật state
echo "$done_n" > "$STATE"
echo "$err_n" > "$ERRSTATE"
