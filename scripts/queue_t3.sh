#!/usr/bin/env bash
# Hàng đợi train T3 DÙNG CHUNG cho nhiều GPU (mặc định GPU 2 và 3).
#
#   bash scripts/queue_t3.sh start     # chạy nền 1 worker / GPU; tự nhận run đang chạy để không chạy trùng
#   bash scripts/queue_t3.sh status    # job nào đã nhận, đang chạy ở GPU nào, tiến độ
#   bash scripts/queue_t3.sh stop      # dừng các worker (KHÔNG dừng run đang train)
#
#   GPUS="2 3" PARALLEL=2 MEM_GB=13 bash scripts/queue_t3.sh start
#
# Danh sách job: logs/launch/queue_relax.txt (đổi bằng QUEUE=...; sửa được, theo thứ tự ưu tiên;
# mỗi dòng "config seed [override ...]"). Mỗi job được nhận đúng một lần bằng `mkdir` nguyên tử
# trong logs/launch/claims/, nên các worker không bao giờ chạy trùng.
# Cấu hình mặc định giống baseline T1 (128 env, batch 16384, 200k bước, eval 1000, lưu 10 000).
# Thời gian đo được: mỗi GPU ~9 bước/s tổng (chạy 2 run song song không nhanh hơn 1 run),
# tức ~6 h / run 200k bước nếu chạy một mình, ~12 h / cặp run song song.

set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-"2 3"}
PARALLEL=${PARALLEL:-2}          # run song song mỗi GPU (2 × 13 GB vừa GPU 32 GB)
MEM_GB=${MEM_GB:-13}             # đỉnh đo được: DGPPO 8.5 GB, ACI vô hướng ~9 GB, CM full 10.1 GB
STEPS=${STEPS:-200000}           # như baseline T1
N_ENV=${N_ENV:-128}
BATCH=${BATCH:-16384}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
ENV_ID=${ENV_ID:-LidarSpread}
PY=${PY:-.venv-jax/bin/python}
QUEUE=${QUEUE:-logs/launch/queue_relax.txt}
CLAIMS=logs/launch/claims
export WANDB_MODE=${WANDB_MODE:-offline} PYTHONUNBUFFERED=1

log() { echo "[$(date +%H:%M:%S)] $*"; }

# tên job: <env>_<config>_seed<s>[_<override đã làm sạch>]
config_env() {  # env khai báo trong configs/t3/<config>.yaml (trống nếu không có)
  sed -n 's/^env: *\([A-Za-z]*\).*/\1/p' "configs/t3/$1.yaml" 2>/dev/null | head -1
}

job_name() {  # env: override `env=...` của job > `env:` trong config > ENV_ID
  local c s o env rest; read -r c s o <<< "$1"
  env=$(grep -oE '(^| )env=[A-Za-z]+' <<< "${o:-}" | tail -1 | sed 's/.*env=//')
  env=${env:-$(config_env "$c")}
  rest=$(sed -E 's/(^| )env=[A-Za-z]+//g; s/^ +//' <<< "${o:-}")
  local n="${env:-$ENV_ID}_${c}_seed${s}"
  [[ -n "$rest" ]] && n+="_$(echo "$rest" | tr ' =.' '__p')"
  echo "$n"
}

# các tiến trình train_cm.py đang chạy: in "pid gpu tên_job"
running_trains() {
  local pid args cfg seed gpu ov
  for pid in $(pgrep -u "$(id -u)" -f "scripts/train_cm.py" || true); do
    args=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
    cfg=$(sed -n 's|.*--config [^ ]*/\([^/ ]*\)\.yaml.*|\1|p' <<< "$args")
    seed=$(sed -n 's|.*--seed \([0-9]*\).*|\1|p' <<< "$args")
    [[ -n "$cfg" && -n "$seed" ]] || continue
    # override riêng của job (bỏ train.*, và env= nếu trùng env của config)
    ov=$(grep -o -- "--set .*" <<< "$args" | sed 's/--set //' | tr ' ' '\n' \
         | grep -v -e '^train\.' -e '^$' | sed "s/^env=$(config_env "$cfg")\$//" | grep -v '^$' \
         | tr '\n' ' ' | sed 's/ $//') || true
    gpu=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
    echo "$pid ${gpu:-?} $(job_name "$cfg $seed ${ov}")"
  done
}

worker() {  # worker <gpu> <pid đang chạy sẵn trên GPU này...>
  local gpu=$1; shift
  local wait_pids="$*" total frac
  total=$(nvidia-smi --id="$gpu" --query-gpu=memory.total --format=csv,noheader,nounits)
  frac=$(python3 -c "print(f'{$MEM_GB*1024/$total:.4f}')")
  busy() { local n p; n=$(jobs -rp | wc -l)
    for p in $wait_pids; do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo "$n"; }
  log "GPU$gpu worker: $PARALLEL slot × $MEM_GB GB (frac $frac), chờ PID: ${wait_pids:-không}"
  local job name config seed overrides
  while IFS= read -r job || [[ -n "$job" ]]; do
    [[ -z "${job// }" || "$job" == \#* ]] && continue
    name=$(job_name "$job")
    [[ -d "$CLAIMS/$name" ]] && continue
    while (( $(busy) >= PARALLEL )); do sleep 15; done
    mkdir "$CLAIMS/$name" 2>/dev/null || continue      # worker khác vừa nhận trước
    read -r config seed overrides <<< "$job"
    log "GPU$gpu bắt đầu $name"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=$frac \
      "$PY" scripts/train_cm.py --config "configs/t3/${config}.yaml" --seed "$seed" \
      --steps "$STEPS" --n-env-train "$N_ENV" --batch-size "$BATCH" \
      --set train.eval_interval="$EVAL_INTERVAL" train.save_interval="$SAVE_INTERVAL" \
      ${overrides:-} < /dev/null > "logs/launch/${name}.log" 2>&1 &
    echo "$! $gpu" > "$CLAIMS/$name/pid"
  done < "$QUEUE"
  wait
  log "GPU$gpu hết job"
}

start() {
  "$PY" -c "import jax; assert jax.__version__.startswith('0.5.')" 2>/dev/null \
    || { echo "Thiếu JAX 0.5.x ở $PY (xem AGENTS.md, mục Môi trường chạy)"; exit 1; }
  mkdir -p "$CLAIMS"
  [[ -f "$QUEUE" ]] || { echo "Thiếu hàng đợi $QUEUE"; exit 1; }

  # nhận các run đang chạy sẵn (vd. từ một lần start trước) để không chạy lại
  declare -A wait_on=()
  while read -r pid gpu name; do
    [[ -n "$pid" ]] || continue
    mkdir -p "$CLAIMS/$name"; echo "$pid $gpu" > "$CLAIMS/$name/pid"
    wait_on[$gpu]="${wait_on[$gpu]:-} $pid"
    log "đã nhận run đang chạy: $name (PID $pid, GPU $gpu)"
  done < <(running_trains)

  for g in $GPUS; do
    nohup bash "$0" _worker "$g" ${wait_on[$g]:-} > "logs/launch/worker_gpu${g}.log" 2>&1 &
    echo "$!" > "logs/launch/worker_gpu${g}.pid"
    log "worker GPU $g chạy nền (PID $!, log: logs/launch/worker_gpu${g}.log)"
  done
  log "xem tiến độ: bash scripts/queue_t3.sh status"
}

status() {
  echo "== Job (hàng đợi $QUEUE)"
  local job name info pid gpu prog
  while IFS= read -r job || [[ -n "$job" ]]; do
    [[ -z "${job// }" || "$job" == \#* ]] && continue
    name=$(job_name "$job")
    if [[ -f "$CLAIMS/$name/pid" ]]; then
      read -r pid gpu < "$CLAIMS/$name/pid"
      if [[ "$pid" == 0 ]]; then printf "  %-48s %s\n" "$name" "bỏ qua"; continue; fi
      prog=$(tr '\r' '\n' < "logs/launch/${name}.log" 2>/dev/null | grep -aoE "[0-9]+/[0-9]+ \[[^]]*\]" | tail -1)
      if kill -0 "$pid" 2>/dev/null; then info="ĐANG CHẠY GPU $gpu  $prog"
      elif grep -aq "Traceback" "logs/launch/${name}.log" 2>/dev/null; then info="LỖI (GPU $gpu, xem log)  $prog"
      else info="xong/dừng (GPU $gpu)  $prog"; fi
    else
      info="chờ"
    fi
    printf "  %-48s %s\n" "$name" "$info"
  done < "$QUEUE"
  echo "== Worker"
  pgrep -u "$(id -u)" -af "queue_t3.sh _worker" | awk '{print "  worker GPU " $NF ": chạy (PID " $1 ")"}' || true
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'
}

stop() {
  local p
  for p in $(pgrep -u "$(id -u)" -f "queue_t3.sh _worker" || true); do
    kill "$p" 2>/dev/null && log "dừng worker PID $p ($(ps -o args= -p "$p" 2>/dev/null | awk '{print $NF}'))"
  done
  rm -f logs/launch/worker_gpu*.pid
  log "các run đang train vẫn chạy tiếp; muốn dừng hẳn: kill <PID> (xem status)"
}

case "${1:-}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  _worker) shift; worker "$@" ;;
  *) sed -n '2,18p' "$0"; exit 1 ;;
esac
