#!/usr/bin/env bash
# T1 — đánh giá (mô phỏng) 1 checkpoint baseline đã train, in reward/cost/safe_rate
# + xuất video. Tự tìm checkpoint mới nhất theo (method, env, seed).
#
#   bash scripts/t1/eval_ckpt.sh <method> <env> <seed> [n_agents] [gpu] [epi]
# Ví dụ:
#   bash scripts/t1/eval_ckpt.sh dgppo LidarSpread 0        # đúng dòng bảng đã report
#   bash scripts/t1/eval_ckpt.sh informarl LidarLine 1 3 1 32
#
# method: dgppo | informarl | informarl_lagr
# env:    LidarSpread | LidarLine
set -eu

METHOD=${1:?method}; ENV=${2:?env}; SEED=${3:?seed}
N=${4:-3}; GPU=${5:-1}; EPI=${6:-32}

REPO=/data/ducbm3/Master/TKPTTT/Project-Swarm-UAV
DGPPO=$REPO/third_party/dgppo
RUNDIR=/data/ducbm3/dgppo_runs/n$N/$ENV/$METHOD

# tìm checkpoint mới nhất khớp seed
CKPT=$(ls -dt "$RUNDIR"/seed${SEED}_* 2>/dev/null | head -1 || true)
if [ -z "${CKPT:-}" ]; then
  echo "Không thấy checkpoint cho $METHOD/$ENV/seed$SEED trong $RUNDIR"; exit 1
fi
echo "Checkpoint: $CKPT"

source /data/ducbm3/dgppo_env/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline WANDB_SILENT=true
export PYTHONPATH=$REPO/scripts/compat:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=$GPU

cd "$DGPPO"
# Mặc định KHÔNG render video (render video đang lỗi PIL trên setup này và làm
# DGPPO rơi vào ipdb gây treo). Muốn thử video: VIDEO=1 bash eval_ckpt.sh ...
# `< /dev/null` đảm bảo kể cả rơi vào ipdb cũng nhận EOF và thoát, không treo.
VIDEO=${VIDEO:-0}
if [ "$VIDEO" = "1" ]; then
  echo "(thử render video — nếu lỗi sẽ tự thoát, không treo)"
  python test.py --path "$CKPT" --epi "$EPI" < /dev/null
  echo "Video (nếu render được) ở: $CKPT/videos/"
else
  python test.py --path "$CKPT" --epi "$EPI" --no-video < /dev/null
fi
