# T1 — Tái hiện baseline (Pha 1)

Tái hiện DGPPO, InforMARL, InforMARL-Lagrangian trên `LidarSpread`, `LidarLine`.

## Môi trường (đã dựng trên máy 5090)

RTX 5090 là Blackwell (sm_120) → **không dùng được** `jax>=0.4.26` như DGPPO ghim.
Phải dùng JAX mới (đã test 0.6.2 chạy được trên 5090). JAX 0.6 bỏ `jax.tree_map`,
nên có shim `scripts/compat/sitecustomize.py` khôi phục alias (nạp qua PYTHONPATH),
**không sửa mã submodule** (giữ pin 51b3b11 sạch).

```bash
cd /data/ducbm3
python3 -m venv dgppo_env && source dgppo_env/bin/activate
pip install -U pip
pip install "jax[cuda12]"                          # JAX mới cho Blackwell
grep -v '^jax' third_party/dgppo/requirements.txt | pip install -r /dev/stdin
pip install jax_dataclasses                          # thiếu trong requirements
pip install -e third_party/dgppo
```

Biến môi trường mỗi lần chạy:
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
export PYTHONPATH=<repo>/scripts/compat:$PYTHONPATH   # shim JAX0.6
export CUDA_VISIBLE_DEVICES=1                          # pin GPU trống
```

## Chạy 1 cấu hình

```bash
python scripts/t1/run_baseline.py --algo dgppo --env LidarSpread -n 3 \
  --seed 0 --steps 200000 --gpu 1 --test-epi 32 \
  --outdir /data/ducbm3/dgppo_runs --csv /data/ducbm3/dgppo_runs/t1_baseline.csv
```
Tự train → test → ghi 1 dòng CSV (method,env,N,obs,seed,steps,reward,cost,safety_rate,train_runtime_s,git_commit,ckpt).

## Chạy pilot (6 run, 1 seed) trong tmux

```bash
tmux new -d -s t1pilot 'bash scripts/t1/run_pilot.sh > /data/ducbm3/dgppo_runs/pilot_console.log 2>&1'
tmux ls; tail -f /data/ducbm3/dgppo_runs/pilot_console.log
```

## Ghi chú hiệu năng

- ~7.4 iter/s steady-state trên 1×RTX5090 → 200k steps ≈ ~7 giờ/run.
- Pilot 6 run tuần tự ≈ ~42h. Grid đầy đủ 5 seed (30 run) ≈ ~9 ngày/1 GPU.
- Kết quả (CSV + checkpoint) ở `/data/ducbm3/dgppo_runs/` (ngoài repo, không commit).

## TODO T1

- [ ] Pilot 1 seed (đang chạy) → bảng so sánh đầu tiên
- [ ] Mở rộng seed (3 hoặc 5) sau khi pilot xác nhận xu hướng
- [ ] Phân tích runtime N=3,5,7 (steps/s, thời gian hội tụ, RAM GPU, thời gian suy luận)
- [ ] So xu hướng với bài gốc; ghi lại mọi chênh lệch
