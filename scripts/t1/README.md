# T1 — Tái hiện baseline (Giai đoạn 1)

Tái hiện **DGPPO**, **InforMARL**, **InforMARL-Lagrangian** trên `LidarSpread`,
`LidarLine` (N=3), lấy số liệu an toàn (safety) + cost để so với bài gốc.

> **Train thật nằm ở đâu?** T1 KHÔNG tự cài lại train loop. Nó gọi thẳng code gốc
> của tác giả trong submodule: `third_party/dgppo/train.py` + `test.py`, với 3
> thuật toán ở `third_party/dgppo/dgppo/algo/{dgppo,informarl,informarl_lagr}.py`
> (mỗi class một method `update()`). `run_baseline.py` chỉ là lớp bọc: train →
> test → ghi CSV. (`scripts/train.py` ở gốc repo là KHUNG cho T2/T3, chưa dùng.)

## 1. Dựng môi trường (máy 5090 / Blackwell)

RTX 5090 là Blackwell (sm_120) → **không dùng được** `jax>=0.4.26` như DGPPO ghim;
phải dùng JAX mới (đã test 0.6.2). JAX 0.6 bỏ `jax.tree_map`, nên có shim
`scripts/compat/sitecustomize.py` khôi phục alias (nạp qua PYTHONPATH) +
trỏ matplotlib vào `imageio-ffmpeg` để render video — **không sửa mã submodule**.

```bash
cd <repo>
python3 -m venv /data/ducbm3/Master/TKPTTT/dgppo_env && source /data/ducbm3/Master/TKPTTT/dgppo_env/bin/activate
pip install -U pip
pip install "jax[cuda12]"                                   # JAX mới cho Blackwell
grep -v '^jax' third_party/dgppo/requirements.txt | pip install -r /dev/stdin
pip install jax_dataclasses imageio-ffmpeg                  # thiếu trong requirements
pip install -e third_party/dgppo
```

## 2. Cấu hình dùng chung — `_env.sh`

Mọi script bash T1 `source scripts/t1/_env.sh` để lấy đường dẫn; **không hardcode**.
Override bằng env-var khi gọi:

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `REPO` | tự suy từ vị trí script | gốc repo |
| `OUTDIR` | `/data/ducbm3/Master/TKPTTT/dgppo_runs` | nơi lưu ckpt/log/CSV **thô** (ngoài repo) |
| `VENV` | `/data/ducbm3/Master/TKPTTT/dgppo_env` | venv JAX-Blackwell |
| `CSV` | `$OUTDIR/t1_baseline_pilot.csv` | file kết quả (xem ghi chú tên ở dưới) |
| `LOG` | `$OUTDIR/seeds_console.log` | log console seed-sweep (monitor đọc) |
| `TOTAL` | `30` | tổng run kỳ vọng (5 seed × 6 config) |

> Tên CSV giữ hậu tố `_pilot` vì file đang chứa **cả** pilot (seed0) lẫn seed-sweep
> (seed1–4); có thể đổi tên sau khi xong 30 run.

## 3. Các script & cách chạy

| Script | Việc |
|--------|------|
| `run_baseline.py` | **Lõi**: 1 cấu hình (train→test→1 dòng CSV). |
| `run_pilot.sh` | Pilot 6 run: 3 algo × 2 env × seed0. |
| `run_seeds.sh` | Seed-sweep: `SEEDS="1 2 3 4"` × 6 config (nối cùng CSV). |
| `monitor_check.sh` | In tiến độ `done=N/30` + báo run/lỗi mới (cho vòng theo dõi). |
| `eval_ckpt.sh` | Mô phỏng/đánh giá 1 checkpoint (+`VIDEO=1` để render .mp4). |
| `coverage_check.py` | Chẩn đoán coverage/goal 1 episode. |
| `collect_best_ckpts.sh` | Gom best-ckpt (final) vào `results/checkpoints/` để commit. |

### Chạy 1 cấu hình
```bash
source /data/ducbm3/Master/TKPTTT/dgppo_env/bin/activate
export PYTHONPATH=<repo>/scripts/compat:$PYTHONPATH   # shim JAX0.6
python scripts/t1/run_baseline.py --algo dgppo --env LidarSpread -n 3 \
  --seed 0 --steps 200000 --gpu 1 --test-epi 32 \
  --outdir /data/ducbm3/Master/TKPTTT/dgppo_runs --csv /data/ducbm3/Master/TKPTTT/dgppo_runs/t1_baseline_pilot.csv
```

### Pilot / seed-sweep trong tmux
```bash
tmux new -d -s t1pilot 'bash scripts/t1/run_pilot.sh'
# hoặc seed-sweep (nên tee ra log để monitor đọc):
tmux send-keys -t uav:1 'bash scripts/t1/run_seeds.sh 2>&1 | tee -a /data/ducbm3/Master/TKPTTT/dgppo_runs/seeds_console.log' Enter
```

### Theo dõi
```bash
bash scripts/t1/monitor_check.sh          # STATUS done=N/30 ... + run/lỗi mới
```

## 4. Checkpoint tốt nhất (commit được — ngoại lệ .gitignore)

Quy ước chung: **không** commit kết quả thô. **Ngoại lệ T1:** best-checkpoint
(final step, ~640 KB/run → ~16 MB cho 30 run) được commit vào
`results/checkpoints/` để chia sẻ + tái lập (đã mở ngoại lệ trong `.gitignore`).

```bash
bash scripts/t1/collect_best_ckpts.sh     # gom final-ckpt + config vào results/checkpoints/
git add results/checkpoints && git commit -m "T1: best checkpoints (final step)"
# load lại: python third_party/dgppo/test.py --path results/checkpoints/<env>/<method>/seed<N>
```

## 5. Ghi chú hiệu năng

- ~8 iter/s trên 1×RTX5090 → 200k steps ≈ ~7 giờ/run.
- Grid đầy đủ 5 seed (30 run) tuần tự ≈ ~9 ngày/1 GPU.
- Checkpoint 1 step chỉ 3 file `.pkl` (~640 KB). Kết quả thô (ckpt mọi step + wandb
  + video) ở `OUTDIR` (ngoài repo). Chỉ code + `pilot_results.*` + best-ckpt lên git.
