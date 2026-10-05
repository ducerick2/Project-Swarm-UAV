# T2 — Hướng dẫn cài môi trường và chạy đánh giá

Tài liệu này hướng dẫn từ đầu đến cuối cách:
1. dựng môi trường;
2. kiểm tra cài đặt đúng;
3. chạy lại toàn bộ đánh giá độ bền vững của DGPPO (mạch T2);
4. tổng hợp kết quả.

Kết quả và phân tích nằm ở `docs/T2_RESULTS.md`; quy tắc hiệu chỉnh và tiêu chí H1 ở `docs/T2_calibration.md`.

Mọi lệnh đều chạy **từ thư mục gốc repo** (`Project-Swarm-UAV`, nhánh `t2-robustness`).

---

## 1. Yêu cầu

| Thành phần | Yêu cầu | Đã kiểm chứng |
|---|---|---|
| Hệ điều hành | Linux x86_64 | Ubuntu, kernel 5.15 |
| GPU | NVIDIA, driver ≥ 525 (CUDA 12); có thể chạy CPU nhưng chậm hơn nhiều | RTX 3080 10 GB, driver 535.183.01 (CUDA 12.2) |
| Python | **3.11** (DGPPO ghim `numpy==1.26.4`, không cài được trên Python ≥ 3.13) | 3.11.0 |
| Dung lượng đĩa | khoảng 6 GB cho venv (các gói CUDA của JAX khá nặng) | |

RTX 5090 (Blackwell) cũng dùng được cấu hình này: JAX 0.6.2 hỗ trợ sm_120. Bản `jax>=0.4.26` mà DGPPO ghim gốc thì không chạy được trên 5090, xem `scripts/t1/README.md` trên nhánh `t1-reproduce`.

## 2. Lấy mã nguồn

### 2.1 Repo và submodule DGPPO

```bash
git clone --recurse-submodules <url-repo-nhom> Project-Swarm-UAV
cd Project-Swarm-UAV
git checkout t2-robustness
git submodule update --init --recursive      # bắt buộc: third_party/dgppo mặc định rỗng
git -C third_party/dgppo log -1 --format=%h  # phải in 51b3b11
```

Không sửa code trong `third_party/dgppo`, vì submodule được ghim cố định. Mọi chỉnh sửa tương thích đặt trong `scripts/compat/sitecustomize.py`. Shim này khôi phục các alias `jax.tree_*` mà JAX 0.6 đã bỏ.

### 2.2 Checkpoint của T1

T2 đánh giá các checkpoint DGPPO mà T1 đã train. Checkpoint nằm ở nhánh `t1-reproduce`, trong `results/checkpoints/<env>/<method>/seed<k>/`. Mặc định, script T2 tìm chúng trong một **worktree cạnh repo**:

```bash
# từ gốc repo (nhánh t2-robustness):
git worktree add ../Project-Swarm-UAV-t1-reproduce t1-reproduce
ls ../Project-Swarm-UAV-t1-reproduce/results/checkpoints/LidarSpread/dgppo/   # seed0 seed1 seed2
```

Nếu checkpoint nằm ở chỗ khác, đặt biến `CKPT_ROOT=<thư mục chứa LidarSpread/ và LidarLine/>` khi chạy (xem mục 5.1).

## 3. Cài môi trường

Đặt venv **ngoài repo**. Trên máy RTX 3080 hiện tại, venv đã dựng sẵn ở `/home/mantd/DGPPO/dgppo_env`, nên bỏ qua được mục này.

### 3.1 Tạo Python 3.11

Có thể dùng conda hoặc python3.11 có sẵn:

```bash
conda create -y -p ~/py311 python=3.11      # hoặc dùng python3.11 có sẵn
~/py311/bin/python -m venv /đường/dẫn/dgppo_env
source /đường/dẫn/dgppo_env/bin/activate
pip install -U pip
```

### 3.2 Cài gói: cách A, đúng phiên bản đã dùng (khuyến nghị)

`scripts/t2/requirements.lock.txt` là `pip freeze` của venv đã chạy ra mọi số liệu trong `docs/T2_RESULTS.md`.

```bash
pip install --timeout 30 --retries 20 -r scripts/t2/requirements.lock.txt
pip install -e third_party/dgppo
```

### 3.3 Cài gói: cách B, lệnh gốc (lấy phiên bản mới nhất thoả ràng buộc)

```bash
pip install --timeout 30 --retries 20 "jax[cuda12]==0.6.2" "numpy==1.26.4" "flax<0.11" optax jraph einops \
    jaxtyping equinox "tensorflow-probability<0.26" scipy matplotlib seaborn colour rich tqdm ipdb pyyaml \
    pytest attrs jax_dataclasses imageio-ffmpeg
pip install -e third_party/dgppo
```

**Ghi chú về các gói:**
- `jax_dataclasses` và `imageio-ffmpeg` không có trong `requirements.txt` của DGPPO, nhưng bắt buộc phải có. Thiếu `jax_dataclasses` thì `import dgppo` lỗi.
- Không dùng `pip install -r third_party/dgppo/requirements.txt` trực tiếp. File đó ghim `jax[cuda12]>=0.4.26` và kéo theo nhiều gói không cần (mujoco, wandb, opencv…).

**Nếu tải chậm hoặc đứt:** các gói `nvidia-cudnn-cu12` (khoảng 770 MB), `nvidia-cublas-cu12` (khoảng 580 MB) và các gói CUDA khác rất nặng.
- `--timeout 30 --retries 20` giúp pip tự nối lại khi mất kết nối.
- Nếu tải bị đứng hẳn, ngắt (Ctrl+C) rồi chạy lại cùng lệnh: các gói đã tải xong sẽ được lấy từ cache của pip.

### 3.4 Kiểm tra cài đặt

```bash
export PYTHONPATH=$PWD/scripts/compat:$PYTHONPATH     # shim JAX 0.6 (nên đặt trong mọi phiên)
python -c "import jax; print(jax.__version__, jax.devices())"
# kỳ vọng: 0.6.2 [CudaDevice(id=0)]   (CPU-only sẽ in [CpuDevice(id=0)])
python -c "import dgppo, jax_dataclasses; print('dgppo ok')"
```

## 4. Kiểm tra code (test)

```bash
export PYTHONPATH=$PWD/scripts/compat:$PYTHONPATH
pytest tests -q                                   # chạy trên CPU (mặc định), khoảng 2.5 phút, kỳ vọng 21 passed
T2_TEST_PLATFORM=gpu pytest tests -q              # chạy trên GPU
pytest tests/test_robust_utils.py -q              # phần không cần JAX (vài giây)
pytest "tests/test_noise_wrapper.py::test_sigma0_matches_test_rollout" -q   # một test cụ thể
```

| File test | Kiểm tra gì |
|---|---|
| `tests/test_noise_wrapper.py` | σ = 0 cho kết quả như `test_rollout` gốc của DGPPO; độ lệch chuẩn nhiễu đo được đúng bằng σ; nhiễu cảm biến không đụng trạng thái thật; cost tính trên trạng thái thật; chạy được dưới `jit` / `vmap` với σ traced |
| `tests/test_robust_utils.py` | khoảng tin cậy Wilson, ghi CSV, gộp tham số CLI/config, suy ra loại dịch chuyển |

Test dùng policy InforMARL khởi tạo ngẫu nhiên nên không cần checkpoint.

**Nghiệm thu đầu-cuối trên checkpoint thật:** ở σ = 0, kết quả phải khớp số liệu T1 trong `scripts/t1/pilot_results.csv` (DGPPO seed0: LidarSpread safe 100%, reward −0.885; LidarLine safe 98.958%, reward −0.28):

```bash
VENV=$VIRTUAL_ENV OUTDIR=/home/mantd/DGPPO/t2_runs GPU=0 SEEDS=0 EPI=32 GRID=configs/t2/sigma0_grid.yaml SWEEP_NAME=check0 \
  bash scripts/t2/run_noise_grid.sh
```

## 5. Chạy đánh giá

### 5.1 Biến cấu hình chung (`scripts/t2/_env.sh`)

Mọi script `.sh` của T2 đọc các biến dưới đây. Biến nào cũng ghi đè được bằng cách đặt trước lệnh, ví dụ `GPU=0 EPI=32 bash scripts/t2/run_noise_grid.sh`.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `VENV` | `/data/ducbm3/dgppo_env` (máy 5090 của nhóm) | venv có JAX + DGPPO. Máy RTX 3080: `/home/mantd/DGPPO/dgppo_env` |
| `CKPT_ROOT` | `../Project-Swarm-UAV-t1-reproduce/results/checkpoints` | thư mục checkpoint T1 |
| `OUTDIR` | `/data/ducbm3/dgppo_runs/t2` (máy 5090) | nơi ghi hồ sơ chạy, **ngoài repo** (không commit). Máy RTX 3080: `/home/mantd/DGPPO/t2_runs` |
| `GPU` | `1` (máy 5090) | chỉ số GPU (`CUDA_VISIBLE_DEVICES`). **Máy RTX 3080: `GPU=0`** |
| `EPI` | `256` | số episode mỗi cấu hình mỗi seed |
| `ENVS` | `LidarSpread LidarLine` | môi trường |
| `METHODS` | `dgppo` | phương pháp (T2 chỉ đánh giá DGPPO) |
| `SEEDS` | `0 1 2` | seed checkpoint |
| `GRID` | `configs/t2/sigma_grid.yaml` | lưới σ (chỉ `run_noise_grid.sh`) |
| `MODES` | `det` (nhiễu) / `det stoch` (dịch chuyển) | policy tất định / lấy mẫu |
| `N_LIST`, `OBS_LIST` | `3 5 7`, `3 5 8` | số agent / số vật cản lúc test (chỉ `run_shift_grid.sh`) |
| `SWEEP_NAME` | `noise` / `shift` | hậu tố tên thư mục lượt quét |

Trên máy RTX 3080, đặt một lần cho cả phiên shell:

```bash
export VENV=/home/mantd/DGPPO/dgppo_env OUTDIR=/home/mantd/DGPPO/t2_runs GPU=0
```

### 5.2 Lượt quét nhiễu (`scripts/t2/run_noise_grid.sh`)

Script chạy mỗi checkpoint (env × seed) qua mọi cặp (σ_w, σ_v) trong `GRID`. Mặc định quét từng yếu tố: σ_w với σ_v = 0, rồi σ_v với σ_w = 0.

```bash
# quét H1: lưới σ chung, 3 seed (khoảng 5 phút trên RTX 3080)
SEEDS="0 1 2" EPI=256 GRID=configs/t2/sigma_grid.yaml SWEEP_NAME=noise_h1 bash scripts/t2/run_noise_grid.sh

# hiệu chỉnh: lưới rộng / lưới mịn, chỉ seed0
SEEDS=0 GRID=configs/t2/calib_grid.yaml SWEEP_NAME=calib bash scripts/t2/run_noise_grid.sh
SEEDS=0 GRID=configs/t2/calib_fine_grid.yaml SWEEP_NAME=calib_fine bash scripts/t2/run_noise_grid.sh
```

### 5.3 Lượt quét dịch chuyển (`scripts/t2/run_shift_grid.sh`)

Script đổi số agent N (giữ số vật cản như lúc train), rồi đổi số vật cản (giữ N như lúc train), với σ = 0, cho cả policy tất định lẫn stochastic.

```bash
SEEDS="0 1 2" EPI=256 SWEEP_NAME=shift bash scripts/t2/run_shift_grid.sh   # khoảng 25 phút trên RTX 3080
```

LidarLine chỉ chạy được tới N = 7. Env đặt các landmark cách nhau (N−2)·6·r, và với diện tích 1.5 thì N = 8 sẽ báo `ValueError`.

### 5.4 Đánh giá một checkpoint (`scripts/t2/eval_robust.py`)

Đây là lõi mà hai script quét gọi. Có thể gọi trực tiếp để thử nhanh.

```bash
export PYTHONPATH=$PWD/scripts/compat:$PYTHONPATH
CK=../Project-Swarm-UAV-t1-reproduce/results/checkpoints/LidarSpread/dgppo/seed0

# vài mức σ tự chọn (tích Descartes σ_w × σ_v)
python scripts/t2/eval_robust.py --path $CK --sigma-w 0 0.2 --sigma-v 0 --epi 64 --gpu 0
# dùng lưới chung
python scripts/t2/eval_robust.py --path $CK --grid configs/t2/sigma_grid.yaml --gpu 0
# dịch chuyển: 5 agent, policy stochastic
python scripts/t2/eval_robust.py --path $CK -n 5 --stochastic --gpu 0
# đọc khối noise / eval từ config theo lược đồ chung
python scripts/t2/eval_robust.py --config configs/example.yaml --gpu 0
# chạy tuần tự từng episode như test.py (để đối chiếu với test.py)
python scripts/t2/eval_robust.py --path $CK --epi 32 --batch 1 --sigma-w 0 --sigma-v 0 --gpu 0
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--path` | (bắt buộc nếu không có `eval.ckpt` trong `--config`) | thư mục checkpoint (`config.yaml`, `models/`) |
| `--config` | — | YAML theo `configs/schema.md`: khối `noise` (`sigma_w`, `sigma_v`, `shift_type`, `shift_level`) và `eval` (`ckpt`, `n_episodes`, `test_seed`, `stochastic`). Tham số dòng lệnh được ưu tiên hơn |
| `-n`, `--num-agents` | N lúc train | số agent lúc test |
| `--obs` | số vật cản lúc train | số vật cản lúc test |
| `--sigma-w`, `--sigma-v` | `0` | danh sách mức σ (tích Descartes) |
| `--grid` | — | YAML lưới σ (`mode: one_at_a_time` hoặc `product`), thay cho `--sigma-*` |
| `--epi` | 256 | số episode |
| `--seed` | 1234 | test seed (giống `test.py`); 32 episode đầu trùng với episode của T1 |
| `--stochastic` | tắt | lấy mẫu action thay vì dùng mode |
| `--step` | step lớn nhất | step checkpoint cần nạp |
| `--batch` | 32 | số episode chạy song song (vmap); `1` = tuần tự như `test.py` |
| `--csv` | `results/t2/robust.csv` (bị `.gitignore` chặn) | CSV gộp (nối thêm); nên trỏ ra ngoài repo, ví dụ `$OUTDIR/adhoc/x.csv` |
| `--run-root` | `<thư mục của --csv>/runs` | nơi tạo thư mục hồ sơ run |
| `--gpu` | — | đặt `CUDA_VISIBLE_DEVICES` |
| `--cpu` | tắt | ép chạy CPU |

**Lưu ý:**
- Một process chỉ dựng được một cấu hình env (N, số vật cản), vì `make_env` của DGPPO sửa tại chỗ dict tham số của class. Muốn thử nhiều N thì gọi nhiều process, như `run_shift_grid.sh` đang làm.
- Không dùng `third_party/dgppo/test.py --stochastic`: lệnh này crash do lỗi của DGPPO. `eval_robust.py --stochastic` tự xử lý chế độ stochastic.

### 5.5 Chạy dài trong tmux (khuyến nghị)

```bash
tmux new -d -s t2 -n noise
tmux send-keys -t t2:noise 'export VENV=/home/mantd/DGPPO/dgppo_env OUTDIR=/home/mantd/DGPPO/t2_runs GPU=0; SEEDS="0 1 2" SWEEP_NAME=noise_h1 bash scripts/t2/run_noise_grid.sh' Enter
tmux new-window -t t2 -n shift
tmux send-keys -t t2:shift 'export VENV=/home/mantd/DGPPO/dgppo_env OUTDIR=/home/mantd/DGPPO/t2_runs GPU=0; SEEDS="0 1 2" SWEEP_NAME=shift bash scripts/t2/run_shift_grid.sh' Enter
tmux attach -t t2            # xem; Ctrl+b rồi d để thoát mà không dừng job
```

Hai lượt quét chạy song song trên một RTX 3080 10 GB vẫn đủ bộ nhớ, vì `XLA_PYTHON_CLIENT_PREALLOCATE=false` được đặt sẵn.

## 6. Tổng hợp và kiểm định

```bash
D=$OUTDIR/sweeps/<thời_gian>_noise_h1

# bảng theo σ (cũng tự sinh ở $D/summary.md khi lượt quét kết thúc)
python scripts/t2/summarize.py $D/episodes.csv --md $D/summary.md --out-csv $D/summary.csv

# kiểm định H1 (mức low/mid/high đọc từ configs/t2/sigma_grid.yaml)
python scripts/t2/h1_test.py $D/episodes.csv --out $D/h1_test.md
python scripts/t2/h1_test.py $D/episodes.csv --seeds 1 2 --out $D/h1_test_seed12.md   # bỏ seed0 (đã dùng để hiệu chỉnh)

# điền lại mọi bảng trong docs/T2_RESULTS.md từ $OUTDIR/sweeps
python scripts/t2/fill_results_tables.py --runs $OUTDIR/sweeps
```

**Về `fill_results_tables.py`:**
- Script thay nội dung giữa các dấu `<!-- BEGIN:x -->` và `<!-- END:x -->` trong tài liệu. Phần viết tay không bị đụng tới.
- Với mỗi loại lượt quét (`*_check0`, `*_calib`, `*_calib_fine`, `*_noise_h1`, `*_shift`), script lấy **thư mục mới nhất**. Nếu chạy lại một lượt quét, bảng sẽ cập nhật theo lần chạy mới; khi đó nhớ đọc lại phần phân tích viết tay cho khớp với số mới.

## 7. Kết quả được ghi ở đâu

Mỗi lượt quét tạo `$OUTDIR/sweeps/<YYYYmmdd-HHMMSS>_<SWEEP_NAME>/` (ngoài repo):

```
sweep.env           toàn bộ biến cấu hình của lượt quét + git describe
<lưới>.yaml         bản sao lưới σ đã dùng
git_diff.patch      thay đổi code (chưa commit) lúc chạy
console.log         toàn bộ output
episodes.csv        mỗi episode một dòng, gộp mọi checkpoint (cột: analysis/robust_csv.py)
summary.md/.csv     bảng tổng hợp theo cấu hình × σ
h1_test*.md         (nếu chạy h1_test.py)
runs/<YYYYmmdd-HHMMSS>_<method>_<env>_seed<k>_N<n>_obs<o>_<det|stoch>/
    run.yaml          lệnh, tham số đã gộp, cặp σ, config + sha256 checkpoint, tham số env thực tế,
                      phiên bản thư viện, GPU, trạng thái (done/failed), bảng kết quả
    ckpt_config.yaml  bản sao config.yaml của checkpoint
    inputs/           bản sao file --grid / --config đã dùng
    code/             bản sao code T2 lúc chạy
    episodes.csv, summary.md/.csv, console.log
```

**Trên máy RTX 3080**, toàn bộ hồ sơ của các lượt chạy đã báo cáo nằm ở `/home/mantd/DGPPO/t2_runs/`. Ngoài `sweeps/` còn có:
- `logs/`: log các lệnh cài đặt, test và debug đã chạy;
- `archive_no_record/`: các lượt chạy trước khi có hệ thống hồ sơ, không dùng làm số liệu báo cáo.

**Không commit kết quả chạy:** hồ sơ có hơn 1000 file, nên chỉ lưu ngoài repo. Repo chỉ chứa code, config, tài liệu và các bảng tổng hợp trong `docs/T2_RESULTS.md`. Khi chia sẻ, nén thư mục `$OUTDIR/sweeps/` lại để gửi. Nếu gọi `eval_robust.py` lẻ mà không đặt `--csv`, kết quả rơi vào `results/t2/` trong repo; thư mục này bị `.gitignore` chặn (`*.csv`, `runs/`), nhưng nên xoá sau khi thử.

## 8. Cấu hình

| File | Nội dung |
|---|---|
| `configs/t2/sigma_grid.yaml` | lưới σ dùng cho quét H1, kèm mức low / mid / high đã hiệu chỉnh (`levels`) |
| `configs/t2/calib_grid.yaml` | lưới rộng dùng để hiệu chỉnh |
| `configs/t2/calib_fine_grid.yaml` | lưới mịn quanh điểm gãy |
| `configs/t2/sigma0_grid.yaml` | chỉ σ = 0 (nghiệm thu) |
| `configs/example.yaml`, `configs/schema.md` | lược đồ cấu hình chung của nhóm: khối `noise` và `eval` |

**Định dạng lưới:**

```yaml
mode: one_at_a_time      # (σ_w, 0) rồi (0, σ_v); hoặc product: mọi tổ hợp
sigma_w: [0.0, 0.1, 0.2]
sigma_v: [0.0, 0.025]
```

**Đơn vị:**
- σ_w theo vận tốc: env clip |v| ≤ 0.5, dt = 0.03;
- σ_v theo độ dài: bán kính agent r = 0.05.

## 9. Lưu ý 

- **Không bật `--xla_gpu_deterministic_ops=true`** trong `XLA_FLAGS`. Với JAX 0.6.2, cờ này làm rollout policy cho kết quả sai (DGPPO σ = 0: 47.9% thay vì 100%). `eval_robust.py` sẽ từ chối chạy nếu cờ này được bật.
- **Kết quả không trùng từng bit giữa các lần chạy, trên cả GPU lẫn CPU.**
  - Nguyên nhân là phép cộng dồn song song của XLA, không phải khởi tạo ngẫu nhiên.
  - Ở σ = 0, khoảng 0.4% episode đổi kết quả an toàn (±0.3 pp).
  - Đây là hành vi bình thường; so sánh ở mức thống kê. Chi tiết: `docs/T2_calibration.md` mục 4.5.
- **Chỉ số GPU khác nhau giữa các máy:** máy 5090 của nhóm mặc định `GPU=1`, máy RTX 3080 dùng `GPU=0`. Sai chỉ số thì JAX báo không thấy GPU hoặc tự rơi về CPU.
- **Cảnh báo `dot_search_space ... All configs were filtered out`** của XLA khi biên dịch là vô hại.
- **`ModuleNotFoundError: jax_dataclasses`:** chạy `pip install jax_dataclasses`.
- **`AttributeError: module 'jax' has no attribute 'tree_map'`:** chưa đặt `PYTHONPATH=$PWD/scripts/compat:$PYTHONPATH`. `eval_robust.py` tự nạp shim này, nhưng `test.py` và các script khác thì cần.
- **`third_party/dgppo` rỗng:** chạy `git submodule update --init --recursive`.
- **`!! thiếu checkpoint …` trong log lượt quét:** kiểm tra `CKPT_ROOT` (mục 2.2).
