# T2 — Hiệu chỉnh mức nhiễu σ_w, σ_v (chỉ DGPPO)

Phạm vi T2: chỉ đánh giá **DGPPO**. InforMARL và InforMARL-Lagrangian thuộc T1, không nằm trong quét, H1 hay báo cáo của T2.

## 1. Thiết lập

- **Checkpoint:** DGPPO seed0, `LidarSpread` và `LidarLine` (N=3, obs=3, step 200000), lấy từ `results/checkpoints/` của nhánh `t1-reproduce`.
- **Episode:** 256 episode, test seed 1234 (cùng khóa với `test.py`; 32 episode đầu trùng T1), policy tất định.
- **Lưới:** `configs/t2/calib_grid.yaml`, quét từng yếu tố (σ_w với σ_v = 0, rồi σ_v với σ_w = 0).
- **Định nghĩa nhiễu:**
  - σ_w cộng vào vận tốc sau mỗi bước (clip |v| ≤ 0.5);
  - σ_v cộng vào vị trí quan sát của agent và điểm hit LiDAR (r = 0.05);
  - xem `envs/noise_wrapper.py`.
- **Chỉ số chính:** `safe_agent` = tỉ lệ (episode, agent) không vi phạm, đúng định nghĩa `safe_rate` của `test.py`.

## 2. Quy tắc chọn mức (viết TRƯỚC khi có kết quả)

Áp riêng cho σ_w và σ_v, dựa trên DGPPO `LidarSpread` seed0:

- **high** = σ nhỏ nhất trên lưới có `safe_agent` < 95% (điểm gãy). Nếu không mức nào dưới 95%, high = mức lớn nhất của lưới và ghi chú là chưa gãy.
- **mid** = điểm lưới gần high/2 nhất (tính theo giá trị σ).
- **low** = điểm lưới gần high/4 nhất, khác 0.
- `LidarLine` dùng để kiểm tra các mức có hợp lý trên env thứ hai không; không dùng để chọn.

## 3. Tiêu chí H1 (chốt trước khi chạy lưới đầy đủ)

**Dữ liệu kiểm định:**
- DGPPO, cả 2 env, seed 0–2, 256 episode;
- cùng tập khóa episode ở mọi mức σ, để so ghép cặp theo episode.

**Tiêu chí:**
- **Chính:** ở mức **mid** (riêng cho σ_w và σ_v), `safe_agent` thấp hơn σ = 0 **ít nhất 5 điểm phần trăm**, kiểm định một phía ghép cặp theo episode, p < 0.05. H1 đúng nếu điều kiện thỏa trên cả hai env.
- **Phân tích độ nhạy** (báo cáo kèm, không thay tiêu chí chính): ngưỡng 2 pp và 10 pp; mức low và high.
- Seed0 đã dùng để hiệu chỉnh, nên báo cáo thêm kết quả chỉ trên seed 1–2.

## 4. Kết quả (chạy 2026-10-05, có hồ sơ đầy đủ)

**Môi trường chạy:**
- RTX 3080, driver 535.183.01, JAX 0.6.2 (`jax[cuda12]`), DGPPO @51b3b11.
- Repo @`1595a0d` cộng các thay đổi T2 chưa commit.
- Venv `/home/mantd/DGPPO/dgppo_env`.

**Nơi lưu:** mỗi lượt quét nằm trong `/home/mantd/DGPPO/t2_runs/sweeps/<thời_gian>_<tên>/` (ngoài repo, không commit), gồm:
- `sweep.env`: toàn bộ biến cấu hình;
- bản sao lưới σ, `git_diff.patch`, `console.log`;
- `episodes.csv`, `summary.md`, `summary.csv`;
- `runs/<run>/`: hồ sơ từng checkpoint gồm `run.yaml` (tham số đã gộp, config checkpoint + sha256 file `.pkl`, tham số env thực tế, phiên bản thư viện, GPU, kết quả), `ckpt_config.yaml`, `code/`, `episodes.csv`, `summary.*`, `console.log`.

| Lượt | Thư mục | Lệnh (từ gốc repo, kèm `VENV=/home/mantd/DGPPO/dgppo_env OUTDIR=/home/mantd/DGPPO/t2_runs GPU=0 SEEDS=0`) |
|---|---|---|
| hiệu chỉnh | `sweeps/20261005-153934_calib` | `EPI=256 GRID=configs/t2/calib_grid.yaml SWEEP_NAME=calib bash scripts/t2/run_noise_grid.sh` |
| lưới mịn | `sweeps/20261005-154058_calib_fine` | `EPI=256 GRID=configs/t2/calib_fine_grid.yaml SWEEP_NAME=calib_fine bash scripts/t2/run_noise_grid.sh` |
| nghiệm thu σ=0 | `sweeps/20261005-154219_check0` | `EPI=32 GRID=configs/t2/sigma0_grid.yaml SWEEP_NAME=check0 bash scripts/t2/run_noise_grid.sh` |

Các lượt chạy trước khi có hệ thống hồ sơ nằm ở `/home/mantd/DGPPO/t2_runs/archive_no_record/`. Chúng cho cùng kết luận và cùng mức đã chọn, nhưng không được dùng làm số liệu báo cáo.

### 4.1 Nghiệm thu σ = 0 (DGPPO seed0, 32 episode)

Khớp `pilot_results.csv` của T1 (máy 5090).

| env | safe_agent (T2) | T1 | reward (T2) | T1 |
|---|---|---|---|---|
| LidarSpread | 100.0% | 100.0% | −0.884 | −0.885 |
| LidarLine | 98.958% | 98.958% | −0.280 | −0.28 |

### 4.2 Lưới hiệu chỉnh (DGPPO seed0, 256 episode, policy tất định)

`safe_agent` tính bằng %, kèm Δ so với σ = 0 tính bằng điểm phần trăm. `viol a/o` là tổng số agent từng va agent khác / va vật cản.

| σ_v | Spread safe_agent (Δ) | Spread viol a/o | Line safe_agent (Δ) | Line viol a/o |
|---|---|---|---|---|
| 0 | 99.1 | 4/3 | 99.5 | 0/4 |
| 0.0125 (0.25r) | 98.8 (−0.3) | 6/3 | 99.2 (−0.3) | 0/6 |
| 0.025 (0.5r) | 99.7 (+0.7) | 2/0 | 99.2 (−0.3) | 0/6 |
| 0.0375 (0.75r) | 99.1 (+0.0) | 2/5 | 98.8 (−0.7) | 0/9 |
| 0.05 (r) | 97.9 (−1.2) | 0/16 | 97.9 (−1.6) | 2/14 |
| 0.075 (1.5r) | **88.9 (−10.2)** | 8/77 | 90.5 (−9.0) | 29/46 |
| 0.1 (2r) | 77.3 (−21.7) | 30/146 | 77.2 (−22.3) | 81/100 |
| 0.15 (3r) | 53.5 (−45.6) | 182/188 | 53.6 (−45.8) | 251/140 |
| 0.2 (4r) | 38.3 (−60.8) | 299/238 | 39.7 (−59.8) | 371/166 |

| σ_w | Spread safe_agent (Δ) | Spread dist2goal | Line safe_agent (Δ) | Line dist2goal |
|---|---|---|---|---|
| 0 | 99.1 | 0.701 | 99.5 | 0.048 |
| 0.01 | 99.0 (−0.1) | 0.700 | 99.2 (−0.3) | 0.048 |
| 0.02 | 99.2 (+0.1) | 0.702 | 99.3 (−0.1) | 0.048 |
| 0.05 | 99.3 (+0.3) | 0.695 | 99.3 (−0.1) | 0.050 |
| 0.1 | 99.5 (+0.4) | 0.692 | 99.3 (−0.1) | 0.052 |
| 0.2 | 98.7 (−0.4) | 0.672 | 98.4 (−1.0) | 0.066 |
| 0.3 | 97.9 (−1.2) | 0.626 | 96.0 (−3.5) | 0.085 |
| 0.5 | **90.6 (−8.5)** | 0.536 | 90.8 (−8.7) | 0.153 |

Lưới mịn (chỉ để tham khảo, không dùng để chọn mức):

| | Spread | Line |
|---|---|---|
| σ_v = 0.055 | 96.2 | 97.1 |
| σ_v = 0.06 | 94.9 | 95.6 |
| σ_v = 0.065 | 93.8 | 95.2 |
| σ_v = 0.07 | 91.4 | 92.4 |
| σ_w = 0.35 | 95.2 | 95.2 |
| σ_w = 0.4 | 94.5 | 94.3 |
| σ_w = 0.45 | 92.6 | 91.8 |

Điểm gãy thực (safe_agent < 95%) nằm ở khoảng σ_v ≈ 0.06–0.065 (≈1.2–1.3r) và σ_w ≈ 0.4.

### 4.3 Mức đã chọn (áp quy tắc mục 2 lên LidarSpread)

| | low | mid | high |
|---|---|---|---|
| σ_v | 0.0125 | 0.0375 | 0.075 |
| σ_w | 0.1 | 0.2 | 0.5 |

**Phá hoà, bổ sung sau khi thấy kết quả:** quy tắc mục 2 không nói gì khi hai điểm lưới cách đều mục tiêu. Có hai trường hợp như vậy:
- low của σ_v: high/4 = 0.01875, cách đều 0.0125 và 0.025;
- mid của σ_w: high/2 = 0.25, cách đều 0.2 và 0.3.

Đã chọn σ **nhỏ hơn**. Lựa chọn này bảo thủ: nó làm H1 khó được xác nhận hơn.

LidarLine cho cùng điểm gãy (σ_v = 0.075 → 90.5%, σ_w = 0.5 → 90.8%), nên các mức này dùng chung cho cả hai env.

### 4.4 Nhận xét

1. **Suy giảm an toàn rất phi tuyến.**
   - Đến σ_v = r, DGPPO gần như không bị ảnh hưởng (≥ 97.9%), rồi sụt mạnh trong khoảng r → 1.5r.
   - Ở σ_v vừa phải, vi phạm chủ yếu là va vật cản (Spread σ_v = 1.5r: 8 va agent / 77 va vật cản). Từ khoảng 3r trở lên, va agent–agent chiếm ưu thế.
2. **Nhiễu động học σ_w ≤ 0.2 gần như không ảnh hưởng.**
   - Mỗi bước, controller đổi được vận tốc tối đa 10·1·0.03 = 0.3, nên bù được nhiễu nhỏ.
   - Riêng seed0 Spread: σ_w lớn làm agent đến gần goal hơn (dist2goal 0.70 → 0.54 ở σ_w = 0.5). Policy seed0 lúc sạch có vẻ bị kẹt, và nhiễu đẩy nó ra.
   - **Hiện tượng này không đúng khi gộp 3 seed:** σ_w lớn làm dist2goal tăng (0.267 → 0.307) và task cost xấu đi. Xem `docs/T2_RESULTS.md` mục 5.5.
3. **Hệ quả cho H1:** với quy tắc "mid = high/2", ở mức mid DGPPO chỉ thay đổi từ +0.0 đến −1.0 điểm phần trăm.
   - Tiêu chí chính (≥ 5 pp ở mid) gần như chắc chắn **không** thỏa. Đây là kết quả của quy tắc đã chốt trước, không chỉnh lại.
   - Nếu nhóm muốn định nghĩa mức theo hiệu ứng (ví dụ mức gây suy giảm 1 / 5 / 10 pp), cần quyết định rõ ràng và ghi là thay đổi sau hiệu chỉnh.

### 4.5 Vấn đề kỹ thuật phát hiện trong lúc hiệu chỉnh

**Kết quả không tất định trên cả GPU lẫn CPU** (XLA, JAX 0.6.2):
- **Không phải do khởi tạo ngẫu nhiên.** Đã kiểm chứng (`/home/mantd/DGPPO/t2_runs/logs/init_check.log`):
  - tham số policy nạp từ file `.pkl` (có sha256 trong `run.yaml`);
  - trạng thái GRU ban đầu toàn số 0;
  - graph ban đầu (vị trí agent, goal, vật cản sinh từ khóa) giống hệt giữa các lần chạy;
  - PRNG của JAX là tất định.
- **Nguồn gốc là sai số làm tròn của các phép cộng dồn song song.** Thứ tự cộng thay đổi giữa các lần chạy:
  - scatter-add trong `GraphsTuple.type_states` và `jraph.segment_sum`, dùng atomic trên GPU;
  - các phép cộng đa luồng trên CPU.
- **Sai lệch ban đầu rất nhỏ** (CPU khoảng 1e-8 đến 1e-6). Nhưng khi chạm một ngưỡng rời rạc thì nhảy vọt: chọn 8 tia LiDAR gần nhất, mask bán kính liên lạc, clip. Ví dụ CPU: cùng một episode lặp 8 lần cho 2 kiểu kết quả, tách nhau ở bước 4 với Δaction = 3.9e-3. Từ đó sai lệch khuếch đại theo thời gian.
- **Đã thử và không khắc phục được:**
  - ép CPU chạy 1 luồng;
  - tắt autotune GPU (`--xla_gpu_autotune_level=0`);
  - `--xla_gpu_deterministic_ops=true` còn cho kết quả **sai** (DGPPO Spread σ = 0: 47.9% thay vì 100%; từng phép đơn lẻ vẫn đúng, chỉ sai khi chạy cả rollout policy trong `lax.scan`). `eval_robust.py` từ chối chạy nếu cờ này được bật.
- **Mức ảnh hưởng:** ở σ = 0 (256 episode × 2 env), khoảng 2/512 episode đổi kết quả an toàn giữa hai lần chạy, safe_agent dao động khoảng ±0.3 pp. Nhiễu này không thiên lệch, nên số liệu tái lập ở mức thống kê, không trùng từng bit. Mọi lần chạy đều có hồ sơ để truy ngược.

**Lỗi bên T1:** cột `cost` trong `scripts/t1/pilot_results.csv` không phải cost trung bình mà là **min** trên các episode. `parse_test_output` trong `run_baseline.py` dùng `.*` tham lam nên bắt nhầm số sau chữ "min/max cost:". Ví dụ DGPPO Spread: ghi −0.649, trung bình thật là −0.561. Cần báo A.

## 5. So với config gốc của T1: những gì đổi và giữ nguyên

Khi đánh giá, T2 **không đổi tham số nào của policy hay môi trường ngoài nhiễu σ_w, σ_v**. Riêng các lượt quét dịch chuyển sau này sẽ cố ý đổi N hoặc số vật cản; khi đó `run.yaml` ghi lại trong `env.differs_from_train`. Mọi giá trị dưới đây đều được ghi trong `run.yaml` của từng run.

| Hạng mục | Lúc train / T1 | T2 khi đánh giá | Giống? |
|---|---|---|---|
| Trọng số policy | checkpoint step 200000 | cùng file `.pkl` (sha256 trong `run.yaml`), không train lại | giống |
| Kiến trúc (GNN layers, GRU, …) | `config.yaml` | đọc từ `config.yaml` giống hệt `test.py` | giống |
| env: id, num_agents, n_obs | LidarSpread/LidarLine, 3, 3 | 3, 3 (từ `config.yaml`) | giống |
| env: n_rays, full_observation | 32, false | 32, false (từ `config.yaml`) | giống |
| env: max_step, dt, area_size, car_radius, comm_radius, top_k_rays, dist2goal, obs_len_range | 128, 0.03, 1.5, 0.05, 0.5, 8, 0.01, [0.1, 0.3] (mặc định) | giống (mặc định, không truyền gì khác) | giống |
| Cách chọn action | `test.py`: tất định (mode của phân phối) | tất định | giống |
| Khóa episode | `test.py --seed 1234` | `jr.split(PRNGKey(1234), 1000)[:epi]`, tách `key_x0` như `test.py` | giống |
| Định nghĩa vi phạm / safe rate | `test.py`: cost ≥ 0, theo agent | giống, tính trên trạng thái thật | giống |
| Số episode đánh giá | T1 báo 32 | 256 (32 episode đầu trùng T1) | khác cỡ mẫu, không phải tham số |
| Cách chạy | tuần tự từng episode, RTX 5090 | vmap 32 episode/lần, RTX 3080 | chỉ khác sai số làm tròn |
| Nhiễu | không có | σ_w, σ_v theo lưới | **chủ đích thay đổi** |

**Ghi chú:**
- Trước khi có hệ thống hồ sơ (các lượt trong `archive_no_record/`), env lấy `n_rays` và `full_observation` theo giá trị mặc định thay vì đọc từ `config.yaml`. Hai giá trị này trùng nhau (32, false), nên kết quả không bị ảnh hưởng.
- Bộ test (`tests/`) dùng `max_step = 32` và một policy InforMARL khởi tạo ngẫu nhiên chỉ để kiểm tra code. Chúng không tạo ra số liệu báo cáo.
