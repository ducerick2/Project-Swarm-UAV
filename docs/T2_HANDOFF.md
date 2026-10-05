# Mô tả công việc: từ T1 (tái hiện baseline) đến T2 (đánh giá độ bền vững)

| | |
|---|---|
| Dự án | Project-Swarm-UAV: học tăng cường đa agent an toàn cho swarm UAV dưới bất định |
| Mạch được mô tả | T1 `t1-reproduce` (thành viên A) và T2 `t2-robustness` (thành viên B) |
| Cập nhật | 05/10/2026 |
| Tài liệu liên quan | T1: `scripts/t1/README.md`, `scripts/t1/pilot_results.md` (nhánh `t1-reproduce`). T2: `docs/T2_RESULTS.md`, `docs/T2_calibration.md`, `docs/T2_setup_and_usage.md` |

---

## 1. Bối cảnh dự án

Dự án nghiên cứu học tăng cường đa agent an toàn (Safe MARL) cho đội UAV khi môi trường có bất định.

- **Phương pháp nền:** DGPPO (ICLR 2025, MIT-REALM). DGPPO huấn luyện chính sách điều khiển kèm một hàm rào chắn rời rạc trên đồ thị (discrete graph CBF) để các agent tránh va chạm.
- **Mã nguồn DGPPO:** nhúng vào repo dưới dạng submodule `third_party/dgppo`, ghim tại commit `51b3b11`.
- **Đóng góp đề xuất:** CM-DGPPO, dùng biên an toàn thích nghi h̃ = h + δ, trong đó δ được cập nhật theo tần suất vi phạm thực tế (kiểu Adaptive Conformal Inference).

**Phân công:**

| Mạch | Nhánh | Thành viên | Nội dung |
|---|---|---|---|
| T1 | `t1-reproduce` | A | Tái hiện 3 baseline (DGPPO, InforMARL, InforMARL-Lagrangian), đo thời gian chạy |
| T2 | `t2-robustness` | B | Đánh giá độ bền vững của DGPPO dưới nhiễu và dịch chuyển phân phối, kiểm định H1 |
| T3 | `t3-cm-dgppo` | C | CM-DGPPO: biên an toàn thích nghi và các biến thể (a)–(e) |
| T4 | `t4-theory-stats` | D | Chứng minh cận lý thuyết, công cụ thống kê |
| T5 | `t5-casestudy` | E | Môi trường CEFC-lite, demo, tổng quan |

Nền tảng chung (giai đoạn 0, trên `main`) gồm: lược đồ cấu hình `configs/`, lớp bao nhiễu `envs/noise_wrapper.py`, định dạng kết quả `analysis/logging_csv.py`, danh sách seed `analysis/seeds.py`.

Theo thoả thuận nhóm, **T1 huấn luyện toàn bộ baseline; T2 chỉ dùng các checkpoint của T1** và không huấn luyện lại.

---

## 2. Công việc đã thực hiện ở T1

### 2.1 Mục tiêu

Tái hiện ba phương pháp baseline của bài DGPPO trên hai môi trường LiDAR, lấy số liệu an toàn và hiệu năng để đối chiếu với bài gốc. Ba phương pháp:
- **DGPPO:** có hàm rào chắn;
- **InforMARL-Lagrangian:** ràng buộc mềm qua nhân tử Lagrange;
- **InforMARL:** không có ràng buộc an toàn.

Các checkpoint thu được là đầu vào cho các mạch sau.

### 2.2 Hạ tầng

- **Huấn luyện:** dùng nguyên mã gốc của tác giả (`third_party/dgppo/train.py`, `test.py`), không viết lại vòng lặp huấn luyện. `scripts/t1/run_baseline.py` bọc quy trình: huấn luyện → đánh giá → ghi một dòng CSV.
- **Tương thích phần cứng:** máy huấn luyện dùng RTX 5090 (kiến trúc Blackwell), không chạy được bản JAX mà DGPPO ghim gốc. T1 chuyển sang JAX 0.6.2 và viết shim `scripts/compat/sitecustomize.py` để:
  - khôi phục các hàm `jax.tree_*` mà JAX 0.6 đã bỏ;
  - trỏ matplotlib tới `imageio-ffmpeg` để xuất video.

  Shim này không sửa mã submodule.
- **Script phụ trợ:**
  - `run_pilot.sh`, `run_seeds.sh`: chạy hàng loạt;
  - `monitor_check.sh`: theo dõi tiến độ;
  - `eval_ckpt.sh`: đánh giá lại một checkpoint, có thể xuất video;
  - `coverage_check.py`: chẩn đoán mức phủ mục tiêu của một episode;
  - `collect_best_ckpts.sh`: gom checkpoint cuối để commit.

### 2.3 Thiết lập thực nghiệm

| Hạng mục | Giá trị |
|---|---|
| Phương pháp | DGPPO, InforMARL, InforMARL-Lagrangian |
| Môi trường | `LidarSpread` (phân tán tới mục tiêu), `LidarLine` (xếp hàng giữa hai mốc) |
| Cấu hình môi trường | 3 agent, 3 vật cản, vùng 1,5 × 1,5, bán kính agent 0,05, LiDAR 32 tia, 128 bước/episode |
| Số bước huấn luyện | 200 000 |
| Tham số chính | batch 16 384; 128 môi trường song song; lr actor 3·10⁻⁴, lr critic 10⁻³; chính sách có GRU; DGPPO: α = 10, cbf_eps = 0,01; InforMARL-Lag: lr_lagr = 10⁻⁷ (λ gần như cố định) |
| Đánh giá | `test.py`, 32 episode, test seed 1234, chính sách tất định |
| Phần cứng | 1 × RTX 5090, khoảng 8 vòng lặp/giây, khoảng 5–7 giờ mỗi lần huấn luyện |

### 2.4 Kết quả (đợt pilot, seed 0)

| Phương pháp | Môi trường | Tỉ lệ an toàn | Phần thưởng | Thời gian huấn luyện |
|---|---|---|---|---|
| DGPPO | LidarSpread | 1,000 | −0,885 | 7,1 giờ |
| DGPPO | LidarLine | 0,990 | −0,280 | 6,8 giờ |
| InforMARL-Lag | LidarSpread | 0,844 | −0,189 | 6,8 giờ |
| InforMARL-Lag | LidarLine | 0,865 | −0,168 | 6,6 giờ |
| InforMARL | LidarSpread | 0,656 | −0,217 | 5,2 giờ |
| InforMARL | LidarLine | 0,771 | −0,190 | 5,2 giờ |

**Nhận xét của T1**, khớp xu hướng bài gốc:
- Thứ tự an toàn nhất quán trên cả hai môi trường: DGPPO (99–100%) > InforMARL-Lagrangian (84–86%) > InforMARL (66–77%).
- Phương pháp an toàn hơn có phần thưởng thấp hơn: có đánh đổi giữa an toàn và hiệu năng.
- DGPPO và InforMARL-Lag chạy chậm hơn InforMARL (khoảng 7 giờ so với khoảng 5 giờ), do phải huấn luyện thêm critic hoặc hàm rào chắn.

### 2.5 Sản phẩm bàn giao

- **Checkpoint cuối** (bước 200 000) của **18 lần huấn luyện** (3 phương pháp × 2 môi trường × seed 0, 1, 2), kèm `config.yaml`, tại `results/checkpoints/<env>/<method>/seed<k>/` trên nhánh `t1-reproduce`. Mỗi checkpoint gồm `actor.pkl`, `Vl.pkl`; DGPPO và InforMARL-Lag có thêm `Vh.pkl`.
- **Số liệu pilot:** `scripts/t1/pilot_results.csv` / `.md`, chỉ seed 0.
- **Script và hướng dẫn dựng môi trường:** `scripts/t1/README.md`.

### 2.6 Lưu ý về số liệu T1

Phát hiện khi T2 kiểm tra lại:

| Vấn đề | Chi tiết | Ảnh hưởng |
|---|---|---|
| Định nghĩa tỉ lệ an toàn | `pilot_results.md` ghi là "tỉ lệ quỹ đạo không vi phạm", nhưng `test.py` tính **theo agent**: tỉ lệ agent không vi phạm. Ví dụ 0,98958 = 95/96, tức 32 episode × 3 agent. | Chỉ là nhãn; số liệu đúng theo định nghĩa của DGPPO |
| Cột `cost` | Là giá trị ràng buộc h, không phải chi phí nhiệm vụ. Ngoài ra, do lỗi đọc kết quả trong `run_baseline.py` (biểu thức chính quy bắt nhầm), cột này ghi **giá trị nhỏ nhất** thay vì trung bình. Ví dụ DGPPO LidarSpread: ghi −0,649, trung bình thật là −0,56. | Cần sửa nếu dùng cột `cost`; tỉ lệ an toàn và phần thưởng không bị ảnh hưởng |
| `test.py --stochastic` | Lỗi trong mã DGPPO gốc: hàm hành động 4 tham số bị gọi với 3 tham số, nên chương trình dừng. | Không đánh giá được chính sách ngẫu nhiên bằng `test.py` |
| Số seed | Pilot chỉ báo seed 0. Checkpoint seed 1–2 có trong repo nhưng chưa có số liệu đánh giá chính thức của T1; seed 3–4 chưa có trong repo. | T2 tự đánh giá seed 0–2 |

---

## 3. Công việc của T2

### 3.1 Mục tiêu và câu hỏi nghiên cứu

**Câu hỏi:** chính sách DGPPO huấn luyện trong điều kiện sạch còn an toàn đến đâu khi điều kiện triển khai khác lúc huấn luyện? Nó bắt đầu mất an toàn ở mức sai lệch nào, và vì sao?

**Giả thuyết H1:** khi cường độ nhiễu động học (σ_w) và nhiễu cảm biến (σ_v) tăng, tỉ lệ an toàn của DGPPO giảm rõ rệt so với khi không có nhiễu. Tiêu chí kiểm định được chốt trước khi chạy thí nghiệm:

> Ở mức nhiễu "vừa", tỉ lệ an toàn giảm ít nhất 5 điểm phần trăm, kiểm định một phía p < 0,05, trên cả hai môi trường.

### 3.2 Phạm vi và nguyên tắc

- **Chỉ đánh giá DGPPO.** InforMARL và InforMARL-Lagrangian thuộc phạm vi T1.
- **Không huấn luyện lại, không đổi cấu hình DGPPO.** Dùng nguyên checkpoint của T1 (seed 0, 1, 2) và chỉ thay đổi điều kiện môi trường lúc triển khai.
- **Không sửa mã DGPPO** trong submodule.
- **Mọi lần chạy đều lưu đủ cấu hình và kết quả** để truy ngược được từng con số.
- **Quyết định thiết kế có nhiều lựa chọn** (cách đặt mức nhiễu, định nghĩa dịch chuyển) được thống nhất với trưởng mạch trước khi thực hiện.

### 3.3 Các loại sai lệch đánh giá

| Loại sai lệch | Ý nghĩa thực tế | Cách mô phỏng |
|---|---|---|
| Nhiễu cảm biến σ_v | sai số định vị (GPS, LiDAR) | cộng nhiễu Gauss vào vị trí các agent và vị trí các điểm LiDAR mà chính sách quan sát; trạng thái thật không đổi |
| Nhiễu động học σ_w | gió, sai số động cơ | cộng nhiễu Gauss vào vận tốc thật của agent sau mỗi bước |
| Số vật cản | môi trường đông chướng ngại vật hơn | 5 hoặc 8 vật cản (huấn luyện với 3) |
| Số agent | đội hình lớn hơn | 5 hoặc 7 agent (huấn luyện với 3), giữ nguyên diện tích |

### 3.4 Nội dung công việc và trạng thái

| # | Công việc | Mô tả | Trạng thái |
|---|---|---|---|
| 1 | Dựng môi trường, kiểm tra quy trình | Venv JAX 0.6.2 trên RTX 3080. Chạy lại đánh giá ở điều kiện gốc trên DGPPO seed 0 và đối chiếu với số liệu T1. | Hoàn thành: khớp tỉ lệ an toàn, phần thưởng lệch ≤ 0,001 |
| 2 | Lớp bao nhiễu | Viết lại `envs/noise_wrapper.py` thành hàm rollout có nhiễu thuần JAX (môi trường DGPPO không theo API kiểu gym). Giữ riêng "đồ thị thật" (dùng tính vi phạm) và "đồ thị quan sát" (đưa cho chính sách). Có test: σ = 0 cho kết quả như rollout gốc; độ lệch chuẩn nhiễu đúng bằng σ. | Hoàn thành |
| 3 | Công cụ đánh giá và lưu hồ sơ | `scripts/t2/eval_robust.py` đánh giá một checkpoint dưới các mức nhiễu hoặc dịch chuyển, ghi CSV mỗi episode một dòng, và tạo hồ sơ chạy đầy đủ (cấu hình, mã băm checkpoint, tham số môi trường, phiên bản thư viện, kết quả). Kèm script quét hàng loạt, tổng hợp bảng và kiểm định H1. | Hoàn thành |
| 4 | Hiệu chỉnh mức nhiễu | Quét lưới rộng σ_v (0–4r) và σ_w (0–0,5) trên DGPPO seed 0. Chọn mức thấp / vừa / cao theo quy tắc ghi trước: "cao" là mức đầu tiên làm an toàn dưới 95%. | Hoàn thành: σ_v = 0,25r / 0,75r / 1,5r; σ_w = 0,1 / 0,2 / 0,5 |
| 5 | Quét nhiễu và kiểm định H1 | DGPPO × 2 môi trường × seed 0–2 × toàn bộ lưới σ × 256 episode; so ghép cặp theo episode. | Hoàn thành |
| 6 | Quét dịch chuyển | Số agent 3 / 5 / 7 và số vật cản 3 / 5 / 8; chính sách tất định và ngẫu nhiên; seed 0–2; 256 episode. | Hoàn thành |
| 7 | Báo cáo | `docs/T2_RESULTS.md`: phương pháp, kết quả, thảo luận, hạn chế, bảng số liệu đầy đủ. | Hoàn thành |
| 8 | Chẩn đoán hàm rào chắn | Nạp `Vh.pkl`, đo tỉ lệ bước vi phạm điều kiện DGCBF và tỉ lệ bước V^h ≤ 0 nhưng vẫn va chạm. | Chưa làm |
| 9 | Đo vi phạm theo hình học thật | Hàm `true_violation` dùng hình chữ nhật vật cản thay cho 8 tia LiDAR. Hàm này cũng là tín hiệu vi phạm cho T3. | Chưa làm |
| 10 | Mở rộng thí nghiệm | Kết hợp nhiều loại sai lệch; tăng số agent kèm tăng diện tích (mật độ không đổi); nhiễu cảm biến độc lập theo từng agent; thêm seed. | Chưa làm |
| 11 | Bàn giao cho các mạch khác | PR `t2-robustness` vào `main`. Gửi T3 chữ ký hàm rollout có nhiễu và các mức nhiễu làm mốc so sánh; gửi T4 dữ liệu CSV; gửi T5 cách bọc môi trường CEFC-lite. | Chưa làm (chờ duyệt) |

### 3.5 Kết quả chính

Chi tiết ở `docs/T2_RESULTS.md`.

- Ở điều kiện huấn luyện, DGPPO an toàn 99,5%.
- An toàn được giữ khi sai lệch nhỏ, rồi suy giảm nhanh khi vượt ngưỡng:
  - nhiễu cảm biến vị trí: ngưỡng khoảng 1 bán kính agent;
  - nhiễu vận tốc: ngưỡng khoảng 0,3.
- Tăng số agent gây hại nhất: 74,6–85,5% với 7 agent. Tăng số vật cản chỉ làm giảm nhẹ.
- H1 không được xác nhận ở mức "vừa" theo tiêu chí đặt trước (chỉ giảm 0,8–1,8 điểm phần trăm), nhưng ở mức "cao" tỉ lệ an toàn giảm 11–14 điểm phần trăm.

### 3.6 Sản phẩm của T2

| Loại | Vị trí |
|---|---|
| Mã nguồn | `envs/noise_wrapper.py`; `analysis/robust_metrics.py`, `robust_csv.py`, `robust_summary.py`, `run_record.py`; `scripts/t2/` |
| Cấu hình | `configs/t2/` (lưới nhiễu, các mức đã hiệu chỉnh); khối `noise` / `eval` trong `configs/example.yaml`, `configs/schema.md` |
| Kiểm thử | `tests/test_noise_wrapper.py`, `tests/test_robust_utils.py` |
| Tài liệu | `docs/T2_RESULTS.md` (báo cáo), `docs/T2_calibration.md` (hiệu chỉnh, tiêu chí H1), `docs/T2_setup_and_usage.md` (cài đặt, cách chạy) |
| Hồ sơ chạy | ngoài repo (hơn 1000 file), máy RTX 3080: `/home/mantd/DGPPO/t2_runs/` |

---

## 4. Quy ước làm việc chung

- Không sửa mã trong `third_party/dgppo`. Mọi điều chỉnh tương thích đặt ngoài submodule.
- Mỗi mạch làm trên nhánh riêng, mở PR vào `main` và có người phản biện. Không đẩy thẳng lên `main`. Cập nhật từ `main` hằng ngày (`git pull --rebase origin main`).
- Không commit kết quả thô. Ngoại lệ duy nhất là checkpoint cuối của T1.
- Mọi lần chạy ghi lại seed, phiên bản mã, cấu hình. Không sửa tay số liệu trong bảng.
- Thay đổi các thành phần dùng chung (lược đồ cấu hình, định dạng CSV) phải được cả nhóm thống nhất qua PR.
