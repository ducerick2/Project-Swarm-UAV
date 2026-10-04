# CM-DGPPO: tham số và lý giải (phục vụ viết báo cáo)

Mọi con số dưới đây là giá trị **đang dùng** trong `configs/t3/` và code `methods/cm_dgppo/`.
Cột "Nguồn" ghi rõ tham số lấy từ tài liệu hay do nhóm chọn, để báo cáo trình bày đúng.

## 1. Ký hiệu

| Ký hiệu | Nghĩa |
|---|---|
| r | bán kính UAV (= `car_radius` của DGPPO) |
| h | hàm ràng buộc (đơn vị khoảng cách): h₁ = 2r − khoảng cách tới UAV gần nhất, h₂ = r − khoảng cách tới điểm LiDAR gần nhất; va chạm khi h > 0 |
| ε | khoảng nới ràng buộc: train với h − ε ≤ 0 (⇔ khoảng cách ≥ 2r − ε và ≥ r − ε) |
| e_t | tỉ lệ cặp (UAV, episode) có va chạm thật trong lô ở bước train t |
| α | mức va chạm chấp nhận (tỉ lệ cặp UAV–episode) |
| η | bước cập nhật ε |
| T | số bước train (= số lần cập nhật chính sách) |

Cập nhật của phương pháp đề xuất (`relax`):

```
ε_{t+1} = clip( ε_t + η (α − e_t),  0,  ε_max )
```

## 2. Bảng tổng hợp

| Tham số | Giá trị | Nguồn | Mục |
|---|---|---|---|
| α | 0.05 | đề cương (`docs/scripts.tex`) + quy ước 95% | 3.1 |
| ε_0 | 0 | thiết kế (bắt đầu từ DGPPO) | 3.2 |
| ε_max | 0.05 (= r) | thiết kế, theo ý nghĩa vật lý | 3.3 |
| η | ε_max / (α τ), τ đo riêng từng env: **1.639·10⁻⁵** (LidarSpread, τ = 61 000), **4.545·10⁻⁵** (LidarLine, τ = 22 000) | công thức, τ đo từ DGPPO T1 | 3.4 |
| Cách đo e_t | theo (UAV, episode), trên `det_rollout` | thiết kế | 3.5 |
| ε_fixed (baseline) | phân vị α của khoảng hở DGPPO trên tập validation: **0.020855** (LidarSpread), **0.016355** (LidarLine) | công thức, dữ liệu DGPPO T1 | 4 |
| Nhiễu σ_w, σ_v | 0, 0 | thiết kế (giống T1) | 5 |
| Cấu hình train và siêu tham số DGPPO | như T1 | code/paper DGPPO, config T1 | 5 |

## 3. Tham số của phương pháp đề xuất (`cm_dgppo_relax`)

### 3.1. α = 0.05: mức va chạm chấp nhận

- **Đề cương** đặt "giữ tần suất vi phạm quanh mức α (ví dụ 5%)", với phân tích độ nhạy dự kiến α ∈ {1%, 5%, 10%}.
- **Nằm giữa các baseline** (số liệu T1, không nhiễu, LidarSpread): InforMARL 0.66–0.74, InforMARL-Lagrangian 0.84–0.91, DGPPO ≈ 1.00. Với mục tiêu safety 1 − α = 0.95, CM-DGPPO vẫn an toàn hơn phương pháp Safe MARL dạng Lagrange, trong khi chừa ngân sách rủi ro để tăng reward so với DGPPO.
- **Ước lượng được**: mỗi lần eval có 256 episode × 3 UAV = 768 cặp, nên ở 5% có khoảng 38 va chạm, sai số chuẩn khoảng ±0.8%. Với α = 1% chỉ còn khoảng 8 va chạm: quá ít để ước lượng, và tín hiệu cập nhật ε quá thưa.
- **Quy ước phổ biến**: độ tin cậy/coverage 95% trong thống kê và conformal prediction; ràng buộc xác suất 5% trong điều khiển có ràng buộc ngẫu nhiên.
- Đo **theo từng UAV** để khớp chỉ số `safety_rate` (định nghĩa của DGPPO: tỉ lệ cặp UAV–episode không vi phạm).

### 3.2. ε_0 = 0: bắt đầu từ DGPPO

ε = 0 chính là DGPPO gốc (có test chứng minh trùng từng bit). Thuật toán chỉ nới ra khi dữ liệu cho thấy còn dư ngân sách rủi ro, tức e_t < α. Đầu quá trình train, chính sách còn va chạm nhiều nên ε giữ ở 0.

### 3.3. ε_max = 0.05 = r: trần của khoảng nới

| ε | UAV–UAV được sát tới | UAV–vật cản được sát tới |
|---|---|---|
| 0 | 0.10 (= 2r, vừa chạm) | 0.05 (= r) |
| 0.02 | 0.08 | 0.03 |
| 0.05 | 0.05 | 0.00 |

Trần bằng một bán kính là giới hạn vật lý tự nhiên. Ở mức này, với vật cản, tâm UAV đã được phép chạm bề mặt vật cản, nên không có lý do nới thêm. Đây là trần **rộng**; ACI sẽ tự dừng sớm hơn nếu va chạm vượt α. Đối chiếu với dữ liệu: phân vị 25% khoảng hở nhỏ nhất của DGPPO (LidarSpread) là 0.055, xấp xỉ r.

### 3.4. η = ε_max / (α τ): bước cập nhật theo tốc độ hội tụ của DGPPO

**Hai tốc độ hội tụ của DGPPO** (log train của T1, 6 run, không nhiễu):

| | Hội tụ khoảng | Cách đo |
|---|---|---|
| **An toàn** | LidarSpread 25k–149k (trung vị **τ = 61 000**); LidarLine 19k–25k (trung vị **τ = 22 000**) | bước đầu tiên t mà TB trượt 10 lần eval (10k bước) của tỉ lệ vi phạm ≤ α **tại t**, và ở dưới α ≥ 90% thời gian còn lại; đọc `eval/unsafe_frac` độ chính xác đầy đủ từ W&B của T1 |
| **Reward** | 33k–177k bước (trung vị ~100k) | bước đạt 90% quãng cải thiện reward cuối cùng |

τ từng seed: LidarSpread 149k / 61k / 25k, LidarLine 25k / 22k / 19k. Dùng **trung vị** vì một run
(LidarSpread seed 0) có các đợt vi phạm nhảy vọt muộn ở bước ~70k và ~130k. τ được đo **riêng cho
từng môi trường** vì hai env hội tụ rất khác nhau (LidarSpread chậm hơn ~3 lần).

**Suy dẫn.** Khi chưa có va chạm (e_t ≈ 0), mỗi bước ε tăng η·α, nên từ 0 lên ε_max mất
ε_max / (η α) bước. Đặt thời gian này bằng τ:

```
ε_max / (η α) = τ      ⇒      η = ε_max / (α τ)
    LidarSpread:  0.05 / (0.05 × 61 000) = 1.639·10⁻⁵
    LidarLine:    0.05 / (0.05 × 22 000) = 4.545·10⁻⁵
```

**Vì sao chọn theo τ của an toàn** (lập luận hai thang thời gian, như các phương pháp nhân tử
Lagrange học song song với chính sách): biến điều chỉnh ràng buộc nên thay đổi không nhanh hơn
chính sách hội tụ. ε phản ứng theo **va chạm**, nên thang thời gian liên quan là của an toàn:
- trong ~τ bước đầu, chính sách đang học tránh va chạm, ε chưa nới vội;
- ε đạt mức nới cần thiết vào khoảng τ đến 2τ (LidarSpread 61k–122k, LidarLine 22k–44k bước);
- phần còn lại của 200k bước dành cho chính sách tối ưu reward dưới mức nới mới, đúng phần reward hội tụ chậm.

Nếu dùng τ của reward (~100k–180k), ε chỉ đạt mức nới ở cuối quá trình train, và chính sách còn quá
ít thời gian khai thác nó.

**Cận dài hạn** (lập luận của Gibbs & Candès 2021, áp vào thang khoảng cách): cộng dồn công thức
cập nhật khi ε không bị kẹp ở 0 hay ε_max,

```
ε_T − ε_0 = η Σ_t (α − e_t)   ⇒   | (1/T') Σ_t e_t − α |  ≤  ε_max / (η T')  =  α τ / T'
```

với T' là độ dài đoạn mà ε **không bị kẹp**. Đầu quá trình train chính sách còn va chạm nhiều nên ε
nằm ở 0 (bị kẹp) — cận không áp dụng cho đoạn đó. Ví dụ nếu ε không bị kẹp trong 150k bước cuối:
α τ / T' ≈ 0.02 (LidarSpread), 0.007 (LidarLine). Cần báo cáo quỹ đạo ε (`cm/eps`) để biết T' thực tế.

**Đoạn văn cho báo cáo:**
> *"DGPPO hội tụ về an toàn sau τ ≈ 61 000 bước trên LidarSpread và 22 000 bước trên LidarLine
> (trung vị 3 lần train mỗi môi trường), trong khi reward tiếp tục cải thiện tới 100 000–200 000 bước. Ta chọn bước cập nhật η = ε_max/(α τ) để
> ngưỡng nới đi từ 0 lên ε_max trong đúng τ bước: ngưỡng chỉ nới sau khi chính sách đã học tránh
> va chạm, và phần lớn quá trình train được dành để tối ưu reward dưới mức nới đó. Khi đó tần
> suất va chạm dài hạn (trên đoạn ε không bị kẹp, độ dài T') lệch α không quá α τ / T'."*

So sánh với lựa chọn theo paper ACI (γ = 0.005 đổi thang, η = γ·ε_max = 0.00025): ε lên trần sau
4 000 bước, tức trước cả khi DGPPO học xong cách tránh va chạm (τ = 22k–61k). Vì vậy chọn theo τ đo được.

Tái tạo: `python scripts/calib_fixed_eps.py --tau-only` (đọc log của T1, không cần GPU).
Code: `cm_eta` trong `configs/t3/cm_dgppo_relax.yaml`.

### 3.5. Cách đo e_t

- **Theo cặp (UAV, episode)**: một cặp tính là vi phạm nếu UAV đó có ít nhất một bước h > 0 (với h₁ hoặc h₂) trong episode. Va chạm UAV–UAV tính cho cả hai UAV.
- **Trên `det_rollout`** (128 episode, chính sách deterministic): đúng chính sách được dùng khi eval.
- **Dùng h thật**, không có ε, tức đếm va chạm thật.
- ε dùng để tạo cost của lô t là ε_t có **trước** khi thấy lô t (đúng thứ tự cập nhật online).

## 4. Baseline ngưỡng cố định (`fixed_relax`): ε_fixed theo công thức

**Công thức** (phân vị α có hiệu chỉnh mẫu hữu hạn, dạng chuẩn trong split conformal prediction):

```
c_1 ≤ c_2 ≤ … ≤ c_n   khoảng hở nhỏ nhất của mỗi cặp (UAV, episode) dưới DGPPO
                      (lấy loại sát hơn trong UAV–UAV và UAV–vật cản; c < 0 là va chạm)
k = ⌊α (n + 1)⌋
ε_fixed = c_k
```

**Dữ liệu**: checkpoint DGPPO của T1 tại 200k bước, 3 seed × 256 episode × 3 UAV, n = 2304, k = ⌊0.05 × 2305⌋ = 115 (tập validation, eval-seed 20000).

| Môi trường | ε_fixed = c_115 | Va chạm hiện tại của DGPPO | Config |
|---|---|---|---|
| LidarSpread | **0.020855** | 0.39% | `fixed_relax.yaml` |
| LidarLine | **0.016355** | 0.78% | `fixed_relax_lidarline.yaml` |

Hiệu chỉnh trên **tập validation** (`--eval-seed 20000`), tách khỏi tập test (`10000`) dùng cho bảng
kết quả, để ngưỡng của baseline không được chọn bằng chính các episode dùng để chấm.

**Ý nghĩa**: nới thêm ε cho phép UAV tiến sát hơn ε. Những lần bay mà DGPPO vốn chỉ còn khoảng hở dưới ε sẽ có nguy cơ chạm. Chọn ε là phân vị α thì khoảng α số cặp chịu nguy cơ này, tức **cùng mức rủi ro α** mà `relax` nhắm tới.

**Vì sao không dùng trung bình của h**: phần lớn thời gian các UAV ở xa nhau, nên trung bình h trên mọi bước rất âm (−0.44 với UAV–UAV) và không phản ánh va chạm. Ngay cả trung bình hay trung vị của *khoảng hở nhỏ nhất* (0.118 / 0.092) cũng quá lớn: nới chừng đó thì khoảng một nửa số cặp sẽ chạm.

**Thống kê khoảng hở nhỏ nhất của DGPPO (LidarSpread, gộp hai loại ràng buộc; đo trên tập 10000, chỉ để minh họa):**

| Trung bình | Trung vị | p25 | p10 | p5 |
|---|---|---|---|---|
| 0.118 | 0.092 | 0.055 | 0.033 | 0.022 |

**Đóng khung cho báo cáo**: `fixed` là **hiệu chỉnh conformal offline, một lần** trên dữ liệu của DGPPO; `relax` là **hiệu chỉnh online** (ACI) ngay trong lúc train, theo dõi trực tiếp tỉ lệ va chạm của chính sách đang học. Giả định của `fixed` là chính sách mới chỉ "dịch sát lại" đúng ε so với DGPPO. Thực tế chính sách được train lại nên giả định này chỉ gần đúng, và đó là động cơ để dùng `relax`.

Tái tạo: `python scripts/calib_fixed_eps.py --alpha 0.05`.

## 5. Môi trường và cấu hình train (giống T1 để so sánh trực tiếp)

| Tham số | Giá trị | Nguồn |
|---|---|---|
| Môi trường | LidarSpread, LidarLine | benchmark DGPPO |
| N (số UAV), n_obs (số vật cản) | 3, 3 | thiết lập chính của DGPPO / T1 |
| r | 0.05 | DGPPO |
| Nhiễu σ_w, σ_v | 0, 0 | giống T1 (đánh giá trong điều kiện bài gốc) |
| T (số bước train) | 200 000 | T1 |
| n_env_train / batch_size | 128 / 16 384 | mặc định DGPPO, T1 |
| eval_interval / save_interval / n_env_test | 1 000 / 10 000 / 32 | T1 |
| GNN layers (actor / V_l / V_h) | 2 / 2 / 1 | mặc định DGPPO |
| lr actor / V_l / V_h | 3e-4 / 1e-3 / 1e-3 | mặc định DGPPO |
| γ, GAE λ, clip, entropy, max grad norm | 0.99, 0.95, 0.25, 0.01, 2.0 | mặc định DGPPO |
| CBF: α (class-κ), ε_cbf, trọng số, lịch | 10, 0.01, 1.0, có lịch | mặc định DGPPO |
| RNN | GRU, 1 lớp, chunk 16 | mặc định DGPPO |
| Seed | 0 (1 seed mỗi phương pháp × môi trường) | giới hạn thời gian |

Khác biệt duy nhất so với DGPPO của T1 là cost an toàn dùng để train: h − ε thay cho h.

## 6. Giao thức đánh giá

| Mục | Thiết lập |
|---|---|
| Chính sách | deterministic, checkpoint cuối (200k bước) |
| Số episode | 256 mỗi run, key cố định `--eval-seed 10000` cho **mọi** phương pháp (cùng tập episode, so sánh cặp) |
| Nhiễu khi eval | 0 |
| Baseline | DGPPO của T1 (3 seed), cùng 256 episode |
| `task_cost` | −tổng reward mỗi episode, trung bình (thấp hơn = tốt hơn) |
| `safety_rate` | tỉ lệ cặp (UAV, episode) không va chạm (định nghĩa DGPPO) |
| `safety_rate_swarm` | tỉ lệ episode không có UAV nào va chạm |
| `clearance_mean`, `clearance_cvar5` | khoảng hở nhỏ nhất mỗi episode: trung bình và trung bình 5% thấp nhất |
| Khi train (W&B) | `cm/eps` (ε), `cm/viol_rate` (e_t), `cm/margin` |

**Tiêu chí thành công**: `relax` có task cost thấp hơn DGPPO trong khi `safety_rate ≥ 1 − α = 0.95`, và đạt được điều đó mà không cần chọn ε trước như `fixed`.

## 7. Giới hạn cần nêu

- Nới ràng buộc nghĩa là **chắc chắn có va chạm thật**; α là mức chấp nhận, không phải 0.
- Cận ở 3.4 chỉ đúng khi ε không bị kẹp ở 0 hay ε_max suốt quá trình; cần báo cáo quỹ đạo ε.
- 1 seed mỗi cấu hình: chưa đủ cho kiểm định thống kê; kết quả mang tính minh họa xu hướng.
- ε_max chọn theo lập luận vật lý, τ đo từ 6 run (dao động lớn: 9k–144k) và định nghĩa "hội tụ" là một lựa chọn; nếu còn thời gian nên thêm độ nhạy (ví dụ η × {0.5, 2}).
- Bài DGPPO cho rằng chi phí của nó đã gần baseline không ràng buộc, nên mức tăng reward có thể nhỏ.

## Tài liệu

- S. Zhang, O. So, M. Black, C. Fan. *Discrete GCBF Proximal Policy Optimization for Multi-agent Safe Optimal Control.* ICLR 2025.
- I. Gibbs, E. Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021.
- J. Lei, M. G'Sell, A. Rinaldo, R. Tibshirani, L. Wasserman. *Distribution-Free Predictive Inference for Regression.* JASA 2018 (phân vị có hiệu chỉnh mẫu hữu hạn trong split conformal).
