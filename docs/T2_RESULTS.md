# Báo cáo T2: Độ bền vững an toàn của DGPPO khi điều kiện triển khai khác điều kiện huấn luyện

| | |
|---|---|
| Mạch công việc | T2 — Đánh giá độ bền vững (nhánh `t2-robustness`) |
| Phương pháp được đánh giá | DGPPO (ICLR 2025) |
| Ngày thực hiện | 05/10/2026 |
| Tài liệu liên quan | `docs/T2_calibration.md` (hiệu chỉnh mức nhiễu, tiêu chí H1), `docs/T2_setup_and_usage.md` (cài đặt, cách chạy) |

## Tóm tắt

Báo cáo đánh giá mức độ an toàn của chính sách DGPPO khi được triển khai trong điều kiện khác với lúc huấn luyện. Chính sách được huấn luyện với 3 agent, 3 vật cản và không có nhiễu. Khi triển khai, bốn loại sai lệch được thử: nhiễu cảm biến vị trí, nhiễu vận tốc, tăng số vật cản và tăng số agent. Chính sách và môi trường được giữ nguyên.

Ở điều kiện như lúc huấn luyện, DGPPO đạt tỉ lệ an toàn 99,5%. Kết quả cho thấy:
- Mức an toàn này được giữ khi sai lệch nhỏ.
- An toàn suy giảm nhanh khi sai lệch vượt một ngưỡng: sai số vị trí khoảng một bán kính agent, nhiễu vận tốc khoảng 0,3 (60% vận tốc tối đa).
- Tăng số agent là sai lệch gây hại nhất: với 7 agent, tỉ lệ an toàn còn 74,6–85,5%.
- Tăng số vật cản chỉ làm giảm nhẹ (còn 96,5–98,2% với 8 vật cản).

Giả thuyết H1, xét theo tiêu chí đã đặt trước (giảm ít nhất 5 điểm phần trăm ở mức nhiễu "vừa"), không được xác nhận, vì mức "vừa" rơi vào vùng DGPPO còn chịu được. Ở mức nhiễu "cao", tỉ lệ an toàn giảm 11–14 điểm phần trăm.

Kết luận chung: bảo đảm an toàn của DGPPO chỉ đúng trong điều kiện giống lúc huấn luyện và không có biên dự phòng cho sai lệch khi triển khai.

---

## 1. Giới thiệu

DGPPO huấn luyện chính sách điều khiển nhiều agent kèm một hàm rào chắn (control barrier function, CBF). Hàm này giúp các agent tránh va chạm với nhau và với vật cản. Trong bài gốc, DGPPO đạt tỉ lệ an toàn gần 100% trên chính điều kiện đã dùng để huấn luyện.

Khi triển khai thực tế cho đội UAV, điều kiện hiếm khi giống hệt lúc huấn luyện: cảm biến định vị có sai số, gió làm lệch chuyển động, số lượng UAV và mật độ chướng ngại vật có thể khác.

**Câu hỏi nghiên cứu:** chính sách DGPPO huấn luyện trong điều kiện sạch còn an toàn đến đâu khi điều kiện triển khai thay đổi? Nó bắt đầu mất an toàn ở mức sai lệch nào, và vì sao?

**Giả thuyết H1:** khi cường độ nhiễu động học (σ_w) và nhiễu cảm biến (σ_v) tăng, tỉ lệ an toàn của DGPPO giảm rõ rệt so với khi không có nhiễu.

## 2. Phương pháp

### 2.1 Đối tượng đánh giá

- **Chính sách:** các checkpoint DGPPO do mạch T1 huấn luyện (bước 200 000, 3 seed huấn luyện: 0, 1, 2). Không huấn luyện lại, không thay đổi tham số nào của chính sách.
- **Môi trường:** hai môi trường của DGPPO.
  - `LidarSpread`: các agent phải phân tán đến các mục tiêu.
  - `LidarLine`: các agent phải xếp thành một hàng giữa hai mốc.
  - Cả hai dùng vùng 1,5 × 1,5, bán kính agent r = 0,05, cảm biến LiDAR 32 tia và tối đa 128 bước mỗi episode.
- **Cấu hình huấn luyện:** 3 agent, 3 vật cản, không nhiễu. Mọi tham số môi trường khác giữ đúng như lúc huấn luyện (đối chiếu chi tiết ở `docs/T2_calibration.md`, mục 5).

### 2.2 Các loại sai lệch được thử

| Loại sai lệch | Ý nghĩa thực tế | Cách mô phỏng | Các mức thử |
|---|---|---|---|
| Nhiễu cảm biến σ_v | sai số định vị (GPS, LiDAR) | cộng nhiễu Gauss vào vị trí các agent và vị trí các điểm LiDAR mà chính sách nhìn thấy; vị trí thật không đổi | 0 đến 4r (0 đến 0,2) |
| Nhiễu động học σ_w | gió, sai số động cơ | cộng nhiễu Gauss vào vận tốc thật của mỗi agent sau mỗi bước | 0 đến 0,5 (vận tốc tối đa là 0,5) |
| Số vật cản | môi trường đông chướng ngại vật hơn | tăng số vật cản lúc triển khai | 3 (như lúc huấn luyện), 5, 8 |
| Số agent | đội hình lớn hơn | tăng số agent lúc triển khai, giữ nguyên diện tích | 3 (như lúc huấn luyện), 5, 7 |

Hai loại nhiễu được thử riêng rẽ. Với nhiễu cảm biến, vi phạm luôn được đo trên vị trí thật. Nhiễu chỉ làm chính sách "nhìn sai", không làm sai thước đo.

### 2.3 Chỉ số đánh giá

- **Tỉ lệ an toàn (safe rate):** tỉ lệ agent không va chạm lần nào trong suốt episode. Đây đúng là chỉ số mà DGPPO dùng (`test.py`). Một agent bị coi là vi phạm nếu khoảng cách tới agent khác nhỏ hơn 2r, hoặc khoảng cách tới vật cản nhỏ hơn r.
- **Tỉ lệ episode an toàn:** tỉ lệ episode mà không agent nào vi phạm. Chỉ số này khắt khe hơn tỉ lệ an toàn.
- **Loại vi phạm:** va chạm giữa các agent, hoặc va chạm với vật cản.
- **Hiệu năng nhiệm vụ:** khoảng cách trung bình từ mục tiêu tới agent gần nhất ở cuối episode (càng nhỏ càng tốt), và chi phí nhiệm vụ (bằng −tổng phần thưởng, càng nhỏ càng tốt).

Mỗi tỉ lệ đi kèm khoảng tin cậy 95% (phương pháp Wilson).

### 2.4 Thiết kế thí nghiệm

- **Số lượng:** mỗi cấu hình chạy 256 episode cho mỗi seed, tức 768 episode khi gộp 3 seed.
- **So sánh công bằng:** mọi cấu hình dùng cùng một tập 256 tình huống ban đầu (vị trí agent, mục tiêu, vật cản). Nhờ vậy có thể so từng episode giữa có nhiễu và không nhiễu.
- **Chọn mức nhiễu:** ba mức "thấp", "vừa", "cao" được chọn trước khi kiểm định, theo quy tắc ghi sẵn trên LidarSpread seed 0:
  - "cao" là mức nhỏ nhất làm tỉ lệ an toàn xuống dưới 95%;
  - "vừa" ≈ một nửa mức cao;
  - "thấp" ≈ một phần tư mức cao.

  Kết quả:

  | | thấp | vừa | cao |
  |---|---|---|---|
  | σ_v | 0,0125 (0,25r) | 0,0375 (0,75r) | 0,075 (1,5r) |
  | σ_w | 0,1 | 0,2 | 0,5 |

- **Kiểm định H1:** so ghép cặp từng episode giữa σ = 0 và mức "vừa", kiểm định hoán vị một phía.

  H1 được xác nhận nếu tỉ lệ an toàn giảm **ít nhất 5 điểm phần trăm** với p < 0,05, trên cả hai môi trường. Tiêu chí này được chốt trước khi chạy thí nghiệm (`docs/T2_calibration.md`, mục 3).

### 2.5 Kiểm tra quy trình đánh giá

Trước khi thêm sai lệch, quy trình đánh giá của T2 được chạy ở điều kiện gốc trên DGPPO seed 0 (32 episode). Kết quả trùng với số liệu mạch T1 đã báo cáo:

| Môi trường | Tỉ lệ an toàn (T2) | (T1) | Phần thưởng (T2) | (T1) |
|---|---|---|---|---|
| LidarSpread | 100% | 100% | −0,884 | −0,885 |
| LidarLine | 98,958% | 98,958% | −0,280 | −0,280 |

Điều này xác nhận quy trình đánh giá không làm thay đổi hành vi của chính sách.

## 3. Kết quả

Các số dưới đây gộp 3 seed. Bảng chi tiết kèm khoảng tin cậy nằm ở Phụ lục A.

### 3.1 Nhiễu cảm biến vị trí

**Bảng 1.** Tỉ lệ an toàn (%) theo nhiễu cảm biến σ_v (r = 0,05 là bán kính agent).

| σ_v | 0 | 0,25r | 0,5r | 0,75r | 1r | 1,5r | 2r | 3r | 4r |
|---|---|---|---|---|---|---|---|---|---|
| LidarSpread | 99,5 | 99,4 | 99,3 | 97,8 | 94,9 | 86,5 | 74,9 | 53,2 | 37,2 |
| LidarLine | 99,5 | 99,4 | 99,4 | 98,7 | 97,3 | 87,8 | 76,0 | 54,8 | 39,5 |

**Quan sát:**
- Khi sai số vị trí nhỏ hơn nửa bán kính agent, DGPPO hầu như không bị ảnh hưởng.
- Từ khoảng 0,75r, tỉ lệ an toàn bắt đầu giảm; khi vượt 1r thì giảm rất nhanh. Ngưỡng mất an toàn (tỉ lệ dưới 95%) nằm ở khoảng 1r với LidarSpread và khoảng 1,1r với LidarLine.
- Ở 4r, chỉ còn khoảng 4 trong 10 agent đi hết episode mà không va chạm.

**Loại vi phạm:**
- Ở mức nhiễu vừa phải (0,75r–1,5r), phần lớn vi phạm là **va vật cản** (82–96% trên LidarSpread). Vị trí vật cản mà agent "thấy" qua LiDAR bị lệch, nên agent đi quá sát và va vào.
- Khi nhiễu rất lớn (từ 3r), **va chạm giữa các agent** chiếm đa số (54–71%), vì các agent không còn biết chính xác vị trí của nhau.

### 3.2 Nhiễu động học (vận tốc)

**Bảng 2.** Tỉ lệ an toàn (%) theo nhiễu vận tốc σ_w (vận tốc tối đa của agent là 0,5).

| σ_w | 0 | 0,01 | 0,02 | 0,05 | 0,1 | 0,2 | 0,3 | 0,5 |
|---|---|---|---|---|---|---|---|---|
| LidarSpread | 99,5 | 99,5 | 99,6 | 99,5 | 99,5 | 97,7 | 95,0 | 85,7 |
| LidarLine | 99,5 | 99,4 | 99,4 | 99,6 | 99,3 | 98,5 | 95,2 | 88,3 |

**Quan sát:**
- Đến σ_w = 0,1 (20% vận tốc tối đa), DGPPO hoàn toàn không bị ảnh hưởng.
- Tỉ lệ an toàn bắt đầu giảm từ 0,2; ngưỡng 95% đạt tới ở khoảng σ_w = 0,3; ở 0,5 còn 85,7–88,3%.
- Vi phạm do nhiễu vận tốc chủ yếu là va vật cản (77–89% trên LidarSpread).

So với nhiễu cảm biến, DGPPO chịu nhiễu vận tốc tốt hơn rõ rệt. Lý do: mỗi bước, bộ điều khiển có thể thay đổi vận tốc tới 0,3, nên các nhiễu vận tốc nhỏ được bù ngay ở bước sau. Ngược lại, khi vị trí quan sát bị sai, chính sách không có cách nào biết mình đang nhìn sai.

### 3.3 Kiểm định giả thuyết H1

**Bảng 3.** Mức giảm tỉ lệ an toàn so với σ = 0 (điểm phần trăm, khoảng tin cậy 95%; 768 cặp episode).

| Loại nhiễu | Mức | LidarSpread | LidarLine |
|---|---|---|---|
| σ_w | thấp (0,1) | 0,0 [−0,3; 0,4] | 0,2 [−0,1; 0,4] |
| σ_w | **vừa (0,2)** | **1,8** [1,2; 2,5], p < 0,001 | **1,0** [0,6; 1,6], p < 0,001 |
| σ_w | cao (0,5) | 13,8 [12,3; 15,4] | 11,2 [9,8; 12,7] |
| σ_v | thấp (0,25r) | 0,1 [−0,2; 0,4] | 0,1 [−0,2; 0,4] |
| σ_v | **vừa (0,75r)** | **1,7** [1,1; 2,3], p < 0,001 | **0,8** [0,4; 1,3], p < 0,001 |
| σ_v | cao (1,5r) | 13,0 [11,5; 14,5] | 11,7 [10,2; 13,3] |

**Kết luận về H1:**
- Ở mức "vừa", tỉ lệ an toàn có giảm và mức giảm có ý nghĩa thống kê, nhưng chỉ 0,8–1,8 điểm phần trăm, chưa đạt ngưỡng 5 điểm. Vì vậy H1 **không được xác nhận** theo tiêu chí đã đặt trước, cho cả hai loại nhiễu.
- Kết luận không đổi khi bỏ seed 0 (seed đã dùng để chọn mức nhiễu).
- Ở mức "cao", mức giảm vượt mọi ngưỡng đã xét (2, 5 và 10 điểm).

### 3.4 Thay đổi số vật cản

**Bảng 4.** Tỉ lệ an toàn (%) khi tăng số vật cản (3 agent, không nhiễu).

| Số vật cản | 3 (như lúc huấn luyện) | 5 | 8 |
|---|---|---|---|
| LidarSpread | 99,6 | 99,5 | 98,2 |
| LidarLine | 99,5 | 98,1 | 96,5 |

Tăng số vật cản lên gần gấp ba chỉ làm tỉ lệ an toàn giảm tối đa khoảng 3 điểm phần trăm. Nguyên nhân có thể là chính sách chỉ nhìn vật cản qua 8 tia LiDAR gần nhất, nên dữ liệu đầu vào của chính sách không phụ thuộc vào số vật cản.

### 3.5 Thay đổi số agent

**Bảng 5.** Tỉ lệ an toàn (%) khi tăng số agent (3 vật cản, không nhiễu, diện tích giữ nguyên).

| Số agent | 3 (như lúc huấn luyện) | 5 | 7 | Tỉ lệ episode an toàn ở 7 agent |
|---|---|---|---|---|
| LidarSpread | 99,6 | 93,9 | 85,5 | 57,8 |
| LidarLine | 99,5 | 88,4 | 74,6 | 33,9 |

**Quan sát:**
- Đây là loại sai lệch gây hại nhất. Với 7 agent trên LidarLine, chỉ khoảng một phần ba số episode diễn ra hoàn toàn không có va chạm.
- Gần như toàn bộ vi phạm là va chạm giữa các agent. Diện tích không đổi nên thêm agent cũng có nghĩa là mật độ cao hơn.

**Bảng 6.** Tỉ lệ an toàn (%) với 7 agent, theo từng seed huấn luyện.

| Môi trường | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| LidarSpread | 63,8 | 95,5 | 97,2 |
| LidarLine | 69,6 | 90,4 | 63,8 |

Mức độ suy giảm khác nhau rất lớn giữa các seed. Cùng thuật toán và cùng cấu hình huấn luyện, có chính sách vẫn giữ được trên 95%, có chính sách chỉ còn khoảng 64%. Con số trung bình ở Bảng 5 che mất sự khác biệt này.

**Chính sách ngẫu nhiên so với tất định:** khi cho chính sách chọn hành động ngẫu nhiên theo phân phối đã học thay vì chọn hành động tốt nhất, kết quả gần như không đổi. Ngoại lệ duy nhất là LidarSpread khi tăng số agent: chính sách ngẫu nhiên an toàn hơn (90,7% so với 85,5% ở 7 agent).

### 3.6 Ảnh hưởng tới hiệu năng nhiệm vụ

Nhiễu không chỉ làm giảm an toàn mà còn làm nhiệm vụ hoàn thành kém hơn:
- LidarSpread, σ_w từ 0 lên 0,5: khoảng cách tới mục tiêu ở cuối episode tăng từ 0,267 lên 0,307; chi phí nhiệm vụ tăng từ 0,482 lên 0,601.
- LidarSpread, σ_v từ 0 lên 4r: khoảng cách tăng lên 0,358; chi phí tăng lên 0,588.

Nghĩa là không có sự đánh đổi kiểu "kém an toàn hơn nhưng làm nhiệm vụ tốt hơn": cả hai cùng xấu đi.

Thêm vật cản cũng làm agent dừng xa mục tiêu hơn (LidarLine: 0,048 → 0,135 với 8 vật cản).

## 4. Thảo luận

**Suy giảm có dạng "vách đá".** Với cả hai loại nhiễu, tỉ lệ an toàn gần như không đổi ở mức nhỏ, rồi rơi nhanh trong một khoảng hẹp: sai số vị trí từ 0,75r đến 2r, nhiễu vận tốc từ 0,2 đến 0,5. Hệ quả thực tế:
- Một hệ thống vận hành gần ngưỡng có thể chuyển từ an toàn sang mất an toàn chỉ vì sai số tăng nhẹ.
- Kết quả "không xác nhận H1" phải được hiểu đúng. Mức "vừa", được chọn bằng một nửa mức "cao", rơi vào phần bằng phẳng của đường cong. Điều đó không có nghĩa DGPPO chịu được nhiễu: vượt ngưỡng, tỉ lệ an toàn giảm 11–60 điểm phần trăm.

**Vì sao nhiễu cảm biến nguy hiểm hơn nhiễu vận tốc.** Hàm rào chắn của DGPPO quyết định dựa trên khoảng cách tương đối tới agent khác và tới vật cản (qua LiDAR).
- Khi vị trí quan sát bị sai, chính sách tin rằng vẫn còn khoảng cách an toàn trong khi thực tế đã quá gần.
- Nhiễu vận tốc thì tác động lên chuyển động thật, và bộ điều khiển phát hiện, bù lại được ở các bước sau.

**Vì sao tăng số agent gây hại nhiều nhất.** Mạng nơ-ron đồ thị của DGPPO cho phép chạy với số agent bất kỳ. Nhưng chính sách chỉ học cách tránh nhau khi có 3 agent. Với 5–7 agent trên cùng diện tích, các tình huống nhiều agent cùng lúc tiến gần nhau chưa từng xuất hiện khi huấn luyện, và chính sách không xử lý được.

**Khác biệt giữa các seed.**
- Seed 0 bền nhất trước nhiễu. Ví dụ ở σ_w = 0,5 trên LidarSpread, seed 0 / 1 / 2 đạt 90,6 / 83,3 / 83,2%. Nhưng chính seed 0 lại kém nhất khi tăng số agent trên LidarSpread (63,8% so với 95–97%). Độ bền trước từng loại sai lệch là tính chất riêng của từng chính sách, không suy ra được từ nhau.
- Vì các mức nhiễu được chọn trên seed 0, mà seed 0 bền nhất, nên các mức này hơi dễ dãi so với các seed còn lại.

**Hàm ý cho triển khai và cho các mạch khác.**
- Để giữ mức an toàn như lúc huấn luyện, sai số định vị cần nhỏ hơn khoảng một nửa bán kính an toàn, và không nên triển khai chính sách huấn luyện với ít agent cho đội hình đông hơn mà không đánh giá lại.
- Bảo đảm an toàn của DGPPO không có biên dự phòng cho sai lệch. Đây là động cơ trực tiếp cho biên an toàn thích nghi của CM-DGPPO (mạch T3): biên tự nới rộng khi tần suất vi phạm thực tế vượt mức cho phép. Các kết quả trong báo cáo này là mốc so sánh cho T3.
- Với sai lệch về số agent, cần kiểm tra riêng xem biên thích nghi có đủ hay không, vì vi phạm đến từ tương tác giữa nhiều agent chứ không phải từ sai số đo.

## 5. Hạn chế

- **Số seed ít:** chỉ có 3 seed huấn luyện, trong khi khác biệt giữa các seed lớn, đặc biệt khi tăng số agent.
- **Chọn mức nhiễu:** các mức nhiễu được chọn trên seed 0, là seed bền nhất trước nhiễu.
- **Nhiễu cảm biến tương quan:** mọi agent nhìn cùng một bản nhiễu. Nhiễu độc lập theo từng agent quan sát chưa được thử.
- **Đo va vật cản:** va chạm với vật cản được đo qua 8 tia LiDAR gần nhất của trạng thái thật (như môi trường gốc của DGPPO), chưa phải hình học chính xác của vật cản.
- **Gộp hai hiệu ứng khi tăng agent:** tăng số agent trên diện tích cố định gộp cả hiệu ứng "nhiều agent" lẫn "mật độ cao".
- **Phạm vi thử:** chưa thử kết hợp nhiều loại sai lệch cùng lúc; chính sách ngẫu nhiên chưa được thử với nhiễu.
- **Tái lập:** kết quả chỉ tái lập ở mức thống kê. Phần mềm tính toán (XLA) cho sai số làm tròn khác nhau giữa các lần chạy, khiến khoảng 0,4% số episode đổi kết quả (Phụ lục B). Mức này nhỏ hơn nhiều so với các hiệu ứng được báo cáo.

## 6. Kết luận

DGPPO giữ được mức an toàn khoảng 99,5% khi điều kiện triển khai chỉ lệch nhỏ so với lúc huấn luyện. Khi sai lệch vượt ngưỡng, an toàn suy giảm nhanh:

| Mức độ ảnh hưởng | Loại sai lệch | Ngưỡng / kết quả |
|---|---|---|
| Nặng nhất | tăng số agent | 85,5% / 74,6% ở 7 agent |
| | nhiễu cảm biến vị trí | ngưỡng khoảng 1 bán kính agent |
| | nhiễu vận tốc | ngưỡng khoảng 0,3 |
| Nhẹ nhất | tăng số vật cản | còn ≥ 96,5% ở 8 vật cản |

Giả thuyết H1 không được xác nhận theo tiêu chí đặt trước ở mức nhiễu "vừa". Tuy vậy, kết quả ở mức "cao" và dạng suy giảm "vách đá" cho thấy DGPPO không bền vững trước sai lệch khi triển khai. Bảo đảm an toàn của nó chỉ đúng trong điều kiện giống lúc huấn luyện.

**Đề xuất:**
1. Định nghĩa lại các mức nhiễu theo mức độ suy giảm thay cho tỉ lệ "một nửa / một phần tư". Nếu thay đổi, cần ghi rõ đây là thay đổi sau khi đã có kết quả hiệu chỉnh.
2. Tăng số seed để ước lượng tốt hơn sự khác biệt giữa các chính sách.
3. Dùng các kết quả này làm mốc so sánh cho CM-DGPPO (T3).
4. Bổ sung chẩn đoán hàm rào chắn (V^h) để giải thích cơ chế vi phạm.

---

## Phụ lục A. Bảng số liệu đầy đủ

Các bảng dưới đây được sinh tự động từ dữ liệu gốc bằng `scripts/t2/fill_results_tables.py`; không sửa tay.

**Cách đọc cột:**
- `safe_agent`: tỉ lệ an toàn (%), kèm khoảng tin cậy Wilson 95%.
- `Δ vs σ=0`: chênh lệch so với không nhiễu (điểm phần trăm).
- `safe_traj`: tỉ lệ episode an toàn.
- `viol agent/obs`: số agent từng va agent khác / từng va vật cản.
- `dist2goal`: khoảng cách tới mục tiêu ở cuối episode.
- `task_cost`: chi phí nhiệm vụ.

**Nguồn dữ liệu** (hồ sơ chạy gồm cấu hình và kết quả, lưu ngoài repo):

<!-- BEGIN:sources -->
- `check0`: `/home/mantd/DGPPO/t2_runs/sweeps/20261005-154219_check0`
- `calib`: `/home/mantd/DGPPO/t2_runs/sweeps/20261005-153934_calib`
- `calib_fine`: `/home/mantd/DGPPO/t2_runs/sweeps/20261005-154058_calib_fine`
- `noise_h1`: `/home/mantd/DGPPO/t2_runs/sweeps/20261005-160338_noise_h1`
- `shift`: `/home/mantd/DGPPO/t2_runs/sweeps/20261005-160338_shift`
<!-- END:sources -->

### A.1 Kiểm tra quy trình ở điều kiện gốc (DGPPO seed 0, 32 episode)

<!-- BEGIN:check0 -->
| env | epi | safe_agent T2 | safe_agent T1 | reward T2 | reward T1 |
|---|---|---|---|---|---|
| LidarLine | 32 | 98.958% | 98.958% | -0.280 | -0.280 |
| LidarSpread | 32 | 100.000% | 100.000% | -0.884 | -0.885 |
<!-- END:check0 -->

### A.2 Hiệu chỉnh mức nhiễu (seed 0, 256 episode)

<!-- BEGIN:calib -->
| env | σ_w | σ_v | epi | safe_agent % [Wilson95] | Δ vs σ=0 (pp) | safe_traj % [Wilson95] | viol agent/obs | dist2goal | task_cost |
|---|---|---|---|---|---|---|---|---|---|
| LidarLine | 0 | 0 | 256 | 99.5 [98.7, 99.8] | +0.0 | 98.4 [96.1, 99.4] | 0/4 | 0.048 | 0.291 |
| LidarLine | 0 | 0.0125 (0.25r) | 256 | 99.2 [98.3, 99.6] | -0.3 | 97.7 [95.0, 98.9] | 0/6 | 0.050 | 0.299 |
| LidarLine | 0 | 0.025 (0.5r) | 256 | 99.2 [98.3, 99.6] | -0.3 | 97.7 [95.0, 98.9] | 0/6 | 0.054 | 0.310 |
| LidarLine | 0 | 0.0375 (0.75r) | 256 | 98.8 [97.8, 99.4] | -0.7 | 96.5 [93.5, 98.1] | 0/9 | 0.055 | 0.317 |
| LidarLine | 0 | 0.05 (1r) | 256 | 97.9 [96.6, 98.7] | -1.6 | 94.1 [90.6, 96.4] | 2/14 | 0.060 | 0.325 |
| LidarLine | 0 | 0.075 (1.5r) | 256 | 90.5 [88.2, 92.4] | -9.0 | 78.1 [72.7, 82.8] | 29/46 | 0.070 | 0.338 |
| LidarLine | 0 | 0.1 (2r) | 256 | 77.2 [74.1, 80.0] | -22.3 | 54.3 [48.2, 60.3] | 81/100 | 0.075 | 0.348 |
| LidarLine | 0 | 0.15 (3r) | 256 | 53.6 [50.1, 57.1] | -45.8 | 27.3 [22.2, 33.1] | 251/140 | 0.095 | 0.367 |
| LidarLine | 0 | 0.2 (4r) | 256 | 39.7 [36.3, 43.2] | -59.8 | 17.2 [13.1, 22.3] | 371/166 | 0.123 | 0.389 |
| LidarLine | 0.01 | 0 | 256 | 99.2 [98.3, 99.6] | -0.3 | 97.7 [95.0, 98.9] | 0/6 | 0.048 | 0.292 |
| LidarLine | 0.02 | 0 | 256 | 99.3 [98.5, 99.7] | -0.1 | 98.0 [95.5, 99.2] | 0/5 | 0.048 | 0.293 |
| LidarLine | 0.05 | 0 | 256 | 99.3 [98.5, 99.7] | -0.1 | 98.0 [95.5, 99.2] | 0/5 | 0.050 | 0.298 |
| LidarLine | 0.1 | 0 | 256 | 99.3 [98.5, 99.7] | -0.1 | 98.0 [95.5, 99.2] | 0/5 | 0.052 | 0.309 |
| LidarLine | 0.2 | 0 | 256 | 98.4 [97.3, 99.1] | -1.0 | 95.7 [92.5, 97.6] | 2/10 | 0.066 | 0.345 |
| LidarLine | 0.3 | 0 | 256 | 96.0 [94.3, 97.1] | -3.5 | 89.5 [85.1, 92.6] | 8/23 | 0.085 | 0.388 |
| LidarLine | 0.5 | 0 | 256 | 90.8 [88.5, 92.6] | -8.7 | 78.1 [72.7, 82.8] | 28/45 | 0.153 | 0.485 |
| LidarSpread | 0 | 0 | 256 | 99.1 [98.1, 99.6] | +0.0 | 98.0 [95.5, 99.2] | 4/3 | 0.701 | 0.887 |
| LidarSpread | 0 | 0.0125 (0.25r) | 256 | 98.8 [97.8, 99.4] | -0.3 | 97.7 [95.0, 98.9] | 6/3 | 0.695 | 0.885 |
| LidarSpread | 0 | 0.025 (0.5r) | 256 | 99.7 [99.1, 99.9] | +0.7 | 99.6 [97.8, 99.9] | 2/0 | 0.691 | 0.879 |
| LidarSpread | 0 | 0.0375 (0.75r) | 256 | 99.1 [98.1, 99.6] | +0.0 | 97.7 [95.0, 98.9] | 2/5 | 0.695 | 0.880 |
| LidarSpread | 0 | 0.05 (1r) | 256 | 97.9 [96.6, 98.7] | -1.2 | 93.8 [90.1, 96.1] | 0/16 | 0.685 | 0.874 |
| LidarSpread | 0 | 0.075 (1.5r) | 256 | 88.9 [86.5, 91.0] | -10.2 | 70.7 [64.9, 75.9] | 8/77 | 0.684 | 0.867 |
| LidarSpread | 0 | 0.1 (2r) | 256 | 77.3 [74.3, 80.2] | -21.7 | 50.8 [44.7, 56.8] | 30/146 | 0.694 | 0.868 |
| LidarSpread | 0 | 0.15 (3r) | 256 | 53.5 [50.0, 57.0] | -45.6 | 23.0 [18.3, 28.6] | 182/188 | 0.750 | 0.900 |
| LidarSpread | 0 | 0.2 (4r) | 256 | 38.3 [34.9, 41.8] | -60.8 | 12.1 [8.7, 16.7] | 299/238 | 0.792 | 0.932 |
| LidarSpread | 0.01 | 0 | 256 | 99.0 [98.0, 99.5] | -0.1 | 97.7 [95.0, 98.9] | 4/4 | 0.700 | 0.888 |
| LidarSpread | 0.02 | 0 | 256 | 99.2 [98.3, 99.6] | +0.1 | 98.4 [96.1, 99.4] | 4/2 | 0.702 | 0.886 |
| LidarSpread | 0.05 | 0 | 256 | 99.3 [98.5, 99.7] | +0.3 | 98.8 [96.6, 99.6] | 4/1 | 0.695 | 0.882 |
| LidarSpread | 0.1 | 0 | 256 | 99.5 [98.7, 99.8] | +0.4 | 98.8 [96.6, 99.6] | 2/2 | 0.692 | 0.871 |
| LidarSpread | 0.2 | 0 | 256 | 98.7 [97.6, 99.3] | -0.4 | 96.5 [93.5, 98.1] | 2/8 | 0.672 | 0.843 |
| LidarSpread | 0.3 | 0 | 256 | 97.9 [96.6, 98.7] | -1.2 | 93.8 [90.1, 96.1] | 0/16 | 0.626 | 0.812 |
| LidarSpread | 0.5 | 0 | 256 | 90.6 [88.4, 92.5] | -8.5 | 76.6 [71.0, 81.3] | 18/54 | 0.536 | 0.757 |
<!-- END:calib -->

Lưới mịn quanh ngưỡng (chỉ để tham khảo, không dùng để chọn mức):

<!-- BEGIN:calib_fine -->
| env | σ_w | σ_v | epi | safe_agent % [Wilson95] | Δ vs σ=0 (pp) | safe_traj % [Wilson95] | viol agent/obs | dist2goal | task_cost |
|---|---|---|---|---|---|---|---|---|---|
| LidarLine | 0 | 0 | 256 | 99.3 [98.5, 99.7] | +0.0 | 98.0 [95.5, 99.2] | 0/5 | 0.048 | 0.291 |
| LidarLine | 0 | 0.055 (1.1r) | 256 | 97.1 [95.7, 98.1] | -2.2 | 92.2 [88.2, 94.9] | 4/18 | 0.062 | 0.328 |
| LidarLine | 0 | 0.06 (1.2r) | 256 | 95.6 [93.9, 96.8] | -3.8 | 88.3 [83.8, 91.7] | 6/28 | 0.066 | 0.331 |
| LidarLine | 0 | 0.065 (1.3r) | 256 | 95.2 [93.4, 96.5] | -4.2 | 87.5 [82.9, 91.0] | 10/27 | 0.067 | 0.334 |
| LidarLine | 0 | 0.07 (1.4r) | 256 | 92.4 [90.4, 94.1] | -6.9 | 81.2 [76.0, 85.6] | 16/44 | 0.068 | 0.336 |
| LidarLine | 0.35 | 0 | 256 | 95.2 [93.4, 96.5] | -4.2 | 88.3 [83.8, 91.7] | 12/25 | 0.101 | 0.413 |
| LidarLine | 0.4 | 0 | 256 | 94.3 [92.4, 95.7] | -5.1 | 85.9 [81.1, 89.7] | 14/30 | 0.114 | 0.436 |
| LidarLine | 0.45 | 0 | 256 | 91.8 [89.6, 93.5] | -7.6 | 80.1 [74.8, 84.5] | 22/42 | 0.131 | 0.461 |
| LidarSpread | 0 | 0 | 256 | 99.1 [98.1, 99.6] | +0.0 | 98.0 [95.5, 99.2] | 4/3 | 0.701 | 0.888 |
| LidarSpread | 0 | 0.055 (1.1r) | 256 | 96.2 [94.6, 97.4] | -2.9 | 89.1 [84.6, 92.3] | 2/27 | 0.684 | 0.872 |
| LidarSpread | 0 | 0.06 (1.2r) | 256 | 94.9 [93.1, 96.3] | -4.2 | 85.9 [81.1, 89.7] | 4/35 | 0.680 | 0.869 |
| LidarSpread | 0 | 0.065 (1.3r) | 256 | 93.8 [91.8, 95.3] | -5.3 | 83.2 [78.1, 87.3] | 4/44 | 0.680 | 0.867 |
| LidarSpread | 0 | 0.07 (1.4r) | 256 | 91.4 [89.2, 93.2] | -7.7 | 76.6 [71.0, 81.3] | 2/64 | 0.676 | 0.864 |
| LidarSpread | 0.35 | 0 | 256 | 95.2 [93.4, 96.5] | -3.9 | 87.1 [82.4, 90.7] | 8/31 | 0.603 | 0.794 |
| LidarSpread | 0.4 | 0 | 256 | 94.5 [92.7, 95.9] | -4.6 | 86.3 [81.6, 90.0] | 14/31 | 0.582 | 0.780 |
| LidarSpread | 0.45 | 0 | 256 | 92.6 [90.5, 94.2] | -6.5 | 82.0 [76.9, 86.2] | 16/42 | 0.556 | 0.765 |
<!-- END:calib_fine -->

### A.3 Nhiễu: gộp 3 seed (768 episode mỗi dòng)

<!-- BEGIN:noise_pooled -->
| env | σ_w | σ_v | epi | safe_agent % [Wilson95] | Δ vs σ=0 (pp) | safe_traj % [Wilson95] | viol agent/obs | dist2goal | task_cost |
|---|---|---|---|---|---|---|---|---|---|
| LidarLine | 0 | 0 | 768 | 99.5 [99.1, 99.7] | +0.0 | 98.6 [97.5, 99.2] | 0/11 | 0.047 | 0.284 |
| LidarLine | 0 | 0.0125 (0.25r) | 768 | 99.4 [99.0, 99.7] | -0.1 | 98.3 [97.1, 99.0] | 0/13 | 0.049 | 0.293 |
| LidarLine | 0 | 0.025 (0.5r) | 768 | 99.4 [99.0, 99.6] | -0.1 | 98.2 [97.0, 98.9] | 0/14 | 0.050 | 0.306 |
| LidarLine | 0 | 0.0375 (0.75r) | 768 | 98.7 [98.1, 99.1] | -0.8 | 96.2 [94.6, 97.4] | 2/28 | 0.051 | 0.312 |
| LidarLine | 0 | 0.05 (1r) | 768 | 97.3 [96.6, 97.9] | -2.2 | 92.6 [90.5, 94.2] | 10/52 | 0.056 | 0.320 |
| LidarLine | 0 | 0.075 (1.5r) | 768 | 87.8 [86.4, 89.1] | -11.7 | 73.3 [70.1, 76.3] | 115/169 | 0.066 | 0.334 |
| LidarLine | 0 | 0.1 (2r) | 768 | 76.0 [74.3, 77.7] | -23.5 | 54.6 [51.0, 58.0] | 285/294 | 0.074 | 0.345 |
| LidarLine | 0 | 0.15 (3r) | 768 | 54.8 [52.8, 56.8] | -44.7 | 30.9 [27.7, 34.2] | 736/419 | 0.097 | 0.367 |
| LidarLine | 0 | 0.2 (4r) | 768 | 39.5 [37.5, 41.5] | -60.1 | 18.5 [15.9, 21.4] | 1143/474 | 0.130 | 0.395 |
| LidarLine | 0.01 | 0 | 768 | 99.4 [99.0, 99.7] | -0.1 | 98.4 [97.3, 99.1] | 0/13 | 0.049 | 0.285 |
| LidarLine | 0.02 | 0 | 768 | 99.4 [99.0, 99.7] | -0.1 | 98.4 [97.3, 99.1] | 0/13 | 0.048 | 0.286 |
| LidarLine | 0.05 | 0 | 768 | 99.6 [99.2, 99.8] | +0.0 | 98.8 [97.8, 99.4] | 0/10 | 0.049 | 0.290 |
| LidarLine | 0.1 | 0 | 768 | 99.3 [98.9, 99.6] | -0.2 | 98.0 [96.8, 98.8] | 0/15 | 0.050 | 0.302 |
| LidarLine | 0.2 | 0 | 768 | 98.5 [97.9, 98.9] | -1.0 | 96.0 [94.3, 97.1] | 6/29 | 0.063 | 0.341 |
| LidarLine | 0.3 | 0 | 768 | 95.2 [94.2, 96.0] | -4.3 | 88.2 [85.7, 90.2] | 36/75 | 0.085 | 0.388 |
| LidarLine | 0.5 | 0 | 768 | 88.3 [86.9, 89.5] | -11.2 | 73.3 [70.1, 76.3] | 115/161 | 0.156 | 0.486 |
| LidarSpread | 0 | 0 | 768 | 99.5 [99.1, 99.7] | +0.0 | 98.8 [97.8, 99.4] | 4/7 | 0.267 | 0.482 |
| LidarSpread | 0 | 0.0125 (0.25r) | 768 | 99.4 [99.0, 99.7] | -0.1 | 98.7 [97.6, 99.3] | 6/7 | 0.265 | 0.492 |
| LidarSpread | 0 | 0.025 (0.5r) | 768 | 99.3 [98.9, 99.6] | -0.2 | 98.3 [97.1, 99.0] | 2/13 | 0.266 | 0.502 |
| LidarSpread | 0 | 0.0375 (0.75r) | 768 | 97.8 [97.2, 98.4] | -1.7 | 93.6 [91.7, 95.1] | 2/48 | 0.269 | 0.511 |
| LidarSpread | 0 | 0.05 (1r) | 768 | 94.9 [93.9, 95.7] | -4.6 | 85.8 [83.2, 88.1] | 12/106 | 0.269 | 0.514 |
| LidarSpread | 0 | 0.075 (1.5r) | 768 | 86.5 [85.1, 87.9] | -13.0 | 66.4 [63.0, 69.7] | 56/262 | 0.275 | 0.522 |
| LidarSpread | 0 | 0.1 (2r) | 768 | 74.9 [73.1, 76.6] | -24.7 | 48.2 [44.7, 51.7] | 200/405 | 0.282 | 0.529 |
| LidarSpread | 0 | 0.15 (3r) | 768 | 53.2 [51.1, 55.2] | -46.4 | 24.5 [21.6, 27.6] | 631/547 | 0.319 | 0.556 |
| LidarSpread | 0 | 0.2 (4r) | 768 | 37.2 [35.2, 39.2] | -62.3 | 12.6 [10.5, 15.2] | 1018/681 | 0.358 | 0.588 |
| LidarSpread | 0.01 | 0 | 768 | 99.5 [99.1, 99.7] | +0.0 | 98.8 [97.8, 99.4] | 4/7 | 0.266 | 0.483 |
| LidarSpread | 0.02 | 0 | 768 | 99.6 [99.3, 99.8] | +0.1 | 99.1 [98.1, 99.6] | 4/5 | 0.267 | 0.482 |
| LidarSpread | 0.05 | 0 | 768 | 99.5 [99.1, 99.7] | -0.0 | 99.0 [98.0, 99.5] | 6/5 | 0.264 | 0.485 |
| LidarSpread | 0.1 | 0 | 768 | 99.5 [99.1, 99.7] | -0.0 | 98.6 [97.5, 99.2] | 2/10 | 0.266 | 0.494 |
| LidarSpread | 0.2 | 0 | 768 | 97.7 [97.0, 98.2] | -1.8 | 93.6 [91.7, 95.1] | 6/47 | 0.273 | 0.520 |
| LidarSpread | 0.3 | 0 | 768 | 95.0 [94.0, 95.8] | -4.6 | 86.5 [83.9, 88.7] | 14/105 | 0.280 | 0.547 |
| LidarSpread | 0.5 | 0 | 768 | 85.7 [84.2, 87.1] | -13.8 | 65.8 [62.3, 69.0] | 79/258 | 0.307 | 0.601 |
<!-- END:noise_pooled -->

### A.4 Nhiễu: tỉ lệ an toàn (%) theo từng seed

<!-- BEGIN:noise_per_seed -->
| env | σ_w | σ_v | seed 0 | seed 1 | seed 2 | gộp [Wilson95] | safe_traj gộp |
|---|---|---|---|---|---|---|---|
| LidarLine | 0 | 0 | 99.3 | 99.7 | 99.5 | 99.5 [99.1, 99.7] | 98.6 |
| LidarLine | 0 | 0.0125 (0.25r) | 99.3 | 99.5 | 99.5 | 99.4 [99.0, 99.7] | 98.3 |
| LidarLine | 0 | 0.025 (0.5r) | 99.3 | 99.1 | 99.7 | 99.4 [99.0, 99.6] | 98.2 |
| LidarLine | 0 | 0.0375 (0.75r) | 99.0 | 98.3 | 98.8 | 98.7 [98.1, 99.1] | 96.2 |
| LidarLine | 0 | 0.05 (1r) | 98.0 | 96.7 | 97.1 | 97.3 [96.6, 97.9] | 92.6 |
| LidarLine | 0 | 0.075 (1.5r) | 90.8 | 87.9 | 84.9 | 87.8 [86.4, 89.1] | 73.3 |
| LidarLine | 0 | 0.1 (2r) | 78.5 | 74.5 | 75.1 | 76.0 [74.3, 77.7] | 54.6 |
| LidarLine | 0 | 0.15 (3r) | 53.6 | 58.3 | 52.5 | 54.8 [52.8, 56.8] | 30.9 |
| LidarLine | 0 | 0.2 (4r) | 40.0 | 42.3 | 36.1 | 39.5 [37.5, 41.5] | 18.5 |
| LidarLine | 0.01 | 0 | 99.5 | 99.7 | 99.1 | 99.4 [99.0, 99.7] | 98.4 |
| LidarLine | 0.02 | 0 | 99.5 | 99.7 | 99.1 | 99.4 [99.0, 99.7] | 98.4 |
| LidarLine | 0.05 | 0 | 99.3 | 99.9 | 99.5 | 99.6 [99.2, 99.8] | 98.8 |
| LidarLine | 0.1 | 0 | 99.3 | 99.7 | 99.0 | 99.3 [98.9, 99.6] | 98.0 |
| LidarLine | 0.2 | 0 | 98.4 | 99.0 | 98.0 | 98.5 [97.9, 98.9] | 96.0 |
| LidarLine | 0.3 | 0 | 96.2 | 95.7 | 93.6 | 95.2 [94.2, 96.0] | 88.2 |
| LidarLine | 0.5 | 0 | 90.9 | 87.2 | 86.7 | 88.3 [86.9, 89.5] | 73.3 |
| LidarSpread | 0 | 0 | 99.1 | 99.7 | 99.7 | 99.5 [99.1, 99.7] | 98.8 |
| LidarSpread | 0 | 0.0125 (0.25r) | 98.8 | 99.7 | 99.7 | 99.4 [99.0, 99.7] | 98.7 |
| LidarSpread | 0 | 0.025 (0.5r) | 99.7 | 99.6 | 98.7 | 99.3 [98.9, 99.6] | 98.3 |
| LidarSpread | 0 | 0.0375 (0.75r) | 99.1 | 98.6 | 95.8 | 97.8 [97.2, 98.4] | 93.6 |
| LidarSpread | 0 | 0.05 (1r) | 97.9 | 96.0 | 90.8 | 94.9 [93.9, 95.7] | 85.8 |
| LidarSpread | 0 | 0.075 (1.5r) | 88.8 | 86.6 | 84.2 | 86.5 [85.1, 87.9] | 66.4 |
| LidarSpread | 0 | 0.1 (2r) | 77.1 | 75.4 | 72.1 | 74.9 [73.1, 76.6] | 48.2 |
| LidarSpread | 0 | 0.15 (3r) | 53.4 | 52.2 | 53.9 | 53.2 [51.1, 55.2] | 24.5 |
| LidarSpread | 0 | 0.2 (4r) | 38.3 | 36.8 | 36.5 | 37.2 [35.2, 39.2] | 12.6 |
| LidarSpread | 0.01 | 0 | 99.0 | 99.9 | 99.7 | 99.5 [99.1, 99.7] | 98.8 |
| LidarSpread | 0.02 | 0 | 99.2 | 99.9 | 99.7 | 99.6 [99.3, 99.8] | 99.1 |
| LidarSpread | 0.05 | 0 | 99.3 | 99.6 | 99.6 | 99.5 [99.1, 99.7] | 99.0 |
| LidarSpread | 0.1 | 0 | 99.5 | 99.7 | 99.2 | 99.5 [99.1, 99.7] | 98.6 |
| LidarSpread | 0.2 | 0 | 98.7 | 98.0 | 96.4 | 97.7 [97.0, 98.2] | 93.6 |
| LidarSpread | 0.3 | 0 | 97.9 | 94.3 | 92.7 | 95.0 [94.0, 95.8] | 86.5 |
| LidarSpread | 0.5 | 0 | 90.6 | 83.3 | 83.2 | 85.7 [84.2, 87.1] | 65.8 |
<!-- END:noise_per_seed -->

### A.5 Nhiễu: loại vi phạm và thời điểm vi phạm đầu tiên

"% vật cản" là tỉ lệ va vật cản trong tổng số vi phạm. "t vi phạm đầu" là trung vị của bước đầu tiên có vi phạm (0–127), tính trên các episode có vi phạm.

<!-- BEGIN:noise_viol -->
| env | σ_w | σ_v | viol agent | viol obs | % vật cản | episode có vi phạm | t vi phạm đầu (trung vị) |
|---|---|---|---|---|---|---|---|
| LidarLine | 0 | 0 | 0 | 11 | 100 | 11 | 77 |
| LidarLine | 0 | 0.0125 (0.25r) | 0 | 13 | 100 | 13 | 30 |
| LidarLine | 0 | 0.025 (0.5r) | 0 | 14 | 100 | 14 | 29 |
| LidarLine | 0 | 0.0375 (0.75r) | 2 | 28 | 93 | 29 | 29 |
| LidarLine | 0 | 0.05 (1r) | 10 | 52 | 84 | 57 | 34 |
| LidarLine | 0 | 0.075 (1.5r) | 115 | 169 | 60 | 205 | 40 |
| LidarLine | 0 | 0.1 (2r) | 285 | 294 | 51 | 349 | 35 |
| LidarLine | 0 | 0.15 (3r) | 736 | 419 | 36 | 531 | 32 |
| LidarLine | 0 | 0.2 (4r) | 1143 | 474 | 29 | 626 | 30 |
| LidarLine | 0.01 | 0 | 0 | 13 | 100 | 12 | 47 |
| LidarLine | 0.02 | 0 | 0 | 13 | 100 | 12 | 55 |
| LidarLine | 0.05 | 0 | 0 | 10 | 100 | 9 | 30 |
| LidarLine | 0.1 | 0 | 0 | 15 | 100 | 15 | 79 |
| LidarLine | 0.2 | 0 | 6 | 29 | 83 | 31 | 48 |
| LidarLine | 0.3 | 0 | 36 | 75 | 68 | 91 | 29 |
| LidarLine | 0.5 | 0 | 115 | 161 | 58 | 205 | 45 |
| LidarSpread | 0 | 0 | 4 | 7 | 64 | 9 | 72 |
| LidarSpread | 0 | 0.0125 (0.25r) | 6 | 7 | 54 | 10 | 78 |
| LidarSpread | 0 | 0.025 (0.5r) | 2 | 13 | 87 | 13 | 21 |
| LidarSpread | 0 | 0.0375 (0.75r) | 2 | 48 | 96 | 49 | 43 |
| LidarSpread | 0 | 0.05 (1r) | 12 | 106 | 90 | 109 | 35 |
| LidarSpread | 0 | 0.075 (1.5r) | 56 | 262 | 82 | 258 | 38 |
| LidarSpread | 0 | 0.1 (2r) | 200 | 405 | 67 | 398 | 33 |
| LidarSpread | 0 | 0.15 (3r) | 631 | 547 | 46 | 580 | 34 |
| LidarSpread | 0 | 0.2 (4r) | 1018 | 681 | 40 | 671 | 31 |
| LidarSpread | 0.01 | 0 | 4 | 7 | 64 | 9 | 72 |
| LidarSpread | 0.02 | 0 | 4 | 5 | 56 | 7 | 74 |
| LidarSpread | 0.05 | 0 | 6 | 5 | 45 | 8 | 76 |
| LidarSpread | 0.1 | 0 | 2 | 10 | 83 | 11 | 58 |
| LidarSpread | 0.2 | 0 | 6 | 47 | 89 | 49 | 38 |
| LidarSpread | 0.3 | 0 | 14 | 105 | 88 | 104 | 43 |
| LidarSpread | 0.5 | 0 | 79 | 258 | 77 | 263 | 48 |
<!-- END:noise_viol -->

### A.6 Kiểm định H1: toàn bộ seed 0–2

drop_pp là mức giảm trung bình (điểm phần trăm) kèm khoảng tin cậy bootstrap 95%. p là giá trị p của kiểm định hoán vị đổi dấu một phía. "H1@x pp" đúng khi mức giảm ≥ x và p < 0,05.

<!-- BEGIN:h1_all -->
Nguồn: `/home/mantd/DGPPO/t2_runs/sweeps/20261005-160338_noise_h1/episodes.csv`

| method | env | mode | trục | mức | σ | cặp | drop_pp [CI95] | p một phía | H1@2pp | H1@5pp | H1@10pp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dgppo | LidarLine | det | sigma_w | low | 0.1 | 768 | +0.17 [-0.09, +0.43] | 0.1739 | ✘ | ✘ | ✘ |
| dgppo | LidarLine | det | sigma_w | mid | 0.2 | 768 | +1.04 [+0.56, +1.56] | 0.0001 | ✘ | ✘ | ✘ |
| dgppo | LidarLine | det | sigma_w | high | 0.5 | 768 | +11.24 [+9.77, +12.72] | 0.0001 | ✔ | ✔ | ✔ |
| dgppo | LidarLine | det | sigma_v | low | 0.0125 | 768 | +0.09 [-0.22, +0.39] | 0.3789 | ✘ | ✘ | ✘ |
| dgppo | LidarLine | det | sigma_v | mid | 0.0375 | 768 | +0.82 [+0.35, +1.30] | 0.0006 | ✘ | ✘ | ✘ |
| dgppo | LidarLine | det | sigma_v | high | 0.075 | 768 | +11.68 [+10.16, +13.28] | 0.0001 | ✔ | ✔ | ✔ |
| dgppo | LidarSpread | det | sigma_w | low | 0.1 | 768 | +0.04 [-0.30, +0.39] | 0.5071 | ✘ | ✘ | ✘ |
| dgppo | LidarSpread | det | sigma_w | mid | 0.2 | 768 | +1.82 [+1.17, +2.52] | 0.0001 | ✘ | ✘ | ✘ |
| dgppo | LidarSpread | det | sigma_w | high | 0.5 | 768 | +13.80 [+12.28, +15.41] | 0.0001 | ✔ | ✔ | ✔ |
| dgppo | LidarSpread | det | sigma_v | low | 0.0125 | 768 | +0.09 [-0.17, +0.39] | 0.3283 | ✘ | ✘ | ✘ |
| dgppo | LidarSpread | det | sigma_v | mid | 0.0375 | 768 | +1.69 [+1.09, +2.34] | 0.0001 | ✘ | ✘ | ✘ |
| dgppo | LidarSpread | det | sigma_v | high | 0.075 | 768 | +12.98 [+11.50, +14.50] | 0.0001 | ✔ | ✔ | ✔ |

**Tiêu chí chính (mức mid, ngưỡng 5 pp, đúng trên cả hai env):**

- dgppo (det), sigma_v: **H1 KHÔNG ĐẠT** (LidarLine: False, LidarSpread: False)
- dgppo (det), sigma_w: **H1 KHÔNG ĐẠT** (LidarLine: False, LidarSpread: False)
<!-- END:h1_all -->

### A.7 Kiểm định H1: chỉ seed 1–2

<!-- BEGIN:h1_seed12 -->
Nguồn: `/home/mantd/DGPPO/t2_runs/sweeps/20261005-160338_noise_h1/episodes.csv` (seed 1 2)

| method | env | mode | trục | mức | σ | cặp | drop_pp [CI95] | p một phía | H1@2pp | H1@5pp | H1@10pp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dgppo | LidarLine | det | sigma_w | low | 0.1 | 512 | +0.26 [+0.00, +0.59] | 0.1052 | ✘ | ✘ | ✘ |
| dgppo | LidarLine | det | sigma_w | mid | 0.2 | 512 | +1.11 [+0.52, +1.82] | 0.0001 | ✘ | ✘ | ✘ |
| dgppo | LidarLine | det | sigma_w | high | 0.5 | 512 | +12.63 [+10.74, +14.58] | 0.0001 | ✔ | ✔ | ✔ |
| dgppo | LidarLine | det | sigma_v | low | 0.0125 | 512 | +0.13 [-0.20, +0.46] | 0.3403 | ✘ | ✘ | ✘ |
| dgppo | LidarLine | det | sigma_v | mid | 0.0375 | 512 | +1.04 [+0.46, +1.69] | 0.0009 | ✘ | ✘ | ✘ |
| dgppo | LidarLine | det | sigma_v | high | 0.075 | 512 | +13.22 [+11.20, +15.30] | 0.0001 | ✔ | ✔ | ✔ |
| dgppo | LidarSpread | det | sigma_w | low | 0.1 | 512 | +0.26 [-0.13, +0.65] | 0.1781 | ✘ | ✘ | ✘ |
| dgppo | LidarSpread | det | sigma_w | mid | 0.2 | 512 | +2.54 [+1.69, +3.45] | 0.0001 | ✔ | ✘ | ✘ |
| dgppo | LidarSpread | det | sigma_w | high | 0.5 | 512 | +16.47 [+14.52, +18.49] | 0.0001 | ✔ | ✔ | ✔ |
| dgppo | LidarSpread | det | sigma_v | low | 0.0125 | 512 | +0.00 [-0.20, +0.20] | 0.7456 | ✘ | ✘ | ✘ |
| dgppo | LidarSpread | det | sigma_v | mid | 0.0375 | 512 | +2.54 [+1.76, +3.39] | 0.0001 | ✔ | ✘ | ✘ |
| dgppo | LidarSpread | det | sigma_v | high | 0.075 | 512 | +14.32 [+12.43, +16.28] | 0.0001 | ✔ | ✔ | ✔ |

**Tiêu chí chính (mức mid, ngưỡng 5 pp, đúng trên cả hai env):**

- dgppo (det), sigma_v: **H1 KHÔNG ĐẠT** (LidarLine: False, LidarSpread: False)
- dgppo (det), sigma_w: **H1 KHÔNG ĐẠT** (LidarLine: False, LidarSpread: False)
<!-- END:h1_seed12 -->

### A.8 Thay đổi số agent / số vật cản: gộp 3 seed

`det` là chính sách tất định (chọn hành động tốt nhất); `stoch` là chính sách ngẫu nhiên (lấy mẫu từ phân phối đã học).

<!-- BEGIN:shift_pooled -->
| env | N | obs | mode | σ_w | σ_v | epi | safe_agent % [Wilson95] | Δ vs σ=0 (pp) | safe_traj % [Wilson95] | viol agent/obs | dist2goal | task_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LidarLine | 3 | 3 | det | 0 | 0 | 768 | 99.5 [99.1, 99.7] | +0.0 | 98.6 [97.5, 99.2] | 0/11 | 0.048 | 0.284 |
| LidarLine | 3 | 3 | stoch | 0 | 0 | 768 | 99.4 [99.0, 99.6] | +0.0 | 98.2 [97.0, 98.9] | 0/14 | 0.055 | 0.304 |
| LidarLine | 3 | 5 | det | 0 | 0 | 768 | 98.1 [97.5, 98.6] | +0.0 | 94.8 [93.0, 96.2] | 4/39 | 0.082 | 0.324 |
| LidarLine | 3 | 5 | stoch | 0 | 0 | 768 | 98.3 [97.6, 98.7] | +0.0 | 95.1 [93.3, 96.4] | 4/36 | 0.087 | 0.342 |
| LidarLine | 3 | 8 | det | 0 | 0 | 768 | 96.5 [95.7, 97.2] | +0.0 | 90.4 [88.1, 92.3] | 8/72 | 0.135 | 0.384 |
| LidarLine | 3 | 8 | stoch | 0 | 0 | 768 | 96.3 [95.5, 97.0] | +0.0 | 90.1 [87.8, 92.0] | 8/78 | 0.139 | 0.398 |
| LidarLine | 5 | 3 | det | 0 | 0 | 768 | 88.4 [87.3, 89.4] | +0.0 | 69.0 [65.7, 72.2] | 387/63 | 0.083 | 0.281 |
| LidarLine | 5 | 3 | stoch | 0 | 0 | 768 | 86.5 [85.4, 87.6] | +0.0 | 65.0 [61.5, 68.3] | 454/65 | 0.087 | 0.294 |
| LidarLine | 7 | 3 | det | 0 | 0 | 768 | 74.6 [73.4, 75.8] | +0.0 | 33.9 [30.6, 37.3] | 1206/191 | 0.101 | 0.289 |
| LidarLine | 7 | 3 | stoch | 0 | 0 | 768 | 74.2 [73.0, 75.3] | +0.0 | 33.6 [30.3, 37.0] | 1236/181 | 0.104 | 0.299 |
| LidarSpread | 3 | 3 | det | 0 | 0 | 768 | 99.6 [99.2, 99.8] | +0.0 | 99.0 [98.0, 99.5] | 4/6 | 0.267 | 0.482 |
| LidarSpread | 3 | 3 | stoch | 0 | 0 | 768 | 99.7 [99.3, 99.8] | +0.0 | 99.0 [98.0, 99.5] | 0/8 | 0.227 | 0.459 |
| LidarSpread | 3 | 5 | det | 0 | 0 | 768 | 99.5 [99.1, 99.7] | +0.0 | 98.4 [97.3, 99.1] | 0/12 | 0.267 | 0.496 |
| LidarSpread | 3 | 5 | stoch | 0 | 0 | 768 | 99.1 [98.6, 99.4] | +0.0 | 97.4 [96.0, 98.3] | 2/19 | 0.233 | 0.476 |
| LidarSpread | 3 | 8 | det | 0 | 0 | 768 | 98.2 [97.6, 98.7] | +0.0 | 95.1 [93.3, 96.4] | 4/37 | 0.277 | 0.517 |
| LidarSpread | 3 | 8 | stoch | 0 | 0 | 768 | 97.4 [96.6, 97.9] | +0.0 | 93.4 [91.4, 94.9] | 18/43 | 0.255 | 0.507 |
| LidarSpread | 5 | 3 | det | 0 | 0 | 768 | 93.9 [93.1, 94.6] | +0.0 | 82.3 [79.4, 84.8] | 129/106 | 0.227 | 0.411 |
| LidarSpread | 5 | 3 | stoch | 0 | 0 | 768 | 96.8 [96.2, 97.3] | +0.0 | 87.8 [85.3, 89.9] | 45/79 | 0.191 | 0.390 |
| LidarSpread | 7 | 3 | det | 0 | 0 | 768 | 85.5 [84.5, 86.4] | +0.0 | 57.8 [54.3, 61.3] | 422/398 | 0.214 | 0.384 |
| LidarSpread | 7 | 3 | stoch | 0 | 0 | 768 | 90.7 [89.9, 91.4] | +0.0 | 61.3 [57.8, 64.7] | 133/377 | 0.177 | 0.358 |
<!-- END:shift_pooled -->

### A.9 Thay đổi số agent / số vật cản: tỉ lệ an toàn (%) theo từng seed

<!-- BEGIN:shift_per_seed -->
| env | mode | N | obs | seed 0 | seed 1 | seed 2 | gộp [Wilson95] | safe_traj gộp |
|---|---|---|---|---|---|---|---|---|
| LidarLine | det | 3 | 3 | 99.5 | 99.7 | 99.3 | 99.5 [99.1, 99.7] | 98.6 |
| LidarLine | det | 3 | 5 | 97.9 | 98.7 | 97.8 | 98.1 [97.5, 98.6] | 94.8 |
| LidarLine | det | 3 | 8 | 96.4 | 96.6 | 96.6 | 96.5 [95.7, 97.2] | 90.4 |
| LidarLine | det | 5 | 3 | 90.3 | 96.1 | 78.8 | 88.4 [87.3, 89.4] | 69.0 |
| LidarLine | det | 7 | 3 | 69.6 | 90.4 | 63.8 | 74.6 [73.4, 75.8] | 33.9 |
| LidarLine | stoch | 3 | 3 | 99.1 | 99.7 | 99.3 | 99.4 [99.0, 99.6] | 98.2 |
| LidarLine | stoch | 3 | 5 | 97.8 | 98.8 | 98.2 | 98.3 [97.6, 98.7] | 95.1 |
| LidarLine | stoch | 3 | 8 | 96.4 | 95.7 | 96.9 | 96.3 [95.5, 97.0] | 90.1 |
| LidarLine | stoch | 5 | 3 | 89.0 | 95.1 | 75.5 | 86.5 [85.4, 87.6] | 65.0 |
| LidarLine | stoch | 7 | 3 | 71.5 | 89.1 | 61.9 | 74.2 [73.0, 75.3] | 33.6 |
| LidarSpread | det | 3 | 3 | 99.1 | 99.9 | 99.7 | 99.6 [99.2, 99.8] | 99.0 |
| LidarSpread | det | 3 | 5 | 99.5 | 99.7 | 99.2 | 99.5 [99.1, 99.7] | 98.4 |
| LidarSpread | det | 3 | 8 | 98.6 | 99.1 | 97.0 | 98.2 [97.6, 98.7] | 95.1 |
| LidarSpread | det | 5 | 3 | 84.6 | 98.4 | 98.8 | 93.9 [93.1, 94.6] | 82.3 |
| LidarSpread | det | 7 | 3 | 63.8 | 95.5 | 97.2 | 85.5 [84.5, 86.4] | 57.8 |
| LidarSpread | stoch | 3 | 3 | 100.0 | 99.7 | 99.2 | 99.7 [99.3, 99.8] | 99.0 |
| LidarSpread | stoch | 3 | 5 | 99.3 | 99.2 | 98.7 | 99.1 [98.6, 99.4] | 97.4 |
| LidarSpread | stoch | 3 | 8 | 98.2 | 98.0 | 95.8 | 97.4 [96.6, 97.9] | 93.4 |
| LidarSpread | stoch | 5 | 3 | 95.2 | 96.6 | 98.6 | 96.8 [96.2, 97.3] | 87.8 |
| LidarSpread | stoch | 7 | 3 | 81.6 | 93.7 | 96.7 | 90.7 [89.9, 91.4] | 61.3 |
<!-- END:shift_per_seed -->

## Phụ lục B. Ghi chú kỹ thuật

**Tính tái lập.** Với JAX 0.6.2, phần mềm XLA cộng dồn các số thực theo thứ tự thay đổi giữa các lần chạy, trên cả GPU lẫn CPU.
- Sai số ban đầu rất nhỏ (khoảng 10⁻⁸ đến 10⁻⁶). Nhưng khi chạm các ngưỡng rời rạc trong môi trường (chọn 8 tia LiDAR gần nhất, phạm vi liên lạc), sai số có thể làm một vài episode đi theo hướng khác.
- Ở điều kiện gốc, khoảng 0,4% số episode đổi kết quả giữa hai lần chạy (dao động khoảng ±0,3 điểm phần trăm).
- Nguyên nhân không phải khởi tạo ngẫu nhiên. Tham số chính sách nạp từ file, trạng thái ban đầu và khóa ngẫu nhiên đã được kiểm tra là giống hệt nhau giữa các lần chạy.
- Tuỳ chọn tất định `--xla_gpu_deterministic_ops` của XLA cho kết quả sai, nên không được sử dụng.

Chi tiết ở `docs/T2_calibration.md`, mục 4.5.

**Sai sót phát hiện trong số liệu T1.** Cột `cost` trong `scripts/t1/pilot_results.csv` (nhánh `t1-reproduce`) ghi giá trị nhỏ nhất thay vì giá trị trung bình, do lỗi đọc kết quả trong `run_baseline.py`. Các cột tỉ lệ an toàn và phần thưởng không bị ảnh hưởng.

## Phụ lục C. Tái lập kết quả

Hướng dẫn cài đặt và toàn bộ lệnh chạy ở `docs/T2_setup_and_usage.md`. Tóm tắt (từ gốc repo, máy RTX 3080):

```bash
export VENV=/home/mantd/DGPPO/dgppo_env OUTDIR=/home/mantd/DGPPO/t2_runs GPU=0 METHODS=dgppo
SEEDS=0 EPI=32 GRID=configs/t2/sigma0_grid.yaml SWEEP_NAME=check0 bash scripts/t2/run_noise_grid.sh          # A.1
SEEDS=0 GRID=configs/t2/calib_grid.yaml SWEEP_NAME=calib bash scripts/t2/run_noise_grid.sh                    # A.2
SEEDS=0 GRID=configs/t2/calib_fine_grid.yaml SWEEP_NAME=calib_fine bash scripts/t2/run_noise_grid.sh          # A.2
SEEDS="0 1 2" GRID=configs/t2/sigma_grid.yaml SWEEP_NAME=noise_h1 bash scripts/t2/run_noise_grid.sh           # A.3–A.5
python scripts/t2/h1_test.py $OUTDIR/sweeps/<…>_noise_h1/episodes.csv --out $OUTDIR/sweeps/<…>_noise_h1/h1_test.md            # A.6
python scripts/t2/h1_test.py $OUTDIR/sweeps/<…>_noise_h1/episodes.csv --seeds 1 2 --out $OUTDIR/sweeps/<…>_noise_h1/h1_test_seed12.md   # A.7
SEEDS="0 1 2" SWEEP_NAME=shift bash scripts/t2/run_shift_grid.sh                                               # A.8–A.9
python scripts/t2/fill_results_tables.py --runs $OUTDIR/sweeps                                                 # điền Phụ lục A
```
