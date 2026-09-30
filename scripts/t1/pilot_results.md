# T1 — Kết quả pilot baseline (Pha 1)

**Thiết lập:** 3 phương pháp × 2 môi trường × N=3 × seed 0 × 200k steps.
Phần cứng: 1× RTX 5090 (GPU1). DGPPO pin tại commit `51b3b11`. JAX 0.6.2 + shim.
Ngày chạy: 2026-09-29 → 2026-10-01 (~40h tổng, tuần tự).

## Bảng kết quả

| Phương pháp | Môi trường | Safety rate ↑ | Reward ↑ | Cost (h) ≤0 | Train |
|---|---|---|---|---|---|
| **DGPPO** | LidarSpread | **1.000** | -0.885 | -0.649 | 7.1h |
| **DGPPO** | LidarLine | **0.990** | -0.280 | -0.655 | 6.8h |
| InforMARL-Lagr | LidarSpread | 0.844 | -0.189 | -0.718 | 6.8h |
| InforMARL-Lagr | LidarLine | 0.865 | -0.168 | -0.653 | 6.6h |
| InforMARL | LidarSpread | 0.656 | -0.217 | -0.646 | 5.2h |
| InforMARL | LidarLine | 0.771 | -0.190 | -0.652 | 5.2h |

(Chỉ số từ `test.py`, 32 episode đánh giá. `safety_rate` = tỉ lệ quỹ đạo không vi phạm;
`reward` = âm của chi phí nhiệm vụ, càng gần 0 càng tốt; `cost` = giá trị ràng buộc h, ≤0 là thỏa.)

## Nhận xét (khớp xu hướng bài gốc DGPPO)

1. **Thứ tự an toàn 3 bậc, nhất quán trên cả 2 môi trường:**
   DGPPO (CBF cứng, ~99–100%) > InforMARL-Lagrangian (ràng buộc mềm, ~84–86%) >
   InforMARL (không ràng buộc, ~66–77%).
2. **Đánh đổi an toàn ↔ hiệu năng:** phương pháp an toàn hơn có reward thấp hơn chút
   (DGPPO reward thấp nhất nhưng safety cao nhất). Đúng như DGPPO chứng minh.
3. **Runtime:** DGPPO và Lagrangian chậm hơn InforMARL thuần (~7h vs ~5h) do có thêm
   critic/CBF; throughput ~8 it/s trên 1×5090.

## Hạn chế của pilot & bước tiếp

- Chỉ **1 seed** → chưa có khoảng tin cậy. Mở rộng 3–5 seed (T4 lo phần thống kê) để báo cáo.
- Chưa phân tích runtime chi tiết N=3,5,7 (steps/s, hội tụ, RAM GPU, thời gian suy luận).
- Dữ liệu thô: `scripts/t1/pilot_results.csv` (+ checkpoint ở `/data/ducbm3/dgppo_runs/`, không commit).
