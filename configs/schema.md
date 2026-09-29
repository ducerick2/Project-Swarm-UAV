# Lược đồ cấu hình (Giai đoạn 0 — cố định)

Mọi thí nghiệm khai báo bằng YAML theo khóa dưới đây. Dùng chung để đổi
môi trường / mức nhiễu / phương pháp mà không sửa code các module khác.

| Khóa | Kiểu | Giá trị |
|---|---|---|
| `method` | str | `dgppo`, `informarl`, `informarl_lag`, `cm_dgppo`, `fixed_margin`, `domain_rand` |
| `env` | str | `LidarSpread`, `LidarLine`, `MPESpread`, `CEFC-lite` |
| `N` | int | 3, 5, 7 |
| `seed` | int | lấy từ `analysis/seeds.py` |
| `noise.sigma_w` | float | nhiễu động học |
| `noise.sigma_v` | float | nhiễu cảm biến |
| `noise.shift` | float | mức dịch chuyển phân phối |
| `margin.alpha` | float | mức vi phạm mục tiêu (vd 0.05) |
| `margin.eta` | float | bước cập nhật biên |
| `margin.delta_max` | float | biên tối đa |
| `margin.per_agent` | bool | biến thể (d) |
| `margin.window_steps` | int | biến thể (e): 0 = theo episode |
| `logging.results_csv` | str | đường dẫn CSV kết quả |
| `logging.backend` | str | `wandb`, `tensorboard`, `none` |
