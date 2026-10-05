# Lược đồ cấu hình (Giai đoạn 0 — cố định)

Mọi thí nghiệm khai báo bằng YAML theo khóa dưới đây. Dùng chung để đổi
môi trường / mức nhiễu / phương pháp mà không sửa code các module khác.

| Khóa | Kiểu | Giá trị |
|---|---|---|
| `method` | str | `dgppo`, `informarl`, `informarl_lag`, `cm_dgppo`, `fixed_margin`, `domain_rand` (checkpoint DGPPO dùng `informarl_lagr`) |
| `env` | str | `LidarSpread`, `LidarLine`, `MPESpread`, `CEFC-lite` |
| `N` | int | 3, 5, 7 |
| `seed` | int | lấy từ `analysis/seeds.py` |
| `noise.sigma_w` | float | std nhiễu động học, cộng vào **vận tốc** agent sau mỗi bước rồi clip (env LiDAR: \|v\| ≤ 0.5) |
| `noise.sigma_v` | float | std nhiễu cảm biến, cộng vào **vị trí** quan sát của agent và điểm hit LiDAR (đơn vị độ dài; r = 0.05); vận tốc quan sát giữ sạch |
| `noise.shift_type` | str | `none`, `n_agents`, `n_obs`: loại dịch chuyển lúc test |
| `noise.shift_level` | int | N lúc test (`n_agents`) hoặc số vật cản lúc test (`n_obs`) |
| `eval.ckpt` | str | thư mục checkpoint (`config.yaml` + `models/`) |
| `eval.n_episodes` | int | số episode đánh giá (mặc định 256) |
| `eval.test_seed` | int | seed sinh khóa episode (mặc định 1234, như `test.py`) |
| `eval.stochastic` | bool | `false` = policy tất định; `true` = lấy mẫu từ policy |

Mức σ dùng chung của nhóm (sau hiệu chỉnh): `configs/t2/sigma_grid.yaml`.
| `margin.alpha` | float | mức vi phạm mục tiêu (vd 0.05) |
| `margin.eta` | float | bước cập nhật biên |
| `margin.delta_max` | float | biên tối đa |
| `margin.per_agent` | bool | biến thể (d) |
| `margin.window_steps` | int | biến thể (e): 0 = theo episode |
| `logging.results_csv` | str | đường dẫn CSV kết quả |
| `logging.backend` | str | `wandb`, `tensorboard`, `none` |
