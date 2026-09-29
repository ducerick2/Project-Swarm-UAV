# Project-Swarm-UAV

Học tăng cường đa agent an toàn cho swarm UAV dưới bất định (Safe MARL / CM-DGPPO).
Bài nền: **DGPPO** (ICLR 2025, JAX) — https://github.com/MIT-REALM/dgppo

## Nền tảng chung (Giai đoạn 0 — cố định, không đổi trong suốt dự án)

| Thành phần | Vị trí | Ghi chú |
|---|---|---|
| Mã nguồn nền DGPPO | `third_party/dgppo` | Pin tại 1 commit cố định (submodule) |
| Lớp bao nhiễu | `envs/noise_wrapper.py` | Đặc tả `NoiseWrapper(env, sigma_w, sigma_v)` |
| Lược đồ cấu hình | `configs/` | YAML: method, env, N, sigma, seed |
| Định dạng kết quả | `analysis/logging_csv.py` | CSV cột cố định |
| Chỉ số + seed dùng chung | `analysis/metrics.py`, `analysis/seeds.py` | Theo định nghĩa DGPPO |

## Cấu trúc thư mục

```
third_party/dgppo/   # submodule DGPPO (pin commit)
envs/                # noise_wrapper (T2 interface), cefc_lite (T5)
methods/cm_dgppo/    # bộ cập nhật biên delta (T3)
configs/             # lược đồ + config mẫu
analysis/            # metrics, csv logger, seeds, stats (T4)
scripts/             # train.py, eval.py
results/             # CSV/artifact (gitignore, dùng DVC/W&B)
```

## Phân công mạch ↔ nhánh

| Nhánh | Thành viên | Mạch |
|---|---|---|
| `main` | cả nhóm | nền tảng chung (protected) |
| `t1-reproduce` | A | Tái hiện DGPPO/InforMARL/InforMARL-Lag + runtime |
| `t2-robustness` | B | NoiseWrapper thật + dịch chuyển phân phối + H1 |
| `t3-cm-dgppo` | C | Biên an toàn thích nghi delta + biến thể (a)-(e) |
| `t4-theory-stats` | D | Chứng minh cận + công cụ thống kê |
| `t5-casestudy` | E | Môi trường CEFC-lite + demo + tổng quan |

## Quy trình

1. Nền tảng chung land lên `main` TRƯỚC khi rẽ nhánh.
2. Mỗi mạch làm trên nhánh riêng, mở PR vào `main`, có người phản biện (theo bảng trong báo cáo).
3. `git pull --rebase origin main` mỗi ngày để tránh trôi nhánh.
4. KHÔNG commit checkpoint/log/CSV kết quả — xem `.gitignore`.

## Môi trường

DGPPO chạy JAX/GPU. Cả nhóm thống nhất version JAX + CUDA (ghi trong `requirements.txt`)
để đảm bảo tái hiện. Mọi lần chạy ghi lại: seed, commit Git, config, log (W&B/TensorBoard).
