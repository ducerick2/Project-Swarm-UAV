"""Danh sách seed dùng chung cho mọi thí nghiệm (Giai đoạn 0).

Giai đoạn 1: 5 seed. Giai đoạn 2 (tích hợp): 20 seed.
KHÔNG đổi danh sách này giữa chừng — nếu đổi phải chạy lại toàn bộ.
"""

SEEDS_PHASE1 = [0, 1, 2, 3, 4]
SEEDS_PHASE2 = list(range(20))  # 0..19


def get_seeds(phase: int = 1):
    return SEEDS_PHASE1 if phase == 1 else SEEDS_PHASE2
