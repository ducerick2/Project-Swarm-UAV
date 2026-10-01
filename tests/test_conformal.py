import numpy as np

from methods.cm_dgppo.conformal import ConformalConfig, OnlineConformal


def run(scores_fn, T=3000, alpha=0.1, gamma=0.02, seed=0):
    rng = np.random.default_rng(seed)
    c = OnlineConformal(ConformalConfig(alpha=alpha, gamma=gamma, window=2000, samples=256), seed=seed)
    errs = [c.update(scores_fn(rng, t))["err"] for t in range(T)]
    return c, np.array(errs)


def test_iid_long_run_coverage():
    c, errs = run(lambda rng, t: rng.normal(size=64))
    assert abs(errs.mean() - 0.1) < 0.01
    assert abs(c.q - 1.2816) < 0.25  # q dao động quanh phân vị 0.9 của N(0, 1)


def test_shift_long_run_coverage():
    # thang điểm số tăng gấp 3 giữa chừng: tần suất miscoverage dài hạn vẫn về alpha
    _, errs = run(lambda rng, t: rng.normal(size=64) * (1.0 if t < 1500 else 3.0))
    assert abs(errs.mean() - 0.1) < 0.01
    assert abs(errs[-1000:].mean() - 0.1) < 0.02


def test_aci_bound_holds_adversarial():
    # cận Gibbs-Candès đúng với mọi chuỗi (ở đây điểm số đổi dấu theo chu kỳ)
    alpha, gamma, T = 0.1, 0.05, 2000
    _, errs = run(lambda rng, t: rng.normal(size=32) + (5.0 if (t // 100) % 2 else -5.0), T=T, alpha=alpha, gamma=gamma)
    bound = (max(alpha, 1 - alpha) + gamma) / (gamma * T)
    assert abs(errs.mean() - alpha) <= bound + 1e-9


def test_zero_scores_keep_zero_margin():
    # không nhiễu => điểm số 0 => q giữ 0 (CM-DGPPO trùng DGPPO)
    c = OnlineConformal(ConformalConfig(alpha=0.05, gamma=0.01))
    for _ in range(50):
        c.update(np.zeros(100))
    assert c.q == 0.0


def test_clip_and_state_roundtrip():
    c = OnlineConformal(ConformalConfig(alpha=0.05, gamma=0.5, q_max=2.0, q_min=-2.0))
    for _ in range(5):
        c.update(np.full(10, 100.0))  # luôn miscover => alpha_t <= 0 => q = q_max
    assert c.q == 2.0 and c.n_clipped > 0
    d = c.state_dict()
    c2 = OnlineConformal(ConformalConfig(alpha=0.05, gamma=0.5, q_max=2.0, q_min=-2.0))
    c2.load_state_dict(d)
    assert c2.q == c.q and c2.alpha_t == c.alpha_t
