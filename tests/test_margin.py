import numpy as np

from methods.cm_dgppo.margin import MarginConfig, MarginUpdater


def test_clip_and_sign():
    m = MarginUpdater(MarginConfig(alpha=0.1, eta=0.5, delta_max=0.05))
    m.update(True)
    assert m.current() == [0.05]           # kẹp ở delta_max
    for _ in range(10):
        m.update(False)
    assert m.current() == [0.0]            # kẹp ở 0


def test_float_and_per_agent():
    m = MarginUpdater(MarginConfig(alpha=0.1, eta=0.01, delta_max=1.0, per_agent=True, n_agents=2))
    m.update([0.6, 0.1])
    np.testing.assert_allclose(m.current(), [0.005, 0.0])
    m2 = MarginUpdater(MarginConfig(alpha=0.1, eta=0.01, delta_max=1.0))
    m2.update(np.float32(0.6))
    np.testing.assert_allclose(m2.current(), [0.005], rtol=1e-6)


def test_converges_to_alpha_monotone():
    # P(vi phạm | delta) = 0.5 * exp(-delta / 0.01): giảm đơn điệu theo delta
    rng = np.random.default_rng(0)
    m = MarginUpdater(MarginConfig(alpha=0.05, eta=0.002, delta_max=0.2))
    errs = []
    for _ in range(5000):
        p = 0.5 * np.exp(-m.current()[0] / 0.01)
        e = float(rng.random(64).__lt__(p).mean())
        errs.append(e)
        m.update(e)
    assert abs(np.mean(errs[1000:]) - 0.05) < 0.01
