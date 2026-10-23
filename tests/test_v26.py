"""主动学习 GP 耦合曲面：拟合精度、批量选点、σ_V 不确定度与对称展开。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool.alsurface import GPSurface, records_from_rows, select_batch, select_spread  # noqa: E402
from mobility_tool.scanflow import PARAM_KEYS  # noqa: E402

CENTER = dict(z=3.6, dx=1.0, dy=0.5, gamma=0.0, pitch=0.0, roll=10.0)
HALF = dict(z=0.4, dx=0.6, dy=0.6, gamma=8.0, pitch=8.0, roll=12.0)


def truth(Q):
    Q = np.atleast_2d(Q)
    u = (Q - np.array([CENTER[k] for k in PARAM_KEYS])) / np.array([HALF[k] for k in PARAM_KEYS])
    h = 80 * np.exp(-1.5 * u[:, 0]) * np.cos(1.2 * u[:, 1] + 0.3) * (1 + 0.2 * u[:, 2]) + 4 * u[:, 5]
    e = -60 * np.exp(-1.4 * u[:, 0]) * np.cos(0.8 * u[:, 2] - 0.2) + 6 * u[:, 1] * u[:, 3]
    return h, e


def sample(n, s, rng):
    c = np.array([CENTER[k] for k in PARAM_KEYS])
    w = np.array([HALF[k] for k in PARAM_KEYS])
    return c + rng.uniform(-1, 1, (n, 6)) * w * s


def test_gp_fit_and_selection():
    rng = np.random.default_rng(3)
    pool = sample(300, 1.0, rng)
    c = np.array([CENTER[k] for k in PARAM_KEYS])
    thermal = c + rng.standard_normal((400, 6)) * np.array([HALF[k] for k in PARAM_KEYS]) * 0.35
    th, te = truth(thermal)
    idx = select_spread(pool, c, HALF, 10)
    assert len(set(idx)) == 10
    for _ in range(4):
        h, e = truth(pool[idx])
        s = GPSurface(records_from_rows(pool[idx], h, e), CENTER, half=HALF, restarts=2)
        new = select_batch(s, pool, thermal, k=10, exclude=idx)
        assert len(set(new)) == 10 and not set(new) & set(idx)
        idx += new
    h, e = truth(pool[idx])
    s = GPSurface(records_from_rows(pool[idx], h, e), CENTER, half=HALF, restarts=2)
    ph, pe = s(thermal, "hole"), s(thermal, "electron")
    assert np.sqrt(((ph - th) ** 2).mean()) < 0.05 * th.std()
    assert abs(ph.std() / th.std() - 1) < 0.03 and abs(pe.std() / te.std() - 1) < 0.03
    u = s.stat_uncertainty(thermal, "hole")
    assert u["std_lo"] <= u["std"] <= u["std_hi"] and u["std_rel"] < 0.05
    assert s.coverage(thermal) > 0.9 and "GP" in s.describe()


def test_active_beats_random_on_sigma():
    rng = np.random.default_rng(7)
    pool = sample(300, 1.0, rng)
    c = np.array([CENTER[k] for k in PARAM_KEYS])
    thermal = c + rng.standard_normal((300, 6)) * np.array([HALF[k] for k in PARAM_KEYS]) * 0.3
    th, _ = truth(thermal)
    idx = select_spread(pool, c, HALF, 8)
    while len(idx) < 24:
        h, e = truth(pool[idx])
        s = GPSurface(records_from_rows(pool[idx], h, e), CENTER, half=HALF, restarts=1)
        idx += select_batch(s, pool, thermal, k=8, exclude=idx)
    h, e = truth(pool[idx])
    al = GPSurface(records_from_rows(pool[idx], h, e), CENTER, half=HALF, restarts=1)
    err_al = np.sqrt(((al(thermal, "hole") - th) ** 2).mean())
    errs = []
    for seed in range(4):
        r = np.random.default_rng(seed).choice(len(pool), 24, replace=False)
        h, e = truth(pool[r])
        g = GPSurface(records_from_rows(pool[r], h, e), CENTER, half=HALF, restarts=1)
        errs.append(np.sqrt(((g(thermal, "hole") - th) ** 2).mean()))
    assert err_al < np.median(errs)


if __name__ == "__main__":
    test_gp_fit_and_selection()
    print("ok test_gp_fit_and_selection")
    test_active_beats_random_on_sigma()
    print("ok test_active_beats_random_on_sigma")
