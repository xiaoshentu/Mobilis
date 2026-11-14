"""Leave-one-out calibration of the GP intervals (ceds al --al-calib loo): the closed-form leave-one-group-out
residuals equal an explicit refit with fixed hyperparameters (constant mean and EHT-type baseline); symmetry images
of one DFT point are left out together; a well-specified GP gives c ≈ 1 and a too-narrow one c > 1; the factor
enters stat_uncertainty."""
import os as _os
_os.environ.setdefault("MOBILIS_LANG", "zh")
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool.alsurface import GPSurface, _GP  # noqa: E402


def brute(gp, X, y, H, grp):
    th, out = gp.theta, []
    for g in np.unique(grp):
        B = np.flatnonzero(grp == g)
        keep = np.setdiff1d(np.arange(len(X)), B)
        i = B[0]
        K = gp._k(X[keep], X[keep], th) + (gp.noise ** 2 + 1e-8) * np.eye(len(keep))
        Ki = np.linalg.inv(K)
        Hk = H[keep]
        A = Hk.T @ Ki @ Hk
        beta = np.linalg.solve(A, Hk.T @ Ki @ y[keep])
        ks = gp._k(X[[i]], X[keep], th)[0]
        m = H[i] @ beta + ks @ Ki @ (y[keep] - Hk @ beta)
        R = H[i] - ks @ Ki @ Hk
        v = gp.signal ** 2 - ks @ Ki @ ks + R @ np.linalg.solve(A, R) + gp.noise ** 2 + 1e-8
        out.append((y[i] - m) / np.sqrt(v))
    return np.array(out)


def test_closed_form():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (30, 3))
    X = np.r_[X, -X[:10]]
    grp = np.r_[np.arange(30), np.arange(10)]
    b = np.sin(2 * X[:, 0]) + X[:, 1]
    y = 2.0 * b + 0.3 * np.cos(3 * X[:, 2]) + 0.05 * rng.standard_normal(len(X))
    for H in (np.c_[np.ones(len(X)), b], None):
        gp = _GP(X, y, H=H, restarts=1)
        z = gp.loo(grp)
        zb = brute(gp, X, y, np.ones((len(X), 1)) if H is None else H, grp)
        assert len(z) == 30 and np.max(np.abs(z - zb)) < 1e-4, np.max(np.abs(z - zb))
    print("closed-form LOO = explicit refit OK")


def test_factor():
    rng = np.random.default_rng(3)
    keys = ("z", "dx", "dy", "gamma", "pitch", "roll")
    center = dict(z=3.5, dx=0.0, dy=0.0, gamma=0.0, pitch=0.0, roll=0.0)
    half = dict(z=0.5, dx=0.8, dy=0.8, gamma=10.0, pitch=10.0, roll=10.0)

    def recs(n, noise):
        R = []
        for _ in range(n):
            q = {k: center[k] + half[k] * rng.uniform(-1, 1) for k in keys}
            v = 40 * np.exp(-(q["z"] - 3.5)) * np.cos(1.2 * q["dx"]) + 5 * q["dy"]
            R.append(dict(q, hole=v + noise * rng.standard_normal(), electron=0.5 * v + noise * rng.standard_normal()))
        return R
    good = GPSurface(recs(60, 0.2), center, half=half, restarts=2)
    c_good = good.loo_factor(floor=0.0)
    assert 0.6 < c_good < 1.6, c_good
    # images: with a C2 operation every DFT point appears twice; both copies are left out together
    s2 = GPSurface(recs(40, 0.2), center, half=half, ops=("E", "C2x"), restarts=1)
    assert len(s2.loo_z("hole")) <= 40 < s2.n
    # a GP whose noise is pinned too small for noisy data is over-confident: c_LOO > 1
    bad = GPSurface(recs(60, 3.0), center, half=half, restarts=2)
    for g in bad.gp.values():
        g.theta[-1] = np.log(0.05)
        g._factor()
    bad._loo_c = None
    assert bad.loo_factor(floor=0.0) > 1.5 * c_good
    Q = np.array([[center[k] + 0.3 * half[k] * rng.standard_normal() for k in keys] for _ in range(200)])
    good.ci_inflate = 1.0
    u1 = good.stat_uncertainty(Q, "hole")
    good.ci_inflate = "loo"
    good._loo_c = 2.5
    u2 = good.stat_uncertainty(Q, "hole")
    assert abs(u2["std_rel"] / u1["std_rel"] - 2.5) < 1e-6
    print(f"c_LOO: well-specified {c_good:.2f}, over-confident {bad.loo_factor(floor=0.0):.2f}; factor applied OK")
    # report: σ_V half width scales with c; ci_inflate restored
    from mobility_tool.alsurface import calib_report
    good.ci_inflate, good._loo_c = None, None
    rows, txt = calib_report({1: good}, {1: Q})
    r = rows[0]
    assert abs(r["rel_hole_2.5"] / r["rel_hole_1.5"] - 2.5 / 1.5) < 1e-6 and good.ci_inflate is None
    assert abs(r["c_loo"] - c_good) < 1e-9 and "c_LOO" in txt
    src = open(os.path.join(ROOT, "mobility_tool", "cli.py"), encoding="utf-8").read()
    assert '"--al-calib-report"' in src and "calib_report(surf, thermal)" in src


if __name__ == "__main__":
    test_closed_form()
    test_factor()
    print("test_v42 ok")
