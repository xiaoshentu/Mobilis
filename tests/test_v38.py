"""v1.25: probabilistic transport-regime assignment (diagnose --uq)."""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool.diagnose import EHT_ERR, diagnose, format_report, regime_probabilities  # noqa: E402

D = os.path.join(ROOT, "tests", "data")


def j_for_ratio(ratio, lam, carrier):
    """Scaled EHT J_max (meV) whose bias-corrected median gives 2|J|/λ = ratio."""
    m, _ = EHT_ERR[carrier]
    return ratio * lam * 1000 / 2 * np.exp(m)


def test_limits_and_normalization():
    for car in ("hole", "electron"):
        far_hop = regime_probabilities([j_for_ratio(0.3, 0.2, car), 5.0], 0.2, 0.5, car, n=20000)
        far_tl = regime_probabilities([j_for_ratio(3.0, 0.2, car)], 0.2, 0.5, car, n=20000)
        band = regime_probabilities([j_for_ratio(3.0, 0.2, car)], 0.2, 0.03, car, n=20000)
        for u in (far_hop, far_tl, band):
            assert abs(sum(u["probabilities"].values()) - 1) < 1e-12
        assert far_hop["probabilities"]["hopping"] > 0.999 and far_hop["decided"]
        assert far_tl["probabilities"]["transient localization"] > 0.999
        assert band["probabilities"]["band-like"] > 0.999 and band["most_probable"] == "band-like"


def test_threshold_is_half_and_width_follows_calibration():
    # at a median 2|J|/λ of exactly 1 the two regimes are equally likely
    for car in ("hole", "electron"):
        u = regime_probabilities([j_for_ratio(1.0, 0.15, car)], 0.15, 0.4, car, n=200000, seed=1)
        assert abs(u["probabilities"]["hopping"] - 0.5) < 0.01 and not u["decided"]
        assert abs(u["two_J_over_lambda_median"] - 1.0) < 0.01
    # at 2|J|/λ = 0.8 holes (EHT error 9 %) are decided, electrons (30 %) are not
    h = regime_probabilities([j_for_ratio(0.8, 0.15, "hole")], 0.15, 0.4, "hole", n=50000, lam_rel=0.05)
    e = regime_probabilities([j_for_ratio(0.8, 0.15, "electron")], 0.15, 0.4, "electron", n=50000, lam_rel=0.05)
    assert h["decided"] and h["probabilities"]["hopping"] > 0.95
    assert not e["decided"] and 0.6 < e["probabilities"]["hopping"] < 0.9
    assert e["dominant"].startswith("coupling")
    # holes with a 10 % uncertainty of λ: λ dominates; with 1 %: the coupling
    assert regime_probabilities([50.0], 0.15, 0.4, "hole", lam_rel=0.10)["dominant"] == "reorganization energy"
    assert regime_probabilities([50.0], 0.15, 0.4, "hole", lam_rel=0.01)["dominant"].startswith("coupling")


def test_reproducible_with_seed():
    a = regime_probabilities([80.0], 0.15, 0.3, "electron", n=5000, seed=7)
    b = regime_probabilities([80.0], 0.15, 0.3, "electron", n=5000, seed=7)
    assert a == b


def test_diagnose_uq_and_cli():
    d = diagnose(os.path.join(D, "naphthalene_benchmark.cif"), 0.187, 0.255, phonons=False, uq=True, n_samples=5000)
    for c in d["carriers"].values():
        p = c["uq"]["probabilities"]
        assert p["hopping"] > 0.99 and p["band-like"] == 0.0
        lo, hi = c["uq"]["two_J_over_lambda_90"]
        assert lo < c["uq"]["two_J_over_lambda_median"] < hi
    assert d["uq_model"]["samples"] == 5000 and "[uq]" in format_report(d)
    r = subprocess.run([sys.executable, "-m", "mobility_tool", "diagnose", os.path.join(D, "naphthalene_benchmark.cif"),
                        "--lam-h", "0.187", "--lam-e", "0.255", "--no-phonons", "--uq", "--uq-samples", "2000"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert r.returncode == 0 and "P(hopping)" in r.stdout


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print(name, "ok")
