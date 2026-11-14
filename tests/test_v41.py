"""photocond: limits of the Braun field factor, the steady-state balance, the μ scaling of Δσ for small yields,
the analytic pulse decay against a numerical integration, and the photocond command."""
import os as _os
_os.environ.setdefault("MOBILIS_LANG", "zh")
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool import photocond as P  # noqa: E402

REF = dict(wavelength=670.0, alpha=1e5, thickness=50.0, refl=0.1, T=300.0, mu_e=16.32, mu_h=6.67, eps_r=4.3,
           eb=0.5, tau_s=80e-15)


def test_field_factor():
    f, b = P.onsager_factor([0.0, 1.0, 1e5], 300.0, 4.3)
    assert f[0] == 1.0 and abs(f[1] - (1 + b[1])) < 1e-9
    # series of J1(2√(−2b))/√(−2b) = Σ (2b)^k / (k! (k+1)!)
    import math
    s = sum((2 * b[2]) ** k / (math.factorial(k) * math.factorial(k + 1)) for k in range(60))
    assert abs(f[2] / s - 1) < 1e-9, (f[2], s)
    print("Braun field factor OK", f)


def test_steady_state():
    r = P.steady_state(intensity=np.array([1e-3, 0.1, 10.0]), field=1e5, tau_t=1e-6, **REF)
    bal = r["k_r"] * r["n"] ** 2 + r["n"] / 1e-6
    assert np.allclose(bal, r["G"], rtol=1e-9)
    r0 = P.steady_state(intensity=0.1, field=1e5, **REF)
    r1 = P.steady_state(intensity=0.1, field=1e5, **dict(REF, mu_e=REF["mu_e"] / 2, mu_h=REF["mu_h"] / 2))
    assert r0["eta"] < 1e-3 and abs(r1["dsigma"] / r0["dsigma"] - 0.5) < 1e-3   # Δσ ∝ μ when η ≪ 1
    i = np.array([1e-2, 1.0])
    d = P.steady_state(intensity=i, field=1e5, **REF)["dsigma"]
    assert abs(np.log(d[1] / d[0]) / np.log(100) - 0.5) < 1e-9                  # Langevin: Δσ ∝ I^0.5
    print("steady state OK")


def test_pulse():
    from scipy.integrate import solve_ivp
    t = np.logspace(-10, -5, 30)
    d, n0 = P.pulse_decay(t, fluence=1e-6, field=1e5, tau_t=2e-7, **REF)
    kr = P.langevin(REF["mu_e"], REF["mu_h"], REF["eps_r"])
    sol = solve_ivp(lambda _, n: -kr * n * n - n / 2e-7, (0, t[-1]), [n0], t_eval=t, rtol=1e-10, atol=1.0,
                    method="LSODA")
    dn = P.E * sol.y[0] * (REF["mu_e"] + REF["mu_h"]) * 1e-4
    assert np.allclose(d, dn, rtol=1e-5), np.max(abs(d / dn - 1))
    print("pulse decay OK")


def test_cli():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "pc.csv")
        p = subprocess.run([sys.executable, "-m", "mobility_tool", "photocond", "--mu-e", "16.32", "--mu-h", "6.67",
                            "--eps", "4.3", "--et", "2.35", "--eopt", "1.85", "--tau-s", "0.08", "--wavelength", "670",
                            "--alpha", "1e5", "--thickness", "50", "--refl", "0.1", "--intensity", "0.01 0.1",
                            "--field", "1e5", "--fluence", "1", "--out", out], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8")
        assert p.returncode == 0, p.stdout + p.stderr
        rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
        assert len(rows) == 2 and os.path.exists(os.path.join(tmp, "pc_pulse.csv"))
        ref = P.steady_state(intensity=0.1, field=1e5, **REF)
        assert abs(float(rows[1]["dsigma_Scm"]) / (float(ref["dsigma"]) / 100) - 1) < 1e-3
        p = subprocess.run([sys.executable, "-m", "mobility_tool", "photocond", "--mu-e", "1", "--mu-h", "1",
                            "--eps", "3", "--wavelength", "500", "--alpha", "1e5"], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8")
        assert p.returncode != 0 and "--eb" in (p.stdout + p.stderr)
    print("photocond command OK")


if __name__ == "__main__":
    test_field_factor()
    test_steady_state()
    test_pulse()
    test_cli()
    print("test_v41 ok")
