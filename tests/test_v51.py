"""Quantum nuclear tunneling (v1.30.0, tunneling.py): the Fermi-golden-rule rate reproduces Marcus when all modes are
classical and Marcus–Levich–Jortner for one high-frequency mode with k_BT ≪ ħω; it exceeds Marcus and the excess
grows at low temperature; mode files of reorg-decomp are read (S column or λ only); KMC with the FGR rates agrees
with the analytic tensor; compare_methods lists fgr_tensor / kmc_fgr; the tunneling command runs."""
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
CIF = os.path.join(ROOT, "tests", "data", "anthracene_COD5000168.cif")
MODES = [(30, 1.2), (60, 0.6), (120, 0.25), (260, 0.12), (600, 0.05), (1180, 0.06), (1400, 0.12), (1600, 0.08)]


def write_modes(path, modes, with_s=True):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "freq_cm-1", "S", "lambda_meV"] if with_s else ["mode", "freq_cm-1", "lambda_meV"])
        for i, (f, s) in enumerate(modes, 1):
            lam = f * 1.239841984e-4 * s * 1000
            w.writerow([i, f, s, lam] if with_s else [i, f, lam])


def main():
    from mobility_tool.kmc import graph_from_crystal, rate_jortner, rate_marcus, run_kmc
    from mobility_tool.tunneling import CM_TO_EV, FGRRate, read_modes, tensor
    J = 0.01
    f = FGRRate([(20.0, 0.2 / (20 * CM_TO_EV))], 300)                          # classical limit
    for dG in (0.0, -0.1, 0.1):
        assert abs(f.rate(J, dG) / rate_marcus(J, 0.2, dG, 300) - 1) < 5e-3, dG
    f = FGRRate([], 300, lam_cl=0.2)
    assert abs(f.rate(J, 0.05) / rate_marcus(J, 0.2, 0.05, 300) - 1) < 5e-3
    hw, li, lo = 0.18, 0.12, 0.05                                            # MLJ limit
    f = FGRRate([(hw / CM_TO_EV, li / hw)], 300, lam_cl=lo)
    for dG in (0.0, -0.1):
        assert abs(f.rate(J, dG) / rate_jortner(J, li, lo, dG, 300, hw) - 1) < 5e-3, dG
    r = [FGRRate(MODES, T).summary()["ratio_to_marcus_dG0"] for T in (100, 200, 300)]
    assert r[0] > r[1] > r[2] > 1.0, r                                       # tunneling, stronger when cold
    f = FGRRate(MODES, 300, classical_below_cm=200)
    assert f.lam_cl > 0 and abs(f.lam - FGRRate(MODES, 300).lam) < 1e-12
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "n.csv"), os.path.join(d, "i.csv")
        write_modes(a, MODES)
        write_modes(b, MODES, with_s=False)
        ma, mb = read_modes(a), read_modes(b)
        assert len(ma) == len(mb) == len(MODES) and all(abs(x[1] - y[1]) < 1e-9 for x, y in zip(ma, mb))
        from mobility_tool.ceds import CEDS  # noqa: F401
        from mobility_tool.crystal import find_molecules, read_cif
        from mobility_tool.workflow import eht_couplings_for_crystal
        crys = read_cif(CIF)
        mols = find_molecules(crys)
        _, coup = eht_couplings_for_crystal(crys, mols, 4.0)
        g = graph_from_crystal(crys, mols, coup, 4.0, carrier="hole")
        f = FGRRate(MODES, 300)
        mu_t = np.linalg.eigvalsh(tensor(g, f))[::-1]
        k = run_kmc(g, f.lam, 300, method="fgr", fgr=f, walkers=600, steps=2500, seed=3)
        assert abs(k["mu_principal"][0] / mu_t[0] - 1) < 0.1, (k["mu_principal"], mu_t)
        from mobility_tool.compare import compare_methods
        res = compare_methods(g, f.lam, 300, methods=("marcus_tensor", "fgr_tensor", "kmc_fgr"), fgr=f, walkers=200,
                              steps=800)
        keys = [row["key"] for row in res["rows"]]
        assert keys == ["marcus_tensor", "fgr_tensor", "kmc_fgr"], keys
        assert res["rows"][1]["mu"] > res["rows"][0]["mu"]
        env = dict(os.environ, MOBILIS_LANG="en", PYTHONIOENCODING="utf-8")
        out = os.path.join(d, "t.csv")
        p = subprocess.run([sys.executable, "-m", "mobility_tool", "tunneling", "--modes", a, b, "--temps", "150,300",
                            "--cif", CIF, "--out", out], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           env=env, timeout=900)
        assert p.returncode == 0, p.stderr
        assert "k_FGR/k_Marcus" in p.stdout and os.path.exists(out), p.stdout
        rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
        assert float(rows[0]["mu_fgr"]) > float(rows[1]["mu_fgr"])            # band-like apparent T dependence
    print(f"limits ok; k_FGR/k_Marcus at 100/200/300 K = {r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f}; KMC/tensor = "
          f"{k['mu_principal'][0] / mu_t[0]:.3f}")


if __name__ == "__main__":
    main()
    print("test_v51 ok")
