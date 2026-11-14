"""exciton: INDO/S–CIS transition charges (benzene symmetry, dipole and oscillator strength), TrESP → point-dipole
limit, Gaussian transition-charge reader, Davydov matrix, diffusion tensor against KMC, MD statistics of a rigid
crystal, and the exciton command on anthracene."""
import os as _os
_os.environ.setdefault("MOBILIS_LANG", "zh")
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool import exciton as EX  # noqa: E402

CIF = os.path.join(ROOT, "tests", "data", "anthracene_COD5000168.cif")


def benzene():
    sym, X = [], []
    for k in range(6):
        a = math.radians(60 * k)
        sym.append("C"); X.append([1.397 * math.cos(a), 1.397 * math.sin(a), 0.0])
    for k in range(6):
        a = math.radians(60 * k)
        sym.append("H"); X.append([2.481 * math.cos(a), 2.481 * math.sin(a), 0.0])
    return sym, np.array(X)


def test_cis_benzene():
    sym, X = benzene()
    r = EX.indo_cis(sym, X, n_states=4, n_occ=3, n_virt=3)
    st = r["states"]
    assert st[0]["f"] < 1e-6 and abs(st[0]["q"].sum()) < 1e-10                   # B2u: dipole-forbidden
    bright = [s for s in st if s["f"] > 0.1]
    assert len(bright) == 2 and abs(bright[0]["E_eV"] - bright[1]["E_eV"]) < 1e-6   # E1u: degenerate pair
    s = bright[0]
    mu_au = np.linalg.norm(s["q"] @ X) / 0.529177
    assert abs(s["f"] - 2 / 3 * s["E_eV"] / 27.2114 * mu_au ** 2) < 1e-9
    assert 4.0 < st[0]["E_eV"] < 5.5 and 6.0 < s["E_eV"] < 8.0
    print("INDO/S–CIS benzene:", [round(x["E_eV"], 2) for x in st], "f", [round(x["f"], 3) for x in st])


def test_tresp_dipole_limit():
    sym, X = benzene()
    q = EX.indo_cis(sym, X, n_states=4, n_occ=3, n_virt=3)["states"][2]["q"]
    mu = q @ X * EX.DEBYE
    surf = EX.TrESPSurface(X, q)
    err = []
    for z in (20.0, 40.0, 80.0):
        v = surf.eval_one([z, 0, 0, 0, 0, 0])["exciton"]
        err.append(abs(v / EX.point_dipole(mu, mu, [0, 0, z]) - 1))
    assert err[1] < 5e-3 and err[2] < err[1] / 3 and err[1] < err[0] / 3, err      # deviation ∝ 1/R²
    q2 = EX.rescale(q, X, 2.0)
    assert abs(np.linalg.norm(q2 @ X) * EX.DEBYE - 2.0) < 1e-9
    print("TrESP → point dipole:", [f"{e:.1e}" for e in err])


def test_reader():
    text = """ Excited State   1:  Singlet-B2U  ...
 ESP charges:
               1
     1  C    0.123000
     2  C   -0.123000
     3  H    0.010000
 Sum of ESP charges =   0.01000
"""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.log")
        open(p, "w").write(text)
        sym, q = EX.read_gaussian_transition_charges(p, n_atoms=3)
        assert sym == ["C", "C", "H"] and np.allclose(q, [0.123, -0.123, 0.01])
        g = os.path.join(tmp, "t.gjf")
        EX.write_tresp_gjf(g, *benzene(), state=1)
        assert "Density=(Transition=1)" in open(g).read()
    print("Gaussian reader / writer OK")


def test_crystal_kmc_md():
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    from mobility_tool.rigidmd import RigidCrystalMD
    c = read_cif(CIF)
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    Y = np.asarray(ce.Y, float)
    r = EX.indo_cis(ce.sym, Y, n_states=3, n_occ=8, n_virt=8)
    q = EX.neutralize(r["states"][1]["q"])
    rows = EX.lattice_couplings(ce, Y, q, 0.39, r_max=20.0)
    w, H = EX.davydov(rows, 2)
    assert np.allclose(H, H.T) and len(w) == 2
    D = EX.diffusion_tensor(rows, 2, 0.15, 300.0)
    L = EX.diffusion_length(D, 1e-10)
    km = EX.kmc_diffusion(rows, 2, 0.15, 300.0, 1e-10, n_walkers=3000, seed=3)
    assert abs(km["msd_cm2"] / (2 * np.trace(D) * 1e-10) - 1) < 4 * km["msd_err"] / km["msd_cm2"] + 0.03
    # MD statistics of an unperturbed crystal: frames equal to the initial structure → σ = 0, ⟨V⟩ = V0
    md = RigidCrystalMD(ce.crys, ce.frames, (ce.sym, ce.Y), (2, 2, 2))
    fr = {"c": np.repeat(md.c0[None], 3, 0), "Q": np.repeat(md.Q0[None], 3, 0)}
    st = EX.md_coupling_stats(ce, md, fr, Y, q, 0.39, recenter=False)
    for (i, j, t), v in st.items():
        assert v["std"] < 1e-8 and abs(v["mean"] - v["V0"]) < 1e-6
        ref = next(x["V"] for x in rows if x["i"] == i and x["j"] == j and x["cell"] == t)
        assert abs(ref - v["V0"]) < 1e-6
    print(f"anthracene: Davydov {w[1] - w[0]:.1f} meV, L3 {L['L3_nm']:.1f} nm, KMC {km['L3_nm']:.1f} nm; MD stats OK")


def test_cli():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "x.json")
        p = subprocess.run([sys.executable, "-m", "mobility_tool", "exciton", "--cif", CIF, "--n-refr", "1.6",
                            "--lambda", "0.15", "--tau", "0.1", "--kmc", "300", "--r-max", "20", "--out", out],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        assert p.returncode == 0, p.stdout + p.stderr
        d = json.load(open(out, encoding="utf-8"))
        assert len(d["davydov_meV"]) == 2 and d["D_static"]["L3_nm"] > 0 and "kmc_static" in d
        g = os.path.join(tmp, "a.gjf")
        p = subprocess.run([sys.executable, "-m", "mobility_tool", "exciton", "--cif", CIF, "--write-gjf", g],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        assert p.returncode == 0 and os.path.exists(g)
    print("exciton command OK")


if __name__ == "__main__":
    test_cis_benzene()
    test_tresp_dipole_limit()
    test_reader()
    test_crystal_kmc_md()
    test_cli()
    print("test_v43 ok")
