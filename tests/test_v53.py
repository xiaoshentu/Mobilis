"""Deformation-potential band transport (v1.31.0): fits of C and E1, Bardeen–Shockley against Boltzmann transport
on parabolic bands (3D and 2D), tight-binding bands and velocities, strained cells and the dp CLI."""
import csv
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool import dp  # noqa: E402
from mobility_tool.mobility import KB_EV  # noqa: E402

HB2_ME = 7.619964      # ħ²/m_e in eV·Å²


def parabolic(m, T, dim, n):
    kT = KB_EV * T
    kmax = 7 * math.sqrt(2 * m * kT / HB2_ME)
    ax = (np.arange(n) + 0.5) / n * 2 * kmax - kmax
    grids = np.meshgrid(*([ax] * dim), indexing="ij")
    k = np.stack([g.ravel() for g in grids], 1)
    E = (HB2_ME / (2 * m) * (k ** 2).sum(1))[:, None]
    V = np.zeros((len(k), 1, 3))
    V[:, 0, :dim] = HB2_ME / m * k
    size = (2 * np.pi) ** dim / (2 * kmax) ** dim          # "cell" whose BZ is the sampled box
    return E, V, size


def check_bs_vs_boltzmann():
    for dim, n, C, unit in ((3, 110, 20.0, "GPa"), (2, 900, 120.0, "N/m")):
        for m, T in ((0.8, 300.0), (2.5, 150.0)):
            E, V, size = parabolic(m, T, dim, n)
            mu_b = dp.boltzmann_dp(E, V, size, C, 2.0, T, dim=dim)[0]
            mu_bs = dp.bardeen_shockley(C, 2.0, m, T, dim=dim)
            print(f"  {dim}D m*={m} T={T:.0f}: Boltzmann {mu_b:.4g}, Bardeen–Shockley {mu_bs:.4g} cm²/Vs")
            assert abs(mu_b / mu_bs - 1) < 0.02


def check_fits():
    V0, C, E1 = 700.0, 15.0, -2.3
    e = np.array([-0.01, -0.005, 0, 0.005, 0.01])
    E = -500 + 0.5 * C * dp.GPA_TO_EV_A3 * V0 * e ** 2 + 3e-3 * e ** 3
    edge = -4.8 + E1 * e + 0.5 * e ** 2
    r = dp.fit_axis(e * 100, E, edge, V0, 3)
    print(f"  fit: C {r['C']:.3f} GPa (15), E1 {r['E1']:.3f} eV (−2.3)")
    assert abs(r["C"] / C - 1) < 1e-3 and abs(r["E1"] - E1) < 1e-3
    # 2D formula evaluated by hand in SI units
    mu = dp.bardeen_shockley(128.0, 3.9, 0.48, 300, dim=2)
    e, hb, me, kT = 1.602176634e-19, 1.054571817e-34, 9.1093837015e-31, 1.380649e-23 * 300
    hand = e * hb ** 3 * 128.0 / (kT * (0.48 * me) ** 2 * (3.9 * e) ** 2) * 1e4
    assert abs(mu / hand - 1) < 1e-6


def check_tb_bands():
    """Simple cubic lattice with one site: ε = 2J Σ cos(k a); minimum curvature → m* = ħ²/(2 J a²)."""
    a, J = 5.0, 0.4
    lat = np.eye(3) * a
    edges = []
    for v in np.vstack([lat, -lat]):
        edges.append((0, 0, v, J, 1.0, (0, 0, 0)))
    g = {"n_sites": 1, "edges": edges}
    E, V, vol = dp.tb_bands(g, lat, "electron", (12, 12, 12))
    ks = (np.array(np.meshgrid(*([np.arange(12) / 12] * 3), indexing="ij")).reshape(3, -1).T) * 2 * np.pi / a
    ref = 2 * J * np.cos(ks * a).sum(1)
    vref = -2 * J * a * np.sin(ks * a)
    assert np.abs(E[:, 0] - ref).max() < 1e-10 and np.abs(V[:, 0] - vref).max() < 1e-10
    assert abs(vol - a ** 3) < 1e-9
    # Boltzmann on the full cosine band approaches the parabolic (BS) result with m* = ħ²/(2Ja²) as k_BT/J → 0;
    # the band flattens above the minimum, so μ lies a little below BS, less so at lower T (converged in n)
    m = HB2_ME / (2 * J * a * a)
    ratio = {}
    for T, n in ((300.0, 64), (100.0, 96)):
        E, V, vol = dp.tb_bands(g, lat, "electron", (n, n, n))
        mu_b = dp.boltzmann_dp(E, V, vol, 10.0, 1.5, T)[0]
        mu_bs = dp.bardeen_shockley(10.0, 1.5, m, T)
        ratio[T] = mu_b / mu_bs
        print(f"  cosine band {T:.0f} K, {n}³ k points: Boltzmann {mu_b:.4g}, BS (m* {m:.3f}) {mu_bs:.4g}")
    assert 0.93 < ratio[300.0] < ratio[100.0] < 1.0


def check_cli():
    env = dict(os.environ, MOBILIS_LANG="zh", PYTHONIOENCODING="utf-8")
    cif = os.path.join(ROOT, "tests", "data", "pentacene_RT.cif")
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, "-m", "mobility_tool", "dp-gen", cif, "--strains=-1,-0.5,0,0.5,1",
                            "--axes", "a,b", "--out", f"{d}/pen"], cwd=ROOT, capture_output=True, text=True, env=env)
        print(r.stdout[-300:], r.stderr[-800:])
        assert r.returncode == 0
        from mobility_tool.crystal import read_cif
        c0 = read_cif(cif)
        c1 = read_cif(f"{d}/pen_ap1.0.cif")
        assert abs(c1["cell"][0] / c0["cell"][0] - 1.01) < 1e-6 and len(c1["frac"]) == len(c0["frac"])
        # fill the table with a synthetic DFT answer and fit it
        V0 = abs(np.linalg.det(c0["lattice"]))
        rows = list(csv.reader(open(f"{d}/pen_strain_table.csv")))
        Cs, E1h, E1e = {"a": 12.0, "b": 18.0}, {"a": -1.2, "b": -2.0}, {"a": 1.5, "b": 2.5}
        for r_ in rows[1:]:
            ax, e = r_[0], float(r_[1])
            r_[2] = f"{-800 + 0.5 * Cs[ax] * dp.GPA_TO_EV_A3 * V0 * e * e:.8f}"
            r_[3] = f"{-4.9 + E1h[ax] * e:.6f}"
            r_[4] = f"{-2.9 + E1e[ax] * e:.6f}"
        with open(f"{d}/tab.csv", "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        r = subprocess.run([sys.executable, "-m", "mobility_tool", "dp", f"{d}/tab.csv", "--cif", cif, "--mass", "1.5",
                            "--out", f"{d}/dp.csv"], cwd=ROOT, capture_output=True, text=True, env=env)
        print(r.stdout[-1500:], r.stderr[-1500:])
        assert r.returncode == 0 and os.path.exists(f"{d}/dp.csv")
        res = list(csv.DictReader(open(f"{d}/dp.csv", encoding="utf-8-sig")))
        ra = next(x for x in res if x["axis"] == "a" and x["carrier"] == "hole")
        assert abs(float(ra["C"]) - 12.0) < 0.01 and abs(float(ra["E1_eV"]) + 1.2) < 1e-3
        assert float(ra["mu_BS_given_mass"]) > 0 and float(ra["mu_boltzmann_TB"]) > 0


if __name__ == "__main__":
    check_fits()
    check_bs_vs_boltzmann()
    check_tb_bands()
    check_cli()
    print("test_v53 ok")
