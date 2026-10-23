"""波函数分析 (基函数归一化与正交、Mulliken / Mayer、密度积分、cube 读写、ESP 表面、NBO 读取)。"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from fake_fchk import _arr, write_fake_fchk  # noqa: E402
from mobility_tool.gaussian_io import read_xyz  # noqa: E402
from mobility_tool.wavefunction import Wavefunction, esp_surface, read_cube, read_nbo, write_cube  # noqa: E402


def test_basis_normalization():
    """单中心：s / p / sp / 6D / 5D / 10F / 7F 全部归一化；5D、7F 之间正交；6D 中 xx·yy = 1/3。"""
    types = [0, 1, -1, 2, -2, 3, -3]
    nbf = 36
    L = ["one-center", "SP RHF", f"{'Number of atoms':<40s}   I     {1:12d}", f"{'Number of alpha electrons':<40s}   I     {1:12d}",
         f"{'Number of basis functions':<40s}   I     {nbf:12d}", f"{'Number of independent functions':<40s}   I     {nbf:12d}"]
    L += _arr("Atomic numbers", "I", [1]) + _arr("Current cartesian coordinates", "R", [0, 0, 0])
    L += _arr("Shell types", "I", types) + _arr("Number of primitives per shell", "I", [1] * 7)
    L += _arr("Shell to atom map", "I", [1] * 7) + _arr("Primitive exponents", "R", [0.8] * 7)
    L += _arr("Contraction coefficients", "R", [1.0] * 7) + _arr("P(S=P) Contraction coefficients", "R", [0, 0, 1.0, 0, 0, 0, 0])
    L += _arr("Alpha Orbital Energies", "R", np.linspace(-1, 1, nbf)) + _arr("Alpha MO coefficients", "R", np.eye(nbf).ravel())
    with tempfile.TemporaryDirectory() as d:
        Path(d, "o.fchk").write_text("\n".join(L) + "\n")
        w = Wavefunction(Path(d, "o.fchk"))
    h = 0.15
    ax = np.arange(-7, 7 + h / 2, h)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    S = np.zeros((nbf, nbf))
    for s0 in range(0, len(pts), 200000):
        phi = w.basis_values(pts[s0:s0 + 200000])
        S += phi @ phi.T * h ** 3
    assert np.abs(np.diag(S) - 1).max() < 1e-6
    assert np.abs(S[14:19, 14:19] - np.eye(5)).max() < 1e-6 and np.abs(S[29:36, 29:36] - np.eye(7)).max() < 1e-6
    assert abs(S[8, 9] - 1 / 3) < 1e-6
    print("基函数 OK: s/p/sp/6D/5D/10F/7F 归一化，5D、7F 正交，6D xx·yy = 1/3")


def test_population_density_cube():
    with tempfile.TemporaryDirectory() as d:
        write_fake_fchk(f"{d}/h2.fchk", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
        w = Wavefunction(f"{d}/h2.fchk")
        r = w.mulliken_mayer()
        assert abs(r["bond_orders"][0, 1] - 1) < 1e-8 and np.abs(r["charges"]).max() < 1e-8
        g = w.grid(spacing_A=0.04, margin_A=3.2)
        n = w.density_values(w.grid_points(g)).sum() * abs(np.linalg.det(g["axes"]))
        assert abs(n - 2) < 2e-3
        write_fake_fchk(f"{d}/bz.fchk", read_xyz(ROOT / "examples/benzene.xyz"))
        wb = Wavefunction(f"{d}/bz.fchk")
        rb = wb.mulliken_mayer()
        assert abs(rb["total_charge"]) < 1e-6
        B = rb["bond_orders"]
        assert 1.3 < B[0, 1] < 1.6 or 1.3 < max(B[0]) < 1.6
        cube = wb.mo_cube(wb.nalpha - 1, spacing_A=0.3)
        write_cube(f"{d}/h.cube", cube)
        c2 = read_cube(f"{d}/h.cube")
        assert c2["data"].shape == cube["data"].shape and np.abs(c2["data"] - cube["data"]).max() < 1e-4
        es = esp_surface(wb.density_cube(0.3), None, 0.001, charges=rb["charges"])
        assert es["extrema"] and es["stats"]["area_A2"] > 50
    print(f"电荷 / 键级 / 密度 OK: H2 键级 1、密度积分 {n:.4f}；苯电荷和 0；cube 读写一致；ESP 表面 {len(es['extrema'])} 个极值")


def test_nbo():
    txt = """ SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS
   1. BD (   2) C   1 - C   2      /  10. BD*(   2) C   3 - C   4            20.16    0.28    0.068
   3. LP (   1) N   7              /  12. BD*(   1) C  13 - H  14             2.10    0.95    0.040
 NATURAL BOND ORBITALS (Summary):
 Summary of Natural Population Analysis:

                                     Natural Population
             Natural    -----------------------------------------------
    Atom No    Charge         Core      Valence    Rydberg      Total
 -----------------------------------------------------------------------
      C    1   -0.20000      1.99900     4.18000    0.02100     6.20000
      H    2    0.20000      0.00000     0.79800    0.00200     0.80000
 =======================================================================
"""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "n.log").write_text(txt)
        r = read_nbo(Path(d, "n.log"))
    assert [x["E2"] for x in r["E2"]] == [20.16, 2.10] and r["E2"][1]["donor_type"] == "LP"
    assert r["E2"][1]["acceptor_atoms"] == [13, 14] and [q for *_, q in r["npa"]] == [-0.2, 0.2]
    print("NBO OK: E(2) 表与自然电荷")


if __name__ == "__main__":
    test_basis_normalization()
    test_population_density_cube()
    test_nbo()
