"""软件内 EHT 转移积分与 VOTCA 式 KMC 的检验（运行: python tests/test_eht_kmc.py）"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mobility_tool.eht import BOHR, build_basis, eht, eht_transfer_integrals, overlap_matrix  # noqa: E402
from mobility_tool.kmc import analytic_bravais, graph_from_paths, rate_jortner, rate_marcus, run_kmc  # noqa: E402

ETH = [("C", 0, 0, 0.667), ("C", 0, 0, -0.667), ("H", 0, 0.923, 1.232), ("H", 0, -0.923, 1.232),
       ("H", 0, 0.923, -1.232), ("H", 0, -0.923, -1.232)]


def test_overlaps():
    for r in (0.74, 1.5):                                     # H 1s–1s 解析 STO 重叠
        sh, n, _ = build_basis(["H", "H"], [[0, 0, 0], [0, 0, r]])
        x = 1.3 * r / BOHR
        assert abs(overlap_matrix(sh, n)[0, 1] - np.exp(-x) * (1 + x + x * x / 3)) < 2e-5
    sh, n, _ = build_basis(["C", "C"], [[0, 0, 0], [0, 0, 1.4]])
    s, p = overlap_matrix(sh, n), 1.625 * 1.4 / BOHR
    assert abs(-s[3, 7] - np.exp(-p) * (-1 - p - p * p / 5 + 2 * p ** 3 / 15 + p ** 4 / 15)) < 2e-5   # σ
    assert abs(s[1, 5] - np.exp(-p) * (1 + p + 2 * p * p / 5 + p ** 3 / 15)) < 2e-5                   # π
    atoms = [("Ir", 0, 0, 0), ("N", 2.0, 0, 0), ("N", -2.0, 0, 0), ("O", 0, 2.05, 0), ("C", 0, -2.0, .3),
             ("Cl", .2, .1, 2.3)]
    sym, x = [a[0] for a in atoms], np.array([a[1:] for a in atoms])
    r1 = eht(sym, x)
    assert np.abs(r1["S"][:9, :9] - np.eye(9)).max() < 1e-10                       # Ir 的 s,p,d 正交归一
    rot = Rotation.from_euler("xyz", [33, -71, 124], degrees=True).as_matrix()
    assert np.abs(eht(sym, x @ rot.T + 1)["e"] - r1["e"]).max() < 1e-8             # d 轨道转动不变
    print("EHT 重叠积分 OK (H/C 解析公式, Ir d 轨道正交归一, 转动不变)")


def test_transfer():
    prev = None
    for d in (3.5, 4.0, 4.5):
        t = eht_transfer_integrals(ETH, [(s, x + d, y, z) for s, x, y, z in ETH])["results"]
        assert abs(abs(t["HOMO"]["J_eff_meV"]) - t["ESD_hole_meV"]) < 0.02 * t["ESD_hole_meV"]
        if prev:
            assert abs(t["HOMO"]["J_eff_meV"]) < prev
        prev = abs(t["HOMO"]["J_eff_meV"])
    rot = Rotation.from_euler("xyz", [20, 50, -10], degrees=True).as_matrix()
    b = [(s, x + 3.8, y, z) for s, x, y, z in ETH]
    r = lambda L: [(s, *(np.array(p) @ rot.T)) for s, *p in L]  # noqa: E731
    j1 = eht_transfer_integrals(ETH, b)["results"]["HOMO"]["J_eff_meV"]
    j2 = eht_transfer_integrals(r(ETH), r(b))["results"]["HOMO"]["J_eff_meV"]
    assert abs(j1 - j2) < 1e-6
    print(f"EHT 转移积分 OK (投影法 = 能级分裂, 随距离单调衰减, 转动不变; 乙烯 3.8 Å J_h = {j1:.1f} meV)")


def test_kmc():
    paths = [dict(vector=[8.709, 0, 0], V=0.07873, count=2), dict(vector=[0.4, 10.13, 0], V=0.01381, count=2),
             dict(vector=[9.109, 10.13, 0], V=0.02277, count=2), dict(vector=[0, 0, 12.0], V=0.004, count=2)]
    g = graph_from_paths(paths, "V")
    assert g["periodic"]
    ana = np.diag(analytic_bravais(g, 0.130, 300))
    r = run_kmc(g, 0.130, 300, walkers=3000, steps=3000, seed=4)
    kmc, err = np.diag(r["mu_tensor_cm2_Vs"]), np.diag(r["mu_err"])
    assert np.all(np.abs(kmc - ana) < 4 * err + 0.02 * ana), (kmc, ana, err)
    assert abs(rate_jortner(0.05, 0.0, 0.2, 0.0, 300) / rate_marcus(0.05, 0.2, 0.0, 300) - 1) < 1e-12
    rf = run_kmc(g, 0.130, 300, field=(1e4, 0, 0), walkers=3000, steps=3000, seed=3)
    assert abs(rf["mu_field_cm2_Vs"] - ana[0]) < 4 * rf["mu_field_err"] + 0.05 * ana[0]
    rd = run_kmc(g, 0.130, 300, sigma=0.06, supercell=5, walkers=1500, steps=2000, seed=5)
    assert rd["mu_avg"] < 0.6 * r["mu_avg"]
    print(f"KMC OK: μ 对角 {np.round(kmc, 3)} vs 解析 {np.round(ana, 3)}; 弱场 μ_F={rf['mu_field_cm2_Vs']:.2f}; "
          f"σ=0.06 eV 时 μ_avg 降至 {rd['mu_avg']:.2f}")


def test_strain():
    from test_crystal import make_cif
    from mobility_tool.crystal import find_molecules, read_cif
    from mobility_tool.workflow import strain_scan
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        make_cif(f"{d}/t.cif")
        c = read_cif(f"{d}/t.cif")
        m = find_molecules(c)
        rows = strain_scan(c, m, "c", [-2, 0, 2], 0.2, 0.2, cutoff=4.5)
    j = [r["J_max_hole"] for r in rows]
    assert j[0] > j[1] > j[2]
    print("应变扫描 OK: 沿 c 压缩 → J 增大:", [round(x, 1) for x in j])


if __name__ == "__main__":
    test_overlaps()
    test_transfer()
    test_kmc()
    test_strain()
