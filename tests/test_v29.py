"""v1.18 回归测试：
1. 视图沿晶轴观察不得修改晶格矢量 (曾把晶格原地归一化成 1 Å，导致近邻、导出与应变扫描全部出错)；
2. 传输平面以晶格矢量给出时，角分辨迁移率不得修改传入的矢量；
3. 多分子晶胞 (跳跃矢量不成 ±v 对，如 P2₁ 晶体) 的零场精确迁移率：均匀化公式与 KMC 一致，成对时退化为旧公式；
4. CIF 空间群名称；
5. 拖动预览绘制与完整绘制使用相同的视野。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def test_view_along_keeps_lattice():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from mobility_tool.render import MolRenderer
    lat = np.array([[6.187, 0, 0], [0, 7.662, 0], [-0.704, 0, 16.195]])
    keep = lat.copy()
    fig = Figure()
    r = MolRenderer(fig.add_axes([0, 0, 1, 1]))
    for i in range(3):
        r.view_along(lat[i], lat[2] if i != 2 else lat[1])
    assert np.allclose(lat, keep), lat
    from mobility_tool.mobility import angular_mobility, plane_angles
    u, v = lat[0], lat[1]
    plane_angles([1.0, 2.0, 3.0], plane=(u, v))
    angular_mobility(np.eye(3), plane=(u, v))
    assert np.allclose(lat, keep), lat
    print("沿晶轴观察 / 晶格矢量作传输平面：晶格不被修改")


def test_exact_multisite():
    from mobility_tool.kmc import analytic_bravais, analytic_equivalent_sites, run_kmc
    # 一维两位点链：A→B 前向 +2 Å (J1)，A→B 后向 −3 Å (J2)；周期 5 Å。A 处的跳跃不成 ±v 对
    J1, J2 = 0.06, 0.02
    edges = [(0, 1, np.array([2.0, 0, 0]), J1, 1.0, (0, 0, 0)), (0, 1, np.array([-3.0, 0, 0]), J2, 1.0, (-1, 0, 0)),
             (1, 0, np.array([-2.0, 0, 0]), J1, 1.0, (0, 0, 0)), (1, 0, np.array([3.0, 0, 0]), J2, 1.0, (1, 0, 0))]
    g = {"n_sites": 2, "edges": edges, "kind": "crystal", "periodic": True}
    lam, T = 0.2, 300.0
    ex = analytic_equivalent_sites(g, lam, T)[0, 0]
    from mobility_tool.kmc import rate_marcus, KB_EV
    k1, k2 = rate_marcus(J1, lam, 0, T), rate_marcus(J2, lam, 0, T)
    a = 5e-8                                                   # 交替速率链的精确解：D = a² k1 k2 / (2 (k1 + k2))
    mu_chain = a * a * k1 * k2 / (2 * (k1 + k2)) / (KB_EV * T)
    assert abs(ex / mu_chain - 1) < 1e-9, (ex, mu_chain)
    naive = (0.5 * (k1 * 4e-16 + k2 * 9e-16)) / (KB_EV * T)    # 旧公式 Σ k r² / 2kT (按位点平均)
    assert naive > 1.5 * ex                                     # 不成对时旧公式明显偏高
    r = run_kmc(g, lam, T, walkers=600, steps=2500, seed=3)
    assert abs(r["mu_tensor_cm2_Vs"][0, 0] / ex - 1) < 4 * r["mu_err"][0, 0] / ex + 0.03, (r["mu_tensor_cm2_Vs"][0, 0], ex)
    # 布拉维格子 / 成对跳跃：与旧公式相同
    gb = {"n_sites": 1, "edges": [(0, 0, np.array([4.0, 0, 0]), 0.05, 1.0, (1, 0, 0)),
                                  (0, 0, np.array([-4.0, 0, 0]), 0.05, 1.0, (-1, 0, 0))], "periodic": True}
    assert np.allclose(analytic_equivalent_sites(gb, lam, T), analytic_bravais(gb, lam, T))
    print(f"多位点精确解 = 交替链解析解 {ex:.4g}；旧公式偏高 {naive / ex:.2f} 倍；KMC {r['mu_tensor_cm2_Vs'][0, 0]:.4g}")


def test_spacegroup_and_preview():
    import tempfile
    from test_crystal import make_cif
    from mobility_tool.crystal import read_cif
    with tempfile.TemporaryDirectory() as d:
        make_cif(f"{d}/t.cif")
        txt = Path(f"{d}/t.cif").read_text()
        if "_symmetry_space_group_name_H-M" not in txt:
            txt = txt.replace("_cell_length_a", "_symmetry_space_group_name_H-M 'P 21/c'\n_cell_length_a", 1)
            Path(f"{d}/t.cif").write_text(txt)
        c = read_cif(f"{d}/t.cif")
        assert c["spacegroup"], c.get("spacegroup")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mobility_tool.gaussian_io import read_xyz
    from mobility_tool.render import MolRenderer
    at = read_xyz(ROOT / "examples/benzene_dimer.xyz")
    fig = Figure(figsize=(6, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    can = FigureCanvasAgg(fig)
    r = MolRenderer(ax)
    r.set_scene([a[0] for a in at], [a[1:] for a in at], reset=True)
    r.draw()
    can.draw()
    lim = (ax.get_xlim(), ax.get_ylim())
    r.draw(fast=True)                         # 预览：数据更新，不新建图元
    can.draw()
    assert np.allclose(lim, (ax.get_xlim(), ax.get_ylim()))
    n_art = len(ax.collections)
    r.rotate(20, 10)
    r.draw(fast=True)
    assert len(ax.collections) == n_art
    r.draw()
    can.draw()
    assert r._preview is None
    print(f"空间群 {c['spacegroup']}；拖动预览与完整绘制视野一致，预览不新建图元")


if __name__ == "__main__":
    test_view_along_keeps_lattice()
    test_exact_multisite()
    test_spacegroup_and_preview()
