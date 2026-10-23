"""LDA 自由原子密度、密度解析导数、IGM / IGMH、AIM 拓扑。"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from fake_fchk import write_fake_fchk  # noqa: E402
from mobility_tool.gaussian_io import read_xyz  # noqa: E402
from mobility_tool.substitution import Z as ZN  # noqa: E402
from mobility_tool.topology import (Promolecule, aim_analysis, fragments_by_connectivity, igm_analysis,  # noqa: E402
                                    igm_surface, wfn_provider)
from mobility_tool.wavefunction import Wavefunction, density_derivs  # noqa: E402

B = 1.8897261


def test_atom_densities():
    from mobility_tool.atomdens import solve_atom
    r = np.linspace(1e-5, 30, 200001)
    for el in ("H", "C", "O", "S", "Ir"):
        p = Promolecule([ZN[el]], [[0, 0, 0]])
        rho, _, _ = p._radial(ZN[el], r)
        assert abs(np.trapezoid(4 * np.pi * r * r * rho, r) - ZN[el]) < 2e-3, el
    ref = {"H": -0.2335, "C": -0.1992, "Ne": -0.4985}               # LDA (VWN) 最高占据本征值
    for el, e in ref.items():
        _, _, eps = solve_atom(ZN[el])
        top = max(eps.values())
        assert abs(top - e) / abs(e) < 0.05, (el, top)
    print("自由原子密度 OK: 归一化、LDA 本征值与文献相差 < 5%")


def test_density_derivatives():
    with tempfile.TemporaryDirectory() as d:
        write_fake_fchk(f"{d}/bz.fchk", read_xyz(ROOT / "examples/benzene.xyz"))
        w = Wavefunction(f"{d}/bz.fchk")
    pts = np.random.default_rng(1).normal(0, 1.5, (25, 3)) * B
    rho, g, hs, G = density_derivs(w, pts, 2, kinetic=True)
    h = 1e-4
    fd = np.stack([(w.density_values(pts + h * e) - w.density_values(pts - h * e)) / (2 * h) for e in np.eye(3)])
    lap = sum((w.density_values(pts + h * e) - 2 * w.density_values(pts) + w.density_values(pts - h * e)) / h ** 2
              for e in np.eye(3))
    assert np.abs(fd - g).max() < 1e-6 and np.abs(lap - hs[:3].sum(0)).max() < 1e-5 and (G >= 0).all()
    print("密度导数 OK: ∇ρ、∇²ρ 与有限差分一致，G ≥ 0")


def _dimer(dz, dx=0.0):
    a = read_xyz(ROOT / "examples/benzene.xyz")
    dim = a + [(s, x + dx, y, z + dz) for s, x, y, z in a]
    sym = [s for s, *_ in dim]
    XA = np.array([x[1:] for x in dim])
    return dim, sym, XA, np.array([ZN[s] for s in sym]), XA * B


def test_igm():
    vals = []
    for dz in (3.5, 4.0, 5.0):
        _, sym, XA, Z, R = _dimer(dz)
        frag = fragments_by_connectivity(sym, XA)
        res = igm_analysis(Z, R, frag, spacing_A=0.2)
        assert res["dg_inter"]["data"].min() >= -1e-12
        vals.append(res["dG_inter"])
    assert vals[0] > vals[1] > vals[2] > 0
    dim, sym, XA, Z, R = _dimer(3.5, 1.0)
    frag = fragments_by_connectivity(sym, XA)
    res = igm_analysis(Z, R, frag, spacing_A=0.15)
    v, f, col, val = igm_surface(res, 0.005)
    assert len(f) > 500 and np.abs(val).max() < 0.02                  # π–π：sign(λ2)ρ 接近 0 (绿色)
    with tempfile.TemporaryDirectory() as d:                           # IGMH：Σ 片段梯度 = ∇ρ (由定义保证)，δG > 0
        write_fake_fchk(f"{d}/dim.fchk", dim)
        w = Wavefunction(f"{d}/dim.fchk")
        rh = igm_analysis(w.Z, w.coords, frag, provider=wfn_provider(w), spacing_A=0.3)
        rh2 = igm_analysis(w.Z, w.coords, frag, provider=wfn_provider(w), spacing_A=0.3, sign=1)
    assert rh["dG_inter"] > 0 and rh2["dG_inter"] > 0 and rh["method"] == "IGMH"
    print(f"IGM OK: δG_inter {vals[0]:.3f} > {vals[1]:.3f} > {vals[2]:.3f}；π–π 等值面为绿色；IGMH 两种形式均可计算")


def test_aim():
    with tempfile.TemporaryDirectory() as d:
        write_fake_fchk(f"{d}/h2.fchk", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
        w = Wavefunction(f"{d}/h2.fchk")
    r = aim_analysis(w.Z, w.coords, wfn_provider(w))
    c = r["cps"][0]
    assert r["counts"]["BCP"] == 1 and abs(c["xyz_bohr"][2] / B - 0.37) < 1e-3 and c["lap"] < 0 and r["ph_ok"]
    assert abs(c["V"] - (c["lap"] / 4 - 2 * c["G"])) < 1e-12
    a = read_xyz(ROOT / "examples/benzene.xyz")
    Zb = np.array([ZN[s] for s, *_ in a])
    Rb = np.array([x[1:] for x in a]) * B
    rb = aim_analysis(Zb, Rb, Promolecule(Zb, Rb), kinetic=False)
    assert rb["counts"]["BCP"] == 12 and rb["counts"]["RCP"] == 1 and rb["ph_ok"]
    _, sym, XA, Z, R = _dimer(3.5)
    frag = fragments_by_connectivity(sym, XA)
    rd = aim_analysis(Z, R, Promolecule(Z, R), frag=frag, kinetic=False)
    inter = [c for c in rd["cps"] if c["type"] == "BCP" and c.get("atoms") and frag[c["atoms"][0]] != frag[c["atoms"][1]]]
    assert rd["ph_ok"] and inter and all(c["lap"] > 0 and c["rho"] < 0.02 for c in inter)
    print(f"AIM OK: H2 键临界点在中点；苯 12 BCP + 1 RCP；苯二聚体 {rd['counts']}，分子间 BCP {len(inter)} 个 (闭壳层)")


if __name__ == "__main__":
    test_atom_densities()
    test_density_derivatives()
    test_igm()
    test_aim()
