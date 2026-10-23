"""CEDS (耦合–能量双曲面) —— 晶体二聚体参数、局域扫描与局域曲面、MC 温度标度、势能校准、
位点相位 (特征标)、TLT。并五苯晶体 (Mattheus 2001, 90 K)；DFT 输出用 fake_adf (EHT) 模拟。"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from fake_adf import run_scan  # noqa: E402
from mobility_tool.ceds import (CEDS, DimerMC, EHTSurface, LocalSurface, cif_ueq, rotvec_matrix,  # noqa: E402
                                summary_table, surfaces_from_project, temperature_series)
from mobility_tool.crystal import find_molecules, read_cif  # noqa: E402
from mobility_tool.scanflow import ScanProject  # noqa: E402
from mobility_tool.surrogate import build_B, molecular_frame  # noqa: E402

CIF = ROOT / "tests" / "data" / "pentacene_90K.cif"
MAIN = (1, 2, 3, 8)                    # 两种人字形二聚体 + 两个独立分子各自沿 a 的平行二聚体


def _setup(d):
    crys = read_cif(CIF)
    mols = find_molecules(crys)
    mono = Path(d) / "pent_cryst.xyz"
    with open(mono, "w") as fh:
        fh.write(f"{len(mols[0]['symbols'])}\npentacene from crystal\n")
        for s, x in zip(mols[0]["symbols"], mols[0]["cart"]):
            fh.write(f"{s} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n")
    p = ScanProject.create(Path(d) / "proj", mono, "pent")
    lv = p.add_level("PW91", "TZP")
    (p.level_dir(lv) / "mono.t21").write_bytes(b"x")
    ce = CEDS(crys, mols, monomer=(p.symbols, p.Y))
    return crys, mols, p, lv, ce


def test_geometry_and_phase():
    from scipy.optimize import linear_sum_assignment
    with tempfile.TemporaryDirectory() as d:
        crys, mols, p, lv, ce = _setup(d)
        assert [q["gid"] for q in ce.pairs][:3] == [1, 2, 3] and len(ce.pairs) >= 8
        rs = {q["gid"]: round(q["r_centroid"], 2) for q in ce.pairs}
        assert rs[1] == 4.69 and rs[2] == 5.16 and rs[3] == rs[8] == 6.24, rs
        lat = np.asarray(crys["lattice"])
        for q in ce.pairs:
            if q["gid"] not in MAIN:
                continue
            axA, cA = ce.frames[q["ref"]]
            XB = np.asarray(mols[q["mol"]]["cart"]) + np.asarray(q["translation"]) @ lat
            Bc = (XB - cA) @ axA                                   # 晶体 B 在 A 坐标系中
            c = q["center"]
            Bf = build_B(ce.Y, c["z"], c["dx"], c["dy"], c["gamma"], c["roll"], c["pitch"])
            hf = [i for i, s in enumerate(ce.sym) if s != "H"]
            hc = [i for i, s in enumerate(mols[q["mol"]]["symbols"]) if s != "H"]
            D = np.linalg.norm(Bf[hf][:, None] - Bc[hc][None], axis=-1)
            r_, c_ = linear_sum_assignment(D)
            rmsd = np.sqrt((D[r_, c_] ** 2).mean())
            assert rmsd < 0.05, (q["gid"], rmsd)
        # 特征标：并五苯 HOMO 对长轴 C2x 对称，LUMO 反对称 (与 DFT 对称检验一致)
        assert ce.characters["hole"]["C2x"] == 1 and ce.characters["electron"]["C2x"] == -1
        # 人字形二聚体的 B 需要绕长轴的二重轴才能落到扫描区域：|V| 与晶体中直接 EHT 一致
        from mobility_tool.eht import eht_transfer_integrals
        surf = EHTSurface(ce.sym, ce.Y)
        q1 = next(q for q in ce.pairs if q["gid"] == 1)
        A = [(s, *x) for s, x in zip(mols[0]["symbols"], mols[0]["cart"])]
        XB = np.asarray(mols[q1["mol"]]["cart"]) + np.asarray(q1["translation"]) @ lat
        B = [(s, *x) for s, x in zip(mols[q1["mol"]]["symbols"], XB)]
        direct = eht_transfer_integrals(A, B, esd=False)["results"]
        for c, k in (("hole", "HOMO"), ("electron", "LUMO")):
            v = surf(q1["q0"], c)[0]
            assert abs(abs(v) - abs(direct[k]["J_eff_meV"])) < 0.05 * abs(v) + 0.5, (c, v, direct[k]["J_eff_meV"])
    print("晶体二聚体参数 (RMSD < 0.05 Å)、特征标与 |V| OK")


def test_mc_temperature_and_calibration():
    with tempfile.TemporaryDirectory() as d:
        crys, mols, p, lv, ce = _setup(d)
        q = ce.pairs[0]
        u = {}
        for T in (100.0, 300.0):
            sim = DimerMC(crys, mols, q["ref"], q["mol"], q["translation"])
            mc = sim.run(T, 600, 5, 2000, seed=3)
            u[T] = np.mean(sim.u_eq(mc))
            assert 0.15 < mc["acc"] < 0.65
        ratio = u[100.0] / u[300.0]
        assert 0.2 < ratio < 0.55, ratio                       # 谐振极限 1/3
        hist = ce.calibrate(2.5 * u[300.0], T=300.0, n_samples=500)
        assert hist[-1][0] < 0.6 and abs(hist[-1][1] / (2.5 * u[300.0]) - 1) < 0.3, hist
        R = rotvec_matrix(np.array([[0.0, 0.0, np.pi / 2]]))[0]
        assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    print(f"MC: U_eq(100 K)/U_eq(300 K) = {ratio:.2f} (谐振 0.33)；势能校准 OK")


def test_local_surface_and_tlt():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        crys, mols, p, lv, ce = _setup(d)
        half = ce.suggest_half(T=300.0, n_samples=300)
        assert set(MAIN) <= set(half) and all(half[g]["z"] < 0.5 for g in MAIN), half
        cen = {g: c for g, c in ce.centers().items() if g in MAIN}
        info = p.add_local("local", lv, cen, n=70, half={g: half[g] for g in MAIN})
        assert info["n_computed"] >= 4 * 60, info
        run_scan(p, "local")
        surf, why = surfaces_from_project(p, ce)
        assert set(surf) == set(MAIN), why
        print("\n".join(f"  G{g}: {why[g]}" for g in MAIN))
        assert all(isinstance(surf[g], LocalSurface) and surf[g].kind == "quadratic" for g in MAIN)
        eht = EHTSurface(ce.sym, ce.Y)
        res = ce.run(surf, T=300.0, n_samples=300, thin=5, burn=1500)
        for g in MAIN:
            r = res[g]
            assert r["coverage"] > 0.85, (g, r["coverage"])    # ±2.5σ 范围：6 维中约 7–10 % 样本略出界
            for c in ("hole", "electron"):
                ve = r[c]["chi"] * eht(r["q"], c)
                vl = r[c]["samples"]
                err = np.sqrt(((ve - vl) ** 2).mean())
                assert err < 0.25 * r[c]["std"] + 0.3, (g, c, err, r[c]["std"])
                assert abs(np.std(ve, ddof=1) / r[c]["std"] - 1) < 0.15
        print(summary_table(ce, res, "hole"))
        for c in ("hole", "electron"):
            t = ce.tlt(res, c, 300.0, n_real=4)
            t0_ = ce.tlt(res, c, 300.0, n_real=1, sigma_scale=0.0)
            assert t["mu_principal"][0] > 0 and np.isfinite(t0_["mu_principal"][0])
            assert t["keep_axes"] == (0, 1)                    # a–b 平面
        rows = temperature_series(ce, surf, [150.0, 300.0], n_samples=150, thin=5, burn=1000, n_real=3)
        assert len(rows) == 2 and all(r["mu_hole"] > 0 for r in rows)
    print(f"局域曲面 (二次多项式) 与 EHT 直接计算一致、TLT、μ(T) OK ({time.time() - t0:.0f} s)")


def test_cif_ueq():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "u.cif"
        f.write_text("data_x\nloop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
                     "_atom_site_fract_z\n_atom_site_U_iso_or_equiv\nC1 C 0.1 0.2 0.3 0.0400(5)\nC2 C 0.2 0.2 0.3 0.0500(5)\n"
                     "H1 H 0.3 0.2 0.3 0.080\n")
        assert abs(cif_ueq(f) - 0.045) < 1e-9
        assert cif_ueq(CIF) is None
    print("CIF U_eq 读取 OK")


def test_product_surface():
    """乘积模型曲面：用 EHT 生成论文 1 那样的 G 网格 (Δx = 0) 与 φ = 52° 的 Δx 扫描，绑定到晶体 T 二聚体后
    与 EHT 直接计算比较 (中心值与 MC 样本)。"""
    from mobility_tool.ceds import ProductSurface, reverse_cfg
    from mobility_tool.scanflow import make_cfg
    with tempfile.TemporaryDirectory() as d:
        crys, mols, p, lv, ce = _setup(d)
        eht = EHTSurface(ce.sym, ce.Y)
        q1 = next(q for q in ce.pairs if q["gid"] == 1)
        from mobility_tool.scanflow import apply_op
        cen = None
        for v in (q1["center"], reverse_cfg(q1["center"])):
            for op in ce.ops:
                c = apply_op(v, op)
                if c["z"] > 4 and 30 < c["roll"] < 70:
                    cen = c
        assert cen is not None
        G = []
        for z in np.arange(cen["z"] - 0.6, cen["z"] + 0.61, 0.2):
            for dy in np.arange(cen["dy"] - 1.0, cen["dy"] + 1.01, 0.5):
                for ph in (cen["roll"] - 8, cen["roll"], cen["roll"] + 8):
                    q = np.array([z, 0.0, dy, 0.0, 0.0, ph])
                    v = eht.eval_one(q)
                    G.append(dict(z=z, dx=0.0, dy=dy, gamma=0.0, pitch=0.0, roll=ph, **v))
        F = []
        for dx in np.arange(0, 3.01, 0.25):
            v = eht.eval_one(np.array([cen["z"], dx, cen["dy"], 0.0, 0.0, cen["roll"]]))
            F.append(dict(z=cen["z"], dx=dx, dy=cen["dy"], gamma=0.0, pitch=0.0, roll=cen["roll"], **v))
        ps = ProductSurface(G, f_records=F, characters=ce.characters, ops=ce.ops)
        b = ps.bind(q1["center"])
        assert b is not None
        for c in ("hole", "electron"):
            v0 = b(q1["q0"], c)[0]
            ve = eht(q1["q0"], c)[0]
            assert abs(abs(v0) - abs(ve)) < 0.05 * abs(ve) + 1.0, (c, v0, ve)
        res = ce.run({1: b}, T=300.0, n_samples=200, thin=5)
        r = res[1]
        assert r["coverage"] > 0.9 and set(r["frozen"]) == {"gamma", "pitch"}
        for c in ("hole", "electron"):
            ve = r[c]["chi"] * eht(r["q"], c)
            assert np.corrcoef(ve, r[c]["samples"])[0, 1] > 0.9, c
    print("乘积模型曲面 (G 网格 × f(Δx)) 与 EHT 一致 OK")


def test_rotmodel_surface():
    """论文转角模型 R2 的重新实现：在 β(Δy) 已知、g 可分离的合成数据上应精确还原。"""
    from mobility_tool.ceds import RotModelSurface
    rec = []
    beta = lambda dy: 1.5 + 0.05 * dy - 0.01 * dy * dy  # noqa: E731
    g = lambda dy, ph: 120 * np.cos(0.9 * dy - 0.4) * (1 + 0.01 * (ph - 52))  # noqa: E731
    for z in np.arange(4.2, 6.01, 0.2):
        for dy in np.arange(-1, 3.01, 0.5):
            for ph in (40.0, 52.04, 65.0):
                v = g(dy, ph) * np.exp(-beta(dy) * (z - 4.8))
                rec.append(dict(z=z, dx=0.0, dy=dy, gamma=0, pitch=0, roll=ph, dmin_A=3.0, hole=v, electron=0.5 * v))
    fx = [dict(z=4.8, dx=x, dy=1.0, gamma=0, pitch=0, roll=52.04, hole=np.cos(1.1 * x), electron=np.cos(1.3 * x))
          for x in np.arange(0, 6.01, 0.25)]
    s = RotModelSurface(rec, f_records=fx)
    for z, dx, dy, ph in ((4.65, 0.2, 1.05, 52.0), (5.1, 2.1, 0.3, 45.0), (4.9, 1.0, 2.2, 60.0)):
        c = dict(z=z, dx=dx, dy=dy, gamma=0, pitch=0, roll=ph)
        ref = g(dy, ph) * np.exp(-beta(dy) * (z - 4.8)) * np.cos(1.1 * dx)
        assert abs(s.value(c, "hole") - ref) < 0.03 * abs(ref) + 0.5, (c, s.value(c, "hole"), ref)
    print("论文转角模型 R2 (β(Δy) 二次、g(Δy, φ) 表、f(Δx)) 还原 OK")


if __name__ == "__main__":
    test_rotmodel_surface()
    test_product_surface()
    test_cif_ueq()
    test_geometry_and_phase()
    test_mc_temperature_and_calibration()
    test_local_surface_and_tlt()
