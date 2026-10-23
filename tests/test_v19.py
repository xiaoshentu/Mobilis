"""全晶体刚体 MD (rigidmd) 与 CEDS 的连接：ceds md 命令、局域结果表曲面、组合曲面。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility_tool.ceds import (CEDS, LocalSurface, OPS, PARAM_KEYS, combined_surfaces,  # noqa: E402
                                local_surfaces_from_csv)
from mobility_tool.crystal import find_molecules, read_cif  # noqa: E402
from mobility_tool.rigidmd import (RigidCrystalMD, export_md_csv, md_pair_params,  # noqa: E402
                                   md_temperature_series)

DATA = os.path.join(os.path.dirname(__file__), "data")
CIF = os.path.join(DATA, "pentacene_RT.cif")


def _ceds():
    c = read_cif(CIF)
    return CEDS(c, find_molecules(c), cutoff=4.0)


class _LinearSurface:
    """测试用：V = V0 + 线性项 (不需要 DFT 数据)。"""
    source, frozen = "test", []

    def __init__(self, q0, v0):
        self.q0, self.v0 = np.asarray(q0, float), v0

    def __call__(self, Q, carrier):
        Q = np.atleast_2d(Q)
        s = 1.0 if carrier == "hole" else -1.2
        return s * (self.v0 + 30 * (Q[:, 0] - self.q0[0]) + 10 * (Q[:, 2] - self.q0[2]))

    def coverage(self, Q):
        return 1.0

    def describe(self):
        return "linear test surface"


def test_general_frame_and_start_params():
    """晶体中的单体 (X 射线 H，分子坐标系不是惯量主轴系) 也能用：初始分子坐标系与 CEDS 完全一致。"""
    ce = _ceds()
    md0 = RigidCrystalMD(ce.crys, ce.frames, (ce.sym, ce.Y), (2, 2, 1))
    # 人为把单体坐标系转 10° 并平移：不再是惯量主轴系 / 质心不在原点；晶体中的原子位置不变
    th = np.radians(10.0)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    shift = np.array([0.3, -0.2, 0.1])
    Y2 = (np.asarray(ce.Y) @ Rz) + shift
    frames2 = [(ax @ Rz, c - (ax @ Rz) @ shift) for ax, c in ce.frames]
    md = RigidCrystalMD(ce.crys, frames2, (ce.sym, Y2), (2, 2, 1))
    assert not np.allclose(md.R, np.eye(3)) and np.linalg.norm(md.com) > 0.1
    assert abs(md.lattice_energy() - md0.lattice_energy()) < 1e-8
    Qm, o = md.mol_frames(md.c0, md.Q0)
    for k in range(md.nk):
        assert np.allclose(Qm[k], frames2[k][0], atol=1e-10) and np.allclose(o[k], frames2[k][1], atol=1e-10)
    md = md0
    Qm, o = md.mol_frames(md.c0, md.Q0)
    for k in range(md.nk):
        assert np.allclose(Qm[k], ce.frames[k][0], atol=1e-10) and np.allclose(o[k], ce.frames[k][1], atol=1e-10)
    fr = {"c": md.c0[None], "Q": md.Q0[None]}
    for p in ce.pairs[:3]:
        q, _ = md_pair_params(md, fr, p, np.diag(OPS[p["g"]]).astype(float), recenter=False, q0=p["q0"])
        assert np.allclose(q, np.asarray(p["q0"])[None], atol=1e-6)


def test_lattice_energy_and_conservation():
    ce = _ceds()
    md = RigidCrystalMD(ce.crys, ce.frames, (ce.sym, ce.Y), (2, 2, 1))
    assert -45 < md.lattice_energy() < -28                       # 并五苯升华焓约 37 kcal/mol
    Fc, tau, E = md.forces(md.c0, md.Q0)
    assert np.abs(Fc).max() < 1e-2                              # 分子在反演中心上：合力 (近似) 为零
    _, st = md.run(150, 40, dt=5.0, gamma=0.0, every=40, seed=3)
    e0 = md.total_energy(st)
    _, st2 = md.run(150, 60, dt=5.0, gamma=0.0, every=60, seed=3, state=st)
    assert abs(md.total_energy(st2) - e0) < 0.05 * md.N * 0.3


def test_md_series_and_export(tmp_path=None):
    import tempfile
    ce = _ceds()
    surf = {p["gid"]: _LinearSurface(p["q0"], 40.0) for p in ce.pairs if p["r_centroid"] < 7.0}
    rows, md = md_temperature_series(ce, surf, [150, 300], supercell=(2, 2, 1), equil_ps=1.0, prod_ps=3.0,
                                     hbar_taus=(0.002,), n_real=2)
    r1, r3 = rows
    g = next(iter(surf))
    sd = [sum(r["res"][k]["hole"]["std"] for k in surf) for r in rows]
    assert sd[1] > sd[0] > 0                                                       # 涨落随温度增大
    assert r3["U_eq"] > r1["U_eq"] > 0
    assert abs(abs(r3["res"][g]["hole"]["mean"]) - 40) < 1                         # 平均结构移回实验结构 (线性曲面：均值 = V0)
    assert r3["mu_hole"] > 0 and r3["mu_static_hole"] > r3["mu_hole"]
    assert "hole_2" in r3["mu_tau"] and np.isfinite(r3["tau_c_fs"])
    out = os.path.join(tmp_path or tempfile.mkdtemp(), "md.csv")
    export_md_csv(ce, rows, out)
    txt = open(out, encoding="utf-8-sig").read()
    assert "hbar_tau_MD_meV" in txt and "sigma_meV" in txt and "mu_hole" in txt


def test_local_csv_and_combined_surfaces():
    """论文中 P 对的局域曲面可直接由 results.csv 重建 (与 ADF 输出解析得到的完全相同)。"""
    ce = _ceds()
    loc = local_surfaces_from_csv(ce, os.path.join(DATA, "pentacene_cedsP_results.csv"))
    assert set(loc) == {3, 7}
    s3 = loc[3]
    assert isinstance(s3, LocalSurface) and s3.stats["hole"]["n"] == 366
    assert abs(s3.stats["hole"]["loo_rmse"] - 8.93) < 0.01
    v0 = s3(np.asarray(next(p for p in ce.pairs if p["gid"] == 3)["q0"]), "hole")[0]
    assert abs(v0 - 34.47) < 0.05
    fake = {1: _LinearSurface(ce.pairs[0]["q0"], 78.0), 3: _LinearSurface(ce.pairs[2]["q0"], 1.0)}
    surf, why = combined_surfaces(ce, product=fake, local=loc)
    assert surf[3] is loc[3] and surf[1] is fake[1] and 4 not in surf          # 局域扫描优先于乘积模型


def test_snapshots_to_adf_and_validation():
    """MD 快照 → 扫描项目 (ADF 批处理) → 模拟 ADF (EHT) → 与 EHT 曲面逐点比较：几何完全一致时误差只来自输出的舍入。
    同时覆盖 g ≠ E 的人字形二聚体 (B 的坐标系需绕自身二重轴) 和命令行 snapshots / validate。"""
    import tempfile
    from pathlib import Path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fake_adf import run_scan
    from mobility_tool import cli
    from mobility_tool.ceds import EHTSurface
    from mobility_tool.rigidmd import run_temperature, snapshot_configs, snapshot_validation
    from mobility_tool.scanflow import ScanProject
    cif90 = os.path.join(DATA, "pentacene_90K.cif")
    with tempfile.TemporaryDirectory() as d:
        crys = read_cif(cif90)
        mols = find_molecules(crys)
        mono = Path(d) / "pent.xyz"
        with open(mono, "w") as fh:
            fh.write(f"{len(mols[0]['symbols'])}\npentacene\n")
            for s_, x in zip(mols[0]["symbols"], mols[0]["cart"]):
                fh.write(f"{s_} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n")
        p = ScanProject.create(Path(d) / "proj", mono, "pent")
        lv = p.add_level("PW91", "TZP")
        (p.level_dir(lv) / "mono.t21").write_bytes(b"x")
        ce = CEDS(crys, mols, monomer=(p.symbols, p.Y))
        md = RigidCrystalMD(crys, ce.frames, (p.symbols, p.Y), (2, 2, 1))
        fr = run_temperature(md, 300.0, dict(equil_ps=0.2, prod_ps=0.4))
        g_of = {q["gid"]: q["g"] for q in ce.pairs}
        gids = {1, 3}
        assert g_of[1] != "E" and g_of[3] == "E"
        cfgs = snapshot_configs(ce, md, fr, gids, n_per_pair=10)
        assert len(cfgs) == 20 and {c["group"] for c in cfgs} == {"1", "3"}
        info = p.add_md_snapshots("snap", lv, cfgs, info={"T": 300.0})
        assert info["n_computed"] == 20 and info["kind"] == "md"
        run_scan(p, "snap")
        recs = [r for r in p.collect("snap") if r["state"] == "done"]
        eht = EHTSurface(ce.sym, ce.Y)
        val = snapshot_validation(ce, recs, {g: eht for g in gids})
        for g in gids:
            for c in ("hole", "electron"):
                v = val[g][c]
                assert v["rmse"] < 0.02 and abs(v["sigma_ratio"] - 1) < 1e-3, (g, c, v["rmse"], v["sigma_ratio"])
        from mobility_tool.ceds import surfaces_from_project
        surfaces_from_project(p, ce)                     # 快照扫描不参与曲面拟合 (此处没有其它数据：不应报错)
        out = Path(d) / "val.csv"
        cli.main(["ceds", "validate", "--cif", cif90, "--project", str(p.root), "--name", "snap", "--source", "eht",
                  "--out", str(out)])
        assert out.exists() and out.with_suffix(".png").exists()
        txt = out.read_text(encoding="utf-8-sig")
        assert "sigma_ratio" in txt and "V_dft_hole" in txt
        cli.main(["ceds", "snapshots", "--cif", cif90, "--project", str(p.root), "--name", "snap2", "--pairs", "3",
                  "--n-snap", "6", "--supercell", "2,2,1", "--md-equil", "0.1", "--md-prod", "0.2", "--temps", "150"])
        assert ScanProject(p.root).scan_info("snap2")["n_computed"] == 6


def test_frames_for_lower_symmetry_molecule():
    """红荧烯 (C2h)：主轴符号约定不能保证各分子一致，坐标系须与参考单体逐原子匹配；共轭主体坐标系给出常用的
    π 堆积描述 (面间距 3.70 Å、长轴滑移 6.15 Å)。并五苯 (D2h) 的参数不受影响。"""
    import tempfile
    from pathlib import Path
    from mobility_tool.crystal import find_dimers
    from mobility_tool.scanflow import ScanProject
    c = read_cif(os.path.join(DATA, "rubrene_QQQCIG09.cif"))
    m = find_molecules(c)
    assert len(m) == 4 and all(len(x["symbols"]) == 70 for x in m)
    for mode in ("all", "core"):
        ce = CEDS(c, m, cutoff=4.0, frame=mode)
        assert max(ce.frame_rmsd) < 0.05, (mode, ce.frame_rmsd)
        for k, (ax, o) in enumerate(ce.frames):                   # 刚性单体放回晶体与实际分子重合
            P = np.asarray(ce.Y) @ ax.T + o
            assert np.abs(np.sort(P, axis=0) - np.sort(np.asarray(m[k]["cart"]), axis=0)).max() < 0.05
        md = RigidCrystalMD(c, ce.frames, (ce.sym, ce.Y), (1, 2, 1))
        assert -60 < md.lattice_energy() < -30
    ce = CEDS(c, m, cutoff=4.0, frame="core")
    g1 = ce.pairs[0]["center"]
    assert abs(abs(g1["z"]) - 3.70) < 0.02 and abs(abs(g1["dx"]) - 6.15) < 0.02 and abs(g1["dy"]) < 0.02
    half = ce.suggest_half(T=300.0, n_samples=300)
    assert all(h["gamma"] < 30 for h in half.values()), half          # 中心在 γ = 180° 时角度须连续展开
    with tempfile.TemporaryDirectory() as d:
        mono = Path(d) / "rub.xyz"
        with open(mono, "w") as fh:
            fh.write(f"70\nrubrene\n")
            for s_, x in zip(m[1]["symbols"], m[1]["cart"]):
                fh.write(f"{s_} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n")
        p = ScanProject.create(Path(d) / "proj", mono, "rub", frame="core")
        assert p.frame == "core"
        cfgs = p.crystal_configs(find_dimers(c, m, ref=0, cutoff=4.0))
        assert max(x["rmsd_heavy_A"] for x in cfgs) < 0.05, [x["rmsd_heavy_A"] for x in cfgs]
    cp = read_cif(CIF)
    cep = CEDS(cp, find_molecules(cp), cutoff=4.0)
    q = cep.pairs[0]["center"]
    assert cep.pairs[0]["g"] == "C2z" and abs(q["z"] - 4.645) < 0.01 and abs(q["roll"] - 52.04) < 0.01


if __name__ == "__main__":
    for f in (test_general_frame_and_start_params, test_lattice_energy_and_conservation, test_md_series_and_export,
              test_local_csv_and_combined_surfaces, test_snapshots_to_adf_and_validation,
              test_frames_for_lower_symmetry_molecule):
        f()
        print("ok", f.__name__)
