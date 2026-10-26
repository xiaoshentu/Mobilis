"""对映体 (手性分子的中心对称晶体)：分子坐标系取非真转动，二聚体加 mirror 标记，B 用镜像单体构建；
ADF 批处理引用镜像片段 mono_m.t21；全晶体 MD 的初始原子位置与晶体一致；ESP 电荷与 Al/Ir 等元素的力场参数。"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from fake_adf import FakeADF  # noqa: E402
from mobility_tool.ceds import CEDS, EHTSurface  # noqa: E402
from mobility_tool.crystal import find_molecules, read_cif  # noqa: E402
from mobility_tool.rigidmd import RigidCrystalMD, snapshot_configs  # noqa: E402
from mobility_tool.scanflow import ScanProject  # noqa: E402
from mobility_tool.surrogate import _matched_rmsd, build_B  # noqa: E402
from mobility_tool.workflow import eht_couplings_for_crystal  # noqa: E402

# 手性的刚性分子：CHFClBr (四面体)
MOL = [("C", 0.0, 0.0, 0.0), ("H", 0.63, 0.63, 0.63), ("F", -0.78, -0.78, 0.78), ("Cl", -1.02, 1.02, -1.02),
       ("Br", 1.12, -1.12, -1.12)]


def chiral_cif(path):
    """P1 晶胞中放一个分子及其反演像 (对映体)。"""
    lat = np.array([[7.4, 0, 0], [0.8, 7.9, 0], [0.5, 0.9, 8.3]])
    inv = np.linalg.inv(lat)
    c1 = np.array([2.0, 2.2, 2.3])
    atoms = []
    for s, *x in MOL:
        atoms.append((s, (np.array(x) + c1) @ inv))
    for s, *x in MOL:
        atoms.append((s, (-(np.array(x) + c1)) @ inv % 1.0))
    a, b, c = (np.linalg.norm(v) for v in lat)
    al = np.degrees(np.arccos(lat[1] @ lat[2] / b / c))
    be = np.degrees(np.arccos(lat[0] @ lat[2] / a / c))
    ga = np.degrees(np.arccos(lat[0] @ lat[1] / a / b))
    L = ["data_chiral", f"_cell_length_a {a:.5f}", f"_cell_length_b {b:.5f}", f"_cell_length_c {c:.5f}",
         f"_cell_angle_alpha {al:.4f}", f"_cell_angle_beta {be:.4f}", f"_cell_angle_gamma {ga:.4f}",
         "_symmetry_space_group_name_H-M 'P 1'", "loop_", "_symmetry_equiv_pos_as_xyz", "x,y,z", "loop_",
         "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
         "_atom_site_fract_z"]
    for i, (s, f) in enumerate(atoms):
        L.append(f"{s}{i + 1} {s} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f}")
    open(path, "w").write("\n".join(L) + "\n")


def test_enantiomer_pairs():
    d = tempfile.mkdtemp()
    cif = os.path.join(d, "chiral.cif")
    chiral_cif(cif)
    crys = read_cif(cif)
    mols = find_molecules(crys)
    assert len(mols) == 2
    ce = CEDS(crys, mols)
    assert ce.mirror == [False, True], ce.mirror
    assert max(ce.frame_rmsd) < 1e-3
    assert any("对映体" in n for n in ce.notes)
    lat = np.asarray(crys["lattice"])
    res, cp = eht_couplings_for_crystal(crys, mols, 4.0)
    eht_by_r = {round(g["members"][0]["r_centroid"], 3): cp[g["id"]] for g in res["groups"]}
    n_m = 0
    for p in ce.pairs:
        axA, cA = ce.frames[p["ref"]]
        Xb = np.asarray(mols[p["mol"]]["cart"]) + np.asarray(p["translation"], float) @ lat
        q = p["q0"]
        Bq = build_B(ce.Y, q[0], q[1], q[2], q[3], q[5], q[4], mirror=p["mirror"])
        assert _matched_rmsd(ce.sym, Bq, mols[p["mol"]]["symbols"], (Xb - cA) @ axA) < 1e-6
        n_m += p["mirror"]
        # 参数 + 镜像标记计算的 EHT 耦合 = 直接用晶体原子计算的耦合 (|V|，相位约定可不同)
        v = EHTSurface(ce.sym, ce.Y, mirror=p["mirror"]).eval_one(q)
        ref = eht_by_r.get(round(p["r_centroid"], 3))
        if ref is not None:
            for c in ("hole", "electron"):
                assert abs(abs(v[c]) - abs(ref[c])) < 0.05 + 1e-3 * abs(ref[c]), (p["gid"], c, v[c], ref[c])
    assert n_m >= 1
    # 全晶体 MD：初始原子位置 = 晶体；分子坐标系还原；短时 NVE 能量守恒
    md = RigidCrystalMD(crys, ce.frames, (ce.sym, ce.Y), (3, 3, 3), rc=8.0, r_sw=7.0)
    X = md.atoms(md.c0, md.Q0)
    assert max(_matched_rmsd(ce.sym, X[k], mols[k]["symbols"], np.asarray(mols[k]["cart"])) for k in range(2)) < 1e-6
    Qm, _ = md.mol_frames(md.c0, md.Q0)
    assert np.abs(Qm[1] - ce.frames[1][0]).max() < 1e-9 and np.linalg.det(Qm[1]) < 0
    fr, st = md.run(100, 200, 2.0, 1e-2, every=200, seed=3)
    e0 = md.total_energy(st)
    fr, st = md.run(100, 200, 1.0, 0.0, every=200, seed=4, state=st)
    assert abs(md.total_energy(st) - e0) / md.N < 0.02
    # 快照构型带 mirror；扫描项目写镜像片段与批处理；模拟 ADF 结果与 EHT 曲面一致
    fr, _ = md.run(100, 400, 2.0, 1e-3, every=20, seed=5)
    gids = [p["gid"] for p in ce.pairs if p["mirror"]][:1] + [p["gid"] for p in ce.pairs if not p["mirror"]][:1]
    cfgs = snapshot_configs(ce, md, fr, gids, n_per_pair=4)
    assert any(c.get("mirror") for c in cfgs)
    assert all(bool(c.get("mirror")) == next(p["mirror"] for p in ce.pairs if str(p["gid"]) == c["group"]) for c in cfgs)
    mono = os.path.join(d, "mono.xyz")
    with open(mono, "w") as fh:
        fh.write(f"{len(ce.sym)}\n\n" + "".join(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(ce.sym, ce.Y)))
    pr = ScanProject.create(os.path.join(d, "proj"), mono, name="chiral")
    lv = pr.add_level()
    pr.add_scan("snap", cfgs, lv, kind="al", reduce=False, clash_min=0.0)
    lvdir = pr.level_dir(lv)
    assert (lvdir / "mono_m.run").exists()
    mrun = (lvdir / "mono_m.run").read_text()
    assert "mirror-image" in mrun
    batch = open(next(iter(sorted(pr.scan_dir("snap").glob("batch_*.run")))), encoding="ascii").read()
    assert "mono_m " in batch and "f=mono_m/1" in batch and "f=mono/1" in batch
    pre = pr.precheck("snap")
    assert any("mono_m" in e for e in pre["errors"])
    fa = FakeADF(pr.symbols, pr.Y)
    for r in pr.manifest("snap"):
        if r["status"] != "ok":
            continue
        p = next(p for p in ce.pairs if str(p["gid"]) == r["group"])
        res_ = fa.results(r)[0]
        q = np.array([r[k] for k in ("z", "dx", "dy", "gamma", "pitch", "roll")])
        v = EHTSurface(ce.sym, ce.Y, mirror=p["mirror"]).eval_one(q)
        assert abs(res_["HOMO"]["J_eff_meV"] - v["hole"]) < 1e-6
        # 批处理中的几何 = 由参数 (含镜像) 重建的几何
        A, B = pr.dimer_atoms(r)
        Bq = build_B(pr.Y, *q[:3], q[3], q[5], q[4], mirror=p["mirror"])
        assert np.abs(np.array([x[1:] for x in B]) - Bq).max() < 1e-9


def test_charges_and_elements():
    from mobility_tool.interaction import UFF
    from mobility_tool.rigidmd import MASS
    for e in ("Al", "Ir", "Pt", "Zn", "Cu"):
        assert e in MASS and e in UFF
    d = tempfile.mkdtemp()
    cif = os.path.join(d, "chiral.cif")
    chiral_cif(cif)
    crys = read_cif(cif)
    ce = CEDS(crys, find_molecules(crys))
    q = np.array([0.1, 0.05, -0.2, 0.15, -0.05])
    md = RigidCrystalMD(crys, ce.frames, (ce.sym, ce.Y), (3, 3, 3), rc=8.0, r_sw=7.0, charges=q + 0.01)
    assert np.allclose(md.q, q + 0.01 - (q + 0.01).mean()) and abs(md.q.sum()) < 1e-12
    assert "ESP" in md.ff_note
    try:
        RigidCrystalMD(crys, ce.frames, (ce.sym, ce.Y), (3, 3, 3), charges=[0.1, 0.2])
        raise AssertionError("电荷数不符时应报错")
    except ValueError:
        pass


if __name__ == "__main__":
    test_enantiomer_pairs()
    print("ok test_enantiomer_pairs")
    test_charges_and_elements()
    print("ok test_charges_and_elements")
