"""v1.24: π frontier-orbital identification in EHT, CIF loops with blank header lines, ZINDO fallback for
molecules related only by an improper operation."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool.crystal import find_dimers, find_molecules, read_cif  # noqa: E402
from mobility_tool.eht import eht_transfer_integrals, monomer, pi_frontier  # noqa: E402

D = os.path.join(ROOT, "tests", "data")


def _ref(path):
    c = read_cif(os.path.join(D, path))
    m = find_molecules(c)
    r = find_dimers(c, m, 0, 4.0)
    return c, m, r, [(s, *x) for s, x in r["atoms_a"]]


def test_blank_lines_in_loop_header():
    c, m, _, _ = _ref("pyrazinacene_790009_blanklines.cif")
    assert len(m) == 4 and all(len(x["symbols"]) == 36 for x in m)


def test_pi_frontier_hydrocarbon_unchanged():
    _, _, r, A = _ref("pentacene_RT.cif")
    mo = monomer([a[0] for a in A], [a[1:] for a in A])
    ih, il, note = pi_frontier(mo, [a[0] for a in A], [a[1:] for a in A])
    assert (note["HOMO_shift"], note["LUMO_shift"]) == (0, 0) and note["pi_HOMO"] > 0.9
    d = r["groups"][0]["members"][0]
    B = [(s, *x) for s, x in d["atoms_b"]]
    a = eht_transfer_integrals(A, B, esd=False)["results"]
    b = eht_transfer_integrals(A, B, esd=False, pi=True)["results"]
    assert abs(a["HOMO"]["J_eff_meV"] - b["HOMO"]["J_eff_meV"]) < 1e-9


def test_pi_frontier_tetraazapyrene():
    # 1,3,6,8-四氮杂芘：EHT 的 HOMO 是 N 孤对 σ 轨道，最高占据 π 轨道是 HOMO-2；π 堆积对 (4.88 Å) 的空穴耦合由小变大
    _, _, r, A = _ref("TAPy_CF2CF3_938130.cif")
    d = min((g["members"][0] for g in r["groups"]), key=lambda x: abs(x["r_centroid"] - 4.88))
    B = [(s, *x) for s, x in d["atoms_b"]]
    a = eht_transfer_integrals(A, B, esd=False)
    b = eht_transfer_integrals(A, B, esd=False, pi=True)
    assert a["pi_frontier"]["HOMO_shift"] == -2 and a["pi_frontier"]["pi_HOMO"] < 0.1
    assert any("--pi" in w for w in a["warnings"]) and any("HOMO-2" in w for w in b["warnings"])
    assert abs(a["results"]["HOMO"]["J_eff_meV"]) < 5 < 20 < abs(b["results"]["HOMO"]["J_eff_meV"])
    assert abs(a["results"]["LUMO"]["J_eff_meV"] - b["results"]["LUMO"]["J_eff_meV"]) < 1e-9


def test_zindo_improper_fallback():
    from mobility_tool.zindo import MOO, dimer_projection
    _, _, r, A = _ref("TAPy_CF2CF3_938130.cif")
    sym = [a[0] for a in A]
    Xa = np.array([a[1:] for a in A])
    moo = MOO(sym, Xa)
    Xm = Xa * np.array([1.0, 1.0, -1.0]) + np.array([0.0, 0.0, 3.5])      # 镜像分子：只能非真转动叠合
    j = moo.coupling_atoms(Xa, Xm, sym_b=sym)
    ref = dimer_projection(sym, Xa, Xm)
    for car in ("hole", "electron"):                                       # 叠合成功走快速路径，失败走二聚体投影
        assert np.isfinite(j[car]) and abs(abs(j[car]) - abs(ref[car])) <= 0.3 * abs(ref[car]) + 2e-3


def test_frame_refinement_pseudo_c3():
    # mer-Alq3: 主惯量近简并，气相优化单体按主轴放到晶体分子上会转错 (晶格能为正)；最小二乘修正后与邻居不再重叠
    from mobility_tool.ceds import CEDS
    from mobility_tool.cli import _read_monomer
    from mobility_tool.rigidmd import RigidCrystalMD
    c = read_cif(os.path.join(D, "Alq3_beta_ja993608k.cif"))
    m = find_molecules(c)
    mono = _read_monomer(os.path.join(D, "Alq3_B3LYP_opt.xyz"), "all")
    ce = CEDS(c, m, cutoff=4.0, monomer=mono, frame="all")
    md = RigidCrystalMD(c, ce.frames, mono, (2, 2, 2))
    assert md.lattice_energy() < -20 and md.mirror_k == [False, True]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("v1.24 π 轨道识别 / CIF 空行 / ZINDO 回退 / 坐标系修正 OK")
