"""转移积分深入分析 (原子自动匹配 / 多能级矩阵 / 原子对分解)、几何检查、红荧烯 CIF、ADF 片段共用。"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import mobility_tool  # noqa: E402,F401
from fake_fchk import write_fake_fchk  # noqa: E402
from mobility_tool.gaussian_io import read_xyz  # noqa: E402


def test_analysis():
    from mobility_tool.eht import eht_transfer_integrals_full
    from mobility_tool.ti_analysis import apta, from_eht, from_fchk, level_matrix
    from mobility_tool.transfer_integral import calc_transfer_integrals
    from mobility_tool.substitution import substitute
    from mobility_tool.ti_analysis import degenerate_levels
    a = read_xyz(ROOT / "examples/benzene_dimer.xyz")
    sa, xa, _, _ = substitute([x[0] for x in a[:12]], [x[1:] for x in a[:12]], [0], "N")   # 吡啶：无简并轨道
    A = [(s_, *p) for s_, p in zip(sa, xa)]
    B = [(s_, x + 0.9, y + 0.4, z + 3.6) for s_, x, y, z in A]
    ref = eht_transfer_integrals_full(A, B, orbitals=("HOMO", "LUMO", "HOMO-1"))["results"]
    benz = from_eht(a[:12], a[12:])
    assert degenerate_levels(benz, 2), "苯的 HOMO/HOMO-1 简并应被识别"
    assert not degenerate_levels(from_eht(A, B), 2)
    d = from_eht(A, B)
    lm = {(r["A"], r["B"]): r for r in level_matrix(d, 2)}
    for lab in ("HOMO", "LUMO", "HOMO-1"):
        assert abs(abs(lm[(lab, lab)]["J_eff_meV"]) - abs(ref[lab]["J_eff_meV"])) < 1e-6   # 轨道符号任意
    r = apta(d, "HOMO")
    assert abs(r["total_meV"] - r["J_eff_meV"]) < 1e-8 and r["U"].shape == (11, 11)
    with tempfile.TemporaryDirectory() as t:
        write_fake_fchk(f"{t}/A.fchk", A)
        write_fake_fchk(f"{t}/B.fchk", B)
        perm = np.random.default_rng(3).permutation(22)
        write_fake_fchk(f"{t}/AB.fchk", [(A + B)[k] for k in perm])        # 二聚体原子顺序打乱
        g = from_fchk(f"{t}/A.fchk", f"{t}/B.fchk", f"{t}/AB.fchk")
        assert g["reordered"]
        gm = {(r_["A"], r_["B"]): r_ for r_ in level_matrix(g, 2)}
        for lab in ("HOMO", "LUMO", "HOMO-1"):
            assert abs(abs(gm[(lab, lab)]["J_eff_meV"]) - abs(ref[lab]["J_eff_meV"])) < 1e-3
        rg = apta(g, "LUMO")
        assert abs(rg["total_meV"] - rg["J_eff_meV"]) < 1e-8
        res = calc_transfer_integrals(f"{t}/A.fchk", f"{t}/B.fchk", f"{t}/AB.fchk")
        assert abs(abs(res["results"]["HOMO"]["J_eff_meV"]) - abs(ref["HOMO"]["J_eff_meV"])) < 1e-3 and res["warnings"]
    print("转移积分深入分析 OK: 多能级矩阵 = 原实现 (吡啶)；苯的简并轨道被识别；fchk 原子顺序打乱时自动匹配；Σu_αβ = J_eff")


def test_geometry_check():
    from mobility_tool.crystal import check_dimer_geometry
    a = read_xyz(ROOT / "examples/benzene_dimer.xyz")
    for bad in ((a[:12], a[:12]), (a[:12] + [a[0]], a[12:]), (a[:12], [(s, x, y, z - 2.9) for s, x, y, z in a[12:]])):
        try:
            check_dimer_geometry(*bad)
            raise AssertionError("未拦截")
        except ValueError:
            pass
    assert check_dimer_geometry(a[:12], a[12:]) is None
    print("几何检查 OK: 同一分子两次 / 重复原子 / 分子重叠均被拦截")


def test_rubrene():
    from mobility_tool.adf import _match_order, _same_fragment
    from mobility_tool.crystal import find_dimers, find_molecules, read_cif
    from mobility_tool.workflow import prepare_dimer_jobs
    from scipy.spatial.distance import pdist
    c = read_cif(ROOT / "tests/data/Rubrene.cif")
    assert len(c["symbols"]) == 280 and pdist(c["frac"] @ c["lattice"]).min() > 0.9
    m = find_molecules(c)
    assert [len(x["symbols"]) for x in m] == [70] * 4
    r = find_dimers(c, m, cutoff=4.0)
    assert len(r["groups"]) == 4
    A = [(s, *x) for s, x in r["atoms_a"]]
    for g in r["groups"]:
        B = [(s, *x) for s, x in g["members"][0]["atoms_b"]]
        p = _match_order(A, B)
        assert p is not None and _same_fragment(A, [B[k] for k in p])
    with tempfile.TemporaryDirectory() as t:
        prepare_dimer_jobs(r, t, "gaussian", method="B3LYP", basis="6-31G(d)")
        prepare_dimer_jobs(r, f"{t}/adf", "adf", functional="PW91", basis="TZP", style="classic")
        assert "1 个单体片段 + 4 个二聚体" in Path(f"{t}/adf/batch_adf.sh").read_text()
    print("红荧烯 OK: 280 原子 / 4 个完整分子 / 4 组二聚体；Gaussian 输入无原子重叠；ADF 4 组共用 1 个单体片段")


if __name__ == "__main__":
    test_analysis()
    test_geometry_check()
    test_rubrene()
