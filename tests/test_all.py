"""用合成数据检验三个模块（不需要安装高斯）。运行: python tests/test_all.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobility_tool import (analyze_reorg, calc_transfer_integrals, compute_mobility,  # noqa: E402
                           generate_reorg_inputs, marcus_rate)
from mobility_tool.gaussian_io import BOHR2ANG, HARTREE2EV  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "benzene.xyz"


def fake_log(energies, freqs=(120.0,), windows=True, charges=None, opt_first=True, homo=-0.21, lumo=-0.01):
    """仿真高斯 log：每个 Link1 步骤有 route 段；Windows 版没有 'Entering Gaussian System'；
    第一个步骤为 opt freq，会额外产生一个内部频率步骤 (Geom=AllCheck ... Freq)。"""
    out = []
    charges = charges or [(0, 1)] * len(energies)
    for i, (e, (c, m)) in enumerate(zip(energies, charges)):
        if not windows:
            out.append(" Entering Gaussian System, Link 0=g16\n")
        route = "#p opt freq b3lyp/6-31g(d)" if (i == 0 and opt_first) else "#p b3lyp/6-311g(2d,2p) geom=check"
        out.append(f" %chk=x.chk\n -------------------------------\n {route}\n -------------------------------\n"
                   f" 1/18=20/1,3;\n ----------\n title {i}\n ----------\n Charge = {c:2d} Multiplicity = {m}\n")
        out.append(f" SCF Done:  E(RB3LYP) =  {e - 0.01:.10f}     A.U. after   12 cycles\n")
        out.append(f" Alpha  occ. eigenvalues --  -10.18920 -10.18900{homo - 0.3:10.5f}{homo:10.5f}\n"
                   f" Alpha virt. eigenvalues --  {lumo:10.5f}   0.05000\n")
        out.append(f" SCF Done:  E(RB3LYP) =  {e:.10f}     A.U. after    3 cycles\n")
        if i == 0 and opt_first:
            out.append(" Optimization completed.\n Normal termination of Gaussian 16 at Sat Oct  3.\n"
                       " Link1:  Proceeding to internal job step number  2.\n -------------------------------\n"
                       " #P Geom=AllCheck Guess=TCheck SCRF=Check GenChk RB3LYP/6-31G(d) Freq\n"
                       " -------------------------------\n")
            out.append(f" SCF Done:  E(RB3LYP) =  {e:.10f}     A.U. after    1 cycles\n")
            out.append(" Frequencies --  " + "  ".join(f"{f:.4f}" for f in freqs) + "\n")
        out.append(" Normal termination of Gaussian 16 at Sat Oct  3 12:00:00 2026.\n")
    return "".join(out)


def test_reorg():
    E = {"nn": -232.000, "cn": -231.650, "an": -231.990, "nc": -231.996, "cc": -231.653,
         "na": -231.997, "aa": -231.9935}
    lh = ((E["nc"] - E["nn"]) + (E["cn"] - E["cc"])) * HARTREE2EV
    le = ((E["na"] - E["nn"]) + (E["an"] - E["aa"])) * HARTREE2EV
    for windows in (True, False):
        with tempfile.TemporaryDirectory() as d:
            generate_reorg_inputs(EXAMPLE, d)
            Path(d, "benzene_neutral.log").write_text(fake_log([E["nn"], E["cn"], E["an"]], windows=windows))
            Path(d, "benzene_cation.log").write_text(fake_log([E["cc"], E["nc"]], windows=windows))
            Path(d, "benzene_anion.log").write_text(fake_log([E["aa"], E["na"]], freqs=(-35.0, 80.0), windows=windows))
            res = analyze_reorg(d)
            assert abs(res["results"]["hole"]["lambda_eV"] - lh) < 1e-9
            assert abs(res["results"]["electron"]["lambda_eV"] - le) < 1e-9
            assert abs(res["ip_ea"]["IP_v"]["eV"] - (E["cn"] - E["nn"]) * HARTREE2EV) < 1e-9
            assert abs(res["ip_ea"]["EA_a"]["eV"] - (E["nn"] - E["aa"]) * HARTREE2EV) < 1e-9
            assert abs(res["orbitals"]["HOMO_eV"] - (-0.21 * HARTREE2EV)) < 1e-6
            assert any("虚频" in w for w in res["warnings"]) and not any("步骤" in w for w in res["warnings"])
    # 单点级别与优化级别不同：每个文件多一个同态单点，能量只取单点级别
    with tempfile.TemporaryDirectory() as d:
        m = generate_reorg_inputs(EXAMPLE, d, method="B3LYP", basis="6-311G(d,p)", sp_basis="6-311G(2d,2p)")
        assert m["separate_sp"] and len(m["files"]["benzene_neutral"]) == 4
        off = 0.5   # 优化级别能量故意偏离，若误用会被发现
        Path(d, "benzene_neutral.log").write_text(fake_log([E["nn"] + off, E["nn"], E["cn"], E["an"]]))
        Path(d, "benzene_cation.log").write_text(fake_log([E["cc"] + off, E["cc"], E["nc"]]))
        Path(d, "benzene_anion.log").write_text(fake_log([E["aa"] + off, E["aa"], E["na"]]))
        res = analyze_reorg(d)
        assert abs(res["results"]["hole"]["lambda_eV"] - lh) < 1e-9
        assert abs(res["ip_ea"]["HEP"]["eV"] - (E["cc"] - E["nc"]) * HARTREE2EV) < 1e-9
    print(f"重组能 / IP / EA OK: λh={lh:.4f} eV, λe={le:.4f} eV（Windows/Linux 格式、单点级别分离均正确）")


def write_fchk(path, coef, energies, nocc, coords_ang):
    nbf = coef.shape[0]

    def arr(name, typ, vals):
        vals = list(vals)
        per = 6 if typ == "I" else 5
        fmt = "{:12d}" if typ == "I" else "{:16.8E}"
        s = f"{name:<40s}   {typ}   N={len(vals):12d}\n"
        for i in range(0, len(vals), per):
            s += "".join(fmt.format(v) for v in vals[i:i + per]) + "\n"
        return s

    txt = "synthetic\nSP        RB3LYP                                                      6-31G(d)\n"
    txt += f"{'Number of atoms':<40s}   I     {len(coords_ang):12d}\n"
    txt += f"{'Route':<40s}   C   N=           2\n#P B3LYP/6-31G(d) nosymm\n"
    txt += f"{'Number of alpha electrons':<40s}   I     {nocc:12d}\n"
    txt += f"{'Number of basis functions':<40s}   I     {nbf:12d}\n"
    txt += f"{'Number of independent functions':<40s}   I     {nbf:12d}\n"
    txt += arr("Current cartesian coordinates", "R", (np.asarray(coords_ang) / BOHR2ANG).ravel())
    txt += arr("Alpha Orbital Energies", "R", energies)
    txt += arr("Alpha MO coefficients", "R", coef.T.ravel())   # 每个 MO 连续存放
    Path(path).write_text(txt)


def test_transfer_integral():
    rng = np.random.default_rng(1)
    na, nb = 8, 7
    n = na + nb
    x = rng.normal(size=(n, n)) * 0.25
    s = x @ x.T + np.eye(n) * 2.0
    dinv = 1 / np.sqrt(np.diag(s))
    s = s * np.outer(dinv, dinv)                       # 对角元为 1 的正定重叠矩阵
    f = rng.normal(size=(n, n)) * 0.2
    f = (f + f.T) / 2 + np.diag(np.linspace(-1.0, 0.5, n))
    eA, cA = eigh(f[:na, :na], s[:na, :na])
    eB, cB = eigh(f[na:, na:], s[na:, na:])
    eD, cD = eigh(f, s)
    coords = rng.normal(size=(5, 3))
    with tempfile.TemporaryDirectory() as d:
        write_fchk(f"{d}/A.fchk", cA, eA, 4, coords[:3])
        write_fchk(f"{d}/B.fchk", cB, eB, 3, coords[3:])
        write_fchk(f"{d}/AB.fchk", cD, eD, 7, coords)
        res = calc_transfer_integrals(f"{d}/A.fchk", f"{d}/B.fchk", f"{d}/AB.fchk", ["HOMO", "LUMO"])
    cfrag = np.zeros((n, n))
    cfrag[:na, :na], cfrag[na:, na:] = cA, cB
    h_ref, s_ref = cfrag.T @ f @ cfrag, cfrag.T @ s @ cfrag
    for lab, ia, ib in (("HOMO", 3, na + 2), ("LUMO", 4, na + 3)):
        j, ea, eb, sab = h_ref[ia, ib], h_ref[ia, ia], h_ref[ib, ib], s_ref[ia, ib]
        jeff = (j - 0.5 * (ea + eb) * sab) / (1 - sab ** 2) * HARTREE2EV * 1000
        got = res["results"][lab]["J_eff_meV"]
        assert abs(got - jeff) < 1e-3 * max(1, abs(jeff)), (lab, got, jeff)
        print(f"转移积分 OK: {lab} J_eff = {got:.4f} meV (参考值 {jeff:.4f} meV)")
    assert not res["warnings"], res["warnings"]


def test_mobility():
    k = marcus_rate(0.05, 0.2, 300.0)
    assert 1e12 < k < 1e14, k
    paths = [{"vector": [5.0, 0, 0], "J_meV": 50, "count": 2}]
    r = compute_mobility(0.2, paths, 300.0, dimension=1)
    d_ref = 0.5 * 2 * k * (5e-8) ** 2          # 1D 随机行走 D = ½ Σ k r²
    assert abs(r["mobility_tensor_cm2_Vs"][0][0] - d_ref / (8.617333262e-5 * 300)) < 1e-12
    print(f"迁移率 OK: k = {k:.3e} s^-1, μ(1D) = {r['mobility_cm2_Vs']:.3f} cm^2/Vs")


if __name__ == "__main__":
    test_reorg()
    test_transfer_integral()
    test_mobility()
    print("\n全部测试通过")
