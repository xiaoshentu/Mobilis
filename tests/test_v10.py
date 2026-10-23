"""相互作用能 (UFF / Gaussian CP / 晶格能)、结构测量、ADF .out 解析与读取链、FCHT 诊断、黄昆因子导入。"""
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import mobility_tool  # noqa: E402,F401


def test_interaction():
    from mobility_tool.gaussian_io import read_xyz
    from mobility_tool.interaction import KCAL2KJ, read_counterpoise, uff_interaction
    a = read_xyz(ROOT / "examples/benzene_dimer.xyz")
    A = a[:12]
    e = [uff_interaction(A, [(s, x, y, z - 3.6 + d) for s, x, y, z in a[12:]])["total"] for d in np.arange(3.0, 8.01, 0.1)]
    k = int(np.argmin(e))
    assert 0 < k < len(e) - 1 and e[0] > e[k] and abs(e[-1]) < abs(e[k]) / 5           # 有极小、近处排斥、远处趋于 0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cp.log"
        p.write_text(" BSSE energy =  0.0023\n complexation energy = -9.87 kcal/mole (raw)\n"
                     " complexation energy = -8.40 kcal/mole (corrected)\n Normal termination of Gaussian 16\n")
        r = read_counterpoise(p)
        assert abs(r["total"] + 8.40 * KCAL2KJ) < 1e-9 and r["normal"]
    print(f"相互作用能 OK: UFF 曲线极小 {3.0 + 0.1 * k:.1f} Å ({e[k]:.1f} kJ/mol)，Counterpoise 解析正确")


def test_measure():
    from mobility_tool.crystal import measure
    assert measure([(0, 0, 0), (0, 0, 3.6)])[1] == 3.6
    assert abs(measure([(1, 0, 0), (0, 0, 0), (0, 1, 0)])[1] - 90) < 1e-9
    assert abs(abs(measure([(1, 0, 0), (0, 0, 0), (0, 0, 1), (-1, 0, 1)])[1]) - 180) < 1e-9
    print("结构测量 OK: 距离 / 键角 / 二面角")


def test_adf_out():
    from mobility_tool.adf import _organize, parse_adf_output, read_adf_transfer_integrals
    t1 = "  V(hole) = -0.02679 a.u.\n  V(electron) = 0.013129 a.u.\n  S(hole) = 0.0498\n"
    t2 = "                hole      electron\n  V  (eV)   -0.72899    0.35726\n"
    for t in (t1, t2):
        r = _organize(parse_adf_output(t))
        assert abs(r["hole"]["V_meV"] + 728.99) < 0.05 and abs(r["electron"]["V_meV"] - 357.26) < 0.05
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "G1.out").write_text(t1)
        (Path(d) / "G1_full.t21").write_bytes(b"\0")
        (Path(d) / "G1_full.txt").write_text("dmpkf: error\n")
        r = read_adf_transfer_integrals(Path(d) / "G1_full.t21")       # 文本无效 → 读 .out
        assert abs(r["hole"]["V_meV"] + 728.99) < 0.05
    from mobility_tool.adf import check_v
    real = _organize(parse_adf_output((ROOT / "tests/data/adf2016_transfer_integrals.out").read_text()))
    assert abs(real["hole"]["V_meV"] - 77.27) < 1e-6 and abs(real["electron"]["V_meV"] + 77.52) < 1e-6
    assert abs(real["hole"]["J_meV"] - 157.64) < 1e-6 and abs(real["hole"]["S"] + 0.01727) < 1e-9
    for v_out, v_calc in check_v(real).values():
        assert abs(v_out - v_calc) < 0.1                     # 由 J、S、e1、e2 重算的 V 与输出一致
    print("ADF 输出解析 OK: ADF2016 真实输出 (V = 77.27 / −77.52 meV，分量重算一致)、Hartree / eV 写法、二进制不可读时改读 .out")


def test_fcht_and_hr():
    from mobility_tool.gaussian_fcht import diagnose_fcht, explain_fcht_failure, fcht_input, read_hr_table
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        g = d / "x_fcht_cation_to_neutral.gjf"
        g.write_text(fcht_input("x_cation.chk", "x_neutral.chk"))
        assert len(diagnose_fcht(g)) == 2
        for st, txt in (("cation", "opt only"), ("neutral", "Harmonic frequencies")):
            (d / f"x_{st}.chk").write_bytes(b"x" * 20000)
            (d / f"x_{st}.log").write_text(f" {txt}\n Normal termination of Gaussian 16\n")
        pr = diagnose_fcht(g)
        assert len(pr) == 1 and "x_cation.log" in pr[0]
        old = time.time() - 3600
        os.utime(d / "x_neutral.log", (old, old))
        assert any("改写" in p for p in diagnose_fcht(g))
        (d / "x_fcht_cation_to_neutral.log").write_text(" FileIO operation on non-existent file.\n Error termination\n")
        assert "内置" in explain_fcht_failure(d / "x_fcht_cation_to_neutral.log")
        for name, body in (("a.csv", "Frequency(cm-1),HRF\n100,0.2\n1600,0.3\n"),
                           ("b.dat", "# mode freq S lambda\n1 100 0.2 20\n2 1600 0.3 480\n"),
                           ("c.txt", "1 100 0.2\n2 1600 0.3\n")):
            (d / name).write_text(body)
            r = read_hr_table(d / name)
            assert [m["freq_cm-1"] for m in r["modes"]] == [100, 1600] and [m["S"] for m in r["modes"]] == [0.2, 0.3]
    print("FCHT 诊断 OK (缺 chk / 无频率 / chk 被改写)，黄昆因子表格导入 OK (带表头 / 注释表头 / 带序号列)")


if __name__ == "__main__":
    test_interaction()
    test_measure()
    test_adf_out()
    test_fcht_and_hr()
