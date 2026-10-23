"""DFT 校准 (形貌模块) 与按载流子分开的 ZINDO 检验统计。"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool import morphology as Mo  # noqa: E402

GRO = os.path.join(ROOT, "tests", "data", "pentacene_300K.gro")
REF = os.path.join(ROOT, "tests", "data", "pentacene_ref.xyz")


def _morph():
    from mobility_tool.cli import _read_monomer
    sym, Y = _read_monomer(REF)
    m = Mo.Morphology(Mo.read_trajectory(GRO, None, {0})[0], sym, Y)
    m.neighbors(4.0)
    m.couplings_eht()
    return m


def test_calibration_recovers_scale():
    m = _morph()
    raw = {c: v.copy() for c, v in m.J_raw.items()}
    idx = m.calibration_pairs(20)
    assert len(idx) == 20 and idx == sorted(idx) and len(set(idx)) == 20
    tot = np.abs(raw["hole"]) + np.abs(raw["electron"])
    assert tot[idx].max() >= 0.9 * tot.max() and tot[idx].min() < np.median(tot[tot > 1e-3])   # 覆盖强弱两端
    rng = np.random.default_rng(1)
    ref = {k: (-2.2 * raw["hole"][k] * (1 + 0.05 * rng.standard_normal()),
               2.4 * raw["electron"][k] * (1 + 0.05 * rng.standard_normal())) for k in idx}
    res = m.calibrate(ref)
    assert abs(res["hole"]["k"] - 2.2) < 0.1 and abs(res["electron"]["k"] - 2.4) < 0.1
    assert res["hole"]["rel_after"] < 0.1 < res["hole"]["rel_before"]
    k0 = idx[0]
    assert m.J["hole"][k0] == ref[k0][0]                                     # 有参考值的对直接用参考值 (带符号)
    other = [k for k in range(len(m.pairs)) if k not in ref][:5]
    assert np.allclose(m.J["hole"][other], res["hole"]["k"] * raw["hole"][other])
    assert "DFT 校准" in m.summary() and "k =" in m.calibration_text()
    m.calibrate(ref, replace=False)
    assert np.allclose(m.J["electron"][idx], m.calib["electron"]["k"] * raw["electron"][idx])
    m.calibrate_scale(2.0, 0.5)
    assert np.allclose(m.J["hole"], 2 * raw["hole"]) and "手动" in m.calibration_text()
    d = tempfile.mkdtemp()
    m.export(os.path.join(d, "x"))
    head = open(os.path.join(d, "x_pairs.csv"), encoding="utf-8-sig").readline()
    assert "J_hole_semi_meV" in head
    m.couplings_eht()
    assert m.calib is None                                                    # 重新计算后清除旧校准


def test_calibration_errors_and_csv():
    m = Mo.Morphology(Mo.read_trajectory(GRO, None, {0})[0], *__import__("mobility_tool.cli", fromlist=["x"])._read_monomer(REF))
    m.neighbors(4.0)
    try:
        m.calibrate({0: (0.01, 0.01)})
        raise AssertionError("未计算半经验 J 时应报错")
    except ValueError as e:
        assert "EHT" in str(e)
    m.couplings_eht()
    try:
        m.calibrate({0: (0.05, 0.05)})
        raise AssertionError("参考对不足时应报错")
    except ValueError as e:
        assert "少于 2 个" in str(e)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "ref.csv")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("pair,J_hole_meV,J_electron_meV\n")
        for k in m.calibration_pairs(8):
            fh.write(f"{k},{3000 * m.J_raw['hole'][k]},{1500 * m.J_raw['electron'][k]}\n")
    res = m.calibrate(m.read_couplings_csv(p))
    assert abs(res["hole"]["k"] - 3) < 1e-9 and abs(res["electron"]["k"] - 1.5) < 1e-9
    r = subprocess.run([sys.executable, "-m", "mobility_tool", "morph", "--gro", GRO, "--monomer", REF,
                        "--calibrate-csv", p, "--site-energy", "none", "--quiet"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert "k = 3.000" in r.stdout and "k = 1.500" in r.stdout and "DFT 校准" in r.stdout
    out = os.path.join(d, "cal")
    r = subprocess.run([sys.executable, "-m", "mobility_tool", "morph", "--gro", GRO, "--monomer", REF,
                        "--export-dimers", out, "--calib-pairs", "6", "--site-energy", "none", "--quiet"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert len([x for x in os.listdir(out) if x.startswith("pair_")]) == 6


def test_benchmark_stats_by_carrier():
    from mobility_tool import zindo as Z
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        d = 10 + 30 * i
        rows.append({"case": f"c{i}", "dft_h": d, "dft_e": -d, "moo_h": 1.1 * d * (1 + 0.02 * rng.standard_normal()),
                     "moo_e": -0.8 * d, "eht_h": 0.45 * d, "eht_e": 0.42 * d})
    st = Z.benchmark_stats(rows)
    by = {(s["key"], s["carrier"]): s for s in st}
    assert abs(by[("moo", "h")]["slope"] - 1.1) < 0.02 and abs(by[("moo", "e")]["slope"] - 0.8) < 1e-9
    assert by[("moo", "e")]["rel_err_scaled"] < 1e-9 and abs(by[("eht", "h")]["rel_err"] - 0.55) < 1e-9
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2)
    Z.plot_benchmark(axs, rows)
    assert axs[0].get_title() == "空穴" and axs[1].get_title() == "电子" and len(axs[1].lines) == 3


if __name__ == "__main__":
    for f in (test_calibration_recovers_scale, test_calibration_errors_and_csv, test_benchmark_stats_by_carrier):
        f()
        print("ok", f.__name__)
