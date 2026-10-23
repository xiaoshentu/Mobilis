"""迁移率方法对比 (compare)：速率模型的极限关系、KMC 与解析解、能量无序、适用性判断、MLJ 参数。"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility_tool.compare import (compare_methods, export_csv, marcus_scalar, marcus_tensor,  # noqa: E402
                                   mlj_from_modes, regime, table_text)
from mobility_tool.kmc import graph_from_paths, rate_jortner, rate_marcus, run_kmc  # noqa: E402

PATHS = [{"name": "a", "vector": [6.0, 0, 0], "Vh_eV": 0.020, "count": 1},
         {"name": "b", "vector": [0, 7.5, 0], "Vh_eV": 0.010, "count": 1},
         {"name": "d", "vector": [3.0, 3.75, 0], "Vh_eV": 0.015, "count": 1}]


def test_rate_limits():
    # S → 0 时 MLJ 退化为经典部分的 Marcus 速率
    j, lam, T = 0.02, 0.15, 300.0
    assert abs(rate_jortner(j, 1e-9, lam, 0.0, T, hw=0.18) / rate_marcus(j, lam, 0.0, T) - 1) < 1e-6
    # 两种速率对 ΔE 的积分都等于 2π/ħ·J² (Franck–Condon 权重归一)
    from mobility_tool.mobility import HBAR_EVS
    dE = np.linspace(-3.0, 2.0, 20001)
    ref = 2 * np.pi / HBAR_EVS * j ** 2
    for k in (rate_marcus(j, lam, dE, T), rate_jortner(j, 0.10, 0.05, dE, T, hw=0.18)):
        assert abs(np.trapezoid(k, dE) / ref - 1) < 1e-3


def test_kmc_matches_analytic_and_miller():
    g = graph_from_paths(PATHS, "Vh_eV")
    lam, T = 0.15, 300.0
    w = np.linalg.eigvalsh(marcus_tensor(g, lam, T))[::-1]
    r = run_kmc(g, lam, T, method="marcus", walkers=600, steps=3000, seed=3)
    assert abs(r["mu_principal"][0] - w[0]) < 4 * r["mu_avg_err"] + 0.05 * w[0]
    r_ma = run_kmc(g, lam, T, method="miller", walkers=600, steps=3000, seed=3)          # σ = 0：与 Marcus 完全相同
    assert np.allclose(r_ma["mu_tensor_cm2_Vs"], r["mu_tensor_cm2_Vs"])
    r_dis = run_kmc(g, lam, T, method="miller", sigma=0.06, walkers=600, steps=3000, seed=3)
    assert r_dis["mu_principal"][0] < 0.6 * r["mu_principal"][0]                          # 能量无序降低迁移率
    # 标量公式按 n 维平均，不等于张量主值
    assert marcus_scalar(g, lam, T, 2) < w[0]


def test_regime_and_verdicts():
    g = graph_from_paths(PATHS, "Vh_eV")
    reg = regime(g, 0.15, 300.0, sigma_E=0.05, sigma_J=0.3)
    assert abs(reg["ratio_2J_lam"] - 2 * 0.02 / 0.15) < 1e-9 and abs(reg["sigma_kT"] - 0.05 / (8.617333e-5 * 300)) < 0.01
    res = compare_methods(g, 0.15, 300.0, "hole", methods=("marcus_tensor", "kmc_marcus", "gdm", "kmc_miller", "band"),
                          sigma_E=0.05, sigma_J=0.3, walkers=300, steps=1500)
    by = {r["key"]: r for r in res["rows"]}
    assert by["marcus_tensor"]["valid"] == "适用"                       # 2J/λ = 0.27
    assert by["band"]["valid"] == "不适用"                               # σ_J/|J| = 0.3
    assert by["gdm"]["mu"] < by["marcus_tensor"]["mu"]
    big = graph_from_paths([dict(p, Vh_eV=10 * p["Vh_eV"]) for p in PATHS], "Vh_eV")
    res2 = compare_methods(big, 0.15, 300.0, "hole", methods=("marcus_tensor",))
    assert res2["rows"][0]["valid"] == "不适用"                          # 2J/λ = 2.7
    txt = table_text({"hole": res})
    assert "适用性" in txt and "Marcus" in txt
    out = os.path.join(tempfile.mkdtemp(), "cmp.csv")
    export_csv({"hole": res}, out)
    assert "applicability" in open(out, encoding="utf-8-sig").read()


def test_crystal_and_ceds_inputs():
    """晶体近邻图 (两个独立分子) 上：全部方法都能运行；TLT 可接受逐键 σ (CEDS 的格式)。"""
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    cif = os.path.join(os.path.dirname(__file__), "data", "pentacene_RT.cif")
    c = read_cif(cif)
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    mean = {1: (77.0, 75.0), 2: (-42.0, -75.5), 3: (32.0, -39.4), 7: (29.0, -37.5)}
    std = {1: (22.0, 15.0), 2: (18.0, 14.6), 3: (13.5, 11.0), 7: (13.8, 11.5)}
    res = {g: {"T": 300.0, "hole": {"mean": mean[g][0], "std": std[g][0]},
               "electron": {"mean": mean[g][1], "std": std[g][1]}} for g in mean}
    g = ce.graph(res, "hole")
    sig = {"edges": [res[gid]["hole"]["std"] / 1000 for gid in g["edge_group"]]}
    out = compare_methods(g, 0.092, 300.0, "hole", sigma_E=0.03, sigma_J=sig, walkers=200, steps=1200, n_real=3,
                          lattice=c["lattice"])
    by = {r["key"]: r for r in out["rows"]}
    assert all(np.isfinite(by[k]["mu"]) and by[k]["mu"] > 0 for k in ("marcus_tensor", "kmc_marcus", "kmc_mlj", "tlt",
                                                                        "kmc_marcus_dis", "kmc_miller", "gdm"))
    assert by["tlt"]["valid"] == "适用" and by["marcus_tensor"]["valid"] == "不适用"   # 2J/λ ≈ 1.7
    assert abs(by["kmc_marcus"]["mu"] / by["marcus_tensor"]["mu"] - 1) < 0.25


def test_mlj_from_modes():
    modes = [[{"freq_cm-1": 300.0, "lambda_meV": 20.0}, {"freq_cm-1": 1400.0, "lambda_meV": 40.0},
              {"freq_cm-1": 1600.0, "lambda_meV": 40.0}]]
    hw, flow = mlj_from_modes(modes)
    assert abs(flow - 0.2) < 1e-9 and abs(hw - 1500 * 1.239841984e-4) < 1e-6


if __name__ == "__main__":
    for f in (test_rate_limits, test_kmc_matches_analytic_and_miller, test_regime_and_verdicts,
              test_crystal_and_ceds_inputs, test_mlj_from_modes):
        f()
        print("ok", f.__name__)
