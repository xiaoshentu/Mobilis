"""v1.20 回归测试：Δ-学习耦合曲面 (EHT 基线 + GP) 与迁移率的不确定度传递。
1. 通用克里金：没有基线时与原来的常数均值 GP 完全相同；有基线时 β 自动吸收基线的缩放与符号；
2. “DFT” = 1.8·V_EHT + 平滑修正 (模拟真实情况：EHT 低估约一半)：同样 12 个训练点，Δ-GP 的误差远小于普通 GP；
3. 后验区间包含 β 的不确定度 (远离训练点时方差不为零)；
4. mu_posterior：曲面不确定度传递到 TLT 迁移率，区间包含后验均值；
5. 旧的主动学习任务 (没有 delta 字段) 仍按常数均值 GP 处理。
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mobility_tool.alsurface import EHTBaseline, GPSurface, _GP, records_from_rows  # noqa: E402
from mobility_tool.gaussian_io import read_geometry  # noqa: E402
from mobility_tool.scanflow import PARAM_KEYS, ScanProject  # noqa: E402

REF = os.path.join(ROOT, "tests", "data", "pentacene_ref.xyz")
CENTER = {"z": 3.6, "dx": 0.8, "dy": 0.4, "gamma": 0.0, "pitch": 0.0, "roll": 0.0}
HALF = {"z": 0.3, "dx": 0.6, "dy": 0.6, "gamma": 6.0, "pitch": 6.0, "roll": 6.0}


def _setup():
    d = tempfile.mkdtemp()
    pr = ScanProject.create(os.path.join(d, "p"), REF, name="p")
    base = EHTBaseline(pr.symbols, pr.Y)
    rng = np.random.default_rng(4)
    c0 = np.array([CENTER[k] for k in PARAM_KEYS])
    h = np.array([HALF[k] for k in PARAM_KEYS])
    Q = c0 + h * rng.uniform(-1, 1, (70, 6))
    B = base.values(Q)

    def dft(Q, Bq):                      # “DFT”：EHT 的 1.8 倍 + 平滑的几何修正 + 常数
        u = (Q - c0) / h
        corr = 6.0 * np.sin(1.3 * u[:, 1]) + 4.0 * u[:, 0] ** 2 - 3.0 * u[:, 2] * u[:, 3]
        return 1.8 * Bq[:, 0] + corr + 5.0, 1.6 * Bq[:, 1] - 0.7 * corr - 4.0
    Vh, Ve = dft(Q, B)
    return base, Q, Vh, Ve


def test_kriging_reduces_to_constant_mean():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (15, 2))
    y = np.sin(2 * X[:, 0]) + X[:, 1]
    g0 = _GP(X, y, restarts=1)
    m0, s0 = g0.predict(X[:3])
    assert g0.H is None and np.all(np.isfinite(m0)) and np.all(s0 >= 0)
    g1 = _GP(X, y, restarts=1, H=np.c_[np.ones(15), 3.0 * y])     # 基线就是答案 (×3)：β₁ → 1/3
    assert abs(g1.beta[1] - 1 / 3) < 0.02, g1.beta
    m1, s1 = g1.predict(X[:3], Hs=np.c_[np.ones(3), 3.0 * y[:3]])
    assert np.allclose(m1, y[:3], atol=0.05)
    far = np.array([[5.0, 5.0]])
    _, sf = g1.predict(far, Hs=np.c_[np.ones(1), [40.0]])          # 远处：方差 = 信号方差 + β 不确定度
    assert sf[0] > s1.max()
    print(f"通用克里金：β₁ = {g1.beta[1]:.3f} (应为 0.333)；远离数据时方差变大")


def test_delta_beats_plain_gp():
    base, Q, Vh, Ve = _setup()
    recs = records_from_rows(Q, Vh, Ve)
    train, test = recs[:12], np.arange(12, 70)
    gp = GPSurface(train, CENTER, half=HALF, reverse=False, restarts=2)
    dg = GPSurface(train, CENTER, half=HALF, reverse=False, restarts=2, baseline=base)
    out = {}
    for c, V in (("hole", Vh), ("electron", Ve)):
        e0 = np.sqrt(((gp(Q[test], c) - V[test]) ** 2).mean())
        e1 = np.sqrt(((dg(Q[test], c) - V[test]) ** 2).mean())
        out[c] = (e0, e1)
        assert e1 < 0.5 * e0, (c, e0, e1)
        b = dg.gp[c].beta
        assert 1.2 < abs(b[1]) < 2.4, b                      # 自动找到约 1.8 / 1.6 倍的缩放
    u = dg.stat_uncertainty(Q[test], "hole", n_draws=100)
    assert u["std_lo"] <= u["std"] * 1.05 and u["std_hi"] >= u["std"] * 0.95
    assert "Δ-GP" in dg.describe()
    chk = base.check(recs)
    assert chk["hole"]["r"] > 0.8
    print("12 个训练点 RMSE (普通 GP → Δ-GP)：" + "；".join(f"{c} {a:.1f} → {b:.1f} meV" for c, (a, b) in out.items()))


def test_mu_posterior_and_old_runs():
    from mobility_tool.alsurface import ALRun, mu_posterior
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    c = read_cif(os.path.join(ROOT, "tests", "data", "pentacene_RT.cif"))
    at = read_geometry(REF)
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    base = EHTBaseline([a[0] for a in at], np.array([a[1:] for a in at]))
    rng = np.random.default_rng(1)
    surf, thermal = {}, {}
    for p in ce.pairs[:2]:
        c0 = np.array([p["center"][k] for k in PARAM_KEYS])
        h = np.array([0.1, 0.15, 0.15, 2.0, 2.0, 3.0])
        Q = c0 + h * rng.standard_normal((16, 6))
        B = base.values(Q)
        surf[p["gid"]] = GPSurface(records_from_rows(Q, 1.8 * B[:, 0], 1.6 * B[:, 1]), p["center"], reverse=False,
                                   restarts=1, baseline=base)
        thermal[p["gid"]] = c0 + h * rng.standard_normal((20, 6))
    used = {p["gid"] for p in ce.pairs[:2]}
    for p in ce.pairs:                     # 其余近邻对：把同组的曲面借给它们，使 TLT 网络完整
        if p["gid"] not in used:
            surf[p["gid"]] = surf[ce.pairs[0]["gid"]]
            thermal[p["gid"]] = thermal[ce.pairs[0]["gid"]]
    r = mu_posterior(ce, surf, thermal, 300.0, n_draws=3, n_real=2)
    for car, v in r.items():
        assert v["lo"] <= v["hi"] and len(v["draws"]) == 3 and v["mu"] > 0, v
    # 旧任务 (state 中没有 delta)：按常数均值 GP
    d = tempfile.mkdtemp()
    pr = ScanProject.create(os.path.join(d, "p"), REF, name="p")
    run = ALRun(pr, "old")
    run.state = {"mirror": {}}
    assert run.delta is False and run.baseline(1) is None
    run.state["delta"] = True
    assert isinstance(run.baseline(1), EHTBaseline)
    print("曲面不确定度 → TLT μ：" + "；".join(f"{c} {v['mu']:.3g} ({v['lo']:.3g}–{v['hi']:.3g})" for c, v in r.items())
          + "；旧任务仍用常数均值 GP")


if __name__ == "__main__":
    test_kriging_reduces_to_constant_mean()
    test_delta_beats_plain_gp()
    test_mu_posterior_and_old_runs()
