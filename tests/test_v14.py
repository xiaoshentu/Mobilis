"""KMC 精度自检 (布拉维格子 / 多分子晶胞 / 温度 / 弱场 / 能量无序)、MSD 记录、平衡起点。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mobility_tool.kmc import exact_1d_ring_disorder, graph_from_paths, run_kmc  # noqa: E402
from mobility_tool.mobility import KB_EV  # noqa: E402
from mobility_tool.validation import kmc_validation  # noqa: E402


def test_selfcheck():
    rows = kmc_validation(quick=True)
    zs = np.array([r[5] for r in rows])
    assert np.abs(zs).max() < 3.5, rows
    print(f"KMC 自检 OK: {len(rows)} 项，|z| 最大 {np.abs(zs).max():.2f}")


def test_disorder_estimators():
    """有能量无序时：Boltzmann 起点 + 固定时间统计 = 系综精确解。"""
    g = graph_from_paths([dict(vector=[5.0, 0, 0], V=0.05, count=2)], "V")
    N, kt = 8, KB_EV * 300
    r = run_kmc(g, 0.2, 300, sigma=0.06, supercell=N, walkers=4000, steps=3000, seed=21)
    E = r["site_energies"].reshape(N, N, N).transpose(1, 2, 0).ravel()
    Ds, Zs = exact_1d_ring_disorder(E, 5e-8, 0.05, 0.2, 300, N)
    exact = float((Zs * Ds).sum() / Zs.sum()) / kt
    mu = r["mu_tensor_cm2_Vs"][0, 0]
    assert r["estimator"] == "time" and abs(mu - exact) < 3 * r["mu_err"][0, 0]
    t, m = r["msd_t"], r["msd_A2"]
    assert len(t) == 40 and (m > 0).all() and abs(m[-1] * 1e-16 / (2 * t[-1]) / kt - mu) < 1e-9
    print(f"能量无序 OK: 固定时间统计 {mu:.4f} ≈ 系综精确 {exact:.4f}；MSD 曲线完整且与扩散张量一致")


if __name__ == "__main__":
    test_selfcheck()
    test_disorder_estimators()
