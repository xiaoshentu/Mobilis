"""KMC 精度自检：与有严格解析解 / 精确解的体系逐项比较。

A–C 布拉维格子 (一维、二维、三维)：μ_ab = Σ k r_a r_b / (2kT)
D   多分子晶胞 (各位点对称等价、跳跃矢量成 ±v 对)：μ_ab = (1/n) Σ_i Σ_j k_ij r_a r_b / (2kT)
E   温度依赖 (Marcus 速率，T = 200 / 300 / 400 K)
F   弱场漂移迁移率 v/F 与零场扩散迁移率 (Einstein 关系)
G   位点能高斯无序：一维环的精确扩散系数 (Derrida, J. Stat. Phys. 1983, 31, 433)，按 Boltzmann 权重做系综平均
z = (KMC − 精确) / 统计误差；|z| < 3 表示在统计误差范围内一致。
"""
import numpy as np

from .kmc import (analytic_bravais, analytic_equivalent_sites, exact_1d_ring_disorder, graph_from_crystal,
                  graph_from_paths, run_kmc)
from .mobility import KB_EV

IRBB_PATHS = [dict(vector=[8.709, 0, 0], V=0.07873, count=2), dict(vector=[0.4, 10.13, 0], V=0.01381, count=2),
              dict(vector=[9.109, 10.13, 0], V=0.02277, count=2)]


def kmc_validation(quick=False, crystal=None, seed=11, on_progress=None):
    """返回 [(名称, 精确值, KMC, 误差, 相对偏差 %, z)]。crystal=(crys, mols, couplings) 时加入晶体图检验。"""
    w, s = (1500, 1500) if quick else (4000, 3000)
    rows = []

    def add(name, kmc, err, exact):
        rows.append((name, float(exact), float(kmc), float(err), 100 * (kmc - exact) / exact,
                     (kmc - exact) / err if err > 0 else 0.0))
        if on_progress:
            on_progress(len(rows), name)
    lam, T = 0.20, 300.0
    g1 = graph_from_paths([dict(vector=[5.0, 0, 0], V=0.05, count=2)], "V")
    r = run_kmc(g1, lam, T, walkers=w, steps=s, seed=seed)
    add("A 一维链 μxx", r["mu_tensor_cm2_Vs"][0, 0], r["mu_err"][0, 0], analytic_bravais(g1, lam, T)[0, 0])
    g2 = graph_from_paths(IRBB_PATHS, "V")
    r = run_kmc(g2, 0.130, T, walkers=w, steps=s, seed=seed + 1)
    ex = analytic_bravais(g2, 0.130, T)
    for a, b, n in ((0, 0, "xx"), (1, 1, "yy"), (0, 1, "xy")):
        add(f"B 二维 (Ir-BB 电子) μ{n}", r["mu_tensor_cm2_Vs"][a, b], r["mu_err"][a, b], ex[a, b])
    g3 = graph_from_paths(IRBB_PATHS + [dict(vector=[1.0, 2.0, 14.0], V=0.004, count=2)], "V")
    r = run_kmc(g3, 0.130, T, walkers=w, steps=s, seed=seed + 2)
    add("C 三维 μzz (弱耦合方向)", r["mu_tensor_cm2_Vs"][2, 2], r["mu_err"][2, 2], analytic_bravais(g3, 0.130, T)[2, 2])
    if crystal is not None:
        crys, mols, coup, cut = crystal
        gR = graph_from_crystal(crys, mols, coup, cut, carrier="hole")
        r = run_kmc(gR, 0.15, T, walkers=w, steps=s, seed=seed + 3)
        ex = np.sort(np.linalg.eigvalsh(analytic_equivalent_sites(gR, 0.15, T)))[::-1]
        km = np.sort(np.linalg.eigvalsh(r["mu_tensor_cm2_Vs"]))[::-1]
        add("D 晶体图 (多分子晶胞) μ 主值", km[0], r["mu_avg_err"] * np.sqrt(3), ex[0])
    for TT in (200.0, 400.0):
        r = run_kmc(g1, lam, TT, walkers=w, steps=s, seed=seed + int(TT))
        add(f"E 一维链 T = {TT:.0f} K", r["mu_tensor_cm2_Vs"][0, 0], r["mu_err"][0, 0], analytic_bravais(g1, lam, TT)[0, 0])
    r = run_kmc(g1, lam, T, field=(1e4, 0, 0), walkers=w, steps=s, seed=seed + 5)
    add("F 漂移迁移率 (10⁴ V/cm) / Einstein", r["mu_field_cm2_Vs"], r["mu_field_err"], analytic_bravais(g1, lam, T)[0, 0])
    N, kt = 8, KB_EV * T
    for sig in ((0.06,) if quick else (0.03, 0.06, 0.09)):
        r = run_kmc(g1, lam, T, sigma=sig, supercell=N, walkers=w + 2000, steps=s, seed=seed + 6)
        E = r["site_energies"].reshape(N, N, N).transpose(1, 2, 0).ravel()
        Ds, Zs = exact_1d_ring_disorder(E, 5e-8, 0.05, lam, T, N)
        add(f"G 能量无序 σ = {sig * 1000:.0f} meV", r["mu_tensor_cm2_Vs"][0, 0], r["mu_err"][0, 0],
            float((Zs * Ds).sum() / Zs.sum()) / kt)
    return rows


def format_rows(rows):
    out = [f"{'检验项目':<28}{'精确':>11}{'KMC':>11}{'误差':>9}{'偏差%':>8}{'z':>7}"]
    for n, ex, km, er, dev, z in rows:
        out.append(f"{n:<28}{ex:11.4g}{km:11.4g}{er:9.2g}{dev:+8.2f}{z:+7.2f}")
    zs = np.array([r[5] for r in rows])
    out.append(f"共 {len(rows)} 项：|z| 最大 {np.abs(zs).max():.2f}，平均 |偏差| {np.mean([abs(r[4]) for r in rows]):.2f}%；"
               + ("全部在 3 倍统计误差内" if np.abs(zs).max() < 3 else "存在超出 3 倍统计误差的项"))
    return "\n".join(out)


def figure(rows, path=None, ax=None):
    """KMC 与精确值对照图 (对数坐标 + 误差棒)。"""
    from matplotlib.figure import Figure
    own = ax is None
    if own:
        fig = Figure(figsize=(5.0, 4.6), dpi=300)
        ax = fig.add_subplot(111)
    ex = np.array([r[1] for r in rows])
    km = np.array([r[2] for r in rows])
    er = np.array([r[3] for r in rows])
    ax.errorbar(ex, km, yerr=er, fmt="o", ms=5, color="#2B4A77", ecolor="#7F8C9F", capsize=3)
    lo, hi = min(ex.min(), km.min()) * 0.5, max(ex.max(), km.max()) * 2
    ax.plot([lo, hi], [lo, hi], "--", color="#C0392B", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Exact μ (cm$^2$ V$^{-1}$ s$^{-1}$)")
    ax.set_ylabel("KMC μ (cm$^2$ V$^{-1}$ s$^{-1}$)")
    if own:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=300)
        return fig
    return ax


def write_csv(rows, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["test", "exact_cm2Vs", "kmc_cm2Vs", "stat_err", "deviation_percent", "z"])
        w.writerows(rows)
