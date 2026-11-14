"""自由原子球对称电子密度：径向 Kohn–Sham 方程 (LDA 交换 + VWN 关联) 自洽求解。

对数格点 r = e^x，令 u(r) = e^{x/2} y(x)，径向方程化为 −y'' + [2r²V + (l+½)²] y = 2r²ε y，
二阶差分后成为稀疏广义本征问题，用移位求逆 Lanczos (eigsh, sigma < 谱下限) 求最低的若干态。
电子组态按 Madelung 顺序填充并作球平均 (开壳层分数占据)。结果用于 IGM / IGMH 的自由原子密度与 Hirshfeld 权重。
"""
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

_ORDER = ["1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p", "6s", "4f", "5d", "6p", "7s", "5f", "6d", "7p"]
_CAP = {"s": 2, "p": 6, "d": 10, "f": 14}
_L = {"s": 0, "p": 1, "d": 2, "f": 3}


def configuration(Z):
    occ, left = [], Z
    for o in _ORDER:
        if left <= 0:
            break
        k = min(_CAP[o[1]], left)
        occ.append((int(o[0]), _L[o[1]], float(k)))
        left -= k
    return occ


def _vwn(rs):
    """VWN5 关联势 (顺磁)。"""
    A, x0, b, c = 0.0310907, -0.10498, 3.72744, 12.9352
    x = np.sqrt(rs)
    X = x * x + b * x + c
    X0 = x0 * x0 + b * x0 + c
    Q = np.sqrt(4 * c - b * b)
    ec = A * (np.log(x * x / X) + 2 * b / Q * np.arctan(Q / (2 * x + b))
              - b * x0 / X0 * (np.log((x - x0) ** 2 / X) + 2 * (b + 2 * x0) / Q * np.arctan(Q / (2 * x + b))))
    dec_dx = A * (2 / x - (2 * x + b) / X - 4 * b / (Q * Q + (2 * x + b) ** 2)
                  - b * x0 / X0 * (2 / (x - x0) - (2 * x + b) / X - 4 * (b + 2 * x0) / (Q * Q + (2 * x + b) ** 2)))
    return ec - x / 6 * dec_dx


@lru_cache(maxsize=None)
def solve_atom(Z, n=3000, rmin=1e-6, rmax=60.0, iters=200, mix=0.35, tol=1e-7):
    """返回 (r, ρ(r), 本征值 {(n,l): ε})。"""
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh
    xs = np.linspace(np.log(rmin / Z), np.log(rmax), n)
    h = xs[1] - xs[0]
    r = np.exp(xs)
    occ = configuration(Z)
    w4 = 4 * np.pi * r ** 3 * h                        # ∫ f 4πr² dr = Σ f · 4π r³ h
    rho = np.full(n, 1e-12)
    V = -Z / r * np.exp(-r * Z ** (1 / 3) / 1.5) - 1.0 / (r + 1.0)        # 初始屏蔽库仑势
    eps = {}
    from scipy.linalg import solve_banded
    prev = {}                                          # l → (本征值数组, 本征向量 n×k)
    Bd = 2 * r ** 2
    for it in range(iters):
        rho_new = np.zeros(n)
        for l in sorted({o[1] for o in occ}):
            shells = sorted(o for o in occ if o[1] == l)
            k = len(shells)
            dmain = 2 / h ** 2 + 2 * r ** 2 * V + (l + 0.5) ** 2
            offd = -np.ones(n - 1) / h ** 2
            if l not in prev or it % 10 == 0:
                A = diags([dmain, offd, offd], [0, 1, -1], format="csc")
                B = diags(Bd, 0, format="csc")
                vals, vecs = eigsh(A, k=k, M=B, sigma=-0.6 * Z * Z - 1.0, which="LM")
                o_ = np.argsort(vals)
                vals, vecs = vals[o_], vecs[:, o_]
            else:
                # 逆迭代 (移位 = 上一步本征值)：(A − σB) z = B y，同一 l 的态之间 B 正交化
                vals0, vecs0 = prev[l]
                vals, vecs = np.empty(k), np.empty((n, k))
                for i in range(k):
                    y = vecs0[:, i]
                    sig = vals0[i] - 1e-6
                    ab = np.zeros((3, n))
                    ab[0, 1:] = offd
                    ab[1] = dmain - sig * Bd
                    ab[2, :-1] = offd
                    for _ in range(2):
                        y = solve_banded((1, 1), ab, Bd * y)
                        for j in range(i):
                            y -= (vecs[:, j] * Bd @ y) * vecs[:, j]
                        y /= np.sqrt(y * Bd @ y)
                    Ay = dmain * y
                    Ay[:-1] += offd * y[1:]
                    Ay[1:] += offd * y[:-1]
                    vals[i] = y @ Ay
                    vecs[:, i] = y
            prev[l] = (vals, vecs)
            for kk, (nq, _, f) in enumerate(shells):
                y = vecs[:, kk]
                u2 = r * y ** 2                           # u² = r y²
                norm = (u2 * r * h).sum()                 # ∫ u² dr = Σ u² r h
                rho_new += f * u2 / norm / (4 * np.pi * r ** 2)
                eps[(nq, l)] = float(vals[kk])
        if it:
            rho_new = mix * rho_new + (1 - mix) * rho
        dn = np.abs(rho_new - rho) @ w4
        rho = rho_new
        q = np.cumsum(rho * w4)
        tail = np.cumsum((rho * 4 * np.pi * r * r * h)[::-1])[::-1]           # ∫_r^∞ 4π r' ρ dr'，dr' = r' h
        VH = q / r + tail
        rs = (3 / (4 * np.pi * np.maximum(rho, 1e-30))) ** (1 / 3)
        Vx = -(3 * rho / np.pi) ** (1 / 3)
        Vc = np.where(rho > 1e-20, _vwn(rs), 0.0)
        V = -Z / r + VH + Vx + Vc
        if it > 5 and dn < tol * Z:
            break
    solve_atom.last_iters = it + 1
    return r, rho, eps


def _table(Z):
    """(r, ln ρ)：随软件发布的预计算数据 → 用户缓存 → 现场计算并缓存。"""
    Z = int(Z)
    asset = Path(__file__).with_name("assets") / "atomdens.npz"
    if asset.exists():
        with np.load(asset) as d:
            if f"r{Z}" in d:
                return d[f"r{Z}"], d[f"lnrho{Z}"]
    cache = Path.home() / ".mobilis" / "atomdens" / f"{Z}.npz"
    if cache.exists():
        with np.load(cache) as d:
            return d["r"], d["lnrho"]
    r, rho, _ = solve_atom(Z)
    keep = rho > 1e-30
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, r=r[keep], lnrho=np.log(rho[keep]))
    except OSError:
        pass
    return r[keep], np.log(rho[keep])


@lru_cache(maxsize=None)
def density_spline(Z):
    """ln ρ(r) 的三次样条 (r 为 bohr)；返回可求 ρ、ρ′、ρ″ 的函数。"""
    from scipy.interpolate import CubicSpline
    rr0, lnr = _table(Z)
    sp = CubicSpline(rr0, lnr)
    rmin, rmax = rr0[0], rr0[-1]

    def f(rr):
        rr = np.clip(rr, rmin, rmax)
        L, L1, L2 = sp(rr), sp(rr, 1), sp(rr, 2)
        p = np.exp(L)
        return p, p * L1, p * (L1 * L1 + L2)
    return f
