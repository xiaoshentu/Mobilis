"""弱相互作用与电子密度拓扑 (独立实现)：

IGM  (Lefebvre et al., Phys. Chem. Chem. Phys. 2017, 19, 17928)：前分子 (自由原子) 密度，无需波函数，秒级
IGMH (Lu & Chen, J. Comput. Chem. 2022, 43, 539；勘误 2022)：分子实际密度按 Hirshfeld 划分到片段
     ∇ρ_F = w_F ∇ρ + s·ρ ∇w_F，w_F = Σ_{A∈F} ρ_A⁰ / Σ_A ρ_A⁰；s = −1 为原文实现 (作图效果更好，见勘误)，s = +1 为数学严格形式
     δg_inter = Σ_F |∇ρ_F| − |∇ρ|；等值面按 sign(λ2)ρ 着色 (蓝：强吸引；绿：范德华 / π–π；红：位阻排斥)
     自由原子密度 ρ_A⁰：本程序自洽求解的球对称 LDA 原子密度 (atomdens.py)；Multiwfn 使用其内置的数值原子密度，
     两者都来自原子计算，数值上可能略有差别
AIM  (Bader, Atoms in Molecules, 1990)：牛顿法搜索临界点，键径追踪，键临界点处 ρ、∇²ρ、椭圆度、G、V、H，
     Espinosa 估计 E ≈ V/2 (Espinosa et al., Chem. Phys. Lett. 1998, 285, 170)，Poincaré–Hopf 检查 n − b + r − c = 1
长度单位 bohr，密度单位 a.u.
"""
import math

import numpy as np

from .gaussian_io import BOHR2ANG

BOHR = 1 / BOHR2ANG
_ORDER = ["1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p", "6s", "4f", "5d", "6p", "7s", "5f", "6d", "7p"]
_CAP = {"s": 2, "p": 6, "d": 10, "f": 14}
_NSTAR = {1: 1.0, 2: 2.0, 3: 3.0, 4: 3.7, 5: 4.0, 6: 4.2, 7: 4.3}


def _group(orb):
    n, l = int(orb[0]), orb[1]
    return (n, "sp") if l in "sp" else (n, l)


def slater_shells(Z):
    """Slater 规则：返回 [(n*, ζ, 电子数)] (每个 Slater 组一项)。"""
    occ, left = {}, Z
    for o in _ORDER:
        if left <= 0:
            break
        k = min(_CAP[o[1]], left)
        occ[o] = k
        left -= k
    groups = {}
    for o, k in occ.items():
        groups[_group(o)] = groups.get(_group(o), 0) + k
    order = sorted(groups, key=lambda g: (g[0], {"sp": 0, "d": 1, "f": 2}[g[1]]))
    out = []
    for g in order:
        n, typ = g
        ne = groups[g]
        s = (ne - 1) * (0.30 if (n == 1) else 0.35)
        for h in order:
            if h == g:
                break
            nh = groups[h]
            if typ == "sp":
                if h[0] == n - 1:
                    s += 0.85 * nh
                elif h[0] <= n - 2:
                    s += 1.00 * nh
                elif h[0] == n:                 # 同主量子数但排在前面的组 (不出现于 sp)
                    s += 0.35 * nh
            else:
                s += 1.00 * nh
        ns = _NSTAR[min(n, 7)]
        out.append((ns, max(Z - s, 0.1) / ns, ne))
    return out


class Promolecule:
    """自由原子密度之和；可按原子 / 片段给出密度、梯度与 Hessian。
    density="lda" (默认)：球对称 LDA 自洽原子密度 (atomdens.py)，远程衰减正确；"slater"：Slater 规则解析近似 (仅作对照)。"""

    def __init__(self, Z, coords_bohr, density="lda"):
        self.Z = np.asarray(Z, int)
        self.R = np.asarray(coords_bohr, float)
        self.kind = density
        if density == "lda":
            from .atomdens import density_spline
            self.spl = {int(z): density_spline(int(z)) for z in set(self.Z.tolist())}
        else:
            self.par = {int(z): slater_shells(int(z)) for z in set(self.Z.tolist())}

    def _radial(self, z, r):
        """ρ(r)、ρ'(r)、ρ''(r)。"""
        if self.kind == "lda":
            return self.spl[z](r)
        rho = np.zeros_like(r)
        d1 = np.zeros_like(r)
        d2 = np.zeros_like(r)
        for ns, zeta, ne in self.par[z]:
            k = 2 * ns - 2
            N2 = (2 * zeta) ** (2 * ns + 1) / math.gamma(2 * ns + 1)
            f = ne * N2 / (4 * np.pi) * r ** k * np.exp(-2 * zeta * r)
            q = k / r - 2 * zeta
            rho += f
            d1 += q * f
            d2 += (q * q - k / r ** 2) * f
        return rho, d1, d2

    def atom_terms(self, pts, order=2):
        """逐原子：ρ_A (nat, n)，∇ρ_A (nat, 3, n)，Hessian (nat, 6, n)。"""
        nat, n = len(self.Z), len(pts)
        rho = np.zeros((nat, n))
        g = np.zeros((nat, 3, n)) if order >= 1 else None
        h = np.zeros((nat, 6, n)) if order >= 2 else None
        for a in range(nat):
            d = pts - self.R[a]
            r = np.maximum(np.sqrt((d ** 2).sum(1)), 1e-8)
            p, p1, p2 = self._radial(int(self.Z[a]), r)
            rho[a] = p
            if order >= 1:
                u = d / r[:, None]
                g[a] = (p1[:, None] * u).T
            if order >= 2:
                for k, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
                    h[a, k] = (p2 - p1 / r) * u[:, i] * u[:, j] + (p1 / r if i == j else 0.0)
        return rho, g, h

    def __call__(self, pts, order=2, kinetic=False):
        rho, g, h = self.atom_terms(pts, order)
        return rho.sum(0), (g.sum(0) if g is not None else None), (h.sum(0) if h is not None else None), None


def wfn_provider(wfn):
    from .wavefunction import density_derivs

    def f(pts, order=2, kinetic=False):
        return density_derivs(wfn, pts, order, kinetic=kinetic)
    return f


def _sym6(h):
    """(6, n) → (n, 3, 3)。"""
    H = np.zeros((h.shape[1], 3, 3))
    for k, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        H[:, i, j] = h[k]
        H[:, j, i] = h[k]
    return H


# ----------------------------------------------------------------------------------------------
# IGM / IGMH
# ----------------------------------------------------------------------------------------------
def fragments_by_connectivity(symbols, coords_A):
    from .viewer import bonds_for
    n = len(symbols)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in bonds_for(symbols, np.asarray(coords_A)):
        parent[find(i)] = find(j)
    roots = {}
    return np.array([roots.setdefault(find(i), len(roots)) for i in range(n)])


def igm_analysis(Z, coords_bohr, frag, provider=None, spacing_A=0.2, margin_A=3.0, region_A=4.0, iso=None,
                 sign=-1, block=4000, on_progress=None):
    """δg_inter 与 sign(λ2)ρ 的格点。provider=None 为 IGM (前分子密度)，否则为 IGMH (provider 为分子密度函数)。
    frag: 每个原子的片段编号 (≥ 2 个片段)。返回格点数据、等值面所需的 cube、δG_inter 积分与散点图数据。"""
    from scipy.ndimage import binary_dilation
    from scipy.spatial import cKDTree
    frag = np.asarray(frag)
    labs = sorted(set(frag.tolist()))
    if len(labs) < 2:
        raise ValueError("至少需要两个片段 (例如二聚体的两个分子)")
    pro = Promolecule(Z, coords_bohr)
    R = pro.R
    h = spacing_A * BOHR
    lo = R.min(0) - margin_A * BOHR
    hi = R.max(0) + margin_A * BOHR
    shape = tuple(np.ceil((hi - lo) / h).astype(int) + 1)
    ii = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    pts = lo + ii * h
    near = np.ones(len(pts), bool)                       # 只算各片段都在 region_A 以内的区域
    for f in labs:
        d, _ = cKDTree(R[frag == f]).query(pts, distance_upper_bound=region_A * BOHR)
        near &= np.isfinite(d)
    idx = np.where(near)[0]
    dg = np.zeros(len(pts))
    rho_g = np.zeros(len(pts))
    for s in range(0, len(idx), block):
        p = pts[idx[s:s + block]]
        ra, ga, _ = pro.atom_terms(p, 1)
        gF = np.stack([ga[frag == f].sum(0) for f in labs])          # 前分子片段梯度 (nF, 3, n)
        if provider is None:
            grad = gF.sum(0)
            rho = ra.sum(0)
        else:
            rho, grad, _, _ = provider(p, 1)
            r0 = np.maximum(ra.sum(0), 1e-300)
            g0 = ga.sum(0)
            gH = []
            for k, f in enumerate(labs):
                rF = ra[frag == f].sum(0)
                w = rF / r0
                dw = (gF[k] * r0 - rF * g0) / r0 ** 2
                gH.append(w * grad + sign * rho * dw)
            gF = np.stack(gH)
        dg[idx[s:s + block]] = np.linalg.norm(gF, axis=1).sum(0) - np.linalg.norm(grad, axis=0)
        rho_g[idx[s:s + block]] = rho
        if on_progress:
            on_progress(min(s + block, len(idx)), len(idx))
    dV = h ** 3
    dG = float(dg.sum() * dV)
    iso = iso if iso is not None else 0.005
    # sign(λ2)ρ：只在等值面附近计算 Hessian
    sl2 = np.zeros(len(pts))
    m = (dg > 0.3 * iso).reshape(shape)
    m = binary_dilation(m, iterations=1).ravel() & near
    sel = np.where(m)[0]
    for s in range(0, len(sel), block):
        p = pts[sel[s:s + block]]
        if provider is None:
            rho, _, hh, _ = pro(p, 2)
        else:
            rho, _, hh, _ = provider(p, 2)
        lam = np.linalg.eigvalsh(_sym6(hh))
        sl2[sel[s:s + block]] = np.sign(lam[:, 1]) * rho
    scat = m & (dg > 1e-4) & (rho_g < 0.1)                 # 只取算过 Hessian 的点
    cube = {"origin": lo, "axes": np.eye(3) * h, "shape": shape, "Z": np.asarray(Z), "coords": R}
    return {"dg_inter": dict(cube, data=dg.reshape(shape), title="dg_inter"),
            "sl2r": dict(cube, data=sl2.reshape(shape), title="sign(lambda2)rho"),
            "iso": iso, "dG_inter": dG, "method": "IGMH" if provider is not None else "IGM (promolecular)",
            "scatter": (sl2[scat], dg[scat]), "n_points": int(len(idx)), "dg_max": float(dg.max()),
            "sign": sign}


def igm_surface(res, iso=None, vmin=-0.05, vmax=0.05):
    """δg_inter 等值面，按 sign(λ2)ρ 着色 (蓝–绿–红)。"""
    from scipy.ndimage import map_coordinates

    from .wavefunction import isosurface
    iso = iso or res["iso"]
    v, f = isosurface(res["dg_inter"], iso)
    if not len(f):
        return v, f, np.zeros((0, 3)), np.zeros(0)
    c = res["sl2r"]
    frac = np.linalg.solve(np.asarray(c["axes"]).T, (v * BOHR - c["origin"]).T)
    val = map_coordinates(c["data"], frac, order=1, mode="nearest")
    fval = val[f].mean(1)
    return v, f, bgr_colors(fval, vmin, vmax), val


def bgr_colors(x, vmin=-0.05, vmax=0.05):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bgr", ["#0000FF", "#00FF00", "#FF0000"])
    return cmap(np.clip((np.asarray(x) - vmin) / (vmax - vmin), 0, 1))[:, :3]


# ----------------------------------------------------------------------------------------------
# AIM 拓扑
# ----------------------------------------------------------------------------------------------
def _newton(provider, x0, iters=80, tol=1e-7, maxstep=0.3):
    x = np.array(x0, float)
    alive = np.ones(len(x), bool)
    for _ in range(iters):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        _, g, h, _ = provider(x[idx], 2)
        H = _sym6(h)
        gn = np.linalg.norm(g, axis=0)
        done = gn < tol
        alive[idx[done]] = False
        act = idx[~done]
        if not len(act):
            break
        try:
            step = -np.linalg.solve(H[~done], g[:, ~done].T[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = -g[:, ~done].T * 0.1
        L = np.linalg.norm(step, axis=1)
        step *= np.minimum(1.0, maxstep / np.maximum(L, 1e-12))[:, None]
        x[act] += step
    _, g, _, _ = provider(x, 1)
    return x, np.linalg.norm(g, axis=0) < 1e-5


def _rings(symbols, R_A):
    """键图中的最小环 (长度 ≤ 8)。"""
    from .viewer import bonds_for
    n = len(symbols)
    adj = [[] for _ in range(n)]
    for i, j in bonds_for(symbols, R_A):
        adj[i].append(j)
        adj[j].append(i)
    rings = set()
    for i in range(n):
        for j in adj[i]:
            if j < i:
                continue
            # 删去 i–j 边后找最短路径
            prev = {i: None}
            q = [i]
            while q and j not in prev:
                nq = []
                for u in q:
                    for v in adj[u]:
                        if (u, v) in ((i, j), (j, i)) or v in prev:
                            continue
                        prev[v] = u
                        nq.append(v)
                q = nq
            if j in prev:
                path, u = [], j
                while u is not None:
                    path.append(u)
                    u = prev[u]
                if len(path) <= 8:
                    rings.add(tuple(sorted(path)))
    return [list(r) for r in rings]


def aim_analysis(Z, coords_bohr, provider, frag=None, inter_A=4.5, kinetic=True):
    """返回临界点列表、键径与 Poincaré–Hopf 检查。provider(pts, order, kinetic) → (ρ, ∇ρ, Hess6, G)。"""
    from .gaussian_io import ELEMENTS
    R = np.asarray(coords_bohr, float)
    sym = [ELEMENTS[int(z)] for z in Z]
    RA = R * BOHR2ANG
    from .viewer import bonds_for
    seeds = [(R[i] + R[j]) / 2 for i, j in bonds_for(sym, RA)]
    if frag is not None:
        frag = np.asarray(frag)
        d = np.linalg.norm(RA[:, None] - RA[None], axis=-1)
        for i in range(len(R)):
            for j in range(i + 1, len(R)):
                if frag[i] != frag[j] and d[i, j] < inter_A:
                    seeds.append((R[i] + R[j]) / 2)
    seeds = np.array(seeds)
    bcp_like, ok = _newton(provider, seeds)
    pts = list(bcp_like[ok])
    # 环 / 笼临界点的初始点：键图中的环心，以及相邻临界点对的中点
    ring_seeds = [R[r].mean(0) for r in _rings(sym, RA)]
    cps0 = np.array(pts) if pts else np.zeros((0, 3))
    if len(cps0):
        dd = np.linalg.norm(cps0[:, None] - cps0[None], axis=-1)
        a, b = np.where(np.triu((dd < 5.0) & (dd > 0.2), 1))
        if len(a) > 4000:
            sel = np.random.default_rng(0).choice(len(a), 4000, replace=False)
            a, b = a[sel], b[sel]
        ring_seeds += list((cps0[a] + cps0[b]) / 2)
    if ring_seeds:
        x2, ok2 = _newton(provider, np.array(ring_seeds))
        pts += list(x2[ok2])
    # 去重、分类
    uniq = []
    for p in pts:
        if np.min(np.linalg.norm(R - p, axis=1)) < 0.1:
            continue                                     # 原子核 (吸引子) 单独计
        if all(np.linalg.norm(p - q) > 0.05 for q in uniq):
            uniq.append(p)
    cps = []
    if uniq:
        U = np.array(uniq)
        rho, g, h, G = provider(U, 2, kinetic=kinetic)
        lam, vec = np.linalg.eigh(_sym6(h))
        for k in range(len(U)):
            if rho[k] < 1e-5:
                continue
            npos = int((lam[k] > 0).sum())
            typ = {1: "BCP", 2: "RCP", 3: "CCP", 0: "NCP"}[npos]
            lap = float(lam[k].sum())
            cp = {"type": typ, "xyz_bohr": U[k], "rho": float(rho[k]), "lap": lap, "lam": lam[k].tolist(),
                  "vec": vec[k][:, 2]}
            if typ == "BCP":
                cp["ellipticity"] = float(lam[k][0] / lam[k][1] - 1) if lam[k][1] != 0 else 0.0
            if G is not None:
                Gk = float(G[k])
                V = lap / 4 - 2 * Gk
                cp.update(G=Gk, V=V, H=Gk + V, E_kcal=0.5 * V * 627.5095)
            cps.append(cp)
    # 键径：从 BCP 沿 ±λ3 本征向量方向追踪 ∇ρ 到原子核
    paths = []
    bcps = [c for c in cps if c["type"] == "BCP"]
    if bcps:
        starts = np.array([c["xyz_bohr"] + s * 0.02 * c["vec"] for c in bcps for s in (1, -1)])
        x = starts.copy()
        trail = [x.copy()]
        alive = np.ones(len(x), bool)
        end = -np.ones(len(x), int)
        for _ in range(800):
            if not alive.any():
                break
            idx = np.where(alive)[0]
            _, g, _, _ = provider(x[idx], 1)
            u = (g / np.maximum(np.linalg.norm(g, axis=0), 1e-14)).T
            mid = x[idx] + 0.5 * 0.04 * u
            _, g2, _, _ = provider(mid, 1)
            u2 = (g2 / np.maximum(np.linalg.norm(g2, axis=0), 1e-14)).T
            x[idx] += 0.04 * u2
            dn = np.linalg.norm(x[idx][:, None] - R[None], axis=-1)
            hit = dn.min(1) < 0.12
            end[idx[hit]] = dn[hit].argmin(1)
            alive[idx[hit]] = False
            trail.append(x.copy())
        T = np.array(trail)
        for k, c in enumerate(bcps):
            a, b = end[2 * k], end[2 * k + 1]
            c["atoms"] = (int(a), int(b)) if a >= 0 and b >= 0 else None
            line = np.vstack([T[::-1, 2 * k], c["xyz_bohr"][None], T[:, 2 * k + 1]]) * BOHR2ANG
            paths.append(line)
    n = len(R)
    cnt = {t: sum(1 for c in cps if c["type"] == t) for t in ("BCP", "RCP", "CCP")}
    ph = n - cnt["BCP"] + cnt["RCP"] - cnt["CCP"]
    return {"cps": cps, "paths": paths, "counts": dict(cnt, NCP=n), "poincare_hopf": ph, "ph_ok": ph == 1,
            "symbols": sym}
