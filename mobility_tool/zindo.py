"""ZINDO/S (INDO/S) 与分子轨道重叠 (MOO) 转移积分 —— VOTCA-CTP 的 izindo 方法。

INDO/S 单体自洽场 (闭壳层)
  价层 Slater 轨道 (H: 1s；B–F: 2s2p；Si–Cl: 3s3p；Br: 4s4p)，零微分重叠 (基组按正交处理)。
  单中心：(ss|ss) = (ss|pp) = F⁰，(pp|pp) = F⁰ + 4F²，(pp|p'p') = F⁰ − 2F²，(sp|sp) = G¹，(pp'|pp') = 3F²
  (参数表中为 Slater–Condon 积分 G¹、F²，计算中按 INDO 惯例取 G¹/3、F²/25)。
  双中心库仑：Mataga–Nishimoto，γ_AB = f_γ / (R + 2f_γ / (F⁰_A + F⁰_B))，f_γ = 1.2 (原子单位)。
  核心积分 U：由价态电离势 I_s、I_p 按 sp 组态构造 (Zerner)。
  共振积分：β_μν = ½(β_A + β_B) S̃_μν；S̃ 为 Slater 轨道重叠，pσ–pσ 乘 1.267、pπ–pπ 乘 0.585，s 相关部分不缩放。
  F_μμ = U_μμ − Σ_B Z_B γ_AB + Σ_B P_B γ_AB + Σ_ν∈A P_νν[(μμ|νν) − ½(μν|μν)]
  F_μν (同一原子) = P_μν[3/2 (μν|μν) − ½ (μμ|νν)]；F_μν (不同原子) = β_μν − ½ P_μν γ_AB
MOO 转移积分 (Kirkpatrick, Int. J. Quantum Chem. 2008, 108, 51)
  单体轨道只算一次，随分子刚性转动、平移；二聚体的分子间 Fock 块只取共振积分 (单体密度冻结，分子间密度为零)：
  J = c_Aᵀ F_AB c_B，F_AB,μν = ½(β_A + β_B) S̃_μν。不做重叠校正 (零微分重叠基组正交)。
二聚体 INDO/S 投影 (检验 MOO)
  对二聚体做完整自洽场，J = c_Aᵀ F_dimer c_B (单体轨道补零)。

参数取自 MOPAC (Apache-2.0) 中 Zerner 的 INDO/S 参数；H、C、N、O、F 的 β 与 Slater 指数与 VOTCA-CTP 的 MOO 一致。
"""
from pathlib import Path

import numpy as np

AU2EV = 27.2114
AU2ANG = 0.529177
F_SIG, F_PI = 1.267, 0.585
F_GAMMA = 1.2

# 元素: (主量子数, 价层轨道数, 核芯电荷, ζ, β, I_s, I_p, F0, G1, F2)
PARAMS = {
    "H": (1, 1, 1, 1.200, -12.0, 13.06, 0.0, 12.85, 0.0, 0.0),
    "B": (2, 4, 3, 1.300, -8.0, 14.00, 8.24, 8.68, 5.401481, 3.480847),
    "C": (2, 4, 4, 1.625, -17.0, 19.42, 10.70, 11.11, 6.897842, 4.509913),
    "N": (2, 4, 5, 1.950, -26.0, 25.58, 13.25, 12.01, 8.958454, 6.459559),
    "O": (2, 4, 6, 2.275, -34.0, 32.49, 15.88, 13.00, 11.815414, 6.902802),
    "F": (2, 4, 7, 2.600, -44.0, 40.14, 18.61, 14.00, 14.484415, 8.593198),
    "Si": (3, 4, 4, 1.520, -9.0, 14.79, 8.10, 7.57, 4.812310, 2.262706),
    "P": (3, 4, 5, 1.730, -15.0, 18.23, 9.98, 8.86, 1.047788, 2.947716),
    "S": (3, 4, 6, 1.925, -15.0, 21.73, 11.92, 10.09, 3.075668, 4.537809),
    "Cl": (3, 4, 7, 2.130, -11.0, 25.29, 13.93, 11.25, 8.802854, 6.447161),
    "Br": (4, 4, 7, 2.638, -8.0, 23.94, 12.44, 9.08, 6.141120, 4.608700),
}


def _p(e):
    if e not in PARAMS:
        raise ValueError(f"INDO/S 参数中没有元素 {e} (现有：{', '.join(PARAMS)})")
    return PARAMS[e]


# ----------------------------------------------------------------------------------------------
# Slater 轨道重叠 (长椭球坐标数值积分，对多项式部分精确)
# ----------------------------------------------------------------------------------------------
_U, _WU = np.polynomial.laguerre.laggauss(32)
_E, _WE = np.polynomial.legendre.leggauss(48)
_FACT = [1.0]
for _k in range(1, 30):
    _FACT.append(_FACT[-1] * _k)


def _norm(n, z):
    return (2 * z) ** n * np.sqrt(2 * z / _FACT[2 * n])


def sto_overlap(n1, l1, z1, n2, l2, z2, R):
    """两个 Slater 轨道沿连线 (z 轴由 A 指向 B) 的重叠分量。R: 原子间距 / bohr (数组)。
    返回 (σ, π)：l1 = l2 = 1 时 π 为 pπ–pπ，其余 π = 0。轨道均取 +z 方向。"""
    R = np.atleast_1d(np.asarray(R, float))
    p = 0.5 * (z1 + z2) * R
    t = (z1 - z2) / (z1 + z2)
    xi = 1.0 + _U[None, :, None] / p[:, None, None]                 # (nR, nu, 1)
    eta = _E[None, None, :]
    w = (_WU[:, None] * _WE[None, :])[None]
    ra = xi + eta
    rb = xi - eta
    base = ra ** (n1 - 1) * rb ** (n2 - 1) * (xi * xi - eta * eta) * np.exp(-p[:, None, None] * t * eta)
    cos_a = (1 + xi * eta) / ra
    cos_b = (xi * eta - 1) / rb
    pref = _norm(n1, z1) * _norm(n2, z2) * (R / 2) ** (n1 + n2 + 1) * np.exp(-p) / p
    ang = {(0, 0): 1.0 / (4 * np.pi), (0, 1): np.sqrt(3) / (4 * np.pi) * cos_b, (1, 0): np.sqrt(3) / (4 * np.pi) * cos_a,
           (1, 1): 3.0 / (4 * np.pi) * cos_a * cos_b}[(l1, l2)]
    sig = pref * 2 * np.pi * (w * base * ang).sum((1, 2))
    pi = np.zeros_like(sig)
    if l1 == 1 and l2 == 1:
        sab = (xi * xi - 1) * (1 - eta * eta) / (ra * rb)
        pi = pref * np.pi * (w * base * 3.0 / (4 * np.pi) * sab).sum((1, 2))
    return sig, pi


class _OverlapTable:
    """每对元素的重叠分量对 R 的样条表 (0.3–40 bohr)，加速大批量计算。"""

    def __init__(self):
        self.tab = {}
        self.grid = np.concatenate([np.linspace(0.3, 6.0, 1200), np.linspace(6.0, 40.0, 1400)[1:]])

    def get(self, ea, eb):
        key = (ea, eb)
        if key not in self.tab:
            from scipy.interpolate import CubicSpline
            na, ka, _, za = _p(ea)[:4]
            nb, kb, _, zb = _p(eb)[:4]
            comps = {}
            for la in ((0,) if ka == 1 else (0, 1)):
                for lb in ((0,) if kb == 1 else (0, 1)):
                    s, pi = sto_overlap(na, la, za, nb, lb, zb, self.grid)
                    comps[(la, lb)] = (CubicSpline(self.grid, s), CubicSpline(self.grid, pi) if la == lb == 1 else None)
            self.tab[key] = comps
        return self.tab[key]


_TABLE = _OverlapTable()


def weighted_overlap_block(sym_a, Xa, sym_b, Xb, weighted=True):
    """两组原子之间的 (加权) 价层重叠矩阵 S̃ 与共振积分矩阵 β (eV)。基函数顺序：每个原子 s, px, py, pz (H 只有 s)。"""
    Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float)
    oa, ob = _offsets(sym_a), _offsets(sym_b)
    S = np.zeros((oa[-1], ob[-1]))
    fs, fp = (F_SIG, F_PI) if weighted else (1.0, 1.0)
    d = Xb[None, :, :] - Xa[:, None, :]
    r = np.linalg.norm(d, axis=-1)
    e = d / np.where(r > 0, r, 1)[..., None]
    rb = r / AU2ANG
    pairs = {}
    for i, a in enumerate(sym_a):
        for j, b in enumerate(sym_b):
            pairs.setdefault((a, b), []).append((i, j))
    for (a, b), lst in pairs.items():
        ii = np.array([x[0] for x in lst])
        jj = np.array([x[1] for x in lst])
        R = rb[ii, jj]
        ok = R < 40.0
        ii, jj, R = ii[ok], jj[ok], R[ok]
        if not len(R):
            continue
        comps = _TABLE.get(a, b)
        ev = e[ii, jj]
        ia, jb = oa[ii], ob[jj]
        S[ia, jb] = comps[(0, 0)][0](R)
        if (0, 1) in comps:
            v = comps[(0, 1)][0](R)
            for c in range(3):
                S[ia, jb + 1 + c] = v * ev[:, c]
        if (1, 0) in comps:
            v = comps[(1, 0)][0](R)
            for c in range(3):
                S[ia + 1 + c, jb] = v * ev[:, c]
        if (1, 1) in comps:
            sg = comps[(1, 1)][0](R) * fs
            pi = comps[(1, 1)][1](R) * fp
            for c1 in range(3):
                for c2 in range(3):
                    S[ia + 1 + c1, jb + 1 + c2] = (sg - pi) * ev[:, c1] * ev[:, c2] + (pi if c1 == c2 else 0.0)
    ba = _beta_vec(sym_a)
    bb = _beta_vec(sym_b)
    return S, 0.5 * (ba[:, None] + bb[None, :]) * S


def _offsets(sym):
    o = [0]
    for e in sym:
        o.append(o[-1] + _p(e)[1])
    return np.array(o)


def _beta_vec(sym):
    return np.concatenate([[_p(e)[4]] * _p(e)[1] for e in sym])


def basis_info(sym):
    """(每个基函数所属原子, 角动量类型 0 = s, 1–3 = px, py, pz)。"""
    at, ty = [], []
    for i, e in enumerate(sym):
        k = _p(e)[1]
        at += [i] * k
        ty += list(range(k))
    return np.array(at), np.array(ty)


# ----------------------------------------------------------------------------------------------
# INDO/S 自洽场
# ----------------------------------------------------------------------------------------------
def _one_center(e):
    n, k, zc, z, beta, Is, Ip, F0, G1, F2 = _p(e)
    G1, F2 = G1 / 3.0, F2 / 25.0                      # 表中为 Slater–Condon 积分，INDO 用 G¹/3、F²/25
    if k == 1:
        return np.array([[F0]]), np.array([[F0]]), np.array([-Is])
    coul = np.full((4, 4), F0)                        # (μμ|νν)
    exch = np.zeros((4, 4))                           # (μν|μν)
    for a in range(1, 4):
        coul[a, a] = F0 + 4 * F2
        for b in range(1, 4):
            if a != b:
                coul[a, b] = F0 - 2 * F2
                exch[a, b] = 3 * F2
        exch[0, a] = exch[a, 0] = G1
    exch[np.diag_indices(4)] = np.diag(coul)
    i_s = min(2, zc)
    i_p = zc - i_s
    Uss = -Is - (i_s - 1) * F0 - i_p * (F0 - G1 / 2)
    i_p2 = max(1, zc - 2)
    i_s2 = zc - i_p2
    Upp = -Ip - (i_p2 - 1) * (F0 - 2 * F2) - i_s2 * (F0 - G1 / 2)
    return coul, exch, np.array([Uss, Upp, Upp, Upp])


def gamma_matrix(sym, X):
    X = np.asarray(X, float)
    F0 = np.array([_p(e)[7] for e in sym])
    R = np.linalg.norm(X[:, None] - X[None], axis=-1) / AU2ANG
    g = F_GAMMA * AU2EV / (R + F_GAMMA * 2 * AU2EV / (F0[:, None] + F0[None, :]))
    np.fill_diagonal(g, F0)
    return g


def indo_scf(sym, X, charge=0, maxiter=300, tol=1e-8, guess=None):
    """闭壳层 INDO/S (DIIS；不收敛时改用阻尼迭代重算)。"""
    try:
        return _indo_scf(sym, X, charge, maxiter, tol, guess, diis=True)
    except RuntimeError:
        return _indo_scf(sym, X, charge, 4 * maxiter, tol, guess, diis=False)


def _indo_scf(sym, X, charge, maxiter, tol, guess, diis):
    """闭壳层 INDO/S。返回 {e (eV), C (列为轨道), F, P, nocc, atom, type, H, gamma, energy}。"""
    sym = list(sym)
    X = np.asarray(X, float)
    atom, typ = basis_info(sym)
    nb = len(atom)
    zc = np.array([_p(e)[2] for e in sym], float)
    nel = int(round(zc.sum())) - charge
    if nel % 2:
        raise ValueError("INDO/S 这里只做闭壳层 (电子数为偶数)")
    nocc = nel // 2
    G = gamma_matrix(sym, X)
    _, Hb = weighted_overlap_block(sym, X, sym, X)
    same = atom[:, None] == atom[None, :]
    H = np.where(same, 0.0, Hb)                                  # 原子内无共振积分
    one = [_one_center(e) for e in sym]
    off = _offsets(sym)
    Uvec = np.concatenate([o[2] for o in one])
    VA = -(G * zc[None, :]).sum(1) + zc * np.diag(G)             # −Σ_{B≠A} Z_B γ_AB
    H[np.diag_indices(nb)] = Uvec + VA[atom]
    Gb = G[atom][:, atom]
    P = guess
    if P is None:                                                 # 中性原子密度叠加 (比核心哈密顿量猜测稳定)
        nb_at = np.array([_p(e)[1] for e in sym], float)
        P = np.diag((zc / nb_at)[atom])
        if charge:
            P *= nel / P.trace()
    from collections import deque
    hist_F, hist_e = deque(maxlen=8), deque(maxlen=8)
    E_old = None
    for it in range(maxiter):
        F = H - 0.5 * P * Gb * (~same)
        q = np.array([P[off[a]:off[a + 1], off[a]:off[a + 1]].trace() for a in range(len(sym))])
        F[np.diag_indices(nb)] += (G * q[None, :]).sum(1)[atom] - (np.diag(G) * q)[atom]
        for a, (coul, exch, _) in enumerate(one):
            s = slice(off[a], off[a + 1])
            Pa = P[s, s]
            d = np.diag(Pa)
            blk = np.diag(coul @ d - 0.5 * (exch @ d))
            od = Pa * (1.5 * exch - 0.5 * coul)
            np.fill_diagonal(od, 0.0)
            F[s, s] += blk + od
        E = 0.5 * float((P * (H + F)).sum())
        err = F @ P - P @ F
        hist_F.append(F.copy())
        hist_e.append(err.copy())
        if diis and len(hist_F) > 2:                              # DIIS
            m = len(hist_F)
            B = -np.ones((m + 1, m + 1))
            B[m, m] = 0
            for i in range(m):
                for j in range(m):
                    B[i, j] = (hist_e[i] * hist_e[j]).sum()
            rhs = np.zeros(m + 1)
            rhs[m] = -1
            try:
                c = np.linalg.solve(B, rhs)[:m]
                F = sum(ci * Fi for ci, Fi in zip(c, hist_F))
            except np.linalg.LinAlgError:
                pass
        w, C = np.linalg.eigh(F)
        Pn = 2 * C[:, :nocc] @ C[:, :nocc].T
        dP = np.abs(Pn - P).max()
        P = Pn if diis else 0.5 * (P + Pn)
        if E_old is not None and abs(E - E_old) < tol and dP < 1e-6:
            break
        E_old = E
    else:
        raise RuntimeError("INDO/S 自洽场不收敛")
    # 收敛后的 Fock 与本征值
    F = H - 0.5 * P * Gb * (~same)
    q = np.array([P[off[a]:off[a + 1], off[a]:off[a + 1]].trace() for a in range(len(sym))])
    F[np.diag_indices(nb)] += (G * q[None, :]).sum(1)[atom] - (np.diag(G) * q)[atom]
    for a, (coul, exch, _) in enumerate(one):
        s = slice(off[a], off[a + 1])
        Pa = P[s, s]
        d = np.diag(Pa)
        od = Pa * (1.5 * exch - 0.5 * coul)
        np.fill_diagonal(od, 0.0)
        F[s, s] += np.diag(coul @ d - 0.5 * (exch @ d)) + od
    w, C = np.linalg.eigh(F)
    return {"e": w, "C": C, "F": F, "P": P, "nocc": nocc, "atom": atom, "type": typ, "H": H, "gamma": G,
            "energy": E, "iterations": it + 1, "symbols": sym, "X": X}


# ----------------------------------------------------------------------------------------------
# 单体轨道：软件内 INDO/S 或 Gaussian ZINDO 输出
# ----------------------------------------------------------------------------------------------
def rotate_mo(C, typ, R):
    """刚性转动 R (x' = R x) 后的轨道系数：p 分量按矢量转动。"""
    C = np.array(C, float)
    p0 = np.nonzero(typ == 1)[0]
    for o in p0:
        C[o:o + 3] = R @ C[o:o + 3]
    return C


def read_gaussian_zindo(path):
    """Gaussian ZINDO 计算的 fchk：返回 {symbols, X (Å), C, e (eV), nocc}。基函数按原子 s, px, py, pz (H 只有 s)；
    若重原子带 d 函数 (9 个)，d 系数丢弃并重新归一化。"""
    from .gaussian_io import BOHR2ANG, read_fchk
    from .substitution import ELEMENTS
    d = read_fchk(path)
    Z = [int(z) for z in d["Atomic numbers"]]
    sym = [ELEMENTS[z - 1] for z in Z]
    X = np.asarray(d["Current cartesian coordinates"], float).reshape(-1, 3) * BOHR2ANG
    nbf = int(d["Number of basis functions"])
    nmo = len(d["Alpha Orbital Energies"])
    C = np.asarray(d["Alpha MO coefficients"], float).reshape(nmo, nbf).T
    e = np.asarray(d["Alpha Orbital Energies"], float) * AU2EV
    per = [_p(s)[1] for s in sym]
    if sum(per) != nbf:
        alt = [1 if s == "H" else (9 if _p(s)[0] >= 3 else 4) for s in sym]
        if sum(alt) != nbf:
            raise ValueError(f"{Path(path).name}: 基函数数 {nbf} 与 ZINDO 价层基组不符 (应为 {sum(per)})")
        keep = []
        o = 0
        for k in alt:
            keep += list(range(o, o + min(k, 4)))
            o += k
        C = C[keep]
        C /= np.linalg.norm(C, axis=0)
    nocc = int(d["Number of alpha electrons"])
    return {"symbols": sym, "X": X, "C": C, "e": e, "nocc": nocc, "source": Path(path).name}


class MOO:
    """VOTCA izindo：参考单体 (sym, Y) 的 ZINDO 轨道只算一次，之后对任意相对位置计算 J。
    orbitals: None = 软件内 INDO/S；或 Gaussian ZINDO 的 fchk 路径 (与 VOTCA 相同的输入)。"""

    def __init__(self, sym, Y, orbitals=None, degeneracy=0.0):
        self.sym, self.Y = list(sym), np.asarray(Y, float)
        self.atom, self.typ = basis_info(self.sym)
        if orbitals is None:
            r = indo_scf(self.sym, self.Y)
            self.C, self.e, self.nocc, self.source = r["C"], r["e"], r["nocc"], "INDO/S (软件内)"
        else:
            g = read_gaussian_zindo(orbitals)
            from .polar import match_atoms
            perm, rmsd = match_atoms(self.sym, self.Y, g["symbols"], g["X"])
            if rmsd > 0.1:
                raise ValueError(f"Gaussian ZINDO 结构与单体不一致 (RMSD {rmsd:.2f} Å)")
            # 把 Gaussian 坐标系中的轨道转到单体坐标系：先按原子重排，再求转动
            Xg = g["X"][perm]
            P = Xg - Xg.mean(0)
            Q = self.Y - self.Y.mean(0)
            U, _, Vt = np.linalg.svd(P.T @ Q)
            dd = np.sign(np.linalg.det(U @ Vt))
            Rm = (U @ np.diag([1, 1, dd]) @ Vt).T                    # Q ≈ P Rmᵀ
            og = _offsets(g["symbols"])
            rows = np.concatenate([np.arange(og[k], og[k + 1]) for k in perm])
            C = g["C"][rows]
            self.C = rotate_mo(C, self.typ, Rm)
            self.e, self.nocc, self.source = g["e"], g["nocc"], f"Gaussian ZINDO ({g['source']})"
        self.degeneracy = float(degeneracy)

    def levels(self, carrier):
        """该载流子的 (近简并) 能级组 (VOTCA：与 HOMO / LUMO 能量差小于 degeneracy 的轨道)。"""
        k = self.nocc - 1 if carrier == "hole" else self.nocc
        if self.degeneracy <= 0:
            return [k]
        return [i for i in range(len(self.e)) if abs(self.e[i] - self.e[k]) < self.degeneracy]

    def coupling(self, R, t, carrier="both"):
        """B = Y·Rᵀ + t (相对 A = Y)。返回 {hole, electron} 的 J (eV)；有简并能级时为 √(ΣJ²/(N_A N_B))。"""
        Xb = self.Y @ np.asarray(R, float).T + np.asarray(t, float)
        _, F = weighted_overlap_block(self.sym, self.Y, self.sym, Xb)
        Cb = rotate_mo(self.C, self.typ, np.asarray(R, float))
        out = {}
        for car in (("hole", "electron") if carrier == "both" else (carrier,)):
            L = self.levels(car)
            M = self.C[:, L].T @ F @ Cb[:, L]
            out[car] = float(M[0, 0]) if len(L) == 1 else float(np.sqrt((M ** 2).mean()))
        return out

    def coupling_atoms(self, Xa, Xb):
        """两分子的实验室坐标 (与单体同原子顺序) → 由刚性叠合求出相对位置后计算 J。"""
        Ra, ta = _fit(self.Y, Xa)
        Rb, tb = _fit(self.Y, Xb)
        Rrel = Ra.T @ Rb
        trel = (tb - ta) @ Ra
        return self.coupling(Rrel, trel)


def _fit(Y, X):
    """X ≈ Y Rᵀ + t。"""
    Yc, Xc = Y - Y.mean(0), X - X.mean(0)
    U, _, Vt = np.linalg.svd(Yc.T @ Xc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, X.mean(0) - Y.mean(0) @ R.T


def dimer_projection(sym, Xa, Xb, degeneracy=0.0):
    """二聚体 INDO/S 自洽场 + 投影：J = c_Aᵀ F_dimer c_B (单体轨道来自各自的 INDO/S 计算)。"""
    A = indo_scf(sym, Xa)
    B = indo_scf(sym, Xb)
    n1 = len(A["P"])
    P0 = np.zeros((2 * n1, 2 * n1))
    P0[:n1, :n1], P0[n1:, n1:] = A["P"], B["P"]                  # 单体密度作初始猜测
    D = indo_scf(list(sym) + list(sym), np.vstack([Xa, Xb]), guess=P0)
    na = len(A["atom"])
    out = {}
    for car in ("hole", "electron"):
        k = A["nocc"] - 1 if car == "hole" else A["nocc"]
        LA = [k] if degeneracy <= 0 else [i for i in range(len(A["e"])) if abs(A["e"][i] - A["e"][k]) < degeneracy]
        kb = B["nocc"] - 1 if car == "hole" else B["nocc"]
        LB = [kb] if degeneracy <= 0 else [i for i in range(len(B["e"])) if abs(B["e"][i] - B["e"][kb]) < degeneracy]
        ca = np.zeros((len(D["F"]), len(LA)))
        cb = np.zeros((len(D["F"]), len(LB)))
        ca[:na] = A["C"][:, LA]
        cb[na:] = B["C"][:, LB]
        M = ca.T @ D["F"] @ cb
        out[car] = float(M[0, 0]) if M.size == 1 else float(np.sqrt((M ** 2).mean()))
    return out


# ----------------------------------------------------------------------------------------------
# 与 Gaussian 的对照：检验集的输入与结果汇总
# ----------------------------------------------------------------------------------------------
def gaussian_zindo_projection(fchk_a, fchk_b, fchk_ab):
    """Gaussian ZINDO 的二聚体投影：F_dimer = C ε Cᵀ (零微分重叠基组正交)，J = c_Aᵀ F c_B。
    要求三个计算用同一取向 (nosymm) 且二聚体原子顺序为 A 后接 B。返回 ({hole, electron} eV, 正交性偏差)。"""
    from .gaussian_io import read_fchk
    A, B, D = (read_fchk(f) for f in (fchk_a, fchk_b, fchk_ab))

    def mo(d):
        nbf = int(d["Number of basis functions"])
        e = np.asarray(d["Alpha Orbital Energies"], float)
        C = np.asarray(d["Alpha MO coefficients"], float).reshape(len(e), nbf).T
        return C, e * AU2EV, int(d["Number of alpha electrons"])
    Ca, ea, na = mo(A)
    Cb, eb, nb = mo(B)
    Cd, ed, _ = mo(D)
    if Cd.shape[0] != Ca.shape[0] + Cb.shape[0]:
        raise ValueError("二聚体基函数数不等于两单体之和 (原子顺序应为 A 后接 B)")
    dev = float(np.abs(Cd.T @ Cd - np.eye(Cd.shape[1])).max())
    F = Cd @ np.diag(ed) @ Cd.T
    out = {}
    for car, ka, kb in (("hole", na - 1, nb - 1), ("electron", na, nb)):
        a = np.zeros(len(F))
        b = np.zeros(len(F))
        a[:Ca.shape[0]] = Ca[:, ka]
        b[Ca.shape[0]:] = Cb[:, kb]
        out[car] = float(a @ F @ b)
    return out, dev


def benchmark_set(sym, Y, crys_pairs=(), extra=True):
    """检验用二聚体：晶体中的近邻对 (由调用方给出 [(名称, Xa, Xb)]) + 共面堆积与滑移构型。"""
    cases = list(crys_pairs)
    if extra:
        for d in (3.4, 3.6, 3.9, 4.3):
            cases.append((f"cofacial_{d:.1f}", Y, Y + np.array([0, 0, d])))
        for dx, dy in ((1.2, 0), (2.4, 0), (0, 1.2), (0, 2.4), (1.2, 1.2), (3.6, 0)):
            cases.append((f"slip_{dx:.1f}_{dy:.1f}", Y, Y + np.array([dx, dy, 3.5])))
        th = np.radians(30)
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        cases.append(("rot30_3.6", Y, Y @ Rz.T + np.array([0, 0, 3.6])))
    return cases


def write_benchmark_inputs(outdir, sym, cases, Y=None, level="B3LYP/6-31G(d)", nproc=8, mem="8GB"):
    """每个构型一个目录：Gaussian ZINDO (A、B、AB，nosymm) 与 DFT 投影法 (A、B、AB) 的输入；
    另写参考单体的 ZINDO 输入 (供 MOO 使用 Gaussian 轨道)、manifest.csv 与批处理脚本。"""
    import csv
    from .gaussian_io import write_gjf, write_xyz
    from .jobs import write_windows_bat
    from .transfer_integral import generate_ti_inputs
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    meth, basis = (level.split("/") + ["6-31G(d)"])[:2]
    nel = sum(_p(e)[2] for e in sym)

    def zjob(stem, atoms, d):
        mult = 1
        write_gjf(d / f"{stem}.gjf", [dict(route="#p ZINDO=(NStates=1,Singlets) nosymm", title=f"{stem} ZINDO",
                                           charge=0, mult=mult, atoms=atoms, chk=f"{stem}.chk", nproc=nproc, mem=mem)])
    if Y is not None:
        atoms = [(s, *map(float, x)) for s, x in zip(sym, Y)]
        zjob("mono_zindo", atoms, out)
        write_xyz(out / "monomer.xyz", atoms, "reference monomer")
    rows = []
    for name, Xa, Xb in cases:
        d = out / name
        d.mkdir(exist_ok=True)
        A = [(s, *map(float, x)) for s, x in zip(sym, Xa)]
        B = [(s, *map(float, x)) for s, x in zip(sym, Xb)]
        write_xyz(d / "dimer.xyz", A + B, f"{name}: first {len(sym)} atoms = A")
        for stem, atoms in (("zindo_A", A), ("zindo_B", B), ("zindo_AB", A + B)):
            zjob(stem, atoms, d)
        generate_ti_inputs(d, dimer=str(d / "dimer.xyz"), n_a=len(sym), name="dft", method=meth, basis=basis,
                           nproc=nproc, mem=mem)
        write_windows_bat(d, ["zindo_A", "zindo_B", "zindo_AB"], name="run_zindo.bat")
        rows.append({"case": name, "n_atoms_A": len(sym), "r_com_A": float(np.linalg.norm(Xb.mean(0) - Xa.mean(0)))})
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["@echo off", "rem 依次运行全部构型的 ZINDO 与 DFT 计算并生成 fchk (子脚本末尾的 pause 由 < nul 跳过)",
             "cd /d %~dp0"]
    if Y is not None:
        write_windows_bat(out, ["mono_zindo"], name="run_mono.bat")
        lines.append("call run_mono.bat < nul")
    for r in rows:
        lines += [f"cd {r['case']}", "call run_zindo.bat < nul", "call run_dft.bat < nul", "cd .."]
    lines += ["echo 全部检验计算结束", "pause"]
    (out / "run_all.bat").write_bytes(("\r\n".join(lines) + "\r\n").encode("gbk", errors="replace"))
    return [r["case"] for r in rows]


def collect_benchmark(outdir, degeneracy=0.0):
    """汇总检验集：软件内 MOO、MOO (Gaussian 单体轨道)、软件内二聚体 INDO/S 投影、Gaussian ZINDO 二聚体投影、
    DFT 投影法与 EHT。返回行列表 (J 单位 meV；缺少的计算为 nan)。"""
    from .eht import eht_transfer_integrals
    from .gaussian_io import read_geometry
    from .transfer_integral import calc_transfer_integrals
    out = Path(outdir)
    mono = read_geometry(out / "monomer.xyz")
    sym = [a[0] for a in mono]
    Y = np.array([a[1:] for a in mono], float)
    moo = MOO(sym, Y, degeneracy=degeneracy)
    gfch = out / "mono_zindo.fchk"
    moo_g = MOO(sym, Y, str(gfch), degeneracy) if gfch.exists() else None
    rows = []
    import csv
    for r in csv.DictReader(open(out / "manifest.csv", encoding="utf-8-sig")):
        d = out / r["case"]
        at = read_geometry(d / "dimer.xyz")
        n = int(r["n_atoms_A"])
        Xa = np.array([a[1:] for a in at[:n]], float)
        Xb = np.array([a[1:] for a in at[n:]], float)
        row = {"case": r["case"], "r_com_A": float(r["r_com_A"])}
        j = moo.coupling_atoms(Xa, Xb)
        row["moo_h"], row["moo_e"] = 1000 * j["hole"], 1000 * j["electron"]
        if moo_g is not None:
            j = moo_g.coupling_atoms(Xa, Xb)
            row["moo_gauss_h"], row["moo_gauss_e"] = 1000 * j["hole"], 1000 * j["electron"]
        j = dimer_projection(sym, Xa, Xb, degeneracy)
        row["indo_dimer_h"], row["indo_dimer_e"] = 1000 * j["hole"], 1000 * j["electron"]
        f = [d / f"zindo_{x}.fchk" for x in ("A", "B", "AB")]
        if all(p.exists() for p in f):
            j, dev = gaussian_zindo_projection(*map(str, f))
            row["gauss_zindo_h"], row["gauss_zindo_e"], row["gauss_orth_dev"] = 1000 * j["hole"], 1000 * j["electron"], dev
        f = [d / f"dft_{x}.fchk" for x in ("A", "B", "AB")]
        if all(p.exists() for p in f):
            res = calc_transfer_integrals(*map(str, f), degeneracy=degeneracy)["results"]
            key = "J_deg_meV" if degeneracy > 0 else "J_eff_meV"
            row["dft_h"], row["dft_e"] = res["HOMO"].get(key), res["LUMO"].get(key)
        e = eht_transfer_integrals([(s, *x) for s, x in zip(sym, Xa)], [(s, *x) for s, x in zip(sym, Xb)],
                                   esd=False)["results"]
        row["eht_h"], row["eht_e"] = e["HOMO"]["J_eff_meV"], e["LUMO"]["J_eff_meV"]
        rows.append(row)
    return rows


BENCH_COLS = [("moo", "ZINDO/MOO (软件内)"), ("moo_gauss", "ZINDO/MOO (Gaussian 单体轨道)"),
              ("indo_dimer", "二聚体 INDO/S 投影 (软件内)"), ("gauss_zindo", "二聚体 ZINDO 投影 (Gaussian)"),
              ("dft", "DFT 投影法"), ("eht", "EHT")]


CARRIER_LABEL = {"h": "空穴", "e": "电子"}


def _bench_xy(rows, ref, key, c):
    x, y = [], []
    for r in rows:
        a, b = r.get(f"{ref}_{c}"), r.get(f"{key}_{c}")
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            x.append(abs(a))
            y.append(abs(b))
    return np.array(x), np.array(y)


def benchmark_stats(rows, ref="dft", rel_min=20.0):
    """各方法相对参考 (默认 DFT；没有 DFT 时用 Gaussian ZINDO 投影) 的 |J| 统计，空穴、电子分开：
    过原点斜率 k、R²、MAE，以及 |J_ref| > rel_min meV 的对的平均相对误差 (直接值与按 k 缩放后)。
    空穴与电子的系统偏差常常方向相反 (ZINDO 相对 B3LYP：空穴偏高、电子偏低)，合在一起统计会掩盖这一点。"""
    if not any(f"{ref}_h" in r for r in rows):
        ref = "gauss_zindo" if any("gauss_zindo_h" in r for r in rows) else "indo_dimer"
    out = []
    for key, lab in BENCH_COLS:
        if key == ref:
            continue
        for c in ("h", "e"):
            x, y = _bench_xy(rows, ref, key, c)
            if len(x) < 3:
                continue
            k = float((x * y).sum() / (x * x).sum())
            r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
            big = x > rel_min
            rel = float(np.mean(np.abs(y[big] - x[big]) / x[big])) if big.any() else float("nan")
            rel_k = float(np.mean(np.abs(y[big] / k - x[big]) / x[big])) if big.any() else float("nan")
            out.append({"method": lab, "key": key, "ref": ref, "carrier": c, "n": len(x), "slope": k, "R2": r2,
                        "MAE_meV": float(np.abs(y - x).mean()), "rel_err": rel, "rel_err_scaled": rel_k})
    return out


def plot_benchmark(axes, rows, ref="dft"):
    """空穴、电子各一个子图 (axes 为两个 Axes)。"""
    st = benchmark_stats(rows, ref)
    if not st:
        return st
    ref = st[0]["ref"]
    marks = {"moo": "o", "moo_gauss": "s", "indo_dimer": "^", "gauss_zindo": "D", "dft": "v", "eht": "x"}
    cols = {"moo": "#2C6FB7", "moo_gauss": "#5AA0D8", "indo_dimer": "#7A5BB0", "gauss_zindo": "#2E8B57",
            "dft": "#444444", "eht": "#C44A32"}
    for ax, c in zip(axes, ("h", "e")):
        hi = 1.0
        for s_ in [s for s in st if s["carrier"] == c]:
            x, y = _bench_xy(rows, ref, s_["key"], c)
            hi = max(hi, x.max(), y.max())
            ax.plot(x, y, marks[s_["key"]], color=cols[s_["key"]], ms=5, mfc="none" if s_["key"] != "eht" else None,
                    label=f"{s_['method']}  k = {s_['slope']:.2f}, R² = {s_['R2']:.3f}")
        ax.plot([0, hi * 1.05], [0, hi * 1.05], "-", color="0.6", lw=0.8)
        ax.set_xlim(0, hi * 1.05)
        ax.set_ylim(0, hi * 1.05)
        ax.set_aspect("equal")
        ax.set_title(CARRIER_LABEL[c], fontsize=11)
        ax.set_xlabel(f"|J| {dict(BENCH_COLS)[ref]} (meV)")
        ax.set_ylabel("|J| (meV)")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    return st
