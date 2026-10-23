"""软件内计算转移积分：扩展 Hückel (EHT) + 二聚体投影 / 能级分裂。不需要任何外部量化程序。

  基组: Slater 轨道 (STO) 用 6 个高斯函数拟合 (见 _sto6g.py)，重叠积分用 Obara–Saika 公式解析计算；
  Hamiltonian: H_μμ = H_ii(价层电离势)，H_μν = K' S_μν (H_μμ + H_νν)/2 (加权 Wolfsberg–Helmholz, K = 1.75)；
  转移积分: 单体轨道在二聚体 AO 基下直接投影 (与 DIPRO 等价)，并给出能级分裂 (ESD) 结果作对照。
EHT 是半经验方法，结果适合快速筛选与趋势比较，定量结论应与 DFT (Gaussian/ADF) 对照 (VOTCA 对 ZINDO 也有同样建议)。
"""
import numpy as np

from ._sto6g import STO6G
from .gaussian_io import read_geometry

BOHR = 0.529177210903
# 元素: (价电子数, [(n, l, H_ii/eV, ζ1, c1, ζ2, c2)])  Hoffmann / YAeHMOP 参数；过渡金属 d 轨道为双 ζ
EHT_PARAMS = {
    "H": (1, [(1, 0, -13.60, 1.300)]),
    "B": (3, [(2, 0, -15.20, 1.300), (2, 1, -8.50, 1.300)]),
    "C": (4, [(2, 0, -21.40, 1.625), (2, 1, -11.40, 1.625)]),
    "N": (5, [(2, 0, -26.00, 1.950), (2, 1, -13.40, 1.950)]),
    "O": (6, [(2, 0, -32.30, 2.275), (2, 1, -14.80, 2.275)]),
    "F": (7, [(2, 0, -40.00, 2.425), (2, 1, -18.10, 2.425)]),
    "Al": (3, [(3, 0, -12.30, 1.167), (3, 1, -6.50, 1.167)]),
    "Si": (4, [(3, 0, -17.30, 1.383), (3, 1, -9.20, 1.383)]),
    "P": (5, [(3, 0, -18.60, 1.600), (3, 1, -14.00, 1.600)]),
    "S": (6, [(3, 0, -20.00, 2.122), (3, 1, -13.30, 1.827)]),
    "Cl": (7, [(3, 0, -26.30, 2.183), (3, 1, -14.20, 1.733)]),
    "Se": (6, [(4, 0, -20.50, 2.440), (4, 1, -14.40, 2.070)]),
    "Br": (7, [(4, 0, -22.07, 2.588), (4, 1, -13.10, 2.131)]),
    "I": (7, [(5, 0, -18.00, 2.679), (5, 1, -12.70, 2.322)]),
    "Cu": (11, [(4, 0, -11.40, 2.200), (4, 1, -6.06, 2.200), (3, 2, -14.00, 5.950, 0.5933, 2.300, 0.5744)]),
    "Ru": (8, [(5, 0, -10.40, 2.078), (5, 1, -6.87, 2.043), (4, 2, -14.90, 5.378, 0.5340, 2.303, 0.6365)]),
    "Os": (8, [(6, 0, -8.50, 2.452), (6, 1, -4.80, 2.429), (5, 2, -11.00, 5.571, 0.6372, 2.416, 0.5598)]),
    "Ir": (9, [(6, 0, -11.36, 2.500), (6, 1, -4.50, 2.200), (5, 2, -12.17, 5.796, 0.6698, 2.557, 0.5860)]),
    "Pt": (10, [(6, 0, -9.077, 2.554), (6, 1, -5.475, 2.554), (5, 2, -12.59, 6.013, 0.6334, 2.696, 0.5513)]),
}
CART = {0: [(0, 0, 0)], 1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]}
_S3 = np.sqrt(3) / 2
D_SPH = np.array([[-0.5, -0.5, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [_S3, -_S3, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]])   # 归一化笛卡尔 d → dz2, dxz, dyz, dx2-y2, dxy
NSPH = {0: 1, 1: 3, 2: 5}
_DF = {0: 1, 1: 1, 2: 3}                  # (2l-1)!!


def _shell_prims(n, l, zetas_coefs):
    a, c = [], []
    for zeta, coef in zetas_coefs:
        a0, c0 = STO6G[(n, l)]
        a += [x * zeta ** 2 for x in a0]
        c += [x * coef for x in c0]
    return np.array(a), np.array(c)


def build_basis(symbols, coords_ang, params=EHT_PARAMS):
    """返回壳层列表 [{atom, center(bohr), l, a, c, hii, key, offset}] 与基函数数、价电子数。"""
    shells, off, nval = [], 0, 0
    xyz = np.asarray(coords_ang, float) / BOHR
    for i, s in enumerate(symbols):
        if s not in params:
            raise ValueError(f"EHT 参数中没有元素 {s}；可在 EHT_PARAMS 中补充")
        val, shl = params[s]
        nval += val
        for k, sh in enumerate(shl):
            n, l, hii, z1 = sh[:4]
            zc = [(z1, 1.0)] if len(sh) == 4 else [(z1, sh[4]), (sh[5], sh[6])]
            a, c = _shell_prims(n, l, zc)
            shells.append({"atom": i, "center": xyz[i], "l": l, "a": a, "c": c, "hii": hii, "key": (s, k),
                           "offset": off})
            off += NSPH[l]
    return shells, off, nval


def _overlap_1d(pa, pb, p, la, lb):
    """Obara–Saika 一维重叠表 (不含 sqrt(π/p) 因子)，返回 S[i][j] (i≤la, j≤lb)。"""
    s = [[None] * (lb + 1) for _ in range(la + 1)]
    s[0][0] = np.ones_like(pa)
    inv = 1 / (2 * p)
    for j in range(1, lb + 1):
        s[0][j] = pb * s[0][j - 1] + ((j - 1) * s[0][j - 2] if j > 1 else 0) * inv
    for i in range(1, la + 1):
        for j in range(0, lb + 1):
            v = pa * s[i - 1][j]
            if i > 1:
                v = v + (i - 1) * s[i - 2][j] * inv
            if j > 0:
                v = v + j * s[i - 1][j - 1] * inv
            s[i][j] = v
    return s


def _self_norm(sh):
    """同中心收缩函数的自重叠 (原始高斯已归一化)：Σ c_i c_j (2√(a_i a_j)/(a_i+a_j))^(l+3/2)。"""
    a, c, l = sh["a"], sh["c"], sh["l"]
    return np.sqrt((c[:, None] * c[None] * (2 * np.sqrt(a[:, None] * a[None]) / (a[:, None] + a[None])) ** (l + 1.5)).sum())


def overlap_matrix(shells, nbf, cutoff_bohr=18.0, shells_b=None, nbf_b=None):
    """重叠矩阵 S (按壳层类型向量化)。给出 shells_b 时返回 shells × shells_b 的矩形块 (用于分子间部分)。"""
    rect = shells_b is not None
    S = np.zeros((nbf, nbf_b if rect else nbf))
    groups, groups_b = {}, {}
    for sh in shells:
        groups.setdefault(sh["key"], []).append(sh)
    for sh in (shells_b or []):
        groups_b.setdefault(sh["key"], []).append(sh)
    keys = list(groups)
    keys_b = list(groups_b)
    pairs = [(ka, kb) for ka in keys for kb in keys_b] if rect else \
        [(ka, kb) for ia, ka in enumerate(keys) for kb in keys[ia:]]
    for ka, kb in pairs:
        if True:
            ga, gb = groups[ka], (groups_b if rect else groups)[kb]
            la, lb = ga[0]["l"], gb[0]["l"]
            A = np.array([g["center"] for g in ga])
            B = np.array([g["center"] for g in gb])
            ii, jj = np.meshgrid(np.arange(len(ga)), np.arange(len(gb)), indexing="ij")
            ii, jj = ii.ravel(), jj.ravel()
            d2 = ((A[ii] - B[jj]) ** 2).sum(1)
            keep = d2 < cutoff_bohr ** 2
            ii, jj = ii[keep], jj[keep]
            if len(ii) == 0:
                continue
            a, ca = ga[0]["a"], ga[0]["c"]
            b, cb = gb[0]["a"], gb[0]["c"]
            p = a[:, None] + b[None, :]                              # (na, nb)
            mu = a[:, None] * b[None, :] / p
            AB = A[ii] - B[jj]                                        # (np, 3)
            base = (np.pi / p) ** 1.5 * np.exp(-mu[None] * (AB ** 2).sum(1)[:, None, None])
            P = (a[None, :, None, None] * A[ii][:, None, None, :] + b[None, None, :, None] * B[jj][:, None, None, :]) \
                / p[None, :, :, None]
            PA = P - A[ii][:, None, None, :]
            PB = P - B[jj][:, None, None, :]
            tabs = [_overlap_1d(PA[..., k], PB[..., k], p[None], la, lb) for k in range(3)]
            blk = np.zeros((len(ii), len(CART[la]), len(CART[lb])))
            for x, ca_ in enumerate(CART[la]):
                na_ = (2 * a / np.pi) ** 0.75 * (4 * a) ** (la / 2) / np.sqrt(np.prod([_DF[t] for t in ca_]))
                for y, cb_ in enumerate(CART[lb]):
                    nb_ = (2 * b / np.pi) ** 0.75 * (4 * b) ** (lb / 2) / np.sqrt(np.prod([_DF[t] for t in cb_]))
                    prim = base * tabs[0][ca_[0]][cb_[0]] * tabs[1][ca_[1]][cb_[1]] * tabs[2][ca_[2]][cb_[2]]
                    w = (ca * na_)[:, None] * (cb * nb_)[None, :]
                    blk[:, x, y] = (prim * w[None]).sum((1, 2))
            if la == 2:
                blk = np.einsum("sc,pcd->psd", D_SPH, blk)
            if lb == 2:
                blk = np.einsum("pcd,td->pct", blk, D_SPH)
            oa = np.array([g["offset"] for g in ga])[ii]
            ob = np.array([g["offset"] for g in gb])[jj]
            r = oa[:, None, None] + np.arange(NSPH[la])[None, :, None]
            c = ob[:, None, None] + np.arange(NSPH[lb])[None, None, :]
            r, c = np.broadcast_to(r, blk.shape), np.broadcast_to(c, blk.shape)
            S[r, c] = blk
            if not rect:
                S[c, r] = blk
    na = np.concatenate([[_self_norm(sh)] * NSPH[sh["l"]] for sh in shells])
    nb_ = na if not rect else np.concatenate([[_self_norm(sh)] * NSPH[sh["l"]] for sh in shells_b])
    return S / np.outer(na, nb_)


def hamiltonian(shells, S, K=1.75, weighted=True):
    hii = np.concatenate([[sh["hii"]] * NSPH[sh["l"]] for sh in shells])
    hs = hii[:, None] + hii[None, :]
    if weighted:
        dl = (hii[:, None] - hii[None, :]) / hs
        kk = K + dl ** 2 + dl ** 4 * (1 - K)
    else:
        kk = np.full_like(hs, K)
    H = kk * S * hs / 2
    np.fill_diagonal(H, hii)
    return H


def eht(symbols, coords, charge=0, **kw):
    """单个体系的 EHT 计算，返回 (轨道能 eV, 系数, H, S, 占据轨道数)。"""
    from scipy.linalg import eigh
    shells, nbf, nval = build_basis(symbols, coords)
    S = overlap_matrix(shells, nbf)
    H = hamiltonian(shells, S, **kw)
    e, c = eigh(H, S)
    nel = nval - charge
    return {"e": e, "C": c, "H": H, "S": S, "nocc": nel // 2, "odd": nel % 2 == 1, "nbf": nbf}


def eht_transfer_integrals_full(atoms_a, atoms_b, charge_a=0, charge_b=0, orbitals=("HOMO", "LUMO"), **kw):
    """atoms: [(元素, x, y, z)] (Å)。返回各轨道的 J_eff、未正交化 J、S、位点能以及能级分裂法结果 (meV)。"""
    import re
    sa, xa = [a[0] for a in atoms_a], [a[1:] for a in atoms_a]
    sb, xb = [a[0] for a in atoms_b], [a[1:] for a in atoms_b]
    A = eht(sa, xa, charge_a, **kw)
    B = eht(sb, xb, charge_b, **kw)
    D = eht(sa + sb, xa + xb, charge_a + charge_b, **kw)
    na, nb = A["nbf"], B["nbf"]
    cf = np.zeros((na + nb, na + nb))
    cf[:na, :na] = A["C"]
    cf[na:, na:] = B["C"]
    h = cf.T @ D["H"] @ cf
    s = cf.T @ D["S"] @ cf
    res, warn = {}, []
    if A["odd"] or B["odd"]:
        warn.append("单体价电子数为奇数，按闭壳层处理 HOMO/LUMO")
    for lab in orbitals:
        m = re.fullmatch(r"(HOMO|LUMO)([+-]\d+)?", lab.upper())
        sh = int(m.group(2) or 0)
        ia = (A["nocc"] - 1 if m.group(1) == "HOMO" else A["nocc"]) + sh
        ib = na + (B["nocc"] - 1 if m.group(1) == "HOMO" else B["nocc"]) + sh
        ea, eb, j, sab = h[ia, ia], h[ib, ib], h[ia, ib], s[ia, ib]
        jeff = (j - 0.5 * (ea + eb) * sab) / (1 - sab ** 2)
        res[lab.upper()] = {"J_eff_meV": jeff * 1000, "J_raw_meV": j * 1000, "S_AB": sab,
                            "e_A_eV": ea, "e_B_eV": eb}
    nd = D["nocc"]
    res["ESD_hole_meV"] = (D["e"][nd - 1] - D["e"][nd - 2]) / 2 * 1000
    res["ESD_electron_meV"] = (D["e"][nd + 1] - D["e"][nd]) / 2 * 1000
    return {"results": res, "warnings": warn, "nbf": na + nb,
            "monomer_gap_eV": float(A["e"][A["nocc"]] - A["e"][A["nocc"] - 1])}


def eht_from_file(dimer=None, n_a=None, mono_a=None, mono_b=None, **kw):
    """从二聚体文件 (前 n_a 个原子为 A) 或两个单体文件计算 (参数同 eht_transfer_integrals)。"""
    if dimer is not None:
        atoms = read_geometry(dimer)
        return eht_transfer_integrals(atoms[:n_a], atoms[n_a:], **kw)
    return eht_transfer_integrals(read_geometry(mono_a), read_geometry(mono_b), **kw)


# ----------------------------------------------------------------------------------------------
# 快速实现：单体缓存 + 只计算分子间重叠块
# ----------------------------------------------------------------------------------------------
_MONO_CACHE = {}
_S3_ = np.sqrt(3.0)


def _geom_key(symbols, coords, charge):
    x = np.asarray(coords, float)
    return (tuple(symbols), charge, tuple(np.round((x - x[0]).ravel(), 3)))


def monomer(symbols, coords, charge=0, **kw):
    """单体 EHT (按“元素 + 相对坐标”缓存：平移等价的分子不重复计算)。"""
    key = _geom_key(symbols, coords, charge) + (tuple(sorted(kw.items())),)
    hit = _MONO_CACHE.get(key)
    shells, nbf, nval = build_basis(symbols, coords)
    if hit is None:
        r = eht(symbols, coords, charge, **kw)
        hit = {"e": r["e"], "C": r["C"], "nocc": r["nocc"], "odd": r["odd"], "nbf": r["nbf"]}
        if len(_MONO_CACHE) > 64:
            _MONO_CACHE.clear()
        _MONO_CACHE[key] = hit
    return dict(hit, shells=shells)


def _d_rotation(R):
    """实球谐 d 函数 (dz2, dxz, dyz, dx2-y2, dxy) 在转动 R 下的表示矩阵：f_k(Rᵀr) = Σ_m D_mk f_m(r)。"""
    def f(v):
        x, y, z = v.T
        return np.stack([z * z - (x * x + y * y) / 2, _S3_ * x * z, _S3_ * y * z, _S3_ / 2 * (x * x - y * y),
                         _S3_ * x * y], 1)
    pts = np.random.default_rng(0).normal(size=(24, 3))
    D, *_ = np.linalg.lstsq(f(pts), f(pts @ R), rcond=None)   # f(Rᵀ r) 的行向量形式: v @ R = (Rᵀ v)ᵀ
    return D


def rotate_coefficients(shells, C, R):
    """分子刚性转动 R 后的 MO 系数 (s 不变, p 按矢量转动, d 按 _d_rotation)。"""
    U = np.zeros((C.shape[0], C.shape[0]))
    Dd = _d_rotation(R)
    for sh in shells:
        o, l = sh["offset"], sh["l"]
        if l == 0:
            U[o, o] = 1
        elif l == 1:
            U[o:o + 3, o:o + 3] = R
        else:
            U[o:o + 5, o:o + 5] = Dd
    return U @ C


def _coupling_block(A, B, **kw):
    sab = overlap_matrix(A["shells"], A["nbf"], shells_b=B["shells"], nbf_b=B["nbf"])
    ha = np.concatenate([[sh["hii"]] * NSPH[sh["l"]] for sh in A["shells"]])
    hb = np.concatenate([[sh["hii"]] * NSPH[sh["l"]] for sh in B["shells"]])
    K = kw.get("K", 1.75)
    hs = ha[:, None] + hb[None, :]
    if kw.get("weighted", True):
        dl = (ha[:, None] - hb[None, :]) / hs
        kk = K + dl ** 2 + dl ** 4 * (1 - K)
    else:
        kk = K
    return sab, kk * sab * hs / 2


def eht_transfer_integrals(atoms_a, atoms_b, charge_a=0, charge_b=0, orbitals=("HOMO", "LUMO"), esd=True,
                           _mono_b=None, _mono_a=None, **kw):
    """atoms: [(元素, x, y, z)] (Å)。返回 J_eff、J、S、位点能 (meV/eV)，esd=True 时另给能级分裂结果。
    与 eht_transfer_integrals_full 数值一致，但单体结果被缓存，且只计算分子间的重叠块。"""
    import re
    sa, xa = [a[0] for a in atoms_a], [a[1:] for a in atoms_a]
    sb, xb = [a[0] for a in atoms_b], [a[1:] for a in atoms_b]
    A = _mono_a or monomer(sa, xa, charge_a, **kw)
    B = _mono_b or monomer(sb, xb, charge_b, **kw)
    sab, hab = _coupling_block(A, B, **kw)
    res, warn = {}, []
    if A["odd"] or B["odd"]:
        warn.append("单体价电子数为奇数，按闭壳层处理 HOMO/LUMO")
    for lab in orbitals:
        m = re.fullmatch(r"(HOMO|LUMO)([+-]\d+)?", lab.upper())
        sh = int(m.group(2) or 0)
        ia = (A["nocc"] - 1 if m.group(1) == "HOMO" else A["nocc"]) + sh
        ib = (B["nocc"] - 1 if m.group(1) == "HOMO" else B["nocc"]) + sh
        ca, cb = A["C"][:, ia], B["C"][:, ib]
        j, s_ = float(ca @ hab @ cb), float(ca @ sab @ cb)
        ea, eb = float(A["e"][ia]), float(B["e"][ib])
        jeff = (j - 0.5 * (ea + eb) * s_) / (1 - s_ ** 2)
        res[lab.upper()] = {"J_eff_meV": jeff * 1000, "J_raw_meV": j * 1000, "S_AB": s_, "e_A_eV": ea, "e_B_eV": eb}
    out = {"results": res, "warnings": warn, "nbf": A["nbf"] + B["nbf"],
           "monomer_gap_eV": float(A["e"][A["nocc"]] - A["e"][A["nocc"] - 1])}
    if esd:
        from scipy.linalg import eigh
        sA = overlap_matrix(A["shells"], A["nbf"])
        sB = overlap_matrix(B["shells"], B["nbf"])
        S = np.block([[sA, sab], [sab.T, sB]])
        H = np.block([[hamiltonian(A["shells"], sA, **kw), hab], [hab.T, hamiltonian(B["shells"], sB, **kw)]])
        nd = (2 * A["nocc"] + A["odd"] + 2 * B["nocc"] + B["odd"]) // 2     # 二聚体占据轨道数
        e = eigh(H, S, eigvals_only=True, subset_by_index=[nd - 2, nd + 1])
        res["ESD_hole_meV"] = (e[1] - e[0]) / 2 * 1000
        res["ESD_electron_meV"] = (e[3] - e[2]) / 2 * 1000
    return out


def clear_cache():
    _MONO_CACHE.clear()
