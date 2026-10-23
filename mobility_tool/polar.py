"""极化位点能 (微静电模型)：原子电荷 + Thole 诱导偶极。

晶体中把一个分子由中性换成带电态 (原子电荷由 q0 变为 q0 + Δq)，体系能量的变化
    ΔE = ΔE_静电 + ΔE_诱导
    ΔE_静电 = Σ_i∈中心 Δq_i φ_环境(i)                         (电荷与周围分子永久电荷，即电荷–四极项)
    ΔE_诱导 = −μ0·ΔF − ½ Δμ·ΔF                                (线性极化)
其中 F0 为全部分子永久电荷 (只计分子间) 在各原子处的电场，μ0 = A⁻¹F0 为中性晶体的诱导偶极；ΔF 为 Δq 在其它
分子原子处的电场，Δμ = A⁻¹ΔF。A = diag(1/α) − T，T 为 Thole 线性阻尼的偶极场张量 (分子内与分子间的诱导偶极
都相互作用，因此单个分子的极化率张量由原子极化率自洽给出)。
固态极化能 P± = −ΔE±：IE_固 = IE_气 − P+，EA_固 = EA_气 + P−；不同晶体学独立分子的 ΔE 之差就是 KMC 中两套
子格子的位点能差。诱导项随团簇半径按 1/R 收敛，用几个半径外推到 R → ∞。

原子极化率与阻尼参数：van Duijnen & Swart, J. Phys. Chem. A 1998, 102, 2399 (线性 Thole 模型, a = 1.662)。
"""
import csv
from pathlib import Path

import numpy as np

COUL = 14.399645                     # e²/(4πε0) / eV·Å
THOLE_A = 1.662
# Å³
THOLE_ALPHA = {"H": 0.514, "C": 1.405, "N": 1.105, "O": 0.862, "F": 0.440, "S": 2.900, "Cl": 2.315, "Br": 3.013,
               "I": 5.415}
DEFAULT_RADII = (14.0, 18.0, 22.0, 26.0, 30.0)


def atomic_polarizabilities(symbols, table=None):
    t = dict(THOLE_ALPHA, **(table or {}))
    miss = sorted({e for e in symbols if e not in t})
    if miss:
        raise ValueError(f"没有 Thole 原子极化率的元素: {', '.join(miss)} (可在 THOLE_ALPHA 中补充)")
    return np.array([t[e] for e in symbols], float)


# ----------------------------------------------------------------------------------------------
# Thole 诱导偶极
# ----------------------------------------------------------------------------------------------
class Thole:
    """N 个可极化原子。X (N,3) / Å，alpha (N,) / Å³，mol (N,) 分子编号 (电荷场只计分子间)。
    几何系数在内存允许时缓存，否则每次矩阵–向量乘积时分块重算。"""

    def __init__(self, X, alpha, mol, a=THOLE_A, block=192, cache_mb=700):
        self.X = np.asarray(X, float)
        self.alpha = np.asarray(alpha, float)
        self.mol = np.asarray(mol)
        self.N = len(self.X)
        self.a = float(a)
        self.block = int(block)
        self.blocks = [(i, min(i + self.block, self.N)) for i in range(0, self.N, self.block)]
        self._s = self.alpha ** (1.0 / 6.0)
        self._cache = {} if self.N * self.N * 6 * 4 / 2 ** 20 <= cache_mb else None
        self.n_matvec = 0

    def _geom(self, i0, i1):
        if self._cache is not None and i0 in self._cache:
            return self._cache[i0]
        d = (self.X[None, :, :] - self.X[i0:i1, None, :])                    # r_j − r_i
        r2 = (d * d).sum(-1)
        idx = np.arange(i0, i1)
        r2[idx - i0, idx] = np.inf
        r = np.sqrt(r2)
        v = r / (self.a * self._s[i0:i1, None] * self._s[None, :])
        v = np.where(np.isfinite(v), v, 2.0)
        near = v < 1.0
        l3 = np.where(near, 4 * v ** 3 - 3 * v ** 4, 1.0)
        l5 = np.where(near, v ** 4, 1.0)
        c3 = l3 / (r2 * r)
        c5 = 3.0 * l5 / (r2 * r2 * r)
        same = self.mol[i0:i1, None] == self.mol[None, :]
        out = (d.astype(np.float32), c3.astype(np.float32), c5.astype(np.float32), same)
        if self._cache is not None:
            self._cache[i0] = out
        return out

    def charge_field(self, q):
        """原子电荷 q (N,) 或 (N,m) 在各原子处的电场 (只计分子间)，(N,3,m) / e Å⁻²。"""
        q = np.asarray(q, float)
        q2 = q[:, None] if q.ndim == 1 else q
        F = np.zeros((self.N, 3, q2.shape[1]))
        for i0, i1 in self.blocks:
            d, c3, _, same = self._geom(i0, i1)
            w = np.where(same, 0.0, c3)                                     # (B,N)
            # F_i = Σ_j q_j (r_i − r_j) λ3/r³ = −Σ_j q_j c3 d
            F[i0:i1] = -np.einsum("bn,bnc,nm->bcm", w, d, q2, optimize=True)
        return F

    def dipole_field(self, mu):
        """诱导偶极 mu (N,3,m) 产生的电场 T·mu (含分子内)，(N,3,m)。"""
        self.n_matvec += 1
        m = mu.shape[2]
        F = np.empty_like(mu)
        muf = mu.reshape(self.N, 3 * m)
        for i0, i1 in self.blocks:
            d, c3, c5, _ = self._geom(i0, i1)
            dm = np.einsum("bnc,ncm->bnm", d, mu, optimize=True)            # (B,N,m)
            t = c5[:, :, None] * dm
            F[i0:i1] = np.matmul(d.transpose(0, 2, 1), t) - (c3 @ muf).reshape(i1 - i0, 3, m)
        return F

    def solve(self, F, tol=1e-7, maxiter=300):
        """(diag(1/α) − T) μ = F，Jacobi 预条件共轭梯度，各列独立收敛。F: (N,3,m)。"""
        F = np.asarray(F, float)
        if F.ndim == 2:
            F = F[:, :, None]
        al = self.alpha[:, None, None]
        mu = al * F
        r = F - (mu / al - self.dipole_field(mu))
        z = al * r
        p = z.copy()
        rz = (r * z).sum((0, 1))
        nb = np.sqrt((F * F).sum((0, 1))) + 1e-300
        it = 0
        for it in range(1, maxiter + 1):
            if np.all(np.sqrt((r * r).sum((0, 1))) / nb < tol):
                break
            Ap = p / al - self.dipole_field(p)
            pAp = (p * Ap).sum((0, 1))
            alpha_k = np.where(pAp != 0, rz / np.where(pAp != 0, pAp, 1), 0.0)
            mu += alpha_k * p
            r -= alpha_k * Ap
            z = al * r
            rz_new = (r * z).sum((0, 1))
            beta = np.where(rz != 0, rz_new / np.where(rz != 0, rz, 1), 0.0)
            p = z + beta * p
            rz = rz_new
        self.iterations = it
        return mu


def molecular_polarizability(symbols, Y, table=None):
    """单个分子的 Thole 极化率张量 (3×3, Å³)。"""
    Y = np.asarray(Y, float)
    th = Thole(Y, atomic_polarizabilities(symbols, table), np.zeros(len(Y), int))
    F = np.zeros((len(Y), 3, 3))
    for k in range(3):
        F[:, k, k] = 1.0
    mu = th.solve(F, tol=1e-10)
    return mu.sum(0)


def read_polarizability(path):
    """Gaussian 输出 (polar 关键词) 中的静态极化率张量，返回 3×3 / Å³。"""
    import re
    text = Path(path).read_text(errors="ignore")
    m = re.findall(r"Exact polarizability:\s*(" + r"\s+".join([r"(-?\d+\.\d+)"] * 6) + ")", text)
    if not m:
        raise ValueError(f"{Path(path).name} 中没有 “Exact polarizability” (需要 polar 关键词)")
    xx, xy, yy, xz, yz, zz = (float(x) for x in m[-1][1:])
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) * 0.529177210903 ** 3


def scale_alpha(symbols, Y, target_mean, table=None):
    """把全部原子极化率乘同一个因子，使分子平均极化率等于 target_mean (Å³，例如 DFT polar 的结果)。
    返回 (原子极化率表, 因子)。"""
    base = dict(THOLE_ALPHA, **(table or {}))

    def mean(s):
        t = {e: s * v for e, v in base.items()}
        return np.trace(molecular_polarizability(symbols, Y, t)) / 3.0
    lo, hi = 0.3, 3.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if mean(mid) < target_mean:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return {e: s * v for e, v in base.items()}, s


# ----------------------------------------------------------------------------------------------
# 电荷
# ----------------------------------------------------------------------------------------------
def _frame(symbols, X):
    from .surrogate import molecular_frame
    Y, _, ax, c = molecular_frame(symbols, X)
    return np.asarray(Y, float)


def match_atoms(ref_symbols, ref_X, symbols, X):
    """把另一份结构的原子对应到参考单体 (两者各自转到分子坐标系，取四种保持手性的轴符号中 RMSD 最小者，再按元素
    做匈牙利匹配)。返回 (perm, RMSD/Å)：参考单体的第 i 个原子对应另一份结构的第 perm[i] 个原子。"""
    from scipy.optimize import linear_sum_assignment
    ref_symbols, symbols = list(ref_symbols), list(symbols)
    if sorted(ref_symbols) != sorted(symbols):
        raise ValueError("电荷文件中的分子与单体组成不同")
    A, B = _frame(ref_symbols, ref_X), _frame(symbols, X)
    rs, ss = np.array(ref_symbols), np.array(symbols)

    def assign(Bs):
        perm = np.zeros(len(rs), int)
        tot = 0.0
        for e in set(ref_symbols):
            i, j = np.where(rs == e)[0], np.where(ss == e)[0]
            D = np.linalg.norm(A[i][:, None] - Bs[j][None], axis=-1)
            r, c = linear_sum_assignment(D)
            perm[i[r]] = j[c]
            tot += (D[r, c] ** 2).sum()
        return perm, float(np.sqrt(tot / len(rs)))
    best = None
    for s in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        Bs = B * np.array(s, float)
        perm, rmsd = assign(Bs)
        for _ in range(8):                                     # 按当前对应做 Kabsch 叠合后重新匹配
            P_ = Bs[perm] - Bs[perm].mean(0)
            U, _, Vt = np.linalg.svd(P_.T @ (A - A.mean(0)))
            d = np.sign(np.linalg.det(U @ Vt))
            Rm = U @ np.diag([1, 1, d]) @ Vt
            Bn = (Bs - Bs[perm].mean(0)) @ Rm + A.mean(0)
            p2, r2 = assign(Bn)
            if r2 >= rmsd - 1e-9:
                break
            Bs, perm, rmsd = Bn, p2, r2
        if best is None or rmsd < best[1]:
            best = (perm, rmsd)
    return best


def charges_from_log(ref_symbols, ref_X, path):
    """Gaussian 输出中的原子电荷 (ESP 优先，否则 Mulliken)，按单体原子顺序排列。返回 (q, 类型, 对应 RMSD)。"""
    from .gaussian_io import read_geometry
    from .interaction import read_charges
    q, kind = read_charges(path)
    atoms = read_geometry(path)
    sym = [a[0] for a in atoms]
    X = np.array([a[1:] for a in atoms], float)
    if len(q) != len(sym):
        raise ValueError(f"{Path(path).name}: 电荷数 ({len(q)}) 与原子数 ({len(sym)}) 不一致")
    perm, rmsd = match_atoms(ref_symbols, ref_X, sym, X)
    return np.asarray(q, float)[perm], kind, rmsd


def frontier_delta_q(symbols, X, carrier):
    """EHT 前线轨道的 Mulliken 布居：空穴 Δq = +p(HOMO)，电子 Δq = −p(LUMO)。"""
    from .eht import NSPH, build_basis, eht
    r = eht(list(symbols), np.asarray(X, float))
    k = r["nocc"] - 1 if carrier == "hole" else r["nocc"]
    c = r["C"][:, k]
    p_bf = c * (r["S"] @ c)
    shells, _, _ = build_basis(list(symbols), np.asarray(X, float))
    p = np.zeros(len(symbols))
    for sh in shells:
        p[sh["atom"]] += p_bf[sh["offset"]:sh["offset"] + NSPH[sh["l"]]].sum()
    p /= p.sum()
    return p if carrier == "hole" else -p


def default_charges(symbols, Y):
    """无 Gaussian 电荷时：中性 = OPLS-AA 芳香 C/H 电荷；离子 = 中性 + EHT 前线轨道布居。"""
    from .rigidmd import force_field
    q0 = force_field(list(symbols), Y)[2]
    hetero = sorted({e for e in symbols if e not in ("C", "H")})
    note = "中性：OPLS-AA 芳香 C/H 电荷；离子：EHT 前线轨道 Mulliken 布居"
    if hetero:
        note += f"；含 {', '.join(hetero)}，建议改用 Gaussian ESP 电荷 (pop=mk)"
    return {"q0": q0, "dq_h": frontier_delta_q(symbols, Y, "hole"), "dq_e": frontier_delta_q(symbols, Y, "electron"),
            "note": note}


def charge_set(symbols, Y, neutral=None, cation=None, anion=None):
    """组装三套电荷 (单体原子顺序)。neutral/cation/anion 为 Gaussian 输出路径 (可缺省，缺省时用默认电荷)。"""
    base = default_charges(symbols, Y)
    out = dict(base)
    notes = []
    if neutral:
        q0, kind, rm = charges_from_log(symbols, Y, neutral)
        out["q0"] = q0
        notes.append(f"中性 {kind} ({Path(neutral).name}, 对应 RMSD {rm:.2f} Å)")
    for path, key, tot, lab in ((cation, "dq_h", 1.0, "阳离子"), (anion, "dq_e", -1.0, "阴离子")):
        if path:
            q, kind, rm = charges_from_log(symbols, Y, path)
            if abs(q.sum() - tot) > 0.05:
                raise ValueError(f"{Path(path).name} 的电荷总和为 {q.sum():+.2f}，不是{lab}")
            out[key] = q - out["q0"]
            notes.append(f"{lab} {kind} ({Path(path).name}, 对应 RMSD {rm:.2f} Å)")
    if notes:
        out["note"] = "；".join(notes) + ("" if (neutral and cation and anion) else "；其余：" + base["note"])
    return out


def write_charge_inputs(symbols, X, prefix, level="B3LYP/6-31G(d)", nproc=8, mem="8GB"):
    """生成中性 / 阳离子 / 阴离子在中性几何下的单点 ESP 电荷输入 (pop=mk)；中性任务同时计算极化率 (polar)，
    用于把 Thole 原子极化率缩放到 DFT 的分子极化率。返回文件列表。"""
    from .gaussian_io import write_gjf
    atoms = [(s, *map(float, x)) for s, x in zip(symbols, np.asarray(X, float))]
    nel = sum(_Z.get(e, 0) for e in symbols)
    out = []
    for tag, ch in (("neutral", 0), ("cation", 1), ("anion", -1)):
        mult = 1 if (nel - ch) % 2 == 0 else 2
        p = Path(f"{prefix}_{tag}.gjf")
        extra = " polar" if ch == 0 else ""
        write_gjf(p, [dict(route=f"#p {level} pop=(mk) iop(6/33=2){extra} scf=(xqc,maxcycle=512)", title=f"{tag} ESP charges",
                           charge=ch, mult=mult, atoms=atoms, chk=p.with_suffix(".chk").name, nproc=nproc, mem=mem)])
        out.append(str(p))
    return out


_Z = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Se": 34, "Br": 35, "I": 53}


# ----------------------------------------------------------------------------------------------
# 团簇与位点能
# ----------------------------------------------------------------------------------------------
def _cell_range(lattice, R):
    rec = np.linalg.inv(np.asarray(lattice, float)).T
    return [int(np.ceil(R * np.linalg.norm(rec[i]))) + 1 for i in range(3)]


def crystal_cluster(lattice, frames, Y, center, R):
    """以晶胞中第 center 个分子为中心、质心距离 ≤ R 的球形团簇。返回 (原子坐标 (M,nat,3), 子格子编号 (M,), 中心序号)。"""
    lat = np.asarray(lattice, float)
    Y = np.asarray(Y, float)
    base = np.array([Y @ np.asarray(ax, float).T + np.asarray(c, float) for ax, c in frames])     # (nk,nat,3)
    cen = base.mean(1)
    c0 = cen[center]
    na, nb, nc = _cell_range(lat, R)
    mols, kk = [], []
    ci = None
    for i in range(-na, na + 1):
        for j in range(-nb, nb + 1):
            for k in range(-nc, nc + 1):
                t = np.array([i, j, k], float) @ lat
                for m in range(len(frames)):
                    if np.linalg.norm(cen[m] + t - c0) <= R + 1e-6:
                        if i == j == k == 0 and m == center:
                            ci = len(mols)
                        mols.append(base[m] + t)
                        kk.append(m)
    return np.array(mols), np.array(kk), ci


def _cluster_energies(Xm, ci, alpha1, q0, dq, on_iter=None):
    """团簇 Xm (M,nat,3)、中心 ci；dq: {"h": Δq, "e": Δq}。返回各载流子的能量分项 (eV)。"""
    M, nat, _ = Xm.shape
    X = Xm.reshape(-1, 3)
    mol = np.repeat(np.arange(M), nat)
    th = Thole(X, np.tile(alpha1, M), mol)
    keys = list(dq)
    Q = np.zeros((M * nat, 1 + len(keys)))
    Q[:, 0] = np.tile(q0, M)
    sl = slice(ci * nat, (ci + 1) * nat)
    for n, k in enumerate(keys):
        Q[sl, 1 + n] = dq[k]
    F = th.charge_field(Q)
    mu = th.solve(F)
    # 中心分子原子处的环境电势 (只计其它分子的永久电荷)
    other = mol != ci
    d = X[sl][:, None, :] - X[other][None, :, :]
    phi = (Q[other, 0][None, :] / np.linalg.norm(d, axis=-1)).sum(1)
    out = {}
    for n, k in enumerate(keys):
        dF, dmu = F[:, :, 1 + n], mu[:, :, 1 + n]
        st = COUL * float(dq[k] @ phi)
        cross = -COUL * float((mu[:, :, 0] * dF).sum())
        pol = -0.5 * COUL * float((dmu * dF).sum())
        out[k] = {"static": st, "cross": cross, "pol": pol, "total": st + cross + pol}
    return out, {"atoms": M * nat, "molecules": M, "iterations": th.iterations, "matvec": th.n_matvec}


def site_energies(lattice, frames, symbols, Y, charges, radii=DEFAULT_RADII, alpha_table=None, on_progress=None):
    """晶胞中各独立分子的极化位点能。charges = charge_set(...)。
    返回 {"sites": [{k, rows: [{R, h: {...}, e: {...}, atoms}], h: {...外推}, e: {...}}], "P_plus", "P_minus", ...}。"""
    alpha1 = atomic_polarizabilities(symbols, alpha_table)
    radii = sorted(float(r) for r in radii)
    alpha_mol = molecular_polarizability(symbols, Y, alpha_table)
    density = len(frames) / abs(np.linalg.det(np.asarray(lattice, float)))
    eps_ll = lorentz_lorenz(float(np.trace(alpha_mol) / 3), density)
    sites = []
    for k in range(len(frames)):
        rows = []
        for R in radii:
            if on_progress:
                on_progress(f"分子 {k + 1}/{len(frames)}，R = {R:g} Å")
            Xm, _, ci = crystal_cluster(lattice, frames, Y, k, R)
            e, info = _cluster_energies(Xm, ci, alpha1, charges["q0"], {"h": charges["dq_h"], "e": charges["dq_e"]})
            rows.append(dict(R=R, **e, **info))
        site = {"k": k, "rows": rows}
        for car in ("h", "e"):
            site[car] = extrapolate(rows, car, eps_ll)
        sites.append(site)
    return finish_site_energies(sites, symbols, Y, alpha_table, alpha_mol, eps_ll, density, radii, charges)


def finish_site_energies(sites, symbols, Y, alpha_table, alpha_mol, eps_ll, density, radii, charges):
    """由各分子的外推结果汇总 P±、子格子位点能差及其误差。"""
    P_plus = -np.mean([s["h"]["total"] for s in sites])
    P_minus = -np.mean([s["e"]["total"] for s in sites])
    return {"eps_LL": eps_ll, "density": density, "sites": sites, "P_plus": float(P_plus), "P_minus": float(P_minus),
            "offsets_h": [s["h"]["total"] + P_plus for s in sites],
            "offsets_e": [s["e"]["total"] + P_minus for s in sites],
            "offsets_unc_h": float(np.sqrt(np.nanmean([s["h"]["elec_unc"] ** 2 for s in sites]))),
            "offsets_unc_e": float(np.sqrt(np.nanmean([s["e"]["elec_unc"] ** 2 for s in sites]))),
            "alpha_mol": alpha_mol, "radii": radii, "note": charges.get("note", "")}


def lorentz_lorenz(alpha_mean, density):
    """Lorentz–Lorenz：(ε − 1)/(ε + 2) = 4πNα/3 (α / Å³，N / Å⁻³)。"""
    y = 4 * np.pi * density * alpha_mean / 3
    return float((1 + 2 * y) / (1 - y)) if y < 0.95 else float("inf")


def extrapolate(rows, car, eps_ll=None):
    """R → ∞ 外推。
    诱导项：渐近形式 P(R) = P∞ − (1 − 1/ε)e²/(2R)。默认用最大半径的值加 Born 尾项，ε 取 Lorentz–Lorenz 估计
    (对并五苯，从 22–34 Å 出发的结果相差 < 0.02 eV)；另给最大三个半径的 1/R 拟合及其斜率对应的 ε 作对照
    (团簇太小时表面效应使斜率偏陡、ε 不合理)。没有 eps_ll 时用 1/R 拟合。
    静电与交叉项：球形截断下条件收敛，随半径摆动 (壳层效应)，取最大几个半径 (≥ 0.7 R_max，至少两个) 的平均，
    标准偏差作为误差。"""
    R = np.array([r["R"] for r in rows])
    pol = np.array([r[car]["pol"] for r in rows])
    i_max = int(np.argmax(R))
    use = np.argsort(R)[-3:]
    b, a = (np.polyfit(1.0 / R[use], pol[use], 1) if len(use) >= 2 else (0.0, pol[i_max]))
    x = 2 * b / COUL
    eps_fit = 1.0 / (1.0 - x) if 0 < x < 1 else float("inf")
    born = None
    if eps_ll is not None and np.isfinite(eps_ll):
        born = pol[i_max] - (1 - 1 / eps_ll) * COUL / (2 * R[i_max])
    if born is not None:
        p_inf, method = float(born), "Born 尾项"
    else:
        p_inf, method = float(a), "1/R 拟合"
    big = np.nonzero(R >= 0.7 * R[i_max])[0]
    if len(big) < 2:
        big = np.argsort(R)[-2:]
    sc = np.array([rows[k][car]["static"] + rows[k][car]["cross"] for k in big])
    st = np.array([rows[k][car]["static"] for k in big])
    out = {"static": float(st.mean()), "cross": float(sc.mean() - st.mean()), "pol": p_inf,
           "pol_Rmax": float(pol[i_max]), "slope": float(b), "eps_fit": float(eps_fit), "pol_fit": float(a),
           "pol_born": float(born) if born is not None else float("nan"), "method": method, "R_max": float(R[i_max]),
           "elec_unc": float(sc.std(ddof=1)) if len(sc) > 1 else float("nan"), "R_avg": [float(R[k]) for k in big]}
    out["total"] = out["static"] + out["cross"] + out["pol"]
    return out


def solid_levels(res, ip_gas=None, ea_gas=None):
    """固态 IE / EA / 输运能隙 (eV)；缺少气相值时返回 None。"""
    out = {}
    if ip_gas is not None:
        out["IE_solid"] = ip_gas - res["P_plus"]
    if ea_gas is not None:
        out["EA_solid"] = ea_gas + res["P_minus"]
    if ip_gas is not None and ea_gas is not None:
        out["E_transport"] = out["IE_solid"] - out["EA_solid"]
    return out


# ----------------------------------------------------------------------------------------------
# MD 快照：热涨落位点能
# ----------------------------------------------------------------------------------------------
def load_traj(path):
    d = np.load(path)
    need = ("c", "Q", "Y_body", "symbols", "supercell", "lattice")
    miss = [k for k in need if k not in d]
    if miss:
        raise ValueError(f"轨迹文件缺少 {miss}；请用“全晶体 MD”保存的 .npz")
    return {"c": d["c"].astype(float), "Q": d["Q"].astype(float), "Y": d["Y_body"].astype(float),
            "symbols": [str(s) for s in d["symbols"]], "supercell": tuple(int(x) for x in d["supercell"]),
            "lattice": d["lattice"].astype(float), "dt_frame_fs": float(d["dt_frame_fs"]) if "dt_frame_fs" in d else 20.0}


def box_shifts(box, R):
    """周期盒子 (行矢量) 中覆盖半径 R 所需的全部平移矢量 (Å)。"""
    ns = _cell_range(box, R)
    return np.array([(i, j, k) for i in range(-ns[0], ns[0] + 1) for j in range(-ns[1], ns[1] + 1)
                     for k in range(-ns[2], ns[2] + 1)], float) @ np.asarray(box, float)


def box_cluster(c, A, shifts, I, R):
    """周期盒子中以分子 I 为中心、质心距离 ≤ R 的团簇。c: (N,3) 质心；A: (N,nat,3) 原子。返回 (Xm, 中心序号)。"""
    d = c[None, :, :] + shifts[:, None, :] - c[I]
    s_, j_ = np.nonzero(np.linalg.norm(d, axis=-1) <= R)
    Xm = A[j_] + shifts[s_][:, None, :]
    ci = int(np.nonzero((j_ == I) & (np.linalg.norm(shifts[s_], axis=1) < 1e-9))[0][0])
    return Xm, ci


def static_site_energies(c, A, box, dq, q0, R=30.0):
    """只计静电项：Δq 与 R 以内其它分子永久电荷的作用 (eV)。返回 (N,) 数组。"""
    shifts = box_shifts(box, R)
    N = len(c)
    out = np.zeros(N)
    for I in range(N):
        d = c[None, :, :] + shifts[:, None, :] - c[I]
        s_, j_ = np.nonzero(np.linalg.norm(d, axis=-1) <= R)
        keep = ~((j_ == I) & (np.linalg.norm(shifts[s_], axis=1) < 1e-9))
        env = (A[j_[keep]] + shifts[s_[keep]][:, None, :]).reshape(-1, 3)
        qe = np.tile(q0, int(keep.sum()))
        r = np.linalg.norm(A[I][:, None, :] - env[None], axis=-1)
        out[I] = COUL * float(dq @ (qe[None, :] / r).sum(1))
    return out


def traj_from_md(md, frames):
    """RigidCrystalMD 与 run 的采样轨迹 → md_site_energies 的输入 (不经过文件)。"""
    return {"c": np.asarray(frames["c"], float), "Q": np.asarray(frames["Q"], float), "Y": np.asarray(md.Y, float),
            "symbols": list(md.sym), "supercell": tuple(md.sc), "lattice": np.asarray(md.lat, float),
            "dt_frame_fs": float(frames.get("dt_frame_fs", 20.0))}


def md_site_energies(traj, charges, n_frames=24, centers=4, R=14.0, nk=None, alpha_table=None, seed=11,
                     on_progress=None):
    """MD 快照中若干分子的位点能 (半径 R 的团簇，用超胞周期像补齐)。charges 必须与轨迹单体同一原子顺序。
    返回 {"E_h", "E_e": (帧, 中心), "k": 子格子, "sigma_h", "sigma_e", "mean_h", "mean_e" (按子格子)}。"""
    tr = load_traj(traj) if isinstance(traj, (str, Path)) else traj
    sym, Yb = tr["symbols"], tr["Y"]
    alpha1 = atomic_polarizabilities(sym, alpha_table)
    box = np.diag(tr["supercell"]) @ tr["lattice"]
    nf, N = tr["c"].shape[:2]
    nk = nk or int(round(N / np.prod(tr["supercell"])))
    rng = np.random.default_rng(seed)
    fidx = np.unique(np.linspace(0, nf - 1, min(n_frames, nf)).round().astype(int))
    # 中心分子：各子格子均匀抽取
    per = max(1, centers // nk)
    cen = np.concatenate([rng.choice(np.arange(k, N, nk), per, replace=False) for k in range(nk)])
    shifts = box_shifts(box, R)
    Eh = np.zeros((len(fidx), len(cen)))
    Ee = np.zeros_like(Eh)
    for a, f in enumerate(fidx):
        c, Q = tr["c"][f], tr["Q"][f]
        A = c[:, None, :] + np.einsum("nij,aj->nai", Q, Yb)                 # (N,nat,3)
        for b, I in enumerate(cen):
            if on_progress:
                on_progress(f"快照 {a + 1}/{len(fidx)}，分子 {b + 1}/{len(cen)}")
            Xm, ci = box_cluster(c, A, shifts, I, R)
            e, _ = _cluster_energies(Xm, ci, alpha1, charges["q0"], {"h": charges["dq_h"], "e": charges["dq_e"]})
            Eh[a, b], Ee[a, b] = e["h"]["total"], e["e"]["total"]
    kcen = cen % nk
    out = {"E_h": Eh, "E_e": Ee, "k": kcen, "centers": cen, "frames": fidx, "R": R, "nk": nk,
           "dt_frame_fs": tr["dt_frame_fs"]}
    for car, E in (("h", Eh), ("e", Ee)):
        means = np.array([E[:, kcen == k].mean() for k in range(nk)])
        dev = E - means[kcen][None, :]
        out[f"mean_{car}"] = means
        out[f"sigma_{car}"] = float(dev.std(ddof=1))
        out[f"dev_{car}"] = dev
    return out


# ----------------------------------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------------------------------
def summary_text(res, levels=None, md=None):
    L = []
    a = np.linalg.eigvalsh(res["alpha_mol"])
    L.append(f"分子极化率 (Thole)：主值 {a[2]:.1f} / {a[1]:.1f} / {a[0]:.1f} Å³，平均 {a.mean():.1f} Å³")
    L.append(f"电荷：{res['note']}")
    L.append("")
    L.append(f"{'分子':>4}{'载流子':>6}{'R/Å':>7}{'静电':>9}{'交叉':>9}{'诱导':>9}{'合计':>9}  (eV)")
    for s in res["sites"]:
        for car, lab in (("h", "空穴"), ("e", "电子")):
            for r in s["rows"]:
                x = r[car]
                L.append(f"{s['k'] + 1:>4}{lab:>6}{r['R']:7.0f}{x['static']:9.3f}{x['cross']:9.3f}{x['pol']:9.3f}"
                         f"{x['total']:9.3f}")
            x = s[car]
            L.append(f"{s['k'] + 1:>4}{lab:>6}{'∞':>7}{x['static']:9.3f}{x['cross']:9.3f}{x['pol']:9.3f}{x['total']:9.3f}")
    L.append("")
    x = res["sites"][0]["h"]
    if "eps_fit" in x:
        L.append(f"外推 ({x['method']})：最大半径 {x['R_max']:g} Å；1/R 拟合斜率对应 ε = "
                 + (f"{x['eps_fit']:.2f}" if np.isfinite(x['eps_fit']) else "∞ (不合理)")
                 + f"，Lorentz–Lorenz ε = {res.get('eps_LL', float('nan')):.2f}")
        if not (1.5 <= x["eps_fit"] <= 8.0):
            L.append("  [提示] 1/R 拟合的斜率不合理 (团簇未进入渐近区，表面效应使斜率偏陡)，不宜直接外推；"
                     "采用 Born 尾项。最大半径加到 26–30 Å 可使两种外推接近")
    L.append(f"固态极化能 P+ = {res['P_plus']:.3f} eV，P− = {res['P_minus']:.3f} eV (R → ∞ 外推)")
    if len(res["sites"]) > 1:
        uh, ue = res.get("offsets_unc_h", float("nan")), res.get("offsets_unc_e", float("nan"))
        L.append("各独立分子的位点能 (相对平均值, meV)：空穴 " +
                 " / ".join(f"{1000 * x:+.0f}" for x in res["offsets_h"]) + f" (± {1000 * uh:.0f})，电子 " +
                 " / ".join(f"{1000 * x:+.0f}" for x in res["offsets_e"]) + f" (± {1000 * ue:.0f})")
        L.append("  误差为静电 + 交叉项在最大几个半径之间的涨落 (球形截断的壳层效应)")
        if all(np.isfinite(u) for u in (uh, ue)) and \
                max(np.ptp(res["offsets_h"]) / max(2 * uh, 1e-9), np.ptp(res["offsets_e"]) / max(2 * ue, 1e-9)) < 2:
            L.append("  [提示] 位点能差与零无显著差别 (小于两倍误差)；KMC 中可只用热涨落 σ_E")
    for k, lab in (("IE_solid", "固态电离能 IE"), ("EA_solid", "固态电子亲和能 EA"), ("E_transport", "输运能隙")):
        if levels and k in levels:
            L.append(f"{lab} = {levels[k]:.2f} eV")
    if md:
        L.append("")
        L.append(f"MD 快照 (R = {md['R']:g} Å，{len(md['frames'])} 帧 × {len(md['centers'])} 个分子)：")
        L.append(f"  热涨落位点能 σ_E：空穴 {1000 * md['sigma_h']:.0f} meV，电子 {1000 * md['sigma_e']:.0f} meV")
        if md["nk"] > 1:
            L.append("  子格子平均位点能差 (meV)：空穴 " +
                     " / ".join(f"{1000 * (x - md['mean_h'].mean()):+.0f}" for x in md["mean_h"]) + "，电子 " +
                     " / ".join(f"{1000 * (x - md['mean_e'].mean()):+.0f}" for x in md["mean_e"]))
    return "\n".join(L)


def export_csv(res, path, md=None, levels=None):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["molecule", "carrier", "R_A", "static_eV", "cross_eV", "induction_eV", "total_eV", "atoms"])
        for s in res["sites"]:
            for car in ("h", "e"):
                for r in s["rows"]:
                    x = r[car]
                    w.writerow([s["k"] + 1, "hole" if car == "h" else "electron", r["R"], f"{x['static']:.5f}",
                                f"{x['cross']:.5f}", f"{x['pol']:.5f}", f"{x['total']:.5f}", r["atoms"]])
                x = s[car]
                w.writerow([s["k"] + 1, "hole" if car == "h" else "electron", "inf", f"{x['static']:.5f}",
                            f"{x['cross']:.5f}", f"{x['pol']:.5f}", f"{x['total']:.5f}", ""])
        w.writerow([])
        w.writerow(["P_plus_eV", f"{res['P_plus']:.5f}", "P_minus_eV", f"{res['P_minus']:.5f}"])
        for k, v in (levels or {}).items():
            w.writerow([k, f"{v:.4f}"])
        if md:
            w.writerow([])
            w.writerow(["md_R_A", md["R"], "sigma_hole_eV", f"{md['sigma_h']:.5f}", "sigma_electron_eV",
                        f"{md['sigma_e']:.5f}"])
            w.writerow(["frame", "molecule_index", "sublattice", "E_hole_eV", "E_electron_eV"])
            for a, f in enumerate(md["frames"]):
                for b, I in enumerate(md["centers"]):
                    w.writerow([int(f), int(I), int(md["k"][b]) + 1, f"{md['E_h'][a, b]:.5f}", f"{md['E_e'][a, b]:.5f}"])


def plot_extrapolation(ax, res):
    cols = {"h": "#2C6FB7", "e": "#C44A32"}
    marks = ["o", "s", "^", "D"]
    for s in res["sites"]:
        for car in ("h", "e"):
            R = np.array([r["R"] for r in s["rows"]])
            y = np.array([-r[car]["total"] for r in s["rows"]])
            lab = ("P+ " if car == "h" else "P− ") + f"分子 {s['k'] + 1}"
            ax.plot(1 / R, y, marks[s["k"] % 4], color=cols[car], ms=6, label=lab, ls="none")
            x = np.linspace(0, 1 / R.min(), 20)
            ex = s[car]
            base = ex["static"] + ex["cross"]
            ax.plot(x, -(base + ex.get("pol_fit", ex["pol"]) + ex["slope"] * x), "-", color=cols[car], lw=1.0,
                    alpha=0.5)
            ax.plot([0], [-ex["total"]], "*", color=cols[car], ms=11, mec="k", mew=0.5)
    ax.set_xlabel("1 / R (Å⁻¹)")
    ax.set_ylabel("极化能 P (eV)")
    ax.text(0.98, 0.02, "★ 采用的 R → ∞ 值", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)


def plot_md_hist(ax, md):
    cols = {"h": "#2C6FB7", "e": "#C44A32"}
    for car, lab in (("h", "空穴"), ("e", "电子")):
        d = 1000 * md[f"dev_{car}"].ravel()
        ax.hist(d, bins=24, histtype="stepfilled", alpha=0.35, color=cols[car],
                label=f"{lab}  σ = {1000 * md[f'sigma_{car}']:.0f} meV")
        ax.hist(d, bins=24, histtype="step", color=cols[car], lw=1.2)
    ax.set_xlabel("位点能涨落 (meV)")
    ax.set_ylabel("次数")
    ax.legend(fontsize=8, frameon=False)
