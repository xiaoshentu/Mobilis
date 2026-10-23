"""跳跃模型之外的输运理论：瞬态局域化理论 (TLT)、紧束缚能带 / 有效质量、动态无序 σ_J。

TLT (Fratini, Ciuchi, Troisi 等；参考开源代码 TransLoc, github.com/CiuK1469/TransLoc):
  在周期超胞上构造 H_ij = J_ij + δ_ij (δ ~ N(0, σ_ij), 热涨落)，对角化后
    L²_ab = (1/Z) Σ_n e^{-βE_n} Σ_m (ħv^a)_nm (ħv^b)_nm · 2 / ((ħ/τ)² + (E_n − E_m)²)
    ħv^a_nm = i Σ_ij c_in H_ij (r_j − r_i)_a c_jm
    μ_ab = e L²_ab / (2 k_B T τ)
  对多个无序构型取平均。空穴用 H = −H(HOMO)，电子用 H = +H(LUMO)；能量最低的态热占据。
紧束缚能带: H(k)_ij = Σ J e^{ik·r}，有效质量 (m*⁻¹)_ab = ħ⁻² ∂²E/∂k_a∂k_b (能带极小处)。
动态无序: 对二聚体中分子 B 作刚体热振动采样 (平移 σ_t, 转动 σ_r)，用软件内 EHT 计算 J 的分布。
"""
import numpy as np

from .mobility import HBAR_EVS, KB_EV

HB2_ME = 7.619964   # ħ²/(m_e·Å²) in eV  →  m*/m_e = 7.62 / (∂²E/∂k² in eV·Å²)


# ----------------------------------------------------------------------------------------------
# 图 → 超胞
# ----------------------------------------------------------------------------------------------
def _bonds(graph, keep_axes):
    """每条物理键只取一个方向 (i<j，或 i==j 时平移为正)。返回 [(i, j, t, v, J)]."""
    out = []
    for (i, j, v, jv, mult, t) in graph["edges"]:
        if t is None:
            raise ValueError("跳跃矢量不构成周期格子，无法建立超胞")
        t = tuple(int(x) for x in t)
        if any(t[a] != 0 for a in range(3) if a not in keep_axes):
            continue
        if (i, j) < (j, i) or (i == j and t > (0, 0, 0)):
            out.append((i, j, t, np.asarray(v, float), float(jv)))
    return out


def bond_index(graph, keep_axes):
    """与 _bonds 顺序相同的原始边序号 (用于按键指定 σ)。"""
    out = []
    for n, (i, j, v, jv, mult, t) in enumerate(graph["edges"]):
        t = tuple(int(x) for x in t)
        if any(t[a] != 0 for a in range(3) if a not in keep_axes):
            continue
        if (i, j) < (j, i) or (i == j and t > (0, 0, 0)):
            out.append(n)
    return out


def supercell_hamiltonian(graph, sizes, carrier="hole", sigma=None, rng=None, keep_axes=(0, 1)):
    """返回 H (N×N, eV)、各键的位移矢量矩阵 D (N×N×3, Å, D_ij = r_j − r_i)。sigma: {键序号: σ} 或标量相对值。"""
    nb = graph["n_sites"]
    L = [sizes[a] if a in keep_axes else 1 for a in range(3)]
    cells = [(a, b, c) for a in range(L[0]) for b in range(L[1]) for c in range(L[2])]
    cid = {c: n for n, c in enumerate(cells)}
    N = nb * len(cells)
    H = np.zeros((N, N))
    D = np.zeros((N, N, 3))
    sgn = -1.0 if carrier == "hole" else 1.0
    bonds = _bonds(graph, keep_axes)
    for c in cells:
        for k, (i, j, t, v, jv) in enumerate(bonds):
            s1 = i + nb * cid[c]
            s2 = j + nb * cid[tuple((c[a] + t[a]) % L[a] for a in range(3))]
            sj = 0.0
            if sigma is not None and rng is not None:
                sj = sigma * abs(jv) if np.isscalar(sigma) else sigma.get(k, 0.0)
            val = sgn * (jv + (rng.normal(0, sj) if sj else 0.0))
            H[s1, s2] += val
            H[s2, s1] += val
            D[s1, s2] += v
            D[s2, s1] -= v
    return H, D, bonds


def tlt_mobility(graph, T=300.0, carrier="hole", sigma=0.3, hbar_tau=0.005, sizes=(24, 24, 1),
                 keep_axes=(0, 1), n_real=20, seed=1, plane=None, sigma_E=0.0):
    """瞬态局域化迁移率。sigma: 相对涨落 σ_J/|J| (标量) 或 {键序号: σ(eV)}；hbar_tau: ħ/τ_in (eV)。
    sigma_E: 位点能 (对角) 涨落的标准差 / eV，各分子独立的高斯分布，与键涨落在同一时间尺度上冻结 (默认 0 = 只有非对角无序)。
    返回平面内迁移率张量、主值与角分布。"""
    rng = np.random.default_rng(seed)
    beta = 1 / (KB_EV * T)
    tau = HBAR_EVS / hbar_tau
    acc = np.zeros((3, 3))
    for _ in range(n_real):
        H, D, bonds = supercell_hamiltonian(graph, sizes, carrier, sigma, rng, keep_axes)
        if sigma_E:
            H[np.diag_indices_from(H)] += rng.normal(0.0, sigma_E, len(H))
        e, c = np.linalg.eigh(H)
        w = np.exp(-beta * (e - e.min()))
        w /= w.sum()
        Ms = [c.T @ (H * D[:, :, a]) @ c for a in range(3)]           # ħv^a_nm (取实部, 乘 i 不影响模方)
        de = e[:, None] - e[None, :]
        kern = 2.0 / (hbar_tau ** 2 + de ** 2)
        for a in range(3):
            for b in range(a, 3):
                val = float((w[:, None] * Ms[a] * Ms[b] * kern).sum())
                acc[a, b] += val
                acc[b, a] = acc[a, b]
    L2 = acc / n_real                                                  # Å²
    mu = L2 * 1e-16 / (2 * KB_EV * T * tau)                            # cm²/Vs
    res = {"mu_tensor_cm2_Vs": mu, "L2_A2": L2, "tau_s": tau, "n_bonds": len(bonds)}
    if plane is not None:
        u, v = plane
    else:
        w_, vec = np.linalg.eigh(mu)
        idx = np.argsort(w_)[::-1][:2]
        u, v = vec[:, idx[0]], vec[:, idx[1]]
    phi = np.linspace(0, 2 * np.pi, 361)
    dirs = np.outer(np.cos(phi), u) + np.outer(np.sin(phi), v)
    res["phi"] = phi
    res["mu_phi"] = np.einsum("ij,jk,ik->i", dirs, mu, dirs)
    w_ = np.linalg.eigvalsh(mu)[::-1]
    res["mu_principal"] = w_
    res["L_A"] = float(np.sqrt(max(np.trace(L2) / max(len(keep_axes), 1), 0)))
    return res


# ----------------------------------------------------------------------------------------------
# 能带 / 有效质量
# ----------------------------------------------------------------------------------------------
def band_energies(graph, kvecs, carrier="hole"):
    """kvecs: (nk, 3) 1/Å。返回 (nk, n_sites) 能带 (载流子能量，越低越先占据)。"""
    nb = graph["n_sites"]
    sgn = -1.0 if carrier == "hole" else 1.0
    out = []
    for k in np.atleast_2d(kvecs):
        Hk = np.zeros((nb, nb), complex)
        for (i, j, v, jv, mult, t) in graph["edges"]:
            Hk[i, j] += sgn * jv * mult * np.exp(1j * k @ np.asarray(v))
        Hk = (Hk + Hk.conj().T) / 2
        out.append(np.linalg.eigvalsh(Hk))
    return np.array(out)


def effective_mass(graph, carrier="hole", lattice=None, n_grid=24):
    """在倒空间网格中找到最低能带的极小点，用有限差分求有效质量张量 (单位 m_e)。"""
    from scipy.optimize import minimize
    if lattice is None:
        from .kmc import _lattice_basis
        basis = list(_lattice_basis([np.asarray(e[2], float) for e in graph["edges"]]) or [])
        while len(basis) < 3:          # 不足三维时补上很长的正交矢量 (该方向能带平坦)
            trial = np.eye(3)[len(basis)] if not basis else np.cross(basis[0], basis[-1] if len(basis) > 1 else
                                                                       np.eye(3)[np.argmin(np.abs(basis[0]))])
            basis.append(trial / np.linalg.norm(trial) * 1e3)
        lat = np.array(basis[:3])
    else:
        lat = np.asarray(lattice, float)
    rec = 2 * np.pi * np.linalg.inv(lat).T
    g = (np.arange(n_grid) / n_grid)
    grid = np.array([[a, b, c] for a in g for b in g for c in (g if graph.get("dim3", True) else [0])])
    ks = grid @ rec
    e = band_energies(graph, ks, carrier)[:, 0]
    k0 = ks[int(np.argmin(e))]
    f = lambda k: band_energies(graph, k, carrier)[0, 0]  # noqa: E731
    k0 = minimize(f, k0, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-12, maxiter=4000)).x
    h = 1e-3
    hess = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            ea, eb = np.eye(3)[a] * h, np.eye(3)[b] * h
            hess[a, b] = (f(k0 + ea + eb) - f(k0 + ea - eb) - f(k0 - ea + eb) + f(k0 - ea - eb)) / (4 * h * h)
    w, v = np.linalg.eigh(hess)
    masses = np.where(np.abs(w) > 1e-6, HB2_ME / np.where(np.abs(w) > 1e-6, w, 1), np.inf)
    allE = band_energies(graph, ks, carrier)
    return {"k_min": k0, "E_min": float(f(k0)), "curvature_eVA2": w, "masses_me": masses, "axes": v.T,
            "bandwidth_eV": float(allE[:, 0].max() - allE[:, 0].min()), "reciprocal": rec, "lattice": lat}


def band_path(graph, rec, carrier="hole", npts=60):
    """沿 Γ→a*/2、Γ→b*/2、Γ→c*/2 的能带。"""
    segs, labels = [], []
    for n, lab in enumerate(("X(a*)", "Y(b*)", "Z(c*)")):
        ks = np.linspace(0, 0.5, npts)[:, None] * rec[n][None]
        segs.append(band_energies(graph, ks, carrier))
        labels.append(lab)
    return segs, labels


def band_mobility(masses_me, hbar_tau=0.005):
    """常弛豫时间近似 μ = eτ/m* (cm²/Vs)。"""
    tau = HBAR_EVS / hbar_tau
    me = 9.1093837e-31
    return np.array([1.602176634e-19 * tau / (m * me) * 1e4 if np.isfinite(m) and m > 0 else 0.0 for m in masses_me])


# ----------------------------------------------------------------------------------------------
# 动态无序
# ----------------------------------------------------------------------------------------------
def _rand_rot(rng, sigma_rad):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    ang = rng.normal(0, sigma_rad)
    from .render import _rot
    return _rot(ax, ang)


def dynamic_disorder(atoms_a, atoms_b, n_samples=30, sigma_t=0.10, sigma_r_deg=1.5, T=300.0, seed=7,
                     on_progress=None):
    """刚体热振动采样（幅度 ∝ √(T/300)）后用 EHT 计算 J 分布。返回均值、标准差与 σ/|<J>|。
    分子 B 只求解一次，每个构型的 MO 系数由 p/d 轨道的转动矩阵直接得到，只需重算分子间重叠块。"""
    from .eht import build_basis, eht_transfer_integrals, monomer, rotate_coefficients
    rng = np.random.default_rng(seed)
    f = np.sqrt(T / 300.0)
    xb = np.array([a[1:] for a in atoms_b], float)
    sb = [a[0] for a in atoms_b]
    cen = xb.mean(0)
    mb = monomer(sb, xb)
    jh, je = [], []
    for k in range(n_samples):
        R = _rand_rot(rng, np.radians(sigma_r_deg) * f)
        x = (xb - cen) @ R.T + cen + rng.normal(0, sigma_t * f, 3)
        shells, _, _ = build_basis(sb, x)
        bk = dict(mb, C=rotate_coefficients(mb["shells"], mb["C"], R), shells=shells)
        r = eht_transfer_integrals(atoms_a, [(s, *p) for s, p in zip(sb, x)], esd=False, _mono_b=bk)["results"]
        jh.append(r["HOMO"]["J_eff_meV"])
        je.append(r["LUMO"]["J_eff_meV"])
        if on_progress:
            on_progress(k + 1, n_samples)
    out = {}
    for key, arr in (("hole", np.array(jh)), ("electron", np.array(je))):
        m, s_ = float(arr.mean()), float(arr.std(ddof=1))
        out[key] = {"mean_meV": m, "std_meV": s_, "ratio": s_ / max(abs(m), 1e-9), "samples": arr}
    return out
