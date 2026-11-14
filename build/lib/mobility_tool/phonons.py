"""声子分辨的动态无序：刚体晶格动力学 → 各晶格模对转移积分涨落 σ_V 与迁移率的贡献 (找出“杀手声子”)。

思路
  1. 力场与 ceds md 相同 (OPLS-AA 芳香 C/H + DSF 静电，或用户 ESP 电荷)；超胞中所有分子为刚体，晶格参数固定为实验值。
     先在固定晶胞中做刚体弛豫 (只调质心与取向)，再用有限差分求力常数；利用晶格平移对称，只需移动参考晶胞中的
     分子 (2 × 6 × Z 次力计算)，得到参考分子与超胞中所有分子之间的 6×6 力常数块。
  2. 在与超胞相容的 q 网格 (na×nb×nc 个点) 上构造动力学矩阵 D(q) (平动用分子质量，转动用实验室系惯量张量做质量加权)，
     对角化得到 6Z 支声子 ω_b(q) 与本征矢；Γ 点的 3 个声学模不计。
  3. 每个唯一二聚体的相对几何 q = (z, Δx, Δy, γ, θ, φ) 对两个分子 12 个刚体坐标的雅可比矩阵 J (数值微分)，
     耦合曲面在实验结构处的梯度 ∇V；模 (q, b) 的非局域电声耦合
         g_qb = ∇V · [J_A ε_A + J_B ε_B exp(2πi q·t)]，ε = M^-1/2 e (每单位正则坐标的位移)。
  4. 热振幅 ⟨|Q|²⟩ = k_BT/ω² (经典) 或 ħ/(2ω) coth(ħω/2k_BT) (量子)，σ²_V = (1/N_q) Σ_qb |g_qb|² ⟨|Q_qb|²⟩
     (线性耦合、谐振近似；与同一超胞中直接对角化 6N 维 Hessian 完全等价)。
  5. 迁移率：σ_V 接到瞬态局域化 (CEDS.tlt)；∂μ/∂σ²_g 用数值微分 (固定随机数)，每支声子的贡献
     Δμ_b ≈ −Σ_g (∂μ/∂σ²_g) σ²_{g,b}，即“去掉这一支迁移率增加多少”；对贡献最大的几支另做精确的去除重算。
  6. 模式特征：每支在分子坐标系 (x 长轴、y 短轴、z 法线) 中的平动 / 转动成分 (Γ 点与 q 平均)，用来辨认长轴滑移等运动。
     由 σ² 加权的平均 ħω 给出 ħ/τ 的估计 (TLT 的非弹性散射率)，可代替经验的 5 meV。

近似：刚体分子 (分子内振动不计；它们主要调制位点能，非局域耦合以低频晶格模为主)、谐振、线性耦合、固定晶胞。
线性-谐振 σ_V 可与 ceds md (非谐、非线性) 的 σ_V 直接比较，偏差即为这些近似的误差。

单位：Å、fs、amu、kcal/mol (与 rigidmd 相同)；频率 cm⁻¹；耦合 meV。
"""
from __future__ import annotations

import numpy as np

from .rigidmd import KB, KCAL2MD

HBAR_MD = 6.350780e-3          # ħ / (amu Å² fs⁻¹)
HBAR_MEV_FS = 658.2119569      # ħ / (meV fs)
C_CM_FS = 2.99792458e-5        # 光速 / (cm fs⁻¹)
CHAR_KEYS = ("T_long", "T_short", "T_normal", "R_long", "R_short", "R_normal")
CHAR_LABEL_EN = {"T_long": "long-axis translation", "T_short": "short-axis translation",
                 "T_normal": "normal translation", "R_long": "rotation about long axis",
                 "R_short": "rotation about short axis", "R_normal": "rotation about normal"}
CHAR_LABEL = {"T_long": "长轴平移", "T_short": "短轴平移", "T_normal": "法向平移",
              "R_long": "绕长轴转动", "R_short": "绕短轴转动", "R_normal": "绕法线转动"}
GRAD_STEP = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])   # 曲面梯度的差分步长 (Å, Å, Å, °, °, °)


def _rotvec(w):
    from .ceds import rotvec_matrix
    return rotvec_matrix(w)


def to_cm(omega):
    """角频率 / fs⁻¹ → 波数 / cm⁻¹ (虚频返回负值)。"""
    omega = np.asarray(omega, float)
    return omega / (2 * np.pi * C_CM_FS)


class RigidLattice:
    """md: rigidmd.RigidCrystalMD (提供力场、超胞与分子)。"""

    def __init__(self, md):
        self.md = md
        self.c = md.c0.copy()
        self.Q = md.Q0.copy()
        self.relaxed = False
        self.relax_info = {}

    # ---- 弛豫 ----
    def _state(self, x):
        N = self.md.N
        x = x.reshape(2, N, 3)
        return self.c_ref + x[0], _rotvec(x[1]) @ self.Q_ref

    def relax(self, ftol=0.02, max_iter=60, on_progress=None):
        """固定晶胞的刚体弛豫 (质心 + 取向)：L-BFGS，每次以当前结构为参考重新展开转动 (梯度 −τ 在参考点精确)。
        ftol: 收敛判据，最大力 / (kcal mol⁻¹ Å⁻¹) 与最大力矩 / (kcal mol⁻¹ rad⁻¹)。"""
        from scipy.optimize import minimize
        md = self.md
        F0, t0, E0 = md.forces(self.c, self.Q)
        hist = [(E0 / md.N, float(np.abs(F0).max()), float(np.abs(t0).max()))]
        for it in range(max_iter):
            self.c_ref, self.Q_ref = self.c.copy(), self.Q.copy()

            def fun(x):
                c, Q = self._state(x)
                F, tau, E = md.forces(c, Q)
                return E, -np.concatenate([F.ravel(), tau.ravel()])

            r = minimize(fun, np.zeros(6 * md.N), jac=True, method="L-BFGS-B",
                         options={"maxiter": 40, "gtol": ftol * 0.5})
            self.c, self.Q = self._state(r.x)
            self.c -= (self.c - md.c0).mean(0)                     # 去掉整体平移
            F, tau, E = md.forces(self.c, self.Q)
            fmax, tmax = float(np.abs(F).max()), float(np.abs(tau).max())
            hist.append((E / md.N, fmax, tmax))
            if on_progress:
                on_progress(f"弛豫 {it + 1}: E = {E / md.N:.3f} kcal/mol/分子，最大力 {fmax:.3g}，最大力矩 {tmax:.3g}")
            if fmax < ftol and tmax < ftol:
                break
        Qm, o = md.mol_frames(self.c, self.Q)
        Qm0, o0 = md.mol_frames(md.c0, md.Q0)
        dcom = np.linalg.norm(self.c - md.c0, axis=1)
        drot = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", Qm, Qm0) - 1) / 2, -1, 1)))
        self.relaxed = True
        self.relax_info = {"E0": hist[0][0], "E": hist[-1][0], "f_max": hist[-1][1], "tau_max": hist[-1][2],
                           "shift_max_A": float(dcom.max()), "rot_max_deg": float(drot.max()), "iterations": len(hist) - 1}
        return self.relax_info

    # ---- 力常数 ----
    def _grad(self, c, Q):
        F, tau, _ = self.md.forces(c, Q)
        return -np.concatenate([F, tau], axis=1)                   # (N, 6)

    def force_constants(self, ht=0.004, hr=0.002, on_progress=None):
        """参考晶胞中 Z 个分子与超胞中全部分子之间的力常数 FC (Z, 6, N, 6) (kcal mol⁻¹ Å⁻²、Å⁻¹ rad⁻¹、rad⁻²)。
        坐标：质心 (实验室系) 与实验室系小转动矢量。"""
        md = self.md
        FC = np.zeros((md.nk, 6, md.N, 6))
        for k in range(md.nk):
            n0 = md.index((0, 0, 0), k)
            for d in range(6):
                g = []
                for s in (1, -1):
                    c, Q = self.c.copy(), self.Q.copy()
                    if d < 3:
                        c[n0, d] += s * ht
                    else:
                        w = np.zeros(3)
                        w[d - 3] = s * hr
                        Q[n0] = _rotvec(w) @ Q[n0]
                    g.append(self._grad(c, Q))
                FC[k, d] = (g[0] - g[1]) / (2 * (ht if d < 3 else hr))
            if on_progress:
                on_progress(f"力常数：分子 {k + 1}/{md.nk}")
        self.FC = FC
        # 对称性检查：Φ(0k,d; Rk',d') 与 Φ(0k',d'; −R k,d) 应相等
        dev = 0.0
        for k in range(md.nk):
            for cl in md.cells:
                for k2 in range(md.nk):
                    a = FC[k, :, md.index(cl, k2), :]
                    b = FC[k2, :, md.index(tuple(-np.array(cl)), k), :].T
                    dev = max(dev, float(np.abs(a - b).max()))
        self.asym = dev / max(np.abs(FC).max(), 1e-12)
        return FC

    def _minv(self):
        """参考晶胞中各分子的 M^-1/2 (6×6)。"""
        md = self.md
        out = []
        for k in range(md.nk):
            n = md.index((0, 0, 0), k)
            I = self.Q[n] @ np.diag(md.I) @ self.Q[n].T
            w, v = np.linalg.eigh(I)
            L = np.zeros((6, 6))
            L[:3, :3] = np.eye(3) / np.sqrt(md.M)
            L[3:, 3:] = v @ np.diag(1 / np.sqrt(w)) @ v.T
            out.append(L)
        return out

    def qgrid(self):
        na, nb, nc = self.md.sc
        return np.array([(i / na, j / nb, l / nc) for i in range(na) for j in range(nb) for l in range(nc)])

    def dynmat(self, q):
        """动力学矩阵 D(q) (6Z × 6Z, fs⁻²)，q 为分数坐标。"""
        md = self.md
        nk = md.nk
        Phi = np.zeros((nk, 6, nk, 6), complex)
        for cl in md.cells:
            ph = np.exp(2j * np.pi * np.dot(q, cl))
            for k2 in range(nk):
                Phi[:, :, k2, :] += self.FC[:, :, md.index(cl, k2), :] * ph
        Phi = Phi.reshape(6 * nk, 6 * nk)
        L = self._minv()
        Lb = np.zeros((6 * nk, 6 * nk))
        for k in range(nk):
            Lb[6 * k:6 * k + 6, 6 * k:6 * k + 6] = L[k]
        D = Lb @ Phi @ Lb * KCAL2MD
        return 0.5 * (D + D.conj().T), Lb

    def modes(self, qs=None):
        """q 网格上的声子。返回 dict：q (Nq,3)、omega (Nq,nb) fs⁻¹ (虚频为负)、freq_cm、eps (Nq, 6Z, nb)：
        每单位正则坐标 (amu^1/2 Å) 时参考晶胞各分子的广义位移 (质心 Å、转动 rad)，复数。"""
        qs = self.qgrid() if qs is None else np.asarray(qs, float)
        om, eps = [], []
        for q in qs:
            D, Lb = self.dynmat(q)
            lam, e = np.linalg.eigh(D)
            om.append(np.sign(lam) * np.sqrt(np.abs(lam)))
            eps.append(Lb @ e)
        om = np.array(om)
        self.mode = {"q": qs, "omega": om, "freq_cm": to_cm(om), "eps": np.array(eps)}
        return self.mode

    def character(self):
        """各模在分子坐标系中的成分 (Nq, nb, 6)，和为 1：平动按 M|u·a|²、转动按 |θ·a|² (aᵀ I a) 计权。"""
        md = self.md
        Qm, _ = md.mol_frames(self.c, self.Q)
        eps = self.mode["eps"]
        Nq, n6, nb = eps.shape
        W = np.zeros((Nq, nb, 6))
        for k in range(md.nk):
            n = md.index((0, 0, 0), k)
            A = Qm[n]                                               # 列：长轴、短轴、法线
            I = self.Q[n] @ np.diag(md.I) @ self.Q[n].T
            Iax = np.einsum("ia,ij,ja->a", A, I, A)
            tr = np.einsum("ia,qib->qba", A, eps[:, 6 * k:6 * k + 3, :])
            ro = np.einsum("ia,qib->qba", A, eps[:, 6 * k + 3:6 * k + 6, :])
            W[:, :, :3] += md.M * np.abs(tr) ** 2
            W[:, :, 3:] += Iax * np.abs(ro) ** 2
        return W / np.maximum(W.sum(-1, keepdims=True), 1e-30)


# ----------------------------------------------------------------------------------------------
# 二聚体几何的雅可比矩阵与耦合梯度
# ----------------------------------------------------------------------------------------------
def pair_jacobians(lat, ceds, ht=0.002, hr=0.001):
    """每个唯一二聚体：参数 (z, Δx, Δy, γ, θ, φ) 对 A、B 两个分子 12 个刚体坐标的雅可比矩阵 (6, 12)，角度 / °。
    A 取参考晶胞中的分子，B 取平移 t 的分子。返回 {gid: (J, k_A, k_B, t)}。"""
    from .ceds import OPS, _wrap, pair_params
    md = lat.md
    out = {}
    for p in ceds.pairs:
        t = np.array(p["translation"])
        nA = md.index((0, 0, 0), p["ref"])
        nB = md.index(tuple(t), p["mol"])
        shift = (t // np.array(md.sc)) @ md.box
        G = np.diag(OPS[p["g"]]).astype(float)

        def q_of(c, Q):
            Qm, o = md.mol_frames(c, Q)
            return pair_params(Qm[nA], o[nA], Qm[nB] @ G, o[nB] + shift)

        J = np.zeros((6, 12))
        for j, n in enumerate((nA, nB)):
            for d in range(6):
                qq = []
                for s in (1, -1):
                    c, Q = lat.c.copy(), lat.Q.copy()
                    if d < 3:
                        c[n, d] += s * ht
                    else:
                        w = np.zeros(3)
                        w[d - 3] = s * hr
                        Q[n] = _rotvec(w) @ Q[n]
                    qq.append(q_of(c, Q))
                dq = qq[0] - qq[1]
                dq[3:] = _wrap(dq[3:])
                J[:, 6 * j + d] = dq / (2 * (ht if d < 3 else hr))
        out[p["gid"]] = (J, p["ref"], p["mol"], t)
    return out


def surface_gradient(ceds, surfaces, carriers=("hole", "electron")):
    """各唯一二聚体在实验结构 q0 处的 V 与 ∇_q V (meV / 参数单位，含特征标 χ)。返回 {gid: {carrier: (V0, grad)}}。"""
    from .ceds import PARAM_KEYS
    out = {}
    for p in ceds.pairs:
        surf = surfaces(p["gid"], p["center"]) if callable(surfaces) else surfaces.get(p["gid"])
        if surf is None:
            continue
        q0 = np.asarray(p["q0"], float)
        Qs = [q0]
        for i in range(6):
            for s in (1, -1):
                q = q0.copy()
                q[i] += s * GRAD_STEP[i]
                Qs.append(q)
        Qs = np.array(Qs)
        frozen = set(getattr(surf, "frozen", []))
        out[p["gid"]] = {}
        for car in carriers:
            chi = ceds.chi(car, p["g"])
            v = chi * np.asarray(surf(Qs, car), float)
            grad = np.array([(v[1 + 2 * i] - v[2 + 2 * i]) / (2 * GRAD_STEP[i]) for i in range(6)])
            for i, k in enumerate(PARAM_KEYS):
                if k in frozen:
                    grad[i] = 0.0
            out[p["gid"]][car] = (float(v[0]), grad)
    return out


def occupation(omega, T, quantum=False):
    """⟨|Q|²⟩ / (amu Å²)。omega / fs⁻¹ (>0)。"""
    omega = np.asarray(omega, float)
    kT = KB * T * KCAL2MD
    if not quantum:
        return kT / omega ** 2
    x = HBAR_MD * omega / (2 * kT)
    return HBAR_MD / (2 * omega) / np.tanh(x)


def mode_couplings(lat, ceds, surfaces, T=300.0, quantum=False, wmin_cm=1.0, carriers=("hole", "electron"),
                   jac=None, grad=None):
    """主要结果：每个模 (q, b) 对每个二聚体 σ²_V 的贡献。返回 dict：
    q、freq_cm (Nq, nb)、char (Nq, nb, 6)、active、occ、
    pairs {gid: {carrier: {V0, sigma, s2 (Nq, nb), s2_branch (nb,), grad}}}、hbar_tau_meV {carrier}、
    branch {freq_gamma, freq_min, freq_max, char_gamma (nb, 6), char_mean (nb, 6)}。"""
    md = lat.md
    jac = jac if jac is not None else pair_jacobians(lat, ceds)
    grad = grad if grad is not None else surface_gradient(ceds, surfaces, carriers)
    qs, om, eps = lat.mode["q"], lat.mode["omega"], lat.mode["eps"]
    fcm = lat.mode["freq_cm"]
    Nq = len(qs)
    active = fcm > wmin_cm
    occ = np.zeros_like(om)
    occ[active] = occupation(om[active], T, quantum)
    pairs = {}
    for g, (J, kA, kB, t) in jac.items():
        if g not in grad:
            continue
        ph = np.exp(2j * np.pi * qs @ t)                        # (Nq,)
        uA = eps[:, 6 * kA:6 * kA + 6, :]                       # (Nq, 6, nb)
        uB = eps[:, 6 * kB:6 * kB + 6, :] * ph[:, None, None]
        dq = np.einsum("pi,qib->qbp", J[:, :6], uA) + np.einsum("pi,qib->qbp", J[:, 6:], uB)   # (Nq, nb, 6)
        pairs[g] = {}
        for car in carriers:
            V0, gr = grad[g][car]
            gm = np.abs(dq @ gr) ** 2                            # |g_qb|²
            s2 = gm * occ / Nq
            pairs[g][car] = {"V0": V0, "sigma": float(np.sqrt(s2.sum())), "s2": s2, "s2_branch": s2.sum(0),
                             "grad": gr}
    ch = lat.character()
    ig = int(np.argmin(np.linalg.norm(qs, axis=1)))
    w = np.where(active, 1.0, 0.0)[..., None]
    out = {"T": T, "quantum": quantum, "q": qs, "freq_cm": fcm, "omega": om, "active": active, "occ": occ,
           "char": ch, "pairs": pairs, "n_imag": int((fcm < -wmin_cm).sum()), "Nq": Nq,
           "branch": {"freq_gamma": fcm[ig], "freq_min": np.where(active, fcm, np.inf).min(0),
                      "freq_max": fcm.max(0), "char_gamma": ch[ig],
                      "char_mean": (ch * w).sum(0) / np.maximum(w.sum(0), 1e-30)}}
    rf = getattr(lat, "info", {}).get("rigid_fraction")
    if rf is not None:                                                 # 外部声子：各支的刚体成分 (q 平均)
        out["branch"]["rigid"] = rf.mean(0)
    out["source"] = "external" if rf is not None else "rigid"
    out["hbar_tau_meV"] = {}
    for car in carriers:
        s2 = sum(v[car]["s2"] for v in pairs.values())
        out["hbar_tau_meV"][car] = float(HBAR_MEV_FS * (om * s2)[active].sum() / max(s2[active].sum(), 1e-30))
    return out


def as_ceds_results(mc, carriers=("hole", "electron"), drop_branches=()):
    """转换成 CEDS.run 的结果格式 (TLT 用)。drop_branches: 要整支去掉的声子支。"""
    res = {}
    for g, d in mc["pairs"].items():
        r = {"gid": g, "T": mc["T"]}
        for car in carriers:
            sb = d[car]["s2_branch"].copy()
            sb[list(drop_branches)] = 0.0
            r[car] = {"V0": d[car]["V0"], "mean": d[car]["V0"], "std": float(np.sqrt(sb.sum())), "ratio": 0.0}
        res[g] = r
    return res


def mobility_contributions(ceds, mc, carrier, hbar_tau=None, n_real=16, eps=0.25, top=4, seed=1):
    """TLT 迁移率与各声子支的贡献。hbar_tau / eV (默认取声子估计)。
    返回 {mu, mu_static, dmu_branch (nb,) 去掉该支时 μ 的线性增量, sens {gid: ∂μ/∂σ²}, top [(支, Δμ 精确)]}。"""
    T = mc["T"]
    ht = hbar_tau if hbar_tau is not None else mc["hbar_tau_meV"][carrier] / 1000.0
    res = as_ceds_results(mc, (carrier,))

    def mu(r):
        return float(ceds.tlt(r, carrier, T, ht, n_real=n_real, seed=seed)["mu_principal"][0])

    mu0 = mu(res)
    mus = float(ceds.tlt(res, carrier, T, ht, n_real=1, seed=seed, sigma_scale=0.0)["mu_principal"][0])
    sens = {}
    for g, d in mc["pairs"].items():
        s2 = d[carrier]["sigma"] ** 2
        if s2 < 1e-6:
            sens[g] = 0.0
            continue
        r2 = {k: ({**v, carrier: dict(v[carrier], std=v[carrier]["std"] * np.sqrt(1 - eps))} if k == g else v)
              for k, v in res.items()}
        sens[g] = (mu(r2) - mu0) / (-eps * s2)                 # ∂μ/∂σ² (通常 < 0)
    nb = len(mc["branch"]["freq_gamma"])
    dmu = np.zeros(nb)
    for g, d in mc["pairs"].items():
        dmu -= sens[g] * d[carrier]["s2_branch"]
    exact = [(int(b), mu(as_ceds_results(mc, (carrier,), drop_branches=[b])) - mu0)
             for b in np.argsort(dmu)[::-1][:top]]
    return {"mu": mu0, "mu_static": mus, "hbar_tau_eV": ht, "dmu_branch": dmu, "sens": sens, "top": exact}


def dominant_character(ch):
    """模式成分 → 主导运动的标签。"""
    i = int(np.argmax(ch))
    return CHAR_KEYS[i], float(ch[i])


def _desc(ch, n=2, en=False):
    lab = CHAR_LABEL_EN if en else CHAR_LABEL
    return ", ".join(f"{lab[CHAR_KEYS[i]]} {ch[i]:.0%}" for i in np.argsort(ch)[::-1][:n])


def band_shares(mc, carrier, edges=(0, 25, 50, 75, 100, 150, 1e9)):
    """σ² (全部二聚体求和) 按频率段的份额。"""
    f = mc["freq_cm"]
    tot = sum(d[carrier]["s2"] for d in mc["pairs"].values())
    s = tot.sum() or 1.0
    return [(edges[i], edges[i + 1], float(tot[(f > edges[i]) & (f <= edges[i + 1])].sum() / s))
            for i in range(len(edges) - 1)]


def character_shares(mc, carrier, weights=None):
    """σ² 按运动类型的份额 (每个模按成分分摊)。weights: {gid: 权重} (默认全部二聚体等权求和)。"""
    tot = sum((weights or {}).get(g, 1.0) * d[carrier]["s2"] for g, d in mc["pairs"].items())
    sh = (mc["char"] * tot[..., None]).sum((0, 1))
    return dict(zip(CHAR_KEYS, sh / max(sh.sum(), 1e-30)))


def summary_text(mc, mob=None, top=6):
    """文字报告：每个二聚体的 σ_V 与贡献最大的声子支、σ² 的频率与运动分布、迁移率贡献最大的声子支。"""
    f = mc["freq_cm"][mc["active"]]
    br = mc["branch"]
    lines = [f"{mc['Nq']} 个 q 点 × {len(br['freq_gamma'])} 支声子，频率 {f.min():.1f}–{f.max():.1f} cm⁻¹；"
             f"{'量子' if mc['quantum'] else '经典'}振幅，T = {mc['T']:g} K"]
    if mc["n_imag"]:
        lines.append(f"[提示] {mc['n_imag']} 个虚频模未计入 (弛豫不完全或力场在实验晶胞下不稳定)")
    for car in ("hole", "electron"):
        if not any(car in d for d in mc["pairs"].values()):
            continue
        lines.append(f"\n[{car}]  ħ/τ (σ² 加权的平均 ħω) = {mc['hbar_tau_meV'][car]:.2f} meV")
        lines.append(f"{'组':<5}{'V0/meV':>9}{'σ/meV':>8}{'σ/|V|':>7}   贡献最大的声子支 (Γ 频率 cm⁻¹, 份额)")
        for g, d in sorted(mc["pairs"].items()):
            x = d[car]
            sb = x["s2_branch"]
            if x["sigma"] < 1e-3:
                continue
            idx = np.argsort(sb)[::-1][:3]
            desc = "; ".join(f"b{i} ({br['freq_gamma'][i]:.0f}, {sb[i] / sb.sum():.0%})" for i in idx)
            lines.append(f"G{g:<4}{x['V0']:9.2f}{x['sigma']:8.2f}{x['sigma'] / max(abs(x['V0']), 1e-9):7.2f}   {desc}")
        bs = band_shares(mc, car)
        lines.append("σ² 频率分布 (cm⁻¹): " + ", ".join(f"{lo:g}–{hi:g} {s:.0%}" if hi < 1e8 else f">{lo:g} {s:.0%}"
                                                     for lo, hi, s in bs))
        cs = character_shares(mc, car)
        lines.append("σ² 运动分布: " + ", ".join(f"{CHAR_LABEL[k]} {v:.0%}" for k, v in cs.items()))
        if mob and car in mob:
            m = mob[car]
            lines.append(f"TLT μ = {m['mu']:.3g} cm²/Vs (σ = 0: {m['mu_static']:.3g})，ħ/τ = {1000 * m['hbar_tau_eV']:.2f} meV")
            dm = m["dmu_branch"]
            ex = dict(m["top"])
            rg = br.get("rigid")
            lines.append(f"{'支':>4}{'Γ cm⁻¹':>9}{'范围':>13}{'Δμ/μ 线性':>11}{'精确':>9}"
                         + (f"{'刚体':>6}" if rg is not None else "") + "   运动 (Γ 点)")
            for b in np.argsort(dm)[::-1][:top]:
                e = f"{ex[b] / m['mu']:+9.0%}" if b in ex else " " * 9
                rng = f"{br['freq_min'][b]:.0f}–{br['freq_max'][b]:.0f}"
                lines.append(f"{b:4d}{br['freq_gamma'][b]:9.1f}{rng:>13}{dm[b] / m['mu']:+11.0%}{e}"
                             + (f"{rg[b]:6.0%}" if rg is not None else "") + f"   {_desc(br['char_gamma'][b])}")
    return "\n".join(lines)


def export_csv(mc, path, mob=None):
    """两节：各声子支 (Γ 频率、范围、成分、各二聚体 σ² 份额、Δμ/μ)；各 (q, 支) 的频率与 σ²。"""
    import csv
    gids = sorted(mc["pairs"])
    br = mc["branch"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        head = ["branch", "freq_gamma_cm", "freq_min_cm", "freq_max_cm"] + [f"gamma_{k}" for k in CHAR_KEYS]
        for car in ("hole", "electron"):
            head += [f"s2_{car}_G{g}_meV2" for g in gids]
            if mob and car in mob:
                head += [f"dmu_rel_{car}"]
        w.writerow(head)
        for b in range(len(br["freq_gamma"])):
            row = [b, f"{br['freq_gamma'][b]:.2f}", f"{br['freq_min'][b]:.2f}", f"{br['freq_max'][b]:.2f}"]
            row += [f"{x:.3f}" for x in br["char_gamma"][b]]
            for car in ("hole", "electron"):
                row += [f"{mc['pairs'][g][car]['s2_branch'][b]:.5g}" for g in gids]
                if mob and car in mob:
                    row += [f"{mob[car]['dmu_branch'][b] / mob[car]['mu']:.4g}"]
            w.writerow(row)
        w.writerow([])
        w.writerow(["group", "carrier", "V0_meV", "sigma_harm_meV", "dmu_dsigma2_cm2Vs_per_meV2"])
        for g in gids:
            for car in ("hole", "electron"):
                d = mc["pairs"][g][car]
                sens = f"{mob[car]['sens'].get(g, 0.0):.4g}" if mob and car in mob else ""
                w.writerow([g, car, f"{d['V0']:.3f}", f"{d['sigma']:.3f}", sens])
        w.writerow([])
        w.writerow(["qa", "qb", "qc", "branch", "freq_cm"] + [f"s2_{c}_meV2" for c in ("hole", "electron")])
        for iq, q in enumerate(mc["q"]):
            for b in range(mc["freq_cm"].shape[1]):
                w.writerow([f"{q[0]:.4f}", f"{q[1]:.4f}", f"{q[2]:.4f}", b, f"{mc['freq_cm'][iq, b]:.3f}"]
                           + [f"{sum(d[c]['s2'][iq, b] for d in mc['pairs'].values()):.5g}" for c in ("hole", "electron")])
    return path


def plot_spectrum(axes, mc, mob=None, carrier="hole", gids=None, width=4.0):
    """(a) 主要二聚体 σ² 的谱密度 (按频率高斯展宽)；(b) 各声子支去除后的 Δμ/μ，横坐标为该支按 σ² 加权的平均频率，
    颜色为全布里渊区平均的主导运动。"""
    f = mc["freq_cm"]
    act = mc["active"]
    x = np.linspace(0, max(f[act].max() * 1.05, 50), 600)
    ax = axes[0]
    gids = gids or sorted(mc["pairs"], key=lambda g: -mc["pairs"][g][carrier]["sigma"])[:4]
    for g in gids:
        s2 = mc["pairs"][g][carrier]["s2"][act]
        y = (s2[:, None] * np.exp(-0.5 * ((x[None] - f[act][:, None]) / width) ** 2)).sum(0) / (width * np.sqrt(2 * np.pi))
        ax.plot(x, y, label=f"G{g} (σ = {mc['pairs'][g][carrier]['sigma']:.1f} meV)")
    ax.set_xlabel("frequency (cm$^{-1}$)")
    ax.set_ylabel("dσ$^2$/dν (meV$^2$ cm)")
    ax.legend(fontsize=7, frameon=False)
    if mob is not None and len(axes) > 1:
        ax = axes[1]
        br = mc["branch"]
        dm = 100 * mob["dmu_branch"] / mob["mu"]
        cols = ["#1f5fae", "#4f8fd8", "#9cc3f0", "#c0392b", "#e07b6a", "#f2b8ad"]
        k = np.argmax(br["char_mean"], 1)
        tot = sum(d[carrier]["s2"] for d in mc["pairs"].values())
        fm = (tot * np.where(act, f, 0)).sum(0) / np.maximum(tot.sum(0), 1e-30)      # 各支按 σ² 加权的平均频率
        for i, key in enumerate(CHAR_KEYS):
            sel = k == i
            if sel.any():
                ax.bar(fm[sel], dm[sel], width=2.5, color=cols[i], label=CHAR_LABEL_EN[key])
        for b in np.argsort(dm)[::-1][:3]:
            ax.annotate(f"b{b}", (fm[b], dm[b]), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
        ax.set_xlabel("σ$^2$-weighted frequency of the branch (cm$^{-1}$)")
        ax.set_ylabel("Δμ/μ when branch removed (%)")
        ax.legend(fontsize=7, frameon=False)
    return axes


# ----------------------------------------------------------------------------------------------
# 外部 (DFT) 声子：phonopy 的 mesh.yaml / qpoints.yaml / band.yaml (含本征矢)
# ----------------------------------------------------------------------------------------------
THZ_TO_CM = 33.35640952
BOHR = 0.529177210903


def read_phonopy_yaml(path, structure=None):
    """读 phonopy 输出 (需要本征矢：phonopy 加 --eigvecs 或 EIGENVECTORS = .TRUE.)。
    返回 dict：lattice (3×3，行，原单位)、symbols、frac、masses、q (Nq,3 约化坐标)、freq_cm (Nq, nb)、
    eig (Nq, nb, nat, 3) 复数 (质量加权、归一)、weights。structure：没有 points 段时 (qpoints.yaml) 另给 phonopy.yaml。"""
    try:
        import yaml
    except ImportError as exc:                                        # pragma: no cover
        raise ImportError("读取 phonopy 文件需要 PyYAML：pip install pyyaml") from exc
    Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(path, encoding="utf-8") as fh:
        d = yaml.load(fh, Loader=Loader)
    st = d
    if "points" not in d and "atoms" not in d:
        if not structure:
            raise ValueError(f"{path} 中没有晶体结构 (points)：请同时给出 phonopy.yaml (--ph-structure)")
        with open(structure, encoding="utf-8") as fh:
            st = yaml.load(fh, Loader=Loader)
    if "points" in st:
        pts = st["points"]
        lat = np.array(st["lattice"], float)
    else:                                                              # phonopy.yaml: primitive_cell / unit_cell
        cell = st.get("primitive_cell") or st.get("unit_cell")
        pts = cell["points"]
        lat = np.array(cell["lattice"], float)
    symbols = [p["symbol"] for p in pts]
    frac = np.array([p["coordinates"] for p in pts], float)
    masses = np.array([p.get("mass", np.nan) for p in pts], float)
    phon = d.get("phonon")
    if not phon:
        raise ValueError(f"{path} 中没有 phonon 段")
    q, f, eig, wt = [], [], [], []
    for item in phon:
        bands = item["band"]
        if "eigenvector" not in bands[0]:
            raise ValueError(f"{path} 中没有本征矢：phonopy 计算时加 --eigvecs (或 EIGENVECTORS = .TRUE.)")
        q.append(item["q-position"])
        f.append([b["frequency"] for b in bands])
        ev = np.array([b["eigenvector"] for b in bands], float)        # (nb, nat, 3, 2)
        eig.append(ev[..., 0] + 1j * ev[..., 1])
        wt.append(item.get("weight", 1))
    return {"lattice": lat, "symbols": symbols, "frac": frac, "masses": masses, "q": np.array(q, float),
            "freq_cm": np.array(f, float) * THZ_TO_CM, "eig": np.array(eig), "weights": np.array(wt, float),
            "length_unit": (st.get("physical_unit", {}) or {}).get("length", "") if isinstance(st, dict) else ""}


def _assign(A, B, symbols_a, symbols_b, lat):
    """A、B 为分数坐标：按元素做最小周期距离的一一对应 (匈牙利算法)。返回 (idx_b, 整数平移 n (A ≈ B[idx] + n), 最大偏差 Å)。"""
    from scipy.optimize import linear_sum_assignment
    idx = np.full(len(A), -1)
    shift = np.zeros((len(A), 3))
    worst = 0.0
    for e in sorted(set(symbols_a)):
        ia = [i for i, s in enumerate(symbols_a) if s == e]
        ib = [i for i, s in enumerate(symbols_b) if s == e]
        if len(ia) > len(ib):
            raise ValueError(f"外部结构中元素 {e} 的原子数少于晶体 ({len(ib)} < {len(ia)})")
        d = A[ia][:, None, :] - B[ib][None, :, :]
        n = np.round(d)
        dist = np.linalg.norm((d - n) @ lat, axis=2)
        r, c = linear_sum_assignment(dist)
        for i, j in zip(r, c):
            idx[ia[i]] = ib[j]
            shift[ia[i]] = n[i, j]
            worst = max(worst, float(dist[i, j]))
    return idx, shift, worst


def external_lattice(md, ceds, ext, length_unit="auto", phase="positions", tol=0.8):
    """把外部声子 (read_phonopy_yaml 的结果) 换算成刚体分子的广义位移，返回与 RigidLattice 用法相同的对象
    (mode_couplings / character 可直接使用)，另含 info：单位、原子对应偏差、各支的刚体成分。
    要求外部计算与 CIF 用同一个晶胞 (可以是 DFT 优化后的晶胞，轴的取向可以不同)。
    phase: "positions" = phonopy 的约定 (相位含原子分数坐标 exp(2πi q·(l + x_k)))；"cell" = 只含晶胞平移。"""
    from .rigidmd import MASS
    lat_c = np.asarray(ceds.crys["lattice"], float)
    lat_e = np.asarray(ext["lattice"], float)
    unit = length_unit
    if unit == "auto":
        r = np.linalg.norm(lat_e, axis=1) / np.linalg.norm(lat_c, axis=1)
        unit = "bohr" if np.all(np.abs(r / (1 / BOHR) - 1) < 0.08) else "angstrom"
    if unit in ("bohr", "au"):
        lat_e = lat_e * BOHR
    pa = np.linalg.norm(lat_e, axis=1) / np.linalg.norm(lat_c, axis=1)
    if np.abs(pa - 1).max() > 0.06:
        raise ValueError(f"外部声子的晶胞与 CIF 不同 (晶格矢量长度之比 {np.round(pa, 3).tolist()})：请用 CIF 的晶胞 "
                         "(相同的 a、b、c 设置) 做声子计算，或检查长度单位 (--ph-length-unit)")
    sym_e = ext["symbols"]
    m_e = ext["masses"]
    if np.isnan(m_e).any():
        m_e = np.array([MASS[s] for s in sym_e])
    nat = len(sym_e)
    # CIF 中每个分子的原子 → 外部结构的原子
    inv_c = np.linalg.inv(lat_c)
    groups, worst = [], 0.0
    for k, mol in enumerate(ceds.mols):
        fa = np.asarray(mol["cart"], float) @ inv_c
        idx, n, w = _assign(fa, ext["frac"], list(mol["symbols"]), sym_e, lat_c)
        worst = max(worst, w)
        groups.append((idx, ext["frac"][idx] + n, np.asarray(mol["cart"], float)))
    used = np.concatenate([g[0] for g in groups])
    if len(set(used.tolist())) != len(used) or len(used) != nat:
        raise ValueError(f"外部结构的原子 ({nat} 个) 与 CIF 晶胞中的分子 ({len(used)} 个原子) 不能一一对应")
    if worst > tol:
        raise ValueError(f"外部结构与 CIF 的原子位置相差 {worst:.2f} Å (> {tol} Å)：请确认是同一晶胞、同一原点")
    # 外部笛卡尔位移 → 分数 → CIF 笛卡尔
    T = np.linalg.inv(lat_e) @ lat_c
    qs = np.asarray(ext["q"], float)
    eig = ext["eig"]                                                   # (Nq, nb, nat, 3)
    Nq, nb = eig.shape[:2]
    eps = np.zeros((Nq, 6 * md.nk, nb), complex)
    rigid = np.zeros((Nq, nb))
    tot = np.zeros((Nq, nb))
    for k, (idx, xf, cart) in enumerate(groups):
        n0 = md.index((0, 0, 0), k)
        m = m_e[idx]
        r = cart - md.c0[n0]
        ph = np.exp(2j * np.pi * (xf @ qs.T)) if phase == "positions" else \
            np.exp(2j * np.pi * (np.floor(xf + 1e-9) @ qs.T))           # (natm, Nq)
        u = eig[:, :, idx, :] / np.sqrt(m)[None, None, :, None]          # (Nq, nb, natm, 3) 外部笛卡尔
        u = (u @ T) * ph.T[:, None, :, None]                            # CIF 笛卡尔，含相位
        # 质量加权最小二乘：u_i ≈ t + θ × r_i (绕 md 的刚体质心)
        A = np.zeros((len(m), 3, 6))
        A[:, :, :3] = np.eye(3)
        for i, ri in enumerate(r):
            A[i, :, 3:] = -np.array([[0, -ri[2], ri[1]], [ri[2], 0, -ri[0]], [-ri[1], ri[0], 0]])
        W = np.repeat(m, 3)
        Am = A.reshape(-1, 6)
        G = Am.T @ (W[:, None] * Am)
        P = np.linalg.solve(G, Am.T * W[None, :])                       # (6, 3 natm)
        x = np.einsum("pj,qbj->qbp", P, u.reshape(Nq, nb, -1))
        eps[:, 6 * k:6 * k + 6, :] = x.transpose(0, 2, 1)
        fit = np.einsum("jp,qbp->qbj", Am, x)
        rigid += np.einsum("j,qbj->qb", W, np.abs(fit) ** 2)
        tot += np.einsum("j,qbj->qb", W, np.abs(u.reshape(Nq, nb, -1)) ** 2)
    om = ext["freq_cm"] * (2 * np.pi * C_CM_FS)
    fcm = ext["freq_cm"].copy()
    ig = np.where(np.linalg.norm(qs - np.round(qs), axis=1) < 1e-8)[0]
    for i in ig:                                                       # Γ 点的三个声学模
        ac = np.argsort(np.abs(fcm[i]))[:3]
        fcm[i, ac] = 0.0
        om[i, ac] = 0.0
    lat = RigidLattice(md)
    lat.mode = {"q": qs, "omega": om, "freq_cm": fcm, "eps": eps}
    frac_rigid = rigid / np.maximum(tot, 1e-30)
    lat.info = {"length_unit": unit, "map_dev_A": worst, "n_q": Nq, "n_branch": nb,
                "rigid_fraction": frac_rigid, "lattice_ratio": pa.tolist()}
    if len(set(np.round(ext["weights"], 6))) > 1:
        raise ValueError("外部 q 网格做了对称性约化 (各 q 点权重不同)：请用 MESH_SYMMETRY = .FALSE. (phonopy --nomeshsym) "
                         "输出完整网格")
    return lat


def gamma_subset(lat):
    """只保留 Γ 点的同一套声子 (外部声子或刚体晶格)，用于“只用 Γ 点”的对照。"""
    qs = lat.mode["q"]
    ig = np.where(np.linalg.norm(qs - np.round(qs), axis=1) < 1e-8)[0]
    if not len(ig):
        raise ValueError("q 网格中没有 Γ 点")
    sub = RigidLattice(lat.md)
    sub.c, sub.Q = lat.c, lat.Q
    sub.mode = {k: v[ig] for k, v in lat.mode.items()}
    if hasattr(lat, "info"):
        sub.info = dict(lat.info, rigid_fraction=lat.info["rigid_fraction"][ig])
    return sub
