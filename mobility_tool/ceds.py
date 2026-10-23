"""CEDS：耦合–能量双曲面法 (Coupling–Energy Dual Surface)。由耦合曲面与刚体涨落求动态无序 σ_V 与迁移率 μ(T)。

思路
  1. 晶体局域势：晶体中二聚体 (A, B) 两个分子作刚体运动 (各 3 个平动 + 3 个转动)，其余近邻分子固定在晶体位置。
     势能 U = E(A, 环境) + E(B, 环境) + E(A, B)，分子间对势默认用 UFF (Lennard-Jones，含色散，kcal/mol)。
     在 U 上做 Metropolis 蒙特卡洛 (温度 T)，得到 A、B 刚体位移的平衡分布 (非谐，含 A–B 相关)。
     平均结构取实验晶体结构：采样的位移减去其平均值后加到晶体几何上 (势能函数只决定涨落)。
  2. 耦合曲面：每个采样构型换算为二聚体参数 q = (z, Δx, Δy, γ, θ, φ) (A 的分子坐标系，与扫描完全相同的约定)，
     由耦合曲面给出 V(q)。曲面来源：
       local      该二聚体附近的 DFT 扫描数据 (局域扫描) 拟合的二次多项式；
       separable  共面网格的可分离模型 (z, Δx, Δy；转角冻结)；
       eht        软件内 EHT 直接计算 (无需 DFT，用于预览与检验曲面精度)。
  3. 由样本得到 ⟨V⟩、σ_V、σ_V/|⟨V⟩| 以及 σ_V² 在各参数上的分配 (线性灵敏度，确定主导涨落的运动)。
  4. 位点轨道相位：每个分子固定一套分子坐标系 (晶体中的取向)，B 的坐标系与扫描区域相差 B 自身的二重轴 g 时，
     V_site = χ(g)·V(q·g)，χ 为前线轨道在 g 下的特征标 (EHT 或 DFT 对称检验给出)。这样 TLT 中各键的符号一致。
  5. 瞬态局域化理论 (TLT)：H_ij = ⟨V⟩_ij + δ_ij，δ ~ N(0, σ_V,ij)，逐温度给出 μ(T)。

近似与局限
  · 刚体近似：分子内振动对 σ_V 的贡献未计入 (低频分子间运动通常占主导)。
  · 局域 Einstein 近似：只有二聚体两个分子运动，环境固定；长波声学模的集体运动未计入。
    rigidmd 模块的全晶体刚体 MD (所有分子同时运动) 用于检验这一近似。
  · UFF 对势较粗糙：用晶体学原子位移参数 (U_eq) 检验涨落幅度 (本模块输出 U_eq 供对照)，必要时缩放势能。
"""
import math
import time

import numpy as np

from .scanflow import KCAL_R, OPS, PARAM_KEYS, PROPER, angles_from_R, apply_op, make_cfg, molecule_ops
from .surrogate import CARRIERS, build_B, molecular_frame, rotation

ANG_KEYS = ("gamma", "pitch", "roll")
DEFAULT_HALF = {"z": 0.5, "dx": 0.8, "dy": 0.8, "gamma": 10.0, "pitch": 10.0, "roll": 10.0}


# ----------------------------------------------------------------------------------------------
# 几何
# ----------------------------------------------------------------------------------------------
def rotvec_matrix(w):
    """转动矢量 (弧度) → 转动矩阵 (Rodrigues)。w: (..., 3)。"""
    w = np.asarray(w, float)
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = w[..., 2], -w[..., 1], w[..., 0]
    I = np.broadcast_to(np.eye(3), K.shape)
    small = th < 1e-12
    th_ = np.where(small, 1.0, th)
    A = np.where(small, 1.0, np.sin(th_) / th_)
    B = np.where(small, 0.5, (1 - np.cos(th_)) / th_ ** 2)
    return I + A * K + B * (K @ K)


def pair_params(axA, cA, axB, cB):
    """两个分子坐标系 (列为轴) → 二聚体参数 (z, Δx, Δy, γ, θ, φ)。axA/axB 可带批量维度。"""
    axA, axB = np.asarray(axA, float), np.asarray(axB, float)
    R = np.swapaxes(axA, -1, -2) @ axB
    t = np.einsum("...ji,...j->...i", axA, np.asarray(cB, float) - np.asarray(cA, float))
    R2 = R.reshape(-1, 3, 3)
    ang = np.array([angles_from_R(r) for r in R2]).reshape(R.shape[:-2] + (3,))
    return np.concatenate([t[..., 2:3], t[..., 0:1], t[..., 1:2], ang], axis=-1)


def choose_g(axA, axB, ops):
    """B 自身的二重轴 g (含 E)，使 R·g 最接近单位矩阵 (与 ScanProject.crystal_configs 相同的取法)。"""
    best = None
    for op in ["E"] + [o for o in PROPER if o in ops]:
        R = axA.T @ (axB * np.array(OPS[op], float))
        ang = math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        if best is None or ang < best[0] - 1e-9:
            best = (ang, op)
    return best[1]


def reverse_cfg(c):
    """二聚体换参考分子 (B 作为 A)：R′ = Rᵀ，t′ = −Rᵀ t。"""
    R = rotation(c["gamma"], c["roll"], c["pitch"])
    t = -R.T @ np.array([c["dx"], c["dy"], c["z"]], float)
    g, p, r = angles_from_R(R.T)
    return make_cfg(t[2], t[0], t[1], gamma=g, pitch=p, roll=r)


def _wrap(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------------------------------
# 晶体近邻图 (全部独立分子为参考，按距离指纹合并为唯一二聚体)
# ----------------------------------------------------------------------------------------------
def crystal_pairs(crys, mols, cutoff=4.0, criterion="atom", fp_tol=0.03):
    """返回 (pairs, edges)。pairs: 唯一二聚体 [{gid, ref, mol, translation, r_centroid, d_min, multiplicity}]；
    edges: 全部有向近邻 [(i, j, 矢量, translation, gid)]。"""
    from .crystal import find_dimers
    pairs, edges = [], []
    for ref in range(len(mols)):
        res = find_dimers(crys, mols, ref=ref, cutoff=cutoff, criterion=criterion)
        for d in res["dimers"]:
            gid = None
            for p in pairs:
                fp = p["fingerprint"]
                if len(fp) == len(d["fingerprint"]) and np.abs(fp - d["fingerprint"]).max() < fp_tol:
                    gid = p["gid"]
                    p["multiplicity"] += 1
                    break
            if gid is None:
                gid = len(pairs) + 1
                pairs.append({"gid": gid, "ref": ref, "mol": d["mol"], "translation": tuple(d["translation"]),
                              "r_centroid": d["r_centroid"], "d_min": d["d_min"], "fingerprint": d["fingerprint"],
                              "multiplicity": 1, "vector": np.asarray(d["vector"], float)})
            edges.append((ref, d["mol"], np.asarray(d["vector"], float), tuple(d["translation"]), gid))
    return pairs, edges


def _mol_xyz(crys, mols, m, t):
    return np.asarray(mols[m]["cart"], float) + np.asarray(t, float) @ np.asarray(crys["lattice"], float)


# ----------------------------------------------------------------------------------------------
# 晶体局域势与蒙特卡洛
# ----------------------------------------------------------------------------------------------
class PairPotential:
    """UFF Lennard-Jones (kcal/mol)，可选原子电荷 (库仑) 与整体缩放。"""

    def __init__(self, scale=1.0, charges=None):
        from .interaction import UFF
        self.UFF, self.scale, self.charges = UFF, float(scale), charges

    def params(self, symbols):
        miss = sorted({e for e in symbols if e not in self.UFF})
        if miss:
            raise ValueError(f"缺少 UFF 参数的元素: {miss}")
        p = np.array([self.UFF[e] for e in symbols], float)
        return p[:, 0], p[:, 1]

    def table(self, sym_i, sym_j):
        xi, di = self.params(sym_i)
        xj, dj = self.params(sym_j)
        return xi[:, None] * xj[None], np.sqrt(di[:, None] * dj[None]) * self.scale

    @staticmethod
    def energy(Xi, Xj, x2, d):
        """x2 = x² (预先平方)。"""
        r2 = (Xi * Xi).sum(1)[:, None] + (Xj * Xj).sum(1)[None, :] - 2.0 * (Xi @ Xj.T)
        q6 = (x2 / r2) ** 3
        return float((d * (q6 * q6 - 2 * q6)).sum())


class DimerMC:
    """二聚体 (A, B) 在晶体环境中的刚体 Metropolis 蒙特卡洛。"""

    def __init__(self, crys, mols, ref, mol, translation, potential=None, env_cutoff=8.0):
        self.pot = potential or PairPotential()
        self.sym = [list(mols[ref]["symbols"]), list(mols[mol]["symbols"])]
        self.X0 = [_mol_xyz(crys, mols, ref, (0, 0, 0)), _mol_xyz(crys, mols, mol, translation)]
        self.c0 = [x[[s != "H" for s in sy]].mean(0) for x, sy in zip(self.X0, self.sym)]
        mobile = {(ref, (0, 0, 0)), (mol, tuple(int(v) for v in translation))}
        env_x, env_s = [], []
        rng = range(-2, 3)
        for m in range(len(mols)):
            for t in ((i, j, k) for i in rng for j in rng for k in rng):
                if (m, t) in mobile:
                    continue
                X = _mol_xyz(crys, mols, m, t)
                dmin = min(np.sqrt(((X[:, None] - Y[None]) ** 2).sum(-1)).min() for Y in self.X0)
                if dmin < env_cutoff:
                    env_x.append(X)
                    env_s += list(mols[m]["symbols"])
        self.n_env = len(env_x)
        env = np.vstack(env_x) if env_x else np.zeros((0, 3))
        env_s = np.array(env_s)
        self.env, self.tab_env = [], []
        for k in range(2):                                     # 每个运动分子只保留 10 Å 内的环境原子
            if not len(env):
                self.env.append(env)
                self.tab_env.append(None)
                continue
            dmin = np.sqrt(((env[:, None] - self.X0[k][None]) ** 2).sum(-1)).min(1)
            sel = dmin < 10.0
            self.env.append(env[sel])
            self.tab_env.append(self.pot.table(self.sym[k], list(env_s[sel])))
        self.tab_ab = self.pot.table(self.sym[0], self.sym[1])

    def coords(self, k, t, w):
        Q = rotvec_matrix(w)
        return (self.X0[k] - self.c0[k]) @ Q.T + self.c0[k] + t

    def _e_env(self, k, X):
        if self.tab_env[k] is None:
            return 0.0
        return self.pot.energy(X, self.env[k], *self.tab_env[k])

    def run(self, T=300.0, n_samples=2000, thin=10, burn=3000, seed=1, step_t=0.03, step_r_deg=0.6, on_progress=None):
        """返回 {'t': (n, 2, 3) Å, 'w': (n, 2, 3) rad, 'E': (n,) kcal/mol, 'acc', 'step_t', 'step_r'}。"""
        rng = np.random.default_rng(seed)
        kT = KCAL_R * T
        st = [np.zeros(3), np.zeros(3)]
        sw = [np.zeros(3), np.zeros(3)]
        X = [self.X0[0].copy(), self.X0[1].copy()]
        e_env = [self._e_env(0, X[0]), self._e_env(1, X[1])]
        e_ab = self.pot.energy(X[0], X[1], *self.tab_ab)
        dt, dr = float(step_t), math.radians(step_r_deg)
        n_tot = burn + n_samples * thin
        out_t, out_w, out_E = [], [], []
        acc_win, acc_all = [], []
        for it in range(n_tot):
            k = it % 2
            t_new = st[k] + rng.normal(0, dt, 3)
            w_new = sw[k] + rng.normal(0, dr, 3)
            Xn = self.coords(k, t_new, w_new)
            en = self._e_env(k, Xn)
            eab = self.pot.energy(Xn, X[1], *self.tab_ab) if k == 0 else self.pot.energy(X[0], Xn, *self.tab_ab)
            dE = (en - e_env[k]) + (eab - e_ab)
            ok = dE <= 0 or rng.random() < math.exp(-dE / kT)
            if ok:
                st[k], sw[k], X[k], e_env[k], e_ab = t_new, w_new, Xn, en, eab
            acc_win.append(ok)
            if it < burn and len(acc_win) == 200:              # 预平衡阶段调整步长，接受率 ≈ 35 %
                a = np.mean(acc_win)
                f = min(max(a / 0.35, 0.5), 2.0)
                dt, dr = dt * f, dr * f
                acc_win = []
            if it >= burn:
                acc_all.append(ok)
                if (it - burn) % thin == thin - 1:
                    out_t.append([st[0].copy(), st[1].copy()])
                    out_w.append([sw[0].copy(), sw[1].copy()])
                    out_E.append(e_env[0] + e_env[1] + e_ab)
            if on_progress and it % 2000 == 0:
                on_progress(it, n_tot)
        return {"t": np.array(out_t), "w": np.array(out_w), "E": np.array(out_E), "acc": float(np.mean(acc_all)),
                "step_t": dt, "step_r": math.degrees(dr), "T": T, "n_env": self.n_env}

    def u_eq(self, mc, recenter=True):
        """分子 A、B 各原子均方位移的平均 U_eq = <|Δr|²>/3 (Å²)，与晶体学原子位移参数对照。"""
        out = []
        for k in range(2):
            t, w = mc["t"][:, k], mc["w"][:, k]
            if recenter:
                t, w = t - t.mean(0), w - w.mean(0)
            Q = rotvec_matrix(w)
            Y = self.X0[k] - self.c0[k]
            d = np.einsum("nij,aj->nai", Q, Y) - Y[None] + t[:, None, :]
            heavy = np.array([s != "H" for s in self.sym[k]])
            out.append(float((d[:, heavy] ** 2).sum(-1).mean() / 3))
        return out


# ----------------------------------------------------------------------------------------------
# 耦合曲面
# ----------------------------------------------------------------------------------------------
def _quad_features(D, kind):
    n, m = D.shape
    cols = [np.ones(n)] + [D[:, i] for i in range(m)]
    if kind in ("quadratic", "diag"):
        cols += [D[:, i] ** 2 for i in range(m)]
    if kind in ("quadratic", "cubic"):
        cols += [D[:, i] * D[:, j] for i in range(m) for j in range(i + 1, m)]
    if kind == "cubic":
        import itertools
        cols += [D[:, i] * D[:, j] * D[:, k] for i, j, k in itertools.combinations_with_replacement(range(m), 3)]
    return np.column_stack(cols)


def _n_terms(m, kind):
    from math import comb
    return {"linear": 1 + m, "diag": 1 + 2 * m, "quadratic": comb(m + 2, 2), "cubic": comb(m + 3, 3)}[kind]


class LocalSurface:
    """局域多项式曲面：在中心 q0 附近 (|Δq_k| ≤ h_k) 的扫描数据上拟合 V(q)。
    记录先按单体点群的整体操作展开 (V 含符号不变)，因此任一对称等价区域的数据都可用。"""

    source = "local"

    def __init__(self, records, center, carriers=CARRIERS, half=None, ops=("E",), min_points=12, reverse=True, degree=2):
        self.half = dict(DEFAULT_HALF, **(half or {}))
        self.center = np.array([center[k] for k in PARAM_KEYS], float)
        rows = []
        for r in records:
            if any(r.get(c) is None for c in carriers):
                continue
            views = [r]
            if reverse:                       # 从 B 看 A：R′ = Rᵀ，t′ = −Rᵀt；同一片段文件下 V_BA = V_AB
                views.append(reverse_cfg(r))
            for v in views:
                for op in ops:
                    q = apply_op(v, op) if op != "E" else v
                    rows.append([q[k] for k in PARAM_KEYS] + [r[c] for c in carriers])
        if not rows:
            raise ValueError("没有可用的扫描数据")
        A = np.array(rows, float)
        D = self._scaled(A[:, :6])
        inside = (np.abs(D) <= 1.0 + 1e-6).all(1)
        A, D = A[inside], D[inside]
        keep = np.unique(np.round(A[:, :6], 5), axis=0, return_index=True)[1]
        A, D = A[keep], D[keep]
        if len(A) < min_points:
            raise ValueError(f"中心附近只有 {len(A)} 个扫描构型 (至少需要 {min_points} 个)")
        self.active = [i for i in range(6) if np.ptp(D[:, i]) > 1e-3]
        self.frozen = [PARAM_KEYS[i] for i in range(6) if i not in self.active]
        Da = D[:, self.active]
        m = len(self.active)
        self.kind = next(k for k in ("cubic", "quadratic", "diag", "linear")
                         if (k != "cubic" or degree >= 3) and len(A) >= 1.4 * _n_terms(m, k)) \
            if len(A) >= 1.4 * _n_terms(m, "linear") else "linear"
        F = _quad_features(Da, self.kind)
        self.coef, self.stats = {}, {}
        for j, c in enumerate(carriers):
            y = A[:, 6 + j]
            lam = 1e-8 * len(y)
            coef = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ y)
            H = F @ np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T)
            res = y - F @ coef
            loo = res / np.clip(1 - np.diag(H), 1e-6, None)
            ss = ((y - y.mean()) ** 2).sum()
            self.coef[c] = coef
            self.stats[c] = {"n": len(y), "rmse": float(np.sqrt((res ** 2).mean())),
                             "loo_rmse": float(np.sqrt((loo ** 2).mean())),
                             "R2": float(1 - (res ** 2).sum() / ss) if ss > 0 else 1.0, "V_range": (float(y.min()), float(y.max()))}
        self.n_points = len(A)
        self.data_range = (D[:, self.active].min(0), D[:, self.active].max(0))

    def _scaled(self, Q):
        D = np.asarray(Q, float) - self.center
        for i, k in enumerate(PARAM_KEYS):
            if k in ANG_KEYS:
                D[:, i] = _wrap(D[:, i])
            D[:, i] /= self.half[k]
        return D

    def __call__(self, Q, carrier):
        D = self._scaled(np.atleast_2d(Q))[:, self.active]
        return _quad_features(D, self.kind) @ self.coef[carrier]

    def coverage(self, Q):
        D = self._scaled(np.atleast_2d(Q))[:, self.active]
        lo, hi = self.data_range
        return float(((D >= lo - 1e-9) & (D <= hi + 1e-9)).all(1).mean())

    def describe(self):
        s = "; ".join(f"{c}: n={v['n']}, LOO-RMSE {v['loo_rmse']:.2f} meV, R² {v['R2']:.4f}" for c, v in self.stats.items())
        fz = f"; 冻结参数 {', '.join(self.frozen)}" if self.frozen else ""
        return f"局域{ {'cubic': '三次', 'quadratic': '二次', 'diag': '对角二次', 'linear': '线性'}[self.kind] }多项式 ({s}{fz})"


class SeparableSurface:
    """共面网格的可分离模型 (z, Δx, Δy)，三个转角冻结在 0。"""

    source = "separable"

    def __init__(self, models):
        self.models = models
        self.frozen = list(ANG_KEYS)

    def __call__(self, Q, carrier):
        Q = np.atleast_2d(Q)
        return np.array([self.models.predict(carrier, "M2i", q[0], q[1], q[2]) for q in Q])

    def coverage(self, Q):
        lo, hi = self.models.range["z"]
        Q = np.atleast_2d(Q)
        return float(((Q[:, 0] >= lo - 0.3) & (Q[:, 0] <= hi + 0.3)).mean())

    def describe(self):
        return "共面网格可分离模型 (转角冻结)"


class EHTSurface:
    """软件内 EHT：由参数 q 构造二聚体 (A 为分子坐标系中的单体) 直接计算 V (meV)。"""

    source = "eht"
    frozen = []

    def __init__(self, symbols, Y):
        from .eht import monomer
        self.sym, self.Y = list(symbols), np.asarray(Y, float)
        self.mono = monomer(self.sym, self.Y)
        self.A = [(s, *p) for s, p in zip(self.sym, self.Y)]
        self._cache = {}

    def eval_one(self, q):
        from .eht import build_basis, eht_transfer_integrals, rotate_coefficients
        key = tuple(np.round(q, 6))
        if key in self._cache:
            return self._cache[key]
        z, dx, dy, g, p, r = q
        R = rotation(g, r, p)
        B = build_B(self.Y, z, dx, dy, g, r, p)
        shells, _, _ = build_basis(self.sym, B)
        mb = dict(self.mono, shells=shells, C=rotate_coefficients(self.mono["shells"], self.mono["C"], R))
        res = eht_transfer_integrals(self.A, [(s, *x) for s, x in zip(self.sym, B)], esd=False, _mono_b=mb)["results"]
        out = {"hole": res["HOMO"]["J_eff_meV"], "electron": res["LUMO"]["J_eff_meV"]}
        self._cache[key] = out
        return out

    def __call__(self, Q, carrier):
        return np.array([self.eval_one(q)[carrier] for q in np.atleast_2d(Q)])

    def coverage(self, Q):
        return 1.0

    def describe(self):
        return "软件内 EHT 直接计算"


def eht_characters(symbols, Y, ops, base=(3.6, 0.8, 0.4, 0.0, 0.0, 0.0)):
    """前线轨道在 B 自身二重轴下的特征标 χ(g) (由 EHT：V(R·g)/V(R) 的符号)。"""
    surf = EHTSurface(symbols, Y)
    q0 = np.array(base, float)
    v0 = surf.eval_one(q0)
    out = {c: {"E": 1} for c in CARRIERS}
    for op in PROPER:
        if op not in ops:
            continue
        G = np.diag(OPS[op]).astype(float)
        g, p, r = angles_from_R(rotation(q0[3], q0[5], q0[4]) @ G)
        v = surf.eval_one(np.array([q0[0], q0[1], q0[2], g, p, r]))
        for c in CARRIERS:
            out[c][op] = int(np.sign(v[c] / v0[c])) if abs(v0[c]) > 1e-3 and abs(abs(v[c]) - abs(v0[c])) < 0.05 * abs(v0[c]) + 0.05 else 0
    return out


# ----------------------------------------------------------------------------------------------
# CEDS 主流程
# ----------------------------------------------------------------------------------------------
class CEDS:
    """crys/mols: 晶体 (crystal.read_cif / find_molecules)；monomer: (symbols, Y) 分子坐标系中的冻结单体
    (默认取晶体中的第一个分子)；surfaces: {gid: 曲面} 或 callable(gid, center) → 曲面；characters: χ 表。"""

    def __init__(self, crys, mols, cutoff=4.0, criterion="atom", monomer=None, potential=None, characters=None,
                 env_cutoff=8.0, frame="all"):
        from .surrogate import frame_atoms, matched_frame
        self.crys, self.mols, self.frame_mode = crys, mols, frame
        self.pairs, self.edges = crystal_pairs(crys, mols, cutoff, criterion)
        if monomer is None:
            s0, x0 = mols[0]["symbols"], mols[0]["cart"]
            Y = molecular_frame(s0, x0, frame_atoms(s0, x0, frame))[0]
            monomer = (list(s0), Y)
        self.sym, self.Y = monomer
        hv = [i for i, e in enumerate(self.sym) if e != "H"]          # X 射线 H 位置偏差大：只用重原子判断点群
        self.ops = molecule_ops([self.sym[i] for i in hv], np.asarray(self.Y)[hv], tol=0.12)
        self.pot = potential or PairPotential()
        self.env_cutoff = env_cutoff
        self.frames, self.frame_rmsd = [], []
        for m in mols:                                  # 各分子的坐标系与参考单体逐原子匹配 (见 matched_frame)
            ax, c, r = matched_frame(self.sym, self.Y, m["symbols"], m["cart"], frame)
            self.frames.append((ax, c))
            self.frame_rmsd.append(r)
        self.notes = []
        if max(self.frame_rmsd) > 0.5:
            self.notes.append(f"单体与晶体中的分子差别较大 (重原子 RMSD 最大 {max(self.frame_rmsd):.2f} Å)：请检查单体是否与晶体"
                              "中的构象一致 (例如取代基的扭转方向)")
        self.characters = characters or eht_characters(self.sym, self.Y, self.ops)
        for p in self.pairs:
            axA, cA = self.frames[p["ref"]]
            axB, cB0 = self.frames[p["mol"]]
            cB = cB0 + np.asarray(p["translation"], float) @ np.asarray(crys["lattice"], float)
            p["g"] = choose_g(axA, axB, self.ops)
            G = np.diag(OPS[p["g"]]).astype(float)
            q = pair_params(axA, cA, axB @ G, cB)
            p["center"] = make_cfg(*q[:3], gamma=q[3], pitch=q[4], roll=q[5])
            p["q0"] = q

    def calibrate(self, target_ueq, T=300.0, n_samples=800, gid=None, iterations=2, seed=11):
        """按晶体学 U_eq (重原子平均，Å²) 缩放对势：谐振近似下 <u²> ∝ kT/势能曲率，故 scale ← scale·U_model/U_target。
        返回 [(scale, U_model)]。只缩放涨落幅度，平均结构仍取实验结构。"""
        p = next((q for q in self.pairs if q["gid"] == gid), self.pairs[0])
        hist = []
        for _ in range(iterations):
            sim = DimerMC(self.crys, self.mols, p["ref"], p["mol"], p["translation"], self.pot, self.env_cutoff)
            mc = sim.run(T, n_samples, 10, 3000, seed)
            u = float(np.mean(sim.u_eq(mc)))
            hist.append((self.pot.scale, u))
            self.pot.scale *= u / float(target_ueq)
        sim = DimerMC(self.crys, self.mols, p["ref"], p["mol"], p["translation"], self.pot, self.env_cutoff)
        mc = sim.run(T, n_samples, 10, 3000, seed)
        hist.append((self.pot.scale, float(np.mean(sim.u_eq(mc)))))
        return hist

    def suggest_half(self, T=300.0, n_samples=600, k=2.5, vmin=3.0, seed=5, floor=None):
        """先做 MC (不需要 DFT)，按各参数涨落的 ±k·σ 确定每个二聚体局域扫描的范围。
        |V0| (EHT 估计) < vmin 的弱耦合二聚体不需要局域扫描。返回 {组号: {参数: 半宽}}。"""
        floor = dict({"z": 0.12, "dx": 0.15, "dy": 0.15, "gamma": 2.0, "pitch": 2.0, "roll": 2.0}, **(floor or {}))
        eht = EHTSurface(self.sym, self.Y)
        out = {}
        for p in self.pairs:
            if max(abs(eht(p["q0"], c)[0]) for c in CARRIERS) < vmin:
                continue
            sim = DimerMC(self.crys, self.mols, p["ref"], p["mol"], p["translation"], self.pot, self.env_cutoff)
            mc = sim.run(T, n_samples, 5, 2000, seed + p["gid"])
            sd = self.sample_params(p, mc).std(0)
            out[p["gid"]] = {key: float(max(k * sd[i], floor[key])) for i, key in enumerate(PARAM_KEYS)}
        return out

    def chi(self, carrier, g):
        return self.characters.get(carrier, {}).get(g, 1) or 1

    def centers(self):
        """每个唯一二聚体在冻结单体参数下的中心构型 (用于生成局域扫描)。"""
        return {p["gid"]: dict(p["center"], r_centroid=p["r_centroid"], g=p["g"]) for p in self.pairs}

    def sample_params(self, p, mc, recenter=True):
        """MC 样本 → 每个样本的二聚体参数 q (n, 6) (B 坐标系已乘 g)。"""
        lat = np.asarray(self.crys["lattice"], float)
        axA, cA = self.frames[p["ref"]]
        axB, cB0 = self.frames[p["mol"]]
        cB = cB0 + np.asarray(p["translation"], float) @ lat
        G = np.diag(OPS[p["g"]]).astype(float)
        t, w = mc["t"], mc["w"]
        if recenter:
            t, w = t - t.mean(0), w - w.mean(0)
        QA, QB = rotvec_matrix(w[:, 0]), rotvec_matrix(w[:, 1])
        # 刚体转动绕各自分子坐标系的原点
        Q = pair_params(QA @ axA, cA + t[:, 0], QB @ (axB @ G), cB + t[:, 1])
        q0 = np.asarray(p["q0"], float)
        Q[:, 3:] = q0[3:] + _wrap(Q[:, 3:] - q0[3:])          # 角度在中心附近连续展开 (中心在 ±180° 附近时)
        return Q

    def run_pair(self, p, surface, T=300.0, n_samples=2000, thin=10, burn=3000, seed=1, recenter=True, mc=None,
                 on_progress=None, max_eval=400):
        if mc is None:
            sim = DimerMC(self.crys, self.mols, p["ref"], p["mol"], p["translation"], self.pot, self.env_cutoff)
            mc = sim.run(T, n_samples, thin, burn, seed, on_progress=on_progress)
            mc["U_eq"] = sim.u_eq(mc, recenter)
        Q = self.sample_params(p, mc, recenter)
        if surface.source == "eht" and len(Q) > max_eval:          # EHT 逐点计算较慢：随机取 max_eval 个样本
            Q = Q[np.random.default_rng(seed).choice(len(Q), max_eval, replace=False)]
        out = {"gid": p["gid"], "T": T, "mc": mc, "q": Q, "source": surface.source, "surface": surface.describe(),
               "frozen": list(getattr(surface, "frozen", [])), "coverage": surface.coverage(Q),
               "q_std": dict(zip(PARAM_KEYS, Q.std(0))), "q0": p["q0"]}
        for c in CARRIERS:
            chi = self.chi(c, p["g"])
            V = chi * surface(Q, c)
            V0 = float(chi * surface(p["q0"], c)[0])
            m, s = float(V.mean()), float(V.std(ddof=1))
            # 线性灵敏度：V ≈ V̄ + Σ g_k δq_k，σ² 的分配 (未冻结的参数)
            act = [i for i, k in enumerate(PARAM_KEYS) if k not in out["frozen"] and Q[:, i].std() > 1e-9]
            share = {}
            if act:
                dQ = Q[:, act] - Q[:, act].mean(0)
                gk, *_ = np.linalg.lstsq(dQ, V - m, rcond=None)
                contrib = gk ** 2 * dQ.var(0)
                tot = contrib.sum() or 1.0
                share = {PARAM_KEYS[i]: float(contrib[j] / tot) for j, i in enumerate(act)}
                out.setdefault("grad", {})[c] = {PARAM_KEYS[i]: float(gk[j]) for j, i in enumerate(act)}
            out[c] = {"V0": V0, "mean": m, "std": s, "ratio": s / max(abs(m), 1e-9), "samples": V, "share": share,
                      "chi": chi}
        return out

    def run(self, surfaces, T=300.0, n_samples=2000, thin=10, burn=None, seed=1, vmin=3.0, recenter=True,
            on_progress=None):
        """对全部唯一二聚体运行。|V0| < vmin (两种载流子) 的二聚体只算中心值 (σ = 0)。
        burn 默认取 max(1500, 采样步数的 1/4)。"""
        burn = burn if burn is not None else max(1500, n_samples * thin // 4)
        res = {}
        for k, p in enumerate(self.pairs):
            surf = surfaces(p["gid"], p["center"]) if callable(surfaces) else surfaces.get(p["gid"])
            if surf is None:
                res[p["gid"]] = {"gid": p["gid"], "skipped": "没有该二聚体的耦合曲面"}
                continue
            v0 = {c: float(self.chi(c, p["g"]) * surf(p["q0"], c)[0]) for c in CARRIERS}
            if max(abs(v) for v in v0.values()) < vmin:
                res[p["gid"]] = {"gid": p["gid"], "T": T, "source": surf.source, "weak": True,
                                 **{c: {"V0": v0[c], "mean": v0[c], "std": 0.0, "ratio": 0.0, "samples": np.array([v0[c]]),
                                        "share": {}, "chi": self.chi(c, p["g"])} for c in CARRIERS}}
                continue
            if on_progress:
                on_progress(f"G{p['gid']} ({k + 1}/{len(self.pairs)})")
            res[p["gid"]] = self.run_pair(p, surf, T, n_samples, thin, burn, seed + p["gid"], recenter)
        return res

    # ---- TLT ----
    def graph(self, res, carrier):
        """位点 = 晶胞中全部分子；键 = 全部有向近邻 (每条物理键两个方向都在，TLT 内部去重)。"""
        edges, groups = [], []
        for (i, j, v, t, gid) in self.edges:
            r = res.get(gid)
            if not r or "skipped" in r:
                continue
            edges.append((i, j, v, r[carrier]["mean"] / 1000.0, 1.0, t))
            groups.append(gid)
        return {"n_sites": len(self.mols), "edges": edges, "edge_group": groups, "kind": "crystal", "periodic": True}

    def tlt(self, res, carrier="hole", T=None, hbar_tau=0.005, sizes=None, keep_axes=None, n_real=20, seed=1,
            sigma_scale=1.0, sigma_E=0.0):
        from .transport import bond_index, tlt_mobility
        T = T or next(iter(v["T"] for v in res.values() if "T" in v))
        g = self.graph(res, carrier)
        if keep_axes is None:
            w = np.zeros(3)
            for e in g["edges"]:
                w += (np.array(e[5]) != 0) * e[3] ** 2
            keep_axes = tuple(sorted(int(i) for i in np.argsort(w)[::-1][:2]))
        if sizes is None:
            L = max(8, int(round(np.sqrt(900 / g["n_sites"]))))
            sizes = tuple(L if a in keep_axes else 1 for a in range(3))
        idx = bond_index(g, keep_axes)
        sigma = {k: sigma_scale * res[g["edge_group"][e]][carrier]["std"] / 1000.0 for k, e in enumerate(idx)}
        out = tlt_mobility(g, T, carrier, sigma, hbar_tau, sizes, keep_axes, n_real, seed, sigma_E=sigma_E)
        out.update(keep_axes=keep_axes, sizes=sizes)
        return out


def temperature_series(ceds, surfaces, temps, carriers=CARRIERS, hbar_tau=0.005, n_samples=1500, thin=10, burn=None,
                       n_real=16, on_progress=None, static=True):
    """逐温度：MC → ⟨V⟩、σ_V → TLT。static=True 时另算 σ = 0 (只有平均耦合) 的 TLT 作对照。"""
    rows = []
    for T in temps:
        if on_progress:
            on_progress(f"T = {T:g} K")
        res = ceds.run(surfaces, T, n_samples, thin, burn)
        row = {"T": T, "res": res}
        for c in carriers:
            t = ceds.tlt(res, c, T, hbar_tau, n_real=n_real)
            row[f"mu_{c}"] = float(t["mu_principal"][0])
            row[f"mu2_{c}"] = float(t["mu_principal"][1]) if len(t["mu_principal"]) > 1 else float("nan")
            row[f"L_{c}"] = t["L_A"]
            if static:
                t0 = ceds.tlt(res, c, T, hbar_tau, n_real=1, sigma_scale=0.0)
                row[f"mu_static_{c}"] = float(t0["mu_principal"][0])
        rows.append(row)
    return rows


def summary_table(ceds, res, carrier):
    lines = [f"{'组':<4}{'r/Å':>7}{'g':>5}{'来源':>10}{'V0':>9}{'⟨V⟩':>9}{'σ_V':>8}{'σ/|V|':>7}{'覆盖':>6}  主要涨落参数"]
    for p in ceds.pairs:
        r = res.get(p["gid"], {})
        if "skipped" in r or carrier not in r:
            lines.append(f"G{p['gid']:<3}{p['r_centroid']:7.2f}{p['g']:>5}  {r.get('skipped', '—')}")
            continue
        d = r[carrier]
        top = sorted(d["share"].items(), key=lambda kv: -kv[1])[:2]
        top_s = ", ".join(f"{k} {100 * v:.0f}%" for k, v in top) if top else ("弱耦合" if r.get("weak") else "—")
        lines.append(f"G{p['gid']:<3}{p['r_centroid']:7.2f}{p['g']:>5}{r['source']:>10}{d['V0']:9.2f}{d['mean']:9.2f}"
                     f"{d['std']:8.2f}{d['ratio']:7.2f}{r.get('coverage', 1.0):6.2f}  {top_s}")
    return "\n".join(lines)


def surfaces_from_project(project, ceds, level=None, half=None, min_points=12, allow_separable=True, eht_fallback=False):
    """由扫描项目给每个晶体二聚体选耦合曲面：局域扫描数据 → 局域多项式；共面且在网格内 → 可分离模型；
    否则 (eht_fallback) 用 EHT 或返回 None。返回 (曲面字典, 说明字典)。"""
    from .scananalysis import PaperModels, computed, scan_roles
    roles = scan_roles(project)
    names = [n for n in project.scans() if level in (None, project.scan_info(n).get("level"))]
    recs = [r for n in names if roles[n] != "md" for r in computed(project.collect(n, write=False))]
    grid = [r for n in names if roles[n] == "grid" for r in project.collect(n, write=False)]
    models = None
    if allow_separable and grid:
        try:
            models = PaperModels(grid)
        except Exception:  # noqa: BLE001
            models = None
    halves = {}                                                  # 局域扫描记录的各中心范围
    for n in names:
        inf = project.scan_info(n).get("info", {})
        for g, cen in (inf.get("centers") or {}).items():
            h = (inf.get("half_by_group") or {}).get(g) or inf.get("half")
            if h:
                halves[tuple(round(float(cen[k]), 3) for k in PARAM_KEYS)] = h
    surf, why = {}, {}
    for p in ceds.pairs:
        c = p["center"]
        key = tuple(round(float(c[k]), 3) for k in PARAM_KEYS)
        h = half or halves.get(key)
        try:
            s = LocalSurface(recs, c, half=h, ops=project.ops, min_points=min_points)
            surf[p["gid"]], why[p["gid"]] = s, s.describe()
            continue
        except ValueError as exc:
            reason = str(exc)
        coplanar = all(abs(_wrap(c[k])) < 3.0 for k in ANG_KEYS)
        if models is not None and coplanar:
            surf[p["gid"]], why[p["gid"]] = SeparableSurface(models), "共面网格可分离模型 (转角冻结)"
        elif eht_fallback:
            surf[p["gid"]], why[p["gid"]] = EHTSurface(ceds.sym, ceds.Y), f"EHT (DFT 数据不足：{reason})"
        else:
            why[p["gid"]] = f"无曲面：{reason}"
    return surf, why


def product_surfaces(ceds, g_csv, fdx_csv=None, slip_csv=None, dy_line=1.0, model="grid", gphi_csvs=()):
    """乘积模型曲面：G(z, Δy, φ) 网格 × f(Δx)，对能映射进数据区域的晶体二聚体绑定。
    返回 {gid: 曲面}；映射不进去的二聚体 (例如平行二聚体) 不在字典中。"""
    G = read_results_csv(g_csv)
    F = read_results_csv(fdx_csv) if fdx_csv else None
    SL = read_results_csv(slip_csv) if slip_csv else None
    if model == "r2":
        ps = RotModelSurface(G + sum((read_results_csv(f) for f in (gphi_csvs or [])), []), f_records=F,
                             slip_records=SL, dy_line=dy_line, characters=ceds.characters, ops=ceds.ops)
    else:
        ps = ProductSurface(G, f_records=F, slip_records=SL, dy_line=dy_line, characters=ceds.characters, ops=ceds.ops)
    out = {}
    for p in ceds.pairs:
        b = ps.bind(p["center"])
        if b is not None:
            out[p["gid"]] = b
    return out


def local_surfaces_from_csv(ceds, paths, degree=2):
    """由局域扫描的结果表 (scan collect 写出的 results.csv，含 group 列) 直接拟合局域曲面，不需要 ADF 输出文件。
    每个组的范围取该组构型相对晶体中心的最大偏离 (×1.001)；拟合用全部表中的数据 (含对称展开)。返回 {gid: 曲面}。"""
    import csv
    recs, groups = [], {}
    for path in ([paths] if isinstance(paths, str) else paths):
        rows = read_results_csv(path)
        with open(path, encoding="utf-8-sig") as fh:
            gid_of = {r["id"]: r.get("group", "") for r in csv.DictReader(fh)}
        for r in rows:
            g = str(gid_of.get(r.get("id"), "")).strip()
            if g:
                groups.setdefault(int(float(g)), []).append(r)
        recs += rows
    out = {}
    for p in ceds.pairs:
        rows = groups.get(p["gid"])
        if not rows:
            continue
        c = p["center"]
        half = {k: max(abs(_wrap(r[k] - c[k])) if k in ANG_KEYS else abs(r[k] - c[k]) for r in rows) * 1.001
                for k in PARAM_KEYS}
        half = {k: max(v, 1e-3) for k, v in half.items()}
        try:
            out[p["gid"]] = LocalSurface(recs, c, half=half, ops=ceds.ops, degree=degree)
        except ValueError:
            pass
    return out


def combined_surfaces(ceds, project=None, level=None, product=None, eht_fallback=False, local=None):
    """给每个晶体二聚体选曲面，优先级：该二聚体的局域扫描 (项目或结果表 local) > 乘积模型 > 项目的可分离模型 > EHT (可选)。
    返回 (曲面字典, 说明字典)。"""
    surf, why = {}, {}
    proj_s, proj_w = (surfaces_from_project(project, ceds, level=level, eht_fallback=False) if project is not None
                      else ({}, {}))
    product, local = product or {}, local or {}
    for p in ceds.pairs:
        g = p["gid"]
        s = proj_s.get(g)
        if g in local:
            surf[g], why[g] = local[g], local[g].describe() + " [结果表]"
        elif isinstance(s, LocalSurface):
            surf[g], why[g] = s, s.describe()
        elif g in product:
            surf[g], why[g] = product[g], product[g].describe()
        elif s is not None:
            surf[g], why[g] = s, proj_w.get(g, s.describe())
        elif eht_fallback:
            surf[g] = EHTSurface(ceds.sym, ceds.Y)
            why[g] = "EHT (没有 DFT 曲面)"
        else:
            why[g] = proj_w.get(g, "无曲面")
    return surf, why


def local_scan_configs(center, n=80, half=None, seed=2026, keys=PARAM_KEYS):
    """以晶体二聚体为中心的局域扫描 (拉丁超立方，各参数 ±h)，另加中心点。"""
    from .scanflow import lhs_blocks
    h = dict(DEFAULT_HALF, **(half or {}))
    blk = {"tag": "local", "n": n}
    for k in PARAM_KEYS:
        blk[k] = (center[k] - h[k], center[k] + h[k]) if k in keys else center[k]
    return [make_cfg(*(center[k] for k in ("z", "dx", "dy")), gamma=center["gamma"], pitch=center["pitch"],
                     roll=center["roll"], set="local")] + lhs_blocks([blk], seed=seed)


def cif_ueq(path, heavy_only=True):
    """读取 CIF 中的 _atom_site_U_iso_or_equiv (或由 _atom_site_aniso_U_11/22/33 求 U_eq) 的平均值 (Å²)。没有则返回 None。"""
    import re
    text = open(path, encoding="utf-8", errors="ignore").read()
    vals = []
    for block in re.split(r"\n\s*loop_\s*\n", text):
        heads = re.findall(r"^\s*(_atom_site\S*)", block, re.M)
        if not heads:
            continue
        lines = [ln for ln in block.splitlines() if ln.strip() and not ln.strip().startswith(("_", "#", "data_"))]
        def col(name):
            return heads.index(name) if name in heads else None
        iu, it, il = col("_atom_site_U_iso_or_equiv"), col("_atom_site_type_symbol"), col("_atom_site_label")
        a11, a22, a33 = col("_atom_site_aniso_U_11"), col("_atom_site_aniso_U_22"), col("_atom_site_aniso_U_33")
        al = col("_atom_site_aniso_label")
        for ln in lines:
            f = ln.split()
            if len(f) != len(heads):
                continue
            lab = f[it] if it is not None else (f[il] if il is not None else (f[al] if al is not None else ""))
            el = re.match(r"[A-Za-z]+", lab).group(0) if re.match(r"[A-Za-z]+", lab) else ""
            if heavy_only and el.upper().startswith("H") and not el.upper().startswith("HE") and len(el) <= 1:
                continue
            num = lambda x: float(re.sub(r"\(.*\)", "", x))  # noqa: E731
            try:
                if iu is not None:
                    vals.append(num(f[iu]))
                elif a11 is not None:
                    vals.append((num(f[a11]) + num(f[a22]) + num(f[a33])) / 3)
            except ValueError:
                continue
        if vals:
            break
    return float(np.mean(vals)) if vals else None


# ----------------------------------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------------------------------
def export_csv(ceds, results_by_T, path, rows=None):
    """results_by_T: {T: run() 的结果}；rows: temperature_series 的结果 (可选，写迁移率)。"""
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["T_K", "group", "r_centroid_A", "g", "source", "carrier", "V0_meV", "V_mean_meV", "sigma_meV",
                    "sigma_over_V", "coverage", "U_eq_A", "U_eq_B"] + [f"share_{k}" for k in PARAM_KEYS] +
                   [f"std_{k}" for k in PARAM_KEYS])
        for T, res in results_by_T.items():
            for p in ceds.pairs:
                r = res.get(p["gid"], {})
                if "skipped" in r or "hole" not in r:
                    continue
                ue = (r.get("mc") or {}).get("U_eq", ["", ""])
                for c in CARRIERS:
                    d = r[c]
                    w.writerow([T, p["gid"], f"{p['r_centroid']:.3f}", p["g"], r.get("source", ""), c, f"{d['V0']:.3f}",
                                f"{d['mean']:.3f}", f"{d['std']:.3f}", f"{d['ratio']:.4f}", f"{r.get('coverage', 1):.3f}",
                                *(f"{u:.5f}" if u != "" else "" for u in ue)] +
                               [f"{d['share'].get(k, 0):.3f}" for k in PARAM_KEYS] +
                               [f"{r.get('q_std', {}).get(k, 0):.4f}" for k in PARAM_KEYS])
        if rows:
            w.writerow([])
            w.writerow(["T_K", "mu_hole", "mu2_hole", "mu_static_hole", "L_hole_A", "mu_electron", "mu2_electron",
                        "mu_static_electron", "L_electron_A"])
            for r in rows:
                w.writerow([r["T"]] + [f"{r.get(k, float('nan')):.5g}" for k in
                                       ("mu_hole", "mu2_hole", "mu_static_hole", "L_hole", "mu_electron", "mu2_electron",
                                        "mu_static_electron", "L_electron")])
    return path


def plot_mu_T(ax, rows):
    from .scananalysis import COL, LAB, MARK
    T = [r["T"] for r in rows]
    for c in CARRIERS:
        if f"mu_{c}" not in rows[0]:
            continue
        ax.plot(T, [r[f"mu_{c}"] for r in rows], "-" + MARK[c], color=COL[c], ms=5,
                mfc="none" if c == "electron" else COL[c], label=f"{LAB[c]}, CEDS + TLT")
        if f"mu_static_{c}" in rows[0]:
            ax.plot(T, [r[f"mu_static_{c}"] for r in rows], "--", color=COL[c], lw=1, label=f"{LAB[c]}, σ = 0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("μ (cm$^2$ V$^{-1}$ s$^{-1}$)")
    ax.legend(fontsize=7)


def plot_hist(ax, ceds, res, carrier, max_groups=4):
    import matplotlib
    cmap = matplotlib.colormaps["tab10"]
    k = 0
    for p in ceds.pairs:
        r = res.get(p["gid"], {})
        if "skipped" in r or r.get("weak") or carrier not in r:
            continue
        d = r[carrier]
        ax.hist(d["samples"], bins=30, alpha=0.5, color=cmap(k), label=f"G{p['gid']} ({p['r_centroid']:.2f} Å): "
                f"{d['mean']:.1f} ± {d['std']:.1f} meV")
        k += 1
        if k >= max_groups:
            break
    ax.set_xlabel("V (meV)")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)


# ----------------------------------------------------------------------------------------------
# 乘积模型曲面：V = f(|Δx|) · G(z, Δy, φ)
# ----------------------------------------------------------------------------------------------
def _cfg(q):
    return make_cfg(q[0], q[1], q[2], gamma=q[3], pitch=q[4], roll=q[5])


def _apply_transform(c, t):
    """t = (是否换参考分子, 整体操作, B 自身二重轴)；返回变换后的构型。"""
    rev, op, g = t
    if rev:
        c = reverse_cfg(c)
    if op != "E":
        c = apply_op(c, op)
    if g != "E":
        from .scanflow import self_op
        c = self_op(c, g)
    return c


def find_transform(center, inside, score, ops, proper=("E",) + PROPER):
    """在 {换参考分子} × {整体操作} × {B 自身二重轴} 中找把中心构型映射到数据区域的变换。"""
    best = None
    for rev in (False, True):
        for op in ops:
            for g in proper:
                if g != "E" and g not in ops:
                    continue
                c = _apply_transform(center, (rev, op, g))
                if not inside(c):
                    continue
                sc = score(c)
                if best is None or sc < best[0]:
                    best = (sc, (rev, op, g), c)
    return best


class ProductSurface:
    """V = f(|Δx|) · G(z, Δy, φ)，γ = θ = 0 (冻结)。
    G：Δx = 0 的 (z × Δy × φ) 网格，每个 φ 上对 (z, Δy) 作双三次样条，φ 方向作拉格朗日插值；
       网格中因接触被排除的点按该 (Δy, φ) 列的 ln|V|–z 线性外推补齐。
    f：长轴滑移函数 V(Δx)/V(0)，取自转动二聚体的 Δx 扫描 (f_records)，或平行二聚体滑移图在 Δy = dy_line 处的一行 (两者只差一个比例因子)。"""

    source = "product"

    def __init__(self, g_records, carriers=CARRIERS, f_records=None, slip_records=None, dy_line=1.0, characters=None,
                 ops=("E",), margin_phi=5.0):
        from scipy.interpolate import CubicSpline, RectBivariateSpline
        self.chars = characters or {c: {"E": 1} for c in carriers}
        self.ops = list(ops)
        rs = [r for r in g_records if abs(r["dx"]) < 1e-6 and abs(r.get("gamma", 0)) < 1e-6 and abs(r.get("pitch", 0)) < 1e-6]
        if not rs:
            raise ValueError("G 网格中没有 Δx = 0、γ = θ = 0 的构型")
        self.zs = np.array(sorted({round(r["z"], 4) for r in rs}))
        self.dys = np.array(sorted({round(r["dy"], 4) for r in rs}))
        self.phis = np.array(sorted({round(r["roll"], 4) for r in rs}))
        if len(self.zs) < 4 or len(self.dys) < 4 or len(self.phis) < 2:
            raise ValueError("G 网格太小 (z、Δy 至少各 4 个值，φ 至少 2 个)")
        self.G, self.filled = {}, 0
        for c in carriers:
            self.G[c] = []
            for ph in self.phis:
                M = np.full((len(self.zs), len(self.dys)), np.nan)
                for r in rs:
                    if abs(r["roll"] - ph) < 1e-3 and r.get(c) is not None:
                        M[np.searchsorted(self.zs, round(r["z"], 4)), np.searchsorted(self.dys, round(r["dy"], 4))] = r[c]
                for j in range(len(self.dys)):                    # 缺失点：ln|V| 对 z 线性外推
                    col = M[:, j]
                    bad = np.isnan(col)
                    if bad.any():
                        ok = ~bad
                        if ok.sum() < 2:
                            raise ValueError(f"G 网格 φ = {ph}、Δy = {self.dys[j]} 的数据太少")
                        zz, vv = self.zs[ok][:4], col[ok][:4]
                        sg = np.sign(vv[0]) or 1.0
                        b, a = np.polyfit(zz, np.log(np.abs(vv) + 1e-6), 1)
                        col[bad] = sg * np.exp(a + b * self.zs[bad])
                        self.filled += int(bad.sum())
                self.G[c].append(RectBivariateSpline(self.zs, self.dys, M, kx=3, ky=3))
        self.f, self.f_source = {}, None
        if f_records:
            fr = [r for r in f_records if r.get(carriers[0]) is not None]
            x = np.array([abs(r["dx"]) for r in fr])
            o = np.argsort(x)
            for c in carriers:
                v = np.array([r[c] for r in fr])[o]
                self.f[c] = CubicSpline(x[o], v / v[0])
            self.f_source = f"转动二聚体的 Δx 扫描 ({len(fr)} 点)"
            self.f_range = float(x.max())
        elif slip_records:
            line = [r for r in slip_records if abs(r["dy"] - dy_line) < 1e-6 and r["dx"] >= -1e-9]
            if len(line) < 4:
                raise ValueError(f"平行滑移图中 Δy = {dy_line} 的数据不足")
            x = np.array([r["dx"] for r in line])
            o = np.argsort(x)
            for c in carriers:
                v = np.array([r[c] for r in line])[o]
                self.f[c] = CubicSpline(x[o], v / v[0])
            zl = line[0]["z"]
            self.f_source = f"平行二聚体滑移图 Δy = {dy_line} Å 的一行" + ("" if not np.isfinite(zl) else f" (z = {zl:.2f} Å)")
            self.f_range = float(x.max())
        self.margin_phi = margin_phi
        self.frozen = ["gamma", "pitch"] + ([] if self.f else ["dx"])

    def inside(self, c):
        return (self.zs[0] - 1e-6 <= c["z"] <= self.zs[-1] + 1e-6 and self.dys[0] - 1e-6 <= c["dy"] <= self.dys[-1] + 1e-6
                and self.phis[0] - self.margin_phi <= c["roll"] <= self.phis[-1] + self.margin_phi
                and abs(_wrap(c["gamma"])) < 15 and abs(_wrap(c["pitch"])) < 15)

    def _G(self, c, z, dy, phi):
        vals = np.array([float(sp(z, dy)[0, 0]) for sp in self.G[c]])
        k = min(len(self.phis), 3)
        idx = np.argsort(np.abs(self.phis - phi))[:k]
        xs, ys = self.phis[idx], vals[idx]
        out = 0.0
        for i in range(k):                                         # 拉格朗日插值 (最近的 k 个 φ)
            w = 1.0
            for j in range(k):
                if j != i:
                    w *= (phi - xs[j]) / (xs[i] - xs[j])
            out += w * ys[i]
        return out

    def value(self, c, carrier):
        g = self._G(carrier, c["z"], c["dy"], c["roll"])
        if self.f:
            g *= float(self.f[carrier](min(abs(c["dx"]), self.f_range)))
        return g

    def bind(self, center):
        """为一个二聚体固定映射到数据区域的变换，返回可调用的曲面。"""
        best = find_transform(center, self.inside,
                              lambda c: abs(_wrap(c["gamma"])) + abs(_wrap(c["pitch"])) + (0 if self.f else 10 * abs(c["dx"])),
                              self.ops)
        if best is None:
            return None
        return _BoundProduct(self, best[1], best[2])

    def describe(self):
        return (f"乘积模型 f(|Δx|)·G(z, Δy, φ)：G 网格 z {self.zs[0]:g}–{self.zs[-1]:g} Å × Δy {self.dys[0]:g}–{self.dys[-1]:g} Å"
                f" × φ {', '.join(f'{p:g}' for p in self.phis)}°" + (f"，补齐 {self.filled // max(len(self.G), 1)} 个缺失点" if self.filled else "")
                + (f"；f 取自{self.f_source}" if self.f else "；Δx 冻结"))


class _BoundProduct:
    source = "product"

    def __init__(self, ps, transform, mapped_center):
        self.ps, self.t, self.mapped = ps, transform, mapped_center
        self.frozen = ps.frozen

    def _chi(self, carrier):
        g = self.t[2]
        return 1 if g == "E" else (self.ps.chars.get(carrier, {}).get(g, 1) or 1)

    def __call__(self, Q, carrier):
        chi = self._chi(carrier)
        out = []
        for q in np.atleast_2d(Q):
            c = _apply_transform(_cfg(q), self.t)
            out.append(chi * self.ps.value(c, carrier))
        return np.array(out)

    def coverage(self, Q):
        n = 0
        Q = np.atleast_2d(Q)
        for q in Q:
            c = _apply_transform(_cfg(q), self.t)
            n += (self.ps.zs[0] <= c["z"] <= self.ps.zs[-1] and self.ps.dys[0] <= c["dy"] <= self.ps.dys[-1]
                  and self.ps.phis[0] <= c["roll"] <= self.ps.phis[-1])
        return n / len(Q)

    def describe(self):
        rev, op, g = self.t
        m = self.mapped
        return (self.ps.describe() + f"；映射 {'换参考分子 + ' if rev else ''}{op}{'' if g == 'E' else ' + B 绕 ' + g}"
                f" → z {m['z']:.3f}, Δx {m['dx']:.3f}, Δy {m['dy']:.3f}, φ {m['roll']:.2f}°")


def read_results_csv(path):
    """读取 parse_results.py / 本软件写出的结果表 (V_h_meV、V_e_meV 或 hole、electron 列)。"""
    import csv
    out = []
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                d = {k: float(r.get(k) or 0.0) for k in PARAM_KEYS}
                if not r.get("z"):                         # slip_full.csv 没有 z 列
                    d["z"] = float("nan")
                h = r.get("V_h_meV", r.get("hole"))
                e = r.get("V_e_meV", r.get("electron"))
                d["hole"] = float(h) if h not in (None, "") else None
                d["electron"] = float(e) if e not in (None, "") else None
                for k in ("S_h", "S_e", "dmin_A"):
                    if r.get(k) not in (None, ""):
                        d[k] = float(r[k])
                d["id"] = r.get("id", "")
            except ValueError:
                continue
            out.append(d)
    return out


class RotModelSurface(ProductSurface):
    """转角工作模型 (R2)：V = f(|Δx|) · g(Δy, φ) · exp[−β(Δy)(z − z0)]，z0 = 4.8 Å。
    β(Δy)：各 (Δy, φ) 列 ln|V|–z 拟合 (z ≥ zlo，|V| ≥ 3 meV，同号，≥ 4 点) 的衰减常数，以 √max|V| 加权作二次拟合
    (只用 max|V| ≥ 30 meV 的列)；g：zlo ≤ z ≤ zhi 的点按 β 折算到 z0 后取平均 (同列变号的不用)，
    φ 方向单调三次 (PCHIP)、Δy 方向自然三次样条，范围外线性外延。只用最近原子距离 ≥ dmin 的点。"""

    source = "rotmodel"

    def __init__(self, g_records, carriers=CARRIERS, f_records=None, slip_records=None, dy_line=1.0, characters=None,
                 ops=("E",), z0=4.8, zlo=4.6, zhi=5.4, dmin=2.6, margin_phi=5.0):
        from scipy.interpolate import CubicSpline, PchipInterpolator
        self.chars = characters or {c: {"E": 1} for c in carriers}
        self.ops = list(ops)
        self.z0, self.margin_phi = z0, margin_phi
        use = [r for r in g_records if abs(r["dx"]) < 1e-6 and r.get("dmin_A", 9.0) >= dmin]
        if not use:
            raise ValueError("没有 Δx = 0 的转角网格数据")
        self.zs = np.array(sorted({round(r["z"], 4) for r in use}))
        self.dys = np.array(sorted({round(r["dy"], 4) for r in use}))
        self.phis = np.array(sorted({round(r["roll"], 4) for r in use}))
        self.beta, self.gtab, self.info = {}, {}, {}
        for c in carriers:
            cols = {}
            for r in use:
                if r["z"] >= zlo - 1e-6 and r.get(c) is not None:
                    cols.setdefault((round(r["roll"], 2), round(r["dy"], 3)), []).append((r["z"], r[c]))
            info = []
            for (ph, dy), pts in cols.items():
                pts.sort()
                z = np.array([p[0] for p in pts])
                v = np.array([p[1] for p in pts])
                sel = (np.abs(v) >= 3.0) & (np.sign(v) == np.sign(v[0]))
                if sel.sum() >= 4:
                    b = -np.polyfit(z[sel], np.log(np.abs(v[sel])), 1)[0]
                    info.append((ph, dy, b, np.abs(v).max()))
            info = np.array(info)
            sel = info[:, 3] >= 30.0
            quad = np.polyfit(info[sel, 1], info[sel, 2], 2, w=np.sqrt(info[sel, 3]))
            self.beta[c] = quad
            self.info[c] = {"n_columns": int(sel.sum()), "beta_at_dy1": float(np.polyval(quad, 1.0))}
            tab = {}
            for r in use:
                if zlo - 1e-6 <= r["z"] <= zhi + 1e-6 and r.get(c) is not None:
                    b = float(np.polyval(quad, r["dy"]))
                    tab.setdefault((round(r["dy"], 3), round(r["roll"], 2)), []).append(r[c] * np.exp(b * (r["z"] - z0)))
            g = {}
            for (dy, ph), v in tab.items():
                v = np.array(v)
                if np.all(np.sign(v) == np.sign(v[0])):
                    g.setdefault(dy, {})[ph] = float(v.mean())
            self.gtab[c] = g
        self._pchip, self._cs = PchipInterpolator, CubicSpline
        self.f, self.f_source = {}, None
        src = f_records or ([r for r in (slip_records or []) if abs(r["dy"] - dy_line) < 1e-6 and r["dx"] >= -1e-9])
        if src:
            x = np.array([abs(r["dx"]) for r in src])
            o = np.argsort(x)
            for c in carriers:
                v = np.array([r[c] for r in src])[o]
                self.f[c] = CubicSpline(x[o], v / v[0], bc_type="natural")
            self.f_range = float(x.max())
            self.f_source = (f"转动二聚体的 Δx 扫描 ({len(src)} 点)" if f_records else
                             f"平行二聚体滑移图 Δy = {dy_line} Å 的一行")
        self.frozen = ["gamma", "pitch"] + ([] if self.f else ["dx"])
        self.filled = 0

    @staticmethod
    def _interp(x, y, xs, kind, P, C):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) == 1:
            return float(y[0])
        if len(x) == 2:
            return float(y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (xs - x[0]))
        f = P(x, y, extrapolate=False) if kind == "pchip" else C(x, y, bc_type="natural", extrapolate=False)
        if x[0] <= xs <= x[-1]:
            return float(f(xs))
        d = f.derivative()
        return float(y[0] + float(d(x[0])) * (xs - x[0])) if xs < x[0] else float(y[-1] + float(d(x[-1])) * (xs - x[-1]))

    def _G(self, c, z, dy, phi):
        tab = self.gtab[c]
        dys = sorted(tab)
        vals = []
        for y in dys:
            ph = sorted(tab[y])
            vals.append(self._interp(ph, [tab[y][p] for p in ph], phi, "pchip", self._pchip, self._cs))
        g = self._interp(dys, vals, dy, "spline", self._pchip, self._cs)
        return g * np.exp(-float(np.polyval(self.beta[c], dy)) * (z - self.z0))

    def describe(self):
        b = ", ".join(f"{c} β(Δy=1) = {v['beta_at_dy1']:.3f} Å⁻¹ ({v['n_columns']} 列)" for c, v in self.info.items())
        return (f"转角模型 R2：f(|Δx|)·g(Δy, φ)·exp[−β(Δy)(z − {self.z0:g})]，φ = {', '.join(f'{p:g}' for p in self.phis)}°；{b}"
                + (f"；f 取自{self.f_source}" if self.f else "；Δx 冻结"))
