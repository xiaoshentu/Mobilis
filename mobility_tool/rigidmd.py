"""全晶体刚体分子动力学 (超胞中所有分子同时运动)，给 CEDS 提供 σ_V，并接到 TLT 得到 μ(T)。

与 ceds.DimerMC (只动二聚体两个分子、其余固定，局域 Einstein 近似) 相比，这里邻居一起运动，包含声学 (集体) 模式，
二聚体相对几何的涨落不再受“笼子固定”的约束；同时用带原子电荷的 OPLS-AA 力场代替不含静电的 UFF。
它给出 Einstein 下限与 U_eq 校准上限之间的独立估计，并由耦合的自相关时间直接估计 TLT 的 ħ/τ。

* 力场：芳香 C/H 用 OPLS-AA (CA/HA，σ、ε 几何平均组合；连 H 的 C −0.115 e，H +0.115 e，稠环 C 0)；
  其他元素用 UFF 的 Lennard-Jones 参数、电荷 0 (会给出提示)。静电用阻尼移位力 (DSF, Fennell & Gezelter 2006)，
  α = 0.2 Å⁻¹；LJ 在 r_sw → rc 之间用五次开关函数平滑截断。分子内相互作用不计 (刚体)。
* 积分：平动 + 转动的 BAOAB Langevin 分裂 (Leimkuhler & Matthews 2013)；转动的自由步用
  Dullweber–Leimkuhler–McLachlan (1997) 对称分裂 R1 R2 R3 R2 R1，在惯量主轴系中进行。
  分子坐标系 (molecular_frame：重原子中心 + 重原子协方差主轴) 与惯量主轴系不同时自动换算。
* 单位：Å、fs、amu、kcal/mol。
* 原子对表是静态的 (晶体中分子不会换邻居)，含超胞自身镜像；每个无序对只列一次。

主要入口
  RigidCrystalMD(crys, frames, monomer, supercell)   建模 → run(T, n_steps, …) 得到轨迹
  md_temperature_series(ceds, surfaces, temps, …)     逐温度 MD → 二聚体参数 → 耦合曲面 → σ_V、τ_c → TLT
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

KCAL2MD = 4.184e-4            # (kcal/mol) → amu Å² fs⁻²
KB = 0.0019872041             # kcal/mol/K
COUL = 332.0637               # kcal Å / (mol e²)
HBAR_MEV_FS = 658.2119569     # ħ / (meV·fs)
MASS = {"H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Si": 28.085, "P": 30.974,
        "S": 32.06, "Cl": 35.45, "Se": 78.971, "Br": 79.904, "I": 126.904, "Li": 6.94, "Be": 9.012, "Mg": 24.305,
        "Al": 26.982, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "Ru": 101.07, "Os": 190.23,
        "Ir": 192.217, "Pt": 195.084, "Au": 196.967}
# OPLS-AA (σ/Å, ε/kcal mol⁻¹)
OPLS = {"CA": (3.55, 0.070), "HA": (2.42, 0.030)}
Q_CH = 0.115

# 默认设置
DEFAULTS = dict(supercell=(4, 3, 2), rc=9.0, r_sw=8.0, skin=1.0, dt=5.0, equil_ps=5.0, prod_ps=30.0, every=4,
                gamma_eq=5e-3, gamma=1e-3)


def force_field(symbols, Y):
    """返回 (σ, ε, q, 说明)。芳香 C/H 用 OPLS-AA，其余元素用 UFF LJ、电荷 0。"""
    from .interaction import UFF
    Y = np.asarray(Y, float)
    sig, eps, q, other = [], [], [], set()
    for i, e in enumerate(symbols):
        if e == "H":
            s, ep = OPLS["HA"]
            qi = Q_CH
        elif e == "C":
            s, ep = OPLS["CA"]
            nH = sum(1 for j, f in enumerate(symbols) if f == "H" and np.linalg.norm(Y[j] - Y[i]) < 1.25)
            qi = -Q_CH * nH
        else:
            if e not in UFF:
                raise ValueError(f"元素 {e} 没有力场参数")
            x, d = UFF[e]
            s, ep, qi = x / 2 ** (1 / 6), d, 0.0
            other.add(e)
        sig.append(s); eps.append(ep); q.append(qi)
    # 与 C 相连的 H 以外的原子 (杂原子) 会使局部电荷不平衡：把整体净电荷均摊回去，保证分子电中性
    q = np.array(q)
    if abs(q.sum()) > 1e-9:
        q -= q.sum() / len(q)
    note = "OPLS-AA (芳香 C/H) + DSF 静电"
    if other:
        note += f"；{', '.join(sorted(other))} 用 UFF LJ、电荷 0 (近似)"
    return np.array(sig), np.array(eps), q, note


def _rot(axis, th):
    c, s = np.cos(th), np.sin(th)
    R = np.zeros(th.shape + (3, 3))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R[..., axis, axis] = 1.0
    R[..., i, i] = c; R[..., j, j] = c
    R[..., i, j] = -s; R[..., j, i] = s
    return R


class RigidCrystalMD:
    """crys: crystal.read_cif；frames: [(ax, c)] 晶胞内各分子的分子坐标系 (列为轴) 与原点 (即 CEDS.frames)；
    monomer: (symbols, Y) 分子坐标系中的刚性单体 (CEDS 用的同一个单体)；supercell: (na, nb, nc)。"""

    def __init__(self, crys, frames, monomer, supercell=(4, 3, 2), rc=9.0, r_sw=8.0, alpha=0.2, skin=1.0,
                 charges=None):
        """charges: 可选的原子电荷 (单体原子顺序，如 Gaussian ESP 电荷)，代替默认的 OPLS 芳香 C/H 电荷；
        含 N、O、金属等极性原子的分子 (Alq3、Ir 配合物) 建议提供。"""
        self.sym = list(monomer[0])
        Ym = np.asarray(monomer[1], float)
        miss = sorted({e for e in self.sym if e not in MASS})
        if miss:
            raise ValueError(f"缺少原子质量: {miss}")
        self.lat = np.asarray(crys["lattice"], float)
        self.sc = tuple(int(n) for n in supercell)
        self.rc, self.r_sw, self.alpha = float(rc), float(r_sw), float(alpha)
        m = np.array([MASS[e] for e in self.sym])
        self.M = m.sum()
        com = (m[:, None] * Ym).sum(0) / self.M
        I = np.einsum("a,ab,ac->bc", m, Ym - com, Ym - com)
        I = np.trace(I) * np.eye(3) - I
        if np.linalg.norm(com) < 1e-3 and np.abs(I - np.diag(np.diag(I))).max() < 1e-3 * np.abs(I).max():
            # 分子坐标系已是惯量主轴系 (如 D2h 的 DFT 单体)：直接使用
            self.R, self.com = np.eye(3), np.zeros(3)
            self.I = np.diag(np.einsum("a,ab,ac->bc", m, Ym, Ym).trace() * np.eye(3)
                             - np.einsum("a,ab,ac->bc", m, Ym, Ym)).copy()
        else:
            w, R = np.linalg.eigh(I)
            for k in range(3):                                     # 主轴尽量与分子轴同向
                j = int(np.argmax(np.abs(R[:, k])))
                if R[j, k] < 0:
                    R[:, k] *= -1
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1
            self.R, self.com, self.I = R, com, w.copy()
        self.Y = (Ym - self.com) @ self.R                           # 主轴系 (体坐标) 中的原子坐标
        self.sig, self.eps, self.q, self.ff_note = force_field(self.sym, Ym)
        if charges is not None:
            q = np.asarray(charges, float).ravel()
            if len(q) != len(self.sym):
                raise ValueError(f"电荷数 ({len(q)}) 与单体原子数 ({len(self.sym)}) 不一致")
            self.q = q - q.sum() / len(q)
            self.ff_note = self.ff_note.split(" + ")[0] + " LJ + 用户电荷 (ESP) DSF 静电" + \
                ("；" + self.ff_note.split("；", 1)[1] if "；" in self.ff_note else "")
            self.ff_note = self.ff_note.replace("、电荷 0 (近似)", " (电荷取自 ESP)")
        na, nb, nc = self.sc
        self.cells = [(a, b, c) for a in range(na) for b in range(nb) for c in range(nc)]
        self._cell_index = {cl: i for i, cl in enumerate(self.cells)}
        self.nk = len(frames)
        Qm0 = np.array([np.asarray(frames[k][0], float) for _ in self.cells for k in range(self.nk)])
        o0 = np.array([np.asarray(frames[k][1], float) + np.array(cl) @ self.lat
                       for cl in self.cells for k in range(self.nk)])
        # 对映体分子 (坐标系为非真转动，见 surrogate.MIRROR)：体坐标取镜像 Yb·D，取向 Q 仍为真转动，积分不变
        D = np.diag([1.0, 1.0, -1.0])
        self.mirror_k = [bool(np.linalg.det(np.asarray(frames[k][0], float)) < 0) for k in range(self.nk)]
        mir = np.array([self.mirror_k[k] for _ in self.cells for k in range(self.nk)])
        self.mirror_n = mir
        self.Rn = np.where(mir[:, None, None], (self.R @ D)[None], self.R[None])
        self.Yn = np.where(mir[:, None, None], (self.Y * np.array([1.0, 1.0, -1.0]))[None], self.Y[None])
        self.Q0 = Qm0 @ self.Rn
        self.c0 = o0 + np.einsum("nij,j->ni", Qm0, self.com)
        self.N = len(self.c0)
        self.box = np.diag(self.sc) @ self.lat                     # 超胞矢量 (行)
        self._build_pairs(skin)

    # ---- 拓扑 ----
    def index(self, cell, k):
        na, nb, nc = self.sc
        return self._cell_index[(cell[0] % na, cell[1] % nb, cell[2] % nc)] * self.nk + k

    def _build_pairs(self, skin):
        nat = len(self.sym)
        X = self.atoms(self.c0, self.Q0)                            # (N, nat, 3)
        Rmol = np.linalg.norm(self.Y, axis=1).max()
        cut = self.rc + skin
        shifts = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
        S = shifts @ self.box
        ii, jj, aa, bb, ss = [], [], [], [], []
        for I in range(self.N):
            for J in range(I, self.N):
                for si, s in enumerate(shifts):
                    if I == J and tuple(s) <= (0, 0, 0):
                        continue
                    d = self.c0[J] + S[si] - self.c0[I]
                    if np.linalg.norm(d) > cut + 2 * Rmol:
                        continue
                    D = np.linalg.norm(X[J][None, :, :] + S[si] - X[I][:, None, :], axis=-1)
                    a, b = np.nonzero(D < cut)
                    ii.append(np.full(len(a), I)); jj.append(np.full(len(a), J))
                    aa.append(a); bb.append(b); ss.append(np.full(len(a), si))
        self.pI, self.pJ = np.concatenate(ii), np.concatenate(jj)
        self.pa, self.pb = np.concatenate(aa), np.concatenate(bb)
        self.pS = S[np.concatenate(ss)]
        self.gi, self.gj = self.pI * nat + self.pa, self.pJ * nat + self.pb
        self.sij = np.sqrt(self.sig[self.pa] * self.sig[self.pb])          # OPLS：几何平均
        self.eij = np.sqrt(self.eps[self.pa] * self.eps[self.pb])
        self.qq = COUL * self.q[self.pa] * self.q[self.pb]
        self.qi = np.nonzero(self.qq != 0)[0]
        self.sij2, self.eij4 = self.sij ** 2, 4 * self.eij
        rc, a = self.rc, self.alpha
        self._ec = erfc(a * rc) / rc
        self._fc = erfc(a * rc) / rc ** 2 + 2 * a / np.sqrt(np.pi) * np.exp(-(a * rc) ** 2) / rc
        self.n_pairs = len(self.pI)

    # ---- 坐标 ----
    def atoms(self, c, Q):
        return c[..., None, :] + np.einsum("...nij,naj->...nai", Q, self.Yn)

    def mol_frames(self, c, Q):
        """质心 c、主轴取向 Q → 分子坐标系 (列为轴) 与原点 (与 molecular_frame / CEDS.frames 一致；对映体为非真转动)。"""
        Qm = np.einsum("...nij,nkj->...nik", Q, self.Rn)
        return Qm, c - np.einsum("...ij,j->...i", Qm, self.com)

    def lattice_energy(self):
        """初始 (实验) 结构的分子间能量 / kcal mol⁻¹ 每分子 (负值；与升华焓比较)。"""
        return self.forces(self.c0, self.Q0)[2] / self.N

    # ---- 力 ----
    def forces(self, c, Q):
        nat = len(self.sym)
        X = self.atoms(c, Q).reshape(-1, 3)
        d = X[self.gj] + self.pS - X[self.gi]
        r2 = (d * d).sum(1)
        r = np.sqrt(r2)
        on = (r < self.rc).astype(float)
        sr6 = (self.sij2 / r2) ** 3
        e_lj = self.eij4 * (sr6 * sr6 - sr6)
        f_lj = 6 * self.eij4 * (2 * sr6 * sr6 - sr6) / r          # −dE/dr
        x = np.clip((r - self.r_sw) / (self.rc - self.r_sw), 0, 1)  # 能量开关 (r_sw → rc)
        S = 1 - x ** 3 * (10 - 15 * x + 6 * x * x)
        dS = -30 * x * x * (1 - x) ** 2 / (self.rc - self.r_sw)
        f = f_lj * S - e_lj * dS
        e = e_lj * S
        a = self.alpha
        qq, rq = self.qq[self.qi], r[self.qi]
        ar = a * rq
        ec = erfc(ar)
        e[self.qi] += on[self.qi] * qq * (ec / rq - self._ec + self._fc * (rq - self.rc))
        f[self.qi] += on[self.qi] * qq * (ec / rq ** 2 + 2 * a / np.sqrt(np.pi) * np.exp(-ar * ar) / rq - self._fc)
        Fv = (f / r)[:, None] * d                                       # 作用在 j 上 (沿 +d)
        F = np.zeros((self.N * nat, 3))
        for k in range(3):
            F[:, k] = np.bincount(self.gj, Fv[:, k], self.N * nat) - np.bincount(self.gi, Fv[:, k], self.N * nat)
        F = F.reshape(self.N, nat, 3)
        Fc = F.sum(1)
        rel = X.reshape(self.N, nat, 3) - c[:, None, :]
        tau = np.cross(rel, F).sum(1)
        return Fc, tau, float(e.sum())

    # ---- 积分 ----
    def _free_rot(self, Q, pi, h):
        for ax, w in ((0, 0.5), (1, 0.5), (2, 1.0), (1, 0.5), (0, 0.5)):
            th = w * h * pi[:, ax] / self.I[ax]
            R = _rot(ax, th)
            Q = Q @ R
            pi = np.einsum("nji,nj->ni", R, pi)                         # π ← Rᵀ π
        return Q, pi

    def run(self, T, n_steps, dt=5.0, gamma=1e-3, every=4, seed=1, state=None, on_frame=None):
        """BAOAB Langevin。gamma/fs⁻¹ (1e-3 = 1 ps⁻¹)。返回 (frames, state)；
        frames: c (帧, N, 3) 质心、Q (帧, N, 3, 3) 主轴取向、E 势能、T_tr / T_rot 平动 / 转动温度。"""
        rng = np.random.default_rng(seed)
        kT = KB * T * KCAL2MD
        if state is None:
            c, Q = self.c0.copy(), self.Q0.copy()
            v = rng.normal(size=(self.N, 3)) * np.sqrt(kT / self.M)
            v -= v.mean(0)
            pi = rng.normal(size=(self.N, 3)) * np.sqrt(kT * self.I)
        else:
            c, Q, v, pi = (np.array(x) for x in state)
        Fc, tau, E = self.forces(c, Q)
        c1 = np.exp(-gamma * dt)
        s1 = np.sqrt(1 - c1 * c1)
        out = {"c": [], "Q": [], "E": [], "T_tr": [], "T_rot": []}
        for step in range(1, n_steps + 1):
            v += 0.5 * dt * KCAL2MD * Fc / self.M
            pi += 0.5 * dt * KCAL2MD * np.einsum("nji,nj->ni", Q, tau)
            c += 0.5 * dt * v
            Q, pi = self._free_rot(Q, pi, 0.5 * dt)
            v = c1 * v + s1 * np.sqrt(kT / self.M) * rng.normal(size=v.shape)
            pi = c1 * pi + s1 * np.sqrt(kT * self.I) * rng.normal(size=pi.shape)
            c += 0.5 * dt * v
            Q, pi = self._free_rot(Q, pi, 0.5 * dt)
            Fc, tau, E = self.forces(c, Q)
            v += 0.5 * dt * KCAL2MD * Fc / self.M
            pi += 0.5 * dt * KCAL2MD * np.einsum("nji,nj->ni", Q, tau)
            if step % every == 0:
                out["c"].append(c.copy()); out["Q"].append(Q.copy()); out["E"].append(E)
                out["T_tr"].append(self.M * (v * v).sum() / (3 * self.N * KB * KCAL2MD))
                out["T_rot"].append((pi * pi / self.I).sum() / (3 * self.N * KB * KCAL2MD))
                if on_frame:
                    on_frame(step, n_steps)
        frames = {k: np.array(v_) for k, v_ in out.items()}
        frames["dt_frame_fs"] = every * dt
        return frames, (c, Q, v, pi)

    def total_energy(self, state):
        c, Q, v, pi = state
        return self.forces(c, Q)[2] + (0.5 * self.M * (v * v).sum() + 0.5 * (pi * pi / self.I).sum()) / KCAL2MD

    def u_eq(self, frames, heavy_only=True):
        """各原子在实验室系中的均方位移 (相对于轨迹平均位置)，取 (重原子) 平均，即模拟的 U_eq / Å²。
        先扣除整个超胞的质心漂移 (Langevin 下整体平动是无约束的随机游走，不属于晶格振动)。"""
        c = frames["c"] - frames["c"].mean(1, keepdims=True)
        X = np.einsum("tnij,naj->tnai", frames["Q"], self.Yn) + c[:, :, None, :]
        u = ((X - X.mean(0)) ** 2).sum(-1).mean(0) / 3.0
        hv = np.array([e != "H" for e in self.sym])
        return float(u[:, hv].mean() if heavy_only else u.mean())

    def save(self, path, frames):
        np.savez_compressed(path, c=frames["c"].astype(np.float32), Q=frames["Q"].astype(np.float32), E=frames["E"],
                            T_tr=frames["T_tr"], T_rot=frames["T_rot"], R=self.R, com=self.com, Y_body=self.Y,
                            mirror_n=self.mirror_n,
                            symbols=np.array(self.sym), supercell=np.array(self.sc), lattice=self.lat,
                            dt_frame_fs=frames.get("dt_frame_fs", 20.0))


def load_frames(path):
    d = np.load(path)
    return {"c": d["c"].astype(float), "Q": d["Q"].astype(float), "E": d["E"], "T_tr": d["T_tr"],
            "T_rot": d["T_rot"], "dt_frame_fs": float(d["dt_frame_fs"]) if "dt_frame_fs" in d else 20.0}


def md_pair_params(md, frames, pair, g_matrix, recenter=True, q0=None):
    """轨迹 → 某唯一二聚体全部超胞实例的参数 q (帧数×实例数, 6)。实例：每个晶胞中的 (ref → mol + translation)。
    recenter=True 时把平均结构移回实验结构 q0 (与 CEDS 的 MC 相同：力场只决定涨落)。返回 (q, 力场平均 q)。"""
    from .ceds import _wrap, pair_params
    A = np.array([md.index(cl, pair["ref"]) for cl in md.cells])
    t = np.array(pair["translation"])
    B = np.array([md.index(tuple(np.array(cl) + t), pair["mol"]) for cl in md.cells])
    shift = np.array([(np.array(cl) + t) // np.array(md.sc) for cl in md.cells]) @ md.box   # 跨越超胞边界的平移
    Qm, o = md.mol_frames(frames["c"], frames["Q"])
    QA, QB = Qm[:, A], Qm[:, B] @ g_matrix
    cA, cB = o[:, A], o[:, B] + shift
    q = pair_params(QA, cA, QB, cB).reshape(-1, 6)
    q_raw_mean = q.mean(0)
    if q0 is not None:
        for i in (3, 4, 5):                                     # 角度均值按 q0 附近展开
            q_raw_mean[i] = q0[i] + _wrap(q[:, i] - q0[i]).mean()
    if recenter and q0 is not None:
        dq = q - q.mean(0)
        for i in (3, 4, 5):
            dq[:, i] = _wrap(q[:, i] - q_raw_mean[i])
        q = np.asarray(q0, float) + dq
    return q, q_raw_mean


def autocorr_time(V, dt_fs, floor=0.05):
    """V: (帧, 实例)。各实例归一化自相关函数取平均，积分到首次低于 floor 处 → 相关时间 / fs。"""
    x = V - V.mean(0)
    n = len(x)
    if n < 4 or not np.any(x):
        return float("nan"), np.ones(1)
    f = np.fft.rfft(x, 2 * n, axis=0)
    a = np.fft.irfft(f * np.conj(f), axis=0)[:n].mean(1)
    a = a / a[0]
    cut = int(np.argmax(a < floor)) or n
    return float(a[:cut].sum() * dt_fs), a[:min(n, 100)]


def md_couplings(ceds, md, frames, surfaces, T, recenter=True, vmin=3.0, max_eval_eht=1200):
    """一段轨迹 → 每个唯一二聚体的 ⟨V⟩、σ_V、参数分配、相关时间；格式与 CEDS.run 相同 (可直接给 tlt / export_csv)。
    EHT 曲面逐点计算较慢：等间隔抽帧，使样本数不超过 max_eval_eht (相关时间按抽帧后的间隔计算，较粗)。"""
    from .ceds import CARRIERS, OPS, PARAM_KEYS
    nf0 = len(frames["c"])
    dtf0 = float(frames.get("dt_frame_fs", 20.0))
    ue = md.u_eq(frames)
    res = {}
    for p in ceds.pairs:
        surf = surfaces(p["gid"], p["center"]) if callable(surfaces) else surfaces.get(p["gid"])
        if surf is None:
            res[p["gid"]] = {"gid": p["gid"], "skipped": "没有该二聚体的耦合曲面"}
            continue
        v0 = {c: float(ceds.chi(c, p["g"]) * surf(p["q0"], c)[0]) for c in CARRIERS}
        if max(abs(v) for v in v0.values()) < vmin:
            res[p["gid"]] = {"gid": p["gid"], "T": T, "source": surf.source, "weak": True,
                             **{c: {"V0": v0[c], "mean": v0[c], "std": 0.0, "ratio": 0.0, "samples": np.array([v0[c]]),
                                    "share": {}, "chi": ceds.chi(c, p["g"]), "tau_fs": float("nan")} for c in CARRIERS}}
            continue
        G = np.diag(OPS[p["g"]]).astype(float)
        fr, nf, dtf = frames, nf0, dtf0
        if surf.source == "eht":
            stride = max(1, int(np.ceil(nf0 * len(md.cells) / max_eval_eht)))
            fr = {"c": frames["c"][::stride], "Q": frames["Q"][::stride]}
            nf, dtf = len(fr["c"]), dtf0 * stride
        q, qm = md_pair_params(md, fr, p, G, recenter=recenter, q0=p["q0"])
        out = {"gid": p["gid"], "T": T, "source": surf.source, "surface": surf.describe(),
               "frozen": list(getattr(surf, "frozen", [])), "coverage": surf.coverage(q),
               "q_std": dict(zip(PARAM_KEYS, q.std(0))), "q0": p["q0"],
               "q_shift": dict(zip(PARAM_KEYS, qm - np.asarray(p["q0"], float))),
               "mc": {"U_eq": [ue, ue]}, "n_samples": len(q)}
        for c in CARRIERS:
            chi = ceds.chi(c, p["g"])
            V = chi * surf(q, c)
            m, s = float(V.mean()), float(V.std(ddof=1))
            act = [i for i, k in enumerate(PARAM_KEYS) if k not in out["frozen"] and q[:, i].std() > 1e-9]
            share = {}
            if act:
                dq = q[:, act] - q[:, act].mean(0)
                gk, *_ = np.linalg.lstsq(dq, V - m, rcond=None)
                con = gk ** 2 * dq.var(0)
                tot = con.sum() or 1.0
                share = {PARAM_KEYS[i]: float(con[j] / tot) for j, i in enumerate(act)}
            tau, _ = autocorr_time(V.reshape(nf, -1), dtf)
            out[c] = {"V0": v0[c], "mean": m, "std": s, "ratio": s / max(abs(m), 1e-9), "samples": V, "share": share,
                      "chi": chi, "tau_fs": tau}
        res[p["gid"]] = out
    return res


def run_temperature(md, T, opts=None):
    """一个温度：平衡 + 采样 (随机数种子 T 与 T+1，与 md_temperature_series 相同)。返回采样轨迹。"""
    o = dict(DEFAULTS, **(opts or {}))
    n_eq = int(round(o["equil_ps"] * 1000 / o["dt"]))
    n_pr = int(round(o["prod_ps"] * 1000 / o["dt"]))
    _, st = md.run(float(T), n_eq, o["dt"], o["gamma_eq"], every=max(n_eq, 1), seed=int(T))
    fr, _ = md.run(float(T), n_pr, o["dt"], o["gamma"], every=o["every"], seed=int(T) + 1, state=st)
    return fr


def md_temperature_series(ceds, surfaces, temps, monomer=None, supercell=None, equil_ps=None, prod_ps=None, dt=None,
                          every=None, hbar_tau=0.005, hbar_taus=(), n_real=16, recenter=True, static=True,
                          save_dir=None, on_progress=None, md=None, **md_kw):
    """逐温度：全晶体刚体 MD → σ_V、τ_c → TLT μ(T)。默认设置见 DEFAULTS，随机数种子为 T 与 T+1。
    hbar_taus: 另外再算的 ħ/τ (eV) 列表 (敏感性检验)。返回 (rows, md)。rows 与 ceds.temperature_series 格式相同，
    另含 U_eq、E_lat、T_tr、T_rot、tau_c_fs、hbar_tau_md_meV、mu_tau (各 ħ/τ 的 μ)。"""
    import os
    o = dict(DEFAULTS)
    for k, v in dict(supercell=supercell, equil_ps=equil_ps, prod_ps=prod_ps, dt=dt, every=every).items():
        if v is not None:
            o[k] = v
    o.update({k: v for k, v in md_kw.items() if v is not None})
    if md is None:
        md = RigidCrystalMD(ceds.crys, ceds.frames, monomer or (ceds.sym, ceds.Y), o["supercell"],
                            rc=o["rc"], r_sw=o["r_sw"], skin=o["skin"])
    rows = []
    for T in temps:
        T = float(T)
        if on_progress:
            on_progress(f"MD T = {T:g} K")
        fr = run_temperature(md, T, o)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            md.save(os.path.join(save_dir, f"md_{T:g}K.npz"), fr)
        res = md_couplings(ceds, md, fr, surfaces, T, recenter=recenter)
        taus = [r[c]["tau_fs"] for r in res.values() if "hole" in r and not r.get("weak") for c in ("hole", "electron")]
        taus = [t for t in taus if np.isfinite(t)]
        tc = float(np.mean(taus)) if taus else float("nan")
        row = {"T": T, "res": res, "U_eq": md.u_eq(fr), "E_lat": float(np.mean(fr["E"]) / md.N),
               "T_tr": float(np.mean(fr["T_tr"])), "T_rot": float(np.mean(fr["T_rot"])),
               "tau_c_fs": tc, "hbar_tau_md_meV": HBAR_MEV_FS / tc if tc == tc else float("nan"), "mu_tau": {}}
        for c in ("hole", "electron"):
            t = ceds.tlt(res, c, T, hbar_tau, n_real=n_real)
            row[f"mu_{c}"] = float(t["mu_principal"][0])
            row[f"mu2_{c}"] = float(t["mu_principal"][1]) if len(t["mu_principal"]) > 1 else float("nan")
            row[f"L_{c}"] = t["L_A"]
            if static:
                t0 = ceds.tlt(res, c, T, hbar_tau, n_real=1, sigma_scale=0.0)
                row[f"mu_static_{c}"] = float(t0["mu_principal"][0])
            for ht in hbar_taus:
                tt = ceds.tlt(res, c, T, ht, n_real=n_real)
                row["mu_tau"][f"{c}_{ht * 1000:g}"] = float(tt["mu_principal"][0])
        rows.append(row)
    return rows, md


def md_summary(rows):
    """MD 结果的文字表。"""
    lines = [f"{'T/K':>6}{'U_eq/Å²':>9}{'μh':>9}{'μh(σ=0)':>9}{'μe':>9}{'μe(σ=0)':>9}{'τc/fs':>8}{'ħ/τc/meV':>9}"]
    for r in rows:
        lines.append(f"{r['T']:6.0f}{r['U_eq']:9.4f}{r['mu_hole']:9.3g}{r.get('mu_static_hole', float('nan')):9.3g}"
                     f"{r['mu_electron']:9.3g}{r.get('mu_static_electron', float('nan')):9.3g}"
                     f"{r['tau_c_fs']:8.0f}{r['hbar_tau_md_meV']:9.2f}")
    if len(rows) >= 2:
        T_ = np.log([r["T"] for r in rows])
        for c, lab in (("hole", "空穴"), ("electron", "电子")):
            lines.append(f"{lab}: μ ∝ T^{np.polyfit(T_, np.log([r[f'mu_{c}'] for r in rows]), 1)[0]:.2f}")
    return "\n".join(lines)


def export_md_csv(ceds, rows, path):
    """CEDS 格式的 σ_V 表 + μ(T) 表，再加 MD 专有的一节 (U_eq、晶格能、相关时间、力场平均结构偏移、各 ħ/τ 的 μ)。"""
    import csv
    from .ceds import PARAM_KEYS, export_csv
    export_csv(ceds, {r["T"]: r["res"] for r in rows}, path, rows)
    keys = sorted({k for r in rows for k in r["mu_tau"]})
    with open(path, "a", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow([])
        w.writerow(["T_K", "U_eq_MD_A2", "E_lat_kcal_mol", "T_tr_K", "T_rot_K", "tau_c_fs", "hbar_tau_MD_meV"]
                   + [f"mu_{k}meV" for k in keys])
        for r in rows:
            w.writerow([r["T"], f"{r['U_eq']:.5f}", f"{r['E_lat']:.3f}", f"{r['T_tr']:.1f}", f"{r['T_rot']:.1f}",
                        f"{r['tau_c_fs']:.1f}", f"{r['hbar_tau_md_meV']:.3f}"] + [f"{r['mu_tau'][k]:.4g}" for k in keys])
        w.writerow([])
        w.writerow(["T_K", "group", "carrier", "tau_c_fs"] + [f"shift_{k}" for k in PARAM_KEYS])
        for r in rows:
            for g, x in r["res"].items():
                if "q_shift" not in x:
                    continue
                for c in ("hole", "electron"):
                    w.writerow([r["T"], g, c, f"{x[c]['tau_fs']:.1f}"] + [f"{x['q_shift'][k]:.4f}" for k in PARAM_KEYS])
    return path


def plot_md_mu_T(ax, rows):
    """MD 的 μ(T) (实线) 与 σ = 0 (虚线)。"""
    from .ceds import plot_mu_T
    plot_mu_T(ax, rows)
    for ln in ax.get_lines():
        ln.set_label(ln.get_label().replace("CEDS + TLT", "MD + TLT"))
    ax.legend(fontsize=7)


def md_half(ceds, md, frames, gids=None, k=3.0, floor=None):
    """由全晶体 MD 的参数涨落确定局域扫描范围：±k·σ_MD (不低于 floor)。返回 {组号: {参数: 半宽}}。"""
    from .ceds import OPS, PARAM_KEYS
    floor = dict({"z": 0.12, "dx": 0.15, "dy": 0.15, "gamma": 2.0, "pitch": 2.0, "roll": 2.0}, **(floor or {}))
    out = {}
    for p in ceds.pairs:
        if gids is not None and p["gid"] not in gids:
            continue
        q, _ = md_pair_params(md, frames, p, np.diag(OPS[p["g"]]).astype(float), recenter=True, q0=p["q0"])
        sd = q.std(0)
        out[p["gid"]] = {key: float(max(k * sd[i], floor[key])) for i, key in enumerate(PARAM_KEYS)}
    return out


# ----------------------------------------------------------------------------------------------
# MD 快照：逐帧直接计算 V，检验耦合曲面
# ----------------------------------------------------------------------------------------------
def snapshot_configs(ceds, md, frames, gids, n_per_pair=120, seed=2026, recenter=True):
    """从轨迹中为每个二聚体组取 n_per_pair 个构型。帧在整段轨迹上等间隔分布 (减小相邻帧的相关)，
    每帧随机取一个超胞实例。参数与 md_couplings 送入曲面的完全相同 (recenter 一致)，
    因此 DFT 结果可与曲面在同一批几何上逐点比较。返回构型字典列表 (group = 组号，variant = 帧/实例)。"""
    from .ceds import OPS, PARAM_KEYS
    from .scanflow import make_cfg
    rng = np.random.default_rng(seed)
    nf, ni = len(frames["c"]), len(md.cells)
    out = []
    for p in ceds.pairs:
        if p["gid"] not in gids:
            continue
        G = np.diag(OPS[p["g"]]).astype(float)
        q, _ = md_pair_params(md, frames, p, G, recenter=recenter, q0=p["q0"])
        q = q.reshape(nf, ni, 6)
        n = min(int(n_per_pair), nf * ni)
        fi = np.unique(np.linspace(0, nf - 1, n).round().astype(int))
        while len(fi) < n:                                       # 帧数少于所需构型数时，同一帧取多个实例
            fi = np.sort(np.concatenate([fi, rng.choice(nf, n - len(fi))]))
        mir = {"mirror": 1} if p.get("mirror") else {}
        for f in fi:
            i = int(rng.integers(ni))
            c = make_cfg(*q[f, i, :3], gamma=q[f, i, 3], pitch=q[f, i, 4], roll=q[f, i, 5], **mir)
            out.append(dict(c, group=str(p["gid"]), variant=f"f{int(f)}i{i}"))
    return out


def snapshot_validation(ceds, records, surfaces, n_boot=2000, seed=7):
    """records：MD 快照扫描的结果 (含 group 与 V)。对每个组比较直接计算的 V 与曲面在同一几何上的预测：
    逐点误差 (MAE、RMSE、最大误差、R²) 与样本统计 (⟨V⟩、σ_V，及 σ 比值的自助法 95% 区间)。"""
    from .ceds import CARRIERS, PARAM_KEYS
    rng = np.random.default_rng(seed)
    by = {}
    for r in records:
        g = str(r.get("group", "")).strip()
        if g and all(r.get(c) is not None for c in CARRIERS):
            by.setdefault(int(float(g)), []).append(r)
    out = {}
    for p in ceds.pairs:
        rows = by.get(p["gid"])
        surf = surfaces(p["gid"], p["center"]) if callable(surfaces) else surfaces.get(p["gid"])
        if not rows or surf is None:
            continue
        Q = np.array([[r[k] for k in PARAM_KEYS] for r in rows], float)
        res = {"gid": p["gid"], "n": len(rows), "surface": surf.describe(), "coverage": surf.coverage(Q), "q": Q}
        for c in CARRIERS:
            vd = np.array([r[c] for r in rows], float)
            vs = np.asarray(surf(Q, c), float)
            e = vs - vd
            ss = ((vd - vd.mean()) ** 2).sum()
            idx = rng.integers(len(vd), size=(n_boot, len(vd)))
            ratio = vs[idx].std(1, ddof=1) / np.maximum(vd[idx].std(1, ddof=1), 1e-12)
            res[c] = {"V_dft": vd, "V_surf": vs, "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                      "max_err": float(np.abs(e).max()), "R2": float(1 - (e ** 2).sum() / ss) if ss > 0 else float("nan"),
                      "mean_dft": float(vd.mean()), "mean_surf": float(vs.mean()),
                      "sigma_dft": float(vd.std(ddof=1)), "sigma_surf": float(vs.std(ddof=1)),
                      "sigma_ratio": float(vs.std(ddof=1) / max(vd.std(ddof=1), 1e-12)),
                      "sigma_ratio_ci": tuple(float(x) for x in np.percentile(ratio, [2.5, 97.5])),
                      "corr": float(np.corrcoef(vd, vs)[0, 1]) if len(vd) > 2 else float("nan")}
        out[p["gid"]] = res
    return out


def validation_table(val):
    lines = [f"{'组':<4}{'载流子':<6}{'n':>5}{'⟨V⟩DFT':>9}{'⟨V⟩曲面':>9}{'σDFT':>8}{'σ曲面':>8}{'σ比值':>7}"
             f"{'95% 区间':>15}{'MAE':>7}{'RMSE':>7}{'R²':>7}{'覆盖':>6}"]
    for g, r in val.items():
        for c, lab in (("hole", "空穴"), ("electron", "电子")):
            d = r[c]
            lo, hi = d["sigma_ratio_ci"]
            lines.append(f"G{g:<3}{lab:<6}{r['n']:5d}{d['mean_dft']:9.2f}{d['mean_surf']:9.2f}{d['sigma_dft']:8.2f}"
                         f"{d['sigma_surf']:8.2f}{d['sigma_ratio']:7.2f}{f'{lo:.2f}–{hi:.2f}':>15}{d['mae']:7.2f}"
                         f"{d['rmse']:7.2f}{d['R2']:7.3f}{r['coverage']:6.2f}")
    return "\n".join(lines)


def export_validation_csv(val, path):
    import csv
    from .ceds import PARAM_KEYS
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "carrier", "n", "mean_dft", "mean_surf", "sigma_dft", "sigma_surf", "sigma_ratio",
                    "ratio_ci_lo", "ratio_ci_hi", "mae", "rmse", "max_err", "R2", "corr", "coverage"])
        for g, r in val.items():
            for c in ("hole", "electron"):
                d = r[c]
                w.writerow([g, c, r["n"]] + [f"{d[k]:.4f}" for k in ("mean_dft", "mean_surf", "sigma_dft", "sigma_surf",
                                                                       "sigma_ratio")]
                           + [f"{x:.4f}" for x in d["sigma_ratio_ci"]]
                           + [f"{d[k]:.4f}" for k in ("mae", "rmse", "max_err", "R2", "corr")] + [f"{r['coverage']:.3f}"])
        w.writerow([])
        w.writerow(["group"] + list(PARAM_KEYS) + ["V_dft_hole", "V_surf_hole", "V_dft_electron", "V_surf_electron"])
        for g, r in val.items():
            for k, q in enumerate(r["q"]):
                w.writerow([g] + [f"{x:.5f}" for x in q] + [f"{r[c][s][k]:.3f}" for c in ("hole", "electron")
                                                          for s in ("V_dft", "V_surf")])
    return path


def plot_validation(axes, val):
    """axes: 两个子图 (空穴、电子)。逐点比较曲面预测与直接计算的 V，各组不同标记。"""
    from .scananalysis import COL
    marks = "osD^v<>p"
    for ax, c in zip(axes, ("hole", "electron")):
        allv = []
        for k, (g, r) in enumerate(val.items()):
            d = r[c]
            ax.plot(d["V_dft"], d["V_surf"], marks[k % len(marks)], ms=3.5, mfc="none", color=COL[c], alpha=0.8,
                    label=f"G{g}: RMSE {d['rmse']:.1f} meV, σ ratio {d['sigma_ratio']:.2f}")
            allv += list(d["V_dft"]) + list(d["V_surf"])
        if allv:
            lo, hi = min(allv), max(allv)
            pad = 0.05 * (hi - lo or 1)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", color="0.5", lw=0.8)
        ax.set_xlabel("V, direct DFT (meV)")
        ax.set_ylabel("V, coupling surface (meV)")
        ax.set_title("Holes" if c == "hole" else "Electrons", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False)
