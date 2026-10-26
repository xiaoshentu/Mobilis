"""基于滑移扫描的转移积分代理模型与堆积反向设计。

坐标约定与 make_dimers.py 完全相同：
    分子坐标系：原点 = 重原子几何中心，x = 长轴，y = 短轴，z = 平面法线 (三阶矩固定符号，右手系)
    二聚体：A 在原点；B 先绕自身中心转动 R = Rz(γ)·Ry(pitch)·Rx(roll)，再平移 (Δx, Δy, z)
    相位一致性：A、B 引用同一个单体片段 (f=mono/1、f=mono/2)，V 的符号可比

代理模型 (V 单位 meV，带符号)：
    M1 可分离    V(z,Δx,Δy) = F_ref(Δx,Δy) · exp[−β (z − z_ref)]          (参考距离的滑移网格 + 距离衰减)
    M2 多距离    在两张滑移网格之间用局部衰减常数 β(Δx,Δy) = ln(V1/V2)/(z2 − z1) 插值 (含距离–滑移耦合)
    M3 EHT 标定  V = a · V_EHT(z,Δx,Δy,γ,…) · exp[−δβ (z − z_ref)]               (可处理分子转动)
堆积设计：一维滑移堆积 (近邻 ±r1) 或二维砖墙 (±r1、±r2)，r = (Δx, Δy, z)；
    Marcus 速率 k = (2π/ħ)V²(4πλkT)^(-1/2) exp(−λ/4kT)，迁移率张量 μ = Σ k r rᵀ / (2kT)，
    在“不碰撞”(d/(r_vdW,A + r_vdW,B) ≥ 阈值) 与“不超出数据范围”约束下全局优化滑移。
"""
import csv
import json
from pathlib import Path

import numpy as np

from .mobility import KB_EV, marcus_rate

VDW = {"H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "S": 1.80, "Cl": 1.75, "Br": 1.85,
       "Si": 2.10, "P": 1.80, "Se": 1.90, "I": 1.98, "Ir": 2.00, "Pt": 1.75}
CARRIERS = ("hole", "electron")
METHODS = {"M1": "M1 可分离 (参考滑移网格 × 距离衰减)", "M2": "M2 多距离 (局部衰减，含距离–滑移耦合)",
           "M3": "M3 EHT 标定 (可处理转动)"}


# ----------------------------------------------------------------------------------------------
# 几何 (与 make_dimers.py 相同)
# ----------------------------------------------------------------------------------------------
def molecular_frame(symbols, xyz, core=None):
    """返回 (分子坐标系下坐标 Y, 平面偏离/Å, 坐标轴 (列), 原点)。
    原点与坐标轴由 core 中的原子 (默认全部重原子) 确定：原点 = 几何中心，x = 长轴，y = 短轴，z = 法线。"""
    xyz = np.asarray(xyz, float)
    if core is None:
        sel = np.array([e != "H" for e in symbols])
    else:
        sel = np.zeros(len(symbols), bool)
        sel[list(core)] = True
    c = xyz[sel].mean(axis=0)
    X = xyz - c
    w, v = np.linalg.eigh(np.cov(X[sel].T))
    ax = v[:, ::-1].copy()
    for k in range(2):
        proj = X[sel] @ ax[:, k]
        s = np.sum(proj ** 3)
        if abs(s) < 1e-6:
            s = proj[np.argmax(np.abs(proj))]
        if s < 0:
            ax[:, k] *= -1
    ax[:, 2] = np.cross(ax[:, 0], ax[:, 1])
    Y = X @ ax
    return Y, float(np.abs(Y[sel, 2]).max()), ax, c


FRAME_MODES = ("all", "core")


def _bond_graph(symbols, xyz, heavy_only=True):
    from .gaussian_io import COVALENT_RADII
    xyz = np.asarray(xyz, float)
    idx = [i for i, e in enumerate(symbols) if not heavy_only or e != "H"]
    r = np.array([COVALENT_RADII.get(symbols[i], 0.8) for i in idx])
    d = np.linalg.norm(xyz[idx][:, None] - xyz[idx][None], axis=-1)
    a, b = np.nonzero(np.triu(d < 1.15 * (r[:, None] + r[None]), 1))
    adj = {i: set() for i in idx}
    for p, q in zip(a, b):
        adj[idx[p]].add(idx[q])
        adj[idx[q]].add(idx[p])
    return adj


def ring_core(symbols, xyz):
    """最大的稠环体系 (由环键连通的重原子)：去掉桥键 (不在任何环上的键) 后最大的连通分量。
    没有环时返回全部重原子。用于以共轭主体定义分子坐标系 (例如红荧烯的并四苯骨架，不含苯基)。"""
    adj = _bond_graph(symbols, xyz)
    nodes = sorted(adj)
    disc, low, bridges, t = {}, {}, set(), [0]

    def dfs(u, parent):                              # Tarjan 桥算法
        disc[u] = low[u] = t[0]
        t[0] += 1
        for v in adj[u]:
            if v not in disc:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.add(frozenset((u, v)))
            elif v != parent:
                low[u] = min(low[u], disc[v])
    import sys
    lim = sys.getrecursionlimit()
    sys.setrecursionlimit(max(lim, 10 * len(nodes) + 100))
    try:
        for u in nodes:
            if u not in disc:
                dfs(u, None)
    finally:
        sys.setrecursionlimit(lim)
    seen, comps = set(), []
    for u in nodes:
        if u in seen:
            continue
        comp, st = set(), [u]
        while st:
            x = st.pop()
            if x in comp:
                continue
            comp.add(x)
            st += [y for y in adj[x] if frozenset((x, y)) not in bridges and y not in comp]
        seen |= comp
        comps.append(sorted(comp))
    rings = [c for c in comps if len(c) >= 3]
    if not rings:
        return nodes
    xyz = np.asarray(xyz, float)
    cen = xyz[nodes].mean(0)
    return min(rings, key=lambda c: (-len(c), np.linalg.norm(xyz[c].mean(0) - cen)))


def frame_atoms(symbols, xyz, mode="all"):
    if mode == "core":
        return ring_core(symbols, xyz)
    if mode != "all":
        raise ValueError(f"未知的坐标系定义 {mode} (可选 all / core)")
    return None


def _matched_rmsd(ref_symbols, P, symbols, xyz):
    from scipy.optimize import linear_sum_assignment
    ref_symbols, symbols = np.asarray(ref_symbols), np.asarray(symbols)
    tot, n = 0.0, 0
    for e in set(ref_symbols.tolist()):
        if e == "H":
            continue
        i, j = np.where(ref_symbols == e)[0], np.where(symbols == e)[0]
        D = np.linalg.norm(P[i][:, None] - xyz[j][None], axis=-1)
        r, c = linear_sum_assignment(D)
        tot += (D[r, c] ** 2).sum()
        n += len(r)
    return float(np.sqrt(tot / max(n, 1)))


def matched_frame(ref_symbols, ref_Y, symbols, xyz, mode="all"):
    """晶体中某个分子的分子坐标系 (ax, 原点)，使参考单体 ref_Y (已在分子坐标系中) 经 ref_Y·axᵀ + 原点 与该分子重合。
    主轴的符号约定对只有部分对称性的分子 (如 C2h 的红荧烯) 不能保证各分子一致，因此在四种保持手性的符号取法中
    选重原子匹配 RMSD 最小的一种 (与默认取法相差不到 0.02 Å 时保留默认)。返回 (ax, 原点, RMSD/Å)。"""
    xyz = np.asarray(xyz, float)
    ref_Y = np.asarray(ref_Y, float)
    _, _, ax, c = molecular_frame(symbols, xyz, frame_atoms(symbols, xyz, mode))
    best = None
    for s in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        a = ax * np.array(s, float)
        r = _matched_rmsd(ref_symbols, ref_Y @ a.T + c, symbols, xyz)
        if best is None:
            best, r0 = (r, a), r
        elif r < best[0]:
            best = (r, a)
    # 对映体：非真转动 (第三轴取反)。只有明显好于真转动时才采用 (非手性分子的镜像与自身重合，不会被选中)
    bm = None
    for s in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        a = ax * np.array(s, float) * MIRROR
        r = _matched_rmsd(ref_symbols, ref_Y @ a.T + c, symbols, xyz)
        if bm is None or r < bm[0]:
            bm = (r, a)
    if bm[0] < 0.5 * best[0] and best[0] > 0.3:
        return bm[1], c, bm[0]
    if r0 <= best[0] + 0.02:
        return ax, c, r0
    return best[1], c, best[0]


def rot(axis, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation(gamma=0.0, roll=0.0, pitch=0.0):
    return rot("z", gamma) @ rot("y", pitch) @ rot("x", roll)


# 对映体 (镜像) 分子：在分子坐标系中把第三个坐标取反。手性分子 (如 mer-Alq3、Ir(C^N)2(acac)) 的中心对称晶体中
# 一半分子是参考单体的镜像，不能只用转动 + 平移重合；这时坐标系 ax 取非真转动 (det = −1)，二聚体参数 q 只描述
# 真转动部分，另加 mirror 标记，B 用镜像单体构建 (ADF 片段也用镜像单体的片段文件 mono_m.t21)。
MIRROR = np.array([1.0, 1.0, -1.0])


def is_mirror(c):
    """构型字典的 mirror 标记 (manifest 中为 "1"/"" 或布尔)。"""
    v = c.get("mirror", "") if isinstance(c, dict) else c
    return str(v).strip().lower() in ("1", "true", "yes", "1.0")


def build_B(Y, z, dx, dy, gamma=0.0, roll=0.0, pitch=0.0, mirror=False):
    Y = np.asarray(Y, float)
    if mirror:
        Y = Y * MIRROR
    return Y @ rotation(gamma, roll, pitch).T + np.array([dx, dy, z])


def clash_ratio(symbols, A, B):
    """(最短原子间距, 最小 d/(rA+rB))。"""
    A, B = np.asarray(A), np.asarray(B)
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    r = np.array([VDW.get(e, 1.7) for e in symbols])
    return float(d.min()), float((d / (r[:, None] + r[None, :])).min())


# ----------------------------------------------------------------------------------------------
# 数据：读入 make_dimers.py / 本程序生成的扫描目录，或 CSV
# ----------------------------------------------------------------------------------------------
def _rec(z, dx, dy, vh, ve, gamma=0.0, roll=0.0, pitch=0.0, rid="", source=""):
    return {"id": rid, "z": float(z), "dx": float(dx), "dy": float(dy), "gamma": float(gamma), "roll": float(roll),
            "pitch": float(pitch), "hole": None if vh is None else float(vh),
            "electron": None if ve is None else float(ve), "source": source}


def load_scan_dir(d):
    """读入扫描目录：manifest.csv (id,z,dx,dy,gamma,roll,pitch,…,status) + outputs/<id>.out (ADF 输出)。"""
    from .adf import _organize, parse_adf_output
    d = Path(d)
    man = d / "manifest.csv"
    if not man.exists():
        raise FileNotFoundError(f"{d} 中没有 manifest.csv (make_dimers.py 或本程序生成的扫描目录)")
    recs, missing = [], 0
    with open(man, encoding="utf-8-sig", errors="ignore") as fh:
        for row in csv.DictReader(fh):
            if row.get("status", "ok") != "ok":
                continue
            rid = row["id"]
            out = next((p for p in (d / "outputs" / f"{rid}.out", d / f"{rid}.out") if p.exists()), None)
            if out is None:
                missing += 1
                continue
            r = _organize(parse_adf_output(out.read_text(errors="ignore")))
            vh, ve = r.get("hole", {}).get("V_meV"), r.get("electron", {}).get("V_meV")
            if vh is None and ve is None:
                missing += 1
                continue
            recs.append(_rec(row["z"], row["dx"], row["dy"], vh, ve, row.get("gamma", 0) or 0, row.get("roll", 0) or 0,
                             row.get("pitch", 0) or 0, rid, d.name))
    return recs, missing


def load_csv(path):
    """CSV：列 z, dx, dy, [gamma, roll, pitch], V_hole/Vh/hole, V_electron/Ve/electron (meV)。"""
    recs = []
    with open(path, encoding="utf-8-sig", errors="ignore") as fh:
        rd = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in rd.fieldnames or []}

        def pick(*names):
            return next((cols[n] for n in names if n in cols), None)
        ch = pick("v_hole", "vh", "hole", "v_hole_mev", "vh_mev")
        ce = pick("v_electron", "ve", "electron", "v_electron_mev", "ve_mev")
        if not (ch or ce) or not all(pick(k) for k in ("z", "dx", "dy")):
            raise ValueError("CSV 需要列 z, dx, dy 以及 V_hole / V_electron (meV)")
        for row in rd:
            g = lambda k: float(row[pick(k)]) if pick(k) and row[pick(k)] not in ("", None) else 0.0  # noqa: E731
            vh = float(row[ch]) if ch and row[ch] not in ("", None) else None
            ve = float(row[ce]) if ce and row[ce] not in ("", None) else None
            recs.append(_rec(g("z"), g("dx"), g("dy"), vh, ve, g("gamma"), g("roll"), g("pitch"), source=Path(path).name))
    return recs


def save_csv(recs, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "z", "dx", "dy", "gamma", "roll", "pitch", "V_hole_meV", "V_electron_meV", "source"])
        for r in recs:
            w.writerow([r["id"], r["z"], r["dx"], r["dy"], r["gamma"], r["roll"], r["pitch"], r["hole"], r["electron"],
                        r["source"]])


# ----------------------------------------------------------------------------------------------
# 软件内 EHT：秒级扫描 (相位一致：所有构型共用同一套单体轨道)
# ----------------------------------------------------------------------------------------------
class EHTCoupler:
    def __init__(self, symbols, Y):
        from .eht import build_basis, monomer
        self.sym, self.Y = list(symbols), np.asarray(Y, float)
        self.mono = monomer(self.sym, self.Y)
        self.atoms_a = [(s, *p) for s, p in zip(self.sym, self.Y)]
        self._bb = build_basis

    def __call__(self, z, dx, dy, gamma=0.0, roll=0.0, pitch=0.0):
        from .eht import eht_transfer_integrals, rotate_coefficients
        R = rotation(gamma, roll, pitch)
        B = build_B(self.Y, z, dx, dy, gamma, roll, pitch)
        shells, _, _ = self._bb(self.sym, B)
        mb = dict(self.mono, shells=shells)
        if gamma or roll or pitch:
            mb["C"] = rotate_coefficients(self.mono["shells"], self.mono["C"], R)
        r = eht_transfer_integrals(self.atoms_a, [(s, *p) for s, p in zip(self.sym, B)], esd=False, _mono_b=mb)["results"]
        return r["HOMO"]["J_eff_meV"], r["LUMO"]["J_eff_meV"]


def eht_scan(symbols, Y, zs, dxs, dys, clash_min=0.75, on_progress=None):
    ec = EHTCoupler(symbols, Y)
    recs, n, tot = [], 0, len(zs) * len(dxs) * len(dys)
    for z in zs:
        for dx in dxs:
            for dy in dys:
                n += 1
                if clash_ratio(symbols, Y, build_B(Y, z, dx, dy))[1] < clash_min:
                    continue
                vh, ve = ec(z, dx, dy)
                recs.append(_rec(z, dx, dy, vh, ve, rid=f"eht_{n:05d}", source="EHT"))
                if on_progress and n % 50 == 0:
                    on_progress(n, tot)
    return recs


# ----------------------------------------------------------------------------------------------
# 代理模型
# ----------------------------------------------------------------------------------------------
class CouplingModel:
    def __init__(self, records, symmetry="D2h", monomer=None):
        self.records = [r for r in records]
        self.symmetry = symmetry
        self.monomer = monomer                         # (symbols, Y)：碰撞判断与 M3 需要
        self.grids, self.splines, self.beta, self.calib = {}, {}, {}, {}
        self.z_ref = None
        self._eht = None

    # ---- 网格 ----
    def _translational(self):
        return [r for r in self.records if abs(r["gamma"]) + abs(r["roll"]) + abs(r["pitch"]) < 1e-9]

    def build_grids(self, min_points=16):
        from scipy.interpolate import griddata
        byz = {}
        for r in self._translational():
            byz.setdefault(round(r["z"], 3), []).append(r)
        grids = {}
        for z, rs in byz.items():
            xs = np.unique(np.round([r["dx"] for r in rs], 4))
            ys = np.unique(np.round([r["dy"] for r in rs], 4))
            if len(xs) < 4 or len(ys) < 4 or len(rs) < min_points:
                continue                                   # z 扫描等一维数据不构成网格
            g = {}
            for c in CARRIERS:
                pts = [(r["dx"], r["dy"], r[c]) for r in rs if r[c] is not None]
                if len(pts) < min_points:
                    continue
                P = np.array(pts)
                if self.symmetry == "D2h":                 # 相同分子的镜面对称: V(±Δx, ±Δy) 相同 (见说明)
                    P = np.vstack([P * [sx, sy, 1] for sx in (1, -1) for sy in (1, -1)])
                    P = np.unique(np.round(P, 6), axis=0)
                gx = np.unique(np.round(P[:, 0], 4))
                gy = np.unique(np.round(P[:, 1], 4))
                G = griddata(P[:, :2], P[:, 2], tuple(np.meshgrid(gx, gy, indexing="ij")), method="linear")
                if np.isnan(G).any():
                    G2 = griddata(P[:, :2], P[:, 2], tuple(np.meshgrid(gx, gy, indexing="ij")), method="nearest")
                    G = np.where(np.isnan(G), G2, G)
                g[c] = (gx, gy, G)
            if g:
                grids[z] = g
        self.grids = grids
        return grids

    def fit(self):
        from scipy.interpolate import RectBivariateSpline
        if not self.grids:
            self.build_grids()
        if not self.grids:
            raise ValueError("数据中没有可用的滑移网格 (至少需要一个 z 下 4×4 以上的 Δx–Δy 网格)")
        self.splines = {}
        for z, g in self.grids.items():
            for c, (gx, gy, G) in g.items():
                k = 3 if len(gx) > 3 and len(gy) > 3 else 1
                self.splines[(z, c)] = RectBivariateSpline(gx, gy, G, kx=k, ky=k, s=0)
        self.z_ref = max(self.grids, key=lambda z: sum(len(v[0]) * len(v[1]) for v in self.grids[z].values()))
        self.beta = {c: self._fit_beta(c) for c in CARRIERS}
        if self.monomer is not None:
            try:
                self.calib = self._fit_eht()
            except Exception:  # noqa: BLE001
                self.calib = {}
        return self

    def _fit_beta(self, c):
        """β：优先用 z 扫描 (Δx=Δy=0 处 ln|V| 对 z 线性拟合)，否则用两张网格的最小二乘比例因子。"""
        line = sorted((r["z"], r[c]) for r in self._translational()
                      if abs(r["dx"]) < 1e-6 and abs(r["dy"]) < 1e-6 and r[c] is not None and abs(r[c]) > 0.5)
        zs = np.unique([z for z, _ in line])
        if len(zs) >= 3:
            z, v = np.array(line).T
            b = -np.polyfit(z, np.log(np.abs(v)), 1)[0]
            return {"value": float(b), "source": f"z 扫描 ({len(z)} 点)"}
        sep = self.separability(c)
        if sep:
            return {"value": sep[0]["beta_eff"], "source": "两张滑移网格的比例因子"}
        return {"value": 1.5, "source": "默认值 (数据不足)"}

    def _fit_eht(self):
        if self._eht is None:
            self._eht = EHTCoupler(*self.monomer)
        out = {}
        sub = self.records if len(self.records) <= 400 else \
            [self.records[i] for i in np.random.default_rng(0).choice(len(self.records), 400, replace=False)]
        cache = [(r, self._eht(r["z"], r["dx"], r["dy"], r["gamma"], r["roll"], r["pitch"])) for r in sub]
        for k, c in enumerate(CARRIERS):
            data = [(r["z"], e[k], r[c]) for r, e in cache if r[c] is not None and abs(e[k]) > 1e-3]
            if len(data) < 5:
                continue
            z, ve, vd = map(np.array, zip(*data))
            best = None
            for db in np.linspace(-1.0, 1.0, 81):         # δβ 一维搜索 + a 的线性最小二乘
                f = ve * np.exp(-db * (z - self.z_ref))
                a = float(f @ vd / (f @ f))
                res = float(((a * f - vd) ** 2).sum())
                if best is None or res < best[0]:
                    best = (res, a, db)
            _, a, db = best
            pred = a * ve * np.exp(-db * (z - self.z_ref))
            r2 = 1 - ((pred - vd) ** 2).sum() / max(((vd - vd.mean()) ** 2).sum(), 1e-12)
            out[c] = {"a": a, "dbeta": float(db), "R2": float(r2), "n": len(z)}
        return out

    # ---- 预测 ----
    def zs(self, c):
        return sorted(z for z in self.grids if c in self.grids[z])

    def _grid_v(self, z, c, dx, dy):
        return self.splines[(z, c)].ev(dx, dy)

    def predict(self, z, dx, dy, carrier="hole", method="M2", gamma=0.0, roll=0.0, pitch=0.0):
        dx, dy = np.broadcast_arrays(np.asarray(dx, float), np.asarray(dy, float))
        z = np.broadcast_to(np.asarray(z, float), dx.shape)
        c = carrier
        if method == "M3" or (gamma or roll or pitch):
            if c not in self.calib:
                raise ValueError("M3 需要单体结构与 EHT 标定 (先“拟合模型”)")
            if self._eht is None:
                self._eht = EHTCoupler(*self.monomer)
            k = CARRIERS.index(c)
            cal = self.calib[c]
            v = np.array([self._eht(zz, xx, yy, gamma, roll, pitch)[k] for zz, xx, yy in
                          zip(z.ravel(), dx.ravel(), dy.ravel())]).reshape(dx.shape)
            return cal["a"] * v * np.exp(-cal["dbeta"] * (z - self.z_ref))
        b = self.beta[c]["value"]
        if method == "M1" or len(self.zs(c)) < 2:
            return self._grid_v(self.z_ref, c, dx, dy) * np.exp(-b * (z - self.z_ref))
        zs = self.zs(c)
        out = np.empty(dx.shape)
        for idx in np.ndindex(dx.shape):
            zz = z[idx]
            if zz <= zs[0]:
                z1, z2 = zs[0], zs[1]
            elif zz >= zs[-1]:
                z1, z2 = zs[-2], zs[-1]
            else:
                j = int(np.searchsorted(zs, zz))
                z1, z2 = zs[j - 1], zs[j]
            v1 = float(self._grid_v(z1, c, dx[idx], dy[idx]))
            v2 = float(self._grid_v(z2, c, dx[idx], dy[idx]))
            if v1 * v2 > 0 and min(abs(v1), abs(v2)) > 0.1:
                bl = float(np.clip(np.log(v1 / v2) / (z2 - z1), 0.3, 4.0))
                ref, zr = (v1, z1) if abs(zz - z1) <= abs(zz - z2) else (v2, z2)
                out[idx] = ref * np.exp(-bl * (zz - zr))
            else:                                            # 节点附近 (变号)：线性插值，外推用全局 β
                t = (zz - z1) / (z2 - z1)
                if 0 <= t <= 1:
                    out[idx] = v1 + t * (v2 - v1)
                else:
                    ref, zr = (v1, z1) if t < 0 else (v2, z2)
                    out[idx] = ref * np.exp(-b * (zz - zr))
        return out

    def data_range(self):
        """滑移网格覆盖的范围 (z 取各网格的最小、最大值；Δx、Δy 含对称扩展)。"""
        t = [r for r in self._translational() if round(r["z"], 3) in self.grids] or self._translational()
        return {"z": (min(r["z"] for r in t), max(r["z"] for r in t)),
                "dx": (-max(abs(r["dx"]) for r in t), max(abs(r["dx"]) for r in t)) if self.symmetry == "D2h"
                else (min(r["dx"] for r in t), max(r["dx"] for r in t)),
                "dy": (-max(abs(r["dy"]) for r in t), max(abs(r["dy"]) for r in t)) if self.symmetry == "D2h"
                else (min(r["dy"] for r in t), max(r["dy"] for r in t))}

    # ---- 检验 ----
    def separability(self, c="hole"):
        """相邻两张滑移网格逐点比较：最小二乘比例因子 k (V2 ≈ k·V1)、等效 β、相关系数、残差、变号点数。"""
        zs = self.zs(c)
        out = []
        for z1, z2 in zip(zs[:-1], zs[1:]):
            gx, gy, G1 = self.grids[z1][c]
            G2 = self.splines[(z2, c)](gx, gy)
            a, b = G1.ravel(), G2.ravel()
            k = float(a @ b / (a @ a))
            big = np.abs(a) > np.percentile(np.abs(a), 75)
            ratio = b[big] / a[big]
            out.append({"z1": z1, "z2": z2, "k": k, "beta_eff": float(-np.log(abs(k)) / (z2 - z1)),
                        "r": float(np.corrcoef(a, b)[0, 1]),
                        "rel_resid": float(np.sqrt(((b - k * a) ** 2).mean()) / max(np.sqrt((b ** 2).mean()), 1e-12)),
                        "ratio_cv": float(ratio.std() / abs(ratio.mean())) if len(ratio) else float("nan"),
                        "sign_flips": int(((a * b) < 0).sum()), "n": int(a.size)})
        return out

    def validate(self, recs, methods=("M1", "M2", "M3")):
        zr = self.data_range()["z"]
        res = {}
        for m in methods:
            if m == "M3" and not self.calib:
                continue
            for c in CARRIERS:
                rows = []
                for r in recs:
                    if r[c] is None:
                        continue
                    try:
                        p = float(self.predict(r["z"], r["dx"], r["dy"], c, m, r["gamma"], r["roll"], r["pitch"]))
                    except ValueError:
                        continue
                    rows.append((r[c], p, zr[0] - 1e-6 <= r["z"] <= zr[1] + 1e-6))
                for grp, sel in (("全部", lambda x: True), ("内插", lambda x: x[2]), ("外推", lambda x: not x[2])):
                    rr = [x for x in rows if sel(x)]
                    if not rr:
                        continue
                    t, p, _ = map(np.array, zip(*rr))
                    err = p - t
                    res[(m, c, grp)] = {"n": len(t), "MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
                                        "max": float(np.abs(err).max()),
                                        "R2": float(1 - (err ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12)),
                                        "sign": float(np.mean(np.sign(t) == np.sign(p))), "pairs": (t, p)}
        return res

    # ---- 保存 ----
    def to_json(self, path):
        g = {str(z): {c: {"dx": v[0].tolist(), "dy": v[1].tolist(), "V": v[2].tolist()} for c, v in gg.items()}
             for z, gg in self.grids.items()}
        doc = {"format": "mobilis-coupling-model-1", "symmetry": self.symmetry, "z_ref": self.z_ref, "beta": self.beta,
               "calib": self.calib, "grids": g, "units": "meV, Å, °",
               "monomer": None if self.monomer is None else {"symbols": list(self.monomer[0]),
                                                            "Y": np.asarray(self.monomer[1]).tolist()},
               "records": self.records}
        Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=1))

    @classmethod
    def from_json(cls, path):
        doc = json.loads(Path(path).read_text())
        mono = doc.get("monomer")
        m = cls(doc.get("records", []), doc.get("symmetry", "D2h"),
                None if not mono else (mono["symbols"], np.array(mono["Y"])))
        m.grids = {float(z): {c: (np.array(v["dx"]), np.array(v["dy"]), np.array(v["V"])) for c, v in gg.items()}
                   for z, gg in doc["grids"].items()}
        m.fit()
        m.calib = doc.get("calib") or m.calib
        return m


# ----------------------------------------------------------------------------------------------
# 堆积 → 迁移率；反向设计
# ----------------------------------------------------------------------------------------------
def packing_mobility(model, params, kind="1d", carrier="hole", lam=0.15, T=300.0, method="M2"):
    """params: 1d → (dx, dy, z)；2d → (dx1, dy1, dx2, dy2, z)。返回迁移率张量、主值、各向异性与各路径 V。"""
    if kind == "1d":
        dx, dy, z = params
        vecs = [np.array([dx, dy, z])]
    else:
        dx1, dy1, dx2, dy2, z = params
        vecs = [np.array([dx1, dy1, z]), np.array([dx2, dy2, z])]
    kt = KB_EV * T
    mu = np.zeros((3, 3))
    vs = []
    for v in vecs:
        V = float(model.predict(v[2], v[0], v[1], carrier, method))
        vs.append(V)
        k = marcus_rate(abs(V) / 1000, lam, T)
        vc = v * 1e-8
        mu += 2 * 0.5 * k * np.outer(vc, vc)                      # ±r 两个近邻
    mu /= kt
    w = np.sort(np.linalg.eigvalsh(mu))[::-1]
    out = {"tensor": mu, "principal": w, "V_meV": vs, "vectors": vecs}
    if kind == "1d":
        u = vecs[0] / np.linalg.norm(vecs[0])
        out["mu_stack"] = float(u @ mu @ u)
        out["mu_max"] = out["mu_stack"]
        out["anisotropy"] = float("inf")
    else:
        nrm = np.cross(vecs[0], vecs[1])
        if np.linalg.norm(nrm) < 1e-6:
            out.update(mu_max=float(w[0]), mu_min=0.0, anisotropy=float("inf"), mu_avg=float(w[0] / 2))
        else:
            e1 = vecs[0] / np.linalg.norm(vecs[0])
            e2 = np.cross(nrm / np.linalg.norm(nrm), e1)
            P = np.array([e1, e2])
            m2 = P @ mu @ P.T
            ww = np.sort(np.linalg.eigvalsh(m2))[::-1]
            out.update(mu_max=float(ww[0]), mu_min=float(ww[1]), mu_avg=float(ww.mean()),
                       anisotropy=float(ww[0] / ww[1]) if ww[1] > 1e-12 else float("inf"))
    return out


OBJECTIVES = {"max_mu": "最大迁移率 (主方向)", "max_avg": "最大平均迁移率 (二维，可限制各向异性)",
              "min_aniso": "最小各向异性 (二维，平均迁移率不低于下限)", "ambipolar": "双极平衡：max min(μh, μe)"}


def optimize_packing(model, kind="1d", objective="max_mu", carrier="hole", lam=0.15, lam_e=None, T=300.0, method="M2",
                     z_range=None, dx_range=None, dy_range=None, clash_min=0.75, aniso_max=None, mu_floor=0.0,
                     seed=2026, maxiter=200):
    """在不碰撞、不超出数据范围的约束下全局优化滑移 (scipy differential_evolution)。"""
    from scipy.optimize import differential_evolution
    rg = model.data_range()
    zr = z_range or rg["z"]
    xr = dx_range or rg["dx"]
    yr = dy_range or rg["dy"]
    bounds = [xr, yr, zr] if kind == "1d" else [xr, yr, xr, yr, zr]
    sym, Y = model.monomer if model.monomer is not None else (None, None)
    lam_e = lam if lam_e is None else lam_e

    def penalty(p):
        if sym is None:
            return 0.0
        vecs = [(p[0], p[1], p[2])] if kind == "1d" else [(p[0], p[1], p[4]), (p[2], p[3], p[4])]
        if kind == "2d":
            vecs.append((p[0] - p[2], p[1] - p[3], 0.0))         # 同层侧向近邻也不能碰撞
        pen = 0.0
        for dx, dy, z in vecs:
            ratio = clash_ratio(sym, Y, build_B(Y, z, dx, dy))[1]
            if ratio < clash_min:
                pen += (clash_min - ratio) * 1e3
        return pen

    def score(p):
        if objective == "ambipolar":
            a = packing_mobility(model, p, kind, "hole", lam, T, method)
            b = packing_mobility(model, p, kind, "electron", lam_e, T, method)
            return min(a["mu_max"], b["mu_max"])
        r = packing_mobility(model, p, kind, carrier, lam if carrier == "hole" else lam_e, T, method)
        if objective == "max_mu":
            return r["mu_max"]
        if objective == "max_avg":
            s = r.get("mu_avg", r["mu_max"])
            if aniso_max and r["anisotropy"] > aniso_max:
                s -= (r["anisotropy"] - aniso_max) * s
            return s
        if objective == "min_aniso":
            s = -np.log(r["anisotropy"]) if np.isfinite(r["anisotropy"]) else -50.0
            if r.get("mu_avg", 0) < mu_floor:
                s -= 10 * (mu_floor - r.get("mu_avg", 0)) / max(mu_floor, 1e-9)
            return s
        raise ValueError(objective)

    hist = []

    def f(p):
        s = score(p)
        hist.append((list(p), s))
        return -s + penalty(p)
    res = differential_evolution(f, bounds, seed=seed, maxiter=maxiter, tol=1e-7, polish=True, popsize=20)
    best = res.x
    out = {"params": best.tolist(), "score": float(score(best)), "clash_penalty": float(penalty(best)),
           "kind": kind, "objective": objective, "method": method, "evaluations": len(hist), "bounds": bounds}
    if objective == "ambipolar":
        out["hole"] = packing_mobility(model, best, kind, "hole", lam, T, method)
        out["electron"] = packing_mobility(model, best, kind, "electron", lam_e, T, method)
    else:
        out[carrier] = packing_mobility(model, best, kind, carrier, lam if carrier == "hole" else lam_e, T, method)
    return out


def landscape(model, z, carrier="hole", method="M2", n=121, quantity="V", lam=0.15, T=300.0, clash_min=0.75):
    """Δx–Δy 平面上的 V 或一维堆积迁移率；碰撞区域标为 NaN。"""
    rg = model.data_range()
    xs = np.linspace(*rg["dx"], n)
    ys = np.linspace(*rg["dy"], max(n // 2, 31))
    X, Yg = np.meshgrid(xs, ys, indexing="ij")
    V = model.predict(z, X, Yg, carrier, method)
    out = V if quantity == "V" else \
        marcus_rate(np.abs(V) / 1000, lam, T) * ((X ** 2 + Yg ** 2 + z ** 2) * 1e-16) / (KB_EV * T)
    if model.monomer is not None:
        sym, Ym = model.monomer
        bad = np.zeros_like(out, bool)
        for i in range(0, len(xs)):
            for j in range(0, len(ys)):
                if clash_ratio(sym, Ym, build_B(Ym, z, xs[i], ys[j]))[1] < clash_min:
                    bad[i, j] = True
        out = np.where(bad, np.nan, out)
    return xs, ys, out


# ----------------------------------------------------------------------------------------------
# 与实际晶体对照：晶体二聚体在分子坐标系中的滑移
# ----------------------------------------------------------------------------------------------
def crystal_slips(dimer_result):
    """把晶体中各近邻二聚体表示为 A 分子坐标系中的 (Δx, Δy, z) 与相对取向 (法线夹角、长轴夹角)。"""
    sa = [s for s, _ in dimer_result["atoms_a"]]
    xa = np.array([x for _, x in dimer_result["atoms_a"]])
    _, _, axA, cA = molecular_frame(sa, xa)
    out = []
    for g in dimer_result["groups"]:
        d = g["members"][0]
        xb = np.array([x for _, x in d["atoms_b"]])
        _, _, axB, cB = molecular_frame([s for s, _ in d["atoms_b"]], xb)
        off = (cB - cA) @ axA
        if off[2] < 0:                                     # 取 z ≥ 0 的等价表示
            off = -off
        ang_n = float(np.degrees(np.arccos(min(1, abs(axA[:, 2] @ axB[:, 2])))))
        ang_l = float(np.degrees(np.arccos(min(1, abs(axA[:, 0] @ axB[:, 0])))))
        out.append({"group": g["id"], "dx": float(off[0]), "dy": float(off[1]), "z": float(off[2]),
                    "normal_angle": ang_n, "long_axis_angle": ang_l,
                    "cofacial": ang_n < 15 and ang_l < 15, "r": d["r_centroid"]})
    return out


# ----------------------------------------------------------------------------------------------
# 生成 ADF2016 计算：验证集 / 最优构型 (与 make_dimers.py 相同格式，共用 mono.t21)
# ----------------------------------------------------------------------------------------------
def lhs_points(n=40, z_inner=(3.5, 3.8), z_outer=((3.3, 3.5), (3.8, 4.2)), dx=(-6, 6), dy=(-3, 3), seed=2026):
    """拉丁超立方抽样：一半在 z 内插区间，一半在外推区间。返回 [(标签, z, dx, dy)]。"""
    from scipy.stats import qmc
    pts = []
    n_in = n // 2
    s = qmc.LatinHypercube(d=3, seed=seed).random(n)
    for i, u in enumerate(s):
        if i < n_in:
            z = z_inner[0] + u[0] * (z_inner[1] - z_inner[0])
            tag = "interp"
        else:
            lo, hi = z_outer[0] if u[0] < 0.4 else z_outer[1]
            uu = u[0] / 0.4 if u[0] < 0.4 else (u[0] - 0.4) / 0.6
            z = lo + uu * (hi - lo)
            tag = "extrap"
        pts.append((tag, round(z, 4), round(dx[0] + u[1] * (dx[1] - dx[0]), 4), round(dy[0] + u[2] * (dy[1] - dy[0]), 4)))
    return pts


def write_adf_scan(outdir, symbols, Y, configs, frag="mono", basis="TZP", xc="PW91", mono_t21=None, clash_min=0.75,
                   prefix="valid"):
    """configs: [(标签, z, dx, dy[, gamma, roll, pitch])]。写 manifest.csv、xyz/、batch_<prefix>.run (ADFjobs) 与 .sh。
    mono_t21: 已有的单体片段文件 (推荐沿用扫描所用的同一个，V 的符号才与扫描数据可比)；为空时同时生成 mono.run。"""
    outdir = Path(outdir)
    (outdir / "outputs").mkdir(parents=True, exist_ok=True)
    (outdir / "xyz").mkdir(exist_ok=True)
    root = outdir.resolve().as_posix()
    kw = (f"SYMMETRY NOSYM\n\nBASIS\ntype {basis}\ncore None\ncreateoutput None\nEND\n\n"
          f"XC\nGGA {xc}\nEND\n\n")
    if mono_t21:
        import shutil
        dst = outdir / f"{frag}.t21"
        if Path(mono_t21).resolve() != dst.resolve():
            shutil.copy(mono_t21, dst)
    else:
        s = '#! /bin/sh\n\n"$ADFBIN/adf" <<eor\n' + f"TITLE {frag} monomer fragment\n\nATOMS\n"
        s += "".join(f"{i} {e:2s} {r[0]:18.10f} {r[1]:18.10f} {r[2]:18.10f}\n" for i, (e, r) in enumerate(zip(symbols, Y), 1))
        s += "END\n\n" + kw + "NOPRINT LOGFILE\n\neor\n"
        s += f'[ -s TAPE21 ] && mv -f TAPE21 "{root}/{frag}.t21"\n'
        (outdir / f"{frag}.run").write_bytes(s.encode("utf-8"))
    rows, body = [], ["#! /bin/sh", "", f'if [ ! -s "{root}/{frag}.t21" ]; then',
                      f'  echo "ERROR: {root}/{frag}.t21 not found, run {frag}.run first"', "  exit 1", "fi", ""]
    for i, cfg in enumerate(configs, 1):
        tag, z, dx, dy = cfg[:4]
        g, ro, pi = (list(cfg[4:7]) + [0.0, 0.0, 0.0])[:3]
        name = f"{prefix}_{i:04d}"
        B = build_B(Y, z, dx, dy, g, ro, pi)
        dmin, ratio = clash_ratio(symbols, Y, B)
        st = "ok" if ratio >= clash_min else "clash"
        rows.append([name, z, dx, dy, g, ro, pi, round(dmin, 4), round(ratio, 4), st, tag])
        if st != "ok":
            continue
        title = f"{name} z={z} dx={dx} dy={dy} gamma={g} roll={ro} pitch={pi}"
        with open(outdir / "xyz" / f"{name}.xyz", "w") as fh:
            fh.write(f"{2 * len(symbols)}\n{title}\n")
            for M in (Y, B):
                for e, r in zip(symbols, M):
                    fh.write(f"{e:2s} {r[0]:14.8f} {r[1]:14.8f} {r[2]:14.8f}\n")
        o = f"{root}/outputs/{name}.out"
        body += [f'if [ ! -s "{o}" ]; then', f'echo "running {name}"', f'"$ADFBIN/adf" <<eor > "{o}" 2>&1',
                 f"TITLE {title}", "", "ATOMS"]
        n = 1
        for kk, M in ((1, Y), (2, B)):
            for e, c in zip(symbols, M):
                body.append(f"{n} {e:2s} {c[0]:18.10f} {c[1]:18.10f} {c[2]:18.10f}  f={frag}/{kk}")
                n += 1
        body += ["END", "", kw.rstrip("\n"), "", "Fragments", f"    {frag} {root}/{frag}.t21", "end", "NOPRINT LOGFILE",
                 "TRANSFERINTEGRALS", "", "eor", "fi", ""]
    body.append('echo "batch finished"')
    text = "\n".join(body) + "\n"
    (outdir / f"batch_{prefix}.run").write_bytes(text.encode("gbk", errors="replace"))
    (outdir / f"batch_{prefix}.sh").write_bytes(text.replace("#! /bin/sh", "#!/bin/sh", 1).encode("utf-8"))
    with open(outdir / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "z", "dx", "dy", "gamma", "roll", "pitch", "dmin_A", "min_ratio_vdW", "status", "set"])
        w.writerows(rows)
    with open(outdir / f"{frag}_frame.xyz", "w") as fh:
        fh.write(f"{len(symbols)}\nmonomer in molecular frame\n")
        for e, r in zip(symbols, Y):
            fh.write(f"{e:2s} {r[0]:14.8f} {r[1]:14.8f} {r[2]:14.8f}\n")
    return {"n": sum(1 for r in rows if r[9] == "ok"), "clash": sum(1 for r in rows if r[9] != "ok"),
            "batch": str(outdir / f"batch_{prefix}.run"), "need_mono": not bool(mono_t21)}
