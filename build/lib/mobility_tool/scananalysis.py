"""扫描数据分析与报告：

距离衰减      |V(z)| = A exp[−β(z − z0)]，ln|V| 对 z 线性最小二乘 (平行、Δx = Δy = 0 的构型)
滑移图        各 z 下第一象限计算值，按 D2h 对称展开到四个象限
可分离性      (i) 两个 z 的滑移图逐点比较：过原点最小二乘比例因子 k、等效 β = −ln k/Δz、r、残差、
                  |V| > 20 % 最大值处的逐点比值 (均值 ± 标准差，按 |Δx|、|Δy| 分箱)、变号点数
              (ii) 固定 z 的滑移矩阵 SVD：第一奇异分量占 Frobenius 范数平方的比例与秩一重构 RMSE
V–S           固定 z 下 V = kS 过原点拟合
解析式        V = A exp(−a_x Δx² − a_y Δy²) cos(k_x Δx) cos(k_y Δy)  (非线性最小二乘，多初值)
模型          M1 = F_zref(|Δx|,|Δy|) e^{−β(z−zref)}；M2 = 最近网格 F_zr e^{−β(z−zr)}；M3 = 解析式 e^{−β(z−zref)}；
              网格由三次样条插值 (第一象限按对称展开)，β 取距离扫描
验证          MAE、RMSE (全部 / 内插 / 外推)、最大误差、R²、符号一致率
层级对照      同一批构型在两个层级 (泛函 / 基组) 下的 V：比例因子、r、RMSE、符号一致率 (允许整体符号不同)
相互作用能    ADF 能量分解 (总成键能)、Gaussian 抵消校正或 UFF；Boltzmann 权重 w ∝ exp(−E/RT) 下的 <V>、√<V²>
重组能        四点法结果 (Gaussian)；Marcus 速率 k = (2π/ħ) V² (4πλk_BT)^(−1/2) exp(−λ/4k_BT)
"""
import base64
import csv
import html
import io
import json
import math
import time
from pathlib import Path

import numpy as np

from .scanflow import KCAL_R, PARAM_KEYS, cfg_key, canonical, symmetry_check, write_records
from .surrogate import CARRIERS, CouplingModel

HOLE_C, ELEC_C = "#2f6db3", "#c4472f"
COL = {"hole": HOLE_C, "electron": ELEC_C}
MARK = {"hole": "o", "electron": "s"}
LAB = {"hole": "Holes", "electron": "Electrons"}
MODELS = ("M1", "M2", "M3")
CZ = {"hole": "空穴", "electron": "电子"}
ROLE_ZH = {"grid": "训练网格", "validation": "验证集", "check": "对称检验", "crystal": "晶体", "compare": "层级对照",
           "local": "CEDS 局域扫描", "md": "MD 快照检验",
           "imported": "导入"}


def _parallel(r):
    return abs(r["gamma"]) + abs(r["pitch"]) + abs(r["roll"]) < 1e-6


def computed(recs):
    """实际计算且完成的构型 (不含对称映射行)。"""
    return [r for r in recs if r.get("state") == "done" and not r.get("rep")]


def _vals(recs, c):
    return [r for r in recs if r.get(c) is not None]


# ----------------------------------------------------------------------------------------------
# 距离衰减、滑移矩阵、可分离性、V–S、解析式
# ----------------------------------------------------------------------------------------------
def distance_fit(recs, c, z0=3.5, vmin=0.5):
    pts = sorted((r["z"], r[c]) for r in computed(recs) if _parallel(r) and abs(r["dx"]) < 1e-6 and abs(r["dy"]) < 1e-6
                 and r.get(c) is not None and abs(r[c]) > vmin and r["z"] > 0)
    z = np.array([p[0] for p in pts])
    if len(np.unique(z)) < 3:
        return None
    v = np.abs([p[1] for p in pts])
    b, a = np.polyfit(z, np.log(v), 1)
    pred = a + b * z
    ss = ((np.log(v) - np.log(v).mean()) ** 2).sum()
    r2 = 1 - ((np.log(v) - pred) ** 2).sum() / ss if ss > 0 else 1.0
    out = {"beta": float(-b), "A_z0": float(np.exp(a + b * z0)), "z0": z0, "R2": float(r2), "n": len(z),
           "z": z.tolist(), "V": [p[1] for p in pts], "range": (float(z.min()), float(z.max()))}
    at = [p[1] for p in pts if abs(p[0] - z0) < 1e-4]
    if at:
        out["V_z0"] = float(at[0])
    return out


def grid_zs(recs, min_points=16):
    """有滑移网格 (≥ 4×4) 的 z 值。"""
    byz = {}
    for r in computed(recs):
        if _parallel(r) and r["z"] > 0:
            byz.setdefault(round(r["z"], 4), []).append(r)
    out = []
    for z, rs in byz.items():
        if len(rs) >= min_points and len({round(r["dx"], 4) for r in rs}) >= 4 and len({round(r["dy"], 4) for r in rs}) >= 4:
            out.append(z)
    return sorted(out)


def grid_matrix(recs, z, c):
    """第一象限计算值组成的矩阵 (行 Δx，列 Δy；缺失为 NaN)。"""
    rs = [r for r in computed(recs) if _parallel(r) and abs(r["z"] - z) < 1e-4 and r["dx"] >= -1e-9 and r["dy"] >= -1e-9
          and r.get(c) is not None]
    xs = sorted({round(r["dx"], 4) for r in rs})
    ys = sorted({round(r["dy"], 4) for r in rs})
    M = np.full((len(xs), len(ys)), np.nan)
    ix, iy = {x: i for i, x in enumerate(xs)}, {y: j for j, y in enumerate(ys)}
    for r in rs:
        M[ix[round(r["dx"], 4)], iy[round(r["dy"], 4)]] = r[c]
    return np.array(xs), np.array(ys), M


def svd_separability(M):
    keep_r = ~np.isnan(M).any(1)
    M2 = M[keep_r]
    keep_c = ~np.isnan(M2).any(0)
    M2 = M2[:, keep_c]
    if min(M2.shape) < 2:
        return None
    u, s, vt = np.linalg.svd(M2, full_matrices=False)
    R1 = s[0] * np.outer(u[:, 0], vt[0])
    return {"share": float(s[0] ** 2 / (s ** 2).sum()), "rmse_rank1": float(np.sqrt(((M2 - R1) ** 2).mean())),
            "shape": M2.shape}


def compare_maps(recs, z1, z2, c, frac=0.2):
    a_ = {(round(r["dx"], 4), round(r["dy"], 4)): r[c] for r in computed(recs)
          if _parallel(r) and abs(r["z"] - z1) < 1e-4 and r.get(c) is not None}
    b_ = {(round(r["dx"], 4), round(r["dy"], 4)): r[c] for r in computed(recs)
          if _parallel(r) and abs(r["z"] - z2) < 1e-4 and r.get(c) is not None}
    keys = sorted(set(a_) & set(b_))
    if len(keys) < 5:
        return None
    a = np.array([a_[k] for k in keys])
    b = np.array([b_[k] for k in keys])
    k = float(a @ b / (a @ a))
    big = np.abs(a) > frac * np.abs(a).max()
    ratio = b[big] / a[big]
    dx = np.abs([kk[0] for kk in keys])[big]
    dy = np.abs([kk[1] for kk in keys])[big]

    def bins(x, edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi + (1e-9 if hi == edges[-1] else 0))
            if m.any():
                out.append((lo, hi, float(ratio[m].mean()), int(m.sum())))
        return out
    return {"z1": z1, "z2": z2, "n": len(keys), "k": k, "beta_eff": float(-math.log(abs(k)) / (z2 - z1)),
            "r": float(np.corrcoef(a, b)[0, 1]), "rmse": float(np.sqrt(((b - k * a) ** 2).mean())),
            "ratio_mean": float(ratio.mean()), "ratio_std": float(ratio.std()), "n_big": int(big.sum()),
            "bins_dx": bins(dx, [0, 2, 4, 6.0001]), "bins_dy": bins(dy, [0, 1, 2, 3.0001]),
            "sign_flips": int((a * b < 0).sum()), "a": a.tolist(), "b": b.tolist()}


def vs_fit(recs, z, c):
    rs = [r for r in computed(recs) if _parallel(r) and abs(r["z"] - z) < 1e-4 and r.get(c) is not None
          and r.get(f"S_{c}") is not None]
    if len(rs) < 5:
        return None
    V = np.array([r[c] for r in rs])
    S = np.array([r[f"S_{c}"] for r in rs])
    if not np.any(S):
        return None
    k = float(V @ S / (S @ S))
    return {"z": z, "n": len(rs), "k": k, "r": float(np.corrcoef(V, S)[0, 1]),
            "rmse": float(np.sqrt(((V - k * S) ** 2).mean())), "V": V.tolist(), "S": S.tolist()}


def analytic(p, dx, dy):
    A, ax, kx, ay, ky = p
    return A * np.exp(-ax * dx ** 2 - ay * dy ** 2) * np.cos(kx * dx) * np.cos(ky * dy)


def analytic_fit(recs, z, c):
    """五参数解析式，用第一象限计算值拟合；多初值 (k_x、k_y 网格) 取最优。"""
    from scipy.optimize import least_squares
    rs = [r for r in computed(recs) if _parallel(r) and abs(r["z"] - z) < 1e-4 and r.get(c) is not None]
    if len(rs) < 10:
        return None
    dx = np.abs([r["dx"] for r in rs])
    dy = np.abs([r["dy"] for r in rs])
    V = np.array([r[c] for r in rs])
    v0 = next((r[c] for r in rs if abs(r["dx"]) < 1e-6 and abs(r["dy"]) < 1e-6), V[np.argmax(np.abs(V))])
    starts = []
    for kx in np.arange(0.2, 2.01, 0.1):
        for ky in np.arange(0.0, 1.61, 0.1):
            p = np.array([v0, 0.02, kx, 0.08, ky])
            starts.append((((analytic(p, dx, dy) - V) ** 2).sum(), p))
    starts.sort(key=lambda t: t[0])
    best = None
    for _, p0 in starts[:12]:
        res = least_squares(lambda p: analytic(p, dx, dy) - V, p0,
                            bounds=([-np.inf, 0, 0, 0, 0], [np.inf, 5, 6, 5, 6]), x_scale="jac")
        if best is None or res.cost < best.cost:
            best = res
    p = best.x
    resid = analytic(p, dx, dy) - V
    dof = max(len(V) - 5, 1)
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac) * (resid @ resid) / dof
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(5, np.nan)
    ss = ((V - V.mean()) ** 2).sum()
    return {"z": z, "n": len(V), "params": dict(zip(("A", "ax", "kx", "ay", "ky"), map(float, p))),
            "stderr": dict(zip(("A", "ax", "kx", "ay", "ky"), map(float, err))),
            "rmse": float(np.sqrt((resid ** 2).mean())), "R2": float(1 - (resid ** 2).sum() / ss) if ss else 1.0,
            "p": p.tolist()}


# ----------------------------------------------------------------------------------------------
# 模型 M1 / M2 / M3
# ----------------------------------------------------------------------------------------------
class PaperModels:
    def __init__(self, train, z_ref=3.5, z0=3.5, symmetry="D2h"):
        par = [r for r in computed(train) if _parallel(r)]
        cm_recs = [{"id": r["id"], "z": r["z"], "dx": r["dx"], "dy": r["dy"], "gamma": 0.0, "roll": 0.0, "pitch": 0.0,
                    "hole": r.get("hole"), "electron": r.get("electron")} for r in par]
        self.cm = CouplingModel(cm_recs, symmetry=symmetry)
        self.cm.build_grids()
        if not self.cm.grids:
            raise ValueError("训练数据中没有滑移网格 (至少需要一个 z 下 4×4 以上的 Δx–Δy 网格)")
        self.cm.fit()
        self.zs = {c: self.cm.zs(c) for c in CARRIERS}
        allz = sorted(self.cm.grids)
        self.z_ref = min(allz, key=lambda z: abs(z - z_ref))
        self.dist = {c: distance_fit(train, c, z0) for c in CARRIERS}
        self.beta = {}
        for c in CARRIERS:
            if self.dist[c]:
                self.beta[c] = (self.dist[c]["beta"], "距离扫描")
            else:
                self.beta[c] = (self.cm.beta[c]["value"], self.cm.beta[c]["source"])
        self.analytic = {c: analytic_fit(train, self.z_ref, c) for c in CARRIERS}
        self.range = {"z": (min(allz), max(allz))}

    def predict(self, c, model, z, dx, dy):
        b = self.beta[c][0]
        if model == "M1":
            return float(self.cm._grid_v(self.z_ref, c, dx, dy)) * math.exp(-b * (z - self.z_ref))
        if model == "M2":
            zr = min(self.zs[c], key=lambda q: abs(q - z))
            return float(self.cm._grid_v(zr, c, dx, dy)) * math.exp(-b * (z - zr))
        if model == "M3":
            a = self.analytic[c]
            if a is None:
                return float("nan")
            return float(analytic(np.array(a["p"]), abs(dx), abs(dy))) * math.exp(-b * (z - self.z_ref))
        if model == "M2i":
            return float(self.cm.predict(z, dx, dy, c, "M2"))
        raise ValueError(model)

    def to_json(self, path, extra=None):
        doc = {"format": "mobilis-paper-models-1", "z_ref": self.z_ref, "grids_z": {c: self.zs[c] for c in CARRIERS},
               "beta": {c: {"value": v, "source": s} for c, (v, s) in self.beta.items()},
               "distance_fit": {c: ({k: v for k, v in d.items() if k not in ("z", "V")} if d else None)
                                for c, d in self.dist.items()},
               "analytic": self.analytic, "models": {"M1": "F_zref(|dx|,|dy|) exp(-beta (z - zref))",
                                                     "M2": "F_zr(|dx|,|dy|) exp(-beta (z - zr)), zr = nearest grid",
                                                     "M3": "A exp(-ax dx^2 - ay dy^2) cos(kx dx) cos(ky dy) exp(-beta (z - zref))"}}
        doc.update(extra or {})
        Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        return path

    def write_grids(self, folder):
        folder = Path(folder)
        for z, g in self.cm.grids.items():
            with open(folder / f"model_grid_z{z:.2f}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["dx", "dy"] + [f"V_{c}_meV" for c in CARRIERS if c in g])
                gx, gy = g[CARRIERS[0]][0], g[CARRIERS[0]][1]
                for i, x in enumerate(gx):
                    for j, y in enumerate(gy):
                        w.writerow([x, y] + [f"{g[c][2][i, j]:.4f}" for c in CARRIERS if c in g])


def validate(models, valid, names=MODELS):
    """返回 (统计表, 逐点预测)。内插 / 外推按 set 标签 (interp / extrap)；没有标签时按网格的 z 范围判断。"""
    lo, hi = models.range["z"]
    pts = [r for r in computed(valid) if _parallel(r)]
    preds = []
    for r in pts:
        tag = r.get("set") if r.get("set") in ("interp", "extrap") else ("interp" if lo - 1e-6 <= r["z"] <= hi + 1e-6 else "extrap")
        row = {"id": r["id"], "set": tag, **{k: r[k] for k in ("z", "dx", "dy")}}
        for c in CARRIERS:
            row[f"DFT_{c}"] = r.get(c)
            for m in names:
                row[f"{m}_{c}"] = models.predict(c, m, r["z"], r["dx"], r["dy"])
        preds.append(row)
    stats = []
    for c in CARRIERS:
        for m in names:
            t = np.array([p[f"DFT_{c}"] for p in preds if p[f"DFT_{c}"] is not None], float)
            q = np.array([p[f"{m}_{c}"] for p in preds if p[f"DFT_{c}"] is not None], float)
            tg = [p["set"] for p in preds if p[f"DFT_{c}"] is not None]
            if not len(t):
                continue
            e = q - t

            def rm(sel):
                s = np.array([x == sel for x in tg])
                return float(np.sqrt((e[s] ** 2).mean())) if s.any() else float("nan")
            stats.append({"carrier": c, "model": m, "n": len(t), "MAE": float(np.abs(e).mean()),
                          "RMSE": float(np.sqrt((e ** 2).mean())), "RMSE_interp": rm("interp"), "RMSE_extrap": rm("extrap"),
                          "max": float(np.abs(e).max()), "R2": float(1 - (e ** 2).sum() / ((t - t.mean()) ** 2).sum()),
                          "sign": float(100 * np.mean(np.sign(t) == np.sign(q))),
                          "n_interp": tg.count("interp"), "n_extrap": tg.count("extrap")})
    return stats, preds


def compare_levels(recA, recB, c):
    a_ = {cfg_key(r): r[c] for r in computed(recA) if r.get(c) is not None}
    b_ = {cfg_key(r): r[c] for r in computed(recB) if r.get(c) is not None}
    keys = sorted(set(a_) & set(b_))
    if len(keys) < 3:
        return None
    a = np.array([a_[k] for k in keys])
    b = np.array([b_[k] for k in keys])
    s = 1.0 if a @ b >= 0 else -1.0                          # 不同层级的片段轨道整体相位可能相反
    k = float(a @ b / (a @ a))
    return {"n": len(keys), "k": k, "r": float(np.corrcoef(a, b)[0, 1]), "rmse": float(np.sqrt(((b - k * a) ** 2).mean())),
            "mae_abs": float(np.abs(np.abs(b) - np.abs(a)).mean()), "global_sign": int(s),
            "sign_agree": float(100 * np.mean(np.sign(a) * s == np.sign(b))), "a": a.tolist(), "b": b.tolist()}


def boltzmann(recs, energies, c, T=300.0):
    """energies: {id: E (kcal/mol)}；对称映射的构型取其代表的能量。返回 <V>、<|V|>、√<V²>、有效构型数与权重最大的构型。"""
    rows = []
    for r in recs:
        if r.get("state") != "done" or r.get(c) is None:
            continue
        e = energies.get(r["rep"] or r["id"])
        if e is None:
            continue
        rows.append((e, r[c], r))
    if len(rows) < 2:
        return None
    E = np.array([x[0] for x in rows])
    V = np.array([x[1] for x in rows])
    w = np.exp(-(E - E.min()) / (KCAL_R * T))
    w /= w.sum()
    order = np.argsort(-w)[:5]
    return {"T": T, "n": len(rows), "V_mean": float(w @ V), "absV_mean": float(w @ np.abs(V)),
            "V_rms": float(np.sqrt(w @ V ** 2)), "n_eff": float(1 / (w ** 2).sum()), "E_min": float(E.min()),
            "V_unweighted_rms": float(np.sqrt((V ** 2).mean())),
            "top": [{"id": rows[i][2]["id"], "E": float(E[i]), "V": float(V[i]), "w": float(w[i]),
                     **{k: rows[i][2][k] for k in ("z", "dx", "dy", "roll")}} for i in order]}


# ----------------------------------------------------------------------------------------------
# 作图 (300 dpi)
# ----------------------------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                                "font.size": 10, "axes.linewidth": 1.0, "xtick.direction": "out",
                                "ytick.direction": "out", "axes.unicode_minus": False, "mathtext.default": "regular",
                                "legend.frameon": False})


def _fig(w, h):
    from matplotlib.figure import Figure
    _style()
    return Figure(figsize=(w, h), dpi=300)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    return path


def _panel_label(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def fig_distance(fits, path):
    fig = _fig(3.6, 3.0)
    ax = fig.add_subplot(111)
    for c in CARRIERS:
        d = fits.get(c)
        if not d:
            continue
        z = np.array(d["z"])
        ax.semilogy(z, np.abs(d["V"]), MARK[c], ms=5, mfc="none" if c == "electron" else COL[c], mec=COL[c],
                    label=f"{LAB[c]}  β = {d['beta']:.3f} Å$^{{-1}}$")
        zz = np.linspace(z.min(), z.max(), 50)
        ax.semilogy(zz, d["A_z0"] * np.exp(-d["beta"] * (zz - d["z0"])), "-", color=COL[c], lw=1.2)
    ax.set_xlabel("z (Å)")
    ax.set_ylabel("|V| (meV)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def _full_map(xs, ys, M):
    """第一象限 → 四象限 (D2h)。"""
    X = np.concatenate([-xs[:0:-1], xs]) if xs[0] == 0 else np.concatenate([-xs[::-1], xs])
    Yv = np.concatenate([-ys[:0:-1], ys]) if ys[0] == 0 else np.concatenate([-ys[::-1], ys])
    top = M[::-1][:-1] if xs[0] == 0 else M[::-1]
    F = np.vstack([top, M])
    left = F[:, ::-1][:, :-1] if ys[0] == 0 else F[:, ::-1]
    return X, Yv, np.hstack([left, F])


def fig_slip_maps(recs, z, path, skeleton=None):
    from matplotlib.colors import TwoSlopeNorm
    fig = _fig(7.2, 3.0)
    for k, c in enumerate(CARRIERS):
        ax = fig.add_subplot(1, 2, k + 1)
        xs, ys, M = grid_matrix(recs, z, c)
        if not len(xs) or np.isnan(M).all():
            continue
        X, Yv, F = _full_map(xs, ys, M)
        vm = np.nanmax(np.abs(F))
        im = ax.pcolormesh(X, Yv, F.T, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vm, vm), shading="nearest", rasterized=True)
        if np.isfinite(F).all():
            ax.contour(X, Yv, F.T, [0], colors="k", linewidths=0.8)
        if skeleton is not None:
            for p, q in skeleton:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="0.35", lw=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("Δx (Å)")
        ax.set_ylabel("Δy (Å)")
        cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02)
        cb.set_label(f"V$_{{{'h' if c == 'hole' else 'e'}}}$ (meV)")
        _panel_label(ax, "(a)" if k == 0 else "(b)")
    return _save(fig, path)


def fig_xy_pairs(pairs, path, xlabel, ylabel, fit=True):
    """pairs: {carrier: (x, y, k)} → 两个子图的散点 + 过原点直线。"""
    fig = _fig(6.4, 3.0)
    for k, c in enumerate(CARRIERS):
        if c not in pairs:
            continue
        ax = fig.add_subplot(1, 2, k + 1)
        x, y, kk = pairs[c]
        x, y = np.asarray(x), np.asarray(y)
        ax.plot(x, y, MARK[c], ms=3.5, mfc="none" if c == "electron" else COL[c], mec=COL[c], alpha=0.85)
        if fit and kk is not None:
            xx = np.linspace(x.min(), x.max(), 2)
            ax.plot(xx, kk * xx, "-", color="0.25", lw=1)
        ax.axhline(0, color="0.8", lw=0.6, zorder=0)
        ax.axvline(0, color="0.8", lw=0.6, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(LAB[c], fontsize=10)
        _panel_label(ax, "(a)" if k == 0 else "(b)")
    return _save(fig, path)


def fig_parity(preds, path, names=MODELS):
    fig = _fig(2.4 * len(names), 4.8)
    for i, c in enumerate(CARRIERS):
        for j, m in enumerate(names):
            ax = fig.add_subplot(2, len(names), i * len(names) + j + 1)
            for tag, mk in (("interp", "o"), ("extrap", "s")):
                t = np.array([p[f"DFT_{c}"] for p in preds if p["set"] == tag and p[f"DFT_{c}"] is not None])
                q = np.array([p[f"{m}_{c}"] for p in preds if p["set"] == tag and p[f"DFT_{c}"] is not None])
                if len(t):
                    ax.plot(t, q, mk, ms=4, mfc=COL[c] if tag == "interp" else "none", mec=COL[c],
                            label="interpolation" if tag == "interp" else "extrapolation")
            allv = [p[f"DFT_{c}"] for p in preds if p[f"DFT_{c}"] is not None]
            if allv:
                lo, hi = min(allv), max(allv)
                pad = 0.05 * (hi - lo + 1)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", color="0.6", lw=0.8, zorder=0)
            ax.set_title(f"{m}, {LAB[c].lower()}", fontsize=9)
            if i == 1:
                ax.set_xlabel("V$_{DFT}$ (meV)")
            if j == 0:
                ax.set_ylabel("V$_{model}$ (meV)")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper left")
    return _save(fig, path)


def fig_energy_map(recs, z, path, key="E_total"):
    from matplotlib.colors import TwoSlopeNorm
    rs = [r for r in computed(recs) if _parallel(r) and abs(r["z"] - z) < 1e-4 and r.get(key) is not None]
    if len(rs) < 9:
        return None
    sub = [dict(r, hole=r[key]) for r in rs]
    xs, ys, M = grid_matrix(sub, z, "hole")
    X, Yv, F = _full_map(xs, ys, M)
    fig = _fig(3.8, 3.0)
    ax = fig.add_subplot(111)
    lo, hi = np.nanmin(F), np.nanmax(F)
    norm = TwoSlopeNorm(0, min(lo, -1e-6), max(hi, 1e-6)) if lo < 0 < hi else None
    im = ax.pcolormesh(X, Yv, F.T, cmap="RdBu_r" if norm else "Blues_r", norm=norm, shading="nearest", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("Δx (Å)")
    ax.set_ylabel("Δy (Å)")
    fig.colorbar(im, ax=ax, shrink=0.6).set_label("E$_{int}$ (kcal mol$^{-1}$)")
    return _save(fig, path)


# ----------------------------------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------------------------------
def _img(path):
    if not path or not Path(path).exists():
        return ""
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{html.escape(Path(path).stem)}"></figure>'


def _table(head, rows):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def scan_roles(project):
    """按扫描类型 / 名称判断用途：grid (训练网格)、validation、check、crystal。"""
    roles = {}
    for n in project.scans():
        s = project.meta["scans"][n]
        role = s.get("role")
        try:
            if project.scan_info(n).get("info", {}).get("clone_of"):
                role = role or "compare"
        except Exception:  # noqa: BLE001
            pass
        if not role:
            kind = s.get("kind", "grid")
            low = n.lower()
            if kind in ("validation", "check", "crystal", "local", "md"):
                role = kind
            elif "valid" in low:
                role = "validation"
            elif "check" in low or "sym" in low:
                role = "check"
            elif "cryst" in low:
                role = "crystal"
            else:
                role = "grid"
        roles[n] = role
    return roles


def _group_imported_checks(project, recs):
    """make_dimers.py 生成的对称检验 (没有 group / variant 列)：按对称等价代表分组。"""
    ops = project.ops
    out = []
    for r in recs:
        if r.get("set") == "check" or not r.get("group"):
            rep, op = canonical({k: r[k] for k in PARAM_KEYS}, ops)
            r = dict(r, set="check", group=str(hash(cfg_key(rep)) % 10 ** 8), variant=op)
        out.append(r)
    return out


def make_report(project, out_dir=None, lam=None, reorg_dir=None, T=300.0, energy="ADF", z0=3.5, tr=lambda s: s,
                on_progress=None, level=None):
    """生成 HTML 报告、图 (300 dpi PNG)、model/ 与 validation/ 目录。lam: {"hole": eV, "electron": eV}。
    level：建模所用的计算层级 (默认取第一个训练网格扫描的层级)；其他层级的扫描只出现在层级对照中。"""
    out_dir = Path(out_dir or project.root / "report")
    figs = out_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    (project.root / "model").mkdir(exist_ok=True)
    (project.root / "validation").mkdir(exist_ok=True)
    roles = scan_roles(project)
    data = {n: project.collect(n) for n in project.scans()}
    lev_of = {n: project.scan_info(n).get("level") for n in data}
    grids = [n for n in data if roles[n] == "grid"]
    level = level or (lev_of[grids[0]] if grids else None)
    for n in data:                                         # 只用同一层级建模与验证
        if roles[n] in ("grid", "validation", "check", "crystal") and level and lev_of[n] not in (level, "imported", None):
            roles[n] = "compare"
    sec, summary = [], []

    def S(title, body):
        sec.append(f"<section><h2>{html.escape(tr(title))}</h2>{body}</section>")

    def step(msg):
        if on_progress:
            on_progress(msg)

    # 1 单体与层级
    rep = project.meta["monomer"]["report"]
    devs = ", ".join(f"{k} {v:.1e}" for k, v in rep["op_deviation_A"].items() if k in rep["ops"])
    body = _table([tr("项目"), tr("内容")], [
        [tr("分子"), f"{rep['formula']} ({rep['n_atoms']} {tr('原子')})"],
        [tr("分子坐标系中的点群"), f"{rep['point_group']} ({', '.join(rep['ops'])})"],
        [tr("对称操作偏差 / Å"), devs or "—"], [tr("平面性 (重原子最大 |z|) / Å"), f"{rep['planarity_A']:.1e}"],
        [tr("C–C 键长范围 / Å"), "—" if not rep["CC_range_A"] else f"{rep['CC_range_A'][0]:.3f}–{rep['CC_range_A'][1]:.3f}"],
        [tr("长 × 宽 (重原子) / Å"), f"{rep['length_A']:.2f} × {rep['width_A']:.2f}"],
        [tr("单体来源"), project.meta["monomer"]["source"]]])
    lev = [[k, v["xc"] + (" + D3(BJ)" if v["d3"] else ""), v["basis"], v["dir"] + "/mono.t21"]
           for k, v in project.meta["levels"].items()]
    body += _table([tr("层级"), tr("泛函"), tr("基组"), tr("片段文件")], lev) if lev else ""
    S("1 单体与计算层级", body)
    summary.append(f"Monomer {rep['formula']}, {rep['point_group']}, planarity {rep['planarity_A']:.1e} A")

    # 2 扫描一览
    rows = []
    for n, recs in data.items():
        comp = [r for r in recs if not r["rep"]]
        info = project.scan_info(n)
        rows.append([n, tr(ROLE_ZH.get(roles[n], roles[n])), info.get("level", ""), len(recs), len(comp), sum(r["state"] == "done" for r in comp),
                     sum(r["state"] == "failed" for r in comp), sum(r["state"] == "pending" for r in comp),
                     sum(r["state"] == "excluded" for r in comp)])
    S("2 扫描一览", _table([tr("扫描"), tr("用途"), tr("层级"), tr("请求构型"), tr("计算构型"), tr("完成"), tr("失败"),
                         tr("待算"), tr("接触筛除")], rows))
    mism = sum(1 for recs in data.values() for r in recs if not r["rep"] and
               any(r.get(f"sign_mismatch_{c}") for c in CARRIERS))
    if mism:
        sec[-1] = sec[-1].replace("</section>", f"<p class='warn'>{tr('ADF 打印的 V 与由 J、S、e1、e2 重算的 V 符号不同的构型数')}："
                                                f"{mism}（{tr('已按重算值取符号')}）</p></section>")

    # 3 对称检验
    chk = [r for n, recs in data.items() if roles[n] == "check" for r in recs]
    if chk:
        step("对称检验")
        if not any(r.get("variant") for r in chk):
            chk = _group_imported_checks(project, chk)
        rows_, ch = symmetry_check(chk)
        t = [[r["group"], f"{r['base']['z']:.3f}, {r['base']['dx']:.3f}, {r['base']['dy']:.3f}, {r['base']['roll']:.2f}",
              r["n_ops"], _f(r["V_hole"]), _f(r["maxdev_hole"], 3), _f(r["V_electron"]), _f(r["maxdev_electron"], 3),
              _f(max(r["absdev_self_hole"] or 0, r["absdev_self_electron"] or 0), 3)] for r in rows_]
        body = _table([tr("组"), "z, Δx, Δy, φ", tr("等价变体数"), "V_h", tr("最大偏差"), "V_e", tr("最大偏差"),
                       tr("自身 C2 的 |V| 偏差")], t)
        chars = "；".join(f"{tr(CZ[c])}: " + ", ".join(f"{op} {'+1' if v > 0 else ('−1' if v < 0 else '?')}" for op, v in d.items())
                         for c, d in ch.items() if d)
        if chars:
            body += f"<p>{tr('前线轨道在 B 自身二重轴下的特征标')}：{html.escape(chars)}</p>"
        S("3 相位一致性：对称等价构型", body)
        mx = max([r["maxdev_hole"] or 0 for r in rows_] + [r["maxdev_electron"] or 0 for r in rows_] + [0])
        summary.append(f"Symmetry check: {len(rows_)} groups, max |dV| between equivalent configurations {mx:.3f} meV")

    # 4–7 训练网格
    train = [r for n, recs in data.items() if roles[n] == "grid" for r in recs]
    models = None
    if train:
        step("距离衰减与滑移图")
        fits = {c: distance_fit(train, c, z0) for c in CARRIERS}
        if any(fits.values()):
            p = fig_distance(fits, figs / "distance_fit.png")
            t = [[tr(CZ[c]), _f(d["beta"], 3), _f(d["A_z0"], 1), _f(d.get("V_z0"), 1), _f(d["R2"], 4), d["n"],
                  f"{d['range'][0]:.2f}–{d['range'][1]:.2f}"] for c, d in fits.items() if d]
            S("4 距离衰减 (共面二聚体)", _table([tr("载流子"), "β / Å⁻¹", f"A(z0={z0}) / meV", f"V(z0) DFT / meV", "R² (ln|V|)",
                                            "n", "z / Å"], t) + _img(p))
            for c, d in fits.items():
                if d:
                    summary.append(f"Distance fit {c}: beta = {d['beta']:.3f} 1/A, R2 = {d['R2']:.4f}")
        zs = grid_zs(train)
        body = ""
        skel = None
        try:
            from .viewer import bonds_for
            Yh = project.Y
            sym = project.symbols
            skel = [(Yh[i], Yh[j]) for i, j in bonds_for(sym, Yh) if sym[i] != "H" and sym[j] != "H"]
        except Exception:  # noqa: BLE001
            pass
        for z in zs:
            p = fig_slip_maps(train, z, figs / f"slip_map_z{z:.2f}.png", skel)
            ext = []
            for c in CARRIERS:
                xs, ys, M = grid_matrix(train, z, c)
                if np.isfinite(M).any():
                    ext.append(f"{tr(CZ[c])}: {np.nanmin(M):.1f} … {np.nanmax(M):.1f} meV")
            body += f"<h3>z = {z:.2f} Å</h3><p>{'；'.join(ext)}</p>" + _img(p)
        if zs:
            S("5 滑移图", body)
        # 可分离性
        body, sep_rows = "", []
        for c in CARRIERS:
            for z in zs:
                sv = svd_separability(grid_matrix(train, z, c)[2])
                if sv:
                    sep_rows.append([tr(CZ[c]), f"{z:.2f}", f"{100 * sv['share']:.2f} %", _f(sv["rmse_rank1"], 2)])
        if sep_rows:
            body += f"<h3>{tr('SVD：滑移矩阵的秩一近似')}</h3>" + _table([tr("载流子"), "z / Å", tr("第一分量占比"),
                                                                     tr("秩一重构 RMSE / meV")], sep_rows)
        cmp_rows, pairs = [], {}
        for z1, z2 in zip(zs[:-1], zs[1:]):
            for c in CARRIERS:
                m = compare_maps(train, z1, z2, c)
                if m:
                    bins = "; ".join(f"|Δy| {lo:g}–{hi:.0f}: {v:.3f}" for lo, hi, v, _ in m["bins_dy"])
                    cmp_rows.append([tr(CZ[c]), f"{z1:.2f} → {z2:.2f}", m["n"], f"{m['k']:.4f}", f"{m['beta_eff']:.3f}",
                                     f"{m['r']:.4f}", _f(m["rmse"], 2), f"{m['ratio_mean']:.3f} ± {m['ratio_std']:.3f}",
                                     m["sign_flips"], bins])
                    pairs[c] = (m["a"], m["b"], m["k"])
                    summary.append(f"Separability {c} {z1}->{z2}: k = {m['k']:.4f}, beta_eff = {m['beta_eff']:.3f}, "
                                   f"r = {m['r']:.4f}, RMSE = {m['rmse']:.2f} meV")
        if cmp_rows:
            p = fig_xy_pairs(pairs, figs / "map_comparison.png", f"V(z = {zs[0]:.2f} Å) (meV)",
                             f"V(z = {zs[1]:.2f} Å) (meV)")
            body += f"<h3>{tr('相邻 z 的滑移图逐点比较')}</h3>" + _table(
                [tr("载流子"), "z / Å", "n", "k", "β_eff / Å⁻¹", "r", "RMSE / meV", tr("|V| > 20% 处比值"), tr("变号点"),
                 tr("按 |Δy| 分箱的比值")], cmp_rows) + _img(p)
        if body:
            S("6 距离与滑移的可分离性", body)
        # V–S
        vs_rows, vpairs = [], {}
        for z in zs:
            for c in CARRIERS:
                v = vs_fit(train, z, c)
                if v:
                    vs_rows.append([tr(CZ[c]), f"{z:.2f}", v["n"], f"{v['k']:.0f}", f"{v['r']:.4f}", _f(v["rmse"], 2)])
                    if z == zs[0]:
                        vpairs[c] = (v["S"], v["V"], v["k"])
        if vs_rows:
            p = fig_xy_pairs(vpairs, figs / "V_vs_S.png", "S", "V (meV)")
            S("7 耦合与轨道重叠 (V = kS)", _table([tr("载流子"), "z / Å", "n", "k / meV", "r", "RMSE / meV"], vs_rows) + _img(p))
        # 模型
        if zs:
            step("拟合模型")
            try:
                models = PaperModels(train, z_ref=z0, z0=z0)
            except ValueError:
                models = None
        if models:
            models.to_json(project.root / "model" / "model.json",
                           {"created": time.strftime("%Y-%m-%d %H:%M"), "train_scans": [n for n in data if roles[n] == "grid"]})
            models.write_grids(project.root / "model")
            t = []
            for k in ("A", "ax", "kx", "ay", "ky"):
                t.append([k] + [(f"{models.analytic[c]['params'][k]:.4g} ± {models.analytic[c]['stderr'][k]:.2g}"
                                 if models.analytic[c] else "—") for c in CARRIERS])
            t.append(["RMSE / meV"] + [_f(models.analytic[c]["rmse"]) if models.analytic[c] else "—" for c in CARRIERS])
            t.append(["R²"] + [_f(models.analytic[c]["R2"], 4) if models.analytic[c] else "—" for c in CARRIERS])
            t.append(["β / Å⁻¹ (" + tr(models.beta["hole"][1]) + ")"] + [f"{models.beta[c][0]:.3f}" for c in CARRIERS])
            S("8 滑移函数的解析式", f"<p>V = A exp(−a<sub>x</sub>Δx² − a<sub>y</sub>Δy²) cos(k<sub>x</sub>Δx) cos(k<sub>y</sub>Δy)，"
                               f"z = {models.z_ref:.2f} Å</p>" + _table([tr("参数"), tr("空穴"), tr("电子")], t))
    # 9 验证
    valid = [r for n, recs in data.items() if roles[n] == "validation" for r in recs]
    if valid and models:
        step("验证集")
        stats, preds = validate(models, valid)
        with open(project.root / "validation" / "predictions.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(preds[0]))
            w.writeheader()
            w.writerows(preds)
        lines = []
        t = []
        for s in stats:
            t.append([tr(CZ[s["carrier"]]), s["model"], _f(s["MAE"]), _f(s["RMSE"]), _f(s["RMSE_interp"]), _f(s["RMSE_extrap"]),
                      _f(s["max"]), _f(s["R2"], 4), _f(s["sign"], 1)])
            lines.append(f"{s['carrier']:<9}{s['model']:<4} MAE {s['MAE']:.2f}  RMSE {s['RMSE']:.2f}  interp {s['RMSE_interp']:.2f}  "
                         f"extrap {s['RMSE_extrap']:.2f}  max {s['max']:.2f}  R2 {s['R2']:.4f}  sign {s['sign']:.1f}%")
        n_i = stats[0]["n_interp"] if stats else 0
        n_e = stats[0]["n_extrap"] if stats else 0
        (project.root / "validation" / "validation_summary.txt").write_text(
            f"Validation set: {len(preds)} dimers ({n_i} interpolation, {n_e} extrapolation)\n" + "\n".join(lines) + "\n")
        p = fig_parity(preds, figs / "parity.png")
        wrong = [p_ for p_ in preds for c in CARRIERS for m in MODELS
                 if p_[f"DFT_{c}"] is not None and np.sign(p_[f"DFT_{c}"]) != np.sign(p_[f"{m}_{c}"])]
        note = ""
        if wrong:
            note = "<p class='warn'>" + tr("符号预测错误的构型") + "：" + "; ".join(
                sorted({f"{p_['id']} (|V_DFT| h {abs(p_['DFT_hole']):.1f}, e {abs(p_['DFT_electron']):.1f} meV)" for p_ in wrong})) + "</p>"
        S("9 随机构型验证", f"<p>{len(preds)} {tr('个构型')}（{tr('内插')} {n_i}，{tr('外推')} {n_e}）</p>" + _table(
            [tr("载流子"), tr("模型"), "MAE", "RMSE", tr("RMSE 内插"), tr("RMSE 外推"), tr("最大误差"), "R²", tr("符号一致 %")], t)
            + note + _img(p))
        for s in stats:
            summary.append(f"Validation {s['carrier']} {s['model']}: RMSE {s['RMSE']:.2f} meV, max {s['max']:.2f}, sign {s['sign']:.0f}%")
    # 10 层级对照
    rows_, pairs = [], {}
    for n in project.scans():
        src = project.scan_info(n).get("info", {}).get("clone_of")
        if src and src in data:
            la, lb = project.scan_info(src).get("level"), project.scan_info(n).get("level")
            for c in CARRIERS:
                m = compare_levels(data[src], data[n], c)
                if m:
                    rows_.append([f"{src} ({la})", f"{n} ({lb})", tr(CZ[c]), m["n"], f"{m['k']:.3f}", f"{m['r']:.4f}",
                                  _f(m["rmse"]), _f(m["mae_abs"]), f"{m['sign_agree']:.1f}", "+" if m["global_sign"] > 0 else "−"])
                    pairs[c] = (m["a"], m["b"], m["k"])
            if pairs:
                p = fig_xy_pairs(pairs, figs / f"level_{n}.png", f"V {la} (meV)", f"V {lb} (meV)")
    for n in project.scans():
        for d, res in project.collect_gaussian_ti(n).items():
            if not res:
                continue
            base = {r["id"]: r for r in data[n] if r["state"] == "done"}
            for c in CARRIERS:
                a = [abs(base[i][c]) for i in res if i in base and base[i][c] is not None]
                b = [abs(res[i][c]) for i in res if i in base and base[i][c] is not None]
                if len(a) >= 3:
                    a, b = np.array(a), np.array(b)
                    k = float(a @ b / (a @ a))
                    rows_.append([f"{n} ({project.scan_info(n).get('level')})", f"Gaussian {d[4:]}", tr(CZ[c]), len(a), f"{k:.3f}",
                                  f"{np.corrcoef(a, b)[0, 1]:.4f}", _f(float(np.sqrt(((b - k * a) ** 2).mean()))),
                                  _f(float(np.abs(b - a).mean())), "|V|", ""])
    if rows_:
        S("10 不同泛函 / 基组的对照", "<p>" + tr("比例因子 k 为过原点最小二乘 V_B ≈ k V_A；不同层级的片段轨道整体相位可能相反，"
                                          "符号一致率按整体符号校正后计算") + "</p>" + _table(
            [tr("层级 A"), tr("层级 B"), tr("载流子"), "n", "k", "r", "RMSE / meV", "MAE(|V|) / meV", tr("符号一致 %"),
             tr("整体符号")], rows_) + "".join(_img(f) for f in sorted(figs.glob("level_*.png"))))
    # 11 相互作用能与 Boltzmann 平均
    step("相互作用能")
    body, brow = "", []
    for n, recs in data.items():
        if roles[n] not in ("grid", "compare") or project.scan_info(n).get("kind") not in ("grid", "imported", None):
            continue
        E, src = {}, ""
        if energy == "ADF":
            E = {r["id"]: r["E_total"] for r in recs if not r["rep"] and r.get("E_total") is not None}
            src = "ADF EDA" + ("" if project.meta["levels"].get(project.scan_info(n).get("level"), {}).get("d3") else
                               " (" + tr("无色散校正") + ")")
        elif energy == "UFF":
            E = project.uff_energies(n)
            src = "UFF"
        else:
            cp = project.collect_cp(n)
            E = next((v for v in cp.values() if v), {})
            src = "Gaussian counterpoise"
        if len(E) < 2:
            continue
        for z in grid_zs(recs):
            sub = [r for r in recs if abs(r["z"] - z) < 1e-4 and _parallel(r)]
            ek = [dict(r, E_sel=E.get(r["rep"] or r["id"])) for r in sub if not r["rep"]]
            p = fig_energy_map(ek, z, figs / f"energy_{n}_z{z:.2f}.png", key="E_sel")
            if p:
                body += f"<h3>{html.escape(n)}, z = {z:.2f} Å ({html.escape(src)})</h3>" + _img(p)
            for c in CARRIERS:
                bz = boltzmann(sub, E, c, T)
                if bz:
                    top = bz["top"][0]
                    brow.append([n, f"{z:.2f}", tr(CZ[c]), src, bz["n"], _f(bz["V_mean"]), _f(bz["V_rms"]),
                                 _f(bz["V_unweighted_rms"]), _f(bz["n_eff"], 1),
                                 f"{top['dx']:.2f}, {top['dy']:.2f} ({top['E']:.2f} kcal/mol)"])
    if brow:
        warn = "" if energy != "ADF" else ("<p class='warn'>" + tr("ADF 的 PW91 等 GGA 泛函不含色散，π–π 构型的能量偏高；"
                                                                  "定量 Boltzmann 权重请用 Gaussian 抵消校正 (含 D3) 或带 D3 的层级") + "</p>")
        S("11 相互作用能与 Boltzmann 平均耦合", f"<p>T = {T:.0f} K，w ∝ exp(−E/RT)</p>" + warn + _table(
            [tr("扫描"), "z / Å", tr("载流子"), tr("能量来源"), "n", "<V>", "√<V²>", tr("不加权 √<V²>"), tr("有效构型数"),
             tr("权重最大的构型 Δx, Δy")], brow) + body)
    # 12 重组能与晶体二聚体
    reorg = None
    if reorg_dir:
        try:
            from .reorganization import analyze_reorg
            reorg = analyze_reorg(reorg_dir)
            lam = {c: reorg["results"][c]["lambda_eV"] for c in CARRIERS if c in reorg["results"]}
        except Exception as exc:  # noqa: BLE001
            S("12 重组能", f"<p class='warn'>{html.escape(str(exc))}</p>")
    if lam:
        from .mobility import marcus_rate
        t = [[tr(CZ[c]), f"{1000 * lam[c]:.1f}"] for c in CARRIERS if c in lam]
        body = _table([tr("载流子"), "λ / meV"], t)
        if reorg:
            body += f"<p>{tr('四点法')}：{html.escape(str(reorg.get('level')))}" + (
                f"，{tr('单点')} {html.escape(str(reorg.get('sp_level')))}" if reorg.get("sp_level") else "") + "</p>"
        cry = [r for n, recs in data.items() if roles[n] == "crystal" for r in recs if not r["rep"]]
        if cry:
            info = {}
            for n in data:
                if roles[n] == "crystal":
                    for pinfo in project.scan_info(n).get("info", {}).get("pairs", []):
                        info[cfg_key(pinfo)] = pinfo
            t = []
            for r in cry:
                pi = info.get(cfg_key(r), {})
                row = [pi.get("variant", r.get("variant", "")), _f(pi.get("r_centroid"), 3),
                       f"{r['z']:.2f}, {abs(r['dx']):.2f}, {abs(r['dy']):.2f}", f"{r['roll']:.1f}", _f(pi.get("rmsd_heavy_A"), 3)]
                for c in CARRIERS:
                    v = r.get(c)
                    row += [_f(abs(v) if v is not None else None), "—" if v is None or c not in lam else
                            f"{marcus_rate(abs(v) / 1000, lam[c], T):.2e}"]
                t.append(row)
            body += f"<h3>{tr('晶体近邻二聚体 (冻结单体)')}</h3>" + _table(
                [tr("二聚体"), "r / Å", "z, |Δx|, |Δy| / Å", "φ / °", "RMSD / Å", "|V_h| / meV", "k_h / s⁻¹", "|V_e| / meV",
                 "k_e / s⁻¹"], t)
        S("12 重组能与 Marcus 速率", body)
        summary += [f"lambda {c}: {1000 * lam[c]:.1f} meV" for c in lam]
    css = ("body{font-family:Segoe UI,Microsoft YaHei,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;"
           "color:#1b1b1b;background:#fff}h1{font-size:22px}h2{font-size:17px;border-bottom:1px solid #ccc;padding-bottom:4px;"
           "margin-top:28px}h3{font-size:14px}table{border-collapse:collapse;margin:8px 0;font-size:13px}th,td{border:1px solid "
           "#d0d0d0;padding:3px 8px;text-align:center}th{background:#eef2f7}img{max-width:100%}figure{margin:8px 0}"
           ".warn{color:#9a3412}.meta{color:#555;font-size:13px}")
    from . import __version__
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(project.meta['name'])} — Mobilis</title>"
           f"<style>{css}</style></head><body><h1>{html.escape(tr('扫描报告'))}：{html.escape(project.meta['name'])}</h1>"
           f"<p class='meta'>{time.strftime('%Y-%m-%d %H:%M')} · Mobilis {__version__} · {html.escape(str(project.root))}</p>"
           + "".join(sec) + "</body></html>")
    if tr("空穴") != "空穴":                                 # 英文界面：全角标点换成半角
        for a, b in (("；", "; "), ("，", ", "), ("：", ": "), ("（", " ("), ("）", ")"), ("、", ", ")):
            doc = doc.replace(a, b)
    path = out_dir / "report.html"
    path.write_text(doc, encoding="utf-8")
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    all_recs = [r for recs in data.values() for r in recs]
    write_records([r for r in all_recs if not r["rep"]], out_dir / "all_results.csv")
    return {"html": path, "summary": summary, "models": models, "figs": sorted(figs.glob("*.png"))}
