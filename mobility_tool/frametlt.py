"""Transient localization with the Hamiltonians of the MD frames (frame-resolved TLT).

The standard TLT input is one Gaussian of width σ_V per bond type, drawn independently for every bond. With a coupling
surface every bond of every MD frame has its own coupling, so the Hamiltonian of a frame can be built directly on the MD
supercell. Three sets of Hamiltonians on the same lattice separate the two approximations of the Gaussian input:

  gaussian   V of every bond drawn independently from N(⟨V⟩, σ_V) of its bond type (standard TLT)
  marginal   V of every bond drawn independently from the MD distribution of its bond type (non-Gaussian, uncorrelated)
  frames     V of every bond taken from the same MD frame (non-Gaussian and correlated between bonds)

marginal − gaussian is the effect of the shape of the distribution; frames − marginal that of the correlations.
The lattice is the MD supercell, so the supercell must be large in the transport plane (≥ 8 cells per axis is
enforced; 12–16 are recommended). Layers of the supercell are not coupled in the TLT Hamiltonian; with two cells
along the third direction each Hamiltonian holds two independent layers.
"""
import numpy as np
from scipy import sparse as _sparse

from .mobility import HBAR_EVS, KB_EV

MIN_CELLS = 8


# ----------------------------------------------------------------------------------------------
# TLT for a given set of Hamiltonians
# ----------------------------------------------------------------------------------------------
def tlt_L2(H, D, T, hbar_tau):
    """Squared transient localization length tensor (Å²) of one Hamiltonian (eV); D[i, j] = r_j − r_i (Å)."""
    beta = 1 / (KB_EV * T)
    e, c = np.linalg.eigh(H)
    w = np.exp(-beta * (e - e.min()))
    w /= w.sum()
    axes = [k for k in range(3) if np.any(D[:, :, k])]
    Ms = {k: c.T @ np.asarray(_sparse.csr_matrix(H * D[:, :, k]) @ c) for k in axes}
    de = e[:, None] - e[None, :]
    wk = w[:, None] * (2.0 / (hbar_tau ** 2 + de ** 2))
    L2 = np.zeros((3, 3))
    for i, a in enumerate(axes):
        Wa = wk * Ms[a]
        for b in axes[i:]:
            L2[a, b] = L2[b, a] = float((Wa * Ms[b]).sum())
    return L2


def _mu_stats(L2s, T, hbar_tau):
    """Mean mobility tensor of a set of realizations, its principal values, and the jackknife standard error of the
    largest one."""
    L2s = np.asarray(L2s)
    f = 1e-16 / (2 * KB_EV * T * (HBAR_EVS / hbar_tau))
    mu = L2s.mean(0) * f
    pv = np.linalg.eigvalsh(mu)[::-1]
    n = len(L2s)
    if n > 1:
        jk = np.array([np.linalg.eigvalsh((L2s.sum(0) - L2s[k]) / (n - 1) * f)[-1] for k in range(n)])
        se = float(np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()))
    else:
        se = float("nan")
    return {"mu_tensor": mu, "mu": float(pv[0]), "mu2": float(pv[1]), "se": se, "n": n,
            "L_A": float(np.sqrt(max(np.trace(L2s.mean(0)) / 2, 0.0)))}


# ----------------------------------------------------------------------------------------------
# bonds of the MD supercell and their mapping onto the coupling surfaces
# ----------------------------------------------------------------------------------------------
def _angles(R):
    """R = R_z(γ) R_y(θ) R_x(φ) → (γ, θ, φ) in degrees for an array of rotation matrices (…, 3, 3)."""
    pitch = np.degrees(np.arcsin(-np.clip(R[..., 2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(R[..., 2, 1], R[..., 2, 2]))
    gamma = np.degrees(np.arctan2(R[..., 1, 0], R[..., 0, 0]))
    return np.stack([gamma, pitch, roll], axis=-1)


def _params(R, t, rev, S, G):
    """Dimer (R, t) in the frame of A → parameters (z, Δx, Δy, γ, θ, φ) after: exchange of the two molecules (rev),
    a symmetry operation S of the monomer applied to the whole dimer, and a twofold axis G of molecule B."""
    if rev:
        Rt = np.swapaxes(R, -1, -2)
        R, t = Rt, -np.einsum("...ij,...j->...i", Rt, t)
    R = (S[:, None] * R * S[None, :]) * G[None, :]
    t = t * S
    return np.concatenate([t[..., 2:3], t[..., 0:1], t[..., 1:2], _angles(R)], axis=-1)


def _dist(q, q0):
    d = np.asarray(q, float) - np.asarray(q0, float)
    d[..., 3:] = (d[..., 3:] + 180.0) % 360.0 - 180.0
    return d


def keep_axes(ceds, v0):
    """The two lattice directions that carry the couplings. v0: {gid: |V| at the crystal geometry}."""
    w = np.zeros(3)
    for (i, j, v, t, gid) in ceds.edges:
        w += (np.array(t) != 0) * float(v0.get(gid, 0.0)) ** 2
    return tuple(sorted(int(i) for i in np.argsort(w)[::-1][:2]))


def crystal_v0(ceds, surfaces):
    from .ceds import CARRIERS
    out = {}
    for p in ceds.pairs:
        s = surfaces(p["gid"], p["center"]) if callable(surfaces) else surfaces.get(p["gid"])
        if s is not None:
            out[p["gid"]] = max(abs(float(s(p["q0"], c)[0])) for c in CARRIERS)
    return out


def bond_types(ceds, surfaces, axes, tol_len=0.08, tol_ang=2.0):
    """Unique physical bonds of the unit cell inside the transport plane, each with the operations that map its
    geometry onto the region of the coupling surface of its pair. Raises ValueError if a bond cannot be mapped."""
    from .scanflow import OPS, PROPER
    if any(getattr(ceds, "mirror", [])):
        raise ValueError("frame-resolved TLT is not available for crystals with enantiomeric molecules")
    lat = np.asarray(ceds.crys["lattice"], float)
    by_gid = {p["gid"]: p for p in ceds.pairs}
    whole = [np.array(OPS[o], float) for o in ceds.ops]
    selfs = [("E", np.ones(3))] + [(o, np.array(OPS[o], float)) for o in PROPER if o in ceds.ops]
    out = []
    for (i, j, v, t, gid) in ceds.edges:
        t = tuple(int(x) for x in t)
        if any(t[a] != 0 for a in range(3) if a not in axes):
            continue
        if not ((i, j) < (j, i) or (i == j and t > (0, 0, 0))):
            continue
        s = surfaces(gid, by_gid[gid]["center"]) if callable(surfaces) else surfaces.get(gid)
        if s is None:
            continue
        axA, cA = ceds.frames[i]
        axB, cB = ceds.frames[j]
        R = np.asarray(axA, float).T @ np.asarray(axB, float)
        tv = np.asarray(axA, float).T @ (np.asarray(cB, float) + np.array(t, float) @ lat - np.asarray(cA, float))
        q0 = np.asarray(by_gid[gid]["q0"], float)
        best = None
        for rev in (False, True):
            for S in whole:
                for name, G in selfs:
                    d = _dist(_params(R, tv, rev, S, G), q0)
                    score = max(np.abs(d[:3]).max() / tol_len, np.abs(d[3:]).max() / tol_ang)
                    if best is None or score < best[0]:
                        best = (score, rev, S, name, G)
        if best[0] > 1.0:
            raise ValueError(f"bond {i}→{j} {t} of pair G{gid} does not map onto the coupling surface "
                             f"(deviation {best[0]:.1f} tolerances)")
        out.append({"i": i, "j": j, "t": t, "v": np.asarray(v, float), "gid": gid, "rev": best[1], "S": best[2],
                    "g": best[3], "G": best[4], "surface": s, "q0": q0})
    return out


def bond_couplings(ceds, md, frames, bonds, recenter=True, progress=None):
    """V of every bond instance: {carrier: [array (frames, cells) per bond type]} in meV, and the site indices."""
    from .ceds import CARRIERS
    Qm, o = md.mol_frames(frames["c"], frames["Q"])
    cells = np.array(md.cells)
    qs, idx = [], []
    for b in bonds:
        A = np.array([md.index(cl, b["i"]) for cl in md.cells])
        tb = cells + np.array(b["t"])
        B = np.array([md.index(tuple(x), b["j"]) for x in tb])
        shift = (tb // np.array(md.sc)) @ md.box
        QA, QB = Qm[:, A], Qm[:, B]
        R = np.einsum("fnji,fnjk->fnik", QA, QB)
        t = np.einsum("fnji,fnj->fni", QA, o[:, B] + shift - o[:, A])
        qs.append(_params(R, t, b["rev"], b["S"], b["G"]))
        idx.append((A, B))
    if recenter:                                    # the force field decides the fluctuations, not the mean structure
        for gid in {b["gid"] for b in bonds}:
            ks = [k for k, b in enumerate(bonds) if b["gid"] == gid]
            q0 = bonds[ks[0]]["q0"]
            d = np.concatenate([_dist(qs[k], q0).reshape(-1, 6) for k in ks])
            m = d.mean(0)
            for k in ks:
                qs[k] = q0 + _dist(qs[k], q0) - m
    V = {c: [] for c in CARRIERS}
    import time
    t0 = time.time()
    for k, (b, q) in enumerate(zip(bonds, qs)):
        if progress and k:
            el = time.time() - t0
            progress(f"frame TLT couplings: bond {k}/{len(bonds)}, {el:.0f} s, about {el / k * (len(bonds) - k):.0f} s left")
        for c in CARRIERS:
            v = ceds.chi(c, b["g"]) * np.asarray(b["surface"](q.reshape(-1, 6), c), float)
            V[c].append(v.reshape(q.shape[:2]))
    return V, idx


# ----------------------------------------------------------------------------------------------
# the comparison
# ----------------------------------------------------------------------------------------------
STAT_FRAMES = 300          # frames used for the coupling statistics (Gaussian / marginal draws, shape, correlations)


def frame_tlt(ceds, md, frames, surfaces, T, hbar_tau=0.005, n_real=24, recenter=True, seed=1, bonds=None, axes=None,
              stride=None, progress=None):
    """Gaussian, marginal and frame-resolved TLT on the MD supercell. Returns {carrier: {kind: stats}, 'bonds': …}.
    The couplings are evaluated on every `stride`-th frame (default: about STAT_FRAMES frames) plus the n_real frames
    whose Hamiltonians enter the frame-resolved TLT; frames 20 fs apart are strongly correlated, so the statistics
    do not change, and with an EHT baseline the evaluation time falls by the same factor."""
    from .ceds import CARRIERS
    if axes is None:
        axes = keep_axes(ceds, crystal_v0(ceds, surfaces))
    if min(md.sc[a] for a in axes) < MIN_CELLS:
        raise ValueError(f"超胞 {md.sc} 太小，不能作为 TLT 晶格：沿晶格方向 {', '.join('abc'[a] for a in axes)} "
                         f"至少需要 {MIN_CELLS} 个晶胞")
    if bonds is None:
        bonds = bond_types(ceds, surfaces, axes)
    nf0 = len(frames["c"])
    sel0 = np.unique(np.linspace(0, nf0 - 1, min(n_real, nf0)).round().astype(int))
    stride = max(1, int(stride if stride else nf0 // STAT_FRAMES))
    use = np.union1d(np.arange(0, nf0, stride), sel0)
    sub = {k: (v[use] if isinstance(v, np.ndarray) and v.ndim and len(v) == nf0 else v) for k, v in frames.items()}
    if progress:
        progress(f"frame TLT: couplings of {len(bonds)} bonds on {len(use)} of {nf0} frames (stride {stride})")
    V, idx = bond_couplings(ceds, md, sub, bonds, recenter, progress)
    nf = nf0
    sel = np.searchsorted(use, sel0)
    N = md.N
    D = np.zeros((N, N, 3))
    for b, (A, B) in zip(bonds, idx):
        np.add.at(D, (A, B), b["v"])
        np.add.at(D, (B, A), -b["v"])
    out = {"axes": axes, "n_frames": nf, "frames_used": len(sel), "n_sites": N, "sizes": tuple(md.sc),
           "stat_frames": int(len(use)), "stride": stride,
           "frame_spacing_fs": float(frames.get("dt_frame_fs", 20.0)) * (sel0[1] - sel0[0] if len(sel0) > 1 else 0),
           "bonds": []}
    for k, b in enumerate(bonds):
        row = {"bond": k + 1, "gid": b["gid"], "i": b["i"], "j": b["j"], "t": b["t"], "g": b["g"], "rev": b["rev"]}
        for c in CARRIERS:
            x = V[c][k].ravel()
            m, s = x.mean(), x.std(ddof=1)
            z = (x - m) / (s or 1.0)
            row[c] = {"mean": float(m), "std": float(s), "skew": float((z ** 3).mean()),
                      "kurt": float((z ** 4).mean() - 3.0)}
        out["bonds"].append(row)
    # correlation between the couplings of two bonds that share a molecule (same frame)
    cor = {c: [] for c in CARRIERS}
    for a in range(len(bonds)):
        for b_ in range(a + 1, len(bonds)):
            shared = None
            for sa, na in ((idx[a][0], "A"), (idx[a][1], "B")):
                for sb, nb in ((idx[b_][0], "A"), (idx[b_][1], "B")):
                    if sa[0] % md.nk == sb[0] % md.nk:              # same molecule of the cell
                        shared = (sa, sb)
            if shared is None:
                continue
            order = np.argsort(shared[1])
            pos = order[np.searchsorted(shared[1][order], shared[0])]
            for c in CARRIERS:
                x, y = V[c][a].ravel(), V[c][b_][:, pos].ravel()
                sx, sy = x.std(), y.std()
                r = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx > 0 and sy > 0 else float("nan")
                cor[c].append({"bonds": (a + 1, b_ + 1), "gids": (bonds[a]["gid"], bonds[b_]["gid"]), "r": float(r)})
    out["correlations"] = cor
    rng = np.random.default_rng(seed)
    for c in CARRIERS:
        sgn = -1.0 if c == "hole" else 1.0
        L2 = {"gaussian": [], "marginal": [], "frames": []}
        mean = [v.mean() for v in V[c]]
        std = [v.std(ddof=1) for v in V[c]]
        nc = len(md.cells)
        for f in sel:
            draws = {"frames": [v[f] for v in V[c]],
                     "gaussian": [rng.normal(mean[k], std[k], nc) for k in range(len(bonds))],
                     "marginal": [rng.choice(V[c][k].ravel(), nc) for k in range(len(bonds))]}
            for kind, vals in draws.items():
                H = np.zeros((N, N))
                for (A, B), v in zip(idx, vals):
                    np.add.at(H, (A, B), sgn * v / 1000.0)
                    np.add.at(H, (B, A), sgn * v / 1000.0)
                L2[kind].append(tlt_L2(H, D, T, hbar_tau))
        out[c] = {kind: _mu_stats(v, T, hbar_tau) for kind, v in L2.items()}
    return out


def report(res):
    """Text table of a frame_tlt result (Chinese with the Chinese interface)."""
    from . import i18n
    if i18n.CLI_LANG == "zh":
        return _report_zh(res)
    lines = [f"Frame-resolved TLT on the MD supercell {'×'.join(map(str, res['sizes']))} ({res['n_sites']} molecules; "
             f"{res['frames_used']} frames, {res['frame_spacing_fs']:.0f} fs apart; coupling statistics from "
             f"{res.get('stat_frames', res['n_frames'])} of {res['n_frames']} frames)",
             f"{'':10s}{'Gaussian':>16s}{'MD marginals':>16s}{'MD frames':>16s}{'shape':>9s}{'correlation':>13s}"]
    for c in ("hole", "electron"):
        g, m, f = (res[c][k] for k in ("gaussian", "marginal", "frames"))
        lines.append(f"{'μ ' + c:10s}" + "".join(f"{x['mu']:9.3g} ±{x['se']:5.2g}" for x in (g, m, f))
                     + f"{100 * (m['mu'] / g['mu'] - 1):+8.1f}%{100 * (f['mu'] / m['mu'] - 1):+12.1f}%")
    lines.append("shape = marginals against Gaussian; correlation = frames against marginals; ± = standard error "
                 "(cm² V⁻¹ s⁻¹)")
    lines.append(f"{'bond':5s}{'pair':>5s}{'⟨V⟩h':>9s}{'σh':>7s}{'skew':>7s}{'kurt':>7s}{'⟨V⟩e':>9s}{'σe':>7s}{'skew':>7s}{'kurt':>7s}")
    for b in res["bonds"]:
        h, e = b["hole"], b["electron"]
        lines.append(f"{b['bond']:<5d}{'G' + str(b['gid']):>5s}{h['mean']:9.2f}{h['std']:7.2f}{h['skew']:7.2f}{h['kurt']:7.2f}"
                     f"{e['mean']:9.2f}{e['std']:7.2f}{e['skew']:7.2f}{e['kurt']:7.2f}")
    for c in ("hole", "electron"):
        cc = res["correlations"][c]
        if cc:
            lines.append(f"correlation of the {c} couplings of bonds sharing a molecule: "
                         + ", ".join(f"{x['bonds'][0]}–{x['bonds'][1]} {x['r']:+.2f}" for x in cc))
    return "\n".join(lines)


def export_csv(rows, path):
    """rows: [(T, frame_tlt result)] → csv with the three mobilities and the bond statistics."""
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["T_K", "carrier", "kind", "mu", "mu2", "se", "L_A", "n_real", "n_sites"])
        for T, r in rows:
            for c in ("hole", "electron"):
                for kind in ("gaussian", "marginal", "frames"):
                    x = r[c][kind]
                    w.writerow([T, c, kind, f"{x['mu']:.5g}", f"{x['mu2']:.5g}", f"{x['se']:.3g}", f"{x['L_A']:.4g}",
                                x["n"], r["n_sites"]])
        w.writerow([])
        w.writerow(["T_K", "bond", "group", "i", "j", "t", "carrier", "V_mean_meV", "sigma_meV", "skewness",
                    "excess_kurtosis"])
        for T, r in rows:
            for b in r["bonds"]:
                for c in ("hole", "electron"):
                    x = b[c]
                    w.writerow([T, b["bond"], b["gid"], b["i"], b["j"], " ".join(map(str, b["t"])), c,
                                f"{x['mean']:.4f}", f"{x['std']:.4f}", f"{x['skew']:.4f}", f"{x['kurt']:.4f}"])
        w.writerow([])
        w.writerow(["T_K", "carrier", "bond_a", "bond_b", "group_a", "group_b", "correlation"])
        for T, r in rows:
            for c in ("hole", "electron"):
                for x in r["correlations"][c]:
                    w.writerow([T, c, *x["bonds"], *x["gids"], f"{x['r']:.4f}"])


def _report_zh(res):
    car = {"hole": "空穴", "electron": "电子"}
    lines = [f"逐帧 TLT：MD 超胞 {'×'.join(map(str, res['sizes']))} ({res['n_sites']} 个分子；{res['frames_used']} 帧，"
             f"相隔 {res['frame_spacing_fs']:.0f} fs；耦合统计取自 {res['n_frames']} 帧中的 "
             f"{res.get('stat_frames', res['n_frames'])} 帧)",
             f"{'':8s}{'高斯抽样':>14s}{'MD 分布抽样':>13s}{'MD 帧':>14s}{'形状':>8s}{'相关':>11s}"]
    for c in ("hole", "electron"):
        g, m, f = (res[c][k] for k in ("gaussian", "marginal", "frames"))
        lines.append(f"{'μ ' + car[c]:8s}" + "".join(f"{x['mu']:9.3g} ±{x['se']:5.2g}" for x in (g, m, f))
                     + f"{100 * (m['mu'] / g['mu'] - 1):+8.1f}%{100 * (f['mu'] / m['mu'] - 1):+12.1f}%")
    lines.append("形状 = MD 分布独立抽样相对高斯抽样；相关 = MD 帧相对 MD 分布独立抽样；± 为标准误差 (cm² V⁻¹ s⁻¹)")
    lines.append(f"{'键':4s}{'二聚体':>4s}{'⟨V⟩h':>9s}{'σh':>7s}{'偏度':>5s}{'峰度':>5s}{'⟨V⟩e':>9s}{'σe':>7s}{'偏度':>5s}{'峰度':>5s}")
    for b in res["bonds"]:
        h, e = b["hole"], b["electron"]
        lines.append(f"{b['bond']:<5d}{'G' + str(b['gid']):>5s}{h['mean']:9.2f}{h['std']:7.2f}{h['skew']:7.2f}{h['kurt']:7.2f}"
                     f"{e['mean']:9.2f}{e['std']:7.2f}{e['skew']:7.2f}{e['kurt']:7.2f}")
    for c in ("hole", "electron"):
        cc = res["correlations"][c]
        if cc:
            lines.append(f"共用分子的键之间{car[c]}耦合的相关系数："
                         + "，".join(f"{x['bonds'][0]}–{x['bonds'][1]} {x['r']:+.2f}" for x in cc))
    return "\n".join(lines)
