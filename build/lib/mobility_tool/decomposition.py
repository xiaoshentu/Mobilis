"""重组能分解（谐振近似，与 DUSHIN 程序思路一致）。

对每个态（中性 / 离子）分别处理：
  λ1 ：中性态势能面上从离子几何弛豫到中性几何 → 用中性态 Hessian，位移 Δx = x(离子) - x(中性)
  λ2 ：离子态势能面上从中性几何弛豫到离子几何 → 用离子态 Hessian，位移 Δx = x(中性) - x(离子)

1) 简正模式分解 / 黄昆因子:
     ΔQ_i = L_i^T M^(1/2) Δx,   λ_i = ½ ω_i² ΔQ_i²,   S_i = λ_i / (ħ ω_i)
2) 内坐标分解 (键长 / 键角 / 二面角):
     Δq = B Δx,  F_q = (B⁺)ᵀ H B⁺,  λ_k = ½ Δq_k (F_q Δq)_k ,  Σ_k λ_k = ½ Δxᵀ P H P Δx
3) 原子分解:  λ_a = ½ Σ_{i∈a} Δx_i (H Δx)_i
两个几何先做质量加权的 Kabsch 叠合以消除平动/转动。
"""
import csv
from pathlib import Path

import numpy as np

from .gaussian_io import BOHR2ANG, COVALENT_RADII, HARTREE2EV, read_fchk_hessian

AMU2ME = 1822.888486209
AU2CM = 219474.6313705


# ----------------------------------------------------------------------------------------------
# 几何工具
# ----------------------------------------------------------------------------------------------
def kabsch_align(ref, mob, weights, return_rotation=False):
    """把 mob 叠合到 ref 上（加权），返回叠合后的 mob（及转动矩阵 R: 新坐标 = (旧-质心)·R + 质心）。"""
    w = np.asarray(weights, float)[:, None]
    cr = (w * ref).sum(0) / w.sum()
    cm = (w * mob).sum(0) / w.sum()
    a = ((mob - cm) * w).T @ (ref - cr)
    u, _, vt = np.linalg.svd(a)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    aligned = (mob - cm) @ rot + cr
    return (aligned, rot) if return_rotation else aligned


def _rigid_basis(coords_bohr, masses_me):
    """质量加权坐标下的平动/转动向量 (正交归一)。"""
    n = len(masses_me)
    sm = np.sqrt(masses_me)
    com = (masses_me[:, None] * coords_bohr).sum(0) / masses_me.sum()
    r = coords_bohr - com
    vecs = []
    for k in range(3):
        v = np.zeros((n, 3))
        v[:, k] = sm
        vecs.append(v.ravel())
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        vecs.append((np.cross(e, r) * sm[:, None]).ravel())
    u, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    return u[:, s > 1e-6 * s.max()]


def normal_modes(hessian, masses_amu, coords_ang):
    """返回 (频率 cm^-1, 简正模式 L(列, 质量加权坐标), ω² a.u., 刚体向量 Q)。"""
    m = np.asarray(masses_amu) * AMU2ME
    sm = np.repeat(np.sqrt(m), 3)
    hm = hessian / np.outer(sm, sm)
    q = _rigid_basis(np.asarray(coords_ang) / BOHR2ANG, m)
    p = np.eye(len(sm)) - q @ q.T
    w2, vec = np.linalg.eigh(p @ hm @ p)
    rigid_overlap = np.linalg.norm(q.T @ vec, axis=0)
    keep = rigid_overlap < 0.5
    keep_idx = np.where(keep)[0]
    # 保证去掉的数目等于刚体自由度
    if len(keep_idx) != len(sm) - q.shape[1]:
        order = np.argsort(rigid_overlap)
        keep_idx = np.sort(order[:len(sm) - q.shape[1]])
    w2, vec = w2[keep_idx], vec[:, keep_idx]
    freq = np.sign(w2) * np.sqrt(np.abs(w2)) * AU2CM
    return freq, vec, w2, q


# ----------------------------------------------------------------------------------------------
# 内坐标
# ----------------------------------------------------------------------------------------------
def find_bonds(symbols, coords_ang, scale=1.2):
    x = np.asarray(coords_ang)
    rad = np.array([COVALENT_RADII.get(s, 1.5) for s in symbols])
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    lim = scale * (rad[:, None] + rad[None])
    i, j = np.where(np.triu((d < lim) & (d > 0.1), 1))
    return list(zip(i.tolist(), j.tolist()))


def build_internal_coords(symbols, coords_ang, scale=1.2, linear_cut=175.0):
    bonds = find_bonds(symbols, coords_ang, scale)
    nb = [set() for _ in symbols]
    for i, j in bonds:
        nb[i].add(j)
        nb[j].add(i)
    x = np.asarray(coords_ang)

    def ang(i, j, k):
        u, v = x[i] - x[j], x[k] - x[j]
        return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))

    angles = []
    for j in range(len(symbols)):
        nbs = sorted(nb[j])
        for a in range(len(nbs)):
            for b in range(a + 1, len(nbs)):
                if ang(nbs[a], j, nbs[b]) < linear_cut:
                    angles.append((nbs[a], j, nbs[b]))
    dihedrals = []
    for j, k in bonds:
        for i in sorted(nb[j] - {k}):
            for l in sorted(nb[k] - {j}):
                if i != l and ang(i, j, k) < linear_cut and ang(j, k, l) < linear_cut:
                    dihedrals.append((i, j, k, l))
    return {"bond": bonds, "angle": angles, "dihedral": dihedrals}


def _q_values(x, ic):
    """x: (N,3)。返回所有内坐标值 (键长与 x 同单位, 角度为弧度)。"""
    out = []
    if ic["bond"]:
        b = np.array(ic["bond"])
        out.append(np.linalg.norm(x[b[:, 0]] - x[b[:, 1]], axis=1))
    if ic["angle"]:
        a = np.array(ic["angle"])
        u, v = x[a[:, 0]] - x[a[:, 1]], x[a[:, 2]] - x[a[:, 1]]
        c = np.einsum("ij,ij->i", u, v) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
        out.append(np.arccos(np.clip(c, -1, 1)))
    if ic["dihedral"]:
        d = np.array(ic["dihedral"])
        b1, b2, b3 = x[d[:, 1]] - x[d[:, 0]], x[d[:, 2]] - x[d[:, 1]], x[d[:, 3]] - x[d[:, 2]]
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2, axis=1)[:, None])
        out.append(np.arctan2(np.einsum("ij,ij->i", m1, n2), np.einsum("ij,ij->i", n1, n2)))
    return np.concatenate(out) if out else np.zeros(0)


def _wrap(dq, n_bond):
    dq = dq.copy()
    dq[n_bond:] = (dq[n_bond:] + np.pi) % (2 * np.pi) - np.pi
    return dq


def wilson_b(x_bohr, ic, h=1e-5):
    n_bond = len(ic["bond"])
    flat = x_bohr.ravel()
    cols = []
    for k in range(flat.size):
        xp, xm = flat.copy(), flat.copy()
        xp[k] += h
        xm[k] -= h
        cols.append(_wrap(_q_values(xp.reshape(-1, 3), ic) - _q_values(xm.reshape(-1, 3), ic), n_bond) / (2 * h))
    return np.array(cols).T


# ----------------------------------------------------------------------------------------------
# 单个态的分解
# ----------------------------------------------------------------------------------------------
def decompose_state(state, other_coords_ang, bond_scale=1.2):
    """state: read_fchk_hessian 的结果 (该态的极小点与 Hessian)；other_coords_ang: 另一态的几何。"""
    sym, x0, mass, hess = state["symbols"], state["coords"], state["masses"], state["hessian"]
    other = kabsch_align(x0, np.asarray(other_coords_ang), mass)
    dx = (other - x0) / BOHR2ANG                     # bohr
    rmsd = float(np.sqrt(((other - x0) ** 2).sum(1).mean()))

    # 1) 简正模式 / 黄昆因子
    freq, modes, w2, _ = normal_modes(hess, mass, x0)
    sm = np.repeat(np.sqrt(mass * AMU2ME), 3)
    dq_nm = modes.T @ (sm * dx.ravel())
    lam_nm = 0.5 * w2 * dq_nm ** 2                   # Hartree
    omega = np.sqrt(np.abs(w2))
    huang_rhys = np.where(w2 > 0, lam_nm / np.where(omega > 0, omega, 1), 0.0)
    mode_rows = [{"mode": i + 1, "freq_cm-1": float(f), "dQ_amu^0.5_bohr": float(q / np.sqrt(AMU2ME)),
                  "S": float(s), "lambda_meV": float(l * HARTREE2EV * 1000)}
                 for i, (f, q, s, l) in enumerate(zip(freq, dq_nm, huang_rhys, lam_nm))]

    # 2) 内坐标
    ic = build_internal_coords(sym, x0, bond_scale)
    x0b = x0 / BOHR2ANG
    bmat = wilson_b(x0b, ic)
    n_bond = len(ic["bond"])
    bplus = bmat.T @ np.linalg.pinv(bmat @ bmat.T, rcond=1e-8)
    fq = bplus.T @ hess @ bplus
    dq_lin = bmat @ dx.ravel()
    contrib = 0.5 * dq_lin * (fq @ dq_lin)
    dq_exact = _wrap(_q_values(x0b + dx, ic) - _q_values(x0b, ic), n_bond)
    labels = []
    for kind in ("bond", "angle", "dihedral"):
        for atoms in ic[kind]:
            lab = "-".join(sym[a] for a in atoms)
            rev = "-".join(sym[a] for a in reversed(atoms))
            labels.append((kind, atoms, min(lab, rev)))
    ic_rows = []
    for (kind, atoms, lab), dqe, c in zip(labels, dq_exact, contrib):
        delta = dqe * BOHR2ANG if kind == "bond" else np.degrees(dqe)
        ic_rows.append({"type": kind, "atoms": "-".join(str(a + 1) for a in atoms), "elements": lab,
                        "delta": float(delta), "lambda_meV": float(c * HARTREE2EV * 1000)})
    by_type = {k: sum(r["lambda_meV"] for r in ic_rows if r["type"] == k) for k in ("bond", "angle", "dihedral")}
    by_label = {}
    for r in ic_rows:
        key = f"{r['type']}:{r['elements']}"
        by_label[key] = by_label.get(key, 0.0) + r["lambda_meV"]

    # 3) 原子分解（只取内部运动部分）
    proj = bmat.T @ np.linalg.pinv(bmat @ bmat.T, rcond=1e-8) @ bmat
    dxi = proj @ dx.ravel()
    atom_c = 0.5 * (dxi * (hess @ dxi)).reshape(-1, 3).sum(1) * HARTREE2EV * 1000
    atom_rows = [{"atom": i + 1, "element": s, "lambda_meV": float(c)} for i, (s, c) in enumerate(zip(sym, atom_c))]

    return {
        "rmsd_A": rmsd,
        "lambda_normal_modes_meV": float(lam_nm.sum() * HARTREE2EV * 1000),
        "lambda_internal_meV": float(contrib.sum() * HARTREE2EV * 1000),
        "n_imaginary": int((w2 < 0).sum()),
        "modes": mode_rows,
        "internal": ic_rows,
        "internal_by_type_meV": by_type,
        "internal_by_label_meV": dict(sorted(by_label.items(), key=lambda kv: -abs(kv[1]))),
        "atoms": atom_rows,
    }


def decompose_reorg(neutral_fchk, ion_fchk, bond_scale=1.2):
    neu, ion = read_fchk_hessian(neutral_fchk), read_fchk_hessian(ion_fchk)
    if neu["symbols"] != ion["symbols"]:
        raise ValueError("两个 fchk 的原子顺序/种类不一致")
    res = {
        "neutral": decompose_state(neu, ion["coords"], bond_scale),   # λ1
        "ion": decompose_state(ion, neu["coords"], bond_scale),       # λ2
    }
    res["duschinsky"] = duschinsky_matrix(neu, ion)
    res["lambda_total_normal_modes_meV"] = res["neutral"]["lambda_normal_modes_meV"] + \
        res["ion"]["lambda_normal_modes_meV"]
    res["lambda_total_internal_meV"] = res["neutral"]["lambda_internal_meV"] + res["ion"]["lambda_internal_meV"]
    if neu.get("energy") is not None and ion.get("energy") is not None:
        res["adiabatic_energy_difference_eV"] = (ion["energy"] - neu["energy"]) * HARTREE2EV
    return res


def duschinsky_matrix(neu, ion):
    """Duschinsky 矩阵 J_ij = L_ion,iᵀ L_neutral,j（行: 离子态模式, 列: 中性态模式）。
    离子几何与其简正模式一起转到中性分子的取向下再比较。"""
    f_n, l_n, _, _ = normal_modes(neu["hessian"], neu["masses"], neu["coords"])
    _, rot = kabsch_align(neu["coords"], ion["coords"], neu["masses"], return_rotation=True)
    x_ion = kabsch_align(neu["coords"], ion["coords"], neu["masses"])
    f_i, l_i, _, _ = normal_modes(ion["hessian"], ion["masses"], ion["coords"])
    nat = len(neu["symbols"])
    l_i_rot = (l_i.T.reshape(-1, nat, 3) @ rot).reshape(l_i.shape[1], -1).T
    # 刚体部分在两个几何下略有不同，按中性几何重新投影掉
    _, _, _, q = normal_modes(neu["hessian"], neu["masses"], neu["coords"])
    l_i_rot = l_i_rot - q @ (q.T @ l_i_rot)
    l_i_rot /= np.linalg.norm(l_i_rot, axis=0)
    jmat = l_i_rot.T @ l_n
    sm = np.repeat(np.sqrt(neu["masses"] * AMU2ME), 3)
    shift = l_n.T @ (sm * ((x_ion - neu["coords"]) / BOHR2ANG).ravel()) / np.sqrt(AMU2ME)
    return {"matrix": jmat, "row_state": "ion", "col_state": "neutral", "row_freq": f_i, "col_freq": f_n,
            "shift_neutral_modes": shift}


def broadened_spectrum(modes, fwhm=20.0, fmin=0.0, fmax=None, npts=2000):
    """把各模式的 λ_i 用 Lorentz 函数展宽，便于作图 (单位 meV/cm^-1)。"""
    f = np.array([m["freq_cm-1"] for m in modes])
    lam = np.array([m["lambda_meV"] for m in modes])
    fmax = fmax or f.max() * 1.05 + 50
    grid = np.linspace(fmin, fmax, npts)
    g = fwhm / 2
    spec = (lam[None, :] * (g / np.pi) / ((grid[:, None] - f[None, :]) ** 2 + g ** 2)).sum(1)
    return grid, spec


def export_decomposition_csv(res, outdir, prefix="reorg_decomp"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    for state in ("neutral", "ion"):
        for part in ("modes", "internal", "atoms"):
            rows = res[state][part]
            p = outdir / f"{prefix}_{state}_{part}.csv"
            with open(p, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            files.append(p)
    return files
