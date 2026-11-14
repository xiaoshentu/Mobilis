"""转移积分深入分析 (按公开方法自行实现)：

1. 原子自动匹配 (思路参考 CATNIP / QC_Tools, Brown et al.)：单体与二聚体的原子顺序可以不同，
   按坐标找到对应原子，把单体分子轨道系数放到二聚体基组的相应位置。
2. 多能级转移积分矩阵 (思路参考 kndo/ti；公式见 Valeev et al., JACS 2006, 128, 9882)：
   对 A 的轨道 i、B 的轨道 j：J_ij = <i|F|j>，S_ij = <i|j>，e_i = <i|F|i>，
   J_eff = (J − ½(e_i + e_j)S)/(1 − S²)，
   e_i,eff = ½[(e_i + e_j) − 2JS + (e_i − e_j)√(1 − S²)]/(1 − S²)  (e_j,eff 取负号)。
3. 原子对转移积分分析 APTA (方法见 Ozawa, Okada, Matsui, Sci. Technol. Adv. Mater. 2024, 25, 2354652；tcal)：
   u_αβ = Σ_{i∈α} Σ_{j∈β} a_i b_j [F_ij − ½(ε_A + ε_B) S_ij] / (1 − S_AB²)，Σ_αβ u_αβ = J_eff。
数据来源：Gaussian 的 fchk (二聚体 S = (C Cᵀ)⁻¹，F = S C ε Cᵀ S) 或软件内 EHT。
"""
import re
from pathlib import Path

import numpy as np

from .gaussian_io import BOHR2ANG, HARTREE2EV, read_fchk

_NFUNC = {0: 1, 1: 3, -1: 4, 2: 6, -2: 5, 3: 10, -3: 7, 4: 15, -4: 9}


def orbital_labels(nocc, nmo, n_levels):
    """H-n … H, L … L+n 的 (标签, 序号)。"""
    out = []
    for k in range(n_levels - 1, -1, -1):
        if nocc - 1 - k >= 0:
            out.append(("HOMO" if k == 0 else f"HOMO-{k}", nocc - 1 - k))
    for k in range(n_levels):
        if nocc + k < nmo:
            out.append(("LUMO" if k == 0 else f"LUMO+{k}", nocc + k))
    return out


def _symbols(Z, n):
    from .substitution import ELEMENTS
    return [ELEMENTS[int(z) - 1] for z in Z] if Z is not None else ["X"] * n


# ----------------------------------------------------------------------------------------------
# Gaussian fchk
# ----------------------------------------------------------------------------------------------
def load_fchk_full(path):
    d = read_fchk(path)
    nbf = d["Number of basis functions"]
    nmo = d.get("Number of independent functions", nbf)
    if nmo != nbf:
        raise ValueError(f"{Path(path).name}: 有基函数被删除 (独立函数 {nmo} ≠ 基函数 {nbf})，请加 iop(3/32=2) 重新计算")
    types = d.get("Shell types")
    smap = d.get("Shell to atom map")
    if types is None or smap is None:
        raise ValueError(f"{Path(path).name} 缺少 Shell types / Shell to atom map")
    atom_of_bf = []
    for t, a in zip(types, smap):
        atom_of_bf += [int(a) - 1] * _NFUNC[int(t)]
    if len(atom_of_bf) != nbf:
        raise ValueError(f"{Path(path).name}: 壳层与基函数数目不一致")
    nums = d.get("Atomic numbers")
    return {"C": np.array(d["Alpha MO coefficients"]).reshape(nmo, nbf).T, "e": np.array(d["Alpha Orbital Energies"]),
            "nocc": int(d["Number of alpha electrons"]), "nbf": nbf, "atom_of_bf": np.array(atom_of_bf),
            "coords": np.array(d["Current cartesian coordinates"]).reshape(-1, 3) * BOHR2ANG,
            "Z": None if nums is None else np.array(nums, int)}


def match_atoms(mono, dimer, tol=0.05):
    """单体每个原子在二聚体中的序号 (按坐标；不要求原子顺序一致，但需 nosymm 保持坐标不变)。"""
    from scipy.spatial import cKDTree
    d, idx = cKDTree(dimer["coords"]).query(mono["coords"])
    if d.max() > tol:
        raise ValueError(f"单体与二聚体坐标对不上 (最大偏差 {d.max():.3f} Å)：请在所有计算中使用 nosymm，"
                         "并确认单体坐标取自二聚体")
    if len(set(idx.tolist())) != len(idx):
        raise ValueError("单体的两个原子匹配到了二聚体的同一个原子")
    if mono["Z"] is not None and dimer["Z"] is not None and np.any(mono["Z"] != dimer["Z"][idx]):
        raise ValueError("按坐标匹配到的原子元素不同")
    return idx


def embed(mono, dimer, atom_map):
    """把单体 MO 系数放入二聚体基组 (返回 nbf_dimer × nmo_mono)。"""
    out = np.zeros((dimer["nbf"], mono["C"].shape[1]))
    for a_mono, a_dim in enumerate(atom_map):
        rows_m = np.where(mono["atom_of_bf"] == a_mono)[0]
        rows_d = np.where(dimer["atom_of_bf"] == a_dim)[0]
        if len(rows_m) != len(rows_d):
            raise ValueError(f"原子 {a_mono + 1} 在单体与二聚体中的基函数数目不同 (基组或 5d/6d 设置不一致?)")
        out[rows_d] = mono["C"][rows_m]
    return out


def from_fchk(fchk_a, fchk_b, fchk_ab):
    """读取 Gaussian 结果，返回统一的分析数据 (单位 Hartree)。"""
    A, B, D = load_fchk_full(fchk_a), load_fchk_full(fchk_b), load_fchk_full(fchk_ab)
    ma, mb = match_atoms(A, D), match_atoms(B, D)
    if len(set(ma.tolist()) & set(mb.tolist())):
        raise ValueError("单体 A 与 B 匹配到了二聚体中相同的原子：两个单体是否重合?")
    reordered = not (np.array_equal(ma, np.arange(len(ma))) and np.array_equal(mb, np.arange(len(ma), len(ma) + len(mb))))
    S = np.linalg.inv(D["C"] @ D["C"].T)
    F = S @ (D["C"] * D["e"]) @ D["C"].T @ S
    return {"cA": embed(A, D, ma), "cB": embed(B, D, mb), "F": F, "S": S, "noccA": A["nocc"], "noccB": B["nocc"],
            "atom_of_bf": D["atom_of_bf"], "atomsA": ma, "atomsB": mb, "reordered": reordered, "units": "Hartree",
            "source": "Gaussian fchk", "coords": D["coords"], "symbols": _symbols(D["Z"], len(D["coords"]))}


# ----------------------------------------------------------------------------------------------
# EHT (软件内)
# ----------------------------------------------------------------------------------------------
def from_eht(atoms_a, atoms_b):
    from .eht import _coupling_block, monomer
    sa, xa = [a[0] for a in atoms_a], [a[1:] for a in atoms_a]
    sb, xb = [a[0] for a in atoms_b], [a[1:] for a in atoms_b]
    A, B = monomer(sa, xa), monomer(sb, xb)
    sab, hab = _coupling_block(A, B)
    na, nb = A["nbf"], B["nbf"]
    atom_bf = np.concatenate([[sh["atom"]] * (1, 3, 5)[sh["l"]] for sh in A["shells"]] +
                             [[len(sa) + sh["atom"]] * (1, 3, 5)[sh["l"]] for sh in B["shells"]])
    # 只需交叉块 (u_αβ) 与单体能量；构造分块矩阵以复用同一套公式
    F = np.zeros((na + nb, na + nb))
    S = np.zeros_like(F)
    F[:na, na:], F[na:, :na] = hab, hab.T
    S[:na, na:], S[na:, :na] = sab, sab.T
    cA = np.zeros((na + nb, A["C"].shape[1]))
    cB = np.zeros((na + nb, B["C"].shape[1]))
    cA[:na], cB[na:] = A["C"], B["C"]
    return {"cA": cA, "cB": cB, "F": F, "S": S, "noccA": A["nocc"], "noccB": B["nocc"], "atom_of_bf": atom_bf,
            "atomsA": np.arange(len(sa)), "atomsB": np.arange(len(sa), len(sa) + len(sb)), "reordered": False,
            "eA_diag": A["e"], "eB_diag": B["e"], "units": "eV", "source": "EHT",
            "coords": np.array(xa + xb, float), "symbols": sa + sb}


# ----------------------------------------------------------------------------------------------
# 多能级矩阵与原子对分析
# ----------------------------------------------------------------------------------------------
def _site(data, which, idx):
    if f"e{which}_diag" in data:                      # EHT：单体本征值 (与 <i|F|i> 相同)
        return float(data[f"e{which}_diag"][idx])
    c = data[f"c{which}"][:, idx]
    return float(c @ data["F"] @ c)


def _to_ev(data, x):
    return x * HARTREE2EV if data["units"] == "Hartree" else x


def degenerate_levels(data, n_levels=2, tol_ev=0.01):
    """返回所选能级中近简并 (能量差 < tol_ev) 的轨道标签组。简并轨道可任意线性组合，单对耦合数值不唯一。"""
    out = []
    for which, nocc in (("A", data["noccA"]), ("B", data["noccB"])):
        labs = orbital_labels(nocc, data[f"c{which}"].shape[1], n_levels + 1)
        es = [(lab, _to_ev(data, _site(data, which, i))) for lab, i in labs]
        for (l1, e1), (l2, e2) in zip(es, es[1:]):
            if abs(e1 - e2) < tol_ev:
                out.append((which, l1, l2))
    return out


def degenerate_coupling(data, carrier="hole", tol_ev=0.01, window=6):
    """VOTCA 的近简并能级耦合：取与 HOMO (空穴) 或 LUMO (电子) 能量差小于 tol_ev 的轨道组 L_A、L_B，
    J_eff = √(Σ_{i∈L_A, j∈L_B} J_eff,ij² / (N_A N_B))，J_eff,ij 为两能级的 Löwdin 正交化耦合。返回 (meV, N_A, N_B)。"""
    out = []
    for which, nocc in (("A", data["noccA"]), ("B", data["noccB"])):
        k = nocc - 1 if carrier == "hole" else nocc
        nmo = data[f"c{which}"].shape[1]
        cand = range(max(0, k - window), min(nmo, k + window + 1))
        ek = _to_ev(data, _site(data, which, k))
        out.append([i for i in cand if abs(_to_ev(data, _site(data, which, i)) - ek) < tol_ev])
    LA, LB = out
    acc = 0.0
    for i in LA:
        a = data["cA"][:, i]
        ei = _site(data, "A", i)
        for j in LB:
            b = data["cB"][:, j]
            ej = _site(data, "B", j)
            jj, ss = float(a @ data["F"] @ b), float(a @ data["S"] @ b)
            acc += ((jj - 0.5 * (ei + ej) * ss) / (1 - ss * ss)) ** 2
    return _to_ev(data, np.sqrt(acc / (len(LA) * len(LB)))) * 1000, len(LA), len(LB)


def level_matrix(data, n_levels=2):
    """多能级转移积分矩阵：[(A 标签, B 标签, S, e_i, e_j, J, J_eff, e_i,eff, e_j,eff)]，能量单位 eV / meV。"""
    la = orbital_labels(data["noccA"], data["cA"].shape[1], n_levels)
    lb = orbital_labels(data["noccB"], data["cB"].shape[1], n_levels)
    rows = []
    for na_, ia in la:
        a = data["cA"][:, ia]
        ei = _site(data, "A", ia)
        for nb_, ib in lb:
            b = data["cB"][:, ib]
            ej = _site(data, "B", ib)
            j = float(a @ data["F"] @ b)
            s = float(a @ data["S"] @ b)
            jeff = (j - 0.5 * (ei + ej) * s) / (1 - s * s)
            r = np.sqrt(1 - s * s)
            ei_eff = 0.5 * ((ei + ej) - 2 * j * s + (ei - ej) * r) / (1 - s * s)
            ej_eff = 0.5 * ((ei + ej) - 2 * j * s - (ei - ej) * r) / (1 - s * s)
            rows.append({"A": na_, "B": nb_, "S": s, "e_i_eV": _to_ev(data, ei), "e_j_eV": _to_ev(data, ej),
                         "J_meV": _to_ev(data, j) * 1000, "J_eff_meV": _to_ev(data, jeff) * 1000,
                         "e_i_eff_eV": _to_ev(data, ei_eff), "e_j_eff_eV": _to_ev(data, ej_eff)})
    return rows


def _index(label, nocc):
    m = re.fullmatch(r"(HOMO|LUMO)([+-]\d+)?", label.upper())
    if not m:
        raise ValueError(f"无法识别的轨道: {label}")
    return (nocc - 1 if m.group(1) == "HOMO" else nocc) + int(m.group(2) or 0)


def apta(data, orb_a="HOMO", orb_b=None):
    """原子对转移积分 u_αβ (meV)。返回矩阵 (A 原子 × B 原子)、各原子贡献与总和 (= J_eff)。"""
    orb_b = orb_b or orb_a
    ia, ib = _index(orb_a, data["noccA"]), _index(orb_b, data["noccB"])
    a, b = data["cA"][:, ia], data["cB"][:, ib]
    ea, eb = _site(data, "A", ia), _site(data, "B", ib)
    s = float(a @ data["S"] @ b)
    M = np.outer(a, b) * (data["F"] - 0.5 * (ea + eb) * data["S"]) / (1 - s * s)
    atom = data["atom_of_bf"]
    A_at, B_at = list(data["atomsA"]), list(data["atomsB"])
    pa = {at: k for k, at in enumerate(A_at)}
    pb = {at: k for k, at in enumerate(B_at)}
    rows = np.array([pa.get(int(x), -1) for x in atom])
    cols = np.array([pb.get(int(x), -1) for x in atom])
    U = np.zeros((len(A_at), len(B_at)))
    ri, ci = np.nonzero(M)
    keep = (rows[ri] >= 0) & (cols[ci] >= 0)
    np.add.at(U, (rows[ri][keep], cols[ci][keep]), M[ri[keep], ci[keep]])
    U = _to_ev(data, U) * 1000
    total = float(U.sum())
    jeff = _to_ev(data, (float(a @ data["F"] @ b) - 0.5 * (ea + eb) * s) / (1 - s * s)) * 1000
    return {"U": U, "per_atom_A": U.sum(1), "per_atom_B": U.sum(0), "total_meV": total, "J_eff_meV": jeff,
            "orbitals": (orb_a, orb_b), "positive_meV": float(U[U > 0].sum()), "negative_meV": float(U[U < 0].sum())}


def top_pairs(res, symbols_a, symbols_b, n=10):
    U = res["U"]
    order = np.argsort(-np.abs(U), axis=None)[:n]
    out = []
    for k in order:
        i, j = np.unravel_index(k, U.shape)
        out.append((f"{symbols_a[i]}{i + 1}", f"{symbols_b[j]}{j + 1}", float(U[i, j])))
    return out


def write_apta_csv(path, res, symbols_a, symbols_b):
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["atom_A", "atom_B", "u_meV"])
        for i in range(res["U"].shape[0]):
            for j in range(res["U"].shape[1]):
                w.writerow([f"{symbols_a[i]}{i + 1}", f"{symbols_b[j]}{j + 1}", f"{res['U'][i, j]:.6f}"])
        w.writerow([])
        w.writerow(["sum", "", f"{res['total_meV']:.6f}"])
