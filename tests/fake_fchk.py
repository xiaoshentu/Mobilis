"""用软件内 EHT 的基组与轨道写出格式与 Gaussian 一致的 fchk (用于在没有 Gaussian 的环境中检验读取与分析)。"""
import numpy as np
from scipy.linalg import eigh

from mobility_tool.eht import build_basis, hamiltonian, overlap_matrix
from mobility_tool.gaussian_io import BOHR2ANG, HARTREE2EV
from mobility_tool.substitution import Z

ELECTRONS = {"H": 1, "C": 4, "N": 5, "O": 6, "S": 6, "F": 7}       # EHT 价电子


def _arr(name, kind, vals, per=5, fmt="{:16.8E}"):
    vals = list(vals)
    lines = [f"{name:<40s}   {kind}   N={len(vals):12d}"]
    f = fmt if kind == "R" else "{:12d}"
    p = per if kind == "R" else 6
    for k in range(0, len(vals), p):
        lines.append("".join(f.format(v) for v in vals[k:k + p]))
    return lines


def write_fake_fchk(path, atoms):
    sym = [a[0] for a in atoms]
    X = np.array([a[1:] for a in atoms], float)
    shells, nbf, _ = build_basis(sym, X)
    S = overlap_matrix(shells, nbf)
    H = hamiltonian(shells, S)
    e, C = eigh(H, S)
    nel = sum(ELECTRONS[s] for s in sym)
    lines = ["fake", "SP        RHF                                                         STO-6G",
             f"{'Number of atoms':<40s}   I     {len(sym):12d}",
             f"{'Number of alpha electrons':<40s}   I     {nel // 2:12d}",
             f"{'Number of basis functions':<40s}   I     {nbf:12d}",
             f"{'Number of independent functions':<40s}   I     {nbf:12d}"]
    lines += _arr("Atomic numbers", "I", [Z[s] for s in sym])
    lines += _arr("Current cartesian coordinates", "R", (X / BOHR2ANG).ravel())
    lines += _arr("Shell types", "I", [{0: 0, 1: 1, 2: -2}[sh["l"]] for sh in shells])
    lines += _arr("Shell to atom map", "I", [sh["atom"] + 1 for sh in shells])
    lines += _arr("Alpha Orbital Energies", "R", e / HARTREE2EV)          # fchk 中为 Hartree
    lines += _arr("Alpha MO coefficients", "R", C.T.ravel())
    # 完整的高斯基组 (STO-6G 收缩，系数对应归一化原函数) 与密度矩阵，供波函数分析使用
    lines += _arr("Number of primitives per shell", "I", [len(sh["a"]) for sh in shells])
    lines += _arr("Primitive exponents", "R", np.concatenate([sh["a"] for sh in shells]))
    lines += _arr("Contraction coefficients", "R", np.concatenate([sh["c"] for sh in shells]))
    lines += _arr("Coordinates of each shell", "R", np.concatenate([np.asarray(sh["center"]) for sh in shells]))
    lines += _arr("Nuclear charges", "R", [float(ELECTRONS[s_]) for s_ in sym])
    nocc = nel // 2
    P = 2 * C[:, :nocc] @ C[:, :nocc].T
    lines += _arr("Total SCF Density", "R", P[np.tril_indices(nbf)])
    open(path, "w").write("\n".join(lines) + "\n")
