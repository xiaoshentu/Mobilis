"""分子轨道的片段组成 (FMO 组成)，由 fchk 计算。

  SCPA:     θ_μ,i = C_μi² / Σ_ν C_νi²
  Mulliken: θ_μ,i = C_μi (S C)_μi ，S = (C Cᵀ)⁻¹ （要求未删除基函数）
片段可用原子编号范围定义（如 "1-12,15"），也可自动划分：金属原子单独成片段，其余按成键拆分为配体。
"""
import re

import numpy as np

from .gaussian_io import ELEMENTS, HARTREE2EV, BOHR2ANG, read_fchk

METALS = {"Ir", "Pt", "Os", "Ru", "Re", "Rh", "Pd", "Au", "Cu", "Zn", "Fe", "Co", "Ni", "Ag", "Cd", "Eu", "Tb"}


def _nfunc(t):
    return 4 if t == -1 else (2 * abs(t) + 1 if t < 0 else (t + 1) * (t + 2) // 2)


def parse_ranges(text, natoms):
    idx = set()
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            idx.update(range(int(a), int(b) + 1))
        else:
            idx.add(int(part))
    bad = [i for i in idx if not 1 <= i <= natoms]
    if bad:
        raise ValueError(f"原子编号超出范围: {bad[:5]}")
    return sorted(i - 1 for i in idx)


def auto_fragments(symbols, coords):
    """金属单独一个片段；去掉金属后按成键拆分剩余部分。"""
    from .crystal import _bonded_components
    metal = [i for i, s in enumerate(symbols) if s in METALS]
    rest = [i for i in range(len(symbols)) if i not in metal]
    frags = {}
    if metal:
        frags["Metal(" + ",".join(symbols[i] for i in metal) + ")"] = metal
    if rest:
        lab = _bonded_components([symbols[i] for i in rest], np.asarray(coords)[rest])
        for k, l in enumerate(dict.fromkeys(lab.tolist())):
            frags[f"Lig{k + 1}"] = [rest[j] for j in range(len(rest)) if lab[j] == l]
    return frags


def orbital_composition(fchk, fragments=None, orbitals=("HOMO-1", "HOMO", "LUMO", "LUMO+1"), method="scpa"):
    d = read_fchk(fchk)
    nbf = d["Number of basis functions"]
    nmo = d.get("Number of independent functions", nbf)
    c = np.array(d["Alpha MO coefficients"]).reshape(nmo, nbf).T
    e = np.array(d["Alpha Orbital Energies"])
    nocc = d["Number of alpha electrons"]
    bf_atom = np.repeat(np.array(d["Shell to atom map"]) - 1, [_nfunc(t) for t in d["Shell types"]])
    if len(bf_atom) != nbf:
        raise ValueError("基函数与壳层信息不匹配")
    symbols = [ELEMENTS[z] for z in d["Atomic numbers"]]
    coords = np.array(d["Current cartesian coordinates"]).reshape(-1, 3) * BOHR2ANG
    natom = len(symbols)
    if not fragments:
        fragments = auto_fragments(symbols, coords)
    if method == "mulliken":
        if nmo != nbf:
            raise ValueError("Mulliken 需要未删除基函数的计算 (iop(3/32=2))，可改用 SCPA")
        sc = np.linalg.inv(c.T)          # S C = (Cᵀ)⁻¹
        weight = c * sc
    else:
        weight = c ** 2 / (c ** 2).sum(0)
    atom_w = np.zeros((natom, nmo))
    np.add.at(atom_w, bf_atom, weight)
    rows = []
    for lab in orbitals:
        m = re.fullmatch(r"(HOMO|LUMO)([+-]\d+)?", lab.upper())
        k = (nocc - 1 if m.group(1) == "HOMO" else nocc) + int(m.group(2) or 0)
        comp = {name: float(atom_w[idx, k].sum() * 100) for name, idx in fragments.items()}
        rows.append({"orbital": lab.upper(), "index": k + 1, "E_eV": float(e[k] * HARTREE2EV), "composition": comp})
    return {"fragments": {k: [i + 1 for i in v] for k, v in fragments.items()}, "rows": rows,
            "method": method, "symbols": symbols}


def format_composition(res):
    names = list(res["fragments"])
    lines = [f"{'轨道':<9}{'E/eV':>9}" + "".join(f"{n[:12]:>14}" for n in names)]
    for r in res["rows"]:
        lines.append(f"{r['orbital']:<9}{r['E_eV']:9.3f}" + "".join(f"{r['composition'][n]:13.2f}%" for n in names))
    lines.append("")
    for n, idx in res["fragments"].items():
        s = ",".join(str(i) for i in idx)
        lines.append(f"{n}: {len(idx)} 个原子 ({s[:80]}{'…' if len(s) > 80 else ''})")
    lines.append(f"方法: {res['method'].upper()}")
    return "\n".join(lines)
