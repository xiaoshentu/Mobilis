"""二聚体投影法 (Dimer Projection, DIPRO) 计算转移积分。

只需三个 fchk 文件（单体 A、单体 B、二聚体 AB）：
  二聚体 MO 系数 C_D 满足 C_D^T S C_D = I，因此 inv(C_D) 给出 AO→MO 的变换，
  单体轨道在二聚体分子轨道上的投影 G^T = inv(C_D) · blockdiag(C_A, C_B)，
  有效 Hamiltonian  H = G ε G^T，重叠  S = G G^T，
  再做 Löwdin 正交化：J_eff = (J - (e_A + e_B) S_AB / 2) / (1 - S_AB^2)。
要求：nosymm（防止坐标被重定向）、iop(3/32=2)（不删除线性相关基函数），
二聚体原子顺序 = A 的原子在前、B 的原子在后，三个计算的方法/基组相同。
"""
import re
from pathlib import Path

import numpy as np

from .gaussian_io import BOHR2ANG, HARTREE2EV, level_and_tail, read_fchk, read_geometry, write_gjf


# ----------------------------------------------------------------------------------------------
# 输入生成
# ----------------------------------------------------------------------------------------------
def generate_ti_inputs(outdir, dimer=None, n_a=None, mono_a=None, mono_b=None, name="dimer",
                       method="B3LYP", basis="6-31G(d)", charge=0, mult=1,
                       nproc=8, mem="8GB", extra="", program="g16", ecp_basis="SDD"):
    if dimer is not None:
        atoms = read_geometry(dimer)
        if not n_a or not 0 < n_a < len(atoms):
            raise ValueError("使用 --dimer 时必须用 --na 指定单体 A 的原子数")
        atoms_a, atoms_b = atoms[:n_a], atoms[n_a:]
    else:
        atoms_a, atoms_b = read_geometry(mono_a), read_geometry(mono_b)
    from .crystal import check_dimer_geometry
    check_dimer_geometry(atoms_a, atoms_b)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    method, *more = method.split()
    extra = " ".join(more + [extra])
    systems = {f"{name}_A": atoms_a, f"{name}_B": atoms_b, f"{name}_AB": atoms_a + atoms_b}
    for stem, atoms in systems.items():
        # 每个元素用的基组在三个计算中相同，因此二聚体基函数 = A 的基函数 + B 的基函数
        lvl, tail = level_and_tail(method, basis, [a[0] for a in atoms], ecp_basis)
        route = f"# {lvl} nosymm iop(3/32=2) scf=tight {extra}"
        write_gjf(outdir / f"{stem}.gjf", [dict(chk=f"{stem}.chk", nproc=nproc, mem=mem, route=route,
                                                title=f"{stem} transfer integral", charge=charge,
                                                mult=mult, atoms=atoms, tail=tail)])
    script = ["#!/bin/bash", "# 运行三个计算并生成 fchk"]
    for stem in systems:
        script += [f"{program} < {stem}.gjf > {stem}.log", f"formchk {stem}.chk {stem}.fchk"]
    (outdir / "run_ti.sh").write_text("\n".join(script) + "\n")
    from .jobs import write_windows_bat
    write_windows_bat(outdir, list(systems), name=f"run_{name}.bat")
    return list(systems)


# ----------------------------------------------------------------------------------------------
# 计算
# ----------------------------------------------------------------------------------------------
def load_fchk_mo(path):
    d = read_fchk(path)
    nbf = d["Number of basis functions"]
    nmo = d.get("Number of independent functions", nbf)
    if nmo != nbf:
        raise ValueError(f"{path}: 独立函数数 {nmo} ≠ 基函数数 {nbf}，基函数被删除了。"
                         "请在关键词中加入 iop(3/32=2) 重新计算。")
    if "Beta MO coefficients" in d:
        print(f"警告: {path} 是开壳层计算，只使用 alpha 轨道。")
    coef = np.array(d["Alpha MO coefficients"]).reshape(nmo, nbf).T   # (AO, MO)
    return {
        "C": coef,
        "e": np.array(d["Alpha Orbital Energies"]),
        "nocc": d["Number of alpha electrons"],
        "nbf": nbf,
        "coords": np.array(d["Current cartesian coordinates"]).reshape(-1, 3) * BOHR2ANG,
    }


def _orbital_index(label, nocc):
    m = re.fullmatch(r"(HOMO|LUMO)([+-]\d+)?", label.upper())
    if not m:
        raise ValueError(f"无法识别的轨道标签: {label}（示例: HOMO, HOMO-1, LUMO, LUMO+1）")
    shift = int(m.group(2) or 0)
    return (nocc - 1 if m.group(1) == "HOMO" else nocc) + shift


def _calc_legacy(fchk_a, fchk_b, fchk_ab, orbitals):
    """原投影法 (要求二聚体中 A 在前、B 在后)，用于缺少壳层信息的 fchk。"""
    A, B, D = load_fchk_mo(fchk_a), load_fchk_mo(fchk_b), load_fchk_mo(fchk_ab)
    if A["nbf"] + B["nbf"] != D["nbf"]:
        raise ValueError(f"基函数数不匹配: A({A['nbf']}) + B({B['nbf']}) ≠ AB({D['nbf']})")
    warnings = []
    ref = np.vstack([A["coords"], B["coords"]])
    if ref.shape != D["coords"].shape or np.abs(ref - D["coords"]).max() > 1e-3:
        raise ValueError("二聚体坐标与 A+B 坐标不一致，且 fchk 缺少壳层信息无法自动匹配原子：请确认原子顺序 (A 在前) 并使用 nosymm")
    na, nb = A["nbf"], B["nbf"]
    c_frag = np.zeros((na + nb, na + nb))
    c_frag[:na, :na] = A["C"]
    c_frag[na:, na:] = B["C"]
    g = np.linalg.solve(D["C"], c_frag).T
    h = (g * D["e"]) @ g.T
    s = g @ g.T
    completeness = np.einsum("ij,ij->i", g, g)
    results = {}
    for label in orbitals:
        ia = _orbital_index(label, A["nocc"])
        ib = na + _orbital_index(label, B["nocc"])
        ea, eb, j, sab = h[ia, ia], h[ib, ib], h[ia, ib], s[ia, ib]
        j_eff = (j - 0.5 * (ea + eb) * sab) / (1 - sab ** 2)
        root = np.sqrt(1 - sab ** 2)
        ea_eff = 0.5 * ((ea + eb) - 2 * j * sab + (ea - eb) * root) / (1 - sab ** 2)
        eb_eff = 0.5 * ((ea + eb) - 2 * j * sab - (ea - eb) * root) / (1 - sab ** 2)
        results[label.upper()] = {"J_eff_meV": j_eff * HARTREE2EV * 1000, "J_raw_meV": j * HARTREE2EV * 1000,
                                  "S_AB": sab, "e_A_eV": ea * HARTREE2EV, "e_B_eV": eb * HARTREE2EV,
                                  "e_A_eff_eV": ea_eff * HARTREE2EV, "e_B_eff_eV": eb_eff * HARTREE2EV,
                                  "projection_completeness": [float(completeness[ia]), float(completeness[ib])]}
    return {"results": results, "warnings": warnings}


def calc_transfer_integrals(fchk_a, fchk_b, fchk_ab, orbitals=("HOMO", "LUMO"), degeneracy=0.0):
    """由 Gaussian fchk 计算转移积分 (投影法)。按坐标自动匹配原子，单体与二聚体的原子顺序可以不同。
    degeneracy > 0 (eV) 时另给 VOTCA 的近简并能级有效耦合 (HOMO / LUMO 的 J_deg_meV 与参与的能级数)。"""
    from .ti_analysis import degenerate_coupling, from_fchk
    try:
        data = from_fchk(fchk_a, fchk_b, fchk_ab)
    except ValueError as exc:
        if "Shell types" in str(exc):
            return _calc_legacy(fchk_a, fchk_b, fchk_ab, orbitals)
        raise
    warnings = ["二聚体原子顺序与单体不同，已按坐标自动匹配原子"] if data["reordered"] else []
    F, S = data["F"], data["S"]
    results = {}
    for label in orbitals:
        ia, ib = _orbital_index(label, data["noccA"]), _orbital_index(label, data["noccB"])
        a, b = data["cA"][:, ia], data["cB"][:, ib]
        ea, eb, j, sab = float(a @ F @ a), float(b @ F @ b), float(a @ F @ b), float(a @ S @ b)
        j_eff = (j - 0.5 * (ea + eb) * sab) / (1 - sab ** 2)
        root = np.sqrt(1 - sab ** 2)
        ea_eff = 0.5 * ((ea + eb) - 2 * j * sab + (ea - eb) * root) / (1 - sab ** 2)
        eb_eff = 0.5 * ((ea + eb) - 2 * j * sab - (ea - eb) * root) / (1 - sab ** 2)
        results[label.upper()] = {
            "J_eff_meV": j_eff * HARTREE2EV * 1000,
            "J_raw_meV": j * HARTREE2EV * 1000,
            "S_AB": sab,
            "e_A_eV": ea * HARTREE2EV, "e_B_eV": eb * HARTREE2EV,
            "e_A_eff_eV": ea_eff * HARTREE2EV, "e_B_eff_eV": eb_eff * HARTREE2EV,
            "projection_completeness": [float(a @ S @ a), float(b @ S @ b)],
        }
    if degeneracy > 0:
        for lab, car in (("HOMO", "hole"), ("LUMO", "electron")):
            if lab in results:
                jd, na_, nb_ = degenerate_coupling(data, car, degeneracy)
                results[lab].update(J_deg_meV=jd, deg_levels=(na_, nb_))
                if na_ > 1 or nb_ > 1:
                    warnings.append(f"{lab} 近简并 (A {na_} 个、B {nb_} 个能级在 {degeneracy * 1000:g} meV 内)："
                                    f"有效耦合 √(ΣJ²/N_A N_B) = {jd:.2f} meV")
    return {"results": results, "warnings": warnings}
