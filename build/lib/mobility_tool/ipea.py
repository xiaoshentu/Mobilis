"""电离势 / 电子亲和势 / 前线轨道。

七个能量 (Hartree)，记号 E_态(几何):
  1 E0_N  中性分子在中性几何 (基态优化能量)
  2 Ec_C  阳离子在阳离子几何 (带正电优化能量)
  3 Ea_A  阴离子在阴离子几何 (带负电优化能量)
  4 Ec_N  阳离子在中性几何   (基态优化后带正电能量)
  5 Ea_N  阴离子在中性几何   (基态优化后带负电能量)
  6 E0_C  中性分子在阳离子几何 (带正电优化后不带电能量)
  7 E0_A  中性分子在阴离子几何 (带负电优化后不带电能量)

  IP(v) = Ec_N - E0_N        IP(a) = Ec_C - E0_N       HEP = Ec_C - E0_C
  EA(v) = E0_N - Ea_N        EA(a) = E0_N - Ea_A       EEP = E0_A - Ea_A
  λh = (E0_C - E0_N) + (Ec_N - Ec_C) = IP(v) - HEP
  λe = (E0_A - E0_N) + (Ea_N - Ea_A) = EEP - EA(v)
"""
from pathlib import Path

import numpy as np

from .gaussian_io import HARTREE2EV, read_fchk, read_log_steps

HARTREE2KJ = 2625.4996394799
HARTREE2KCAL = 627.509474

ENERGY_KEYS = [
    ("E0_N", "1 中性@中性几何 (基态优化)"),
    ("Ec_C", "2 阳离子@阳离子几何 (正电优化)"),
    ("Ea_A", "3 阴离子@阴离子几何 (负电优化)"),
    ("Ec_N", "4 阳离子@中性几何"),
    ("Ea_N", "5 阴离子@中性几何"),
    ("E0_C", "6 中性@阳离子几何"),
    ("E0_A", "7 中性@阴离子几何"),
]

QUANTITIES = [  # (键, 说明, 公式所需能量, 计算函数)
    ("IP_v", "垂直电离势 IP(v)", ("Ec_N", "E0_N"), lambda e: e["Ec_N"] - e["E0_N"]),
    ("IP_a", "绝热电离势 IP(a)", ("Ec_C", "E0_N"), lambda e: e["Ec_C"] - e["E0_N"]),
    ("HEP", "空穴抽取势 HEP", ("Ec_C", "E0_C"), lambda e: e["Ec_C"] - e["E0_C"]),
    ("EA_v", "垂直电子亲和势 EA(v)", ("E0_N", "Ea_N"), lambda e: e["E0_N"] - e["Ea_N"]),
    ("EA_a", "绝热电子亲和势 EA(a)", ("E0_N", "Ea_A"), lambda e: e["E0_N"] - e["Ea_A"]),
    ("EEP", "电子抽取势 EEP", ("E0_A", "Ea_A"), lambda e: e["E0_A"] - e["Ea_A"]),
    ("lambda1_h", "λ1 空穴 (中性势能面)", ("E0_C", "E0_N"), lambda e: e["E0_C"] - e["E0_N"]),
    ("lambda2_h", "λ2 空穴 (阳离子势能面)", ("Ec_N", "Ec_C"), lambda e: e["Ec_N"] - e["Ec_C"]),
    ("lambda_h", "空穴重组能 λh", ("E0_C", "E0_N", "Ec_N", "Ec_C"),
     lambda e: e["E0_C"] - e["E0_N"] + e["Ec_N"] - e["Ec_C"]),
    ("lambda1_e", "λ1 电子 (中性势能面)", ("E0_A", "E0_N"), lambda e: e["E0_A"] - e["E0_N"]),
    ("lambda2_e", "λ2 电子 (阴离子势能面)", ("Ea_N", "Ea_A"), lambda e: e["Ea_N"] - e["Ea_A"]),
    ("lambda_e", "电子重组能 λe", ("E0_A", "E0_N", "Ea_N", "Ea_A"),
     lambda e: e["E0_A"] - e["E0_N"] + e["Ea_N"] - e["Ea_A"]),
]


def ip_ea_from_energies(energies):
    """energies: {E0_N: Hartree, ...}，可以只给一部分，能算的都算。返回 {键: {name, eV, kJ/mol, kcal/mol}}。"""
    e = {k: float(v) for k, v in energies.items() if v not in (None, "")}
    out = {}
    for key, name, need, fn in QUANTITIES:
        if all(n in e for n in need):
            v = fn(e)
            out[key] = {"name": name, "eV": v * HARTREE2EV, "kJ/mol": v * HARTREE2KJ, "kcal/mol": v * HARTREE2KCAL}
    return out


def frontier_orbitals(path, step=0):
    """从 fchk 或 log 读取 HOMO/LUMO (eV)。log 默认取第一个计算步骤 (中性优化) 的最后一次布居分析。"""
    path = Path(path)
    if path.suffix.lower() in (".fchk", ".fch"):
        d = read_fchk(path)
        ea = np.array(d["Alpha Orbital Energies"])
        na = d["Number of alpha electrons"]
        occ, virt = list(ea[:na]), list(ea[na:])
        if "Beta Orbital Energies" in d:
            eb = np.array(d["Beta Orbital Energies"])
            nb = d["Number of beta electrons"]
            occ += list(eb[:nb])
            virt += list(eb[nb:])
    else:
        steps = read_log_steps(path)
        orb = steps[step]["orbitals"] if step < len(steps) else None
        if not orb:
            raise ValueError(f"{path} 的第 {step + 1} 步中没有找到轨道能量")
        occ = orb["Alpha occ."] + orb["Beta occ."]
        virt = orb["Alpha virt."] + orb["Beta virt."]
    occ, virt = sorted(occ), sorted(virt)
    homo, lumo = occ[-1] * HARTREE2EV, virt[0] * HARTREE2EV
    res = {"HOMO_eV": homo, "LUMO_eV": lumo, "gap_eV": lumo - homo,
           "Koopmans_IP_eV": -homo, "Koopmans_EA_eV": -lumo}
    if len(occ) > 1:
        res["HOMO-1_eV"] = occ[-2] * HARTREE2EV
    if len(virt) > 1:
        res["LUMO+1_eV"] = virt[1] * HARTREE2EV
    return res


def format_ip_ea(res, orbitals=None):
    lines = [f"{'物理量':<24}{'eV':>12}{'kJ/mol':>12}{'kcal/mol':>12}"]
    for key, *_ in QUANTITIES:
        if key in res:
            r = res[key]
            lines.append(f"{r['name']:<24}{r['eV']:12.4f}{r['kJ/mol']:12.3f}{r['kcal/mol']:12.3f}")
    if orbitals:
        lines.append("")
        lines.append(f"HOMO = {orbitals['HOMO_eV']:.4f} eV   LUMO = {orbitals['LUMO_eV']:.4f} eV   "
                     f"Gap = {orbitals['gap_eV']:.4f} eV")
        lines.append(f"Koopmans 近似: IP ≈ -E(HOMO) = {orbitals['Koopmans_IP_eV']:.4f} eV, "
                     f"EA ≈ -E(LUMO) = {orbitals['Koopmans_EA_eV']:.4f} eV")
    return "\n".join(lines)
