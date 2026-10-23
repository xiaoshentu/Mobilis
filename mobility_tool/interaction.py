"""分子间相互作用能（二聚体结合能）与能量框架 / 晶格能。

三种方法：
  UFF 力场  ：E = Σ D_ij[(x_ij/r)^12 − 2(x_ij/r)^6] (+ 可选 Coulomb，原子电荷取自 Gaussian 输出)，秒级、定性；
  xtb GFN2 ：E_int = E(AB) − E(A) − E(B)，含 D4 色散，半定量；
  Gaussian ：Counterpoise=2 (抵消校正 BSSE)，推荐 B3LYP-D3(BJ) 或 ωB97X-D，用于发表。
能量框架：晶体中每对近邻分子画一根圆柱，半径 ∝ |E_int|（参考 CrystalExplorer 的 energy frameworks）。
晶格能近似：E_latt ≈ ½ Σ_近邻 E_int（只含所选近邻，截断越大越接近完整值）。
"""
import re
import subprocess
from pathlib import Path

import numpy as np

KCAL2KJ = 4.184
HARTREE2KJ = 2625.4996
COULOMB_KJ = 1389.35458        # e²/(4πε0) in kJ/mol·Å

# UFF 非键参数 (Rappé et al., JACS 1992)：x_i / Å, D_i / kcal mol⁻¹
UFF = {"H": (2.886, 0.044), "B": (4.083, 0.180), "C": (3.851, 0.105), "N": (3.660, 0.069), "O": (3.500, 0.060),
       "F": (3.364, 0.050), "Si": (4.295, 0.402), "P": (4.147, 0.305), "S": (4.035, 0.274), "Cl": (3.947, 0.227),
       "Se": (4.205, 0.291), "Br": (4.189, 0.251), "I": (4.500, 0.339), "Cu": (3.495, 0.005), "Zn": (2.763, 0.124),
       "Ru": (2.963, 0.056), "Os": (3.120, 0.037), "Ir": (3.731, 0.073), "Pt": (2.754, 0.080), "Au": (3.293, 0.039)}


def _xyz(atoms):
    return [a[0] for a in atoms], np.array([a[1:] for a in atoms], float)


# ----------------------------------------------------------------------------------------------
# UFF
# ----------------------------------------------------------------------------------------------
def uff_interaction(atoms_a, atoms_b, charges=None):
    """返回 {'vdw', 'elec', 'total'} (kJ/mol)。charges: 单体原子电荷 (A、B 为同一分子时共用)，或 (qa, qb)。"""
    sa, xa = _xyz(atoms_a)
    sb, xb = _xyz(atoms_b)
    miss = sorted({e for e in sa + sb if e not in UFF})
    if miss:
        raise ValueError(f"缺少 UFF 参数的元素: {miss}")
    pa = np.array([UFF[e] for e in sa])
    pb = np.array([UFF[e] for e in sb])
    r = np.linalg.norm(xa[:, None] - xb[None], axis=-1)
    x = np.sqrt(pa[:, 0][:, None] * pb[:, 0][None])
    d = np.sqrt(pa[:, 1][:, None] * pb[:, 1][None]) * KCAL2KJ
    q6 = (x / r) ** 6
    vdw = float((d * (q6 * q6 - 2 * q6)).sum())
    elec = 0.0
    if charges is not None:
        qa, qb = charges if isinstance(charges, tuple) else (charges, charges)
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        if len(qa) != len(sa) or len(qb) != len(sb):
            raise ValueError("原子电荷数目与分子原子数不符")
        elec = float((COULOMB_KJ * qa[:, None] * qb[None] / r).sum())
    return {"vdw": vdw, "elec": elec, "total": vdw + elec, "method": "UFF" + (" + 静电" if charges is not None else "")}


def read_charges(log):
    """从 Gaussian 输出读取原子电荷：优先 ESP 拟合电荷 (pop=mk / chelpg)，否则 Mulliken。"""
    text = Path(log).read_text(errors="ignore")
    for head in (r"ESP charges:", r"Charges from ESP fit", r"Mulliken charges:", r"Mulliken atomic charges:"):
        idx = [m.end() for m in re.finditer(head, text)]
        if not idx:
            continue
        block = text[idx[-1]:].splitlines()
        q = []
        for line in block[2:] if head.startswith("Charges from") else block[1:]:
            p = line.split()
            if len(p) >= 3 and p[0].isdigit():
                q.append(float(p[2]))
            elif q:
                break
        if q:
            return np.array(q), head.split()[0]
    raise ValueError("Gaussian 输出中没有找到 ESP 或 Mulliken 电荷")


# ----------------------------------------------------------------------------------------------
# xtb
# ----------------------------------------------------------------------------------------------
def _xtb_energy(atoms, exe, wd, name, gfn=2, charge=0, timeout=3600):
    from .gaussian_io import write_xyz
    wd = Path(wd) / name
    wd.mkdir(parents=True, exist_ok=True)
    write_xyz(wd / "mol.xyz", atoms, name)
    flags = 0x08000000 if __import__("sys").platform.startswith("win") else 0
    r = subprocess.run([exe, "mol.xyz", "--sp", "--gfn", str(gfn), "--chrg", str(charge)], cwd=wd,
                       capture_output=True, timeout=timeout, creationflags=flags)
    text = r.stdout.decode(errors="ignore")
    (wd / "xtb.out").write_text(text)
    m = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", text)
    if not m:
        raise RuntimeError(f"xtb 没有输出总能量 (见 {wd / 'xtb.out'})")
    return float(m[-1])


def xtb_interaction(atoms_a, atoms_b, exe, workdir, gfn=2, cache=None):
    """GFN-xTB 结合能 (kJ/mol)。cache: {几何键: 能量}，平移等价的单体只算一次。"""
    from .eht import _geom_key
    cache = {} if cache is None else cache

    def mono(atoms, tag):
        s, x = _xyz(atoms)
        key = _geom_key(s, x, 0) + (gfn,)
        if key not in cache:
            cache[key] = _xtb_energy(atoms, exe, workdir, f"{tag}_{len(cache)}", gfn)
        return cache[key]
    ea, eb = mono(atoms_a, "A"), mono(atoms_b, "B")
    eab = _xtb_energy(list(atoms_a) + list(atoms_b), exe, workdir, f"AB_{abs(hash(tuple(map(tuple, atoms_b)))) % 10**8}",
                      gfn)
    return {"total": (eab - ea - eb) * HARTREE2KJ, "E_AB": eab, "E_A": ea, "E_B": eb, "method": f"GFN{gfn}-xTB"}


# ----------------------------------------------------------------------------------------------
# Gaussian counterpoise
# ----------------------------------------------------------------------------------------------
def write_counterpoise_gjf(path, atoms_a, atoms_b, functional="B3LYP", basis="6-31G(d)", dispersion="GD3BJ",
                           nproc=8, mem="8GB", title="dimer counterpoise", extra=""):
    from .crystal import check_dimer_geometry
    check_dimer_geometry(atoms_a, atoms_b)
    disp = f" em={dispersion}" if dispersion and dispersion.lower() not in ("none", "无") else ""
    path = Path(path)
    lines = [f"%nprocshared={nproc}", f"%mem={mem}", f"%chk={path.stem}.chk",
             f"# {functional}/{basis}{disp} counterpoise=2 nosymm {extra}".rstrip(), "", title, "", "0 1 0 1 0 1"]
    for frag, atoms in ((1, atoms_a), (2, atoms_b)):
        for s, x, y, z in atoms:
            lines.append(f" {s}(Fragment={frag}) {x:14.8f} {y:14.8f} {z:14.8f}")
    from .gaussian_io import gaussian_basis_fix
    path.write_text(gaussian_basis_fix("\n".join(lines)) + "\n\n")
    return path


def read_counterpoise(log):
    """读取 Gaussian 抵消校正结果 (kJ/mol)。"""
    text = Path(log).read_text(errors="ignore")
    corr = re.findall(r"complexation energy\s*=\s*(-?\d+\.\d+)\s*kcal/mole\s*\(corrected\)", text)
    raw = re.findall(r"complexation energy\s*=\s*(-?\d+\.\d+)\s*kcal/mole\s*\(raw\)", text)
    bsse = re.findall(r"BSSE energy\s*=\s*(-?\d+\.\d+)", text)
    if not corr:
        raise ValueError(f"{Path(log).name} 中没有 Counterpoise 结果 (计算未完成或不是 counterpoise 任务)")
    return {"total": float(corr[-1]) * KCAL2KJ, "raw": float(raw[-1]) * KCAL2KJ if raw else None,
            "BSSE": float(bsse[-1]) * HARTREE2KJ if bsse else None, "method": "Gaussian CP",
            "normal": "Normal termination" in text[-3000:]}


def prepare_counterpoise_jobs(dimer_result, outdir, **kw):
    """为结构页的全部近邻二聚体组生成 G<n>_cp.gjf 与 Windows 批处理。"""
    from .jobs import write_windows_bat
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    a = [(s, *x) for s, x in dimer_result["atoms_a"]]
    names = []
    for g in dimer_result["groups"]:
        b = [(s, *x) for s, x in g["members"][0]["atoms_b"]]
        nm = f"G{g['id']}_cp"
        write_counterpoise_gjf(outdir / f"{nm}.gjf", a, b, title=f"G{g['id']} counterpoise", **kw)
        names.append(nm)
    (outdir / "run_cp.sh").write_text("#!/bin/bash\n" + "\n".join(f"g16 < {n}.gjf > {n}.log" for n in names) + "\n")
    write_windows_bat(outdir, names, name="run_cp.bat")
    return names


def collect_counterpoise(outdir):
    out, err = {}, {}
    for f in sorted(Path(outdir).glob("G*_cp.log")) + sorted(Path(outdir).glob("G*_cp.out")):
        gid = int(re.search(r"G(\d+)_cp", f.name).group(1))
        try:
            out[gid] = read_counterpoise(f)
        except Exception as exc:  # noqa: BLE001
            err[gid] = str(exc)
    return out, err


# ----------------------------------------------------------------------------------------------
# 晶体: 全部组 / 晶格能 / 能量框架
# ----------------------------------------------------------------------------------------------
def group_energies(dimer_result, method="uff", charges=None, xtb_exe=None, workdir=None, on_progress=None):
    a = [(s, *x) for s, x in dimer_result["atoms_a"]]
    out, cache = {}, {}
    for k, g in enumerate(dimer_result["groups"]):
        b = [(s, *x) for s, x in g["members"][0]["atoms_b"]]
        if method == "uff":
            out[g["id"]] = uff_interaction(a, b, charges)
        else:
            out[g["id"]] = xtb_interaction(a, b, xtb_exe, Path(workdir) / "xtb_int", cache=cache)
        if on_progress:
            on_progress(k + 1, len(dimer_result["groups"]))
    return out


def lattice_energy(dimer_result, energies):
    """E_latt ≈ ½ Σ_近邻 E_int (kJ/mol)；只含 dimer_result 中的近邻。"""
    tot, miss = 0.0, set()
    for d in dimer_result["dimers"]:
        e = energies.get(d["group"])
        if e is None:
            miss.add(d["group"])
            continue
        tot += e["total"] if isinstance(e, dict) else e
    return tot / 2, sorted(miss)


def framework_edges(crys, mols, energies, cutoff=4.0, criterion="atom", ncell=(1, 1, 1)):
    """能量框架的圆柱：[(起点, 终点, E_int)]，覆盖 ncell 个晶胞中的全部分子。"""
    from .kmc import graph_from_crystal
    e_meV = {g: (v["total"] if isinstance(v, dict) else v) * 1000 for g, v in energies.items()}
    graph = graph_from_crystal(crys, mols, {g: {"hole": e, "electron": e} for g, e in e_meV.items()}, cutoff,
                               criterion, "hole", signed=True)
    lat = np.asarray(crys["lattice"])
    cents = [np.asarray(m["cart"]).mean(0) for m in mols]
    edges, seen = [], set()
    for T in np.array([[i, j, k] for i in range(ncell[0]) for j in range(ncell[1]) for k in range(ncell[2])]):
        for (i, j, v, e, mult, t) in graph["edges"]:
            p = cents[i] + T @ lat
            q = p + np.asarray(v)
            key = tuple(np.round(np.sort(np.stack([p, q]), axis=0).ravel(), 2))
            if key in seen:
                continue
            seen.add(key)
            edges.append((p, q, float(e)))
    return edges, cents
