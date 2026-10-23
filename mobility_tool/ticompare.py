"""转移积分三种方法对比：软件内 EHT、Gaussian DFT (DIPRO)、ADF 片段法。

单个二聚体或 crystal 子命令导出的二聚体目录 (dimers.json + G<n>.xyz [+ G<n>_A/B/AB.fchk] [+ G<n>_full 结果])。
"""
import json
from pathlib import Path

import numpy as np

from .eht import eht_transfer_integrals
from .gaussian_io import read_geometry

METHODS = ("EHT", "Gaussian", "ADF")


def _gaussian(prefix):
    from .transfer_integral import calc_transfer_integrals
    for ext in (".fchk", ".fch"):
        fa = Path(f"{prefix}_A{ext}")
        if fa.exists():
            r = calc_transfer_integrals(fa, Path(f"{prefix}_B{ext}"), Path(f"{prefix}_AB{ext}"))["results"]
            return {"hole": r["HOMO"]["J_eff_meV"], "electron": r["LUMO"]["J_eff_meV"]}
    return None


def _adf(stem, ams_info=None):
    from .adf import read_adf_transfer_integrals
    last = None
    for cand in (Path(f"{stem}_full.txt"), Path(f"{stem}_full.json"),
                                     Path(f"{stem}_full.results/adf.rkf"), Path(f"{stem}_full.t21"), Path(f"{stem}.out")):
        if not (cand.exists() and cand.stat().st_size > 0):
            continue
        try:
            r = read_adf_transfer_integrals(cand, ams_info)
        except Exception as exc:  # noqa: BLE001
            last = exc
            continue
        if r.get("hole", {}).get("V_meV") is not None or r.get("electron", {}).get("V_meV") is not None:
            return {"hole": r.get("hole", {}).get("V_meV"), "electron": r.get("electron", {}).get("V_meV")}
    if last:
        raise last
    return None


def compare_dimer(dimer, n_a, gaussian_prefix=None, adf_stem=None, label="dimer", charge=0, xtb_exe=None,
                  xtb_dir=None, ams_info=None):
    atoms = read_geometry(dimer)
    e = eht_transfer_integrals(atoms[:n_a], atoms[n_a:], charge, charge)["results"]
    row = {"label": label, "r_centroid": float(np.linalg.norm(
        np.mean([a[1:] for a in atoms[n_a:]], 0) - np.mean([a[1:] for a in atoms[:n_a]], 0))),
        "EHT": {"hole": e["HOMO"]["J_eff_meV"], "electron": e["LUMO"]["J_eff_meV"]},
        "EHT_ESD": {"hole": e["ESD_hole_meV"], "electron": e["ESD_electron_meV"]}, "errors": []}
    if xtb_exe:
        from .programs import xtb_dipro
        try:
            r = xtb_dipro(atoms[:n_a], atoms[n_a:], xtb_exe, xtb_dir or (Path(dimer).parent / f"xtb_{label}"),
                          charge=2 * charge)
            row["xtb"] = {"hole": r["hole"], "electron": r["electron"]}
        except Exception as exc:  # noqa: BLE001
            row["errors"].append(f"xtb: {exc}")
    for meth, fn, arg in (("Gaussian", _gaussian, gaussian_prefix), ("ADF", _adf, adf_stem)):
        if arg:
            try:
                row[meth] = fn(arg) if meth == "Gaussian" else fn(arg, ams_info)
            except Exception as exc:  # noqa: BLE001
                row["errors"].append(f"{meth}: {exc}")
    return row


def compare_dir(d, xtb_exe=None, ams_info=None):
    d = Path(d)
    info = json.loads((d / "dimers.json").read_text())
    rows = []
    for g in info["groups"]:
        stem = d / f"G{g['id']}"
        row = compare_dimer(d / g["file"], info["n_atoms_A"], str(stem), str(stem), label=f"G{g['id']}",
                            xtb_exe=xtb_exe, ams_info=ams_info)
        row["multiplicity"] = g["multiplicity"]
        rows.append(row)
    return rows


def statistics(rows, ref="Gaussian", test="EHT"):
    """以 ref 为基准评价 test：相关系数、比例因子 (|J_ref| ≈ k |J_test|，过原点最小二乘)、平均相对偏差。"""
    out = {}
    for car in ("hole", "electron"):
        pairs = [(abs(r[ref][car]), abs(r[test][car])) for r in rows
                 if r.get(ref) and r.get(test) and r[ref].get(car) is not None and r[test].get(car) is not None]
        if len(pairs) < 2:
            continue
        a, b = np.array(pairs).T
        k = float((a * b).sum() / (b * b).sum())
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        mre = float(np.mean(np.abs(k * b - a) / np.maximum(a, 1e-6)))
        out[car] = {"n": len(pairs), "pearson_r": r, "scale_k": k, "mean_rel_err_after_scaling": mre,
                    "mean_ratio": float(np.mean(b / np.maximum(a, 1e-9)))}
    return out


def format_rows(rows):
    lines = [f"{'二聚体':<8}{'r/Å':>8}  " + "".join(f"{m + '(h)':>12}{m + '(e)':>12}" for m in METHODS)
             + f"{'ESD(h)':>10}{'ESD(e)':>10}"]
    for r in rows:
        cells = ""
        for m in METHODS:
            for car in ("hole", "electron"):
                v = (r.get(m) or {}).get(car)
                cells += f"{v:12.2f}" if v is not None else f"{'—':>12}"
        lines.append(f"{r['label']:<8}{r['r_centroid']:8.3f}  {cells}{r['EHT_ESD']['hole']:10.2f}"
                     f"{r['EHT_ESD']['electron']:10.2f}")
    lines.append("单位 meV；EHT 为软件内计算 (投影法)，ESD 为 EHT 二聚体能级分裂的一半；xtb 为 GFN1-xTB DIPRO "
                 "(输出精度 1 meV)；符号不影响 Marcus 速率。")
    return "\n".join(lines)
