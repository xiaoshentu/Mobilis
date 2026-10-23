"""串联各模块的批处理流程：晶体二聚体 → 批量生成输入 → 批量读取转移积分 → 迁移率。"""
import json
import re
from pathlib import Path

from .adf import generate_adf_batch, read_adf_transfer_integrals
from .crystal import export_dimers
from .transfer_integral import calc_transfer_integrals, generate_ti_inputs


def prepare_dimer_jobs(dimer_result, outdir, program="gaussian", **settings):
    """导出所有等价组的二聚体，并为每组生成 Gaussian 或 ADF 输入（名称 G1, G2, ...）。"""
    outdir = Path(outdir)
    files = export_dimers(dimer_result, outdir)
    n_a = len(dimer_result["atoms_a"])
    if program in ("eht", "xtb"):       # 软件内 / xtb 直接计算，不需要输入文件
        return files
    if program != "gaussian":       # ADF: 一个批量脚本依次计算全部二聚体 (ADFjobs 中可直接运行 batch_adf.run)
        from .gaussian_io import read_geometry
        chunk = settings.pop("chunk", 0)
        entries = []
        for f in files:
            at = read_geometry(f)
            entries.append((f.stem, at[:n_a], at[n_a:]))
        return generate_adf_batch(outdir, entries, chunk=chunk, **settings) and files
    for f in files:
        name = f.stem
        generate_ti_inputs(outdir, dimer=f, n_a=n_a, name=name, **settings)
    if program == "gaussian":   # 合并成一个总脚本
        lines = ["#!/bin/bash"]
        for f in files:
            for s in ("A", "B", "AB"):
                stem = f"{f.stem}_{s}"
                lines += [f"g16 < {stem}.gjf > {stem}.log", f"formchk {stem}.chk {stem}.fchk"]
        (outdir / "run_ti.sh").write_text("\n".join(lines) + "\n")
        from .jobs import write_windows_bat
        write_windows_bat(outdir, [f"{f.stem}_{s}" for f in files for s in ("A", "B", "AB")], name="run_ti.bat")
    return files


def collect_group_couplings(outdir, program="gaussian", xtb_exe=None, ams_info=None):
    """批量读取 G<n> 的转移积分，返回 {组号: {'hole': J_meV, 'electron': J_meV}}。"""
    outdir = Path(outdir)
    out, errors = {}, {}
    if program == "xtb":
        from .gaussian_io import read_geometry
        from .programs import find_xtb, xtb_dipro
        exe = xtb_exe or find_xtb()
        if not exe:
            raise FileNotFoundError("没有找到 xtb，请在 工具 → 程序路径设置 中指定")
        info = json.loads((outdir / "dimers.json").read_text())
        for g in info["groups"]:
            try:
                at = read_geometry(outdir / g["file"])
                r = xtb_dipro(at[:info["n_atoms_A"]], at[info["n_atoms_A"]:], exe, outdir / f"xtb_G{g['id']}")
                out[g["id"]] = {"hole": r["hole"], "electron": r["electron"]}
            except Exception as exc:  # noqa: BLE001
                errors[g["id"]] = str(exc)
        return out, errors
    if program == "eht":
        from .eht import eht_from_file
        info = json.loads((outdir / "dimers.json").read_text())
        for g in info["groups"]:
            try:
                r = eht_from_file(outdir / g["file"], info["n_atoms_A"], esd=False)["results"]
                out[g["id"]] = {"hole": r["HOMO"]["J_eff_meV"], "electron": r["LUMO"]["J_eff_meV"]}
            except Exception as exc:  # noqa: BLE001
                errors[g["id"]] = str(exc)
        return out, errors
    if program == "gaussian":
        for fa in sorted(list(outdir.glob("G*_A.fchk")) + list(outdir.glob("G*_A.fch"))):
            m = re.match(r"G(\d+)_A\.fchk?$", fa.name)
            if not m:
                continue
            gid = int(m.group(1))
            try:
                r = calc_transfer_integrals(fa, fa.with_name(f"G{gid}_B{fa.suffix}"), fa.with_name(f"G{gid}_AB{fa.suffix}"))
                out[gid] = {"hole": r["results"]["HOMO"]["J_eff_meV"],
                            "electron": r["results"]["LUMO"]["J_eff_meV"]}
            except Exception as exc:  # noqa: BLE001
                errors[gid] = str(exc)
    else:
        cands = {}
        for f in list(outdir.glob("G*_full.txt")) + list(outdir.glob("G*.out")) + \
                list(outdir.glob("G*_full.results/adf.rkf")) + list(outdir.glob("G*_full.t21")):   # 文本优先
            nm = f.parent.name if f.name == "adf.rkf" else f.name
            m = re.fullmatch(r"G(\d+)(?:_full)?(?:\.txt|\.out|\.t21|\.results)", nm)
            if m:
                cands.setdefault(int(m.group(1)), []).append(f)
        for gid, fs in sorted(cands.items()):
            last = None
            for f in fs:
                try:
                    if f.suffix == ".txt" and f.stat().st_size == 0:
                        continue
                    r = read_adf_transfer_integrals(f, ams_info)
                    if r.get("hole", {}).get("V_meV") is None and r.get("electron", {}).get("V_meV") is None:
                        continue
                    out[gid] = {"hole": r.get("hole", {}).get("V_meV"), "electron": r.get("electron", {}).get("V_meV")}
                    break
                except Exception as exc:  # noqa: BLE001
                    last = exc
            if gid not in out:
                errors[gid] = str(last)
    return out, errors


def eht_couplings_for_crystal(crys, mols, cutoff=4.0, criterion="atom", ref=0, on_progress=None, dimer_result=None):
    """软件内 (EHT) 计算参考分子所有二聚体组的 J。返回 (二聚体结果, {组号: {hole, electron, ...}})。
    给出 dimer_result (如按方向选择的结果) 时直接使用它，不重新搜索。"""
    from .crystal import find_dimers
    from .eht import eht_transfer_integrals
    res = dimer_result or find_dimers(crys, mols, ref=ref, cutoff=cutoff, criterion=criterion)
    a_atoms = [(s, *x) for s, x in res["atoms_a"]]
    out = {}
    for n, g in enumerate(res["groups"]):
        d = g["members"][0]
        r = eht_transfer_integrals(a_atoms, [(s, *x) for s, x in d["atoms_b"]], esd=False)["results"]
        out[g["id"]] = {"hole": r["HOMO"]["J_eff_meV"], "electron": r["LUMO"]["J_eff_meV"],
                        "r_centroid": d["r_centroid"], "multiplicity": len(g["members"])}
        if on_progress:
            on_progress(n + 1, len(res["groups"]))
    return res, out


def mobility_from_dimers(res, couplings, lam_e, lam_h, T=300.0):
    """由近邻二聚体与各组 J 计算两种载流子的迁移率张量（解析，布拉维近似）。"""
    from .mobility import compute_mobility
    out = {}
    for car, lam in (("electron", lam_e), ("hole", lam_h)):
        paths = [{"name": f"G{d['group']}", "vector": d["vector"], "J_meV": abs(couplings[d["group"]][car]),
                  "count": 1} for d in res["dimers"] if couplings.get(d["group"], {}).get(car) is not None]
        if paths:
            r = compute_mobility(lam, paths, T)
            out[car] = {"mu_max": r["mobility_principal_cm2_Vs"][0], "mu_avg": r["mobility_tensor_average_cm2_Vs"],
                        "mu_scalar": r["mobility_cm2_Vs"], "tensor": r["mobility_tensor_cm2_Vs"]}
    return out


def strain_scan(crys, mols, axis, values_percent, lam_e, lam_h, T=300.0, cutoff=4.0, on_progress=None):
    """沿 a/b/c (或 'all' 等比) 缩放晶胞，逐点用 EHT 计算 J 和迁移率。"""
    from .crystal import strain_structure
    rows = []
    for k, pct in enumerate(values_percent):
        f = 1 + pct / 100.0
        sc = {"a": (f, 1, 1), "b": (1, f, 1), "c": (1, 1, f), "all": (f, f, f)}[axis]
        c2, m2 = strain_structure(crys, mols, sc)
        res, coup = eht_couplings_for_crystal(c2, m2, cutoff)
        mob = mobility_from_dimers(res, coup, lam_e, lam_h, T)
        rows.append({"strain_percent": pct, "n_dimers": len(res["dimers"]),
                     "J_max_hole": max((abs(v["hole"]) for v in coup.values()), default=0),
                     "J_max_electron": max((abs(v["electron"]) for v in coup.values()), default=0),
                     **{f"mu_{c}_{q}": mob.get(c, {}).get(f"mu_{q}") for c in ("electron", "hole") for q in ("max", "avg")}})
        if on_progress:
            on_progress(k + 1, len(values_percent))
    return rows
