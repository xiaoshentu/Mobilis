"""四点法重组能 + IP/EA：生成高斯输入、解析输出。

对应 7 个能量（见 ipea.py）:
  {name}_neutral.gjf : 中性优化(+freq) → [中性单点] → 阳离子单点 → 阴离子单点   (能量 1, 4, 5)
  {name}_cation.gjf  : 阳离子优化(+freq) → [阳离子单点] → 中性单点           (能量 2, 6)
  {name}_anion.gjf   : 阴离子优化(+freq) → [阴离子单点] → 中性单点           (能量 3, 7)
方括号中的步骤只在“单点计算级别”与优化级别不同时生成，此时全部能量都取单点级别，
保证 7 个能量处在同一级别（例如 B3LYP/6-311G(d,p) 优化，B3LYP/6-311G(2d,2p) 算能量）。
"""
import json
from pathlib import Path

from .gaussian_io import level_and_tail, read_geometry, read_log_steps, solvent_kw, write_gjf
from .ipea import HARTREE2EV, ip_ea_from_energies

CARRIERS = {"hole": ("cation", +1), "electron": ("anion", -1)}
STATE_KEY = {("neutral", "neutral"): "E0_N", ("cation", "cation"): "Ec_C", ("anion", "anion"): "Ea_A",
             ("cation", "neutral"): "Ec_N", ("anion", "neutral"): "Ea_N",
             ("neutral", "cation"): "E0_C", ("neutral", "anion"): "E0_A"}

# 界面中的常用选项（均可手动输入其它值）
FUNCTIONALS = ["B3LYP", "B3LYP em=gd3bj", "PBE0", "PBE1PBE em=gd3bj", "M062X", "CAM-B3LYP",
               "wB97XD", "BHandHLYP", "B97D3", "PBEPBE", "HF"]
BASIS_SETS = ["6-31G(d)", "6-31G(d,p)", "6-311G(d,p)", "6-311G(2d,2p)", "6-311+G(d,p)", "def2-SVP",
              "def2-TZVP", "def2-TZVPP", "cc-pVDZ", "cc-pVTZ"]
MANIFEST = "reorg_manifest.json"


def _split_method(method, extra):
    method, *more = method.split()
    return method, " ".join(more + [extra]).strip()


def generate_reorg_inputs(geom, outdir, name=None, method="B3LYP", basis="6-31G(d)",
                          carriers=("hole", "electron"), charge=0, mult=1, ion_mult=None,
                          nproc=8, mem="8GB", freq=True, extra="", program="g16",
                          sp_method=None, sp_basis=None, solvent="", ecp_basis="SDD"):
    atoms = read_geometry(geom)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = name or Path(geom).stem
    ion_mult = ion_mult or (2 if mult == 1 else mult - 1)
    method, extra_opt = _split_method(method, extra)
    sp_m, extra_sp = _split_method(sp_method or method, extra if sp_method else extra_opt)
    level = f"{method}/{basis}"
    sp_level = f"{sp_m}/{sp_basis or basis}"
    separate_sp = sp_level.lower() != level.lower() or extra_sp != extra_opt
    opt_kw = "opt freq" if freq else "opt"
    common = dict(nproc=nproc, mem=mem)
    elems = [a[0] for a in atoms]
    opt_route, opt_tail = level_and_tail(method, basis, elems, ecp_basis)
    sp_route, sp_tail = level_and_tail(sp_m, sp_basis or basis, elems, ecp_basis)
    solv = solvent_kw(solvent)
    q = {"neutral": (charge, mult), "cation": (charge + 1, ion_mult), "anion": (charge - 1, ion_mult)}
    manifest = {"name": name, "level": level + (f" {solv}" if solv else ""),
                "sp_level": (sp_level if separate_sp else level) + (f" {solv}" if solv else ""),
                "separate_sp": separate_sp, "charge": charge, "files": {}}

    def sp_job(stem, state, geom_state):
        c, m = q[state]
        return dict(oldchk=f"{stem}.chk", chk=f"{stem}_sp_{state}.chk",
                    route=f"# {sp_route} geom=check {solv} {extra_sp}",
                    title=f"{name} {state} single point at {geom_state} geometry ({sp_level})",
                    charge=c, mult=m, tail=sp_tail, **common)

    ions = [CARRIERS[c][0] for c in carriers]
    for opt_state, others in [("neutral", ions)] + [(ion, ["neutral"]) for ion in ions]:
        stem = f"{name}_{opt_state}"
        c, m = q[opt_state]
        jobs = [dict(chk=f"{stem}.chk", route=f"# {opt_kw} {opt_route} {solv} {extra_opt}",
                     title=f"{name} {opt_state} opt", charge=c, mult=m, atoms=atoms, tail=opt_tail, **common)]
        steps = [{"state": opt_state, "geom": opt_state, "role": "opt"}]
        if separate_sp:
            jobs.append(sp_job(stem, opt_state, opt_state))
            steps.append({"state": opt_state, "geom": opt_state, "role": "sp"})
        for st in others:
            jobs.append(sp_job(stem, st, opt_state))
            steps.append({"state": st, "geom": opt_state, "role": "sp"})
        write_gjf(outdir / f"{stem}.gjf", jobs)
        manifest["files"][stem] = steps

    (outdir / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    script = ["#!/bin/bash", "# 依次运行重组能 / IP / EA 计算（可按需改为提交到作业系统）"]
    for s in manifest["files"]:
        script += [f"{program} < {s}.gjf > {s}.log", f"formchk {s}.chk {s}.fchk"]
    script.append("# *_neutral.fchk / *_cation.fchk / *_anion.fchk 含 Hessian，可用于重组能分解")
    (outdir / "run_reorg.sh").write_text("\n".join(script) + "\n")
    from .jobs import write_windows_bat
    write_windows_bat(outdir, list(manifest["files"]), name="run_reorg.bat")
    return manifest


def _find_log(outdir, stem):
    for ext in (".log", ".out", ".LOG", ".OUT"):
        p = Path(outdir) / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def analyze_reorg(outdir):
    outdir = Path(outdir)
    manifest = json.loads((outdir / MANIFEST).read_text())
    separate = manifest.get("separate_sp", False)
    energies, warnings, orbitals = {}, [], None
    for stem, steps in manifest["files"].items():
        log = _find_log(outdir, stem)
        if log is None:
            warnings.append(f"缺少输出文件 {stem}.log/.out")
            continue
        blocks = read_log_steps(log)
        if len(blocks) < len(steps):
            warnings.append(f"{log.name}: 只找到 {len(blocks)} 个计算步骤，预期 {len(steps)} 个（计算未完成？）")
        for step, blk in zip(steps, blocks):
            tag = f"{log.name} [{step['state']}@{step['geom']} {step.get('role', '')}]"
            if blk["error"] or not blk["normal"]:
                warnings.append(f"{tag}: 未正常结束")
            if blk["opt_failed"]:
                warnings.append(f"{tag}: 优化未收敛")
            if blk["n_imag"]:
                warnings.append(f"{tag}: 存在 {blk['n_imag']} 个虚频 (最低 {blk['min_freq']:.1f} cm^-1)")
            if step["state"] == "neutral" and step["geom"] == "neutral" and blk["orbitals"]:
                orbitals = blk["orbitals"]       # 有单点步骤时会被单点级别覆盖
            if blk["energy"] is None:
                warnings.append(f"{tag}: 没有找到 SCF 能量")
                continue
            if separate and step.get("role") == "opt":
                continue
            energies[STATE_KEY[(step["state"], step["geom"])]] = blk["energy"]

    ipea = ip_ea_from_energies(energies)
    results = {}
    for carrier in ("hole", "electron"):
        suf = "h" if carrier == "hole" else "e"
        if f"lambda_{suf}" in ipea:
            r = {"lambda1_eV": ipea[f"lambda1_{suf}"]["eV"], "lambda2_eV": ipea[f"lambda2_{suf}"]["eV"],
                 "lambda_eV": ipea[f"lambda_{suf}"]["eV"]}
            for k in (("IP_v", "IP_a", "HEP") if carrier == "hole" else ("EA_v", "EA_a", "EEP")):
                if k in ipea:
                    r[f"{k}_eV"] = ipea[k]["eV"]
            results[carrier] = r
    orb = None
    if orbitals:
        occ = sorted(orbitals["Alpha occ."] + orbitals["Beta occ."])
        virt = sorted(orbitals["Alpha virt."] + orbitals["Beta virt."])
        if occ and virt:
            orb = {"HOMO_eV": occ[-1] * HARTREE2EV, "LUMO_eV": virt[0] * HARTREE2EV,
                   "gap_eV": (virt[0] - occ[-1]) * HARTREE2EV,
                   "Koopmans_IP_eV": -occ[-1] * HARTREE2EV, "Koopmans_EA_eV": -virt[0] * HARTREE2EV}
    return {"name": manifest["name"], "level": manifest["level"], "sp_level": manifest.get("sp_level"),
            "energies_Ha": energies, "ip_ea": ipea, "orbitals": orb, "results": results, "warnings": warnings}
