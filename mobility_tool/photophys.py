"""激发态与光物理：S0/S1/T1 优化输入、TD-DFT 输出解析、吸收/发射光谱、Stokes 位移、态间 RMSD、k_r/k_nr。

生成的三个文件（S1、T1 用 %oldchk 读取 S0 的优化结构，运行器按 S0 → S1 → T1 顺序执行）:
  {name}_S0.gjf : S0 opt freq → TD(nstates) 单点 (吸收, 单点级别)
  {name}_S1.gjf : S1 opt td(root=1) → TD 单点 (S1 几何下的荧光能量)
  {name}_T1.gjf : T1 opt td(triplets,root=1) → TD(triplets) 单点 (T1 几何下的磷光能量)
"""
import re
from pathlib import Path

import numpy as np

from .decomposition import kabsch_align
from .gaussian_io import (ELEMENTS, MASSES, level_and_tail, read_geometry, solvent_kw, split_log_steps,
                          write_gjf)

NM_EV = 1239.841984


# ----------------------------------------------------------------------------------------------
# 输入
# ----------------------------------------------------------------------------------------------
def generate_excited_inputs(geom, outdir, name=None, method="B3LYP", basis="6-311G(d,p)", sp_basis="def2-TZVP",
                            solvent="", nstates=10, charge=0, mult=1, nproc=8, mem="8GB", ecp_basis="SDD",
                            freq_excited=False):
    atoms = read_geometry(geom)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = name or Path(geom).stem
    method, *more = method.split()
    extra = " ".join(more)
    elems = [a[0] for a in atoms]
    opt_lvl, opt_tail = level_and_tail(method, basis, elems, ecp_basis)
    sp_lvl, sp_tail = level_and_tail(method, sp_basis or basis, elems, ecp_basis)
    solv = solvent_kw(solvent)
    c = dict(nproc=nproc, mem=mem, charge=charge, mult=mult)
    fx = " freq" if freq_excited else ""
    s0 = f"{name}_S0"
    jobs = {
        s0: [dict(chk=f"{s0}.chk", route=f"# opt freq {opt_lvl} {solv} {extra}", title=f"{name} S0 opt",
                  atoms=atoms, tail=opt_tail, **c),
             dict(oldchk=f"{s0}.chk", chk=f"{s0}_td.chk", route=f"# td(nstates={nstates}) {sp_lvl} geom=check {solv} {extra}",
                  title=f"{name} absorption at S0 geometry", tail=sp_tail, **c)],
        f"{name}_S1": [dict(oldchk=f"{s0}.chk", chk=f"{name}_S1.chk",
                            route=f"# opt{fx} td(nstates={min(nstates, 6)},root=1) {opt_lvl} geom=check {solv} {extra}",
                            title=f"{name} S1 opt", tail=opt_tail, **c),
                       dict(oldchk=f"{name}_S1.chk", chk=f"{name}_S1_td.chk",
                            route=f"# td(nstates={min(nstates, 6)}) {sp_lvl} geom=check {solv} {extra}",
                            title=f"{name} fluorescence at S1 geometry", tail=sp_tail, **c)],
        f"{name}_T1": [dict(oldchk=f"{s0}.chk", chk=f"{name}_T1.chk",
                            route=f"# opt{fx} td(triplets,nstates={min(nstates, 6)},root=1) {opt_lvl} geom=check {solv} {extra}",
                            title=f"{name} T1 opt", tail=opt_tail, **c),
                       dict(oldchk=f"{name}_T1.chk", chk=f"{name}_T1_td.chk",
                            route=f"# td(triplets,nstates={min(nstates, 6)}) {sp_lvl} geom=check {solv} {extra}",
                            title=f"{name} phosphorescence at T1 geometry", tail=sp_tail, **c)],
    }
    for stem, js in jobs.items():
        write_gjf(outdir / f"{stem}.gjf", js)
    from .jobs import write_windows_bat
    write_windows_bat(outdir, list(jobs), name="run_excited.bat")
    (outdir / "run_excited.sh").write_text("#!/bin/bash\n" + "\n".join(
        f"g16 < {s}.gjf > {s}.log && formchk {s}.chk {s}.fchk" for s in jobs) + "\n")
    return list(jobs)


# ----------------------------------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------------------------------
_ES = re.compile(r"Excited State\s+(\d+):\s+(\S+)\s+(-?[\d.]+)\s+eV\s+(-?[\d.]+)\s+nm\s+f=\s*(-?[\d.]+)")
_TR = re.compile(r"^\s+(\d+[AB]?)\s*->\s*(\d+[AB]?)\s+(-?[\d.]+)")


def parse_excitations(text):
    """返回最后一组激发态 [{n, mult, E_eV, nm, f, configs:[(from, to, coef, %)]}]。"""
    states, sets, cur = [], [], None
    for line in text.splitlines():
        m = _ES.search(line)
        if m:
            n = int(m.group(1))
            if n == 1 and states:
                sets.append(states)
                states = []
            cur = {"n": n, "mult": m.group(2).split("-")[0], "E_eV": float(m.group(3)), "nm": float(m.group(4)),
                   "f": float(m.group(5)), "configs": []}
            states.append(cur)
            continue
        if cur is not None:
            t = _TR.match(line)
            if t:     # 只取激发 (->)，退激发 (<-) 不计；闭壳层贡献 = 2c²
                coef = float(t.group(3))
                cur["configs"].append((t.group(1), t.group(2), coef, 200 * coef ** 2))
    if states:
        sets.append(states)
    return sets[-1] if sets else []


def _geometry_in(block):
    blocks = re.findall(r"(?:Standard|Input) orientation:.*?-{5,}\n.*?-{5,}\n(.*?)\n\s*-{5,}", block, re.S)
    if not blocks:
        return None
    out = []
    for line in blocks[-1].splitlines():
        p = line.split()
        out.append((ELEMENTS[int(p[1])], float(p[3]), float(p[4]), float(p[5])))
    return out


def read_log(path):
    """按 Link1 步骤返回 [{route, states, geometry}]。"""
    text = Path(path).read_text(errors="ignore")
    return [{"route": r, "states": parse_excitations(b), "geometry": _geometry_in(b),
             "normal": "Normal termination" in b} for r, b in split_log_steps(text)]


def homo_index_label(configs, homo):
    """把轨道编号转换为 HOMO-n / LUMO+n 记号。"""
    def lab(k):
        k = int(re.match(r"\d+", k).group(0))
        return "HOMO" if k == homo else f"HOMO-{homo - k}" if k < homo else "LUMO" if k == homo + 1 \
            else f"LUMO+{k - homo - 1}"
    return [(f"{lab(a)}→{lab(b)}", pct) for a, b, _, pct in configs]


def _find(outdir, stem):
    for ext in (".log", ".out", ".LOG", ".OUT"):
        p = Path(outdir) / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def analyze_excited(outdir, name, homo=None):
    """汇总吸收 / 荧光 / 磷光 (汇总表)、Stokes 位移、S0–S1 与 S0–T1 RMSD。"""
    res = {"name": name, "warnings": []}
    steps = {}
    for st in ("S0", "S1", "T1"):
        p = _find(outdir, f"{name}_{st}")
        if p is None:
            res["warnings"].append(f"缺少 {name}_{st}.log/.out")
            continue
        steps[st] = read_log(p)
        from .gaussian_io import explain_gaussian_error
        err = explain_gaussian_error(p.read_text(errors="ignore"))
        if err:
            res["warnings"].append(f"{p.name}: {err}")
        if homo is None:       # HOMO 编号 = alpha 电子数（闭壳层基态）
            m = re.search(r"(\d+)\s+alpha electrons", p.read_text(errors="ignore"))
            if m and st == "S0":
                homo = int(m.group(1))
        for k, s in enumerate(steps[st]):
            if not s["normal"]:
                res["warnings"].append(f"{p.name} 第 {k + 1} 步未正常结束")

    def last_states(st):
        for s in reversed(steps.get(st, [])):
            if s["states"]:
                return s["states"]
        return None

    rows = []
    for st, kind, trans in (("S0", "Absorption", "S0→S1"), ("S1", "Fluorescence", "S1→S0"),
                            ("T1", "Phosphorescence", "T1→S0")):
        states = last_states(st)
        if not states:
            continue
        s1 = states[0]
        cfg = sorted(s1["configs"], key=lambda c: -c[3])[:3]
        rows.append({"kind": kind, "transition": trans, "E_eV": s1["E_eV"], "nm": s1["nm"], "f": s1["f"],
                     "configs": homo_index_label(cfg, homo) if homo else [(f"{a}→{b}", p) for a, b, _, p in cfg]})
        if st == "S0":
            res["absorption_states"] = states
    res["table"] = rows
    if not rows:
        detail = "\n".join(w for w in res["warnings"] if "Gaussian 计算出错" in w)
        raise ValueError("没有读到任何激发态 (TD-DFT) 结果。" + ("\n" + detail if detail else
                         "请确认 S0 / S1 / T1 输出中包含 td 计算并正常结束"))
    e = {r["kind"]: r["E_eV"] for r in rows}
    if "Absorption" in e and "Fluorescence" in e:
        res["stokes_shift_eV"] = e["Absorption"] - e["Fluorescence"]
    geo = {st: steps[st][0]["geometry"] for st in steps if steps[st] and steps[st][0]["geometry"]}
    for st in ("S1", "T1"):
        if "S0" in geo and st in geo and len(geo[st]) == len(geo["S0"]):
            res[f"rmsd_S0_{st}"] = rmsd(geo["S0"], geo[st])["rmsd"]
    res["geometries"] = geo
    return res


# ----------------------------------------------------------------------------------------------
# 光谱、RMSD、速率
# ----------------------------------------------------------------------------------------------
def broaden(energies_ev, intensities, fwhm_ev=0.3, nm_range=(250, 850), npts=1200, normalize=True):
    """能量空间高斯展宽，返回 (波长 nm, 强度)。"""
    nm = np.linspace(*nm_range, npts)
    e = NM_EV / nm
    sig = fwhm_ev / (2 * np.sqrt(2 * np.log(2)))
    y = np.zeros_like(e)
    for ei, fi in zip(energies_ev, intensities):
        y += fi * np.exp(-0.5 * ((e - ei) / sig) ** 2)
    if normalize and y.max() > 0:
        y = y / y.max()
    return nm, y


def absorption_spectrum(states, fwhm_ev=0.3, nm_range=(250, 850)):
    return broaden([s["E_eV"] for s in states], [s["f"] for s in states], fwhm_ev, nm_range)


def emission_band(e_ev, fwhm_ev=0.18, nm_range=(450, 900)):
    return broaden([e_ev], [1.0], fwhm_ev, nm_range)


def rmsd(atoms_a, atoms_b, heavy_only=False, mass_weighted=False):
    """叠合后的 RMSD (Å)。atoms: [(元素, x, y, z)]，两者原子顺序须一致。"""
    sym = [a[0] for a in atoms_a]
    if [b[0] for b in atoms_b] != sym:
        raise ValueError("两个结构的原子顺序/种类不一致")
    xa = np.array([a[1:] for a in atoms_a], float)
    xb = np.array([b[1:] for b in atoms_b], float)
    w = np.array([MASSES.get(s, 12.0) for s in sym]) if mass_weighted else np.ones(len(sym))
    aligned = kabsch_align(xa, xb, w)
    sel = np.array([s != "H" for s in sym]) if heavy_only else np.ones(len(sym), bool)
    dev = np.linalg.norm(aligned - xa, axis=1)
    return {"rmsd": float(np.sqrt((dev[sel] ** 2).mean())), "aligned": aligned, "deviation": dev, "symbols": sym,
            "reference": xa}


def decay_rates(plqy, tau_us):
    """k_r = Φ/τ, k_nr = (1-Φ)/τ (s^-1)；τ 以 μs 输入。"""
    tau = float(tau_us) * 1e-6
    return {"k_r": float(plqy) / tau, "k_nr": (1 - float(plqy)) / tau}
