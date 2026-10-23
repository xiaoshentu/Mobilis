"""调用 ADF 的片段方法 (TRANSFERINTEGRALS) 计算转移积分。

流程：先分别计算两个单体（片段），再以两个片段为基计算二聚体，ADF 直接给出
  e1, e2 (位点能), J (转移积分), S (重叠), V = (J - S(e1+e2)/2)/(1-S²)
支持两种输入格式：
  ams     — AMS2020 及以后版本 ($AMSBIN/ams，结果在 xxx.results/adf.rkf)
  classic — ADF2019 及以前版本 ($ADFBIN/adf，结果在 TAPE21)
"""
import json

import numpy as np
import re
from pathlib import Path

from .gaussian_io import HARTREE2EV, read_geometry

ADF_FUNCTIONALS = {
    "PW91": ("GGA", "PW91"), "PBE": ("GGA", "PBE"), "BP86": ("GGA", "BP86"), "BLYP": ("GGA", "BLYP"),
    "B3LYP": ("Hybrid", "B3LYP"), "PBE0": ("Hybrid", "PBE0"), "BHandHLYP": ("Hybrid", "BHandHLYP"),
    "LDA": ("LDA", "VWN"),
}
ADF_BASIS = ["DZ", "DZP", "TZP", "TZ2P", "QZ4P"]


def _xc_block(functional, indent):
    kind, name = ADF_FUNCTIONALS.get(functional.upper(), ADF_FUNCTIONALS.get(functional, ("GGA", functional)))
    return f"{indent}XC\n{indent}  {kind} {name}\n{indent}End"


def _atoms_lines(atoms, frag, style):
    tag = "" if frag is None else (f" adf.f={frag}" if style == "ams" else f" f={frag}")
    return [f"    {s:<3s}{x:16.8f}{y:16.8f}{z:16.8f}{tag}" for s, x, y, z in atoms]


def _dmpkf_lines(binvar, kf, txt):
    """用 dmpkf 导出 TransferIntegrals 为文本；按节导出为空时改为整体导出 (不同版本参数不同)。"""
    return [f'"{binvar}/dmpkf" "{kf}" TransferIntegrals > "{txt}" 2>&1',
            f'grep -q "hole" "{txt}" 2>/dev/null || "{binvar}/dmpkf" "{kf}" > "{txt}" 2>&1']


def _ams_job(name, atoms_lines, engine_lines, charge=0):
    sys_lines = ["System", "  Atoms", *atoms_lines, "  End"]
    if charge:
        sys_lines.append(f"  Charge {charge}")
    sys_lines.append("End")
    return "\n".join([f'AMS_JOBNAME={name} "$AMSBIN/ams" --delete-old-results <<eor', "Task SinglePoint",
                      *sys_lines, "", "Engine ADF", *engine_lines, "EndEngine", "eor",
                      f'[ -f {name}.results/adf.rkf ] || {{ echo "[错误] {name} 计算失败，请查看 {name}.results/ams.log"; exit 1; }}',
                      ""])


def _classic_job(atoms_block, body, tape, label, out):
    """ADF2019 及以前 (含 ADF2016) 的单个计算，写法与在 ADFjobs 中验证可用的 .run 脚本一致：
    全部使用绝对路径 (正斜杠)；输出写入单独的 .out；已完成的步骤 (结果文件存在) 自动跳过，可中断后续算。"""
    return [f'if [ ! -s "{tape}" ]; then',
            f'echo "running {label}"',
            "rm -f TAPE21 TAPE13 logfile",
            f'"$ADFBIN/adf" <<eor > "{out}" 2>&1', *atoms_block, *body, "End Input", "eor",
            f'[ -s TAPE21 ] || {{ echo "[错误] {label} 计算失败 (没有生成 TAPE21)，请查看 {out}"; exit 1; }}',
            f'mv -f TAPE21 "{tape}"', "rm -f TAPE13 logfile", "fi", ""]


def generate_adf_inputs(outdir, dimer=None, n_a=None, mono_a=None, mono_b=None, name="dimer",
                        functional="PW91", basis="TZP", style="ams", numerical_quality="Good",
                        extra="", atoms=None):
    """style: "ams" (AMS2020+) 或 "classic" (ADF2019 及以前, 含 ADF2016)。
    同时写出 {name}_adf_job.json 记录全部参数，运行时若发现版本不符可自动按正确格式重新生成。"""
    import json
    if atoms is not None:
        atoms_a, atoms_b = atoms
    elif dimer is not None:
        at = read_geometry(dimer)
        if not n_a or not 0 < n_a < len(at):
            raise ValueError("使用二聚体文件时必须指定单体 A 的原子数")
        atoms_a, atoms_b = at[:n_a], at[n_a:]
    else:
        atoms_a, atoms_b = read_geometry(mono_a), read_geometry(mono_b)
    from .crystal import check_dimer_geometry
    check_dimer_geometry(atoms_a, atoms_b)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fa, fb, full = f"{name}_frag1", f"{name}_frag2", f"{name}_full"
    text = ["#!/bin/sh", f"# ADF 转移积分 ({'AMS2020+' if style == 'ams' else 'ADF2019 及以前 / ADF2016'})：两个片段 + 二聚体",
            'cd "$(dirname "$0")"', ""]

    if style == "ams":
        common = ["  Basis", f"    Type {basis}", "    Core None", "  End", _xc_block(functional, "  "),
                  "  Symmetry NOSYM", f"  NumericalQuality {numerical_quality}"]
        common += [f"  {line}" for line in extra.splitlines() if line.strip()]
        text.append(_ams_job(fa, _atoms_lines(atoms_a, None, style), common))
        text.append(_ams_job(fb, _atoms_lines(atoms_b, None, style), common))
        full_engine = common + ["  Fragments", f"    frag1 {fa}.results/adf.rkf",
                                f"    frag2 {fb}.results/adf.rkf", "  End", "  TransferIntegrals Yes"]
        text.append(_ams_job(full, _atoms_lines(atoms_a, "frag1", style) + _atoms_lines(atoms_b, "frag2", style),
                             full_engine))
        result_file = f"{full}.results/adf.rkf"
        text += _dmpkf_lines("$AMSBIN", (outdir.resolve() / result_file).as_posix(), (outdir.resolve() / f"{full}.txt").as_posix()) + [""]
    else:
        ab = lambda f: (outdir.resolve() / f).as_posix()            # noqa: E731  绝对路径 (正斜杠)
        common = ["Units", " length Angstrom", "End", "Symmetry NOSYM", "", "Basis", f" type {basis}", " core None",
                  " createoutput None", "End", "", _xc_block(functional, ""), "", "NOPRINT LOGFILE"]
        common += [line for line in extra.splitlines() if line.strip()]
        for stem, atoms_, lab in ((fa, atoms_a, "片段 1"), (fb, atoms_b, "片段 2")):
            text += _classic_job([f"TITLE {stem}", "", "Atoms", *_atoms_lines(atoms_, None, style), "End", ""], common,
                                 ab(f"{stem}.t21"), f"{stem} ({lab})", ab(f"{stem}.out"))
        full_body = ["Units", " length Angstrom", "End", "Symmetry NOSYM", "", _xc_block(functional, ""), "",
                     "Fragments", f" frag1 {ab(fa + '.t21')}", f" frag2 {ab(fb + '.t21')}", "End",
                     "NOPRINT LOGFILE", "TRANSFERINTEGRALS"]
        text += _classic_job([f"TITLE {full}", "", "Atoms", *_atoms_lines(atoms_a, "frag1", style),
                              *_atoms_lines(atoms_b, "frag2", style), "End", ""], full_body,
                             ab(f"{full}.t21"), f"{full} (二聚体)", ab(f"{full}.out"))
        result_file = ab(f"{full}.t21")
        text += _dmpkf_lines("$ADFBIN", result_file, ab(full + ".txt")) + [""]

    text.append('echo "ADF 转移积分计算完成"')
    run = outdir / f"run_{name}_adf.sh"
    body = ("\n".join(text) + f"\n# 结果文件: {result_file}\n")
    run.write_bytes(body.encode("utf-8"))
    if style == "classic":           # ADFjobs (ADF2016 等) 直接打开运行的版本，与 .sh 内容相同
        (outdir / f"{name}_adf.run").write_bytes(
            body.replace("#!/bin/sh", "#! /bin/sh", 1).encode("gbk", errors="replace"))   # 与 ADFjobs 中可用的脚本同为 GBK
    job = {"name": name, "functional": functional, "basis": basis, "style": style,
           "numerical_quality": numerical_quality, "extra": extra,
           "atoms_a": [list(a) for a in atoms_a], "atoms_b": [list(b) for b in atoms_b]}
    (outdir / f"{name}_adf_job.json").write_text(json.dumps(job, ensure_ascii=False))
    return {"script": str(run), "result_file": str(outdir / result_file), "text_dump": str(outdir / f"{full}.txt")}


def regenerate_adf_job(script, style):
    """按记录的参数用另一种格式 (ams / classic) 重新生成脚本 (单个或批量)。"""
    import json
    script = Path(script)
    if script.name.startswith("batch_"):
        prefix = script.stem.rsplit("_", 1)[0] if script.stem.rsplit("_", 1)[-1].isdigit() else script.stem
        jf = script.parent / f"{prefix}_job.json"
        if not jf.exists():
            return None
        job = json.loads(jf.read_text())
        entries = [(n, [tuple(x) for x in a], [tuple(x) for x in b]) for n, a, b in job["entries"]]
        return generate_adf_batch(script.parent, entries, job["functional"], job["basis"], style,
                                  job.get("numerical_quality", "Good"), job.get("extra", ""), job.get("chunk", 0), prefix)
    name = script.name[len("run_"):-len("_adf.sh")]
    jf = script.parent / f"{name}_adf_job.json"
    if not jf.exists():
        return None
    job = json.loads(jf.read_text())
    atoms = ([tuple(a) for a in job["atoms_a"]], [tuple(b) for b in job["atoms_b"]])
    return generate_adf_inputs(script.parent, name=name, functional=job["functional"], basis=job["basis"],
                               style=style, numerical_quality=job.get("numerical_quality", "Good"),
                               extra=job.get("extra", ""), atoms=atoms)


def script_style(script):
    t = Path(script).read_text(errors="ignore")
    return "ams" if "$AMSBIN/ams" in t else "classic"


# ----------------------------------------------------------------------------------------------
# 结果读取
# ----------------------------------------------------------------------------------------------
_KEY_RE = re.compile(r"^(Vtot|V|J|S|e1|e2)\((.+)\)$")


def check_v(res):
    """用 V = (J − S(e1+e2)/2)/(1 − S²) 由分量重算 V，与输出的 V 比较。返回 {通道: (V输出, V重算) meV}。"""
    chk = {}
    for c, d in res.items():
        if all(k in d for k in ("J_meV", "S", "e1_eV", "e2_eV", "V_meV")):
            v = (d["J_meV"] - d["S"] * (d["e1_eV"] + d["e2_eV"]) * 1000 / 2) / (1 - d["S"] ** 2)
            chk[c] = (d["V_meV"], v)
    return chk


def _organize(raw):
    """把 TransferIntegrals 节的 {'V(hole)': x, ...} 整理为 {'hole': {'V_meV':...}}。值为 Hartree。"""
    out = {}
    for key, val in raw.items():
        m = _KEY_RE.match(key.strip())
        if not m:
            continue
        q, chan = m.groups()
        val = float(val[0] if isinstance(val, (list, tuple)) else val)
        d = out.setdefault(chan, {})
        if q == "S":
            d["S"] = val
        else:
            d[f"{q}_meV" if q in ("V", "Vtot", "J") else f"{q}_eV"] = val * HARTREE2EV * (1000 if q in ("V", "Vtot", "J") else 1)
    return out


def read_adf_transfer_integrals(path, ams_info=None):
    """读取 adf.rkf / TAPE21 (需要 PLAMS: AMS 自带的 amspython 或 pip install plams)，
    或读取 dmpkf 文本 / 本程序导出的 json。"""
    path = Path(path)
    if path.suffix == ".json":
        return _organize(json.loads(path.read_text()))
    if path.suffix in (".txt", ".dmp"):
        return _organize(_parse_text_dump(path.read_text(errors="ignore")))
    if path.suffix == ".out":
        raw = parse_adf_output(path.read_text(errors="ignore"))
        if not raw:
            raise ValueError(f"{path.name} 中没有找到转移积分 (V(hole) 等)")
        return _organize(raw)
    tried = []
    res_dir = path.parent.name.endswith(".results")
    stem = path.parent.name[:-len(".results")] if res_dir else path.stem
    base = path.parent.parent if res_dir else path.parent
    # 1) 脚本运行结束时用 dmpkf 导出的文本
    txt = base / f"{stem}.txt"
    if txt.exists() and txt.stat().st_size > 0:
        raw = _parse_text_dump(txt.read_text(errors="ignore"))
        if raw:
            return _organize(raw)
        tried.append(f"{txt.name} 中没有 TransferIntegrals")
    else:
        tried.append(f"{txt.name} 不存在或为空 (dmpkf 未成功导出)")
    # 2) ADF 文本输出 (.out)：批量脚本中为 G1.out，单个任务为 *_full.out
    name = stem[:-len("_full")] if stem.endswith("_full") else stem
    for out in (base / f"{name}.out", base / f"{stem}.out", base / f"{stem}.results" / "adf.out"):
        if out.exists():
            raw = parse_adf_output(out.read_text(errors="ignore"))
            if raw:
                return _organize(raw)
            tried.append(f"{out.name} 中没有识别出转移积分")
    # 3) PLAMS
    try:
        from scm.plams import KFFile
        return _organize(KFFile(str(path)).read_section("TransferIntegrals"))
    except ImportError:
        tried.append("没有 PLAMS")
    except Exception as exc:  # noqa: BLE001
        tried.append(f"PLAMS 读取失败: {exc}")
    # 4) 调用安装目录中的 dmpkf
    if ams_info:
        from .programs import dump_kf
        dump = dump_kf(ams_info, path, "TransferIntegrals")
        raw = _parse_text_dump(dump)
        if raw:
            return _organize(raw)
        tried.append("dmpkf 输出中没有 TransferIntegrals: " + " ".join(dump.split())[:160])
    else:
        tried.append("未指定 ADF 安装目录，无法调用 dmpkf")
    raise ImportError(f"无法读取 {path.name}：" + "；".join(tried) +
                      "。请把一个二聚体的 .out 文件发给开发者以完善读取")


_VAR = r"(Vtot|V|J|S|e1|e2)"
_NUM = r"(-?\d+\.\d*(?:[EeDd][-+]?\d+)?)"


def parse_adf_output(text):
    """从 ADF 输出 (.out) 中读取转移积分 (启发式)。支持
       V(hole) = -0.0123 [eV|a.u.]     以及表格行      V  (eV)   -0.0123   0.0456   (空穴、电子两列)
    返回 Hartree 为单位的 {'V(hole)': ...}，与 TAPE21 一致。"""
    from .gaussian_io import HARTREE2EV
    raw = {}
    # ADF2016 等版本的标准格式 (“Charge transfer integrals relevant for hole or electron mobility calculations”):
    #   V for hole transfer:           0.07727 eV
    #   J(hole) Charge transfer integral HOMO fragment 1 - HOMO fragment 2:       0.15764 eV
    #   S(hole) Overlap integral HOMO fragment 1 - HOMO fragment 2:      -0.01727
    num = r"(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)"
    unit = r"\s*(eV|a\.u\.|au|hartree|Hartree)?\s*$"
    std = {}
    for m in re.finditer(r"^\s*V for (hole|electron) transfer\s*:\s*" + num + unit, text, re.M):
        std[f"V({m.group(1)})"] = (float(m.group(2).replace("D", "E")), m.group(3))
    for m in re.finditer(r"^\s*(e1|e2|J|S)\((hole|electron)\)[^\n]*:\s*" + num + unit, text, re.M):
        std[f"{m.group(1)}({m.group(2)})"] = (float(m.group(3).replace("D", "E")), m.group(4))
    if std:
        for k, (v, u) in std.items():
            is_ev = (u or "").lower() == "ev" or (u is None and not k.startswith("S") and abs(v) > 0.5 and k[0] == "e")
            raw[k] = v / HARTREE2EV if (is_ev and not k.startswith("S")) else v
        return raw
    for line in text.splitlines():
        low = line.lower()
        unit_ev = "ev" in low.replace("level", "")
        for m in re.finditer(_VAR + r"\s*\(\s*(hole|electron)\s*\)\s*[:=]?\s*" + _NUM, line, re.I):
            var, chan, val = m.group(1), m.group(2).lower(), float(m.group(3).replace("D", "E"))
            var = {"vtot": "Vtot", "v": "V", "j": "J", "s": "S", "e1": "e1", "e2": "e2"}[var.lower()]
            raw[f"{var}({chan})"] = val / HARTREE2EV if (unit_ev and var != "S") else val
        m = re.match(r"^\s*" + _VAR + r"\b[^\d\-]*" + _NUM + r"\s+" + _NUM + r"\s*$", line)
        if m and "(" not in line.split()[0]:
            var = m.group(1)
            for chan, val in (("hole", m.group(2)), ("electron", m.group(3))):
                v = float(val.replace("D", "E"))
                raw.setdefault(f"{var}({chan})", v / HARTREE2EV if (unit_ev and var != "S") else v)
    return raw


def _parse_text_dump(text):
    """解析 dmpkf 输出或任何 'V(hole) ... 数值' 形式的文本。"""
    raw, lines = {}, text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        m = re.match(r"^((?:Vtot|V|J|S|e1|e2)\([^)]*\))\s*[:=]?\s*(-?\d+\.\d*(?:[EeDd][-+]?\d+)?)?\s*$", s)
        if not m:
            continue
        if m.group(2):
            raw[m.group(1)] = float(m.group(2).replace("D", "E"))
            continue
        for nxt in lines[i + 1:i + 4]:       # dmpkf: 变量名下一两行才是数值
            nums = re.findall(r"-?\d+\.\d+(?:[EeDd][-+]?\d+)?", nxt)
            if nums:
                raw[m.group(1)] = float(nums[-1].replace("D", "E"))
                break
    return raw


# ----------------------------------------------------------------------------------------------
# 批量计算：一个脚本内顺序计算全部二聚体 (与 make_dimers.py 的 batch_*.run 相同的思路)
# ----------------------------------------------------------------------------------------------
def _same_fragment(atoms1, atoms2, tol=0.03):
    """两个分子能否只通过平移 + 纯转动 (不含镜像) 重合 (原子顺序相同)。可以则共用同一个片段文件。"""
    if [a[0] for a in atoms1] != [b[0] for b in atoms2]:
        return False
    x1 = np.array([a[1:] for a in atoms1], float)
    x2 = np.array([b[1:] for b in atoms2], float)
    p, q = x1 - x1.mean(0), x2 - x2.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    R = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ R - q) ** 2).sum(1).mean())) < tol


def _match_order(ref, atoms, tol=0.05):
    """寻找原子置换 perm，使 atoms[perm] 能由 ref 经平移 + 纯转动得到 (晶体中原子顺序不同的对称等价分子)。
    做法：按惯性主轴对齐 (只取保持手性的 4 种朝向)，再按元素做最优匹配 (匈牙利算法)，最后用 Kabsch 检验。"""
    from scipy.optimize import linear_sum_assignment
    sr, sa = [a[0] for a in ref], [a[0] for a in atoms]
    if sorted(sr) != sorted(sa):
        return None
    xr = np.array([a[1:] for a in ref], float)
    xa = np.array([a[1:] for a in atoms], float)
    cr, ca = xr.mean(0), xa.mean(0)
    _, vr = np.linalg.eigh(np.cov((xr - cr).T))
    _, va = np.linalg.eigh(np.cov((xa - ca).T))
    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        Rb = va * np.array(signs)
        if np.linalg.det(Rb) * np.linalg.det(vr) < 0:
            Rb[:, 2] *= -1
        y = (xa - ca) @ Rb @ vr.T                       # atoms 转到 ref 的主轴坐标系
        perm = np.empty(len(sr), int)
        ok = True
        for el in set(sr):
            ir = [k for k, e in enumerate(sr) if e == el]
            ia = [k for k, e in enumerate(sa) if e == el]
            d = np.linalg.norm((xr[ir] - cr)[:, None] - y[ia][None], axis=-1)
            r, c = linear_sum_assignment(d)
            if d[r, c].max() > 0.5:
                ok = False
                break
            perm[np.array(ir)[r]] = np.array(ia)[c]
        if ok and _same_fragment(ref, [atoms[k] for k in perm], tol):
            return perm
    return None


def generate_adf_batch(outdir, entries, functional="PW91", basis="TZP", style="classic", numerical_quality="Good",
                       extra="", chunk=0, prefix="batch_adf"):
    """entries: [(名称, atoms_a, atoms_b)]。生成 {prefix}[_k].sh (本程序运行) 与 {prefix}[_k].run (ADFjobs 运行)：
    先计算各个不同的单体片段 (能转动重合的分子共用一个)，再依次计算全部二聚体；
    每个二聚体结束后用 dmpkf 导出 {名称}_full.txt，已完成的步骤自动跳过 (可中断后续算)。"""
    import json
    from .crystal import check_dimer_geometry
    for name, a, b in entries:
        try:
            check_dimer_geometry(a, b)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from None
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ab = lambda f: (outdir.resolve() / f).as_posix()            # noqa: E731
    frags = []                                                     # [(名称, atoms)]

    def frag_of(atoms):
        """返回 (片段名, 按片段原子顺序排列的分子)。原子顺序不同但对称等价的分子也共用片段。"""
        for nm, at in frags:
            if _same_fragment(at, atoms):
                return nm, atoms
            perm = _match_order(at, atoms)
            if perm is not None:
                return nm, [atoms[k] for k in perm]
        nm = f"frag{len(frags) + 1}"
        frags.append((nm, atoms))
        return nm, atoms
    plan = []
    for name, a, b in entries:
        fa, a2 = frag_of(a)
        fb, b2 = frag_of(b)
        plan.append((name, a2, b2, fa, fb))
    t21 = (lambda f: ab(f"{f}.t21")) if style == "classic" else (lambda f: ab(f"{f}.results/adf.rkf"))

    if style == "classic":
        common = ["Units", " length Angstrom", "End", "Symmetry NOSYM", "", "Basis", f" type {basis}", " core None",
                  " createoutput None", "End", "", _xc_block(functional, ""), "", "NOPRINT LOGFILE"]
        common += [line for line in extra.splitlines() if line.strip()]

        def frag_step(nm, atoms):
            return _classic_job([f"TITLE {nm} monomer fragment", "", "Atoms", *_atoms_lines(atoms, None, style), "End",
                                 ""], common, t21(nm), f"{nm} (单体片段)", ab(f"{nm}.out"))

        def dimer_step(name, a, b, fa, fb):
            tag_a, tag_b = (f"{fa}/1", f"{fb}/2") if fa == fb else (fa, fb)
            fr = [f" {f} {t21(f)}" for f in dict.fromkeys([fa, fb])]
            body = ["Units", " length Angstrom", "End", "Symmetry NOSYM", "", _xc_block(functional, ""), "",
                    "Fragments", *fr, "End", "NOPRINT LOGFILE", "TRANSFERINTEGRALS"]
            st = _classic_job([f"TITLE {name}", "", "Atoms", *_atoms_lines(a, tag_a, style), *_atoms_lines(b, tag_b, style),
                               "End", ""], body, ab(f"{name}_full.t21"), f"{name} (二聚体)", ab(f"{name}.out"))
            return st[:-2] + _dmpkf_lines("$ADFBIN", ab(name + "_full.t21"), ab(name + "_full.txt")) + ["fi", ""]
        res_of = lambda name: ab(f"{name}_full.t21")             # noqa: E731
    else:
        common = ["  Basis", f"    Type {basis}", "    Core None", "  End", _xc_block(functional, "  "),
                  "  Symmetry NOSYM", f"  NumericalQuality {numerical_quality}"]
        common += [f"  {line}" for line in extra.splitlines() if line.strip()]

        def frag_step(nm, atoms):
            return [f'if [ ! -s "{t21(nm)}" ]; then', _ams_job(nm, _atoms_lines(atoms, None, style), common), "fi", ""]

        def dimer_step(name, a, b, fa, fb):
            na, nb = (fa, fa + "b") if fa == fb else (fa, fb)
            eng = common + ["  Fragments", f"    {na} {t21(fa)}", f"    {nb} {t21(fb)}", "  End", "  TransferIntegrals Yes"]
            return [f'if [ ! -s "{ab(name + "_full.results/adf.rkf")}" ]; then',
                    _ams_job(f"{name}_full", _atoms_lines(a, na, style) + _atoms_lines(b, nb, style), eng),
                    *_dmpkf_lines("$AMSBIN", ab(name + "_full.results/adf.rkf"), ab(name + "_full.txt")), "fi", ""]
        res_of = lambda name: ab(f"{name}_full.results/adf.rkf")  # noqa: E731

    chunk = chunk if chunk and chunk > 0 else len(plan)
    scripts = []
    for k in range(0, len(plan), chunk):
        part = plan[k:k + chunk]
        need = list(dict.fromkeys([p[3] for p in part] + [p[4] for p in part]))
        tag = prefix + (f"_{k // chunk + 1}" if chunk < len(plan) else "")
        text = ["#!/bin/sh", f"# ADF 批量转移积分 ({'ADF2019 及以前 / ADF2016' if style == 'classic' else 'AMS2020+'})："
                f"{len(need)} 个单体片段 + {len(part)} 个二聚体",
                f"# 可在 ADFjobs 中直接运行 {tag}.run，或在本程序“运行任务”中运行 {tag}.sh；已完成的步骤自动跳过", ""]
        text += [f'cd "{outdir.resolve().as_posix()}" || exit 1', ""]     # ADFjobs 可能在别处执行脚本
        for nm in need:
            text += frag_step(nm, dict(frags)[nm])
        for name, a, b, fa, fb in part:
            text += dimer_step(name, a, b, fa, fb)
        text.append('echo "batch finished"')
        text += [f"# 结果文件: {res_of(p[0])}" for p in part]
        body = "\n".join(text) + "\n"
        (outdir / f"{tag}.sh").write_bytes(body.encode("utf-8"))
        (outdir / f"{tag}.run").write_bytes(body.replace("#!/bin/sh", "#! /bin/sh", 1).encode("gbk", errors="replace"))
        scripts.append(str(outdir / f"{tag}.sh"))
    job = {"batch": True, "prefix": prefix, "chunk": chunk, "functional": functional, "basis": basis, "style": style,
           "numerical_quality": numerical_quality, "extra": extra,
           "entries": [[n, [list(x) for x in a], [list(x) for x in b]] for n, a, b in entries]}
    (outdir / f"{prefix}_job.json").write_text(json.dumps(job, ensure_ascii=False))
    return {"scripts": scripts, "fragments": [f for f, _ in frags],
            "shared": sum(1 for p in plan if p[3] == p[4]), "results": [res_of(p[0]) for p in plan]}
