"""用 Gaussian 的 FC/FCHT 功能分解重组能、计算黄昆因子和 Duschinsky 矩阵。

输入模板（两个态先分别 opt+freq，得到 initial.chk / final.chk）:
    %oldchk=final.chk
    #p geom=allcheck freq(readfc,fcht,readfcht)

    initial=source=chk final=source=chk spectroscopy=onephotonemission
    print=(huangrhys,matrix=JK)
    [RedDim=ClearLowFreq=100.0]

    initial.chk
    final.chk

按 Sobereva 的说明，重组能分解应使用“末态极小点”的简正模式，因此对每种载流子生成两个任务:
    {name}_fcht_{ion}_to_neutral : initial=离子, final=中性 → 分解 λ1 (中性势能面)
    {name}_fcht_neutral_to_{ion} : initial=中性, final=离子 → 分解 λ2 (离子势能面)
λ_i = S_i · h c ν_i   (ν 以 cm^-1 计时 λ_i[cm^-1] = S_i · ν_i)
"""
import csv
import re
from pathlib import Path

import numpy as np

CM2EV = 1 / 8065.543937
CM2KJ = 0.01196266

_D = re.compile(r"-?\d*\.\d+(?:[DdEe][-+]?\d+)?")


def _f(s):
    return float(s.replace("D", "E").replace("d", "e"))


# ----------------------------------------------------------------------------------------------
# 输入
# ----------------------------------------------------------------------------------------------
def fcht_input(initial_chk, final_chk, nproc=8, mem="8GB", clear_low_freq=None, chk=None):
    lines = [f"%oldchk={final_chk}"]
    if chk:
        lines.append(f"%chk={chk}")
    lines += [f"%nprocshared={nproc}", f"%mem={mem}", "#p geom=allcheck freq(readfc,fcht,readfcht)", "",
              "initial=source=chk final=source=chk spectroscopy=onephotonemission",
              "print=(huangrhys,matrix=JK)"]
    if clear_low_freq:
        lines.append(f"RedDim=ClearLowFreq={float(clear_low_freq):.1f}")
    lines += ["", initial_chk, final_chk, "", ""]
    return "\n".join(lines)


def generate_fcht_inputs(outdir, name, carriers=("hole", "electron"), nproc=8, mem="8GB", clear_low_freq=None):
    """在重组能目录中，为每种载流子生成两个 FCHT 任务。"""
    outdir = Path(outdir)
    ions = {"hole": "cation", "electron": "anion"}
    jobs = []
    for c in carriers:
        ion = ions[c]
        for ini, fin, part in ((ion, "neutral", "lambda1"), ("neutral", ion, "lambda2")):
            stem = f"{name}_fcht_{ini}_to_{fin}"
            (outdir / f"{stem}.gjf").write_text(
                fcht_input(f"{name}_{ini}.chk", f"{name}_{fin}.chk", nproc, mem, clear_low_freq, f"{stem}.chk"))
            jobs.append({"stem": stem, "carrier": c, "part": part, "initial": ini, "final": fin})
    lines = ["#!/bin/bash", "# 需在 run_reorg.sh 完成后运行（要用到优化+频率的 chk）"]
    lines += [f"g16 < {j['stem']}.gjf > {j['stem']}.log" for j in jobs]
    (outdir / "run_fcht.sh").write_text("\n".join(lines) + "\n")
    from .jobs import write_windows_bat
    write_windows_bat(outdir, [j["stem"] for j in jobs], fchk_stems=[], name="run_fcht.bat")
    return jobs


# ----------------------------------------------------------------------------------------------
# 输出解析
# ----------------------------------------------------------------------------------------------
def _freq_blocks(text):
    """返回按出现顺序的频率列表组（每遇到一次 'and normal coordinates:' 开始新的一组）。"""
    blocks, cur = [], None
    for line in text.splitlines():
        if "and normal coordinates:" in line:
            cur = []
            blocks.append(cur)
        elif cur is not None and line.strip().startswith("Frequencies --"):
            cur += [_f(v) for v in _D.findall(line.split("--", 1)[1])]
    return [b for b in blocks if b]


def _parse_matrix(lines):
    """解析高斯分块打印的矩阵 (每块 ≤5 列，列号行 + 数据行)。"""
    entries, cols = {}, None
    for line in lines:
        tok = line.split()
        if not tok:
            continue
        if all(t.isdigit() for t in tok):
            cols = [int(t) for t in tok]
            continue
        if cols and tok[0].isdigit():
            vals = _D.findall(line[line.index(tok[0]) + len(tok[0]):])
            for c, v in zip(cols, vals):
                entries[(int(tok[0]), c)] = _f(v)
    if not entries:
        return None
    n = max(max(k) for k in entries)
    m = np.zeros((max(k[0] for k in entries), max(k[1] for k in entries)))
    for (i, j), v in entries.items():
        m[i - 1, j - 1] = v
    return m if n else None


def _section(text, start_pat, end_pats):
    m = re.search(start_pat, text)
    if not m:
        return []
    rest = text[m.end():]
    end = len(rest)
    for p in end_pats:
        e = re.search(p, rest)
        if e:
            end = min(end, e.start())
    return rest[:end].splitlines()


def read_fcht_log(path, freq_block=0):
    """解析 FCHT 输出：频率、黄昆因子、λ_i、Duschinsky 矩阵 J、位移矢量 K。
    freq_block: 与黄昆因子配对的频率组（默认第一组，即本任务读入的末态频率，与 gau_reorg.py 一致）。"""
    text = Path(path).read_text(errors="ignore")
    hr = [(int(i), _f(v)) for i, v in re.findall(r"Mode num\.\s*(\d+)\s*-\s*Factor:\s*([-+\d.DdEe]+)", text)]
    if not hr:
        raise ValueError(f"{path} 中没有找到 Huang-Rhys 因子（需要 print=(huangrhys,matrix=JK)）")
    hr_dict = dict(hr)                       # 若打印多次取最后一次
    nmode = max(hr_dict)
    blocks = _freq_blocks(text)
    dof = re.search(r"Deg\. of freedom\s+(\d+)", text)
    nfreq = int(dof.group(1)) if dof else nmode
    if not blocks:
        raise ValueError(f"{path} 中没有找到频率")
    freqs = blocks[min(freq_block, len(blocks) - 1)][:nfreq]
    warnings = []
    if len(freqs) != nmode:
        warnings.append(f"频率数 {len(freqs)} 与黄昆因子数 {nmode} 不同（是否使用了 RedDim？），按顺序配对")
    modes = []
    for i in range(1, nmode + 1):
        s = hr_dict.get(i, 0.0)
        nu = freqs[i - 1] if i - 1 < len(freqs) else float("nan")
        modes.append({"mode": i, "freq_cm-1": nu, "S": s, "lambda_cm-1": s * nu,
                      "lambda_meV": s * nu * CM2EV * 1000, "lambda_kJ/mol": s * nu * CM2KJ})
    jmat = _parse_matrix(_section(text, r"Duschinsky [Tt]ransformation", [r"Shift [Vv]ector", r"Huang-Rhys"]))
    kvec = None
    sec = _section(text, r"Shift [Vv]ector", [r"Huang-Rhys", r"\n\s*\n\s*\n", r"={5,}"])
    vals = [(int(t[0]), _f(t[1])) for t in (line.split() for line in sec) if len(t) == 2 and t[0].isdigit()]
    if vals:
        kvec = np.array([v for _, v in sorted(vals)])
    return {
        "modes": modes,
        "lambda_meV": float(sum(m["lambda_meV"] for m in modes if m["freq_cm-1"] == m["freq_cm-1"])),
        "lambda_kJ/mol": float(sum(m["lambda_kJ/mol"] for m in modes if m["freq_cm-1"] == m["freq_cm-1"])),
        "duschinsky": jmat, "shift": kvec, "freq_blocks": blocks, "warnings": warnings,
    }


# ----------------------------------------------------------------------------------------------
# 导出 (CSV 与 Origin 可直接读取的格式)
# ----------------------------------------------------------------------------------------------
def export_fcht(res, outdir, prefix):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    p = outdir / f"{prefix}_黄昆因子.csv"           # Frequency,HRF 无表头，可直接给原画图脚本用
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows([(m["freq_cm-1"], m["S"]) for m in res["modes"]])
    files.append(p)
    p = outdir / f"{prefix}_重组能分解.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["No.", "Frequency/cm-1", "Huang-Rhys factor", "lambda_i/kJ*mol-1", "lambda_i/eV"])
        for m in res["modes"]:
            w.writerow([m["mode"], m["freq_cm-1"], m["S"], m["lambda_kJ/mol"], m["lambda_meV"] / 1000])
        w.writerow(["Total", "", "", res["lambda_kJ/mol"], res["lambda_meV"] / 1000])
    files.append(p)
    p = outdir / f"{prefix}.hr.txt"                  # Origin 棒状图格式
    lines = ["0\t0"]
    for m in res["modes"]:
        lines += [f"{m['freq_cm-1']}\t0", f"{m['freq_cm-1']}\t{m['S']}", f"{m['freq_cm-1']}\t0"]
    lines.append("4000\t0")
    p.write_text("\n".join(lines) + "\n")
    files.append(p)
    if res.get("duschinsky") is not None:
        p = outdir / f"{prefix}_矩阵.csv"
        np.savetxt(p, res["duschinsky"], delimiter=",", fmt="%.6e")
        files.append(p)
        p = outdir / f"{prefix}.mat.txt"
        np.savetxt(p, res["duschinsky"], delimiter="\t", fmt="%.6e")
        files.append(p)
    return files


# ----------------------------------------------------------------------------------------------
# FCHT 任务诊断 (FileIO operation on non-existent file 等) 与黄昆因子结果导入
# ----------------------------------------------------------------------------------------------
def fcht_chk_files(gjf):
    """FCHT 输入中用到的 chk: %oldchk 与 FCHT 节中列出的初态 / 末态 chk。"""
    text = Path(gjf).read_text(errors="ignore")
    names = re.findall(r"^%oldchk=(\S+)", text, re.M | re.I)
    names += [ln.strip() for ln in text.splitlines() if ln.strip().lower().endswith(".chk") and "=" not in ln]
    return list(dict.fromkeys(names))


def diagnose_fcht(path, produced=()):
    """检查 FCHT 任务 (gjf 或失败的 log) 所需的 chk。produced: 将由同批其它任务生成的 chk (不检查)。
    返回问题列表 (空表示未发现问题)。"""
    path = Path(path)
    gjf = path if path.suffix.lower() in (".gjf", ".com") else path.with_suffix(".gjf")
    probs = []
    if not gjf.exists():
        return [f"找不到输入文件 {gjf.name}，无法检查所用的 chk"]
    for name in fcht_chk_files(gjf):
        if Path(name).name in {Path(p).name for p in produced}:
            continue
        chk = (gjf.parent / name) if not Path(name).is_absolute() else Path(name)
        if not chk.exists():
            probs.append(f"{name} 不在任务目录 {gjf.parent} 中 (FCHT 必须与两个态的 chk 放在同一目录)")
            continue
        if chk.stat().st_size < 10000:
            probs.append(f"{name} 只有 {chk.stat().st_size} 字节，可能已损坏或未写完 (磁盘空间不足?)")
        log = next((chk.with_suffix(s) for s in (".log", ".out") if chk.with_suffix(s).exists()), None)
        if log is None:
            probs.append(f"找不到生成 {name} 的计算输出 ({chk.stem}.log)，无法确认其中有频率数据")
            continue
        t = log.read_text(errors="ignore")
        if "Harmonic frequencies" not in t:
            probs.append(f"{log.name} 中没有频率计算结果：{name} 必须来自 opt freq (或 freq) 任务")
        if "Normal termination" not in t[-3000:]:
            probs.append(f"{log.name} 没有正常结束，{name} 中的数据可能不完整")
        if chk.stat().st_mtime > log.stat().st_mtime + 120:
            probs.append(f"{name} 在 {log.name} 结束后又被改写过 (可能被其它任务覆盖，频率数据已丢失)")
        if re.search(r"NImag=([1-9])", t.replace("\n ", "")):
            probs.append(f"{log.name} 有虚频，不是能量极小点，FCHT 结果不可靠")
    return probs


def explain_fcht_failure(log):
    """失败的 FCHT 输出：给出原因与建议。"""
    text = Path(log).read_text(errors="ignore")
    if "FileIO operation on non-existent file" not in text and "Error termination" not in text:
        return None
    msg = [f"{Path(log).name} 计算失败"]
    if "FileIO operation on non-existent file" in text:
        msg.append("Gaussian 读取 chk 时找不到需要的数据 (FileIO operation on non-existent file)。")
    probs = diagnose_fcht(log)
    msg += (["发现的问题:"] + [f"  · {p}" for p in probs]) if probs else \
        ["两个 chk 文件看起来正常；请确认它们由同一版本的 Gaussian (如都为 G16) 生成，且计算时磁盘空间充足。"]
    msg.append("也可改用“内置”分解：直接读取两个态的 fchk (含 Hessian)，不需要 FCHT 任务。")
    return "\n".join(msg)


_FREQ_KEYS = ("freq", "frequency", "omega", "ω", "wavenumber", "cm-1", "cm^-1", "ν", "nu", "频率")
_HR_KEYS = ("hr", "huang", "huang-rhys", "hrf", "s", "factor", "黄昆")


def _modes(freqs, s):
    modes = []
    for i, (nu, si) in enumerate(zip(freqs, s), 1):
        modes.append({"mode": i, "freq_cm-1": float(nu), "S": float(si), "lambda_cm-1": si * nu,
                      "lambda_meV": si * nu * CM2EV * 1000, "lambda_kJ/mol": si * nu * CM2KJ})
    return {"modes": modes, "lambda_meV": float(sum(m["lambda_meV"] for m in modes)),
            "lambda_kJ/mol": float(sum(m["lambda_kJ/mol"] for m in modes)), "duschinsky": None, "shift": None,
            "warnings": []}


def read_hr_table(path):
    """读取黄昆因子结果：Gaussian FCHT 输出，或 CSV / TXT / DAT 表格 (频率 cm⁻¹ 与黄昆因子 S 两列，
    可带表头与序号列，如 MOMAP、DUSHIN 或本程序导出的文件)。返回与 read_fcht_log 相同的结构。"""
    path = Path(path)
    text = path.read_text(errors="ignore")
    if "Mode num." in text and "Factor:" in text:
        return read_fcht_log(path)
    if "Error termination" in text:
        raise ValueError(explain_fcht_failure(path) or f"{path.name} 计算失败")
    rows, header = [], None
    for line in text.splitlines():
        toks = [t for t in re.split(r"[,\s;\t]+", line.strip().lstrip("#!%;").strip()) if t]
        if not toks:
            continue
        nums = []
        for t in toks:
            try:
                nums.append(float(t.replace("D", "E")))
            except ValueError:
                nums = None
                break
        if nums and len(nums) >= 2:
            rows.append(nums)
        elif not rows:
            header = [t.lower() for t in toks]
    if not rows:
        raise ValueError(f"{path.name} 中没有数值表格")
    ncol = min(len(r) for r in rows)
    data = np.array([r[:ncol] for r in rows])
    fcol = scol = None
    if header and len(header) >= ncol:
        for j, h in enumerate(header[:ncol]):
            if fcol is None and any(k in h for k in _FREQ_KEYS):
                fcol = j
            elif scol is None and (h in _HR_KEYS or any(k in h for k in _HR_KEYS[:4]) or "黄昆" in h):
                scol = j
    if fcol is None or scol is None:          # 按数值特征判断：跳过序号列；频率 10–5000，S ≥ 0 且多数 < 10
        idx = [j for j in range(ncol) if np.allclose(data[:, j], np.arange(1, len(data) + 1))
               or np.allclose(data[:, j], np.arange(len(data)))]
        cand = [j for j in range(ncol) if j not in idx]
        fcol = fcol if fcol is not None else next(
            (j for j in cand if np.median(data[:, j]) > 20 and data[:, j].max() < 6000 and data[:, j].min() > 0), None)
        scol = scol if scol is not None else next(
            (j for j in cand if j != fcol and data[:, j].min() >= 0 and np.median(data[:, j]) < 10), None)
    if fcol is None or scol is None:
        raise ValueError(f"无法在 {path.name} 中识别频率列与黄昆因子列：请使用“频率,黄昆因子”两列格式")
    return _modes(data[:, fcol], data[:, scol])
