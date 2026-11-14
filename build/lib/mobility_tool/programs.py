"""外部程序 (Gaussian / ADF-AMS / xtb) 的安装位置检测、环境设置与调用。

ADF 的 .sh 脚本用 AMS 自带的 bash (或系统 bash / sh) 执行，并设置 AMSHOME、AMSBIN、AMSRESOURCES、
SCMLICENSE、SCM_TMPDIR 等环境变量 (Windows 不能直接运行 .sh)；xtb 用于 DIPRO 转移积分。
"""
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

IS_WIN = sys.platform.startswith("win")
_AMS_GLOBS = [r"C:\AMS*", r"D:\AMS*", r"E:\AMS*", r"C:\Program Files\AMS*", r"C:\ADF*", r"D:\ADF*",
              "/opt/ams*", "/opt/adf*", "/usr/local/ams*", str(Path.home() / "ams*"), str(Path.home() / "adf*"),
              str(Path.home() / "AMS*")]
_XTB_GLOBS = [r"C:\xtb*\bin\xtb.exe", r"D:\xtb*\bin\xtb.exe", r"C:\Program Files\xtb*\bin\xtb.exe",
              str(Path.home() / "xtb*" / "bin" / "xtb"), "/opt/xtb*/bin/xtb", str(Path.home() / ".local/bin/xtb")]


# ----------------------------------------------------------------------------------------------
# AMS / ADF
# ----------------------------------------------------------------------------------------------
_EXE_AMS = ("ams", "ams.exe", "ams.bat")
_EXE_ADF = ("adf", "adf.exe", "adf.bat")


def _ams_home_ok(p):
    p = Path(p)
    return p.is_dir() and (p / "bin").is_dir() and any((p / "bin" / n).exists() for n in _EXE_AMS + _EXE_ADF)


def _version(home):
    m = re.search(r"(20\d\d)(?:\.(\d+))?", Path(home).name)
    if m:
        return m.group(1) + (f".{m.group(2)}" if m.group(2) else "")
    for f in ("Install/version", "VERSION", "version.txt"):
        q = Path(home) / f
        if q.exists():
            return q.read_text(errors="ignore").strip().splitlines()[0][:20]
    return "?"


def find_ams(user_path=None):
    """返回 {'home', 'bin', 'shell', 'kind'('ams'|'classic')} 或 None。user_path 可以是安装目录或其 bin 目录。"""
    cands = []
    if user_path:
        p = Path(user_path)
        if p.is_file():                                 # 直接选了 bin 中的 adf / ams 程序
            p = p.parent
        cands += [p, p.parent] if p.name.lower() == "bin" else [p]
        if p.is_dir() and not _ams_home_ok(p):          # 选了上一级目录时，向下最多找三层 (限 400 个目录)
            level, seen = [p], 0
            for _ in range(3):
                nxt = []
                for q in level:
                    try:
                        subs = sorted((r for r in q.iterdir() if r.is_dir() and not r.name.startswith(".")),
                                      reverse=True)
                    except OSError:
                        continue
                    for r in subs:
                        seen += 1
                        if seen > 400:
                            break
                        if _ams_home_ok(r):
                            cands.append(r)
                        nxt.append(r)
                level = nxt
    for env in ("AMSHOME", "ADFHOME"):
        if os.environ.get(env):
            cands.append(Path(os.environ[env]))
    for g in _AMS_GLOBS:
        cands += [Path(x) for x in sorted(glob.glob(g), reverse=True)]
    for c in cands:
        if _ams_home_ok(c):
            b = c / "bin"
            kind = "ams" if any((b / n).exists() for n in _EXE_AMS) else "classic"
            return {"home": str(c), "bin": str(b), "shell": find_shell(c), "kind": kind, "version": _version(c)}
    return None


def explain_ams(path):
    """说明为什么没有识别出 ADF / AMS 安装 (给出检查过的位置)。"""
    if not path:
        return "没有找到 ADF/AMS：请点“浏览”选择安装目录 (如 C:\\ADF2016.101)，或点“选择 adf 程序文件…”"
    p = Path(path)
    if not p.exists():
        return f"路径不存在: {p}"
    if p.is_file():
        p = p.parent
    home = p.parent if p.name.lower() == "bin" else p
    b = home / "bin"
    if not b.is_dir():
        return (f"所选目录 {home} 中没有 bin 文件夹。请选择 ADF 的安装目录 (其中应有 bin\\adf 或 bin\\ams)，"
                "或点“选择 adf 程序文件…”直接选中 bin 中的 adf 程序")
    names = ", ".join(sorted(x.name for x in b.iterdir())[:12])
    return f"{b} 中没有 adf / ams 程序 (已有: {names} …)。请确认这是 ADF 的安装目录"


def find_shell(ams_home=None):
    """执行 .sh 脚本用的 shell：优先 AMS 自带的 bash，其次 Git-Bash / 系统 sh。"""
    cands = []
    if ams_home:
        h = Path(ams_home)
        for sub in ("bin", "msys/usr/bin", "msys/bin", "bin/msys/usr/bin", "bin/msys/bin", "msys64/usr/bin",
                    "msys64/bin", "Tools/msys/bin", "tools/bin"):
            cands += [h / sub / n for n in ("bash.exe", "sh.exe", "bash", "sh")]
        if IS_WIN:                       # 在安装目录内再搜索一层 (不同版本位置不同)
            for pat in ("*/sh.exe", "*/*/sh.exe", "*/*/*/sh.exe", "*/bash.exe", "*/*/bash.exe", "*/*/*/bash.exe"):
                cands += sorted(h.glob(pat))
    cands += [Path(r"C:\Program Files\Git\bin\bash.exe"), Path(r"C:\Program Files\Git\usr\bin\bash.exe")]
    for n in ("bash", "sh"):
        w = shutil.which(n)
        if w:
            cands.append(Path(w))
    for c in cands:
        if c.is_file() and not _is_wsl_stub(c):
            return str(c)
    return None


def _is_wsl_stub(p):
    """Windows 的 System32\\bash.exe / WindowsApps\\bash.exe 是 WSL 启动器，未安装发行版时只会提示安装 WSL。"""
    if not IS_WIN:
        return False
    s = str(p).lower().replace("/", "\\")
    return "\\system32\\" in s or "\\sysnative\\" in s or "\\windowsapps\\" in s


def ams_env(info, scratch=None):
    env = dict(os.environ)
    home = Path(info["home"])
    fwd = (lambda p: Path(p).as_posix()) if IS_WIN else str
    env["AMSHOME"] = env["ADFHOME"] = fwd(home)
    env["AMSBIN"] = env["ADFBIN"] = fwd(home / "bin")
    env["AMSRESOURCES"] = env["ADFRESOURCES"] = fwd(home / "atomicdata")
    for lic in (home / "license.txt", Path(os.environ.get("APPDATA", "")) / "SCM" / "license.txt",
                Path.home() / ".scm" / "license.txt"):
        if lic.is_file():
            env["SCMLICENSE"] = fwd(lic)
            break
    if scratch:
        Path(scratch).mkdir(parents=True, exist_ok=True)
        env["SCM_TMPDIR"] = fwd(scratch)
    env["PATH"] = str(home / "bin") + os.pathsep + env.get("PATH", "")
    if info.get("nscm"):
        env["NSCM"] = str(info["nscm"])                  # ADF / AMS 并行进程数
    return env


def adf_result_files(script):
    """脚本末尾 “# 结果文件:” 列出的全部结果文件 (批量脚本有多个)。"""
    return [Path(script).parent / m for m in re.findall(r"#\s*结果文件:\s*(\S+)", Path(script).read_text())]


def adf_result_file(script):
    r = adf_result_files(script)
    return r[-1] if r else None


def _run_shell(info, script, cwd, scratch=None, stop_event=None, log=None):
    import time
    flags = 0x08000000 if IS_WIN else 0
    fh = open(log, "wb") if log else subprocess.DEVNULL
    try:
        proc = subprocess.Popen([info["shell"], Path(script).name], cwd=cwd, env=ams_env(info, scratch),
                                stdout=fh, stderr=subprocess.STDOUT, creationflags=flags)
        while proc.poll() is None:
            if stop_event is not None and stop_event.is_set():
                proc.terminate()
                return None
            time.sleep(0.3)
        return proc.returncode
    finally:
        if log:
            fh.close()


def run_adf_script(info, script, scratch=None, stop_event=None):
    """用 shell 执行 run_*_adf.sh，输出写入同名 .log；成功条件：生成了结果文件 (adf.rkf / .t21)。
    若脚本格式 (AMS / 旧版 ADF) 与安装版本不符，会按记录的参数自动重新生成。"""
    from .adf import regenerate_adf_job, script_style
    script = Path(script).resolve()
    if not info.get("shell"):
        return False, None, ("找不到可执行 .sh 脚本的 bash/sh。可改为在 ADFjobs 中打开同名 .run 文件运行，"
                             "并在本窗口点“监视 ADFjobs 结果”，算完自动读取；或安装 Git for Windows")
    note = ""
    if script_style(script) != info["kind"]:
        r = regenerate_adf_job(script, info["kind"])
        if r is None:
            return False, None, (f"脚本为 {'AMS2020+' if script_style(script) == 'ams' else 'ADF2019 及以前'} 格式，"
                                 f"与安装的 ADF {info.get('version', '')} 不符；请在转移积分页选择正确版本后重新生成")
        note = f"已按 ADF {info.get('version', '')} 的格式自动重新生成脚本; "
    log = script.with_suffix(".log")
    rc = _run_shell(info, script, script.parent, scratch, stop_event, log)
    if rc is None:
        return False, log, "已手动停止"
    res = adf_result_files(script)
    done = [r for r in res if r.exists()]
    if res and len(done) == len(res):
        return True, log, note + (f"结果: {res[0].name}" if len(res) == 1 else f"全部 {len(res)} 个二聚体完成")
    tail = "\n".join(log.read_text(errors="ignore").splitlines()[-10:])
    return False, log, note + f"完成 {len(done)} / {len(res)} 个结果文件，返回码 {rc}\n{tail}"


def dump_kf(info, kf_path, section):
    """调用安装目录中的 dmpkf 导出 KF 文件 (adf.rkf / TAPE21) 的某一节为文本。"""
    import tempfile
    kf_path = Path(kf_path).resolve()
    with tempfile.TemporaryDirectory() as d:
        sh = Path(d) / "dump.sh"
        var = "AMSBIN" if info["kind"] == "ams" else "ADFBIN"
        sh.write_bytes((f'"${var}/dmpkf" "{kf_path.as_posix()}" {section}\n'
                        f'"${var}/dmpkf" "{kf_path.as_posix()}"\n').encode())
        log = Path(d) / "dump.txt"
        _run_shell(info, sh, d, log=log)
        return log.read_text(errors="ignore")


def test_adf_environment(info, workdir):
    """用 H2 单点 (SZ 基组) 检测 ADF 能否运行；返回 (是否成功, 日志末尾, 秒)。"""
    import time
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    if info["kind"] == "ams":
        body = ['AMS_JOBNAME=h2test "$AMSBIN/ams" --delete-old-results <<eor', "Task SinglePoint", "System",
                "  Atoms", "    H 0 0 0", "    H 0 0 0.74", "  End", "End", "Engine ADF", "  Basis", "    Type SZ",
                "  End", "EndEngine", "eor", "[ -f h2test.results/adf.rkf ] && echo ADF_TEST_OK"]
    else:
        body = ["rm -f TAPE21", '"$ADFBIN/adf" <<eor', "Atoms", "  H 0 0 0", "  H 0 0 0.74", "End", "Basis",
                " Type SZ", " Core None", "End", "End Input", "eor", "[ -f TAPE21 ] && echo ADF_TEST_OK",
                "rm -f TAPE21 TAPE13 logfile"]
    sh = wd / "adf_env_test.sh"
    sh.write_bytes(("#!/bin/sh\n" + "\n".join(body) + "\n").encode())
    # 同一测试的 ADFjobs 版本：可在 ADFjobs 中打开运行，与本程序的结果对照
    (wd / "adf_env_test.run").write_bytes(("#! /bin/sh\n" + "\n".join(body) + "\n").encode("gbk", errors="replace"))
    env = ams_env(info, wd / "scratch")
    b = Path(info["bin"])
    diag = [f"shell: {info.get('shell') or '未找到'}",
            f"{'AMSBIN' if info['kind'] == 'ams' else 'ADFBIN'}: {env['AMSBIN']}",
            "程序: " + ", ".join(n for n in ("adf", "adf.exe", "adf.bat", "ams", "ams.exe", "dmpkf", "dmpkf.exe")
                               if (b / n).exists()),
            f"许可证: {env.get('SCMLICENSE', '未找到 license.txt (若 ADF 报许可证错误，请把 license.txt 放到安装目录)')}",
            f"临时目录: {env.get('SCM_TMPDIR')}"]
    if info.get("shell") and "Git" in info["shell"]:
        diag.append("注意: 使用的是 Git-Bash，而不是 ADF 自带的 shell；若失败，请改在 ADFjobs 中运行 .run 文件")
    log = wd / "adf_env_test.log"
    t0 = time.time()
    if not info.get("shell"):
        return False, "\n".join(diag + ["找不到 bash/sh：请在 ADFjobs 中运行 " + str(wd / "adf_env_test.run")]), 0.0
    rc = _run_shell(info, sh, wd, log=log)
    text = log.read_text(errors="ignore")
    ok = "ADF_TEST_OK" in text
    out = diag + [f"返回码: {rc}", "---- ADF 输出 (末尾) ----"] + text.splitlines()[-25:]
    if not ok:
        out.append(f"---- 对照: 请在 ADFjobs 中打开 {wd / 'adf_env_test.run'} 运行；能成功则问题在本程序的 shell / 环境变量")
    return ok, "\n".join(out), time.time() - t0


# ----------------------------------------------------------------------------------------------
# xtb (DIPRO)
# ----------------------------------------------------------------------------------------------
def find_xtb(user_path=None):
    cands = []
    if user_path:
        p = Path(user_path)
        cands += [p] if p.is_file() else [p / "xtb.exe", p / "xtb", p / "bin" / "xtb.exe", p / "bin" / "xtb"]
    w = shutil.which("xtb")
    if w:
        cands.append(Path(w))
    for g in _XTB_GLOBS:
        cands += [Path(x) for x in sorted(glob.glob(g), reverse=True)]
    return next((str(c) for c in cands if c.is_file()), None)


def parse_dipro(text):
    """解析 xtb --dipro 输出：各轨道对的 |J(AB,eff)| 与总耦合 (eV)。"""
    pairs = {}
    for m in re.finditer(r"Fragment 1 (\S+)\s+Fragment 2 (\S+)\s*\n.*?\|J\(AB,eff\)\|:\s*(-?[\d.]+)\s*eV", text, re.S):
        pairs[(m.group(1), m.group(2))] = float(m.group(3))
    tot = {}
    for key, pat in (("hole", r"total \|J\(AB,eff\)\| for hole transport.*?:\s*([\d.]+)\s*eV"),
                     ("electron", r"total \|J\(AB,eff\)\| for charge transport \(unocc.*?:\s*([\d.]+)\s*eV"),
                     ("ct", r"total \|J\(AB,eff\)\| for charge transfer.*?:\s*([\d.]+)\s*eV")):
        m = re.search(pat, text)
        if m:
            tot[key] = float(m.group(1))
    if not pairs and not tot:
        raise ValueError("xtb 输出中没有 DIPRO 结果（需要 xtb ≥ 6.7，且两个片段识别正确）")
    return {"pairs": pairs, "total": tot, "normal": "normal termination of dipro" in text}


def xtb_dipro(atoms_a, atoms_b, xtb_exe, workdir, gfn=1, charge=0, threshold=0.1, timeout=1800):
    """用 xtb 计算二聚体 DIPRO 转移积分；返回 meV (HOMO-HOMO / LUMO-LUMO 及窗口内总耦合)。"""
    from .gaussian_io import write_xyz
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    na, nt = len(atoms_a), len(atoms_a) + len(atoms_b)
    write_xyz(wd / "dimer.xyz", list(atoms_a) + list(atoms_b), "dimer for xtb DIPRO")
    (wd / "xcontrol").write_text(f"$split\nfragment1: 1-{na}\nfragment2: {na + 1}-{nt}\n$end\n")
    (wd / ".UHFfrag").write_text("0 0\n")
    cmd = [xtb_exe, "dimer.xyz", "--dipro", str(threshold), "--gfn", str(gfn), "--chrg", str(charge),
           "-I", "xcontrol"]
    flags = 0x08000000 if IS_WIN else 0
    r = subprocess.run(cmd, cwd=wd, capture_output=True, timeout=timeout, creationflags=flags)
    text = r.stdout.decode(errors="ignore")
    (wd / "xtb_dipro.out").write_text(text)
    res = parse_dipro(text)
    hh = res["pairs"].get(("HOMO", "HOMO"))
    ll = res["pairs"].get(("LUMO", "LUMO"))
    return {"hole": hh * 1000 if hh is not None else res["total"].get("hole", 0) * 1000,
            "electron": ll * 1000 if ll is not None else res["total"].get("electron", 0) * 1000,
            "total_hole": res["total"].get("hole", float("nan")) * 1000,
            "total_electron": res["total"].get("electron", float("nan")) * 1000,
            "normal": res["normal"], "output": str(wd / "xtb_dipro.out")}
