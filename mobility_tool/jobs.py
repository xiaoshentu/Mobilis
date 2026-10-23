"""在本机直接运行高斯任务（Windows / Linux），并生成更稳妥的批处理脚本。

Windows 版高斯常见的“双击 bat 没反应/闪退”原因：
  1. g16 不在 PATH 中，或没有设置 GAUSS_EXEDIR；
  2. 没有设置临时目录 GAUSS_SCRDIR（或目录不存在）；
  3. 路径里有中文或空格（例如“桌面\\计算迁移率软件”），高斯无法打开文件；
  4. bat 不是从输入文件所在目录启动（双击时当前目录可能不对）。
本模块逐项处理这些问题。
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

IS_WIN = sys.platform.startswith("win")
COMMON_DIRS = [r"C:\G16W", r"D:\G16W", r"E:\G16W", r"C:\G09W", r"D:\G09W", r"C:\Program Files\G16W",
               r"C:\Program Files (x86)\G16W", "/opt/g16", "/usr/local/g16", str(Path.home() / "g16")]


def find_gaussian(user_path=None):
    """返回 {'exe': 高斯可执行文件, 'formchk': formchk, 'exedir': 目录} 或 None。"""
    cands = []
    if user_path:
        p = Path(user_path)
        cands.append(p if p.is_file() else None)
        if p.is_dir():
            cands += [p / n for n in ("g16.exe", "g09.exe", "g16", "g09")]
    for env in ("GAUSS_EXEDIR",):
        if os.environ.get(env):
            d = Path(os.environ[env].split(os.pathsep)[0])
            cands += [d / n for n in ("g16.exe", "g09.exe", "g16", "g09")]
    for n in ("g16", "g09"):
        w = shutil.which(n)
        if w:
            cands.append(Path(w))
    for d in COMMON_DIRS:
        cands += [Path(d) / n for n in ("g16.exe", "g09.exe", "g16", "g09")]
    for c in cands:
        if c and c.is_file():
            exedir = c.parent
            fc = next((exedir / n for n in ("formchk.exe", "formchk") if (exedir / n).is_file()), None)
            fc = fc or shutil.which("formchk")
            return {"exe": str(c), "formchk": str(fc) if fc else None, "exedir": str(exedir)}
    return None


def path_problems(path):
    """检查路径中会让 Windows 版高斯出错的字符。"""
    s = str(Path(path).resolve())
    probs = []
    if any(ord(ch) > 127 for ch in s):
        probs.append(f"路径含中文或其它非英文字符: {s}")
    if " " in s and IS_WIN:
        probs.append(f"路径含空格: {s}")
    return probs


def gjf_chks(gjf):
    """gjf 中各 Link1 步骤写出的 %chk 文件（取第一个，即优化/主计算的 chk）。"""
    return re.findall(r"^%chk=(\S+)", Path(gjf).read_text(errors="ignore"), re.M | re.I)


def order_jobs(gjfs):
    """FCHT 任务依赖重组能任务的 chk，排到最后。"""
    return sorted(gjfs, key=lambda p: (1 if "_fcht_" in Path(p).name else 0, Path(p).name))


def output_name(gjf):
    return Path(gjf).with_suffix(".out" if IS_WIN else ".log")


def build_env(info, scratch):
    env = dict(os.environ)
    env["GAUSS_EXEDIR"] = info["exedir"] + (os.pathsep + env["GAUSS_EXEDIR"] if env.get("GAUSS_EXEDIR") else "")
    env["GAUSS_SCRDIR"] = str(scratch)
    env["PATH"] = info["exedir"] + os.pathsep + env.get("PATH", "")
    return env


def run_one(info, gjf, scratch, on_line=None, stop_event=None):
    """运行一个 gjf（阻塞），返回 (是否成功, 输出文件, 信息)。"""
    gjf = Path(gjf).resolve()
    cwd = gjf.parent
    out = output_name(gjf)
    Path(scratch).mkdir(parents=True, exist_ok=True)
    env = build_env(info, scratch)
    flags = 0x08000000 if IS_WIN else 0     # CREATE_NO_WINDOW
    if IS_WIN:
        proc = subprocess.Popen([info["exe"], gjf.name, out.name], cwd=cwd, env=env, creationflags=flags,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        fin, fout = open(gjf, "rb"), open(out, "wb")
        proc = subprocess.Popen([info["exe"]], cwd=cwd, env=env, stdin=fin, stdout=fout, stderr=subprocess.PIPE)
    while proc.poll() is None:
        if stop_event is not None and stop_event.is_set():
            proc.terminate()
            return False, out, "已手动停止"
        time.sleep(0.5)
    if not IS_WIN:
        fin.close()
        fout.close()
    text = out.read_text(errors="ignore") if out.exists() else ""
    n_link = len(re.findall(r"^--Link1--", gjf.read_text(errors="ignore"), re.M)) + 1
    n_ok = text.count("Normal termination")
    if "Error termination" in text or n_ok < n_link:
        tail = "\n".join(text.splitlines()[-6:])
        err = proc.stdout.read().decode(errors="ignore")[-400:] if IS_WIN and proc.stdout else ""
        if not IS_WIN and proc.stderr:
            err = proc.stderr.read().decode(errors="ignore")[-400:]
        from .gaussian_io import explain_gaussian_error
        why = explain_gaussian_error(text) or ""
        return False, out, (f"未正常结束 ({n_ok}/{n_link} 步)\n{why}" if why else
                            f"未正常结束 ({n_ok}/{n_link} 步)\n{tail}\n{err}").strip()
    msgs = []
    if info.get("formchk"):
        for chk in dict.fromkeys(gjf_chks(gjf)[:1]):
            c = cwd / chk
            if c.exists():
                fch = c.with_suffix(".fchk")
                r = subprocess.run([info["formchk"], c.name, fch.name], cwd=cwd, env=env, creationflags=flags,
                                   capture_output=True)
                msgs.append(f"formchk {c.name} → {fch.name}" + ("" if fch.exists() else " 失败"))
    if on_line:
        on_line("\n".join(msgs))
    return True, out, "; ".join(msgs) or "完成"


class JobQueue:
    """后台线程中依次运行一组任务 (.gjf → Gaussian, run_*_adf.sh → ADF/AMS)；通过回调报告进度。"""

    def __init__(self, info, gjfs, scratch, on_status, on_done=None, ams_info=None):
        self.info, self.gjfs, self.scratch = info, order_jobs(gjfs), scratch
        self.ams_info = ams_info
        self.on_status, self.on_done = on_status, on_done
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _run(self):
        results = []
        for g in self.gjfs:
            if self.stop_event.is_set():
                self.on_status(g, "已取消", "")
                continue
            self.on_status(g, "运行中", "")
            t0 = time.time()
            try:
                if str(g).endswith(".sh"):
                    if not self.ams_info:
                        raise RuntimeError("未设置 ADF/AMS 安装位置（工具 → 程序路径设置）")
                    from .programs import run_adf_script
                    ok, out, msg = run_adf_script(self.ams_info, g, self.scratch, self.stop_event)
                else:
                    if not self.info:
                        raise RuntimeError("未找到高斯程序（工具 → 程序路径设置）")
                    ok, out, msg = run_one(self.info, g, self.scratch, stop_event=self.stop_event)
            except Exception as exc:  # noqa: BLE001
                ok, out, msg = False, None, str(exc)
            self.on_status(g, f"完成 {time.time() - t0:.0f}s" if ok else "失败", msg)
            results.append((g, ok, msg))
        if self.on_done:
            self.on_done(results)


def write_windows_bat(outdir, stems, fchk_stems=None, name="run.bat"):
    """生成更稳妥的 Windows 批处理：自动切到所在目录、设置环境变量与临时目录、检查 g16、逐个检查是否正常结束。"""
    fchk_stems = stems if fchk_stems is None else fchk_stems
    lines = [
        "@echo off", "setlocal",
        "rem ===== 如果高斯不在下面的目录，请修改 G16DIR =====",
        r"if not defined G16DIR set G16DIR=C:\G16W",
        "cd /d %~dp0",
        "set GAUSS_EXEDIR=%G16DIR%",
        "set GAUSS_SCRDIR=%~dp0scratch",
        "if not exist \"%GAUSS_SCRDIR%\" mkdir \"%GAUSS_SCRDIR%\"",
        "set PATH=%G16DIR%;%PATH%",
        "where g16 >nul 2>nul || (echo [错误] 找不到 g16.exe，请修改本文件开头的 G16DIR & pause & exit /b 1)",
        "echo 当前目录: %CD%",
        "echo 注意: 路径中如含中文或空格，Windows 版高斯可能无法运行，请把文件夹放到如 D:\\calc\\ 这样的路径。",
    ]
    for s in stems:
        lines += [f"echo ---- 运行 {s}.gjf ----", f"g16 {s}.gjf {s}.out",
                  f"findstr /c:\"Error termination\" {s}.out >nul && echo [警告] {s} 出错，请查看 {s}.out"]
        if s in fchk_stems:
            lines.append(f"if exist {s}.chk formchk {s}.chk {s}.fchk")
    lines += ["echo 全部任务结束", "pause"]
    p = Path(outdir) / name
    # 中文 Windows 的 cmd 默认 GBK 编码；UTF-8 的 bat 里有中文时会被错误解析
    p.write_bytes(("\r\n".join(lines) + "\r\n").encode("gbk", errors="replace"))
    return p
