# -*- coding: utf-8 -*-
"""给已生成的 ADF 批处理 (.run) 补上片段映射容差 (v1.20.4 之前生成的输入)。

用法: python -m mobility_tool.fix_adf_tol <目录或 .run 文件> [--tol 0.1]
递归处理目录下所有 .run，原文件备份为 .run.bak。已完成的任务 (有 .done 或结果) 照常被批处理跳过。
"""
import argparse
import re
import shutil
from pathlib import Path

PAT = re.compile(r"^([ \t]*SYMMETRY[ \t]+NOSYM)[ \t]*(?:tol[ \t]*=[ \t]*[^\s]+)?[ \t]*(\r?)$", re.I | re.M)


def fix_file(p, tol):
    with open(p, encoding="utf-8", errors="replace", newline="") as fh:
        t = fh.read()
    new, n = PAT.subn(lambda m: f"{m.group(1)} tol={tol}{m.group(2)}", t)
    if n and new != t:
        shutil.copyfile(p, p.with_suffix(p.suffix + ".bak"))
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--tol", default="0.1")
    a = ap.parse_args(argv)
    P = Path(a.path)
    files = [P] if P.is_file() else sorted(P.rglob("*.run"))
    tot = 0
    for f in files:
        n = fix_file(f, a.tol)
        if n:
            print(f"{f}: {n} 处")
        tot += n
    print(f"共修改 {tot} 处 SYMMETRY 行，{len(files)} 个文件")


if __name__ == "__main__":
    main()
