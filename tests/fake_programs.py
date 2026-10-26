"""测试用的假 Gaussian (g16 / formchk) 与假 ADF (AMS2020+ 的 ams，ADF2016 的 adf) 可执行文件。
smoke_gui.py 开始时调用 install()，在 /tmp 下生成：
  /tmp/fakeg/bin/{g16,formchk}       g16 从标准输入读 gjf，每个 Link1 步骤输出 Normal termination，并生成 %chk 文件
  /tmp/fakeams/bin/{ams,adf,dmpkf}   ams 生成 $AMS_JOBNAME.results/adf.rkf
  /tmp/fake2016/ADF2016.101/bin/{adf,dmpkf}   adf 在当前目录生成 TAPE21 并输出 NORMAL TERMINATION
"""
import os
import stat
from pathlib import Path

G16 = r'''#!/usr/bin/env python3
import re, sys
# Linux 版: g16 < x.gjf > x.log；Windows 版: g16.exe x.gjf x.out (见 jobs.py)
text = open(sys.argv[1]).read() if len(sys.argv) > 2 else sys.stdin.read()
if len(sys.argv) > 2:
    sys.stdout = open(sys.argv[2], "w")
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}
steps = re.split(r"^--Link1--\s*$", text, flags=re.M)
geom = []
for n, s in enumerate(steps):
    m = re.search(r"^%chk=(\S+)", s, re.M | re.I)
    if m:
        open(m.group(1), "w").write("fake checkpoint\n")
    print(" Entering Gaussian System, Link 0=g16")
    print(" (fake Gaussian for tests: no calculation performed)")
    r = re.search(r"^(#[^\n]*)", s, re.M)
    if r:
        print(" " + "-" * 20)
        print(" " + r.group(1))
        print(" " + "-" * 20)
    at = re.findall(r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", s, re.M)
    geom = at or geom
    q = re.search(r"^\s*(-?\d+)\s+(\d+)\s*$", s, re.M)
    chg = int(q.group(1)) if q else 0
    if q:
        print(f" Charge = {chg} Multiplicity = {q.group(2)}")
    if geom:                     # 输出几何 (略作“优化”：坐标不变) 与假的 SCF 能量
        print("                         Standard orientation:")
        print(" " + "-" * 69)
        print(" Center     Atomic      Atomic             Coordinates (Angstroms)")
        print(" Number     Number       Type             X           Y           Z")
        print(" " + "-" * 69)
        for i, (e, x, y, z) in enumerate(geom, 1):
            print(f" {i:6d} {Z.get(e, 6):10d} {0:11d} {float(x):14.6f}{float(y):12.6f}{float(z):12.6f}")
        print(" " + "-" * 69)
    print(f" SCF Done:  E(RB3LYP) =  {-230.0 + 0.25 * abs(chg) + 0.002 * n:.8f}     A.U. after   10 cycles")
    print(" Normal termination of Gaussian 16 at Mon Jan  1 00:00:00 2024.")
'''
FORMCHK = r'''#!/usr/bin/env python3
import sys
open(sys.argv[2], "w").write("fake formatted checkpoint\n")
'''
AMS = r'''#!/bin/sh
cat > /dev/null
name="${AMS_JOBNAME:-ams}"
rm -rf "$name.results"
mkdir -p "$name.results"
echo "fake rkf" > "$name.results/adf.rkf"
echo "NORMAL TERMINATION" > "$name.results/ams.log"
echo "NORMAL TERMINATION"
'''
ADF = r'''#!/bin/sh
input=$(cat)
echo "fake TAPE21" > TAPE21
case "$input" in *TRANSFERINTEGRALS*|*TransferIntegrals*)
cat <<'TI'
 Charge transfer integrals relevant for hole or electron mobility calculations

 V for hole transfer:          0.08500 eV
 V for electron transfer:          0.04200 eV

 e1(hole) Site energy HOMO fragment 1:      -4.50000 eV
 e2(hole) Site energy HOMO fragment 2:      -4.50000 eV
 J(hole) Charge transfer integral HOMO fragment 1 - HOMO fragment 2:      0.20000 eV
 S(hole) Overlap integral HOMO fragment 1 - HOMO fragment 2:       0.02500

 e1(electron) Site energy LUMO fragment 1:      -2.80000 eV
 e2(electron) Site energy LUMO fragment 2:      -2.80000 eV
 J(electron) Charge transfer integral LUMO fragment 1 - LUMO fragment 2:      -0.02800 eV
 S(electron) Overlap integral LUMO fragment 1 - LUMO fragment 2:       -0.02500
TI
;;
esac
echo " NORMAL TERMINATION"
'''
DMPKF = '#!/bin/sh\necho "fake dmpkf"\n'


def _put(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def fake_gaussian(directory):
    """在 directory 中放一份假 g16 / formchk，返回可执行文件路径 (Windows 下为调用当前 Python 的 .bat)。"""
    import sys
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        _put(d / "g16", G16)
        _put(d / "formchk", FORMCHK)
        return d / "g16"
    for n, t in (("g16", G16), ("formchk", FORMCHK)):
        (d / f"{n}.py").write_text(t)
        (d / f"{n}.bat").write_text(f'@"{sys.executable}" "%~dp0{n}.py" %*\r\n')
    return d / "g16.bat"


def install(root="/tmp"):
    r = Path(root)
    _put(r / "fakeg/bin/g16", G16)
    _put(r / "fakeg/bin/formchk", FORMCHK)
    for n, t in (("ams", AMS), ("adf", ADF), ("dmpkf", DMPKF)):
        _put(r / "fakeams/bin" / n, t)
    for n, t in (("adf", ADF), ("dmpkf", DMPKF)):
        _put(r / "fake2016/ADF2016.101/bin" / n, t)
    return r


if __name__ == "__main__":
    print(install())
