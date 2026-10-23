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
text = sys.stdin.read()
steps = re.split(r"^--Link1--\s*$", text, flags=re.M)
for s in steps:
    m = re.search(r"^%chk=(\S+)", s, re.M | re.I)
    if m:
        open(m.group(1), "w").write("fake checkpoint\n")
    print(" Entering Gaussian System, Link 0=g16")
    print(" (fake Gaussian for tests: no calculation performed)")
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
