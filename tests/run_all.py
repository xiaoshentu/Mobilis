"""依次运行 tests/ 下的全部测试脚本 (不需要 Gaussian / ADF)。用法: python tests/run_all.py [--quick] [--ioenc gbk]"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLOW = {"smoke_gui.py", "test_v18.py"}


def main():
    quick = "--quick" in sys.argv
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")       # 中文 Windows 控制台 (GBK)
    # --ioenc gbk: 让子进程按中文 Windows 控制台的编码输出，用于在 Linux / CI 上发现编码问题
    ioenc = sys.argv[sys.argv.index("--ioenc") + 1] if "--ioenc" in sys.argv else "utf-8"
    env = dict(os.environ, PYTHONIOENCODING=ioenc)
    env.pop("PYTHONUTF8", None)
    files = sorted(p for p in HERE.glob("test_*.py"))
    files.append(HERE / "smoke_gui.py")
    bad = []
    for f in files:
        if quick and f.name in SLOW:
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, str(f)], cwd=HERE.parent, capture_output=True, text=True, encoding=ioenc, errors="replace", env=env)
        ok = r.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'} {f.name:24s} {time.time() - t0:6.1f} s", flush=True)
        if not ok:
            bad.append(f.name)
            print((r.stdout + r.stderr)[-3000:])
    print("全部通过" if not bad else f"失败: {', '.join(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
