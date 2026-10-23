"""依次运行 tests/ 下的全部测试脚本 (不需要 Gaussian / ADF)。用法: python tests/run_all.py [--quick]"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLOW = {"smoke_gui.py", "test_v18.py"}


def main():
    quick = "--quick" in sys.argv
    files = sorted(p for p in HERE.glob("test_*.py"))
    files.append(HERE / "smoke_gui.py")
    bad = []
    for f in files:
        if quick and f.name in SLOW:
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, str(f)], cwd=HERE.parent, capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok = r.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'} {f.name:24s} {time.time() - t0:6.1f} s", flush=True)
        if not ok:
            bad.append(f.name)
            print((r.stdout + r.stderr)[-3000:])
    print("全部通过" if not bad else f"失败: {', '.join(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
