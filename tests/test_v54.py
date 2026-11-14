"""Project files, tutorials and packaging (v1.31.0): templates parse with the real argument parser, a project runs
and records its status, the tutorial commands are valid, and the wheel contains every module, the data files and
both entry points."""
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mobility_tool import project as P  # noqa: E402

ENV = dict(os.environ, PYTHONIOENCODING="utf-8", MOBILIS_LANG="zh", PYTHONPATH=str(ROOT))


def mt(args, cwd):
    return subprocess.run([sys.executable, "-m", "mobility_tool"] + args, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", env=ENV)


def check_templates_and_run():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "pentacene.cif").write_bytes((ROOT / "tests/data/pentacene_RT.cif").read_bytes())
        for t in P.TEMPLATES:
            f = f"{d}/{t}.mobilis.json"
            r = mt(["project", "new", f, "--template", t, "--set", "cif=pentacene.cif"], d)
            assert r.returncode == 0, r.stderr
            bad = P.check(f)
            print(f"  template {t}: {len(P.load(f)['steps'])} steps, argument errors {bad}")
            assert not bad
        r = mt(["project", "run", f"{d}/crystal.mobilis.json", "--only", "tlt_hole,kmc_hole"], d)
        print(r.stdout[-500:], r.stderr[-500:])
        assert r.returncode == 0
        st = json.loads((Path(d) / "crystal.status.json").read_text(encoding="utf-8"))
        assert st["tlt_hole"]["returncode"] == 0 and (Path(d) / "logs/kmc_hole.log").exists()
        r = mt(["project", "show", f"{d}/crystal.mobilis.json"], d)
        assert "完成" in r.stdout and "未运行" in r.stdout
        # a wrong argument is found by check, an undefined variable as well
        data = P.load(f"{d}/crystal.mobilis.json")
        data["steps"].append({"name": "bad", "command": "tlt", "args": ["${cif}", "--no-such-option"]})
        data["steps"].append({"name": "bad2", "command": "kmc", "args": ["${undefined}"]})
        P.save(f"{d}/bad.mobilis.json", data)
        assert P.check(f"{d}/bad.mobilis.json")


def check_tutorials():
    """Every `python -m mobility_tool …` line of the tutorials parses (tutorial files are run in test_v55)."""
    tut = sorted((ROOT / "docs" / "tutorials").glob("*.md"))
    assert len(tut) >= 4, tut
    n = 0
    for f in tut:
        for line in f.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:\$ )?python -m mobility_tool (.+)$", line)
            if not m or "<" in line:
                continue
            import shlex
            args = shlex.split(m.group(1))
            env = dict(ENV, MOBILIS_PARSE_ONLY="1")
            r = subprocess.run([sys.executable, "-m", "mobility_tool"] + args, cwd=ROOT / "examples", env=env,
                               capture_output=True, text=True, encoding="utf-8")
            ok = r.returncode == 0 and ("MOBILIS_PARSE_OK" in r.stdout or "-h" in args or "--version" in args)
            assert ok, (f.name, line, r.stderr[-400:])
            n += 1
    print(f"  tutorials: {len(tut)} files, {n} commands parse")
    assert n >= 15


def check_tutorial_demos():
    """The tutorial commands that use the bundled demo files run to the end."""
    import shlex
    import shutil
    with tempfile.TemporaryDirectory() as d:
        ex = Path(d) / "examples"
        shutil.copytree(ROOT / "examples", ex)
        n = 0
        for f in sorted((ROOT / "docs" / "tutorials").glob("*.md")):
            for line in f.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\s*python -m mobility_tool (.+)$", line)
                if not m or not any(k in line for k in ("tvcf_demo", "dp_demo", "dp-gen", "project ")):
                    continue
                r = mt(shlex.split(m.group(1)), ex)
                assert r.returncode == 0, (f.name, line, r.stdout[-800:], r.stderr[-800:])
                n += 1
        assert (ex / "tadf.json").exists() and (ex / "bz_spectrum.csv").exists() and (ex / "dp_results.csv").exists()
        print(f"  tutorial demos: {n} commands run")


def check_wheel():
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "--no-build-isolation",
                            "-w", d, "-q"], capture_output=True, text=True)
        if r.returncode != 0:
            if "No module named" in r.stderr or "setuptools" in r.stderr and "not" in r.stderr:
                print("  wheel: setuptools / wheel not installed, skipped")
                return
            raise AssertionError(r.stderr[-2000:])
        whl = next(Path(d).glob("mobilis-*.whl"))
        names = zipfile.ZipFile(whl).namelist()
        mods = {p.name for p in (ROOT / "mobility_tool").glob("*.py")}
        missing = [m for m in mods if f"mobility_tool/{m}" not in names]
        assert not missing, missing
        assert "mobility_tool/assets/atomdens.npz" in names and "mobility_tool/assets/mobilis.ico" in names
        ep = next(n for n in names if n.endswith("entry_points.txt"))
        txt = zipfile.ZipFile(whl).read(ep).decode()
        assert "mobilis = mobility_tool.cli:main" in txt and "mobilis-gui = mobility_tool.gui:main" in txt
        print(f"  wheel {whl.name}: {len(names)} files, all {len(mods)} modules, data files and entry points")


if __name__ == "__main__":
    check_templates_and_run()
    check_wheel()
    check_tutorials()
    check_tutorial_demos()
    print("test_v54 ok")
