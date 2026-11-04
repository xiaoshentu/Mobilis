"""English command line (v1.24): help of every subcommand and option, translated messages, Chinese with --lang zh."""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAN = re.compile(r"[一-鿿]")
CMDS = ("reorg-gen reorg-analyze ti-gen ti-calc mobility reorg-decomp adf-gen adf-read crystal ipea orbitals fcht-gen "
        "fcht-read mobility2d mobility3d excited-gen excited-read rmsd orbcomp run eht-ti compare-ti kmc strain-scan tlt "
        "vibronic kmc-validate scan ceds polar morph zindo cif2gro gui diagnose").split()


def _run(args, lang="en"):
    env = dict(os.environ, MOBILIS_LANG=lang, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "mobility_tool", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8")


def test_help_english_and_complete():
    top = _run(["-h"]).stdout
    assert not HAN.search(top) and "charge transport" in top
    for c in CMDS:
        out = _run([c, "-h"]).stdout
        assert out and not HAN.search(out), (c, [l for l in out.splitlines() if HAN.search(l)][:3])
    # every option has a help text
    import argparse
    sys.path.insert(0, ROOT)
    from mobility_tool import cli
    cap = {}
    orig = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        cap["p"] = self
        raise SystemExit
    argparse.ArgumentParser.parse_args = grab
    try:
        cli.main(["-h"])
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    missing = []
    for act in cap["p"]._actions:
        if isinstance(act, argparse._SubParsersAction):
            for name, sp in act.choices.items():
                missing += [f"{name} {a.dest}" for a in sp._actions if not a.help and not isinstance(a, argparse._HelpAction)]
    assert not missing, missing


def test_messages_and_chinese_switch():
    out = _run(["mobility", "--lam", "0.2", "--path", "5.0,30,2"]).stdout
    assert not HAN.search(out), out
    assert "Deng–Goddard" in out and "lower by 2.0×" in out and "random walk" in out
    zh = _run(["--lang", "zh", "mobility", "--lam", "0.2", "--path", "5.0,30,2"], lang="").stdout
    assert "偏小" in zh
    assert HAN.search(_run(["crystal", "-h"], lang="zh").stdout)
    cif = os.path.join(ROOT, "tests", "data", "pentacene_RT.cif")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, "-m", "mobility_tool", "crystal", cif, "-o", os.path.join(d, "dm")], cwd=ROOT,
                           env=dict(os.environ, MOBILIS_LANG="en", PYTHONIOENCODING="utf-8"), capture_output=True,
                           text=True, encoding="utf-8")
        assert r.returncode == 0 and not HAN.search(r.stdout) and "equivalent groups" in r.stdout, r.stdout


if __name__ == "__main__":
    test_help_english_and_complete()
    test_messages_and_chinese_switch()
    print("ok")
