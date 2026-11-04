"""Comprehensive test run of Mobilis (no Gaussian or ADF needed).

    python run_test.py            # writes output/ and compares it with expected_output/
    python run_test.py --keep     # keep an existing output/ folder

Steps
  1  KMC against exact solutions (Table 2 and Figure 3 of the paper), including the crystal graph of α-perylene
  2  one-dimensional chain: random-walk mobility against the Deng–Goddard / Wen expression (Section 6.1)
  3  neighbour dimers of anthracene and the Gaussian inputs for the couplings and reorganization energies
  4  built-in EHT couplings, crystal-graph KMC and transient localization for anthracene
  5  anisotropy benchmark from the stored B3LYP results: couplings of 26 dimer groups (DFT, ZINDO, EHT),
     108 mobility tensors (3 coupling sources x 2 rate expressions x 3 estimators x 6 crystal-carrier cases),
     Figures 4, 6 and 7 of the paper
"""
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                       # Mobilis source folder (contains mobility_tool/)
OUT = HERE / "output"
EXP = HERE / "expected_output"
BENCH = HERE / "benchmark"
ENV = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""), MOBILIS_LANG="en",
           PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")


def _show(args, cwd):
    """Command line as written to the log: 'python' for the interpreter, paths relative to the working folder."""
    out = []
    for a in args:
        a = str(a)
        if a == sys.executable:
            a = "python"
        elif os.path.isabs(a):
            a = os.path.relpath(a, cwd)
        out.append(f'"{a}"' if " " in a else a)
    return " ".join(out)


def run(step, args, cwd, log):
    t0 = time.time()
    r = subprocess.run(args, cwd=cwd, env=ENV, capture_output=True, text=True, encoding="utf-8")
    text = r.stdout + r.stderr
    log.write(f"\n===== {step}\n$ {_show(args, cwd)}\n{text}")
    print(f"{step:60s} {'ok' if r.returncode == 0 else 'FAILED'}  ({time.time() - t0:.0f} s)")
    if r.returncode:
        print(text[-2000:])
        sys.exit(1)
    return text


def mob(*a):
    return [sys.executable, "-m", "mobility_tool", *a]


def numbers_close(a, b, rtol=1e-6, path=""):
    """Recursive comparison of JSON-like data; returns a list of differences."""
    bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                bad.append(f"{path}/{k}: missing")
            else:
                bad += numbers_close(a[k], b[k], rtol, f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        for i, (x, y) in enumerate(zip(a, b)):
            bad += numbers_close(x, y, rtol, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isclose(a, b, rel_tol=rtol, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))):
            bad.append(f"{path}: {a} vs {b}")
    elif a != b:
        bad.append(f"{path}: {a!r} vs {b!r}")
    return bad


def main():
    if OUT.exists() and "--keep" not in sys.argv:
        shutil.rmtree(OUT)
    OUT.mkdir(exist_ok=True)
    cif = BENCH / "1 structure"
    with open(OUT / "run_test.log", "w", encoding="utf-8") as log:
        run("1  KMC against exact solutions (12 cases)",
            mob("kmc-validate", "--cif", str(cif / "perylene.cif"), "--out", "kmc_validation.csv"), OUT, log)
        run("2  1D chain: random walk vs Deng-Goddard / Wen",
            mob("mobility", "--lam", "0.2", "--path", "5.0,30,2", "--json", "chain_1d.json"), OUT, log)
        run("3  anthracene: neighbour dimers and Gaussian inputs",
            mob("crystal", str(cif / "anthracene.cif"), "--cutoff", "4.0", "-o", "anthracene_inputs", "--program", "gaussian",
                "--method", "B3LYP", "--basis", "6-31G(d)", "--nproc", "16", "--mem", "32GB"), OUT, log)
        run("   anthracene: reorganization-energy inputs",
            mob("reorg-gen", "anthracene_inputs/G1_A.gjf", "-o", "anthracene_inputs/reorg", "--name", "anthracene",
                "--carrier", "both"), OUT, log)
        run("4  anthracene: EHT couplings + crystal-graph KMC (holes)",
            mob("kmc", str(cif / "anthracene.cif"), "--lam", "0.137", "--carrier", "hole"), OUT, log)
        run("   anthracene: transient localization (holes)",
            mob("tlt", str(cif / "anthracene.cif"), "--carrier", "hole"), OUT, log)
        run("5  benchmark couplings (B3LYP / ZINDO / EHT, 26 groups)",
            [sys.executable, str(BENCH / "scripts" / "couplings.py")], OUT, log)
        run("   benchmark mobility tensors (108 combinations)",
            [sys.executable, str(BENCH / "scripts" / "models.py")], OUT, log)
        run("   Figures 4, 6 and 7", [sys.executable, str(BENCH / "scripts" / "make_figs.py")], OUT, log)

    if not EXP.exists():
        print("no expected_output/ folder: nothing to compare")
        return
    bad = []
    for name in ("chain_1d.json", "couplings.json", "models.json", "fig_data.json"):
        bad += [f"{name}: {d}" for d in numbers_close(json.load(open(OUT / name, encoding="utf-8")),
                                                      json.load(open(EXP / name, encoding="utf-8")))]
    rows = lambda p: [r for r in csv.reader(open(p, encoding="utf-8-sig"))][1:]
    for x, y in zip(rows(OUT / "kmc_validation.csv"), rows(EXP / "kmc_validation.csv")):
        bad += [f"kmc_validation.csv {x[0]}: {a} vs {b}" for a, b in zip(x[1:], y[1:])
                if not math.isclose(float(a), float(b), rel_tol=1e-6)]
    g_out = json.load(open(OUT / "anthracene_inputs" / "dimers.json", encoding="utf-8"))
    g_exp = json.load(open(EXP / "anthracene_inputs" / "dimers.json", encoding="utf-8"))
    if len(g_out["groups"]) != len(g_exp["groups"]):
        bad.append("anthracene dimer groups differ")
    print("\nComparison with expected_output/: " + ("all results agree" if not bad else f"{len(bad)} differences"))
    for d in bad[:20]:
        print("  " + d)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
