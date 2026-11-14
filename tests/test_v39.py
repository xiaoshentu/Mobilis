"""db: coupling-surface database. A crystal is run through the active-learning loop with the simulated ADF, its MD
statistics are computed, and the project is stored with `db add`; `db list` and `db export` read it back."""
import os as _os
_os.environ.setdefault("MOBILIS_LANG", "zh")
import csv
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from fake_adf import run_scan  # noqa: E402
from mobility_tool.alsurface import ALRun  # noqa: E402
from mobility_tool.scanflow import ScanProject  # noqa: E402

CIF = os.path.join(ROOT, "tests", "data", "pentacene_RT.cif")
REF = os.path.join(ROOT, "tests", "data", "pentacene_ref.xyz")


def cli(*args, ok=True):
    r = subprocess.run([sys.executable, "-m", "mobility_tool", *args], cwd=ROOT, capture_output=True, encoding="utf-8",
                       errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"}, text=True, timeout=1500)
    if ok:
        assert r.returncode == 0, r.stderr[-3000:] + r.stdout[-2000:]
    return r.stdout + r.stderr, r.returncode


def test_db():
    d = tempfile.mkdtemp()
    proj = os.path.join(d, "pent")
    pr = ScanProject.create(proj, REF, name="pent")
    pr.add_level()
    common = ["ceds", "al", "--cif", CIF, "--project", proj, "--name", "al1", "--pairs", "1,3", "--temps", "300",
              "--supercell", "3,2,1", "--md-equil", "0.5", "--md-prod", "3", "--al-pool", "150", "--al-init", "8",
              "--al-batch", "8", "--al-target", "0.2", "--al-mu-draws", "0"]
    cli(*common)
    for _ in range(6):
        run = ALRun(ScanProject(proj), "al1")
        run_scan(ScanProject(proj), run.state["rounds"][-1]["scan"])
        out, _rc = cli(*common)
        if "全部近邻对已收敛" in out:
            break
    else:
        raise AssertionError("not converged:\n" + out)
    run = ALRun(ScanProject(proj), "al1")
    n_al = sum(len(i) for r in run.state["rounds"] for i in r["selected"].values())
    md = os.path.join(d, "pent_md.csv")
    cli("ceds", "md", "--cif", CIF, "--project", proj, "--al-surfaces", "al1", "--temps", "300", "--supercell", "3,2,1",
        "--md-equil", "0.5", "--md-prod", "3", "--out", md)
    lam = os.path.join(d, "pent_lambda.json")
    json.dump({"name": "pent", "level": "B3LYP/6-31G(d)", "results": {"hole": {"lambda_eV": 0.092},
                                                                   "electron": {"lambda_eV": 0.132}}}, open(lam, "w"))
    db = os.path.join(d, "db")
    out, _rc = cli("db", "add", db, "pentacene_114447", "--project", proj, "--cif", CIF, "--ccdc", "114447", "--al", "al1",
                   "--lambda", lam, "--md-csv", md, "--note", "test")
    assert (f"{n_al} 个 DFT 二聚体" in out or f"{n_al} DFT dimers" in out) and "--validation" in out, out
    rec = json.load(open(os.path.join(db, "pentacene_114447", "record.json"), encoding="utf-8"))
    assert rec["schema"] == "mobilis-db-1" and rec["structure"]["ccdc"] == "114447"
    assert abs(rec["structure"]["cell"][0] - 6.265) < 0.01
    assert rec["active_learning"]["n_dft_total"] == n_al == rec["dimers"]["n"]
    assert rec["active_learning"]["delta_learning"] is True and rec["active_learning"]["rounds"] >= 1
    assert rec["reorganization"]["lambda_hole_meV"] == 92.0
    th = rec["thermal"]
    assert {"pairs", "mobility", "md"} <= set(th) and th["mobility"][0]["mu_hole"] > 0
    assert any(r["carrier"] == "hole" and r["sigma_meV"] > 0 for r in th["pairs"])
    rows = list(csv.DictReader(open(os.path.join(db, "pentacene_114447", "dimers.csv"), encoding="utf-8-sig")))
    assert len(rows) == n_al and {"z", "roll", "hole", "electron", "group"} <= set(rows[0])
    xyz = open(os.path.join(db, "pentacene_114447", "structures.xyz")).read()
    assert xyz.count("group=") == n_al and xyz.splitlines()[0].strip() == "72"
    assert os.path.exists(os.path.join(db, "pentacene_114447", "files", "pentacene_RT.cif"))
    # a second add of the same name is refused unless --overwrite; --no-cif leaves the structure file out
    out, rc = cli("db", "add", db, "pentacene_114447", "--project", proj, ok=False)
    assert rc != 0 and ("already holds a record" in out or "已有记录" in out)
    cli("db", "add", db, "pentacene_114447", "--project", proj, "--cif", CIF, "--no-cif", "--al", "al1", "--overwrite")
    assert not os.path.exists(os.path.join(db, "pentacene_114447", "files", "pentacene_RT.cif"))
    rec = json.load(open(os.path.join(db, "pentacene_114447", "record.json"), encoding="utf-8"))
    assert "thermal" not in rec and rec["structure"]["spacegroup"] is not None
    cli("db", "add", db, "copy2", "--project", proj, "--al", "al1", "--md-csv", md)
    out, _rc = cli("db", "list", db)
    assert "pentacene_114447" in out and "copy2" in out and "C22H14" in out, out
    out, _rc = cli("db", "export", db)
    allrows = list(csv.DictReader(open(os.path.join(db, "all_dimers.csv"), encoding="utf-8-sig")))
    assert len(allrows) == 2 * n_al and {r["crystal"] for r in allrows} == {"pentacene_114447", "copy2"}
    assert os.path.exists(os.path.join(db, "all_dimers_summary.csv"))
    # a project without finished dimers is refused and leaves no folder behind
    empty = os.path.join(d, "empty")
    ScanProject.create(empty, REF, name="empty").add_level()
    out, rc = cli("db", "add", db, "empty", "--project", empty, ok=False)
    assert rc != 0 and ("no finished dimer" in out or "没有算完的二聚体" in out) and not os.path.exists(os.path.join(db, "empty"))


if __name__ == "__main__":
    test_db()
    print("test_v39 ok")
