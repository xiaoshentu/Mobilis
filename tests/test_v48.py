"""ORCA / xtb levels (v1.29.0): Molden reader and dimer projection with phase alignment of B; a scan project with
an xtb and an ORCA level writes the reference, B and AB jobs, and collects signed couplings from Molden files.
The Molden files are produced here with the built-in EHT (same basis in A, B and AB, B's orbital signs randomized),
so the collected couplings must equal the EHT baseline couplings, sign included."""
import os as _os
_os.environ.setdefault("MOBILIS_LANG", "zh")
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CIF = os.path.join(ROOT, "tests", "data", "anthracene_COD5000168.cif")


def eht_molden(path, sym, X, rng=None):
    from mobility_tool import eht as E
    from mobility_tool.molden import HARTREE_EV, write_molden
    r = E.eht(sym, X)
    shells, nbf, _ = E.build_basis(sym, X)
    C = r["C"] * (rng.choice([-1.0, 1.0], r["C"].shape[1]) if rng is not None else 1.0)
    occ = np.r_[np.full(r["nocc"], 2.0), np.zeros(nbf - r["nocc"])]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_molden(path, sym, X, [(s["atom"], s["l"], list(zip(s["a"], s["c"]))) for s in shells], C,
                 r["e"] / HARTREE_EV, occ)


def main():
    from mobility_tool.alsurface import EHTBaseline
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    from mobility_tool.molden import read_molden
    from mobility_tool.scanflow import ScanProject, make_cfg
    c = read_cif(CIF)
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    sym, Y = list(ce.sym), np.asarray(ce.Y, float)
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        xyz = Path(tmp) / "mono.xyz"
        xyz.write_text(f"{len(sym)}\nanthracene\n" + "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(sym, Y)))
        pr = ScanProject.create(Path(tmp) / "proj", xyz)
        Y = pr.Y
        sym = pr.symbols
        base = EHTBaseline(sym, Y)
        for engine in ("xtb", "orca"):
            lv = pr.add_qc_level(engine, method="B3LYP D3BJ", basis="def2-SVP", gfn=1)
            ld = pr.level_dir(lv)
            assert (ld / "run_mono.sh").exists() and (ld / "run_mono.bat").exists()
            if engine == "orca":
                inp = (ld / "mono.inp").read_text()
                assert "! B3LYP D3BJ def2-SVP" in inp and "orca_2mkl" not in inp
                assert "-molden" in (ld / "run_mono.sh").read_text()
            else:
                assert "--molden" in (ld / "run_mono.sh").read_text()
            cfgs = [make_cfg(3.6, 1.2, 0.4, 10, 3, 5), make_cfg(3.5, 0.0, 2.5), make_cfg(3.7, 4.0, -1.0, -20, 4, 30),
                    make_cfg(3.6, 2.0, 1.0, 50, -5, 52)]
            name = f"s_{engine}"
            info = pr.add_scan(name, cfgs, lv, reduce=False)
            d = pr.scan_dir(name)
            sh = (d / f"{info['batches'][0]}.sh").read_text()
            assert ("ORCA_EXE" in sh) == (engine == "orca") or "XTB_EXE" in sh
            pc = pr.precheck(name)
            assert not pc["ok"] and "reference monomer" in pc["errors"][0]
            eht_molden(pr.t21(lv), sym, Y)                                   # the reference monomer
            rows = [r for r in pr.manifest(name) if r["status"] == "ok" and not r["rep"]]
            for r in rows:
                B = pr.B_coords(r)
                eht_molden(d / "outputs" / r["id"] / "B.molden.input", sym, B, rng)
                eht_molden(d / "outputs" / r["id"] / "AB.molden.input", sym + sym, np.vstack([Y, B]))
            pc = pr.precheck(name)
            assert pc["ok"] and pc["n_done"] == len(rows) == pc["n_total"], pc
            recs = [x for x in pr.collect(name) if x["state"] == "done"]
            assert len(recs) == len(rows)
            worst = 0.0
            for x in recs:
                q = np.array([x[k] for k in ("z", "dx", "dy", "gamma", "pitch", "roll")], float)
                hb, lb = base._one(q)
                worst = max(worst, abs(x["hole"] - hb), abs(x["electron"] - lb))
                assert abs(abs(x["phase_B_hole"]) - 1) < 1e-6
            assert worst < 1e-4, worst
            print(f"{engine}: {len(recs)} dimers, max |V - EHT baseline| = {worst:.2e} meV")
        m = read_molden(pr.t21(lv))
        assert m["nbf"] == m["nmo"] and m["nocc"] == 33
    src = open(os.path.join(ROOT, "mobility_tool", "cli.py"), encoding="utf-8").read()
    assert '"--engine"' in src


if __name__ == "__main__":
    main()
    print("test_v48 ok")
