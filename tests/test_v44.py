"""Lattice-stability check of the rigid-body MD (ceds md): a stable trajectory passes; a trajectory whose molecules
slide off their sites, or whose U_eq grows during the run, or whose U_eq is far above the crystallographic value, is
flagged; the flags go to the summary text and the CSV."""
import os as _os
_os.environ.setdefault("MOBILIS_LANG", "zh")
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CIF = os.path.join(ROOT, "tests", "data", "anthracene_COD5000168.cif")


def main():
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    from mobility_tool.rigidmd import RigidCrystalMD, stability_check, stability_text
    c = read_cif(CIF)
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    md = RigidCrystalMD(ce.crys, ce.frames, (ce.sym, ce.Y), (2, 2, 2))
    rng = np.random.default_rng(0)
    n, N = 200, md.N
    c0 = md.R.copy() if hasattr(md, "R") and np.shape(md.R) == (N, 3) else np.zeros((N, 3))
    Q0 = np.repeat(np.eye(3)[None, None], N, 1)[0]
    Q = np.repeat(Q0[None], n, 0)
    cs = c0[None] + 0.12 * rng.standard_normal((n, N, 3))
    st = stability_check(md, {"c": cs, "Q": Q}, ueq_ref=md.u_eq({"c": cs, "Q": Q}))
    assert st["stable"] and abs(st["ratio_ref"] - 1) < 1e-9 and abs(st["growth"] - 1) < 0.2, st
    assert "晶格稳定" in stability_text(300, st)
    # sliding: half of the molecules move 1.5 Å along x during the run
    slide = cs.copy()
    slide[:, ::2, 0] += np.linspace(0, 1.5, n)[:, None]
    s2 = stability_check(md, {"c": slide, "Q": Q}, ueq_ref=st["U_eq"])
    assert not s2["stable"] and {"drift", "ratio_ref"} <= set(s2["flags"]), s2["flags"]
    assert "[警告]" in stability_text(300, s2)
    # growing amplitude only
    grow = c0[None] + (0.08 + 0.12 * np.linspace(0, 1, n))[:, None, None] * rng.standard_normal((n, N, 3))
    s3 = stability_check(md, {"c": grow, "Q": Q})
    assert "growth" in s3["flags"] and "ratio_ref" not in s3["flags"], s3
    # temperature series: U_eq ∝ T passes, a jump of U_eq/T is flagged (DNTT 250 → 275 K: 0.029 → 0.12 Å²)
    from mobility_tool.rigidmd import series_check
    rows = [{"T": T, "U_eq": u, "stability": {"flags": [], "stable": True}}
            for T, u in ((200, 0.028), (250, 0.029), (275, 0.12), (300, 0.14))]
    assert series_check(rows) == [275, 300] and not rows[2]["stability"]["stable"]
    assert "anharmonic" in rows[3]["stability"]["flags"] and rows[1]["stability"]["stable"]
    rub = [{"T": T, "U_eq": u, "stability": {"flags": [], "stable": True}}
           for T, u in ((100, 0.00399), (200, 0.00796), (300, 0.01145))]
    assert series_check(rub) == []
    src = open(os.path.join(ROOT, "mobility_tool", "cli.py"), encoding="utf-8").read()
    assert "ueq_ref=ueq_ref" in src and "stability_text" in src
    print(f"stable: growth {st['growth']:.2f}; sliding: {s2['flags']} drift {s2['drift_max_A']:.2f} Å; growing: {s3['flags']}")


if __name__ == "__main__":
    main()
    print("test_v44 ok")
