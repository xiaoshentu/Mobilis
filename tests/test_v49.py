"""EHT speed-up (v1.29.0): the compiled overlap kernel (_eht_fast, numba when installed; tested here as plain Python)
equals eht.overlap_matrix to machine precision, including the primitive screening; EHTBaseline._one gives the same
couplings as eht_transfer_integrals; shells with d functions fall back to numpy; a failing kernel falls back to numpy
with a warning instead of stopping the run."""
import os as _os
_os.environ.setdefault("MOBILIS_LANG", "zh")
import contextlib
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CIF = os.path.join(ROOT, "tests", "data", "anthracene_COD5000168.cif")


def main():
    from mobility_tool import _eht_fast as F
    from mobility_tool import eht as E
    from mobility_tool.alsurface import EHTBaseline
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    from mobility_tool.surrogate import build_B, rotation
    c = read_cif(CIF)
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    sym, Y = list(ce.sym), np.asarray(ce.Y, float)
    rng = np.random.default_rng(2)
    worst = 0.0
    for k in range(6):
        q = np.array([3.5, 1.0, 0.5, 0.0, 0.0, 0.0]) + rng.normal(0, [0.4, 2, 2, 30, 10, 30])
        B = build_B(Y, q[0], q[1], q[2], q[3], q[5], q[4], mirror=(k == 5))
        sa, na, _ = E.build_basis(sym, Y)
        sb, nb, _ = E.build_basis(sym, B)
        ref = E.overlap_matrix(sa, na, shells_b=sb, nbf_b=nb)
        fast = F.overlap_block(sa, na, sb, nb, python=True)
        worst = max(worst, np.abs(ref - fast).max())
    assert worst < 1e-14, worst
    b = EHTBaseline(sym, Y)
    for q in rng.normal([3.6, 1.2, 0.4, 5, 2, 3], [0.3, 1, 1, 20, 5, 20], (4, 6)):
        z, dx, dy, ga, pi, ro = q
        Bc = build_B(Y, z, dx, dy, ga, ro, pi)
        sh, _, _ = E.build_basis(sym, Bc)
        mb = dict(b.mono_b, shells=sh, C=E.rotate_coefficients(b.mono_b["shells"], b.mono_b["C"], rotation(ga, ro, pi)))
        r = E.eht_transfer_integrals(b.A, [(s, *p) for s, p in zip(sym, Bc)], esd=False, _mono_b=mb)["results"]
        h, l = b._one(q)
        assert abs(h - r["HOMO"]["J_eff_meV"]) < 1e-9 and abs(l - r["LUMO"]["J_eff_meV"]) < 1e-9
    cu, _, _ = E.build_basis(["Cu", "H"], [[0, 0, 0], [0, 0, 1.5]])
    assert not F.usable(cu, cu) and F.usable(sa, sb)
    # a failing compiled kernel → numpy, with a warning
    have, ob = F.HAVE_NUMBA, F.overlap_block
    try:
        F.HAVE_NUMBA = True
        F.overlap_block = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no compiler"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sab, _ = E._coupling_block(dict(shells=sa, nbf=na), dict(shells=sb, nbf=nb))
        assert "numba EHT kernel failed" in buf.getvalue() and not F.HAVE_NUMBA
        assert np.allclose(sab, E.overlap_matrix(sa, na, shells_b=sb, nbf_b=nb))
    finally:
        F.HAVE_NUMBA, F.overlap_block = have, ob
    assert os.path.exists(os.path.join(ROOT, "tools_bench_eht.py"))
    print(f"kernel = numpy to {worst:.1e}; numba installed: {have}")


if __name__ == "__main__":
    main()
    print("test_v49 ok")
