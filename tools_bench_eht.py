"""Speed of the EHT baseline on this computer: numpy and (if installed) numba, and their agreement.
Usage: python tools_bench_eht.py <crystal.cif> [n_configurations]   (pip install numba for the compiled kernel)"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from mobility_tool import _eht_fast as F
    from mobility_tool.alsurface import EHTBaseline
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    c = read_cif(sys.argv[1])
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    rng = np.random.default_rng(0)
    Q = np.array([3.6, 1.5, 0.5, 5.0, 2.0, 3.0]) + rng.normal(0, 0.4, (n, 6))
    print(f"{len(ce.sym)} atoms per molecule, {n} configurations; numba installed: {F.HAVE_NUMBA}")
    have = F.HAVE_NUMBA
    res = {}
    for backend in (["numpy", "numba"] if have else ["numpy"]):
        F.HAVE_NUMBA = backend == "numba"
        b = EHTBaseline(ce.sym, ce.Y)
        b._one(Q[0])                                   # compile / warm up
        t = time.time()
        res[backend] = np.array([b._one(q) for q in Q])
        print(f"  {backend:6s} {(time.time() - t) / n * 1000:7.2f} ms per configuration")
    F.HAVE_NUMBA = have
    if len(res) == 2:
        print(f"  largest difference numba − numpy: {np.abs(res['numba'] - res['numpy']).max():.1e} meV")
    else:
        print("  pip install numba  →  compiled overlap kernel (typically several times faster)")


if __name__ == "__main__":
    main()
