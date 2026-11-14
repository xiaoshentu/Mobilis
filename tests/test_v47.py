"""v1.28.3: the EHT worker processes run with one BLAS/LAPACK thread each (otherwise J workers start J × J threads
and the parallel EHT becomes slower than one process); the main process keeps its own thread settings; the
parallel EHT gives the same couplings as the serial one."""
import os as _os
_os.environ.setdefault("MOBILIS_LANG", "zh")
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CIF = os.path.join(ROOT, "tests", "data", "anthracene_COD5000168.cif")


def main():
    from mobility_tool import alsurface as A
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    c = read_cif(CIF)
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    before = {k: os.environ.get(k) for k in A._THREAD_VARS}
    A.EHT_JOBS, A.EHT_PARALLEL_MIN = 2, 20
    b = A.EHTBaseline(ce.sym, ce.Y)
    rng = np.random.default_rng(3)
    Q = np.array([3.6, 1.2, 0.4, 4.0, 2.0, 3.0]) + rng.normal(0, 0.3, (60, 6))
    par = b.values(Q)
    ser = np.array([b._one(q) for q in Q])
    assert np.allclose(par, ser, atol=1e-9), np.abs(par - ser).max()
    pool = A._eht_pool(b, 2)
    with A._single_thread_env():
        seen = list(pool.map(A._worker_threads, range(4)))
    assert all(v == "1" for d in seen for v in d.values()), seen
    after = {k: os.environ.get(k) for k in A._THREAD_VARS}
    assert after == before, (before, after)
    for p in list(A._POOLS.values()):
        p.shutdown()
    print("workers: one BLAS thread; main process unchanged; parallel = serial")


if __name__ == "__main__":
    main()
    print("test_v47 ok")
