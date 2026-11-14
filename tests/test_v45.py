"""EHT baseline of the Δ-GP surfaces in worker processes (spawn start method, as on Windows): large batches give the
same couplings as one process; small batches stay in the main process; the CLI sets the number of processes."""
import os as _os
_os.environ.setdefault("MOBILIS_LANG", "zh")
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    from mobility_tool import alsurface as A
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    c = read_cif(os.path.join(ROOT, "tests", "data", "pentacene_RT.cif"))
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    q0 = np.asarray(ce.pairs[0]["q0"], float)
    Q = q0 + 0.05 * np.random.default_rng(0).standard_normal((60, 6))
    ref = A.EHTBaseline(ce.sym, ce.Y).values(Q)
    A.EHT_JOBS, A.EHT_PARALLEL_MIN = 2, 40
    b = A.EHTBaseline(ce.sym, ce.Y)
    t = time.time()
    v = b.values(np.r_[Q, Q[:5]])
    assert np.allclose(v[:60], ref, atol=1e-9) and np.allclose(v[60:], ref[:5], atol=1e-9)
    assert len(A._POOLS) == 1 and len(b.cache) == 60
    small = A.EHTBaseline(ce.sym, ce.Y)
    small.values(Q[:10])
    assert len(A._POOLS) == 1                     # below the threshold: no new pool
    for p in A._POOLS.values():
        p.shutdown()
    A._POOLS.clear()
    A.EHT_JOBS = 1
    r = subprocess.run([sys.executable, "-c", "import sys; sys.argv=['m','--help']; import runpy; "
                        "runpy.run_module('mobility_tool', run_name='__main__')"], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0
    src = open(os.path.join(ROOT, "mobility_tool", "__main__.py"), encoding="utf-8").read()
    assert 'if __name__ == "__main__"' in src         # spawn workers import __main__ without running the CLI
    print(f"parallel EHT = serial ({time.time() - t:.1f} s for 60 configurations with 2 processes)")


if __name__ == "__main__":
    main()
    print("test_v45 ok")
