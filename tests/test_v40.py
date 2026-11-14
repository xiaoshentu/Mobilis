"""Frame-resolved TLT (frametlt.py): bond mapping on the MD supercell, agreement of the representative bond with
md_pair_params, identical results of the three variants without fluctuations, and the supercell check of ceds md."""
import os as _os
_os.environ.setdefault("MOBILIS_LANG", "zh")
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool import frametlt as FT  # noqa: E402
from mobility_tool import rigidmd as RM  # noqa: E402
from mobility_tool.ceds import CEDS, OPS  # noqa: E402
from mobility_tool.crystal import find_molecules, read_cif  # noqa: E402

CIF = os.path.join(ROOT, "tests", "data", "pentacene_RT.cif")


class Lin:
    """Analytic coupling surface around the crystal geometry of a pair (meV)."""
    source = "test"

    def __init__(self, q0, a, b):
        self.q0, self.a, self.b = np.asarray(q0, float), a, b

    def __call__(self, q, c):
        d = FT._dist(np.atleast_2d(q), self.q0)
        s = 1.0 if c == "hole" else -0.7
        return s * (self.a + self.b * (3 * d[:, 0] + 2 * d[:, 1] - 2.5 * d[:, 2] + 0.04 * d[:, 5] + 0.6 * d[:, 1] ** 2))

    def describe(self):
        return "test"

    def coverage(self, q):
        return 1.0


def test_frame_tlt():
    crys = read_cif(CIF)
    ce = CEDS(crys, find_molecules(crys))
    amp = {1: 75.0, 2: -42.0, 3: 33.0, 7: 31.0}
    surf = {g: Lin(next(p["q0"] for p in ce.pairs if p["gid"] == g), a, 9.0) for g, a in amp.items()}
    axes = FT.keep_axes(ce, FT.crystal_v0(ce, surf))
    assert axes == (0, 1)
    bonds = FT.bond_types(ce, surf, axes)
    assert sorted(b["gid"] for b in bonds) == [1, 1, 2, 2, 3, 7]          # 6 in-plane bonds per cell (Z = 2)
    md = RM.RigidCrystalMD(crys, ce.frames, (ce.sym, ce.Y), (8, 8, 1))
    _, st = md.run(300.0, 60, 5.0, 5e-3, every=60, seed=300)
    fr, _ = md.run(300.0, 120, 5.0, 1e-3, every=10, seed=301, state=st)
    V, idx = FT.bond_couplings(ce, md, fr, bonds, recenter=False)
    for p in ce.pairs:                                  # the bond of the representative dimer gives the same couplings
        if p["gid"] not in surf:
            continue
        q, _ = RM.md_pair_params(md, fr, p, np.diag(OPS[p["g"]]).astype(float), recenter=False, q0=p["q0"])
        ref = ce.chi("hole", p["g"]) * surf[p["gid"]](q, "hole")
        k = [k for k, b in enumerate(bonds) if (b["i"], b["j"], b["t"]) == (p["ref"], p["mol"], tuple(p["translation"]))]
        if k:
            assert np.allclose(np.sort(V["hole"][k[0]].ravel()), np.sort(ref), atol=1e-8), p["gid"]
        else:                                           # reversed representative: same distribution within sampling
            same = [k for k, b in enumerate(bonds) if b["gid"] == p["gid"]]
            assert abs(V["hole"][same[0]].mean() - ref.mean()) < 0.3 * ref.std() + 0.5
    for k, b in enumerate(bonds):                      # every bond fluctuates around the coupling of its pair
        exp = ce.chi("hole", b["g"]) * amp[b["gid"]]
        assert abs(V["hole"][k].mean() - exp) < 8.0, (b["gid"], V["hole"][k].mean(), exp)
    flat = {g: Lin(s.q0, s.a, 0.0) for g, s in surf.items()}
    r0 = FT.frame_tlt(ce, md, fr, flat, 300.0, n_real=3)
    for c in ("hole", "electron"):
        m = [r0[c][k]["mu"] for k in ("gaussian", "marginal", "frames")]
        assert m[0] > 0 and max(m) - min(m) < 1e-9 * m[0], m
    r = FT.frame_tlt(ce, md, fr, surf, 300.0, n_real=4)
    # stride: the frame-resolved Hamiltonians are the same frames; the statistics come from every 3rd frame
    msg = []
    r3 = FT.frame_tlt(ce, md, fr, surf, 300.0, n_real=4, stride=3, progress=msg.append)
    assert r3["stride"] == 3 and r3["stat_frames"] < len(fr["c"]) and any("stride 3" in m for m in msg)
    for c in ("hole", "electron"):
        assert abs(r3[c]["frames"]["mu"] / r[c]["frames"]["mu"] - 1) < 0.05, (c, r3[c]["frames"]["mu"], r[c]["frames"]["mu"])  # recentering mean from the subset
        for k_ in range(len(bonds)):
            assert abs(r3["bonds"][k_][c]["std"] / r["bonds"][k_][c]["std"] - 1) < 0.25
    for c in ("hole", "electron"):
        for k in ("gaussian", "marginal", "frames"):
            assert np.isfinite(r[c][k]["mu"]) and 0 < r[c][k]["mu"] < r0[c]["frames"]["mu"] * 1.05, (c, k, r[c][k]["mu"])
    assert len(r["bonds"]) == 6 and r["correlations"]["hole"]
    txt = FT.report(r)
    assert "MD frames" in txt and "correlation" in txt
    out = os.path.join(ROOT, "tests", "_v40_frames.csv")
    FT.export_csv([(300.0, r)], out)
    assert "frames" in open(out, encoding="utf-8-sig").read()
    os.remove(out)
    try:
        FT.frame_tlt(ce, RM.RigidCrystalMD(crys, ce.frames, (ce.sym, ce.Y), (4, 3, 1)), fr, surf, 300.0)
        raise AssertionError("small supercell accepted")
    except ValueError as e:
        assert "too small" in str(e) or "太小" in str(e)
    p = subprocess.run([sys.executable, "-m", "mobility_tool", "ceds", "md", "--cif", CIF, "--source", "eht",
                        "--temps", "300", "--supercell", "4,3,2", "--frame-tlt"], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert p.returncode != 0 and ("--frame-tlt uses the MD supercell" in (p.stdout + p.stderr) or "--frame-tlt 以 MD 超胞" in (p.stdout + p.stderr)), p.stdout + p.stderr


if __name__ == "__main__":
    test_frame_tlt()
    print("test_v40 ok")
