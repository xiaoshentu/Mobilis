"""ceds md, v1.28.2: (1) the coupling statistics of large pairs report progress and remaining time and give the same
numbers as without progress; (2) trajectories are saved with their settings and reused when crystal, supercell and
MD settings agree, and run again when a setting changes or reuse is switched off."""
import os as _os
_os.environ.setdefault("MOBILIS_LANG", "zh")
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CIF = os.path.join(ROOT, "tests", "data", "anthracene_COD5000168.cif")


class FakeSurface:
    source = "gp"
    frozen = []

    def __call__(self, Q, carrier):
        Q = np.atleast_2d(Q)
        return 40.0 + 30.0 * Q[:, 0] - 5.0 * Q[:, 1] + (2.0 if carrier == "electron" else 0.0)

    def describe(self):
        return "fake"

    def coverage(self, Q):
        return 1.0


def main():
    from mobility_tool import rigidmd as RM
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    c = read_cif(CIF)
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    surf = {p["gid"]: FakeSurface() for p in ce.pairs}
    md = RM.RigidCrystalMD(ce.crys, ce.frames, (ce.sym, ce.Y), (2, 2, 2))
    opts = dict(equil_ps=0.02, prod_ps=0.06, dt=5.0, every=2)
    fr = RM.run_temperature(md, 300.0, opts)
    # (1) progress
    RM.PROGRESS_MIN = 10
    msgs = []
    r1 = RM.md_couplings(ce, md, fr, surf, 300.0, progress=msgs.append)
    r0 = RM.md_couplings(ce, md, fr, surf, 300.0)
    for g, r in r0.items():
        if "hole" in r:
            assert abs(r["hole"]["std"] - r1[g]["hole"]["std"]) < 1e-12, g
    assert any("coupling statistics G" in m and "configurations" in m for m in msgs), msgs[:3]
    assert any("100%" in m and "min left" in m for m in msgs), msgs[-3:]
    # (2) trajectory reuse
    with tempfile.TemporaryDirectory() as d:
        log = []
        kw = dict(md=md, save_dir=d, on_progress=log.append, static=False, n_real=2, **opts)
        rows1, _ = RM.md_temperature_series(ce, surf, [300], **kw)
        assert os.path.exists(os.path.join(d, "md_300K.npz")) and os.path.exists(os.path.join(d, "md_300K.json"))
        assert any("trajectory saved" in m for m in log)
        log.clear()
        rows2, _ = RM.md_temperature_series(ce, surf, [300], **kw)
        assert any("reusing the saved trajectory" in m for m in log), log[:3]
        assert not any("steps)" in m for m in log), "MD should not run again"
        assert abs(rows1[0]["U_eq"] - rows2[0]["U_eq"]) < 1e-5
        log.clear()
        RM.md_temperature_series(ce, surf, [300], **dict(kw, prod_ps=0.08))      # changed setting → new MD
        assert not any("reusing" in m for m in log) and any("trajectory saved" in m for m in log)
        log.clear()
        RM.md_temperature_series(ce, surf, [300], **dict(kw, prod_ps=0.08, reuse_traj=False))
        assert not any("reusing" in m for m in log)
        # a trajectory saved by an older version (npz only): reused after checking supercell, lattice and frames
        os.remove(os.path.join(d, "md_300K.json"))
        log.clear()
        RM.md_temperature_series(ce, surf, [300], **dict(kw, prod_ps=0.08))
        assert any("reusing" in m and "older version" in m for m in log), log[:3]
        assert os.path.exists(os.path.join(d, "md_300K.json"))
        os.remove(os.path.join(d, "md_300K.json"))
        log.clear()
        RM.md_temperature_series(ce, surf, [300], **dict(kw, prod_ps=0.08, every=4))   # other frame interval
        assert not any("reusing" in m for m in log)
        log.clear()
        RM.md_temperature_series(ce, surf, [300], **dict(kw, prod_ps=0.08))
        meta = RM.traj_meta(md, 300, dict(opts, prod_ps=0.08))
        assert RM.load_trajectory(os.path.join(d, "md_300K.npz"), md, meta) is not None
        meta["charges"] = "x"
        assert RM.load_trajectory(os.path.join(d, "md_300K.npz"), md, meta) is None
    # (3) cost estimate before the MD
    import contextlib
    import io
    import types
    from mobility_tool import cli

    class Base:
        def _one(self, q):
            return (0.0, 0.0)
    sb = {p["gid"]: types.SimpleNamespace(baseline=Base()) for p in ce.pairs}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli._md_cost_note(ce, sb, md, types.SimpleNamespace(md_prod=30.0, md_dt=5.0, md_every=4))
    n_exp = len(ce.pairs) * 1500 * len(md.cells)
    assert f"约 {n_exp} 次 EHT 计算" in buf.getvalue(), buf.getvalue()
    from mobility_tool import i18n
    i18n.CLI_LANG = "en"
    assert f"about {n_exp} EHT evaluations" in i18n.tr_cli(buf.getvalue()), i18n.tr_cli(buf.getvalue())
    src = open(os.path.join(ROOT, "mobility_tool", "cli.py"), encoding="utf-8").read()
    assert "--rerun-md" in src and "--no-save-traj" in src and "reuse_traj=not a.rerun_md" in src
    print(f"progress messages: {len(msgs)}; trajectory reuse ok")


if __name__ == "__main__":
    main()
    print("test_v46 ok")
