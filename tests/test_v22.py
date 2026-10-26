"""无定形 / MD 形貌 (morphology)：读写 .gro/.pdb、元素推断、映射 (乱序、跨边界、联合原子)、近邻、EHT J、位点能、KMC、
方法对比与命令行。"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool import morphology as Mo  # noqa: E402

CIF = os.path.join(ROOT, "tests", "data", "pentacene_RT.cif")
GRO = os.path.join(ROOT, "tests", "data", "pentacene_300K.gro")
REF = os.path.join(ROOT, "tests", "data", "pentacene_ref.xyz")


def _crystal():
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    c = read_cif(CIF)
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    return c, ce


def _shuffled_wrapped(fr, nat, seed=3, drop_h=False):
    """把每个残基内的原子顺序打乱 (各残基相同)、原子折回盒子；可去掉 H (联合原子)。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(nat)
    els = fr["elements"][:nat]
    if drop_h:
        perm = np.array([k for k in perm if els[k] != "H"])
    M = len(fr["X"]) // nat
    idx = np.concatenate([m * nat + perm for m in range(M)])
    X = fr["X"][idx]
    f = X @ np.linalg.inv(fr["box"])
    f -= np.floor(f)
    return {"names": [fr["names"][k] for k in idx], "resid": fr["resid"][idx], "resname": [fr["resname"][k] for k in idx],
            "X": f @ fr["box"], "box": fr["box"], "title": "shuffled"}


def test_gro_roundtrip_and_elements():
    assert [Mo.guess_element(n) for n in ("C12", "H3", "CL1", "BR", "S1", "N10", "HA")] == ["C", "H", "Cl", "Br", "S", "N", "H"]
    assert Mo.guess_element("CA", {"CA": "C"}) == "C" and Mo.guess_element("C4X", {"C4": "S"}) == "S"
    c, ce = _crystal()
    fr = Mo.crystal_frame(c["lattice"], ce.frames, ce.sym, ce.Y, (2, 1, 1), "PEN")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.gro")
    Mo.write_gro(p, [fr, fr])
    back = Mo.read_gro(p)
    assert len(back) == 2 and np.abs(back[1]["X"] - fr["X"]).max() < 0.006
    assert np.allclose(back[0]["box"], fr["box"], atol=1e-4)                  # 三斜盒子
    assert len(Mo.split_residues(back[0])) == 4
    assert len(Mo.read_gro(GRO, {1})) == 1 and len(Mo.read_gro(GRO)) == 3
    # PDB (CRYST1 + 元素列)
    from mobility_tool.crystal import read_cif
    a_, b_, c_ = np.linalg.norm(fr["box"], axis=1)
    B = fr["box"]
    ang = [np.degrees(np.arccos(B[1] @ B[2] / (b_ * c_))), np.degrees(np.arccos(B[0] @ B[2] / (a_ * c_))),
           np.degrees(np.arccos(B[0] @ B[1] / (a_ * b_)))]
    L = [f"CRYST1{a_:9.3f}{b_:9.3f}{c_:9.3f}{ang[0]:7.2f}{ang[1]:7.2f}{ang[2]:7.2f} P 1           1"]
    for k, (nm, r, x, e) in enumerate(zip(fr["names"], fr["resid"], fr["X"], fr["elements"])):
        L.append(f"HETATM{k + 1:5d} {nm:<4s} PEN A{r:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00          {e:>2s}")
    L.append("END")
    pp = os.path.join(d, "x.pdb")
    open(pp, "w").write("\n".join(L) + "\n")
    pdb = Mo.read_pdb(pp)[0]
    assert np.abs(pdb["X"] - fr["X"]).max() < 0.002 and np.allclose(pdb["box"], fr["box"], atol=2e-3)
    assert Mo.Morphology(pdb, ce.sym, ce.Y).M == 4
    del read_cif


def test_mapping_neighbors_and_eht():
    from mobility_tool.eht import eht_transfer_integrals
    c, ce = _crystal()
    fr = Mo.crystal_frame(c["lattice"], ce.frames, ce.sym, ce.Y, (4, 3, 2), "PEN")
    m = Mo.Morphology(_shuffled_wrapped(fr, len(ce.sym)), ce.sym, ce.Y)
    assert m.M == 48 and m.fit_rmsd.max() < 1e-6
    p = m.neighbors(4.0)
    assert len(p) == 6 * 48                                                   # 每个分子 12 个近邻
    Jh, Je = m.couplings_eht()
    for k in (0, 5, 17):
        a, b = m.pair_atoms(k)
        r = eht_transfer_integrals([(s, *x) for s, x in zip(ce.sym, a)], [(s, *x) for s, x in zip(ce.sym, b)],
                                   esd=False)["results"]
        assert abs(abs(Jh[k]) * 1000 - abs(r["HOMO"]["J_eff_meV"])) < 1e-6
        assert abs(abs(Je[k]) * 1000 - abs(r["LUMO"]["J_eff_meV"])) < 1e-6
    # 联合原子 (无 H)：重原子叠合，得到同样的几何与 J
    ua = Mo.Morphology(_shuffled_wrapped(fr, len(ce.sym), drop_h=True), ce.sym, ce.Y)
    assert ua.united_atom and np.abs(ua.X - m.X).max() < 1e-6
    ua.neighbors(4.0)
    assert len(ua.pairs) == len(p)
    # MD 几何模式 (全原子) 与 qm 模式在刚性晶体中一致
    md = Mo.Morphology(_shuffled_wrapped(fr, len(ce.sym)), ce.sym, ce.Y, geometry="md")
    md.neighbors(4.0)
    Jh2, _ = md.couplings_eht()
    assert np.allclose(np.sort(np.abs(Jh2)), np.sort(np.abs(Jh)), atol=1e-8)


def test_kmc_matches_crystal_analytic():
    from mobility_tool.compare import marcus_tensor
    c, ce = _crystal()
    m = Mo.Morphology(Mo.crystal_frame(c["lattice"], ce.frames, ce.sym, ce.Y, (4, 3, 2), "PEN"), ce.sym, ce.Y)
    m.neighbors(4.0)
    m.couplings_eht()
    w = np.linalg.eigvalsh(marcus_tensor(m.graph("hole"), 0.092, 300.0))[::-1]
    r = m.kmc("hole", 0.092, 300.0, use_energies=False, walkers=800, steps=3000)
    assert abs(r["mu_principal"][0] - w[0]) < 4 * 3 * r["err"] + 0.06 * w[0]


def test_site_energies_export_and_compare():
    from mobility_tool.cli import _read_monomer
    from mobility_tool.compare import compare_methods
    from mobility_tool.polar import default_charges
    sym, Y = _read_monomer(REF)
    m = Mo.Morphology(Mo.read_gro(GRO, {1})[0], sym, Y)
    m.neighbors(4.0)
    m.couplings_eht()
    ch = default_charges(sym, Y)
    E = m.site_energies(ch, "static")
    assert abs(E["hole"].mean()) < 1e-12 and 0.005 < E["hole"].std() < 0.2
    r, C = m.energy_correlation("hole", dr=3.0)
    assert np.isfinite(C).any()
    d = tempfile.mkdtemp()
    files = m.export(os.path.join(d, "m"))
    assert all(os.path.exists(f) for f in files)
    # 读入 J：把导出的 J 改成固定值再读回
    rows = open(files[1], encoding="utf-8-sig").read().splitlines()
    pj = os.path.join(d, "j.csv")
    open(pj, "w").write("pair,J_hole_meV,J_electron_meV\n" + "\n".join(f"{k},10,20" for k in range(len(rows) - 1)))
    m2 = Mo.Morphology(Mo.read_gro(GRO, {1})[0], sym, Y)
    m2.neighbors(4.0)
    m2.import_couplings(pj)
    assert np.allclose(m2.J["hole"], 0.01) and np.allclose(m2.J["electron"], 0.02)
    made = m.export_dimers(os.path.join(d, "dimers"), range(2))
    assert len(made) == 2 and os.path.exists(os.path.join(made[1], "dimer_AB.gjf"))
    r0 = m.kmc("hole", 0.092, 300.0, use_energies=False, walkers=300, steps=1500)
    r1 = m.kmc("hole", 0.092, 300.0, walkers=300, steps=1500)
    assert r1["mu_avg"] < r0["mu_avg"]
    out = compare_methods(m.graph("hole"), 0.092, 300.0, "hole", sigma_E=float(E["hole"].std()), walkers=200, steps=1000,
                          polar={"offsets": E["hole"], "sigma": 0.0})
    keys = {x["key"] for x in out["rows"]}
    assert {"kmc_marcus", "kmc_polar", "gdm", "kmc_marcus_dis"} <= keys
    assert not keys & {"marcus_scalar", "marcus_tensor", "tlt", "band"}
    assert "含空间关联" in next(x["note"] for x in out["rows"] if x["key"] == "kmc_polar")


def test_cli_morph_and_cif2gro():
    d = tempfile.mkdtemp()
    run = lambda *a: subprocess.run([sys.executable, "-m", "mobility_tool", *a], cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},  # noqa: E731
                                    text=True, timeout=900)
    g = os.path.join(d, "x.gro")
    p = run("cif2gro", "--cif", CIF, "--supercell", "2,2,1", "--resname", "PEN", "--out", g)
    assert p.returncode == 0, p.stderr
    assert len(Mo.read_gro(g)[0]["X"]) == 8 * 36
    p = run("morph", "--gro", GRO, "--frame", "2", "--monomer", REF, "--lam-h", "0.092", "--walkers", "200",
            "--steps", "800", "--out", os.path.join(d, "m"), "--quiet")
    assert p.returncode == 0, p.stderr
    assert "加位点能" in p.stdout and os.path.exists(os.path.join(d, "m.png"))


if __name__ == "__main__":
    for f in (test_gro_roundtrip_and_elements, test_mapping_neighbors_and_eht, test_kmc_matches_crystal_analytic,
              test_site_energies_export_and_compare, test_cli_morph_and_cif2gro):
        f()
        print("ok", f.__name__)
