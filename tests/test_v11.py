"""原子 / 基团取代 (单分子、晶体对称取代、分子位于对称中心、仅所选分子)。"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import mobility_tool  # noqa: E402,F401
from mobility_tool.crystal import find_dimers, find_molecules, read_cif  # noqa: E402
from mobility_tool.gaussian_io import read_xyz  # noqa: E402
from mobility_tool.substitution import (GROUPS, attach_group, crystal_min_contact, neighbors,  # noqa: E402
                                        substitute, substitute_crystal, write_cif_p1)


def symmetry_dev(crys, mols):
    lat = np.asarray(crys["lattice"])
    inv = np.linalg.inv(lat)
    S = [s for m in mols for s in m["symbols"]]
    F = np.vstack([np.asarray(m["cart"]) @ inv for m in mols]) % 1
    worst = 0.0
    for R, t in crys["symops"]:
        img = (F @ np.asarray(R).T + t) % 1
        d = img[:, None] - F[None]
        d -= np.round(d)
        dd = np.linalg.norm(d @ lat, axis=-1)
        for i in range(len(S)):
            worst = max(worst, dd[i, [j for j in range(len(S)) if S[j] == S[i]]].min())
    return worst


def test_molecule():
    a = read_xyz(ROOT / "examples/benzene.xyz")
    sym, X = [x[0] for x in a], np.array([x[1:] for x in a])
    ci = sym.index("C")
    hi = [k for k in neighbors(sym, X, ci) if sym[k] == "H"][0]
    s1, X1, _, _ = substitute(sym, X, [ci], "N")
    assert (s1.count("C"), s1.count("N"), s1.count("H")) == (5, 1, 5)          # 吡啶
    s2, X2, _, _ = substitute(s1, X1, [s1.index("N")], "C")
    assert (s2.count("C"), s2.count("H")) == (6, 6)                             # 补回 H
    s3, X3, _, _ = substitute(sym, X, [hi], "F")
    j = neighbors(s3, X3, hi)[0]
    assert abs(np.linalg.norm(X3[hi] - X3[j]) - 1.33) < 0.01
    for g in GROUPS:
        s4, X4, _, _ = attach_group(sym, X, hi, g)
        assert len(s4) == 11 + len(GROUPS[g][0]) + (3 if g == "OCH3" else 0)
        d = np.linalg.norm(X4[:, None] - X4[None], axis=-1) + np.eye(len(X4)) * 9
        assert d.min() > 0.9, g                                                 # 没有原子重叠
    print("单分子取代 OK: 吡啶 (删 H)、补 H、C–F 键长、12 种基团几何")


def test_crystal_general():
    from test_crystal import make_cif
    with tempfile.TemporaryDirectory() as d:
        make_cif(f"{d}/t.cif")
        c = read_cif(f"{d}/t.cif")
        m = find_molecules(c)
        s0, X0 = m[0]["symbols"], np.asarray(m[0]["cart"])
        ci = s0.index("C")
        hi = [k for k in neighbors(s0, X0, ci) if s0[k] == "H"][0]
        for kw, at in [(dict(new="N"), [ci]), (dict(new="Br"), [hi])] + [(dict(group=g), [hi]) for g in GROUPS]:
            c2, m2, _ = substitute_crystal(c, m, 0, at, **kw)
            assert symmetry_dev(c2, m2) < 1e-6, kw
        c3, m3, notes = substitute_crystal(c, m, 0, [hi], new="Cl", symmetric=False)
        assert sum(x["symbols"].count("Cl") for x in m3) == 1 and len(c3["symops"]) == 1
        c2, m2, _ = substitute_crystal(c, m, 0, [ci], new="N")
        assert find_dimers(c2, m2, cutoff=4.5)["groups"]
        write_cif_p1(f"{d}/s.cif", c2, m2)
        assert {len(x["symbols"]) for x in find_molecules(read_cif(f"{d}/s.cif"))} == {11}
        assert np.isfinite(crystal_min_contact(c2, m2)[0])
    print("晶体对称取代 OK: P2₁/c 中元素与 12 种基团取代后对称性严格保持；仅所选分子模式；导出 P1 CIF")


def test_site_symmetry():
    """分子位于反演中心 (P-1, Z=1，不对称单元为半个苯)：取代一个 H 时其反演对称的 H 必须同时取代。"""
    a = read_xyz(ROOT / "examples/benzene.xyz")
    X = np.array([x[1:] for x in a])
    X -= X.mean(0)
    sym = [x[0] for x in a]
    cell = np.diag([7.5, 8.0, 6.5])
    half = [k for k in range(len(sym)) if (X[k] @ [1, 0.37, 0.11]) > 0]       # 每对反演原子只取一个
    lines = ["data_t", "_cell_length_a 7.5", "_cell_length_b 8.0", "_cell_length_c 6.5", "_cell_angle_alpha 90",
             "_cell_angle_beta 90", "_cell_angle_gamma 90", "loop_", "_symmetry_equiv_pos_as_xyz", "'x, y, z'",
             "'-x, -y, -z'", "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
             "_atom_site_fract_y", "_atom_site_fract_z"]
    for k in half:
        f = X[k] @ np.linalg.inv(cell) + 0.5
        lines.append(f"{sym[k]}{k} {sym[k]} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f}")
    with tempfile.TemporaryDirectory() as d:
        Path(f"{d}/p1.cif").write_text("\n".join(lines) + "\n")
        c = read_cif(f"{d}/p1.cif")
        m = find_molecules(c)
        assert len(m) == 1 and len(m[0]["symbols"]) == 12
        s0, X0 = m[0]["symbols"], np.asarray(m[0]["cart"])
        hi = s0.index("H")
        for kw in (dict(new="F"), dict(group="CH3"), dict(group="CN")):
            c2, m2, notes = substitute_crystal(c, m, 0, [hi], **kw)
            assert symmetry_dev(c2, m2) < 1e-6, kw
            el = kw.get("new") or GROUPS[kw["group"]][0][0][0]
            assert sum(1 for s in m2[0]["symbols"] if s == el) >= 2, (kw, m2[0]["symbols"])
    print("位点对称 OK: 分子位于反演中心时，所选原子与其反演对称原子同时取代 (F、CH₃、CN)，对称性保持")


if __name__ == "__main__":
    test_molecule()
    test_crystal_general()
    test_site_symmetry()
