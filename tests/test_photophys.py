"""检验激发态解析、Stokes 位移、RMSD、光谱、k_r/k_nr 以及轨道片段组成。"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobility_tool.gaussian_io import ELEMENTS, read_xyz  # noqa: E402
from mobility_tool.orbcomp import orbital_composition, parse_ranges  # noqa: E402
from mobility_tool.photophys import (absorption_spectrum, analyze_excited, decay_rates, emission_band,  # noqa
                                     generate_excited_inputs, parse_excitations, rmsd)

EX = Path(__file__).resolve().parents[1] / "examples" / "benzene.xyz"


def orient(atoms):
    lines = ["                         Standard orientation:",
             " ---------------------------------------------------------------------",
             " Center     Atomic      Atomic             Coordinates (Angstroms)",
             " Number     Number       Type             X           Y           Z",
             " ---------------------------------------------------------------------"]
    for i, (s, x, y, z) in enumerate(atoms):
        lines.append(f" {i + 1:6d} {ELEMENTS.index(s):10d} {0:11d} {x:15.6f}{y:12.6f}{z:12.6f}")
    lines.append(" ---------------------------------------------------------------------")
    return "\n".join(lines) + "\n"


def td_block(states):
    out = []
    for n, (mult, e, f, cfg) in enumerate(states, 1):
        out.append(f" Excited State   {n}:      {mult}-A      {e:.4f} eV  {1239.841984 / e:.2f} nm  f={f:.4f}  <S**2>=0.000")
        out += [f"     {a} ->{b}         {c: .5f}" for a, b, c in cfg]
        out.append("")
    return "\n".join(out) + "\n"


def fake_log(geom_opt, td_sets_opt, td_sp):
    route1 = " -------\n #p opt td b3lyp/genecp\n -------\n"
    route2 = " -------\n #p td b3lyp/def2tzvp geom=check\n -------\n"
    s = route1
    for g, td in zip(geom_opt, td_sets_opt):         # 优化过程中多次打印
        s += orient(g) + td_block(td)
    s += " Normal termination of Gaussian 16\n" + route2 + orient(geom_opt[-1]) + td_block(td_sp)
    return s + " Normal termination of Gaussian 16\n"


def test_photophys():
    atoms = read_xyz(EX)
    rot = Rotation.from_euler("z", 40, degrees=True).as_matrix()
    x = np.array([a[1:] for a in atoms])
    rng = np.random.default_rng(0)
    disp = rng.normal(size=x.shape) * 0.05
    s1 = [(a[0], *p) for a, p in zip(atoms, (x + disp) @ rot.T + 2.0)]
    expect = np.sqrt(((disp - disp.mean(0)) ** 2).sum(1).mean())
    r = rmsd(atoms, s1)["rmsd"]
    assert abs(r - expect) < 0.01, (r, expect)       # 小位移下近似等于去掉平动后的位移
    assert rmsd(atoms, [(a[0], *p) for a, p in zip(atoms, x @ rot.T - 3)])["rmsd"] < 1e-9
    cfg = [("140", "141", 0.69), ("139", "141", 0.12)]
    with tempfile.TemporaryDirectory() as d:
        stems = generate_excited_inputs(EX, d, name="m", basis="6-311G(d,p)", sp_basis="def2-TZVP",
                                        solvent="dichloromethane")
        assert stems == ["m_S0", "m_S1", "m_T1"]
        t = (Path(d) / "m_T1.gjf").read_text()
        assert "td(triplets" in t and "%oldchk=m_S0.chk" in t and "scrf=(pcm,solvent=dichloromethane)" in t
        Path(d, "m_S0.log").write_text(fake_log([atoms], [[]], [("Singlet", 2.41, 0.0001, cfg),
                                                                  ("Singlet", 3.10, 0.35, cfg)]))
        Path(d, "m_S1.log").write_text(fake_log([atoms, s1], [[("Singlet", 2.30, 0.01, cfg)]] * 2,
                                                [("Singlet", 2.10, 0.0242, cfg)]))
        Path(d, "m_T1.log").write_text(fake_log([atoms, s1], [[("Triplet", 1.80, 0.0, cfg)]] * 2,
                                                [("Triplet", 1.65, 0.0, cfg)]))
        res = analyze_excited(d, "m", homo=140)
    tab = {r["kind"]: r for r in res["table"]}
    assert abs(tab["Absorption"]["E_eV"] - 2.41) < 1e-9 and abs(tab["Fluorescence"]["E_eV"] - 2.10) < 1e-9
    assert abs(tab["Phosphorescence"]["E_eV"] - 1.65) < 1e-9
    assert abs(res["stokes_shift_eV"] - 0.31) < 1e-9
    assert tab["Absorption"]["configs"][0][0] == "HOMO→LUMO" and abs(tab["Absorption"]["configs"][0][1] - 95.22) < 0.01
    assert abs(res["rmsd_S0_S1"] - r) < 1e-4 and not res["warnings"]
    nm, y = absorption_spectrum(res["absorption_states"], 0.3)
    assert abs(nm[np.argmax(y)] - 1239.841984 / 3.10) < 2
    nm, y = emission_band(1.65, 0.18)
    assert abs(nm[np.argmax(y)] - 751.4) < 1
    k = decay_rates(0.21, 1.4)
    assert abs(k["k_r"] - 1.50e5) < 1e3 and abs(k["k_nr"] - 5.64e5) < 1e3
    print(f"光物理 OK: 吸收 2.41 / 荧光 2.10 / 磷光 1.65 eV, Stokes 0.31 eV, RMSD(S0-S1) {r:.3f} Å, "
          f"k_r={k['k_r']:.2e}, k_nr={k['k_nr']:.2e}")


def write_shell_fchk(path, c, e, nocc, shells, shell_atom, z, coords):
    def arr(name, typ, vals):
        vals = list(vals)
        per, fmt = (6, "{:12d}") if typ == "I" else (5, "{:16.8E}")
        s = f"{name:<40s}   {typ}   N={len(vals):12d}\n"
        for i in range(0, len(vals), per):
            s += "".join(fmt.format(v) for v in vals[i:i + per]) + "\n"
        return s
    nbf = c.shape[0]
    t = "x\nSP        RB3LYP\n"
    for k, v in (("Number of atoms", len(z)), ("Number of alpha electrons", nocc), ("Number of basis functions", nbf),
                 ("Number of independent functions", nbf)):
        t += f"{k:<40s}   I     {v:12d}\n"
    t += arr("Atomic numbers", "I", z) + arr("Current cartesian coordinates", "R", np.ravel(coords))
    t += arr("Shell types", "I", shells) + arr("Shell to atom map", "I", shell_atom)
    t += arr("Alpha Orbital Energies", "R", e) + arr("Alpha MO coefficients", "R", c.T.ravel())
    Path(path).write_text(t)


def test_orbcomp():
    shells, shell_atom = [0, 1, 0, -2, 0, 0], [1, 1, 2, 2, 3, 4]     # s,p | s,5d | s | s → 4+6+1+1 = 12
    nbf = 12
    rng = np.random.default_rng(4)
    x = rng.normal(size=(nbf, nbf)) * 0.2
    s = x @ x.T + 2 * np.eye(nbf)
    dd = 1 / np.sqrt(np.diag(s))
    s *= np.outer(dd, dd)
    f = rng.normal(size=(nbf, nbf))
    e, c = eigh((f + f.T) / 2, s)
    z = [6, 77, 1, 1]
    coords = [[0, 0, 0], [0, 0, 2.0], [0, 1.1, -0.5], [5, 5, 5]]
    with tempfile.TemporaryDirectory() as d:
        write_shell_fchk(f"{d}/o.fchk", c, e, 3, shells, shell_atom, z, np.array(coords) / 0.529177210903)
        frags = {"F1": parse_ranges("1,3", 4), "F2": parse_ranges("2", 4), "F3": parse_ranges("4", 4)}
        mul = orbital_composition(f"{d}/o.fchk", frags, method="mulliken")
        sc = orbital_composition(f"{d}/o.fchk", frags, method="scpa")
        auto = orbital_composition(f"{d}/o.fchk")
    bf_atom = np.repeat(np.array(shell_atom) - 1, [1, 3, 1, 5, 1, 1])
    ref = c * (s @ c)
    for row in mul["rows"]:
        k = row["index"] - 1
        assert abs(sum(row["composition"].values()) - 100) < 1e-4
        exp_f2 = ref[bf_atom == 1, k].sum() * 100
        assert abs(row["composition"]["F2"] - exp_f2) < 1e-3, (row["composition"]["F2"], exp_f2)
    assert all(abs(sum(r["composition"].values()) - 100) < 1e-6 for r in sc["rows"])
    assert list(auto["fragments"])[0].startswith("Metal(Ir)")
    print("轨道组成 OK: Mulliken 与直接计算一致，SCPA/Mulliken 均归一，自动片段:", list(auto["fragments"]))


if __name__ == "__main__":
    test_photophys()
    test_orbcomp()
