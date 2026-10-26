"""ZINDO/S 与 MOO (zindo)：Slater 重叠、INDO/S 自洽场、MOO 与二聚体投影、Gaussian ZINDO 输入路径、
近简并能级耦合、形貌中的 ZINDO 耦合与命令行。"""
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool import zindo as Z  # noqa: E402
from mobility_tool.gaussian_io import BOHR2ANG  # noqa: E402

CIF = os.path.join(ROOT, "tests", "data", "pentacene_RT.cif")
REF = os.path.join(ROOT, "tests", "data", "pentacene_ref.xyz")
ZNUM = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}


def _benzene():
    S, X = [], []
    for k in range(6):
        t = k * np.pi / 3
        S.append("C"); X.append([1.397 * np.cos(t), 1.397 * np.sin(t), 0.0])
    for k in range(6):
        t = k * np.pi / 3
        S.append("H"); X.append([2.481 * np.cos(t), 2.481 * np.sin(t), 0.0])
    return S, np.array(X)


def _pentacene():
    from mobility_tool.cli import _read_monomer
    return _read_monomer(REF)


def _rot(a, b, c):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("zyx", [a, b, c]).as_matrix()


def write_zindo_fchk(path, sym, X, C, e_ev, nocc):
    """Gaussian 风格的 fchk (ZINDO：价层基组，轨道系数按 MO 连续存放)。"""
    def arr(name, typ, vals):
        vals = list(vals)
        per = 6 if typ == "I" else 5
        fmt = "{:12d}" if typ == "I" else "{:16.8E}"
        s = f"{name:<40s}   {typ}   N={len(vals):12d}\n"
        for i in range(0, len(vals), per):
            s += "".join(fmt.format(v) for v in vals[i:i + per]) + "\n"
        return s
    nbf = C.shape[0]
    txt = "synthetic zindo\nSP        RZINDO\n"
    txt += f"{'Number of atoms':<40s}   I     {len(sym):12d}\n"
    txt += f"{'Number of alpha electrons':<40s}   I     {nocc:12d}\n"
    txt += f"{'Number of basis functions':<40s}   I     {nbf:12d}\n"
    txt += arr("Atomic numbers", "I", [ZNUM[s] for s in sym])
    txt += arr("Current cartesian coordinates", "R", (np.asarray(X) / BOHR2ANG).ravel())
    txt += arr("Alpha Orbital Energies", "R", np.asarray(e_ev) / Z.AU2EV)
    txt += arr("Alpha MO coefficients", "R", C.T.ravel())
    Path(path).write_text(txt)


def test_sto_overlap_analytic():
    R = np.array([1.0, 3.0, 7.0])
    s, _ = Z.sto_overlap(1, 0, 1.2, 1, 0, 1.2, R)
    p = 1.2 * R
    assert np.abs(s - np.exp(-p) * (1 + p + p * p / 3)).max() < 1e-12

    def Ak(k, p):
        return math.exp(-p) * sum(math.factorial(k) / math.factorial(k - j) / p ** (j + 1) for j in range(k + 1))
    for Rv in (2.0, 5.0):
        p = 1.625 * Rv
        sg, pi = Z.sto_overlap(2, 1, 1.625, 2, 1, 1.625, [Rv])
        assert abs(pi[0] - p ** 5 / 120 * (5 * Ak(4, p) - 6 * Ak(2, p) + Ak(0, p))) < 1e-12
        assert abs(sg[0] - p ** 5 / 120 * (5 * Ak(4, p) - 18 * Ak(2, p) + 5 * Ak(0, p))) < 1e-12
    # 不同指数 (C–H) 与样条表一致
    sg, _ = Z.sto_overlap(2, 0, 1.625, 1, 0, 1.2, [3.0])
    S, _ = Z.weighted_overlap_block(["C"], [[0, 0, 0]], ["H"], [[0, 0, 3.0 * Z.AU2ANG]])
    assert abs(S[0, 0] - sg[0]) < 1e-7


def test_indo_benzene_and_invariance():
    S, X = _benzene()
    r = Z.indo_scf(S, X)
    n = r["nocc"]
    e = r["e"]
    assert abs(e[n - 1] - e[n - 2]) < 1e-6 and abs(e[n] - e[n + 1]) < 1e-6        # e1g / e2u 简并
    assert -10.0 < e[n - 1] < -8.0 and 0.0 < e[n] < 2.0
    qC = 4 - np.trace(r["P"][np.ix_(r["atom"] == 0, r["atom"] == 0)])
    assert abs(qC) < 0.05
    r2 = Z.indo_scf(S, X @ _rot(0.3, 1.1, -0.4).T + 2.0)
    assert abs(r2["energy"] - r["energy"]) < 1e-6 and np.allclose(r2["e"], r["e"], atol=1e-6)


def test_moo_vs_dimer_projection_and_invariance():
    sym, Y = _pentacene()
    moo = Z.MOO(sym, Y)
    for shift, tol in (([0, 0, 3.6], 0.02), ([1.2, 0, 3.5], 0.15), ([0, 1.2, 3.5], 0.06)):
        Xb = Y + np.array(shift)
        jm = moo.coupling_atoms(Y, Xb)
        jd = Z.dimer_projection(sym, Y, Xb)
        for c in ("hole", "electron"):
            assert abs(abs(jm[c]) - abs(jd[c])) < tol * abs(jd[c]) + 0.008, (shift, c, jm[c], jd[c])   # 节点附近用绝对误差
    # 整体转动、平移二聚体：|J| 不变
    Xb = Y + np.array([0.7, 0.4, 3.6])
    j0 = moo.coupling_atoms(Y, Xb)
    R, t = _rot(0.5, -0.8, 1.9), np.array([3.0, -1.0, 2.0])
    j1 = moo.coupling_atoms(Y @ R.T + t, Xb @ R.T + t)
    assert all(abs(abs(j0[c]) - abs(j1[c])) < 1e-6 for c in j0)


def test_gaussian_zindo_paths():
    """Gaussian ZINDO 的 fchk (原子顺序打乱、分子在另一取向) → MOO 与软件内 INDO/S 一致；
    Gaussian 二聚体投影的读取与软件内二聚体投影一致。"""
    sym, Y = _pentacene()
    d = tempfile.mkdtemp()
    R = _rot(0.9, 0.3, -1.2)
    perm = np.random.default_rng(4).permutation(len(sym))
    Xg = (Y @ R.T + [1.0, 2.0, -3.0])[perm]
    sg = [sym[k] for k in perm]
    g = Z.indo_scf(sg, Xg)
    write_zindo_fchk(f"{d}/mono.fchk", sg, Xg, g["C"], g["e"], g["nocc"])
    own, gau = Z.MOO(sym, Y), Z.MOO(sym, Y, f"{d}/mono.fchk")
    assert "Gaussian" in gau.source
    for shift in ([0, 0, 3.6], [1.2, 0.4, 3.5]):
        a, b = own.coupling_atoms(Y, Y + shift), gau.coupling_atoms(Y, Y + shift)
        assert all(abs(abs(a[c]) - abs(b[c])) < 1e-6 for c in a)
    Xb = Y + np.array([0.0, 1.2, 3.5])
    A, B = Z.indo_scf(sym, Y), Z.indo_scf(sym, Xb)
    D = Z.indo_scf(sym + sym, np.vstack([Y, Xb]))
    write_zindo_fchk(f"{d}/A.fchk", sym, Y, A["C"], A["e"], A["nocc"])
    write_zindo_fchk(f"{d}/B.fchk", sym, Xb, B["C"], B["e"], B["nocc"])
    write_zindo_fchk(f"{d}/AB.fchk", sym + sym, np.vstack([Y, Xb]), D["C"], D["e"], D["nocc"])
    jg, dev = Z.gaussian_zindo_projection(f"{d}/A.fchk", f"{d}/B.fchk", f"{d}/AB.fchk")
    jd = Z.dimer_projection(sym, Y, Xb)
    assert dev < 1e-6 and all(abs(abs(jg[c]) - abs(jd[c])) < 1e-6 for c in jg)


def test_degenerate_coupling():
    """简并轨道可任意幺正混合：单对耦合随之改变，VOTCA 的有效耦合 √(ΣJ²/N_A N_B) 不变。"""
    from mobility_tool.ti_analysis import degenerate_coupling, from_eht, level_matrix
    S, X = _benzene()
    B = X + np.array([0.9, 0.4, 3.6])
    d = from_eht([(s, *x) for s, x in zip(S, X)], [(s, *x) for s, x in zip(S, B)])
    k = d["noccB"] - 1
    eB = np.array(d["eB_diag"])
    assert abs(eB[k] - eB[k - 1]) < 1e-6
    moo = Z.MOO(S, X, degeneracy=0.01)
    assert len(moo.levels("hole")) == 2
    C0, cB0 = moo.C.copy(), d["cB"].copy()
    vals, singles, zvals, zsingle = [], [], [], []
    for ang in (0, 25, 70):
        th = np.radians(ang)
        U = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        d["cB"] = cB0.copy()
        d["cB"][:, [k - 1, k]] = cB0[:, [k - 1, k]] @ U
        jd, na, nb = degenerate_coupling(d, "hole", 0.01)
        assert na == nb == 2
        vals.append(jd)
        singles.append([r["J_eff_meV"] for r in level_matrix(d, 1) if r["A"] == "HOMO" and r["B"] == "HOMO"][0])
        L = moo.levels("hole")
        moo.C = C0.copy()
        moo.C[:, L] = C0[:, L] @ U                      # 同时改变 A、B 的规范 (同一单体轨道)
        zvals.append(moo.coupling(np.eye(3), [0.9, 0.4, 3.6])["hole"])
        moo.degeneracy = 0.0
        zsingle.append(moo.coupling(np.eye(3), [0.9, 0.4, 3.6])["hole"])
        moo.degeneracy = 0.01
    assert np.ptp(vals) < 1e-3 * abs(vals[0]) and np.ptp(singles) > 0.05 * abs(singles[0])   # 两能级 Löwdin 校正带来 ~1e-4 的偏差
    assert np.ptp(zvals) < 1e-9 and np.ptp(zsingle) > 0.05 * abs(zsingle[0])


def test_morphology_zindo_and_cli():
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    from mobility_tool.morphology import Morphology, crystal_frame
    c = read_cif(CIF)
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    m = Morphology(crystal_frame(c["lattice"], ce.frames, ce.sym, ce.Y, (3, 3, 2), "PEN"), ce.sym, ce.Y)
    m.neighbors(4.0)
    Jh, Je = m.couplings_zindo()
    moo = Z.MOO(ce.sym, ce.Y)
    for k in (0, 7, 30):
        a, b = m.pair_atoms(k)
        r = moo.coupling_atoms(a, b)
        assert abs(abs(r["hole"]) - abs(Jh[k])) < 1e-6 and abs(abs(r["electron"]) - abs(Je[k])) < 1e-6
    assert m.J_source.startswith("ZINDO") and np.abs(Jh).max() > 0.05
    d = tempfile.mkdtemp()
    run = lambda *a: subprocess.run([sys.executable, "-m", "mobility_tool", *a], cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},  # noqa: E731
                                    text=True, timeout=900)
    from mobility_tool.gaussian_io import write_xyz
    sym, Y = _pentacene()
    write_xyz(f"{d}/dimer.xyz", [(s, *x) for s, x in zip(sym, Y)] + [(s, *x) for s, x in zip(sym, Y + [0, 0, 3.6])])
    p = run("zindo", "pair", "--dimer", f"{d}/dimer.xyz")
    assert p.returncode == 0, p.stderr
    assert "ZINDO/MOO" in p.stdout and "二聚体 INDO/S" in p.stdout
    p = run("zindo", "bench-inputs", "--cif", CIF, "--out", f"{d}/bench")
    assert p.returncode == 0, p.stderr
    assert (Path(d) / "bench/cofacial_3.4/zindo_AB.gjf").exists() and (Path(d) / "bench/run_all.bat").exists()
    assert "ZINDO" in (Path(d) / "bench/mono_zindo.gjf").read_text()
    p = run("zindo", "bench", "--dir", f"{d}/bench")
    assert p.returncode == 0, p.stderr
    assert "R²" in p.stdout and (Path(d) / "bench/zindo_benchmark.png").exists()
    p = run("morph", "--gro", os.path.join(ROOT, "tests/data/pentacene_300K.gro"), "--monomer", REF, "--coupling",
            "zindo", "--site-energy", "none", "--quiet")
    assert p.returncode == 0, p.stderr
    assert "ZINDO/MOO" in p.stdout


if __name__ == "__main__":
    for f in (test_sto_overlap_analytic, test_indo_benzene_and_invariance, test_moo_vs_dimer_projection_and_invariance,
              test_gaussian_zindo_paths, test_degenerate_coupling, test_morphology_zindo_and_cli):
        f()
        print("ok", f.__name__)
