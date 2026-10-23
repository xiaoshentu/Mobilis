"""极化位点能 (polar)：Thole 求解、线性响应能量、电荷对应、晶体 P±、MD σ_E、KMC 子格子位点能与命令行。"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool import polar as P  # noqa: E402
from mobility_tool.kmc import rate_marcus, run_kmc  # noqa: E402
from mobility_tool.mobility import KB_EV  # noqa: E402

CIF = os.path.join(ROOT, "tests", "data", "pentacene_RT.cif")


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
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    c = read_cif(CIF)
    return c, CEDS(c, find_molecules(c), cutoff=4.0)


def test_thole_solver_and_polarizability():
    a, r = 1.405, 4.0                                   # 两原子沿键轴：μ_tot = 2α/(1 − 2α/r³) (r > s，无阻尼)
    th = P.Thole(np.array([[0, 0, 0], [0, 0, r]]), np.array([a, a]), np.array([0, 0]))
    F = np.zeros((2, 3, 1)); F[:, 2, 0] = 1.0
    assert abs(th.solve(F, tol=1e-12)[:, 2, 0].sum() - 2 * a / (1 - 2 * a / r ** 3)) < 1e-9
    S, X = _benzene()
    A = P.molecular_polarizability(S, X)
    assert abs(A[0, 0] - A[1, 1]) < 1e-6 and A[2, 2] < A[0, 0]                    # 面内 > 面外
    assert 8.0 < np.trace(A) / 3 < 10.0
    table, s = P.scale_alpha(S, X, 10.39)
    assert abs(np.trace(P.molecular_polarizability(S, X, table)) / 3 - 10.39) < 0.01 and s > 1
    log = os.path.join(tempfile.mkdtemp(), "p.log")
    open(log, "w").write(" Exact polarizability:  80.000   0.000  80.000   0.000   0.000  40.000\n")
    assert abs(P.read_polarizability(log)[2, 2] - 40 * 0.529177210903 ** 3) < 1e-9


def test_linear_response_energy_matches_direct():
    c, ce = _pentacene()
    ch = P.default_charges(ce.sym, ce.Y)
    assert abs(ch["q0"].sum()) < 1e-9 and abs(ch["dq_h"].sum() - 1) < 1e-9 and abs(ch["dq_e"].sum() + 1) < 1e-9
    a1 = P.atomic_polarizabilities(ce.sym)
    Xm, _, ci = P.crystal_cluster(c["lattice"], ce.frames, ce.Y, 0, 8.0)
    e, _ = P._cluster_energies(Xm, ci, a1, ch["q0"], {"h": ch["dq_h"]})
    M, nat, _ = Xm.shape
    X = Xm.reshape(-1, 3)
    mol = np.repeat(np.arange(M), nat)
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    inter = mol[:, None] != mol[None]

    def U(q):
        th = P.Thole(X, np.tile(a1, M), mol)
        F = th.charge_field(q)[:, :, 0]
        mu = th.solve(F, tol=1e-11)[:, :, 0]
        return 0.5 * P.COUL * np.sum(np.where(inter, q[:, None] * q[None] / D, 0)) - 0.5 * P.COUL * (mu * F).sum()
    q0 = np.tile(ch["q0"], M)
    q1 = q0.copy()
    q1[ci * nat:(ci + 1) * nat] += ch["dq_h"]
    assert abs((U(q1) - U(q0)) - e["h"]["total"]) < 1e-6


def test_match_atoms_permuted_rotated():
    _, ce = _pentacene()
    S, X = list(ce.sym), np.asarray(ce.Y, float)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(S))
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("zyx", [0.7, -1.1, 2.3]).as_matrix()
    Xb = (X @ R.T + [3, -2, 1])[perm]
    Sb = [S[i] for i in perm]
    p, rmsd = P.match_atoms(S, X, Sb, Xb)
    assert rmsd < 1e-6 and all(Sb[p[i]] == S[i] for i in range(len(S)))
    assert np.allclose(np.linalg.norm(Xb[p] - Xb[p].mean(0), axis=1), np.linalg.norm(X - X.mean(0), axis=1))


def test_crystal_site_energies():
    c, ce = _pentacene()
    ch = P.default_charges(ce.sym, ce.Y)
    res = P.site_energies(c["lattice"], ce.frames, ce.sym, ce.Y, ch, radii=(7, 9, 11))
    assert res["P_plus"] > 0.3 and res["P_minus"] > 0.3
    assert abs(sum(res["offsets_h"])) < 1e-9 and len(res["offsets_h"]) == 2
    for s in res["sites"]:
        for car in ("h", "e"):
            x = s[car]
            assert abs(x["total"] - (x["static"] + x["cross"] + x["pol"])) < 1e-12
            assert x["pol"] < s["rows"][-1][car]["pol"]            # 外推后诱导项更负
    lv = P.solid_levels(res, 6.59, 1.39)
    assert abs(lv["E_transport"] - (6.59 - 1.39 - res["P_plus"] - res["P_minus"])) < 1e-9
    txt = P.summary_text(res, lv)
    assert "P+" in txt and "输运能隙" in txt
    out = os.path.join(tempfile.mkdtemp(), "pol.csv")
    P.export_csv(res, out, levels=lv)
    assert "P_plus_eV" in open(out, encoding="utf-8-sig").read()


def test_kmc_sublattice_offsets_exact_chain():
    """交替两位点链：D = 2a²/(1/k↑ + 1/k↓)。"""
    a, J, lam, T, dE = 4.0, 0.02, 0.15, 300.0, 0.05
    e = [(0, 1, np.array([a, 0, 0]), J, 1, (0, 0, 0)), (0, 1, np.array([-a, 0, 0]), J, 1, (-1, 0, 0)),
         (1, 0, np.array([a, 0, 0]), J, 1, (1, 0, 0)), (1, 0, np.array([-a, 0, 0]), J, 1, (0, 0, 0))]
    g = {"n_sites": 2, "edges": e, "periodic": True, "kind": "chain"}
    base = run_kmc(g, lam, T, walkers=400, steps=1500, seed=5)
    same = run_kmc(g, lam, T, walkers=400, steps=1500, seed=5, site_offsets=[0.1, 0.1])
    assert np.allclose(base["mu_tensor_cm2_Vs"], same["mu_tensor_cm2_Vs"])
    r = run_kmc(g, lam, T, walkers=2000, steps=3000, seed=5, site_offsets=[0.0, dE])
    ku, kd = rate_marcus(J, lam, dE, T), rate_marcus(J, lam, -dE, T)
    D = 2 * (a * 1e-8) ** 2 / (1 / ku + 1 / kd)
    mu = D / (KB_EV * T)
    assert abs(r["mu_tensor_cm2_Vs"][0, 0] / mu - 1) < 0.06
    assert r["mu_tensor_cm2_Vs"][0, 0] < 0.8 * base["mu_tensor_cm2_Vs"][0, 0]


def test_compare_polar_and_md_sigma():
    from mobility_tool.compare import compare_methods
    from mobility_tool.rigidmd import RigidCrystalMD, run_temperature
    c, ce = _pentacene()
    mean = {1: 77.0, 2: -42.0, 3: 32.0, 7: 29.0}
    res = {g: {"T": 300.0, "hole": {"mean": m, "std": 15.0}, "electron": {"mean": m, "std": 15.0}} for g, m in mean.items()}
    g = ce.graph(res, "hole")
    out = compare_methods(g, 0.092, 300.0, "hole", methods=("kmc_marcus", "kmc_polar"), walkers=300, steps=1500,
                          polar={"offsets": [-0.03, 0.03], "sigma": 0.03})
    by = {r["key"]: r for r in out["rows"]}
    assert by["kmc_polar"]["mu"] < by["kmc_marcus"]["mu"] and "σ_E = 30 meV" in by["kmc_polar"]["note"]
    assert "kmc_polar" not in {r["key"] for r in compare_methods(g, 0.092, 300.0, "hole", methods=("kmc_polar",))["rows"]}
    md = RigidCrystalMD(c, ce.frames, (ce.sym, ce.Y), (2, 2, 1))
    tr = P.traj_from_md(md, run_temperature(md, 300.0, dict(equil_ps=0.2, prod_ps=0.2)))
    ch = P.default_charges(tr["symbols"], tr["Y"])
    r = P.md_site_energies(tr, ch, n_frames=3, centers=2, R=8.0, nk=2)
    assert r["E_h"].shape == (3, 2) and 0 < r["sigma_h"] < 0.5 and set(r["k"].tolist()) == {0, 1}


def test_cli_polar():
    d = tempfile.mkdtemp()
    run = lambda *a: subprocess.run([sys.executable, "-m", "mobility_tool", "polar", "--cif", CIF, *a], cwd=ROOT,  # noqa: E731
                                    capture_output=True, text=True, timeout=600)
    p = run("--gjf", os.path.join(d, "pent"))
    assert p.returncode == 0, p.stderr
    txt = open(os.path.join(d, "pent_neutral.gjf")).read()
    assert "pop=(mk)" in txt and "polar" in txt and "0 1" in txt
    assert "1 2" in open(os.path.join(d, "pent_cation.gjf")).read()
    p = run("--radii", "7,9", "--ip-gas", "6.59", "--out", os.path.join(d, "pol.csv"), "--quiet")
    assert p.returncode == 0, p.stderr
    assert "P+" in p.stdout and os.path.exists(os.path.join(d, "pol.png"))


def test_extrapolation_rules():
    """默认 Born 尾项 (ε 给定时)；1/R 拟合作对照，没有 ε 时采用。小团簇的拟合斜率偏陡，Born 更接近真值。"""
    def rows(radii, pinf, eps, extra=0.0):
        out = []
        for R in radii:
            e_pol = -pinf + (1 - 1 / eps) * P.COUL / (2 * R) + extra / R ** 2     # 诱导能 = −P(R)
            out.append({"R": R, "h": {"static": 0.0, "cross": 0.0, "pol": e_pol}})
        return out
    ex = P.extrapolate(rows((18, 22, 26, 30), 1.0, 3.5), "h", eps_ll=3.5)          # 渐近区：两种外推都精确
    assert ex["method"] == "Born 尾项" and abs(ex["pol"] + 1.0) < 1e-9
    assert abs(ex["pol_fit"] + 1.0) < 1e-9 and abs(ex["eps_fit"] - 3.5) < 1e-6
    ex = P.extrapolate(rows((18, 22, 26, 30), 1.0, 3.5), "h")                      # 没有 ε：用 1/R 拟合
    assert ex["method"] == "1/R 拟合" and abs(ex["pol"] + 1.0) < 1e-9
    ex = P.extrapolate(rows((10, 14, 18, 22), 1.0, 3.5, extra=60.0), "h", eps_ll=3.5)    # 小团簇：斜率偏陡
    assert ex["method"] == "Born 尾项" and abs(ex["pol"] - ex["pol_born"]) < 1e-12
    assert abs(ex["pol"] + 1.0) < abs(ex["pol_fit"] + 1.0)                              # Born 比外推更接近真值
    assert 3.0 < P.lorentz_lorenz(42.8, 2 / 692.0) < 5.0


if __name__ == "__main__":
    for f in (test_thole_solver_and_polarizability, test_linear_response_energy_matches_direct,
              test_match_atoms_permuted_rotated, test_crystal_site_energies, test_kmc_sublattice_offsets_exact_chain,
              test_compare_polar_and_md_sigma, test_cli_polar, test_extrapolation_rules):
        f()
        print("ok", f.__name__)
