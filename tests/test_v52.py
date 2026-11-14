"""TVCF rates (v1.31.0): FC, HT and promoting-mode correlation functions with frequency change and Duschinsky
rotation against exact sums over vibrational states; displaced oscillators against tunneling.FGRRate; detailed
balance between forward and reverse rates; TADF kinetics; CLI on fchk files."""
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility_tool import tvcf  # noqa: E402
from mobility_tool.mobility import HBAR_EVS, KB_EV  # noqa: E402

TAU_FS = 60.0


def ho(w, x, nmax):
    """Harmonic-oscillator eigenfunctions φ_n(x) (ħ = m = 1) and their derivatives, n = 0..nmax."""
    phi = np.zeros((nmax + 2,) + x.shape)
    phi[0] = (w / np.pi) ** 0.25 * np.exp(-w * x ** 2 / 2)
    for n in range(nmax + 1):
        phi[n + 1] = math.sqrt(2 * w / (n + 1)) * x * phi[n] - (math.sqrt(n / (n + 1)) * phi[n - 1] if n else 0)
    d = np.zeros_like(phi[:nmax + 1])
    for n in range(nmax + 1):
        d[n] = math.sqrt(w / 2) * ((math.sqrt(n) * phi[n - 1] if n else 0) - math.sqrt(n + 1) * phi[n + 1])
    return phi[:nmax + 1], d


def exact(pair, T, dEs, G=None, g0=1.0, N=None, ni=6, nf=16, L=26.0, npts=260):
    """Σ_v P_v |⟨f u|Ô|i v⟩|² 2π g(ΔE + ε_v − ε_u) / ħ with a Gaussian g of σ = √2/τ (the TVCF damping)."""
    n = pair.n
    xs = np.linspace(-L, L, npts)
    dx = xs[1] - xs[0]
    grid = np.meshgrid(*([xs] * n), indexing="ij")
    X = np.stack([g.ravel() for g in grid])                     # (n, M) initial coordinates
    Xf = pair.J @ X + pair.D[:, None]
    beta = 1 / (KB_EV * T)
    # initial states
    import itertools
    iv = list(itertools.product(range(ni), repeat=n))
    fu = list(itertools.product(range(nf), repeat=n))
    phi_i, dphi_i = zip(*[ho(pair.w_i[k], X[k], ni) for k in range(n)])
    phi_f = [ho(pair.w_f[k], Xf[k], nf)[0] for k in range(n)]
    Psi_f = np.array([np.prod([phi_f[k][u[k]] for k in range(n)], 0) for u in fu])
    e_i = np.array([sum(pair.w_i[k] * (v[k] + 0.5) for k in range(n)) for v in iv])
    e_f = np.array([sum(pair.w_f[k] * (u[k] + 0.5) for k in range(n)) for u in fu])
    P = np.exp(-beta * (e_i - e_i.min()))
    P /= P.sum()
    vol = dx ** n
    sig = math.sqrt(2) / (TAU_FS / tvcf.HBAR_FS)
    out = np.zeros(len(dEs))
    for a, v in enumerate(iv):
        psi = np.prod([phi_i[k][v[k]] for k in range(n)], 0)
        if N is not None:
            dpsi = sum(N[k] * np.prod([(dphi_i[m][v[m]] if m == k else phi_i[m][v[m]]) for m in range(n)], 0)
                       for k in range(n))
            M2 = (Psi_f @ dpsi * vol) ** 2
        elif G is not None:
            M2 = (Psi_f @ ((g0 + G @ X) * psi) * vol) ** 2
        else:
            M2 = (Psi_f @ psi * vol) ** 2
        for b, dE in enumerate(dEs):
            x = dE + e_i[a] - e_f
            out[b] += P[a] * (M2 * np.exp(-x ** 2 / (2 * sig ** 2))).sum() / (math.sqrt(2 * np.pi) * sig)
    return 2 * np.pi * out / HBAR_EVS


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def check_against_exact():
    T = 300.0
    cases = {
        "1D, frequency change": tvcf.Pair([0.15], [0.11], None, [2.2]),
        "2D, Duschinsky 25°": tvcf.Pair([0.15, 0.08], [0.12, 0.10], rot(math.radians(25)), [2.0, 1.6]),
    }
    for name, p in cases.items():
        dEs = [0.25, 0.4]
        cor = tvcf.Correlator(p, T, TAU_FS)
        G = np.array([0.7, -0.4][:p.n])
        N = np.array([0.3, 0.5][:p.n])
        C = cor.run({"fc": "fc", "ht": ("coord", 0.8, G), "nac": ("nac", N)})
        kw = dict(ni=5 if p.n == 2 else 8, nf=15 if p.n == 2 else 30, npts=200 if p.n == 2 else 1200)
        ref = {"fc": exact(p, T, dEs, **kw), "ht": exact(p, T, dEs, G=G, g0=0.8, **kw),
               "nac": exact(p, T, dEs, N=N, **kw)}
        for k in ("fc", "ht", "nac"):
            got = cor.rate(C[k], dEs)
            dev = np.abs(got / ref[k] - 1)
            print(f"  {name:22s} {k:4s}  TVCF {got[0]:.4e} {got[1]:.4e}   exact {ref[k][0]:.4e} {ref[k][1]:.4e}"
                  f"   max dev {dev.max():.2e}")
            assert dev.max() < 2e-3, (name, k, dev)


def check_displaced_vs_fgr():
    from mobility_tool.tunneling import FGRRate
    f = np.array([1600.0, 1200.0, 450.0, 120.0])
    S = np.array([0.6, 0.3, 0.4, 0.8])
    p = tvcf.Pair.displaced(f, S)
    for T in (100.0, 300.0):
        cor = tvcf.Correlator(p, T, 500.0)
        rho = cor.run()["fc"]
        fgr = FGRRate(list(zip(f, S)), T, damping_fs=500.0, emax=1.2, de=0.001)
        for dE in (0.0, 0.3, 0.8):
            a = cor.rate(rho, dE)[0]
            b = float(fgr.k_over_v2(-dE))
            print(f"  displaced T={T:.0f} ΔE={dE}: TVCF {a:.4e}  FGRRate {b:.4e}")
            assert abs(a / b - 1) < 5e-3


def check_detailed_balance_and_T0():
    """Eight modes, frequency changes and a full Duschinsky rotation: k(i→f, ΔE) / k(f→i, −ΔE) = e^{βΔE} Z_f/Z_i."""
    rng = np.random.default_rng(3)
    wi = np.array([0.17, 0.13, 0.09, 0.05, 0.03, 0.02, 0.012, 0.008])
    wf = wi * rng.uniform(0.85, 1.15, 8)
    Q, _ = np.linalg.qr(np.eye(8) + rng.normal(size=(8, 8)) * 0.15)
    Q *= np.sign(np.diag(Q))
    D = np.sqrt(2 * np.array([0.5, 0.3, 0.4, 0.6, 1.0, 1.2, 1.5, 2.0]) / wf)
    p = tvcf.Pair(wi, wf, Q, D)
    q = p.swapped()
    T = 300.0
    beta = 1 / (KB_EV * T)
    lz = lambda w: (-beta * w / 2 - np.log1p(-np.exp(-beta * w))).sum()  # noqa: E731
    c, cq = tvcf.Correlator(p, T, 1000.0), tvcf.Correlator(q, T, 1000.0)
    rf, rb = c.run()["fc"], cq.run()["fc"]
    for dE in (0.1, 0.3):
        ratio = c.rate(rf, dE)[0] / cq.rate(rb, -dE)[0]
        expect = math.exp(beta * dE + lz(p.w_f) - lz(p.w_i))
        print(f"  detailed balance ΔE={dE}: k_fwd/k_bwd = {ratio:.4g}, expected {expect:.4g}")
        assert abs(ratio / expect - 1) < 5e-3
    c0 = tvcf.Correlator(p, 0.0, 500.0)
    c1 = tvcf.Correlator(p, 1.0, 500.0)
    a, b = c0.rate(c0.run()["fc"], 0.4)[0], c1.rate(c1.run()["fc"], 0.4)[0]
    assert abs(a / b - 1) < 1e-6, (a, b)


def check_kinetics_and_radiative():
    k = tvcf.tadf_kinetics(k_rS=1e7, k_icS=1e6, k_isc=5e7, k_risc=1e5, k_rT=0, k_nrT=1e3)
    # independent integration of the rate equations
    from scipy.integrate import solve_ivp
    M = np.array([[-(1e7 + 1e6 + 5e7), 1e5], [5e7, -(1e5 + 1e3)]])
    sol = solve_ivp(lambda t, y: M @ y, (0, 2e-3), [1.0, 0.0], method="LSODA", rtol=1e-10, atol=1e-14,
                    dense_output=True)
    ts = np.concatenate([np.linspace(0, 1e-6, 20001), np.linspace(1e-6, 2e-3, 200001)[1:]])
    y = sol.sol(ts)[0]
    total = np.trapezoid(1e7 * y, ts) if hasattr(np, "trapezoid") else np.trapz(1e7 * y, ts)
    print(f"  TADF: PLQY {k['PLQY_total']:.4f} (rate equations {total:.4f}), τ_p {k['tau_prompt_s']:.3e} s, "
          f"τ_d {k['tau_delayed_s']:.3e} s")
    assert abs(k["PLQY_total"] / total - 1) < 2e-3
    # radiative: zero displacement → k_r = K_RAD ΔE³ μ²
    p = tvcf.Pair([0.15, 0.1], [0.15, 0.1], None, [0.0, 0.0])
    r = tvcf.radiative(p, 300.0, 2.5, 1.0)
    print(f"  k_r(no displacement) = {r['k_r']:.4e}, Einstein {tvcf.K_RAD * 2.5 ** 3:.4e}")
    assert abs(r["k_r"] / (tvcf.K_RAD * 2.5 ** 3) - 1) < 5e-3


def make_fchks(d):
    """Benzene with a model force field: S0, S1 (displaced, softer), T1 (displaced differently), S1 rotated."""
    from scipy.spatial.transform import Rotation
    from test_decomposition import EXAMPLE, model_hessian, write_freq_fchk
    from mobility_tool.gaussian_io import BOHR2ANG, read_xyz
    atoms = read_xyz(EXAMPLE)
    sym = [a[0] for a in atoms]
    x0 = np.array([a[1:] for a in atoms])
    h0, b0 = model_hessian(sym, x0)
    rng = np.random.default_rng(5)
    out = {}
    for name, scale, e_ev, seed in (("s1", 0.9, 3.0, 1), ("t1", 0.95, 2.85, 2)):
        r = np.random.default_rng(seed)
        dq = r.normal(size=b0.shape[0]) * 0.012
        dx = b0.T @ np.linalg.pinv(b0 @ b0.T) @ dq
        x = x0 + dx.reshape(-1, 3) * BOHR2ANG
        h, _ = model_hessian(sym, x)
        h = h * scale
        out[name] = (x, h, -232.0 + e_ev / 27.211386245988, dx)
    write_freq_fchk(f"{d}/s0.fchk", sym, x0, h0, -232.0)
    for name, (x, h, e, _) in out.items():
        write_freq_fchk(f"{d}/{name}.fchk", sym, x, h, e)
    rot = Rotation.from_euler("xyz", [20, -35, 60], degrees=True).as_matrix()
    x, h, e, _ = out["s1"]
    R = np.kron(np.eye(len(sym)), rot)
    write_freq_fchk(f"{d}/s1_rot.fchk", sym, x @ rot.T + np.array([1.0, 2.0, -1.0]), R @ h @ R.T, e)
    nac = rng.normal(size=(len(sym), 3)) * 0.05
    np.savetxt(f"{d}/nac.txt", nac)
    np.savetxt(f"{d}/nac_rot.txt", nac @ rot.T)
    return out, h0, sym


def check_fchk_and_cli():
    from mobility_tool.gaussian_io import HARTREE2EV
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as d:
        out, h0, sym = make_fchks(d)
        p = tvcf.Pair.from_fchk(f"{d}/s1.fchk", f"{d}/s0.fchk")
        dx = out["s1"][3]
        lam_f = 0.5 * dx @ h0 @ dx * HARTREE2EV
        lam_i = 0.5 * dx @ out["s1"][1] @ dx * HARTREE2EV
        print(f"  fchk: λ_f {p.lambda_f():.5f} (expected {lam_f:.5f}), λ_i {p.lambda_i():.5f} ({lam_i:.5f}), "
              f"ΔE {p.dE_fchk:.4f} eV, |J Jᵀ − 1| {np.abs(p.J @ p.J.T - np.eye(p.n)).max():.1e}")
        assert abs(p.lambda_f() / lam_f - 1) < 1e-3 and abs(p.lambda_i() / lam_i - 1) < 1e-3
        assert abs(p.dE_fchk - 3.0) < 1e-6
        # rotated and translated S1 (geometry, Hessian and NAC): identical rates
        q = tvcf.Pair.from_fchk(f"{d}/s1_rot.fchk", f"{d}/s0.fchk")
        na = p.project_cartesian(np.loadtxt(f"{d}/nac.txt"))
        nb = q.project_cartesian(np.loadtxt(f"{d}/nac_rot.txt"))
        ra = tvcf.decay_rates(p, 300, 3.0, mu=1.0, nac=na)
        rb = tvcf.decay_rates(q, 300, 3.0, mu=1.0, nac=nb)
        print(f"  benzene model S1→S0: k_r {ra['k_r']:.4e} / {rb['k_r']:.4e}, k_IC {ra['k_IC']:.4e} / {rb['k_IC']:.4e}"
              f" s⁻¹, {ra['n_time_steps']} steps")
        assert abs(ra["k_r"] / rb["k_r"] - 1) < 1e-6 and abs(ra["k_IC"] / rb["k_IC"] - 1) < 1e-4
        env = dict(os.environ, MOBILIS_LANG="zh", PYTHONIOENCODING="utf-8")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run([sys.executable, "-m", "mobility_tool", "tvcf", "--initial", f"{d}/s1.fchk", "--final",
                            f"{d}/s0.fchk", "--fosc", "0.05", "--nac", f"{d}/nac.txt", "--temps", "100,300",
                            "--out", f"{d}/b"], cwd=root, capture_output=True, text=True, env=env)
        print(r.stdout[-600:], r.stderr[-1500:])
        assert r.returncode == 0 and os.path.exists(f"{d}/b_rates.csv") and os.path.exists(f"{d}/b_spectrum.csv")
        r = subprocess.run([sys.executable, "-m", "mobility_tool", "tadf", "--s0", f"{d}/s0.fchk", "--s1",
                            f"{d}/s1.fchk", "--t1", f"{d}/t1.fchk", "--fosc", "0.02", "--soc", "0.8", "--soc-t1s0",
                            "0.3", "--out", f"{d}/tadf.json"], cwd=root, capture_output=True, text=True, env=env)
        print(r.stdout[-900:], r.stderr[-1500:])
        assert r.returncode == 0 and "PLQY" in r.stdout
        import json
        res = json.load(open(f"{d}/tadf.json"))["results"][0]
        # detailed balance between ISC and RISC through the same S1/T1 pair (1/3 for the triplet sublevels)
        assert res["k_RISC"] < res["k_ISC"]


if __name__ == "__main__":
    check_against_exact()
    check_displaced_vs_fgr()
    check_detailed_balance_and_T0()
    check_kinetics_and_radiative()
    check_fchk_and_cli()
    print("test_v52 ok")
