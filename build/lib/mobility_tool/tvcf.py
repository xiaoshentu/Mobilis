# -*- coding: utf-8 -*-
"""Thermal vibration correlation function (TVCF) rates for excited-state decay: radiative rate and emission spectrum
(Franck–Condon and Herzberg–Teller), internal conversion (promoting modes, non-adiabatic coupling), intersystem
crossing and reverse intersystem crossing (spin–orbit coupling, FC and HT), and the S1/T1 kinetics of TADF emitters.

Harmonic potentials with frequency changes and Duschinsky rotation, Q_f = J Q_i + D (i: initial, f: final state),
following Q. Peng, Y. Yi, Z. Shuai, J. Shao, J. Chem. Phys. 126, 114302 (2007); Y. Niu, Q. Peng, C. Deng, X. Gao,
Z. Shuai, J. Phys. Chem. A 114, 7817 (2010); Q. Peng, Y. Niu, Q. Shi, X. Gao, Z. Shuai, J. Chem. Theory Comput. 9,
1132 (2013).

Every rate is a golden-rule expression  k = (1/ħ) ∫ dt e^{iΔE t/ħ} C(t),

    C(t) = Z_i⁻¹ Tr[ e^{−iτ_i h_i} Ô₁ e^{−iτ_f h_f} Ô₂ ],    τ_f = t,  τ_i = −iħβ − t,

with ΔE = E_i − E_f the adiabatic energy gap and Ô = 1 (FC), a coordinate (HT) or a momentum (promoting mode). In
the coordinate representation the trace is a Gaussian integral over two copies of the nuclear coordinates (x, y);
with u = (x + y)/√2 and v = (x − y)/√2 it factorizes into two N × N problems,

    A_u = −i[(B_i − A_i) + Jᵀ(B_f − A_f)J],   A_v = −i[(B_i + A_i) + Jᵀ(B_f + A_f)J],
    a = ω / sin(ωτ),  b = ω / tan(ωτ)   (Mehler kernels),

so that ρ(t) = C_FC(t) = √W · exp[c + ½ b_uᵀ A_u⁻¹ b_u] and the HT / promoting-mode correlations follow from the first
and second moments of that Gaussian. The branch of √W is followed continuously in t from ρ(0) = 1. A Gaussian damping
exp[−(t/τ_d)²] (τ_d = 500 fs, an energy broadening of ≈ 2 meV) makes the integrals converge.

Units inside: ħ = 1, energies in eV, time in ħ/eV (0.658 fs), mass-weighted coordinates q with h = ½(p² + ω²q²).
Checked in tests/test_v52.py against exact sums over vibrational states (one and two modes, frequency change,
Duschinsky rotation, HT and promoting modes), against tunneling.FGRRate for displaced oscillators, and against
detailed balance between forward and reverse rates."""
import csv
import math

import numpy as np

from .mobility import HBAR_EVS, KB_EV

EV_AU = 27.211386245988             # eV per hartree
AMU_ME = 1822.888486209
HBAR_FS = 0.6582119569              # fs per (ħ/eV)
CM_EV = 1 / 8065.543937
DEBYE_AU = 0.393430307              # e·a0 per debye
# Einstein A coefficient: k_r = K_RAD · n² · (E/eV)³ · |μ/(e a0)|²   (s⁻¹)
_E, _A0, _EPS0, _HBAR_JS, _C = 1.602176634e-19, 5.29177210903e-11, 8.8541878128e-12, 1.054571817e-34, 2.99792458e8
K_RAD = _E ** 3 * (_E * _A0) ** 2 / (3 * math.pi * _EPS0 * _HBAR_JS ** 4 * _C ** 3)
Q_AU_TO_EV = 1 / math.sqrt(EV_AU)   # q (ħ = 1, eV units) per mass-weighted a.u. coordinate (√m_e · a0)


# ----------------------------------------------------------------------------------------------
# harmonic pair of electronic states
# ----------------------------------------------------------------------------------------------
class Pair:
    """Two harmonic electronic states: frequencies ħω_i, ħω_f (eV), Duschinsky matrix J and shift D (Q_f = J Q_i + D,
    D in q units). Built from two Gaussian fchk files, from a displaced-oscillator mode table or directly."""

    def __init__(self, w_i, w_f, J=None, D=None, label_i="i", label_f="f", warnings=()):
        self.w_i = np.asarray(w_i, float)
        self.w_f = np.asarray(w_f, float)
        n = len(self.w_i)
        if len(self.w_f) != n:
            raise ValueError("the two states need the same number of modes")
        self.J = np.eye(n) if J is None else np.asarray(J, float)
        self.D = np.zeros(n) if D is None else np.asarray(D, float)
        if np.any(self.w_i <= 0) or np.any(self.w_f <= 0):
            raise ValueError("all frequencies must be positive (imaginary or zero modes: use floor_cm)")
        self.label_i, self.label_f = label_i, label_f
        self.warnings = list(warnings)
        off = self.J - np.diag(np.diag(self.J))
        self.diagonal = bool(np.abs(off).max() < 1e-12) if n else True
        self.detJ = abs(float(np.linalg.det(self.J))) if n else 1.0

    @property
    def n(self):
        return len(self.w_i)

    def huang_rhys_f(self):
        """Huang–Rhys factors of the final-state modes, S = ω_f D² / 2."""
        return 0.5 * self.w_f * self.D ** 2

    def lambda_f(self):
        """Reorganization energy on the final surface (eV)."""
        return float(0.5 * (self.w_f ** 2 * self.D ** 2).sum())

    def lambda_i(self):
        """Reorganization energy on the initial surface (eV): displacement of the final minimum in Q_i."""
        di = np.linalg.solve(self.J, -self.D)
        return float(0.5 * (self.w_i ** 2 * di ** 2).sum())

    def swapped(self):
        """The same two states with initial and final exchanged (Q_i = J⁻¹(Q_f − D))."""
        Jinv = np.linalg.inv(self.J)
        return Pair(self.w_f, self.w_i, Jinv, -Jinv @ self.D, self.label_f, self.label_i, self.warnings)

    @classmethod
    def displaced(cls, freq_cm, S, label_i="i", label_f="f"):
        """Same frequencies in both states, no rotation, Huang–Rhys factors S."""
        w = np.asarray(freq_cm, float) * CM_EV
        S = np.asarray(S, float)
        return cls(w, w, None, np.sqrt(2 * S / w), label_i, label_f)

    @classmethod
    def from_fchk(cls, initial, final, floor_cm=0.0, label_i=None, label_f=None):
        """Normal modes of both states from their freq fchk files (same atoms, same order). The final geometry and
        its modes are rotated onto the initial one (mass-weighted Kabsch). Imaginary modes are taken as real with
        a warning; modes below floor_cm are raised to floor_cm."""
        from pathlib import Path
        from .decomposition import kabsch_align, normal_modes
        from .gaussian_io import BOHR2ANG, read_fchk_hessian
        si, sf = read_fchk_hessian(initial), read_fchk_hessian(final)
        if si["symbols"] != sf["symbols"]:
            raise ValueError("the two fchk files have different atoms or atom order")
        mass = si["masses"]
        _, li, w2i, q = normal_modes(si["hessian"], mass, si["coords"])
        _, lf, w2f, _ = normal_modes(sf["hessian"], sf["masses"], sf["coords"])
        xf, rot = kabsch_align(si["coords"], sf["coords"], mass, return_rotation=True)
        nat = len(mass)
        lf = (lf.T.reshape(-1, nat, 3) @ rot).reshape(lf.shape[1], -1).T
        lf = lf - q @ (q.T @ lf)
        lf /= np.linalg.norm(lf, axis=0)
        J = lf.T @ li
        sm = np.repeat(np.sqrt(mass * AMU_ME), 3)
        D = lf.T @ (sm * ((si["coords"] - xf) / BOHR2ANG).ravel()) * Q_AU_TO_EV
        warn = []
        out = []
        for w2, lab in ((w2i, "initial"), (w2f, "final")):
            w = np.sqrt(np.abs(w2)) * EV_AU
            if (w2 < 0).any():
                warn.append(f"{int((w2 < 0).sum())} imaginary mode(s) of the {lab} state taken as real")
            low = w < floor_cm * CM_EV
            if floor_cm > 0 and low.any():
                warn.append(f"{int(low.sum())} mode(s) of the {lab} state below {floor_cm:g} cm⁻¹ raised to it")
                w = np.where(low, floor_cm * CM_EV, w)
            if (w <= 0).any():
                raise ValueError(f"zero frequency in the {lab} state: use floor_cm")
            out.append(w)
        p = cls(out[0], out[1], J, D, label_i or Path(initial).stem, label_f or Path(final).stem, warn)
        p.symbols, p.coords_i, p.modes_i, p.masses = si["symbols"], si["coords"], li, mass
        e_i, e_f = si.get("energy"), sf.get("energy")
        p.dE_fchk = (e_i - e_f) * EV_AU if e_i is not None and e_f is not None else None
        return p

    def project_cartesian(self, vec_au, mass_weighted=False):
        """Per-mode components (q units) of a Cartesian derivative vector given in a.u. per bohr (e.g. a
        non-adiabatic coupling ⟨Φ_f|∂Φ_i/∂R⟩ or a gradient), on the initial-state normal modes (from_fchk only).
        ∂/∂q_k = Σ_a L_ak / √m_a ∂/∂x_a."""
        if not hasattr(self, "modes_i"):
            raise ValueError("Cartesian vectors need a pair built from fchk files")
        v = np.asarray(vec_au, float).reshape(-1)
        if v.size != 3 * len(self.masses):
            raise ValueError(f"vector has {v.size // 3} atoms, molecule has {len(self.masses)}")
        sm = np.repeat(np.sqrt(self.masses * AMU_ME), 3)
        n_au = self.modes_i.T @ (v if mass_weighted else v / sm)
        return n_au / Q_AU_TO_EV


# ----------------------------------------------------------------------------------------------
# correlation functions
# ----------------------------------------------------------------------------------------------
def _f_kernel(w, t):
    """Mehler kernel coefficients of the final state at real time t: b − a = −ω tan(ωt/2), b + a = ω cot(ωt/2),
    and log of ω/(2πi sin ωt)."""
    x = np.outer(t, w)
    tm = -w * np.tan(x / 2)
    tp = w / np.tan(x / 2)
    logpre = np.log(w / (2j * np.pi * np.sin(x)))
    return tm, tp, logpre


def _i_kernel(w, t, beta):
    """Initial-state coefficients at τ_i = −iβ − t, written with z = e^{−βω} e^{iωt} so that T → 0 is stable:
    b = iω(1+z²)/(1−z²), a = 2iωz/(1−z²); log of ω/(2πi sin ωτ_i)/Z_i²."""
    eb = np.exp(-beta * w) if np.isfinite(beta) else np.zeros_like(w)
    ph = np.exp(1j * np.outer(t, w))
    z = eb * ph
    den = 1 - z ** 2
    b = 1j * w * (1 + z ** 2) / den
    a = 2j * w * z / den
    logpre = np.log(w * ph * (1 - eb) ** 2 / (np.pi * den))
    return b - a, b + a, a, b, logpre


class Correlator:
    """Correlation functions of a Pair at temperature T (K) on a time grid t_k = (k + ½) dt, k ≥ 0 (ħ/eV).
    Operators (dict name → spec):
        "fc"                                   ρ(t)
        ("coord", g0, G)   G: (N,) or (N, m)   ⟨(g0 + Gᵀq)(g0 + Gᵀq)⟩  (Herzberg–Teller; g0 scalar or (m,))
        ("nac", Nvec)      Nvec: (N,)          Σ_kl N_k N_l ⟨P_l … P_k⟩  (promoting modes, initial-state coordinates)
    run() returns {name: C(t)}; the FC factor ρ(t) is included in every C."""

    def __init__(self, pair, T=300.0, damping_fs=500.0, dt_fs=None, tmax_fs=None, batch=None, emax=4.0, tol=1e-13):
        """emax: largest |ΔE| (eV) the rates and spectra will be evaluated at; it sets the time step together with the
        highest frequency. tol: the integration stops once every |C(t)| has stayed below tol × its maximum for a
        whole batch (the correlation of a large molecule decays long before the damping does)."""
        self.pair = pair
        self.T = float(T)
        self.beta = 1 / (KB_EV * T) if T > 0 else np.inf
        self.tau = damping_fs / HBAR_FS
        wmax = max(pair.w_i.max(), pair.w_f.max()) if pair.n else 0.1
        self.tol = tol
        self.batch = batch or (4096 if pair.diagonal else max(16, min(1024, int(1.2e8 / max(pair.n, 1) ** 2 / 16))))
        if dt_fs:
            dt = dt_fs / HBAR_FS
        else:
            # short-time expansion log ρ ≈ −iω_c t − σ²t²/2: the spectrum of ρ lies within ω_c ± σ, so the grid has to
            # resolve |ω_c| + 10σ + |ΔE| (aliasing below e⁻⁵⁰)
            h = 0.004
            lw, ex, _ = self._chunk(np.array([h, 2 * h]), {"fc": "fc"})
            ph = np.unwrap(np.angle(np.exp(1j * lw.imag)))
            ph -= 2 * np.pi * np.round(ph[0] / (2 * np.pi))
            L = 0.5 * (lw.real + 1j * ph) + ex
            a1 = (4 * L[0] - L[1]) / (2 * h)
            b2 = (L[1] - 2 * L[0]) / (2 * h * h)
            wc, sig = abs(a1), math.sqrt(2 * abs(b2))
            dt = min(0.15, 2 * np.pi / wmax / 40, np.pi / (wc + 10 * sig + abs(emax) + 0.5))
        tmax = (tmax_fs / HBAR_FS) if tmax_fs else 6.0 * self.tau
        self.dt = dt
        self.t = (np.arange(int(np.ceil(tmax / dt))) + 0.5) * dt
        self.damp = np.exp(-(self.t / self.tau) ** 2)

    def run(self, ops=None):
        ops = {"fc": "fc"} if ops is None else dict(ops)
        ops.setdefault("fc", "fc")
        logw, expo, moments = [], [], {k: [] for k in ops if ops[k] != "fc"}
        peak = {k: 0.0 for k in ops}
        quiet = 0
        for a in range(0, len(self.t), self.batch):
            sl = slice(a, a + self.batch)
            lw, ex, mom = self._chunk(self.t[sl], ops)
            logw.append(lw)
            expo.append(ex)
            for k in mom:
                moments[k].append(mom[k])
            if self.tol:
                amp = np.exp(0.5 * lw.real + ex.real)
                small = True
                for k in ops:
                    m = amp * (np.abs(mom[k]) if k in mom else 1.0)
                    cur = float(m.max())
                    small &= cur < self.tol * max(peak[k], cur, 1e-300) or cur == 0.0
                    peak[k] = max(peak[k], cur)
                quiet = quiet + 1 if small and a > 0 else 0
                if quiet >= 2:
                    break
        logw, expo = np.concatenate(logw), np.concatenate(expo)
        nt = len(logw)
        self.t, self.damp = self.t[:nt], self.damp[:nt]
        # continuous branch of √W from ρ(0) = 1
        mag = np.exp(0.5 * logw.real)
        phase = np.unwrap(np.angle(np.exp(1j * logw.imag)))
        phase -= 2 * np.pi * np.round(phase[0] / (2 * np.pi))
        rho = mag * np.exp(0.5j * phase) * np.exp(expo)
        return {k: rho if ops[k] == "fc" else rho * np.concatenate(moments[k]) for k in ops}

    def _chunk(self, t, ops):
        p = self.pair
        n = p.n
        tmf, tpf, lpf = _f_kernel(p.w_f, t)
        tmi, tpi, ai, bi, lpi = _i_kernel(p.w_i, t, self.beta)
        J, D = p.J, p.D
        r2 = math.sqrt(2.0)
        if p.diagonal:
            j = np.diag(J)
            Au = -1j * (tmi + j ** 2 * tmf)
            Av = -1j * (tpi + j ** 2 * tpf)
            bu = r2 * 1j * j * tmf * D
            logdet = np.log(Au).sum(1) + np.log(Av).sum(1)
            mu = bu / Au
            expo = 1j * (tmf * D ** 2).sum(1) + 0.5 * (bu * mu).sum(1)
            su = lambda vec1, vec2: (vec1 * vec2 / Au).sum(-1)  # noqa: E731
            sv = lambda vec1, vec2: (vec1 * vec2 / Av).sum(-1)  # noqa: E731
        else:
            JT = J.T
            Au = -1j * ((JT[None] * tmf[:, None, :]) @ J)
            Av = -1j * ((JT[None] * tpf[:, None, :]) @ J)
            idx = np.arange(n)
            Au[:, idx, idx] += -1j * tmi
            Av[:, idx, idx] += -1j * tpi
            bu = r2 * 1j * ((tmf * D) @ J)                 # Jᵀ (T_f D)
            su0, ldu = np.linalg.slogdet(Au)
            sv0, ldv = np.linalg.slogdet(Av)
            logdet = np.log(su0) + ldu + np.log(sv0) + ldv
            mu = np.linalg.solve(Au, bu[..., None])[..., 0]
            expo = 1j * (tmf * D ** 2).sum(1) + 0.5 * (bu * mu).sum(1)

            def su(vec1, vec2):
                return (vec1 * np.linalg.solve(Au, vec2[..., None])[..., 0]).sum(-1)

            def sv(vec1, vec2):
                return (vec1 * np.linalg.solve(Av, vec2[..., None])[..., 0]).sum(-1)
        logw = lpf.sum(1) + lpi.sum(1) + 2 * n * np.log(2 * np.pi) + 2 * np.log(p.detJ) - logdet
        mom = {}
        for name, spec in ops.items():
            if spec == "fc":
                continue
            kind = spec[0]
            if kind == "coord":
                g0 = np.atleast_1d(np.asarray(spec[1], float))
                G = np.asarray(spec[2], float).reshape(n, -1)
                tot = 0
                for c in range(G.shape[1]):
                    # L1 = g0 + Gᵀy, L2 = g0 + Gᵀx  →  u: G/√2 each, v: −G/√2 and +G/√2
                    gu = np.broadcast_to(G[:, c] / r2, mu.shape)
                    m1 = g0[c % len(g0)] + (gu * mu).sum(-1)
                    tot = tot + m1 * m1 + su(gu, gu) - sv(gu, gu)
                mom[name] = tot
            elif kind == "nac":
                N = np.asarray(spec[1], float)
                # L1 = −Nᵀ(B_i y − A_i x): α1 = A_i N, γ1 = −B_i N
                a1, g1 = ai * N, -bi * N
                # L2 = −Nᵀ Jᵀ(B_f (Jx + D) − A_f (Jy + D)): α2 = −Jᵀ B_f J N, γ2 = Jᵀ A_f J N, δ2 = −NᵀJᵀ(B_f − A_f)D
                bf = (tpf + tmf) / 2
                af = (tpf - tmf) / 2
                JN = J @ N
                a2 = -(bf * JN) @ J if not p.diagonal else -(bf * JN) * np.diag(J)
                g2 = (af * JN) @ J if not p.diagonal else (af * JN) * np.diag(J)
                d2 = -(JN * tmf * D).sum(-1)
                p1, r1 = (a1 + g1) / r2, (a1 - g1) / r2
                p2, r2v = (a2 + g2) / r2, (a2 - g2) / r2
                mom[name] = (p1 * mu).sum(-1) * ((p2 * mu).sum(-1) + d2) + su(p1, p2) + sv(r1, r2v)
            else:
                raise ValueError(f"unknown operator {spec!r}")
        return logw, expo, mom

    # ------------------------------------------------------------------------------------------
    def rate(self, C, dE):
        """(1/ħ) ∫ dt e^{iΔE t} C(t) e^{−(t/τ)²}  (s⁻¹ per unit coupling² in eV²) at gap(s) ΔE = E_i − E_f (eV)."""
        dE = np.atleast_1d(np.asarray(dE, float))
        ph = np.exp(1j * np.outer(dE, self.t))
        val = 2 * (ph @ (C * self.damp)).real * self.dt
        return val / HBAR_EVS

    def lineshape(self, C, dE, E):
        """Spectral density σ(E) = (1/2π) ∫ dt e^{i(ΔE − E)t} C(t) e^{−(t/τ)²} (per eV) on photon energies E."""
        E = np.asarray(E, float)
        ph = np.exp(1j * np.outer(dE - E, self.t))
        return (ph @ (C * self.damp)).real * self.dt / np.pi


def _trapz(y, x):
    y, x = np.asarray(y), np.asarray(x)
    return float(((y[1:] + y[:-1]) * np.diff(x)).sum() / 2)


# ----------------------------------------------------------------------------------------------
# rates
# ----------------------------------------------------------------------------------------------
def radiative(pair, T, dE, mu0_au, ht=None, n_refr=1.0, damping_fs=500.0, e_grid=None, corr=None):
    """Spontaneous emission rate k_r (s⁻¹) and emission spectrum from the initial state. mu0_au: transition dipole
    (e·a0; scalar or 3-vector); ht: dμ/dq per mode, (N, 3) in e·a0 per q unit (see ht_from_au)."""
    mu0 = np.atleast_1d(np.asarray(mu0_au, float))
    if mu0.size == 1:
        mu0 = np.array([0.0, 0.0, float(mu0[0])])
    G = np.zeros((pair.n, 3)) if ht is None else np.asarray(ht, float).reshape(pair.n, 3)
    corr = corr or Correlator(pair, T, damping_fs)
    C = corr.run({"fc": "fc", "mu": ("coord", mu0, G), "mu_fc": ("coord", mu0, np.zeros((pair.n, 3)))})
    lam = pair.lambda_f()
    if e_grid is None:
        e_grid = np.linspace(max(dE - 6 * lam - 1.0, 0.05), dE + 0.5, 1500)
    sig = corr.lineshape(C["mu"], dE, e_grid)
    sig_fc = corr.lineshape(C["mu_fc"], dE, e_grid)
    rate = K_RAD * n_refr ** 2 * _trapz(e_grid ** 3 * sig, e_grid)
    rate_fc = K_RAD * n_refr ** 2 * _trapz(e_grid ** 3 * sig_fc, e_grid)
    spec = e_grid ** 3 * np.clip(sig, 0, None)
    k = int(np.argmax(spec))
    return {"k_r": float(rate), "k_r_FC": float(rate_fc), "E_eV": e_grid, "spectrum": spec / max(spec.max(), 1e-300),
            "peak_eV": float(e_grid[k]), "peak_nm": 1239.841984 / float(e_grid[k])}


def internal_conversion(pair, T, dE, nac_q, damping_fs=500.0, corr=None):
    """k_IC (s⁻¹) from the non-adiabatic coupling vector in initial-state normal coordinates (q units, eV^½;
    Pair.project_cartesian converts a Cartesian vector)."""
    corr = corr or Correlator(pair, T, damping_fs)
    C = corr.run({"nac": ("nac", np.asarray(nac_q, float))})
    return float(corr.rate(C["nac"], dE)[0])


def spin_orbit(pair, T, dE, soc_cm, ht=None, damping_fs=500.0, corr=None, average=1.0):
    """ISC rate (s⁻¹), FC with ⟨S|H_SO|T⟩ (cm⁻¹; Σ over triplet sublevels of |·|², i.e. the root-sum-square) plus
    optional HT dH_SO/dq per mode (cm⁻¹ per q unit). average = 1/3 for T → S (average over the triplet sublevels)."""
    v0 = float(soc_cm) * CM_EV
    G = np.zeros(pair.n) if ht is None else np.asarray(ht, float) * CM_EV
    corr = corr or Correlator(pair, T, damping_fs)
    C = corr.run({"fc": "fc", "soc": ("coord", v0, G)})
    return {"k": float(corr.rate(C["soc"], dE)[0]) * average,
            "k_FC": float(corr.rate(C["fc"], dE)[0]) * v0 ** 2 * average}


def decay_rates(pair, T, dE, mu=None, ht_mu=None, nac=None, soc=None, ht_soc=None, soc_average=1.0, n_refr=1.0,
                damping_fs=500.0, e_grid=None):
    """All rates of one initial → final pair from a single TVCF run (any of the couplings may be None):
    mu (e·a0, scalar or 3-vector) [+ ht_mu (N, 3) per q unit] → k_r, k_r_FC and the emission spectrum;
    nac (N,) per q unit → k_IC;  soc (cm⁻¹) [+ ht_soc (N,) cm⁻¹ per q unit] → k_ISC, k_ISC_FC (× soc_average)."""
    corr = Correlator(pair, T, damping_fs, emax=abs(dE) + 0.6)
    ops = {"fc": "fc"}
    if mu is not None:
        mu0 = np.atleast_1d(np.asarray(mu, float))
        mu0 = np.array([0.0, 0.0, float(mu0[0])]) if mu0.size == 1 else mu0
        G = np.zeros((pair.n, 3)) if ht_mu is None else np.asarray(ht_mu, float).reshape(pair.n, 3)
        ops["mu"] = ("coord", mu0, G)
        if ht_mu is not None:
            ops["mu_fc"] = ("coord", mu0, np.zeros((pair.n, 3)))
    if nac is not None:
        ops["nac"] = ("nac", np.asarray(nac, float))
    if soc is not None:
        G = np.zeros(pair.n) if ht_soc is None else np.asarray(ht_soc, float) * CM_EV
        ops["soc"] = ("coord", float(soc) * CM_EV, G)
    C = corr.run(ops)
    out = {"T": float(T), "dE_eV": float(dE), "n_time_steps": len(corr.t), "dt_fs": corr.dt * HBAR_FS,
           "k_FC_per_eV2": float(corr.rate(C["fc"], dE)[0])}
    if mu is not None:
        if e_grid is None:
            e_grid = np.linspace(max(dE - 4 * pair.lambda_f() - 1.0, 0.05), dE + 0.5, 1500)
        sig = corr.lineshape(C["mu"], dE, e_grid)
        out["k_r"] = K_RAD * n_refr ** 2 * _trapz(e_grid ** 3 * sig, e_grid)
        out["k_r_FC"] = out["k_r"] if ht_mu is None else \
            K_RAD * n_refr ** 2 * _trapz(e_grid ** 3 * corr.lineshape(C["mu_fc"], dE, e_grid), e_grid)
        spec = e_grid ** 3 * np.clip(sig, 0, None)
        k = int(np.argmax(spec))
        out.update(E_eV=e_grid, spectrum=spec / max(spec.max(), 1e-300), peak_eV=float(e_grid[k]),
                   peak_nm=1239.841984 / float(e_grid[k]))
    if nac is not None:
        out["k_IC"] = float(corr.rate(C["nac"], dE)[0])
    if soc is not None:
        out["k_ISC"] = float(corr.rate(C["soc"], dE)[0]) * soc_average
        out["k_ISC_FC"] = out["k_FC_per_eV2"] * (float(soc) * CM_EV) ** 2 * soc_average
    return out


def mu_from_fosc(f, e_vert_ev):
    """|μ| (e·a0) from an oscillator strength f = (2/3) ΔE |μ|² (a.u.)."""
    return math.sqrt(3 * f / (2 * e_vert_ev / EV_AU))


def ht_from_au(pair, dmu_dQ_au, amu=True):
    """Convert per-mode derivatives given per mass-weighted a.u. coordinate (per √amu·a0 if amu, else per √m_e·a0)
    to per q unit."""
    g = np.asarray(dmu_dQ_au, float)
    if amu:
        g = g / math.sqrt(AMU_ME)
    return g / Q_AU_TO_EV


def tadf_kinetics(k_rS, k_icS, k_isc, k_risc, k_rT=0.0, k_nrT=0.0):
    """S1/T1 kinetics: prompt and delayed lifetimes (s), quantum yields of prompt and delayed fluorescence."""
    kS = k_rS + k_icS + k_isc
    kT = k_risc + k_rT + k_nrT
    M = np.array([[-kS, k_risc], [k_isc, -kT]])
    ev = np.sort(np.linalg.eigvals(M).real)
    phi_pf = k_rS / kS if kS > 0 else 0.0
    phi_isc = k_isc / kS if kS > 0 else 0.0
    phi_risc = k_risc / kT if kT > 0 else 0.0
    loop = phi_isc * phi_risc
    phi_df = phi_pf * loop / (1 - loop) if loop < 1 else float("inf")
    phi_ph = phi_isc * (k_rT / kT if kT > 0 else 0.0) / (1 - loop) if loop < 1 else 0.0
    return {"tau_prompt_s": float(-1 / ev[0]) if ev[0] < 0 else float("inf"),
            "tau_delayed_s": float(-1 / ev[1]) if ev[1] < 0 else float("inf"),
            "PLQY_prompt": float(phi_pf), "PLQY_delayed": float(phi_df), "PLQY_phos": float(phi_ph),
            "PLQY_total": float(phi_pf + phi_df + phi_ph), "Phi_ISC": float(phi_isc), "Phi_RISC": float(phi_risc)}


class Emitter:
    """S0 / S1 / T1 of one molecule from three freq fchk files, with the couplings needed for its decay channels.
    rates(T) returns k_r(S1), k_IC(S1), k_ISC, k_RISC, k_p(T1), k_nr(T1) and the S1/T1 kinetics."""

    def __init__(self, s0, s1, t1, soc, fosc=None, e_vert=None, mu_s1=None, mu_t1=None, soc_t1s0=0.0, nac=None,
                 e_s1=None, e_t1=None, n_refr=1.0, floor_cm=0.0, damping_fs=500.0):
        self.s1s0 = Pair.from_fchk(s1, s0, floor_cm=floor_cm, label_i="S1", label_f="S0")
        self.s1t1 = Pair.from_fchk(s1, t1, floor_cm=floor_cm, label_i="S1", label_f="T1")
        self.t1s0 = Pair.from_fchk(t1, s0, floor_cm=floor_cm, label_i="T1", label_f="S0")
        self.e_s1 = e_s1 if e_s1 is not None else self.s1s0.dE_fchk
        self.e_t1 = e_t1 if e_t1 is not None else self.t1s0.dE_fchk
        if self.e_s1 is None or self.e_t1 is None:
            raise ValueError("no total energies in the fchk files: give the adiabatic S1 and T1 energies")
        self.mu = mu_from_fosc(fosc, e_vert or self.e_s1) if fosc is not None else (mu_s1 or 0.0)
        self.mu_t1 = mu_t1
        self.soc, self.soc_t1s0 = float(soc), float(soc_t1s0 or 0.0)
        self.nac = self.s1s0.project_cartesian(read_vector(nac, len(self.s1s0.masses))) if nac else None
        self.n_refr, self.damping_fs = n_refr, damping_fs
        self.warnings = sorted(set(self.s1s0.warnings + self.s1t1.warnings + self.t1s0.warnings))

    def summary(self):
        return {"E_S1_eV": self.e_s1, "E_T1_eV": self.e_t1, "dE_ST_eV": self.e_s1 - self.e_t1,
                "lambda_S1S0_eV": self.s1s0.lambda_f(), "lambda_S1T1_eV": self.s1t1.lambda_f(),
                "lambda_T1S0_eV": self.t1s0.lambda_f(), "n_modes": self.s1s0.n}

    def rates(self, T):
        kw = dict(damping_fs=self.damping_fs)
        r1 = decay_rates(self.s1s0, T, self.e_s1, mu=self.mu, nac=self.nac, n_refr=self.n_refr, **kw)
        r2 = decay_rates(self.s1t1, T, self.e_s1 - self.e_t1, soc=self.soc, **kw)
        r3 = decay_rates(self.s1t1.swapped(), T, self.e_t1 - self.e_s1, soc=self.soc, soc_average=1 / 3, **kw)
        r4 = decay_rates(self.t1s0, T, self.e_t1, soc=self.soc_t1s0 or None, mu=self.mu_t1 or None,
                         soc_average=1 / 3, n_refr=self.n_refr, **kw) if (self.soc_t1s0 or self.mu_t1) else {}
        k = {"k_r_S1": r1.get("k_r", 0.0), "k_IC_S1": r1.get("k_IC", 0.0), "k_ISC": r2["k_ISC"],
             "k_RISC": r3["k_ISC"], "k_p_T1": r4.get("k_r", 0.0), "k_nr_T1": r4.get("k_ISC", 0.0)}
        kin = tadf_kinetics(k["k_r_S1"], k["k_IC_S1"], k["k_ISC"], k["k_RISC"], k["k_p_T1"], k["k_nr_T1"])
        return {"T": float(T), **k, **kin, "peak_nm": r1.get("peak_nm"), "E_eV": r1.get("E_eV"),
                "spectrum": r1.get("spectrum")}


# ----------------------------------------------------------------------------------------------
# input helpers
# ----------------------------------------------------------------------------------------------
def read_vector(path, natoms=None):
    """Cartesian vector (one row per atom, the last three numbers of each row: x y z). Understands ORCA's
    'CARTESIAN NON-ADIABATIC COUPLINGS' block; otherwise every row with ≥ 3 numbers is taken."""
    import re
    text = open(path, encoding="utf-8", errors="ignore").read()
    num = re.compile(r"[-+]?\d*\.\d+(?:[EeDd][-+]?\d+)?")
    m = re.search(r"CARTESIAN NON-ADIABATIC COUPLINGS[^\n]*\n", text)
    lines = text[m.end():].splitlines() if m else text.splitlines()
    rows = []
    for line in lines:
        vals = num.findall(line)
        if len(vals) >= 3:
            rows.append([float(v.replace("D", "E")) for v in vals[-3:]])
            if natoms and len(rows) == natoms:
                break
        elif rows and m:
            if line.strip() and not line.strip().startswith(("-", "=")):
                break
    if not rows:
        raise ValueError(f"{path}: no Cartesian vector found")
    return np.array(rows)


def read_mode_table(path, ncol):
    """Per-mode table: rows 'mode, v1[, v2, v3]' (header allowed); returns (N, ncol)."""
    out = []
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.reader(fh):
            try:
                vals = [float(x) for x in r if x.strip()]
            except ValueError:
                continue
            if len(vals) >= ncol + 1:
                out.append(vals[1:ncol + 1])
            elif len(vals) == ncol:
                out.append(vals)
    if not out:
        raise ValueError(f"{path}: no numeric rows")
    return np.array(out)
