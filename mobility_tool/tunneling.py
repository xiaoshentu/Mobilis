# -*- coding: utf-8 -*-
"""Hopping rate with quantum nuclear tunneling over all intramolecular modes (Fermi golden rule, displaced harmonic
oscillators; G. Nan, X. Yang, L. Wang, Z. Shuai, Y. Zhao, Phys. Rev. B 79, 115203 (2009); Z. Shuai, W. Li, J. Ren,
Y. Jiang, H. Geng, J. Chem. Phys. 153, 080902 (2020)).

    k(ΔG) = |V|²/ħ² ∫ dt exp[ i ΔG t/ħ − G(t) ]
    G(t)  = Σ_j S_j [ (2n̄_j + 1)(1 − cos ω_j t) − i sin ω_j t ]  +  λ_cl k_B T t²/ħ² − i λ_cl t/ħ

S_j are the Huang–Rhys factors of both molecules that change charge state: the modes of the neutral molecule
(relaxation on the neutral surface, λ₁) and those of the ion (λ₂), i.e. the two files written by `reorg-decomp`.
n̄_j = 1/(exp(ħω_j/k_BT) − 1). λ_cl is a classical (outer-sphere) reorganization energy; modes below an optional
frequency cut can be moved into it. ΔG = E_final − E_initial.

Limits (tests/test_v51.py): all modes classical → Marcus; one high-frequency mode at k_BT ≪ ħω plus classical λ_cl →
Marcus–Levich–Jortner. Quantum modes give larger rates than Marcus at the same λ, the more so at low temperature
(tunneling through the barrier instead of activation over it).

A Gaussian damping exp[−(t/τ_d)²] (τ_d = 500 fs by default, i.e. an extra broadening of ≈ 2 meV) makes the time
integral converge when only a few quantum modes are present; it has no visible effect for real molecules."""
import csv

import numpy as np

from .mobility import HBAR_EVS, KB_EV

C_CM_S = 2.99792458e10
CM_TO_EV = 1.239841984e-4


def read_modes(path):
    """(frequency / cm⁻¹, Huang–Rhys factor S) of every mode with ω > 0 and S > 0 from a reorg-decomp modes CSV
    (columns freq_cm-1 and S, or freq_cm-1 and lambda_meV)."""
    out = []
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            f = float(r.get("freq_cm-1") or r.get("freq") or 0)
            if f <= 0:
                continue
            if r.get("S") not in (None, ""):
                s = float(r["S"])
            else:
                s = float(r["lambda_meV"]) / 1000 / (f * CM_TO_EV)
            if s > 0:
                out.append((f, s))
    if not out:
        raise ValueError(f"{path}: no modes with frequency > 0 and Huang–Rhys factor > 0")
    return out


class FGRRate:
    """Quantum-nuclear-tunneling hopping rate for a set of modes [(ω / cm⁻¹, S)], temperature T / K and classical
    reorganization energy lam_cl / eV. rate(J, ΔG) in s⁻¹ (J, ΔG in eV, scalars or arrays)."""

    def __init__(self, modes, T, lam_cl=0.0, classical_below_cm=0.0, damping_fs=500.0, emax=1.5, de=0.002):
        self.T = float(T)
        kt = KB_EV * self.T
        q = [(f, s) for f, s in modes if f >= classical_below_cm]
        low = sum(f * CM_TO_EV * s for f, s in modes if f < classical_below_cm)
        self.lam_cl = float(lam_cl) + low
        self.lam_q = float(sum(f * CM_TO_EV * s for f, s in q))
        self.lam = self.lam_q + self.lam_cl
        if self.lam <= 0:
            raise ValueError("reorganization energy must be positive")
        w = np.array([2 * np.pi * C_CM_S * f for f, _ in q])                  # rad/s
        S = np.array([s for _, s in q])
        hw = np.array([f * CM_TO_EV for f, _ in q])
        n = 1.0 / np.expm1(hw / kt) if len(q) else np.zeros(0)
        # time grid: step resolves the highest frequency, length covers the damping
        wmax = w.max() if len(w) else 0.0
        dt = min(0.25e-15, 2 * np.pi / max(wmax, 1e12) / 40)
        tau = damping_fs * 1e-15
        t = np.arange(0.0, 3.0 * tau + dt, dt)
        G = np.zeros(len(t), complex)
        for k0 in range(0, len(w), 32):
            wk, sk, nk = w[k0:k0 + 32, None], S[k0:k0 + 32, None], n[k0:k0 + 32, None]
            G += (sk * ((2 * nk + 1) * (1 - np.cos(wk * t)) - 1j * np.sin(wk * t))).sum(0)
        G += self.lam_cl * kt * t ** 2 / HBAR_EVS ** 2 - 1j * self.lam_cl * t / HBAR_EVS
        f = np.exp(-G - (t / tau) ** 2)
        wts = np.full(len(t), dt)
        wts[0] = dt / 2
        self._t, self._fw = t, f * wts
        self.eps = np.arange(-emax, emax + de / 2, de)
        self.table = self._k_over_v2(self.eps)
        self.modes_q = len(q)

    def _k_over_v2(self, eps):
        eps = np.atleast_1d(np.asarray(eps, float))
        out = np.empty(len(eps))
        for a in range(0, len(eps), 64):
            ph = np.exp(1j * np.outer(eps[a:a + 64], self._t) / HBAR_EVS)
            out[a:a + 64] = 2 * (ph @ self._fw).real / HBAR_EVS ** 2
        return np.maximum(out, 0.0)

    def k_over_v2(self, dG):
        """k/|V|² (s⁻¹ eV⁻²) at ΔG / eV (interpolated in log space inside the table, direct outside)."""
        dG = np.asarray(dG, float)
        inside = (dG >= self.eps[0]) & (dG <= self.eps[-1])
        lt = np.log(np.maximum(self.table, 1e-300))
        out = np.exp(np.interp(dG, self.eps, lt))
        if np.any(~inside):
            out = np.where(inside, out, self._k_over_v2(np.ravel(dG))[0] if dG.ndim == 0 else
                           self._k_over_v2(dG.ravel()).reshape(dG.shape))
        return out

    def rate(self, J, dG=0.0):
        return np.asarray(J, float) ** 2 * self.k_over_v2(dG)

    def marcus(self, J, dG=0.0):
        """Marcus rate with the same total λ (for comparison)."""
        kt = KB_EV * self.T
        return 2 * np.pi / HBAR_EVS * np.asarray(J, float) ** 2 / np.sqrt(4 * np.pi * self.lam * kt) * \
            np.exp(-(np.asarray(dG, float) + self.lam) ** 2 / (4 * self.lam * kt))

    def summary(self):
        return {"T": self.T, "lambda_eV": self.lam, "lambda_quantum_eV": self.lam_q, "lambda_classical_eV": self.lam_cl,
                "n_quantum_modes": self.modes_q, "ratio_to_marcus_dG0": float(self.k_over_v2(0.0) /
                                                                             (self.marcus(1.0, 0.0)))}


def tensor(graph, fgr):
    """Zero-field random-walk mobility tensor μ_ab = (1/n) Σ k r_a r_b / (2k_BT) (cm² V⁻¹ s⁻¹) with FGR rates at
    ΔG = 0 (equivalent sites)."""
    c0 = float(fgr.k_over_v2(0.0))
    mu = np.zeros((3, 3))
    for (_, _, v, j, m, _t) in graph["edges"]:
        vc = np.asarray(v, float) * 1e-8
        mu += 0.5 * c0 * j ** 2 * m * np.outer(vc, vc)
    return mu / (KB_EV * fgr.T) / graph["n_sites"]


def modes_from_files(paths):
    out = []
    for p in paths:
        out += read_modes(p)
    return out
