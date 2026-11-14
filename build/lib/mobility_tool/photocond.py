"""Photoconductivity of a molecular semiconductor under laser illumination.

Chain of the model (all inputs in the units of the function arguments; SI inside):

1. generation of singlet excitons   g = Φ (1 − R)(1 − e^{−αd}) / d,   Φ = I λ / (h c)
2. free-carrier yield               η = k_d / (k_d + k_S),  k_S = 1/τ_S (fission, radiative and non-radiative decay)
   Braun dissociation rate          k_d = 3 k_L / (4π a³) · exp(−E_b / kT) · I₁(2√(2b)) / √(2b)
                                    b = e³ F / (8π ε₀ ε_r k² T²),  a = e² / (4π ε₀ ε_r E_b)
3. recombination                    k_r = ζ k_L,  k_L = e (μ_e + μ_h) / (ε₀ ε_r)   (Langevin)
   steady state                     η g = k_r n² + n / τ_t
4. photoconductivity                Δσ = e n (μ_e + μ_h),   J = Δσ F

Pulsed excitation: n(0) = η · fluence · λ/(hc) · (1 − R)(1 − e^{−αd}) / d and dn/dt = −k_r n² − n/τ_t.

The mobilities are the ground-state (polaron) mobilities of the crystal: photogenerated carriers relax to the band
edge or polaron states within picoseconds, long before they recombine.
"""
import csv
import math

import numpy as np
from scipy.special import i1e

E = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
H = 6.62607015e-34
C = 2.99792458e8


def langevin(mu_e, mu_h, eps_r):
    """Langevin recombination constant / m³ s⁻¹ (mobilities in cm² V⁻¹ s⁻¹)."""
    return E * (mu_e + mu_h) * 1e-4 / (EPS0 * eps_r)


def capture_radius(eb_ev, eps_r):
    """Electron–hole distance / m at which the Coulomb energy equals E_b."""
    return E / (4 * math.pi * EPS0 * eps_r * eb_ev)


def onsager_factor(field_vcm, T, eps_r):
    """Braun field factor I₁(2√(2b))/√(2b) (→ 1 at zero field)."""
    b = E ** 3 * np.asarray(field_vcm, float) * 100.0 / (8 * math.pi * EPS0 * eps_r * (KB * T) ** 2)
    x = np.sqrt(2 * b)
    z = 2 * x
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        f = np.where(x > 1e-6, i1e(z) * np.exp(z) / np.where(x > 1e-6, x, 1.0), 1.0 + b)
    return f, b


def braun_rate(field_vcm, T, eb_ev, eps_r, mu_e, mu_h, a_m=None):
    """Field-assisted dissociation rate / s⁻¹ of a bound electron–hole pair (Braun 1984)."""
    a = a_m or capture_radius(eb_ev, eps_r)
    f, _ = onsager_factor(field_vcm, T, eps_r)
    return 3 * langevin(mu_e, mu_h, eps_r) / (4 * math.pi * a ** 3) * math.exp(-eb_ev * E / (KB * T)) * f


def yield_free(field_vcm, T, eb_ev, eps_r, mu_e, mu_h, tau_s, a_m=None):
    """Free-carrier yield η per absorbed photon."""
    kd = braun_rate(field_vcm, T, eb_ev, eps_r, mu_e, mu_h, a_m)
    return kd / (kd + 1.0 / tau_s), kd


def absorbed_fraction(alpha_cm, d_nm, refl):
    return (1 - refl) * (1 - math.exp(-alpha_cm * d_nm * 1e-7))


def generation(intensity_wcm2, wavelength_nm, alpha_cm, d_nm, refl):
    """Exciton generation rate averaged over the film / m⁻³ s⁻¹ (intensity in W cm⁻²)."""
    phi = np.asarray(intensity_wcm2, float) * 1e4 * wavelength_nm * 1e-9 / (H * C)
    return phi * absorbed_fraction(alpha_cm, d_nm, refl) / (d_nm * 1e-9)


def steady_density(G, k_r, tau_t=None):
    """Steady carrier density / m⁻³ from G = k_r n² + n/τ_t."""
    G = np.asarray(G, float)
    if not tau_t:
        return np.sqrt(G / k_r)
    a = 1.0 / tau_t
    return (-a + np.sqrt(a * a + 4 * k_r * G)) / (2 * k_r)


def steady_state(*, intensity, wavelength, alpha, thickness, refl, field, T, mu_e, mu_h, eps_r, eb, tau_s,
                 zeta=1.0, tau_t=None, a_m=None):
    """Steady photoconductivity. Returns a dict of arrays broadcast over intensity / field."""
    eta, kd = yield_free(field, T, eb, eps_r, mu_e, mu_h, tau_s, a_m)
    g = generation(intensity, wavelength, alpha, thickness, refl)
    k_r = zeta * langevin(mu_e, mu_h, eps_r)
    n = steady_density(eta * g, k_r, tau_t)
    dsig = E * n * (mu_e + mu_h) * 1e-4          # S/m
    J = dsig * np.asarray(field, float) * 100.0  # A/m²
    lifetime = n / (eta * g)
    return {"eta": eta, "k_d": kd, "g": g, "G": eta * g, "k_r": k_r, "n": n, "dsigma": dsig, "J": J,
            "lifetime": lifetime}


def pulse_decay(t, *, fluence, wavelength, alpha, thickness, refl, field, T, mu_e, mu_h, eps_r, eb, tau_s,
                zeta=1.0, tau_t=None, a_m=None):
    """Δσ(t) / S m⁻¹ after a short pulse (fluence in J cm⁻²); t in s."""
    eta, _ = yield_free(field, T, eb, eps_r, mu_e, mu_h, tau_s, a_m)
    n0 = eta * fluence * 1e4 * wavelength * 1e-9 / (H * C) * absorbed_fraction(alpha, thickness, refl) / (thickness * 1e-9)
    k_r = zeta * langevin(mu_e, mu_h, eps_r)
    t = np.asarray(t, float)
    if tau_t:
        a = 1.0 / tau_t
        n = a * n0 * np.exp(-a * t) / (a + k_r * n0 * (1 - np.exp(-a * t)))
    else:
        n = n0 / (1 + k_r * n0 * t)
    return E * n * (mu_e + mu_h) * 1e-4, n0


def export_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
