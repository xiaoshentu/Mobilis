# -*- coding: utf-8 -*-
"""Band transport limited by acoustic-phonon scattering in deformation-potential (DP) theory.

    Bardeen & Shockley, Phys. Rev. 80, 72 (1950):
        3D:  μ = 2√(2π) e ħ⁴ C / [3 (m*)^{5/2} (k_BT)^{3/2} E₁²]
        2D:  μ = e ħ³ C_2D / (k_BT m* m_d E₁²),   m_d = √(m_x m_y)
    with the elastic constant C = (1/V₀) ∂²E/∂ε² (2D: per area) and the deformation potential E₁ = ∂E_edge/∂ε, both
    from periodic DFT calculations of the strained crystal (band edges referenced to the vacuum level or a core
    level, otherwise E₁ is meaningless).

    Boltzmann transport on the tight-binding band of the crystal (couplings from EHT, ADF or any other source) with
    the DP relaxation time of elastic, isotropic acoustic scattering, as in L. Tang, M. Long, D. Wang, Z. Shuai,
    Sci. China Ser. B 52, 1646 (2009) and J. Xi, M. Long, L. Tang, D. Wang, Z. Shuai, Nanoscale 4, 4348 (2012):
        1/τ_β(ε) = 2π k_BT E₁β² g(ε) / (ħ C_β),     g: density of states per volume and spin
        μ_ββ = (e / k_BT) ⟨τ_β(ε_k) v_β(k)²⟩     (non-degenerate carriers, Boltzmann average over the bands)
    For a parabolic band this reproduces the Bardeen–Shockley expression (tests/test_v53.py).

Units: energies eV, lengths Å, C in GPa (3D) or N/m (2D), μ in cm² V⁻¹ s⁻¹."""
import csv
import math

import numpy as np

from .mobility import HBAR_EVS, KB_EV

E_CH = 1.602176634e-19
ME = 9.1093837015e-31
HBAR = 1.054571817e-34
GPA_TO_EV_A3 = 1e9 / E_CH * 1e-30          # GPa → eV/Å³
NM_TO_EV_A2 = 1.0 / E_CH * 1e-20           # N/m → eV/Å²


# ----------------------------------------------------------------------------------------------
# fits of the strained calculations
# ----------------------------------------------------------------------------------------------
def fit_axis(strain, energy_ev, edge_ev, size, dim=3):
    """Elastic constant and deformation potential of one strain direction. strain: fractions (or %, detected by
    |ε| > 0.2); energy_ev: total energy per cell; edge_ev: band edge; size: V₀ / Å³ (3D) or S₀ / Å² (2D).
    Returns C (GPa or N/m), E₁ (eV) and the fit residuals."""
    e = np.asarray(strain, float)
    if np.abs(e).max() > 0.2:
        e = e / 100
    E = np.asarray(energy_ev, float)
    pE = np.polyfit(e, E, 2)
    d2 = 2 * pE[0]
    if dim == 3:
        C = d2 / size / GPA_TO_EV_A3
    else:
        C = d2 / size / NM_TO_EV_A2
    out = {"C": float(C), "C_unit": "GPa" if dim == 3 else "N/m", "rms_energy_meV":
           float(1000 * np.sqrt(np.mean((np.polyval(pE, e) - E) ** 2)))}
    if edge_ev is not None:
        ed = np.asarray(edge_ev, float)
        pB = np.polyfit(e, ed, 1 if len(e) < 4 else 2)
        out["E1"] = float(pB[-2])                         # slope at ε = 0
        out["rms_edge_meV"] = float(1000 * np.sqrt(np.mean((np.polyval(pB, e) - ed) ** 2)))
    return out


def read_strain_table(path):
    """CSV with columns axis, strain, energy_eV, and any of vbm_eV / cbm_eV, plus volume_A3 (3D) or area_A2 (2D)
    of the unstrained cell (one value is enough). Returns {axis: rows}."""
    data = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
            data.setdefault(r.get("axis", "a"), []).append(r)
    if not data:
        raise ValueError(f"{path}: no rows")
    return data


def fit_table(path, dim=3, size=None):
    """C and E₁ (holes: VBM, electrons: CBM) for every axis of a strain table."""
    data = read_strain_table(path)
    out = {}
    for ax, rows in data.items():
        st = [float(r["strain"]) for r in rows]
        en = [float(r["energy_ev"]) for r in rows]
        sz = size
        if sz is None:
            key = "volume_a3" if dim == 3 else "area_a2"
            vals = [float(r[key]) for r in rows if r.get(key)]
            if not vals:
                raise ValueError(f"{path}: column {key} (unstrained cell) is missing")
            sz = vals[int(np.argmin(np.abs(st)))] if len(vals) == len(st) else vals[0]
        res = {}
        for car, col in (("hole", "vbm_ev"), ("electron", "cbm_ev")):
            if all(r.get(col) for r in rows):
                res[car] = fit_axis(st, en, [float(r[col]) for r in rows], sz, dim)
        if not res:
            res["none"] = fit_axis(st, en, None, sz, dim)
        out[ax] = res
    return out


# ----------------------------------------------------------------------------------------------
# Bardeen–Shockley
# ----------------------------------------------------------------------------------------------
def bardeen_shockley(C, E1, m_eff, T=300.0, dim=3, m_d=None):
    """μ / cm² V⁻¹ s⁻¹. C in GPa (3D) or N/m (2D), E1 in eV, masses in m_e."""
    kT = KB_EV * T * E_CH
    E1j = E1 * E_CH
    m = m_eff * ME
    if dim == 3:
        mu = 2 * math.sqrt(2 * math.pi) * E_CH * HBAR ** 4 * (C * 1e9) / (3 * m ** 2.5 * kT ** 1.5 * E1j ** 2)
    else:
        md = (m_d if m_d is not None else m_eff) * ME
        mu = E_CH * HBAR ** 3 * C / (kT * m * md * E1j ** 2)
    return mu * 1e4


# ----------------------------------------------------------------------------------------------
# Boltzmann transport on a band structure
# ----------------------------------------------------------------------------------------------
def tb_bands(graph, lattice, carrier="hole", n=(32, 32, 32)):
    """Tight-binding bands of a crystal graph on a Γ-centred k grid: energies (nk, nb) in eV (carrier energies,
    lowest = band edge), group velocities ħv = ∂ε/∂k (nk, nb, 3) in eV·Å by Hellmann–Feynman, cell volume Å³."""
    lat = np.asarray(lattice, float)
    rec = 2 * np.pi * np.linalg.inv(lat).T
    axes = [np.arange(k) / k for k in n]
    frac = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    ks = frac @ rec
    nb = graph["n_sites"]
    sgn = -1.0 if carrier == "hole" else 1.0
    E = np.empty((len(ks), nb))
    V = np.empty((len(ks), nb, 3))
    edges = graph["edges"]
    vec = np.array([np.asarray(e[2], float) for e in edges])
    amp = np.array([sgn * e[3] * e[4] for e in edges])
    ii = np.array([e[0] for e in edges])
    jj = np.array([e[1] for e in edges])
    for a in range(0, len(ks), 2048):
        kk = ks[a:a + 2048]
        ph = np.exp(1j * kk @ vec.T) * amp                       # (nk, ne)
        H = np.zeros((len(kk), nb, nb), complex)
        dH = np.zeros((len(kk), 3, nb, nb), complex)
        for e in range(len(edges)):
            H[:, ii[e], jj[e]] += ph[:, e]
            dH[:, :, ii[e], jj[e]] += 1j * ph[:, e, None] * vec[e]
        H = (H + np.conj(np.swapaxes(H, 1, 2))) / 2
        dH = (dH + np.conj(np.swapaxes(dH, 2, 3))) / 2
        w, U = np.linalg.eigh(H)
        E[a:a + 2048] = w
        V[a:a + 2048] = np.einsum("kib,kxij,kjb->kbx", U.conj(), dH, U).real
    return E, V, abs(float(np.linalg.det(lat)))


def boltzmann_dp(E, V, volume, C, E1, T=300.0, directions=None, dim=3, smear=0.5):
    """Mobilities μ_ββ (cm² V⁻¹ s⁻¹) along unit vectors `directions` (default x, y, z) from band energies E (nk, nb,
    eV), velocities ħv (nk, nb, 3, eV·Å) and the cell volume (Å³), with C (GPa) and E1 (eV) per direction (scalars or
    one per direction). dim = 2: `volume` is the cell area (Å²) and C is in N/m. The DOS is a histogram with bins of k_BT/20 smoothed over ±1 bin."""
    kT = KB_EV * T
    dirs = np.eye(3) if directions is None else np.asarray(directions, float)
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    nd = len(dirs)
    C = np.broadcast_to(np.asarray(C, float), (nd,))
    E1 = np.broadcast_to(np.asarray(E1, float), (nd,))
    e = E.ravel() - E.min()
    v = V.reshape(-1, 3)
    nk = E.shape[0]
    de = kT / 40
    top = min(e.max(), 30 * kT)
    sel = e <= top
    bins = np.arange(0, top + 2 * de, de)
    h, _ = np.histogram(e[sel], bins=bins)
    sig = max(smear, 1e-3) * kT
    x = np.arange(-int(4 * sig / de), int(4 * sig / de) + 1) * de
    ker = np.exp(-x ** 2 / (2 * sig ** 2))
    ker = ker / ker.sum()
    h = np.convolve(h, ker, mode="same") / np.convolve(np.ones(len(h)), ker, mode="same")   # edge-corrected
    g = h / (nk * volume * de)                                    # states / (eV Å³) (2D: Å²) per spin
    idx = np.minimum((e[sel] / de).astype(int), len(g) - 1)
    gk = np.maximum(g[idx], 1e-300)
    w = np.exp(-e[sel] / kT)
    Z = w.sum()
    out = []
    for d in range(nd):
        cu = C[d] * (GPA_TO_EV_A3 if dim == 3 else NM_TO_EV_A2)
        inv_tau = 2 * np.pi * kT * E1[d] ** 2 * gk / (HBAR_EVS * cu)                    # 1/s
        vb = (v[sel] @ dirs[d]) * 1e-10 / HBAR_EVS                                      # m/s
        mu = (w * vb ** 2 / inv_tau).sum() / Z / kT                                      # m² / (V s)
        out.append(mu * 1e4)
    return np.array(out)


def write_strained(crys, path_prefix, strains_percent, axes=("a", "b", "c"), fmt="cif"):
    """Homogeneously strained cells (fractional coordinates fixed) for periodic DFT, plus a template of the strain
    table to fill in. Returns the written files."""
    from pathlib import Path
    lat0 = np.asarray(crys["lattice"], float)
    frac = crys["frac"] if "frac" in crys else np.asarray([a[1:] for a in crys["atoms"]], float) @ np.linalg.inv(lat0)
    syms = crys["symbols"] if "symbols" in crys else [a[0] for a in crys["atoms"]]
    files = []
    pre = Path(path_prefix)
    pre.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ax in axes:
        k = "abc".index(ax)
        for s in strains_percent:
            lat = lat0.copy()
            lat[k] = lat[k] * (1 + s / 100)
            name = f"{pre.name}_{ax}{s:+.1f}".replace("+", "p").replace("-", "m")
            p = pre.parent / (name + (".vasp" if fmt == "poscar" else ".cif"))
            _write_cell(p, lat, syms, frac, fmt, name)
            files.append(p)
            rows.append([ax, s / 100, "", "", "", f"{abs(np.linalg.det(lat0)):.4f}"])
    tab = pre.parent / f"{pre.name}_strain_table.csv"
    with open(tab, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "strain", "energy_eV", "vbm_eV", "cbm_eV", "volume_A3"])
        w.writerows(rows)
    files.append(tab)
    return files


def _write_cell(path, lat, syms, frac, fmt, title):
    if fmt == "poscar":
        order = sorted(set(syms), key=syms.index)
        with open(path, "w") as fh:
            fh.write(f"{title}\n1.0\n")
            for v in lat:
                fh.write(" ".join(f"{x:14.8f}" for x in v) + "\n")
            fh.write(" ".join(order) + "\n" + " ".join(str(syms.count(s)) for s in order) + "\nDirect\n")
            for s in order:
                for sym, f in zip(syms, frac):
                    if sym == s:
                        fh.write(" ".join(f"{x:14.8f}" for x in f) + "\n")
        return
    a, b, c = (np.linalg.norm(v) for v in lat)
    al = math.degrees(math.acos(np.dot(lat[1], lat[2]) / (b * c)))
    be = math.degrees(math.acos(np.dot(lat[0], lat[2]) / (a * c)))
    ga = math.degrees(math.acos(np.dot(lat[0], lat[1]) / (a * b)))
    with open(path, "w") as fh:
        fh.write(f"data_{title}\n_symmetry_space_group_name_H-M 'P 1'\n")
        for k, v in (("a", a), ("b", b), ("c", c), ("alpha", al), ("beta", be), ("gamma", ga)):
            fh.write(f"_cell_{'length_' if len(k) == 1 else 'angle_'}{k} {v:.6f}\n")
        fh.write("loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
                 "_atom_site_fract_z\n")
        for i, (s, f) in enumerate(zip(syms, frac)):
            fh.write(f"{s}{i + 1} {s} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f}\n")
