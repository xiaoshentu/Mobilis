"""Singlet-exciton transport in molecular crystals: transition charges, TrESP couplings, hopping rates,
diffusion tensor and diffusion length, kinetic Monte Carlo with a finite lifetime, and the thermal fluctuations of the
excitonic couplings on rigid-body MD frames.

Transition charges
  * in-code INDO/S–CIS (ZINDO/S Hamiltonian of zindo.py, singlet CIS in an active space of occupied × virtual
    orbitals). Two-electron integrals in the ZDO approximation: two-centre (μμ|νν) = γ_AB (Mataga–Nishimoto), one-centre
    Coulomb and exchange integrals of INDO. Transition charge of atom A: q_A = √2 Σ_ia X_ia Σ_μ∈A C_μi C_μa.
  * Gaussian: ESP charges fitted to the transition density (TD + Pop=ESP + Density=Transition=n), read from the log.
  Either set can be rescaled to a given transition dipole (from TD-DFT or experiment), as usual for TrESP.
Coupling (TrESP, Madjet, Abdurahman & Renger 2006):  V = f · Σ_I∈A Σ_J∈B q_I q_J / |r_I − r_J|,  f = screening factor.
Rates: Marcus with zero driving force, k = (2π/ħ) V² (4πλkT)^(−1/2) exp(−λ/4kT); λ = exciton reorganization energy
  (S0/S1 four-point energies). Dynamic disorder enters through ⟨V²⟩ = ⟨V⟩² + σ_V².
Diffusion: D = (1/2N) Σ_i Σ_j k_ij r_ij r_ijᵀ (random walk over all molecules of the cell); L along a principal axis
  = (2 D_k τ)^(1/2), three-dimensional L = (2 tr D · τ)^(1/2) with τ the exciton lifetime.
"""
import math

import numpy as np

from .zindo import AU2ANG, AU2EV, _offsets, _one_center, gamma_matrix, indo_scf

COUL = 14.399645          # e² / (4πε₀) in eV Å
DEBYE = 4.803204          # e Å → D
HBAR = 6.582119569e-16    # eV s
KB = 8.617333262e-5       # eV / K


# ----------------------------------------------------------------------------------------------
# INDO/S–CIS
# ----------------------------------------------------------------------------------------------
def _integrals(sym, X):
    """Γ (basis × basis Coulomb matrix: γ_AB between atoms, one-centre (μμ|νν) on the same atom) and the list of
    one-centre exchange integrals (μ, ν, (μν|μν)) with μ < ν."""
    from .zindo import basis_info
    atom, _ = basis_info(sym)
    G = gamma_matrix(sym, X)
    Gam = G[atom][:, atom].copy()
    off = _offsets(sym)
    ex = []
    for a, e in enumerate(sym):
        coul, exch, _ = _one_center(e)
        s = slice(off[a], off[a + 1])
        Gam[s, s] = coul
        n = off[a + 1] - off[a]
        for m in range(n):
            for k in range(m + 1, n):
                if exch[m, k]:
                    ex.append((off[a] + m, off[a] + k, exch[m, k]))
    return Gam, ex


def indo_cis(sym, X, n_states=4, n_occ=12, n_virt=12, scf=None):
    """Singlet INDO/S–CIS. Returns {"states": [{"E_eV", "f", "mu_D" (vector), "q" (transition charges per atom),
    "main" (leading excitations)}], "scf": scf}. Energies relative to the ground state."""
    sym = list(sym)
    X = np.asarray(X, float)
    scf = scf or indo_scf(sym, X)
    C, eps, nocc = scf["C"], scf["e"], scf["nocc"]
    occ = list(range(max(0, nocc - n_occ), nocc))
    vir = list(range(nocc, min(C.shape[1], nocc + n_virt)))
    Co, Cv = C[:, occ], C[:, vir]
    no, nv = len(occ), len(vir)
    Gam, ex = _integrals(sym, X)
    # Coulomb-type products
    T_ov = np.einsum("mi,ma->mia", Co, Cv).reshape(len(C), -1)          # (μ, ia)
    T_oo = np.einsum("mi,mj->mij", Co, Co).reshape(len(C), -1)
    T_vv = np.einsum("ma,mb->mab", Cv, Cv).reshape(len(C), -1)
    iajb = T_ov.T @ Gam @ T_ov
    ijab = (T_oo.T @ Gam @ T_vv).reshape(no, no, nv, nv)
    if ex:
        mu = np.array([e[0] for e in ex]); nu = np.array([e[1] for e in ex]); K = np.array([e[2] for e in ex])
        U_ov = (Co[mu][:, :, None] * Cv[nu][:, None, :] + Co[nu][:, :, None] * Cv[mu][:, None, :]).reshape(len(ex), -1)
        iajb += U_ov.T @ (K[:, None] * U_ov)
        U_oo = (Co[mu][:, :, None] * Co[nu][:, None, :] + Co[nu][:, :, None] * Co[mu][:, None, :]).reshape(len(ex), -1)
        U_vv = (Cv[mu][:, :, None] * Cv[nu][:, None, :] + Cv[nu][:, :, None] * Cv[mu][:, None, :]).reshape(len(ex), -1)
        ijab += (U_oo.T @ (K[:, None] * U_vv)).reshape(no, no, nv, nv)
    de = (eps[vir][None, :] - eps[occ][:, None]).ravel()
    A = np.diag(de) + 2 * iajb - ijab.transpose(0, 2, 1, 3).reshape(no * nv, no * nv)
    w, V = np.linalg.eigh(A)
    from .zindo import basis_info
    atom, _ = basis_info(sym)
    states = []
    for k in range(min(n_states, len(w))):
        x = V[:, k]
        dens = np.sqrt(2) * (T_ov @ x)                                    # transition density on basis functions
        q = np.bincount(atom, weights=dens, minlength=len(sym))
        mu_eA = q @ X
        f = 2.0 / 3.0 * (w[k] / AU2EV) * float(mu_eA @ mu_eA) / AU2ANG ** 2
        top = np.argsort(-x ** 2)[:3]
        main = [(f"H-{no - 1 - j // nv}" if no - 1 - j // nv else "H") + "→" +
                (f"L+{j % nv}" if j % nv else "L") + f" {x[j] ** 2:.2f}" for j in top]
        states.append({"E_eV": float(w[k]), "f": float(f), "mu_D": mu_eA * DEBYE, "q": q, "main": main})
    return {"states": states, "scf": scf, "active": (no, nv)}


# ----------------------------------------------------------------------------------------------
# Transition charges from Gaussian (TD + Pop=ESP + Density=Transition=n)
# ----------------------------------------------------------------------------------------------
def write_tresp_gjf(path, sym, X, state=1, method="B3LYP", basis="6-31G(d)", nproc=8, mem="8GB", title="TrESP"):
    """Gaussian input: TD-DFT, then ESP charges fitted to the transition density of the given state."""
    lines = [f"%nprocshared={nproc}", f"%mem={mem}",
             f"#p TD(NStates={max(5, state)},Root={state}) {method}/{basis} Pop=(ESP,ReadRadii) "
             f"Density=(Transition={state}) IOp(6/33=2)", "", title, "", "0 1"]
    lines += [f"{s:<3s}{x:14.6f}{y:14.6f}{z:14.6f}" for s, (x, y, z) in zip(sym, np.asarray(X, float))]
    lines += ["", ""]
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines))


def read_gaussian_transition_charges(path, n_atoms=None):
    """ESP-fitted charges of the transition density: the last 'ESP charges:' (or 'Charges from ESP fit') block of the
    log. Returns (symbols, q)."""
    text = open(path, errors="replace").read().splitlines()
    starts = [i for i, l in enumerate(text) if l.strip().startswith(("ESP charges:", "Charges from ESP fit"))]
    if not starts:
        raise ValueError(f"{path}: no ESP charge block (run with Pop=ESP Density=Transition=n)")
    i = starts[-1] + 1
    while i < len(text) and len(text[i].split()) < 3:          # column-number header line
        i += 1
    sym, q = [], []
    while i < len(text):
        p = text[i].split()
        if len(p) < 3 or not p[0].isdigit():
            break
        sym.append(p[1]); q.append(float(p[2]))
        i += 1
    if n_atoms and len(q) != n_atoms:
        raise ValueError(f"{path}: {len(q)} charges, expected {n_atoms}")
    return sym, np.array(q)


def rescale(q, X, mu_target_D):
    """Scale transition charges to a transition-dipole magnitude (Debye)."""
    mu = np.linalg.norm(np.asarray(q) @ np.asarray(X, float)) * DEBYE
    return np.asarray(q) * (mu_target_D / mu if mu > 0 else 1.0)


def neutralize(q):
    q = np.asarray(q, float)
    return q - q.mean()


# ----------------------------------------------------------------------------------------------
# TrESP coupling surface (same interface as ceds.EHTSurface: q = (z, dx, dy, γ, θ, φ) → V / meV)
# ----------------------------------------------------------------------------------------------
class TrESPSurface:
    """Excitonic coupling from transition charges q on the reference monomer Y (molecular frame). Returns the same
    value for every carrier key, so that the CEDS / MD machinery can be used with carrier = "hole"."""

    source = "tresp"
    frozen = []

    def __init__(self, Y, q, screening=1.0, mirror=False):
        from .ceds import MIRROR
        self.Y = np.asarray(Y, float)
        self.q = np.asarray(q, float)
        self.f = float(screening)
        self.mirror = bool(mirror)
        self._M = MIRROR

    def eval_one(self, qv):
        from .ceds import build_B
        z, dx, dy, g, p, r = qv
        B = build_B(self.Y, z, dx, dy, g, r, p, mirror=self.mirror)
        d = np.linalg.norm(self.Y[:, None, :] - B[None, :, :], axis=-1)
        v = 1000.0 * self.f * COUL * float(self.q @ (1.0 / d) @ self.q)
        return {"hole": v, "electron": v, "exciton": v}

    def __call__(self, Q, carrier="exciton"):
        return np.array([self.eval_one(q)["exciton"] for q in np.atleast_2d(Q)])

    def coverage(self, Q):
        return 1.0

    def describe(self):
        return f"TrESP (screening {self.f:g})"


def point_dipole(mu_a_D, mu_b_D, r_A, screening=1.0):
    """Point-dipole coupling / meV between dipoles (Debye) at separation r (Å)."""
    ma, mb, r = (np.asarray(x, float) for x in (mu_a_D, mu_b_D, r_A))
    d = np.linalg.norm(r)
    e = r / d
    return 1000.0 * screening * COUL * (ma @ mb - 3 * (ma @ e) * (mb @ e)) / DEBYE ** 2 / d ** 3


# ----------------------------------------------------------------------------------------------
# Crystal: couplings of all neighbours, Davydov splitting, rates, diffusion
# ----------------------------------------------------------------------------------------------
def crystal_molecules(ceds):
    """(axes, origin) of every molecule of the cell: the atoms of molecule m are origin + Y axesᵀ (CEDS frames)."""
    if any(getattr(ceds, "mirror", [])):
        raise ValueError("exciton couplings are not available for crystals with enantiomeric molecules")
    return [(np.asarray(ax, float), np.asarray(c, float)) for ax, c in ceds.frames]


def tresp_pair(Y, q, Ra, ca, Rb, cb, screening=1.0):
    A = ca + Y @ Ra.T
    B = cb + Y @ Rb.T
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return 1000.0 * screening * COUL * float(q @ (1.0 / d) @ q)


def lattice_couplings(ceds, Y, q, screening=1.0, r_max=30.0):
    """TrESP couplings of molecule i (each molecule of the cell) to every molecule within r_max (centroid distance,
    Å). Returns rows {i, j, cell (3 ints), r (vector Å), V (meV)}."""
    lat = np.asarray(ceds.crys["lattice"], float)
    mols = crystal_molecules(ceds)
    nmax = [int(math.ceil(r_max / np.linalg.norm(lat[k]))) + 1 for k in range(3)]
    rows = []
    for i, (Ri, ci) in enumerate(mols):
        for j, (Rj, cj) in enumerate(mols):
            for a in range(-nmax[0], nmax[0] + 1):
                for b in range(-nmax[1], nmax[1] + 1):
                    for c in range(-nmax[2], nmax[2] + 1):
                        if i == j and a == b == c == 0:
                            continue
                        t = a * lat[0] + b * lat[1] + c * lat[2]
                        r = cj + t - ci
                        if np.linalg.norm(r) > r_max:
                            continue
                        rows.append({"i": i, "j": j, "cell": (a, b, c), "r": r,
                                     "V": tresp_pair(Y, q, Ri, ci, Rj, cj + t, screening)})
    return rows


def davydov(rows, n_mol):
    """k = 0 exciton band energies (meV) relative to the monomer: eigenvalues of the n_mol × n_mol matrix
    H_ij = Σ_cells V(i, j, cell). For two molecules per cell the Davydov splitting is their difference."""
    H = np.zeros((n_mol, n_mol))
    for r in rows:
        H[r["i"], r["j"]] += r["V"] / (1 if r["i"] != r["j"] else 1)
    H = 0.5 * (H + H.T)
    w = np.linalg.eigvalsh(H)
    return w, H


def marcus_rate(V_meV, lam_eV, T, dG_eV=0.0):
    V = np.asarray(V_meV, float) / 1000.0
    kT = KB * T
    return 2 * math.pi / HBAR * V ** 2 / math.sqrt(4 * math.pi * lam_eV * kT) * np.exp(-(dG_eV + lam_eV) ** 2 / (4 * lam_eV * kT))


def diffusion_tensor(rows, n_mol, lam_eV, T, v2=None):
    """D (cm² s⁻¹) = (1/2N) Σ_ij k_ij r_ij r_ijᵀ. v2: optional {(i, j, cell): ⟨V²⟩ / meV²} replacing V²."""
    D = np.zeros((3, 3))
    for r in rows:
        V2 = v2.get((r["i"], r["j"], r["cell"]), r["V"] ** 2) if v2 else r["V"] ** 2
        k = marcus_rate(math.sqrt(V2), lam_eV, T)
        D += k * np.outer(r["r"], r["r"])
    D /= 2 * n_mol
    return D * 1e-16                       # Å² → cm²


def diffusion_length(D_cm2s, tau_s):
    """Principal values (cm² s⁻¹), axes, L along each axis = (2 D_k τ)^(1/2) and 3D L = (2 tr D τ)^(1/2), in nm."""
    w, U = np.linalg.eigh(D_cm2s)
    idx = np.argsort(-w)
    w, U = w[idx], U[:, idx]
    L = np.sqrt(2 * np.clip(w, 0, None) * tau_s) * 1e7
    return {"D": w, "axes": U.T, "L_nm": L, "L3_nm": float(math.sqrt(2 * max(np.trace(D_cm2s), 0) * tau_s) * 1e7)}


def kmc_diffusion(rows, n_mol, lam_eV, T, tau_s, n_walkers=2000, seed=1, v2=None, max_steps=2_000_000):
    """Kinetic Monte Carlo of excitons with an exponential lifetime τ, all walkers advanced together. Returns the mean
    square displacement at decay (cm²) with its standard error, L3 = ⟨r²⟩^1/2 (nm) and the mean number of hops. For
    long lifetimes the walk is stopped after max_steps hops of the slowest walker and ⟨r²⟩ is extrapolated linearly in
    time (exact for a random walk)."""
    rng = np.random.default_rng(seed)
    by_i = [[] for _ in range(n_mol)]
    for r in rows:
        V2 = v2.get((r["i"], r["j"], r["cell"]), r["V"] ** 2) if v2 else r["V"] ** 2
        by_i[r["i"]].append((r["j"], np.asarray(r["r"], float), float(marcus_rate(math.sqrt(V2), lam_eV, T))))
    m = max(len(x) for x in by_i)
    J = np.zeros((n_mol, m), int); RV = np.zeros((n_mol, m, 3)); CUM = np.ones((n_mol, m)); KT = np.zeros(n_mol)
    for i, lst in enumerate(by_i):
        k = np.array([x[2] for x in lst])
        J[i, :len(lst)] = [x[0] for x in lst]
        RV[i, :len(lst)] = [x[1] for x in lst]
        CUM[i, :len(lst)] = np.cumsum(k) / k.sum()
        KT[i] = k.sum()
    site = rng.integers(n_mol, size=n_walkers)
    pos = np.zeros((n_walkers, 3))
    t = np.zeros(n_walkers)
    life = rng.exponential(tau_s, size=n_walkers)
    alive = np.ones(n_walkers, bool)
    hops = np.zeros(n_walkers)
    steps = 0
    while alive.any() and steps < max_steps:
        idx = np.flatnonzero(alive)
        dt = rng.exponential(1.0 / KT[site[idx]])
        dead = t[idx] + dt > life[idx]
        alive[idx[dead]] = False
        go = idx[~dead]
        t[go] += dt[~dead]
        u = rng.random(len(go))
        c = (CUM[site[go]] < u[:, None]).sum(1)
        pos[go] += RV[site[go], c]
        site[go] = J[site[go], c]
        hops[go] += 1
        steps += 1
    r2 = (pos ** 2).sum(1)
    if alive.any():                              # unfinished walkers: scale ⟨r²⟩ to the full lifetime
        r2[alive] *= life[alive] / np.maximum(t[alive], 1e-30)
    r2 = r2 * 1e-16
    return {"msd_cm2": float(r2.mean()), "msd_err": float(r2.std() / math.sqrt(len(r2))),
            "L3_nm": float(math.sqrt(r2.mean()) * 1e7), "hops": float(hops.mean()), "finished": float(1 - alive.mean())}


def _mean_rotation(R):
    """Rotation closest to the arithmetic mean of rotation matrices (SVD projection)."""
    U, _, Vt = np.linalg.svd(R.mean(0))
    M = U @ Vt
    if np.linalg.det(M) < 0:
        U[:, -1] *= -1
        M = U @ Vt
    return M


def md_coupling_stats(ceds, md, frames, Y, q, screening=1.0, recenter=True, edges=None):
    """⟨V⟩ and σ_V (meV) of the TrESP coupling of every contact neighbour (ceds.edges) over rigid-body MD frames
    and all cells of the supercell. With recenter, the relative pose of each dimer is taken as the crystal pose times
    the MD fluctuation about its mean (as in the CEDS MD, the force field sets the fluctuations, not the mean structure).
    Returns {(i, j, cell): {"mean", "std", "V0"}}."""
    lat = np.asarray(ceds.crys["lattice"], float)
    Qm, o = md.mol_frames(frames["c"], frames["Q"])
    cells = np.array(md.cells)
    out = {}
    for (i, j, v, t, gid) in (edges if edges is not None else ceds.edges):
        t = tuple(int(x) for x in t)
        axA, cA = ceds.frames[i]
        axB, cB = ceds.frames[j]
        R0 = np.asarray(axA, float).T @ np.asarray(axB, float)
        t0 = np.asarray(axA, float).T @ (np.asarray(cB, float) + np.array(t, float) @ lat - np.asarray(cA, float))
        A = np.array([md.index(cl, i) for cl in md.cells])
        tb = cells + np.array(t)
        B = np.array([md.index(tuple(x), j) for x in tb])
        shift = (tb // np.array(md.sc)) @ md.box
        QA, QB = Qm[:, A], Qm[:, B]
        R = np.einsum("fnji,fnjk->fnik", QA, QB).reshape(-1, 3, 3)
        tv = np.einsum("fnji,fnj->fni", QA, o[:, B] + shift - o[:, A]).reshape(-1, 3)
        if recenter:
            Rm = _mean_rotation(R)
            R = np.einsum("ij,jk,nkl->nil", R0, Rm.T, R)
            tv = t0 + tv - tv.mean(0)
        V = np.empty(len(R))
        for s0 in range(0, len(R), 2000):
            Bx = np.einsum("nij,aj->nai", R[s0:s0 + 2000], Y) + tv[s0:s0 + 2000, None, :]
            d = np.linalg.norm(Y[None, :, None, :] - Bx[:, None, :, :], axis=-1)
            V[s0:s0 + 2000] = 1000.0 * screening * COUL * np.einsum("a,nab,b->n", q, 1.0 / d, q)
        V0 = 1000.0 * screening * COUL * float(q @ (1.0 / np.linalg.norm(Y[:, None] - (Y @ R0.T + t0)[None], axis=-1)) @ q)
        out[(i, j, t)] = {"mean": float(V.mean()), "std": float(V.std()), "V0": V0, "gid": gid}
    return out
