"""Transport-regime diagnosis from a crystal structure (command `diagnose`).

From the CIF and the reorganization energies, with built-in EHT couplings and rigid-body lattice dynamics only
(a few seconds per crystal), the routine reports for each carrier:

  * the largest coupling J and the ratio 2|J|/λ (nonadiabatic hopping needs 2|J|/λ < 1, Section 6.2 of the paper);
  * the thermal fluctuation σ_J/|J| of the leading pairs from harmonic phonons (ceds phonons), which sets the
    dynamic disorder of transient localization theory (Fratini et al., Adv. Funct. Mater. 26 (2016) 2292;
    Nat. Mater. 16 (2017) 998);
  * the ratio of the largest interlayer to the largest in-plane coupling (hopping underestimates the
    interlayer mobility when it is small, Section 6.5);
  * whether the frontier orbitals are π orbitals;
  * whether J_max exceeds 2k_BT in the hopping range (then the TLT estimate is reported for comparison);

and recommends a transport model and the Mobilis command for it. The thresholds are those used in the paper and
in the cited work; they mark where a model stops being justified, not sharp physical transitions.

With uq=True (option --uq) the regime is also given as a probability. The largest coupling is drawn from the error
distribution of the largest scaled EHT coupling against the largest B3LYP dimer-projection coupling of the same crystal
and carrier (four-crystal benchmark, Section 5.2), λ from a log-normal distribution of relative width lam_rel, and σ_J/|J| from a log-normal of width
SIGMA_REL (harmonic phonons against MD, Section 5.4). Hopping is the probability of 2|J_max|/λ < 1, transient
localization that of 2|J_max|/λ ≥ 1 with σ_J/|J| ≥ 0.1, the band-like limit the rest. The variance of ln(2|J_max|/λ)
is split into its coupling and λ parts, which tells which input limits the assignment.
"""
import json

import numpy as np

# EHT → B3LYP dimer projection, through-origin factors of the four-crystal benchmark (Section 5.2)
K_EHT = {"hole": 2.87, "electron": 2.14}
HOP_LIMIT = 1.0          # 2|J|/λ below which nonadiabatic (Marcus) hopping is used
BAND = (0.6, 1.5)        # with EHT couplings (factor ~1.5 from DFT) the regime is not decided inside this range
INTERLAYER_WEAK = 0.2    # J_inter / J_intra below which the hopping interlayer mobility is not reliable
SIGMA_LOW = 0.1          # σ_J/|J| below which dynamic disorder is weak (band-like limit of TLT)
# ln(J_max,EHT·k / J_max,DFT): mean and standard deviation over naphthalene, anthracene, α-perylene and phenanthrene
EHT_ERR = {"hole": (-0.057, 0.088), "electron": (-0.059, 0.297)}
SIGMA_REL = 0.15         # log-normal width of the harmonic σ_J/|J| (phonons against MD: 7-22 %, Section 5.4)
LAM_REL = 0.10           # default relative uncertainty of λ (functional dependence of four-point λ)
P_DECIDED = 0.9          # probability above which one model is recommended


def _supercell(lat, length=16.0, nmax=4):
    L = np.linalg.norm(np.asarray(lat, float), axis=1)
    return tuple(int(min(nmax, max(2, np.ceil(length / x)))) for x in L)


def _layer_normal(vectors, r):
    """Normal of the molecular layer: the plane of the two shortest non-parallel neighbour vectors
    (the herringbone or brickwork layer), independent of the carrier."""
    order = np.argsort(np.asarray(r))
    v1 = np.asarray(vectors[order[0]], float)
    for k in order[1:]:
        v2 = np.asarray(vectors[k], float)
        n = np.cross(v1, v2)
        if np.linalg.norm(n) > 0.25 * np.linalg.norm(v1) * np.linalg.norm(v2):
            return n / np.linalg.norm(n)
    return None


def diagnose(cif, lam_h, lam_e, T=300.0, cutoff=4.0, supercell=None, pi=False, phonons=True, uq=False, n_samples=20000,
             lam_rel=LAM_REL, seed=0):
    from .ceds import CEDS, EHTSurface
    from .crystal import find_molecules, read_cif
    from .eht import monomer, pi_frontier
    crys = read_cif(cif)
    mols = find_molecules(crys)
    big = max(len(m["symbols"]) for m in mols)
    n_solv = sum(len(m["symbols"]) != big for m in mols)
    mols = [m for m in mols if len(m["symbols"]) == big]
    ce = CEDS(crys, mols, cutoff=cutoff, frame="all")
    out = {"cif": str(cif), "T": T, "Z": len(mols), "solvent_removed": n_solv, "lambda": {"hole": lam_h, "electron": lam_e},
           "notes": list(ce.notes), "warnings": []}
    mo = monomer(ce.sym, ce.Y)
    _, _, piinfo = pi_frontier(mo, ce.sym, ce.Y)
    out["pi_frontier"] = piinfo
    surfs = {m: EHTSurface(ce.sym, ce.Y, mirror=m) for m in (False, True)}
    surf = {p["gid"]: surfs[bool(p.get("mirror"))] for p in ce.pairs}
    V0 = {}
    for p in ce.pairs:
        s = surf[p["gid"]]
        V0[p["gid"]] = {c: float(np.asarray(s(np.array([p["q0"]]), c)).ravel()[0]) for c in ("hole", "electron")}
    sig = {}
    if phonons:
        from . import phonons as PH
        from .rigidmd import RigidCrystalMD
        sc = supercell or _supercell(crys["lattice"])
        md = RigidCrystalMD(crys, ce.frames, (ce.sym, ce.Y), sc)
        lat = PH.RigidLattice(md)
        lat.relax()
        lat.force_constants()
        lat.modes()
        mc = PH.mode_couplings(lat, ce, surf, T=T)
        sig = {g: {c: d[c]["sigma"] for c in ("hole", "electron")} for g, d in mc["pairs"].items()}
        out["lattice_energy_kcal_mol"] = float(md.lattice_energy())
        out["supercell"] = list(sc)
        out["hbar_tau_meV"] = {c: float(v) for c, v in mc.get("hbar_tau_meV", {}).items()}
        if md.lattice_energy() > 0:
            out["warnings"].append("positive lattice energy of the force field at the experimental structure: σ_J not reliable")
    pairs = []
    for p in ce.pairs:
        g = p["gid"]
        row = {"gid": g, "r": float(p["r_centroid"]), "vector": np.asarray(p["vector"], float).tolist(),
               "mult": int(p.get("multiplicity", 1))}
        for c in ("hole", "electron"):
            row[f"J_{c}"] = K_EHT[c] * abs(V0[g][c])
            row[f"sigma_{c}"] = K_EHT[c] * sig[g][c] if g in sig else None
        pairs.append(row)
    out["pairs"] = pairs
    vec = [r["vector"] for r in pairs]
    res = {}
    for car, lam in (("hole", lam_h), ("electron", lam_e)):
        J = np.array([r[f"J_{car}"] for r in pairs])
        k = int(np.argmax(J))
        jmax = float(J[k])
        ratio = 2 * jmax / 1000.0 / lam
        # σ_J/|J| of the leading pathways (J ≥ 0.5 J_max), weighted by J²
        lead = [i for i in range(len(pairs)) if J[i] >= 0.5 * jmax and pairs[i][f"sigma_{car}"] is not None]
        s_rel = (float(np.sqrt(sum(pairs[i][f"sigma_{car}"] ** 2 for i in lead) / sum(J[i] ** 2 for i in lead)))
                 if lead else None)
        n = _layer_normal(vec, [r["r"] for r in pairs])
        inter = None
        if n is not None:
            intra = [J[i] for i in range(len(pairs)) if abs(np.dot(vec[i], n)) < 2.0]
            outp = [J[i] for i in range(len(pairs)) if abs(np.dot(vec[i], n)) >= 2.0]
            inter = (max(outp) / max(intra)) if outp and max(intra) > 0 else 0.0
        if ratio < BAND[0]:
            regime = "hopping"
            cmd = (f"kmc {cif} --lam {lam} --carrier {car}  (exact crystal-graph tensor; or mobility with DFT couplings, "
                   "random-walk estimator)")
        elif ratio < BAND[1]:
            regime = "intermediate (hopping limit 2|J|/λ = 1 within the EHT uncertainty)"
            cmd = (f"kmc {cif} --lam {lam} --carrier {car}  and  tlt {cif} --carrier {car} --sigma "
                   f"{s_rel if s_rel is not None else 0.3:.2f}; decide with DFT couplings of the leading pairs")
        elif s_rel is not None and s_rel < SIGMA_LOW:
            regime = "band-like (weak dynamic disorder)"
            cmd = f"tlt {cif} --carrier {car} --sigma {s_rel:.2f}  (TLT tends to band transport as σ_J → 0)"
        else:
            regime = "transient localization"
            cmd = f"tlt {cif} --carrier {car} --sigma {s_rel if s_rel is not None else 0.3:.2f}"
        warn = []
        if not regime.startswith(("transient", "band")) and inter is not None and inter < INTERLAYER_WEAK:
            warn.append(f"interlayer/in-plane coupling ratio {inter:.2f}: the hopping mobility normal to the layers is "
                        "unreliable (it depends on in-plane delocalization, Section 6.5)")
        if regime.startswith(("transient", "band")) and inter is not None and inter < INTERLAYER_WEAK:
            warn.append(f"transport is two-dimensional (interlayer/in-plane {inter:.2f}): use --plane for the layer")
        kT = 8.617333e-5 * T * 1000.0
        if regime == "hopping" and jmax > 2 * kT:
            warn.append(f"J_max = {jmax:.0f} meV exceeds 2k_BT = {2 * kT:.0f} meV: the carrier is not confined to one "
                        f"molecule between thermal fluctuations; compare with tlt {cif} --carrier {car} --sigma "
                        f"{s_rel if s_rel is not None else 0.3:.2f}")
        if not regime.startswith(("transient", "band")) and s_rel is not None and s_rel > 1.0:
            warn.append(f"σ_J/|J| = {s_rel:.2f}: couplings change sign thermally; use ⟨J²⟩ (ceds md) for the rates")
        fr = piinfo["pi_HOMO"] if car == "hole" else piinfo["pi_LUMO"]
        if fr < 0.5:
            sh = piinfo["HOMO_shift"] if car == "hole" else piinfo["LUMO_shift"]
            warn.append(f"the EHT {'HOMO' if car == 'hole' else 'LUMO'} has {fr:.0%} π character (π orbital at "
                        f"{'HOMO' if car == 'hole' else 'LUMO'}{sh:+d}); check the orbital order of a DFT monomer and "
                        "use --pi if the π orbital is the frontier one")
        res[car] = {"J_max_meV": jmax, "pair_of_J_max": pairs[k]["gid"], "two_J_over_lambda": ratio,
                    "sigma_rel": s_rel, "interlayer_ratio": inter, "regime": regime, "command": cmd, "warnings": warn}
        if uq:
            res[car]["uq"] = regime_probabilities(J, lam, s_rel, car, n_samples, lam_rel, seed)
    out["carriers"] = res
    if uq:
        out["uq_model"] = {"eht_ln_error": EHT_ERR, "lam_rel": lam_rel, "sigma_rel_width": SIGMA_REL, "samples": n_samples,
                           "seed": seed}
    return out


def regime_probabilities(J_meV, lam, s_rel, carrier, n=20000, lam_rel=LAM_REL, seed=0):
    """Monte Carlo probabilities of the transport regimes for one carrier.

    J_meV: scaled EHT couplings of all pairs (meV). The largest one is corrected by the benchmark error of J_max,
    J_max = J_max,EHT·k · exp(-m + s·z) with (m, s) = EHT_ERR[carrier]; λ and σ_J/|J| are log-normal around their values."""
    rng = np.random.default_rng(seed)
    m, s = EHT_ERR[carrier]
    jmax = float(np.max(np.asarray(J_meV, float))) * np.exp(-m + s * rng.standard_normal(n))
    lam_s = lam * np.exp(lam_rel * rng.standard_normal(n))
    r = 2 * jmax / 1000.0 / lam_s
    hop = r < 1.0
    if s_rel is None:
        p = {"hopping": float(hop.mean()), "transient localization": float((~hop).mean()), "band-like": 0.0}
    else:
        sr = s_rel * np.exp(SIGMA_REL * rng.standard_normal(n))
        p = {"hopping": float(hop.mean()), "transient localization": float((~hop & (sr >= SIGMA_LOW)).mean()),
             "band-like": float((~hop & (sr < SIGMA_LOW)).mean())}
    lo, med, hi = (float(x) for x in np.percentile(r, [5, 50, 95]))
    var = {"coupling (EHT calibration)": float(np.var(np.log(jmax))), "reorganization energy": lam_rel ** 2}
    best = max(p, key=p.get)
    return {"probabilities": p, "two_J_over_lambda_median": med, "two_J_over_lambda_90": [lo, hi],
            "variance_ln_ratio": var, "dominant": max(var, key=var.get), "most_probable": best,
            "decided": p[best] >= P_DECIDED}


def format_report(d):
    L = [f"Transport-regime diagnosis: {d['cif']}",
         f"  {d['Z']} molecules per cell" + (f", {d['solvent_removed']} solvent molecules removed" if d["solvent_removed"] else "")
         + f"; T = {d['T']:g} K; couplings: built-in EHT × k (k_h = {K_EHT['hole']}, k_e = {K_EHT['electron']})"]
    if "lattice_energy_kcal_mol" in d:
        L.append(f"  lattice dynamics: supercell {'×'.join(map(str, d['supercell']))}, lattice energy "
                 f"{d['lattice_energy_kcal_mol']:.1f} kcal/mol per molecule")
    for n in d["notes"]:
        L.append(f"  [note] {n}")
    for w in d["warnings"]:
        L.append(f"  [warning] {w}")
    L.append("")
    L.append(f"  {'pair':>4} {'r/Å':>6} {'J_h':>7} {'σ_h':>6} {'J_e':>7} {'σ_e':>6}   (meV)")
    for r in d["pairs"]:
        sh = f"{r['sigma_hole']:6.1f}" if r["sigma_hole"] is not None else "     -"
        se = f"{r['sigma_electron']:6.1f}" if r["sigma_electron"] is not None else "     -"
        L.append(f"  G{r['gid']:<3} {r['r']:6.2f} {r['J_hole']:7.1f} {sh} {r['J_electron']:7.1f} {se}")
    for car, c in d["carriers"].items():
        L.append("")
        s = f"{c['sigma_rel']:.2f}" if c["sigma_rel"] is not None else "-"
        it = f"{c['interlayer_ratio']:.2f}" if c["interlayer_ratio"] is not None else "-"
        L.append(f"[{car}] J_max = {c['J_max_meV']:.1f} meV (G{c['pair_of_J_max']}), λ = {d['lambda'][car]:.3f} eV, "
                 f"2|J|/λ = {c['two_J_over_lambda']:.2f}, σ_J/|J| = {s}, interlayer/in-plane = {it}")
        L.append(f"   regime: {c['regime']}")
        L.append(f"   command: python -m mobility_tool {c['command']}")
        for w in c["warnings"]:
            L.append(f"   [warning] {w}")
        if "uq" in c:
            u = c["uq"]; pr = u["probabilities"]
            L.append(f"   [uq] P(hopping) = {pr['hopping']:.2f}, P(transient localization) = {pr['transient localization']:.2f}, "
                     f"P(band-like) = {pr['band-like']:.2f}")
            L.append(f"        2|J|/λ = {u['two_J_over_lambda_median']:.2f} (90 %: {u['two_J_over_lambda_90'][0]:.2f}-"
                     f"{u['two_J_over_lambda_90'][1]:.2f}); main source of uncertainty: {u['dominant']}")
            L.append("        " + (f"assignment: {u['most_probable']} (P >= {P_DECIDED})" if u["decided"] else
                                 f"borderline (most probable: {u['most_probable']}): compare hopping and TLT, or compute "
                                 "the leading couplings with DFT"))
    L.append("")
    L.append("Angle-resolved mobilities: use the random-walk tensor, not the formula of Wen et al. (Section 6.1).")
    return "\n".join(L)


def write_json(d, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False, default=float)
