# -*- coding: utf-8 -*-
"""Molden files (ORCA via orca_2mkl -molden, xtb --molden, and most other programs) and the dimer projection
method on them.

Only what the projection needs is read: atoms, the shell structure ([GTO]: which atom, which angular momentum, how
many functions) and the MO coefficients, energies and occupations ([MO]). The projection uses the dimer MO
coefficients C_D and energies only (H = G ε Gᵀ, S = G Gᵀ with Gᵀ = C_D⁻¹ blockdiag(C_A, C_B)), so normalization and
sign conventions of the basis functions cancel as long as the three calculations come from the same program with the
same basis: the AO basis of the dimer is that of A followed by that of B.

Phase of B. A separate calculation of monomer B gives its orbitals an arbitrary sign. With `T` (the 3×3 matrix that
maps the frame of A onto B: B = A Tᵀ + t, a rotation, or a rotation times a mirror for enantiomer pairs) the p
coefficients of B's orbital are compared with those of the reference orbital of A transformed by T, and B's orbital
is flipped when they point the other way. Together with a fixed reference calculation of A this gives couplings with
consistent signs over a whole scan, as the frozen fragments do in ADF."""
import re

import numpy as np

HARTREE_EV = 27.211386245988
BOHR = 0.529177210903
_NCART = {0: 1, 1: 3, 2: 6, 3: 10, 4: 15}
_NPURE = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}
_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
_Z = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg",
      13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 34: "Se", 35: "Br", 53: "I"}


def read_molden(path):
    """Returns dict(atoms=[(symbol, x, y, z) Å], shells=[(atom index, l, n functions)], C (nbf × nmo), e (Hartree),
    occ, nbf, nmo, nocc (closed shell: number of orbitals with occupation ≥ 1))."""
    text = open(path, encoding="utf-8", errors="ignore").read()
    sec = {}
    parts = re.split(r"^\s*\[([^\]]+)\]([^\n]*)\n", text, flags=re.M)
    for k in range(1, len(parts) - 1, 3):
        sec[parts[k].strip().upper()] = (parts[k + 1].strip(), parts[k + 2])
    if "MO" not in sec or "GTO" not in sec:
        raise ValueError(f"{path}: not a Molden file with [GTO] and [MO] sections")
    pure_d = any(k in sec for k in ("5D", "5D7F", "5D10F"))
    pure_f = any(k in sec for k in ("7F", "5D7F"))
    pure_g = "9G" in sec
    atoms = []
    if "ATOMS" in sec:
        unit, body = sec["ATOMS"]
        f = BOHR if "AU" in unit.upper() else 1.0
        for ln in body.splitlines():
            t = ln.split()
            if len(t) >= 6:
                z = int(t[2])
                atoms.append((_Z.get(z, re.sub(r"\d", "", t[0]).capitalize()), float(t[3]) * f, float(t[4]) * f,
                              float(t[5]) * f))
    shells, atom = [], None
    lines = sec["GTO"][1].splitlines()
    i = 0
    while i < len(lines):
        t = lines[i].split()
        i += 1
        if not t:
            atom = None
            continue
        if atom is None and len(t) >= 1 and t[0].isdigit():
            atom = int(t[0]) - 1
            continue
        if t[0].lower() in _L and len(t) >= 2:
            l = _L[t[0].lower()]
            nprim = int(float(t[1]))
            pure = (l == 2 and pure_d) or (l == 3 and pure_f) or (l == 4 and pure_g)
            shells.append((atom, l, (_NPURE if pure else _NCART)[l]))
            i += nprim
        elif t[0].lower() == "sp":
            raise ValueError(f"{path}: sp shells are not supported (use a basis without sp shells)")
    nbf = sum(s[2] for s in shells)
    mos, cur = [], None
    for ln in sec["MO"][1].splitlines():
        s = ln.strip()
        if not s:
            continue
        m = re.match(r"(\w+)\s*=\s*(\S+)", s)
        if m and not s[0].isdigit():
            key, val = m.group(1).lower(), m.group(2)
            if key == "ene":
                if cur is None or cur["c"]:
                    cur = {"e": 0.0, "occ": 0.0, "spin": "alpha", "c": {}}
                    mos.append(cur)
                cur["e"] = float(val.replace("D", "E"))
            elif key in ("occup", "spin", "sym"):
                if cur is None or cur["c"]:
                    cur = {"e": 0.0, "occ": 0.0, "spin": "alpha", "c": {}}
                    mos.append(cur)
                if key == "occup":
                    cur["occ"] = float(val.replace("D", "E"))
                elif key == "spin":
                    cur["spin"] = val.lower()
            continue
        t = s.split()
        if cur is not None and len(t) >= 2:
            cur["c"][int(t[0]) - 1] = float(t[1].replace("D", "E"))
    alpha = [m for m in mos if m["spin"].startswith("a")]
    if len(alpha) != len(mos):
        print(f"[warning] {path}: open-shell file, only the alpha orbitals are used", flush=True)
    C = np.zeros((nbf, len(alpha)))
    for j, m in enumerate(alpha):
        for k, v in m["c"].items():
            if k < nbf:
                C[k, j] = v
    e = np.array([m["e"] for m in alpha])
    occ = np.array([m["occ"] for m in alpha])
    order = np.argsort(e, kind="stable")
    C, e, occ = C[:, order], e[order], occ[order]
    nocc = int((occ > 0.5).sum())
    return {"atoms": atoms, "shells": shells, "C": C, "e": e, "occ": occ, "nbf": nbf, "nmo": C.shape[1],
            "nocc": nocc, "path": str(path)}


def p_blocks(shells):
    """AO indices of every p shell, (n_p_shells, 3), in file order (x, y, z)."""
    out, off = [], 0
    for _, l, n in shells:
        if l == 1:
            out.append([off, off + 1, off + 2])
        off += n
    return np.array(out, int).reshape(-1, 3)


def frame_map(XA, XB):
    """3×3 T with XB − ⟨XB⟩ ≈ (XA − ⟨XA⟩) Tᵀ (least squares; a rotation, or rotation × mirror)."""
    A = np.asarray(XA, float) - np.mean(XA, 0)
    B = np.asarray(XB, float) - np.mean(XB, 0)
    X, *_ = np.linalg.lstsq(A, B, rcond=None)
    return X.T


def orbital_index(label, nocc):
    m = re.fullmatch(r"(HOMO|LUMO)([+-]\d+)?", label.upper())
    if not m:
        raise ValueError(f"orbital label {label}: use HOMO, HOMO-1, LUMO, LUMO+1, ...")
    return (nocc - 1 if m.group(1) == "HOMO" else nocc) + int(m.group(2) or 0)


def phase_sign(ref, mo_b, idx, T):
    """+1 / −1 that aligns orbital idx of B with the same orbital of the reference A transformed by T, and the
    normalized agreement (|value| close to 1 when B is the rigidly moved A)."""
    pb = p_blocks(mo_b["shells"])
    pa = p_blocks(ref["shells"])
    if len(pa) == 0 or pa.shape != pb.shape:
        return 1.0, float("nan")
    va = ref["C"][pa, idx] @ np.asarray(T, float).T          # (k, 3) p vectors of A moved onto B
    vb = mo_b["C"][pb, idx]
    num = float((va * vb).sum())
    den = float(np.sqrt((va * va).sum() * (vb * vb).sum())) or 1.0
    return (1.0 if num >= 0 else -1.0), num / den


def projection(mo_a, mo_b, mo_d, orbitals=("HOMO", "LUMO"), ref=None, T=None):
    """Dimer projection from three Molden calculations (A, B, AB with the atoms of A first). With ref (the reference
    calculation of A in its own frame; default mo_a) and T (frame of A → frame of B), A is phased like ref and B like
    ref moved by T. Returns {label: J_eff_meV, J_raw_meV, S_AB, e_A_eV, e_B_eV, phase_B, completeness}."""
    na, nb = mo_a["nbf"], mo_b["nbf"]
    if na + nb != mo_d["nbf"]:
        raise ValueError(f"basis functions: A {na} + B {nb} ≠ AB {mo_d['nbf']} (same basis and atom order needed)")
    if mo_d["nmo"] != mo_d["nbf"]:
        raise ValueError(f"{mo_d.get('path', 'dimer')}: {mo_d['nmo']} orbitals for {mo_d['nbf']} basis functions; "
                         "linearly dependent functions were removed (ORCA: %scf Sthresh 1e-10 end, or a basis "
                         "without diffuse functions)")
    if mo_a["nmo"] != na or mo_b["nmo"] != nb:
        raise ValueError("monomer calculations with removed basis functions are not supported")
    cf = np.zeros((na + nb, na + nb))
    cf[:na, :na] = mo_a["C"]
    cf[na:, na:] = mo_b["C"]
    g = np.linalg.solve(mo_d["C"], cf).T
    h = (g * mo_d["e"]) @ g.T
    s = g @ g.T
    ref = ref or mo_a
    out = {}
    for lab in orbitals:
        ia = orbital_index(lab, mo_a["nocc"])
        ib = orbital_index(lab, mo_b["nocc"])
        sa, _ = phase_sign(ref, mo_a, ia, np.eye(3)) if ref is not mo_a else (1.0, 1.0)
        sb, agree = phase_sign(ref, mo_b, ib, T) if T is not None else (1.0, float("nan"))
        sign = sa * sb
        ea, eb = h[ia, ia], h[na + ib, na + ib]
        j, sab = sign * h[ia, na + ib], sign * s[ia, na + ib]
        jeff = (j - 0.5 * (ea + eb) * sab) / (1 - sab ** 2)
        out[lab.upper()] = {"J_eff_meV": jeff * HARTREE_EV * 1000, "J_raw_meV": j * HARTREE_EV * 1000, "S_AB": sab,
                            "e_A_eV": ea * HARTREE_EV, "e_B_eV": eb * HARTREE_EV, "phase_B": agree,
                            "completeness": [float(s[ia, ia]), float(s[na + ib, na + ib])]}
    return out


def write_molden(path, symbols, coords_ang, shells_spec, C, e, occ):
    """Minimal Molden writer (tests and conversions): shells_spec = [(atom, l, [(exponent, coefficient), ...])]."""
    inv = {v: k for k, v in _Z.items()}
    lines = ["[Molden Format]", "[Atoms] Angs"]
    for k, (s, r) in enumerate(zip(symbols, coords_ang), 1):
        lines.append(f"{s:2s} {k:4d} {inv.get(s, 0):3d} {r[0]:14.8f} {r[1]:14.8f} {r[2]:14.8f}")
    lines.append("[GTO]")
    by_atom = {}
    for a, l, prims in shells_spec:
        by_atom.setdefault(a, []).append((l, prims))
    for a in range(len(symbols)):
        lines.append(f"{a + 1:4d} 0")
        for l, prims in by_atom.get(a, []):
            lines.append(f" {'spdfg'[l]} {len(prims):3d} 1.00")
            lines += [f" {x:18.10E} {c:18.10E}" for x, c in prims]
        lines.append("")
    if any(l >= 2 for _, l, _ in shells_spec):
        lines.append("[5D]")
    lines.append("[MO]")
    for j in range(C.shape[1]):
        lines += [" Sym= A", f" Ene= {e[j]:.10f}", " Spin= Alpha", f" Occup= {occ[j]:.6f}"]
        lines += [f"{k + 1:6d} {C[k, j]:20.12E}" for k in range(C.shape[0])]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
