# -*- coding: utf-8 -*-
"""Benchmark: coupling source × rate expression × tensor estimator, against time-of-flight data."""
import json, sys
import numpy as np
from mobility_tool.crystal import read_cif, find_molecules, find_dimers, symmetrize_tensor
from mobility_tool.kmc import graph_from_crystal, rate_marcus, rate_jortner, run_kmc
from mobility_tool.mobility import KB_EV

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent   # repository folder Mobilis_anisotropy_benchmark
B = str(ROOT) + "/"
C = json.load(open("couplings.json"))
HW, LAM_S = 0.1736, 0.020           # MLJ: one effective mode of 1400 cm⁻¹ carries λ − λ_s; λ_s is the classical low-frequency part
EXP = {  # 300 K except phenanthrene (230 K); orthogonal a, b, c′ (c* for perylene); None = no data
    ("naphthalene", "hole"): {"T": 300, "aa": 1.0, "bb": 1.47, "cc": None},
    ("naphthalene", "electron"): {"T": 300, "aa": 0.6, "bb": 0.64, "cc": 0.4},
    ("anthracene", "hole"): {"T": 300, "aa": 1.140, "bb": 2.930, "cc": 0.846, "ac": -0.366},
    ("anthracene", "electron"): {"T": 300, "aa": 1.683, "bb": 1.156, "cc": 0.384, "ac": -0.324},
    ("perylene", "electron"): {"T": 300, "aa": 2.3, "bb": 5.53, "cc": 0.27},
    ("phenanthrene", "hole"): {"T": 230, "aa": 6.0, "bb": 8.1, "cc": 2.1},
}

# EHT calibration: one through-origin scale factor per carrier (all dimer groups with DFT > 1 meV)
K = {}
for car, s in (("hole", "h"), ("electron", "e")):
    x = np.array([g[f"eht_{s}"] for m in C.values() for g in m["groups"] if g[f"dft_{s}"] > 1])
    y = np.array([g[f"dft_{s}"] for m in C.values() for g in m["groups"] if g[f"dft_{s}"] > 1])
    K[car] = float((x @ y) / (x @ x))
print("EHT scale factors", K)


def rate(j_ev, lam, T, model):
    if model == "marcus":
        return rate_marcus(j_ev, lam, 0.0, T)
    return rate_jortner(j_ev, lam - LAM_S, LAM_S, 0.0, T, HW)


def tensors(mol, car, src, model, T):
    crys = read_cif(B + f"1 structure/{mol}.cif")
    mols = find_molecules(crys)
    s = "h" if car == "hole" else "e"
    jg = {}
    for g in C[mol]["groups"]:
        j = g[f"dft_{s}"] if src == "DFT" else g[f"zindo_{s}"] if src == "ZINDO" else K[car] * g[f"eht_{s}"]
        jg[g["group"]] = {car: j}
    lam = C[mol]["lambda"][car]
    kt = KB_EV * T
    # neighbours of one reference molecule (formula of Wen et al. and single-site random-walk tensor)
    res = find_dimers(crys, mols, ref=0, cutoff=4.0)
    v = np.array([d["vector"] for d in res["dimers"]]) * 1e-8
    k = np.array([rate(jg[d["group"]][car] / 1000, lam, T, model) for d in res["dimers"]])
    ops = res["point_ops"]
    rw = symmetrize_tensor(np.einsum("i,ia,ib->ab", 0.5 * k, v, v) / kt, ops)
    P = k / k.sum()
    wen = symmetrize_tensor(np.einsum("i,ia,ib->ab", k * P, v, v) / (2 * kt), ops)
    # exact random walk on the full crystal graph (homogenized, = long-time limit of KMC)
    G = graph_from_crystal(crys, mols, jg, 4.0, carrier=car)
    n = G["n_sites"]
    E = [(int(i), int(jj), np.asarray(vv, float) * 1e-8, rate(jj_, lam, T, model)) for (i, jj, vv, jj_, m, _t) in G["edges"]]
    L = np.zeros((n, n)); b = np.zeros((n, 3))
    for i, jj, vc, kk in E:
        b[i] += kk * vc; L[i, i] -= kk; L[i, jj] += kk
    chi = np.linalg.lstsq(L, -b, rcond=None)[0]
    D = np.zeros((3, 3))
    for i, jj, vc, kk in E:
        u = vc + chi[jj] - chi[i]
        D += 0.5 * kk * np.outer(u, u)
    exact = D / n / kt
    return {"wen": wen, "rw": rw, "exact": exact, "graph": G, "lam": lam}


def comps(M):
    return {"aa": M[0, 0], "bb": M[1, 1], "cc": M[2, 2], "ac": M[0, 2]}


rows = []
for (mol, car), ex in EXP.items():
    T = ex["T"]
    for src in ("DFT", "ZINDO", "EHT"):
        for model in ("marcus", "mlj"):
            t = tensors(mol, car, src, model, T)
            for est in ("wen", "rw", "exact"):
                c = comps(t[est])
                err = [np.log10(c[q] / ex[q]) for q in ("aa", "bb", "cc") if ex.get(q)]
                rows.append({"mol": mol, "car": car, "T": T, "src": src, "rate": model, "est": est, **{q: float(c[q]) for q in c},
                             "log_err": [float(e) for e in err], "rms_log": float(np.sqrt(np.mean(np.square(err))))})
                print(f"{mol:12s} {car:8s} {src:5s} {model:6s} {est:5s}  aa {c['aa']:7.3f} bb {c['bb']:7.3f} cc {c['cc']:7.3f}"
                      f"  rms log10 {rows[-1]['rms_log']:.2f}", flush=True)
json.dump({"K_eht": K, "HW": HW, "LAM_S": LAM_S, "exp": {f"{m}|{c}": v for (m, c), v in EXP.items()}, "rows": rows},
          open("models.json", "w"), indent=1)
