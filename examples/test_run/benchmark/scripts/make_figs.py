# -*- coding: utf-8 -*-
"""Figures 4, 6 and 7 of the paper, computed and drawn with Mobilis functions (mobility_3d, symmetrize_tensor, angular_mobility, plots.*)."""
import json, sys
import numpy as np
from mobility_tool.crystal import read_cif, find_molecules, find_dimers, symmetrize_tensor
from mobility_tool.mobility import mobility_3d, angular_mobility
from mobility_tool import plots

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent   # repository folder Mobilis_anisotropy_benchmark
B = str(ROOT) + "/"
C = json.load(open("couplings.json"))
M = json.load(open("models.json"))
K = M["K_eht"]
EXP = {tuple(k.split("|")): v for k, v in M["exp"].items()}
NAME = {"naphthalene": "Naphthalene", "anthracene": "Anthracene", "perylene": "α-Perylene", "phenanthrene": "Phenanthrene"}
SH = {"naphthalene": "NAP", "anthracene": "ANT", "perylene": "PER", "phenanthrene": "PHE"}

# Figure 4: coupling parity plot
rows = []
for mol, d in C.items():
    for g in d["groups"]:
        rows.append(dict(g, eht_scaled_h=K["hole"] * g["eht_h"], eht_scaled_e=K["electron"] * g["eht_e"]))
plots.figure_coupling_parity(rows, "Fig4_coupling_parity.png")

# Figure 6: angle-resolved mobility in the ab plane (corrected / Wen et al. / experiment)
cases, table = [], []
for (mol, car), ex in EXP.items():
    crys = read_cif(B + f"1 structure/{mol}.cif"); mols = find_molecules(crys)
    res = find_dimers(crys, mols, ref=0, cutoff=4.0)
    s = "h" if car == "hole" else "e"
    J = {g["group"]: g[f"dft_{s}"] / 1000.0 for g in C[mol]["groups"]}
    paths = [{"name": f"G{d['group']}", "vector": d["vector"], "V": J[d["group"]], "count": 1} for d in res["dimers"]]
    lam = C[mol]["lambda"][car]
    out = {}
    for model in ("tensor", "wen"):
        r = mobility_3d(lam, paths, ex["T"], "V", model)
        Mt = symmetrize_tensor(r["M"], res["point_ops"])
        phi, mu = angular_mobility(Mt, "xy")
        out[model] = (phi, mu, Mt)
    phi = out["tensor"][0]
    sel = phi <= np.pi + 1e-9
    case = {"label": f"{NAME[mol]}, {car}s" + (f" ({ex['T']} K)" if ex["T"] != 300 else ""), "phi": phi[sel],
            "mu_tensor": out["tensor"][1][sel], "mu_wen": out["wen"][1][sel], "exp": {}}
    if ex.get("ac") is not None:             # full measured tensor (anthracene): experimental curve in the ab plane
        case["exp_curve"] = ex["aa"] * np.cos(phi[sel]) ** 2 + ex["bb"] * np.sin(phi[sel]) ** 2
    else:
        if ex.get("aa"):
            case["exp"][0.0] = ex["aa"]
        if ex.get("bb"):
            case["exp"][90.0] = ex["bb"]
    cases.append(case)
    for model in ("tensor", "wen"):
        Mt = out[model][2]
        table.append({"mol": mol, "car": car, "model": model, "aa": Mt[0, 0], "bb": Mt[1, 1], "cc": Mt[2, 2]})
plots.figure_angular_benchmark(cases, "Fig5_angular_benchmark.png")

# Figure 7: error of each choice (log10 calc/exp)
R = M["rows"]
def pick(src, rate, est):
    out = {}
    for (mol, car), ex in EXP.items():
        r = next(x for x in R if x["mol"] == mol and x["car"] == car and x["src"] == src and x["rate"] == rate and x["est"] == est)
        for q, lab in (("aa", "a"), ("bb", "b"), ("cc", "c′")):
            if ex.get(q):
                out[f"{SH[mol]} {car[0]} {lab}"] = float(np.log10(r[q] / ex[q]))
    return out
groups = [("(a) tensor estimator (DFT, Marcus)", [("Wen", pick("DFT", "marcus", "wen")), ("corrected", pick("DFT", "marcus", "rw")),
                                                 ("exact (KMC)", pick("DFT", "marcus", "exact"))]),
          ("(b) couplings (Marcus, exact)", [("B3LYP", pick("DFT", "marcus", "exact")), ("ZINDO", pick("ZINDO", "marcus", "exact")),
                                             ("EHT scaled", pick("EHT", "marcus", "exact"))]),
          ("(c) rate (B3LYP, exact)", [("Marcus", pick("DFT", "marcus", "exact")), ("MLJ", pick("DFT", "mlj", "exact"))])]
plots.figure_benchmark_errors(groups, "Fig6_benchmark_errors.png")
json.dump({"angular_table": table, "groups": groups}, open("fig_data.json", "w"), indent=1, default=float)
for t, vs in groups:
    print(t, {n: round(float(np.sqrt(np.mean(np.square(list(v.values()))))), 2) for n, v in vs})
