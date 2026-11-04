# -*- coding: utf-8 -*-
"""Four benchmark crystals: B3LYP (Gaussian dimer projection), built-in EHT and built-in ZINDO/MOO couplings of every dimer group."""
import glob, json, sys
import numpy as np
from mobility_tool.crystal import read_cif, find_molecules, find_dimers
from mobility_tool.eht import eht_transfer_integrals
from mobility_tool.zindo import MOO

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent   # repository folder Mobilis_anisotropy_benchmark
B = str(ROOT) + "/"
SRC = {m: (None, m) for m in ("naphthalene", "anthracene", "perylene", "phenanthrene")}
out = {}
for mol, (d, tag) in SRC.items():
    lam = json.load(open(B + f"2 reorganization energies/{tag}_lambda.json"))["results"]
    dft = {}
    for car in ("hole", "electron"):
        r = json.load(open(B + f"3 transferintegrals/{tag}_mobility_{car}.json"))[0]
        for p in r["pathways"]:
            g = int(p["name"].split("-")[0][1:])
            dft.setdefault(g, {"r": p["r_A"]})[car] = p["J_meV"]
    crys = read_cif(B + f"1 structure/{mol}.cif")
    mols = find_molecules(crys)
    res = find_dimers(crys, mols, ref=0, cutoff=4.0)
    A = [(s, *x) for s, x in res["atoms_a"]]
    sym = [a[0] for a in A]
    Xa = np.array([a[1:] for a in A])
    moo = MOO(sym, Xa)
    rows = []
    for g in res["groups"]:
        dm = g["members"][0]
        Bat = [(s, *x) for s, x in dm["atoms_b"]]
        Xb = np.array([a[1:] for a in Bat])
        e = eht_transfer_integrals(A, Bat, esd=False)["results"]
        z = moo.coupling_atoms(Xa, Xb, sym_b=[a[0] for a in Bat])
        row = {"group": g["id"], "r": dm["r_centroid"], "mult": len(g["members"]),
               "eht_h": abs(e["HOMO"]["J_eff_meV"]), "eht_e": abs(e["LUMO"]["J_eff_meV"]),
               "zindo_h": abs(z["hole"]) * 1000, "zindo_e": abs(z["electron"]) * 1000}
        dg = dft.get(g["id"])
        if dg is None or abs(dg["r"] - dm["r_centroid"]) > 0.01:
            raise SystemExit(f"{mol} G{g['id']}: group index or distance mismatch {dg} vs {dm['r_centroid']:.3f}")
        row["dft_h"], row["dft_e"] = dg["hole"], dg["electron"]
        rows.append(row)
        print(f"{mol:13s} G{g['id']} r {dm['r_centroid']:6.3f}  h: DFT {row['dft_h']:7.2f} ZINDO {row['zindo_h']:7.2f} EHT {row['eht_h']:7.2f}"
              f"   e: DFT {row['dft_e']:7.2f} ZINDO {row['zindo_e']:7.2f} EHT {row['eht_e']:7.2f}", flush=True)
    out[mol] = {"lambda": {"hole": lam["hole"]["lambda_eV"], "electron": lam["electron"]["lambda_eV"]}, "groups": rows,
                "point_ops": res.get("point_ops")}
json.dump(out, open("couplings.json", "w"), indent=1)
