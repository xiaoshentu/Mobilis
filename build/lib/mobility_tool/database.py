"""Coupling-surface database: one folder per crystal with the DFT dimer data and the derived transport quantities.

Layout of a database folder::

    <db>/<key>/record.json      metadata, levels of theory, active-learning summary, thermal statistics, mobility
    <db>/<key>/dimers.csv       every finished DFT dimer: geometry (z, dx, dy, gamma, pitch, roll), V, J, S, site
                                energies and the energy decomposition (same columns as all_results.csv)
    <db>/<key>/structures.xyz   Cartesian coordinates of those dimers (multi-frame xyz)
    <db>/<key>/files/           copies of the input files given on the command line (CIF, lambda json, csv files)

Crystal structures taken from the CSD are not redistributable: give the deposition number with ``ccdc`` and leave the
CIF out with ``copy_cif=False`` when the database is going to be published.
"""
import csv
import hashlib
import io
import json
import shutil
import time
from pathlib import Path

SCHEMA = "mobilis-db-1"
RECORD = "record.json"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() and "." not in str(v) and "e" not in str(v).lower() else f


def read_sections(path):
    """A Mobilis result csv with several tables separated by blank lines → list of tables (lists of dict rows)."""
    text = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
    out = []
    for blk in text.split("\n\n"):
        blk = blk.strip()
        if blk:
            out.append([{k: _num(v) for k, v in r.items()} for r in csv.DictReader(io.StringIO(blk))])
    return [t for t in out if t]


def _md_tables(path):
    """Tables of a ``ceds md`` csv, named by their content."""
    names = {"sigma_meV": "pairs", "mu_hole": "mobility", "U_eq_MD_A2": "md", "shift_z": "pair_dynamics"}
    out = {}
    for t in read_sections(path):
        key = next((n for col, n in names.items() if col in t[0]), f"table{len(out) + 1}")
        out[key] = t
    return out


def _lambda(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    res = d.get("results", {})
    out = {"level": d.get("level"), "sp_level": d.get("sp_level"), "warnings": d.get("warnings", [])}
    for c in ("hole", "electron"):
        if c in res and res[c].get("lambda_eV") is not None:
            out[f"lambda_{c}_meV"] = round(1000.0 * res[c]["lambda_eV"], 2)
    return out


def _al_summary(project, name):
    from .alsurface import ALRun
    run = ALRun(project, name)
    if not run.exists:
        raise FileNotFoundError(f"no active-learning task '{name}' in {project.root}")
    st = run.state
    last = st["history"][-1] if st.get("history") else []
    out = {"name": name, "T_K": st.get("T"), "goal": st.get("goal", "sigma"), "target": st.get("target"),
           "mu_target": st.get("mu_target"), "delta_learning": bool(st.get("delta", False)),
           "baseline": "EHT" if st.get("delta") else None, "rounds": len(st.get("rounds", [])),
           "seeds": list(st.get("seeds", [])), "pairs": [int(g) for g in st.get("pairs", [])],
           "n_dft": {str(r["gid"]): int(r["n"]) for r in last if "gid" in r and "n" in r},
           "converged": sorted(int(g) for g in st.get("converged", {})),
           "scans": [r["scan"] for r in st.get("rounds", [])]}
    out["n_dft_total"] = sum(out["n_dft"].values())
    if st.get("mu_history"):
        m = st["mu_history"][-1]
        out["mu_posterior"] = {"mu": m.get("mu"), "half_width": m.get("half"), "target": m.get("target"),
                               "converged": m.get("converged")}
    return out


def add_record(db, key, project, cif=None, al=None, reorg=None, md_csv=None, validation=None, ccdc="", note="",
               copy_cif=True, overwrite=False):
    """Store one crystal. Returns the record (dict)."""
    from . import __version__
    from .scanflow import ScanProject, write_records
    pr = project if isinstance(project, ScanProject) else ScanProject(project)
    d = Path(db) / key
    if (d / RECORD).exists() and not overwrite:
        raise FileExistsError(f"{d} already holds a record; use overwrite to replace it")
    if d.exists() and overwrite:
        shutil.rmtree(d)
    (d / "files").mkdir(parents=True)

    recs = [r for r in pr.collect_all() if r.get("state") == "done" and not r.get("rep")]
    if not recs:
        shutil.rmtree(d)
        raise ValueError(f"{pr.root} has no finished dimer calculation")
    write_records(recs, d / "dimers.csv")
    with open(d / "structures.xyz", "w") as fh:
        for r in recs:
            A, B = pr.dimer_atoms(r)
            vh = "" if r["hole"] is None else f" Vh={r['hole']:.3f}"
            ve = "" if r["electron"] is None else f" Ve={r['electron']:.3f}"
            fh.write(f"{len(A) + len(B)}\n{r['scan']} {r['id']} group={r.get('group', '')} z={r['z']} dx={r['dx']} "
                     f"dy={r['dy']} gamma={r['gamma']} pitch={r['pitch']} roll={r['roll']}{vh}{ve} meV\n")
            for s, x, y, z in A + B:
                fh.write(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")

    by_scan, by_group = {}, {}
    for r in recs:
        by_scan[r["scan"]] = by_scan.get(r["scan"], 0) + 1
        g = str(r.get("group", ""))
        by_group[g] = by_group.get(g, 0) + 1
    rep = pr.meta["monomer"].get("report", {})
    rec = {"schema": SCHEMA, "key": key, "created": time.strftime("%Y-%m-%d %H:%M"),
           "software": f"Mobilis {__version__}", "note": note,
           "molecule": {"formula": rep.get("formula"), "point_group": rep.get("point_group"),
                        "n_atoms": len(pr.meta["monomer"]["symbols"]), "frame": pr.meta.get("frame", "all")},
           "structure": {"ccdc": str(ccdc or "")},
           "levels": pr.meta.get("levels", {}),
           "dimers": {"n": len(recs), "by_scan": by_scan, "by_group": by_group,
                      "energy_columns": sorted({k for r in recs for k in r if k.startswith("E_")})},
           "files": {}}

    def keep(path, name=None):
        p = Path(path)
        tgt = d / "files" / (name or p.name)
        shutil.copy(p, tgt)
        rec["files"][f"files/{tgt.name}"] = _sha256(tgt)
        return tgt

    if cif:
        from .crystal import read_cif
        crys = read_cif(cif)
        rec["structure"].update({"cell": [round(float(x), 5) for x in crys["cell"]],
                                 "spacegroup": crys.get("spacegroup", ""), "source_file": Path(cif).name,
                                 "sha256": _sha256(cif)})
        if copy_cif:
            keep(cif)
    if reorg:
        rec["reorganization"] = _lambda(reorg)
        keep(reorg)
    if al:
        rec["active_learning"] = _al_summary(pr, al)
    if md_csv:
        rec["thermal"] = _md_tables(md_csv)
        keep(md_csv)
    if validation:
        t = read_sections(validation)
        rec["validation"] = t[0]
        keep(validation)
    for f in ("dimers.csv", "structures.xyz"):
        rec["files"][f] = _sha256(d / f)
    (d / RECORD).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def records(db):
    out = []
    for p in sorted(Path(db).glob(f"*/{RECORD}")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def _mu300(rec):
    rows = rec.get("thermal", {}).get("mobility", [])
    if not rows:
        return None, None, None
    r = min(rows, key=lambda x: abs(float(x["T_K"]) - 300.0))
    return r["T_K"], r.get("mu_hole"), r.get("mu_electron")


def summary_rows(db):
    out = []
    for r in records(db):
        al, lam = r.get("active_learning", {}), r.get("reorganization", {})
        T, mh, me = _mu300(r)
        out.append({"key": r["key"], "formula": r["molecule"].get("formula"), "ccdc": r["structure"].get("ccdc", ""),
                    "n_dimers": r["dimers"]["n"], "pairs": len([g for g in r["dimers"]["by_group"] if g]),
                    "al_dft": al.get("n_dft_total", ""), "al_rounds": al.get("rounds", ""),
                    "lambda_h_meV": lam.get("lambda_hole_meV", ""), "lambda_e_meV": lam.get("lambda_electron_meV", ""),
                    "T_K": "" if T is None else T, "mu_h": "" if mh is None else mh, "mu_e": "" if me is None else me,
                    "validated": "yes" if r.get("validation") else "", "software": r.get("software", "")})
    return out


def summary_table(rows):
    if not rows:
        return "(empty database)"
    cols = ["key", "formula", "ccdc", "n_dimers", "pairs", "al_dft", "al_rounds", "lambda_h_meV", "lambda_e_meV",
            "T_K", "mu_h", "mu_e", "validated"]

    def fmt(v):
        return f"{v:.3g}" if isinstance(v, float) else str(v)
    w = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in cols}
    lines = ["  ".join(c.ljust(w[c]) for c in cols)]
    lines += ["  ".join(fmt(r[c]).ljust(w[c]) for c in cols) for r in rows]
    return "\n".join(lines)


def export(db, out):
    """One table with the dimers of every crystal (column ``crystal`` first) for model training, plus <out>_summary.csv."""
    out = Path(out)
    rows, keys = [], ["crystal"]
    for r in records(db):
        with open(Path(db) / r["key"] / "dimers.csv", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                row = {"crystal": r["key"], **row}
                for k in row:
                    if k not in keys:
                        keys.append(k)
                rows.append(row)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    srows = summary_rows(db)
    sp = out.with_name(out.stem + "_summary.csv")
    if srows:
        with open(sp, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(srows[0]))
            w.writeheader()
            w.writerows(srows)
    return len(rows), len(srows), sp
