# -*- coding: utf-8 -*-
"""Project files (*.mobilis.json): one file that keeps every setting of a study and can run it again.

    {
      "mobilis_project": 1,
      "name": "pentacene",
      "variables": {"cif": "pentacene.cif", "T": "300"},
      "steps": [
        {"name": "couplings", "command": "eht-ti", "args": ["${cif}", "--out", "eht.json"]},
        {"name": "tlt_hole",  "command": "tlt",    "args": ["${cif}", "--carrier", "hole", "--temp", "${T}"]}
      ],
      "gui": {"mob_tab.lam_h": "0.092", ...}          # written by the GUI (File → Save project), optional
    }

`project run` executes the steps in order as `python -m mobility_tool <command> <args>` in the folder of the
project file (relative paths are relative to it), writes each step's output to logs/<name>.log and the status of
every step to <project>.status.json. `${name}` is replaced by the variables; --set name=value overrides them."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

VERSION = 1

TEMPLATES = {
    "crystal": {
        "description": "charge transport in a molecular crystal with EHT couplings (no external program needed)",
        "variables": {"cif": "crystal.cif", "T": "300", "lam_h": "0.10", "lam_e": "0.15"},
        "steps": [
            {"name": "diagnose", "command": "diagnose", "args": ["${cif}", "--lam-h", "${lam_h}", "--lam-e", "${lam_e}",
                                                             "--temp", "${T}", "--json", "diagnose.json"]},
            {"name": "tlt_hole", "command": "tlt", "args": ["${cif}", "--carrier", "hole", "--temp", "${T}"]},
            {"name": "tlt_electron", "command": "tlt", "args": ["${cif}", "--carrier", "electron", "--temp", "${T}"]},
            {"name": "kmc_hole", "command": "kmc", "args": ["${cif}", "--lam", "${lam_h}", "--carrier", "hole",
                                                             "--temp", "${T}"]},
            {"name": "kmc_electron", "command": "kmc", "args": ["${cif}", "--lam", "${lam_e}", "--carrier",
                                                                 "electron", "--temp", "${T}"]},
        ],
    },
    "emitter": {
        "description": "S1/T1 kinetics of an emitter from S0/S1/T1 freq fchk files (TVCF)",
        "variables": {"s0": "S0.fchk", "s1": "S1.fchk", "t1": "T1.fchk", "fosc": "0.05", "soc": "0.5",
                      "T": "300"},
        "steps": [
            {"name": "tadf", "command": "tadf", "args": ["--s0", "${s0}", "--s1", "${s1}", "--t1", "${t1}", "--fosc",
                                                         "${fosc}", "--soc", "${soc}", "--temps", "${T}", "--out",
                                                         "tadf.json"]},
            {"name": "emission", "command": "tvcf", "args": ["--initial", "${s1}", "--final", "${s0}", "--fosc",
                                                             "${fosc}", "--temps", "77,${T}", "--out", "S1_S0"]},
        ],
    },
    "deformation": {
        "description": "deformation-potential band mobility: strained cells, then the filled strain table",
        "variables": {"cif": "crystal.cif", "table": "crystal_dp_strain_table.csv", "T": "300"},
        "steps": [
            {"name": "strained_cells", "command": "dp-gen", "args": ["${cif}", "--strains=-1,-0.5,0,0.5,1"]},
            {"name": "mobility", "command": "dp", "args": ["${table}", "--cif", "${cif}", "--temp", "${T}",
                                                           "--out", "dp_results.csv"]},
        ],
    },
}


def new_project(path, template="crystal", name=None, variables=None):
    t = TEMPLATES[template]
    data = {"mobilis_project": VERSION, "name": name or Path(path).name.split(".")[0], "template": template,
            "description": t["description"], "variables": dict(t["variables"], **(variables or {})),
            "steps": [dict(s) for s in t["steps"]]}
    save(path, data)
    return data


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "mobilis_project" not in data:
        raise ValueError(f"{path} is not a Mobilis project file")
    if int(data["mobilis_project"]) > VERSION:
        raise ValueError(f"{path} was written by a newer Mobilis (project format {data['mobilis_project']})")
    data.setdefault("variables", {})
    data.setdefault("steps", [])
    return data


def save(path, data):
    from . import __version__
    data = dict(data)
    data["mobilis_project"] = VERSION
    data["written_by"] = f"Mobilis {__version__}"
    data["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand(args, variables):
    out = []
    for a in args:
        def rep(m):
            if m.group(1) not in variables:
                raise KeyError(f"variable ${{{m.group(1)}}} is not defined")
            return str(variables[m.group(1)])
        out.append(_VAR.sub(rep, str(a)))
    return out


def plan(data, only=None, start=None, overrides=None):
    """Steps to run, with the variables substituted: [(name, argv)]."""
    var = dict(data["variables"], **(overrides or {}))
    steps = data["steps"]
    names = [s["name"] for s in steps]
    if start:
        if start not in names:
            raise KeyError(f"no step {start!r} (steps: {', '.join(names)})")
        steps = steps[names.index(start):]
    if only:
        miss = [n for n in only if n not in names]
        if miss:
            raise KeyError(f"no step {', '.join(miss)} (steps: {', '.join(names)})")
        steps = [s for s in steps if s["name"] in only]
    return [(s["name"], [s["command"]] + expand(s.get("args", []), var)) for s in steps if not s.get("skip")]


def run(path, only=None, start=None, overrides=None, echo=True, keep_going=False):
    """Run the steps; returns the list of step results (name, argv, returncode, seconds, log file)."""
    path = Path(path).resolve()
    data = load(path)
    root = path.parent
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    status_file = root / (path.name.split(".")[0] + ".status.json")
    try:
        status = json.loads(status_file.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        status = {}
    results = []
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for name, argv in plan(data, only, start, overrides):
        log = logs / f"{name}.log"
        if echo:
            print(f"==> [{name}] mobility_tool {' '.join(argv)}", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(f"# mobility_tool {' '.join(argv)}\n")
            p = subprocess.Popen([sys.executable, "-m", "mobility_tool"] + argv, cwd=root, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace")
            for line in p.stdout:
                fh.write(line)
                if echo:
                    print("    " + line, end="", flush=True)
            rc = p.wait()
        dt = time.time() - t0
        results.append({"name": name, "argv": argv, "returncode": rc, "seconds": round(dt, 1), "log": str(log)})
        status[name] = {"returncode": rc, "seconds": round(dt, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "argv": argv}
        status_file.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        if echo:
            print(f"<== [{name}] {'ok' if rc == 0 else 'failed (exit code %d)' % rc}, {dt:.1f} s, log {log.name}",
                  flush=True)
        if rc != 0 and not keep_going:
            break
    return results


def check(path, overrides=None):
    """Parse every step with the real argument parser (nothing is run). Returns [(step, error message)]."""
    path = Path(path).resolve()
    data = load(path)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MOBILIS_PARSE_ONLY="1")
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    bad = []
    try:
        steps = plan(data, overrides=overrides)
    except KeyError as exc:
        return [("variables", str(exc))]
    for name, argv in steps:
        r = subprocess.run([sys.executable, "-m", "mobility_tool"] + argv, cwd=path.parent, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or "MOBILIS_PARSE_OK" not in r.stdout:
            bad.append((name, (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else "?"))
    return bad


# ----------------------------------------------------------------------------------------------
# GUI settings (every tk variable and file field of every page)
# ----------------------------------------------------------------------------------------------
def _tabs(app, create=()):
    """(attribute name, page) of the pages that exist (pages are created on first use); `create`: attribute names
    of pages to create now."""
    out = [("struct_tab", app.struct_tab)] if hasattr(app, "struct_tab") else []
    made = getattr(app, "_tabs", {})
    for name, (attr, _cls) in getattr(type(app), "TAB_CLASSES", {}).items():
        if name in made:
            out.append((attr, made[name]))
        elif attr in create:
            out.append((attr, getattr(app, attr)))
    return out


def gui_fields(app, create=()):
    from .gui import FileEntry, tk
    out = {}
    for tname, tab in _tabs(app, create):
        for aname, obj in sorted(vars(tab).items()):
            if aname.startswith("_"):
                continue
            if isinstance(obj, (getattr(tk, "Variable", FileEntry), FileEntry)) or \
                    type(obj).__name__ in ("StringVar", "BooleanVar", "MappedVar", "IntVar", "DoubleVar"):
                out[f"{tname}.{aname}"] = obj
    return out


def gui_collect(app):
    vals = {}
    for k, obj in gui_fields(app).items():
        try:
            v = obj.get()
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, (str, int, float, bool)):
            vals[k] = v if not isinstance(v, str) else str(v)
    return vals


def gui_apply(app, values):
    """Set the saved values; returns the keys that no longer exist in this version."""
    fields = gui_fields(app, create={k.split(".", 1)[0] for k in values})
    unknown = []
    for k, v in values.items():
        obj = fields.get(k)
        if obj is None:
            unknown.append(k)
            continue
        try:
            obj.set(v)
        except Exception:  # noqa: BLE001
            unknown.append(k)
    return unknown
