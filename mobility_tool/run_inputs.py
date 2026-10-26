"""“运行 Gaussian / ADF”的输入生成 (不依赖界面：对话框、流程图、命令行与测试共用)。
v = 对话框的取值 (dict)；ctx = 结构快照：molecule (原子, 名称)、dimer (A, B)、dimer_res、crystal、ams_home。"""
from pathlib import Path

import numpy as np

from . import gaussian_io as _gio
from .workflow import prepare_dimer_jobs as _prepare_dimer_jobs


class _G:
    read_geometry = staticmethod(_gio.read_geometry)
    write_xyz = staticmethod(_gio.write_xyz)
    prepare_dimer_jobs = staticmethod(_prepare_dimer_jobs)


G = _G


def add_input(files, txt):
    """附加输入行：加在每个 gjf 计算步骤的末尾。"""
    txt = (txt or "").strip()
    if not txt:
        return
    for f in files:
        if Path(f).suffix.lower() not in (".gjf", ".com"):
            continue
        parts = Path(f).read_text().split("--Link1--")
        Path(f).write_text("--Link1--\n".join(p.rstrip("\n") + "\n\n" + txt + "\n\n" for p in parts))


def _molecule(v, ctx):
    if v.get("source") == "结构文件":
        if not v.get("file"):
            raise ValueError("请选择结构文件")
        return G.read_geometry(v["file"]), Path(v["file"]).stem
    if not ctx.get("molecule"):
        raise ValueError("主窗口中还没有结构：请先 文件 → 打开结构")
    return ctx["molecule"]


def _dimer(v, ctx):
    if v.get("source") == "二聚体文件":
        from .crystal import split_dimer
        if not v.get("file"):
            raise ValueError("请选择二聚体文件")
        n_a, _r, atoms = split_dimer(G.read_geometry(v["file"]))
        return atoms[:n_a], atoms[n_a:]
    if not ctx.get("dimer"):
        raise ValueError("请在主窗口中恰好选中两个分子 (左键点第一个，Ctrl+左键点第二个)，"
                         "或把“来源”改为二聚体文件 / 晶体全部近邻二聚体")
    return ctx["dimer"]


def _need_dimer_res(ctx):
    if not ctx.get("dimer_res"):
        raise ValueError("请先在主窗口搜索近邻二聚体 (构建 → 搜索近邻二聚体)")
    return ctx["dimer_res"]


def gaussian_generate(outdir, v, ctx):
    """按“运行 Gaussian”的设置写出输入文件；返回说明文字。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name, calc = v["name"], v["calc"]
    nproc, mem, prog = int(v.get("nproc") or 8), v.get("mem") or "8GB", v.get("program") or "g16"
    car = {"空穴": ("hole",), "电子": ("electron",)}.get(v.get("carriers"), ("hole", "electron"))
    charge, mult = int(v.get("charge") or 0), int(v.get("mult") or 1)

    def geom_file():
        if v.get("source") == "结构文件":
            return _molecule(v, ctx) and v["file"]
        atoms, _ = _molecule(v, ctx)
        p = outdir / f"{name}_input.xyz"
        G.write_xyz(p, atoms, f"{name} (Mobilis)")
        return str(p)
    if calc in ("opt", "reorg"):
        from .reorganization import generate_reorg_inputs
        generate_reorg_inputs(geom_file(), outdir, name=name, method=v["method"], basis=v["basis"],
                              carriers=car if calc == "reorg" else (), charge=charge, mult=mult,
                              ion_mult=int(v.get("ion_mult") or 2) if calc == "reorg" else None, nproc=nproc, mem=mem,
                              freq=bool(v.get("freq")) or calc == "opt", extra=v.get("extra", ""), program=prog,
                              sp_method=v.get("sp_method") or None, sp_basis=v.get("sp_basis") or None,
                              solvent=v.get("solvent", ""), ecp_basis=v.get("ecp", "SDD"))
        return ""
    if calc == "excited":
        from .photophys import generate_excited_inputs
        generate_excited_inputs(geom_file(), outdir, name=name, method=v["method"], basis=v["ex_basis"],
                                sp_basis=v["ex_sp_basis"], solvent=v.get("solvent", ""), nstates=int(v.get("nstates") or 10),
                                charge=charge, mult=mult, nproc=nproc, mem=mem, ecp_basis=v.get("ecp", "SDD"),
                                freq_excited=bool(v.get("freq_excited")))
        return ""
    if calc == "esp":
        from .polar import write_charge_inputs
        atoms, _ = _molecule(v, ctx)
        write_charge_inputs([a[0] for a in atoms], np.array([a[1:] for a in atoms], float),
                            str(outdir / name), level=f"{v['method']}/{v['basis']}", nproc=nproc, mem=mem)
        from .jobs import write_windows_bat
        write_windows_bat(outdir, [f"{name}_{t}" for t in ("neutral", "cation", "anion")], name=f"run_{name}_esp.bat")
        return "MD 静电用中性的 ESP 电荷 (--md-charges)"
    if calc == "fcht":
        from .gaussian_fcht import generate_fcht_inputs
        d = v.get("reorg_dir") or ctx.get("reorg_dir")
        if not d:
            raise ValueError("请选择四点法重组能目录")
        generate_fcht_inputs(d, name, carriers=car, nproc=nproc, mem=mem)
        return "FCHT 输入写在重组能目录中"
    if calc == "ti":
        if v.get("source") == "晶体全部近邻二聚体":
            files = G.prepare_dimer_jobs(_need_dimer_res(ctx), outdir, program="gaussian", method=v["method"],
                                         basis=v["basis"], nproc=nproc, mem=mem, extra=v.get("extra", ""))
            return f"{len(files)} 个等价二聚体组；近邻信息见 dimers.json"
        a, b = _dimer(v, ctx)
        p = outdir / f"{name}_dimer.xyz"
        G.write_xyz(p, list(a) + list(b), f"dimer nA={len(a)}")
        from .transfer_integral import generate_ti_inputs
        generate_ti_inputs(outdir, dimer=p, n_a=len(a), name=name, method=v["method"], basis=v["basis"],
                           charge=charge, mult=mult, nproc=nproc, mem=mem, extra=v.get("extra", ""), program=prog,
                           ecp_basis=v.get("ecp", "SDD"))
        return ""
    if calc == "cp":
        from .interaction import prepare_counterpoise_jobs, write_counterpoise_gjf
        kw = dict(functional=v["method"], basis=v["basis"], dispersion=v.get("disp", "GD3BJ"), nproc=nproc, mem=mem)
        if v.get("source") == "晶体全部近邻二聚体":
            dr = _need_dimer_res(ctx)
            names = prepare_counterpoise_jobs(dr, outdir, **kw)
            import json as _json
            (outdir / "dimers.json").write_text(_json.dumps({"neighbors": [
                {"group": x["group"], "vector": list(map(float, x["vector"]))} for x in dr["dimers"]]},
                ensure_ascii=False))
            return f"{len(names)} 个等价二聚体组"
        a, b = _dimer(v, ctx)
        write_counterpoise_gjf(outdir / f"{name}_cp.gjf", a, b, extra=v.get("extra", ""), **kw)
        from .jobs import write_windows_bat
        write_windows_bat(outdir, [f"{name}_cp"], name=f"run_{name}_cp.bat")
        return ""
    raise ValueError(f"未知计算类型 {calc}")


def adf_style(v, ctx):
    s = v.get("style", "")
    if s.startswith("AMS"):
        return "ams"
    if s.startswith("ADF2019"):
        return "classic"
    from .programs import find_ams
    info = find_ams(ctx.get("ams_home") or None)
    return info["kind"] if info else "ams"


def adf_generate(outdir, v, ctx):
    """按“运行 ADF”的设置写出输入文件；返回说明文字。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = v["name"]
    kw = dict(functional=v["functional"], basis=v["basis"], style=adf_style(v, ctx), numerical_quality=v["quality"])
    if v["calc"] == "crystal":
        files = G.prepare_dimer_jobs(_need_dimer_res(ctx), outdir, program="adf", chunk=int(v.get("chunk") or 0), **kw)
        return f"{len(files)} 个等价二聚体组；在 ADFjobs 中打开 batch_adf*.run 或在任务管理器中运行"
    a, b = _dimer(v, ctx)
    p = outdir / f"{name}_dimer.xyz"
    G.write_xyz(p, list(a) + list(b), f"dimer nA={len(a)}")
    from .adf import generate_adf_inputs
    generate_adf_inputs(outdir, dimer=p, n_a=len(a), name=name, extra=v.get("extra", ""), **kw)
    return ""
