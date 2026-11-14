"""命令行入口:  python -m mobility_tool <子命令> ..."""
import argparse
import os
import sys

import numpy as np
import json
from pathlib import Path

from .adf import generate_adf_inputs, read_adf_transfer_integrals
from .crystal import find_dimers, find_molecules, read_cif
from .decomposition import decompose_reorg, export_decomposition_csv
from .gaussian_fcht import export_fcht, generate_fcht_inputs, read_fcht_log
from .ipea import ENERGY_KEYS, format_ip_ea, frontier_orbitals, ip_ea_from_energies
from .plots import figure_hr_duschinsky, figure_reorg_decomposition
from .mobility import compute_mobility, mobility_from_config
from .workflow import collect_group_couplings, prepare_dimer_jobs
from .reorganization import analyze_reorg, generate_reorg_inputs
from .transfer_integral import calc_transfer_integrals, generate_ti_inputs


def _common_gauss(p):
    p.add_argument("--method", default="B3LYP", help="泛函/方法 (默认 B3LYP)")
    p.add_argument("--basis", default="6-31G(d)", help="基组 (默认 6-31G(d))")
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--mem", default="8GB")
    p.add_argument("--extra", default="", help='额外关键词, 如 "em=gd3bj"')
    p.add_argument("--program", default="g16", help="运行脚本中的高斯命令 (g16/g09)")


def _print_warnings(ws):
    for w in ws:
        print(f"  [警告] {w}")


def cmd_reorg_gen(a):
    carriers = {"hole": ["hole"], "electron": ["electron"], "both": ["hole", "electron"]}[a.carrier]
    m = generate_reorg_inputs(a.geom, a.outdir, a.name, a.method, a.basis, carriers, a.charge, a.mult,
                              a.ion_mult, a.nproc, a.mem, not a.no_freq, a.extra, a.program,
                              a.sp_method, a.sp_basis, a.solvent, a.ecp_basis)
    print(f"已在 {a.outdir} 生成: " + ", ".join(f"{s}.gjf" for s in m["files"]) + " 以及 run_reorg.sh")


def cmd_reorg_analyze(a):
    res = analyze_reorg(a.dir)
    print(f"\n重组能结果  {res['name']}  ({res['level']})")
    _print_warnings(res["warnings"])
    for carrier, r in res["results"].items():
        cn = "空穴" if carrier == "hole" else "电子"
        print(f"\n  {cn}:  λ1 = {r['lambda1_eV']:.4f} eV   λ2 = {r['lambda2_eV']:.4f} eV   "
              f"λ = {r['lambda_eV']:.4f} eV ({r['lambda_eV'] * 1000:.1f} meV)")
        for k, v in r.items():
            if k.startswith(("IP", "EA")):
                print(f"        {k} = {v:.4f}")
    if not res["results"]:
        print("  没有得到完整的四点能量，请检查计算是否完成。")
    print("\n" + format_ip_ea(res["ip_ea"], res["orbitals"]))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n结果已写入 {a.json}")


def cmd_ti_gen(a):
    stems = generate_ti_inputs(a.outdir, a.dimer, a.na, a.mono_a, a.mono_b, a.name, a.method, a.basis,
                               a.charge, a.mult, a.nproc, a.mem, a.extra, a.program)
    print(f"已在 {a.outdir} 生成: " + ", ".join(f"{s}.gjf" for s in stems) + " 以及 run_ti.sh")


def cmd_ti_calc(a):
    if a.prefix:
        fa, fb, fab = (f"{a.prefix}_{s}.fchk" for s in ("A", "B", "AB"))
    else:
        fa, fb, fab = a.a, a.b, a.ab
    res = calc_transfer_integrals(fa, fb, fab, [o.strip() for o in a.orbitals.split(",")])
    print(f"\n转移积分 (二聚体投影法)  A={fa}  B={fb}  AB={fab}")
    _print_warnings(res["warnings"])
    print(f"  {'轨道':<8}{'J_eff/meV':>12}{'J_raw/meV':>12}{'S_AB':>12}{'e_A/eV':>11}{'e_B/eV':>11}")
    for lab, r in res["results"].items():
        print(f"  {lab:<8}{r['J_eff_meV']:12.3f}{r['J_raw_meV']:12.3f}{r['S_AB']:12.5f}"
              f"{r['e_A_eV']:11.4f}{r['e_B_eV']:11.4f}")
    print("  (空穴传输用 HOMO, 电子传输用 LUMO)")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"结果已写入 {a.json}")


def _print_mobility(res):
    print(f"\nT = {res['temperature_K']:.2f} K   λ = {res['lambda_eV']:.4f} eV   n = {res['dimension']}")
    print(f"  {'路径':<10}{'r/Å':>9}{'J/meV':>10}{'数目':>6}{'k/s^-1':>13}{'P':>9}")
    for p in res["pathways"]:
        print(f"  {p['name']:<10}{p['r_A']:9.3f}{p['J_meV']:10.3f}{p['count']:6d}"
              f"{p['k_s-1']:13.4e}{p['P']:9.4f}")
    print(f"  扩散系数 D = {res['D_cm2_s']:.4e} cm^2/s   [无规行走 D = Σ k r²/2n]")
    print(f"  迁移率   μ = {res['mobility_cm2_Vs']:.4f} cm^2/(V·s)")
    if "mobility_DG_cm2_Vs" in res:
        print(f"  (对比) Deng–Goddard 式 D = Σ r²kP/2n：μ = {res['mobility_DG_cm2_Vs']:.4f} cm^2/(V·s)，"
              f"偏小 {res['mobility_cm2_Vs'] / max(res['mobility_DG_cm2_Vs'], 1e-300):.1f} 倍")
    if "mobility_principal_cm2_Vs" in res:
        w = res["mobility_principal_cm2_Vs"]
        print(f"  迁移率张量主值 = {w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f} cm^2/(V·s)  "
              f"平均 {res['mobility_tensor_average_cm2_Vs']:.4f}")


def cmd_mobility(a):
    if a.dimers:
        if a.lam is None:
            raise SystemExit("使用 --dimers 时需要 --lam")
        from .crystal import mobility_pathways_from_dimers
        coup, errors = collect_group_couplings(a.dimers, a.program)
        for g, e in errors.items():
            print(f"  [警告] G{g}: {e}")
        key = "hole" if a.carrier == "hole" else "electron"
        paths = mobility_pathways_from_dimers(Path(a.dimers) / "dimers.json",
                                              {g: abs(v[key]) for g, v in coup.items() if v.get(key) is not None})
        info = json.loads((Path(a.dimers) / "dimers.json").read_text(encoding="utf-8"))
        ops = info.get("point_ops")
        if a.cif:
            from .crystal import point_ops_cart, read_cif
            ops = point_ops_cart(read_cif(a.cif))
        if not ops:
            print("  [提示] dimers.json 中没有晶体点群 (v1.21.2 以前导出)；每个晶胞多于两个分子时请加 --cif 以对称化张量")
        results = [compute_mobility(a.lam, paths, a.temp, a.dim, point_ops=ops)]
    elif a.config:
        results = mobility_from_config(a.config)
    else:
        if a.lam is None or not a.path:
            raise SystemExit("请提供 --config，或同时提供 --lam 和至少一个 --path r,J_meV[,count]")
        paths = []
        for i, s in enumerate(a.path):
            v = [float(x) for x in s.split(",")]
            paths.append({"name": f"P{i + 1}", "r": v[0], "J_meV": v[1], "count": int(v[2]) if len(v) > 2 else 1})
        results = [compute_mobility(a.lam, paths, a.temp, a.dim)]
    for r in results:
        _print_mobility(r)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n结果已写入 {a.json}")


def cmd_decomp(a):
    res = decompose_reorg(a.neutral, a.ion, a.bond_scale)
    print("\n重组能分解 (谐振近似, meV)        简正模式     内坐标    RMSD/Å")
    for key, lab in (("neutral", "λ1 (中性态势能面)"), ("ion", "λ2 (离子态势能面)")):
        r = res[key]
        print(f"  {lab:<22}{r['lambda_normal_modes_meV']:12.2f}{r['lambda_internal_meV']:11.2f}{r['rmsd_A']:10.4f}")
        if r["n_imaginary"]:
            print(f"  [警告] 存在 {r['n_imaginary']} 个虚频")
    print(f"  {'合计':<22}{res['lambda_total_normal_modes_meV']:12.2f}{res['lambda_total_internal_meV']:11.2f}")
    for key, lab in (("neutral", "λ1"), ("ion", "λ2")):
        r = res[key]
        print(f"\n{lab} 按内坐标类型: " + ", ".join(f"{k} {v:.2f}" for k, v in r["internal_by_type_meV"].items()))
        print(f"{lab} 贡献最大的 {a.top} 个振动模式:\n   频率/cm-1    黄昆因子S    λ_i/meV")
        for m in sorted(r["modes"], key=lambda m: -m["lambda_meV"])[:a.top]:
            print(f"   {m['freq_cm-1']:9.1f}{m['S']:13.4f}{m['lambda_meV']:11.3f}")
        print(f"{lab} 贡献最大的 {a.top} 个内坐标:")
        for x in sorted(r["internal"], key=lambda x: -abs(x["lambda_meV"]))[:a.top]:
            print(f"   {x['type']:<9}{x['atoms']:<14}{x['elements']:<12}Δ={x['delta']:+9.4f}  {x['lambda_meV']:9.3f}")
    if a.csv:
        files = export_decomposition_csv(res, a.csv)
        print(f"\n已导出 {len(files)} 个 CSV 到 {a.csv}")


def cmd_adf_gen(a):
    r = generate_adf_inputs(a.outdir, a.dimer, a.na, a.mono_a, a.mono_b, a.name, a.functional, a.basis,
                            a.style, a.quality)
    print(f"已生成 {r['script']}\n计算完成后读取: {r['result_file']}")


def cmd_adf_read(a):
    res = read_adf_transfer_integrals(a.file)
    print(f"\n{'通道':<28}{'V/meV':>11}{'J/meV':>11}{'S':>10}{'e1/eV':>10}{'e2/eV':>10}")
    for ch, r in res.items():
        print(f"{ch:<28}{r.get('V_meV', float('nan')):11.3f}{r.get('J_meV', float('nan')):11.3f}"
              f"{r.get('S', float('nan')):10.5f}{r.get('e1_eV', float('nan')):10.4f}{r.get('e2_eV', float('nan')):10.4f}")


def cmd_crystal(a):
    crys = read_cif(a.cif)
    for w in crys["warnings"]:
        print(f"  [警告] {w}")
    mols = find_molecules(crys)
    print(f"晶胞内 {len(mols)} 个分子，原子数 {[len(m['symbols']) for m in mols]}")
    res = find_dimers(crys, mols, ref=a.ref - 1, cutoff=a.cutoff,
                      criterion="centroid" if a.centroid else "atom")
    print(f"近邻 {len(res['dimers'])} 个，等价组 {len(res['groups'])} 个")
    for g in res["groups"]:
        d = g["members"][0]
        print(f"  G{g['id']:<3} 数目 {len(g['members'])}  质心距 {d['r_centroid']:7.3f} Å  最短原子距 {d['d_min']:6.3f} Å")
    if a.outdir:
        if a.program == "gaussian":
            st = dict(method=a.method, basis=a.basis, nproc=a.nproc, mem=a.mem)
        else:
            st = dict(functional=a.method, basis=a.basis)
        prepare_dimer_jobs(res, a.outdir, program=a.program, **st)
        print(f"已在 {a.outdir} 导出二聚体与 {a.program} 输入；计算后用 mobility --dimers {a.outdir} 计算迁移率")


def cmd_ipea(a):
    if a.dir:
        from .reorganization import analyze_reorg
        r = analyze_reorg(a.dir)
        energies, orb = r["energies_Ha"], r["orbitals"]
    else:
        energies = {k: v for (k, _), v in zip(ENERGY_KEYS, a.energies or []) if v is not None}
        orb = None
    if a.orbitals:
        orb = frontier_orbitals(a.orbitals)
    print(format_ip_ea(ip_ea_from_energies(energies), orb))


def cmd_orbitals(a):
    for k, v in frontier_orbitals(a.file).items():
        print(f"{k:<16}{v:10.4f} eV")


def cmd_fcht_gen(a):
    import json as _json
    man = _json.loads((Path(a.dir) / "reorg_manifest.json").read_text())
    carriers = [c for c, ion in (("hole", "cation"), ("electron", "anion")) if f"{man['name']}_{ion}" in man["files"]]
    jobs = generate_fcht_inputs(a.dir, man["name"], carriers, a.nproc, a.mem, a.clear_low_freq)
    print("已生成: " + ", ".join(j["stem"] + ".gjf" for j in jobs) + "  以及 run_fcht.sh / run_fcht.bat")


def cmd_fcht_read(a):
    r = read_fcht_log(a.log, a.freq_block)
    for w in r["warnings"]:
        print(f"  [警告] {w}")
    print(f"{'No.':>4}{'频率/cm-1':>12}{'黄昆因子S':>13}{'λi/kJ·mol-1':>14}{'λi/meV':>10}")
    for m in r["modes"]:
        print(f"{m['mode']:4d}{m['freq_cm-1']:12.2f}{m['S']:13.5f}{m['lambda_kJ/mol']:14.4f}{m['lambda_meV']:10.3f}")
    print(f"内重组能 Σλi = {r['lambda_kJ/mol']:.4f} kJ/mol = {r['lambda_meV'] / 1000:.5f} eV")
    if a.out:
        stem = Path(a.log).stem
        files = export_fcht(r, a.out, stem)
        f = [m["freq_cm-1"] for m in r["modes"]]
        figure_hr_duschinsky(f, [m["S"] for m in r["modes"]], r["duschinsky"], Path(a.out) / f"{stem}_HR_Duschinsky.png")
        figure_reorg_decomposition(f, [m["lambda_meV"] for m in r["modes"]], Path(a.out) / f"{stem}_decomposition.png",
                                   fwhm=20)
        print(f"已导出 {len(files)} 个数据文件和 2 张 300 dpi 图到 {a.out}")


def cmd_mobility2d(a):
    import csv as _csv
    from .mobility import angular_mobility_2d
    rows = list(_csv.DictReader(open(a.table, encoding="utf-8-sig")))
    norm = lambda d: {k.strip().lower(): v for k, v in d.items() if k}  # noqa: E731
    paths = []
    for r in map(norm, rows):
        get = lambda *ks: next((r[k] for k in ks if k in r and r[k] not in ("", None)), None)  # noqa: E731
        paths.append({"name": get("name", "pathway") or f"P{len(paths) + 1}", "r": float(get("r", "d")),
                      "theta": float(get("theta", "θ", "Θ") or 0), "gamma": float(get("gamma", "γ") or 0),
                      "Ve": float(get("ve")) / a.scale if get("ve") else None,
                      "Vh": float(get("vh")) / a.scale if get("vh") else None, "count": int(float(get("count") or 1))})
    systems = []
    for key, lam, lab in (("Ve", a.lam_e, "电子"), ("Vh", a.lam_h, "空穴")):
        if lam is None or not any(p[key] is not None for p in paths):
            continue
        r = angular_mobility_2d(lam, [p for p in paths if p[key] is not None], a.temp, key, model=a.model)
        print(f"\n{lab} (λ={lam} eV, T={a.temp} K)")
        print(f"{'路径':<8}{'r/Å':>9}{'θ/°':>10}{'W/s-1':>13}{'P':>9}")
        for q in r["pathways"]:
            print(f"{q['name']:<8}{q['r_A']:9.3f}{q['theta_deg']:10.3f}{q['W_s-1']:13.4e}{q['P']:9.4f}")
        print(f"μmax = {r['mu_max']:.4g} cm²/Vs (φ={r['phi_max_deg']:.1f}°), μmin = {r['mu_min']:.4g}, "
              f"各向异性 {r['anisotropy']:.3g}")
        systems.append((key, r))
    if a.plot and systems:
        from .plots import figure_angular
        d = {"label": a.label, "phi": systems[0][1]["phi"]}
        for key, r in systems:
            d["mu_e" if key == "Ve" else "mu_h"] = r["mu"]
            d["lam_e" if key == "Ve" else "lam_h"] = r["lambda_eV"]
        figure_angular([d], a.plot)
        print(f"图已保存 {a.plot}")


def cmd_mobility3d(a):
    """三维各向异性迁移率：CSV (name, x, y, z, Ve, Vh, count)。"""
    import csv as _csv
    from .mobility import crystal_direction, mobility_3d
    from .plots import E_COLOR, H_COLOR, figure_mobility_3d, mobility_3d_html
    rows = list(_csv.DictReader(open(a.table, encoding="utf-8-sig")))
    norm = lambda d: {k.strip().lower(): v for k, v in d.items() if k}  # noqa: E731
    paths = []
    for r in map(norm, rows):
        g = lambda k: r.get(k) if r.get(k) not in ("", None) else None  # noqa: E731
        paths.append({"name": g("name") or f"P{len(paths) + 1}", "vector": [float(r["x"]), float(r["y"]), float(r["z"])],
                      "Ve": float(g("ve")) / a.scale if g("ve") else None, "Vh": float(g("vh")) / a.scale if g("vh") else None,
                      "count": int(float(g("count") or 1))})
    lat = None
    if a.cif:
        from .crystal import read_cif
        lat = read_cif(a.cif)["lattice"]
    systems = []
    for key, lam, lab, col in (("Ve", a.lam_e, "电子", E_COLOR), ("Vh", a.lam_h, "空穴", H_COLOR)):
        if lam is None or not any(p[key] is not None for p in paths):
            continue
        r = mobility_3d(lam, [p for p in paths if p[key] is not None], a.temp, key, a.model)
        print(f"\n{lab} (λ = {lam} eV, T = {a.temp} K, {a.model})：主值 " + " / ".join(f"{v:.4g}" for v in r["principal"])
              + f" cm²/Vs，各向异性 {r['anisotropy']:.3g}")
        for k, (v, d) in enumerate(zip(r["principal"], r["axes"])):
            line = f"  μ{k + 1} = {v:9.4g}  ({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f})"
            if lat is not None:
                f, ang = crystal_direction(d, lat)
                line += f"  ≈ [{f[0]:.2f} {f[1]:.2f} {f[2]:.2f}]"
            print(line)
        systems.append({"label": f"{lab} (λ = {lam * 1000:.0f} meV)", "res": r, "color": col})
    if a.plot and systems:
        figure_mobility_3d(systems, a.plot, lat, a.plane)
        h = str(Path(a.plot).with_suffix(".html"))
        mobility_3d_html(h, systems, lat, a.plane, title=a.label or "三维各向异性迁移率")
        print(f"图已保存 {a.plot} 与 {Path(h).name} (浏览器中可旋转)")


def cmd_excited_gen(a):
    from .photophys import generate_excited_inputs
    st = generate_excited_inputs(a.geom, a.outdir, a.name, a.method, a.basis, a.sp_basis, a.solvent, a.nstates,
                                 a.charge, a.mult, a.nproc, a.mem)
    print("已生成: " + ", ".join(f"{x}.gjf" for x in st) + "（按 S0→S1→T1 顺序运行）")


def cmd_excited_read(a):
    from .photophys import analyze_excited
    r = analyze_excited(a.dir, a.name)
    for w in r["warnings"]:
        print(f"  [警告] {w}")
    for t in r["table"]:
        cfg = "; ".join(f"{x}({p:.1f}%)" for x, p in t["configs"])
        print(f"{t['kind']:<16}{t['transition']:<8}{t['E_eV']:8.3f} eV{t['nm']:9.2f} nm  f={t['f']:.4f}  {cfg}")
    for k in ("stokes_shift_eV", "rmsd_S0_S1", "rmsd_S0_T1"):
        if k in r:
            print(f"{k} = {r[k]:.3f}")


def cmd_rmsd(a):
    from .photophys import rmsd
    from .gaussian_io import read_geometry
    r = rmsd(read_geometry(a.a), read_geometry(a.b), a.heavy)
    print(f"RMSD = {r['rmsd']:.4f} Å")


def cmd_orbcomp(a):
    from .orbcomp import format_composition, orbital_composition, parse_ranges
    from .gaussian_io import read_fchk
    frags = None
    if a.frag:
        n = read_fchk(a.fchk)["Number of atoms"]
        frags = {f.split(":")[0]: parse_ranges(f.split(":")[1], n) for f in a.frag}
    print(format_composition(orbital_composition(a.fchk, frags, a.orbitals.split(","), a.method)))


def cmd_run(a):
    import threading
    from .jobs import JobQueue, find_gaussian, path_problems
    info = find_gaussian(a.gaussian)
    if not info:
        raise SystemExit("没有找到高斯程序，请用 --gaussian 指定 g16.exe 路径")
    for pr in path_problems(a.dir):
        print(f"[警告] {pr}")
    gj = sorted(Path(a.dir).glob(a.pattern))
    done = threading.Event()
    q = JobQueue(info, gj, a.scratch or str(Path(a.dir) / "scratch"),
                 lambda g, st, m: print(f"{Path(g).name:<40}{st:<10}{(m.splitlines() or [''])[0][:80]}"),
                 lambda r: done.set())
    q.start()
    done.wait()


def _gui(a):
    from . import gui
    if a.lang:
        cfg = gui.load_config()
        cfg["lang"] = a.lang
        gui.save_config(cfg)
    gui.main()


def cmd_eht_ti(a):
    from .eht import eht_from_file
    r = eht_from_file(a.dimer, a.na, a.mono_a, a.mono_b,
                      orbitals=[o.strip() for o in a.orbitals.split(",")], pi=a.pi)
    for w in r["warnings"]:
        print(f"  [警告] {w}")
    print(f"软件内 EHT ({r['nbf']} 个基函数)")
    for lab, q in r["results"].items():
        if isinstance(q, dict):
            print(f"  {lab:<9} J_eff = {q['J_eff_meV']:10.3f} meV   S = {q['S_AB']:.5f}")
    print(f"  ESD: 空穴 {r['results']['ESD_hole_meV']:.3f} meV, 电子 {r['results']['ESD_electron_meV']:.3f} meV")


def cmd_compare_ti(a):
    from .ticompare import compare_dir, format_rows, statistics
    rows = compare_dir(a.dir)
    print(format_rows(rows))
    for ref in ("Gaussian", "ADF"):
        for car, q in statistics(rows, ref=ref).items():
            print(f"EHT vs {ref} ({car}, n={q['n']}): r={q['pearson_r']:.3f}, k={q['scale_k']:.3f}, "
                  f"标定后相对偏差 {q['mean_rel_err_after_scaling'] * 100:.1f}%")


def cmd_kmc(a):
    from .crystal import find_molecules, read_cif
    from .kmc import graph_from_crystal, run_kmc
    from .workflow import eht_couplings_for_crystal
    crys = read_cif(a.cif)
    mols = find_molecules(crys)
    res0, coup = eht_couplings_for_crystal(crys, mols, a.cutoff, pi=a.pi)
    for w in res0.get("eht_warnings", []):
        print(f"  [警告] {w}")
    lam = a.lam
    g = graph_from_crystal(crys, mols, coup, a.cutoff, carrier=a.carrier)
    fgr = None
    if a.rate == "fgr":
        from .tunneling import FGRRate, modes_from_files
        if not a.modes:
            raise SystemExit("--rate fgr 需要 --modes (reorg-decomp 写出的 neutral 与 ion 模式文件)")
        fgr = FGRRate(modes_from_files(a.modes), a.temp, lam_cl=a.lam_out, classical_below_cm=a.classical_below)
        sm = fgr.summary()
        print(f"量子核隧穿速率：λ = {1000 * sm['lambda_eV']:.0f} meV (量子 {1000 * sm['lambda_quantum_eV']:.0f}，"
              f"经典 {1000 * sm['lambda_classical_eV']:.0f})，{sm['n_quantum_modes']} 个量子模，"
              f"ΔG = 0 时 k/k_Marcus = {sm['ratio_to_marcus_dG0']:.3f}")
    r = run_kmc(g, lam, a.temp, method=a.rate, lam_out=a.lam_out, hw=a.hw, carrier=a.carrier, sigma=a.sigma,
                supercell=a.supercell, field=[float(x) for x in a.field.split()], walkers=a.walkers, steps=a.steps,
                fgr=fgr)
    print(f"KMC ({a.carrier}, {a.rate}, EHT 转移积分, 位点 {r['n_sites']}, 跳跃 {r['hops']:,})")
    print("迁移率主值 / cm²V⁻¹s⁻¹: " + ", ".join(f"{w:.4g}" for w in r["mu_principal"]) +
          f"   平均 {r['mu_avg']:.4g} ± {r['mu_avg_err']:.2g}")
    if "mu_field_cm2_Vs" in r:
        print(f"场迁移率 {r['mu_field_cm2_Vs']:.4g} ± {r['mu_field_err']:.2g}")


def cmd_tunneling(a):
    """Quantum-nuclear-tunneling hopping (tunneling.FGRRate) against Marcus and MLJ; optional crystal mobility."""
    import csv as _csv
    from .compare import mlj_from_modes
    from .kmc import rate_jortner
    from .tunneling import FGRRate, modes_from_files, tensor
    modes = modes_from_files(a.modes)
    temps = [float(x) for x in str(a.temps).replace(";", ",").split(",") if x.strip()]
    rows_m = [{"freq_cm-1": f, "lambda_meV": f * 1.239841984e-4 * s * 1000} for f, s in modes]
    hw, f_low = mlj_from_modes([rows_m])
    lam_tot = sum(f * 1.239841984e-4 * s for f, s in modes) + a.lam_out
    print(f"{len(modes)} 个模式，λ_内 = {1000 * (lam_tot - a.lam_out):.1f} meV，λ_外 = {1000 * a.lam_out:.1f} meV；"
          f"MLJ 有效模 ħω = {1000 * hw:.0f} meV (低于 1000 cm⁻¹ 的模式占 λ 的 {100 * f_low:.0f}%，按经典处理)")
    graph = None
    if a.cif:
        from .crystal import find_molecules, read_cif
        from .kmc import graph_from_crystal, run_kmc
        from .compare import marcus_tensor
        from .workflow import eht_couplings_for_crystal
        crys = read_cif(a.cif)
        mols = find_molecules(crys)
        _, coup = eht_couplings_for_crystal(crys, mols, a.cutoff, pi=a.pi)
        graph = graph_from_crystal(crys, mols, coup, a.cutoff, carrier=a.carrier)
    head = f"{'T/K':>6}{'k_FGR/k_Marcus':>16}{'k_MLJ/k_Marcus':>16}"
    if graph:
        head += f"{'μ_FGR':>10}{'μ_Marcus':>10}" + (f"{'μ_KMC':>10}" if a.kmc else "")
    print(head)
    out = []
    for T in temps:
        fgr = FGRRate(modes, T, lam_cl=a.lam_out, classical_below_cm=a.classical_below)
        km = float(fgr.marcus(1.0))
        r_mlj = rate_jortner(1.0, (1 - f_low) * (lam_tot - a.lam_out), f_low * (lam_tot - a.lam_out) + a.lam_out,
                             0.0, T, hw) / km
        row = {"T_K": T, "lambda_eV": fgr.lam, "k_fgr_over_marcus": fgr.summary()["ratio_to_marcus_dG0"],
               "k_mlj_over_marcus": float(r_mlj), "k_fgr_per_meV2_s": float(fgr.k_over_v2(0.0)) * 1e-6}
        line = f"{T:6.0f}{row['k_fgr_over_marcus']:16.3f}{row['k_mlj_over_marcus']:16.3f}"
        if graph:
            mf = np.linalg.eigvalsh(tensor(graph, fgr))[::-1]
            mm = np.linalg.eigvalsh(marcus_tensor(graph, fgr.lam, T))[::-1]
            row.update(mu_fgr=float(mf[0]), mu_fgr_2=float(mf[1]), mu_marcus=float(mm[0]), mu_marcus_2=float(mm[1]))
            line += f"{mf[0]:10.3g}{mm[0]:10.3g}"
            if a.kmc:
                r = run_kmc(graph, fgr.lam, T, method="fgr", fgr=fgr, carrier=a.carrier, sigma=a.sigma)
                row.update(mu_kmc=float(r["mu_principal"][0]), mu_kmc_err=float(r["mu_avg_err"]))
                line += f"{r['mu_principal'][0]:10.3g}"
        print(line)
        out.append(row)
    if graph:
        print("μ：最大主值 / cm² V⁻¹ s⁻¹ (EHT 转移积分，零场无规行走张量；Marcus 与 FGR 用相同的 λ)")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8-sig") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"已写出 {a.out}")


def _tvcf_pair(initial, final, modes, floor_cm):
    from .tvcf import Pair
    if modes:
        from .gaussian_fcht import read_hr_table
        m = read_hr_table(modes)["modes"]
        ok = [r for r in m if r["freq_cm-1"] > 0 and r["S"] >= 0]
        return Pair.displaced([max(r["freq_cm-1"], floor_cm) for r in ok], [r["S"] for r in ok])
    if not (initial and final):
        raise SystemExit("需要 --initial 与 --final 两个 fchk 文件，或 --modes 黄昆因子表")
    return Pair.from_fchk(initial, final, floor_cm=floor_cm)


def _tvcf_couplings(a, pair):
    """Couplings of the tvcf command from its options (None where not given)."""
    from .tvcf import ht_from_au, mu_from_fosc, read_mode_table, read_vector
    mu = None
    if a.mu:
        mu = np.array([float(x) for x in a.mu.replace(";", ",").split(",") if x.strip()])
        if a.mu_debye:
            from .tvcf import DEBYE_AU
            mu = mu * DEBYE_AU
    elif a.fosc is not None:
        mu = mu_from_fosc(a.fosc, a.e_vert or a.de_used)
    ht_mu = ht_from_au(pair, read_mode_table(a.ht_dipole, 3)) if a.ht_dipole else None
    nac = None
    if a.nac:
        nac = pair.project_cartesian(read_vector(a.nac, len(getattr(pair, "masses", [])) or None))
    elif a.nac_modes:
        nac = ht_from_au(pair, read_mode_table(a.nac_modes, 1)[:, 0])
    ht_soc = ht_from_au(pair, read_mode_table(a.ht_soc, 1)[:, 0]) if a.ht_soc else None
    for name, v in (("--ht-dipole", ht_mu), ("--nac-modes", nac), ("--ht-soc", ht_soc)):
        if v is not None and len(v) != pair.n:
            raise SystemExit(f"{name}: {len(v)} 个模式，体系有 {pair.n} 个")
    return mu, ht_mu, nac, a.soc, ht_soc


def _tvcf_header(pair, dE):
    print(f"{pair.n} 个振动模式；ΔE (绝热) = {dE:.4f} eV；λ (末态面) = {1000 * pair.lambda_f():.1f} meV，"
          f"λ (初态面) = {1000 * pair.lambda_i():.1f} meV；Duschinsky 矩阵非对角元最大 "
          f"{np.abs(pair.J - np.diag(np.diag(pair.J))).max():.3f}")
    for w in pair.warnings:
        print("注意：" + w)


def cmd_tvcf(a):
    """Excited-state decay rates from thermal vibration correlation functions (tvcf.py)."""
    import csv as _csv
    from .tvcf import decay_rates
    pair = _tvcf_pair(a.initial, a.final, a.modes, a.floor_cm)
    dE = a.de if a.de is not None else getattr(pair, "dE_fchk", None)
    if dE is None:
        raise SystemExit("请用 --de 给出绝热能隙 (eV，初态减末态)")
    a.de_used = dE
    mu, ht_mu, nac, soc, ht_soc = _tvcf_couplings(a, pair)
    if mu is None and nac is None and soc is None:
        raise SystemExit("至少给出一种耦合：--mu / --fosc (辐射)、--nac / --nac-modes (内转换) 或 --soc (系间窜越)")
    _tvcf_header(pair, dE)
    temps = [float(x) for x in str(a.temps).replace(";", ",").split(",") if x.strip()]
    rows = []
    for T in temps:
        r = decay_rates(pair, T, dE, mu, ht_mu, nac, soc, ht_soc, 1 / 3 if a.triplet_initial else 1.0, a.n_refr,
                        a.damping_fs)
        parts = [f"T = {T:.0f} K"]
        if mu is not None:
            parts.append(f"k_r = {r['k_r']:.3e} s⁻¹" + (f" (FC {r['k_r_FC']:.3e})" if ht_mu is not None else "") +
                         f"，发射峰 {r['peak_nm']:.0f} nm")
        if nac is not None:
            parts.append(f"k_IC = {r['k_IC']:.3e} s⁻¹")
        if soc is not None:
            parts.append(f"k_ISC = {r['k_ISC']:.3e} s⁻¹" + (f" (FC {r['k_ISC_FC']:.3e})" if ht_soc is not None else ""))
        print("；".join(parts))
        rows.append(r)
    if a.out:
        keys = [k for k in ("T", "dE_eV", "k_r", "k_r_FC", "peak_nm", "k_IC", "k_ISC", "k_ISC_FC", "k_FC_per_eV2")
                if k in rows[0]]
        with open(a.out + "_rates.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = _csv.writer(fh)
            w.writerow(keys)
            for r in rows:
                w.writerow([r[k] for k in keys])
        files = [a.out + "_rates.csv"]
        if mu is not None:
            with open(a.out + "_spectrum.csv", "w", newline="", encoding="utf-8-sig") as fh:
                w = _csv.writer(fh)
                w.writerow(["E_eV", "wavelength_nm"] + [f"I_{r['T']:.0f}K" for r in rows])
                for k, e in enumerate(rows[0]["E_eV"]):
                    w.writerow([f"{e:.5f}", f"{1239.841984 / e:.2f}"] + [f"{r['spectrum'][k]:.6e}" for r in rows])
            files.append(a.out + "_spectrum.csv")
        print("已写出 " + "、".join(files))


def cmd_tadf(a):
    """S1/T1 kinetics of a (TADF) emitter: k_r, k_IC, k_ISC, k_RISC, k_nr(T1), k_p by TVCF, then lifetimes and PLQY."""
    import json as _json
    from .tvcf import DEBYE_AU, Emitter
    f = DEBYE_AU if a.mu_debye else 1.0
    em = Emitter(a.s0, a.s1, a.t1, a.soc, a.fosc, a.e_vert, (a.mu_s1 or 0.0) * f, (a.mu_t1 or 0.0) * f, a.soc_t1s0,
                 a.nac or None, a.e_s1, a.e_t1, a.n_refr, a.floor_cm, a.damping_fs)
    sm = em.summary()
    print(f"S1 = {sm['E_S1_eV']:.3f} eV, T1 = {sm['E_T1_eV']:.3f} eV, ΔE_ST = {1000 * sm['dE_ST_eV']:.0f} meV; "
          f"λ(S1→S0) = {1000 * sm['lambda_S1S0_eV']:.0f} meV, λ(S1→T1) = {1000 * sm['lambda_S1T1_eV']:.0f} meV, "
          f"λ(T1→S0) = {1000 * sm['lambda_T1S0_eV']:.0f} meV")
    for w in em.warnings:
        print("注意：" + w)
    res = []
    for T in [float(x) for x in str(a.temps).replace(";", ",").split(",") if x.strip()]:
        r = em.rates(T)
        print(f"T = {T:.0f} K: k_r = {r['k_r_S1']:.2e}, k_IC = {r['k_IC_S1']:.2e}, k_ISC = {r['k_ISC']:.2e}, "
              f"k_RISC = {r['k_RISC']:.2e}, k_p = {r['k_p_T1']:.2e}, k_nr(T1) = {r['k_nr_T1']:.2e} s⁻¹")
        print(f"        τ_瞬时 = {r['tau_prompt_s'] * 1e9:.3g} ns，τ_延迟 = {r['tau_delayed_s'] * 1e6:.3g} μs，"
              f"PLQY = {r['PLQY_total']:.3f} (瞬时 {r['PLQY_prompt']:.3f}，延迟 {r['PLQY_delayed']:.3f})")
        if r["peak_nm"]:
            print(f"        S1 发射峰 {r['peak_nm']:.0f} nm")
        res.append({k: v for k, v in r.items() if k not in ("E_eV", "spectrum")})
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            _json.dump({**sm, "SOC_cm-1": a.soc, "results": res}, fh, indent=1)
        print(f"已写出 {a.out}")

def cmd_dp_gen(a):
    """Strained cells for the periodic DFT calculations of deformation-potential theory."""
    from .crystal import read_cif
    from .dp import write_strained
    crys = read_cif(a.cif)
    strains = [float(x) for x in a.strains.replace(";", ",").split(",") if x.strip()]
    files = write_strained(crys, a.out or Path(a.cif).with_suffix("").as_posix() + "_dp", strains,
                           [x.strip() for x in a.axes.split(",") if x.strip()], a.format)
    print(f"已写出 {len(files) - 1} 个应变晶胞 ({a.format}) 和待填写的表格 {files[-1]}")
    print("对每个晶胞做周期性 DFT (固定晶胞、优化原子或固定分数坐标)，在表中填入总能量与价带顶 / 导带底 "
          "(以真空能级或芯能级为参考)，然后运行 dp <表格>")


def cmd_dp(a):
    """Deformation-potential mobilities: C and E1 from the strain table, Bardeen–Shockley, Boltzmann on the TB band."""
    import csv as _csv
    from . import dp
    fits = dp.fit_table(a.table, a.dim, a.size)
    masses = {}
    if a.mass:
        vals = [float(x) for x in a.mass.replace(";", ",").split(",") if x.strip()]
        for k, ax in enumerate(fits):
            masses[ax] = vals[min(k, len(vals) - 1)]
    tb = {}
    if a.cif:
        from .crystal import find_molecules, read_cif
        from .kmc import graph_from_crystal
        from .transport import HB2_ME, effective_mass
        from .workflow import eht_couplings_for_crystal
        crys = read_cif(a.cif)
        mols = find_molecules(crys)
        _, coup = eht_couplings_for_crystal(crys, mols, a.cutoff, pi=a.pi)
        lat = np.asarray(crys["lattice"], float)
        for car in ("hole", "electron"):
            g = graph_from_crystal(crys, mols, coup, a.cutoff, carrier=car, signed=True)
            em = effective_mass(g, car, lat, 16)
            inv = em["axes"].T @ np.diag([1 / m if np.isfinite(m) and m > 0 else 0.0 for m in em["masses_me"]]) \
                @ em["axes"]
            n = tuple(a.kgrid if (np.linalg.norm(lat[i]) < 30) else 1 for i in range(3))
            E, V, vol = dp.tb_bands(g, lat, car, n)
            tb[car] = {"inv_mass": inv, "E": E, "V": V, "vol": vol, "bandwidth": em["bandwidth_eV"]}
    rows = []
    print(f"{'轴':>4}{'载流子':>10}{'C':>10}{'E1/eV':>9}{'m*/m_e':>9}{'μ_BS':>10}{'μ_BS(TB m*)':>13}{'μ_Boltzmann':>13}")
    for ax, res in fits.items():
        for car, f in res.items():
            if car == "none":
                print(f"{ax:>4}{'—':>8}{f['C']:10.3g}  (表中没有价带顶 / 导带底)")
                continue
            d = np.eye(3)["abc".index(ax)] if ax in "abc" else np.eye(3)[0]
            row = {"axis": ax, "carrier": car, "C": f["C"], "C_unit": f["C_unit"], "E1_eV": f["E1"],
                   "rms_energy_meV": f["rms_energy_meV"], "rms_edge_meV": f["rms_edge_meV"]}
            line = f"{ax:>4}{car:>10}{f["C"]:10.3g}{f["E1"]:9.3f}"
            m = masses.get(ax)
            if m:
                row["mass_given"] = m
                row["mu_BS_given_mass"] = dp.bardeen_shockley(f["C"], f["E1"], m, a.temp, a.dim)
                line += f"{m:9.3g}{row['mu_BS_given_mass']:10.4g}"
            else:
                line += f"{'':9}{'':10}"
            if car in tb and ax in "abc":
                lat = np.asarray(read_cif(a.cif)["lattice"], float)
                u = lat["abc".index(ax)] / np.linalg.norm(lat["abc".index(ax)])
                im = float(u @ tb[car]["inv_mass"] @ u)
                if im > 0:
                    row["mass_TB"] = 1 / im
                    row["mu_BS_TB_mass"] = dp.bardeen_shockley(f["C"], f["E1"], 1 / im, a.temp, a.dim)
                mu_b = dp.boltzmann_dp(tb[car]["E"], tb[car]["V"], tb[car]["vol"], f["C"], f["E1"], a.temp, [u])[0]
                row["mu_boltzmann_TB"] = float(mu_b)
                line += f"{row.get('mu_BS_TB_mass', float('nan')):13.4g}{mu_b:13.4g}"
            print(line)
            rows.append(row)
    print(f"C 的单位：{'GPa' if a.dim == 3 else 'N/m'}；μ / cm² V⁻¹ s⁻¹，T = {a.temp:g} K。")
    if tb:
        print("紧束缚能带由 EHT 转移积分得到 (带宽：空穴 "
              f"{1000 * tb['hole']['bandwidth']:.0f} meV，电子 {1000 * tb['electron']['bandwidth']:.0f} meV)；"
              "带宽与 k_BT 相近时能带图像本身不可靠，应与 TLT 结果对照")
    if a.out:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(a.out, "w", newline="", encoding="utf-8-sig") as fh:
            w = _csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"已写出 {a.out}")


def cmd_project(a):
    """Project files: new / run / show."""
    from . import project as P
    if a.action == "new":
        var = dict(kv.split("=", 1) for kv in a.set)
        d = P.new_project(a.file, a.template, variables=var)
        print(f"已写出项目文件 {a.file} (模板 {a.template}：{d['description']})")
        print("修改其中的 variables (文件名、温度、重组能等)，然后运行：python -m mobility_tool project run " + a.file)
        return
    data = P.load(a.file)
    if a.action == "show":
        print(f"项目 {data.get('name', '')}  ({data.get('written_by', '')}, {data.get('saved', '')})")
        for k, v in data["variables"].items():
            print(f"  ${{{k}}} = {v}")
        status = {}
        try:
            status = json.loads((Path(a.file).resolve().parent / (Path(a.file).name.split(".")[0] +
                                                                  ".status.json")).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        for name, argv in P.plan(data):
            st = status.get(name)
            tag = "未运行" if not st else ("完成" if st["returncode"] == 0 else "失败") + f" {st['finished']}"
            print(f"  [{tag}] {name}: mobility_tool {' '.join(argv)}")
        if data.get("gui"):
            print(f"  界面设置 {len(data['gui'])} 项 (界面中 文件 → 打开项目 可恢复)")
        return
    over = dict(kv.split("=", 1) for kv in a.set)
    only = [x for x in a.only.split(",") if x] if a.only else None
    if a.action == "check":
        bad = P.check(a.file, over)
        for name, msg in bad:
            print(f"  [错误] {name}: {msg}")
        print("项目检查通过：全部步骤的参数有效" if not bad else f"项目检查：{len(bad)} 个步骤有错误")
        if bad:
            raise SystemExit(1)
        return
    res = P.run(a.file, only, a.start or None, over, keep_going=a.keep_going)
    bad = [r["name"] for r in res if r["returncode"]]
    print(f"项目运行结束：{len(res) - len(bad)} 步完成" + (f"，失败：{', '.join(bad)}" if bad else ""))
    if bad:
        raise SystemExit(1)


def cmd_strain(a):
    from .crystal import find_molecules, read_cif
    from .workflow import strain_scan
    crys = read_cif(a.cif)
    mols = find_molecules(crys)
    lo, hi, st = (float(x) for x in a.range.split(":"))
    rows = strain_scan(crys, mols, a.axis, list(np.arange(lo, hi + 1e-9, st)), a.lam_e, a.lam_h, a.temp, a.cutoff)
    print(f"{'应变/%':>8}{'Jh,max':>10}{'Je,max':>10}{'μh,max':>10}{'μe,max':>10}")
    for r in rows:
        print(f"{r['strain_percent']:8.2f}{r['J_max_hole']:10.2f}{r['J_max_electron']:10.2f}"
              f"{r['mu_hole_max'] or 0:10.4g}{r['mu_electron_max'] or 0:10.4g}")


def cmd_db(a):
    """Coupling-surface database: add a crystal, list the records, export the dimers of all crystals."""
    from . import database as dbm
    if a.action == "add":
        if not a.key or not a.project:
            raise SystemExit("db add needs a record name and --project")
        try:
            r = dbm.add_record(a.db, a.key, a.project, cif=a.cif, al=a.al, reorg=a.reorg, md_csv=a.md_csv,
                               validation=a.validation, ccdc=a.ccdc, note=a.note, copy_cif=not a.no_cif,
                               overwrite=a.overwrite)
        except (FileExistsError, FileNotFoundError, ValueError) as e:
            raise SystemExit(str(e))
        al = r.get("active_learning", {})
        print(f"已保存 {a.key}：{len(r['dimers']['by_scan'])} 个扫描中的 {r['dimers']['n']} 个 DFT 二聚体"
              + (f"；主动学习 {al['name']}：{al['n_dft_total']} 个二聚体，{al['rounds']} 轮" if al else "")
              + (f"；能量列 {', '.join(r['dimers']['energy_columns'])}" if r['dimers']['energy_columns'] else ""))
        missing = [n for n, v in (("--cif", a.cif), ("--al", a.al), ("--lambda", a.reorg), ("--md-csv", a.md_csv),
                                  ("--validation", a.validation)) if not v]
        if missing:
            print("Not given (the record can be replaced later with --overwrite): " + ", ".join(missing))
        print("记录文件：", os.path.join(a.db, a.key, dbm.RECORD))
    elif a.action == "list":
        print(dbm.summary_table(dbm.summary_rows(a.db)))
    elif a.action == "export":
        out = a.out or os.path.join(a.db, "all_dimers.csv")
        n, k, sp = dbm.export(a.db, out)
        print(f"{k} 个晶体的 {n} 个二聚体已写入 {out}；汇总：{sp}")


def cmd_diagnose(a):
    from .diagnose import diagnose, format_report, write_json
    sc = tuple(int(x) for x in a.supercell.replace(",", " ").split()) if a.supercell else None
    d = diagnose(a.cif, a.lam_h, a.lam_e, T=a.temp, cutoff=a.cutoff, supercell=sc, phonons=not a.no_phonons,
                 uq=a.uq, n_samples=a.uq_samples, lam_rel=a.lam_rel, seed=a.seed)
    print(format_report(d))
    if a.json:
        write_json(d, a.json)
        print(f"Written {a.json}")


def _mu_from_md_csv(path, T):
    """μ_e, μ_h at the temperature closest to T from the μ(T) block of a ceds md csv."""
    import csv as _csv
    import io
    text = open(path, encoding="utf-8-sig").read().replace("\r\n", "\n").strip()
    for blk in text.split("\n\n"):
        rows = list(_csv.DictReader(io.StringIO(blk.strip())))
        if rows and "mu_hole" in rows[0] and "mu_electron" in rows[0]:
            r = min(rows, key=lambda x: abs(float(x["T_K"]) - T))
            return float(r["mu_electron"]), float(r["mu_hole"]), float(r["T_K"])
    raise SystemExit(f"{path}: no mobility block (columns T_K, mu_hole, mu_electron)")


def cmd_photocond(a):
    """Photogenerated carriers and photoconductivity under illumination (steady state or pulse)."""
    import numpy as np
    from . import photocond as PC
    if a.md_csv:
        mu_e, mu_h, Tm = _mu_from_md_csv(a.md_csv, a.temp)
        print(f"Mobilities from {a.md_csv} at {Tm:g} K: μe = {mu_e:.3g}, μh = {mu_h:.3g} cm² V⁻¹ s⁻¹")
    else:
        if a.mu_e is None or a.mu_h is None:
            raise SystemExit("photocond needs --mu-e and --mu-h, or --md-csv")
        mu_e, mu_h = a.mu_e, a.mu_h
    if a.eb is not None:
        eb = a.eb
    elif a.et is not None and a.eopt is not None:
        eb = a.et - a.eopt
    else:
        raise SystemExit("photocond needs --eb, or --et and --eopt")
    if eb <= 0:
        raise SystemExit(f"exciton binding energy {eb:.3f} eV must be positive")
    lst = lambda x: [float(v) for v in str(x).replace(",", " ").split()]
    I, F = lst(a.intensity), lst(a.field)
    par = dict(wavelength=a.wavelength, alpha=a.alpha, thickness=a.thickness, refl=a.refl, T=a.temp, mu_e=mu_e,
               mu_h=mu_h, eps_r=a.eps, eb=eb, tau_s=a.tau_s * 1e-12, zeta=a.zeta,
               tau_t=a.tau_trap * 1e-9 if a.tau_trap else None)
    a_nm = PC.capture_radius(eb, a.eps) * 1e9
    kl = PC.langevin(mu_e, mu_h, a.eps) * 1e6
    print(f"E_b = {eb:.3f} eV, ε_r = {a.eps:g}, e–h distance a = {a_nm:.2f} nm, k_L = {kl:.3g} cm³ s⁻¹, "
          f"absorbed fraction {PC.absorbed_fraction(a.alpha, a.thickness, a.refl):.3f}")
    if a_nm < 0.4:
        print("[note] a < 0.4 nm: E_b is larger than a Coulomb-bound pair on neighbouring molecules; check E_t and E_opt")
    rows = []
    print(f"{'I/W cm⁻²':>10s} {'F/V cm⁻¹':>10s} {'η':>10s} {'G/cm⁻³s⁻¹':>11s} {'n/cm⁻³':>10s} {'τ/s':>9s} "
          f"{'Δσ/S cm⁻¹':>11s} {'J/A cm⁻²':>10s}")
    for f in F:
        for i in I:
            r = PC.steady_state(intensity=i, field=f, **par)
            row = {"intensity_Wcm2": i, "field_Vcm": f, "T_K": a.temp, "eta": float(r["eta"]),
                   "k_d_s": float(r["k_d"]), "G_cm3s": float(r["G"]) * 1e-6, "n_cm3": float(r["n"]) * 1e-6,
                   "lifetime_s": float(r["lifetime"]), "dsigma_Scm": float(r["dsigma"]) / 100,
                   "J_Acm2": float(r["J"]) * 1e-4}
            rows.append(row)
            print(f"{i:10.3g} {f:10.3g} {row['eta']:10.3g} {row['G_cm3s']:11.3g} {row['n_cm3']:10.3g} "
                  f"{row['lifetime_s']:9.3g} {row['dsigma_Scm']:11.3g} {row['J_Acm2']:10.3g}")
    if len(I) >= 2:
        print("\nIntensity dependence Δσ ∝ I^k (local exponent between neighbouring intensities; "
              "k = 0.5: bimolecular Langevin recombination, k = 1: monomolecular/trap recombination)")
        for f in F:
            rr = [r for r in rows if r["field_Vcm"] == f]
            ks = [np.log(rr[j + 1]["dsigma_Scm"] / rr[j]["dsigma_Scm"]) / np.log(rr[j + 1]["intensity_Wcm2"] /
                  rr[j]["intensity_Wcm2"]) for j in range(len(rr) - 1) if rr[j + 1]["intensity_Wcm2"] != rr[j]["intensity_Wcm2"]]
            if ks:
                print(f"  F = {f:g} V/cm: k = " + ", ".join(f"{k:.2f}" for k in ks))
                for j, k in enumerate(ks):
                    rr[j + 1]["exponent_k"] = float(k)
        for r in rows:
            r.setdefault("exponent_k", "")
    if a.fluence:
        t = np.logspace(-12, -3, 37)
        ds, n0 = PC.pulse_decay(t, fluence=a.fluence * 1e-6, field=F[0], **par)
        print(f"\nPulse {a.fluence:g} μJ cm⁻² at {F[0]:g} V cm⁻¹: n0 = {n0 * 1e-6:.3g} cm⁻³")
        for tt, d in zip(t[::4], ds[::4]):
            print(f"  t = {tt:9.3g} s   Δσ = {d / 100:.3g} S cm⁻¹")
        if a.out:
            import os
            fp = os.path.splitext(a.out)[0] + "_pulse.csv"
            PC.export_csv([{"t_s": float(x), "dsigma_Scm": float(d) / 100} for x, d in zip(t, ds)], fp)
            print("written", fp)
    if a.out:
        PC.export_csv(rows, a.out)
        print("written", a.out)


def cmd_exciton(a):
    """Singlet-exciton transport of a crystal: transition charges → TrESP couplings → Davydov splitting, Marcus
    hopping rates, diffusion tensor and length, KMC with finite lifetime; optional rigid-body MD for the thermal
    fluctuations of the couplings."""
    import json as _json
    import numpy as np
    from . import exciton as EX
    from .ceds import CEDS
    from .crystal import find_molecules, read_cif
    crys = read_cif(a.cif)
    ce = CEDS(crys, find_molecules(crys), cutoff=a.cutoff)
    nmol = len(ce.frames)
    Y = np.asarray(ce.Y, float)
    print(f"{a.cif}: {nmol} molecules per cell, {len(ce.sym)} atoms per molecule, {len(ce.pairs)} unique contact dimers")
    if a.write_gjf:
        EX.write_tresp_gjf(a.write_gjf, ce.sym, Y, state=a.gauss_state, method=a.method, basis=a.basis)
        print(f"Gaussian input for transition charges written to {a.write_gjf} (molecular frame of this crystal)")
        return
    if a.charges and a.charges.lower() != "indo":
        sym, q = EX.read_gaussian_transition_charges(a.charges, n_atoms=len(ce.sym))
        if [x.upper() for x in sym] != [x.upper() for x in ce.sym]:
            raise SystemExit("atom order of the Gaussian log differs from the molecule of the crystal; generate the input "
                             "with --write-gjf")
        src = f"Gaussian ESP fit of the transition density ({a.charges})"
    else:
        r = EX.indo_cis(ce.sym, Y, n_states=max(4, (a.state or 0) + 1), n_occ=a.active, n_virt=a.active)
        print("INDO/S–CIS singlet states of the monomer (active space "
              f"{r['active'][0]} × {r['active'][1]}):")
        for k, st in enumerate(r["states"]):
            print(f"  S{k + 1}  {st['E_eV']:5.2f} eV  f = {st['f']:.3f}  |μ| = {np.linalg.norm(st['mu_D']):.2f} D  "
                  + ", ".join(st["main"]))
        k = a.state if a.state is not None else next((i for i, st in enumerate(r["states"]) if st["f"] >= 0.05), 0)
        q = r["states"][k]["q"]
        src = f"INDO/S–CIS S{k + 1}"
    q = EX.neutralize(q)
    mu0 = np.linalg.norm(q @ Y) * EX.DEBYE
    if a.mu:
        q = EX.rescale(q, Y, a.mu)
    f_scr = a.screening if a.screening is not None else (1.0 / a.n_refr ** 2 if a.n_refr else 1.0)
    print(f"Transition charges: {src}; |μ| = {mu0:.2f} D" + (f" rescaled to {a.mu:.2f} D" if a.mu else "")
          + f"; screening factor {f_scr:.3f}")
    rows = EX.lattice_couplings(ce, Y, q, f_scr, r_max=a.r_max)
    w, H = EX.davydov(rows, nmol)
    print(f"\nk = 0 exciton band (relative to the monomer, meV): " + ", ".join(f"{x:.1f}" for x in w)
          + (f";  Davydov splitting {w[-1] - w[0]:.1f} meV ({8.0655 * (w[-1] - w[0]):.0f} cm⁻¹)" if nmol > 1 else "")
          + f"  (lattice sum to {a.r_max:g} Å)")
    strong = sorted(rows, key=lambda r_: -abs(r_["V"]))
    seen, show = set(), []
    for r_ in strong:
        key = (round(abs(r_["V"]), 2), round(float(np.linalg.norm(r_["r"])), 2))
        if key in seen:
            continue
        seen.add(key); show.append(r_)
        if len(show) >= a.n_show:
            break
    print(f"{'pair':>14s} {'r / Å':>7s} {'V / meV':>9s}")
    for r_ in show:
        print(f"{r_['i']}→{r_['j']} {str(r_['cell']):>9s} {np.linalg.norm(r_['r']):7.2f} {r_['V']:9.2f}")
    v2, md_stats = None, None
    if a.md_ps:
        from .rigidmd import RigidCrystalMD, run_temperature, DEFAULTS
        sc = tuple(int(x) for x in a.supercell.replace(",", " ").split())
        md = RigidCrystalMD(ce.crys, ce.frames, (ce.sym, ce.Y), sc)
        o = dict(DEFAULTS, equil_ps=a.md_equil, prod_ps=a.md_ps)
        print(f"\nRigid-body MD {sc[0]}×{sc[1]}×{sc[2]} ({md.N} molecules), {a.temp:g} K, {a.md_equil:g} + {a.md_ps:g} ps")
        fr = run_temperature(md, a.temp, o, report=lambda m: print("  " + m, flush=True))
        md_stats = EX.md_coupling_stats(ce, md, fr, Y, q, f_scr)
        v2 = {k: v["mean"] ** 2 + v["std"] ** 2 for k, v in md_stats.items()}
        print(f"{'pair':>14s} {'V0':>8s} {'⟨V⟩':>8s} {'σ_V':>7s} {'σ/|⟨V⟩|':>8s}")
        done = set()
        for (i, j, t), v in sorted(md_stats.items(), key=lambda kv: -abs(kv[1]["V0"])):
            key = (round(abs(v["V0"]), 1),)
            if key in done:
                continue
            done.add(key)
            print(f"{i}→{j} {str(t):>9s} {v['V0']:8.2f} {v['mean']:8.2f} {v['std']:7.2f} "
                  f"{v['std'] / max(abs(v['mean']), 1e-9):8.2f}")
    out = {"cif": a.cif, "charges": src, "mu_D": float(np.linalg.norm(q @ Y) * EX.DEBYE), "screening": f_scr,
           "davydov_meV": [float(x) for x in w], "T": a.temp, "lambda_eV": a.lam, "tau_ns": a.tau}
    print(f"\nMarcus hopping, λ = {a.lam:g} eV, T = {a.temp:g} K, lifetime τ = {a.tau:g} ns")
    for lab, vv in (("static couplings", None), ("with thermal fluctuations ⟨V²⟩", v2)):
        if lab.startswith("with") and vv is None:
            continue
        D = EX.diffusion_tensor(rows, nmol, a.lam, a.temp, v2=vv)
        L = EX.diffusion_length(D, a.tau * 1e-9)
        print(f"  {lab}: D = " + ", ".join(f"{x:.3g}" for x in L["D"]) + " cm² s⁻¹ (principal); "
              f"L = " + ", ".join(f"{x:.1f}" for x in L["L_nm"]) + f" nm; L_3D = {L['L3_nm']:.1f} nm")
        for k, ax in enumerate(L["axes"]):
            frac = np.linalg.solve(np.asarray(ce.crys["lattice"], float).T, ax)
            print(f"     axis {k + 1}: " + " ".join(f"{x:+.2f}" for x in ax) + "  (fractional "
                  + " ".join(f"{x:+.2f}" for x in frac / np.abs(frac).max()) + ")")
        out[("D_static" if vv is None else "D_md")] = {"principal_cm2s": [float(x) for x in L["D"]],
                                                       "L_nm": [float(x) for x in L["L_nm"]], "L3_nm": L["L3_nm"]}
        if a.kmc:
            km = EX.kmc_diffusion(rows, nmol, a.lam, a.temp, a.tau * 1e-9, n_walkers=a.kmc, v2=vv)
            print(f"     KMC ({a.kmc} excitons): ⟨r²⟩^1/2 at decay = {km['L3_nm']:.1f} nm "
                  f"(analytic L_3D {L['L3_nm']:.1f} nm), {km['hops']:.0f} hops per exciton")
            out[("kmc_static" if vv is None else "kmc_md")] = km
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            _json.dump(out, fh, indent=1, ensure_ascii=False)
        print("written", a.out)


def cmd_tlt(a):
    from .crystal import find_molecules, read_cif
    from .kmc import graph_from_crystal
    from .transport import effective_mass, tlt_mobility
    from .workflow import eht_couplings_for_crystal
    crys = read_cif(a.cif)
    mols = find_molecules(crys)
    res0, coup = eht_couplings_for_crystal(crys, mols, a.cutoff, pi=a.pi)
    for w in res0.get("eht_warnings", []):
        print(f"  [警告] {w}")
    g = graph_from_crystal(crys, mols, coup, a.cutoff, carrier=a.carrier, signed=True)
    keep = tuple("abc".index(x) for x in a.plane)
    L = a.size or max(8, int(round(np.sqrt(600 / g["n_sites"]))))
    r = tlt_mobility(g, a.temp, a.carrier, a.sigma, a.htau / 1000, tuple(L if i in keep else 1 for i in range(3)),
                     keep, a.realizations)
    print(f"TLT ({a.carrier}, σ/|J|={a.sigma}, ħ/τ={a.htau} meV, 平面 {a.plane}): μ 主值 " +
          ", ".join(f"{w:.4g}" for w in r["mu_principal"][:2]) + f" cm²/Vs, L ≈ {r['L_A']:.1f} Å")
    em = effective_mass(g, a.carrier, crys["lattice"], 16)
    print(f"紧束缚带宽 {em['bandwidth_eV'] * 1000:.1f} meV, 有效质量 " +
          ", ".join("∞" if not np.isfinite(m) else f"{m:.3g}" for m in em["masses_me"]) + " m_e")


def cmd_vibronic(a):
    from .gaussian_fcht import read_fcht_log
    from .vibronic import emission_spectrum, energy_gap_law
    r = read_fcht_log(a.log)
    f, s = [m["freq_cm-1"] for m in r["modes"]], [m["S"] for m in r["modes"]]
    sp = emission_spectrum(f, s, e_vert=a.ev, T=a.temp, fwhm_ev=a.fwhm)
    eg = energy_gap_law(f, s, sp["E00_eV"])
    print(f"E00 {sp['E00_eV']:.3f} eV, 峰 {sp['peak_nm']:.1f} nm, FWHM {sp['fwhm_cm']:.0f} cm⁻¹, ΣS {sp['S_total']:.3f}")
    print(f"能隙定律: S_M={eg['S_M']:.3f}, ħω_M={eg['hw_M_cm']:.0f} cm⁻¹, γ={eg['gamma']:.2f}, "
          f"k_nr/C²={eg['knr_over_C2']:.3e} s⁻¹eV⁻²")


def cmd_kmc_validate(a):
    from .validation import figure, format_rows, kmc_validation, write_csv
    cr = None
    if a.cif:
        from .workflow import eht_couplings_for_crystal
        crys = read_cif(a.cif)
        mols = find_molecules(crys)
        _, coup = eht_couplings_for_crystal(crys, mols, a.cutoff)
        cr = (crys, mols, coup, a.cutoff)
    rows = kmc_validation(quick=a.quick, crystal=cr)
    from .i18n import tr_cli
    rows = [(tr_cli(r[0]),) + tuple(r[1:]) for r in rows]
    print(format_rows(rows))
    if a.out:
        write_csv(rows, a.out)
        import matplotlib
        matplotlib.use("Agg")
        figure(rows, str(Path(a.out).with_suffix(".png")))
        print("已写出", a.out, "及对照图")


def _pairs(s):
    out = []
    for part in str(s).replace("，", ";").split(";"):
        v = [float(x) for x in part.replace(",", " ").split()]
        if v:
            out.append(tuple(v[:2]))
    return out


def cmd_scan(a):
    """批量扫描工作流。"""
    from .scanflow import ScanProject, grid_configs
    act = a.action
    if act == "new":
        pr = ScanProject.create(a.project, a.mono, a.name, frame=a.frame)
        r = pr.meta["monomer"]["report"]
        print(f"已新建 {pr.root}: {r['formula']}, 点群 {r['point_group']} ({', '.join(r['ops'])}), 平面性 {r['planarity_A']:.1e} Å，"
              f"分子坐标系 {'共轭主体' if a.frame == 'core' else '全部重原子'}")
        return
    pr = ScanProject(a.project)
    lv = getattr(a, "level", None) or next(iter(pr.meta["levels"]), None)
    if act == "level" and a.engine != "adf":
        lv = pr.add_qc_level(a.engine, method=a.method, basis=a.qc_basis, gfn=a.gfn, nprocs=a.nprocs,
                             maxcore=a.maxcore)
        ld = pr.level_dir(lv)
        print(f"层级 {lv}：参考单体 {ld / 'run_mono.sh'} (Windows: run_mono.bat) → mono.molden.input"
              + ("" if pr.t21(lv).exists() else "  (run it first)"))
        print("  every dimer: calculations of B and AB, coupling by dimer projection on the Molden files; set ORCA_EXE / "
              "XTB_EXE to the full program path if it is not on PATH")
        return
    if act == "level":
        lv = pr.add_level(a.xc, a.basis, a.d3)
        if a.import_t21:
            pr.import_fragment(lv, a.import_t21)
        print(f"层级 {lv}: {pr.level_dir(lv) / 'mono.run'}" + ("" if pr.t21(lv).exists() else "  (请先运行，生成 mono.t21)"))
        if a.mirror:
            pm = pr._write_mono_job(lv, mirror=True)
            print(f"镜像单体 (对映体分子)：{pm}" + ("" if pr.t21(lv, mirror=True).exists() else "  (请先运行，生成 mono_m.t21)"))
    elif act == "grid":
        info = pr.add_scan(a.name, grid_configs({"z": a.z, "dx": a.dx, "dy": a.dy, "gamma": a.gamma, "pitch": a.pitch,
                                                 "roll": a.roll}), lv, clash_min=a.clash, chunk=a.chunk,
                           reduce=not a.no_reduce)
        print(f"{a.name}: 请求 {info['n_requested']}，计算 {info['n_computed']}，对称映射 {info['n_symmetry']}，"
              f"接触筛除 {info['n_clash']}；批处理 {', '.join(info['batches'])}")
    elif act == "valid":
        dx, dy = _pairs(a.dx)[0], _pairs(a.dy)[0]
        bl = []
        if a.interp:
            bl.append({"tag": "interp", "n": a.interp, "z": _pairs(a.zi)[0], "dx": dx, "dy": dy})
        ze = _pairs(a.ze)
        if a.extrap and ze:
            w = [hi - lo for lo, hi in ze]
            k = [int(a.extrap * x / sum(w)) for x in w]
            k[0] += a.extrap - sum(k)
            bl += [{"tag": "extrap", "n": m, "z": zr, "dx": dx, "dy": dy} for zr, m in zip(ze, k) if m]
        info = pr.add_validation(a.name, lv, bl, seed=a.seed, clash_min=a.clash, chunk=a.chunk)
        print(f"{a.name}: {info['n_computed']} 个验证构型；批处理 {', '.join(info['batches'])}")
    elif act == "check":
        bases = []
        for part in a.bases.split(";"):
            v = [float(x) for x in part.replace(",", " ").split()] + [0.0] * 4
            if len(v) > 4:
                bases.append({"z": v[0], "dx": v[1], "dy": v[2], "roll": v[3]})
        info = pr.add_check(a.name, lv, bases)
        print(f"{a.name}: {info['n_computed']} 个对称变体；批处理 {', '.join(info['batches'])}")
    elif act == "clone":
        info = pr.clone(a.name, a.level)
        print(f"{info['name']}: {info['n_computed']} 个构型，层级 {a.level}")
    elif act == "import":
        n = pr.import_scan(a.dir, a.name, a.level)
        print(f"已导入 {n}")
    elif act == "status":
        for r in pr.status_table():
            print(r)
    elif act == "precheck":
        r = pr.precheck(a.name)
        print(f"{a.name}: {r['n_done']} / {r['n_total']} 已完成，待算 {r['n_pending']}，失败 {r['n_failed']}")
        for e in r["errors"]:
            print("[错误]", e)
        for w in r["warnings"]:
            print("[提示]", w)
    elif act == "collect":
        for n in ([a.name] if a.name else pr.scans()):
            recs = pr.collect(n)
            print(f"{n}: {sum(r['state'] == 'done' for r in recs if not r['rep'])} 个构型已解析 → "
                  f"{pr.scan_dir(n) / 'results.csv'}")
    elif act in ("cp", "ti"):
        fn = pr.gaussian_cp if act == "cp" else pr.gaussian_ti
        out, k = fn(a.name, functional=a.functional, basis=a.basis, stride=a.stride)
        print(f"{k} 个构型的 Gaussian 输入: {out}")
    elif act == "report":
        import matplotlib
        matplotlib.use("Agg")
        from .scananalysis import make_report
        lam = {c: v for c, v in (("hole", a.lam_h), ("electron", a.lam_e)) if v is not None} or None
        res = make_report(pr, lam=lam, reorg_dir=a.reorg_dir, T=a.temp, energy=a.energy, z0=a.z0, level=a.level,
                          on_progress=lambda m: print("  …", m))
        print("\n".join(res["summary"]))
        print("报告:", res["html"])
    elif act == "archive":
        path, n = pr.archive(a.out or None, include_outputs=not a.no_outputs)
        print(f"已归档 {n} 个文件: {path}")


def cmd_ceds(a):
    """CEDS：晶体局域势 MC → σ_V(T) → TLT μ(T)。"""
    import matplotlib
    matplotlib.use("Agg")
    from .ceds import (CEDS, EHTSurface, PairPotential, cif_ueq, export_csv, summary_table, surfaces_from_project,
                       temperature_series)
    from .crystal import find_molecules, read_cif
    crys = read_cif(a.cif)
    mols = find_molecules(crys)
    proj = None
    if a.project:
        from .scanflow import ScanProject
        proj = ScanProject(a.project)
    mono = (proj.symbols, proj.Y) if proj and sorted(proj.symbols) == sorted(mols[0]["symbols"]) else None
    frame = proj.frame if proj else a.frame
    if a.monomer:
        mono = _read_monomer(a.monomer, frame)
        if sorted(mono[0]) != sorted(mols[0]["symbols"]):
            raise SystemExit(f"--monomer 的组成与晶体中的分子不同")
    ce = CEDS(crys, mols, cutoff=a.cutoff, monomer=mono, potential=PairPotential(scale=a.scale), frame=frame)
    print(f"{len(ce.pairs)} 个唯一二聚体；特征标 {ce.characters}；分子坐标系 {'共轭主体' if frame == 'core' else '全部重原子'}")
    for n_ in ce.notes:
        print("[提示]", n_)
    temps = [float(x) for x in a.temps.replace(",", " ").split()]
    target = a.ueq or (cif_ueq(a.cif) if a.ueq_from_cif else None)
    if target and a.action == "md":
        print(f"[提示] MD 用 OPLS-AA 力场，不做 U_eq 校准；晶体学 U_eq = {target:.4f} Å² 只作对照")
    elif target:
        h = ce.calibrate(target, T=temps[-1])
        print(f"势能缩放 {h[0][0]:.3f} → {h[-1][0]:.3f}，U_eq {h[0][1]:.4f} → {h[-1][1]:.4f} Å² (目标 {target:.4f})")
    if a.action == "local":
        if not proj:
            raise SystemExit("生成局域扫描需要 --project")
        if a.md_half:
            from .rigidmd import RigidCrystalMD, load_frames, md_half, run_temperature
            sc = tuple(int(x) for x in a.supercell.replace(",", " ").split())
            md = RigidCrystalMD(ce.crys, ce.frames, mono or (proj.symbols, proj.Y), sc, rc=a.md_rc, r_sw=a.md_rc - 1.0,
                                charges=_md_charges(a, mono or (proj.symbols, proj.Y)))
            if a.traj:
                fr = load_frames(a.traj)
            else:
                print(f"运行 MD 确定扫描范围：T = {temps[-1]:g} K，超胞 {'×'.join(map(str, sc))} ({md.N} 个分子) …", flush=True)
                fr = run_temperature(md, temps[-1], dict(equil_ps=a.md_equil, prod_ps=a.md_prod, dt=a.md_dt,
                                                         every=a.md_every))
                if a.save_traj:
                    import os
                    os.makedirs(a.save_traj, exist_ok=True)
                    md.save(os.path.join(a.save_traj, f"md_{temps[-1]:g}K.npz"), fr)
            mc_half = ce.suggest_half(T=temps[-1])
            half = {g: h for g, h in md_half(ce, md, fr, set(mc_half), k=a.md_k).items()}
            print(f"扫描范围 = ±{a.md_k:g}σ (全晶体 MD)")
        else:
            half = ce.suggest_half(T=temps[-1])
        if a.pairs:
            if a.pairs.strip().lower() == "parallel":
                want = {p["gid"] for p in ce.pairs if p["r_centroid"] < 7.0 and abs(p["center"]["roll"]) < 5.0}
            else:
                want = {int(x) for x in a.pairs.replace(",", " ").split()}
            half = {g: h for g, h in half.items() if g in want}
        print(f"{'组':<4}{'r/Å':>8}{'z':>8}{'Δx':>8}{'Δy':>8}{'φ':>8}")
        for p in ce.pairs:
            q = p["center"]
            print(f"G{p['gid']:<3}{p['r_centroid']:8.3f}{q['z']:8.2f}{q['dx']:8.2f}{q['dy']:8.2f}{q['roll']:8.2f}"
                  + ("   ← 局域扫描" if p["gid"] in half else ""))
        cen = {g: c for g, c in ce.centers().items() if g in half}
        lv = a.level or next(iter(proj.meta["levels"]))
        info = proj.add_local(a.name, lv, cen, n=a.n_local, half=half)
        print(f"局域扫描 {a.name}: {len(half)} 个二聚体，{info['n_computed']} 个构型；批处理 {', '.join(info['batches'])}")
        for g, hh in half.items():
            print(f"  G{g}: " + ", ".join(f"{k} ±{v:.2f}" for k, v in hh.items()))
        return
    if a.action == "snapshots":
        _ceds_snapshots(a, ce, proj, temps, mono)
        return
    if a.action == "al":
        _ceds_al(a, ce, proj, temps, mono)
        return
    if a.g_csv or a.local_csv:
        from .ceds import combined_surfaces, local_surfaces_from_csv, product_surfaces
        prod = (product_surfaces(ce, a.g_csv, a.fdx_csv, a.slip_csv, a.dy_line, a.model, a.gphi_csv or ())
                if a.g_csv else {})
        loc = local_surfaces_from_csv(ce, a.local_csv) if a.local_csv else {}
        surf, why = combined_surfaces(ce, proj if a.source == "project" else None, a.level, prod, a.eht_fallback, loc)
        for g, w in why.items():
            print(f"  G{g}: {w}")
        if a.action == "run" and not a.sigma_only and len(surf) < len([p for p in ce.pairs if p["r_centroid"] < 7.0]):
            print("[提示] 部分近邻二聚体没有曲面 (例如平行二聚体)，TLT 不完整：只输出 σ_V。平行二聚体可用 "
                  "ceds local 生成局域扫描后加 --project 一起使用 (加 --sigma-only 可省略此提示)")
            a.sigma_only = True
        if a.action == "md" and len(surf) < len([p for p in ce.pairs if p["r_centroid"] < 7.0]):
            print("[提示] 部分近邻二聚体没有曲面，TLT 只包含有曲面的键 (μ 偏低或方向不完整)；σ_V 不受影响")
    elif a.source == "eht" or not proj:
        s_ = {m: EHTSurface(ce.sym, ce.Y, mirror=m) for m in (False, True)}
        surf = {p["gid"]: s_[bool(p.get("mirror"))] for p in ce.pairs}
    else:
        surf, why = surfaces_from_project(proj, ce, level=a.level, eht_fallback=a.eht_fallback)
        for g, w in why.items():
            print(f"  G{g}: {w}")
    if a.al_surfaces:
        if not proj:
            raise SystemExit("--al-surfaces 需要 --project")
        from .alsurface import ALRun
        run = ALRun(proj, a.al_surfaces)
        if not run.exists:
            raise SystemExit(f"项目中没有主动学习任务 {a.al_surfaces}")
        gp = run.surfaces(ce)
        surf = dict(surf or {})
        surf.update(gp)
        for g, s_ in gp.items():
            print(f"  G{g}: {s_.describe()}")
    if a.action == "md":
        _ceds_md(a, ce, surf, temps, mono)
        return
    if a.action == "phonons":
        _ceds_phonons(a, ce, surf, temps, mono)
        return
    if a.action == "validate":
        _ceds_validate(a, ce, proj, surf)
        return
    if a.sigma_only:
        res_T = {}
        for T in temps:
            res_T[T] = ce.run(surf, T, n_samples=a.samples)
            for c in ("hole", "electron"):
                print(f"\n[{c}] T = {T:g} K")
                print(summary_table(ce, res_T[T], c))
        if a.out:
            export_csv(ce, res_T, a.out)
            print("已写出", a.out)
        return
    rows = temperature_series(ce, surf, temps, hbar_tau=a.htau / 1000, n_samples=a.samples)
    for c in ("hole", "electron"):
        print(f"\n[{c}] T = {temps[-1]:g} K")
        print(summary_table(ce, rows[-1]["res"], c))
    print(f"\n{'T/K':>6}{'mu_h':>10}{'mu_h(s=0)':>11}{'mu_e':>10}{'mu_e(s=0)':>11}")
    for r in rows:
        print(f"{r['T']:6.0f}{r['mu_hole']:10.3g}{r['mu_static_hole']:11.3g}{r['mu_electron']:10.3g}{r['mu_static_electron']:11.3g}")
    if a.out:
        export_csv(ce, {r["T"]: r["res"] for r in rows}, a.out, rows)
        print("已写出", a.out)


def cmd_polar(a):
    """极化位点能：固态极化能 P±、各独立分子的位点能差、MD 快照的热涨落 σ_E。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
                                             "DejaVu Sans"], "axes.unicode_minus": False})
    from . import polar as P
    from .ceds import CEDS
    from .crystal import find_molecules, read_cif
    crys = read_cif(a.cif)
    mols = find_molecules(crys)
    mono = _read_monomer(a.monomer, a.frame) if a.monomer else None
    if mono and sorted(mono[0]) != sorted(mols[0]["symbols"]):
        raise SystemExit("--monomer 的组成与晶体中的分子不同")
    ce = CEDS(crys, mols, cutoff=4.0, monomer=mono, frame=a.frame)
    if a.gjf:
        for f in P.write_charge_inputs(ce.sym, ce.Y, a.gjf, level=a.level or "B3LYP/6-31G(d)"):
            print("已生成", f)
        return
    ch = P.charge_set(ce.sym, ce.Y, a.neutral, a.cation, a.anion)
    table = None
    if a.alpha_mean or a.polar_log:
        target = a.alpha_mean or float(np.trace(P.read_polarizability(a.polar_log)) / 3)
        table, s = P.scale_alpha(ce.sym, ce.Y, target)
        print(f"原子极化率缩放 ×{s:.3f}，分子平均极化率 = {target:.1f} Å³")
    radii = [float(x) for x in a.radii.replace(",", " ").split()]
    say = (lambda m: print("  " + m, flush=True)) if not a.quiet else None
    res = P.site_energies(crys["lattice"], ce.frames, ce.sym, ce.Y, ch, radii, table, on_progress=say)
    md = None
    if a.traj:
        tr = P.load_traj(a.traj)
        chm = P.charge_set(tr["symbols"], tr["Y"], a.neutral, a.cation, a.anion)
        md = P.md_site_energies(tr, chm, a.md_frames, a.md_centers, a.md_r, nk=len(ce.frames), alpha_table=table,
                                on_progress=say)
    lv = P.solid_levels(res, a.ip_gas, a.ea_gas)
    print(P.summary_text(res, lv, md))
    if a.out:
        P.export_csv(res, a.out, md, lv)
        fig, axs = plt.subplots(1, 2 if md else 1, figsize=(10 if md else 5.5, 4.2), squeeze=False)
        P.plot_extrapolation(axs[0, 0], res)
        if md:
            P.plot_md_hist(axs[0, 1], md)
        fig.tight_layout()
        fig.savefig(str(Path(a.out).with_suffix(".png")), dpi=200)
        print("已写出", a.out, "与", Path(a.out).with_suffix(".png").name)


def _element_map(text):
    out = {}
    for item in (text or "").replace(";", ",").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def cmd_cif2gro(a):
    """晶体超胞写成 .gro (GROMACS 起始结构 / 形貌流程测试)。"""
    from .ceds import CEDS
    from .crystal import find_molecules, read_cif
    from .morphology import crystal_frame, write_gro
    crys = read_cif(a.cif)
    mols = find_molecules(crys)
    mono = _read_monomer(a.monomer, a.frame) if a.monomer else None
    ce = CEDS(crys, mols, cutoff=4.0, monomer=mono, frame=a.frame)
    sc = tuple(int(x) for x in a.supercell.replace(",", " ").split())
    fr = crystal_frame(crys["lattice"], ce.frames, ce.sym, ce.Y, sc, a.resname)
    write_gro(a.out, [fr])
    print(f"已写出 {a.out}：{len(fr['X'])} 个原子，{len(fr['X']) // len(ce.sym)} 个分子 (残基名 {a.resname})")


def cmd_morph(a):
    """MD 形貌 (GROMACS) → 映射 → 近邻 → J → 位点能 → KMC。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
                                             "DejaVu Sans"], "axes.unicode_minus": False})
    from . import morphology as Mo
    from .polar import charge_set
    say = (lambda m: print("  " + m, flush=True)) if not a.quiet else None
    frames = Mo.read_trajectory(a.gro, a.traj or None, {a.frame})
    if not frames:
        raise SystemExit(f"形貌中没有这一帧 (序号 {a.frame})")
    sym, Y = _read_monomer(a.monomer)
    m = Mo.Morphology(frames[0], sym, Y, a.resname or None, _element_map(a.elements), a.geometry)
    m.neighbors(a.cutoff)
    if a.import_j:
        print(f"读入 {m.import_couplings(a.import_j)} 个转移积分：{a.import_j}")
    elif a.coupling == "eht":
        m.couplings_eht(say)
    elif a.coupling == "zindo":
        m.couplings_zindo(a.zindo_orbitals or None, (a.degeneracy or 0) / 1000, say)
    if a.export_dimers:
        if a.calib_pairs:
            idx = m.calibration_pairs(a.calib_pairs)
        else:
            idx = range(min(len(m.pairs), a.max_pairs)) if a.max_pairs else None
        lvl = (a.level or "B3LYP/6-31G(d)").split("/")
        made = m.export_dimers(a.export_dimers, idx, method=lvl[0], basis=lvl[1] if len(lvl) > 1 else "6-31G(d)")
        print(f"已为 {len(made)} 个近邻对写出 Gaussian 投影法输入：{a.export_dimers}"
              + ("  (按 |J| 分布挑选，用于 --calibrate)" if a.calib_pairs else ""))
    if a.collect_dft:
        print(f"读回 {m.collect_dft(a.collect_dft)} 对 Gaussian 结果 (直接替换这些对的 J)")
    if a.calibrate or a.calibrate_csv:
        ref = m.read_dft(a.calibrate) if a.calibrate else m.read_couplings_csv(a.calibrate_csv)
        print(f"读入 {len(ref)} 对参考转移积分，校准 {m.J_raw_source if getattr(m, 'J_raw', None) is not None else ''}：")
        m.calibrate(ref, replace=not a.calib_no_replace)
        print("  " + m.calibration_text().replace("\n", "\n  "))
    elif a.scale_h is not None or a.scale_e is not None:
        m.calibrate_scale(a.scale_h if a.scale_h is not None else 1.0, a.scale_e if a.scale_e is not None else 1.0)
        print("  " + m.calibration_text().replace("\n", "\n  "))
    if a.site_energy != "none":
        ch = charge_set(sym, Y, a.neutral, a.cation, a.anion)
        m.site_energies(ch, a.site_energy, a.site_r or None, on_progress=say)
    print(m.summary())
    kres = {}
    if getattr(m, "J", None) is not None and (a.lam_h or a.lam_e):
        for car, lam in (("hole", a.lam_h), ("electron", a.lam_e)):
            if not lam:
                continue
            r0 = m.kmc(car, lam, a.temp, 0.0, a.walkers, a.steps, use_energies=False)
            line = f"{'空穴' if car == 'hole' else '电子'} KMC (T = {a.temp:g} K, λ = {lam * 1000:.0f} meV)：只有 J 的无序 μ = {r0['mu_avg']:.3g}"
            if getattr(m, "E", None) is not None:
                r1 = m.kmc(car, lam, a.temp, 0.0, a.walkers, a.steps)
                line += f"；加位点能 μ = {r1['mu_avg']:.3g} ± {3 * r1['err']:.2g}"
                kres[car] = (r0, r1)
                if a.field:
                    rf = m.kmc(car, lam, a.temp, a.field, a.walkers, a.steps)
                    line += f"；场迁移率 ({a.field:g} V/cm) μ = {rf['mu_avg']:.3g}"
            line += " cm²/Vs"
            print(line)
    if a.out:
        files = m.export(a.out)
        fig = plt.figure(figsize=(9.5, 7))
        Mo.plot_overview(fig, m)
        fig.savefig(a.out + ".png", dpi=200)
        print("已写出", ", ".join(Path(f).name for f in files), "与", Path(a.out + ".png").name)


def _crystal_pairs(crys, frames, sym, Y, cutoff=4.0):
    """晶体中的全部不等价近邻对 (按质心距离去重)：[(名称, Xa, Xb)]。"""
    from .morphology import Morphology, crystal_frame
    m = Morphology(crystal_frame(crys["lattice"], frames, sym, Y, (3, 3, 2), "MOL"), sym, Y)
    m.neighbors(cutoff)
    seen, out = set(), []
    for k, (i, j, n, v, dm) in enumerate(m.pairs):
        key = round(float(np.linalg.norm(v)), 2)
        if key in seen:
            continue
        seen.add(key)
        a, b = m.pair_atoms(k)
        out.append((f"crystal_{key:.2f}", a, b))
    return out


def cmd_zindo(a):
    """ZINDO/MOO 转移积分：单个二聚体、检验集输入与结果汇总。"""
    from . import zindo as Z
    deg = (a.degeneracy or 0.0) / 1000
    if a.action == "pair":
        from .gaussian_io import read_geometry
        at = read_geometry(a.dimer)
        n = a.na or len(at) // 2
        sa = [x[0] for x in at[:n]]
        sb = [x[0] for x in at[n:]]
        Xa = np.array([x[1:] for x in at[:n]], float)
        Xb = np.array([x[1:] for x in at[n:]], float)
        moo = Z.MOO(sa, Xa, a.orbitals or None, deg)
        j = moo.coupling_atoms(Xa, Xb, sym_b=sb)
        Xb = moo._ordered(Xb, sb)
        print(f"单体轨道：{moo.source}")
        print(f"ZINDO/MOO        空穴 {1000 * j['hole']:9.3f}  电子 {1000 * j['electron']:9.3f} meV")
        if not a.orbitals:
            jd = Z.dimer_projection(sa, Xa, Xb, deg)
            print(f"二聚体 INDO/S 投影 空穴 {1000 * jd['hole']:9.3f}  电子 {1000 * jd['electron']:9.3f} meV")
        return
    if a.action == "bench-inputs":
        from .ceds import CEDS
        from .crystal import find_molecules, read_cif
        if not a.cif:
            raise SystemExit("bench-inputs 需要 --cif")
        crys = read_cif(a.cif)
        mols = find_molecules(crys)
        mono = _read_monomer(a.monomer, a.frame) if a.monomer else None
        ce = CEDS(crys, mols, cutoff=4.0, monomer=mono, frame=a.frame)
        cases = Z.benchmark_set(ce.sym, ce.Y, _crystal_pairs(crys, ce.frames, ce.sym, ce.Y, a.cutoff))
        names = Z.write_benchmark_inputs(a.out, ce.sym, cases, ce.Y, a.level or "B3LYP/6-31G(d)")
        print(f"已写出 {len(names)} 个检验构型到 {a.out}：{', '.join(names)}")
        print("每个构型：Gaussian ZINDO (zindo_A/B/AB) 与 DFT 投影法 (dft_A/B/AB)；另有参考单体 mono_zindo.gjf。"
              "在 Windows 上运行 run_all.bat，然后用 zindo bench --dir 汇总")
        return
    if a.action == "bench":
        import csv
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
                                                 "DejaVu Sans"], "axes.unicode_minus": False})
        rows = Z.collect_benchmark(a.dir, deg)
        keys = ["case", "r_com_A"] + [f"{k}_{c}" for k, _ in Z.BENCH_COLS for c in ("h", "e")] + ["gauss_orth_dev"]
        out = a.out or str(Path(a.dir) / "zindo_benchmark.csv")
        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for r in rows:
                w.writerow([r.get(k, "") if isinstance(r.get(k, ""), str) else
                            (f"{r[k]:.4f}" if k in r and r[k] is not None else "") for k in keys])
        print(f"{'构型':<18}" + "".join(f"{k:>14}" for k, _ in Z.BENCH_COLS) + "   (空穴 J / meV)")
        for r in rows:
            print(f"{r['case']:<18}" + "".join(f"{r[k + '_h']:14.2f}" if r.get(k + "_h") is not None else f"{'—':>14}"
                                                for k, _ in Z.BENCH_COLS))
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.6))
        st = Z.plot_benchmark(axs, rows)
        if st:
            print(f"\n相对 {dict(Z.BENCH_COLS)[st[0]['ref']]} 的 |J|：斜率 k 为过原点拟合；相对误差只计 |J_ref| > 20 meV 的对")
        for c in ("h", "e"):
            for s_ in [x for x in st if x["carrier"] == c]:
                print(f"  {Z.CARRIER_LABEL[c]}  {s_['method']:<28} k {s_['slope']:.3f}  R² {s_['R2']:.3f}  "
                      f"MAE {s_['MAE_meV']:5.1f} meV  相对误差 {100 * s_['rel_err']:3.0f}%  "
                      f"(按 k 缩放后 {100 * s_['rel_err_scaled']:3.0f}%)  n = {s_['n']}")
        fig.tight_layout()
        fig.savefig(str(Path(out).with_suffix(".png")), dpi=200)
        print("已写出", out, "与", Path(out).with_suffix(".png").name)


def _read_monomer(path, frame="all"):
    """单体：xyz / gjf / log，或扫描项目的 project.json (其中的 DFT 优化单体)。转到分子坐标系。"""
    import json
    from .surrogate import frame_atoms, molecular_frame
    if str(path).lower().endswith(".json"):
        m = json.load(open(path, encoding="utf-8"))["monomer"]
        sym, X = list(m["symbols"]), np.asarray(m["Y"], float)
    else:
        from .gaussian_io import read_geometry
        atoms = read_geometry(path)
        sym, X = [a_[0] for a_ in atoms], np.array([a_[1:] for a_ in atoms], float)
    Y, _, ax, cen = molecular_frame(sym, X, frame_atoms(sym, X, frame))
    if np.linalg.norm(cen) < 1e-3 and np.allclose(np.abs(ax), np.eye(3), atol=1e-4):
        Y = X.copy()                     # 已在分子坐标系中 (如 project.json / mono_frame.xyz)：原样保留
    return sym, Y


def _md_charges(a, monomer):
    """--md-charges：Gaussian 输出中的原子电荷 (ESP 优先)，按单体原子顺序；未给出时返回 None (用默认力场电荷)。"""
    path = getattr(a, "md_charges", None)
    if not path:
        return None
    from .polar import charges_from_log
    sym, Y = monomer
    q, kind, rmsd = charges_from_log(sym, Y, path)
    print(f"MD 原子电荷：{Path(path).name} 的 {kind} 电荷 ({len(q)} 个原子，与单体对应 RMSD {rmsd:.2f} Å)", flush=True)
    if rmsd > 0.3:
        print("  [注意] 电荷文件的几何与单体差别较大，请确认是同一分子", flush=True)
    return q


def _md_pairs(a, ce):
    if a.pairs.strip():
        if a.pairs.strip().lower() == "parallel":
            return {p["gid"] for p in ce.pairs if p["r_centroid"] < 7.0 and abs(p["center"]["roll"]) < 5.0}
        return {int(x) for x in a.pairs.replace(",", " ").split()}
    return {p["gid"] for p in ce.pairs if p["r_centroid"] < 7.0}


def _ceds_snapshots(a, ce, proj, temps, mono):
    """ceds snapshots：从 MD 轨迹取二聚体构型，写成扫描项目中的 ADF 批处理 (逐帧直接计算 V)。"""
    from .rigidmd import RigidCrystalMD, load_frames, run_temperature, snapshot_configs
    if proj is None:
        raise SystemExit("ceds snapshots 需要 --project")
    if mono is not None and sorted(mono[0]) == sorted(proj.symbols) and not np.allclose(np.asarray(mono[1]), proj.Y):
        print("[提示] --monomer 与扫描项目的单体不同；构型按项目单体生成，MD 用 --monomer 的单体")
    sc = tuple(int(x) for x in a.supercell.replace(",", " ").split())
    md = RigidCrystalMD(ce.crys, ce.frames, mono or (proj.symbols, proj.Y), sc, rc=a.md_rc, r_sw=a.md_rc - 1.0,
                                charges=_md_charges(a, mono or (proj.symbols, proj.Y)))
    T = temps[-1]
    if a.traj:
        fr = load_frames(a.traj)
        if fr["c"].shape[1] != md.N:
            raise SystemExit(f"轨迹中有 {fr['c'].shape[1]} 个分子，与超胞 {sc} ({md.N} 个) 不符")
        src = a.traj
    else:
        print(f"运行 MD：T = {T:g} K，超胞 {'×'.join(map(str, sc))} ({md.N} 个分子)，{a.md_equil:g} + {a.md_prod:g} ps …",
              flush=True)
        fr = run_temperature(md, T, dict(equil_ps=a.md_equil, prod_ps=a.md_prod, dt=a.md_dt, every=a.md_every))
        src = "md"
        if a.save_traj:
            import os
            os.makedirs(a.save_traj, exist_ok=True)
            md.save(os.path.join(a.save_traj, f"md_{T:g}K.npz"), fr)
    gids = _md_pairs(a, ce)
    cfgs = snapshot_configs(ce, md, fr, gids, n_per_pair=a.n_snap)
    lv = a.level or next(iter(proj.meta["levels"]))
    info = proj.add_md_snapshots(a.name, lv, cfgs, info={"T": T, "n_per_pair": a.n_snap, "pairs": sorted(gids),
                                                         "supercell": list(sc), "trajectory": src,
                                                         "n_frames": int(len(fr["c"]))})
    print(f"MD 快照 {a.name}: {len(gids)} 个二聚体 × {a.n_snap} 个构型 = {info['n_computed']} 个 ADF 任务 (层级 {lv})；"
          f"批处理 {', '.join(info['batches'])}")
    print("在 ADFjobs 中运行这些批处理，收集结果后用 ceds validate 与耦合曲面比较")


def _ceds_al(a, ce, proj, temps, mono):
    """ceds al：主动学习耦合曲面。第一次运行：全晶体 MD → 候选池 → 第 1 轮 ADF 批处理；之后每次运行：
    读回已完成的结果，拟合 GP，按 σ_V 的后验不确定度判断收敛，未收敛的近邻对选出下一批写成新一轮。"""
    from .alsurface import CI_INFLATE_DELTA, ALRun, al_report
    if proj is None:
        raise SystemExit("ceds al 需要 --project 与 --name")
    a.al_mu_target_set = getattr(a, "al_mu_target", None) is not None
    if not a.al_mu_target_set:
        a.al_mu_target = 0.01
    run = ALRun(proj, a.name)
    if not run.exists and getattr(a, "al_calib_report", False):
        raise SystemExit(f"项目中没有主动学习任务 {a.name}：--al-calib-report 只用于已有的任务 (检查 --project 与 --name)")
    if not run.exists:
        from .rigidmd import RigidCrystalMD, load_frames, run_temperature, snapshot_configs
        sc = tuple(int(x) for x in a.supercell.replace(",", " ").split())
        md = RigidCrystalMD(ce.crys, ce.frames, mono or (proj.symbols, proj.Y), sc, rc=a.md_rc, r_sw=a.md_rc - 1.0,
                                charges=_md_charges(a, mono or (proj.symbols, proj.Y)))
        T = temps[-1]
        if a.traj:
            fr = load_frames(a.traj)
            if fr["c"].shape[1] != md.N:
                raise SystemExit(f"轨迹中有 {fr['c'].shape[1]} 个分子，与超胞 {sc} ({md.N} 个) 不符")
        else:
            print(f"运行 MD 生成候选构型：T = {T:g} K，超胞 {'×'.join(map(str, sc))} ({md.N} 个分子) …", flush=True)
            fr = run_temperature(md, T, dict(equil_ps=a.md_equil, prod_ps=a.md_prod, dt=a.md_dt, every=a.md_every))
            run.dir.mkdir(parents=True, exist_ok=True)
            md.save(str(run.dir / f"{a.name}_md_{T:g}K.npz"), fr)
        gids = _md_pairs(a, ce)
        cfgs = snapshot_configs(ce, md, fr, gids, n_per_pair=a.al_pool)
        lv = a.level or next(iter(proj.meta["levels"]))
        seeds = [x for x in (a.al_seed or "").replace(",", " ").split() if x]
        for sname in seeds:
            if sname not in proj.meta.get("scans", {}):
                raise SystemExit(f"种子扫描 {sname} 不在项目中 (已有: {', '.join(proj.meta.get('scans', {})) or '无'})")
        info = run.start(ce, cfgs, lv, T, init=a.al_init, batch=a.al_batch, target=a.al_target,
                         info={"supercell": list(sc), "n_frames": int(len(fr["c"]))}, seeds=seeds, abs_floor=a.al_abs,
                         delta=a.al_delta != "off", goal="mu" if a.al_goal == "mu" else "sigma",
                         mu_target=a.al_mu_target, hbar_tau=a.htau / 1000.0)
        print(f"主动学习 {a.name}：{len(gids)} 个近邻对，候选池每对 {a.al_pool} 个 MD 构型；"
              + ("Δ-学习 (EHT 基线 + GP)" if run.delta else "GP (常数均值)")
              + (f"；以迁移率为目标 (μ 的 95% 区间半宽 ≤ {100 * a.al_mu_target:g}%)" if run.goal == "mu" else ""))
        if "seeded" in info:
            print("种子数据 (已算完的构型): " + ", ".join(f"G{g} {n} 个" for g, n in sorted(info["seeded"].items())) + "\n")
            print(al_report(info["rows"]))
            if not info["scan"]:
                print("\n种子数据已足够，全部近邻对已收敛。用这些曲面计算 μ(T)：")
                print(f"  python -m mobility_tool ceds md --cif {a.cif} --project {a.project} --al-surfaces {a.name}")
                return
        print(f"第 1 轮 {info['n']} 个 ADF 任务 → 扫描 {info['scan']}，批处理 {', '.join(info['batches'])}")
        print("在 ADFjobs 中运行这些批处理；算完后再次运行同一条命令，软件会拟合并给出下一轮 (或报告收敛)")
        return
    if a.al_delta in ("on", "off") and run.delta != (a.al_delta == "on"):
        run.set_delta(a.al_delta == "on")
        print(f"已{'开启' if run.delta else '关闭'} Δ-学习 (EHT 基线)；已算的 DFT 数据全部保留，从这一轮起按新曲面拟合与选点\n")
    if a.al_goal in ("sigma", "mu") and (run.goal != a.al_goal or (a.al_goal == "mu" and a.al_mu_target_set)):
        run.set_goal(a.al_goal, a.al_mu_target if a.al_goal == "mu" else None)
        print(f"选点与停止判据改为{'迁移率 μ 的后验区间' if run.goal == 'mu' else '各近邻对 σ_V 的后验区间'}；"
              "已算的 DFT 数据全部保留\n")
    if a.al_calib in ("fixed", "loo") and run.calib != a.al_calib:
        run.set_calib(a.al_calib)
        print("区间校准改为" + ("每个近邻对的留一法残差 (c = 标准化留一残差的均方根，≥ 1)" if a.al_calib == "loo"
                                else f"固定因子 c = {CI_INFLATE_DELTA:g}") + "；已算的 DFT 数据全部保留\n")
    if getattr(a, "al_calib_report", False):
        import json as _json
        from .alsurface import calib_report
        surf = run.surfaces(ce)
        _, thermal = run.pool()
        cr, txt = calib_report(surf, thermal)
        print("\n区间校准检查 (留一法：每次去掉一个 DFT 构型及其对称像)：")
        print(txt)
        fn = str(run.dir / f"{a.name}_calib_report.json")
        with open(fn, "w", encoding="utf-8") as fh:
            _json.dump(cr, fh, indent=1, ensure_ascii=False)
        print(f"已写出 {fn}")
        print("只做了报告：没有写出新一轮，也没有改变主动学习的状态")
        return
    rebuilt = run.rebuild_missing()
    if rebuilt:
        print(f"[提示] 扫描目录 {', '.join(rebuilt)} 不存在 (被删除？)：已按保存的选点重新写出 (构型与编号不变)。"
              "运行其中的批处理，算完后再运行本命令\n")
    try:
        rows, new = run.step(ce, force=a.force)
    except ValueError as e:
        raise SystemExit(str(e))
    if run.delta:
        chk = run.baseline_check(run.records()[0])
        if chk:
            print("EHT 基线与已算 DFT 的相关性：" + "；".join(
                f"G{g} " + "/".join(f"{v['r']:.2f}" for v in cc.values()) for g, cc in sorted(chk.items())))
            bad = [g for g, cc in chk.items() if any(v["r"] < 0.8 for v in cc.values())]
            if bad:
                print(f"[提示] {', '.join(f'G{g}' for g in bad)} 的 EHT 基线相关性低 (r < 0.8)，Δ-学习的优势会变小；"
                      "曲面仍然正确 (GP 学习全部差值)，必要时可用 --al-delta off 对比")
    crit = (f"μ 的 95% 区间半宽 ≤ {100 * run.state.get('mu_target', 0.01):g}%" if run.goal == "mu"
            else f"σ_V 的 95% 区间半宽 ≤ {100 * run.state['target']:.0f}%")
    print(f"主动学习 {a.name}：第 {len(run.state['rounds']) - (1 if new else 0)} 轮结果 (收敛判据：{crit})\n")
    print(al_report(rows))
    if getattr(run, "last_mu", None):
        lm = run.last_mu
        print(f"\nTLT 迁移率 ({run.state['T']:g} K, ħ/τ = {1000 * run.state.get('hbar_tau', 0.005):g} meV，线性化后验)：" + "；".join(
            f"{'空穴' if c == 'hole' else '电子'} {lm['mu'][c]:.3g} cm²/Vs ± {100 * lm['half'][c]:.1f}%" for c in lm["mu"]))
    if new:
        print(f"\n下一轮：{new['n']} 个 ADF 任务 → 扫描 {new['scan']}，批处理 {', '.join(new['batches'])}")
    else:
        n = sum(r["n"] for r in rows)
        print(f"\n全部近邻对已收敛，共 {n} 个 DFT 计算。")
        if a.al_mu_draws > 0:
            print(f"耦合曲面不确定度 → TLT 迁移率 ({run.state['T']:g} K, ħ/τ = {a.htau:g} meV, "
                  f"{a.al_mu_draws} 次后验抽样) …", flush=True)
            ci = run.mu_ci(ce, hbar_tau=a.htau / 1000.0, n_draws=a.al_mu_draws)
            for c, v in ci.items():
                print(f"  {'空穴' if c == 'hole' else '电子'} μ = {v['mu']:.3g} cm²/Vs  (95% 区间 {v['lo']:.3g} – {v['hi']:.3g}，"
                      f"仅含耦合曲面误差)")
        print("用这些曲面计算 μ(T)：")
        print(f"  python -m mobility_tool ceds md --cif {a.cif} --project {a.project} --al-surfaces {a.name}")
    if a.al_budget:
        from .alsurface import budget_report, error_budget
        sE = {}
        for c, x in zip(("hole", "electron"), (a.al_budget_sigma_e or "").replace(",", " ").split()):
            sE[c] = float(x) / 1000.0
        surf = run.surfaces(ce)
        _, thermal = run.pool()
        print(f"\n误差预算 ({run.state['T']:g} K, ħ/τ = {a.htau:g} meV，每项只改变一个因素) …", flush=True)
        b = error_budget(ce, surf, {g: thermal[g] for g in surf}, run.state["T"], a.htau / 1000.0, sigma_E=sE or None,
                         n_boot=a.al_budget, n_seeds=8)
        print(budget_report(b))
        print("± 为 95% 区间相对半宽 (数值误差)；× 为改用该参数 / 模型后 μ 的比值。\n"
              "热运动样本一项对应主动学习的热样本数；ceds md --al-surfaces 用整段轨迹，这一项随样本数约按 1/√N 减小，"
              "TLT 无序实现一项随实现数按 1/√n 减小")


def _ceds_validate(a, ce, proj, surf):
    """ceds validate：MD 快照的直接计算结果与耦合曲面逐点比较。"""
    import os
    from .rigidmd import export_validation_csv, plot_validation, snapshot_validation, validation_table
    if proj is None:
        raise SystemExit("ceds validate 需要 --project 与 --name (MD 快照扫描)")
    recs = [r for r in proj.collect(a.name, write=True) if r.get("state") == "done" and not r.get("rep")]
    if not recs:
        raise SystemExit(f"{a.name} 中还没有算完的构型")
    val = snapshot_validation(ce, recs, surf)
    if not val:
        raise SystemExit("没有同时具有快照数据和耦合曲面的二聚体")
    print(f"MD 快照检验 {a.name}：{len(recs)} 个构型\n")
    print(validation_table(val))
    if a.out:
        export_validation_csv(val, a.out)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
        plot_validation(axes, val)
        fig.tight_layout()
        png = os.path.splitext(a.out)[0] + ".png"
        fig.savefig(png, dpi=300)
        print("已写出", a.out, "和", png)


def _md_cost_note(ce, surf, md, a):
    """Before the MD: how many EHT-baseline evaluations the coupling statistics will need per temperature, and a time
    estimate from three timed evaluations (Δ-GP surfaces only; fixed scans and plain GP surfaces are fast)."""
    import time
    from . import alsurface
    try:
        n_fr = int(round(a.md_prod * 1000 / a.md_dt)) // max(int(a.md_every), 1)
        ncell = len(md.cells)
        base = []
        for p in ce.pairs:
            sf = surf(p["gid"], p["center"]) if callable(surf) else surf.get(p["gid"])
            b = getattr(sf, "baseline", None)
            if b is not None and hasattr(b, "_one"):
                base.append((b, p["q0"]))
        if not base:
            return
        b, q0 = base[0]
        t0 = time.time()
        for k in range(3):
            b._one(np.asarray(q0, float) + 1e-3 * (k + 1))
        dt = (time.time() - t0) / 3
        n = len(base) * n_fr * ncell
        jobs = max(int(alsurface.EHT_JOBS or 1), 1)
        print(f"  每次 MD 后的耦合统计：约 {n} 次 EHT 计算 ({len(base)} 个二聚体 × {n_fr} 帧 × {ncell} 个晶胞)，"
              f"每次 {dt * 1000:.0f} ms，{jobs} 个进程约需 {n * dt / jobs / 60:.0f} 分钟 (弱耦合对会跳过，此为上限)")
    except Exception:                                   # the estimate must never stop a run
        return


def _ceds_md(a, ce, surf, temps, mono):
    """ceds md：全晶体刚体 MD → σ_V(T)、τ_c → TLT μ(T)。"""
    import os
    import time
    from .ceds import summary_table
    from .rigidmd import (DEFAULTS, RigidCrystalMD, export_md_csv, md_summary, md_temperature_series,
                          plot_md_mu_T)
    sc = tuple(int(x) for x in a.supercell.replace(",", " ").split())
    md = RigidCrystalMD(ce.crys, ce.frames, mono or (ce.sym, ce.Y), sc, rc=a.md_rc, r_sw=a.md_rc - 1.0,
                                charges=_md_charges(a, mono or (ce.sym, ce.Y)))
    if a.frame_tlt:                                     # check the lattice and the bond mapping before the MD
        from . import frametlt as FT
        try:
            ax = FT.keep_axes(ce, FT.crystal_v0(ce, surf))
            if min(sc[k] for k in ax) < FT.MIN_CELLS:
                raise ValueError(f"--frame-tlt 以 MD 超胞作为 TLT 晶格：--supercell 沿 {' 和 '.join('abc'[k] for k in ax)} "
                                 f"至少需要 {FT.MIN_CELLS} 个晶胞 (建议 12–16)，当前为 {a.supercell}")
            nb = len(FT.bond_types(ce, surf, ax))
        except ValueError as e:
            raise SystemExit(str(e))
        print(f"Frame-resolved TLT: transport plane {''.join('abc'[k] for k in ax)}, {nb} bonds per cell, "
              f"{a.frame_tlt} frames per temperature")
    n_steps = (a.md_equil + a.md_prod) * 1000 / a.md_dt
    print(f"全晶体刚体 MD：超胞 {'×'.join(map(str, sc))} = {md.N} 个分子，{md.n_pairs} 个原子对；{md.ff_note}")
    print(f"  晶格能 (实验结构) {md.lattice_energy():.2f} kcal/mol 每分子；每个温度 {a.md_equil:g} ps 平衡 + "
          f"{a.md_prod:g} ps 采样 (Δt = {a.md_dt:g} fs，{n_steps:.0f} 步)")
    t0 = time.time()
    hts = [float(x) / 1000 for x in a.htau_list.replace(",", " ").split()] if a.htau_list else []
    from .ceds import cif_ueq
    try:
        ueq_ref = a.ueq or cif_ueq(a.cif)
    except Exception:
        ueq_ref = None
    _md_cost_note(ce, surf, md, a)
    traj_dir = a.save_traj or (None if a.no_save_traj else os.path.join(
        os.getcwd(), f"md_traj_{os.path.splitext(os.path.basename(a.cif or 'crystal'))[0]}_{'x'.join(map(str, sc))}"))
    if traj_dir:
        print(f"  trajectories: {traj_dir} (reused when the same run is started again; --rerun-md to force a new MD)")
    rows, _ = md_temperature_series(ce, surf, temps, md=md, equil_ps=a.md_equil, prod_ps=a.md_prod, dt=a.md_dt,
                                    every=a.md_every, hbar_tau=a.htau / 1000, hbar_taus=hts, recenter=not a.raw_geometry,
                                    save_dir=traj_dir, frame_tlt=a.frame_tlt, ueq_ref=ueq_ref,
                                    frame_stride=a.frame_stride or None, reuse_traj=not a.rerun_md,
                                    on_progress=lambda m: print(f"  {m}  (已用 {time.time() - t0:.0f} s)", flush=True))
    for c in ("hole", "electron"):
        print(f"\n[{c}] T = {temps[-1]:g} K (MD)")
        print(summary_table(ce, rows[-1]["res"], c))
    print(f"\nMD + TLT (ħ/τ = {a.htau:g} meV){'，力场平均结构' if a.raw_geometry else ''}")
    print(md_summary(rows))
    from .rigidmd import series_check, stability_text
    series_check(rows)
    print("\n晶格稳定性" + ("" if ueq_ref else " (CIF 中没有 U_iso / U_aniso，只检查 U_eq 增长与分子偏移)") + "：")
    for r in rows:
        print("  " + stability_text(r["T"], r["stability"]))
    bad = [r["T"] for r in rows if not r["stability"]["stable"]]
    if bad:
        print(f"[警告] {', '.join(f'{t:g}' for t in bad)} K 的结果来自失稳的晶格，不能用于 μ(T)")
    if hts:
        print("各 ħ/τ 的 μ: " + "; ".join(f"T={r['T']:g}: " + ", ".join(f"{k} {v:.3g}" for k, v in r["mu_tau"].items())
                                          for r in rows))
    if a.frame_tlt:
        from . import frametlt as FT
        for r in rows:
            print(f"\nT = {r['T']:g} K")
            print(FT.report(r["frame_tlt"]))
        if a.out:
            fp = os.path.splitext(a.out)[0] + "_frames.csv"
            FT.export_csv([(r["T"], r["frame_tlt"]) for r in rows], fp)
            print("Frame-resolved TLT written to", fp)
    if a.out:
        export_md_csv(ce, rows, a.out)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        plot_md_mu_T(ax, rows)
        ax.set_title("CEDS: rigid-body crystal MD + TLT", fontsize=9)
        fig.tight_layout()
        png = os.path.splitext(a.out)[0] + ".png"
        fig.savefig(png, dpi=300)
        print("已写出", a.out, "和", png)
    print(f"用时 {time.time() - t0:.0f} s；默认设置 {DEFAULTS}")


def _ceds_phonons(a, ce, surf, temps, mono):
    """ceds phonons：刚体晶格动力学 → 各声子支对 σ_V 与 TLT 迁移率的贡献 (杀手声子)。"""
    import os
    import time
    from . import phonons as PH
    from .rigidmd import RigidCrystalMD
    sc = tuple(int(x) for x in a.supercell.replace(",", " ").split())
    md = RigidCrystalMD(ce.crys, ce.frames, mono or (ce.sym, ce.Y), sc, rc=a.md_rc, r_sw=a.md_rc - 1.0,
                        charges=_md_charges(a, mono or (ce.sym, ce.Y)))
    t0 = time.time()
    if not a.ph_external:
        print(f"刚体晶格动力学：超胞 {'×'.join(map(str, sc))} = {md.N} 个分子 (q 网格 {len(md.cells)} 点)；{md.ff_note}")
    if a.ph_external:
        ext = PH.read_phonopy_yaml(a.ph_external, a.ph_structure or None)
        lat = PH.external_lattice(md, ce, ext, length_unit=a.ph_length_unit, phase=a.ph_phase)
        inf = lat.info
        low = lat.mode["freq_cm"] < 200.0
        print(f"外部声子 {Path(a.ph_external).name}：{inf['n_q']} 个 q 点 × {inf['n_branch']} 支；长度单位 {inf['length_unit']}；"
              f"与 CIF 的原子对应最大偏差 {inf['map_dev_A']:.2f} Å，晶格矢量长度比 "
              f"{', '.join(f'{x:.3f}' for x in inf['lattice_ratio'])}")
        print(f"  200 cm⁻¹ 以下的模平均刚体成分 {inf['rigid_fraction'][low].mean():.0%} (分子内形变对耦合的影响不计)")
        ig = int(np.argmin(np.linalg.norm(lat.mode["q"] - np.round(lat.mode["q"]), axis=1)))
        fg = np.sort(lat.mode["freq_cm"][ig])
        print(f"  Γ 点最低频率 (cm⁻¹): {', '.join(f'{x:.1f}' for x in fg[3:15])} …")
        if (lat.mode["freq_cm"] < -1.0).any():
            print(f"[提示] 外部声子有 {(lat.mode['freq_cm'] < -1.0).sum()} 个虚频模，不计入")
    else:
        lat = PH.RigidLattice(md)
        r = lat.relax()
        print(f"  固定晶胞刚体弛豫：E {r['E0']:.2f} → {r['E']:.2f} kcal/mol 每分子，质心最大位移 {r['shift_max_A']:.3f} Å，"
              f"最大转角 {r['rot_max_deg']:.2f}°")
        lat.force_constants()
        lat.modes()
        print(f"  Γ 点频率 (cm⁻¹): {', '.join(f'{x:.1f}' for x in lat.mode['freq_cm'][0][3:])}  (已用 {time.time() - t0:.0f} s)")
    jac = PH.pair_jacobians(lat, ce)
    grad = PH.surface_gradient(ce, surf)
    gamma_only = PH.gamma_subset(lat) if a.ph_gamma else None
    base = os.path.splitext(a.out)[0] if a.out else ""
    for T in temps:
        mc = PH.mode_couplings(lat, ce, surf, T=T, quantum=a.ph_quantum, jac=jac, grad=grad)
        mob = None
        if not a.sigma_only:
            mob = {c: PH.mobility_contributions(ce, mc, c, hbar_tau=None if a.ph_htau_auto else a.htau / 1000,
                                                n_real=a.ph_real) for c in ("hole", "electron")}
        print(f"\n========== T = {T:g} K ==========")
        print(PH.summary_text(mc, mob))
        if gamma_only is not None:
            mg = PH.mode_couplings(gamma_only, ce, surf, T=T, quantum=a.ph_quantum, jac=jac, grad=grad)
            print("\n-- 只用 Γ 点 (常见的单胞做法，对照) --")
            for c in ("hole", "electron"):
                print(f"[{c}] " + ", ".join(f"G{g} σ = {d[c]['sigma']:.2f}" for g, d in sorted(mg["pairs"].items())
                                          if d[c]["sigma"] > 0.5))
                cs = PH.character_shares(mg, c)
                print("    σ² 运动分布: " + ", ".join(f"{PH.CHAR_LABEL[k]} {v:.0%}" for k, v in cs.items()))
        if base:
            tag = f"_{T:g}K" if len(temps) > 1 else ""
            PH.export_csv(mc, f"{base}{tag}.csv", mob)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for c in ("hole", "electron"):
                fig, axs = plt.subplots(1, 2 if mob else 1, figsize=(9 if mob else 4.5, 3.4), squeeze=False)
                PH.plot_spectrum(axs[0], mc, mob[c] if mob else None, carrier=c)
                axs[0, 0].set_title(f"{c}, {T:g} K", fontsize=9)
                fig.tight_layout()
                fig.savefig(f"{base}{tag}_{c}.png", dpi=300)
                plt.close(fig)
            print(f"已写出 {base}{tag}.csv 与 {base}{tag}_hole/electron.png")
    print(f"用时 {time.time() - t0:.0f} s")


def _fill_missing_help(parser, cmd=""):
    from ._cli_i18n import OPTION_HELP
    for act in parser._actions:
        if isinstance(act, argparse._SubParsersAction):
            if cmd and not act.help:
                act.help = OPTION_HELP.get(act.dest)
            for name, sp in act.choices.items():
                _fill_missing_help(sp, name)
        elif not act.help and not isinstance(act, argparse._HelpAction):
            key = act.option_strings[-1] if act.option_strings else act.dest
            act.help = OPTION_HELP.get(f"{cmd} {key}") or OPTION_HELP.get(key)


def _translate_parser(parser):
    from .i18n import tr_cli
    if parser.description:
        parser.description = tr_cli(parser.description)
    for act in parser._actions:
        if act.help and act.help is not argparse.SUPPRESS:
            act.help = tr_cli(act.help)
        if isinstance(act, argparse._VersionAction) and act.version:
            act.version = tr_cli(act.version)
        if isinstance(act, argparse._SubParsersAction):
            for ca in act._choices_actions:
                if ca.help:
                    ca.help = tr_cli(ca.help)
            for sp in act.choices.values():
                _translate_parser(sp)


def main(argv=None):
    from . import __affiliation__, __author__, __version__
    from . import i18n
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "--lang":          # python -m mobility_tool --lang zh <subcommand> ...
        os.environ["MOBILIS_LANG"] = argv[1]
        argv = argv[2:]
    i18n.CLI_LANG = i18n.cli_lang_from_env()
    if not isinstance(sys.stdout, i18n.TranslatingStream):
        sys.stdout = i18n.TranslatingStream(sys.stdout)
        sys.stderr = i18n.TranslatingStream(sys.stderr)
    ap = argparse.ArgumentParser(prog="mobility_tool", description="Mobilis 迁跃 — 有机半导体电荷输运与光物理计算平台")
    ap.add_argument("--version", action="version",
                    version=f"Mobilis {__version__}  (开发者: {__author__}, {__affiliation__})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("reorg-gen", help="生成四点法重组能高斯输入")
    p.add_argument("geom", help="结构文件 (xyz / gjf / log)")
    p.add_argument("-o", "--outdir", default="reorg")
    p.add_argument("--name")
    p.add_argument("--carrier", choices=["hole", "electron", "both"], default="both")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--mult", type=int, default=1)
    p.add_argument("--ion-mult", type=int, default=None, help="离子态多重度 (默认 2)")
    p.add_argument("--no-freq", action="store_true", help="优化时不做频率计算")
    p.add_argument("--sp-method", help="单点能量用的泛函 (默认同优化)")
    p.add_argument("--sp-basis", help="单点能量用的基组, 如 6-311G(2d,2p)")
    p.add_argument("--solvent", default="", help="PCM 溶剂, 如 dichloromethane")
    p.add_argument("--ecp-basis", default="SDD", help="重原子赝势基组 (Pople 基组 + Ir 等时自动 GenECP)")
    _common_gauss(p)
    p.set_defaults(func=cmd_reorg_gen)

    p = sub.add_parser("reorg-analyze", help="解析高斯输出得到重组能")
    p.add_argument("dir", help="reorg-gen 生成的目录 (含 log 文件)")
    p.add_argument("--json", help="把结果保存为 json")
    p.set_defaults(func=cmd_reorg_analyze)

    p = sub.add_parser("ti-gen", help="生成转移积分高斯输入 (A, B, AB)")
    p.add_argument("--dimer", help="二聚体结构文件, 前 na 个原子为单体 A")
    p.add_argument("--na", type=int, help="单体 A 的原子数")
    p.add_argument("--mono-a", help="单体 A 结构文件 (与 --mono-b 一起使用)")
    p.add_argument("--mono-b", help="单体 B 结构文件")
    p.add_argument("-o", "--outdir", default="ti")
    p.add_argument("--name", default="dimer")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--mult", type=int, default=1)
    _common_gauss(p)
    p.set_defaults(func=cmd_ti_gen)

    p = sub.add_parser("ti-calc", help="由 fchk 计算转移积分")
    p.add_argument("--prefix", help="例如 ti/dimer → 读取 ti/dimer_A/B/AB.fchk")
    p.add_argument("-a", help="单体 A 的 fchk")
    p.add_argument("-b", help="单体 B 的 fchk")
    p.add_argument("-ab", help="二聚体的 fchk")
    p.add_argument("--orbitals", default="HOMO,LUMO", help="如 HOMO-1,HOMO,LUMO,LUMO+1")
    p.add_argument("--json")
    p.set_defaults(func=cmd_ti_calc)

    p = sub.add_parser("mobility", help="Marcus 理论计算迁移率")
    p.add_argument("--config", help="json 配置文件 (见 examples/mobility_config.json)")
    p.add_argument("--lam", type=float, help="重组能 / eV")
    p.add_argument("--path", action="append", help="跳跃路径 'r(Å),J(meV)[,数目]', 可重复")
    p.add_argument("--temp", type=float, default=298.15)
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--dimers", help="crystal 子命令导出的目录 (自动读取各组 J)")
    p.add_argument("--program", choices=["gaussian", "adf"], default="gaussian")
    p.add_argument("--carrier", choices=["hole", "electron"], default="hole")
    p.add_argument("--cif", help="--dimers：从这个 CIF 取晶体点群对称化张量 (旧的 dimers.json 没有记录点群时)")
    p.add_argument("--json")
    p.set_defaults(func=cmd_mobility)

    p = sub.add_parser("reorg-decomp", help="重组能分解: 振动模式(黄昆因子)/内坐标/原子")
    p.add_argument("neutral", help="中性分子 freq 计算的 fchk")
    p.add_argument("ion", help="离子 freq 计算的 fchk")
    p.add_argument("--bond-scale", type=float, default=1.2)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--csv", help="导出 CSV 的目录")
    p.set_defaults(func=cmd_decomp)

    p = sub.add_parser("adf-gen", help="生成 ADF 片段法转移积分输入")
    p.add_argument("--dimer")
    p.add_argument("--na", type=int)
    p.add_argument("--mono-a")
    p.add_argument("--mono-b")
    p.add_argument("-o", "--outdir", default="adf_ti")
    p.add_argument("--name", default="dimer")
    p.add_argument("--functional", default="PW91")
    p.add_argument("--basis", default="TZP")
    p.add_argument("--style", choices=["ams", "classic"], default="ams", help="ams: AMS2020+; classic: ADF2019-")
    p.add_argument("--quality", default="Good")
    p.set_defaults(func=cmd_adf_gen)

    p = sub.add_parser("adf-read", help="读取 ADF 转移积分 (adf.rkf / TAPE21 / dmpkf 文本)")
    p.add_argument("file")
    p.set_defaults(func=cmd_adf_read)

    p = sub.add_parser("crystal", help="从 CIF 搜索近邻二聚体并批量生成输入")
    p.add_argument("cif")
    p.add_argument("--ref", type=int, default=1, help="参考分子序号 (从 1 开始)")
    p.add_argument("--cutoff", type=float, default=4.0)
    p.add_argument("--centroid", action="store_true", help="用质心距离判据 (默认最短原子间距)")
    p.add_argument("-o", "--outdir")
    p.add_argument("--program", choices=["gaussian", "adf"], default="gaussian")
    p.add_argument("--method", default="B3LYP", help="Gaussian 方法或 ADF 泛函")
    p.add_argument("--basis", default="6-31G(d)")
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--mem", default="8GB")
    p.set_defaults(func=cmd_crystal)

    p = sub.add_parser("ipea", help="计算 IP/EA/HEP/EEP/重组能")
    p.add_argument("--dir", help="reorg-gen 生成并已算完的目录")
    p.add_argument("--energies", type=float, nargs=7, metavar="E",
                   help="手动给 7 个能量(Hartree): E0(N) E+(C) E-(A) E+(N) E-(N) E0(C) E0(A)")
    p.add_argument("--orbitals", help="读取 HOMO/LUMO 的 fchk 或 log")
    p.set_defaults(func=cmd_ipea)

    p = sub.add_parser("orbitals", help="从 fchk/log 读取 HOMO/LUMO")
    p.add_argument("file")
    p.set_defaults(func=cmd_orbitals)

    p = sub.add_parser("fcht-gen", help="在重组能目录生成 Gaussian FCHT (黄昆因子/Duschinsky) 输入")
    p.add_argument("dir")
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--mem", default="8GB")
    p.add_argument("--clear-low-freq", type=float, help="RedDim=ClearLowFreq 的值, 如 100")
    p.set_defaults(func=cmd_fcht_gen)

    p = sub.add_parser("fcht-read", help="解析 FCHT 输出: 重组能分解/黄昆因子/Duschinsky 矩阵")
    p.add_argument("log")
    p.add_argument("--freq-block", type=int, default=0, help="与黄昆因子配对的频率组 (默认 0)")
    p.add_argument("-o", "--out", help="导出 csv / Origin 数据和插图的目录")
    p.set_defaults(func=cmd_fcht_read)

    p = sub.add_parser("mobility2d", help="二维角分辨迁移率: 读取 CSV 表 (name,d,theta,Ve,Vh)")
    p.add_argument("table")
    p.add_argument("--lam-e", type=float)
    p.add_argument("--lam-h", type=float)
    p.add_argument("--temp", type=float, default=300)
    p.add_argument("--scale", type=float, default=1.0, help="表中 V 的单位换算到 eV 的除数 (meV 用 1000)")
    p.add_argument("--plot", help="保存极坐标图的文件名")
    p.add_argument("--model", choices=["tensor", "wen"], default="tensor",
                   help="tensor: 修正式 (无规行走张量的投影，默认)；wen: Wen 原式 (含跳跃概率 P，系统性偏低，仅供与文献对比)")
    p.add_argument("--label", default="")
    p.set_defaults(func=cmd_mobility2d)

    p = sub.add_parser("mobility3d", help="三维各向异性迁移率: 读取 CSV 表 (name,x,y,z,Ve,Vh,count)")
    p.add_argument("table")
    p.add_argument("--lam-e", type=float)
    p.add_argument("--lam-h", type=float)
    p.add_argument("--temp", type=float, default=300)
    p.add_argument("--scale", type=float, default=1.0, help="表中 V 的单位换算到 eV 的除数 (meV 用 1000)")
    p.add_argument("--model", choices=["wen", "tensor"], default="tensor",
                   help="wen: Wen 公式的三维推广 (与二维图一致)；tensor: 无规行走迁移率张量")
    p.add_argument("--cif", help="晶体 (画出晶轴 a、b、c，并给出主轴的晶体学方向)")
    p.add_argument("--plane", default="xy", help="画出截面的传输平面 (xy / xz / yz)")
    p.add_argument("--plot", help="保存三维图的文件名 (同时写出同名 .html)")
    p.add_argument("--label", default="")
    p.set_defaults(func=cmd_mobility3d)

    p = sub.add_parser("excited-gen", help="生成 S0/S1/T1 优化与 TD 单点输入")
    p.add_argument("geom")
    p.add_argument("-o", "--outdir", default="excited")
    p.add_argument("--name")
    p.add_argument("--method", default="B3LYP")
    p.add_argument("--basis", default="6-311G(d,p)")
    p.add_argument("--sp-basis", default="def2-TZVP")
    p.add_argument("--solvent", default="")
    p.add_argument("--nstates", type=int, default=10)
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--mult", type=int, default=1)
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--mem", default="8GB")
    p.set_defaults(func=cmd_excited_gen)

    p = sub.add_parser("excited-read", help="读取吸收/荧光/磷光、Stokes 位移、RMSD")
    p.add_argument("dir")
    p.add_argument("name")
    p.set_defaults(func=cmd_excited_read)

    p = sub.add_parser("rmsd", help="两个结构叠合后的 RMSD")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--heavy", action="store_true")
    p.set_defaults(func=cmd_rmsd)

    p = sub.add_parser("orbcomp", help="轨道片段组成 (SCPA/Mulliken)")
    p.add_argument("fchk")
    p.add_argument("--frag", action="append", help='如 "Ir:1" 或 "L1:2-30"，可重复；不给则自动划分')
    p.add_argument("--orbitals", default="HOMO-1,HOMO,LUMO,LUMO+1")
    p.add_argument("--method", choices=["scpa", "mulliken"], default="scpa")
    p.set_defaults(func=cmd_orbcomp)

    p = sub.add_parser("run", help="在本机依次运行目录中的高斯任务并 formchk")
    p.add_argument("dir")
    p.add_argument("--gaussian", help="g16.exe 路径或所在目录")
    p.add_argument("--scratch")
    p.add_argument("--pattern", default="*.gjf")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("eht-ti", help="软件内 EHT 计算转移积分 (无需外部程序)")
    p.add_argument("--dimer")
    p.add_argument("--na", type=int)
    p.add_argument("--mono-a")
    p.add_argument("--mono-b")
    p.add_argument("--orbitals", default="HOMO,LUMO")
    p.add_argument("--pi", action="store_true", help="用最高占据 / 最低空的 π 轨道代替 HOMO/LUMO (氮杂芳烃等孤对 σ 轨道落在 HOMO 附近时)")
    p.set_defaults(func=cmd_eht_ti)

    p = sub.add_parser("compare-ti", help="对比 EHT / Gaussian / ADF 三种方法的转移积分")
    p.add_argument("dir", help="含 dimers.json 的二聚体目录")
    p.set_defaults(func=cmd_compare_ti)

    p = sub.add_parser("kmc", help="CIF → EHT 转移积分 → 晶体近邻图 KMC (VOTCA 式)")
    p.add_argument("cif")
    p.add_argument("--lam", type=float, required=True, help="内重组能 / eV")
    p.add_argument("--carrier", choices=["hole", "electron"], default="hole")
    p.add_argument("--rate", choices=["marcus", "jortner", "fgr"], default="marcus")
    p.add_argument("--modes", nargs="+", default=[],
                   help="--rate fgr: mode files of reorg-decomp (<prefix>_neutral_modes.csv and <prefix>_ion_modes.csv)")
    p.add_argument("--classical-below", type=float, default=0.0,
                   help="--rate fgr: treat modes below this frequency / cm⁻¹ classically (default 0: all quantum)")
    p.add_argument("--lam-out", type=float, default=0.0)
    p.add_argument("--hw", type=float, default=0.2)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--supercell", type=int, default=4)
    p.add_argument("--field", default="0 0 0", help="V/cm, 如 '1e4 0 0'")
    p.add_argument("--temp", type=float, default=300)
    p.add_argument("--cutoff", type=float, default=4.0)
    p.add_argument("--walkers", type=int, default=2000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--pi", action="store_true", help="用最高占据 / 最低空的 π 轨道代替 HOMO/LUMO (氮杂芳烃等孤对 σ 轨道落在 HOMO 附近时)")
    p.set_defaults(func=cmd_kmc)

    p = sub.add_parser("tunneling", help="Hopping with quantum nuclear tunneling over all intramolecular modes "
                       "(Fermi golden rule): rate against Marcus and MLJ versus temperature, and with --cif the "
                       "mobility tensor and KMC")
    p.add_argument("--modes", nargs="+", required=True,
                   help="mode files of reorg-decomp: <prefix>_neutral_modes.csv and <prefix>_ion_modes.csv")
    p.add_argument("--temps", default="100,150,200,250,300", help="temperatures / K, e.g. 100,200,300")
    p.add_argument("--lam-out", type=float, default=0.0, help="classical (outer-sphere) reorganization energy / eV")
    p.add_argument("--classical-below", type=float, default=0.0,
                   help="treat modes below this frequency / cm⁻¹ classically (default 0: all quantum)")
    p.add_argument("--cif", default="", help="crystal: mobility tensor with EHT couplings (FGR and Marcus, same λ)")
    p.add_argument("--carrier", choices=["hole", "electron"], default="hole")
    p.add_argument("--cutoff", type=float, default=4.0)
    p.add_argument("--kmc", action="store_true", help="--cif: also run KMC with the FGR rates (checks the tensor)")
    p.add_argument("--sigma", type=float, default=0.0, help="--kmc: Gaussian site-energy disorder / eV")
    p.add_argument("--pi", action="store_true", help="use the highest occupied / lowest unoccupied π orbitals")
    p.add_argument("--out", default="", help="write the table to this csv file")
    p.set_defaults(func=cmd_tunneling)

    p = sub.add_parser("tvcf", help="Excited-state decay rates from thermal vibration correlation functions "
                       "(harmonic, frequency changes and Duschinsky rotation): radiative rate and emission spectrum "
                       "(FC and Herzberg–Teller), internal conversion (non-adiabatic coupling) and intersystem "
                       "crossing (spin–orbit coupling)")
    p.add_argument("--initial", default="", help="freq fchk of the initial state (e.g. S1 or T1)")
    p.add_argument("--final", default="", help="freq fchk of the final state (e.g. S0, or T1 for ISC)")
    p.add_argument("--modes", default="", help="instead of fchk files: table of frequencies and Huang–Rhys factors "
                   "(displaced oscillators, same frequencies in both states)")
    p.add_argument("--de", type=float, help="adiabatic energy gap E_initial − E_final / eV (default: from the fchk "
                   "total energies)")
    p.add_argument("--temps", default="300", help="temperatures / K, e.g. 77,300")
    p.add_argument("--mu", default="", help="transition dipole (a.u., e·a0): |μ| or μx,μy,μz → radiative rate")
    p.add_argument("--mu-debye", action="store_true", help="--mu (and --mu-s1 / --mu-t1) are in debye")
    p.add_argument("--fosc", type=float, help="oscillator strength instead of --mu (with --e-vert)")
    p.add_argument("--e-vert", type=float, help="vertical energy / eV belonging to --fosc (default: --de)")
    p.add_argument("--ht-dipole", default="", help="Herzberg–Teller: table 'mode, dμx, dμy, dμz' (a.u. per √amu·a0, "
                   "initial-state normal modes in increasing frequency)")
    p.add_argument("--nac", default="", help="non-adiabatic coupling ⟨Φ_f|∂Φ_i/∂R⟩ at the initial geometry, one row "
                   "per atom (bohr⁻¹; ORCA output or any table) → internal conversion")
    p.add_argument("--nac-modes", default="", help="non-adiabatic coupling per initial-state mode, table 'mode, "
                   "value' (a.u. per √amu·a0)")
    p.add_argument("--soc", type=float, help="spin–orbit coupling / cm⁻¹ (root of the sum of |⟨S|H_SO|T_m⟩|² over the "
                   "triplet sublevels) → intersystem crossing")
    p.add_argument("--ht-soc", default="", help="spin-vibronic (HT) part of the SOC: table 'mode, dSOC' (cm⁻¹ per "
                   "√amu·a0)")
    p.add_argument("--triplet-initial", action="store_true", help="the initial state is the triplet (T → S): "
                   "average over its three sublevels (factor 1/3)")
    p.add_argument("--n-refr", type=float, default=1.0, help="refractive index of the medium (k_r ∝ n²)")
    p.add_argument("--floor-cm", type=float, default=0.0, help="raise frequencies below this value / cm⁻¹ to it")
    p.add_argument("--damping-fs", type=float, default=500.0, help="Gaussian damping time / fs (Gaussian "
                   "broadening σ ≈ 0.93 eV·fs / τ)")
    p.add_argument("--out", default="", help="prefix of the output files (<prefix>_rates.csv, <prefix>_spectrum.csv)")
    p.set_defaults(func=cmd_tvcf)

    p = sub.add_parser("tadf", help="S1/T1 kinetics of an emitter (TADF or conventional) from three freq fchk files: "
                       "k_r, k_IC, k_ISC, k_RISC and the T1 decay by TVCF, then lifetimes and quantum yields")
    p.add_argument("--s0", required=True, help="freq fchk of S0")
    p.add_argument("--s1", required=True, help="freq fchk of S1 (TD-DFT opt + freq)")
    p.add_argument("--t1", required=True, help="freq fchk of T1")
    p.add_argument("--e-s1", type=float, help="adiabatic S1 energy above S0 / eV (default: from the fchk files)")
    p.add_argument("--e-t1", type=float, help="adiabatic T1 energy above S0 / eV (default: from the fchk files)")
    p.add_argument("--fosc", type=float, help="S1 → S0 oscillator strength")
    p.add_argument("--e-vert", type=float, help="vertical S1 energy / eV belonging to --fosc (default: --e-s1)")
    p.add_argument("--mu-s1", type=float, help="instead of --fosc: |μ(S1→S0)| / a.u.")
    p.add_argument("--mu-t1", type=float, help="|μ(T1→S0)| / a.u. from SOC-TDDFT → phosphorescence rate")
    p.add_argument("--mu-debye", action="store_true", help="--mu-s1 / --mu-t1 are in debye")
    p.add_argument("--soc", type=float, required=True, help="S1–T1 spin–orbit coupling / cm⁻¹")
    p.add_argument("--soc-t1s0", type=float, default=0.0, help="T1–S0 spin–orbit coupling / cm⁻¹ → non-radiative "
                   "T1 decay")
    p.add_argument("--nac", default="", help="S1–S0 non-adiabatic coupling at the S1 geometry (bohr⁻¹, one row per "
                   "atom) → internal conversion")
    p.add_argument("--temps", default="300")
    p.add_argument("--n-refr", type=float, default=1.0, help="refractive index of the medium (k_r ∝ n²)")
    p.add_argument("--floor-cm", type=float, default=0.0, help="raise frequencies below this value / cm⁻¹ to it")
    p.add_argument("--damping-fs", type=float, default=500.0, help="Gaussian damping time / fs")
    p.add_argument("--out", default="", help="write the results to this JSON file")
    p.set_defaults(func=cmd_tadf)

    p = sub.add_parser("dp-gen", help="Strained cells for deformation-potential theory (periodic DFT inputs) and the "
                       "strain table to fill in")
    p.add_argument("cif")
    p.add_argument("--strains", default="-1,-0.5,0,0.5,1", help="strains / %%, written as --strains=-1,-0.5,0,0.5,1")
    p.add_argument("--axes", default="a,b,c", help="lattice vectors to strain, e.g. a,b")
    p.add_argument("--format", choices=["cif", "poscar"], default="cif", help="file format of the strained cells")
    p.add_argument("--out", default="", help="prefix of the written files")
    p.set_defaults(func=cmd_dp_gen)

    p = sub.add_parser("dp", help="Band mobility limited by acoustic phonons (deformation-potential theory): C and "
                       "E1 from the strain table, Bardeen–Shockley, and Boltzmann transport on the tight-binding band "
                       "of the crystal")
    p.add_argument("table", help="strain table: axis, strain, energy_eV, vbm_eV, cbm_eV, volume_A3 (2D: area_A2)")
    p.add_argument("--dim", type=int, choices=[2, 3], default=3, help="3: bulk crystal (C in GPa); 2: monolayer "
                   "(C in N/m)")
    p.add_argument("--size", type=float, help="volume / Å³ (2D: area / Å²) of the unstrained cell, if not in the table")
    p.add_argument("--mass", default="", help="effective mass(es) / m_e for Bardeen–Shockley, one value or one per "
                   "axis of the table")
    p.add_argument("--cif", default="", help="crystal: tight-binding band from EHT couplings → effective masses and "
                   "Boltzmann transport")
    p.add_argument("--cutoff", type=float, default=4.0)
    p.add_argument("--pi", action="store_true", help="use the highest occupied / lowest unoccupied π orbitals")
    p.add_argument("--kgrid", type=int, default=48, help="k points per periodic direction for Boltzmann transport")
    p.add_argument("--temp", type=float, default=300.0)
    p.add_argument("--out", default="", help="write the results to this csv file")
    p.set_defaults(func=cmd_dp)

    p = sub.add_parser("project", help="Project files (*.mobilis.json) that keep all settings of a study and run it "
                       "again: new (from a template), show, check (arguments of every step, nothing is run), run")
    p.add_argument("action", choices=["new", "show", "check", "run"])
    p.add_argument("file", help="project file, e.g. pentacene.mobilis.json")
    p.add_argument("--template", choices=["crystal", "emitter", "deformation"], default="crystal",
                   help="new: crystal (transport with EHT couplings), emitter (TVCF S1/T1 kinetics) or deformation "
                   "(deformation-potential band mobility)")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="new / run: set a variable of the project, e.g. --set cif=pentacene.cif")
    p.add_argument("--only", default="", help="run: only these steps (comma-separated names)")
    p.add_argument("--start", default="", help="run: start from this step")
    p.add_argument("--keep-going", action="store_true", help="run: continue after a failed step")
    p.set_defaults(func=cmd_project)

    p = sub.add_parser("strain-scan", help="晶胞缩放 (应变) 扫描: EHT 转移积分 → 迁移率")
    p.add_argument("cif")
    p.add_argument("--axis", choices=["a", "b", "c", "all"], default="c")
    p.add_argument("--range", default="-3:3:1")
    p.add_argument("--lam-e", type=float, default=0.2)
    p.add_argument("--lam-h", type=float, default=0.2)
    p.add_argument("--temp", type=float, default=300)
    p.add_argument("--cutoff", type=float, default=4.0)
    p.set_defaults(func=cmd_strain)

    p = sub.add_parser("db", help="Coupling-surface database: store the DFT dimers and transport results of a crystal "
                                  "(add), list the records (list), merge all dimers into one table (export)")
    p.add_argument("action", choices=["add", "list", "export"])
    p.add_argument("db", help="Database folder")
    p.add_argument("key", nargs="?", help="add: record name of the crystal, e.g. rubrene_605654")
    p.add_argument("--project", help="add: scan project of the crystal (folder with project.json)")
    p.add_argument("--cif", help="add: crystal structure (cell and space group are stored; the file is copied)")
    p.add_argument("--no-cif", action="store_true", help="add: do not copy the CIF (structures that may not be redistributed)")
    p.add_argument("--ccdc", default="", help="add: CCDC / COD deposition number")
    p.add_argument("--al", help="add: name of the active-learning task (ceds al --name)")
    p.add_argument("--lambda", dest="reorg", help="add: reorganization-energy result (<name>_lambda.json)")
    p.add_argument("--md-csv", help="add: result csv of ceds md (pair statistics, mobility against temperature)")
    p.add_argument("--validation", help="add: result csv of ceds validate (surfaces against direct snapshots)")
    p.add_argument("--note", default="", help="add: free text stored in the record")
    p.add_argument("--overwrite", action="store_true", help="add: replace an existing record of the same name")
    p.add_argument("--out", help="export: output csv (default <db>/all_dimers.csv)")
    p.set_defaults(func=cmd_db)

    p = sub.add_parser("photocond", help="Photoconductivity under illumination: exciton generation, Onsager–Braun "
                                         "free-carrier yield, Langevin/trap recombination, steady Δσ or pulse decay")
    g = p.add_argument_group("transport (from ceds md or given)")
    g.add_argument("--md-csv", help="result csv of ceds md: μe, μh at --temp")
    g.add_argument("--mu-e", type=float, help="electron mobility / cm² V⁻¹ s⁻¹")
    g.add_argument("--mu-h", type=float, help="hole mobility / cm² V⁻¹ s⁻¹")
    g.add_argument("--eps", type=float, required=True, help="relative permittivity (polar: Lorentz–Lorenz)")
    g = p.add_argument_group("excitons")
    g.add_argument("--eb", type=float, help="exciton binding energy / eV")
    g.add_argument("--et", type=float, help="solid-state transport gap / eV (band edges)")
    g.add_argument("--eopt", type=float, help="optical gap / eV (E_b = E_t − E_opt)")
    g.add_argument("--tau-s", type=float, default=1000.0,
                   help="singlet-exciton lifetime / ps (fission, radiative and non-radiative decay; default 1000)")
    g = p.add_argument_group("sample and illumination")
    g.add_argument("--wavelength", type=float, required=True, help="excitation wavelength / nm")
    g.add_argument("--alpha", type=float, required=True, help="absorption coefficient at the wavelength / cm⁻¹")
    g.add_argument("--thickness", type=float, default=100.0, help="film thickness / nm")
    g.add_argument("--refl", type=float, default=0.0, help="reflectance")
    g.add_argument("--intensity", default="0.1", help="intensity / W cm⁻², one or several values")
    g.add_argument("--field", default="1e5", help="electric field / V cm⁻¹, one or several values")
    g.add_argument("--temp", type=float, default=300.0, help="temperature / K")
    g.add_argument("--fluence", type=float, help="pulse energy density / μJ cm⁻² (adds the decay of Δσ after a pulse)")
    g = p.add_argument_group("recombination")
    g.add_argument("--zeta", type=float, default=1.0, help="reduction factor of Langevin recombination")
    g.add_argument("--tau-trap", type=float, help="trap-assisted (monomolecular) lifetime / ns")
    p.add_argument("--out", help="result csv")
    p.set_defaults(func=cmd_photocond)

    p = sub.add_parser("exciton", help="Singlet-exciton transport: transition charges (INDO/S–CIS or Gaussian), TrESP "
                                       "couplings, Davydov splitting, Marcus rates, diffusion tensor and length, KMC, "
                                       "thermal fluctuations of the couplings from rigid-body MD")
    p.add_argument("--cif", required=True)
    p.add_argument("--cutoff", type=float, default=4.0, help="contact distance for the dimers / Å (default 4.0)")
    g = p.add_argument_group("transition charges")
    g.add_argument("--charges", default="indo", help="'indo' (in-code INDO/S–CIS, default) or a Gaussian log with "
                                                     "Pop=ESP Density=Transition=n")
    g.add_argument("--state", type=int, help="INDO/S state index from 0 (default: lowest state with f ≥ 0.05)")
    g.add_argument("--active", type=int, default=12, help="occupied and virtual orbitals in the CIS (default 12 each)")
    g.add_argument("--mu", type=float, help="rescale the transition charges to this transition dipole / D")
    g.add_argument("--screening", type=float, help="screening factor of the Coulomb coupling (default 1, or 1/n²)")
    g.add_argument("--n-refr", type=float, help="refractive index; screening = 1/n² unless --screening is given")
    g.add_argument("--write-gjf", help="only write a Gaussian input for the transition charges (molecule in the frame of "
                                       "this crystal) and stop")
    g.add_argument("--gauss-state", type=int, default=1, help="excited state for --write-gjf (default 1)")
    g.add_argument("--method", default="B3LYP"); g.add_argument("--basis", default="6-31G(d)")
    g = p.add_argument_group("transport")
    g.add_argument("--lambda", dest="lam", type=float, default=0.15, help="exciton reorganization energy / eV")
    g.add_argument("--tau", type=float, default=10.0, help="exciton lifetime / ns")
    g.add_argument("--temp", type=float, default=300.0)
    g.add_argument("--r-max", type=float, default=30.0, help="range of the coupling and lattice sums / Å")
    g.add_argument("--kmc", type=int, default=0, help="KMC check with this number of excitons (0 = off)")
    g.add_argument("--n-show", type=int, default=8, help="couplings listed")
    g = p.add_argument_group("thermal fluctuations (rigid-body MD)")
    g.add_argument("--md-ps", type=float, default=0.0, help="MD sampling time / ps (0 = static couplings only)")
    g.add_argument("--md-equil", type=float, default=5.0, help="MD equilibration / ps")
    g.add_argument("--supercell", default="4,3,2")
    p.add_argument("--out", help="result json")
    p.set_defaults(func=cmd_exciton)

    p = sub.add_parser("diagnose", help="Transport regime from a CIF: 2|J|/λ, phonon σ_J/|J|, interlayer coupling, "
                                        "π frontier orbitals -> recommended model and command (a few seconds)")
    p.add_argument("cif", help="Crystal structure (CIF)")
    p.add_argument("--lam-h", type=float, required=True, help="Hole reorganization energy / eV")
    p.add_argument("--lam-e", type=float, required=True, help="Electron reorganization energy / eV")
    p.add_argument("--temp", type=float, default=300.0, help="Temperature / K")
    p.add_argument("--cutoff", type=float, default=4.0, help="Neighbour criterion: shortest atom-atom distance / Å")
    p.add_argument("--supercell", default="", help="Supercell of the lattice dynamics, e.g. 3,3,2 (default: >= 16 Å)")
    p.add_argument("--no-phonons", action="store_true", help="Skip the lattice dynamics (no σ_J/|J|)")
    p.add_argument("--uq", action="store_true",
                   help="Regime probabilities from the EHT calibration error, λ and σ_J/|J| uncertainties (Monte Carlo)")
    p.add_argument("--uq-samples", type=int, default=20000, help="Monte Carlo samples of --uq")
    p.add_argument("--lam-rel", type=float, default=0.10, help="Relative uncertainty of λ for --uq")
    p.add_argument("--seed", type=int, default=0, help="Random seed of --uq")
    p.add_argument("--json", default="", help="Write the full result to this JSON file")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("tlt", help="CIF → EHT 转移积分 → 瞬态局域化 (TLT) 迁移率与有效质量")
    p.add_argument("cif")
    p.add_argument("--carrier", choices=["hole", "electron"], default="hole")
    p.add_argument("--sigma", type=float, default=0.3, help="σ_J/|J|")
    p.add_argument("--htau", type=float, default=5.0, help="ħ/τ_in / meV")
    p.add_argument("--plane", default="ab")
    p.add_argument("--size", type=int, default=0)
    p.add_argument("--realizations", type=int, default=10)
    p.add_argument("--temp", type=float, default=300)
    p.add_argument("--cutoff", type=float, default=4.0)
    p.add_argument("--pi", action="store_true", help="用最高占据 / 最低空的 π 轨道代替 HOMO/LUMO (氮杂芳烃等孤对 σ 轨道落在 HOMO 附近时)")
    p.set_defaults(func=cmd_tlt)

    p = sub.add_parser("vibronic", help="由 FCHT 黄昆因子计算振动分辨磷光光谱与能隙定律")
    p.add_argument("log")
    p.add_argument("--ev", type=float, required=True, help="垂直发射能 / eV")
    p.add_argument("--temp", type=float, default=300)
    p.add_argument("--fwhm", type=float, default=0.05)
    p.set_defaults(func=cmd_vibronic)

    p = sub.add_parser("kmc-validate", help="KMC 精度自检：与精确解逐项比较 (可写出 CSV 与对照图)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--cif", help="多分子晶胞：用这个晶体 (软件内 EHT 转移积分) 加一项晶体图 KMC 与均匀化精确解的比较")
    p.add_argument("--cutoff", type=float, default=4.0, help="近邻判据：最短原子间距 / Å")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_kmc_validate)

    p = sub.add_parser("scan", help="批量扫描工作流：项目 / 层级 / 网格 / 验证集 / 对称检验 / 结果 / 报告 / 归档")
    ss = p.add_subparsers(dest="action", required=True)
    q = ss.add_parser("new", help="新建扫描项目")
    q.add_argument("project"); q.add_argument("--mono", required=True); q.add_argument("--name")
    q.add_argument("--frame", choices=["all", "core"], default="all",
                   help="分子坐标系：all = 全部重原子；core = 最大稠环体系 (共轭主体，如红荧烯的并四苯骨架)")
    q = ss.add_parser("level", help="添加计算层级 (生成 mono.run)")
    q.add_argument("project"); q.add_argument("--xc", default="PW91"); q.add_argument("--basis", default="TZP")
    q.add_argument("--d3", action="store_true"); q.add_argument("--import-t21", default="")
    q.add_argument("--engine", choices=["adf", "orca", "xtb"], default="adf",
                   help="program of the level: adf (fragment orbitals, default), orca (any DFT method and basis, free "
                        "for academic use) or xtb (GFN-xTB, seconds per dimer); orca / xtb couplings come from the "
                        "dimer projection on Molden files with the phase of B aligned to the reference monomer")
    q.add_argument("--method", default="B3LYP D3BJ", help="orca: method keywords, e.g. \"B3LYP D3BJ\" or \"PBE0\"")
    q.add_argument("--qc-basis", default="def2-SVP", help="orca: basis set (no diffuse functions), e.g. def2-SVP")
    q.add_argument("--gfn", type=int, choices=[1, 2], default=1, help="xtb: GFN1 (default) or GFN2")
    q.add_argument("--nprocs", type=int, default=4, help="orca: processes per calculation")
    q.add_argument("--maxcore", type=int, default=2000, help="orca: memory per process / MB")
    q.add_argument("--mirror", action="store_true",
                   help="同时生成镜像单体的片段计算 mono_m.run (手性分子的中心对称晶体，如 Alq3、Ir 配合物)")
    for nm, hp in (("grid", "参数网格 (z / Δx / Δy / γ / θ / φ，写法 起:止:步 或 a,b,c)"), ("valid", "随机验证集 (拉丁超立方)"),
                   ("check", "对称等价检验")):
        q = ss.add_parser(nm, help=hp)
        q.add_argument("project"); q.add_argument("name"); q.add_argument("--level")
        q.add_argument("--clash", type=float, default=0.75); q.add_argument("--chunk", type=int, default=100)
        if nm == "grid":
            q.add_argument("--z", default="3.5"); q.add_argument("--dx", default="0"); q.add_argument("--dy", default="0")
            q.add_argument("--gamma", default="0"); q.add_argument("--pitch", default="0"); q.add_argument("--roll", default="0")
            q.add_argument("--no-reduce", action="store_true")
        elif nm == "valid":
            q.add_argument("--interp", type=int, default=20); q.add_argument("--zi", default="3.5,3.8")
            q.add_argument("--extrap", type=int, default=20); q.add_argument("--ze", default="3.3,3.5;3.8,4.2")
            q.add_argument("--dx", default="-6,6"); q.add_argument("--dy", default="-3,3"); q.add_argument("--seed", type=int, default=2026)
        else:
            q.add_argument("--bases", default="3.5 1.2 0.7 0; 3.6 2.0 0.4 25", help="z Δx Δy φ; …")
    q = ss.add_parser("clone", help="同一批构型换层级重算 (泛函 / 基组对照)")
    q.add_argument("project"); q.add_argument("name"); q.add_argument("--level", required=True)
    q = ss.add_parser("import", help="导入 make_dimers.py 生成的扫描目录")
    q.add_argument("project"); q.add_argument("dir"); q.add_argument("--name"); q.add_argument("--level")
    q = ss.add_parser("status", help="全部扫描的进度")
    q.add_argument("project")
    q = ss.add_parser("precheck", help="运行前检查 / 修复完成标记")
    q.add_argument("project"); q.add_argument("name")
    q = ss.add_parser("collect", help="解析输出，写 results.csv")
    q.add_argument("project"); q.add_argument("name", nargs="?")
    for nm, hp in (("cp", "Gaussian 抵消校正相互作用能输入"), ("ti", "Gaussian 转移积分输入 (层级对照)")):
        q = ss.add_parser(nm, help=hp)
        q.add_argument("project"); q.add_argument("name"); q.add_argument("--functional", default="B3LYP")
        q.add_argument("--basis", default="6-31G(d)"); q.add_argument("--stride", type=int, default=1)
    q = ss.add_parser("report", help="分析与报告 (距离衰减、滑移图、可分离性、V–S、模型、验证、层级对照、能量、重组能)")
    q.add_argument("project"); q.add_argument("--lam-h", type=float); q.add_argument("--lam-e", type=float)
    q.add_argument("--reorg-dir"); q.add_argument("--temp", type=float, default=300.0)
    q.add_argument("--energy", choices=["ADF", "CP", "UFF"], default="ADF"); q.add_argument("--z0", type=float, default=3.5)
    q.add_argument("--level")
    q = ss.add_parser("archive", help="归档 (all_structures.xyz、all_results.csv、README、原始输出)")
    q.add_argument("project"); q.add_argument("--out"); q.add_argument("--no-outputs", action="store_true")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("ceds", help="CEDS 动态无序与迁移率–温度 (晶体局域势 MC + 耦合曲面 + TLT)")
    p.add_argument("action", choices=["run", "local", "md", "snapshots", "validate", "al", "phonons"],
                   help="run: 局域 MC 计算 σ_V 与 μ(T)；local: 为晶体二聚体生成 DFT 局域扫描；md: 全晶体刚体 MD 计算 σ_V 与 μ(T)；"
                        "snapshots: 从 MD 轨迹取构型，生成逐帧直接计算的 ADF 批处理；validate: 快照结果与耦合曲面比较；"
                        "al: 主动学习耦合曲面 (第一次运行写第 1 轮 ADF 批处理；算完后再次运行，拟合并写下一轮，直到收敛)；"
                        "phonons: 刚体晶格动力学，各声子支对 σ_V 与迁移率的贡献 (杀手声子)")
    p.add_argument("--cif", required=True)
    p.add_argument("--project", help="扫描项目 (耦合曲面来源 / 局域扫描写入处)")
    p.add_argument("--monomer", help="刚性单体 (xyz / gjf / log 或 project.json)；默认取扫描项目的单体或晶体中的分子")
    p.add_argument("--frame", choices=["all", "core"], default="all",
                   help="没有扫描项目时的分子坐标系定义 (有项目时取项目的设置)")
    p.add_argument("--name", default="ceds_local", help="局域扫描名称")
    p.add_argument("--level")
    p.add_argument("--source", choices=["project", "eht"], default="project")
    p.add_argument("--eht-fallback", action="store_true")
    p.add_argument("--temps", default="100 150 200 250 300")
    p.add_argument("--samples", type=int, default=1500)
    p.add_argument("--n-local", type=int, default=150)
    p.add_argument("--htau", type=float, default=5.0, help="ħ/τ_in / meV")
    p.add_argument("--cutoff", type=float, default=4.0)
    p.add_argument("--scale", type=float, default=1.0, help="对势缩放")
    p.add_argument("--ueq", type=float, help="目标 U_eq / Å² (按晶体学原子位移参数校准涨落幅度)")
    p.add_argument("--ueq-from-cif", action="store_true", help="从 CIF 读取 U_iso / U_aniso 作为目标")
    p.add_argument("--g-csv", help="乘积模型：Δx = 0 的 G(z, Δy, φ) 网格结果表 (results.csv)")
    p.add_argument("--fdx-csv", help="乘积模型：转动二聚体的 Δx 扫描结果表 (给出 f(Δx))")
    p.add_argument("--slip-csv", help="乘积模型：没有 Δx 扫描时，用平行二聚体滑移图 (results.csv 或 slip_full.csv) 的一行给 f(Δx)")
    p.add_argument("--dy-line", type=float, default=1.0, help="取 f(Δx) 的平行滑移图行 Δy / Å")
    p.add_argument("--model", choices=["grid", "r2"], default="grid",
                   help="grid: G 网格双三次 × f(Δx)；r2: 转角工作模型 f·g(Δy,φ)·exp[−β(Δy)(z−4.8)]")
    p.add_argument("--gphi-csv", nargs="*", help="r2 模型：加密 φ 扫描的结果表 (与 G 网格一起拟合 g)")
    p.add_argument("--local-csv", nargs="*", help="局域扫描结果表 (含 group 列的 results.csv)：没有 ADF 输出文件时直接拟合局域曲面")
    p.add_argument("--sigma-only", action="store_true", help="只计算 σ_V，不做 TLT")
    p.add_argument("--pairs", default="", help='local / snapshots：只处理这些组，如 "3,7"；"parallel" = 全部平行近邻 '
                                               '(snapshots 默认取全部近邻)')
    p.add_argument("--out", default="", help="结果 CSV (md 时另存同名 .png)")
    g = p.add_argument_group("md (全晶体刚体 MD)")
    g.add_argument("--supercell", default="4,3,2", help="超胞 na,nb,nc")
    g.add_argument("--md-charges", default=None,
                   help="MD 静电用的原子电荷：Gaussian 输出 (pop=mk 的 ESP 电荷优先)；含 N/O/金属的分子建议给出")
    g.add_argument("--md-equil", type=float, default=5.0, help="每个温度的平衡时间 / ps")
    g.add_argument("--md-prod", type=float, default=30.0, help="每个温度的采样时间 / ps")
    g.add_argument("--md-dt", type=float, default=5.0, help="时间步长 / fs")
    g.add_argument("--md-every", type=int, default=4, help="每隔多少步存一帧")
    g.add_argument("--md-rc", type=float, default=9.0, help="非键截断 / Å (LJ 开关从 rc−1 开始)")
    g.add_argument("--htau-list", default="", help='另外计算这些 ħ/τ (meV) 的 μ，如 "2,10"')
    g.add_argument("--raw-geometry", action="store_true", help="不把平均结构移回实验结构 (直接用力场平均结构)")
    g.add_argument("--frame-tlt", type=int, nargs="?", const=24, default=0, metavar="N_FRAMES",
                   help="md: TLT with the Hamiltonians of N_FRAMES MD frames on the MD supercell (default 24), compared "
                        "with independent Gaussian draws and with independent draws from the MD distributions on the "
                        "same lattice. Needs a supercell of at least 8 cells along the two transport directions, e.g. "
                        "--supercell 12,12,2 (each layer of the supercell is one TLT lattice)")
    g.add_argument("--frame-stride", type=int, default=0, metavar="K",
                   help="md --frame-tlt: evaluate the coupling statistics on every K-th MD frame (default: about 300 "
                        "frames, i.e. K = 5 for 30 ps)")
    g.add_argument("--jobs", type=int, default=0,
                   help="EHT baseline of Δ-GP surfaces: number of processes for large batches (default: all CPU cores)")
    g.add_argument("--save-traj", default="", help="把每个温度的轨迹存为 npz 的目录")
    g.add_argument("--no-save-traj", action="store_true",
                   help="md: do not keep the trajectories (by default they are saved to md_traj_<cif>_<supercell> in "
                        "the working folder and reused when the same run is started again)")
    g.add_argument("--rerun-md", action="store_true",
                   help="md: run the MD again even if a saved trajectory with the same settings exists")
    g.add_argument("--traj", default="", help="snapshots：直接使用已保存的轨迹 (npz)，不再运行 MD")
    g.add_argument("--n-snap", type=int, default=120, help="snapshots：每个二聚体取多少个构型")
    g.add_argument("--md-half", action="store_true", help="local：按全晶体 MD 的涨落 (±k·σ) 确定局域扫描范围")
    g.add_argument("--md-k", type=float, default=3.0, help="local --md-half：范围取 ±k·σ_MD")
    g = p.add_argument_group("phonons (声子分辨的动态无序；力场、--supercell、--md-charges、--md-rc 与 md 相同)")
    g.add_argument("--ph-quantum", action="store_true", help="声子振幅用量子 (Bose–Einstein) 占据，默认经典 k_BT/ω²")
    g.add_argument("--ph-htau-auto", action="store_true",
                   help="TLT 的 ħ/τ 取声子估计 (σ² 加权的平均 ħω)，代替 --htau")
    g.add_argument("--ph-gamma", action="store_true", help="另外给出只用 Γ 点的结果作对照")
    g.add_argument("--ph-real", type=int, default=16, help="TLT 实现数 (各声子支贡献的数值微分用同一组随机数)")
    g.add_argument("--ph-external", default="",
                   help="外部 (DFT) 声子：phonopy 的 mesh.yaml / qpoints.yaml / band.yaml，需含本征矢 (--eigvecs)；"
                        "网格需完整 (MESH_SYMMETRY = .FALSE.)，晶胞与 CIF 相同。CRYSTAL、VASP、QE 等经 phonopy 接口得到")
    g.add_argument("--ph-structure", default="", help="--ph-external 文件中没有结构时 (qpoints.yaml)，给出 phonopy.yaml")
    g.add_argument("--ph-length-unit", choices=["auto", "angstrom", "bohr"], default="auto",
                   help="外部文件的长度单位 (auto：与 CIF 晶胞比较判断)")
    g.add_argument("--ph-phase", choices=["positions", "cell"], default="positions",
                   help="本征矢的相位约定：positions = phonopy (含原子分数坐标)；cell = 只含晶胞平移")
    g = p.add_argument_group("al (主动学习耦合曲面)")
    g.add_argument("--al-init", type=int, default=12, help="第 1 轮每个近邻对的构型数")
    g.add_argument("--al-abs", type=float, default=0.3,
                   help="收敛的绝对下限 / meV：σ_V 区间半宽小于它也算收敛 (弱耦合二聚体)")
    g.add_argument("--al-seed", default=None,
                   help="已有扫描 (逗号分隔) 中算完的构型作为初始训练数据，例如先前的局部扫描 rub_local；"
                        "只在第一次运行时读取设置，之后每轮自动包含")
    g.add_argument("--al-batch", type=int, default=10, help="之后每轮每个未收敛近邻对的构型数")
    g.add_argument("--al-target", type=float, default=0.03, help="收敛判据：σ_V 的相对不确定度 (95%% 区间半宽)")
    g.add_argument("--al-pool", type=int, default=600, help="候选池：每个近邻对从 MD 中取多少个构型")
    g.add_argument("--al-delta", choices=["auto", "on", "off"], default="auto",
                   help="Δ-学习：GP 均值取 β₀ + β₁·V_EHT，只学习 DFT 与 EHT 之差 (新任务默认开启；已有任务保持原设置，"
                        "on / off 可切换)")
    g.add_argument("--al-goal", choices=["auto", "sigma", "mu"], default="auto",
                   help="选点与停止的目标：sigma = 各近邻对 σ_V 的区间 (默认)；mu = TLT 迁移率的区间，"
                        "在全部近邻对中按 μ 的后验方差选点 (v1.21)。已有任务保持原设置，sigma / mu 可切换")
    g.add_argument("--al-calib", choices=["auto", "fixed", "loo"], default="auto",
                   help="后验区间的校准：fixed = 固定因子 c = 2.0 (默认，在并五苯上标定)；loo = 每个近邻对由 DFT 点的"
                        "留一法标准化残差自行给出 c。两种方式都会在结果表中列出 c_LOO。已有任务保持原设置")
    g.add_argument("--al-mu-target", type=float, default=None,
                   help="--al-goal mu 的停止判据：μ 的 95%% 区间相对半宽 (默认 0.01)")
    g.add_argument("--al-calib-report", action="store_true",
                   help="报告各近邻对的留一法校准因子 c_LOO 与 c = 1.5/2.0/2.5/c_LOO 下 σ_V 的 95%% 区间 (只报告，不改变选点)")
    g.add_argument("--al-budget", type=int, nargs="?", const=40, default=0, metavar="N_BOOT",
                   help="al：给出迁移率的误差预算 (曲面后验、热运动样本、TLT 实现数、ħ/τ、σ_E 各自的影响)；"
                        "N_BOOT = 热运动样本的自助抽样次数 (默认 40)")
    g.add_argument("--al-budget-sigma-e", default="",
                   help='--al-budget 中位点能无序 σ_E / meV，空穴、电子 (如 "31,46"，来自 polar)')
    g.add_argument("--al-mu-draws", type=int, default=30,
                   help="收敛后把曲面不确定度传递到 TLT 迁移率的后验抽样次数 (0 = 不计算)")
    g.add_argument("--al-surfaces", default="", help="run / md：用这个主动学习任务拟合的 GP 曲面 (覆盖其它来源)")
    g.add_argument("--force", action="store_true", help="al：上一轮还有未完成的构型时也继续")
    p.set_defaults(func=cmd_ceds)

    p = sub.add_parser("polar", help="极化位点能 (原子电荷 + Thole 诱导偶极)：P±、固态 IE/EA、位点能差与 σ_E")
    p.add_argument("--cif", required=True)
    p.add_argument("--monomer", help="刚性单体 (xyz / gjf / log 或 project.json)；默认取晶体中的分子")
    p.add_argument("--frame", choices=["all", "core"], default="all")
    p.add_argument("--gjf", default="", help="只生成 ESP 电荷与极化率的 Gaussian 输入 (给出文件名前缀)")
    p.add_argument("--level", help="--gjf 的方法/基组 (默认 B3LYP/6-31G(d))")
    p.add_argument("--neutral", help="中性分子的 Gaussian 输出 (pop=mk 的 ESP 电荷)")
    p.add_argument("--cation", help="阳离子 (中性几何) 的 Gaussian 输出")
    p.add_argument("--anion", help="阴离子 (中性几何) 的 Gaussian 输出")
    p.add_argument("--polar-log", help="含 polar 结果的 Gaussian 输出：把原子极化率缩放到 DFT 分子极化率")
    p.add_argument("--alpha-mean", type=float, help="直接给出目标分子平均极化率 / Å³")
    p.add_argument("--radii", default="14,18,22,26,30",
                   help="团簇半径 / Å (最大半径 ≥ 26 Å 时诱导项按 1/R 拟合外推，否则用 Born 尾项；"
                        "快速估计可用 14,18,22)")
    p.add_argument("--ip-gas", type=float, help="气相电离能 / eV (给出则计算固态 IE)")
    p.add_argument("--ea-gas", type=float, help="气相电子亲和能 / eV")
    p.add_argument("--traj", help="全晶体 MD 轨迹 (ceds md --save-traj 的 npz)：计算热涨落 σ_E")
    p.add_argument("--md-frames", type=int, default=24)
    p.add_argument("--md-centers", type=int, default=4, help="每帧计算的分子数 (各子格子均分)")
    p.add_argument("--md-r", type=float, default=14.0, help="快照团簇半径 / Å")
    p.add_argument("--out", default="", help="结果 CSV (另存同名 .png)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_polar)

    p = sub.add_parser("morph", help="无定形 / MD 形貌：GROMACS 形貌 → 分子映射 → 近邻 → 转移积分 → 位点能 → KMC")
    p.add_argument("--gro", required=True, help="形貌 (.gro 可多帧，或 .pdb)；有 --traj 时作为拓扑")
    p.add_argument("--traj", help=".xtc / .trr / .dcd 轨迹 (需要 MDAnalysis 或 mdtraj)")
    p.add_argument("--frame", type=int, default=0, help="使用第几帧 (从 0 开始)")
    p.add_argument("--monomer", required=True, help="参考单体 (DFT 优化结构：xyz / gjf / log / project.json)")
    p.add_argument("--resname", help="要映射的残基名 (默认：重原子数与单体相同、数量最多的残基)")
    p.add_argument("--elements", help='原子名到元素的对照，如 "CA=C,HA=H"')
    p.add_argument("--geometry", choices=["qm", "md"], default="qm", help="qm: 叠合后的参考单体；md: MD 原子坐标")
    p.add_argument("--cutoff", type=float, default=4.0, help="近邻判据：最短重原子间距 / Å")
    p.add_argument("--coupling", choices=["eht", "zindo", "none"], default="eht",
                   help="eht: 软件内 EHT；zindo: ZINDO/MOO (VOTCA izindo)")
    p.add_argument("--zindo-orbitals", help="ZINDO/MOO 使用 Gaussian ZINDO 单体计算的 fchk (默认软件内 INDO/S)")
    p.add_argument("--degeneracy", type=float, default=0.0, help="近简并能级阈值 / meV (0 = 不合并)")
    p.add_argument("--import-j", help="读入转移积分 CSV (pair, J_hole_meV, J_electron_meV)")
    p.add_argument("--export-dimers", help="为近邻对写出 Gaussian 投影法输入的目录")
    p.add_argument("--max-pairs", type=int, default=0, help="--export-dimers 只写前 N 对 (0 = 全部)")
    p.add_argument("--collect-dft", help="读回 --export-dimers 目录中的 Gaussian 结果 (fchk)")
    p.add_argument("--level", help="Gaussian 方法/基组 (默认 B3LYP/6-31G(d))")
    p.add_argument("--calib-pairs", type=int, default=0,
                   help="--export-dimers 只写 N 个按 |J| 分布挑选的代表性对 (DFT 校准用，常用 15–30)")
    p.add_argument("--calibrate", help="用该目录中已完成的 Gaussian 二聚体结果校准 EHT / ZINDO：空穴、电子各拟合一个比例 k")
    p.add_argument("--calibrate-csv", help="同 --calibrate，参考 J 来自 CSV (pair, J_hole_meV, J_electron_meV)")
    p.add_argument("--calib-no-replace", action="store_true", help="校准后有参考值的对也用 k × 半经验值 (默认直接用参考值)")
    p.add_argument("--scale-h", type=float, help="手动给定空穴比例系数 k (J ← k·J_半经验)")
    p.add_argument("--scale-e", type=float, help="手动给定电子比例系数 k")
    p.add_argument("--site-energy", choices=["none", "static", "polar"], default="static",
                   help="位点能：static = 原子电荷静电；polar = 静电 + Thole 诱导 (每个分子约 0.5 s)")
    p.add_argument("--site-r", type=float, help="位点能截断半径 / Å (static 默认 30，polar 默认 14)")
    p.add_argument("--neutral"); p.add_argument("--cation"); p.add_argument("--anion")
    p.add_argument("--lam-h", type=float, help="空穴重组能 / eV (给出则运行 KMC)")
    p.add_argument("--lam-e", type=float, help="电子重组能 / eV")
    p.add_argument("--temp", type=float, default=300.0)
    p.add_argument("--field", type=float, default=0.0, help="另算场迁移率的电场 / V cm⁻¹ (沿 x、y、z 平均)")
    p.add_argument("--walkers", type=int, default=1000)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--out", default="", help="输出前缀 (写出 _sites.csv、_pairs.csv 与 .png)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_morph)

    p = sub.add_parser("zindo", help="ZINDO/MOO 转移积分 (VOTCA izindo)：单个二聚体、与 Gaussian 对照的检验集")
    p.add_argument("action", choices=["pair", "bench-inputs", "bench"],
                   help="pair: 计算一个二聚体；bench-inputs: 写出检验集的 Gaussian ZINDO 与 DFT 输入；bench: 汇总检验结果")
    p.add_argument("--dimer", help="pair：二聚体结构 (前 --na 个原子为 A)")
    p.add_argument("--na", type=int, default=0)
    p.add_argument("--orbitals", help="pair：Gaussian ZINDO 单体 fchk (默认软件内 INDO/S)")
    p.add_argument("--degeneracy", type=float, default=0.0, help="近简并能级阈值 / meV")
    p.add_argument("--cif", help="bench-inputs：晶体 (取全部不等价近邻对)")
    p.add_argument("--monomer", help="bench-inputs：参考单体 (DFT 优化结构)")
    p.add_argument("--frame", choices=["all", "core"], default="all")
    p.add_argument("--cutoff", type=float, default=4.0)
    p.add_argument("--level", help="bench-inputs：DFT 方法/基组 (默认 B3LYP/6-31G(d))")
    p.add_argument("--out", default="", help="bench-inputs：输出目录；bench：结果 CSV")
    p.add_argument("--dir", help="bench：检验集目录")
    p.set_defaults(func=cmd_zindo)

    p = sub.add_parser("cif2gro", help="晶体超胞写成 GROMACS .gro (起始结构 / 形貌流程测试)")
    p.add_argument("--cif", required=True)
    p.add_argument("--monomer", help="用这个单体 (如 DFT 优化结构) 替换晶体中的分子")
    p.add_argument("--frame", choices=["all", "core"], default="all")
    p.add_argument("--supercell", default="4,3,2")
    p.add_argument("--resname", default="MOL")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_cif2gro)

    p = sub.add_parser("gui", help="启动图形界面")
    p.add_argument("--lang", choices=["zh", "en"], help="界面语言 (默认使用上次的设置)")
    p.set_defaults(func=_gui)

    _fill_missing_help(ap)
    _translate_parser(ap)                               # English: Chinese sources → English; Chinese: the reverse
    ap.epilog = i18n.tr_cli("Chinese help and messages: python -m mobility_tool --lang zh <subcommand> ... "
                            "(or MOBILIS_LANG=zh)")
    args = ap.parse_args(argv)
    if os.environ.get("MOBILIS_PARSE_ONLY") == "1":    # project check: arguments only, nothing is run
        print("MOBILIS_PARSE_OK " + args.cmd)
        return
    if hasattr(args, "jobs"):                           # EHT baseline: worker processes for large batches
        from . import alsurface as _als
        _als.EHT_JOBS = args.jobs if args.jobs and args.jobs > 0 else (os.cpu_count() or 1)
    args.func(args)


if __name__ == "__main__":
    main()
