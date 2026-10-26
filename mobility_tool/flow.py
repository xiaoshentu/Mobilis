"""流程图 (多步串联计算)。

一个流程 = 有序的阶段列表；每个阶段是 {"type": ..., "settings": {...}}：
  gaussian    运行 Gaussian (设置与“运行 Gaussian”对话框相同：opt / reorg / excited / ti / cp / esp / fcht)
  adf         运行 ADF (ti / crystal)
  dimers      软件内：搜索近邻二聚体
  eht         软件内：EHT 计算全部二聚体 J
  mobility    软件内：Marcus 迁移率 (解析)
  kmc         软件内：动力学蒙特卡洛迁移率
前一阶段的结果放进共享上下文 ctx，后一阶段直接使用：
  opt → ctx["molecule"] (优化后的结构)；reorg → ctx["lambda"]、ctx["reorg_dir"]、优化后的中性结构；
  ti (晶体) / adf crystal / eht → ctx["couplings"]；dimers / eht → ctx["dimer_res"]；mobility / kmc → ctx["results"]。
流程可保存为 JSON (*.mflow)，在命令行中也能运行：python -m mobility_tool.flow my.mflow 工作目录
"""
import json
import threading
import time
from pathlib import Path

FLOW_VERSION = 1

STAGE_TYPES = {   # type: (名称, 说明)
    "gaussian": ("Gaussian", "运行 Gaussian：设置与“运行 Gaussian”对话框相同"),
    "adf": ("ADF", "运行 ADF：二聚体或晶体全部近邻的转移积分"),
    "dimers": ("搜索近邻二聚体", "软件内：在晶体中搜索参考分子的近邻二聚体"),
    "eht": ("EHT 转移积分", "软件内：EHT 计算全部近邻二聚体的 J (秒级，适合筛选)"),
    "mobility": ("Marcus 迁移率", "软件内：由 λ 与各二聚体 J 计算迁移率张量 (解析)"),
    "kmc": ("KMC 迁移率", "软件内：动力学蒙特卡洛模拟迁移率"),
}

GAUSS_CALC = {"opt": "结构优化", "reorg": "重组能 (四点法)", "excited": "激发态", "ti": "转移积分",
              "cp": "相互作用能 (BSSE)", "esp": "ESP 电荷", "fcht": "FCHT 黄昆因子"}
ADF_CALC = {"ti": "二聚体转移积分", "crystal": "晶体全部近邻"}

DEFAULTS = {
    "dimers": {"cutoff": 4.0, "criterion": "atom", "ref": 1},
    "eht": {"cutoff": 4.0, "criterion": "atom", "ref": 1},
    "mobility": {"T": 300.0, "lam_h": 0.10, "lam_e": 0.15, "use_flow_lambda": True},
    "kmc": {"T": 300.0, "lam_h": 0.10, "lam_e": 0.15, "use_flow_lambda": True, "carrier": "hole",
            "walkers": 1000, "steps": 3000, "seed": 1},
}


def new_stage(kind, settings=None):
    s = dict(DEFAULTS.get(kind, {}))
    s.update(settings or {})
    return {"type": kind, "settings": s}


def stage_title(st):
    kind, s = st["type"], st.get("settings", {})
    if kind == "gaussian":
        t = GAUSS_CALC.get(s.get("calc"), s.get("calc", "?"))
        if s.get("calc") in ("ti", "cp"):
            t += " (晶体全部近邻)" if s.get("source") == "晶体全部近邻二聚体" else " (二聚体)"
        return f"Gaussian · {t}"
    if kind == "adf":
        return f"ADF · {ADF_CALC.get(s.get('calc'), s.get('calc', '?'))}"
    return STAGE_TYPES[kind][0]


def stage_summary(st):
    """阶段框中第二行的简短说明。"""
    kind, s = st["type"], st.get("settings", {})
    if kind == "gaussian":
        m = s.get("method", "")
        b = s.get("ex_basis") if s.get("calc") == "excited" else s.get("basis", "")
        return f"{m}/{b}" if m else "双击设置"
    if kind == "adf":
        return f"{s.get('functional', '')}/{s.get('basis', '')}" if s.get("functional") else "双击设置"
    if kind in ("dimers", "eht"):
        return f"截断 {s.get('cutoff', 4.0)} Å"
    if kind == "mobility":
        return f"T = {s.get('T', 300)} K" + ("，λ 取自流程" if s.get("use_flow_lambda") else "")
    if kind == "kmc":
        return f"{'空穴' if s.get('carrier') == 'hole' else '电子'}，{s.get('walkers')} 个载流子"
    return ""


def save_flow(path, stages, title=""):
    Path(path).write_text(json.dumps({"mobilis_flow": FLOW_VERSION, "title": title, "stages": stages},
                                     ensure_ascii=False, indent=2), encoding="utf-8")


def load_flow(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "stages" not in d:
        raise ValueError(f"{path} 不是 Mobilis 流程文件")
    for st in d["stages"]:
        if st.get("type") not in STAGE_TYPES:
            raise ValueError(f"未知阶段类型 {st.get('type')}")
    return d.get("title", ""), d["stages"]


def check_flow(stages):
    """运行前检查阶段之间的依赖，返回提示列表 (空 = 没有问题)。"""
    probs, have = [], set()
    for n, st in enumerate(stages, 1):
        kind, s = st["type"], st.get("settings", {})
        tag = f"阶段 {n} ({stage_title(st)})"
        if kind in ("gaussian", "adf") and not s.get("calc"):
            probs.append(f"{tag}：还没有设置，请双击阶段打开设置")
        need_dimers = (kind == "gaussian" and s.get("source") == "晶体全部近邻二聚体") or \
                      (kind == "adf" and s.get("calc") == "crystal")
        if need_dimers and "dimers" not in have:
            probs.append(f"{tag}：需要近邻二聚体，请在它之前加入“搜索近邻二聚体”阶段 (或先在主窗口搜索)")
        if kind in ("mobility", "kmc") and "couplings" not in have:
            probs.append(f"{tag}：需要各二聚体的 J，请在它之前加入 EHT / Gaussian 晶体转移积分 / ADF 晶体阶段")
        if kind in ("dimers", "eht"):
            have.add("dimers")
        if kind == "eht" or (kind == "adf" and s.get("calc") == "crystal") or \
                (kind == "gaussian" and s.get("calc") == "ti" and s.get("source") == "晶体全部近邻二聚体"):
            have.add("couplings")
    return probs


class FlowRunner:
    """在后台线程中依次运行各阶段。
    on_status(序号, 状态, 信息)：状态为 running / done / failed / skipped；on_log(文字)；on_done(ctx, 是否成功)。
    回调在后台线程中调用，界面需自行转回主线程。"""

    def __init__(self, stages, workdir, ctx, gaussian=None, ams=None, scratch=None,
                 on_status=None, on_log=None, on_done=None):
        self.stages, self.workdir, self.ctx = stages, Path(workdir), dict(ctx)
        self.gaussian, self.ams = gaussian, ams
        self.scratch = Path(scratch) if scratch else self.workdir / "scratch"
        self.on_status = on_status or (lambda *a: None)
        self.on_log = on_log or (lambda *a: None)
        self.on_done = on_done or (lambda *a: None)
        self.stop_event = threading.Event()
        self.thread = None
        self.ctx.setdefault("results", {})

    # ---- 控制 ----
    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()

    def run(self):
        ok = True
        self.workdir.mkdir(parents=True, exist_ok=True)
        t_all = time.time()
        for n, st in enumerate(self.stages):
            if self.stop_event.is_set():
                self.on_status(n, "skipped", "已停止")
                ok = False
                continue
            if not ok:
                self.on_status(n, "skipped", "前面的阶段失败")
                continue
            self.on_status(n, "running", "")
            self.on_log(f"── 阶段 {n + 1}：{stage_title(st)} ──")
            t0 = time.time()
            try:
                msg = getattr(self, f"_run_{st['type']}")(n, st.get("settings", {})) or "完成"
                self.on_status(n, "done", msg)
                self.on_log(f"   {msg}  ({time.time() - t0:.1f} s)")
            except Exception as exc:  # noqa: BLE001
                ok = False
                self.on_status(n, "failed", str(exc))
                self.on_log(f"   [失败] {exc}")
        self._write_summary(ok, time.time() - t_all)
        self.on_done(self.ctx, ok)
        return ok

    def _write_summary(self, ok, dt):
        res = self.ctx.get("results", {})
        out = {"ok": ok, "seconds": round(dt, 1), "lambda": self.ctx.get("lambda"),
               "couplings_meV": {str(k): {c: v.get(c) for c in ("hole", "electron")}
                                 for k, v in (self.ctx.get("couplings") or {}).items()},
               "results": _jsonable(res)}
        try:
            (self.workdir / "flow_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                            encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def _stage_dir(self, n, st_name):
        d = self.workdir / f"stage{n + 1}_{st_name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- 外部程序 ----
    def _run_gjfs(self, d, gjfs):
        from .jobs import find_gaussian, order_jobs, run_one
        info = self.gaussian or find_gaussian()
        if not info:
            raise FileNotFoundError("没有找到 Gaussian：请在 Gaussian → 程序路径设置 中指定 g16 / g09")
        for k, g in enumerate(order_jobs(gjfs), 1):
            if self.stop_event.is_set():
                raise RuntimeError("已手动停止")
            self.on_log(f"   [{k}/{len(gjfs)}] {Path(g).name}")
            ok, out, msg = run_one(info, g, self.scratch, stop_event=self.stop_event)
            if not ok:
                raise RuntimeError(f"{Path(g).name}: {msg}")

    def _run_gaussian(self, n, s):
        from .run_inputs import gaussian_generate
        calc = s.get("calc", "opt")
        v = dict(s)
        v.setdefault("name", "job")
        if calc == "fcht":
            v["reorg_dir"] = v.get("reorg_dir") or self.ctx.get("reorg_dir")
            d = Path(v["reorg_dir"] or ".")
            v["name"] = self.ctx.get("reorg_name", v["name"])
        else:
            d = self._stage_dir(n, f"g_{calc}")
        t0 = time.time() - 1
        note = gaussian_generate(d, v, self.ctx)
        _add_input_text(d, t0, v.get("addin"))
        gjfs = sorted(p for p in d.glob("*.gjf") if p.stat().st_mtime >= t0)
        if not gjfs:
            raise RuntimeError("没有生成 gjf 文件")
        self.on_log(f"   写出 {len(gjfs)} 个输入文件到 {d}" + (f"；{note}" if note else ""))
        self._run_gjfs(d, gjfs)
        return self._post_gaussian(calc, d, v)

    def _post_gaussian(self, calc, d, v):
        from .gaussian_io import read_log_geometry
        from .jobs import output_name
        name = v["name"]
        if calc == "opt":
            log = output_name(d / f"{name}_neutral.gjf")
            atoms = read_log_geometry(log)
            self.ctx["molecule"] = (atoms, name)
            return f"优化完成，后续阶段使用优化后的结构 ({len(atoms)} 个原子)"
        if calc == "reorg":
            from .reorganization import analyze_reorg
            self.ctx["reorg_dir"], self.ctx["reorg_name"] = str(d), name
            r = analyze_reorg(d)
            lam = {c: r["results"][c]["lambda_eV"] for c in ("hole", "electron") if c in r.get("results", {})}
            if not lam:
                raise RuntimeError("没有读到重组能：" + "；".join(r.get("warnings", [])[:3]))
            self.ctx["lambda"] = dict(self.ctx.get("lambda") or {}, **lam)
            try:
                self.ctx["molecule"] = (read_log_geometry(output_name(d / f"{name}_neutral.gjf")), name)
            except Exception:  # noqa: BLE001
                pass
            return "λ = " + "，".join(f"{'空穴' if c == 'hole' else '电子'} {x * 1000:.1f} meV" for c, x in lam.items())
        if calc == "ti":
            if v.get("source") == "晶体全部近邻二聚体":
                from .workflow import collect_group_couplings
                coup, err = collect_group_couplings(d, "gaussian")
                return self._set_couplings(coup, err)
            from .transfer_integral import calc_transfer_integrals
            f = [d / f"{name}_{x}.fchk" for x in ("A", "B", "AB")]
            r = calc_transfer_integrals(*f)["results"]
            self.ctx["dimer_J"] = {"hole": r["HOMO"]["J_eff_meV"], "electron": r["LUMO"]["J_eff_meV"]}
            return f"J(空穴) = {r['HOMO']['J_eff_meV']:.2f} meV，J(电子) = {r['LUMO']['J_eff_meV']:.2f} meV"
        if calc == "esp":
            self.ctx["charges_dir"] = str(d)
        return "计算完成，结果在 " + str(d)

    def _set_couplings(self, coup, err):
        if not coup:
            raise RuntimeError("没有读到任何二聚体的 J：" + "；".join(f"G{k}: {e}" for k, e in list(err.items())[:3]))
        self.ctx["couplings"] = coup
        msg = f"读到 {len(coup)} 个二聚体组的 J"
        if err:
            msg += f"；{len(err)} 组失败 ({', '.join(f'G{k}' for k in err)})"
        return msg

    def _run_adf(self, n, s):
        from .run_inputs import adf_generate
        from .programs import find_ams, run_adf_script
        v = dict(s)
        v.setdefault("name", "job")
        d = self._stage_dir(n, f"adf_{v.get('calc', 'ti')}")
        t0 = time.time() - 1
        note = adf_generate(d, v, self.ctx)
        info = self.ams or find_ams(self.ctx.get("ams_home") or None)
        if not info:
            raise FileNotFoundError("没有找到 ADF / AMS：请在 ADF → 程序路径设置 中指定安装目录")
        scripts = sorted(p for p in d.glob("*.sh") if p.stat().st_mtime >= t0 and p.name.startswith("run"))
        self.on_log(f"   写出输入到 {d}" + (f"；{note}" if note else ""))
        for k, sc in enumerate(scripts, 1):
            if self.stop_event.is_set():
                raise RuntimeError("已手动停止")
            self.on_log(f"   [{k}/{len(scripts)}] {sc.name}")
            ok, _log, msg = run_adf_script(info, sc, self.scratch, stop_event=self.stop_event)
            if not ok:
                raise RuntimeError(f"{sc.name}: {msg}")
        if v.get("calc") == "crystal":
            from .workflow import collect_group_couplings
            coup, err = collect_group_couplings(d, "adf", ams_info=info)
            return self._set_couplings(coup, err)
        from .adf import read_adf_transfer_integrals
        res = next((p for p in list(d.glob("*.results/adf.rkf")) + list(d.glob("*.t21")) + list(d.glob("*.out"))), None)
        if res is None:
            return "计算完成，结果在 " + str(d)
        r = read_adf_transfer_integrals(res, info)
        self.ctx["dimer_J"] = {c: r.get(c, {}).get("V_meV") for c in ("hole", "electron")}
        return "V(空穴) = {}，V(电子) = {} meV".format(*(self.ctx["dimer_J"][c] for c in ("hole", "electron")))

    # ---- 软件内 ----
    def _crystal(self):
        if not self.ctx.get("crystal"):
            raise ValueError("流程需要晶体结构：请先在主窗口打开 CIF")
        return self.ctx["crystal"]

    def _run_dimers(self, n, s):
        from .crystal import find_dimers
        crys, mols = self._crystal()
        res = find_dimers(crys, mols, ref=int(s.get("ref", 1)) - 1, cutoff=float(s.get("cutoff", 4.0)),
                          criterion=s.get("criterion", "atom"))
        self.ctx["dimer_res"] = res
        self.ctx["cutoff"], self.ctx["criterion"] = float(s.get("cutoff", 4.0)), s.get("criterion", "atom")
        return f"{len(res['dimers'])} 个近邻，{len(res['groups'])} 个等价组"

    def _run_eht(self, n, s):
        from .workflow import eht_couplings_for_crystal
        crys, mols = self._crystal()
        cut, crit = float(s.get("cutoff", 4.0)), s.get("criterion", "atom")
        dr = self.ctx.get("dimer_res") if s.get("use_flow_dimers", True) else None
        res, coup = eht_couplings_for_crystal(crys, mols, cut, crit, ref=int(s.get("ref", 1)) - 1, dimer_result=dr)
        self.ctx["dimer_res"], self.ctx["couplings"] = res, coup
        self.ctx.setdefault("cutoff", cut)
        self.ctx.setdefault("criterion", crit)
        return f"EHT 计算 {len(coup)} 个二聚体组的 J (最大 |J| 空穴 " \
               f"{max(abs(c['hole']) for c in coup.values()):.1f} meV)"

    def _lambdas(self, s):
        lam = dict(self.ctx.get("lambda") or {}) if s.get("use_flow_lambda", True) else {}
        return {"hole": float(lam.get("hole", s.get("lam_h", 0.1))),
                "electron": float(lam.get("electron", s.get("lam_e", 0.15)))}

    def _need_couplings(self):
        if not self.ctx.get("couplings") or not self.ctx.get("dimer_res"):
            raise ValueError("还没有各二聚体的 J：请在此阶段之前加入 EHT 或晶体转移积分阶段")
        return self.ctx["dimer_res"], self.ctx["couplings"]

    def _run_mobility(self, n, s):
        import numpy as np
        from .workflow import mobility_from_dimers
        res, coup = self._need_couplings()
        lam, T = self._lambdas(s), float(s.get("T", 300))
        mob = mobility_from_dimers(res, coup, lam["electron"], lam["hole"], T)
        exact = {}
        if self.ctx.get("crystal"):
            from .kmc import analytic_equivalent_sites, graph_from_crystal
            crys, mols = self.ctx["crystal"]
            for car in ("hole", "electron"):
                try:
                    g = graph_from_crystal(crys, mols, coup, self.ctx.get("cutoff", 4.0),
                                           self.ctx.get("criterion", "atom"), carrier=car)
                    if g["edges"] and not g["missing_groups"]:
                        t = np.asarray(analytic_equivalent_sites(g, lam[car], T))
                        exact[car] = {"tensor": t.tolist(), "mu_avg": float(np.trace(t) / 3),
                                      "mu_max": float(np.linalg.eigvalsh(t).max())}
                except Exception:  # noqa: BLE001
                    pass
        self.ctx["results"]["mobility"] = {"lambda_eV": lam, "T": T, "marcus": mob, "exact": exact}
        parts = []
        for car in ("hole", "electron"):
            r = exact.get(car) or mob.get(car)
            if r:
                parts.append(f"{'空穴' if car == 'hole' else '电子'} μ_avg = {r['mu_avg']:.3g} cm²/V·s")
        return "；".join(parts) or "没有可用的跳跃路径"

    def _run_kmc(self, n, s):
        from .kmc import graph_from_crystal, run_kmc
        res, coup = self._need_couplings()
        crys, mols = self._crystal()
        car = s.get("carrier", "hole")
        lam, T = self._lambdas(s)[car], float(s.get("T", 300))
        g = graph_from_crystal(crys, mols, coup, self.ctx.get("cutoff", 4.0), self.ctx.get("criterion", "atom"),
                               carrier=car)
        r = run_kmc(g, lam, T, carrier=car, walkers=int(s.get("walkers", 1000)), steps=int(s.get("steps", 3000)),
                    seed=int(s.get("seed", 1)))
        import numpy as np
        t = np.asarray(r["mu_tensor_cm2_Vs"])
        self.ctx["results"].setdefault("kmc", {})[car] = {"tensor": t.tolist(), "mu_avg": float(np.trace(t) / 3),
                                                          "lambda_eV": lam, "T": T}
        return f"KMC {'空穴' if car == 'hole' else '电子'} μ_avg = {np.trace(t) / 3:.3g} cm²/V·s"


def _add_input_text(d, t0, txt):
    if not (txt or "").strip():
        return
    from .run_inputs import add_input
    add_input([p for p in Path(d).glob("*.gjf") if p.stat().st_mtime >= t0], txt)


def _jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main(argv=None):
    """命令行运行流程：python -m mobility_tool.flow 流程.mflow 工作目录 [--cif 晶体.cif] [--xyz 分子.xyz]"""
    import argparse
    ap = argparse.ArgumentParser(description="运行 Mobilis 流程 (*.mflow)")
    ap.add_argument("flow")
    ap.add_argument("workdir")
    ap.add_argument("--cif")
    ap.add_argument("--xyz")
    a = ap.parse_args(argv)
    _title, stages = load_flow(a.flow)
    ctx = {}
    if a.cif:
        from .crystal import find_molecules, read_cif
        crys = read_cif(a.cif)
        ctx["crystal"] = (crys, find_molecules(crys))
    if a.xyz:
        from .gaussian_io import read_geometry
        ctx["molecule"] = (read_geometry(a.xyz), Path(a.xyz).stem)
    for p in check_flow(stages):
        print("[提示]", p)
    r = FlowRunner(stages, a.workdir, ctx, on_log=print,
                   on_status=lambda n, st, m: st in ("failed",) and print(f"阶段 {n + 1} 失败: {m}"))
    return 0 if r.run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
