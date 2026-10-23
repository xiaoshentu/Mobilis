"""迁移率方法对比：同一套输入 (近邻图中的 J、重组能 λ、能量无序 σ、耦合涨落 σ_J) 用不同的输运模型计算迁移率，
并按输运区间 (2J/λ、σ/kT、σ_J/J、Ioffe–Regel 极限) 给出各方法的适用性判断。

方法
  marcus_scalar   Marcus 速率 + 文献常用标量公式 D = (1/2n) Σ r² k P，P = k / Σk
  marcus_tensor   Marcus 速率 + 无规行走严格张量 D_ab = ½ Σ k r_a r_b (位点等价)
  kmc_marcus      Marcus 速率的动力学蒙特卡洛 (零场扩散张量)
  kmc_marcus_dis  同上，加位点能高斯无序 σ (Boltzmann 平衡起点)
  kmc_polar       同上，位点能取极化模型 (polar)：两套独立分子的位点能差 + MD 快照的热涨落 σ_E
  kmc_mlj         Marcus–Levich–Jortner 速率 (一个有效量子模 ħω，经典部分 λ_s) 的 KMC
  kmc_miller      Miller–Abrahams 速率的 KMC (ν0 取该对的 Marcus 零偏速率；只有 σ > 0 时与 Marcus 不同)
  gdm             Bässler 高斯无序模型的零场解析估计 μ = μ(σ=0) exp[−(2σ/3kT)²]
  kmc_field       Marcus 速率的 KMC 场迁移率 μ = <v>·F̂/|F| (类似飞行时间法)
  tlt             瞬态局域化理论 (热涨落 σ_J，弛豫时间 ħ/τ)
  band            紧束缚能带 + 常弛豫时间 μ = eτ/m*
外部结果 (例如 CEDS 的局域 MC / 全晶体 MD) 可作为附加行一并列出。
"""
import time

import numpy as np

from .mobility import HBAR_EVS, KB_EV

METHOD_NAMES = {
    "marcus_scalar": "Marcus 标量公式 (文献常用)",
    "marcus_tensor": "Marcus 无规行走张量 (解析)",
    "kmc_marcus": "Marcus KMC (零场扩散)",
    "kmc_marcus_dis": "Marcus KMC + 能量无序 σ",
    "kmc_polar": "Marcus KMC + 极化位点能",
    "kmc_mlj": "Marcus–Levich–Jortner KMC",
    "kmc_miller": "Miller–Abrahams KMC + 能量无序 σ",
    "gdm": "Bässler GDM 解析估计",
    "kmc_field": "Marcus KMC 场迁移率",
    "tlt": "瞬态局域化理论 (TLT)",
    "band": "能带 + 常弛豫时间",
}
FAMILY = {"marcus_scalar": "跳跃", "marcus_tensor": "跳跃", "kmc_marcus": "跳跃", "kmc_marcus_dis": "跳跃 + 静态无序",
          "kmc_polar": "跳跃 + 静电位点能",
          "kmc_mlj": "跳跃 (量子振动)", "kmc_miller": "跳跃 + 静态无序", "gdm": "跳跃 + 静态无序",
          "kmc_field": "跳跃", "tlt": "瞬态局域化", "band": "能带"}
DEFAULT_METHODS = tuple(METHOD_NAMES)
E_HBAR = 1.0 / HBAR_EVS                  # e/ħ in 1/(V·s)


def _rate(j, lam, T):
    return 2 * np.pi / HBAR_EVS * j ** 2 / np.sqrt(4 * np.pi * lam * KB_EV * T) * np.exp(-lam / (4 * KB_EV * T))


def marcus_tensor(graph, lam, T):
    """位点等价时的零场严格解 μ_ab = (1/n) Σ k r_a r_b / (2kT) (cm²/Vs)。"""
    mu = np.zeros((3, 3))
    for (_, _, v, j, m, _t) in graph["edges"]:
        vc = np.asarray(v, float) * 1e-8
        mu += 0.5 * _rate(j, lam, T) * m * np.outer(vc, vc)
    return mu / (KB_EV * T) / graph["n_sites"]


def marcus_scalar(graph, lam, T, dim=3):
    """文献常用标量公式：每个位点 D = (1/2n) Σ_i r_i² k_i P_i，P_i = k_i/Σk，再对位点平均 (cm²/Vs)。"""
    Ds = []
    for s in range(graph["n_sites"]):
        es = [e for e in graph["edges"] if e[0] == s]
        k = np.array([_rate(e[3], lam, T) * e[4] for e in es])
        r2 = np.array([np.dot(e[2], e[2]) * 1e-16 for e in es])
        if k.sum() <= 0:
            continue
        Ds.append((r2 * k * (k / k.sum())).sum() / (2 * dim))
    return float(np.mean(Ds) / (KB_EV * T)) if Ds else float("nan")


def keep_axes(graph):
    w = np.zeros(3)
    for e in graph["edges"]:
        if e[5] is not None:
            w += (np.array(e[5]) != 0) * e[3] ** 2
    return tuple(sorted(int(i) for i in np.argsort(w)[::-1][:2]))


def regime(graph, lam, T, sigma_E=0.0, sigma_J=None):
    """输运区间指标：J_max、2J_max/λ、σ/kT、σ_J/|J| (按 J² 加权)。"""
    J = np.array([abs(e[3]) for e in graph["edges"]])
    jmax = float(J.max()) if len(J) else 0.0
    # 无定形形貌中 J 分布很宽，个别大 J 不代表整体：用 90% 分位数判断区间
    jref = float(np.percentile(J, 90)) if (len(J) and graph.get("kind") == "amorphous") else jmax
    out = {"J_max_meV": 1000 * jmax, "lam_eV": lam, "ratio_2J_lam": 2 * jref / lam if lam > 0 else float("inf"),
           "sigma_kT": sigma_E / (KB_EV * T), "kT_meV": 1000 * KB_EV * T}
    if isinstance(sigma_J, dict) and sigma_J.get("edges") is not None:
        s = np.asarray(sigma_J["edges"], float)
        w = J ** 2
        out["sigmaJ_rel"] = float(np.sum(w * s / np.maximum(J, 1e-12)) / max(w.sum(), 1e-30))
    elif sigma_J is not None and not isinstance(sigma_J, dict):
        out["sigmaJ_rel"] = float(sigma_J)
    return out


def _verdict(key, reg, mu_band=None, a_A=None):
    """返回 (适用 / 边界 / 不适用 / 对照, 说明)。"""
    r = reg["ratio_2J_lam"]
    s = reg.get("sigma_kT", 0.0)
    sj = reg.get("sigmaJ_rel")
    if key in ("marcus_scalar", "marcus_tensor", "kmc_marcus", "kmc_field", "kmc_mlj", "kmc_marcus_dis", "kmc_polar"):
        if r < 0.3:
            v, why = "适用", f"2J/λ = {r:.2f} ≪ 1，非绝热小极化子跳跃成立"
        elif r < 1.0:
            v, why = "边界", f"2J/λ = {r:.2f}，J 与 λ 相当，跳跃图像只作参考 (常低估迁移率)"
        else:
            v, why = "不适用", f"2J/λ = {r:.2f} > 1，电荷离域于多个分子，跳跃模型失效"
        if key == "kmc_marcus" and r < 1.0:
            why += "；扩散张量应与解析张量一致 (检验 KMC)"
        if key == "kmc_mlj":
            why += "；高频分子内振动按量子处理，低温下比 Marcus 可靠"
        if key == "kmc_marcus_dis" and s > 0:
            why += f"；σ/kT = {s:.1f}"
        if key == "kmc_polar":
            why += "；位点能由原子电荷 + Thole 极化计算"
        return v, why
    if key in ("kmc_miller", "gdm"):
        if s <= 0:
            return "对照", "σ = 0 时与 Marcus 相同"
        v = "适用" if s >= 2 else "边界"
        why = f"σ/kT = {s:.1f}；唯象模型，适合静态无序主导的无定形薄膜，晶体中只作对照"
        if key == "gdm":
            why += "；解析式基于三维立方格子 (Bässler 1993)"
        return v, why
    if key == "tlt":
        if r < 0.3:
            return "边界", f"2J/λ = {r:.2f}：电荷强局域，跳跃模型更合适"
        if sj is None or sj <= 0:
            return "边界", "未给出耦合涨落 σ_J"
        return "适用", f"2J/λ = {r:.2f}、σ_J/|J| ≈ {sj:.2f}：分子间热涨落主导，适合高迁移率晶体"
    if key == "band":
        if sj is not None and sj > 0.15:
            return "不适用", f"σ_J/|J| ≈ {sj:.2f}，热涨落破坏布洛赫态 (能带图像失效)"
        if mu_band is not None and a_A:
            mu_ir = E_HBAR * (a_A * 1e-8) ** 2
            if mu_band < 3 * mu_ir:
                return "不适用", f"μ 低于 Ioffe–Regel 极限的 3 倍 (e a²/ħ ≈ {mu_ir:.1f} cm²/Vs)，平均自由程短于晶格间距"
        return "边界", "需要 σ_J ≪ |J| 且 μ ≫ e a²/ħ；室温有机晶体很少满足"
    return "", ""


def compare_methods(graph, lam, T=300.0, carrier="hole", methods=DEFAULT_METHODS, sigma_E=0.0, sigma_J=0.3,
                    hbar_tau=0.005, mlj_hw=0.18, mlj_lam_s=None, field=1e4, walkers=400, steps=2500, supercell=4,
                    n_real=12, lattice=None, dim=None, seed=1, extra=(), on_progress=None, polar=None):
    """graph：kmc.graph_from_crystal / graph_from_paths / CEDS.graph 的近邻图 (J 为 eV，可带符号)；lam：重组能 (eV)。
    sigma_J：TLT 的耦合涨落，标量 = 相对值 σ_J/|J|，或 {"edges": 与 graph["edges"] 对齐的 σ (eV)}。
    mlj_lam_s：MLJ 的经典部分 λ_s (默认 0.3λ)，量子部分 λ − λ_s 由一个有效模 ħω 承担。
    extra：附加结果 [{name, mu, mu2?, family?, note?, valid?}] (例如 CEDS)。
    polar：极化模型的位点能 {"offsets": 各晶胞位点的位点能 (eV), "sigma": 热涨落 σ_E (eV)}，用于 kmc_polar。
    无定形形貌 (graph["kind"] == "amorphous"，整个周期盒子为一个晶胞) 不计算依赖平移对称性的方法
    (标量公式、解析张量、TLT、能带)，GDM 的 μ(σ=0) 改用 KMC。
    返回 {"rows", "regime", "inputs"}。"""
    from .kmc import run_kmc
    from .transport import bond_index, band_mobility, effective_mass, tlt_mobility
    rows = []
    reg = regime(graph, lam, T, sigma_E, sigma_J)
    keep = keep_axes(graph)
    dim = dim or (2 if graph.get("kind") == "crystal" else 3)
    lam_s = mlj_lam_s if mlj_lam_s is not None else 0.3 * lam
    dist = [np.linalg.norm(e[2]) for e in graph["edges"] if abs(e[3]) == max(abs(x[3]) for x in graph["edges"])]
    a_A = float(np.mean(dist)) if dist else None
    periodic = bool(graph.get("periodic"))

    def add(key, mu, mu2=float("nan"), err=float("nan"), note="", t0=None, **kw):
        v, why = _verdict(key, reg, kw.get("mu_band"), a_A)
        rows.append({"key": key, "name": METHOD_NAMES[key], "family": FAMILY[key], "mu": float(mu), "mu2": float(mu2),
                     "err": float(err), "note": note, "valid": v, "why": why,
                     "seconds": time.time() - t0 if t0 else 0.0})

    def prog(k):
        if on_progress:
            on_progress(METHOD_NAMES[k])

    amorphous = graph.get("kind") == "amorphous"
    kmc_kw = dict(carrier=carrier, supercell=1 if amorphous else supercell, walkers=walkers, steps=steps, seed=seed)
    mu0_kmc = {}

    def mu_zero():
        """σ = 0 的参考迁移率：晶体用解析张量；无定形形貌用 KMC (没有平移对称性，解析张量不成立)。"""
        if not amorphous:
            return np.linalg.eigvalsh(marcus_tensor(graph, lam, T))[::-1]
        if "w" not in mu0_kmc:
            mu0_kmc["w"] = run_kmc(graph, lam, T, method="marcus", **kmc_kw)["mu_principal"]
        return mu0_kmc["w"]
    for key in methods:
        t0 = time.time()
        if amorphous and key in ("marcus_scalar", "marcus_tensor", "tlt", "band"):
            continue
        prog(key)
        try:
            if key == "marcus_scalar":
                add(key, marcus_scalar(graph, lam, T, dim), note=f"n = {dim} 维", t0=t0)
            elif key == "marcus_tensor":
                w = np.linalg.eigvalsh(marcus_tensor(graph, lam, T))[::-1]
                add(key, w[0], w[1], t0=t0)
            elif key == "kmc_marcus":
                r = run_kmc(graph, lam, T, method="marcus", **kmc_kw)
                add(key, r["mu_principal"][0], r["mu_principal"][1], r["mu_avg_err"] * 3, t0=t0)
            elif key == "kmc_marcus_dis":
                if sigma_E <= 0 or not periodic:
                    continue
                r = run_kmc(graph, lam, T, method="marcus", sigma=sigma_E, **kmc_kw)
                add(key, r["mu_principal"][0], r["mu_principal"][1], r["mu_avg_err"] * 3, note=f"σ = {sigma_E * 1000:.0f} meV",
                    t0=t0)
            elif key == "kmc_polar":
                if not polar or not periodic:
                    continue
                offs = np.asarray(polar["offsets"], float)
                sp = float(polar.get("sigma") or 0.0)
                r = run_kmc(graph, lam, T, method="marcus", sigma=sp, site_offsets=offs, **kmc_kw)
                note = (f"逐个分子的位点能 (σ = {1000 * offs.std():.0f} meV，含空间关联)" if amorphous else
                        f"子格子位点能差 {1000 * np.ptp(offs):.0f} meV，σ_E = {1000 * sp:.0f} meV")
                add(key, r["mu_principal"][0], r["mu_principal"][1], r["mu_avg_err"] * 3, note=note, t0=t0)
            elif key == "kmc_mlj":
                r = run_kmc(graph, max(lam - lam_s, 0.0), T, method="jortner", lam_out=lam_s, hw=mlj_hw, **kmc_kw)
                add(key, r["mu_principal"][0], r["mu_principal"][1], r["mu_avg_err"] * 3,
                    note=f"ħω = {mlj_hw * 1000:g} meV，λ_s = {lam_s * 1000:.0f} meV，S = {(lam - lam_s) / mlj_hw:.2f}", t0=t0)
            elif key == "kmc_miller":
                if sigma_E <= 0 or not periodic:
                    continue
                r = run_kmc(graph, lam, T, method="miller", sigma=sigma_E, **kmc_kw)
                add(key, r["mu_principal"][0], r["mu_principal"][1], r["mu_avg_err"] * 3, note=f"σ = {sigma_E * 1000:.0f} meV",
                    t0=t0)
            elif key == "gdm":
                if sigma_E <= 0:
                    continue
                w = mu_zero()
                f = np.exp(-(2 * sigma_E / (3 * KB_EV * T)) ** 2)
                add(key, w[0] * f, w[1] * f, note=f"exp[−(2σ/3kT)²] = {f:.3g}", t0=t0)
            elif key == "kmc_field":
                if not field:
                    continue
                Fv = np.zeros(3)
                if amorphous:
                    Fv[0] = float(field)
                else:
                    ev, evec = np.linalg.eigh(marcus_tensor(graph, lam, T))
                    Fv = evec[:, -1] * float(field)
                r = run_kmc(graph, lam, T, method="marcus", field=Fv, **kmc_kw)
                add(key, r.get("mu_field_cm2_Vs", float("nan")), err=3 * r.get("mu_field_err", float("nan")),
                    note=f"|F| = {field:g} V/cm，" + ("沿 x" if amorphous else "沿解析张量最大主轴"), t0=t0)
            elif key == "tlt":
                if graph.get("kind") != "crystal" and not periodic:
                    continue
                L = max(8, int(round(np.sqrt(600 / graph["n_sites"]))))
                sizes = tuple(L if a in keep else 1 for a in range(3))
                if isinstance(sigma_J, dict):
                    idx = bond_index(graph, keep)
                    sig = {k: float(sigma_J["edges"][e]) for k, e in enumerate(idx)}
                else:
                    sig = float(sigma_J)
                r = tlt_mobility(graph, T, carrier, sig, hbar_tau, sizes, keep, n_real)
                add(key, r["mu_principal"][0], r["mu_principal"][1] if len(r["mu_principal"]) > 1 else float("nan"),
                    note=f"ħ/τ = {hbar_tau * 1000:g} meV，L ≈ {r['L_A']:.0f} Å", t0=t0)
            elif key == "band":
                em = effective_mass(graph, carrier, lattice, n_grid=16 if graph["n_sites"] > 2 else 24)
                mus = sorted([m for m in band_mobility(em["masses_me"], hbar_tau) if np.isfinite(m)], reverse=True)
                if not mus:
                    continue
                add(key, mus[0], mus[1] if len(mus) > 1 else float("nan"),
                    note=f"带宽 {em['bandwidth_eV'] * 1000:.0f} meV，τ = ħ/({hbar_tau * 1000:g} meV)", t0=t0, mu_band=mus[0])
        except Exception as exc:  # noqa: BLE001
            rows.append({"key": key, "name": METHOD_NAMES[key], "family": FAMILY[key], "mu": float("nan"),
                         "mu2": float("nan"), "err": float("nan"), "note": f"失败: {exc}", "valid": "", "why": "",
                         "seconds": time.time() - t0})
    for x in extra:
        rows.append({"key": x.get("key", "extra"), "name": x["name"], "family": x.get("family", "其它"),
                     "mu": float(x["mu"]), "mu2": float(x.get("mu2", float("nan"))), "err": float(x.get("err", float("nan"))),
                     "note": x.get("note", ""), "valid": x.get("valid", ""), "why": x.get("why", ""), "seconds": 0.0})
    return {"rows": rows, "regime": reg,
            "inputs": {"T": T, "carrier": carrier, "lam": lam, "sigma_E": sigma_E,
                       "sigma_J": "按键给出" if isinstance(sigma_J, dict) else sigma_J, "hbar_tau": hbar_tau,
                       "mlj_hw": mlj_hw, "mlj_lam_s": lam_s, "field": field, "walkers": walkers, "steps": steps}}


def table_text(result_by_carrier):
    """{carrier: compare_methods 的结果} → 文字表。"""
    lines = []
    for car, res in result_by_carrier.items():
        reg = res["regime"]
        lines.append(f"[{'空穴' if car == 'hole' else '电子'}]  T = {res['inputs']['T']:g} K，λ = {reg['lam_eV'] * 1000:.0f} meV，"
                     f"J_max = {reg['J_max_meV']:.1f} meV，2J/λ = {reg['ratio_2J_lam']:.2f}"
                     + (f"，σ/kT = {reg['sigma_kT']:.1f}" if reg['sigma_kT'] > 0 else "")
                     + (f"，σ_J/|J| ≈ {reg['sigmaJ_rel']:.2f}" if reg.get('sigmaJ_rel') else ""))
        lines.append(f"{'方法':<30}{'μ_max':>10}{'μ_2':>10}{'±':>8}  {'适用性':<6}  说明")
        for r in res["rows"]:
            mu = "—" if not np.isfinite(r["mu"]) else f"{r['mu']:.3g}"
            mu2 = "" if not np.isfinite(r["mu2"]) else f"{r['mu2']:.3g}"
            er = "" if not np.isfinite(r["err"]) else f"{r['err']:.2g}"
            lines.append(f"{r['name']:<30}{mu:>10}{mu2:>10}{er:>8}  {r['valid']:<6}  {r['why'] or r['note']}"
                         + (f" ({r['note']})" if r['why'] and r['note'] else ""))
        lines.append("")
    lines.append("μ 单位 cm²V⁻¹s⁻¹，为迁移率张量的最大主值 (μ_2 为第二主值)；± 为 KMC 统计误差的 3 倍。")
    return "\n".join(lines)


def export_csv(result_by_carrier, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["carrier", "method_key", "method", "family", "mu_max_cm2_Vs", "mu_2", "err_3sigma", "applicability",
                    "reason", "note", "seconds"])
        for car, res in result_by_carrier.items():
            for r in res["rows"]:
                w.writerow([car, r["key"], r["name"], r["family"], f"{r['mu']:.6g}", f"{r['mu2']:.6g}", f"{r['err']:.3g}",
                            r["valid"], r["why"], r["note"], f"{r['seconds']:.1f}"])
        w.writerow([])
        w.writerow(["carrier", "T_K", "lambda_eV", "J_max_meV", "2J_over_lambda", "sigma_over_kT", "sigmaJ_rel"])
        for car, res in result_by_carrier.items():
            g = res["regime"]
            w.writerow([car, res["inputs"]["T"], g["lam_eV"], f"{g['J_max_meV']:.3f}", f"{g['ratio_2J_lam']:.4f}",
                        f"{g['sigma_kT']:.3f}", g.get("sigmaJ_rel", "")])
    return path


def plot_comparison(ax, result_by_carrier):
    """水平条形图 (对数坐标)：每个方法一行，空穴与电子两组。"""
    from .scananalysis import COL
    names = []
    for res in result_by_carrier.values():
        for r in res["rows"]:
            if r["name"] not in names:
                names.append(r["name"])
    y = np.arange(len(names))
    cars = list(result_by_carrier)
    h = 0.8 / max(len(cars), 1)
    for k, car in enumerate(cars):
        val = {r["name"]: r["mu"] for r in result_by_carrier[car]["rows"]}
        ok = {r["name"]: r["valid"] for r in result_by_carrier[car]["rows"]}
        v = np.array([val.get(n, np.nan) for n in names])
        yy = y + (k - (len(cars) - 1) / 2) * h
        m = np.isfinite(v) & (v > 0)
        alpha = np.array([0.35 if ok.get(n) in ("不适用",) else 1.0 for n in names])
        for i in np.nonzero(m)[0]:
            ax.barh(yy[i], v[i], h * 0.9, color=COL[car], alpha=alpha[i])
        ax.barh([np.nan], [np.nan], color=COL[car], label="空穴" if car == "hole" else "电子")
    allv = [r["mu"] for res in result_by_carrier.values() for r in res["rows"] if np.isfinite(r["mu"]) and r["mu"] > 0]
    if allv:
        ax.set_xlim(min(allv) / 2.5, max(allv) * 1.6)
    from .i18n import tr
    ax.set_yticks(y, [tr(n) for n in names])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("μ (cm$^2$ V$^{-1}$ s$^{-1}$)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="x", color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)


def mlj_from_modes(mode_sets, cutoff_cm=1000.0):
    """由重组能分解的各模式 (分解页 λ1 / λ2 的 modes：freq_cm-1、lambda_meV) 得到 MLJ 参数：
    低于 cutoff 的模式按经典处理 (占总 λ 的比例 f_low)，高于 cutoff 的模式合并为一个有效量子模
    ħω_eff = Σ λ_i ω_i / Σ λ_i。返回 (ħω_eff / eV, f_low)。"""
    lo = hi = hiw = 0.0
    for modes in mode_sets:
        for m in modes:
            f, lm = float(m["freq_cm-1"]), float(m["lambda_meV"])
            if f <= 0 or lm <= 0:
                continue
            if f < cutoff_cm:
                lo += lm
            else:
                hi += lm
                hiw += lm * f
    if hi <= 0:
        raise ValueError("没有高频模式的重组能贡献")
    return hiw / hi * 1.239841984e-4, lo / (lo + hi)
