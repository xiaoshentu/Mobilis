"""仿 VOTCA-CTP 的电荷输运模拟：近邻图 + 速率 (Marcus / Jortner / Miller–Abrahams) + 动力学蒙特卡洛 (KMC)。

  速率 (VOTCA 手册式 2.31, 2.34):
    Marcus : ω = 2π/ħ J² (4πλkT)^(-1/2) exp[-(ΔE + λ)² / 4λkT]，λ = λ_int + λ_out
    Jortner: ω = 2π/ħ J² (4πλ_out kT)^(-1/2) Σ_N e^(-S) S^N/N! exp[-(ΔE + Nħω + λ_out)² / 4λ_out kT]，S = λ_int/ħω
    Miller–Abrahams: ω = ν0 · min(1, exp(−ΔE/kT))；ν0 取该对的 Marcus 零偏速率 (无序为零时与 Marcus 相同)
  外重组能: 常数，或球模型 (式 2.3) λ_out = c_p e²/(4πε0) (1/2R_i + 1/2R_j − 1/r_ij)
  位点能差: ΔE = E_j − E_i (高斯无序 σ) − q F·r_ij (外电场)
  零场: 扩散张量 D_ab = <Δr_a Δr_b>/(2t)，μ = D/kT (式 2.44, 2.45)；有场: μ = <v>·F̂/|F| (式 2.46)
"""
import numpy as np

from .mobility import HBAR_EVS, KB_EV

COULOMB_EV_A = 14.3996454784   # e²/(4πε0) in eV·Å


def rate_marcus(j, lam, dE, T):
    kt = KB_EV * T
    return 2 * np.pi / HBAR_EVS * j ** 2 / np.sqrt(4 * np.pi * lam * kt) * np.exp(-(dE + lam) ** 2 / (4 * lam * kt))


def rate_jortner(j, lam_int, lam_out, dE, T, hw=0.2, nmax=20):
    from math import factorial
    kt = KB_EV * T
    lam_out = max(lam_out, 1e-4)
    s = lam_int / hw
    tot = 0.0
    for n in range(nmax + 1):
        tot = tot + np.exp(-s) * s ** n / factorial(n) * np.exp(-(dE + n * hw + lam_out) ** 2 / (4 * lam_out * kt))
    return 2 * np.pi / HBAR_EVS * j ** 2 / np.sqrt(4 * np.pi * lam_out * kt) * tot


def lambda_out_spheres(pekar, r_i, r_j, r_ij):
    return COULOMB_EV_A * pekar * (1 / (2 * r_i) + 1 / (2 * r_j) - 1 / np.asarray(r_ij))


# ----------------------------------------------------------------------------------------------
# 近邻图
# ----------------------------------------------------------------------------------------------
def graph_from_paths(paths, key):
    """布拉维格子 (每个晶胞一个分子)。paths: [{vector 或 r/theta/gamma, key: J(eV), count}]。
    缺少 −r 的路径自动补上反向跳跃（细致平衡），数目在 ±r 间平分。"""
    vecs, js = [], []
    for p in paths:
        if p.get(key) in (None, ""):
            continue
        if p.get("vector") is not None:
            v = np.asarray(p["vector"], float)
        else:
            th, ga = np.radians(p.get("theta", 0)), np.radians(p.get("gamma", 0))
            v = p["r"] * np.array([np.cos(ga) * np.cos(th), np.cos(ga) * np.sin(th), np.sin(ga)])
        vecs.append((v, float(p[key]), int(p.get("count", 1))))
    edges = []
    for v, j, c in vecs:
        has_neg = any(np.allclose(-v, w, atol=1e-3) for w, _, _ in vecs)
        if has_neg:
            edges.append([0, 0, v, j, float(c), None])
        else:
            edges += [[0, 0, v, j, c / 2, None], [0, 0, -v, j, c / 2, None]]
    basis = _lattice_basis([e[2] for e in edges])
    ok = basis is not None
    if ok:
        bm = np.array(basis)
        for e in edges:
            coef, *_ = np.linalg.lstsq(bm.T, e[2], rcond=None)
            t = np.round(coef).astype(int)
            if np.abs(coef - t).max() > 0.05 or np.linalg.norm(bm.T @ t - e[2]) > 0.05:
                ok = False
                break
            e[5] = tuple(t) + (0,) * (3 - len(t))
    if not ok:
        for e in edges:
            e[5] = None
    return {"n_sites": 1, "edges": [tuple(e) for e in edges], "kind": "bravais", "periodic": ok}


def _lattice_basis(vectors, tol=1e-3):
    """从跳跃矢量中选出最短的线性无关矢量作为格矢基 (1–3 个)。"""
    vs = sorted((np.asarray(v, float) for v in vectors), key=np.linalg.norm)
    basis = []
    for v in vs:
        if np.linalg.norm(v) < tol:
            continue
        trial = basis + [v]
        if np.linalg.matrix_rank(np.array(trial), tol=1e-3 * np.linalg.norm(v)) == len(trial):
            basis.append(v)
        if len(basis) == 3:
            break
    return basis or None


def graph_from_crystal(crys, mols, j_by_group, cutoff=4.0, criterion="atom", carrier="hole", signed=False):
    """晶胞内全部分子作为位点；对每个分子搜索近邻，按等价二聚体组 (与以分子 1 为参考导出的组号一致) 赋 J。"""
    from .crystal import find_dimers
    base = find_dimers(crys, mols, ref=0, cutoff=cutoff, criterion=criterion)
    fps = [(g["id"], g["fingerprint"]) for g in base["groups"]]
    edges, missing = [], set()
    for ref in range(len(mols)):
        res = base if ref == 0 else find_dimers(crys, mols, ref=ref, cutoff=cutoff, criterion=criterion)
        for d in res["dimers"]:
            gid = next((g for g, fp in fps if len(fp) == len(d["fingerprint"])
                        and np.abs(fp - d["fingerprint"]).max() < 0.03), None)
            j = (j_by_group.get(gid) or {}).get(carrier) if gid is not None else None
            if j is None:
                missing.add(gid)
                continue
            edges.append((ref, d["mol"], np.asarray(d["vector"]), (float(j) if signed else abs(float(j))) / 1000.0, 1.0,
                          tuple(d["translation"])))
    return {"n_sites": len(mols), "edges": edges, "kind": "crystal", "missing_groups": sorted(missing, key=str),
            "n_groups": len(fps), "periodic": True}


# ----------------------------------------------------------------------------------------------
# KMC
# ----------------------------------------------------------------------------------------------
def run_kmc(graph, lam_int, T=300.0, method="marcus", lam_out=0.0, lam_out_model="constant", pekar=0.05,
            radius=5.0, hw=0.2, field=(0.0, 0.0, 0.0), carrier="hole", sigma=0.0, supercell=4, walkers=2000,
            steps=3000, seed=1, n_record=40, burn_in=None, start="boltzmann", estimator=None, site_offsets=None):
    """返回扩散张量、迁移率张量、场迁移率、MSD 曲线等。field 单位 V/cm；sigma 为位点能高斯无序 (eV)；
    site_offsets 为晶胞中各位点 (子格子) 的确定位点能 (eV，载流子能量，例如极化模型给出的两套分子之差)，可与 sigma 叠加。
    有能量无序时，载流子按 Boltzmann 分布 p_i ∝ exp(−E_i/kT) 放置，并先运行 burn_in 步 (默认 steps 的 20%)
    达到平衡后再开始统计，避免初始高能暂态使迁移率偏高。"""
    rng = np.random.default_rng(seed)
    q = 1.0 if carrier == "hole" else -1.0
    F = np.asarray(field, float) * 1e-8                       # V/cm → V/Å
    nbase = graph["n_sites"]
    use_cells = sigma > 0
    if use_cells and not graph.get("periodic"):
        raise ValueError("这些跳跃矢量不构成周期格子，无法放置位点能无序；请用晶体近邻图或给出完整的三维矢量")
    N = supercell if use_cells else 1
    cells = [(i, j, k) for i in range(N) for j in range(N) for k in range(N)]
    cidx = {c: n for n, c in enumerate(cells)}
    nsite = nbase * len(cells)
    energies = rng.normal(0.0, sigma, nsite) if use_cells else np.zeros(nsite)
    offs = None
    if site_offsets is not None:
        offs = np.asarray(site_offsets, float)
        if offs.shape != (nbase,):
            raise ValueError(f"site_offsets 应有 {nbase} 个值 (每个晶胞位点一个)")
        energies = energies + np.tile(offs - offs.mean(), len(cells))
    out = [[] for _ in range(nsite)]
    for c in cells:
        for (i, j, v, jv, mult, t) in graph["edges"]:
            si = i + nbase * cidx[c]
            tc = tuple((c[x] + t[x]) % N for x in range(3)) if use_cells else c
            out[si].append((j + nbase * cidx[tc], v, jv, mult))
    deg = max(len(o) for o in out)
    tgt = np.zeros((nsite, deg), int)
    disp = np.zeros((nsite, deg, 3))
    rate = np.zeros((nsite, deg))
    kt = KB_EV * T
    for si, lst in enumerate(out):
        for k, (sj, v, jv, mult) in enumerate(lst):
            dE = energies[sj] - energies[si] - q * F @ v
            lo = lam_out if lam_out_model == "constant" else float(lambda_out_spheres(pekar, radius, radius,
                                                                                       np.linalg.norm(v)))
            if method == "jortner":
                w = rate_jortner(jv, lam_int, lo, dE, T, hw)
            elif method == "miller":
                w = rate_marcus(jv, lam_int + lo, 0.0, T) * (np.exp(-dE / kt) if dE > 0 else 1.0)
            else:
                w = rate_marcus(jv, lam_int + lo, dE, T)
            tgt[si, k], disp[si, k], rate[si, k] = sj, v, w * mult
    ktot = rate.sum(1)
    if np.any(ktot <= 0):
        raise ValueError("存在没有出射跳跃的位点（检查 J 是否都为零）")
    cum = np.cumsum(rate, 1) / ktot[:, None]
    disordered = sigma > 0 or (offs is not None and np.ptp(offs) > 0)
    if disordered and start == "boltzmann":          # 平衡起点：Boltzmann 分布
        pe = np.exp(-(energies - energies.min()) / kt)
        site = rng.choice(nsite, size=walkers, p=pe / pe.sum())
    else:
        site = rng.integers(0, nsite, walkers)
    nburn = (steps // 5 if burn_in is None else int(burn_in)) if (disordered or np.linalg.norm(F) > 0) else 0
    for _ in range(nburn):                           # 平衡期：只更新位点，不计位移与时间
        u = rng.random(walkers)
        k = np.minimum((cum[site] < u[:, None]).sum(1), deg - 1)
        site = tgt[site, k]
    pos = np.zeros((walkers, 3))
    t = np.zeros(walkers)
    estimator = estimator or ("time" if disordered else "steps")
    if estimator == "time":
        # 固定模拟时间：所有载流子模拟到同一时刻 t_max，D = <r²>/(2 t_max)，即实验 (系综) 意义上的扩散系数。
        # t_max 取“平均每载流子 steps 次跳跃”所需时间 (按平衡态平均跳跃时间估计)。
        pe = np.exp(-(energies - energies.min()) / kt)
        pe /= pe.sum()
        t_max = steps / float(pe @ ktot)
        active = np.ones(walkers, bool)
        tk = t_max * np.arange(1, n_record + 1) / n_record           # 记录 MSD 的时刻
        tk[-1] = t_max                                               # 避免浮点舍入使最后一点略大于 t_max 而漏记
        msd_sum = np.zeros(n_record)
        nxt = np.zeros(walkers, int)                                 # 每个载流子下一个待记录时刻的序号

        def record(idx, t_new, p_hold):
            """t_new 之前的检查点处，载流子仍停在 p_hold。"""
            while len(idx):
                ok = (nxt[idx] < n_record)
                ok[ok] = tk[nxt[idx[ok]]] <= t_new[ok]
                if not ok.any():
                    break
                ii = idx[ok]
                np.add.at(msd_sum, nxt[ii], (p_hold[ok] ** 2).sum(1))
                nxt[ii] += 1
                idx, t_new, p_hold = ii, t_new[ok], p_hold[ok]
        n_iter = 0
        while active.any() and n_iter < 50 * steps:
            n_iter += 1
            idx = np.where(active)[0]
            dt = -np.log(rng.random(len(idx))) / ktot[site[idx]]
            go = t[idx] + dt <= t_max
            stop = idx[~go]
            record(stop, np.full(len(stop), t_max), pos[stop].copy())
            active[stop] = False
            mv = idx[go]
            record(mv, t[mv] + dt[go], pos[mv].copy())
            u = rng.random(len(mv))
            k = np.minimum((cum[site[mv]] < u[:, None]).sum(1), deg - 1)
            t[mv] += dt[go]
            pos[mv] += disp[site[mv], k]
            site[mv] = tgt[site[mv], k]
        t[:] = t_max
        steps_done = n_iter
        rec_t, rec_msd = list(tk), list(msd_sum / walkers)
    rec_every = max(1, steps // n_record)
    if estimator == "steps":
        rec_t, rec_msd = [], []
    for step in range(1, (steps if estimator == "steps" else 0) + 1):
        u = rng.random(walkers)
        k = (cum[site] < u[:, None]).sum(1)
        k = np.minimum(k, deg - 1)
        t += -np.log(rng.random(walkers)) / ktot[site]
        pos += disp[site, k]
        site = tgt[site, k]
        if step % rec_every == 0:
            rec_t.append(t.mean())
            rec_msd.append((pos ** 2).sum(1).mean())
    tt = t.sum()
    r_cm = pos * 1e-8
    D = (r_cm[:, :, None] * r_cm[:, None, :]).sum(0) / (2 * tt)          # 用总时间加权的无偏估计
    if np.linalg.norm(F) > 0:
        drift = r_cm.sum(0) / tt                                          # cm/s
        D_c = ((r_cm - drift * t[:, None])[:, :, None] * (r_cm - drift * t[:, None])[:, None, :]).sum(0) / (2 * tt)
        D = D_c
    mu = D / kt
    # 统计误差：按 20 组独立轨迹分块估计
    nb = 20 if walkers >= 40 else 2
    blocks = np.array_split(np.arange(walkers), nb)
    mus = []
    for bl in blocks:
        rb = r_cm[bl]
        if np.linalg.norm(F) > 0:
            rb = rb - (r_cm.sum(0) / tt) * t[bl][:, None]
        mus.append((rb[:, :, None] * rb[:, None, :]).sum(0) / (2 * t[bl].sum()) / kt)
    mu_err = np.std(mus, axis=0, ddof=1) / np.sqrt(nb)
    w, v = np.linalg.eigh(mu)
    res = {"D_tensor_cm2_s": D, "mu_tensor_cm2_Vs": mu, "mu_principal": w[::-1], "mu_axes": v[:, ::-1].T,
           "mu_avg": float(np.trace(mu) / 3), "mu_err": mu_err, "mu_avg_err": float(np.sqrt((np.diag(mu_err) ** 2).sum()) / 3),
           "hops": walkers * steps, "sim_time_s": float(t.mean()),
           "msd_t": np.array(rec_t), "msd_A2": np.array(rec_msd), "n_sites": nsite,
           "mean_rate": float(ktot.mean()), "site_energies": energies, "burn_in": nburn, "estimator": estimator}
    if np.linalg.norm(F) > 0:
        vel = r_cm.sum(0) / tt
        fh = F / np.linalg.norm(F)
        res["velocity_cm_s"] = vel
        res["mu_field_cm2_Vs"] = float(q * vel @ fh / (np.linalg.norm(F) * 1e8))
        vb = [q * (r_cm[bl].sum(0) / t[bl].sum()) @ fh / (np.linalg.norm(F) * 1e8) for bl in blocks]
        res["mu_field_err"] = float(np.std(vb, ddof=1) / np.sqrt(nb))
    return res


def analytic_equivalent_sites(graph, lam_int, T=300.0, lam_out=0.0):
    """多分子晶胞 (各位点能量相同) 的零场精确迁移率张量，周期格子上随机行走的均匀化公式：
    先解修正位移 χ_i：Σ_j k_ij (r_ij + χ_j − χ_i) = 0 (每个位点的局部平均位移为零)，
    再 D = (1/2n) Σ_ij k_ij (r_ij + χ_j − χ_i)(r_ij + χ_j − χ_i)ᵀ，μ = D / kT。
    每个位点的跳跃矢量成 ±v 对时 χ = 0，退化为 μ = (1/n) Σ k r r / (2kT)；P2₁ 等晶体 (如 DNTT、并五苯) 的
    分子间跳跃不成对，必须用这个一般式。"""
    kt = KB_EV * T
    n = int(graph["n_sites"])
    E = [(int(i), int(jj), np.asarray(v, float) * 1e-8, rate_marcus(j, lam_int + lam_out, 0.0, T) * m)
         for (i, jj, v, j, m, _t) in graph["edges"]]
    L = np.zeros((n, n))
    b = np.zeros((n, 3))
    for i, jj, vc, k in E:
        b[i] += k * vc
        L[i, i] -= k
        L[i, jj] += k
    chi = np.linalg.lstsq(L, -b, rcond=None)[0]                 # 规范自由度 (整体平移) 由最小范数解确定
    D = np.zeros((3, 3))
    for i, jj, vc, k in E:
        u = vc + chi[jj] - chi[i]
        D += 0.5 * k * np.outer(u, u)
    return D / n / kt


def exact_1d_ring_disorder(energies, a_cm, J, lam, T, N):
    """一维环 (长 N) 上带位点能无序的精确扩散系数 (Derrida 1983)：D = a² N² / Σ_i 1/(π_i w_{i,i+1})，Σπ = 1。
    energies 按环排列 (长度为 N 的整数倍，每 N 个为一个环)。返回各环的 D 与 Boltzmann 权重。"""
    kt = KB_EV * T
    Ds, Zs = [], []
    for r in range(len(energies) // N):
        e = np.asarray(energies[r * N:(r + 1) * N])
        w = np.exp(-(e - e.min()) / kt)
        Z = w.sum()
        pi = w / Z
        c = [pi[i] * rate_marcus(J, lam, e[(i + 1) % N] - e[i], T) for i in range(N)]
        Ds.append(a_cm ** 2 * N ** 2 / np.sum(1 / np.array(c)))
        Zs.append(Z * np.exp(-e.min() / kt))
    return np.array(Ds), np.array(Zs)


def analytic_bravais(graph, lam_int, T=300.0, lam_out=0.0):
    """布拉维格子零场解析解 μ_ab = Σ k r_a r_b / (2kT)，用于检验 KMC。"""
    kt = KB_EV * T
    mu = np.zeros((3, 3))
    for (_, _, v, j, m, _t) in graph["edges"]:
        k = rate_marcus(j, lam_int + lam_out, 0.0, T) * m
        vc = np.asarray(v) * 1e-8
        mu += 0.5 * k * np.outer(vc, vc)
    return mu / kt
