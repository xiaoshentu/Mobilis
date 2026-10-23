"""主动学习耦合曲面 (CEDS 的 GPR 版本)：高斯过程回归拟合二聚体耦合 V(q)，按热运动分布选下一批 DFT 构型。

为什么
  CEDS 需要每个近邻对在热运动范围内的耦合曲面。规则网格或随机局域扫描要几百个 DFT 二聚体 (并五苯 P 对 273 个)，
  多数落在热运动很少到达的区域。这里用高斯过程 (GP) 代替多项式：
    · GP 同时给出预测值与不确定度，可以只在“热运动常到、而曲面还不确定”的地方补算；
    · 不确定度可以直接传递到 σ_V 与 ⟨V⟩ (后验抽样)，作为停止判据：σ_V 的相对不确定度 < 目标值即可停止。

做法
  1. 坐标：二聚体参数 q = (z, Δx, Δy, γ, θ, φ)，以该近邻对的晶体几何为中心、按半宽 h 缩放 (u = (q − q0)/h，角度取周期)。
     训练数据按单体点群与 A↔B 互换展开 (与 ceds.LocalSurface 相同)，只保留缩放后 |u| ≤ 1.5 的点。
  2. GP：常数均值 + 各向异性平方指数核 + 白噪声，超参数由边际似然最大化 (L-BFGS，多次随机起点)。空穴、电子各一个 GP。
  3. 选点 (批量 IMSE)：候选池 = 热运动样本中的构型 (例如 MD 快照，只需几何、不需要 DFT)；
     每次选使“热运动样本上的后验方差总和”下降最多的候选，两种载流子按各自信号方差归一后相加；
     同一批内用固定超参数的方差更新做贪心选择 (方差只依赖输入位置，不需要已知 V)。
  4. 停止判据：在热运动样本上对 GP 后验联合抽样，得到 ⟨V⟩、σ_V 的分布与置信区间。

接口与 ceds 的其它曲面相同：surface(Q, carrier) → V (meV)；coverage(Q)；describe()。
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

from .scanflow import PARAM_KEYS, apply_op
from .surrogate import CARRIERS

ANG = ("gamma", "pitch", "roll")
DEFAULT_HALF = {"z": 0.6, "dx": 0.8, "dy": 0.8, "gamma": 10.0, "pitch": 10.0, "roll": 15.0}


def _wrap(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def _q(r):
    return [float(r[k]) for k in PARAM_KEYS]


class _GP:
    """常数均值 + 各向异性平方指数核 + 白噪声。X 已缩放到 O(1)。"""

    def __init__(self, X, y, restarts=4, seed=0, min_noise=0.05):
        self.X, self.y = np.asarray(X, float), np.asarray(y, float)
        self.mu = float(self.y.mean())
        self.min_noise = min_noise
        d = self.X.shape[1]
        sd = float(self.y.std()) or 1.0
        rng = np.random.default_rng(seed)
        best = None
        starts = [np.r_[np.log(np.full(d, 0.7)), np.log(sd), np.log(max(0.02 * sd, 0.3))]]
        for _ in range(restarts):
            starts.append(np.r_[np.log(rng.uniform(0.3, 2.0, d)), np.log(sd * rng.uniform(0.5, 2.0)),
                                np.log(max(sd * rng.uniform(0.005, 0.05), 0.2))])
        for th in starts:
            try:
                r = minimize(self._nll, th, method="L-BFGS-B",
                             bounds=[(np.log(0.05), np.log(20.0))] * d + [(np.log(0.05 * sd), np.log(20 * sd)),
                                                                         (np.log(self.min_noise), np.log(0.5 * sd + 1))])
            except Exception:  # noqa: BLE001
                continue
            if best is None or r.fun < best.fun:
                best = r
        self.theta = best.x if best is not None else starts[0]
        self._factor()

    def _k(self, A, B, th):
        ls, s = np.exp(th[:-2]), np.exp(th[-2])
        D = (A[:, None, :] - B[None, :, :]) / ls
        return s * s * np.exp(-0.5 * (D * D).sum(-1))

    def _nll(self, th):
        K = self._k(self.X, self.X, th) + (np.exp(th[-1]) ** 2 + 1e-8) * np.eye(len(self.X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        r = self.y - self.mu
        a = solve_triangular(L, r, lower=True)
        return float(0.5 * a @ a + np.log(np.diag(L)).sum())

    def _factor(self):
        K = self._k(self.X, self.X, self.theta) + (self.noise ** 2 + 1e-8) * np.eye(len(self.X))
        self.cf = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.cf, self.y - self.mu)

    @property
    def noise(self):
        return float(np.exp(self.theta[-1]))

    @property
    def signal(self):
        return float(np.exp(self.theta[-2]))

    @property
    def lengths(self):
        return np.exp(self.theta[:-2])

    def predict(self, Xs, cov=False):
        Ks = self._k(np.asarray(Xs, float), self.X, self.theta)
        m = self.mu + Ks @ self.alpha
        v = cho_solve(self.cf, Ks.T)
        if cov:
            C = self._k(Xs, Xs, self.theta) - Ks @ v
            return m, C
        var = self.signal ** 2 - np.einsum("ij,ji->i", Ks, v)
        return m, np.sqrt(np.clip(var, 0, None))

    def post_cov(self, A, B):
        """后验协方差 cov(f(A), f(B))。"""
        KaX = self._k(A, self.X, self.theta)
        KbX = self._k(B, self.X, self.theta)
        return self._k(A, B, self.theta) - KaX @ cho_solve(self.cf, KbX.T)


class GPSurface:
    """以晶体二聚体为中心的 GP 耦合曲面。records: [{z, dx, dy, gamma, pitch, roll, hole, electron}] (meV)。
    ops: 单体点群操作 (例如 CEDS.ops)，用于展开训练数据；half: 各参数的缩放半宽。"""

    source = "gp"

    def __init__(self, records, center, half=None, ops=("E",), carriers=CARRIERS, reverse=True, keep=1.5,
                 restarts=4, seed=0):
        from .ceds import reverse_cfg
        self.half = dict(DEFAULT_HALF, **(half or {}))
        self.center = np.array([center[k] for k in PARAM_KEYS], float)
        self.carriers = tuple(carriers)
        rows = []
        for r in records:
            if any(r.get(c) is None or not np.isfinite(r.get(c)) for c in carriers):
                continue
            views = [r] + ([reverse_cfg(r)] if reverse else [])
            for v in views:
                for op in ops:
                    q = apply_op(v, op) if op != "E" else v
                    rows.append(_q(q) + [float(r[c]) for c in carriers])
        if not rows:
            raise ValueError("没有可用的训练数据")
        A = np.array(rows, float)
        U = self.scale(A[:, :6])
        ok = (np.abs(U) <= keep).all(1)
        A, U = A[ok], U[ok]
        _, idx = np.unique(np.round(U, 4), axis=0, return_index=True)
        A, U = A[np.sort(idx)], U[np.sort(idx)]
        if len(A) < 4:
            raise ValueError(f"中心附近只有 {len(A)} 个训练构型")
        self.U, self.n = U, len(A)
        self.gp = {c: _GP(U, A[:, 6 + j], restarts=restarts, seed=seed) for j, c in enumerate(carriers)}
        self.frozen = []

    # ---- 坐标
    def scale(self, Q):
        D = np.atleast_2d(np.asarray(Q, float)) - self.center
        for i, k in enumerate(PARAM_KEYS):
            if k in ANG:
                D[:, i] = _wrap(D[:, i])
            D[:, i] /= self.half[k]
        return D

    # ---- 曲面接口
    def __call__(self, Q, carrier):
        return self.gp[carrier].predict(self.scale(Q))[0]

    def predict(self, Q, carrier):
        return self.gp[carrier].predict(self.scale(Q))

    def coverage(self, Q):
        U = self.scale(Q)
        return float((np.abs(U) <= 1.0 + 1e-9).all(1).mean())

    def describe(self):
        s = "; ".join(f"{c}: 噪声 {g.noise:.2f} meV, 长度 [{', '.join(f'{x:.2f}' for x in g.lengths)}]"
                      for c, g in self.gp.items())
        return f"GP 曲面 ({self.n} 个训练点，含对称展开；{s})"

    # ---- 不确定度
    # 后验区间用 CI_INFLATE 放大：并五苯真实 DFT 回测中，未放大的 95% 区间只覆盖 34/52 例 (GP 低估模型误差)，×1.5 后约 90%
    def stat_uncertainty(self, Q, carrier, n_draws=400, max_points=300, seed=1):
        """热运动样本 Q 上 ⟨V⟩、σ_V 的后验分布：返回 dict(mean, std, mean_lo, mean_hi, std_lo, std_hi, std_rel)。"""
        rng = np.random.default_rng(seed)
        Q = np.atleast_2d(Q)
        if len(Q) > max_points:
            Q = Q[rng.choice(len(Q), max_points, replace=False)]
        m, C = self.gp[carrier].predict(self.scale(Q), cov=True)
        C = 0.5 * (C + C.T) + 1e-6 * np.eye(len(C))
        w, V = np.linalg.eigh(C)
        L = V * np.sqrt(np.clip(w, 0, None))
        F = m[None, :] + rng.standard_normal((n_draws, len(m))) @ L.T
        mu, sd = F.mean(1), F.std(1, ddof=1)
        out = {"mean": float(m.mean()), "std": float(m.std(ddof=1)), "std_post": float(sd.mean()),
               "mean_lo": float(np.percentile(mu, 2.5)), "mean_hi": float(np.percentile(mu, 97.5)),
               "std_lo": float(np.percentile(sd, 2.5)), "std_hi": float(np.percentile(sd, 97.5))}
        k = CI_INFLATE
        for a in ("mean", "std"):
            c = out[a] if a == "mean" else out["std_post"]
            out[a + "_lo"] = c - k * (c - out[a + "_lo"])
            out[a + "_hi"] = c + k * (out[a + "_hi"] - c)
        out["std_rel"] = (out["std_hi"] - out["std_lo"]) / 2 / max(out["std"], 1e-9)
        return out


def select_batch(surface, pool, thermal, k=10, exclude=(), max_thermal=300, seed=0):
    """批量 IMSE 选点：从候选池 pool (n, 6) 中选 k 个，使热运动样本 thermal 上的后验方差总和下降最多。
    两种载流子按各自信号方差归一后相加。返回 pool 的下标列表。"""
    rng = np.random.default_rng(seed)
    T = np.atleast_2d(thermal)
    if len(T) > max_thermal:
        T = T[rng.choice(len(T), max_thermal, replace=False)]
    Up, Ut = surface.scale(pool), surface.scale(T)
    chosen, excl = [], set(int(i) for i in exclude)
    # 每种载流子：候选–热样本的后验协方差 Cpt、候选方差 vp；贪心选点后做秩一更新
    state = {}
    for c, g in surface.gp.items():
        Cpt = g.post_cov(Up, Ut)
        Cpp = g.post_cov(Up, Up)
        state[c] = [Cpt, Cpp, g.noise ** 2, g.signal ** 2]
    for _ in range(min(k, len(pool) - len(excl))):
        score = np.zeros(len(pool))
        for c, (Cpt, Cpp, n2, s2) in state.items():
            vp = np.clip(np.diag(Cpp), 1e-12, None)
            score += (Cpt ** 2).sum(1) / (vp + n2) / s2
        for i in excl.union(chosen):
            score[i] = -np.inf
        j = int(np.argmax(score))
        chosen.append(j)
        for c, st in state.items():
            Cpt, Cpp, n2, _ = st
            d = Cpp[j, j] + n2
            st[0] = Cpt - np.outer(Cpp[:, j], Cpt[j]) / d
            st[1] = Cpp - np.outer(Cpp[:, j], Cpp[j]) / d
    return chosen


def select_spread(pool, thermal_center, half, k, seed=0):
    """起始批次 (还没有 GP 时)：在缩放坐标中做最远点采样，第一个点取最接近中心的构型。"""
    U = np.atleast_2d(pool) - thermal_center
    for i, key in enumerate(PARAM_KEYS):
        if key in ANG:
            U[:, i] = _wrap(U[:, i])
        U[:, i] /= half.get(key, DEFAULT_HALF[key])
    chosen = [int(np.argmin((U * U).sum(1)))]
    d = ((U - U[chosen[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        j = int(np.argmax(d))
        chosen.append(j)
        d = np.minimum(d, ((U - U[j]) ** 2).sum(1))
    return chosen


def records_from_rows(Q, V_hole, V_electron):
    return [dict(zip(PARAM_KEYS, map(float, q)), hole=float(h), electron=float(e)) for q, h, e in zip(Q, V_hole, V_electron)]


# ----------------------------------------------------------------------------------------------
# 与扫描项目连接的主动学习流程 (ceds al)
# ----------------------------------------------------------------------------------------------
CI_INFLATE = 1.5
HALF_FLOOR = {"z": 0.15, "dx": 0.2, "dy": 0.2, "gamma": 2.0, "pitch": 2.0, "roll": 3.0}


class ALRun:
    """一个主动学习任务的状态 (存于 项目/al/<name>.json 与 <name>_pool.npz)。
    每一轮写成扫描项目中的一个扫描 <name>_r<k> (普通 ADF 批处理)；算完后再调用 step() 拟合、判断收敛、选下一批。

    候选池与热运动样本都来自同一段全晶体刚体 MD 的二聚体几何 (只需几何，不需要 DFT)：
    候选池 = 每个近邻对 n_pool 个构型；热运动样本 = 其中随机 n_thermal 个，用于选点 (IMSE) 和停止判据。"""

    def __init__(self, project, name):
        from pathlib import Path
        self.proj, self.name = project, name
        self.dir = Path(project.root) / "al"
        self.state_path = self.dir / f"{name}.json"
        self.pool_path = self.dir / f"{name}_pool.npz"
        self.state = None
        if self.state_path.exists():
            import json
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    @property
    def exists(self):
        return self.state is not None

    def _save(self):
        import json
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")

    def pool(self):
        d = np.load(self.pool_path)
        return {int(k.split("_")[1]): d[k] for k in d.files if k.startswith("pool_")}, \
               {int(k.split("_")[1]): d[k] for k in d.files if k.startswith("thermal_")}

    # ---- 第 1 轮
    def start(self, ceds, configs, level, T, n_thermal=300, init=12, batch=10, target=0.03, k_half=2.5, seed=0,
              info=None):
        """configs: rigidmd.snapshot_configs 的结果 (group = 组号)。写第 1 轮扫描并保存状态。"""
        if self.exists:
            raise ValueError(f"主动学习任务 {self.name} 已经存在 ({self.state_path})")
        rng = np.random.default_rng(seed)
        pools, thermal, halves = {}, {}, {}
        for c in configs:
            pools.setdefault(int(c["group"]), []).append(_q(c))
        arrays = {}
        for g, Q in pools.items():
            Q = np.array(Q, float)
            pools[g] = Q
            t = Q[rng.choice(len(Q), min(n_thermal, len(Q)), replace=False)]
            thermal[g] = t
            sd = Q.std(0)
            halves[g] = {k: float(max(k_half * sd[i], HALF_FLOOR[k])) for i, k in enumerate(PARAM_KEYS)}
            arrays[f"pool_{g}"], arrays[f"thermal_{g}"] = Q, t
        self.dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.pool_path, **arrays)
        centers = {p["gid"]: {k: float(p["center"][k]) for k in PARAM_KEYS} for p in ceds.pairs if p["gid"] in pools}
        self.state = {"name": self.name, "T": float(T), "level": level, "pairs": sorted(pools), "centers": centers,
                      "half": {str(g): h for g, h in halves.items()}, "init": init, "batch": batch, "target": target,
                      "rounds": [], "converged": {}, "history": [], "info": info or {}}
        sel = {}
        for g, Q in pools.items():
            c0 = np.array([centers[g][k] for k in PARAM_KEYS])
            sel[g] = select_spread(Q, c0, halves[g], min(init, len(Q)))
        return self._write_round(sel)

    def _write_round(self, sel):
        k = len(self.state["rounds"]) + 1
        scan = f"{self.name}_r{k}"
        pools, _ = self.pool()
        cfgs = []
        from .scanflow import make_cfg
        for g, idx in sel.items():
            for i in idx:
                q = pools[g][i]
                cfgs.append(dict(make_cfg(*q[:3], gamma=q[3], pitch=q[4], roll=q[5]), set="al", group=str(g),
                                 variant=f"pool{int(i)}"))
        info = self.proj.add_scan(scan, cfgs, self.state["level"], kind="al", reduce=False, clash_min=0.0,
                                  info={"al": self.name, "round": k})
        self.state["rounds"].append({"scan": scan, "selected": {str(g): [int(i) for i in idx] for g, idx in sel.items()}})
        self._save()
        return {"scan": scan, "n": len(cfgs), "batches": info.get("batches", [])}

    # ---- 数据与曲面
    def records(self):
        """全部轮次中已算完的构型：{组号: [记录]}；另返回各轮未完成 / 失败的数目。"""
        out, pending, failed = {}, 0, 0
        for r in self.state["rounds"]:
            for rec in self.proj.collect(r["scan"], write=True):
                if rec.get("rep"):
                    continue
                st = rec.get("state")
                if st == "done":
                    out.setdefault(int(rec["group"]), []).append(rec)
                elif st == "pending":
                    pending += 1
                elif st == "failed":
                    failed += 1
        return out, pending, failed

    def surface(self, ceds, gid, recs, restarts=3):
        return GPSurface(recs, self.state["centers"][str(gid)] if str(gid) in self.state["centers"]
                         else self.state["centers"][gid], half=self.state["half"][str(gid)], ops=ceds.ops,
                         restarts=restarts)

    def surfaces(self, ceds):
        """用全部已完成的数据拟合每个近邻对的 GP 曲面：{组号: GPSurface} (可直接用于 ceds run / md)。"""
        recs, _, _ = self.records()
        return {g: self.surface(ceds, g, rs) for g, rs in recs.items() if len(rs) >= 4}

    # ---- 下一轮
    def step(self, ceds, force=False):
        """拟合、判断收敛；未收敛的对选出下一批并写成新一轮扫描。返回 (报告行, 新轮次信息或 None)。"""
        recs, pending, failed = self.records()
        if pending and not force:
            raise ValueError(f"上一轮还有 {pending} 个构型没有算完 (失败 {failed} 个)；算完并 collect 后再继续 (或加 --force)")
        pools, thermal = self.pool()
        tgt = float(self.state["target"])
        rows, sel = [], {}
        used = {}
        for r in self.state["rounds"]:
            for g, idx in r["selected"].items():
                used.setdefault(int(g), set()).update(idx)
        for g in self.state["pairs"]:
            rs = recs.get(g, [])
            row = {"gid": g, "n": len(rs)}
            if len(rs) < 4:
                row["status"] = "数据不足"
                rows.append(row)
                sel[g] = [i for i in range(len(pools[g])) if i not in used.get(g, set())][:self.state["batch"]]
                continue
            s = self.surface(ceds, g, rs)
            ok = True
            for c in ("hole", "electron"):
                u = s.stat_uncertainty(thermal[g], c)
                row[c] = u
                half_mean = (u["mean_hi"] - u["mean_lo"]) / 2
                ok &= u["std_rel"] <= tgt and half_mean <= max(tgt * abs(u["mean"]), 1.0)
            row["status"] = "收敛" if ok else "继续"
            rows.append(row)
            if ok:
                self.state["converged"][str(g)] = len(rs)
            else:
                self.state["converged"].pop(str(g), None)
                sel[g] = select_batch(s, pools[g], thermal[g], k=self.state["batch"], exclude=sorted(used.get(g, set())))
        self.state["history"].append([{k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()})
                                       for k, v in r.items()} for r in rows])
        sel = {g: idx for g, idx in sel.items() if idx}
        new = self._write_round(sel) if sel else None
        if not new:
            self._save()
        return rows, new


def al_report(rows):
    L = [f"{'组':<5}{'DFT 点':>7}{'状态':>8}   {'空穴 ⟨V⟩ ± / σ_V ±':<30}{'电子 ⟨V⟩ ± / σ_V ±':<30}"]
    for r in rows:
        s = f"G{r['gid']:<4}{r['n']:>7}{r['status']:>8}   "
        for c in ("hole", "electron"):
            u = r.get(c)
            if u:
                s += f"{u['mean']:7.1f}±{(u['mean_hi'] - u['mean_lo']) / 2:4.1f} / {u['std']:5.1f}±{100 * u['std_rel']:3.0f}%   "
            else:
                s += f"{'—':<30}"
        L.append(s)
    return "\n".join(L)
