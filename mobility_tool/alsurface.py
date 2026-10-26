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
    """各向异性平方指数核 + 白噪声。X 已缩放到 O(1)。
    H = None：常数均值 (取训练数据平均值)，与 v1.19 以前相同。
    H (n, p)：均值为基函数的线性组合 m(x) = h(x)·β，β 取平坦先验 (通用克里金)：β 由广义最小二乘估计，
    超参数用限制似然 (REML)，预测方差包含 β 的不确定度。Δ-学习用 h(x) = (1, V_EHT(x))。"""

    def __init__(self, X, y, restarts=4, seed=0, min_noise=0.05, H=None):
        self.X, self.y = np.asarray(X, float), np.asarray(y, float)
        self.H = None if H is None else np.asarray(H, float)
        self.mu = float(self.y.mean())
        self.min_noise = min_noise
        d = self.X.shape[1]
        if self.H is None:
            sd = float(self.y.std()) or 1.0
        else:                                        # 初值与界限按基函数最小二乘拟合后的残差定
            b0 = np.linalg.lstsq(self.H, self.y, rcond=None)[0]
            sd = float((self.y - self.H @ b0).std()) or 1.0
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

    def _gls(self, L):
        """给定 K 的 Cholesky 因子：返回 (β, 残差, HᵀK⁻¹H 的 Cholesky 因子)。"""
        Hw = solve_triangular(L, self.H, lower=True)
        yw = solve_triangular(L, self.y, lower=True)
        A = Hw.T @ Hw + 1e-10 * np.eye(self.H.shape[1])
        LA = np.linalg.cholesky(A)
        beta = np.linalg.solve(A, Hw.T @ yw)
        return beta, self.y - self.H @ beta, LA

    def _nll(self, th):
        K = self._k(self.X, self.X, th) + (np.exp(th[-1]) ** 2 + 1e-8) * np.eye(len(self.X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        if self.H is None:
            r = self.y - self.mu
            a = solve_triangular(L, r, lower=True)
            return float(0.5 * a @ a + np.log(np.diag(L)).sum())
        try:
            _, r, LA = self._gls(L)
        except np.linalg.LinAlgError:
            return 1e10
        a = solve_triangular(L, r, lower=True)
        return float(0.5 * a @ a + np.log(np.diag(L)).sum() + np.log(np.diag(LA)).sum())

    def _factor(self):
        K = self._k(self.X, self.X, self.theta) + (self.noise ** 2 + 1e-8) * np.eye(len(self.X))
        self.cf = cho_factor(K, lower=True)
        if self.H is None:
            self.alpha = cho_solve(self.cf, self.y - self.mu)
            return
        L = np.linalg.cholesky(K)
        self.beta, r, LA = self._gls(L)
        self.alpha = cho_solve(self.cf, r)
        self._LA = LA
        self._KiH = cho_solve(self.cf, self.H)

    @property
    def noise(self):
        return float(np.exp(self.theta[-1]))

    @property
    def signal(self):
        return float(np.exp(self.theta[-2]))

    @property
    def lengths(self):
        return np.exp(self.theta[:-2])

    def _mean(self, n, Hs):
        if self.H is None:
            return self.mu
        if Hs is None:
            raise ValueError("带基线的 GP 预测时需要给出基函数 Hs")
        return np.asarray(Hs, float) @ self.beta

    def _Rb(self, Ks, Hs):
        """β 不确定度项的系数 R = Hs − Ks K⁻¹ H，按 (HᵀK⁻¹H)^(-1/2) 白化。"""
        R = np.asarray(Hs, float) - Ks @ self._KiH
        return solve_triangular(self._LA, R.T, lower=True)          # (p, n)

    def predict(self, Xs, cov=False, Hs=None):
        Xs = np.asarray(Xs, float)
        Ks = self._k(Xs, self.X, self.theta)
        m = self._mean(len(Xs), Hs) + Ks @ self.alpha
        v = cho_solve(self.cf, Ks.T)
        Rb = None if self.H is None else self._Rb(Ks, Hs)
        if cov:
            C = self._k(Xs, Xs, self.theta) - Ks @ v
            if Rb is not None:
                C = C + Rb.T @ Rb
            return m, C
        var = self.signal ** 2 - np.einsum("ij,ji->i", Ks, v)
        if Rb is not None:
            var = var + (Rb * Rb).sum(0)
        return m, np.sqrt(np.clip(var, 0, None))

    def post_cov(self, A, B, Ha=None, Hb=None):
        """后验协方差 cov(f(A), f(B))。"""
        KaX = self._k(A, self.X, self.theta)
        KbX = self._k(B, self.X, self.theta)
        C = self._k(A, B, self.theta) - KaX @ cho_solve(self.cf, KbX.T)
        if self.H is not None:
            C = C + self._Rb(KaX, Ha).T @ self._Rb(KbX, Hb)
        return C


class GPSurface:
    """以晶体二聚体为中心的 GP 耦合曲面。records: [{z, dx, dy, gamma, pitch, roll, hole, electron}] (meV)。
    ops: 单体点群操作 (例如 CEDS.ops)，用于展开训练数据；half: 各参数的缩放半宽。
    baseline: Δ-学习的基线 (例如 EHTBaseline)，baseline(Q, carrier) → V (meV)。给出时 GP 的均值为
    β₀ + β₁·V_base(q)，GP 只学习 DFT 与 (缩放后的) 基线之差；β 由数据估计 (自动处理基线的整体缩放与符号)。"""

    source = "gp"

    def __init__(self, records, center, half=None, ops=("E",), carriers=CARRIERS, reverse=True, keep=1.5,
                 restarts=4, seed=0, baseline=None):
        from .ceds import reverse_cfg
        self.half = dict(DEFAULT_HALF, **(half or {}))
        self.center = np.array([center[k] for k in PARAM_KEYS], float)
        self.carriers = tuple(carriers)
        self.baseline = baseline
        self.ci_inflate = None                 # None：按是否有基线取 CI_INFLATE / CI_INFLATE_DELTA
        rows = []
        recs = [r for r in records if not any(r.get(c) is None or not np.isfinite(r.get(c)) for c in carriers)]
        if baseline is not None and recs:              # 基线在原始构型上计算，与 V 一起复制到对称展开的各个像
            Q0 = np.array([_q(r) for r in recs], float)
            B0 = {c: np.asarray(baseline(Q0, c), float) for c in carriers}
        for n, r in enumerate(recs):
            views = [r] + ([reverse_cfg(r)] if reverse else [])
            for v in views:
                for op in ops:
                    q = apply_op(v, op) if op != "E" else v
                    rows.append(_q(q) + [float(r[c]) for c in carriers]
                                + ([float(B0[c][n]) for c in carriers] if baseline is not None else []))
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
        nc = len(carriers)
        if baseline is None:
            self.gp = {c: _GP(U, A[:, 6 + j], restarts=restarts, seed=seed) for j, c in enumerate(carriers)}
        else:
            self.gp = {c: _GP(U, A[:, 6 + j], restarts=restarts, seed=seed,
                              H=np.c_[np.ones(len(A)), A[:, 6 + nc + j]]) for j, c in enumerate(carriers)}
        self.frozen = []

    # ---- 坐标
    def scale(self, Q):
        D = np.atleast_2d(np.asarray(Q, float)) - self.center
        for i, k in enumerate(PARAM_KEYS):
            if k in ANG:
                D[:, i] = _wrap(D[:, i])
            D[:, i] /= self.half[k]
        return D

    @property
    def inflate(self):
        if self.ci_inflate is not None:
            return self.ci_inflate
        return CI_INFLATE_DELTA if self.baseline is not None else CI_INFLATE

    def H(self, Q, carrier):
        """预测点的基函数 (1, V_base)；没有基线时为 None。"""
        if self.baseline is None:
            return None
        b = np.asarray(self.baseline(np.atleast_2d(np.asarray(Q, float)), carrier), float)
        return np.c_[np.ones(len(b)), b]

    # ---- 曲面接口
    def __call__(self, Q, carrier):
        return self.gp[carrier].predict(self.scale(Q), Hs=self.H(Q, carrier))[0]

    def predict(self, Q, carrier):
        return self.gp[carrier].predict(self.scale(Q), Hs=self.H(Q, carrier))

    def coverage(self, Q):
        U = self.scale(Q)
        return float((np.abs(U) <= 1.0 + 1e-9).all(1).mean())

    def describe(self):
        s = "; ".join(f"{c}: 噪声 {g.noise:.2f} meV, 长度 [{', '.join(f'{x:.2f}' for x in g.lengths)}]"
                      + (f", V = {g.beta[0]:+.1f} + {g.beta[1]:.3f}·V_EHT" if self.baseline is not None else "")
                      for c, g in self.gp.items())
        kind = "Δ-GP 曲面 (EHT 基线)" if self.baseline is not None else "GP 曲面"
        return f"{kind} ({self.n} 个训练点，含对称展开；{s})"

    # ---- 不确定度
    # 后验区间用 CI_INFLATE 放大：并五苯真实 DFT 回测中，未放大的 95% 区间只覆盖 34/52 例 (GP 低估模型误差)，×1.5 后约 90%
    def stat_uncertainty(self, Q, carrier, n_draws=400, max_points=300, seed=1):
        """热运动样本 Q 上 ⟨V⟩、σ_V 的后验分布：返回 dict(mean, std, mean_lo, mean_hi, std_lo, std_hi, std_rel)。"""
        rng = np.random.default_rng(seed)
        Q = np.atleast_2d(Q)
        if len(Q) > max_points:
            Q = Q[rng.choice(len(Q), max_points, replace=False)]
        m, C = self.gp[carrier].predict(self.scale(Q), cov=True, Hs=self.H(Q, carrier))
        C = 0.5 * (C + C.T) + 1e-6 * np.eye(len(C))
        w, V = np.linalg.eigh(C)
        L = V * np.sqrt(np.clip(w, 0, None))
        F = m[None, :] + rng.standard_normal((n_draws, len(m))) @ L.T
        mu, sd = F.mean(1), F.std(1, ddof=1)
        out = {"mean": float(m.mean()), "std": float(m.std(ddof=1)), "std_post": float(sd.mean()),
               "mean_lo": float(np.percentile(mu, 2.5)), "mean_hi": float(np.percentile(mu, 97.5)),
               "std_lo": float(np.percentile(sd, 2.5)), "std_hi": float(np.percentile(sd, 97.5))}
        k = self.inflate
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
        Hp, Ht = surface.H(pool, c), surface.H(T, c)
        Cpt = g.post_cov(Up, Ut, Hp, Ht)
        Cpp = g.post_cov(Up, Up, Hp, Hp)
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


class EHTBaseline:
    """Δ-学习的基线：软件内 EHT 计算二聚体 (A 在原点、B 按 q 放置) 的 HOMO / LUMO 有效耦合 (meV)。
    与 ADF 片段法相同，B 用同一套单体轨道旋转得到，因此相位与 ADF 一致 (相差的整体缩放和符号由 β₁ 吸收)。
    每个构型约 10 ms；结果按坐标缓存。mirror=True：B 为镜像单体 (对映体近邻对)。"""

    def __init__(self, symbols, Y, mirror=False, nd=5):
        from .eht import monomer
        from .surrogate import MIRROR
        self.sym, self.Y = list(symbols), np.asarray(Y, float)
        self.mirror, self.nd = bool(mirror), nd
        self.A = [(s, *p) for s, p in zip(self.sym, self.Y)]
        self.mono = monomer(self.sym, self.Y)
        self.mono_b = monomer(self.sym, self.Y * MIRROR) if self.mirror else self.mono
        self.cache = {}

    def _one(self, q):
        from .eht import build_basis, eht_transfer_integrals, rotate_coefficients
        from .surrogate import build_B, rotation
        z, dx, dy, ga, pi, ro = map(float, q)
        R = rotation(ga, ro, pi)
        B = build_B(self.Y, z, dx, dy, ga, ro, pi, mirror=self.mirror)
        shells, _, _ = build_basis(self.sym, B)
        mb = dict(self.mono_b, shells=shells, C=rotate_coefficients(self.mono_b["shells"], self.mono_b["C"], R))
        r = eht_transfer_integrals(self.A, [(s, *p) for s, p in zip(self.sym, B)], esd=False, _mono_b=mb)["results"]
        return r["HOMO"]["J_eff_meV"], r["LUMO"]["J_eff_meV"]

    def values(self, Q):
        out = np.empty((len(Q), 2))
        for i, q in enumerate(np.atleast_2d(np.asarray(Q, float))):
            key = tuple(np.round(q, self.nd))
            if key not in self.cache:
                self.cache[key] = self._one(q)
            out[i] = self.cache[key]
        return out

    def __call__(self, Q, carrier):
        return self.values(Q)[:, 0 if carrier == "hole" else 1]

    def check(self, records, carriers=CARRIERS):
        """与已有 DFT 数据的一致性：每种载流子的相关系数与缩放系数 (相关系数低于 0.8 时不宜用作基线)。"""
        Q = np.array([_q(r) for r in records], float)
        B = self.values(Q)
        out = {}
        for j, c in enumerate(carriers):
            y = np.array([float(r[c]) for r in records])
            if len(y) >= 3 and B[:, j].std() > 1e-9:
                a = np.polyfit(B[:, j], y, 1)
                out[c] = {"r": float(np.corrcoef(B[:, j], y)[0, 1]), "slope": float(a[0]), "intercept": float(a[1]),
                          "resid_std": float(np.std(y - np.polyval(a, B[:, j]))), "std": float(y.std())}
        return out


def records_from_rows(Q, V_hole, V_electron):
    return [dict(zip(PARAM_KEYS, map(float, q)), hole=float(h), electron=float(e)) for q, h, e in zip(Q, V_hole, V_electron)]


# ----------------------------------------------------------------------------------------------
# 与扫描项目连接的主动学习流程 (ceds al)
# ----------------------------------------------------------------------------------------------
CI_INFLATE = 1.5
# Δ-GP 的后验区间更窄、低估得更多：并五苯 400 个 MD 快照回测 (N ≥ 16，四个近邻对 × 两种载流子 × 各 N)
# 在 ×1.5 时只覆盖 72%，需要再放大约 2.3 倍才达到 95%。取 3.4 (只影响区间与收敛判据，不影响曲面本身)。
CI_INFLATE_DELTA = 3.4
ABS_FLOOR = 0.3          # meV：σ_V、⟨V⟩ 区间半宽的绝对收敛下限 (弱耦合二聚体)
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
              info=None, seeds=(), abs_floor=ABS_FLOOR, delta=True):
        """configs: rigidmd.snapshot_configs 的结果 (group = 组号)。写第 1 轮扫描并保存状态。
        seeds: 项目中已有扫描的名称 (如先前的局部扫描)；其中已算完的构型直接作为训练数据，
        已有足够数据的近邻对第 1 轮直接按 IMSE 补点 (或已收敛则不再计算)。"""
        if self.exists:
            raise ValueError(f"主动学习任务 {self.name} 已经存在 ({self.state_path})")
        rng = np.random.default_rng(seed)
        pools, thermal, halves, mirror = {}, {}, {}, {}
        for c in configs:
            pools.setdefault(int(c["group"]), []).append(_q(c))
            mirror[int(c["group"])] = bool(str(c.get("mirror", "")).strip() in ("1", "True", "true"))
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
                      "rounds": [], "converged": {}, "history": [], "info": info or {}, "seeds": list(seeds),
                      "mirror": {str(g): m for g, m in mirror.items()}, "abs_floor": float(abs_floor),
                      "delta": bool(delta)}
        if seeds:
            recs = self._seed_records()[0]
            if any(len(recs.get(g, [])) >= 4 for g in pools):
                self._save()
                rows, new = self.step(ceds, force=True, init_spread=init)
                return {"seeded": {g: len(recs.get(g, [])) for g in pools}, "rows": rows,
                        "scan": new["scan"] if new else None, "n": new["n"] if new else 0,
                        "batches": new["batches"] if new else []}
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
                mir = {"mirror": 1} if self.state.get("mirror", {}).get(str(g)) else {}
                cfgs.append(dict(make_cfg(*q[:3], gamma=q[3], pitch=q[4], roll=q[5], **mir), set="al", group=str(g),
                                 variant=f"pool{int(i)}"))
        info = self.proj.add_scan(scan, cfgs, self.state["level"], kind="al", reduce=False, clash_min=0.0,
                                  info={"al": self.name, "round": k})
        self.state["rounds"].append({"scan": scan, "selected": {str(g): [int(i) for i in idx] for g, idx in sel.items()}})
        self._save()
        return {"scan": scan, "n": len(cfgs), "batches": info.get("batches", [])}

    # ---- 数据与曲面
    def _seed_records(self):
        """种子扫描中已算完的构型 (未算完的不计入待完成数，避免旧的大批量扫描阻塞主动学习)。"""
        out, n = {}, 0
        for sc in self.state.get("seeds", []):
            try:
                rows = self.proj.collect(sc, write=True)
            except Exception as e:                       # noqa: BLE001
                raise ValueError(f"种子扫描 {sc} 无法读取: {e}") from None
            for rec in rows:
                if rec.get("rep") or rec.get("state") != "done":
                    continue
                try:
                    g = int(rec["group"])
                except (TypeError, ValueError, KeyError):
                    continue
                out.setdefault(g, []).append(rec)
                n += 1
        return out, n

    def records(self):
        """全部轮次 (及种子扫描) 中已算完的构型：{组号: [记录]}；另返回各轮未完成 / 失败的数目。"""
        out, pending, failed = {}, 0, 0
        if self.state.get("seeds"):
            out = {g: list(v) for g, v in self._seed_records()[0].items()}
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

    @property
    def delta(self):
        """Δ-学习 (EHT 基线)。v1.20 以前建立的任务没有此项，按原来的常数均值 GP 处理。"""
        return bool(self.state.get("delta", False))

    def set_delta(self, on):
        self.state["delta"] = bool(on)
        self._save()

    def baseline(self, gid):
        if not self.delta:
            return None
        mirror = bool(self.state.get("mirror", {}).get(str(gid)))
        if not hasattr(self, "_base"):
            self._base = {}
        if mirror not in self._base:
            self._base[mirror] = EHTBaseline(self.proj.symbols, self.proj.Y, mirror=mirror)
        return self._base[mirror]

    def surface(self, ceds, gid, recs, restarts=3):
        mirror = bool(self.state.get("mirror", {}).get(str(gid)))
        return GPSurface(recs, self.state["centers"][str(gid)] if str(gid) in self.state["centers"]
                         else self.state["centers"][gid], half=self.state["half"][str(gid)], ops=ceds.ops,
                         restarts=restarts, reverse=not mirror, baseline=self.baseline(gid))

    def baseline_check(self, recs):
        """各近邻对已算 DFT 数据与 EHT 基线的相关性：{组号: {载流子: {r, slope, ...}}}。"""
        return {g: self.baseline(g).check(rs) for g, rs in recs.items() if self.delta and len(rs) >= 4}

    def mu_ci(self, ceds, T=None, hbar_tau=0.005, n_draws=30, n_real=12, carriers=CARRIERS, seed=3):
        """由 GP 后验传递到 TLT 迁移率：热运动样本上联合抽样 → (⟨V⟩, σ_V) → TLT μ。
        返回 {载流子: {mu, lo, hi, draws}} (主方向迁移率，cm²/Vs；区间为 95%)。"""
        recs, _, _ = self.records()
        _, thermal = self.pool()
        surf = {g: self.surface(ceds, g, rs) for g, rs in recs.items() if len(rs) >= 4}
        return mu_posterior(ceds, surf, thermal, T or self.state["T"], hbar_tau, n_draws, n_real, carriers, seed)

    def surfaces(self, ceds):
        """用全部已完成的数据拟合每个近邻对的 GP 曲面：{组号: GPSurface} (可直接用于 ceds run / md)。"""
        recs, _, _ = self.records()
        return {g: self.surface(ceds, g, rs) for g, rs in recs.items() if len(rs) >= 4}

    # ---- 下一轮
    def step(self, ceds, force=False, init_spread=None):
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
                c0 = np.array([self.state["centers"][str(g)][k] if str(g) in self.state["centers"]
                               else self.state["centers"][g][k] for k in PARAM_KEYS])
                half = self.state["half"][str(g)]
                free = [i for i in range(len(pools[g])) if i not in used.get(g, set())]
                k = init_spread or self.state["init"]
                pick = select_spread(pools[g][free], c0, half, min(k, len(free)))
                sel[g] = [free[i] for i in pick]
                continue
            s = self.surface(ceds, g, rs)
            ok = True
            for c in ("hole", "electron"):
                u = s.stat_uncertainty(thermal[g], c)
                row[c] = u
                half_mean = (u["mean_hi"] - u["mean_lo"]) / 2
                half_std = (u["std_hi"] - u["std_lo"]) / 2
                floor = float(self.state.get("abs_floor", ABS_FLOOR))
                # 相对判据对很弱的耦合 (σ_V < 1 meV) 没有意义：区间半宽小于绝对下限 (默认 0.3 meV) 也算收敛
                ok &= (u["std_rel"] <= tgt or half_std <= floor) and half_mean <= max(tgt * abs(u["mean"]), floor, 1.0)
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


def mu_posterior(ceds, surfaces, thermal, T, hbar_tau=0.005, n_draws=30, n_real=12, carriers=CARRIERS, seed=3,
                 max_points=200):
    """把耦合曲面的后验不确定度传递到 TLT 迁移率。
    surfaces: {组号: GPSurface}；thermal: {组号: 热运动样本 (n, 6)}。每次抽样中，每个近邻对在热运动样本上
    联合抽取一条曲面 → ⟨V⟩、σ_V → TLT (各次抽样用相同的随机数，差别只来自耦合曲面)。
    返回 {载流子: {"mu": 后验均值曲面的 μ, "lo", "hi": 95% 区间, "draws": [...]}}。"""
    rng = np.random.default_rng(seed)
    draws = {}
    for g, s in surfaces.items():
        Q = np.atleast_2d(thermal[g])
        if len(Q) > max_points:
            Q = Q[rng.choice(len(Q), max_points, replace=False)]
        draws[g] = {}
        for c in carriers:
            m, C = s.gp[c].predict(s.scale(Q), cov=True, Hs=s.H(Q, c))
            C = 0.5 * (C + C.T) + 1e-6 * np.eye(len(C))
            w, V = np.linalg.eigh(C)
            L = V * np.sqrt(np.clip(w, 0, None))
            F = m[None, :] + rng.standard_normal((n_draws, len(m))) @ L.T
            # 与 stat_uncertainty 相同：后验区间按 CI_INFLATE 放大 (GP 低估模型误差)
            F = m[None, :] + s.inflate * (F - m[None, :])
            draws[g][c] = (float(m.mean()), float(m.std(ddof=1)), F.mean(1), F.std(1, ddof=1))
    out = {}
    for c in carriers:
        def mu_of(stat):
            res = {g: {"T": T, c: {"mean": stat[g][0], "std": stat[g][1]}} for g in stat}
            return float(ceds.tlt(res, c, T, hbar_tau, n_real=n_real, seed=1)["mu_principal"][0])
        mu0 = mu_of({g: draws[g][c][:2] for g in draws})
        vals = [mu_of({g: (draws[g][c][2][k], draws[g][c][3][k]) for g in draws}) for k in range(n_draws)]
        out[c] = {"mu": mu0, "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
                  "draws": vals}
    return out


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
