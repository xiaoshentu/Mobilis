"""MD 形貌 (GROMACS 等) → 分子映射 → 近邻列表 → 转移积分 → 位点能 → 周期盒子中的 KMC。

读取
  .gro (可多帧，三斜盒子)、.pdb (多 MODEL，CRYST1)；.xtc / .trr / .dcd 需要本机装有 MDAnalysis 或 mdtraj。
  分子按残基拆分 (残基号或残基名变化处断开)，跨周期边界的分子按最小像补齐。
映射
  每个分子用 DFT 优化的参考单体刚性叠合 (重原子 Kabsch)：原子对应关系只在第一个分子上求一次 (同一残基类型的原子
  顺序相同)，联合原子力场 (没有 H) 也可映射。几何可取映射后的参考单体 (默认) 或 MD 中的原子坐标。
近邻
  两分子 (含周期像) 的最短重原子间距 < 截断 (默认 4 Å)，与晶体页的近邻判据一致。
转移积分
  软件内 EHT (参考单体的轨道随分子转动，速度与 CEDS 相同)；或导出二聚体的 Gaussian 投影法输入，计算后读回。
位点能
  静电 (原子电荷，球形截断) 或 静电 + Thole 诱导 (polar 模块的微静电模型)。
KMC
  整个周期盒子作为一个晶胞：位点 = 分子，位点能 = 上面的位点能，跳跃矢量带周期像，零场扩散与场迁移率。
"""
import csv
from pathlib import Path

import numpy as np

NM = 10.0
_TWO = {"CL": "Cl", "BR": "Br", "SI": "Si", "SE": "Se", "IR": "Ir", "PT": "Pt", "ZN": "Zn", "CU": "Cu", "LI": "Li"}
_ONE = set("HBCNOFPSKI")


def guess_element(name, mapping=None):
    """由原子名推断元素 (GROMACS 原子名如 C12、H3、CL1)。mapping 可指定 {原子名或前缀: 元素}。"""
    n = name.strip()
    if mapping:
        if n in mapping:
            return mapping[n]
        for k in sorted(mapping, key=len, reverse=True):
            if n.upper().startswith(k.upper()):
                return mapping[k]
    s = "".join(ch for ch in n if ch.isalpha()).upper()
    if not s:
        raise ValueError(f"无法由原子名 {name!r} 推断元素")
    if s[:2] in _TWO:
        return _TWO[s[:2]]
    if s[0] in _ONE:
        return s[0]
    raise ValueError(f"无法由原子名 {name!r} 推断元素，请用元素对照指定")


# ----------------------------------------------------------------------------------------------
# 读写
# ----------------------------------------------------------------------------------------------
def _gro_box(vals):
    v = [float(x) * NM for x in vals]
    if len(v) == 3:
        return np.diag(v)
    if len(v) == 9:
        # v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
        return np.array([[v[0], v[3], v[4]], [v[5], v[1], v[6]], [v[7], v[8], v[2]]])
    raise ValueError("gro 盒子行应有 3 或 9 个数")


def read_gro(path, frames=None):
    """读取 .gro (可多帧)。frames: 帧序号集合 (None = 全部)。返回 [{names, resid, resname, X (Å), box (行矢量, Å), title}]。"""
    lines = Path(path).read_text(errors="ignore").splitlines()
    out, i, k = [], 0, 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        title = lines[i]
        n = int(lines[i + 1].split()[0])
        body = lines[i + 2:i + 2 + n]
        boxl = lines[i + 2 + n]
        i += 3 + n
        if frames is not None and k not in frames:
            k += 1
            continue
        first = body[0]
        p1 = first.index(".", 20)
        w = first.index(".", p1 + 1) - p1                     # 坐标字段宽度 (默认 8)
        resid, resn, names = [], [], []
        X = np.empty((n, 3))
        for a, ln in enumerate(body):
            resid.append(int(ln[0:5]))
            resn.append(ln[5:10].strip())
            names.append(ln[10:15].strip())
            X[a] = [float(ln[20 + w * c:20 + w * (c + 1)]) for c in range(3)]
        out.append({"names": names, "resid": np.array(resid), "resname": resn, "X": X * NM,
                    "box": _gro_box(boxl.split()), "title": title.strip()})
        k += 1
    if not out:
        raise ValueError(f"{Path(path).name} 中没有读到坐标")
    return out


def write_gro(path, frames):
    """frames: [{names, resid, resname, X (Å), box (Å), title}]。"""
    L = []
    for f in frames:
        X = np.asarray(f["X"], float) / NM
        L.append(f.get("title", "Mobilis"))
        L.append(f"{len(X):5d}")
        for a, (r, rn, nm, x) in enumerate(zip(f["resid"], f["resname"], f["names"], X)):
            L.append(f"{int(r) % 100000:5d}{rn[:5]:<5s}{nm[:5]:>5s}{(a + 1) % 100000:5d}{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}")
        b = np.asarray(f["box"], float) / NM
        vals = [b[0, 0], b[1, 1], b[2, 2]]
        if np.abs(b - np.diag(np.diag(b))).max() > 1e-6:
            vals += [b[0, 1], b[0, 2], b[1, 0], b[1, 2], b[2, 0], b[2, 1]]
        L.append(" ".join(f"{v:10.5f}" for v in vals))
    Path(path).write_text("\n".join(L) + "\n")


def crystal_frame(lattice, frames, symbols, Y, supercell=(1, 1, 1), resname="MOL"):
    """晶体超胞 → 形貌帧 (可写成 .gro 作为 GROMACS 起始结构，或直接用于测试流程)。
    frames: 晶胞中各分子的 (坐标轴, 原点) (即 CEDS.frames)；Y: 分子坐标系中的单体。"""
    lat = np.asarray(lattice, float)
    sc = tuple(int(n) for n in supercell)
    Y = np.asarray(Y, float)
    cnt, names = {}, []
    for e in symbols:
        cnt[e] = cnt.get(e, 0) + 1
        names.append(f"{e.upper()}{cnt[e]}")
    X, resid = [], []
    r = 0
    for a in range(sc[0]):
        for b in range(sc[1]):
            for c in range(sc[2]):
                t = np.array([a, b, c], float) @ lat
                for ax, o in frames:
                    r += 1
                    X.append(Y @ np.asarray(ax, float).T + np.asarray(o, float) + t)
                    resid += [r] * len(symbols)
    nat = len(symbols)
    return {"names": names * r, "resid": np.array(resid), "resname": [resname] * (nat * r),
            "X": np.concatenate(X), "box": np.diag(sc) @ lat, "title": f"{resname} crystal {sc[0]}x{sc[1]}x{sc[2]}",
            "elements": list(symbols) * r}


def _cryst1(ln):
    a, b, c = (float(ln[6 + 9 * k:15 + 9 * k]) for k in range(3))
    al, be, ga = (np.radians(float(ln[33 + 7 * k:40 + 7 * k])) for k in range(3))
    from .crystal import cell_matrix
    return np.asarray(cell_matrix(a, b, c, np.degrees(al), np.degrees(be), np.degrees(ga)), float)


def read_pdb(path, frames=None):
    out, cur, box, k = [], None, None, 0

    def close():
        nonlocal cur, k
        if cur and cur["names"]:
            if frames is None or k in frames:
                if box is None:
                    raise ValueError("PDB 中没有 CRYST1 (周期盒子)")
                out.append({"names": cur["names"], "resid": np.array(cur["resid"]), "resname": cur["resname"],
                            "X": np.array(cur["X"]), "box": box, "title": Path(path).stem, "elements": cur["el"]})
            k += 1
        cur = None
    for ln in Path(path).read_text(errors="ignore").splitlines():
        if ln.startswith("CRYST1"):
            box = _cryst1(ln)
        elif ln.startswith(("ATOM", "HETATM")):
            if cur is None:
                cur = {"names": [], "resid": [], "resname": [], "X": [], "el": []}
            cur["names"].append(ln[12:16].strip())
            cur["resname"].append(ln[17:21].strip())
            cur["resid"].append(int(ln[22:26]))
            cur["X"].append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            e = ln[76:78].strip()
            cur["el"].append(e.capitalize() if e else None)
        elif ln.startswith(("ENDMDL", "END")):
            close()
    close()
    if not out:
        raise ValueError(f"{Path(path).name} 中没有读到坐标")
    return out


def read_trajectory(top, traj=None, frames=None):
    """top: .gro / .pdb (或 MDAnalysis 能读的拓扑)；traj: .xtc / .trr / .dcd (可选)。"""
    if traj is None:
        suf = Path(top).suffix.lower()
        if suf == ".gro":
            return read_gro(top, frames)
        if suf == ".pdb":
            return read_pdb(top, frames)
        raise ValueError("形貌文件应为 .gro 或 .pdb；轨迹 (.xtc / .trr / .dcd) 需同时给出拓扑")
    base = read_trajectory(top, None, {0})[0]
    try:
        import MDAnalysis as mda
        u = mda.Universe(str(top), str(traj))
        out = []
        for k, ts in enumerate(u.trajectory):
            if frames is not None and k not in frames:
                continue
            from MDAnalysis.lib.mdamath import triclinic_vectors
            out.append(dict(base, X=u.atoms.positions.astype(float).copy(), box=np.asarray(triclinic_vectors(ts.dimensions))))
        return out
    except ImportError:
        pass
    try:
        import mdtraj
        t = mdtraj.load(str(traj), top=str(top))
        return [dict(base, X=t.xyz[k] * NM, box=t.unitcell_vectors[k] * NM) for k in range(t.n_frames)
                if frames is None or k in frames]
    except ImportError:
        raise ValueError("读取 .xtc / .trr / .dcd 需要 MDAnalysis 或 mdtraj (pip install MDAnalysis)；"
                         "也可先用 gmx trjconv -f traj.xtc -s topol.tpr -o frames.gro 转为 .gro") from None


# ----------------------------------------------------------------------------------------------
# 分子与映射
# ----------------------------------------------------------------------------------------------
def split_residues(frame):
    """按残基拆分：残基号或残基名变化处断开 (残基号超过 99999 回绕也能正确处理)。返回 [(残基名, 原子序号数组)]。"""
    r, rn = frame["resid"], frame["resname"]
    cut = [0] + [a for a in range(1, len(r)) if r[a] != r[a - 1] or rn[a] != rn[a - 1]] + [len(r)]
    return [(rn[cut[k]], np.arange(cut[k], cut[k + 1])) for k in range(len(cut) - 1)]


def unwrap(X, box):
    """把一个分子的原子按最小像移到第一个原子附近 (分子尺寸须小于半个盒长)。"""
    inv = np.linalg.inv(box)
    d = X - X[0]
    f = d @ inv
    f -= np.round(f)
    return X[0] + f @ box


def kabsch_batch(P, Q):
    """把参考 Q (n,3) 叠合到各分子 P (M,n,3)：P ≈ Q·Rᵀ + t。返回 (R (M,3,3), t (M,3), RMSD (M,))。"""
    Pc = P - P.mean(1, keepdims=True)
    q0 = Q.mean(0)
    Qc = Q - q0
    H = np.einsum("na,mnb->mab", Qc, Pc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("mab,mbc->mac", Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))))
    D = np.zeros((len(P), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1
    D[:, 2, 2] = d
    R = np.einsum("mab,mbc,mcd->mad", Vt.transpose(0, 2, 1), D, U.transpose(0, 2, 1))
    t = P.mean(1) - np.einsum("mab,b->ma", R, q0)
    fit = np.einsum("na,mba->mnb", Q, R) + t[:, None, :]
    rmsd = np.sqrt(((fit - P) ** 2).sum(-1).mean(1))
    return R, t, rmsd


class Morphology:
    """一个形貌帧中的全部同类分子。ref: (元素, 坐标) 参考单体 (DFT 优化)。
    geometry: "qm" = 用叠合后的参考单体 (默认)；"md" = 用 MD 原子坐标 (需含 H)。"""

    def __init__(self, frame, ref_symbols, ref_Y, resname=None, elements=None, geometry="qm"):
        self.ref_sym = list(ref_symbols)
        self.ref_Y = np.asarray(ref_Y, float)
        self.box = np.asarray(frame["box"], float)
        self.inv = np.linalg.inv(self.box)
        self.geometry = geometry
        res = split_residues(frame)
        heavy_ref = [k for k, e in enumerate(self.ref_sym) if e != "H"]
        names = frame["names"]
        els = frame.get("elements")

        def elem(a):
            if els is not None and els[a]:
                return els[a]
            return guess_element(names[a], elements)
        counts = {}
        for rn, idx in res:
            counts.setdefault(rn, []).append(idx)
        if resname:
            if resname not in counts:
                raise ValueError(f"形貌中没有残基 {resname}；现有：{', '.join(sorted(counts))}")
            pick = resname
        else:
            nh = len(heavy_ref)
            ok = [rn for rn, lst in counts.items()
                  if sum(1 for a in lst[0] if elem(a) != "H") == nh]
            if not ok:
                raise ValueError("没有与参考单体重原子数相同的残基；请指定残基名并检查参考单体")
            pick = max(ok, key=lambda rn: len(counts[rn]))
        self.resname = pick
        mols = counts[pick]
        self.n_other = sum(len(v) for k, v in counts.items() if k != pick)
        sym0 = [elem(a) for a in mols[0]]
        if any(len(m) != len(mols[0]) for m in mols):
            raise ValueError(f"残基 {pick} 的各分子原子数不同")
        self.atom_index = np.array(mols)                            # (M, nat_md)
        Xmd = np.array([unwrap(frame["X"][m], self.box) for m in mols])
        from .polar import match_atoms
        full = sorted(sym0) == sorted(self.ref_sym)
        heavy_md = [k for k, e in enumerate(sym0) if e != "H"]
        if full:
            perm, rm0 = match_atoms(self.ref_sym, self.ref_Y, sym0, Xmd[0])
            self.perm = perm
            hp = perm[heavy_ref]
        else:
            if sorted(sym0[k] for k in heavy_md) != sorted(self.ref_sym[k] for k in heavy_ref):
                raise ValueError(f"残基 {pick} 的组成与参考单体不同 (MD: {''.join(sorted(sym0))})")
            if geometry == "md":
                raise ValueError("MD 中没有氢原子 (联合原子力场)，只能用参考单体几何 (qm)")
            p_h, rm0 = match_atoms([self.ref_sym[k] for k in heavy_ref], self.ref_Y[heavy_ref],
                                   [sym0[k] for k in heavy_md], Xmd[0][heavy_md])
            hp = np.array(heavy_md)[p_h]
            self.perm = None
        self.united_atom = not full
        self.R, self.t, self.fit_rmsd = kabsch_batch(Xmd[:, hp], self.ref_Y[heavy_ref])
        if geometry == "md":
            self.X = Xmd[:, self.perm]
        else:
            self.X = np.einsum("na,mba->mnb", self.ref_Y, self.R) + self.t[:, None, :]
        self.com = self.X.mean(1)
        self.M = len(self.X)
        self.nat = len(self.ref_sym)
        self.resid = frame["resid"][self.atom_index[:, 0]]
        self.pairs = None

    # ---- 近邻 ----
    def neighbors(self, cutoff=4.0):
        """最短重原子间距 < cutoff 的分子对 (含周期像)。返回 [(i, j, 平移整数矢量, 质心矢量 v, 最短距离)]。"""
        hv = np.array([e != "H" for e in self.ref_sym])
        Rmol = np.linalg.norm(self.X[0] - self.com[0], axis=1).max()
        lim = cutoff + 2 * Rmol
        from .polar import _cell_range
        ns = _cell_range(self.box, lim)
        ints = np.array([(a, b, c) for a in range(-ns[0], ns[0] + 1) for b in range(-ns[1], ns[1] + 1)
                         for c in range(-ns[2], ns[2] + 1)], int)
        S = ints @ self.box
        Xh = self.X[:, hv]
        out = []
        for i in range(self.M):
            d = self.com[None, i:, :] + S[:, None, :] - self.com[i]           # (S, M−i, 3)
            s_, jj = np.nonzero(np.linalg.norm(d, axis=-1) < lim)
            for s, jr in zip(s_, jj):
                j = i + jr
                if j == i and tuple(ints[s]) <= (0, 0, 0):
                    continue
                B = Xh[j] + S[s]
                dmin = np.sqrt(((Xh[i][:, None, :] - B[None]) ** 2).sum(-1).min())
                if dmin < cutoff:
                    out.append((i, j, tuple(int(x) for x in ints[s]), self.com[j] + S[s] - self.com[i], float(dmin)))
        self.pairs = out
        self.cutoff = cutoff
        return out

    def pair_atoms(self, k):
        i, j, n, _, _ = self.pairs[k]
        sh = np.array(n) @ self.box
        return self.X[i], self.X[j] + sh

    # ---- 转移积分 ----
    def couplings_eht(self, on_progress=None):
        """全部近邻对的 EHT 转移积分 (eV，带符号)。返回 (J_h, J_e)。"""
        from .eht import build_basis, eht_transfer_integrals, monomer, rotate_coefficients
        if self.pairs is None:
            raise ValueError("请先搜索近邻")
        Jh, Je = np.zeros(len(self.pairs)), np.zeros(len(self.pairs))
        if self.geometry == "qm":
            ref = monomer(self.ref_sym, self.ref_Y)
            A = [(s, *p) for s, p in zip(self.ref_sym, self.ref_Y)]
        else:
            monos = {}
        for k, (i, j, n, v, _) in enumerate(self.pairs):
            if on_progress and k % 200 == 0:
                on_progress(f"EHT 转移积分 {k}/{len(self.pairs)}")
            sh = np.array(n) @ self.box
            if self.geometry == "qm":
                Rrel = self.R[i].T @ self.R[j]
                Yb = self.ref_Y @ Rrel.T + (self.t[j] + sh - self.t[i]) @ self.R[i]
                mb = dict(ref, shells=build_basis(self.ref_sym, Yb)[0],
                          C=rotate_coefficients(ref["shells"], ref["C"], Rrel))
                r = eht_transfer_integrals(A, [(s, *p) for s, p in zip(self.ref_sym, Yb)], esd=False, _mono_b=mb)
            else:
                for m in (i, j):
                    if m not in monos:
                        monos[m] = monomer(self.ref_sym, self.X[m] - self.com[m])
                ma = dict(monos[i], shells=build_basis(self.ref_sym, self.X[i])[0])
                mb = dict(monos[j], shells=build_basis(self.ref_sym, self.X[j] + sh)[0])
                r = eht_transfer_integrals([(s, *p) for s, p in zip(self.ref_sym, self.X[i])],
                                           [(s, *p) for s, p in zip(self.ref_sym, self.X[j] + sh)],
                                           esd=False, _mono_a=ma, _mono_b=mb)
            Jh[k] = r["results"]["HOMO"]["J_eff_meV"] / 1000
            Je[k] = r["results"]["LUMO"]["J_eff_meV"] / 1000
        self.J = {"hole": Jh, "electron": Je}
        self.J_source = "EHT"
        self._keep_raw()
        return Jh, Je

    def couplings_zindo(self, orbitals=None, degeneracy=0.0, on_progress=None):
        """全部近邻对的 ZINDO/MOO 转移积分 (eV)：参考单体的 ZINDO 轨道 (软件内 INDO/S 或 Gaussian ZINDO fchk)
        随各分子的刚性叠合转动、平移 (VOTCA izindo)。"""
        from .zindo import MOO
        if self.pairs is None:
            raise ValueError("请先搜索近邻")
        moo = MOO(self.ref_sym, self.ref_Y, orbitals, degeneracy)
        Jh, Je = np.zeros(len(self.pairs)), np.zeros(len(self.pairs))
        for k, (i, j, n, v, _) in enumerate(self.pairs):
            if on_progress and k % 500 == 0:
                on_progress(f"ZINDO/MOO 转移积分 {k}/{len(self.pairs)}")
            sh = np.array(n) @ self.box
            Rrel = self.R[i].T @ self.R[j]
            r = moo.coupling(Rrel, (self.t[j] + sh - self.t[i]) @ self.R[i])
            Jh[k], Je[k] = r["hole"], r["electron"]
        self.J = {"hole": Jh, "electron": Je}
        self.J_source = "ZINDO/MOO" + ("" if orbitals is None else " (Gaussian ZINDO 轨道)")
        self._keep_raw()
        return Jh, Je

    # ---- DFT 校准 ----
    def _keep_raw(self):
        """保存半经验 (EHT / ZINDO) 的原始 J，供校准使用；重新计算后清除旧的校准。"""
        self.J_raw = {c: v.copy() for c, v in self.J.items()}
        self.J_raw_source = self.J_source
        self.calib = None

    def calibration_pairs(self, n=20, jmin=1e-3):
        """挑出 n 个近邻对做 DFT 校准：在 |J| > jmin (eV) 的对中，按空穴、电子 |J| 之和排序后等间隔取，
        使参考值覆盖从弱到强的整个范围 (只取前 N 对往往集中在同一类构型)。返回排好序的对序号。"""
        raw = getattr(self, "J_raw", None)
        if raw is None:
            raise ValueError("请先用 EHT 或 ZINDO 计算全部近邻对的转移积分，再挑选校准用的二聚体")
        tot = np.abs(raw["hole"]) + np.abs(raw["electron"])
        cand = np.where(tot > jmin)[0]
        cand = cand[np.argsort(tot[cand])]
        if len(cand) <= n:
            return sorted(int(k) for k in cand)
        pick = np.unique(np.round(np.linspace(0, len(cand) - 1, n)).astype(int))
        return sorted(int(k) for k in cand[pick])

    def read_dft(self, outdir):
        """读 export_dimers 目录中已完成的 Gaussian 结果：{对序号: (J_空穴, J_电子)} (eV，带符号)。"""
        from .transfer_integral import calc_transfer_integrals
        out = {}
        for k in range(len(self.pairs)):
            d = Path(outdir) / f"pair_{k:05d}"
            f = [d / f"dimer_{x}.fchk" for x in ("A", "B", "AB")]
            if not all(p.exists() for p in f):
                continue
            r = calc_transfer_integrals(*map(str, f))
            res = r.get("results", r)
            out[k] = (res["HOMO"]["J_eff_meV"] / 1000, res["LUMO"]["J_eff_meV"] / 1000)
        return out

    def read_couplings_csv(self, path):
        """参考转移积分 CSV (pair, J_hole_meV, J_electron_meV) → {对序号: (J_空穴, J_电子)} (eV)。"""
        out = {}
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            g = lambda key: float(r[key]) / 1000 if str(r.get(key, "")).strip() else np.nan  # noqa: E731
            out[int(r["pair"])] = (g("J_hole_meV"), g("J_electron_meV"))
        return out

    def calibrate(self, ref, replace=True, jmin=1e-3, label="DFT"):
        """用少量参考 (DFT) 转移积分校准半经验 J。对 |J| 做过原点最小二乘 |J_ref| ≈ k·|J_半经验|，
        空穴、电子各一个 k；全部近邻对 J ← k·J_半经验，有参考值的对 (replace=True) 直接取参考值。
        只用 |J_ref| > jmin (eV) 的对拟合 (很弱的对相对误差没有意义)。返回 {"hole": {...}, "electron": {...}}。"""
        raw = getattr(self, "J_raw", None)
        if raw is None:
            raise ValueError("请先用 EHT 或 ZINDO 计算全部近邻对的转移积分，再做校准")
        J, res = {}, {}
        for ci, c in enumerate(("hole", "electron")):
            idx = [k for k, v in ref.items() if 0 <= k < len(self.pairs) and np.isfinite(v[ci]) and abs(v[ci]) > jmin]
            if len(idx) < 2:
                raise ValueError(f"{'空穴' if c == 'hole' else '电子'}：|J| > {jmin * 1000:g} meV 的参考对少于 2 个，无法校准")
            x = np.abs(raw[c][idx])
            y = np.abs([ref[k][ci] for k in idx])
            k = float((x * y).sum() / (x * x).sum())
            ss = float(((y - y.mean()) ** 2).sum())
            res[c] = {"k": k, "n": len(idx),
                      "R2": 1 - float(((y - k * x) ** 2).sum()) / ss if ss > 0 else float("nan"),
                      "rel_before": float(np.mean(np.abs(x - y) / y)), "rel_after": float(np.mean(np.abs(k * x - y) / y)),
                      "mae_before_meV": 1000 * float(np.abs(x - y).mean()),
                      "mae_after_meV": 1000 * float(np.abs(k * x - y).mean())}
            J[c] = k * raw[c]
            if replace:                                  # 参考值本身更可靠：|J| 很小的对也直接替换
                for kk, v in ref.items():
                    if 0 <= kk < len(self.pairs) and np.isfinite(v[ci]):
                        J[c][kk] = v[ci]
        self.J = J
        n = max(r["n"] for r in res.values())
        self.calib = {"mode": label, "replace": replace, **res}
        self.J_source = f"{self.J_raw_source} × k ({label} 校准，{n} 对)"
        return res

    def calibrate_scale(self, k_hole=1.0, k_electron=1.0):
        """直接给定比例系数 (例如从另一组二聚体的检验得到)：J ← k·J_半经验。"""
        raw = getattr(self, "J_raw", None)
        if raw is None:
            raise ValueError("请先用 EHT 或 ZINDO 计算全部近邻对的转移积分")
        self.J = {"hole": k_hole * raw["hole"], "electron": k_electron * raw["electron"]}
        self.calib = {"mode": "手动", "replace": False, "hole": {"k": float(k_hole), "n": 0},
                      "electron": {"k": float(k_electron), "n": 0}}
        self.J_source = f"{self.J_raw_source} × k (手动 k_h = {k_hole:g}，k_e = {k_electron:g})"
        return self.calib

    def calibration_text(self):
        c = getattr(self, "calib", None)
        if not c:
            return "未校准"
        L = []
        for car, lab in (("hole", "空穴"), ("electron", "电子")):
            r = c[car]
            if r.get("n"):
                L.append(f"{lab}：k = {r['k']:.3f} (n = {r['n']}，R² = {r['R2']:.3f})；|J| 相对误差 "
                         f"{100 * r['rel_before']:.0f}% → {100 * r['rel_after']:.0f}%，"
                         f"MAE {r['mae_before_meV']:.1f} → {r['mae_after_meV']:.1f} meV")
            else:
                L.append(f"{lab}：k = {r['k']:.3f} (手动)")
        if c.get("replace"):
            L.append("有参考值的对直接使用参考值，其余对使用 k × 半经验 J")
        return "\n".join(L)

    def export_dimers(self, outdir, idx=None, method="B3LYP", basis="6-31G(d)", nproc=8, mem="8GB"):
        """为所选近邻对写出 Gaussian 投影法输入 (每对一个目录 pair_00012)。"""
        from .gaussian_io import write_xyz
        from .transfer_integral import generate_ti_inputs
        outdir = Path(outdir)
        idx = range(len(self.pairs)) if idx is None else idx
        made = []
        for k in idx:
            a, b = self.pair_atoms(k)
            d = outdir / f"pair_{k:05d}"
            d.mkdir(parents=True, exist_ok=True)
            atoms = [(s, *p) for s, p in zip(self.ref_sym, a)] + [(s, *p) for s, p in zip(self.ref_sym, b)]
            write_xyz(d / "dimer.xyz", atoms, f"pair {k}: molecules {self.pairs[k][0]} {self.pairs[k][1]}")
            generate_ti_inputs(d, dimer=str(d / "dimer.xyz"), n_a=self.nat, name="dimer", method=method, basis=basis,
                               nproc=nproc, mem=mem)
            made.append(str(d))
        return made

    def collect_dft(self, outdir):
        """读回 export_dimers 目录中的 Gaussian 结果 (dimer_A/B/AB.fchk)，直接替换这些对的 J；没有结果的对保留原值。
        返回读到的对数。只算了一部分对时，用 calibrate() 把校准系数推广到其余的对更合理。"""
        if getattr(self, "J", None) is None:
            self.J = {"hole": np.full(len(self.pairs), np.nan), "electron": np.full(len(self.pairs), np.nan)}
        got = self.read_dft(outdir)
        for k, (jh, je) in got.items():
            self.J["hole"][k], self.J["electron"][k] = jh, je
        n = len(got)
        self.J_source = "Gaussian" if n == len(self.pairs) else f"Gaussian {n} 对 + 其余 {getattr(self, 'J_source', '')}"
        return n

    def import_couplings(self, path):
        """读入 CSV (pair, J_hole_meV, J_electron_meV)，例如其它程序算出的转移积分。"""
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        if getattr(self, "J", None) is None:
            self.J = {"hole": np.zeros(len(self.pairs)), "electron": np.zeros(len(self.pairs))}
        for r in rows:
            k = int(r["pair"])
            self.J["hole"][k] = float(r["J_hole_meV"]) / 1000
            self.J["electron"][k] = float(r["J_electron_meV"]) / 1000
        self.J_source = Path(path).name
        self.calib = None
        return len(rows)

    # ---- 位点能 ----
    def site_energies(self, charges, mode="static", R=None, alpha_table=None, on_progress=None):
        """mode: "static" (原子电荷，R 默认 30 Å) 或 "polar" (静电 + Thole 诱导，R 默认 14 Å)。
        返回 {"hole": (M,), "electron": (M,)} (eV，载流子能量，已扣除平均值) 与原始值。"""
        from . import polar as P
        if mode == "static":
            R = R or 30.0
            Eh = P.static_site_energies(self.com, self.X, self.box, charges["dq_h"], charges["q0"], R)
            Ee = P.static_site_energies(self.com, self.X, self.box, charges["dq_e"], charges["q0"], R)
        elif mode == "polar":
            R = R or 14.0
            a1 = P.atomic_polarizabilities(self.ref_sym, alpha_table)
            shifts = P.box_shifts(self.box, R)
            Eh, Ee = np.zeros(self.M), np.zeros(self.M)
            for I in range(self.M):
                if on_progress and I % 10 == 0:
                    on_progress(f"极化位点能 {I}/{self.M}")
                Xm, ci = P.box_cluster(self.com, self.X, shifts, I, R)
                e, _ = P._cluster_energies(Xm, ci, a1, charges["q0"], {"h": charges["dq_h"], "e": charges["dq_e"]})
                Eh[I], Ee[I] = e["h"]["total"], e["e"]["total"]
        else:
            raise ValueError("位点能方式应为 static 或 polar")
        self.E_raw = {"hole": Eh, "electron": Ee}
        self.E = {"hole": Eh - Eh.mean(), "electron": Ee - Ee.mean()}
        self.E_mode, self.E_R = mode, R
        return self.E

    def energy_correlation(self, carrier, dr=2.0, rmax=None):
        """位点能空间关联 C(r) = <δE_i δE_j> / σ² (按质心最小像距离分箱)。"""
        E = self.E[carrier]
        rmax = rmax or 0.5 * min(np.linalg.norm(self.box, axis=1))
        f = (self.com[:, None] - self.com[None]) @ self.inv
        f -= np.round(f)
        r = np.linalg.norm(f @ self.box, axis=-1)
        iu = np.triu_indices(self.M, 1)
        r, p = r[iu], (E[:, None] * E[None])[iu] / E.var()
        bins = np.arange(0, rmax + dr, dr)
        k = np.digitize(r, bins) - 1
        C = np.array([p[k == b].mean() if np.any(k == b) else np.nan for b in range(len(bins) - 1)])
        return 0.5 * (bins[1:] + bins[:-1]), C

    # ---- KMC ----
    def graph(self, carrier, jmin=1e-6):
        J = self.J[carrier]
        edges, groups = [], []
        for k, (i, j, n, v, _) in enumerate(self.pairs):
            if not np.isfinite(J[k]) or abs(J[k]) < jmin:
                continue
            edges.append((i, j, np.asarray(v, float), float(J[k]), 1, n))
            edges.append((j, i, -np.asarray(v, float), float(J[k]), 1, tuple(-x for x in n)))
            groups += [k, k]
        return {"n_sites": self.M, "edges": edges, "edge_group": groups, "kind": "amorphous", "periodic": True,
                "lattice": self.box}

    def kmc(self, carrier, lam, T=300.0, field=0.0, walkers=1000, steps=4000, seed=1, use_energies=True,
            method="marcus"):
        """零场扩散迁移率 (field = 0) 或沿 x、y、z 三个方向的场迁移率平均 (field / V cm⁻¹)。"""
        from .kmc import run_kmc
        g = self.graph(carrier)
        offs = self.E[carrier] if (use_energies and getattr(self, "E", None) is not None) else None
        kw = dict(method=method, carrier=carrier, supercell=1, walkers=walkers, steps=steps, seed=seed,
                  site_offsets=offs)
        if not field:
            r = run_kmc(g, lam, T, **kw)
            return {"mu_avg": r["mu_avg"], "mu_principal": r["mu_principal"], "err": r["mu_avg_err"], "raw": r}
        mus, errs = [], []
        for ax in range(3):
            F = np.zeros(3)
            F[ax] = field
            r = run_kmc(g, lam, T, field=F, **kw)
            mus.append(r["mu_field_cm2_Vs"])
            errs.append(r["mu_field_err"])
        return {"mu_avg": float(np.mean(mus)), "mu_axes": mus, "err": float(np.sqrt(np.sum(np.square(errs)))) / 3}

    # ---- 汇总与导出 ----
    def summary(self):
        L = [f"残基 {self.resname}：{self.M} 个分子 (另有 {self.n_other} 个其它残基，作为背景忽略)；"
             f"盒子 {' × '.join(f'{x:.1f}' for x in np.linalg.norm(self.box, axis=1))} Å；"
             f"{'联合原子 (无 H)' if self.united_atom else '全原子'}；几何 = "
             f"{'叠合后的参考单体' if self.geometry == 'qm' else 'MD 原子坐标'}",
             f"参考单体叠合 RMSD (重原子)：平均 {self.fit_rmsd.mean():.2f} Å，最大 {self.fit_rmsd.max():.2f} Å"]
        if self.pairs is not None:
            nb = np.bincount(np.concatenate([[p[0] for p in self.pairs], [p[1] for p in self.pairs]]),
                             minlength=self.M) if self.pairs else np.zeros(self.M)
            L.append(f"近邻 (最短重原子距离 < {self.cutoff:g} Å)：{len(self.pairs)} 对，每个分子平均 {nb.mean():.1f} 个，"
                     f"最少 {nb.min()}")
        if getattr(self, "J", None) is not None:
            for c, lab in (("hole", "空穴"), ("electron", "电子")):
                a = 1000 * np.abs(self.J[c][np.isfinite(self.J[c])])
                if len(a):
                    L.append(f"{lab} |J| ({self.J_source})：中位数 {np.median(a):.1f} meV，"
                             f"均方根 {np.sqrt((a ** 2).mean()):.1f} meV，最大 {a.max():.0f} meV，"
                             f"> 10 meV 的占 {np.mean(a > 10):.0%}")
        if getattr(self, "calib", None):
            L.append("DFT 校准：" + self.calibration_text().replace("\n", "；"))
        if getattr(self, "E", None) is not None:
            L.append(f"位点能 ({'静电' if self.E_mode == 'static' else '静电 + 诱导'}，R = {self.E_R:g} Å)："
                     f"σ_空穴 = {1000 * self.E['hole'].std():.0f} meV，σ_电子 = {1000 * self.E['electron'].std():.0f} meV")
            if self.pairs:
                for c, lab in (("hole", "空穴"), ("electron", "电子")):
                    dE = np.array([self.E[c][j] - self.E[c][i] for i, j, *_ in self.pairs])
                    L.append(f"  近邻对 ΔE 的标准差 ({lab})：{1000 * dE.std():.0f} meV")
        return "\n".join(L)

    def export(self, prefix):
        """写出 prefix_sites.csv、prefix_pairs.csv。"""
        prefix = str(prefix)
        with open(prefix + "_sites.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["molecule", "resid", "x_A", "y_A", "z_A", "fit_rmsd_A", "E_hole_eV", "E_electron_eV"])
            for m in range(self.M):
                e = self.E if getattr(self, "E", None) is not None else None
                w.writerow([m, int(self.resid[m]), *(f"{x:.3f}" for x in self.com[m]), f"{self.fit_rmsd[m]:.3f}",
                            f"{e['hole'][m]:.5f}" if e else "", f"{e['electron'][m]:.5f}" if e else ""])
        with open(prefix + "_pairs.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            raw = getattr(self, "J_raw", None) if getattr(self, "calib", None) else None
            w.writerow(["pair", "i", "j", "na", "nb", "nc", "r_com_A", "d_min_A", "J_hole_meV", "J_electron_meV"]
                       + (["J_hole_semi_meV", "J_electron_semi_meV"] if raw is not None else []))
            J = getattr(self, "J", None)
            for k, (i, j, n, v, dm) in enumerate(self.pairs or []):
                w.writerow([k, i, j, *n, f"{np.linalg.norm(v):.3f}", f"{dm:.3f}",
                            f"{1000 * J['hole'][k]:.3f}" if J is not None else "",
                            f"{1000 * J['electron'][k]:.3f}" if J is not None else ""]
                           + ([f"{1000 * raw['hole'][k]:.3f}", f"{1000 * raw['electron'][k]:.3f}"] if raw is not None else []))
        return [prefix + "_sites.csv", prefix + "_pairs.csv"]


def plot_overview(fig, morph, kmc=None):
    """2×2：|J|² 分布、位点能分布、位点能空间关联、近邻质心距离分布。"""
    cols = {"hole": "#2C6FB7", "electron": "#C44A32"}
    labs = {"hole": "空穴", "electron": "电子"}
    ax = fig.subplots(2, 2)
    a = ax[0, 0]
    if getattr(morph, "J", None) is not None:
        for c in ("hole", "electron"):
            j = np.abs(morph.J[c][np.isfinite(morph.J[c])]) * 1000
            j = j[j > 1e-4]
            a.hist(np.log10(j ** 2), bins=40, histtype="step", color=cols[c], lw=1.4, label=labs[c])
        a.set_xlabel("log₁₀(J² / meV²)")
        a.set_ylabel("对数")
        a.legend(fontsize=8, frameon=False)
    a.set_title("(a) 转移积分", fontsize=9, loc="left")
    a = ax[0, 1]
    if getattr(morph, "E", None) is not None:
        for c in ("hole", "electron"):
            a.hist(1000 * morph.E[c], bins=30, histtype="step", color=cols[c], lw=1.4,
                   label=f"{labs[c]} σ = {1000 * morph.E[c].std():.0f} meV")
        a.set_xlabel("位点能 (meV)")
        a.set_ylabel("分子数")
        a.legend(fontsize=8, frameon=False)
    a.set_title("(b) 位点能", fontsize=9, loc="left")
    a = ax[1, 0]
    if getattr(morph, "E", None) is not None:
        for c in ("hole", "electron"):
            r, C = morph.energy_correlation(c)
            a.plot(r, C, "o-", ms=3, color=cols[c], label=labs[c])
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xlabel("质心距离 (Å)")
        a.set_ylabel("C(r)")
        a.legend(fontsize=8, frameon=False)
    a.set_title("(c) 位点能空间关联", fontsize=9, loc="left")
    a = ax[1, 1]
    if morph.pairs:
        a.hist([np.linalg.norm(p[3]) for p in morph.pairs], bins=30, color="0.55")
        a.set_xlabel("近邻质心距离 (Å)")
        a.set_ylabel("对数")
    a.set_title("(d) 近邻", fontsize=9, loc="left")
    fig.tight_layout()
