"""波函数分析 (直接由 Gaussian fchk 计算，不依赖 Multiwfn)：

* 基组：s / p / sp / d (5D、6D) / f (7F、10F)，函数顺序与归一化遵循 Gaussian 约定
* 分子轨道、电子密度的格点值与 Gaussian cube 文件读写
* Mulliken 电荷 (Mulliken, J. Chem. Phys. 1955, 23, 1833) 与 Mayer 键级 (Mayer, Chem. Phys. Lett. 1983, 97, 270)
* 分子表面静电势：电子密度 0.001 a.u. 等值面 (marching cubes) 上插值 ESP，给出表面极值点与统计量
  严格 ESP 由 Gaussian 自带的 cubegen 计算；无 cubegen 时可用原子电荷近似 (仅作定性参考)
"""
import re
import subprocess
from pathlib import Path

import numpy as np

from .gaussian_io import BOHR2ANG, HARTREE2EV, read_fchk

BOHR = 1 / BOHR2ANG                 # Å → bohr
KCAL = 627.5095                     # Hartree → kcal/mol
_DF = {0: 1, 1: 1, 2: 3, 3: 15, 4: 105}                       # (2L−1)!!
_CART = {1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
         2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
         3: [(3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 2, 0), (2, 1, 0), (2, 0, 1), (1, 0, 2), (0, 1, 2), (0, 2, 1),
             (1, 1, 1)]}
_NFUNC = {0: 1, 1: 3, -1: 4, 2: 6, -2: 5, 3: 10, -3: 7}


def _dfact(n):
    return 1 if n <= 0 else n * _dfact(n - 2)


# ----------------------------------------------------------------------------------------------
# 读取
# ----------------------------------------------------------------------------------------------
class Wavefunction:
    def __init__(self, path):
        d = read_fchk(path)
        self.path = str(path)
        self.Z = np.array(d["Atomic numbers"], int)
        self.zeff = np.array(d.get("Nuclear charges", self.Z), float)       # 有赝势时为价电子数
        self.coords = np.array(d["Current cartesian coordinates"]).reshape(-1, 3)    # bohr
        self.charge = int(d.get("Charge", 0))
        self.mult = int(d.get("Multiplicity", 1))
        self.nalpha = int(d["Number of alpha electrons"])
        self.nbeta = int(d.get("Number of beta electrons", self.nalpha))
        self.nbf = int(d["Number of basis functions"])
        self.nmo = int(d.get("Number of independent functions", self.nbf))
        self.ea = np.array(d["Alpha Orbital Energies"])
        self.Ca = np.array(d["Alpha MO coefficients"]).reshape(self.nmo, self.nbf).T
        self.open_shell = "Beta MO coefficients" in d
        if self.open_shell:
            self.eb = np.array(d["Beta Orbital Energies"])
            self.Cb = np.array(d["Beta MO coefficients"]).reshape(self.nmo, self.nbf).T
        self.P = _tri(d["Total SCF Density"], self.nbf) if "Total SCF Density" in d else None
        self.Ps = _tri(d["Spin SCF Density"], self.nbf) if "Spin SCF Density" in d else None
        self.types = [int(t) for t in d["Shell types"]]
        if any(abs(t) > 3 for t in self.types):
            self.max_l = max(abs(t) for t in self.types)
        nprim = [int(n) for n in d["Number of primitives per shell"]]
        exps = np.array(d["Primitive exponents"])
        cc = np.array(d["Contraction coefficients"])
        cp = np.array(d.get("P(S=P) Contraction coefficients", np.zeros_like(cc)))
        smap = [int(a) - 1 for a in d["Shell to atom map"]]
        scrd = np.array(d.get("Coordinates of each shell", self.coords[smap].ravel())).reshape(-1, 3)
        self.shells, k, off = [], 0, 0
        for t, n, a, xyz in zip(self.types, nprim, smap, scrd):
            self.shells.append({"type": t, "a": exps[k:k + n], "c": cc[k:k + n], "cp": cp[k:k + n], "atom": a,
                                "center": xyz, "offset": off})
            k += n
            off += _NFUNC.get(t, 2 * abs(t) + 1 if t < 0 else (abs(t) + 1) * (abs(t) + 2) // 2)
        self.atom_of_bf = np.concatenate([[s["atom"]] * _NFUNC.get(s["type"], 0) for s in self.shells]).astype(int)
        self.symbols = [_symbol(z) for z in self.Z]

    # ---- 轨道信息 ----
    def orbitals(self, spin="alpha"):
        e = self.ea if spin == "alpha" else self.eb
        nocc = self.nalpha if spin == "alpha" else self.nbeta
        out = []
        for i, en in enumerate(e):
            k = i - (nocc - 1)
            lab = "HOMO" if k == 0 else (f"HOMO{k}" if k < 0 else ("LUMO" if k == 1 else f"LUMO+{k - 1}"))
            occ = (1.0 if (self.open_shell or spin == "beta") else 2.0) if i < nocc else 0.0
            out.append({"index": i + 1, "label": lab, "energy_eV": float(en * HARTREE2EV), "occ": occ})
        return out

    def gap_eV(self):
        return float((self.ea[self.nalpha] - self.ea[self.nalpha - 1]) * HARTREE2EV)

    # ---- 基函数格点值 ----
    def basis_values(self, pts):
        """pts: (n, 3) bohr。返回 (nbf, n)。"""
        out = np.zeros((self.nbf, len(pts)))
        for sh in self.shells:
            t, o = sh["type"], sh["offset"]
            d = pts - sh["center"]
            r2 = (d ** 2).sum(1)
            x, y, z = d.T
            if t in (0, -1):
                out[o] = _radial(sh["a"], sh["c"], 0, r2)
                if t == -1:
                    rp = _radial(sh["a"], sh["cp"], 1, r2)
                    out[o + 1], out[o + 2], out[o + 3] = x * rp, y * rp, z * rp
                continue
            L = abs(t)
            if t > 0:                                        # 笛卡尔函数，各自归一化
                rad = _radial(sh["a"], sh["c"], L, r2, cart=True)
                for k, (lx, ly, lz) in enumerate(_CART[L]):
                    norm = np.sqrt(_DF[L] / (_dfact(2 * lx - 1) * _dfact(2 * ly - 1) * _dfact(2 * lz - 1)))
                    out[o + k] = rad * norm * x ** lx * y ** ly * z ** lz
            else:                                            # 纯球谐函数 (以 x^L 的归一化为基准)
                rad = _radial(sh["a"], sh["c"], L, r2)
                for k, poly in enumerate(_pure(L, x, y, z)):
                    out[o + k] = rad * poly
        return out

    def mo_values(self, pts, indices, spin="alpha", chunk=20000):
        C = (self.Ca if spin == "alpha" else self.Cb)[:, indices]
        out = np.zeros((len(indices), len(pts)))
        for s in range(0, len(pts), chunk):
            out[:, s:s + chunk] = C.T @ self.basis_values(pts[s:s + chunk])
        return out

    def density_values(self, pts, chunk=20000):
        out = np.zeros(len(pts))
        occ_a = self.Ca[:, :self.nalpha]
        occ_b = (self.Cb if self.open_shell else self.Ca)[:, :self.nbeta]
        for s in range(0, len(pts), chunk):
            phi = self.basis_values(pts[s:s + chunk])
            out[s:s + chunk] = ((occ_a.T @ phi) ** 2).sum(0) + ((occ_b.T @ phi) ** 2).sum(0)
        return out

    # ---- 格点 ----
    def grid(self, spacing_A=0.25, margin_A=4.0):
        lo = self.coords.min(0) - margin_A * BOHR
        hi = self.coords.max(0) + margin_A * BOHR
        h = spacing_A * BOHR
        n = np.ceil((hi - lo) / h).astype(int) + 1
        return {"origin": lo, "axes": np.eye(3) * h, "shape": tuple(n)}

    def grid_points(self, g):
        i, j, k = np.meshgrid(*[np.arange(n) for n in g["shape"]], indexing="ij")
        idx = np.stack([i.ravel(), j.ravel(), k.ravel()], 1)
        return g["origin"] + idx @ g["axes"]

    def mo_cube(self, index, spin="alpha", spacing_A=0.25):
        g = self.grid(spacing_A)
        v = self.mo_values(self.grid_points(g), [index], spin)[0]
        return dict(g, data=v.reshape(g["shape"]), Z=self.Z, coords=self.coords,
                    title=f"MO {index + 1} ({spin})", zeff=self.zeff)

    def density_cube(self, spacing_A=0.25):
        g = self.grid(spacing_A)
        v = self.density_values(self.grid_points(g))
        return dict(g, data=v.reshape(g["shape"]), Z=self.Z, coords=self.coords, title="electron density",
                    zeff=self.zeff)

    # ---- 布居与键级 ----
    def overlap(self):
        """重叠矩阵：没有删除线性相关基函数时 S = (C Cᵀ)⁻¹ (C 为全部 MO 系数)，精确且不需积分。"""
        if self.nmo != self.nbf:
            raise ValueError("该计算删除了线性相关的基函数 (独立函数数 < 基函数数)，无法由 MO 系数还原重叠矩阵；"
                             "请加 iop(3/32=2) 重新计算")
        return np.linalg.inv(self.Ca @ self.Ca.T)

    def densities(self):
        if self.P is not None:
            P = self.P
            Ps = self.Ps if self.Ps is not None else np.zeros_like(P)
        else:
            Pa = self.Ca[:, :self.nalpha] @ self.Ca[:, :self.nalpha].T
            Cb = self.Cb if self.open_shell else self.Ca
            Pb = Cb[:, :self.nbeta] @ Cb[:, :self.nbeta].T
            P, Ps = Pa + Pb, Pa - Pb
        return (P + Ps) / 2, (P - Ps) / 2

    def mulliken_mayer(self):
        """返回 Mulliken 电荷 (e)、原子自旋布居与 Mayer 键级矩阵。"""
        S = self.overlap()
        Pa, Pb = self.densities()
        PSa, PSb = Pa @ S, Pb @ S
        PS = PSa + PSb
        nat = len(self.Z)
        pop = np.zeros(nat)
        spin = np.zeros(nat)
        np.add.at(pop, self.atom_of_bf, np.diag(PS))
        np.add.at(spin, self.atom_of_bf, np.diag(PSa - PSb))
        q = self.zeff - pop
        Ma = PSa * PSa.T
        Mb = PSb * PSb.T
        M = 2 * (Ma + Mb)                                   # 闭壳层时即 Σ (PS)_μν (PS)_νμ
        B = np.zeros((nat, nat))
        np.add.at(B, (self.atom_of_bf[:, None].repeat(self.nbf, 1), self.atom_of_bf[None, :].repeat(self.nbf, 0)), M)
        np.fill_diagonal(B, 0.0)
        return {"charges": q, "spin": spin, "bond_orders": B, "total_charge": float(q.sum())}


def _symbol(z):
    from .substitution import ELEMENTS
    return ELEMENTS[int(z) - 1]


def _tri(vals, n):
    M = np.zeros((n, n))
    M[np.tril_indices(n)] = vals
    return M + np.tril(M, -1).T


def _radial(a, c, L, r2, cart=False):
    """收缩径向部分 Σ c_i N_i exp(−a_i r²)，N 为 x^L 型原函数的归一化常数；整体再按收缩函数归一化。"""
    a = np.asarray(a, float)
    c = np.asarray(c, float)
    N = (2 * a / np.pi) ** 0.75 * (4 * a) ** (L / 2) / np.sqrt(_DF[L])
    s = (c[:, None] * c[None] * (2 * np.sqrt(a[:, None] * a[None]) / (a[:, None] + a[None])) ** (L + 1.5)).sum()
    c = c / np.sqrt(s)
    return (c * N) @ np.exp(-np.outer(a, r2))


def _pure(L, x, y, z):
    """Gaussian 的纯球谐顺序：0, +1, −1, +2, −2, (+3, −3)；以 x^L 的归一化为基准时下列多项式均已归一化。"""
    if L == 2:
        s3 = np.sqrt(3.0)
        return [(2 * z * z - x * x - y * y) / 2, s3 * x * z, s3 * y * z, s3 / 2 * (x * x - y * y), s3 * x * y]
    if L == 3:
        r2 = x * x + y * y
        return [z * (2 * z * z - 3 * r2) / 2, np.sqrt(3 / 8) * x * (4 * z * z - r2), np.sqrt(3 / 8) * y * (4 * z * z - r2),
                np.sqrt(15) / 2 * z * (x * x - y * y), np.sqrt(15) * x * y * z, np.sqrt(5 / 8) * x * (x * x - 3 * y * y),
                np.sqrt(5 / 8) * y * (3 * x * x - y * y)]
    raise ValueError(f"暂不支持 L = {L} 的纯球谐函数 (g 及以上)")


# ----------------------------------------------------------------------------------------------
# cube 文件
# ----------------------------------------------------------------------------------------------
def write_cube(path, cube):
    o, ax, shape, data = cube["origin"], cube["axes"], cube["shape"], cube["data"]
    lines = ["Mobilis cube", cube.get("title", ""), f"{len(cube['Z']):5d}{o[0]:12.6f}{o[1]:12.6f}{o[2]:12.6f}"]
    for n, v in zip(shape, ax):
        lines.append(f"{n:5d}{v[0]:12.6f}{v[1]:12.6f}{v[2]:12.6f}")
    zeff = cube.get("zeff", cube["Z"])
    for z, q, r in zip(cube["Z"], zeff, cube["coords"]):
        lines.append(f"{int(z):5d}{float(q):12.6f}{r[0]:12.6f}{r[1]:12.6f}{r[2]:12.6f}")
    flat = data.reshape(shape[0] * shape[1], shape[2])
    for row in flat:
        for k in range(0, len(row), 6):
            lines.append("".join(f"{v:13.5E}" for v in row[k:k + 6]))
    Path(path).write_text("\n".join(lines) + "\n")


def read_cube(path):
    L = Path(path).read_text(errors="ignore").splitlines()
    p = L[2].split()
    nat = int(p[0])
    origin = np.array(list(map(float, p[1:4])))
    shape, axes = [], []
    for k in range(3):
        q = L[3 + k].split()
        shape.append(abs(int(q[0])))
        axes.append(list(map(float, q[1:4])))
    if int(L[3].split()[0]) < 0:
        raise ValueError("cube 中长度单位为 Å (负的格点数)，请用 bohr 单位的 cube")
    Z, coords = [], []
    for k in range(abs(nat)):
        q = L[6 + k].split()
        Z.append(int(q[0]))
        coords.append(list(map(float, q[2:5])))
    start = 6 + abs(nat) + (1 if nat < 0 else 0)
    vals = np.array(" ".join(L[start:]).replace("D", "E").split(), float)
    n = shape[0] * shape[1] * shape[2]
    if len(vals) < n:
        raise ValueError(f"{Path(path).name}: 格点数据不完整")
    return {"origin": origin, "axes": np.array(axes), "shape": tuple(shape), "data": vals[:n].reshape(shape),
            "Z": np.array(Z), "coords": np.array(coords), "title": L[1].strip()}


# ----------------------------------------------------------------------------------------------
# 等值面与表面静电势
# ----------------------------------------------------------------------------------------------
def isosurface(cube, value):
    """返回 (顶点 Å, 三角面)。value 可为负 (轨道负相位)。"""
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise ImportError("等值面需要 scikit-image：pip install scikit-image") from exc
    data = cube["data"]
    if not (data.min() < value < data.max()):
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    v, f, _, _ = marching_cubes(data, level=value)
    xyz = cube["origin"] + v @ cube["axes"]
    return xyz * BOHR2ANG, f.astype(int)


def esp_surface(dens_cube, esp_cube=None, iso=0.001, charges=None):
    """电子密度等值面上的静电势 (kcal/mol)。esp_cube 为同一格点的 ESP cube；否则用原子电荷近似。"""
    from scipy.ndimage import map_coordinates
    verts, faces = isosurface(dens_cube, iso)
    if not len(verts):
        raise ValueError("密度等值面为空：请检查 cube 或等值")
    vb = verts * BOHR
    if esp_cube is not None:
        if tuple(esp_cube["shape"]) != tuple(dens_cube["shape"]) or np.abs(esp_cube["origin"] - dens_cube["origin"]).max() > 1e-4:
            raise ValueError("密度与 ESP 的 cube 格点不一致 (请用同一 fchk、同样的格点参数生成)")
        frac = np.linalg.solve(np.asarray(esp_cube["axes"]).T, (vb - esp_cube["origin"]).T)
        esp = map_coordinates(esp_cube["data"], frac, order=1, mode="nearest")
        method = "ESP cube (cubegen)"
    else:
        if charges is None:
            raise ValueError("没有 ESP cube 时需要原子电荷")
        R = dens_cube["coords"]
        esp = (np.asarray(charges)[None] / np.linalg.norm(vb[:, None] - R[None], axis=-1)).sum(1)
        method = "原子电荷近似"
    esp = esp * KCAL
    ext = surface_extrema(verts, faces, esp)
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    fval = esp[faces].mean(1)
    stats = {"min": float(esp.min()), "max": float(esp.max()), "mean": float((fval * area).sum() / area.sum()),
             "area_A2": float(area.sum()), "pos_area_A2": float(area[fval > 0].sum()),
             "neg_area_A2": float(area[fval < 0].sum()),
             "sigma2_tot": float((area * (fval - (fval * area).sum() / area.sum()) ** 2).sum() / area.sum())}
    return {"verts": verts, "faces": faces, "esp": esp, "extrema": ext, "stats": stats, "method": method}


def surface_extrema(verts, faces, val, rings=3, merge_A=0.6):
    """网格上的局部极值：在 rings 层邻域内为最小 / 最大的顶点，并合并相距 < merge_A 的点。"""
    n = len(verts)
    nb = [set() for _ in range(n)]
    for a, b, c in faces:
        nb[a].update((b, c))
        nb[b].update((a, c))
        nb[c].update((a, b))
    out = []
    for i in range(n):
        ring, front = {i}, {i}
        for _ in range(rings):
            front = set().union(*(nb[k] for k in front)) - ring
            ring |= front
        ring.discard(i)
        if not ring:
            continue
        vr = val[list(ring)]
        if val[i] < vr.min():
            out.append(("min", i))
        elif val[i] > vr.max():
            out.append(("max", i))
    kept = []
    for kind, i in sorted(out, key=lambda t: val[t[1]] if t[0] == "min" else -val[t[1]]):
        if all(np.linalg.norm(verts[i] - verts[j]) > merge_A for _, j, _ in kept):
            kept.append((kind, i, float(val[i])))
    return [{"kind": k, "xyz": verts[i], "value": v} for k, i, v in kept]


# ----------------------------------------------------------------------------------------------
# Gaussian cubegen
# ----------------------------------------------------------------------------------------------
def find_cubegen(gaussian_exe=None):
    import shutil
    cands = []
    if gaussian_exe:
        d = Path(gaussian_exe).parent
        cands += [d / "cubegen.exe", d / "cubegen"]
    w = shutil.which("cubegen")
    if w:
        cands.append(Path(w))
    return next((str(c) for c in cands if c.is_file()), None)


def run_cubegen(exe, fchk, kind, out, npts="-2", cwd=None):
    """kind: 'density=scf' / 'potential=scf' / 'MO=homo' 等。npts: -1 粗 / -2 中 / -3 细 (Gaussian 约定)。"""
    import sys
    flags = 0x08000000 if sys.platform.startswith("win") else 0
    r = subprocess.run([exe, "0", kind, str(fchk), str(out), str(npts), "h"], cwd=cwd, capture_output=True,
                       timeout=3600, creationflags=flags)
    if not Path(out).exists():
        raise RuntimeError(f"cubegen 失败：{r.stdout.decode(errors='ignore')[-300:]} {r.stderr.decode(errors='ignore')[-300:]}")
    return out


# ----------------------------------------------------------------------------------------------
# NBO (Gaussian 输出中的 NBO 3.x / 7 结果)
# ----------------------------------------------------------------------------------------------
_E2_LINE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*/\s*(\d+)\.\s*(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


def read_nbo(log):
    """读取二阶微扰能 E(2) 表 (kcal/mol) 与自然电荷 (NPA)。"""
    text = Path(log).read_text(errors="ignore")
    k = text.rfind("SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS")
    e2 = []
    if k >= 0:
        block = text[k:]
        end = block.find("NATURAL BOND ORBITALS (Summary)")
        for line in (block[:end] if end > 0 else block).splitlines():
            m = _E2_LINE.match(line)
            if m:
                e2.append({"donor": f"{m.group(1)}. {m.group(2)}", "acceptor": f"{m.group(3)}. {m.group(4)}",
                           "E2": float(m.group(5)), "dE": float(m.group(6)), "F": float(m.group(7)),
                           "donor_atoms": [int(x) for x in re.findall(r"[A-Z][a-z]?\s*(\d+)", m.group(2))],
                           "acceptor_atoms": [int(x) for x in re.findall(r"[A-Z][a-z]?\s*(\d+)", m.group(4))],
                           "donor_type": m.group(2).split("(")[0].strip(), "acceptor_type": m.group(4).split("(")[0].strip()})
    npa = []
    k = text.rfind("Summary of Natural Population Analysis")
    if k >= 0:
        for line in text[k:].splitlines()[6:]:
            p = line.split()
            if len(p) >= 3 and p[1].isdigit():
                try:
                    npa.append((p[0], int(p[1]), float(p[2])))
                except ValueError:
                    break
            elif npa:
                break
    if not e2 and not npa:
        raise ValueError(f"{Path(log).name} 中没有 NBO 结果 (计算时需加 pop=nbo 或 pop=nbo7)")
    return {"E2": e2, "npa": npa}


# ----------------------------------------------------------------------------------------------
# 基函数的解析导数 (用于密度梯度 / Hessian：IGMH、AIM)
# 每个基函数写成 Σ_k d_k x^l y^m z^n · Σ_p c_p exp(−a_p r²)；对单项式逐项求导
# ----------------------------------------------------------------------------------------------
def _monomials(t):
    """返回该壳层各函数的 [(系数, (l, m, n))]，以及角动量 L。系数已含笛卡尔函数的单独归一化或球谐组合系数。"""
    if t == 0:
        return 0, [[(1.0, (0, 0, 0))]]
    if t == 1:
        return 1, [[(1.0, e)] for e in _CART[1]]
    L = abs(t)
    if t > 0:
        out = []
        for (lx, ly, lz) in _CART[L]:
            nrm = np.sqrt(_DF[L] / (_dfact(2 * lx - 1) * _dfact(2 * ly - 1) * _dfact(2 * lz - 1)))
            out.append([(nrm, (lx, ly, lz))])
        return L, out
    s3 = np.sqrt(3.0)
    if L == 2:
        return 2, [[(1.0, (0, 0, 2)), (-0.5, (2, 0, 0)), (-0.5, (0, 2, 0))], [(s3, (1, 0, 1))], [(s3, (0, 1, 1))],
                   [(s3 / 2, (2, 0, 0)), (-s3 / 2, (0, 2, 0))], [(s3, (1, 1, 0))]]
    if L == 3:
        c1, c2, c3 = np.sqrt(3 / 8), np.sqrt(15), np.sqrt(5 / 8)
        return 3, [[(1.0, (0, 0, 3)), (-1.5, (2, 0, 1)), (-1.5, (0, 2, 1))],
                   [(4 * c1, (1, 0, 2)), (-c1, (3, 0, 0)), (-c1, (1, 2, 0))],
                   [(4 * c1, (0, 1, 2)), (-c1, (2, 1, 0)), (-c1, (0, 3, 0))],
                   [(c2 / 2, (2, 0, 1)), (-c2 / 2, (0, 2, 1))], [(c2, (1, 1, 1))],
                   [(c3, (3, 0, 0)), (-3 * c3, (1, 2, 0))], [(3 * c3, (2, 1, 0)), (-c3, (0, 3, 0))]]
    raise ValueError(f"暂不支持 L = {L}")


def _contraction(a, c, L):
    a = np.asarray(a, float)
    c = np.asarray(c, float)
    N = (2 * a / np.pi) ** 0.75 * (4 * a) ** (L / 2) / np.sqrt(_DF[L])
    s = (c[:, None] * c[None] * (2 * np.sqrt(a[:, None] * a[None]) / (a[:, None] + a[None])) ** (L + 1.5)).sum()
    return c / np.sqrt(s) * N


def _poly1d(x, l, a, order):
    """x^l e 的 0、1、2 阶导数中的多项式因子 (e = exp(−a x²) 另乘)。"""
    p0 = x ** l
    if order == 0:
        return p0, None, None
    p1 = (l * x ** (l - 1) if l > 0 else 0.0) - 2 * a * x ** (l + 1)
    if order == 1:
        return p0, p1, None
    p2 = (l * (l - 1) * x ** (l - 2) if l > 1 else 0.0) - 2 * a * (2 * l + 1) * x ** l + 4 * a * a * x ** (l + 2)
    return p0, p1, p2


def _shell_parts(sh):
    """[(L, 函数单项式列表, 收缩系数 (含归一化), 指数, 函数在壳层中的起始序号)]；SP 壳层拆成 s、p 两部分。"""
    t = sh["type"]
    if t == -1:
        return [(0, [[(1.0, (0, 0, 0))]], _contraction(sh["a"], sh["c"], 0), sh["a"], 0),
                (1, [[(1.0, e)] for e in _CART[1]], _contraction(sh["a"], sh["cp"], 1), sh["a"], 1)]
    L, funcs = _monomials(t)
    return [(L, funcs, _contraction(sh["a"], sh["c"], L), sh["a"], 0)]


def basis_derivs(wfn, pts, order=2, cutoff=1e-10):
    """基函数及其导数。返回 (val (nbf, n), grad (3, nbf, n) 或 None, hess (6, nbf, n) 或 None)，hess 顺序 xx yy zz xy xz yz。
    按与点集包围球的距离筛选壳层 (c·exp(−a r²) < cutoff 的壳层跳过)。"""
    n = len(pts)
    val = np.zeros((wfn.nbf, n))
    grad = np.zeros((3, wfn.nbf, n)) if order >= 1 else None
    hess = np.zeros((6, wfn.nbf, n)) if order >= 2 else None
    cen = pts.mean(0)
    rad = np.sqrt(((pts - cen) ** 2).sum(1).max()) if n else 0.0
    lncut = -np.log(cutoff)
    for sh in wfn.shells:
        dmin = max(0.0, np.linalg.norm(sh["center"] - cen) - rad)
        if (np.min(sh["a"]) * dmin * dmin) > lncut + 10:
            continue
        d = pts - sh["center"]
        x, y, z = d.T
        r2 = (d ** 2).sum(1)
        for L, funcs, cc, aa, start in _shell_parts(sh):
            for cp, a in zip(cc, aa):
                e = cp * np.exp(-a * r2)
                cache = {}

                def P(ax, l, xx=None):
                    key = (ax, l)
                    if key not in cache:
                        cache[key] = _poly1d((x, y, z)[ax], l, a, order)
                    return cache[key]
                for k, func in enumerate(funcs):
                    row = sh["offset"] + start + k
                    for coef, (l, m, nn) in func:
                        X, Y, Z = P(0, l), P(1, m), P(2, nn)
                        ce = coef * e
                        val[row] += ce * X[0] * Y[0] * Z[0]
                        if order >= 1:
                            grad[0, row] += ce * X[1] * Y[0] * Z[0]
                            grad[1, row] += ce * X[0] * Y[1] * Z[0]
                            grad[2, row] += ce * X[0] * Y[0] * Z[1]
                        if order >= 2:
                            hess[0, row] += ce * X[2] * Y[0] * Z[0]
                            hess[1, row] += ce * X[0] * Y[2] * Z[0]
                            hess[2, row] += ce * X[0] * Y[0] * Z[2]
                            hess[3, row] += ce * X[1] * Y[1] * Z[0]
                            hess[4, row] += ce * X[1] * Y[0] * Z[1]
                            hess[5, row] += ce * X[0] * Y[1] * Z[1]
    return val, grad, hess


def density_derivs(wfn, pts, order=2, block=3000, kinetic=False):
    """ρ、∇ρ (3, n)、Hessian (6, n)；kinetic=True 时另给 Lagrange 动能密度 G = ½ Σ n_i |∇ψ_i|² (a.u.)。"""
    n = len(pts)
    rho = np.zeros(n)
    g = np.zeros((3, n)) if order >= 1 else None
    h = np.zeros((6, n)) if order >= 2 else None
    G = np.zeros(n) if kinetic else None
    sets = [(wfn.Ca[:, :wfn.nalpha], 1.0), ((wfn.Cb if wfn.open_shell else wfn.Ca)[:, :wfn.nbeta], 1.0)]
    order_b = max(order, 1 if kinetic else 0)
    for s in range(0, n, block):
        p = pts[s:s + block]
        v, gr, he = basis_derivs(wfn, p, order_b)
        for C, occ in sets:
            psi = C.T @ v
            rho[s:s + block] += occ * (psi ** 2).sum(0)
            if order_b >= 1:
                dpsi = np.einsum("mi,amn->ain", C, gr)
                if g is not None:
                    g[:, s:s + block] += 2 * occ * (psi[None] * dpsi).sum(1)
                if kinetic:
                    G[s:s + block] += 0.5 * occ * (dpsi ** 2).sum((0, 1))
            if order >= 2:
                d2 = np.einsum("mi,amn->ain", C, he)
                pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
                for k, (i, j) in enumerate(pairs):
                    h[k, s:s + block] += 2 * occ * (dpsi[i] * dpsi[j] + psi * d2[k]).sum(0)
    return rho, g, h, G
