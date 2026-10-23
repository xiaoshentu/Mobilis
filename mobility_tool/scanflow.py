"""批量扫描工作流：由一个冻结的单体构造二聚体，在参数网格上批量计算 ADF 片段轨道转移积分。

二聚体参数 (与 make_dimers.py 一致)：
    r_B = R_z(γ) R_y(θ) R_x(φ) r_A + (Δx, Δy, z)ᵀ
    分子坐标系：原点 = 重原子几何中心，x = 长轴，y = 短轴，z = 平面法线 (协方差主轴，三阶矩定符号，右手系)
    γ = gamma (绕 z)，θ = pitch (绕 y)，φ = roll (绕 x，长轴)

项目结构：
    <项目>/project.json                    单体坐标、单体对称性、计算层级、扫描列表
    <项目>/monomer/mono_frame.xyz          分子坐标系下的冻结单体
    <项目>/monomer/<层级>/mono.run / run_mono_adf.sh  单体片段计算 → mono.t21 (每个层级一个，所有二聚体共用)
    <项目>/scans/<扫描>/manifest.csv       全部构型 (含接触筛选、对称约简的对应关系)
    <项目>/scans/<扫描>/batch_*.run / .sh  分块批处理 (ADFjobs 用 .run，本软件“运行任务”用 .sh)
    <项目>/scans/<扫描>/outputs/<id>.out   ADF 输出；<id>.done 为完成标记 (断点续算)

相位一致性：同一层级的所有二聚体引用同一个 mono.t21 (绝对路径)，ADF 用刚性转动 + 平移把它映射到两个分子上，
V 的符号在构型之间可比。对称约简与对称检验只使用单体所在点群中使二聚体整体变换的操作 S：
    R′ = S R S，t′ = S t   (A 在 S 下不变)，由 S 联系的两个构型 V (含符号) 严格相同。
B 绕自身对称轴转动 (R′ = R g) 得到同一几何、|V| 相同，V 的符号给出前线轨道在 g 下的特征标。
"""
import csv
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np

from .surrogate import CARRIERS, build_B, clash_ratio, molecular_frame, rotation

PARAM_KEYS = ("z", "dx", "dy", "gamma", "pitch", "roll")
XC_LINES = {"PW91": "GGA PW91", "PBE": "GGA PBE", "BP86": "GGA BP86", "BLYP": "GGA BLYP",
            "B3LYP": "Hybrid B3LYP", "PBE0": "Hybrid PBE0"}
D3_OK = {"PBE", "BP86", "BLYP", "B3LYP", "PBE0"}           # ADF 中有 Grimme D3(BJ) 参数的泛函
BASES = ["DZ", "DZP", "TZP", "TZ2P"]
OPS = {"E": (1, 1, 1), "C2x": (1, -1, -1), "C2y": (-1, 1, -1), "C2z": (-1, -1, 1),
       "i": (-1, -1, -1), "σxy": (1, 1, -1), "σxz": (1, -1, 1), "σyz": (-1, 1, 1)}
PROPER = ("C2x", "C2y", "C2z")
MANIFEST_COLS = ["id", "z", "dx", "dy", "gamma", "roll", "pitch", "dmin_A", "min_ratio_vdW", "status", "set", "rep",
                 "op", "group", "variant"]
KCAL_R = 0.0019872043                                      # kcal mol⁻¹ K⁻¹


# ----------------------------------------------------------------------------------------------
# 构型参数与对称操作
# ----------------------------------------------------------------------------------------------
def _norm_angle(a):
    a = (float(a) + 180.0) % 360.0 - 180.0
    return 180.0 if a <= -180.0 + 1e-9 else (0.0 if abs(a) < 1e-9 else a)


def make_cfg(z, dx, dy, gamma=0.0, pitch=0.0, roll=0.0, **extra):
    c = {"z": round(float(z), 6), "dx": round(float(dx), 6), "dy": round(float(dy), 6),
         "gamma": round(_norm_angle(gamma), 6), "pitch": round(_norm_angle(pitch), 6),
         "roll": round(_norm_angle(roll), 6)}
    for k in ("z", "dx", "dy"):
        if c[k] == 0:
            c[k] = 0.0
    c.update(extra)
    return c


def cfg_R(c):
    return rotation(c["gamma"], c["roll"], c["pitch"])          # R_z(γ) R_y(θ) R_x(φ)


def angles_from_R(R):
    """R = R_z(γ) R_y(θ) R_x(φ) → (γ, θ, φ) / 度。"""
    sp = -float(np.clip(R[2, 0], -1.0, 1.0))
    pitch = math.degrees(math.asin(sp))
    if math.cos(math.radians(pitch)) > 1e-8:
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        gamma = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:                                                       # 万向锁：令 φ = 0
        roll = 0.0
        gamma = math.degrees(math.atan2(-R[0, 1], R[1, 1]))
    return gamma, pitch, roll


def apply_op(c, op):
    """二聚体整体作对称操作 S (A 不变)：R′ = S R S，t′ = S t。"""
    S = np.diag(OPS[op]).astype(float)
    R = S @ cfg_R(c) @ S
    t = S @ np.array([c["dx"], c["dy"], c["z"]])
    g, p, r = angles_from_R(R)
    return make_cfg(t[2], t[0], t[1], g, p, r)


def self_op(c, op):
    """B 绕自身的二重轴转动 (同一几何)：R′ = R g。"""
    G = np.diag(OPS[op]).astype(float)
    g, p, r = angles_from_R(cfg_R(c) @ G)
    return make_cfg(c["z"], c["dx"], c["dy"], g, p, r)


def cfg_key(c, nd=4):
    return tuple(round(float(c[k]), nd) + 0.0 for k in PARAM_KEYS)


def molecule_ops(symbols, Y, tol=0.05):
    """分子坐标系中使单体不变的操作 (D2h 的子群)。"""
    from scipy.spatial import cKDTree
    Y = np.asarray(Y, float)
    out = []
    for name, d in OPS.items():
        Yp = Y * np.array(d)
        ok = True
        for el in set(symbols):
            idx = [i for i, s in enumerate(symbols) if s == el]
            dist, _ = cKDTree(Y[idx]).query(Yp[idx])
            if dist.max() > tol:
                ok = False
                break
        if ok:
            out.append(name)
    return out


def op_deviation(symbols, Y, op):
    from scipy.spatial import cKDTree
    Y = np.asarray(Y, float)
    Yp = Y * np.array(OPS[op])
    dev = 0.0
    for el in set(symbols):
        idx = [i for i, s in enumerate(symbols) if s == el]
        dev = max(dev, cKDTree(Y[idx]).query(Yp[idx])[0].max())
    return float(dev)


def canonical(c, ops):
    """对称等价构型中的代表 (优先 z ≥ 0、Δx ≥ 0、Δy ≥ 0、φ ≥ 0 …)。返回 (代表, 所用操作)。"""
    def rank(q):
        return tuple(q[k] < -1e-6 for k in ("z", "dx", "dy", "roll", "gamma", "pitch")) + cfg_key(q)
    orbit = [(apply_op(c, op), op) for op in ops]
    return min(orbit, key=lambda t: rank(t[0]))


def parse_values(s):
    """'0:6:0.25' → 0, 0.25, …, 6；'1,2,3' → [1, 2, 3]；'3.5' → [3.5]。"""
    if isinstance(s, (int, float)):
        return [float(s)]
    if isinstance(s, (list, tuple)):
        return [float(x) for x in s]
    s = str(s).strip().replace("，", ",").replace("：", ":")
    if not s:
        return [0.0]
    if ":" in s:
        p = [float(x) for x in s.split(":")]
        if len(p) != 3 or p[2] <= 0:
            raise ValueError(f"范围写法应为 起点:终点:步长，例如 0:6:0.25 (收到 {s})")
        n = int(round((p[1] - p[0]) / p[2]))
        if n < 0 or n > 100000:
            raise ValueError(f"范围 {s} 不合理")
        return [round(p[0] + i * p[2], 6) for i in range(n + 1)]
    return [float(x) for x in s.split(",") if x.strip()]


def grid_configs(spec):
    """spec: {z, dx, dy, gamma, pitch, roll} → 各取值的笛卡尔积。"""
    import itertools
    vals = [parse_values(spec.get(k, "0")) for k in PARAM_KEYS]
    return [make_cfg(*v) for v in itertools.product(*vals)]


def lhs_blocks(blocks, seed=2026):
    """拉丁超立方抽样。blocks: [{"tag": "interp", "n": 20, "z": (3.5, 3.8), "dx": (-6, 6), "dy": (-3, 3), ...}]。
    每个参数可给区间或固定值；同一种子可完全重现。"""
    from scipy.stats import qmc
    out = []
    for k, b in enumerate(blocks):
        keys = [p for p in PARAM_KEYS if isinstance(b.get(p), (list, tuple)) and len(b[p]) == 2]
        n = int(b["n"])
        u = qmc.LatinHypercube(d=max(len(keys), 1), seed=seed + 97 * k).random(n) if n else np.zeros((0, 1))
        for row in u:
            v = {p: float(b.get(p, 0.0)) if not isinstance(b.get(p), (list, tuple)) else 0.0 for p in PARAM_KEYS}
            for j, p in enumerate(keys):
                lo, hi = b[p]
                v[p] = round(lo + row[j] * (hi - lo), 4)
            out.append(make_cfg(v["z"], v["dx"], v["dy"], v["gamma"], v["pitch"], v["roll"], set=b.get("tag", "valid")))
    return out


# ----------------------------------------------------------------------------------------------
# 单体
# ----------------------------------------------------------------------------------------------
def monomer_report(symbols, Y):
    Y = np.asarray(Y, float)
    heavy = np.array([s != "H" for s in symbols])
    ops = molecule_ops(symbols, Y)
    devs = {op: op_deviation(symbols, Y, op) for op in OPS if op != "E"}
    from .viewer import bonds_for
    cc = [float(np.linalg.norm(Y[i] - Y[j])) for i, j in bonds_for(symbols, Y) if symbols[i] == "C" and symbols[j] == "C"]
    el = {}
    for s in symbols:
        el[s] = el.get(s, 0) + 1
    formula = "".join(f"{e}{n if n > 1 else ''}" for e, n in sorted(el.items(), key=lambda kv: (kv[0] != "C", kv[0] != "H", kv[0])))
    group = {8: "D2h", 4: "C2v/C2h/D2", 2: "Cs/C2/Ci", 1: "C1"}.get(len(ops), "?")
    return {"formula": formula, "n_atoms": len(symbols), "ops": ops, "point_group": group,
            "op_deviation_A": devs, "planarity_A": float(np.abs(Y[heavy, 2]).max()),
            "length_A": float(np.ptp(Y[heavy, 0])), "width_A": float(np.ptp(Y[heavy, 1])),
            "CC_range_A": [min(cc), max(cc)] if cc else None}


# ----------------------------------------------------------------------------------------------
# ADF 输入与批处理 (ADF2016 / ADF2019 以前的 classic 格式，与 make_dimers.py 相同)
# ----------------------------------------------------------------------------------------------
def keyword_block(xc, basis, d3=False):
    disp = "\nDispersion Grimme3 BJDAMP" if d3 else ""
    return (f"SYMMETRY NOSYM\n\nBASIS\ntype {basis}\ncore None\ncreateoutput None\nEND\n\n"
            f"XC\n{XC_LINES[xc]}{disp}\nEND\n\n")


def _atoms_lines(symbols, X, frag=None):
    out, n = [], 1
    for k, M in enumerate(X, 1):
        for e, r in zip(symbols, M):
            tag = f"  f={frag}/{k}" if frag else ""
            out.append(f"{n} {e:2s} {r[0]:18.10f} {r[1]:18.10f} {r[2]:18.10f}{tag}")
            n += 1
    return out


def _ascii_path_warning(p):
    s = str(p)
    if any(ord(ch) > 127 for ch in s) or " " in s:
        return f"路径 {s} 含中文或空格：ADF 自带的 sh 可能无法处理，建议把项目放到纯英文路径 (如 C:/adfscan) 下"
    return None


# ----------------------------------------------------------------------------------------------
# 结果解析
# ----------------------------------------------------------------------------------------------
_EDA = {"pauli": r"Total Pauli Repulsion:", "elstat": r"Electrostatic Interaction:",
        "steric": r"Total Steric Interaction:", "orbital": r"Total Orbital Interactions:",
        "dispersion": r"Dispersion Energy:", "total": r"Total Bonding Energy:"}
_NUM = r"\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"


def parse_eda(text):
    """ADF 片段计算的能量分解 (kcal/mol)：Pauli、静电、立体、轨道相互作用、色散、总成键能。"""
    out = {}
    for k, lab in _EDA.items():
        m = re.findall(lab + _NUM, text)
        if m:
            out[k] = float(m[-1][2])
    return out


def parse_adf_out(path):
    """单个 ADF 输出：V、J、S、e1、e2 (两种载流子)、由分量重算的 V、能量分解与是否正常结束。"""
    from .adf import _organize, parse_adf_output
    text = Path(path).read_text(errors="ignore")
    res = {"normal": "NORMAL TERMINATION" in text, "eda": parse_eda(text), "carriers": {}}
    try:
        org = _organize(parse_adf_output(text))
    except Exception:  # noqa: BLE001
        org = {}
    for c in CARRIERS:
        d = org.get(c)
        if not d or "V_meV" not in d:
            continue
        r = {"V_adf": float(d["V_meV"])}
        for k_out, k_in in (("J", "J_meV"), ("S", "S"), ("e1", "e1_eV"), ("e2", "e2_eV")):
            if k_in in d:
                r[k_out] = float(d[k_in])
        if all(k in r for k in ("J", "S", "e1", "e2")):
            r["V_recalc"] = (r["J"] - r["S"] * (r["e1"] + r["e2"]) * 1000 / 2) / (1 - r["S"] ** 2)
            # 数值取 ADF 打印值 (精度更高)，符号取由分量重算的值
            s = np.sign(r["V_recalc"]) if abs(r["V_recalc"]) > 1e-9 else np.sign(r["V_adf"])
            r["V"] = float(abs(r["V_adf"]) * (s if s != 0 else 1.0))
            r["sign_mismatch"] = bool(np.sign(r["V_adf"]) != np.sign(r["V_recalc"]) and abs(r["V_recalc"]) > 0.05)
        else:
            r["V"] = r["V_adf"]
        res["carriers"][c] = r
    res["ok"] = bool(res["carriers"]) and res["normal"]
    return res


# ----------------------------------------------------------------------------------------------
# 项目
# ----------------------------------------------------------------------------------------------
class ScanProject:
    FILE = "project.json"

    def __init__(self, root):
        self.root = Path(root).resolve()
        p = self.root / self.FILE
        if not p.exists():
            raise FileNotFoundError(f"{self.root} 不是扫描项目 (没有 {self.FILE})")
        self.meta = json.loads(p.read_text(encoding="utf-8"))
        self._ops = None

    # ---- 基本 ----
    @classmethod
    def create(cls, root, monomer_file, name=None, frame="all"):
        """frame：分子坐标系的定义。all = 全部重原子；core = 最大的稠环体系 (共轭主体，不含苯基等取代基)。"""
        from .gaussian_io import read_geometry
        from .surrogate import frame_atoms
        root = Path(root).resolve()
        if (root / cls.FILE).exists():
            raise FileExistsError(f"{root} 已经是一个扫描项目，请直接打开")
        atoms = read_geometry(monomer_file)
        sym = [a[0] for a in atoms]
        X = np.array([a[1:] for a in atoms], float)
        Y, _, ax, cen = molecular_frame(sym, X, frame_atoms(sym, X, frame))
        if np.linalg.norm(cen) < 1e-3 and np.allclose(np.abs(ax), np.eye(3), atol=1e-4):
            Y = X.copy()      # 输入已在分子坐标系中 (如扫描目录的 mono_frame.xyz)：原样保留，与已算的 mono.t21 完全一致
        (root / "monomer").mkdir(parents=True, exist_ok=True)
        (root / "scans").mkdir(exist_ok=True)
        with open(root / "monomer" / "mono_frame.xyz", "w") as fh:
            fh.write(f"{len(sym)}\nfrozen monomer in molecular frame (from {Path(monomer_file).name})\n")
            for e, r in zip(sym, Y):
                fh.write(f"{e:2s} {r[0]:16.10f} {r[1]:16.10f} {r[2]:16.10f}\n")
        try:
            shutil.copy(monomer_file, root / "monomer" / ("source_" + Path(monomer_file).name))
        except (OSError, shutil.SameFileError):
            pass
        from . import __version__
        meta = {"format": "mobilis-scan-project-1", "name": name or root.name, "created": time.strftime("%Y-%m-%d %H:%M"),
                "software": f"Mobilis {__version__}",
                "monomer": {"source": str(Path(monomer_file).resolve()), "symbols": sym, "Y": Y.round(10).tolist(),
                            "report": monomer_report(sym, Y)}, "frame": frame,
                "levels": {}, "scans": {}}
        (root / cls.FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return cls(root)

    def save(self):
        (self.root / self.FILE).write_text(json.dumps(self.meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @property
    def symbols(self):
        return list(self.meta["monomer"]["symbols"])

    @property
    def frame(self):
        return self.meta.get("frame", "all")

    @property
    def Y(self):
        return np.array(self.meta["monomer"]["Y"], float)

    @property
    def ops(self):
        if self._ops is None:
            self._ops = molecule_ops(self.symbols, self.Y)
        return self._ops

    def B_coords(self, c):
        return build_B(self.Y, c["z"], c["dx"], c["dy"], c["gamma"], c["roll"], c["pitch"])

    def dimer_atoms(self, c):
        sym = self.symbols
        return [(s, *p) for s, p in zip(sym, self.Y)], [(s, *p) for s, p in zip(sym, self.B_coords(c))]

    # ---- 计算层级 (每个层级一个单体片段) ----
    def add_level(self, xc="PW91", basis="TZP", d3=False):
        if xc not in XC_LINES:
            raise ValueError(f"不支持的泛函 {xc}；可选 {', '.join(XC_LINES)}")
        if basis not in BASES:
            raise ValueError(f"不支持的基组 {basis}；可选 {', '.join(BASES)}")
        if d3 and xc not in D3_OK:
            raise ValueError(f"ADF 中没有 {xc} 的 Grimme D3(BJ) 参数；请取消色散校正，或改用 {', '.join(sorted(D3_OK))}")
        name = f"{xc}_{basis}" + ("_D3BJ" if d3 else "")
        d = self.root / "monomer" / name
        d.mkdir(parents=True, exist_ok=True)
        body = _atoms_lines(self.symbols, [self.Y])
        kw = keyword_block(xc, basis, d3)
        title = f"TITLE mono fragment {name} ({self.meta['monomer']['report']['formula']})\n\nATOMS\n"
        run = '#! /bin/sh\n\n"$ADFBIN/adf" <<eor\n' + title + "\n".join(body) + "\nEND\n\n" + kw + "NOPRINT LOGFILE\n\neor\n"
        (d / "mono.run").write_bytes(run.encode("ascii"))
        dp = d.as_posix()
        sh = (f'#!/bin/sh\ncd "{dp}"\n"$ADFBIN/adf" <<eor > "{dp}/mono.out" 2>&1\n' + title + "\n".join(body) +
              "\nEND\n\n" + kw + "NOPRINT LOGFILE\n\neor\n" + f'if [ -s TAPE21 ]; then mv -f TAPE21 "{dp}/mono.t21"; fi\n')
        (d / "run_mono_adf.sh").write_bytes((sh + "# 结果文件: mono.t21\n").encode("utf-8"))
        self.meta["levels"][name] = {"xc": xc, "basis": basis, "d3": bool(d3), "dir": f"monomer/{name}"}
        self.save()
        return name

    def level_dir(self, level):
        if level not in self.meta["levels"]:
            raise KeyError(f"没有计算层级 {level}；请先添加")
        return self.root / self.meta["levels"][level]["dir"]

    def t21(self, level):
        return self.level_dir(level) / "mono.t21"

    def import_fragment(self, level, t21_path):
        """沿用已有的 mono.t21 (例如已有扫描所用的同一个文件)，保证与已有数据相位一致。"""
        dst = self.t21(level)
        if Path(t21_path).resolve() != dst.resolve():
            shutil.copy(t21_path, dst)
        (dst.parent / "imported_from.txt").write_text(str(Path(t21_path).resolve()) + "\n")
        return dst

    # ---- 扫描 ----
    def scan_dir(self, name):
        s = self.meta["scans"].get(name)
        if s is None:
            raise KeyError(f"没有扫描 {name}")
        p = Path(s["path"])
        return p if p.is_absolute() else self.root / p

    def scans(self):
        return list(self.meta["scans"])

    def _new_scan_dir(self, name):
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
            raise ValueError("扫描名称只能用英文字母、数字、下划线、点和连字符 (ADF 的路径要求)")
        if name in self.meta["scans"]:
            raise FileExistsError(f"扫描 {name} 已存在")
        d = self.root / "scans" / name
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f"{d} 已存在且不为空")
        (d / "outputs").mkdir(parents=True, exist_ok=True)
        (d / "xyz").mkdir(exist_ok=True)
        return d

    def add_scan(self, name, configs, level, kind="grid", clash_min=0.75, chunk=100, reduce=True, info=None,
                 prefix=None):
        """configs：make_cfg 字典列表 (可带 set / group / variant)。按单体对称性约简，筛除原子接触过近的构型，
        写 manifest.csv、xyz/ 与分块批处理。返回统计。"""
        self.level_dir(level)
        d = self._new_scan_dir(name)
        prefix = prefix or re.sub(r"[^A-Za-z0-9]", "", name)[:12] or "cfg"
        ops = self.ops if reduce else ["E"]
        sym = self.symbols
        req = []                                                # 第一遍：每个请求的代表构型
        for c in configs:
            own = make_cfg(**{k: c[k] for k in PARAM_KEYS})
            rep, op = canonical(own, ops) if reduce else (own, "E")
            req.append((own, rep, op, cfg_key(rep), cfg_key(own) == cfg_key(rep),
                        {k: c.get(k, "") for k in ("set", "group", "variant")}))
        owner = {}                                              # 代表构型 → 本身就是它的第一个请求 (保留其标签)
        for n, (_, _, _, key, is_rep, _) in enumerate(req):
            if is_rep and key not in owner:
                owner[key] = n
        rows, reps, n_id = [], {}, 0
        for n, (own, rep, op, key, is_rep, base) in enumerate(req):
            if key not in reps:                                 # 第二遍：新的代表构型，实际计算
                n_id += 1
                rid = f"{prefix}_{n_id:04d}"
                dmin, ratio = clash_ratio(sym, self.Y, self.B_coords(rep))
                status = "ok" if ratio >= clash_min else "clash"
                reps[key] = (rid, status)
                lab = req[owner[key]][5] if key in owner else {"set": base["set"], "group": "", "variant": ""}
                rows.append(dict(rep, id=rid, dmin_A=round(dmin, 4), min_ratio_vdW=round(ratio, 4), status=status,
                                 rep="", op="", **lab))
                if status == "ok":
                    self._write_xyz(d / "xyz" / f"{rid}.xyz", rep, rid)
            if owner.get(key) == n:
                continue                                        # 该请求就是计算行本身
            rid, status = reps[key]                             # 等价构型或重复请求：映射到代表，不重复计算
            rows.append(dict(own, id=f"{rid}~{len(rows)}", dmin_A="", min_ratio_vdW="",
                             status="sym" if status == "ok" else "clash", rep=rid, op=op, **base))
        with open(d / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in MANIFEST_COLS})
        comp = [r for r in rows if r["status"] == "ok" and not r["rep"]]
        info = dict(info or {})
        lev = self.meta["levels"][level]
        scan = {"name": name, "kind": kind, "level": level, "xc": lev["xc"], "basis": lev["basis"], "d3": lev["d3"],
                "created": time.strftime("%Y-%m-%d %H:%M"), "clash_min": clash_min, "chunk": chunk, "reduce": reduce,
                "n_requested": len(configs), "n_computed": len(comp),
                "n_clash": sum(1 for r in rows if r["status"] == "clash" and not r["rep"]),
                "n_symmetry": sum(1 for r in rows if r["rep"]), "ops": ops, "info": info}
        (d / "scan.json").write_text(json.dumps(scan, ensure_ascii=False, indent=1), encoding="utf-8")
        self.meta["scans"][name] = {"path": f"scans/{name}", "level": level, "kind": kind}
        self.save()
        nb = self.write_batches(name)
        scan["batches"] = nb
        return scan

    def _write_xyz(self, path, c, title):
        A, B = self.dimer_atoms(c)
        with open(path, "w") as fh:
            fh.write(f"{len(A) + len(B)}\n{title} z={c['z']} dx={c['dx']} dy={c['dy']} gamma={c['gamma']} "
                     f"pitch={c['pitch']} roll={c['roll']}\n")
            for s, x, y, z in A + B:
                fh.write(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")

    def scan_info(self, name):
        return json.loads((self.scan_dir(name) / "scan.json").read_text(encoding="utf-8"))

    def manifest(self, name):
        with open(self.scan_dir(name) / "manifest.csv", encoding="utf-8-sig", errors="ignore") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            for k in PARAM_KEYS:
                r[k] = float(r.get(k) or 0.0)
            r.setdefault("rep", "")
            r.setdefault("set", "")
        return rows

    def write_batches(self, name):
        """(重新) 生成分块批处理：所有构型引用本层级的同一个 mono.t21 (绝对路径)；已完成 (.done) 的构型自动跳过。"""
        d = self.scan_dir(name)
        info = self.scan_info(name)
        level = info["level"]
        t21 = self.t21(level).resolve().as_posix()
        lev = self.meta["levels"][level]
        kw = keyword_block(lev["xc"], lev["basis"], lev["d3"]).rstrip("\n")
        for old in list(d.glob("batch_*.run")) + list(d.glob("batch_*.sh")):
            old.unlink()
        comp = [r for r in self.manifest(name) if r["status"] == "ok" and not r["rep"]]
        chunk = max(1, int(info.get("chunk") or len(comp) or 1))
        root = d.resolve().as_posix()
        names = []
        for k in range(0, len(comp), chunk):
            part = comp[k:k + chunk]
            nb = f"batch_{name}_{k // chunk + 1}"
            lines = ["#! /bin/sh", "",
                     f"# Mobilis scan {name}, part {k // chunk + 1}/{(len(comp) - 1) // chunk + 1}, level {level}",
                     f"# all dimers use the same fragment file (phase-consistent signs of V): {t21}",
                     f'if [ ! -s "{t21}" ]; then',
                     f'  echo "ERROR: {t21} not found, run monomer/{level}/mono.run first"', "  exit 1", "fi", ""]
            for r in part:
                out = f"{root}/outputs/{r['id']}.out"
                done = f"{root}/outputs/{r['id']}.done"
                X = [self.Y, self.B_coords(r)]
                lines += [f'if [ ! -f "{done}" ]; then', f'echo "running {r["id"]}"',
                          f'"$ADFBIN/adf" <<eor > "{out}" 2>&1',
                          f"TITLE {r['id']} z={r['z']} dx={r['dx']} dy={r['dy']} gamma={r['gamma']} roll={r['roll']} "
                          f"pitch={r['pitch']}", "", "ATOMS"] + _atoms_lines(self.symbols, X, "mono") + \
                         ["END", "", kw, "", "Fragments", f"    mono {t21}", "end", "NOPRINT LOGFILE",
                          "TRANSFERINTEGRALS", "", "eor", f'if [ $? -eq 0 ]; then echo ok > "{done}"; fi', "fi", ""]
            lines.append('echo "batch finished"')
            text = "\n".join(lines) + "\n"
            (d / f"{nb}.run").write_bytes(text.encode("ascii", errors="replace"))
            sh = text.replace("#! /bin/sh", f'#!/bin/sh\ncd "{root}"', 1) + \
                "".join(f"# 结果文件: outputs/{r['id']}.done\n" for r in part)       # “运行任务”窗口据此显示进度
            (d / f"{nb}.sh").write_bytes(sh.encode("utf-8"))
            names.append(nb)
        return names

    def import_scan(self, path, name=None, level=None):
        """登记已有的扫描目录 (make_dimers.py 生成：manifest.csv + outputs/)，只读，用于分析、报告与归档。"""
        path = Path(path).resolve()
        if not (path / "manifest.csv").exists():
            raise FileNotFoundError(f"{path} 中没有 manifest.csv")
        name = name or path.name
        if name in self.meta["scans"]:
            raise FileExistsError(f"扫描 {name} 已存在")
        self.meta["scans"][name] = {"path": str(path), "level": level or "imported", "kind": "imported", "external": True}
        self.save()
        if not (path / "scan.json").exists():
            (path / "scan.json").write_text(json.dumps({"name": name, "kind": "imported", "level": level or "imported",
                                                        "imported": True}, ensure_ascii=False, indent=1), encoding="utf-8")
        return name

    # ---- 运行前检查 / 修复 ----
    def output_path(self, name, rid):
        d = self.scan_dir(name)
        for p in (d / "outputs" / f"{rid}.out", d / f"{rid}.out"):
            if p.exists():
                return p
        return d / "outputs" / f"{rid}.out"

    def precheck(self, name, repair=True):
        """运行前检查：片段文件、路径、批处理、已完成 / 待算 / 失败的构型。repair=True 时修正完成标记：
        已完整算完但没有 .done 的补上标记，有 .done 但结果不完整的删除标记以便重算。"""
        d = self.scan_dir(name)
        info = self.scan_info(name)
        errors, warns = [], []
        level = info.get("level")
        if level in self.meta["levels"]:
            t21 = self.t21(level)
            if not t21.exists() or t21.stat().st_size == 0:
                errors.append(f"缺少单体片段文件 {t21}：请先运行 monomer/{level}/mono.run (ADFjobs)，或在“运行任务”中运行 run_mono_adf.sh")
            else:
                mo = t21.parent / "mono.out"
                if mo.exists() and "NORMAL TERMINATION" not in mo.read_text(errors="ignore"):
                    errors.append(f"{mo} 没有正常结束，片段文件不可靠")
                run = t21.parent / "mono.run"
                if run.exists() and t21.stat().st_mtime + 1 < run.stat().st_mtime and \
                        not (t21.parent / "imported_from.txt").exists():
                    warns.append("mono.t21 比 mono.run 旧：可能来自修改单体之前的计算，请确认")
        elif not info.get("imported"):
            errors.append(f"扫描所属的计算层级 {level} 不在项目中")
        for p in (self.root, d):
            w = _ascii_path_warning(p)
            if w:
                warns.append(w)
        batches = sorted(d.glob("batch_*.run"))
        if not batches and not info.get("imported"):
            errors.append("没有批处理文件：请重新生成")
        rows = self.manifest(name)
        comp = [r for r in rows if r["status"] == "ok" and not r["rep"]]
        done, pending, failed, repaired, partial = [], [], [], 0, 0
        for r in comp:
            out = self.output_path(name, r["id"])
            mark = out.with_suffix(".done")
            if out.exists():
                res = parse_adf_out(out)
                if res["ok"]:
                    done.append(r["id"])
                    if not mark.exists() and repair:
                        mark.write_text("ok\n")
                        repaired += 1
                    continue
                if mark.exists():
                    failed.append(r["id"])
                    if repair:
                        mark.unlink()
                else:
                    partial += 1
                pending.append(r["id"])
            else:
                if mark.exists() and repair:
                    mark.unlink()
                pending.append(r["id"])
        sizes = [self.output_path(name, i).stat().st_size for i in done[:50]]
        est = (np.mean(sizes) * len(pending) / 1e6) if sizes else None
        if est is not None:
            try:
                free = shutil.disk_usage(d).free / 1e6
                if free < 2 * est:
                    warns.append(f"磁盘剩余 {free:.0f} MB，预计还需约 {est:.0f} MB")
            except OSError:
                pass
        if failed:
            warns.append(f"{len(failed)} 个构型的输出不完整或出错，已删除完成标记，重新运行批处理时会重算：{', '.join(failed[:8])}"
                         + (" …" if len(failed) > 8 else ""))
        if partial:
            warns.append(f"{partial} 个构型的输出不完整 (可能被中断)，重新运行时会重算")
        if repaired:
            warns.append(f"已为 {repaired} 个已算完但缺少标记的构型补上完成标记，续算时会跳过")
        return {"errors": errors, "warnings": warns, "n_total": len(comp), "n_done": len(done), "n_pending": len(pending),
                "n_failed": len(failed), "pending": pending, "batches": [b.name for b in batches],
                "est_MB": est, "ok": not errors}

    # ---- 结果 ----
    def collect(self, name, write=True):
        """读取全部输出。按对称约简的对应关系展开到所有请求的构型；写 results.csv。"""
        d = self.scan_dir(name)
        rows = self.manifest(name)
        parsed, recs = {}, []
        for r in rows:
            if r["status"] == "ok" and not r["rep"]:
                out = self.output_path(name, r["id"])
                parsed[r["id"]] = parse_adf_out(out) if out.exists() else None
        for r in rows:
            src = r["rep"] or r["id"]
            res = parsed.get(src)
            rec = {"scan": name, "id": r["id"], "rep": r["rep"], "set": r.get("set", ""), "group": r.get("group", ""),
                   "variant": r.get("variant", ""), "op": r.get("op", ""), "status": r["status"],
                   "dmin_A": r.get("dmin_A", ""), "source": name}
            rec.update({k: r[k] for k in PARAM_KEYS})
            for c in CARRIERS:
                rec[c] = None
            if r["status"] == "clash":
                rec["state"] = "excluded"
            elif res is None:
                rec["state"] = "pending"
            elif not res["ok"]:
                rec["state"] = "failed"
            else:
                rec["state"] = "done"
                for c, v in res["carriers"].items():
                    rec[c] = v["V"]
                    for k in ("J", "S", "e1", "e2", "V_adf", "V_recalc", "sign_mismatch"):
                        if k in v:
                            rec[f"{k}_{c}"] = v[k]
                for k, v in res["eda"].items():
                    rec[f"E_{k}"] = v
            recs.append(rec)
        if write:
            write_records(recs, d / "results.csv")
        return recs

    def collect_all(self):
        out = []
        for n in self.scans():
            try:
                out += self.collect(n, write=False)
            except Exception:  # noqa: BLE001
                continue
        return out

    def status_table(self):
        rows = []
        for n in self.scans():
            try:
                info = self.scan_info(n)
                recs = self.collect(n, write=False)
            except Exception as exc:  # noqa: BLE001
                rows.append({"name": n, "error": str(exc)})
                continue
            comp = [r for r in recs if not r["rep"] and r["status"] == "ok"]
            rows.append({"name": n, "kind": info.get("kind", ""), "level": info.get("level", ""),
                         "n": len(comp), "done": sum(r["state"] == "done" for r in comp),
                         "failed": sum(r["state"] == "failed" for r in comp),
                         "excluded": sum(1 for r in recs if r["state"] == "excluded" and not r["rep"])})
        return rows

    # ---- 派生扫描 ----
    def clone(self, name, level, new_name=None):
        """同一批构型换一个计算层级 (泛函 / 基组对照)。"""
        rows = [r for r in self.manifest(name) if not r["rep"]]
        cfgs = [make_cfg(**{k: r[k] for k in PARAM_KEYS}, set=r.get("set", ""), group=r.get("group", ""),
                         variant=r.get("variant", "")) for r in rows if r["status"] == "ok"]
        info = self.scan_info(name)
        new_name = new_name or re.sub(r"[^A-Za-z0-9_.\-]", "", f"{name}_{level}")
        return self.add_scan(new_name, cfgs, level, kind=info.get("kind", "grid"), clash_min=info.get("clash_min", 0.75),
                             chunk=info.get("chunk", 100), reduce=False, info={"clone_of": name},
                             prefix=re.sub(r"[^A-Za-z0-9]", "", name)[:12])

    def add_validation(self, name, level, blocks, seed=2026, clash_min=0.75, chunk=100, max_tries=20):
        """随机验证集 (拉丁超立方，固定种子可重现)；接触过近的构型用后续样本替换，保证各组数目。"""
        cfgs = []
        sym, Y = self.symbols, self.Y
        for k, b in enumerate(blocks):
            got, tries = [], 0
            while len(got) < int(b["n"]) and tries < max_tries:
                cand = lhs_blocks([dict(b, n=int(b["n"]) * (2 if tries else 1))], seed=seed + 1009 * k + 7 * tries)
                for c in cand:
                    if len(got) >= int(b["n"]):
                        break
                    if clash_ratio(sym, Y, self.B_coords(c))[1] >= clash_min:
                        got.append(c)
                tries += 1
            cfgs += got
        return self.add_scan(name, cfgs, level, kind="validation", clash_min=clash_min, chunk=chunk, reduce=False,
                             info={"seed": seed, "blocks": [{k: (list(v) if isinstance(v, tuple) else v)
                                                             for k, v in b.items()} for b in blocks]})

    def add_check(self, name, level, bases, include_self=True, chunk=100):
        """对称等价检验：每个基准构型生成单体点群中全部整体操作的变体 (V 含符号应相同)，以及 B 绕自身二重轴的变体
        (|V| 应相同，符号给出前线轨道的特征标)。"""
        cfgs = []
        for g, b in enumerate(bases, 1):
            b = make_cfg(**{k: b.get(k, 0.0) for k in PARAM_KEYS})
            seen = set()
            for op in self.ops:
                c = apply_op(b, op)
                if cfg_key(c) in seen:
                    continue
                seen.add(cfg_key(c))
                cfgs.append(dict(c, set="check", group=str(g), variant=op))
            if include_self:
                for op in PROPER:
                    if op in self.ops:
                        cfgs.append(dict(self_op(b, op), set="check", group=str(g), variant="self:" + op))
        return self.add_scan(name, cfgs, level, kind="check", reduce=False, chunk=chunk,
                             info={"bases": [make_cfg(**{k: b.get(k, 0.0) for k in PARAM_KEYS}) for b in bases]})

    def crystal_configs(self, dimer_result):
        """把晶体中的近邻二聚体换算为冻结单体的参数：A、B 各自的分子坐标系，B 在单体点群允许的
        几种坐标轴取法中选与 A 最接近的一种，R = R_z(γ)R_y(θ)R_x(φ)。返回构型与冻结单体重建的 RMSD。"""
        sa = [s for s, _ in dimer_result["atoms_a"]]
        xa = np.array([x for _, x in dimer_result["atoms_a"]], float)
        if sorted(sa) != sorted(self.symbols):
            raise ValueError("晶体中的分子与项目的单体不同 (元素组成不一致)")
        from .surrogate import matched_frame
        axA, cA, _ = matched_frame(self.symbols, self.Y, sa, xa, self.frame)
        out = []
        for g in dimer_result["groups"]:
            d = g["members"][0]
            sb = [s for s, _ in d["atoms_b"]]
            xb = np.array([x for _, x in d["atoms_b"]], float)
            axB, cB, _ = matched_frame(self.symbols, self.Y, sb, xb, self.frame)
            best = None
            for op in self.ops:
                sgn = np.array(OPS[op], float)
                if np.prod(sgn) < 0:
                    continue                                  # 只取保持手性的坐标轴取法
                R = axA.T @ (axB * sgn)
                ang = math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
                if best is None or ang < best[0]:
                    best = (ang, R)
            R = best[1]
            t = axA.T @ (cB - cA)
            gm, pt, rl = angles_from_R(R)
            c = make_cfg(t[2], t[0], t[1], gm, pt, rl, set="crystal", group=str(g["id"]),
                         variant=g.get("label", f"G{g['id']}"))
            Bf = self.B_coords(c)
            Bc = (xb - cA) @ axA                                # 晶体 B 在 A 坐标系中
            from scipy.optimize import linear_sum_assignment
            rmsd2 = 0.0
            for el in set(sb):
                ii = [i for i, s in enumerate(self.symbols) if s == el]
                jj = [j for j, s in enumerate(sb) if s == el]
                D = np.linalg.norm(Bf[ii][:, None] - Bc[jj][None], axis=-1)
                r_, c_ = linear_sum_assignment(D)
                if el != "H":
                    rmsd2 += (D[r_, c_] ** 2).sum()
            nh = sum(1 for s in sb if s != "H")
            c["rmsd_heavy_A"] = float(np.sqrt(rmsd2 / nh))
            c["r_centroid"] = float(d["r_centroid"])
            out.append(c)
        return out

    def add_local(self, name, level, centers, n=80, half=None, seed=2026, clash_min=0.7, chunk=100):
        """CEDS 局域扫描：centers = {组号: 中心构型 (z, Δx, Δy, γ, θ, φ)}，每个中心做拉丁超立方采样 (各参数 ±h)。
        所有构型共用本层级的 mono.t21；结果用于拟合各晶体二聚体附近的局域耦合曲面。"""
        from .ceds import DEFAULT_HALF, local_scan_configs
        per = isinstance(half, dict) and half and all(isinstance(v, dict) for v in half.values())
        hb = {str(g): dict(DEFAULT_HALF, **((half.get(g) or half.get(str(g)) or {}) if per else (half or {})))
              for g in centers}
        cfgs = []
        for k, (gid, c) in enumerate(sorted(centers.items(), key=lambda kv: str(kv[0]))):
            for q in local_scan_configs(c, n=n, half=hb[str(gid)], seed=seed + 31 * k):
                if clash_ratio(self.symbols, self.Y, self.B_coords(q))[1] >= clash_min:
                    cfgs.append(dict(q, set="local", group=str(gid)))
        info = {"centers": {str(g): {k: c[k] for k in PARAM_KEYS} for g, c in centers.items()},
                "half_by_group": hb, "n": n, "seed": seed}
        return self.add_scan(name, cfgs, level, kind="local", reduce=False, chunk=chunk, clash_min=clash_min, info=info)

    def add_md_snapshots(self, name, level, configs, info=None, chunk=100):
        """MD 快照：每个构型是轨迹中某一帧、某个超胞实例的二聚体几何 (group = 晶体二聚体组号)。
        只用于检验耦合曲面 (逐帧直接计算 V)，不参与曲面拟合。不做对称约简，也不按接触距离筛除，
        以免改变 MD 样本的分布。"""
        cfgs = [dict(c, set="md") for c in configs]
        return self.add_scan(name, cfgs, level, kind="md", reduce=False, chunk=chunk, clash_min=0.0, info=info or {})

    def add_crystal(self, name, level, dimer_result, chunk=100):
        cfgs = self.crystal_configs(dimer_result)
        info = {"pairs": [{k: c[k] for k in PARAM_KEYS + ("group", "variant", "rmsd_heavy_A", "r_centroid")} for c in cfgs]}
        return self.add_scan(name, cfgs, level, kind="crystal", reduce=False, chunk=chunk, clash_min=0.0, info=info)

    # ---- Gaussian：抵消校正相互作用能 / 不同泛函的转移积分 ----
    def _select(self, name, ids=None, stride=1):
        recs = [r for r in self.manifest(name) if r["status"] == "ok" and not r["rep"]]
        if ids:
            want = set(ids)
            recs = [r for r in recs if r["id"] in want]
        return recs[::max(1, int(stride))]

    def gaussian_cp(self, name, functional="B3LYP", basis="6-31G(d)", dispersion="GD3BJ", ids=None, stride=1,
                    nproc=8, mem="8GB"):
        """为扫描中的构型生成 Gaussian 抵消校正 (counterpoise=2) 输入，用于相互作用能。"""
        from .interaction import write_counterpoise_gjf
        from .jobs import write_windows_bat
        tag = re.sub(r"[^A-Za-z0-9]", "", f"{functional}{basis}{dispersion or ''}")[:24]
        out = self.scan_dir(name) / f"cp_{tag}"
        out.mkdir(exist_ok=True)
        names = []
        for r in self._select(name, ids, stride):
            A, B = self.dimer_atoms(r)
            nm = f"{r['id']}_cp"
            write_counterpoise_gjf(out / f"{nm}.gjf", A, B, functional=functional, basis=basis, dispersion=dispersion,
                                   nproc=nproc, mem=mem, title=f"{r['id']} counterpoise")
            names.append(nm)
        (out / "run_cp.sh").write_text("#!/bin/bash\n" + "\n".join(f"g16 < {n}.gjf > {n}.log" for n in names) + "\n")
        write_windows_bat(out, names, name="run_cp.bat")
        (out / "level.json").write_text(json.dumps({"functional": functional, "basis": basis, "dispersion": dispersion}))
        return out, len(names)

    def collect_cp(self, name):
        """读取扫描下所有 cp_* 目录的抵消校正结果 (kcal/mol)。返回 {目录: {id: E}}。"""
        from .interaction import read_counterpoise
        out = {}
        for d in sorted(self.scan_dir(name).glob("cp_*")):
            res = {}
            for f in sorted(d.glob("*_cp.log")) + sorted(d.glob("*_cp.out")):
                try:
                    res[f.name[:-len("_cp" + f.suffix)]] = read_counterpoise(f)["total"] / 4.184
                except Exception:  # noqa: BLE001
                    continue
            out[d.name] = res
        return out

    def gaussian_ti(self, name, functional="B3LYP", basis="6-31G(d)", ids=None, stride=1, nproc=8, mem="8GB"):
        """同一批构型的 Gaussian 投影法转移积分输入 (泛函 / 基组对照)。单体 A 只算一次，每个构型算 B 与 AB。"""
        from .gaussian_io import write_gjf
        from .jobs import write_windows_bat
        tag = re.sub(r"[^A-Za-z0-9]", "", f"{functional}{basis}")[:24]
        out = self.scan_dir(name) / f"gti_{tag}"
        out.mkdir(exist_ok=True)
        route = f"# {functional}/{basis} nosymm iop(3/32=2) scf=tight"
        A0 = [(s, *p) for s, p in zip(self.symbols, self.Y)]
        write_gjf(out / "A.gjf", [dict(chk="A.chk", nproc=nproc, mem=mem, route=route, title="monomer A", charge=0,
                                       mult=1, atoms=A0)])
        names = ["A"]
        for r in self._select(name, ids, stride):
            A, B = self.dimer_atoms(r)
            write_gjf(out / f"{r['id']}_B.gjf", [dict(chk=f"{r['id']}_B.chk", nproc=nproc, mem=mem, route=route,
                                                       title=f"{r['id']} B", charge=0, mult=1, atoms=B)])
            write_gjf(out / f"{r['id']}_AB.gjf", [dict(chk=f"{r['id']}_AB.chk", nproc=nproc, mem=mem, route=route,
                                                        title=f"{r['id']} AB", charge=0, mult=1, atoms=A + B)])
            names += [f"{r['id']}_B", f"{r['id']}_AB"]
        (out / "run_ti.sh").write_text("#!/bin/bash\n" + "\n".join(f"g16 < {n}.gjf > {n}.log && formchk {n}.chk {n}.fchk"
                                                                    for n in names) + "\n")
        write_windows_bat(out, names, name="run_ti.bat")
        (out / "level.json").write_text(json.dumps({"functional": functional, "basis": basis}))
        return out, len(names) // 2

    def collect_gaussian_ti(self, name):
        """读取 gti_* 目录 (A.fchk + <id>_B.fchk + <id>_AB.fchk)。返回 {目录: {id: {hole, electron}}} (meV)。"""
        from .transfer_integral import calc_transfer_integrals
        out = {}
        for d in sorted(self.scan_dir(name).glob("gti_*")):
            res = {}
            a = d / "A.fchk"
            if a.exists():
                for f in sorted(d.glob("*_AB.fchk")):
                    rid = f.name[:-len("_AB.fchk")]
                    b = d / f"{rid}_B.fchk"
                    if not b.exists():
                        continue
                    try:
                        r = calc_transfer_integrals(a, b, f)["results"]
                        res[rid] = {"hole": r["HOMO"]["J_eff_meV"], "electron": r["LUMO"]["J_eff_meV"]}
                    except Exception:  # noqa: BLE001
                        continue
            out[d.name] = res
        return out

    def uff_energies(self, name):
        """软件内 UFF 相互作用能 (kcal/mol，仅范德华)，秒级，用于定性比较。"""
        from .interaction import uff_interaction
        out = {}
        for r in self._select(name):
            A, B = self.dimer_atoms(r)
            out[r["id"]] = uff_interaction(A, B)["total"] / 4.184
        return out

    # ---- 归档 ----
    def archive(self, path=None, include_outputs=True, extra_dirs=()):
        """归档：all_structures.xyz (多帧，注释行含扫描名、参数与 V)、all_results.csv、README、项目全部文件。"""
        import zipfile
        path = Path(path or self.root / f"archive_{time.strftime('%Y%m%d_%H%M')}.zip")
        recs = self.collect_all()
        tmp = self.root / "_archive_tmp"
        tmp.mkdir(exist_ok=True)
        write_records([r for r in recs if not r["rep"]], tmp / "all_results.csv")
        with open(tmp / "all_structures.xyz", "w") as fh:
            for r in recs:
                if r["rep"] or r["status"] != "ok":
                    continue
                A, B = self.dimer_atoms(r)
                vh = "" if r["hole"] is None else f" Vh={r['hole']:.3f}"
                ve = "" if r["electron"] is None else f" Ve={r['electron']:.3f}"
                fh.write(f"{len(A) + len(B)}\n{r['scan']} {r['id']} z={r['z']} dx={r['dx']} dy={r['dy']} "
                         f"gamma={r['gamma']} pitch={r['pitch']} roll={r['roll']}{vh}{ve} meV\n")
                for s, x, y, z in A + B:
                    fh.write(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")
        (tmp / "README.txt").write_text(self.readme(recs), encoding="utf-8")
        n = 0
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in tmp.iterdir():
                z.write(f, f.name)
            z.write(self.root / self.FILE, self.FILE)
            dirs = [self.root / "monomer"] + [self.scan_dir(s) for s in self.scans()] + \
                   [self.root / x for x in ("model", "validation", "report")] + [Path(x) for x in extra_dirs]
            for d in dirs:
                if not d.exists():
                    continue
                base = d.name if d.parent != self.root / "scans" else f"scans/{d.name}"
                if d == self.root / "monomer":
                    base = "monomer"
                for f in d.rglob("*"):
                    if f.is_dir() or f.suffix == ".zip":
                        continue
                    if not include_outputs and "outputs" in f.relative_to(d).parts and f.suffix == ".out":
                        continue
                    z.write(f, f"{base}/{f.relative_to(d).as_posix()}")
                    n += 1
        shutil.rmtree(tmp, ignore_errors=True)
        return path, n

    def readme(self, recs=None):
        recs = recs if recs is not None else self.collect_all()
        m = self.meta
        rep = m["monomer"]["report"]
        lines = [f"Mobilis scan project: {m['name']}", f"Created {m['created']} with {m['software']}", "",
                 f"Monomer: {rep['formula']} ({rep['n_atoms']} atoms), point group in its frame: {rep['point_group']} "
                 f"(operations {', '.join(rep['ops'])}), planarity {rep['planarity_A']:.1e} A",
                 f"Source geometry: {m['monomer']['source']}", "",
                 "Dimer parameters: r_B = Rz(gamma) Ry(pitch) Rx(roll) r_A + (dx, dy, z); frame: origin at the heavy-atom",
                 "center, x long axis, y short axis, z plane normal. Angles in degrees, lengths in Angstrom, V in meV.", "",
                 "Levels (one fragment file mono.t21 per level, shared by all dimers of that level):"]
        for k, v in m["levels"].items():
            lines.append(f"  {k}: ADF classic, XC {XC_LINES[v['xc']]}{' + D3(BJ)' if v['d3'] else ''}, basis {v['basis']} "
                         f"(all-electron, NOSYM), fragment {v['dir']}/mono.t21")
        lines += ["", "Scans:"]
        for n in self.scans():
            rs = [r for r in recs if r["scan"] == n and not r["rep"]]
            lines.append(f"  {n}: {len(rs)} configurations, {sum(r['state'] == 'done' for r in rs)} done, "
                         f"{sum(r['state'] == 'excluded' for r in rs)} excluded by contact criterion")
        lines += ["", "Files: all_results.csv (all couplings, J, S, e1, e2 and the ADF energy decomposition in kcal/mol),",
                  "all_structures.xyz (every computed dimer, one frame each), scans/<name>/manifest.csv (all requested",
                  "configurations incl. symmetry mapping), scans/<name>/outputs (raw ADF output), monomer/ (fragment inputs",
                  "and files), model/ validation/ report/ (if present).",
                  "V: magnitude from the printed ADF value, sign from V = (J - S(e1+e2)/2)/(1 - S^2)."]
        return "\n".join(lines) + "\n"


def write_records(recs, path):
    keys = ["scan", "id", "set", "group", "variant", "state", "status", "rep", "op"] + list(PARAM_KEYS) + \
           ["dmin_A", "hole", "electron"]
    extra = sorted({k for r in recs for k in r if k not in keys and k != "source"})
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + extra, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow({k: ("" if r.get(k) is None else r.get(k, "")) for k in keys + extra})


def read_records(path):
    out = []
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                if k in PARAM_KEYS or k in CARRIERS or k.startswith(("J_", "S_", "e1_", "e2_", "V_", "E_")):
                    r[k] = float(v) if v not in ("", None) else None
            out.append(r)
    return out


# ----------------------------------------------------------------------------------------------
# 对称检验分析
# ----------------------------------------------------------------------------------------------
def symmetry_check(recs):
    """按基准构型分组：整体对称操作的变体 V (含符号) 的最大偏差；B 自身二重轴变体 |V| 偏差与符号 (特征标)。"""
    groups = {}
    for r in recs:
        if r.get("set") == "check" and r["state"] == "done":
            groups.setdefault(r["group"], []).append(r)
    rows = []
    chars = {c: {} for c in CARRIERS}
    for g, rs in sorted(groups.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
        base = next((r for r in rs if r["variant"] == "E"), rs[0])
        row = {"group": g, "base": {k: base[k] for k in PARAM_KEYS}, "n_ops": 0}
        for c in CARRIERS:
            v0 = base[c]
            ops = [r[c] for r in rs if not r["variant"].startswith("self:") and r[c] is not None]
            row["n_ops"] = len(ops)
            row[f"V_{c}"] = v0
            row[f"maxdev_{c}"] = float(max(abs(v - v0) for v in ops)) if ops and v0 is not None else None
            for r in rs:
                if r["variant"].startswith("self:") and r[c] is not None and v0 is not None and abs(v0) > 0.5:
                    op = r["variant"][5:]
                    row[f"{op}_{c}"] = r[c]
                    chars[c].setdefault(op, []).append(np.sign(r[c] / v0))
            row[f"absdev_self_{c}"] = max((abs(abs(r[c]) - abs(v0)) for r in rs
                                           if r["variant"].startswith("self:") and r[c] is not None and v0 is not None),
                                          default=None)
        rows.append(row)
    character = {c: {op: (int(np.sign(np.mean(v))) if v and abs(np.mean(v)) > 0.99 else 0) for op, v in chars[c].items()}
                 for c in CARRIERS}
    return rows, character
