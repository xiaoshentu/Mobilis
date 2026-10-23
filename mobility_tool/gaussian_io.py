"""高斯 (Gaussian) 文件读写：几何结构读取、gjf 输入生成、log 与 fchk 解析。"""
import math
import re
from pathlib import Path

HARTREE2EV = 27.211386245988
BOHR2ANG = 0.529177210903

ELEMENTS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr"
).split()
# 原子质量 (amu)
MASSES = {
    "H": 1.00783, "He": 4.0026, "Li": 7.016, "Be": 9.0122, "B": 11.0093, "C": 12.0, "N": 14.0031,
    "O": 15.9949, "F": 18.9984, "Ne": 19.9924, "Na": 22.9898, "Mg": 23.985, "Al": 26.9815,
    "Si": 27.9769, "P": 30.9738, "S": 31.9721, "Cl": 34.9689, "Ar": 39.9624, "K": 38.9637,
    "Ca": 39.9626, "Ti": 47.948, "Fe": 55.9349, "Co": 58.9332, "Ni": 57.9353, "Cu": 62.9296,
    "Zn": 63.9291, "Ga": 68.9256, "Ge": 73.9212, "As": 74.9216, "Se": 79.9165, "Br": 78.9183,
    "Kr": 83.9115, "Ru": 101.9043, "Pd": 105.9035, "Ag": 106.9051, "Sn": 119.9022, "Te": 129.9062,
    "I": 126.9045, "Ir": 192.9629, "Pt": 194.9648, "Au": 196.9666, "Os": 191.9615, "Re": 186.9558,
    "Rh": 102.9055, "Hg": 201.9706, "Pb": 207.9767, "Eu": 152.9212, "Tb": 158.9254, "Cd": 113.9034,
    "W": 183.9509, "Mo": 97.9054, "Mn": 54.938, "Cr": 51.9405, "V": 50.944, "Sc": 44.9559, "Sb": 120.9038,
    "Bi": 208.9804, "Tl": 204.9744, "In": 114.9039, "Cs": 132.9055, "Ba": 137.9052, "Rb": 84.9118, "Sr": 87.9056,
}
# 共价半径 (Å)，用于判断成键
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Ti": 1.60, "Fe": 1.32, "Co": 1.26, "Ni": 1.24,
    "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Ru": 1.46, "Pd": 1.39, "Ag": 1.45, "Sn": 1.39, "Te": 1.38, "I": 1.39, "Ir": 1.41, "Pt": 1.36,
    "Au": 1.36, "Os": 1.44, "Re": 1.51, "Rh": 1.42, "Hg": 1.32, "Pb": 1.46, "Eu": 1.98, "Tb": 1.94,
    "Cd": 1.44, "W": 1.62, "Mo": 1.54, "Mn": 1.39, "Cr": 1.39, "V": 1.53, "Sc": 1.70, "Sb": 1.39,
    "Bi": 1.48, "Tl": 1.45, "In": 1.42, "Cs": 2.44, "Ba": 2.15, "Rb": 2.20, "Sr": 1.95,
}


# ----------------------------------------------------------------------------------------------
# 几何结构
# ----------------------------------------------------------------------------------------------
def _norm_symbol(s):
    s = re.sub(r"[^A-Za-z0-9]", "", s)
    if s.isdigit():
        return ELEMENTS[int(s)]
    s = re.match(r"[A-Za-z]+", s).group(0)
    return s[0].upper() + s[1:].lower()


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    atoms = []
    for line in lines[2:2 + n]:
        p = line.split()
        atoms.append((_norm_symbol(p[0]), float(p[1]), float(p[2]), float(p[3])))
    return atoms


def read_gjf(path):
    """读取 gjf/com 中的笛卡尔坐标（电荷/多重度行之后直到空行）。"""
    lines = Path(path).read_text().splitlines()
    atoms, reading = [], False
    for i, line in enumerate(lines):
        if not reading:
            if re.match(r"^\s*-?\d+\s+\d+(\s+-?\d+\s+\d+)*\s*$", line) and i > 0 and lines[i - 1].strip() == "":
                reading = True
            continue
        if line.strip() == "":
            break
        p = line.split()
        # 兼容 "C  0  x y z" (冻结标记) 格式
        xyz = [float(v) for v in p[-3:]]
        atoms.append((_norm_symbol(p[0].split("(")[0].split("-")[0]), *xyz))
    if not atoms:
        raise ValueError(f"未能在 {path} 中找到笛卡尔坐标")
    return atoms


def read_log_geometry(path):
    """读取高斯 log 中最后一个 Standard/Input orientation 的几何结构。"""
    text = Path(path).read_text(errors="ignore")
    blocks = re.findall(r"(?:Standard|Input) orientation:.*?-{5,}\n.*?-{5,}\n(.*?)\n\s*-{5,}", text, re.S)
    if not blocks:
        raise ValueError(f"{path} 中未找到几何结构")
    atoms = []
    for line in blocks[-1].splitlines():
        p = line.split()
        atoms.append((ELEMENTS[int(p[1])], float(p[3]), float(p[4]), float(p[5])))
    return atoms


def read_geometry(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".xyz":
        return read_xyz(path)
    if suffix in (".gjf", ".com", ".gjc"):
        return read_gjf(path)
    if suffix in (".log", ".out"):
        return read_log_geometry(path)
    if suffix in (".fchk", ".fch"):
        d = read_fchk(path)
        xyz = [v * BOHR2ANG for v in d["Current cartesian coordinates"]]
        return [(ELEMENTS[z], *xyz[3 * i:3 * i + 3]) for i, z in enumerate(d["Atomic numbers"])]
    raise ValueError(f"不支持的结构文件格式: {suffix}（支持 xyz / gjf / com / log / out / fchk）")


def write_xyz(path, atoms, comment=""):
    lines = [str(len(atoms)), comment]
    lines += [f"{s:<3s}{x:16.8f}{y:16.8f}{z:16.8f}" for s, x, y, z in atoms]
    Path(path).write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------
# gjf 输入
# ----------------------------------------------------------------------------------------------
def format_job(route, title, charge, mult, atoms=None, chk=None, nproc=None, mem=None, tail="",
               oldchk=None):
    lines = []
    if oldchk:
        lines.append(f"%oldchk={oldchk}")
    if chk:
        lines.append(f"%chk={chk}")
    if nproc:
        lines.append(f"%nprocshared={nproc}")
    if mem:
        lines.append(f"%mem={mem}")
    route = " ".join(route.split())
    lines.append(route if route.startswith("#") else "# " + route)
    lines += ["", title, "", f"{charge} {mult}"]
    if atoms:
        lines += [f"{s:<3s}{x:16.8f}{y:16.8f}{z:16.8f}" for s, x, y, z in atoms]
    lines.append("")
    if tail.strip():
        lines += [tail.strip(), ""]
    return "\n".join(lines) + "\n"


def gaussian_basis_fix(text):
    """把 ORCA / 文献常用的基组写法换成 Gaussian 能识别的写法：def2-TZVP → Def2TZVP (Gaussian 不接受连字符)。"""
    return re.sub(r"(?i)\bdef2-([a-z]+)\b", lambda m: "Def2" + m.group(1).upper(), text)


def write_gjf(path, jobs):
    """jobs: format_job 的参数字典列表，多个任务以 --Link1-- 串联。"""
    text = "--Link1--\n".join(format_job(**j) for j in jobs)
    Path(path).write_text(gaussian_basis_fix(text) + "\n")


_G_ERRORS = [
    (r"QPErr --- A syntax error was detected", "输入行有语法错误",
     "检查关键词拼写；基组写法须为 Gaussian 格式，如 Def2TZVP、Def2SVP (不能写 def2-TZVP)，"
     "溶剂名如 dichloromethane、toluene"),
    (r"Small interatomic distances encountered|Problem with the distance matrix", "原子之间距离过近",
     "结构中有重叠或重复的原子，或二聚体中两个分子交叠；检查结构文件"),
    (r"Convergence failure -- run terminated|SCF has not converged", "SCF 不收敛",
     "可加 scf=(xqc,maxcycle=512)，或先用小基组计算再以 guess=read 读入"),
    (r"Erroneous write|Write error in NtrExt1|No space left", "写文件失败 (临时目录磁盘空间不足)",
     "清理临时目录或用 %RWF 指定更大的磁盘；“程序路径设置”中可更改 Gaussian 临时目录"),
    (r"galloc: could not allocate memory|Out-of-memory", "内存不足", "减小 %mem，或减少 %nprocshared"),
    (r"The combination of multiplicity\s+\d+ and\s+\d+ electrons is impossible", "电荷与多重度不匹配",
     "检查电荷 / 多重度：偶数电子用单重态 (1)，奇数电子用二重态 (2)"),
    (r"FileIO operation on non-existent file", "读取 chk 时找不到数据", "确认 %oldchk 指定的 chk 存在于任务目录且来自正常结束的计算"),
    (r"Error in internal coordinate system|FormBX had a problem", "内坐标系统出错 (优化中断)",
     "可加 opt=cartesian 或从最后的结构重新优化"),
    (r"Atom\s+\d+ has no basis functions|There are no basis functions", "某些原子没有基组",
     "重原子需要赝势基组 (GenECP)，或该基组不包含此元素"),
]


def explain_gaussian_error(text):
    """识别 Gaussian 输出中的常见错误，返回中文说明 (含出错行) 或 None。"""
    if "Error termination" not in text and "QPErr" not in text:
        return None
    lines = text.splitlines()
    for pat, what, fix in _G_ERRORS:
        m = re.search(pat, text)
        if m:
            msg = f"Gaussian 计算出错：{what}。建议：{fix}。"
            if "QPErr" in pat:
                k = next(i for i, l in enumerate(lines) if "QPErr" in l)
                bad = [l.rstrip() for l in lines[k + 1:k + 3]]
                msg += "\n出错位置 (Gaussian 用 ' 标出)：\n" + "\n".join(bad)
                if re.search(r"(?i)def2-", "\n".join(lines[max(0, k - 4):k + 2])):
                    msg += "\n本例原因：基组写成了 def2-xxx，Gaussian 中应写为 Def2xxx (新版软件生成输入时已自动转换)"
            return msg
    k = next((i for i, l in enumerate(lines) if "Error termination" in l), len(lines))
    return "Gaussian 计算出错。出错前的输出：\n" + "\n".join(l.rstrip() for l in lines[max(0, k - 6):k + 1])


# ----------------------------------------------------------------------------------------------
# log 解析
# ----------------------------------------------------------------------------------------------
_SCF_RE = re.compile(r"SCF Done:\s+E\([^)]*\)\s*=\s*(-?\d+\.\d+)")
_ROUTE_RE = re.compile(r"^ -{5,}\n( #[^\n]*(?:\n (?!-{5,})[^\n]*)*)\n -{5,}$", re.M)
_NUM_RE = re.compile(r"-?\d+\.\d+")


def _orbitals(blk):
    """返回该步最后一次布居分析的 alpha/beta 占据与空轨道能量 (Hartree)。"""
    sets, cur, prev_eig = [], None, False
    for line in blk.splitlines():
        m = re.match(r"\s*(Alpha|Beta)\s+(occ\.|virt\.)\s+eigenvalues --(.*)", line)
        if m:
            if not prev_eig:
                cur = {"Alpha occ.": [], "Alpha virt.": [], "Beta occ.": [], "Beta virt.": []}
                sets.append(cur)
            cur[f"{m.group(1)} {m.group(2)}"] += [float(v) for v in _NUM_RE.findall(m.group(3))]
            prev_eig = True
        else:
            prev_eig = False
    return sets[-1] if sets else None


def split_log_steps(text):
    """按 route 段切分 log 的各个计算步骤（--Link1-- 任务）。
    不依赖 'Entering Gaussian System'（Windows 版高斯不输出该行）；
    'opt freq' 自动产生的内部频率步骤 (Geom=AllCheck ... Freq) 并入前一步。"""
    hits = list(_ROUTE_RE.finditer(text))
    if not hits:
        return [("", text)]
    blocks = []
    for k, m in enumerate(hits):
        route = " ".join(m.group(1).split())
        body = text[m.start():hits[k + 1].start() if k + 1 < len(hits) else len(text)]
        internal_freq = re.search(r"geom=allcheck", route, re.I) and re.search(r"\bfreq\b", route, re.I) \
            and blocks and re.search(r"\bopt\b", blocks[-1][0], re.I) and re.search(r"\bfreq\b", blocks[-1][0], re.I)
        if internal_freq:
            blocks[-1] = (blocks[-1][0], blocks[-1][1] + body)
        else:
            blocks.append((route, body))
    blocks[0] = (blocks[0][0], text[:hits[0].start()] + blocks[0][1])
    return blocks


def read_log_steps(path):
    """返回每个计算步骤的能量、是否正常结束、虚频数、轨道能等信息。"""
    text = Path(path).read_text(errors="ignore")
    steps = []
    for route, blk in split_log_steps(text):
        energies = _SCF_RE.findall(blk)
        freqs = []
        for line in re.findall(r"Frequencies --\s+(.*)", blk):
            freqs += [float(v) for v in _NUM_RE.findall(line)]
        cm = re.findall(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", blk)
        steps.append({
            "route": route,
            "energy": float(energies[-1]) if energies else None,
            "normal": "Normal termination" in blk,
            "error": "Error termination" in blk,
            "opt_failed": "Optimization stopped" in blk,
            "n_imag": sum(1 for f in freqs if f < 0),
            "min_freq": min(freqs) if freqs else None,
            "charge_mult": (int(cm[-1][0]), int(cm[-1][1])) if cm else None,
            "orbitals": _orbitals(blk),
        })
    return steps


# ----------------------------------------------------------------------------------------------
# fchk 解析
# ----------------------------------------------------------------------------------------------
def read_fchk(path):
    """读取格式化检查点文件，返回 {关键字: 值}，整数/实数数组转为 list。"""
    lines = Path(path).read_text(errors="ignore").splitlines()
    data, i = {}, 2
    while i < len(lines):
        line = lines[i]
        key, rest = line[:40].strip(), line[40:].split()
        i += 1
        if not key or not rest:
            continue
        typ = rest[0]
        if len(rest) >= 3 and rest[1] == "N=":
            n = int(rest[2])
            if typ in ("I", "R"):
                vals = []
                while len(vals) < n:
                    vals += lines[i].split()
                    i += 1
                conv = int if typ == "I" else float
                data[key] = [conv(v) for v in vals[:n]]
            else:  # C / H / L 数组：按每行条目数跳过
                per_line = {"C": 5, "H": 9, "L": 72}.get(typ, 5)
                i += math.ceil(n / per_line)
        elif len(rest) >= 2:
            try:
                data[key] = int(rest[1]) if typ == "I" else float(rest[1]) if typ == "R" else rest[1]
            except ValueError:
                data[key] = rest[1]
    return data


def read_fchk_hessian(path):
    """从 freq 计算的 fchk 读取: 元素符号、坐标(Å)、质量(amu)、笛卡尔 Hessian (Hartree/Bohr^2)。"""
    import numpy as np
    d = read_fchk(path)
    if "Cartesian Force Constants" not in d:
        raise ValueError(f"{path} 中没有 Hessian (Cartesian Force Constants)，请确认是 freq 计算的 fchk")
    z = d["Atomic numbers"]
    n3 = 3 * len(z)
    tri = np.array(d["Cartesian Force Constants"])
    h = np.zeros((n3, n3))
    h[np.tril_indices(n3)] = tri
    h = h + h.T - np.diag(np.diag(h))
    coords = np.array(d["Current cartesian coordinates"]).reshape(-1, 3) * BOHR2ANG
    symbols = [ELEMENTS[i] for i in z]
    masses = np.array(d.get("Real atomic weights") or [MASSES[s] for s in symbols])
    return {"symbols": symbols, "coords": coords, "masses": masses, "hessian": h,
            "energy": d.get("Total Energy")}


# ----------------------------------------------------------------------------------------------
# 混合基组 / 赝势 (GenECP) 与溶剂
# ----------------------------------------------------------------------------------------------
SOLVENTS = ["", "dichloromethane", "chloroform", "toluene", "tetrahydrofuran", "acetonitrile", "methanol",
            "dimethylsulfoxide", "water", "chlorobenzene", "n,n-dimethylformamide"]


def _z(sym):
    return ELEMENTS.index(sym) if sym in ELEMENTS else 99


def needs_ecp(basis, elements):
    """Pople / Dunning 基组不含 Kr 以后的元素，此时需要 GenECP；def2 系列在高斯中自带赝势。"""
    b = basis.lower().replace(" ", "")
    pople_like = b.startswith(("6-", "3-", "sto", "cc-p", "aug-cc", "4-"))
    return pople_like and any(_z(e) > 36 for e in elements)


def basis_spec(basis, elements, ecp_basis="SDD", ecp_elements=None, force_gen=False):
    """返回 (写在 route 中的基组记号, 写在分子坐标后的基组/赝势段)。
    elements: 体系中的元素；ecp_elements: 用赝势的元素（默认自动取原子序数 > 36 的元素）。"""
    elements = sorted(set(elements), key=_z)
    heavy = sorted(set(ecp_elements) if ecp_elements else {e for e in elements if _z(e) > 36}, key=_z)
    heavy = [e for e in heavy if e in elements]
    if not heavy or not (force_gen or needs_ecp(basis, elements)):
        return basis, ""
    light = [e for e in elements if e not in heavy]
    lines = []
    if light:
        lines += [" ".join(light) + " 0", basis, "****"]
    lines += [" ".join(heavy) + " 0", ecp_basis, "****", "", " ".join(heavy) + " 0", ecp_basis]
    return "genecp", "\n".join(lines)


def solvent_kw(solvent, model="pcm"):
    solvent = (solvent or "").strip()
    return f"scrf=({model},solvent={solvent})" if solvent else ""


def level_and_tail(method, basis, atoms_or_elements, ecp_basis="SDD"):
    """method/basis 的 route 写法及对应的基组段。"""
    elems = [a[0] if isinstance(a, (tuple, list)) else a for a in atoms_or_elements]
    token, tail = basis_spec(basis, elems, ecp_basis)
    return f"{method}/{token}", tail
