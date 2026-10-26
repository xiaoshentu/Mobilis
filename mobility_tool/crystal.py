"""晶体结构处理：读取 CIF → 展开对称操作 → 拼出完整分子 → 搜索近邻二聚体 → 合并等价二聚体。

等价判据：两个二聚体的全部分子间原子距离排序后逐一相同（容差内），等价二聚体的转移积分相同，
只需计算一次，但在迁移率计算中每个近邻都按自己的跳跃矢量计入。
"""
import json
import re
import shlex
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .gaussian_io import COVALENT_RADII, write_xyz


# ----------------------------------------------------------------------------------------------
# CIF
# ----------------------------------------------------------------------------------------------
def _num(v):
    v = re.sub(r"\(.*?\)", "", str(v))
    try:
        return float(v)
    except ValueError:
        return None


def _cif_tokens(text):
    """返回 (标签字典, 循环列表)。忽略 ; ; 多行文本。"""
    lines, skip = [], False
    for line in text.splitlines():
        if line.startswith(";"):
            skip = not skip
            continue
        if not skip and not line.lstrip().startswith("#"):
            lines.append(line)
    tags, loops, i = {}, [], 0
    while i < len(lines):
        s = lines[i].strip()
        if s.lower() == "loop_":
            i += 1
            heads = []
            while i < len(lines) and lines[i].strip().startswith("_"):
                heads.append(lines[i].strip().split()[0].lower())
                i += 1
            vals = []
            while i < len(lines):
                t = lines[i].strip()
                if not t:
                    i += 1
                    if vals:
                        continue
                    continue
                if t.startswith("_") or t.lower().startswith(("loop_", "data_")):
                    break
                try:
                    vals += shlex.split(t, posix=True)
                except ValueError:
                    vals += t.split()
                i += 1
            n = len(heads)
            rows = [vals[k:k + n] for k in range(0, len(vals) - n + 1, n)]
            loops.append({"heads": heads, "rows": rows})
            continue
        if s.startswith("_"):
            parts = s.split(None, 1)
            if len(parts) == 2:
                tags[parts[0].lower()] = parts[1].strip().strip("'\"")
            elif i + 1 < len(lines):
                tags[parts[0].lower()] = lines[i + 1].strip().strip("'\"")
                i += 1
        i += 1
    return tags, loops


def _parse_symop(op):
    op = op.replace(" ", "").lower()
    comps = op.split(",")
    if len(comps) != 3 or not re.fullmatch(r"[xyz0-9+\-*/.,]+", op):
        raise ValueError(f"无法解析对称操作: {op}")
    rot, trans = np.zeros((3, 3)), np.zeros(3)
    for r, expr in enumerate(comps):
        f = lambda x, y, z, e=expr: eval(e, {"__builtins__": {}}, {"x": x, "y": y, "z": z})  # noqa: E731
        t = f(0.0, 0.0, 0.0)
        trans[r] = t
        for c, e in enumerate(np.eye(3)):
            rot[r, c] = f(*e) - t
    return rot, trans


def cell_matrix(a, b, c, alpha, beta, gamma):
    al, be, ga = np.radians([alpha, beta, gamma])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    return np.array([[a, 0, 0], [b * np.cos(ga), b * np.sin(ga), 0], [cx, cy, np.sqrt(c ** 2 - cx ** 2 - cy ** 2)]])


def read_cif(path):
    text = Path(path).read_text(errors="ignore")
    blocks = re.split(r"(?im)^data_", text)
    text = next((b for b in blocks if "_atom_site_fract_x" in b.lower()), text)
    tags, loops = _cif_tokens(text)
    try:
        cell = [_num(tags[k]) for k in ("_cell_length_a", "_cell_length_b", "_cell_length_c",
                                        "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")]
    except KeyError as exc:
        raise ValueError(f"CIF 缺少晶胞参数 {exc}") from exc
    warnings = []
    ops = ["x,y,z"]
    for lp in loops:
        for key in ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz"):
            if key in lp["heads"]:
                ops = [row[lp["heads"].index(key)] for row in lp["rows"]]
    if ops == ["x,y,z"]:
        warnings.append("CIF 中未找到对称操作，只使用 x,y,z")
    site = next((lp for lp in loops if "_atom_site_fract_x" in lp["heads"]), None)
    if site is None:
        raise ValueError("CIF 中没有 _atom_site_fract_x 坐标")
    h = site["heads"]
    get = lambda row, k: row[h.index(k)] if k in h else None  # noqa: E731
    atoms = []
    for row in site["rows"]:
        sym = get(row, "_atom_site_type_symbol") or get(row, "_atom_site_label")
        sym = re.match(r"[A-Za-z]{1,2}", sym).group(0)
        sym = sym[0].upper() + sym[1:].lower()
        if sym not in COVALENT_RADII and sym[0] in COVALENT_RADII:
            sym = sym[0]
        occ = _num(get(row, "_atom_site_occupancy") or 1) or 1.0
        grp = (get(row, "_atom_site_disorder_group") or ".").strip()
        if grp not in (".", "?", "1") or occ < 0.5:
            continue
        if occ < 1.0:
            warnings.append(f"原子 {get(row, '_atom_site_label')} 占有率 {occ}，已保留（无序结构请先手动处理）")
        frac = [_num(get(row, k)) for k in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")]
        atoms.append((sym, np.array(frac)))

    lat = cell_matrix(*cell)
    # 全部对称操作一次生成，再用周期性 KD-tree 去重
    asym_sym = [a for a, _ in atoms]
    asym_f = np.array([f for _, f in atoms])
    allf, alls = [], []
    for op in ops:
        rot, trans = _parse_symop(op)
        allf.append((asym_f @ rot.T + trans) % 1.0)
        alls += asym_sym
    allf = np.vstack(allf) % 1.0
    allf[allf > 1 - 1e-9] = 0.0
    tree = cKDTree(allf, boxsize=1.0)
    tol = 0.3 / max(cell[:3])                                  # 0.3 Å 换算为分数坐标 (保守)
    keep = np.ones(len(allf), bool)
    for i, j in tree.query_pairs(tol, output_type="ndarray"):
        if keep[i] and keep[j] and alls[i] == alls[j]:
            d = allf[j] - allf[i]
            d -= np.round(d)
            if np.linalg.norm(d @ lat) < 0.3:
                keep[j] = False
    syms = [s_ for s_, k in zip(alls, keep) if k]
    fracs = allf[keep]
    sg = next((str(tags[k]).strip().strip("'\"") for k in ("_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M",
                                                            "_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m")
               if k in tags and str(tags[k]).strip() not in ("?", ".")), "")
    return {"cell": cell, "lattice": lat, "symbols": syms, "frac": np.array(fracs), "n_symops": len(ops),
            "spacegroup": " ".join(sg.split()),
            "symops": [tuple(np.asarray(x, float).tolist() for x in _parse_symop(o)) for o in ops],
            "warnings": warnings}


# ----------------------------------------------------------------------------------------------
# 分子与二聚体
# ----------------------------------------------------------------------------------------------
def _bonded_components(symbols, cart, scale=1.2):
    rad = np.array([COVALENT_RADII.get(s, 1.5) for s in symbols])
    tree = cKDTree(cart)
    pairs = tree.query_pairs(2 * scale * rad.max(), output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(len(symbols))
    d = np.linalg.norm(cart[pairs[:, 0]] - cart[pairs[:, 1]], axis=1)
    ok = d < scale * (rad[pairs[:, 0]] + rad[pairs[:, 1]])
    p = pairs[ok]
    g = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(len(symbols),) * 2)
    return connected_components(g, directed=False)[1]


def find_molecules(crys, scale=1.2):
    """返回晶胞内完整分子列表（质心落在 [0,1)³ 中的那些）。"""
    lat, frac, syms = crys["lattice"], crys["frac"], crys["symbols"]
    n = len(syms)
    for rng in (1, 2):
        shifts = np.array([[i, j, k] for i in range(-rng, rng + 1) for j in range(-rng, rng + 1)
                           for k in range(-rng, rng + 1)])
        big_f = (frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        big_s = syms * len(shifts)
        labels = _bonded_components(big_s, big_f @ lat, scale)
        mols = []
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            cen = big_f[idx].mean(0)
            if np.all(cen >= -1e-6) and np.all(cen < 1 - 1e-6):
                mols.append({"symbols": [big_s[i] for i in idx], "cart": big_f[idx] @ lat,
                             "centroid": cen @ lat})
        if sum(len(m["symbols"]) for m in mols) == n:
            return mols
    raise ValueError("无法从晶胞拼出完整分子（原子数不匹配），请检查 CIF 是否有无序或缺少氢原子")


def find_dimers(crys, molecules, ref=0, cutoff=4.0, criterion="atom", fp_tol=0.02):
    """搜索参考分子周围的近邻。criterion='atom' 用最短原子间距, 'centroid' 用质心距离。"""
    lat = crys["lattice"]
    ra = molecules[ref]
    rmax = max(np.linalg.norm(m["cart"] - m["centroid"], axis=1).max() for m in molecules)
    dimers = []
    rng = range(-3, 4)
    for mi, m in enumerate(molecules):
        for t in ((i, j, k) for i in rng for j in rng for k in rng):
            if mi == ref and t == (0, 0, 0):
                continue
            shift = np.array(t) @ lat
            cen = m["centroid"] + shift
            vec = cen - ra["centroid"]
            dc = np.linalg.norm(vec)
            if criterion == "centroid":
                if dc > cutoff:
                    continue
            elif dc > cutoff + 2 * rmax:
                continue
            dist = cdist(ra["cart"], m["cart"] + shift)
            dmin = dist.min()
            if criterion == "atom" and dmin > cutoff:
                continue
            dimers.append({"mol": mi, "translation": list(t), "vector": vec.tolist(), "r_centroid": float(dc),
                           "d_min": float(dmin), "fingerprint": np.sort(dist.ravel()),
                           "atoms_b": list(zip(m["symbols"], (m["cart"] + shift).tolist()))})
    dimers.sort(key=lambda d: d["r_centroid"])
    groups = []
    for d in dimers:
        for g in groups:
            ref_fp = g["fingerprint"]
            if len(ref_fp) == len(d["fingerprint"]) and np.abs(ref_fp - d["fingerprint"]).max() < fp_tol:
                g["members"].append(d)
                break
        else:
            groups.append({"fingerprint": d["fingerprint"], "members": [d]})
    for gi, g in enumerate(groups):
        g["id"] = gi + 1
        for d in g["members"]:
            d["group"] = gi + 1
    atoms_a = list(zip(ra["symbols"], ra["cart"].tolist()))
    return {"ref": ref, "cutoff": cutoff, "criterion": criterion, "atoms_a": atoms_a,
            "dimers": dimers, "groups": groups}


def export_dimers(result, outdir, prefix="G"):
    """每组等价二聚体导出一个 xyz（A 在前），并写 dimers.json 记录全部近邻。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    atoms_a = [(s, *xyz) for s, xyz in result["atoms_a"]]
    files = []
    for g in result["groups"]:
        d = g["members"][0]
        atoms_b = [(s, *xyz) for s, xyz in d["atoms_b"]]
        p = outdir / f"{prefix}{g['id']}.xyz"
        write_xyz(p, atoms_a + atoms_b, f"dimer group {g['id']}: nA={len(atoms_a)} r={d['r_centroid']:.3f} A")
        files.append(p)
    info = {
        "n_atoms_A": len(atoms_a), "cutoff": result["cutoff"], "criterion": result["criterion"],
        "groups": [{"id": g["id"], "label": g.get("label", f"{prefix}{g['id']}"), "file": f"{prefix}{g['id']}.xyz",
                    "multiplicity": len(g["members"]),
                    "r_centroid": g["members"][0]["r_centroid"], "d_min": g["members"][0]["d_min"]}
                   for g in result["groups"]],
        "neighbors": [{"group": d["group"], "vector": d["vector"], "r_centroid": d["r_centroid"],
                       "d_min": d["d_min"], "translation": d["translation"], "mol": d["mol"]}
                      for d in result["dimers"]],
    }
    (outdir / "dimers.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return files


def mobility_pathways_from_dimers(dimers_json, j_by_group_meV):
    """由 dimers.json 和每组的 J (meV) 生成迁移率计算所需的路径列表（每个近邻一条）。"""
    info = json.loads(Path(dimers_json).read_text()) if not isinstance(dimers_json, dict) else dimers_json
    paths = []
    for i, nb in enumerate(info["neighbors"]):
        j = j_by_group_meV.get(nb["group"], j_by_group_meV.get(str(nb["group"])))
        if j is None:
            continue
        paths.append({"name": f"G{nb['group']}-{i + 1}", "vector": nb["vector"], "J_meV": float(j), "count": 1})
    return paths


def split_dimer(atoms):
    """把二聚体结构按成键拆成两个分子。返回 (n_a, 是否需要重排, 重排后的原子列表)。"""
    syms = [a[0] for a in atoms]
    xyz = np.array([a[1:] for a in atoms], float)
    labels = _bonded_components(syms, xyz)
    uniq = list(dict.fromkeys(labels.tolist()))
    if len(uniq) != 2:
        raise ValueError(f"按成键找到 {len(uniq)} 个分子，二聚体应正好包含 2 个分子")
    ia = [i for i, l in enumerate(labels) if l == uniq[0]]
    ib = [i for i, l in enumerate(labels) if l == uniq[1]]
    reordered = [atoms[i] for i in ia + ib]
    return len(ia), ia != list(range(len(ia))), reordered


def strain_structure(crys, mols, scales=(1.0, 1.0, 1.0)):
    """按比例缩放晶格矢量 a、b、c（应变），分子作为刚体随质心的分数坐标平移，内部几何不变。"""
    sc = np.asarray(scales, float)
    lat = crys["lattice"]
    new_lat = lat * sc[:, None]
    inv = np.linalg.inv(lat)
    new_mols = []
    for m in mols:
        c_new = (m["centroid"] @ inv) @ new_lat
        shift = c_new - m["centroid"]
        new_mols.append({"symbols": m["symbols"], "cart": m["cart"] + shift, "centroid": c_new})
    a, b, c, al, be, ga = crys["cell"]
    new = dict(crys, lattice=new_lat, cell=(a * sc[0], b * sc[1], c * sc[2], al, be, ga))
    return new, new_mols


def stacking_geometry(atoms_a, atoms_b, contact=4.5):
    """二聚体接触区的 π 堆积几何：接触原子 (非氢，距对方 < contact) 拟合平面，
    给出面间距、两平面夹角、滑移距离和堆积类型 (面对面 / 滑移堆积 / 边对面 / 其它)。"""
    xa = np.array([a[1:] for a in atoms_a if a[0] != "H"], float)
    xb = np.array([b[1:] for b in atoms_b if b[0] != "H"], float)
    d = cdist(xa, xb)
    ca, cb = xa[d.min(1) < contact], xb[d.min(0) < contact]
    if len(ca) < 3 or len(cb) < 3:
        return {"type": "无 π 接触", "d_plane": None, "angle": None, "slip": None, "n_contact": (len(ca), len(cb))}

    def plane(x):
        c = x.mean(0)
        _, sv, vt = np.linalg.svd(x - c)
        return c, vt[2], sv[2] / max(sv[0], 1e-9)
    c1, n1, flat1 = plane(ca)
    c2, n2, flat2 = plane(cb)
    ang = float(np.degrees(np.arccos(min(1.0, abs(n1 @ n2)))))
    dc = c2 - c1
    dpl = float(abs(dc @ n1) + abs(dc @ n2)) / 2
    slip = float(np.sqrt(max(np.linalg.norm(dc) ** 2 - dpl ** 2, 0)))
    if ang < 20 and dpl < 4.2:
        kind = "面对面 π 堆积" if slip < 2.0 else "滑移 π 堆积"
    elif ang > 50:
        kind = "边对面 (T 型 / 鱼骨)"
    else:
        kind = "倾斜堆积"
    return {"type": kind, "d_plane": dpl, "angle": ang, "slip": slip, "d_min": float(d.min()),
            "n_contact": (len(ca), len(cb)), "planarity": (float(flat1), float(flat2))}



def measure(points):
    """2 点距离 (Å)、3 点角度、4 点二面角 (°)。返回 (类型, 数值, 标签)。"""
    p = np.asarray(points, float)
    if len(p) == 2:
        v = float(np.linalg.norm(p[1] - p[0]))
        return "distance", v, f"{v:.3f} Å"
    if len(p) == 3:
        a, b = p[0] - p[1], p[2] - p[1]
        v = float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
        return "angle", v, f"{v:.2f}°"
    if len(p) == 4:
        b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
        b1n = b1 / np.linalg.norm(b1)
        v_ = b0 - (b0 @ b1n) * b1n
        w_ = b2 - (b2 @ b1n) * b1n
        v = float(np.degrees(np.arctan2(np.cross(b1n, v_) @ w_, v_ @ w_)))
        return "dihedral", v, f"{v:.2f}°"
    return None, None, ""


def check_dimer_geometry(atoms_a, atoms_b=None, intra=0.5, inter=0.8, warn=1.6):
    """写出计算输入前的几何检查。严重问题抛出 ValueError (指出原子与距离)，较近接触返回提醒文字。"""
    def chk_intra(atoms, tag):
        x = np.array([a[1:] for a in atoms], float)
        if len(x) < 2:
            return
        d = cdist(x, x) + np.eye(len(x)) * 99
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] < intra:
            raise ValueError(f"{tag}中原子 {atoms[i][0]}{i + 1} 与 {atoms[j][0]}{j + 1} 相距仅 {d[i, j]:.3f} Å："
                             "结构中有重叠或重复的原子 (CIF 中的无序位置、重复的对称拷贝等)，请检查结构")
    chk_intra(atoms_a, "单体 A " if atoms_b is not None else "分子")
    if atoms_b is None:
        return None
    chk_intra(atoms_b, "单体 B ")
    xa = np.array([a[1:] for a in atoms_a], float)
    xb = np.array([b[1:] for b in atoms_b], float)
    if np.linalg.norm(xa.mean(0) - xb.mean(0)) < 1.0:
        raise ValueError("两个单体的中心几乎重合 (相距 < 1 Å)：可能把同一个分子选了两次，请重新选择二聚体")
    d = cdist(xa, xb)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    if d[i, j] < inter:
        raise ValueError(f"单体 A 的 {atoms_a[i][0]}{i + 1} 与单体 B 的 {atoms_b[j][0]}{j + 1} 相距仅 {d[i, j]:.3f} Å："
                         "两个分子发生重叠，Gaussian 会报“原子距离过近”。请检查二聚体是否选错或 CIF 是否有无序")
    if d[i, j] < warn:
        return f"[提醒] 分子间最近接触 {atoms_a[i][0]}{i + 1}…{atoms_b[j][0]}{j + 1} = {d[i, j]:.2f} Å，偏近"
    return None


def _long_axis(atoms):
    x = np.array([p for s, p in atoms if s != "H"], float)
    w, v = np.linalg.eigh(np.cov((x - x.mean(0)).T))
    return v[:, -1]


def transport_plane_normal(crys, dimer_result, plane="auto"):
    """传输平面的法向量。plane: 'ab' / 'bc' / 'ac'，或 'auto' (由两个最近且不共线的近邻方向张成，即分子层)。"""
    lat = np.asarray(crys["lattice"], float)
    pairs = {"ab": (0, 1), "bc": (1, 2), "ac": (0, 2)}
    if plane in pairs:
        i, j = pairs[plane]
        n = np.cross(lat[i], lat[j])
        return n / np.linalg.norm(n)
    vs = [np.asarray(d["vector"]) for d in sorted(dimer_result["dimers"], key=lambda d: d["r_centroid"])]
    v1 = vs[0]
    for v in vs[1:]:
        if abs(v @ v1) / (np.linalg.norm(v) * np.linalg.norm(v1)) < 0.95:
            n = np.cross(v1, v)
            return n / np.linalg.norm(n)
    raise ValueError("近邻方向都共线，无法确定传输平面")


def plane_name(crys, normal, tol=0.05):
    """法向量若与某个晶面 (ab / bc / ac) 一致，返回其名称。"""
    lat = np.asarray(crys["lattice"], float)
    for name, (i, j) in {"ab": (0, 1), "bc": (1, 2), "ac": (0, 2)}.items():
        n = np.cross(lat[i], lat[j])
        if abs(abs(n @ normal) / np.linalg.norm(n) - 1) < tol:
            return name
    return None


def paper_dimers(crys, molecules, ref=0, plane="auto", cutoff=15.0, criterion="centroid", split_directions=True,
                 drop_collinear=True, plane_tol=0.6, contact=4.0):
    """按方向选择二聚体 (Deng et al., Nat. Protoc. 2015, 10, 632 中二维角分辨迁移率的做法)：
    按 criterion / cutoff 搜索近邻；plane 为 'auto' (分子层) / 'ab' / 'bc' / 'ac' 时只保留该平面内的近邻，'3d' 保留全部；
    同一直线上只保留最近的一个；对称等价但方向不同的二聚体分别列出 (v 与 −v 为同一方向)。
    命名：P = 最近的 π 堆积方向；T = 其它有紧密接触 (最短原子距离 ≤ contact) 的方向；L = 远程 (无紧密接触)；
    I = 层间 (plane='3d' 时位于分子层之外)。返回结构与 find_dimers 相同。"""
    full = find_dimers(crys, molecules, ref=ref, cutoff=cutoff, criterion=criterion)
    if not full["dimers"]:
        raise ValueError("截断范围内没有近邻")
    n = transport_plane_normal(crys, full, "auto" if plane == "3d" else plane)
    in_plane = lambda d: abs(np.asarray(d["vector"]) @ n) < plane_tol      # noqa: E731
    dims = full["dimers"] if plane == "3d" else [d for d in full["dimers"] if in_plane(d)]
    if drop_collinear:                                    # 同一直线上只保留最近的 (去掉第二、第三近邻)
        keep = []
        for d in sorted(dims, key=lambda x: x["r_centroid"]):
            u = np.asarray(d["vector"]) / d["r_centroid"]
            if any(abs(u @ (np.asarray(k["vector"]) / k["r_centroid"])) > 0.999 and k["r_centroid"] < d["r_centroid"] - 0.1
                   for k in keep):
                continue
            keep.append(d)
        dims = keep
    a_atoms = [(s, *p) for s, p in full["atoms_a"]]
    buckets = {}
    for d in dims:
        u = np.asarray(d["vector"]) / d["r_centroid"]
        k = u * np.sign(u[np.argmax(np.abs(u))])           # v 与 −v 视为同一方向
        key = (d["group"], tuple(np.round(k, 2))) if split_directions else (d["group"],)
        buckets.setdefault(key, []).append(d)
    groups = []
    for key, members in sorted(buckets.items(), key=lambda kv: (kv[1][0]["r_centroid"], kv[0])):
        d0 = members[0]
        geo = stacking_geometry(a_atoms, [(s, *p) for s, p in d0["atoms_b"]])
        groups.append({"fingerprint": d0["fingerprint"], "members": members, "stacking": geo["type"]})
    pi_types = ("面对面 π 堆积", "滑移 π 堆积")
    first_pi = next((g for g in groups if g["stacking"] in pi_types and in_plane(g["members"][0])), None)
    for g in groups:
        d0 = g["members"][0]
        if not in_plane(d0):
            g["kind"] = "I"
        elif g is first_pi:
            g["kind"] = "P"
        elif d0["d_min"] <= contact:
            g["kind"] = "T"
        else:
            g["kind"] = "L"
    counts = {}
    for g in groups:
        counts[g["kind"]] = counts.get(g["kind"], 0) + 1
    seen = {}
    for gi, g in enumerate(groups):
        g["id"] = gi + 1
        seen[g["kind"]] = seen.get(g["kind"], 0) + 1
        g["label"] = g["kind"] + (str(seen[g["kind"]]) if counts[g["kind"]] > 1 else "")
        for d in g["members"]:
            d["group"] = gi + 1
    out_dims = [d for g in groups for d in g["members"]]
    return {"ref": ref, "cutoff": cutoff, "criterion": criterion, "atoms_a": full["atoms_a"], "dimers": out_dims,
            "groups": groups, "mode": "paper", "plane_normal": n.tolist(),
            "plane": "三维" if plane == "3d" else (plane_name(crys, n) or "自定义")}


def remove_groups(dimer_result, group_ids):
    """从二聚体结果中移除指定的组，并重新编号 (保留标签)。"""
    groups = [g for g in dimer_result["groups"] if g["id"] not in set(group_ids)]
    for gi, g in enumerate(groups):
        g["id"] = gi + 1
        for d in g["members"]:
            d["group"] = gi + 1
    return dict(dimer_result, groups=groups, dimers=[d for g in groups for d in g["members"]])
