"""原子 / 基团取代：单分子或晶体中的分子 (可按晶体对称性取代全部等价位置)。

元素取代   末端原子 (只连一个原子，如 H→F) 沿原键方向按共价半径调整键长；
           价态补偿：重原子取代时按价态差删除或补加 H (如 芳环 C–H → N 删除 H；N → C–H 补 H)。
基团取代   把末端原子 (通常是 H) 换成 −F −Cl −Br −I −CH3 −CF3 −OH −OCH3 −NH2 −CN −NO2 −C≡CH；
           平面基团与母体局部平面共面，四面体基团绕键轴转到离其它原子最远的取向。
晶体       用 CIF 的对称操作找出参考分子中所选原子在每个分子上的像 (含分子自身位于对称元素上时的分子内等价原子)，
           逐个分子取代，晶体对称性保持不变；也可只改所选分子 (结果为 P1)。
取代后的结构未经优化：单分子请在“重组能”页优化；晶体中为保持原堆积的刚性取代 (用于筛选与趋势分析)。
"""
import numpy as np

ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
            "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
            "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
            "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
# 共价半径 / Å (Cordero et al., Dalton Trans. 2008；C 取 sp3 值 0.76)
_COV = [0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,
        2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16,
        2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40,
        2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,
        1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
        2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69]
COV = {e: (_COV[i] if i < len(_COV) else 1.6) for i, e in enumerate(ELEMENTS)}
Z = {e: i + 1 for i, e in enumerate(ELEMENTS)}
VALENCE = {"H": 1, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1, "Si": 4, "P": 3, "S": 2, "Cl": 1, "Ge": 4, "As": 3,
           "Se": 2, "Br": 1, "Sn": 4, "Sb": 3, "Te": 2, "I": 1, "Al": 3, "Ga": 3}

# 基团模板：(原子, x, y, z)，锚原子在原点，母体在 −x 方向；y 位于母体局部平面内；BOND = 与 C(芳) 的键长
_T = np.radians
GROUPS = {
    "F": ([("F", 0, 0, 0)], 1.35, "plane"), "Cl": ([("Cl", 0, 0, 0)], 1.74, "plane"),
    "Br": ([("Br", 0, 0, 0)], 1.90, "plane"), "I": ([("I", 0, 0, 0)], 2.10, "plane"),
    "CH3": ([("C", 0, 0, 0)] + [("H", 1.09 * np.cos(_T(70.53)), 1.09 * np.sin(_T(70.53)) * np.cos(_T(a)),
                                 1.09 * np.sin(_T(70.53)) * np.sin(_T(a))) for a in (0, 120, 240)], 1.51, "rotor"),
    "CF3": ([("C", 0, 0, 0)] + [("F", 1.35 * np.cos(_T(70.53)), 1.35 * np.sin(_T(70.53)) * np.cos(_T(a)),
                                 1.35 * np.sin(_T(70.53)) * np.sin(_T(a))) for a in (0, 120, 240)], 1.50, "rotor"),
    "OH": ([("O", 0, 0, 0), ("H", 0.96 * np.cos(_T(71.5)), 0.96 * np.sin(_T(71.5)), 0)], 1.36, "plane"),
    "OCH3": ([("O", 0, 0, 0), ("C", 1.43 * np.cos(_T(63)), 1.43 * np.sin(_T(63)), 0)], 1.36, "plane+methyl"),
    "NH2": ([("N", 0, 0, 0), ("H", 1.01 * np.cos(_T(60)), 1.01 * np.sin(_T(60)), 0),
             ("H", 1.01 * np.cos(_T(60)), -1.01 * np.sin(_T(60)), 0)], 1.38, "plane"),
    "CN": ([("C", 0, 0, 0), ("N", 1.16, 0, 0)], 1.44, "plane"),
    "NO2": ([("N", 0, 0, 0), ("O", 1.22 * np.cos(_T(62.5)), 1.22 * np.sin(_T(62.5)), 0),
             ("O", 1.22 * np.cos(_T(62.5)), -1.22 * np.sin(_T(62.5)), 0)], 1.47, "plane"),
    "C≡CH": ([("C", 0, 0, 0), ("C", 1.20, 0, 0), ("H", 2.26, 0, 0)], 1.43, "plane"),
}
GROUP_LABELS = {"F": "−F", "Cl": "−Cl", "Br": "−Br", "I": "−I", "CH3": "−CH₃ 甲基", "CF3": "−CF₃ 三氟甲基",
                "OH": "−OH 羟基", "OCH3": "−OCH₃ 甲氧基", "NH2": "−NH₂ 氨基", "CN": "−CN 氰基", "NO2": "−NO₂ 硝基",
                "C≡CH": "−C≡CH 乙炔基"}


def neighbors(symbols, xyz, i, scale=1.25):
    xyz = np.asarray(xyz, float)
    d = np.linalg.norm(xyz - xyz[i], axis=1)
    lim = scale * (COV.get(symbols[i], 1.5) + np.array([COV.get(e, 1.5) for e in symbols]))
    return [j for j in np.where((d < lim) & (d > 0.1))[0]]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _local_frame(symbols, xyz, i, j):
    """x: j→i 键方向；y: 位于 j 的局部平面内 (由 j 的另一个近邻确定)；z: 法线。"""
    x = _unit(xyz[i] - xyz[j])
    others = [k for k in neighbors(symbols, xyz, j) if k != i]
    ref = xyz[others[0]] - xyz[j] if others else np.array([0.0, 0.0, 1.0])
    y = ref - (ref @ x) * x
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(x, [0, 0, 1.0]) if abs(x[2]) < 0.9 else np.cross(x, [0, 1.0, 0])
    y = _unit(y)
    return x, y, np.cross(x, y)


# ----------------------------------------------------------------------------------------------
# 单分子
# ----------------------------------------------------------------------------------------------
def substitute(symbols, xyz, indices, new, adjust_bond=True, fix_h=True):
    """把 indices 处原子换成元素 new。返回 (symbols, xyz, 说明列表, 旧序号→新序号映射)。"""
    if new not in COV:
        raise ValueError(f"未知元素: {new}")
    sym = list(symbols)
    X = np.array(xyz, float)
    notes, remove, add = [], set(), []
    for i in sorted(set(int(k) for k in indices)):
        old = sym[i]
        nb = neighbors(sym, X, i)
        sym[i] = new
        if adjust_bond and len(nb) == 1:                      # 末端原子：沿原键方向调整键长
            j = nb[0]
            L = COV[sym[j]] + COV[new]
            X[i] = X[j] + _unit(X[i] - X[j]) * L
            notes.append(f"{old}{i + 1} → {new} (末端，键长 {sym[j]}–{new} = {L:.2f} Å)")
        else:
            notes.append(f"{old}{i + 1} → {new}")
        if fix_h and old != "H" and new != "H" and old in VALENCE and new in VALENCE:
            hs = [k for k in nb if sym[k] == "H" and k not in remove]
            heavy = [k for k in nb if sym[k] != "H"]
            dv = VALENCE[old] - VALENCE[new]
            if dv > 0 and hs:                                  # 价态降低：删除多余的 H (如 C–H → N)
                for k in hs[:dv]:
                    remove.add(k)
                notes.append(f"  删除 {min(dv, len(hs))} 个与 {new}{i + 1} 相连的 H")
            elif dv < 0 and len(heavy) in (2, 3) and not hs:  # 价态升高：在键角平分线外侧补 H (如 N → C–H)
                d = -sum(_unit(X[k] - X[i]) for k in heavy)
                if np.linalg.norm(d) > 0.2:
                    add.append(("H", X[i] + _unit(d) * (COV[new] + COV["H"])))
                    notes.append(f"  在 {new}{i + 1} 上补 1 个 H")
    keep = [k for k in range(len(sym)) if k not in remove]
    mapping = {old: new_i for new_i, old in enumerate(keep)}
    sym2 = [sym[k] for k in keep] + [a for a, _ in add]
    X2 = np.vstack([X[keep]] + [p[None] for _, p in add]) if add else X[keep]
    return sym2, X2, notes, mapping


def attach_group(symbols, xyz, index, group):
    """把末端原子 index (只连一个原子，通常为 H) 换成基团 group。返回 (symbols, xyz, 说明, 映射)。"""
    if group not in GROUPS:
        raise ValueError(f"未知基团: {group}")
    sym = list(symbols)
    X = np.array(xyz, float)
    nb = neighbors(sym, X, index)
    if len(nb) != 1:
        raise ValueError(f"{sym[index]}{index + 1} 不是末端原子 (连接 {len(nb)} 个原子)，只能对末端原子 (如 H) 做基团取代")
    j = nb[0]
    tmpl, bond, mode = GROUPS[group]
    if sym[j] != "C":
        bond = COV[sym[j]] + COV[tmpl[0][0]]
    x, y, zv = _local_frame(sym, X, index, j)
    anchor = X[j] + x * bond
    others = np.delete(X, index, axis=0)

    def place(angle):
        c, s = np.cos(angle), np.sin(angle)
        yy, zz = c * y + s * zv, -s * y + c * zv
        return np.array([anchor + p[1] * x + p[2] * yy + p[3] * zz for p in tmpl])
    if mode == "rotor":
        best = max(np.radians(np.arange(0, 120, 5)), key=lambda a: np.linalg.norm(
            place(a)[1:, None] - others[None], axis=-1).min())
        pos = place(best)
    else:
        pos = place(0.0)
        if len(tmpl) > 1:                                      # 平面基团：两种朝向中选离其它原子更远的
            alt = place(np.pi)
            if np.linalg.norm(alt[1:, None] - others[None], axis=-1).min() > \
                    np.linalg.norm(pos[1:, None] - others[None], axis=-1).min():
                pos = alt
    new_atoms = [(t[0], p) for t, p in zip(tmpl, pos)]
    if mode == "plane+methyl":                                 # OCH3: 甲基碳上补 3 个 H
        o, cm = pos[0], pos[1]
        ax = _unit(cm - o)
        u = _unit(np.cross(ax, zv)) if np.linalg.norm(np.cross(ax, zv)) > 1e-6 else y
        w = np.cross(ax, u)
        for a in (0, 120, 240):
            t = np.radians(a)
            new_atoms.append(("H", cm + 1.09 * (np.cos(_T(70.53)) * ax + np.sin(_T(70.53)) * (np.cos(t) * u + np.sin(t) * w))))
    sym[index] = new_atoms[0][0]
    X[index] = new_atoms[0][1]
    sym += [a for a, _ in new_atoms[1:]]
    X = np.vstack([X] + [p[None] for _, p in new_atoms[1:]])
    mapping = {k: k for k in range(len(symbols))}
    return sym, X, [f"{symbols[index]}{index + 1} → {GROUP_LABELS.get(group, group)}"], mapping


def min_contact(symbols, xyz, new_idx):
    """新加入原子与其它原子 (非成键) 的最短距离，用于提示空间冲突。"""
    X = np.asarray(xyz)
    best = np.inf
    for i in new_idx:
        bonded = set(neighbors(symbols, X, i)) | {i}
        d = np.linalg.norm(X - X[i], axis=1)
        mask = np.array([k not in bonded for k in range(len(X))])
        if mask.any():
            best = min(best, d[mask].min())
    return best


# ----------------------------------------------------------------------------------------------
# 晶体：对称等价位置
# ----------------------------------------------------------------------------------------------
def mapping_ops(crys, mols, ref, tol=0.15):
    """找出把参考分子映射到各分子的对称操作 (含晶格平移)。返回 {分子序号: [(R, t, 平移, 原子置换)]}。"""
    lat = np.asarray(crys["lattice"], float)
    inv = np.linalg.inv(lat)
    ops = crys.get("symops") or [(np.eye(3).tolist(), [0, 0, 0])]
    fr = np.asarray(mols[ref]["cart"]) @ inv
    sref = mols[ref]["symbols"]
    out = {}
    for m, mol in enumerate(mols):
        if sorted(mol["symbols"]) != sorted(sref):
            continue
        fm = np.asarray(mol["cart"]) @ inv
        for R, t in ops:
            R, t = np.asarray(R, float), np.asarray(t, float)
            img = fr @ R.T + t
            shift = np.round(fm.mean(0) - img.mean(0))
            img = img + shift
            d = np.linalg.norm((img[:, None] - fm[None]) @ lat, axis=-1)
            perm = d.argmin(1)
            if d[np.arange(len(fr)), perm].max() < tol and len(set(perm.tolist())) == len(perm) and \
                    all(sref[k] == mol["symbols"][perm[k]] for k in range(len(perm))):
                out.setdefault(m, []).append((R, t, shift, perm))
    return out


def equivalent_sites(crys, mols, ref, atoms, tol=0.15):
    """参考分子 ref 中原子 atoms 在各分子上的对称像 {分子序号: [原子序号]} (含分子自身位于对称元素上时的分子内等价原子)。"""
    return {m: sorted({int(perm[a]) for (_, _, _, perm) in lst for a in atoms})
            for m, lst in mapping_ops(crys, mols, ref, tol).items()}


def _op_cart(lat, R, t, shift, X):
    inv = np.linalg.inv(lat)
    return ((np.asarray(X) @ inv) @ R.T + t + shift) @ lat


def substitute_crystal(crys, mols, ref, atoms, new=None, group=None, symmetric=True, adjust_bond=True, fix_h=True):
    """在晶体中取代。symmetric=True：先在参考分子上取代 (分子自身的位点对称像同时取代)，再用对称操作
    复制到全部等价分子 — 结果严格保持晶体对称性；否则只改参考分子 (结果为 P1)。返回 (新 crys, 新 mols, 说明)。"""
    lat = np.asarray(crys["lattice"], float)
    sym, X = list(mols[ref]["symbols"]), np.asarray(mols[ref]["cart"], float)
    ops = mapping_ops(crys, mols, ref) if symmetric else {ref: [(np.eye(3), np.zeros(3), np.zeros(3),
                                                                np.arange(len(sym)))]}
    site_ops = [o for o in ops.get(ref, []) if not np.allclose(o[3], np.arange(len(sym)))]
    targets = sorted({int(a) for a in atoms})
    notes = []
    if group:
        n0 = len(sym)
        done = set()
        for a in targets:
            if a in done:
                continue
            k0 = len(sym)
            sym, X, nt, _ = attach_group(sym, X, a, group)
            notes += nt
            done.add(a)
            g_pos = np.vstack([X[a][None], X[k0:]])
            g_sym = [sym[a]] + sym[k0:]
            for R, t, sh, perm in site_ops:                     # 分子内位点对称像：由对称操作生成，严格对称
                b = int(perm[a])
                if b in done or b >= n0:
                    continue
                img = _op_cart(lat, R, t, sh, g_pos)
                sym[b], X[b] = g_sym[0], img[0]
                sym += g_sym[1:]
                X = np.vstack([X, img[1:]])
                done.add(b)
                notes.append(f"  位点对称等价位置 {mols[ref]['symbols'][b]}{b + 1} 同时取代")
    else:
        allsites = set(targets)
        for R, t, sh, perm in site_ops:
            allsites.update(int(perm[a]) for a in targets)
        if len(allsites) > len(targets):
            notes.append(f"  分子位于对称元素上：其内部等价位置 {sorted(k + 1 for k in allsites - set(targets))} 同时取代")
        sym, X, nt, _ = substitute(sym, X, sorted(allsites), new, adjust_bond, fix_h)
        notes = nt + notes
    new_ref = {"symbols": sym, "cart": X}
    new_mols = []
    for m, mol in enumerate(mols):
        if m == ref:
            new_mols.append({**mol, "symbols": sym, "cart": X, "centroid": X.mean(0)})
        elif m in ops and symmetric:
            R, t, sh, _ = ops[m][0]
            Xm = _op_cart(lat, R, t, sh, new_ref["cart"])
            new_mols.append({**mol, "symbols": list(sym), "cart": Xm, "centroid": Xm.mean(0)})
        else:
            new_mols.append(dict(mol))
    changed = sum(1 for m in range(len(mols)) if m == ref or (symmetric and m in ops))
    notes.append(f"共修改 {changed}/{len(mols)} 个分子" + ("" if symmetric else "（仅所选分子，晶体对称性降低为 P1）"))
    if symmetric and changed < len(mols):
        notes.append("  有分子与参考分子不对称等价 (Z' > 1)，未修改；如需修改请以该分子为参考再取代一次")
    allsym = [s_ for m in new_mols for s_ in m["symbols"]]
    allx = np.vstack([np.asarray(m["cart"]) for m in new_mols])
    new_crys = dict(crys, symbols=allsym, frac=(allx @ np.linalg.inv(lat)) % 1.0)
    if not symmetric:
        new_crys["symops"] = [(np.eye(3).tolist(), [0.0, 0.0, 0.0])]
        new_crys["n_symops"] = 1
    return new_crys, new_mols, notes


def write_cif_p1(path, crys, mols, title="substituted"):
    """以 P1 写出 (完整分子坐标)。"""
    from .crystal import cell_matrix  # noqa: F401
    a, b, c, al, be, ga = crys["cell"]
    lat = np.asarray(crys["lattice"], float)
    inv = np.linalg.inv(lat)
    lines = [f"data_{title}", f"_cell_length_a {a}", f"_cell_length_b {b}", f"_cell_length_c {c}",
             f"_cell_angle_alpha {al}", f"_cell_angle_beta {be}", f"_cell_angle_gamma {ga}",
             "_symmetry_space_group_name_H-M 'P 1'", "loop_", "_symmetry_equiv_pos_as_xyz", "'x, y, z'", "loop_",
             "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    n = 0
    for m in mols:
        for s, x in zip(m["symbols"], np.asarray(m["cart"]) @ inv):
            n += 1
            lines.append(f"{s}{n} {s} {x[0] % 1:.6f} {x[1] % 1:.6f} {x[2] % 1:.6f}")
    from pathlib import Path
    Path(path).write_text("\n".join(lines) + "\n")


def crystal_min_contact(crys, mols, vdw_ratio=0.75):
    """周期性条件下分子间最短原子距离及其与范德华半径之和的比值。返回 (d_min, ratio_min, 描述)。"""
    from scipy.spatial import cKDTree
    VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
           "B": 1.92, "Si": 2.10, "P": 1.80, "Se": 1.90}
    lat = np.asarray(crys["lattice"], float)
    S, Xs, owner = [], [], []
    for k, m in enumerate(mols):
        S += list(m["symbols"])
        Xs.append(np.asarray(m["cart"]))
        owner += [k] * len(m["symbols"])
    X = np.vstack(Xs)
    owner = np.array(owner)
    shifts = np.array([[i, j, kk] for i in (-1, 0, 1) for j in (-1, 0, 1) for kk in (-1, 0, 1)]) @ lat
    big = np.vstack([X + s for s in shifts])
    big_owner = np.concatenate([owner * 27 + n for n in range(27)])
    home = np.concatenate([owner * 27 + 13])                        # 位移 (0,0,0) 的序号为 13
    tree = cKDTree(big)
    best = (np.inf, np.inf, "")
    for i, p in enumerate(X):
        for j in tree.query_ball_point(p, 4.0):
            if big_owner[j] == home[i]:
                continue
            d = np.linalg.norm(big[j] - p)
            sj = S[j % len(S)]
            r = d / (VDW.get(S[i], 1.7) + VDW.get(sj, 1.7))
            if r < best[1]:
                best = (d, r, f"{S[i]}…{sj} {d:.2f} Å")
    return best
