"""基于 matplotlib 的分子 / 晶体三维显示（球棍模型、晶胞框、二聚体高亮、原子贡献着色）。"""
import itertools

import matplotlib
import numpy as np

from .gaussian_io import COVALENT_RADII

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
                                          "WenQuanYi Micro Hei", "DejaVu Sans"]

def font_chain(names):
    """只保留本机已安装的字体，末尾补 DejaVu Sans (matplotlib 自带)；用作 font.family 时逐字回退。"""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    out = [n for n in dict.fromkeys(names) if n in have]
    return out + ([] if "DejaVu Sans" in out else ["DejaVu Sans"])


matplotlib.rcParams["font.family"] = font_chain(matplotlib.rcParams["font.sans-serif"])   # ⁻ 等用 DejaVu 显示
matplotlib.rcParams["axes.unicode_minus"] = False

CPK = {"H": "#E8E8E8", "C": "#505050", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "S": "#E6C619",
       "Cl": "#1FF01F", "Br": "#A62929", "I": "#940094", "P": "#FF8000", "B": "#FFB5B5", "Si": "#F0C8A0",
       "Se": "#FFA100", "Ir": "#175487", "Pt": "#D0D0E0", "Zn": "#7D80B0", "Cu": "#C88033"}


def bonds_for(symbols, coords, scale=1.2):
    from scipy.spatial import cKDTree
    x = np.asarray(coords, float)
    if len(x) < 2:
        return []
    rad = np.array([COVALENT_RADII.get(s, 1.5) for s in symbols])
    pairs = cKDTree(x).query_pairs(2 * scale * rad.max(), output_type="ndarray")
    if len(pairs) == 0:
        return []
    d = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
    ok = (d < scale * (rad[pairs[:, 0]] + rad[pairs[:, 1]])) & (d > 0.1)
    return pairs[ok]


def _cell_edges(lattice, origin=np.zeros(3)):
    a, b, c = lattice
    corners = [origin + i * a + j * b + k * c for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    edges = []
    for p, q in itertools.combinations(range(8), 2):
        if bin(p ^ q).count("1") == 1:
            edges.append((corners[p], corners[q]))
    return edges


def draw_structure(ax, symbols, coords, lattice=None, highlight=None, atom_values=None, show_h=True,
                   labels=False, title=None, atom_size=1.0, cmap="coolwarm", cell_range=None, all_cells=False):
    """highlight: [(原子索引列表, 颜色), ...]；atom_values: 每个原子的数值（按颜色映射着色）。"""
    ax.cla()
    x = np.asarray(coords, float)
    sym = list(symbols)
    mask = np.array([show_h or s != "H" for s in sym])
    colors = [CPK.get(s, "#FF1493") for s in sym]
    alpha = np.full(len(sym), 1.0)
    if highlight:
        alpha[:] = 0.15
        for idx, col in highlight:
            for i in idx:
                colors[i] = col
                alpha[i] = 1.0
    sc_mappable = None
    if atom_values is not None:
        v = np.asarray(atom_values, float)
        vmax = np.abs(v).max() or 1.0
        cm = matplotlib.colormaps[cmap]
        colors = [cm(0.5 + 0.5 * val / vmax) for val in v]
        sc_mappable = matplotlib.cm.ScalarMappable(cmap=cm, norm=matplotlib.colors.Normalize(-vmax, vmax))

    for i, j in bonds_for(sym, x):
        if not (mask[i] and mask[j]):
            continue
        a = min(alpha[i], alpha[j])
        mid = (x[i] + x[j]) / 2
        for p, col in ((i, colors[i]), (j, colors[j])):
            seg = np.array([x[p], mid])
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=col, lw=2.5, alpha=a, solid_capstyle="round")
    size = np.array([(60 if s == "H" else 140) * atom_size for s in sym])
    idx = np.where(mask)[0]
    rgba = np.array([matplotlib.colors.to_rgba(colors[i], alpha[i]) for i in idx])
    ax.scatter(x[idx, 0], x[idx, 1], x[idx, 2], s=size[idx], c=rgba, edgecolors="k",
               linewidths=0.3, depthshade=highlight is None)
    if labels:
        for i in idx:
            ax.text(*x[i], f"{sym[i]}{i + 1}", fontsize=7)
    box_pts = []
    if lattice is not None:
        lat = np.asarray(lattice)
        rng = cell_range or ((0, 1), (0, 1), (0, 1))
        origin = np.array([r[0] for r in rng]) @ lat
        span = lat * np.array([r[1] - r[0] for r in rng])[:, None]
        if all_cells and any(r[1] - r[0] > 1 for r in rng):
            for i in range(*rng[0]):
                for j in range(*rng[1]):
                    for k in range(*rng[2]):
                        for p, q in _cell_edges(lat, np.array([i, j, k]) @ lat):
                            ax.plot(*zip(p, q), color="#9DB4D3", lw=0.6)
        for p, q in _cell_edges(span, origin):
            ax.plot(*zip(p, q), color="#1f77b4", lw=1.3)
            box_pts += [p, q]
        for vec, lab in zip(lat, "abc"):
            ax.text(*(origin + vec * 1.08), lab, color="#1f77b4", fontsize=11, weight="bold")
    pts = x[idx] if not box_pts else np.vstack([x[idx], np.array(box_pts)])
    lo, hi = pts.min(0), pts.max(0)
    ctr, half = (lo + hi) / 2, (hi - lo).max() / 2 + 1.0
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x / Å")
    ax.set_ylabel("y / Å")
    ax.set_zlabel("z / Å")
    if title:
        ax.set_title(title)
    return sc_mappable
