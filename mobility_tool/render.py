"""分子 / 晶体渲染器（MedeA / VESTA 风格）。

不使用 matplotlib 的 3D 坐标轴，而是自行投影：球体由多层圆盘叠出明暗与高光，键为两色圆柱并带高光条，
所有图元按深度排序后放进一个 PolyCollection（画家算法），支持透视、多种显示样式、背景与 a/b/c 坐标指示。
鼠标交互由调用方连接：rotate(dx, dy)、zoom(f)、pan(dx, dy)、pick(x, y)。
"""
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

from .gaussian_io import COVALENT_RADII

JMOL = {"H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "B": "#FFB5B5", "C": "#909090", "N": "#3050F8",
        "O": "#FF0D0D", "F": "#90E050", "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0",
        "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F", "K": "#8F40D4", "Fe": "#E06633", "Co": "#F090A0",
        "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0", "Se": "#FFA100", "Br": "#A62929", "Ru": "#248F8F",
        "Rh": "#0A7D8C", "Pd": "#006985", "Ag": "#C0C0C0", "I": "#940094", "Os": "#266696", "Ir": "#175487",
        "Pt": "#D0D0E0", "Au": "#FFD123", "Eu": "#61FFC7", "Tb": "#30FFC7"}
VDW = {"H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Si": 2.10, "P": 1.80, "S": 1.80,
       "Cl": 1.75, "Se": 1.90, "Br": 1.85, "I": 1.98, "Ir": 2.00, "Pt": 1.75, "Cu": 1.40, "Zn": 1.39, "Au": 1.66}
STYLES = {"球棍": "ballstick", "棍状": "stick", "空间填充": "spacefill", "线框": "wireframe"}
BACKGROUNDS = {"深蓝渐变 (MedeA 风格)": ("#091428", "#2E4468"), "白色": ("#FFFFFF", "#FFFFFF"),
               "浅灰渐变": ("#FFFFFF", "#D5DBE5"), "黑色": ("#000000", "#000000")}
_CIRC = np.stack([np.cos(np.linspace(0, 2 * np.pi, 20, endpoint=False)),
                  np.sin(np.linspace(0, 2 * np.pi, 20, endpoint=False))], 1)
# 球体分层: (半径比例, 向光源偏移, 颜色处理)
_LAYERS = [(1.00, 0.00, ("dark", 0.50)), (0.90, 0.06, ("mix", 0.0)), (0.62, 0.17, ("white", 0.22)),
           (0.24, 0.33, ("white", 0.80))]
_LIGHT = np.array([-0.7071, 0.7071])


def _shade(rgb, how, k, bg=None):
    """rgb 可为 (3,) 或 (N, 3)，向量化。"""
    rgb = np.asarray(rgb, float)
    if how == "dark":
        return rgb * k
    if how == "white":
        return rgb + (1 - rgb) * k
    return rgb * 0.92


def _circle(nv):
    t = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    return np.stack([np.cos(t), np.sin(t)], 1)


def _rot(axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = axis
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


class MolRenderer:
    def __init__(self, ax):
        self.ax = ax
        self.R = np.eye(3)
        self.zoom = 1.0
        self.pan = np.zeros(2)
        self.style = "ballstick"
        self.background = "深蓝渐变 (MedeA 风格)"
        self.perspective = True
        self.labels = False
        self.show_h = True
        self.symbols, self.xyz = [], np.zeros((0, 3))
        self.bonds = np.zeros((0, 2), int)
        self.colors = None
        self.lattice, self.cell_range, self.all_cells = None, None, False
        self.center, self.radius = np.zeros(3), 10.0
        self.title = ""
        self.cylinders = []          # 额外圆柱 [(p, q, 半径, 颜色)]，如能量框架
        self.meshes = []             # 等值面网格 [{"verts": (N,3) Å, "faces": (M,3), "color": rgb 或 "face_rgb": (M,3), "alpha": 0.7}]
        self.annotations = []        # 文字标注 [(xyz, 文字, 颜色)]
        self.marks = []              # 测量模式选中的原子序号
        self.mark_label = ""
        self._artists = []
        self._screen = None

    # ---------------- 场景 ----------------
    def set_scene(self, symbols, xyz, colors=None, lattice=None, cell_range=None, all_cells=False, reset=False,
                  bonds=None):
        self.symbols = list(symbols)
        self.xyz = np.asarray(xyz, float).reshape(-1, 3)
        if bonds is None:
            from .viewer import bonds_for
            bonds = bonds_for(self.symbols, self.xyz)
        self.bonds = np.asarray(bonds, int).reshape(-1, 2)
        self.colors = colors
        self.lattice = None if lattice is None else np.asarray(lattice, float)
        self.cell_range, self.all_cells = cell_range, all_cells
        pts = [self.xyz] if len(self.xyz) else []
        for p, q, _ in self._cell_segments():
            pts.append(np.array([p, q]))
        allp = np.vstack(pts) if pts else np.zeros((1, 3))
        if reset or not hasattr(self, "_has_view"):
            self.center = (allp.min(0) + allp.max(0)) / 2
            self.radius = max(np.linalg.norm(allp - self.center, axis=1).max(), 3.0) + 1.5
            self.zoom, self.pan = 1.0, np.zeros(2)
            self._has_view = True

    def _cell_segments(self):
        if self.lattice is None:
            return []
        lat = self.lattice
        rng = self.cell_range or ((0, 1), (0, 1), (0, 1))
        segs = []

        def box(origin, vecs, tag):
            corners = {(i, j, k): origin + i * vecs[0] + j * vecs[1] + k * vecs[2]
                       for i in (0, 1) for j in (0, 1) for k in (0, 1)}
            for a in corners:
                for d in range(3):
                    if a[d] == 0:
                        b = list(a)
                        b[d] = 1
                        segs.append((corners[a], corners[tuple(b)], tag))
        if self.all_cells:
            for i in range(*rng[0]):
                for j in range(*rng[1]):
                    for k in range(*rng[2]):
                        box(np.array([i, j, k]) @ lat, lat, "inner")
        span = lat * np.array([r[1] - r[0] for r in rng])[:, None]
        box(np.array([r[0] for r in rng]) @ lat, span, "outer")
        return segs

    # ---------------- 相机 ----------------
    def rotate(self, dx, dy, speed=0.01):
        ang = np.hypot(dx, dy) * speed
        if ang == 0:
            return
        axis = np.array([-dy, dx, 0.0]) / np.hypot(dx, dy)     # 屏幕坐标中的转轴
        self.R = _rot(axis, ang) @ self.R

    def view_along(self, vec, up=None):
        """沿某方向观察 (vec 指向观察者)。"""
        z = np.asarray(vec, float)
        z /= np.linalg.norm(z)
        up = np.array([0, 0, 1.0]) if up is None else np.asarray(up, float)
        if abs(up @ z) > 0.95:
            up = np.array([0, 1.0, 0])
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        self.R = np.array([x, y, z])
        self.zoom, self.pan = 1.0, np.zeros(2)

    def reset(self):
        self.R = np.eye(3)
        self.zoom, self.pan = 1.0, np.zeros(2)

    def project(self, pts):
        p = (np.asarray(pts) - self.center) @ self.R.T
        if self.perspective:
            d = 6 * self.radius
            s = d / np.maximum(d - p[:, 2], 1e-3)
        else:
            s = np.ones(len(p))
        return p[:, :2] * s[:, None], p[:, 2], s

    # ---------------- 半径与颜色 ----------------
    def _radii(self):
        if self.style == "spacefill":
            return np.array([VDW.get(e, 1.8) for e in self.symbols])
        if self.style == "stick":
            return np.full(len(self.symbols), 0.16)
        if self.style == "wireframe":
            return np.full(len(self.symbols), 0.05)
        return np.array([0.26 * VDW.get(e, 1.8) + 0.04 for e in self.symbols])

    def _bond_radius(self):
        return {"ballstick": 0.11, "stick": 0.16, "spacefill": 0.0, "wireframe": 0.035}[self.style]

    def _atom_rgb(self):
        if self.colors is not None:
            return np.array([to_rgb(c) for c in self.colors])
        return np.array([to_rgb(JMOL.get(e, "#FF1493")) for e in self.symbols])

    # ---------------- 绘制 ----------------
    def draw(self, fast=False):
        ax = self.ax
        for a in self._artists:
            try:
                a.remove()
            except Exception:  # noqa: BLE001
                pass
        self._artists = []
        top, bottom = BACKGROUNDS.get(self.background, ("#FFFFFF", "#FFFFFF"))
        dark = np.mean(to_rgb(top)) < 0.5
        grad = np.linspace(0, 1, 256)[:, None, None] * (np.array(to_rgb(bottom)) - np.array(to_rgb(top))) + \
            np.array(to_rgb(top))
        self._artists.append(ax.imshow(grad, extent=(0, 1, 0, 1), transform=ax.transAxes, aspect="auto", zorder=-10,
                                       interpolation="bicubic"))
        ax.set_axis_off()
        n = len(self.symbols)
        vis = np.array([self.show_h or e != "H" for e in self.symbols], bool) if n else np.zeros(0, bool)
        scr, depth, sc = self.project(self.xyz) if n else (np.zeros((0, 2)), np.zeros(0), np.zeros(0))
        rad = self._radii() * sc if n else np.zeros(0)
        rgb = self._atom_rgb() if n else np.zeros((0, 3))
        self._screen = (scr, depth, rad, vis)
        polys, cols, keys = [], [], []
        item = 0
        big = int(vis.sum()) > 2500
        layers = _LAYERS[1:2] if fast else ([_LAYERS[0], _LAYERS[1], _LAYERS[3]] if big else _LAYERS)
        nv = 10 if fast else (14 if big else 20)
        circ = _circle(nv)
        # 原子
        idx = np.where(vis)[0]
        for li, (rf, off, (how, k)) in enumerate(layers):
            c = _shade(rgb[idx], how, k) if len(idx) else np.zeros((0, 3))
            ctr = scr[idx] + (_LIGHT * off)[None] * rad[idx, None]
            polys.append(ctr[:, None, :] + circ[None] * (rad[idx] * rf)[:, None, None])
            cols.append(np.clip(c, 0, 1))
            keys.append(np.stack([depth[idx], item + np.arange(len(idx)), np.full(len(idx), li)], 1))
        item += n
        # 键 (两半，各两层: 主体 + 高光条)
        rb = self._bond_radius()
        if rb > 0 and len(self.bonds):
            b = self.bonds[vis[self.bonds[:, 0]] & vis[self.bonds[:, 1]]]
            for half in (0, 1):
                i, j = (b[:, 0], b[:, 1]) if half == 0 else (b[:, 1], b[:, 0])
                p0, p1 = scr[i], (scr[i] + scr[j]) / 2
                d = p1 - p0
                ln = np.linalg.norm(d, axis=1)
                ok = ln > 1e-6
                i, j, p0, p1, d, ln = i[ok], j[ok], p0[ok], p1[ok], d[ok], ln[ok]
                nrm = np.stack([-d[:, 1], d[:, 0]], 1) / ln[:, None]
                w = rb * (sc[i] + sc[j]) / 2
                z = (depth[i] * 0.75 + depth[j] * 0.25) - 1e-3
                for li, (wf, sh, how, k) in enumerate([(1.0, 0.0, "dark", 0.80), (0.38, 0.32, "white", 0.35)]
                                                      [:1 if fast else 2]):
                    sgn = np.sign((nrm * _LIGHT).sum(1))
                    o = nrm * (sgn * sh * w)[:, None]
                    ww = (w * wf)[:, None]
                    quad = np.stack([p0 + o + nrm * ww, p1 + o + nrm * ww, p1 + o - nrm * ww, p0 + o - nrm * ww], 1)
                    polys.append(np.concatenate([quad, np.repeat(quad[:, -1:], nv - 4, 1)], 1))
                    cols.append(np.clip(_shade(rgb[i], how, k), 0, 1) if len(i) else np.zeros((0, 3)))
                    keys.append(np.stack([z, item + np.arange(len(i)), np.full(len(i), li)], 1))
                item += len(i)
        # 额外圆柱 (能量框架等)
        if self.cylinders:
            P = np.array([c[0] for c in self.cylinders], float)
            Q = np.array([c[1] for c in self.cylinders], float)
            Rr = np.array([c[2] for c in self.cylinders], float)
            Cc = np.array([to_rgb(c[3]) for c in self.cylinders])
            sp, zp, sps = self.project(P)
            sq, zq, sqs = self.project(Q)
            d = sq - sp
            ln = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            nrm = np.stack([-d[:, 1], d[:, 0]], 1) / ln[:, None]
            w = Rr * (sps + sqs) / 2
            sgn = np.sign((nrm * _LIGHT).sum(1))
            for li, (wf, sh, how, k) in enumerate([(1.0, 0.0, "dark", 0.85), (0.35, 0.33, "white", 0.4)][:1 if fast else 2]):
                o = nrm * (sgn * sh * w)[:, None]
                ww = (w * wf)[:, None]
                quad = np.stack([sp + o + nrm * ww, sq + o + nrm * ww, sq + o - nrm * ww, sp + o - nrm * ww], 1)
                polys.append(np.concatenate([quad, np.repeat(quad[:, -1:], nv - 4, 1)], 1))
                cols.append(np.clip(_shade(Cc, how, k), 0, 1))
                keys.append(np.stack([(zp + zq) / 2, item + np.arange(len(P)), np.full(len(P), li)], 1))
            item += len(P)
        # 等值面网格 (按面深度与原子、键一起排序；双面 Lambert 明暗)
        for mesh in self.meshes:
            V = np.asarray(mesh["verts"], float)
            Fc = np.asarray(mesh["faces"], int)
            if not len(Fc):
                continue
            sel = np.arange(len(Fc))
            if fast and len(Fc) > 4000:
                sel = sel[::max(2, len(Fc) // 4000)]
            Fc = Fc[sel]
            sv, zv, _ = self.project(V)
            cam = (V - self.center) @ self.R.T
            tri3 = cam[Fc]
            nrm = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            light = np.array([-0.35, 0.45, 0.82])
            shade = 0.38 + 0.62 * np.abs(nrm @ (light / np.linalg.norm(light)))
            base = np.asarray(mesh["face_rgb"], float)[sel] if "face_rgb" in mesh else \
                np.tile(np.array(to_rgb(mesh.get("color", "#4A90D9"))), (len(Fc), 1))
            rgb = np.clip(base * shade[:, None] + 0.12 * shade[:, None] ** 8, 0, 1)
            a = float(mesh.get("alpha", 0.75))
            tri = sv[Fc]
            polys.append(np.concatenate([tri, np.repeat(tri[:, -1:], nv - 3, 1)], 1))
            cols.append(np.hstack([rgb, np.full((len(Fc), 1), a)]))
            keys.append(np.stack([zv[Fc].mean(1), item + np.arange(len(Fc)), np.zeros(len(Fc))], 1))
            item += len(Fc)
        # 晶胞边框
        segs = self._cell_segments()
        if segs:
            line_rgb = np.array(to_rgb("#E8EEF7" if dark else "#2A2F38"))
            inner_rgb = line_rgb * 0.6 + np.array(to_rgb(bottom)) * 0.4
            P = np.array([s[0] for s in segs])
            Q = np.array([s[1] for s in segs])
            outer = np.array([s[2] == "outer" for s in segs])
            sp, zp, sp_s = self.project(P)
            sq, zq, sq_s = self.project(Q)
            d = sq - sp
            ln = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            nrm = np.stack([-d[:, 1], d[:, 0]], 1) / ln[:, None]
            w = np.where(outer, 0.045, 0.022) * (sp_s + sq_s) / 2
            quad = np.stack([sp + nrm * w[:, None], sq + nrm * w[:, None], sq - nrm * w[:, None],
                             sp - nrm * w[:, None]], 1)
            polys.append(np.concatenate([quad, np.repeat(quad[:, -1:], nv - 4, 1)], 1))
            cols.append(np.where(outer[:, None], line_rgb, inner_rgb))
            keys.append(np.stack([(zp + zq) / 2, item + np.arange(len(segs)), np.zeros(len(segs))], 1))
        if polys:
            P = np.concatenate(polys)
            C = np.concatenate([c if c.shape[1] == 4 else np.hstack([c, np.ones((len(c), 1))]) for c in cols])
            K = np.concatenate(keys)
            order = np.lexsort((K[:, 2], K[:, 1], K[:, 0]))
            pc = PolyCollection(P[order], facecolors=C[order], edgecolors=C[order], linewidths=0.3, antialiaseds=True,
                                zorder=2)                         # 描边同色，消除三角面之间的接缝
            ax.add_collection(pc)
            self._artists.append(pc)
        # 视野
        fig = ax.figure
        w_in, h_in = fig.get_size_inches()
        bb = ax.get_position()
        asp = (w_in * bb.width) / max(h_in * bb.height, 1e-6)
        L = self.radius / self.zoom
        ax.set_xlim(self.pan[0] - L * asp, self.pan[0] + L * asp)
        ax.set_ylim(self.pan[1] - L, self.pan[1] + L)
        ax.set_aspect("equal", adjustable="datalim")
        # 原子编号
        if self.labels and n and not fast:
            front = np.argsort(-depth)[:300]
            tc = "#FFFFFF" if dark else "#111111"
            for k in front:
                if vis[k]:
                    self._artists.append(ax.text(scr[k, 0], scr[k, 1], f"{self.symbols[k]}{k + 1}", fontsize=7,
                                                 color=tc, ha="center", va="center", zorder=5))
        # 文字标注 (表面极值、键级等)
        for xyz, text, color in self.annotations:
            sp_, zp_, _ = self.project(np.asarray(xyz, float)[None])
            self._artists.append(ax.scatter([sp_[0, 0]], [sp_[0, 1]], s=18, color=color, edgecolors="white",
                                            linewidths=0.6, zorder=9))
            self._artists.append(ax.text(sp_[0, 0], sp_[0, 1], " " + text, color=color, fontsize=8, zorder=10,
                                         ha="left", va="bottom", fontweight="bold"))
        # 测量标注
        if self.marks and n:
            mk = [k for k in self.marks if k < n]
            hl = "#FFD54F"
            for k in mk:
                self._artists.append(ax.add_patch(__import__("matplotlib.patches", fromlist=["Circle"]).Circle(
                    scr[k], rad[k] * 1.35 + 0.05, fill=False, ec=hl, lw=2.2, zorder=8)))
            if len(mk) > 1:
                xy = scr[mk]
                self._artists += ax.plot(xy[:, 0], xy[:, 1], ls="--", color=hl, lw=1.8, zorder=8)
                if self.mark_label:
                    mid = xy[-2:].mean(0)
                    self._artists.append(ax.text(mid[0], mid[1], " " + self.mark_label, color=hl, fontsize=11,
                                                 fontweight="bold", zorder=9, ha="left", va="bottom"))
        self._draw_triad(dark)
        if self.title:
            self._artists.append(ax.text(0.5, 0.97, self.title, transform=ax.transAxes, ha="center", va="top",
                                         fontsize=10, color="#FFFFFF" if dark else "#1F2937", zorder=6))

    def _draw_triad(self, dark):
        ax = self.ax
        o = np.array([0.075, 0.09])
        if self.lattice is not None:
            vecs, labs = self.lattice, ["a", "b", "c"]
        else:
            vecs, labs = np.eye(3), ["x", "y", "z"]
        cols = ["#E53935", "#43A047", "#1E88E5"]
        for v, lab, c in zip(vecs, labs, cols):
            u = (v / np.linalg.norm(v)) @ self.R.T
            end = o + u[:2] * 0.06
            self._artists.append(ax.annotate("", xy=end, xytext=o, xycoords="axes fraction",
                                             arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0), zorder=7))
            self._artists.append(ax.text(*(o + u[:2] * 0.078), lab, transform=ax.transAxes, color=c, fontsize=11,
                                         fontweight="bold", ha="center", va="center", zorder=7))

    # ---------------- 拾取 ----------------
    def pick(self, x, y):
        if self._screen is None or x is None:
            return None
        scr, depth, rad, vis = self._screen
        if not len(scr):
            return None
        d = np.hypot(scr[:, 0] - x, scr[:, 1] - y)
        hit = np.where((d <= np.maximum(rad, 0.25)) & vis)[0]
        if not len(hit):
            return None
        return int(hit[np.argmax(depth[hit])])

    def save(self, path, dpi=300):
        self.ax.figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
