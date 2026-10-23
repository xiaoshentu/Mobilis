"""生成 Mobilis 图标 (PNG 多尺寸 + Windows ICO)。运行: python tools_make_icon.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Polygon
from matplotlib.colors import to_rgb
from pathlib import Path
from PIL import Image

OUT = Path(__file__).parent / "mobility_tool" / "assets"


def sphere(ax, x, y, r, color, z=3):
    c = np.array(to_rgb(color))
    layers = [(1.00, 0.00, c * 0.45), (0.93, 0.05, c * 0.80), (0.70, 0.14, c + (1 - c) * 0.15),
              (0.42, 0.25, c + (1 - c) * 0.45), (0.16, 0.38, c + (1 - c) * 0.92)]
    for rf, off, col in layers:
        ax.add_patch(Circle((x - off * r, y + off * r), r * rf, color=np.clip(col, 0, 1), zorder=z, lw=0))


def make(size=1024):
    fig = Figure(figsize=(1, 1), dpi=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_alpha(0)
    # 背景: 圆角方块 + 竖直渐变 (用剪裁的渐变图像实现)
    box = FancyBboxPatch((0.04, 0.04), 0.92, 0.92, boxstyle="round,pad=0,rounding_size=0.20", lw=0, fc="none")
    ax.add_patch(box)
    top, bot = np.array(to_rgb("#0B1F3A")), np.array(to_rgb("#3A6EA5"))
    grad = np.linspace(0, 1, 256)[:, None, None] * (bot - top) + top
    im = ax.imshow(grad, extent=(0.04, 0.96, 0.04, 0.96), origin="upper", zorder=0, interpolation="bicubic")
    im.set_clip_path(box)
    # 晶格点 (淡)
    for gx in np.linspace(0.20, 0.80, 4):
        for gy in np.linspace(0.20, 0.80, 4):
            ax.add_patch(Circle((gx, gy), 0.012, color=(1, 1, 1, 0.13), zorder=1, lw=0))
    # 两个分子 (球) 与跳跃箭头
    sphere(ax, 0.30, 0.36, 0.135, "#4FA3E0")
    sphere(ax, 0.70, 0.36, 0.135, "#F2A541")
    arr = FancyArrowPatch((0.30, 0.53), (0.68, 0.53), connectionstyle="arc3,rad=-0.6",
                          arrowstyle="-|>,head_length=0.55,head_width=0.38", mutation_scale=9, lw=2.6,
                          color="#FFFFFF", zorder=4, capstyle="round")
    ax.add_patch(arr)
    # 磷光闪光 (四角星)
    cx, cy, R, r = 0.79, 0.80, 0.085, 0.022
    pts = []
    for k in range(8):
        ang = np.pi / 2 + k * np.pi / 4
        rad = R if k % 2 == 0 else r
        pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    ax.add_patch(Polygon(pts, closed=True, color="#FFE38A", zorder=5, lw=0))
    ax.add_patch(Circle((cx, cy), 0.018, color="#FFFFFF", zorder=6, lw=0))
    return fig


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig = make(1024)
    fig.savefig(OUT / "mobilis_1024.png", dpi=1024, transparent=True)
    img = Image.open(OUT / "mobilis_1024.png").convert("RGBA")
    for n in (256, 64, 48, 40, 32):
        img.resize((n, n), Image.LANCZOS).save(OUT / f"mobilis_{n}.png")
    img.save(OUT / "mobilis.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("已生成:", sorted(p.name for p in OUT.iterdir()))
