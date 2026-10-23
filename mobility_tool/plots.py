"""出版用插图 (粗边框、外向刻度、(a)(b) 标签、300 dpi)。
无需 pandas / seaborn / Origin，数据直接来自本程序的分解结果。"""
import logging

import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)    # 缺少的候选字体不必逐个提示

PRIMARY = "#2B4A77"
# 近似 seaborn.diverging_palette(240, 40, s=70, l=45, center="light")
DIVERGING = LinearSegmentedColormap.from_list("steady", ["#2F5F9E", "#7FA3CF", "#F2F2F2", "#D2A26E", "#9C6526"])

# 字体按 font.family 列表逐字回退：拉丁字符用 Arial，中文自动改用雅黑 / 黑体 / Noto
STYLE = {"font.family": ["Arial", "Helvetica", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
         "font.size": 16, "axes.linewidth": 2.0, "axes.unicode_minus": False}


def _style_axes(ax):
    ax.tick_params(axis="both", which="major", labelsize=16, length=8, width=2.0, direction="out")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", length=4, width=1.5, direction="out")
    for sp in ax.spines.values():
        sp.set_linewidth(2.0)


def _label(ax, text):
    ax.text(-0.15, 1.05, text, transform=ax.transAxes, fontsize=24, fontweight="bold", va="top", ha="right")


def draw_hr_bars(ax, freqs, hr, threshold=1e-4, width=15):
    f, s = np.asarray(freqs, float), np.asarray(hr, float)
    keep = s > threshold
    ax.bar(f[keep], s[keep], width=width, color=PRIMARY, align="center")
    ax.set_xlabel("Frequency (cm$^{-1}$)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Huang-Rhys Factor", fontsize=20, fontweight="bold")
    if keep.any():
        ax.set_xlim(0, f[keep].max() * 1.05)
        ax.set_ylim(0, s[keep].max() * 1.05)
    _style_axes(ax)


def draw_duschinsky(ax, matrix, fig=None, colorbar=True, tick_step=None,
                    xlabel="Initial State Normal Modes", ylabel="Final State Normal Modes"):
    m = np.asarray(matrix, float)
    amax = np.abs(m).max()
    vlim = round(amax, 1) if amax < 0.9 else 1.0
    vlim = vlim or amax or 1.0
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    n0, n1 = m.shape
    im = ax.imshow(m, cmap=DIVERGING, norm=norm, interpolation="nearest", origin="upper",
                   extent=(0.5, n1 + 0.5, n0 + 0.5, 0.5))
    step = tick_step or (50 if max(m.shape) > 120 else 20 if max(m.shape) > 40 else 5)
    ax.set_xticks(np.arange(step, n1 + 1, step))
    ax.set_yticks(np.arange(step, n0 + 1, step))
    ax.set_xlabel(xlabel, fontsize=20, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=20, fontweight="bold")
    _style_axes(ax)
    if colorbar and fig is not None:
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.2)
        cb = fig.colorbar(im, cax=cax, orientation="vertical")
        cb.set_label("Matrix Element ($J_{ij}$)", size=18, fontweight="bold")
        cb.ax.tick_params(labelsize=14)
    return im


def figure_hr_duschinsky(freqs, hr, matrix, path=None, figsize=(16, 7), dpi=300, threshold=1e-4):
    """(a) 黄昆因子柱状图 + (b) Duschinsky 矩阵热图。"""
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=figsize, dpi=dpi)
        ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"wspace": 0.3})
        draw_hr_bars(ax1, freqs, hr, threshold)
        _label(ax1, "(a)")
        if matrix is not None:
            draw_duschinsky(ax2, matrix, fig)
            _label(ax2, "(b)")
        else:
            ax2.set_visible(False)
        if path:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def figure_reorg_decomposition(freqs, lam_meV, path=None, fwhm=None, title=None, figsize=(9, 6), dpi=300):
    """振动模式对重组能的贡献（棒状图，可加 Lorentz 展宽曲线）。"""
    f, lam = np.asarray(freqs, float), np.asarray(lam_meV, float)
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=figsize, dpi=dpi)
        ax = fig.subplots()
        ax.vlines(f, 0, lam, color=PRIMARY, lw=2.5)
        ax.set_xlabel("Frequency (cm$^{-1}$)", fontsize=20, fontweight="bold")
        ax.set_ylabel("$\\lambda_i$ (meV)", fontsize=20, fontweight="bold")
        ax.set_ylim(0, max(lam.max() * 1.1, 1e-6))
        ax.set_xlim(0, f.max() * 1.05 + 50)
        _style_axes(ax)
        if fwhm:
            grid = np.linspace(0, f.max() * 1.05 + 50, 3000)
            g = fwhm / 2
            spec = (lam[None] * (g / np.pi) / ((grid[:, None] - f[None]) ** 2 + g ** 2)).sum(1)
            ax2 = ax.twinx()
            ax2.plot(grid, spec, color="#C0392B", lw=1.8)
            ax2.set_ylabel("Broadened (meV$\\cdot$cm)", fontsize=16, color="#C0392B")
            ax2.set_ylim(0, spec.max() * 1.1)
        ax.set_title(title or f"$\\Sigma\\lambda_i$ = {lam.sum():.1f} meV", fontsize=18)
        if path:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def draw_energy_levels(ax, ipea, orbitals):
    """能级示意：HOMO/LUMO 与 -IP / -EA。"""
    items = []
    if orbitals:
        items += [("HOMO", orbitals["HOMO_eV"], "#2B4A77"), ("LUMO", orbitals["LUMO_eV"], "#C0392B")]
    for key, lab, col in (("IP_v", "-IP(v)", "#1F77B4"), ("IP_a", "-IP(a)", "#6BAED6"),
                          ("EA_v", "-EA(v)", "#D62728"), ("EA_a", "-EA(a)", "#FC9272")):
        if key in ipea:
            items.append((lab, -ipea[key]["eV"], col))
    for i, (lab, e, col) in enumerate(items):
        x0 = 0 if lab in ("HOMO", "LUMO") else 1.4
        ax.hlines(e, x0, x0 + 1, color=col, lw=3)
        ax.text(x0 + 1.05, e, f"{lab} {e:.2f}", va="center", fontsize=10)
    ax.set_xlim(-0.2, 3.6)
    ax.set_xticks([0.5, 1.9])
    ax.set_xticklabels(["轨道能", "-IP / -EA"])
    ax.set_ylabel("能量 / eV")


# ----------------------------------------------------------------------------------------------
# 其它插图
# ----------------------------------------------------------------------------------------------
E_COLOR, H_COLOR = "#2B5DA8", "#B22222"


def draw_angular(ax, phi, mu_e=None, mu_h=None, lam_e=None, lam_h=None, title=True, legend=True):
    """电子 (蓝) / 空穴 (红) 角分辨迁移率极坐标图，0° 在上方、顺时针。"""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for mu, col, lab, lam in ((mu_e, E_COLOR, "Electron", lam_e), (mu_h, H_COLOR, "Hole", lam_h)):
        if mu is None:
            continue
        name = f"{lab} ($\\lambda_{{{lab[0].lower()}}}$ = {lam:.3f} eV)" if lam is not None else lab
        ax.plot(phi, mu, color=col, lw=1.8, label=name)
        ax.fill(phi, mu, color=col, alpha=0.18)
    ax.set_thetagrids(range(0, 360, 45), [f"{a}°" for a in range(0, 360, 45)], fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    if title:
        ax.set_title("Angular Dependent Mobility $\\mu_\\phi$ (cm$^2$V$^{-1}$s$^{-1}$)", fontsize=10, pad=14)
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, frameon=False)


def figure_angular(systems, path=None, dpi=300):
    """systems: [{label, phi, mu_e, mu_h, lam_e, lam_h}] → 一行多个极坐标子图。"""
    with matplotlib.rc_context({**STYLE, "font.size": 10, "axes.linewidth": 1.0}):
        n = len(systems)
        fig = Figure(figsize=(3.6 * n, 4.4), dpi=dpi)
        for i, s in enumerate(systems):
            ax = fig.add_subplot(1, n, i + 1, projection="polar")
            draw_angular(ax, s["phi"], s.get("mu_e"), s.get("mu_h"), s.get("lam_e"), s.get("lam_h"))
            ax.text(0.5, -0.32, s.get("label", ""), transform=ax.transAxes, ha="center", fontsize=13)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def figure_three_panel(freqs, hr, matrix, lam_ev, path=None, fmax=None, dpi=300, label=None):
    """三联图：(a) 黄昆因子 (b) Duschinsky 矩阵 (c) 各模式重组能 (eV)。"""
    f = np.asarray(freqs, float)
    with matplotlib.rc_context({**STYLE, "font.size": 12}):
        fig = Figure(figsize=(18, 5.6), dpi=dpi)
        axs = fig.subplots(1, 3, gridspec_kw={"wspace": 0.42})
        draw_hr_bars(axs[0], f, hr)
        _label(axs[0], "(a)")
        if matrix is not None:
            draw_duschinsky(axs[1], matrix, fig)
            _label(axs[1], "(b)")
        else:
            axs[1].set_visible(False)
        lam = np.asarray(lam_ev, float)
        fm = fmax or min(f.max() * 1.05, 1700)
        keep = f <= fm
        axs[2].bar(f[keep], lam[keep], width=max(fm / 300, 3), color=PRIMARY)
        axs[2].set_xlim(0, fm)
        axs[2].set_xlabel("Frequency (cm$^{-1}$)", fontsize=20, fontweight="bold")
        axs[2].set_ylabel("Reorganization Energy (eV)", fontsize=20, fontweight="bold")
        _style_axes(axs[2])
        _label(axs[2], "(c)")
        if label:
            fig.suptitle(label, fontsize=22, fontweight="bold", y=0.02)
        if path:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def draw_overlay(ax, symbols, ref_xyz, mov_xyz, rmsd=None, colors=("#2B6CB0", "#E07B39")):
    """两个几何（已叠合）的球棍叠加图。"""
    from .viewer import bonds_for
    ax.cla()
    for xyz, col in ((ref_xyz, colors[0]), (mov_xyz, colors[1])):
        x = np.asarray(xyz)
        for i, j in bonds_for(symbols, x):
            ax.plot(*zip(x[i], x[j]), color=col, lw=1.8)
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=[12 if s == "H" else 28 for s in symbols], color=col)
    allx = np.vstack([ref_xyz, mov_xyz])
    c, h = allx.mean(0), (allx.max(0) - allx.min(0)).max() / 2 + 0.5
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_zlim(c[2] - h, c[2] + h)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if rmsd is not None:
        ax.set_title(f"rmsd={rmsd:.3f}Å", y=-0.05, fontsize=12)


def draw_spectra(ax, curves, xlabel="Wavelength (nm)"):
    """吸收 / 发射光谱：curves = [{label, x, y, kind('abs'|'emi'), color}]，吸收实线 / 发射虚线，双 y 轴。"""
    ax2 = ax.twinx()
    for c in curves:
        target = ax if c.get("kind", "abs") == "abs" else ax2
        target.plot(c["x"], c["y"], ls="-" if c.get("kind", "abs") == "abs" else "--", lw=1.8,
                    color=c.get("color"), label=c["label"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized Absorbance (S$_0$→S$_n$)")
    ax2.set_ylabel("Normalized Phosphorescence (T$_1$→S$_0$)")
    ax.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=max(1, (len(l1) + len(l2)) // 2), loc="upper center",
                  bbox_to_anchor=(0.5, 1.16), frameon=False)
    return ax2


# ----------------------------------------------------------------------------------------------
# 三维各向异性迁移率
# ----------------------------------------------------------------------------------------------
def _carrier_cmap(color):
    return LinearSegmentedColormap.from_list("carrier", ["#EEF1F5", color])


def _plane_vectors(plane):
    if plane is None:
        return None
    if isinstance(plane, str):
        axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
        return [np.array(axes[c], float) for c in plane]
    u = np.asarray(plane[0], float)
    u = u / np.linalg.norm(u)
    v = np.asarray(plane[1], float)
    v = v - (v @ u) * u
    return [u, v / np.linalg.norm(v)]


def view_azimuth(res, oblique=20.0):
    """视线垂直于最大迁移率方向 (再略斜一点)，使快轴在图中横向展开、不被透视缩短。"""
    d = np.asarray(res["dir_max"], float)
    if np.hypot(d[0], d[1]) < 0.3:                 # 快轴接近 z 时任意方位角均可
        return -55.0
    return float(np.degrees(np.arctan2(d[1], d[0])) + 90.0 - oblique)


def draw_mobility_3d(ax, res, color=H_COLOR, lattice=None, plane=None, title=None, n=61, colorbar=True,
                     elev=22, azim=None, alpha=0.5):
    """三维迁移率曲面：半径 = μ(n̂)，颜色 = μ；虚线为主轴，灰色箭头为晶轴 a、b、c，实线为传输平面内的截面
    (与二维角分辨图相同)。"""
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from .mobility import mobility_on_sphere
    nv, mu = mobility_on_sphere(res["M"], n, 2 * n - 1)
    X, Y, Z = (mu * nv[..., k] for k in range(3))
    cmap = _carrier_cmap(color)
    norm = Normalize(0, res["mu_max"])
    fc = cmap(norm(mu))
    fc[..., 3] = alpha
    ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.plot_wireframe(X, Y, Z, rstride=max(1, n // 10), cstride=max(1, n // 6), color=color, linewidth=0.25,
                      alpha=0.35)
    lim = 1.3 * res["mu_max"]
    for k, (val, d) in enumerate(zip(res["principal"], res["axes"])):
        ax.plot(*np.array([-val * d, val * d]).T, "--", color="#333333", lw=0.9)
    if lattice is not None:
        for lab, a in zip("abc", np.asarray(lattice, float)):
            e = a / np.linalg.norm(a) * 1.18 * res["mu_max"]
            ax.quiver(0, 0, 0, *e, color="#666666", lw=1.0, arrow_length_ratio=0.08)
            ax.text(*(e * 1.08), lab, color="#444444", fontsize=10, fontstyle="italic")
    pv = _plane_vectors(plane)
    if pv is not None:
        t = np.linspace(0, 2 * np.pi, 361)
        d = np.outer(np.cos(t), pv[0]) + np.outer(np.sin(t), pv[1])
        r = np.einsum("ia,ab,ib->i", d, res["M"], d)
        ax.plot(*(d * r[:, None]).T, color=color, lw=2.2)
    ax.view_init(elev=elev, azim=view_azimuth(res) if azim is None else azim)
    for setlim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x", labelpad=-6)
    ax.set_ylabel("y", labelpad=-6)
    ax.set_zlabel("z", labelpad=-6)
    ax.tick_params(labelsize=7, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
    if title:
        ax.set_title(title, fontsize=10)
    if colorbar:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, label="μ (cm$^2$V$^{-1}$s$^{-1}$)")


def figure_mobility_3d(systems, path=None, lattice=None, plane=None, dpi=300, elev=22, azim=None):
    """systems: [{label, res, color}] → 一行多个三维迁移率曲面。"""
    with matplotlib.rc_context({**STYLE, "font.size": 10, "axes.linewidth": 1.0}):
        nsys = len(systems)
        if azim is None:                           # 各子图共用一个视角，便于对比
            azim = view_azimuth(max(systems, key=lambda x: x["res"]["mu_max"])["res"])
        fig = Figure(figsize=(4.8 * nsys, 4.6), dpi=dpi)
        for i, s_ in enumerate(systems):
            ax = fig.add_subplot(1, nsys, i + 1, projection="3d")
            draw_mobility_3d(ax, s_["res"], s_.get("color", H_COLOR), lattice, plane, s_.get("label"), elev=elev,
                             azim=azim)
        fig.subplots_adjust(left=0.0, right=0.95, wspace=0.12)
        if path:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def mobility_3d_html(path, systems, lattice=None, plane=None, title="三维各向异性迁移率", n=61):
    """可在浏览器中旋转的三维迁移率图 (plotly.js，需联网加载脚本)。systems: [{label, res, color}]。"""
    import json
    from .mobility import mobility_on_sphere
    traces, layout = [], {"title": {"text": title, "x": 0.02}, "showlegend": False,
                          "margin": {"l": 0, "r": 0, "t": 50, "b": 0}, "paper_bgcolor": "#FFFFFF"}
    k = len(systems)
    pv = _plane_vectors(plane)
    for i, s_ in enumerate(systems):
        scene = "scene" if i == 0 else f"scene{i + 1}"
        res, col = s_["res"], s_.get("color", H_COLOR)
        nv, mu = mobility_on_sphere(res["M"], n, 2 * n - 1)
        X, Y, Z = (mu * nv[..., c] for c in range(3))
        traces.append({"type": "surface", "x": X.round(5).tolist(), "y": Y.round(5).tolist(), "z": Z.round(5).tolist(),
                       "surfacecolor": mu.round(5).tolist(), "colorscale": [[0, "#EEF1F5"], [1, col]], "cmin": 0,
                       "cmax": res["mu_max"], "opacity": 0.9, "scene": scene,
                       "colorbar": {"title": {"text": "μ (cm²/Vs)"}, "x": (i + 1) / k - 0.02, "len": 0.6},
                       "hovertemplate": "μ = %{surfacecolor:.3g} cm²/Vs<extra></extra>", "name": s_.get("label", "")})
        for val, d in zip(res["principal"], res["axes"]):
            seg = np.array([-val * d, val * d])
            traces.append({"type": "scatter3d", "mode": "lines", "x": seg[:, 0].tolist(), "y": seg[:, 1].tolist(),
                           "z": seg[:, 2].tolist(), "line": {"color": "#333333", "dash": "dash", "width": 3},
                           "scene": scene, "hovertemplate": f"主轴 μ = {val:.3g}<extra></extra>"})
        if lattice is not None:
            for lab, a in zip("abc", np.asarray(lattice, float)):
                e = a / np.linalg.norm(a) * 1.18 * res["mu_max"]
                traces.append({"type": "scatter3d", "mode": "lines+text", "x": [0, e[0]], "y": [0, e[1]], "z": [0, e[2]],
                               "text": ["", lab], "line": {"color": "#777777", "width": 4},
                               "textfont": {"size": 15, "color": "#444444"}, "scene": scene, "hoverinfo": "skip"})
        if pv is not None:
            t = np.linspace(0, 2 * np.pi, 361)
            dd = np.outer(np.cos(t), pv[0]) + np.outer(np.sin(t), pv[1])
            r = np.einsum("ia,ab,ib->i", dd, res["M"], dd)
            P = dd * r[:, None]
            traces.append({"type": "scatter3d", "mode": "lines", "x": P[:, 0].round(5).tolist(),
                           "y": P[:, 1].round(5).tolist(), "z": P[:, 2].round(5).tolist(),
                           "line": {"color": col, "width": 6}, "scene": scene, "hoverinfo": "skip"})
        lim = 1.3 * res["mu_max"]
        ax_ = {"range": [-lim, lim], "showbackground": False, "zeroline": False}
        layout[scene] = {"domain": {"x": [i / k, (i + 1) / k], "y": [0, 1]}, "aspectmode": "cube",
                         "xaxis": dict(ax_, title="x"), "yaxis": dict(ax_, title="y"), "zaxis": dict(ax_, title="z"),
                         "annotations": []}
        layout.setdefault("annotations", []).append(
            {"text": f"{s_.get('label', '')}  μmax = {res['mu_max']:.3g} cm²/Vs", "x": (i + 0.5) / k, "y": 0.98,
             "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 14}})
    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>html,body{{margin:0;height:100%;background:#fff;font-family:"Microsoft YaHei",Arial,sans-serif}}
#p{{width:100%;height:100vh}}</style></head>
<body><div id="p"></div>
<script>Plotly.newPlot("p", {json.dumps(traces)}, {json.dumps(layout)}, {{responsive: true, displaylogo: false}});</script>
</body></html>
"""
    from pathlib import Path as _P
    _P(path).write_text(html, encoding="utf-8")
    return path
