"""图形界面：python -m mobility_tool gui

选项卡: 结构/晶体、重组能、IP/EA、重组能分解/黄昆因子、转移积分、迁移率、激发态/光谱、输运理论、相互作用能、
代理模型、波函数分析、批量扫描工作流 (含 CEDS)。
只依赖 Python 自带的 tkinter 和 matplotlib。
"""
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog as _filedialog
from tkinter import messagebox as _messagebox
from tkinter import ttk as _ttk

from . import i18n
from .i18n import tr as T


# ---------------- 界面翻译层: 包装 ttk 控件 / 对话框，使所有显示文字经过 T() ----------------
def _tr_kw(kw):
    if isinstance(kw.get("wraplength"), int):            # 按 100% 屏幕缩放设计 → 高分屏 / 大字号下同比放大
        from .theme import px as _px
        kw["wraplength"] = _px(kw["wraplength"])
    if "text" in kw:
        kw["text"] = T(kw["text"])
    if "title" in kw:
        kw["title"] = T(kw["title"])
    if "filetypes" in kw:
        kw["filetypes"] = [(T(a), b) for a, b in kw["filetypes"]]
    if "values" in kw and kw.get("values") is not None and not isinstance(kw["values"], str):
        kw["values"] = [T(v) if isinstance(v, str) else v for v in kw["values"]]
    return kw


_CACHE = {}


def _translating(cls):
    if cls in _CACHE:
        return _CACHE[cls]

    class C(cls):
        def __init__(self, *a, **kw):
            if cls.__name__ in ("Button", "Label", "Checkbutton", "Radiobutton") and isinstance(kw.get("width"), int) \
                    and kw["width"] > 0 and "text" in kw:
                kw["width"] = -kw["width"]          # 负值 = 最小宽度：译文较长时不截断 ("Browse" 而非 "Brows")
            orig = kw.get("text")
            kw = _tr_kw(kw)
            master = a[0] if a else kw.get("master")
            router = getattr(master, "__dict__", {}).get("_router")      # 只认显式设置的属性 (测试桩的 __getattr__ 不算)
            if router is not None:                  # 分区面板：分区成为标签页，其余控件归入当前分区
                master, kw = router.route(cls.__name__, kw, orig)
                if a:
                    a = (master,) + tuple(a[1:])
                else:
                    kw["master"] = master
            super().__init__(*a, **kw)

        def configure(self, *a, **kw):
            return super().configure(*a, **_tr_kw(kw))

        config = configure

        def heading(self, col, *a, **kw):
            return super().heading(col, *a, **_tr_kw(kw))

        def column(self, col, *a, **kw):
            from .theme import px as _px
            for k in ("width", "minwidth"):
                if isinstance(kw.get(k), int):
                    kw[k] = _px(kw[k])
            try:
                fn = super().column
            except AttributeError:            # 测试用的桩控件没有 column
                return None
            return fn(col, *a, **kw)

        def insert(self, *a, **kw):
            if "values" in kw:
                kw["values"] = [T(v) if isinstance(v, str) else v for v in kw["values"]]
            return super().insert(*a, **kw)

        def item(self, iid, *a, **kw):
            if "values" in kw:
                kw["values"] = [T(v) if isinstance(v, str) else v for v in kw["values"]]
            return super().item(iid, *a, **kw)

        def add(self, child, **kw):
            return super().add(child, **_tr_kw(kw))
    C.__name__ = cls.__name__
    _CACHE[cls] = C
    return C


class _TTK:
    WRAP = {"Label", "Button", "Checkbutton", "Radiobutton", "LabelFrame", "Treeview", "Notebook", "Combobox",
            "Frame", "Entry", "Spinbox", "Scrollbar", "Separator", "Progressbar", "Scale", "PanedWindow"}

    def __getattr__(self, name):
        obj = getattr(_ttk, name)
        return _translating(obj) if name in self.WRAP else obj


ttk = _TTK()


class _Dialogs:
    def __init__(self, mod):
        self.mod = mod

    def __getattr__(self, name):
        fn = getattr(self.mod, name)

        def wrapped(*a, **kw):
            a = [T(x) if isinstance(x, str) else x for x in a]
            return fn(*a, **_tr_kw(kw))
        return wrapped


filedialog = _Dialogs(_filedialog)
messagebox = _Dialogs(_messagebox)


class MappedVar(tk.StringVar):
    """下拉框: 显示翻译后的文字，程序读取到的仍是原始 (中文) 取值。"""

    def __init__(self, value="", mapping=None):
        # 显示值经过 T() 翻译 (与下拉框选项一致)；读取时译文与原文都映射回原始取值
        self._map = {k: T(v) if isinstance(v, str) else v for k, v in (mapping or {}).items()}
        self._rev = {v: k for k, v in (mapping or {}).items()}
        self._rev.update({v: k for k, v in self._map.items()})
        super().__init__(value=self._map.get(str(value), str(value)))

    def get(self):
        v = super().get()
        return self._rev.get(v, v)

    def set(self, value):
        super().set(self._map.get(str(value), str(value)))

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from . import viewer  # noqa: E402,F401  (设置中文字体)
from .adf import ADF_BASIS, ADF_FUNCTIONALS, generate_adf_inputs, read_adf_transfer_integrals  # noqa: E402
from .crystal import find_dimers, find_molecules, read_cif, split_dimer  # noqa: E402
from .decomposition import broadened_spectrum, decompose_reorg, export_decomposition_csv  # noqa: E402
from .gaussian_fcht import export_fcht, generate_fcht_inputs, read_fcht_log  # noqa: E402
from .ipea import ENERGY_KEYS, format_ip_ea, frontier_orbitals, ip_ea_from_energies  # noqa: E402
from .gaussian_io import SOLVENTS  # noqa: E402
from .orbcomp import auto_fragments, format_composition, orbital_composition, parse_ranges  # noqa: E402
from .photophys import (absorption_spectrum, analyze_excited, decay_rates, emission_band,  # noqa: E402
                        generate_excited_inputs, rmsd)
from .plots import (draw_angular, draw_duschinsky, draw_energy_levels, draw_overlay, draw_spectra,  # noqa
                    figure_angular, figure_hr_duschinsky, figure_reorg_decomposition, figure_three_panel)
from .gaussian_io import BOHR2ANG, HARTREE2EV, read_geometry, write_xyz  # noqa: E402
from .mobility import angular_mobility, angular_mobility_2d, compute_mobility, plane_angles  # noqa: E402
from .reorganization import BASIS_SETS, FUNCTIONALS, analyze_reorg, generate_reorg_inputs  # noqa: E402
from .transfer_integral import calc_transfer_integrals, generate_ti_inputs  # noqa: E402
from .theme import (FONT_SCALES, PALETTE, ScrollPanel, SectionPanel, apply_theme, enable_hidpi, set_scale, stripe,  # noqa: E402
                    styled_text, sz)
from .viewer import draw_structure  # noqa: E402
from .jobs import JobQueue, find_gaussian, path_problems  # noqa: E402
from . import __affiliation__, __author__, __version__  # noqa: E402

APP_NAME = "Mobilis"
APP_CN = "迁跃"
APP_SUBTITLE = "有机半导体电荷输运与光物理计算平台"
APP_DEVELOPER = f"{__author__}，{__affiliation__.split(' (')[0]}"

CONFIG = Path.home() / ".mobility_tool.json"


def load_config():
    try:
        return json.loads(CONFIG.read_text())
    except Exception:  # noqa: BLE001
        return {}


def save_config(cfg):
    try:
        CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        pass
from .workflow import collect_group_couplings, prepare_dimer_jobs  # noqa: E402

STRUCT_TYPES = [("结构文件", "*.xyz *.gjf *.com *.log *.out *.fchk *.cif"), ("所有文件", "*.*")]
FCHK_TYPES = [("fchk", "*.fchk *.fch"), ("所有文件", "*.*")]


# ==============================================================================================
# 通用小部件
# ==============================================================================================
_APP = None          # 主窗口 (用于最近文件 / 工作目录)


def _cfg():
    return _APP.config_data if _APP is not None else {}


def record_recent(path):
    """记录最近使用的文件 / 目录 (最多 40 个，保存在配置中)。"""
    if not path or _APP is None:
        return
    p = str(Path(path))
    rec = [x for x in _cfg().get("recent_files", []) if x != p]
    _cfg()["recent_files"] = [p] + rec[:39]
    try:
        save_config(_cfg())
    except Exception:  # noqa: BLE001
        pass


def file_status(path, mode="open"):
    """文件状态标记: ✓ 可用 / 正常结束, ✗ 不存在 / 出错, … 计算未完成。返回 (符号, 颜色, 说明)。"""
    if not path:
        return "", PALETTE["muted"], ""
    p = Path(path)
    if mode == "save":
        return "", PALETTE["muted"], ""
    if not p.exists():
        return "✗", "#C0392B", "不存在"
    if mode == "dir" or p.is_dir():
        return "✓", "#2E8B57", "目录"
    if p.suffix.lower() in (".log", ".out"):
        try:
            with open(p, "rb") as fh:
                fh.seek(max(0, p.stat().st_size - 6000))
                tail = fh.read().decode("utf-8", "ignore")
        except OSError:
            return "?", PALETTE["muted"], ""
        if "Normal termination" in tail or "normal termination" in tail or "NORMAL TERMINATION" in tail:
            return "✓", "#2E8B57", "正常结束"
        if "Error termination" in tail or "ERROR" in tail:
            return "✗", "#C0392B", "出错结束"
        return "…", "#D98E04", "未完成或不是计算输出"
    return "✓", "#2E8B57", "可用"


class FileEntry(ttk.Frame):
    """文件 / 目录选择框：下拉列出最近使用的同类文件与工作目录中的匹配文件，右侧显示文件状态。"""

    def __init__(self, parent, label, mode="open", filetypes=None, width=34, command=None):
        super().__init__(parent)
        self.var = tk.StringVar()
        self.mode, self.filetypes, self.command = mode, filetypes or [("所有文件", "*.*")], command
        ttk.Label(self, text=label, width=-12).pack(side="left")          # 负宽度 = 最小宽度，英文标签不被截断
        self.box = ttk.Combobox(self, textvariable=self.var, width=min(width, 16), postcommand=self._fill)
        self.box.pack(side="left", fill="x", expand=True)
        try:
            self.box.bind("<<ComboboxSelected>>", self._chosen)
            self.box.bind("<FocusOut>", lambda e: self._check())
        except Exception:  # noqa: BLE001
            pass
        ttk.Button(self, text="浏览", width=5, command=self.browse).pack(side="left", padx=2)
        self.state_lbl = ttk.Label(self, text="", width=2)
        self.state_lbl.pack(side="left")

    # ---- 下拉候选 ----
    def _patterns(self):
        pats = [p for _, ps in self.filetypes for p in ps.split() if p not in ("*.*", "*")]
        return pats

    def _matches(self, p):
        import fnmatch
        pats = self._patterns()
        name = Path(p).name.lower()
        return not pats or any(fnmatch.fnmatch(name, q.lower()) for q in pats)

    def candidates(self, limit=30):
        out = []
        for r in _cfg().get("recent_files", []):
            rp = Path(r)
            try:
                if self.mode == "dir":
                    if rp.is_dir():
                        out.append(str(rp))
                elif rp.is_file() and self._matches(rp):
                    out.append(str(rp))
            except OSError:              # 无权限 / 已删除的临时目录
                continue
        wd = _cfg().get("workdir")
        try:
            wd_ok = bool(wd) and Path(wd).is_dir()
        except OSError:
            wd_ok = False
        if wd_ok:
            try:
                base = Path(wd)
                if self.mode == "dir":
                    found = [q for q in base.iterdir() if q.is_dir()]
                else:
                    found = [q for q in list(base.iterdir()) + [x for d in base.iterdir() if d.is_dir()
                                                                for x in d.iterdir()] if q.is_file() and self._matches(q)]
                found.sort(key=lambda q: q.stat().st_mtime, reverse=True)
                out += [str(q) for q in found]
            except OSError:
                pass
        return list(dict.fromkeys(out))[:limit]

    def _fill(self):
        try:
            self.box.configure(values=self.candidates())
        except Exception:  # noqa: BLE001
            pass

    def _chosen(self, event=None):
        p = self.get()
        self._check()
        record_recent(p)
        if self.command and p:
            self.command(p)

    def _check(self):
        sym, col, tip = file_status(self.get(), self.mode)
        try:
            self.state_lbl.configure(text=sym, foreground=col)
            self.state_tip = tip
        except Exception:  # noqa: BLE001
            pass

    def browse(self):
        # 指定父窗口：从“运行任务”等子窗口打开时，对话框不会跑到主窗口后面
        kw = {}
        try:
            kw["parent"] = self.winfo_toplevel()
        except Exception:  # noqa: BLE001
            pass
        cur = self.get() or _cfg().get("workdir", "")
        if cur:
            c = Path(cur)
            kw["initialdir"] = str(c if c.is_dir() else c.parent)
        if self.mode == "dir":
            p = filedialog.askdirectory(**kw)
        elif self.mode == "save":
            p = filedialog.asksaveasfilename(filetypes=self.filetypes, **kw)
        else:
            p = filedialog.askopenfilename(filetypes=self.filetypes, **kw)
        if p:
            self.var.set(p)
            self._check()
            if self.mode != "save":
                record_recent(p)
            if self.command:
                self.command(p)

    def get(self):
        return self.var.get().strip()

    def set(self, v):
        self.var.set(str(v))
        self._check()


TIPS = {
    "泛函": "交换相关泛函，可附加色散校正，如 B3LYP em=gd3bj；Ir 等过渡金属配合物常用 B3LYP / PBE0",
    "优化基组": "几何优化所用基组；含 Ir、Pt 等重原子时自动对其使用赝势基组 (GenECP)",
    "重原子赝势基组": "重原子 (Z > 36) 使用的赝势基组，如 LANL2DZ、SDD、def2-TZVP",
    "单点基组": "转移积分单点计算所用基组；6-31G(d) 是常用的折中选择",
    "单点泛函(空=同上)": "重组能单点能所用泛函；留空则与优化相同",
    "单点基组(空=同上)": "重组能单点能所用基组，如用更大的 6-311G(d,p) 提高精度；留空则与优化相同",
    "溶剂 (PCM)": "隐式溶剂模型；固态薄膜中通常用气相 (留空)",
    "溶剂 (PCM, 空=气相)": "隐式溶剂模型；固态薄膜中通常用气相 (留空)",
    "截断距离 / Å": "两个分子最短原子间距小于该值才算近邻；4 Å 包含主要的 π–π 与 C–H···π 接触",
    "判据": "近邻判据：按最短原子间距 (推荐) 或质心距离",
    "电子重组能 λe / eV": "电子的内重组能 (四点法)，来自“重组能”页",
    "空穴重组能 λh / eV": "空穴的内重组能 (四点法)，来自“重组能”页",
    "外重组能 λout / eV": "周围介质极化贡献，有机晶体中通常取 0–0.1 eV",
    "温度 / K": "计算温度；Marcus 速率与迁移率均依赖温度",
    "ħ/τ_in / meV": "TLT 中的分子间振动能标 (非弹性散射率)，一般取 5–10 meV",
    "σ_J / |J|": "转移积分热涨落的相对大小；可用“动态无序”计算得到，常见 0.3–0.5",
    "超胞边长 L": "TLT 超胞边长 (格点数)；0 表示自动选择约 600 个格点",
    "无序 σ / eV": "位点能量的高斯无序宽度 (KMC)；晶体中一般 < 0.05 eV",
    "ħω / eV (Jortner)": "Marcus–Levich–Jortner 模型中的有效高频振动量子，通常 0.15–0.2 eV",
    "PLQY Φ (0–1)": "光致发光量子产率；k_r = Φ/τ，k_nr = (1 − Φ)/τ",
    "寿命 τ / μs": "激发态 (磷光) 寿命",
    "展宽 FWHM / eV": "振动分辨光谱中每个振动峰的高斯展宽",
    "能量 / eV": "垂直发射能或 0-0 能量，例如来自 T1 优化结构的 TD-DFT 结果",
    "电子耦合 C / meV (可空)": "能隙定律中的非绝热电子耦合；留空则只比较 k_nr/C² 的相对大小",
    "参考分子序号": "以第几个分子为中心搜索近邻",
    "成键判据倍数": "两原子距离 < 倍数 × (共价半径之和) 视为成键，用于拼出完整分子",
}


def field(parent, row, label, default="", width=14, values=None, col=0):
    lab = ttk.Label(parent, text=label)
    lab.grid(row=row, column=col, sticky="w", padx=2, pady=1)
    if values is not None:
        var = MappedVar(default, {str(v): T(str(v)) for v in values})
        w = ttk.Combobox(parent, textvariable=var, values=list(values), width=width)   # 显示值由包装层翻译
    else:
        var = tk.StringVar(value=str(default))
        w = ttk.Entry(parent, textvariable=var, width=width)
    w.grid(row=row, column=col + 1, sticky="we", padx=2, pady=1)
    if label in TIPS:
        from .theme import ToolTip
        ToolTip(lab, T(TIPS[label]))
        ToolTip(w, T(TIPS[label]))
    return var


class PlotPanel(ttk.Frame):
    def __init__(self, parent, projection=None, figsize=(6, 5)):
        super().__init__(parent, style="Card.TFrame", padding=6)
        self.fig = Figure(figsize=figsize, dpi=100, facecolor=PALETTE["plot_bg"])
        self.projection = projection
        self.ax = self.fig.add_subplot(111, projection=projection)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        tb = NavigationToolbar2Tk(self.canvas, self)
        tb.update()
        self.toolbar = tb
        for w in [tb] + list(tb.winfo_children() or []):
            try:
                w.configure(background=PALETTE["plot_bg"])
            except Exception:  # noqa: BLE001
                pass
        widget = self.canvas.get_tk_widget()
        try:
            widget.configure(highlightthickness=0, background=PALETTE["plot_bg"])
        except Exception:  # noqa: BLE001
            pass
        widget.pack(fill="both", expand=True)
        orig = self.canvas.draw_idle

        def draw_idle(*a, **k):
            if i18n.LANG != "zh":
                import matplotlib.text as _mt
                for txt in self.fig.findobj(_mt.Text):
                    s_ = txt.get_text()
                    if s_:
                        txt.set_text(T(s_))
            return orig(*a, **k)
        self.canvas.draw_idle = draw_idle

    def new_axes(self, projection=None):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111, projection=projection)
        return self.ax

    def draw(self):
        self.fig.tight_layout()
        self.canvas.draw_idle()


class ViewerPanel(ttk.Frame):
    """晶体 / 分子显示区：自绘渲染器 (render.MolRenderer)，无 matplotlib 坐标轴与工具栏。"""

    def __init__(self, parent, figsize=(7, 6)):
        super().__init__(parent, style="Card.TFrame", padding=0)
        from .render import MolRenderer
        self.fig = Figure(figsize=figsize, dpi=100, facecolor="#091428")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.renderer = MolRenderer(self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.toolbar = None
        w = self.canvas.get_tk_widget()
        try:
            w.configure(highlightthickness=0)
        except Exception:  # noqa: BLE001
            pass
        w.pack(fill="both", expand=True)
        self._full_job = None

    def zoom(self, factor, full=None):
        """滚轮缩放：先画快速预览，停止滚动 200 ms 后再画完整图像 (full 为完整重绘函数，默认 renderer.draw)。"""
        self.renderer.zoom *= factor
        self.renderer.draw(fast=True)
        self.canvas.draw_idle()
        if self._full_job:
            try:
                self.after_cancel(self._full_job)
            except Exception:  # noqa: BLE001
                pass

        def _full():
            self._full_job = None
            (full or self.renderer.draw)()
            self.canvas.draw_idle()
        try:
            self._full_job = self.after(200, _full)
        except Exception:  # noqa: BLE001
            _full()


class EditableTree(ttk.Treeview):
    """双击单元格可编辑的表格。"""

    def __init__(self, parent, columns, widths=None, **kw):
        super().__init__(parent, columns=columns, show="headings", **kw)
        for i, c in enumerate(columns):
            self.heading(c, text=c)
            self.column(c, width=(widths or [80] * len(columns))[i], anchor="center")
        self.bind("<Double-1>", self._edit)

    def _edit(self, event):
        item, col = self.identify_row(event.y), self.identify_column(event.x)
        if not item or not col:
            return
        box = self.bbox(item, col)
        if not box:
            return
        x, y, w, h = box
        ci = int(col[1:]) - 1
        var = tk.StringVar(value=self.item(item, "values")[ci])
        ent = ttk.Entry(self, textvariable=var)
        ent.place(x=x, y=y, width=w, height=h)
        ent.focus_set()

        def done(_=None):
            vals = list(self.item(item, "values"))
            vals[ci] = var.get()
            self.item(item, values=vals)
            ent.destroy()

        ent.bind("<Return>", done)
        ent.bind("<FocusOut>", done)

    def rows(self):
        return [list(self.item(i, "values")) for i in self.get_children()]

    def clear(self):
        self.delete(*self.get_children())


def safe(app):
    """回调装饰器：出错时弹窗并写日志。"""
    def deco(fn):
        def wrapper(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as exc:  # noqa: BLE001
                app.log(traceback.format_exc())
                messagebox.showerror("出错了", str(exc))
        return wrapper
    return deco


# ==============================================================================================
# 结构 / 晶体
# ==============================================================================================
class StructureTab(ttk.Frame):
    """结构 / 晶体查看器。鼠标：滚轮缩放，右键拖动旋转，中键平移，左键点选分子（Ctrl/Shift 多选）。"""
    SEL_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    def __init__(self, parent, app):
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.atoms, self.crys, self.mols, self.dimer_res = None, None, None, None
        self.scene, self.sel, self._busy = [], [], False
        self._bond_cache, self.scene_bonds = {}, None
        # 主窗口布局：三维结构占满主窗口，结构工具 (显示 / 构建 / 分子 / 二聚体) 作为右侧可隐藏的分区面板
        panel = SectionPanel(self, width=430)
        panel.pack(side="right", fill="y", padx=(0, 6), pady=6)
        self.panel = panel
        left = panel.inner
        self.plot = ViewerPanel(self, figsize=(7, 6))
        self.plot.pack(side="left", fill="both", expand=True, padx=(6, 4), pady=6)
        self.rend = self.plot.renderer
        S = safe(app)
        self._drag = None
        self.plot.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.plot.canvas.mpl_connect("button_press_event", S(self.on_press))
        self.plot.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.plot.canvas.mpl_connect("button_release_event", S(self.on_release))
        self.plot.canvas.mpl_connect("resize_event", self._on_resize)
        self._resize_job, self._drag_pending = None, False

        ttk.Button(left, text="打开结构文件 (xyz/gjf/log/fchk/cif)…", style="Accent.TButton",
                   command=S(self.open_file)).pack(fill="x")
        self.info = ttk.Label(left, text="未加载结构", wraplength=380, justify="left")
        self.info.pack(fill="x", pady=4)
        ttk.Label(left, text="鼠标: 左/右键拖动旋转 · 滚轮缩放 · 中键平移 · 左键单击选分子 (Ctrl/Shift 多选)",
                  foreground=PALETTE["muted"], wraplength=380).pack(fill="x")
        vf = ttk.LabelFrame(left, text="显示")
        vf.pack(fill="x", pady=2)
        from .render import BACKGROUNDS, STYLES
        self.style_var = field(vf, 0, "样式", "球棍", values=list(STYLES), width=14)
        self.bg_var = field(vf, 1, "背景", "深蓝渐变", values=list(BACKGROUNDS), width=14)
        self.persp = tk.BooleanVar(value=True)
        ttk.Checkbutton(vf, text="透视", variable=self.persp, command=S(self.redraw)).grid(row=0, column=2, sticky="w")
        self.meas_mode = tk.BooleanVar(value=False)
        self.measure_pts = []
        ttk.Checkbutton(vf, text="测量模式：依次点原子 (2 距离 / 3 键角 / 4 二面角)", variable=self.meas_mode,
                        command=S(self.toggle_measure)).grid(row=4, column=0, columnspan=3, sticky="w")
        ttk.Button(vf, text="应用", command=S(self.redraw)).grid(row=1, column=2, sticky="we")
        vb = ttk.Frame(vf)
        vb.grid(row=2, column=0, columnspan=3, sticky="we")
        ttk.Label(vb, text="观察方向").pack(side="left", padx=(2, 6))
        for txt, vec in (("a", 0), ("b", 1), ("c", 2), ("x", "x"), ("y", "y"), ("z", "z")):
            ttk.Button(vb, text=txt, width=3, command=S(lambda v=vec: self.view_along(v))).pack(side="left")
        vb2 = ttk.Frame(vf)
        vb2.grid(row=3, column=0, columnspan=3, sticky="we")
        ttk.Button(vb2, text="重置视角", command=S(lambda: self.redraw(keep_view=False))).pack(side="left", fill="x",
                                                                                        expand=True)
        ttk.Button(vb2, text="保存高清图片…", command=S(self.save_image)).pack(side="left", fill="x", expand=True)
        opt = ttk.Frame(left)
        opt.pack(fill="x", pady=2)
        self.show_h = tk.BooleanVar(value=True)
        self.show_lab = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="显示氢", variable=self.show_h, command=S(self.redraw)).pack(side="left")
        ttk.Checkbutton(opt, text="原子编号", variable=self.show_lab, command=S(self.redraw)).pack(side="left")
        self.mode = tk.StringVar(value="cell")
        ttk.Radiobutton(opt, text="晶胞", value="cell", variable=self.mode, command=S(self.rebuild)).pack(side="left")
        ttk.Radiobutton(opt, text="近邻簇", value="cluster", variable=self.mode,
                        command=S(self.rebuild)).pack(side="left")
        sf = ttk.LabelFrame(left, text="原子取代 (单分子 / 晶体中的分子)")
        sf.pack(fill="x", pady=2)
        self.sub_mode = tk.BooleanVar(value=False)
        self.sub_sel, self.sub_undo = [], []
        ttk.Checkbutton(sf, text="取代模式：点击原子选择 / 取消 (晶体中须在同一分子内)", variable=self.sub_mode,
                        command=S(self.toggle_sub)).pack(anchor="w")
        self.sub_info = ttk.Label(sf, text="", foreground=PALETTE["muted"], wraplength=400)
        self.sub_info.pack(anchor="w")
        sb1 = ttk.Frame(sf)
        sb1.pack(fill="x")
        ttk.Button(sb1, text="换成元素 (周期表)…", style="Accent.TButton", command=S(self.pick_element)
                   ).pack(side="left", fill="x", expand=True)
        from .substitution import GROUP_LABELS
        self.sub_group = MappedVar("CH3", GROUP_LABELS)
        ttk.Combobox(sb1, textvariable=self.sub_group, values=list(GROUP_LABELS), state="readonly", width=12
                     ).pack(side="left", padx=2)
        ttk.Button(sb1, text="换成基团", command=S(self.apply_group)).pack(side="left")
        so = ttk.Frame(sf)
        so.pack(fill="x")
        self.sub_bond = tk.BooleanVar(value=True)
        self.sub_h = tk.BooleanVar(value=True)
        self.sub_sym = tk.BooleanVar(value=True)
        ttk.Checkbutton(so, text="调整键长", variable=self.sub_bond).pack(side="left")
        ttk.Checkbutton(so, text="自动增减氢", variable=self.sub_h).pack(side="left")
        ttk.Checkbutton(so, text="晶体：所有对称等价位置", variable=self.sub_sym).pack(side="left")
        sb2 = ttk.Frame(sf)
        sb2.pack(fill="x")
        for txt, cmd in (("撤销", self.undo_sub), ("清除选择", self.clear_sub), ("导出结构…", self.export_sub),
                         ("送去重组能计算", self.sub_to_reorg)):
            ttk.Button(sb2, text=txt, command=S(cmd)).pack(side="left", fill="x", expand=True)
        rf = ttk.LabelFrame(left, text="显示范围 (晶胞编号，可为负)")
        rf.pack(fill="x", pady=2)
        self.rng = {}
        for row, ax_ in enumerate("abc"):
            ttk.Label(rf, text=f"{ax_}:").grid(row=row, column=0)
            v0, v1 = tk.StringVar(value="0"), tk.StringVar(value="1")
            ttk.Spinbox(rf, from_=-4, to=4, width=4, textvariable=v0, command=S(self.rebuild)).grid(row=row, column=1)
            ttk.Label(rf, text="→").grid(row=row, column=2)
            ttk.Spinbox(rf, from_=-3, to=5, width=4, textvariable=v1, command=S(self.rebuild)).grid(row=row, column=3)
            self.rng[ax_] = (v0, v1)
        qb = ttk.Frame(rf)
        qb.grid(row=0, column=4, rowspan=3, padx=6)
        for txt, r0, r1 in (("1×1×1", 0, 1), ("2×2×2", 0, 2), ("3×3×3", 0, 3), ("中心 3×3×3", -1, 2)):
            ttk.Button(qb, text=txt, command=S(lambda a=r0, b=r1: self.set_range(a, b))).pack(fill="x")
        self.all_cells = tk.BooleanVar(value=True)
        ttk.Checkbutton(rf, text="显示每个晶胞的边框", variable=self.all_cells, command=S(self.redraw)).grid(
            row=3, column=0, columnspan=4, sticky="w")

        st = ttk.LabelFrame(left, text="晶胞大小调整 (晶格缩放 / 应变)")
        st.pack(fill="x", pady=2)
        sf = ttk.Frame(st)
        sf.pack(fill="x")
        self.strain = {}
        for k, ax_ in enumerate("abc"):
            self.strain[ax_] = field(sf, 0, f"{ax_} / %", 0, width=6, col=2 * k)
        ttk.Label(st, text="分子保持刚性、随质心分数坐标平移；可配合 EHT 研究应变对 J 和迁移率的影响",
                  foreground=PALETTE["muted"], wraplength=380).pack(anchor="w")
        bb = ttk.Frame(st)
        bb.pack(fill="x")
        ttk.Button(bb, text="应用缩放", command=S(self.apply_strain)).pack(side="left", fill="x", expand=True)
        ttk.Button(bb, text="恢复原始晶胞", command=S(self.reset_strain)).pack(side="left", fill="x", expand=True)
        sf2 = ttk.Frame(st)
        sf2.pack(fill="x")
        self.scan_axis = field(sf2, 0, "扫描方向", "c", values=["a", "b", "c", "all"], width=6)
        self.scan_vals = field(sf2, 0, "范围 %", "-3:3:1", width=8, col=2)
        ttk.Button(st, text="应变扫描 (EHT → μ)", command=S(self.strain_scan)).pack(fill="x")

        mf = ttk.LabelFrame(left, text="分子列表（点选后可导出）")
        mf.pack(fill="both", expand=True, pady=4)
        cols = ("#", "标签", "化学式", "原子数")
        self.mtree = ttk.Treeview(mf, columns=cols, show="headings", height=7, selectmode="extended")
        for c, w in zip(cols, (36, 90, 150, 60)):
            self.mtree.heading(c, text=c)
            self.mtree.column(c, width=w, anchor="center")
        self.mtree.pack(fill="both", expand=True)
        self.mtree.bind("<<TreeviewSelect>>", S(self.on_tree_select))
        bf = ttk.Frame(mf)
        bf.pack(fill="x", pady=2)
        ttk.Button(bf, text="导出选中分子 xyz…", command=S(self.export_selected)).pack(side="left", fill="x", expand=True)
        ttk.Button(bf, text="选中两分子→二聚体", command=S(self.export_pair)).pack(side="left", fill="x", expand=True)
        ttk.Button(bf, text="清除选择", command=S(lambda: self.set_selection([]))).pack(side="left")

        lf = ttk.LabelFrame(left, text="晶体二聚体搜索")
        lf.pack(fill="x", pady=6)
        self.ref = field(lf, 0, "参考分子序号", 1)
        self.cut = field(lf, 1, "截断距离 / Å", 4.0)
        self.crit = field(lf, 2, "判据", "最短原子间距", values=["最短原子间距", "质心距离"])
        self.sel_mode = field(lf, 3, "选择方式", "全部近邻 (按对称等价分组)",
                              values=["全部近邻 (按对称等价分组)", "按方向 (传输平面内，Deng 2015)"], width=24)
        self.plane = field(lf, 4, "传输平面", "自动 (分子层)", values=["自动 (分子层)", "bc", "ab", "ac", "三维 (含层间)"],
                           width=24)
        ttk.Button(lf, text="搜索近邻二聚体", style="Accent.TButton", command=S(self.search)).grid(
            row=5, column=0, columnspan=2, sticky="we")
        ttk.Button(lf, text="移除选中组", command=S(self.remove_selected_groups)).grid(row=6, column=0, columnspan=2,
                                                                                    sticky="we")
        cols = ("组", "r质心/Å", "d最短/Å", "向量 (Å)")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (40, 70, 70, 170)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", S(self.on_dimer_select))
        ttk.Button(left, text="软件内计算全部二聚体 J (EHT) → 迁移率表", style="Accent.TButton",
                   command=S(self.eht_all)).pack(fill="x", pady=2)
        ttk.Button(left, text="导出等价二聚体 + 生成转移积分输入…", command=S(self.export)).pack(fill="x", pady=2)
        ttk.Button(left, text="把近邻矢量送到迁移率表", command=S(self.to_mobility)).pack(fill="x")
        ttk.Button(left, text="整个场景另存为 xyz…", command=S(self.save_xyz)).pack(fill="x", pady=2)

    # ---------------- 载入 / 场景 ----------------
    def open_file(self):
        p = filedialog.askopenfilename(filetypes=STRUCT_TYPES, initialdir=self.app.config_data.get("workdir") or None)
        if not p:
            return
        self.open_path(p)

    def open_path(self, p):
        record_recent(p)
        self.atoms = self.crys = self.mols = self.dimer_res = None
        self.tree.delete(*self.tree.get_children())
        if p.lower().endswith(".cif"):
            self.crys = read_cif(p)
            self.mols = find_molecules(self.crys)
            self.crys0, self.mols0 = self.crys, self.mols
            a, b, c, al, be, ga = self.crys["cell"]
            nat = sorted({len(m["symbols"]) for m in self.mols})
            self.info.config(text=f"{Path(p).name}\n晶胞 a={a:.3f} b={b:.3f} c={c:.3f} α={al:.1f} β={be:.1f} "
                                  f"γ={ga:.1f}\n对称操作 {self.crys['n_symops']} 个，晶胞内 {len(self.mols)} 个分子 "
                                  f"(每个 {nat} 个原子)")
            for w in self.crys["warnings"]:
                self.app.log("CIF: " + w)
            self.mode.set("cell")
        else:
            self.atoms = read_geometry(p)
            self.info.config(text=f"{Path(p).name}\n原子数 {len(self.atoms)}")
        self.app.current_structure = p
        self.app.log(f"已加载 {p}")
        try:                                            # 主窗口标题栏：名称 · 化学式 · 空间群 · 晶胞
            if self.mols is not None:
                a, b, c, al, be, ga = self.crys["cell"]
                f_ = formula(self.mols[0]["symbols"])
                self.app.set_structure_title(f"{Path(p).stem}  ·  ({f_})×{len(self.mols)}  ·  "
                                             f"{self.crys.get('spacegroup') or 'P?'}  ·  a={a:.3f} b={b:.3f} c={c:.3f} Å  "
                                             f"α={al:.1f} β={be:.1f} γ={ga:.1f}°")
            else:
                self.app.set_structure_title(f"{Path(p).stem}  ·  {formula([x[0] for x in self.atoms])}  ·  "
                                             f"{len(self.atoms)} {T('原子')}")
        except Exception:  # noqa: BLE001
            pass
        self.rebuild(keep_view=False)

    def build_scene(self):
        from .crystal import _bonded_components
        scene = []
        if self.atoms is not None:
            sym = [a[0] for a in self.atoms]
            xyz = np.array([a[1:] for a in self.atoms])
            lab = _bonded_components(sym, xyz)
            for k, l in enumerate(dict.fromkeys(lab.tolist())):
                idx = np.where(lab == l)[0]
                scene.append({"label": f"M{k + 1}", "symbols": [sym[i] for i in idx], "cart": xyz[idx], "src": idx})
        elif self.mols is not None and self.mode.get() == "cluster" and self.dimer_res:
            ra = self.dimer_res["atoms_a"]
            scene.append({"label": "参考", "symbols": [s for s, _ in ra], "cart": np.array([x for _, x in ra]),
                          "mol": int(self.ref.get()) - 1})
            for i, d in enumerate(self.dimer_res["dimers"]):
                scene.append({"label": f"{self.group_name(d['group'])}-{i + 1}", "symbols": [s for s, _ in d["atoms_b"]],
                              "cart": np.array([x for _, x in d["atoms_b"]]), "mol": d.get("mol")})
        elif self.mols is not None:
            lat = self.crys["lattice"]
            rg = self.cell_range()
            multi = any(r[1] - r[0] > 1 or r[0] != 0 for r in rg)
            for i in range(*rg[0]):
                for j in range(*rg[1]):
                    for k in range(*rg[2]):
                        shift = np.array([i, j, k]) @ lat
                        for m, mol in enumerate(self.mols):
                            tag = f"{m + 1}" + (f"[{i},{j},{k}]" if multi else "")
                            scene.append({"label": tag, "symbols": mol["symbols"], "cart": mol["cart"] + shift, "mol": m})
        return scene

    def _scene_bonds(self):
        """按分子模板缓存成键 (同一分子的平移 / 对称拷贝只检测一次)。"""
        from .viewer import bonds_for
        cache = self._bond_cache
        out, off = [], 0
        for m in self.scene:
            x = np.asarray(m["cart"])
            key = (tuple(m["symbols"]), tuple(np.round(np.sort(np.linalg.norm(x - x.mean(0), axis=1)), 2)))
            b = cache.get(key)
            if b is None:
                b = np.asarray(bonds_for(m["symbols"], x), int).reshape(-1, 2)
                cache[key] = b
            out.append(b + off)
            off += len(m["symbols"])
        self._bond_cache = cache if len(cache) < 200 else {}
        return np.vstack(out) if out else np.zeros((0, 2), int)

    def rebuild(self, keep_view=False):
        self.scene = self.build_scene()
        self.scene_bonds = self._scene_bonds()
        self.sel = []
        self._busy = True
        self.mtree.delete(*self.mtree.get_children())
        for i, m in enumerate(self.scene):
            self.mtree.insert("", "end", values=(i + 1, m["label"], formula(m["symbols"]), len(m["symbols"])))
        stripe(self.mtree)
        self._busy = False
        if self.mode.get() == "cluster" and self.dimer_res and self.mols is not None:
            self.sel = [0]
        self.redraw(keep_view=keep_view)

    # ---------------- 绘制与交互 ----------------
    def redraw(self, keep_view=True):
        if not self.scene:
            return
        from .render import BACKGROUNDS, JMOL, STYLES
        from matplotlib.colors import to_rgb
        syms, xyz, owner = [], [], []
        for i, m in enumerate(self.scene):
            syms += m["symbols"]
            xyz.append(m["cart"])
            owner += [i] * len(m["symbols"])
        self.scene_xyz, self.scene_owner, self.scene_syms = np.vstack(xyz), np.array(owner), syms
        colors = None
        if self.sel:
            top, bot = BACKGROUNDS.get(self.bg_var.get(), ("#FFFFFF", "#FFFFFF"))
            bg = (np.array(to_rgb(top)) + np.array(to_rgb(bot))) / 2
            base = np.array([to_rgb(JMOL.get(e, "#FF1493")) for e in syms])
            colors = list(base * 0.35 + bg * 0.65)
            for n, k in enumerate(self.sel):
                c = np.array(to_rgb(self.SEL_COLORS[n % len(self.SEL_COLORS)]))
                for i in np.where(self.scene_owner == k)[0]:
                    colors[i] = c * 0.75 + base[i] * 0.25 if syms[i] != "H" else c * 0.4 + 0.6
            colors = [tuple(np.clip(c, 0, 1)) for c in colors]
        lattice = self.crys["lattice"] if (self.mols is not None and self.mode.get() == "cell") else None
        r = self.rend
        r.style = STYLES.get(self.style_var.get(), "ballstick")
        r.background = self.bg_var.get()
        r.perspective = self.persp.get()
        r.show_h = self.show_h.get()
        r.labels = self.show_lab.get()
        r.title = T("选中: " + ", ".join(self.scene[k]["label"] for k in self.sel)) if self.sel else ""
        if self.scene_bonds is None or (len(self.scene_bonds) and
                                                           self.scene_bonds.max() >= len(syms)):
            self.scene_bonds = self._scene_bonds()
        r.set_scene(syms, self.scene_xyz, colors=colors, lattice=lattice,
                    cell_range=self.cell_range() if lattice is not None else None,
                    all_cells=self.all_cells.get(), reset=not keep_view, bonds=self.scene_bonds)
        if not keep_view:
            r.reset()
            if lattice is not None:
                r.view_along(lattice[2] + 0.35 * lattice[0] - 0.25 * lattice[1])
        r.draw()
        self.plot.canvas.draw_idle()

    def cell_range(self):
        out = []
        for ax_ in "abc":
            a, b = (int(float(v.get() or 0)) for v in self.rng[ax_])
            if b <= a:
                b = a + 1
            out.append((a, b))
        if np.prod([b - a for a, b in out]) * max(len(self.mols or []), 1) > 600:
            raise ValueError("显示的分子太多（>600），请缩小范围")
        return out

    def set_range(self, a, b):
        for ax_ in "abc":
            self.rng[ax_][0].set(str(a))
            self.rng[ax_][1].set(str(b))
        self.mode.set("cell")
        self.rebuild(keep_view=False)

    def apply_strain(self):
        from .crystal import strain_structure
        if self.mols is None:
            raise ValueError("请先打开 CIF 晶体文件")
        sc = [1 + float(self.strain[a].get() or 0) / 100 for a in "abc"]
        self.crys, self.mols = strain_structure(self.crys0, self.mols0, sc)
        self.dimer_res = None
        self.tree.delete(*self.tree.get_children())
        a, b, c = self.crys["cell"][:3]
        self.app.log(f"晶胞已缩放为 a={a:.3f} b={b:.3f} c={c:.3f} Å (相对原始 {', '.join(f'{(x - 1) * 100:+.2f}%' for x in sc)})")
        self.mode.set("cell")
        self.rebuild()

    def reset_strain(self):
        for a in "abc":
            self.strain[a].set("0")
        self.apply_strain()

    def eht_all(self):
        from .workflow import eht_couplings_for_crystal
        if self.mols is None:
            raise ValueError("请先打开 CIF 晶体文件")
        crit = "atom" if self.crit.get().startswith("最短") else "centroid"
        _a = (self.crys, self.mols, float(self.cut.get()), crit, int(self.ref.get()) - 1)
        _keep = self.dimer_res if (self.dimer_res and self.dimer_res.get("mode") == "paper") else None
        def _work():
            return eht_couplings_for_crystal(*_a[:4], ref=_a[4], dimer_result=_keep)

        def _done(_v):
            res, coup = _v
            self.dimer_res = res
            self.tree.delete(*self.tree.get_children())
            for d in res["dimers"]:
                v = ", ".join(f"{x:.2f}" for x in d["vector"])
                self.tree.insert("", "end", values=(f"G{d['group']}", f"{d['r_centroid']:.3f}", f"{d['d_min']:.3f}", v))
            stripe(self.tree)
            self.app.data["crystal_couplings"] = coup
            mt = self.app.mob_tab
            sc = mt._scale() / 1000.0
            mt.set_paths([{"name": f"{self.group_name(d['group'])}-{i + 1}", "vector": d["vector"],
                           "Ve": abs(coup[d["group"]]["electron"]) * sc, "Vh": abs(coup[d["group"]]["hole"]) * sc}
                          for i, d in enumerate(res["dimers"])])
            for gid, v in coup.items():
                self.app.log(f"G{gid}: r={v['r_centroid']:.3f} Å ×{v['multiplicity']}  J_h={v['hole']:.3f}  "
                             f"J_e={v['electron']:.3f} meV (EHT)")
            self.app.log("EHT 转移积分已填入迁移率表；可在“迁移率”页直接计算或运行 KMC")
            self.mode.set("cluster")
            self.rebuild(keep_view=False)
        self.app.run_bg("软件内计算全部二聚体 J (EHT)…", _work, _done)

    def strain_scan(self):
        from .workflow import strain_scan
        if self.mols is None:
            raise ValueError("请先打开 CIF 晶体文件")
        a, b, c = (float(x) for x in self.scan_vals.get().split(":"))
        vals = list(np.arange(a, b + 1e-9, c))
        mt = self.app.mob_tab
        _a = (self.crys0, self.mols0, self.scan_axis.get(), vals, float(mt.lam_e.get()), float(mt.lam_h.get()),
              float(mt._temps()[0]), float(self.cut.get()))
        def _work():
            return strain_scan(*_a)

        def _done(_v):
            rows = _v
            self.app.data["strain_scan"] = rows
            mt.show_strain_scan(rows, self.scan_axis.get())
            self.app.show_page("迁移率")
        self.app.run_bg("应变扫描 (EHT → μ)…", _work, _done)

    # ---------------- 原子取代 ----------------
    def toggle_sub(self):
        if self.sub_mode.get() and self.meas_mode.get():
            self.meas_mode.set(False)
            self.toggle_measure()
        self.clear_sub()
        self.app.log("取代模式：在三维图中点击要取代的原子 (再次点击取消)，然后选择元素或基团"
                     if self.sub_mode.get() else "已退出取代模式")

    def _scene_atom(self, k):
        """场景原子序号 → (场景分子序号, 分子内原子序号)。"""
        owner = int(self.scene_owner[k])
        start = int(np.where(self.scene_owner == owner)[0][0])
        return owner, k - start

    def toggle_sub_atom(self, k):
        owner, local = self._scene_atom(k)
        if self.sub_sel and self.atoms is None and owner != self.sub_sel[0][0]:
            raise ValueError("晶体中一次只能在同一个分子内选择原子 (其它分子的对称等价位置会自动处理)")
        key = (owner, local, int(k))
        if any(x[2] == k for x in self.sub_sel):
            self.sub_sel = [x for x in self.sub_sel if x[2] != k]
        else:
            self.sub_sel.append(key)
        self._show_sub()

    def _show_sub(self):
        names = [f"{self.scene_syms[x[2]]}{x[1] + 1}" for x in self.sub_sel]
        txt = (T("已选") + f" {len(names)}: " + ", ".join(names)) if names else ""
        if names and self.atoms is None:
            txt += f"  ({T('分子')} {self.scene[self.sub_sel[0][0]]['label']})"
        try:
            self.sub_info.config(text=txt)
        except Exception:  # noqa: BLE001
            pass
        self.rend.marks, self.rend.mark_label = [x[2] for x in self.sub_sel], ""
        self._redraw_view()

    def clear_sub(self):
        self.sub_sel = []
        self._show_sub()

    def pick_element(self):
        if not self.sub_sel:
            raise ValueError("请先打开取代模式并在三维图中点击要取代的原子")
        PeriodicTableDialog(self, lambda el: safe(self.app)(lambda: self.apply_sub(new=el))())

    def apply_group(self):
        if not self.sub_sel:
            raise ValueError("请先打开取代模式并在三维图中点击要取代的原子 (基团取代需选末端原子，如 H)")
        self.apply_sub(group=self.sub_group.get())

    def apply_sub(self, new=None, group=None):
        from .substitution import attach_group, crystal_min_contact, min_contact, substitute, substitute_crystal
        if not self.sub_sel:
            raise ValueError("没有选择原子")
        what = new or group
        if self.atoms is not None:
            sym = [a[0] for a in self.atoms]
            X = np.array([a[1:] for a in self.atoms], float)
            idx = sorted({int(self.scene[o]["src"][loc]) for o, loc, _ in self.sub_sel})
            self.sub_undo.append(("atoms", list(self.atoms)))
            n0 = len(sym)
            if group:
                notes = []
                for i in sorted(idx, reverse=True):
                    sym, X, nt, _ = attach_group(sym, X, i, group)
                    notes += nt
            else:
                sym, X, notes, _ = substitute(sym, X, idx, new, self.sub_bond.get(), self.sub_h.get())
            self.atoms = [(s_, *p) for s_, p in zip(sym, X)]
            added = list(range(n0, len(sym))) + (idx if group else [])
            close = min_contact(sym, X, [i for i in added if i < len(sym)]) if added else np.inf
            if close < 1.6:
                notes.append(f"[注意] 新原子与其它原子最近 {close:.2f} Å，可能存在空间冲突")
        else:
            m = self.scene[self.sub_sel[0][0]].get("mol")
            if m is None:
                raise ValueError("无法确定所选原子属于哪个分子")
            atoms = sorted({loc for _, loc, _ in self.sub_sel})
            before = crystal_min_contact(self.crys, self.mols)
            self.sub_undo.append(("crystal", self.crys, self.mols, self.crys0, self.mols0))
            self.crys, self.mols, notes = substitute_crystal(self.crys, self.mols, m, atoms, new=new, group=group,
                                                             symmetric=self.sub_sym.get(),
                                                             adjust_bond=self.sub_bond.get(), fix_h=self.sub_h.get())
            self.crys0, self.mols0 = self.crys, self.mols
            after = crystal_min_contact(self.crys, self.mols)
            if after[1] < min(0.75, before[1] - 0.02):
                notes.append(f"[注意] 取代后与相邻分子最近接触 {after[2]} (仅为范德华半径和的 {after[1]:.0%})："
                             "原堆积容纳不下该基团，建议进行晶体结构优化后再计算")
            self.dimer_res = None
            self.app.data.pop("crystal_couplings", None)
            self.tree.delete(*self.tree.get_children())
        self.app.log(f"原子取代 ({what})：" + "；".join(n.strip() for n in notes))
        self.app.log("取代后的结构未经优化：单分子请在“重组能”页优化；晶体为保持原堆积的刚性取代，"
                     "二聚体与转移积分需重新搜索、计算")
        self.sub_sel = []
        self.rebuild(keep_view=True)
        self._show_sub()
        self.app.data["substituted"] = True

    def undo_sub(self):
        if not self.sub_undo:
            raise ValueError("没有可撤销的取代")
        last = self.sub_undo.pop()
        if last[0] == "atoms":
            self.atoms = last[1]
        else:
            _, self.crys, self.mols, self.crys0, self.mols0 = last
            self.dimer_res = None
            self.tree.delete(*self.tree.get_children())
        self.sub_sel = []
        self.rebuild(keep_view=True)
        self.app.log("已撤销上一次取代")

    def export_sub(self, path=None):
        from .substitution import write_cif_p1
        if self.atoms is None and self.mols is None:
            raise ValueError("请先打开结构")
        p = path or filedialog.asksaveasfilename(
            defaultextension=".xyz" if self.atoms is not None else ".cif",
            filetypes=[("xyz", "*.xyz"), ("Gaussian 输入", "*.gjf"), ("CIF (P1)", "*.cif")],
            initialdir=self.app.config_data.get("workdir") or None)
        if not p:
            return None
        p = Path(p)
        if self.atoms is not None:
            atoms = self.atoms
        else:
            ref = int(self.ref.get()) - 1
            atoms = [(s_, *x) for s_, x in zip(self.mols[ref]["symbols"], self.mols[ref]["cart"])]
        if p.suffix.lower() == ".cif":
            if self.mols is None:
                raise ValueError("单分子请导出为 xyz 或 gjf")
            write_cif_p1(p, self.crys, self.mols, p.stem)
        elif p.suffix.lower() in (".gjf", ".com"):
            from .gaussian_io import write_gjf
            write_gjf(p, [{"route": "# opt freq B3LYP/6-31G(d)", "title": p.stem, "charge": 0, "mult": 1,
                           "atoms": atoms, "chk": p.stem + ".chk"}])
        else:
            write_xyz(p, atoms, "substituted structure (Mobilis)")
        record_recent(p)
        self.app.log(f"已导出 {p}" + ("" if self.atoms is not None or p.suffix.lower() == ".cif"
                                    else " (参考分子)"))
        return p

    def sub_to_reorg(self):
        """把取代后的分子 (单分子或晶体中的参考分子) 写成 xyz，填入“重组能”页 (优化后再做其它计算)。"""
        base = Path(self.app.config_data.get("workdir") or Path.home())
        p = self.export_sub(base / "substituted_molecule.xyz")
        self.app.reorg_tab.geom.set(p)
        try:
            self.app.reorg_tab.name.set("substituted")
        except Exception:  # noqa: BLE001
            pass
        self.app.show_page("重组能")
        self.app.log("取代后的分子已填入“重组能”页：生成输入并优化后，可继续计算 IP/EA、激发态与转移积分")

    def toggle_measure(self):
        self.measure_pts = []
        self.rend.marks, self.rend.mark_label = [], ""
        self.app.log("测量模式: 依次点击原子，2 个显示距离、3 个显示键角、4 个显示二面角；第 5 次点击重新开始"
                     if self.meas_mode.get() else "已退出测量模式")
        self._redraw_view()

    def add_measure_point(self, k):
        from .crystal import measure
        if len(self.measure_pts) >= 4 or k in self.measure_pts:
            self.measure_pts = []
        self.measure_pts.append(int(k))
        pts = [self.scene_xyz[i] for i in self.measure_pts]
        kind, val, lab = measure(pts)
        names = "–".join(f"{self.scene_syms[i]}{i + 1}" for i in self.measure_pts)
        self.rend.marks, self.rend.mark_label = list(self.measure_pts), lab
        if kind:
            self.app.log(f"{ {'distance': '距离', 'angle': '键角', 'dihedral': '二面角'}[kind] } {names} = {lab}")
            self.last_measure = (kind, val)
        self._redraw_view()

    def _on_resize(self, event=None):
        if not self.scene:
            return
        try:
            if self._resize_job:
                self.after_cancel(self._resize_job)
            self._resize_job = self.after(150, self._redraw_view)
        except Exception:  # noqa: BLE001
            self._redraw_view()

    def _redraw_view(self, fast=False):
        self._resize_job = None
        self._drag_pending = False
        if self.scene:
            self.rend.draw(fast=fast)
            self.plot.canvas.draw_idle()

    def view_along(self, which):
        lat = self.crys["lattice"] if self.crys is not None else np.eye(3)
        v = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}.get(which) if isinstance(which, str) else lat[which]
        up = lat[2] if not isinstance(which, str) and which != 2 else lat[1]
        self.rend.view_along(v, up)
        self.rend.draw()
        self.plot.canvas.draw_idle()

    def save_image(self):
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"),
                                                                            ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if p:
            self.rend.save(p, dpi=300)
            self.app.log(f"已保存 {p} (300 dpi)")

    def on_scroll(self, event):
        if event.inaxes is not self.plot.ax:
            return
        self.plot.zoom(1.15 if event.button == "up" else 1 / 1.15)

    def pick_atom(self, x, y):
        return self.rend.pick(x, y)

    def on_press(self, event):
        if event.inaxes is not self.plot.ax or not self.scene:
            return
        self._drag = {"button": event.button, "x": event.x, "y": event.y, "moved": False, "key": event.key or ""}

    def on_motion(self, event):
        d = self._drag
        if not d or event.x is None:
            return
        dx, dy = event.x - d["x"], event.y - d["y"]
        if not d["moved"] and abs(dx) + abs(dy) < 4:
            return
        d["moved"] = True
        d["x"], d["y"] = event.x, event.y
        if d["button"] == 2:
            ax = self.plot.ax
            (x0, y0), (x1, y1) = ax.transData.inverted().transform([(0, 0), (dx, dy)])
            self.rend.pan -= np.array([x1 - x0, y1 - y0])
        else:
            self.rend.rotate(dx, dy)
        if not self._drag_pending:              # 限帧：上一帧画完前不再排队
            self._drag_pending = True
            try:
                self.after(16, lambda: self._redraw_view(fast=True))
            except Exception:  # noqa: BLE001
                self._redraw_view(fast=True)

    def on_release(self, event):
        d, self._drag = self._drag, None
        if not d:
            return
        if d["moved"]:
            self.rend.draw()
            self.plot.canvas.draw_idle()
            return
        if d["button"] != 1:
            return
        k = self.pick_atom(event.xdata, event.ydata)
        if k is None:
            return
        if self.sub_mode.get():
            self.toggle_sub_atom(k)
            return
        if self.meas_mode.get():
            self.add_measure_point(k)
            return
        mol = int(self.scene_owner[k])
        key = d["key"]
        if "control" in key or "ctrl" in key or "shift" in key:
            sel = [m for m in self.sel if m != mol] if mol in self.sel else self.sel + [mol]
        else:
            sel = [mol]
        self.set_selection(sel)
        self.app.log(f"选中原子 {self.scene_syms[k]}{k + 1} → 分子 {self.scene[mol]['label']} "
                     f"({formula(self.scene[mol]['symbols'])})")

    def set_selection(self, sel):
        self.sel = list(sel)
        self._busy = True
        kids = self.mtree.get_children()
        self.mtree.selection_set([kids[i] for i in self.sel if i < len(kids)])
        self._busy = False
        self.redraw()

    def on_tree_select(self, _=None):
        if self._busy:
            return
        kids = list(self.mtree.get_children())
        chosen = [kids.index(i) for i in self.mtree.selection()]
        if set(chosen) == set(self.sel):
            return                          # 由程序设置选择引起的事件，避免重复绘制
        keep = [k for k in self.sel if k in chosen] + [k for k in chosen if k not in self.sel]
        self.sel = keep
        self.redraw()

    def on_dimer_select(self, _=None):
        sel = self.tree.selection()
        if not sel or not self.dimer_res:
            return
        if self.mode.get() != "cluster":
            self.mode.set("cluster")
            self.rebuild()
        self.set_selection([0, self.tree.index(sel[0]) + 1])

    # ---------------- 导出 ----------------
    def _sel_atoms(self, ks):
        return [(s, *x) for k in ks for s, x in zip(self.scene[k]["symbols"], self.scene[k]["cart"])]

    def export_selected(self):
        if not self.sel:
            raise ValueError("请先在三维图中左键点选分子，或在分子列表中选择")
        p = filedialog.asksaveasfilename(defaultextension=".xyz", filetypes=[("xyz", "*.xyz")])
        if p:
            labs = ", ".join(self.scene[k]["label"] for k in self.sel)
            write_xyz(p, self._sel_atoms(self.sel), f"molecules {labs}")
            self.app.log(f"已导出 {len(self.sel)} 个分子 ({labs}) 到 {p}")

    def export_pair(self):
        if len(self.sel) != 2:
            raise ValueError("请恰好选中两个分子（左键点第一个，Ctrl+左键点第二个）")
        p = filedialog.asksaveasfilename(defaultextension=".xyz", filetypes=[("xyz", "*.xyz")])
        if not p:
            return
        a, b = self.sel
        na = len(self.scene[a]["symbols"])
        ca, cb = self.scene[a]["cart"].mean(0), self.scene[b]["cart"].mean(0)
        write_xyz(p, self._sel_atoms([a, b]), f"dimer nA={na} r={np.linalg.norm(cb - ca):.3f} A")
        self.app.ti_tab.dimer.set(p)
        self.app.ti_tab.na.set(na)
        self.app.log(f"二聚体已保存 {p}（A={self.scene[a]['label']}, B={self.scene[b]['label']}, "
                     f"质心距 {np.linalg.norm(cb - ca):.3f} Å），已填入“转移积分”页")

    def save_xyz(self):
        if not self.scene:
            raise ValueError("没有结构")
        p = filedialog.asksaveasfilename(defaultextension=".xyz", filetypes=[("xyz", "*.xyz")])
        if p:
            write_xyz(p, self._sel_atoms(range(len(self.scene))), "scene")
            self.app.log(f"已保存 {p}")

    def search(self):
        if self.mols is None:
            raise ValueError("请先打开 CIF 晶体文件")
        ref = int(self.ref.get()) - 1
        crit = "atom" if self.crit.get().startswith("最短") else "centroid"
        if self.sel_mode.get().startswith("按方向"):
            from .crystal import paper_dimers
            pl = {"自动 (分子层)": "auto", "三维 (含层间)": "3d"}.get(self.plane.get(), self.plane.get())
            self.dimer_res = paper_dimers(self.crys, self.mols, ref=ref, plane=pl, cutoff=float(self.cut.get()),
                                          criterion=crit)
            self.app.log(f"按方向选择：传输平面 {self.dimer_res['plane']}，" + "、".join(
                f"{g['label']} ({g['members'][0]['r_centroid']:.2f} Å)" for g in self.dimer_res["groups"]) +
                "。P = 最近的 π 堆积方向，T = 其它紧密接触方向，L = 远程，I = 层间")
        else:
            self.dimer_res = find_dimers(self.crys, self.mols, ref=ref, cutoff=float(self.cut.get()), criterion=crit)
        self.fill_dimer_table()
        self.app.log(f"找到 {len(self.dimer_res['dimers'])} 个近邻，{len(self.dimer_res['groups'])} 组二聚体")
        self.mode.set("cluster")
        self.rebuild(keep_view=False)

    def group_name(self, gid):
        g = next((x for x in self.dimer_res["groups"] if x["id"] == gid), None)
        return g.get("label", f"G{gid}") if g else f"G{gid}"

    def fill_dimer_table(self):
        self.tree.delete(*self.tree.get_children())
        for d in self.dimer_res["dimers"]:
            v = ", ".join(f"{x:.2f}" for x in d["vector"])
            self.tree.insert("", "end", values=(self.group_name(d["group"]), f"{d['r_centroid']:.3f}", f"{d['d_min']:.3f}", v))
        stripe(self.tree)

    def remove_selected_groups(self):
        """从二聚体表中移除选中行所属的组。"""
        from .crystal import remove_groups
        if not self.dimer_res:
            raise ValueError("请先搜索近邻二聚体")
        kids = list(self.tree.get_children())
        rows = [kids.index(i) for i in self.tree.selection() if i in kids]
        if not rows:
            raise ValueError("请先在二聚体表中选择要移除的行")
        gids = {self.dimer_res["dimers"][k]["group"] for k in rows}
        names = [self.group_name(g) for g in sorted(gids)]
        self.dimer_res = remove_groups(self.dimer_res, gids)
        self.app.data.pop("crystal_couplings", None)
        self.fill_dimer_table()
        self.rebuild(keep_view=True)
        self.app.log(f"已移除 {', '.join(names)}，剩余 {len(self.dimer_res['groups'])} 组二聚体")

    def export(self):
        if not self.dimer_res:
            raise ValueError("请先搜索近邻二聚体")
        d = filedialog.askdirectory(title="选择输出目录")
        if not d:
            return
        program, settings = self.app.ti_tab.settings()
        files = prepare_dimer_jobs(self.dimer_res, d, program=program, **settings)
        self.app.log(f"已导出 {len(files)} 组二聚体到 {d}（{program}），近邻信息见 dimers.json")
        if program == "adf":
            nat = 2 * len(self.dimer_res["atoms_a"])
            big = (f"\n\n注意：每个二聚体 {nat} 个原子，{settings.get('basis', 'TZP')} 全电子计算量较大，ADF2016 单个二聚体"
                   "可能需要数小时。可改用 DZP 基组，或在“工具 → 程序路径设置”中设置 ADF 并行核数。") if nat > 100 else ""
            msg = (f"已在 {d} 生成 {len(files)} 组二聚体的批量计算脚本 batch_adf.run (一次算完全部二聚体)。\n\n"
                   "运行方式 (任选其一)：\n"
                   "1. 在 ADFjobs 中打开 batch_adf.run 运行；同时可在“运行任务”窗口点“监视 ADFjobs 结果”，算完自动读取；\n"
                   "2. 在“运行任务”窗口选中 batch_adf.sh 运行，结束后自动读取并填入“迁移率”页。" + big)
            self.app.log(msg.replace("\n\n", " ").replace("\n", " "))
            if messagebox.askyesno("完成", msg + "\n\n现在打开“运行任务”窗口吗？"):
                self.app.open_runner(d)
            return
        messagebox.showinfo("完成", f"已在 {d} 生成 {len(files)} 组二聚体输入。\n计算完成后在“迁移率”页"
                                    f"点击“从二聚体目录导入”即可自动读取 J。")

    def to_mobility(self):
        if not self.dimer_res:
            raise ValueError("请先搜索近邻二聚体")
        rows = [{"name": f"{self.group_name(d['group'])}-{i + 1}", "vector": d["vector"]}
                for i, d in enumerate(self.dimer_res["dimers"])]
        self.app.mob_tab.set_paths(rows)
        self.app.log("近邻矢量已填入迁移率表，请填写各组 Ve / Vh")


def formula(symbols):
    """Hill 记号化学式。"""
    from collections import Counter
    c = Counter(symbols)
    order = (["C", "H"] if "C" in c else []) + sorted(k for k in c if k not in (("C", "H") if "C" in c else ()))
    return "".join(f"{e}{c[e] if c[e] > 1 else ''}" for e in order)


# ==============================================================================================
# 重组能
# ==============================================================================================
class ReorgTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        S = safe(app)
        panel = SectionPanel(self)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner
        self.geom = FileEntry(left, "分子结构", filetypes=STRUCT_TYPES)
        self.geom.pack(fill="x")
        ttk.Button(left, text="使用“结构”页当前文件", command=lambda: self.geom.set(app.current_structure or "")
                   ).pack(anchor="e")
        g = ttk.LabelFrame(left, text="计算设置 (Gaussian)")
        g.pack(fill="x", pady=4)
        self.method = field(g, 0, "泛函", "B3LYP", values=FUNCTIONALS, width=18)
        self.basis = field(g, 1, "基组", "6-31G(d)", values=BASIS_SETS, width=18)
        self.carrier = field(g, 2, "载流子", "空穴+电子", values=["空穴+电子", "空穴", "电子"], width=18)
        self.charge = field(g, 3, "中性电荷", 0)
        self.mult = field(g, 4, "中性多重度", 1)
        self.ion_mult = field(g, 5, "离子多重度", 2)
        self.nproc = field(g, 6, "核数", 8)
        self.mem = field(g, 7, "内存", "8GB")
        self.extra = field(g, 8, "额外关键词", "", width=18)
        self.program = field(g, 9, "高斯命令", "g16")
        self.sp_method = field(g, 10, "单点泛函(空=同上)", "", values=[""] + FUNCTIONALS, width=18)
        self.sp_basis = field(g, 11, "单点基组(空=同上)", "", values=[""] + BASIS_SETS, width=18)
        self.solvent = field(g, 13, "溶剂 (PCM, 空=气相)", "", values=SOLVENTS, width=18)
        self.ecp = field(g, 14, "重原子赝势基组", "SDD", width=18)
        ttk.Label(g, text="含 Ir 等重原子且用 Pople 基组时自动改为 GenECP", foreground=PALETTE["muted"]).grid(
            row=15, column=0, columnspan=2, sticky="w")
        self.freq = tk.BooleanVar(value=True)
        ttk.Checkbutton(g, text="优化后做频率计算 (重组能分解必需)", variable=self.freq).grid(row=12, column=0,
                                                                                columnspan=2, sticky="w")
        self.outdir = FileEntry(left, "输出目录", mode="dir")
        self.outdir.pack(fill="x")
        nf = ttk.Frame(left)
        nf.pack(fill="x")
        self.name = field(nf, 0, "任务名(可空)", "")
        ttk.Button(left, text="生成高斯输入文件", style="Accent.TButton", command=S(self.generate)).pack(fill="x", pady=4)
        ttk.Button(left, text="▶ 在本机运行这些任务…", command=S(lambda: app.open_runner(self.resdir.get()))).pack(fill="x")

        a = ttk.LabelFrame(left, text="结果分析")
        a.pack(fill="x", pady=6)
        self.resdir = FileEntry(a, "结果目录", mode="dir")
        self.resdir.pack(fill="x")
        ttk.Button(a, text="读取 log 计算重组能", style="Accent.TButton", command=S(self.analyze)).pack(fill="x")
        ttk.Button(a, text="把 λ 送到迁移率页", command=S(self.send)).pack(fill="x", pady=2)

        self.text = styled_text(self, app.mono, wrap="none", size=10)
        self.text.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.result = None

    def generate(self):
        carriers = {"空穴+电子": ("hole", "electron"), "空穴": ("hole",), "电子": ("electron",)}[self.carrier.get()]
        out = self.outdir.get() or str(Path(self.geom.get()).parent / "reorg")
        m = generate_reorg_inputs(self.geom.get(), out, self.name.get() or None, self.method.get(), self.basis.get(), carriers,
                                  int(self.charge.get()), int(self.mult.get()), int(self.ion_mult.get()),
                                  int(self.nproc.get()), self.mem.get(), self.freq.get(), self.extra.get(),
                                  self.program.get(), self.sp_method.get().strip() or None,
                                  self.sp_basis.get().strip() or None, self.solvent.get().strip(),
                                  self.ecp.get().strip() or "SDD")
        self.resdir.set(out)
        self.text.delete("1.0", "end")
        for stem in m["files"]:
            self.text.insert("end", f"======== {stem}.gjf ========\n" + (Path(out) / f"{stem}.gjf").read_text() + "\n")
        self.text.insert("end", "======== run_reorg.sh ========\n" + (Path(out) / "run_reorg.sh").read_text())
        self.app.log(f"重组能输入已生成于 {out}")

    def analyze(self):
        self.result = analyze_reorg(self.resdir.get())
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"重组能 (四点法)  {self.result['name']}   优化: {self.result['level']}   "
                        f"能量: {self.result['sp_level']}\n\n")
        for w in self.result["warnings"]:
            t.insert("end", f"[警告] {w}\n")
        for car, r in self.result["results"].items():
            t.insert("end", f"\n{'空穴' if car == 'hole' else '电子'}:\n")
            for k, v in r.items():
                t.insert("end", f"   {k:<26s}{v:18.8f}\n")
        if not self.result["results"]:
            t.insert("end", "\n未得到完整的四点能量。\n")
        t.insert("end", "\n" + format_ip_ea(self.result["ip_ea"], self.result["orbitals"]) + "\n")
        self.app.data["reorg"] = self.result
        self.app.ipea_tab.load_result(self.result)

    def send(self):
        if not self.result:
            raise ValueError("请先读取重组能结果")
        self.app.mob_tab.set_lambda(self.result["results"])


# ==============================================================================================
# 重组能分解
# ==============================================================================================
class DecompTab(ttk.Frame):
    PLOTS = ["各振动模式 λ_i", "黄昆因子 S_i", "Duschinsky 矩阵", "内坐标类型", "主要内坐标 (前 20)", "原子贡献 (三维)"]

    def __init__(self, parent, app):
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.sets, self.res = None, None
        S = safe(app)
        panel = SectionPanel(self)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner
        self.source = tk.StringVar(value="builtin")
        sf = ttk.Frame(left)
        sf.pack(fill="x")
        ttk.Label(sf, text="方法:").pack(side="left")
        ttk.Radiobutton(sf, text="内置 (读 fchk Hessian)", value="builtin", variable=self.source).pack(side="left")
        ttk.Radiobutton(sf, text="Gaussian FCHT 输出", value="fcht", variable=self.source).pack(side="left")

        b1 = ttk.LabelFrame(left, text="内置方法：两个态 opt+freq 的 fchk")
        b1.pack(fill="x", pady=2)
        self.fn = FileEntry(b1, "中性 fchk", filetypes=FCHK_TYPES)
        self.fn.pack(fill="x")
        self.fi = FileEntry(b1, "离子 fchk", filetypes=FCHK_TYPES)
        self.fi.pack(fill="x")
        b2 = ttk.LabelFrame(left, text="FCHT 方法：freq(readfc,fcht,readfcht) 的 log")
        b2.pack(fill="x", pady=2)
        self.l1 = FileEntry(b2, "λ1 (离子→中性)", filetypes=[("log", "*.log *.out"), ("所有文件", "*.*")])
        self.l1.pack(fill="x")
        self.l2 = FileEntry(b2, "λ2 (中性→离子)", filetypes=[("log", "*.log *.out"), ("所有文件", "*.*")])
        self.l2.pack(fill="x")
        g2 = ttk.Frame(b2)
        g2.pack(fill="x")
        self.redim = field(g2, 0, "RedDim 低频截断/cm⁻¹ (空=不用)", "")
        ttk.Button(b2, text="在重组能目录生成 FCHT 输入…", command=S(self.gen_fcht)).pack(fill="x")

        row = ttk.Frame(left)
        row.pack(fill="x", pady=2)
        self.auto_car = field(row, 0, "载流子", "空穴", values=["空穴", "电子"], width=6)
        ttk.Button(row, text="从重组能目录自动填文件…", command=S(self.autofind)).grid(row=0, column=2, padx=4)
        g = ttk.Frame(left)
        g.pack(fill="x")
        self.scale = field(g, 0, "成键判据倍数", 1.2)
        self.fwhm = field(g, 1, "展宽 FWHM/cm⁻¹", 20)
        ttk.Button(left, text="开始分解", style="Accent.TButton", command=S(self.run)).pack(fill="x", pady=4)
        v = ttk.LabelFrame(left, text="作图")
        v.pack(fill="x")
        self.state_var = tk.StringVar(value="neutral")
        ttk.Radiobutton(v, text="λ1 (中性态势能面, 中性态模式)", value="neutral", variable=self.state_var,
                        command=S(self.draw)).pack(anchor="w")
        ttk.Radiobutton(v, text="λ2 (离子态势能面, 离子态模式)", value="ion", variable=self.state_var,
                        command=S(self.draw)).pack(anchor="w")
        self.kind = MappedVar(self.PLOTS[0], {p: T(p) for p in self.PLOTS})
        cb = ttk.Combobox(v, textvariable=self.kind, values=self.PLOTS, state="readonly")
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", S(lambda e: self.draw()))
        bf = ttk.Frame(left)
        bf.pack(fill="x", pady=4)
        ttk.Button(bf, text="导出插图 (a)(b)…", command=S(self.export_fig)).pack(side="left", fill="x", expand=True)
        ttk.Button(self._hr_parent(bf), text="读取黄昆因子结果… (FCHT 输出 / CSV / TXT / DAT)",
                   command=S(self.import_hr)).pack(fill="x", pady=(3, 0))
        ttk.Button(bf, text="导出数据 (CSV/Origin)…", command=S(self.export)).pack(side="left", fill="x", expand=True)
        self.text = styled_text(left, app.mono, width=56, height=16, size=9)
        self.text.pack(fill="both", expand=True)
        self.plot = PlotPanel(self, figsize=(7, 6))
        self.plot.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

    def _carrier_ion(self):
        return "cation" if self.auto_car.get() == "空穴" else "anion"

    def gen_fcht(self):
        d = filedialog.askdirectory(title="选择重组能计算目录 (含 reorg_manifest.json)")
        if not d:
            return
        man = json.loads((Path(d) / "reorg_manifest.json").read_text())
        carriers = [c for c, ion in (("hole", "cation"), ("electron", "anion")) if f"{man['name']}_{ion}" in man["files"]]
        redim = self.redim.get().strip()
        jobs = generate_fcht_inputs(d, man["name"], carriers, clear_low_freq=float(redim) if redim else None)
        self.app.log(f"已生成 {len(jobs)} 个 FCHT 输入和 run_fcht.sh / run_fcht.bat：" + ", ".join(j["stem"] for j in jobs))
        self.source.set("fcht")

    def autofind(self):
        d = filedialog.askdirectory(title="选择重组能计算目录")
        if not d:
            return
        ion, d = self._carrier_ion(), Path(d)
        n, i = sorted(list(d.glob("*_neutral.fchk")) + list(d.glob("*_neutral.fch"))), \
            sorted(list(d.glob(f"*_{ion}.fchk")) + list(d.glob(f"*_{ion}.fch")))
        if n and i:
            self.fn.set(n[0])
            self.fi.set(i[0])
        l1, l2 = sorted(d.glob(f"*_fcht_{ion}_to_neutral.log")), sorted(d.glob(f"*_fcht_neutral_to_{ion}.log"))
        if l1:
            self.l1.set(l1[0])
        if l2:
            self.l2.set(l2[0])
        if not (n and i) and not (l1 or l2):
            raise FileNotFoundError(f"在 {d} 中没找到 *_neutral.fchk / *_{ion}.fchk 或 FCHT 的 log")
        self.app.log(f"已自动填入 {d} 中的文件")

    def run(self):
        """内置方法的 Hessian 分解较耗时：放到后台线程计算，完成后在界面线程显示。"""
        if self.source.get() == "builtin":
            fn, fi, sc = self.fn.get(), self.fi.get(), float(self.scale.get())
            self.app.run_bg("重组能分解", lambda: decompose_reorg(fn, fi, sc), lambda res: self._run(res))
            return
        return self._run()

    def _run(self, pre=None):
        t = self.text
        t.delete("1.0", "end")
        if self.source.get() == "builtin":
            self.res = pre if pre is not None else decompose_reorg(self.fn.get(), self.fi.get(), float(self.scale.get()))
            jm = self.res["duschinsky"]["matrix"]
            self.sets = {"neutral": dict(self.res["neutral"], duschinsky=jm.T),
                         "ion": dict(self.res["ion"], duschinsky=jm)}
            t.insert("end", "重组能分解 (谐振近似, meV)   简正模式   内坐标   RMSD/Å\n")
            for key, lab in (("neutral", "λ1 中性态"), ("ion", "λ2 离子态")):
                r = self.sets[key]
                t.insert("end", f" {lab:<12}{r['lambda_normal_modes_meV']:16.2f}{r['lambda_internal_meV']:9.2f}"
                                f"{r['rmsd_A']:9.4f}\n")
            t.insert("end", f" 合计        {self.res['lambda_total_normal_modes_meV']:16.2f}"
                            f"{self.res['lambda_total_internal_meV']:9.2f}\n")
        else:
            self.res, self.sets = None, {}
            for key, entry in (("neutral", self.l1), ("ion", self.l2)):
                if entry.get():
                    try:
                        r = read_fcht_log(entry.get())
                    except ValueError as exc:
                        from .gaussian_fcht import explain_fcht_failure
                        raise ValueError(explain_fcht_failure(entry.get()) or str(exc)) from None
                    self.sets[key] = {"modes": r["modes"], "duschinsky": r["duschinsky"], "fcht": r,
                                      "lambda_normal_modes_meV": r["lambda_meV"], "n_imaginary": 0}
                    for w in r["warnings"]:
                        t.insert("end", f"[警告] {w}\n")
            if not self.sets:
                raise ValueError("请至少给出一个 FCHT 输出文件")
            t.insert("end", "Gaussian FCHT: λ_i = S_i·hν_i\n")
            for key, lab in (("neutral", "λ1 中性态"), ("ion", "λ2 离子态")):
                if key in self.sets:
                    r = self.sets[key]["fcht"]
                    t.insert("end", f" {lab}: {r['lambda_meV']:.2f} meV = {r['lambda_kJ/mol']:.3f} kJ/mol\n")
            if "neutral" not in self.sets:
                self.state_var.set("ion")
            if self.kind.get() in self.PLOTS[3:]:
                self.kind.set(self.PLOTS[0])
        reorg = self.app.data.get("reorg")
        if reorg:
            for car, r in reorg["results"].items():
                t.insert("end", f" 四点法 λ({car}) = {r['lambda_eV'] * 1000:.2f} meV "
                                f"(λ1 {r['lambda1_eV'] * 1000:.2f}, λ2 {r['lambda2_eV'] * 1000:.2f})\n")
        for key, lab in (("neutral", "λ1"), ("ion", "λ2")):
            if key not in self.sets:
                continue
            r = self.sets[key]
            if r.get("n_imaginary"):
                t.insert("end", f"[警告] {lab} 所用态有 {r['n_imaginary']} 个虚频，结果不可靠\n")
            if "internal_by_type_meV" in r:
                t.insert("end", f"\n{lab} 按类型: " + ", ".join(f"{k} {v:.2f}" for k, v in
                                                              r["internal_by_type_meV"].items()))
            t.insert("end", f"\n{lab} 贡献最大的振动模式:\n   频率/cm-1      S       λ_i/meV\n")
            for m in sorted(r["modes"], key=lambda m: -m["lambda_meV"])[:8]:
                t.insert("end", f"   {m['freq_cm-1']:9.1f}{m['S']:10.4f}{m['lambda_meV']:11.3f}\n")
        self.app.log("重组能分解完成")
        self.draw()

    @staticmethod
    def _hr_parent(bf):
        return bf.master

    def import_hr(self, path=None):
        """导入已有的黄昆因子结果 (FCHT 输出、MOMAP / DUSHIN / 本程序导出的表格)，存入当前选择的态 (λ1 或 λ2)。"""
        from .gaussian_fcht import read_hr_table
        p = path or filedialog.askopenfilename(
            title=T("选择黄昆因子结果文件"), initialdir=self.app.config_data.get("workdir") or None,
            filetypes=[("黄昆因子结果", "*.log *.out *.csv *.txt *.dat"), ("所有文件", "*.*")])
        if not p:
            return None
        record_recent(p)
        r = read_hr_table(p)
        key = self.state_var.get() if self.state_var.get() in ("neutral", "ion") else "neutral"
        if not self.sets or self.res is not None:          # 清掉内置分解的数据，避免混用
            self.sets, self.res = {}, None
        self.sets[key] = {"modes": r["modes"], "duschinsky": r["duschinsky"], "fcht": r,
                          "lambda_normal_modes_meV": r["lambda_meV"], "n_imaginary": 0}
        lab = "λ1" if key == "neutral" else "λ2"
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"已读取 {Path(p).name} 作为 {lab}: {len(r['modes'])} 个振动模式，"
                        f"λ = Σ S_i·hν_i = {r['lambda_meV']:.2f} meV ({r['lambda_kJ/mol']:.3f} kJ/mol)\n")
        t.insert("end", f"\n{lab} 贡献最大的振动模式:\n   频率/cm-1      S       λ_i/meV\n")
        for m in sorted(r["modes"], key=lambda m: -m["lambda_meV"])[:8]:
            t.insert("end", f"   {m['freq_cm-1']:9.1f}{m['S']:10.4f}{m['lambda_meV']:11.3f}\n")
        if r["duschinsky"] is None and self.kind.get() in self.PLOTS[3:]:
            self.kind.set(self.PLOTS[0])
        self.app.log(f"黄昆因子结果已读取 ({lab})，可用于作图、导出和“输运理论与光物理扩展”页的振动分辨光谱")
        self.draw()
        return r

    def _current(self):
        key = self.state_var.get()
        if not self.sets or key not in self.sets:
            raise ValueError("该态没有数据")
        return self.sets[key], ("λ1" if key == "neutral" else "λ2")

    def draw(self):
        if not self.sets:
            return
        r, lab = self._current()
        kind = self.kind.get()
        if kind in self.PLOTS[3:] and "internal" not in r:
            raise ValueError("内坐标 / 原子分解只有内置方法 (fchk) 才有")
        if kind == self.PLOTS[5]:
            ax = self.plot.new_axes("3d")
            st = Path(self.fn.get() if self.state_var.get() == "neutral" else self.fi.get())
            atoms = read_geometry(st)
            sm = draw_structure(ax, [a[0] for a in atoms], [a[1:] for a in atoms],
                                atom_values=[a["lambda_meV"] for a in r["atoms"]], title=f"{lab} 原子贡献 (meV)")
            self.plot.fig.colorbar(sm, ax=ax, shrink=0.6, label="meV")
            self.plot.canvas.draw_idle()
            return
        ax = self.plot.new_axes()
        if kind == self.PLOTS[2]:
            if r.get("duschinsky") is None:
                raise ValueError("没有 Duschinsky 矩阵（FCHT 需要 print=(huangrhys,matrix=JK)）")
            draw_duschinsky(ax, r["duschinsky"], self.plot.fig)
            ax.set_title(f"{lab}: 末态={'中性' if lab == 'λ1' else '离子'}，初态={'离子' if lab == 'λ1' else '中性'}",
                         fontsize=10)
        elif kind in self.PLOTS[:2]:
            f = np.array([m["freq_cm-1"] for m in r["modes"]])
            y = np.array([m["lambda_meV"] if kind == self.PLOTS[0] else m["S"] for m in r["modes"]])
            ax.vlines(f, 0, y, color="#2B4A77", lw=1.5)
            ax.set_xlabel("频率 / cm$^{-1}$")
            if kind == self.PLOTS[0]:
                ax.set_ylabel("λ$_i$ / meV")
                grid, spec = broadened_spectrum(r["modes"], float(self.fwhm.get()))
                ax2 = ax.twinx()
                ax2.plot(grid, spec, color="#d62728", lw=1)
                ax2.set_ylabel("展宽谱 / meV·cm", color="#d62728")
                ax.set_title(f"{lab} 振动模式分解  Σ = {y.sum():.1f} meV")
            else:
                ax.set_ylabel("黄昆因子 S$_i$")
                ax.set_title(f"{lab} Huang–Rhys 因子  ΣS = {y.sum():.3f}")
        elif kind == self.PLOTS[3]:
            names = {"bond": "键长", "angle": "键角", "dihedral": "二面角"}
            items = r["internal_by_type_meV"]
            ax.bar([names[k] for k in items], list(items.values()), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
            ax.set_ylabel("λ / meV")
            ax.set_title(f"{lab} 按内坐标类型")
        else:
            top = sorted(r["internal"], key=lambda x: -abs(x["lambda_meV"]))[:20][::-1]
            unit = {"bond": "Å", "angle": "°", "dihedral": "°"}
            labels = [f"{x['type'][0].upper()}({x['atoms']}) {x['elements']} Δ={x['delta']:+.3f}{unit[x['type']]}"
                      for x in top]
            ax.barh(range(len(top)), [x["lambda_meV"] for x in top], color="#1f77b4")
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("λ / meV")
            ax.set_title(f"{lab} 贡献最大的内坐标")
        self.plot.draw()

    def export_fig(self):
        r, lab = self._current()
        d = filedialog.askdirectory(title="选择图片保存目录")
        if not d:
            return
        tag = "lambda1_neutral" if lab == "λ1" else "lambda2_ion"
        f = [m["freq_cm-1"] for m in r["modes"]]
        p1 = Path(d) / f"Huang_and_Duschinsky_{tag}.png"
        figure_hr_duschinsky(f, [m["S"] for m in r["modes"]], r.get("duschinsky"), p1)
        p2 = Path(d) / f"reorg_decomposition_{tag}.png"
        figure_reorg_decomposition(f, [m["lambda_meV"] for m in r["modes"]], p2, fwhm=float(self.fwhm.get()))
        p3 = Path(d) / f"Fig3_style_{tag}.png"
        figure_three_panel(f, [m["S"] for m in r["modes"]], r.get("duschinsky"),
                           [m["lambda_meV"] / 1000 for m in r["modes"]], p3)
        self.app.log(f"已保存 300 dpi 图片: {p1.name}, {p2.name}, {p3.name}（三联图）")

    def export(self):
        if not self.sets:
            raise ValueError("请先进行分解")
        d = filedialog.askdirectory(title="选择导出目录")
        if not d:
            return
        n = 0
        if self.res is not None:
            n += len(export_decomposition_csv(self.res, d))
            (Path(d) / "reorg_decomp_summary.json").write_text(json.dumps(
                {k: v for k, v in self.res.items() if not isinstance(v, dict)}, indent=2, ensure_ascii=False))
        for key, r in self.sets.items():
            tag = "lambda1_neutral" if key == "neutral" else "lambda2_ion"
            fake = r.get("fcht") or {
                "modes": [dict(m, **{"lambda_kJ/mol": m["lambda_meV"] / 1000 * 96.48533}) for m in r["modes"]],
                "lambda_meV": r["lambda_normal_modes_meV"],
                "lambda_kJ/mol": r["lambda_normal_modes_meV"] / 1000 * 96.48533, "duschinsky": r.get("duschinsky")}
            n += len(export_fcht(fake, d, tag))
        self.app.log(f"已导出 {n} 个数据文件到 {d}（含 黄昆因子.csv / 矩阵.csv / .hr.txt / .mat.txt，可直接用于 Origin）")


# ==============================================================================================
# IP / EA
# ==============================================================================================
class IPEATab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.result, self.orbitals = None, None
        S = safe(app)
        panel = SectionPanel(self)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner
        ttk.Button(left, text="从重组能结果目录读取 7 个能量…", command=S(self.from_dir)).pack(fill="x")
        e = ttk.LabelFrame(left, text="七个能量 / Hartree（也可手动填写或粘贴）")
        e.pack(fill="x", pady=4)
        self.vars = {}
        for i, (k, lab) in enumerate(ENERGY_KEYS):
            self.vars[k] = field(e, i, lab, "", width=20)
        ttk.Button(left, text="计算 IP / EA / HEP / EEP / 重组能", style="Accent.TButton", command=S(self.compute)).pack(fill="x", pady=4)
        o = ttk.LabelFrame(left, text="前线轨道 (HOMO / LUMO)")
        o.pack(fill="x", pady=4)
        self.orbfile = FileEntry(o, "fchk 或 log", filetypes=[("fchk/log", "*.fchk *.fch *.log *.out"),
                                                              ("所有文件", "*.*")])
        self.orbfile.pack(fill="x")
        ttk.Button(o, text="读取 HOMO / LUMO", command=S(self.read_orb)).pack(fill="x")
        oc = ttk.LabelFrame(left, text="轨道片段组成")
        oc.pack(fill="x", pady=4)
        ttk.Label(oc, text="每行一个片段，如  Fragment1: 1-5,60", foreground=PALETTE["muted"]).pack(anchor="w")
        self.frag_text = styled_text(oc, app.mono, height=5, size=9)
        self.frag_text.pack(fill="x")
        of = ttk.Frame(oc)
        of.pack(fill="x")
        self.orbs = field(of, 0, "轨道", "HOMO-1,HOMO,LUMO,LUMO+1", width=22)
        self.comp_method = field(of, 1, "方法", "SCPA", values=["SCPA", "Mulliken"], width=22)
        bb = ttk.Frame(oc)
        bb.pack(fill="x")
        ttk.Button(bb, text="自动划分 (金属/配体)", command=S(self.auto_frag)).pack(side="left", fill="x", expand=True)
        ttk.Button(bb, text="计算组成", style="Accent.TButton", command=S(self.comp)).pack(side="left", fill="x",
                                                                                         expand=True)
        ttk.Button(left, text="导出结果 (CSV)…", command=S(self.export)).pack(fill="x", pady=4)
        ttk.Label(left, text="IP(v)=E+(N)-E0(N)  IP(a)=E+(C)-E0(N)  HEP=E+(C)-E0(C)\n"
                             "EA(v)=E0(N)-E-(N)  EA(a)=E0(N)-E-(A)  EEP=E0(A)-E-(A)\n"
                             "λh=IP(v)-HEP   λe=EEP-EA(v)", foreground=PALETTE["muted"], justify="left").pack(anchor="w")
        right = ttk.Frame(self, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.text = styled_text(right, app.mono, height=18, wrap="none", size=10)
        self.text.pack(fill="x", pady=(0, 8))
        self.plot = PlotPanel(right, figsize=(5, 4))
        self.plot.pack(fill="both", expand=True)

    def load_result(self, res):
        for k, v in res["energies_Ha"].items():
            self.vars[k].set(f"{v:.10f}")
        self.orbitals = res.get("orbitals")
        self.compute()

    def from_dir(self):
        d = filedialog.askdirectory(title="选择重组能计算目录 (含 reorg_manifest.json)")
        if d:
            res = analyze_reorg(d)
            for w in res["warnings"]:
                self.app.log("[警告] " + w)
            self.app.data["reorg"] = res
            self.load_result(res)
            neu = sorted(Path(d).glob("*_neutral.fchk"))
            if neu:
                self.orbfile.set(neu[0])

    def compute(self):
        energies = {k: float(v.get()) for k, v in self.vars.items() if str(v.get()).strip()}
        if not energies:
            raise ValueError("请先填写能量")
        self.result = ip_ea_from_energies(energies)
        self.text.delete("1.0", "end")
        self.text.insert("end", format_ip_ea(self.result, self.orbitals) + "\n")
        missing = [lab for k, lab in ENERGY_KEYS if k not in energies]
        if missing:
            self.text.insert("end", "\n未填写: " + "; ".join(missing) + "\n")
        ax = self.plot.new_axes()
        draw_energy_levels(ax, self.result, self.orbitals)
        self.plot.draw()

    def read_orb(self):
        self.orbitals = frontier_orbitals(self.orbfile.get())
        if any(str(v.get()).strip() for v in self.vars.values()):
            self.compute()
        else:
            self.text.delete("1.0", "end")
            self.text.insert("end", "\n".join(f"{k:<16}{v:10.4f}" for k, v in self.orbitals.items()) + "\n")

    def _frags(self, natoms):
        frags = {}
        for line in self.frag_text.get("1.0", "end").splitlines():
            if ":" in line:
                name, rng = line.split(":", 1)
                if rng.strip():
                    frags[name.strip()] = parse_ranges(rng, natoms)
        return frags

    def auto_frag(self):
        from .gaussian_io import read_fchk as _rf, ELEMENTS as _EL, BOHR2ANG as _B
        d = _rf(self.orbfile.get())
        sym = [_EL[z] for z in d["Atomic numbers"]]
        xyz = np.array(d["Current cartesian coordinates"]).reshape(-1, 3) * _B
        fr = auto_fragments(sym, xyz)
        self.frag_text.delete("1.0", "end")
        for k, v in fr.items():
            self.frag_text.insert("end", f"{k}: {compress_ranges([i + 1 for i in v])}\n")

    def comp(self):
        f = self.orbfile.get()
        if not f.lower().endswith((".fchk", ".fch")):
            raise ValueError("轨道组成需要 fchk 文件")
        from .gaussian_io import read_fchk as _rf
        natoms = _rf(f)["Number of atoms"]
        res = orbital_composition(f, self._frags(natoms) or None,
                                  [o.strip() for o in self.orbs.get().split(",") if o.strip()],
                                  self.comp_method.get().lower())
        self.text.delete("1.0", "end")
        self.text.insert("end", format_composition(res) + "\n")
        self.app.log("轨道组成计算完成")

    def export(self):
        if not self.result:
            raise ValueError("请先计算")
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("csv", "*.csv")])
        if not p:
            return
        import csv
        with open(p, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["物理量", "eV", "kJ/mol", "kcal/mol"])
            for r in self.result.values():
                w.writerow([r["name"], r["eV"], r["kJ/mol"], r["kcal/mol"]])
            if self.orbitals:
                for k, v in self.orbitals.items():
                    w.writerow([k, v, "", ""])
        self.app.log(f"已导出 {p}")


# ==============================================================================================
# 激发态 / 光谱
# ==============================================================================================
class PhotoTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.systems, self.current = [], None
        S = safe(app)
        panel = SectionPanel(self)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner
        g = ttk.LabelFrame(left, text="生成 S0 / S1 / T1 计算 (TD-DFT)")
        g.pack(fill="x")
        self.geom = FileEntry(g, "分子结构", filetypes=STRUCT_TYPES)
        self.geom.pack(fill="x")
        gf = ttk.Frame(g)
        gf.pack(fill="x")
        self.method = field(gf, 0, "泛函", "B3LYP", values=FUNCTIONALS, width=18)
        self.basis = field(gf, 1, "优化基组", "6-311G(d,p)", values=BASIS_SETS, width=18)
        self.sp_basis = field(gf, 2, "单点基组", "def2-TZVP", values=BASIS_SETS, width=18)
        self.solvent = field(gf, 3, "溶剂 (PCM)", "dichloromethane", values=SOLVENTS, width=18)
        self.nstates = field(gf, 4, "激发态数", 10, width=18)
        self.charge = field(gf, 5, "电荷 / 多重度", "0 1", width=18)
        self.nproc = field(gf, 6, "核数 / 内存", "8 8GB", width=18)
        self.outdir = FileEntry(g, "输出目录", mode="dir")
        self.outdir.pack(fill="x")
        ttk.Button(g, text="生成输入", style="Accent.TButton", command=S(self.generate)).pack(fill="x", pady=2)
        ttk.Button(g, text="▶ 在本机运行…", command=S(lambda: app.open_runner(self.outdir.get()))).pack(fill="x")

        r = ttk.LabelFrame(left, text="读取结果 (吸收 / 荧光 / 磷光 / Stokes / RMSD)")
        r.pack(fill="x", pady=4)
        self.resdir = FileEntry(r, "结果目录", mode="dir")
        self.resdir.pack(fill="x")
        rf = ttk.Frame(r)
        rf.pack(fill="x")
        self.name = field(rf, 0, "任务名 (空=自动)", "", width=18)
        self.label = field(rf, 1, "图例名称", "", width=18)
        ttk.Button(r, text="读取并加入光谱对比", style="Accent.TButton", command=S(self.analyze)).pack(fill="x")

        sp = ttk.LabelFrame(left, text="吸收与发射光谱")
        sp.pack(fill="x", pady=4)
        sf = ttk.Frame(sp)
        sf.pack(fill="x")
        self.fwhm_abs = field(sf, 0, "吸收 FWHM / eV", 0.3)
        self.fwhm_emi = field(sf, 1, "磷光 FWHM / eV", 0.18)
        self.nm_range = field(sf, 2, "波长范围 / nm", "300-800")
        self.sys_lab = ttk.Label(sp, text="对比体系: (空)", foreground=PALETTE["muted"], wraplength=380)
        self.sys_lab.pack(anchor="w")
        bf = ttk.Frame(sp)
        bf.pack(fill="x")
        for txt, cmd in (("刷新光谱", self.draw_spectra), ("清空", self.clear), ("导出光谱图…", self.export_spectra)):
            ttk.Button(bf, text=txt, command=S(cmd)).pack(side="left", fill="x", expand=True)

        rm = ttk.LabelFrame(left, text="结构叠合 RMSD")
        rm.pack(fill="x", pady=4)
        self.ga = FileEntry(rm, "结构 1", filetypes=STRUCT_TYPES)
        self.ga.pack(fill="x")
        self.gb = FileEntry(rm, "结构 2", filetypes=STRUCT_TYPES)
        self.gb.pack(fill="x")
        self.heavy = tk.BooleanVar(value=False)
        ttk.Checkbutton(rm, text="只计重原子", variable=self.heavy).pack(anchor="w")
        rb = ttk.Frame(rm)
        rb.pack(fill="x")
        ttk.Button(rb, text="计算 RMSD", command=S(self.calc_rmsd)).pack(side="left", fill="x", expand=True)
        ttk.Button(rb, text="S0–S1 / S0–T1 叠合图", command=S(self.state_overlays)).pack(side="left", fill="x",
                                                                                      expand=True)

        k = ttk.LabelFrame(left, text="辐射 / 非辐射速率 (Table S5)")
        k.pack(fill="x", pady=4)
        kf = ttk.Frame(k)
        kf.pack(fill="x")
        self.plqy = field(kf, 0, "PLQY Φ (0–1)", 0.21)
        self.tau = field(kf, 1, "寿命 τ / μs", 1.4)
        ttk.Button(k, text="计算 k_r, k_nr", command=S(self.rates)).pack(fill="x")

        right = ttk.Frame(self, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.plot = PlotPanel(right, figsize=(6.5, 4.6))
        self.plot.pack(fill="both", expand=True)
        self.text = styled_text(right, app.mono, height=13, wrap="none", size=9)
        self.text.pack(fill="x", pady=(8, 0))

    def generate(self):
        q, m = (int(v) for v in self.charge.get().split())
        npr, mem = self.nproc.get().split()
        out = self.outdir.get() or str(Path(self.geom.get()).parent / "excited")
        stems = generate_excited_inputs(self.geom.get(), out, None, self.method.get(), self.basis.get(),
                                        self.sp_basis.get(), self.solvent.get().strip(), int(self.nstates.get()),
                                        q, m, int(npr), mem)
        self.outdir.set(out)
        self.resdir.set(out)
        self.text.delete("1.0", "end")
        for st in stems:
            self.text.insert("end", f"======== {st}.gjf ========\n")
            body = (Path(out) / f"{st}.gjf").read_text().splitlines()
            self.text.insert("end", "\n".join(l for l in body if l.startswith(("%", "#", "--Link1"))) + "\n")
        self.app.log(f"激发态输入已生成于 {out}（S1/T1 会读取 S0 的 chk，需按 S0→S1→T1 顺序运行）")

    def _name(self, d):
        n = self.name.get().strip()
        if n:
            return n
        hits = sorted(list(Path(d).glob("*_S0.log")) + list(Path(d).glob("*_S0.out")))
        if not hits:
            raise FileNotFoundError("目录中没有 *_S0.log / *_S0.out")
        return hits[0].stem[:-3]

    def analyze(self):
        d = self.resdir.get()
        name = self._name(d)
        res = analyze_excited(d, name)
        res["label"] = self.label.get().strip() or name
        self.current = res
        self.systems = [x for x in self.systems if x["label"] != res["label"]] + [res]
        t = self.text
        t.delete("1.0", "end")
        for w in res["warnings"]:
            t.insert("end", f"[警告] {w}\n")
        t.insert("end", f"{res['label']}\n{'类型':<16}{'跃迁':<8}{'E/eV':>8}{'λ/nm':>9}{'f':>8}   主要组态\n")
        for r in res["table"]:
            cfg = "; ".join(f"{a}({p:.1f}%)" for a, p in r["configs"])
            t.insert("end", f"{r['kind']:<16}{r['transition']:<8}{r['E_eV']:8.3f}{r['nm']:9.2f}{r['f']:8.4f}   {cfg}\n")
        for k, lab in (("stokes_shift_eV", "Stokes 位移 / eV"), ("rmsd_S0_S1", "RMSD S0–S1 / Å"),
                       ("rmsd_S0_T1", "RMSD S0–T1 / Å")):
            if k in res:
                t.insert("end", f"{lab}: {res[k]:.3f}\n")
        self.sys_lab.config(text="对比体系: " + ", ".join(x["label"] for x in self.systems))
        self.draw_spectra()

    def _curves(self):
        lo, hi = (float(v) for v in self.nm_range.get().split("-"))
        colors = ["#B22222", "#1F4E9E", "#D2691E", "#6A3D9A", "#2E8B57", "#8B8000", "#008B8B"]
        curves = []
        for i, sy in enumerate(self.systems):
            col = colors[i % len(colors)]
            if sy.get("absorption_states"):
                x, y = absorption_spectrum(sy["absorption_states"], float(self.fwhm_abs.get()), (lo, hi))
                curves.append({"label": f"Abs: {sy['label']}", "x": x, "y": y, "kind": "abs", "color": col})
            ph = next((r for r in sy["table"] if r["kind"] == "Phosphorescence"), None)
            if ph:
                x, y = emission_band(ph["E_eV"], float(self.fwhm_emi.get()), (lo, hi))
                curves.append({"label": f"Emi: {sy['label']}", "x": x, "y": y, "kind": "emi", "color": col})
        return curves

    def draw_spectra(self):
        if not self.systems:
            return
        ax = self.plot.new_axes()
        draw_spectra(ax, self._curves())
        self.plot.draw()

    def clear(self):
        self.systems = []
        self.sys_lab.config(text="对比体系: (空)")

    def export_spectra(self):
        if not self.systems:
            raise ValueError("请先读取结果")
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                                                                            ("SVG", "*.svg")])
        if p:
            from matplotlib.figure import Figure as _F
            fig = _F(figsize=(9, 6), dpi=300)
            draw_spectra(fig.add_subplot(111), self._curves())
            fig.tight_layout()
            fig.savefig(p, dpi=300, bbox_inches="tight")
            import csv as _csv
            with open(Path(p).with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as fh:
                w = _csv.writer(fh)
                cs = self._curves()
                w.writerow(["nm"] + [c["label"] for c in cs])
                for k in range(len(cs[0]["x"])):
                    w.writerow([cs[0]["x"][k]] + [c["y"][k] for c in cs])
            self.app.log(f"光谱图与数据已导出: {p}")

    def _overlay(self, pairs):
        ax = None
        self.plot.fig.clf()
        n = len(pairs)
        for i, (lab, a, b) in enumerate(pairs):
            r = rmsd(a, b, heavy_only=self.heavy.get())
            ax = self.plot.fig.add_subplot(1, n, i + 1, projection="3d")
            draw_overlay(ax, r["symbols"], r["reference"], r["aligned"], r["rmsd"])
            ax.text2D(0.5, 0.95, lab, transform=ax.transAxes, ha="center", fontsize=11)
            self.text.insert("end", f"{lab}: RMSD = {r['rmsd']:.4f} Å（最大原子偏移 {r['deviation'].max():.3f} Å, "
                                    f"原子 {int(np.argmax(r['deviation'])) + 1}）\n")
        self.plot.canvas.draw_idle()

    def calc_rmsd(self):
        a, b = read_geometry(self.ga.get()), read_geometry(self.gb.get())
        self._overlay([(f"{Path(self.ga.get()).stem} vs {Path(self.gb.get()).stem}", a, b)])

    def state_overlays(self):
        if not self.current:
            raise ValueError("请先读取激发态结果")
        g = self.current["geometries"]
        pairs = [(f"S0 vs {st}", g["S0"], g[st]) for st in ("S1", "T1") if "S0" in g and st in g]
        if not pairs:
            raise ValueError("结果中缺少几何结构")
        self._overlay(pairs)

    def rates(self):
        k = decay_rates(float(self.plqy.get()), float(self.tau.get()))
        self.text.insert("end", f"Φ = {self.plqy.get()}, τ = {self.tau.get()} μs → k_r = {k['k_r']:.3e} s⁻¹, "
                                f"k_nr = {k['k_nr']:.3e} s⁻¹\n")


# ==============================================================================================
# 转移积分
# ==============================================================================================
class TITab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        S = safe(app)
        panel = SectionPanel(self)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner
        self.prog = tk.StringVar(value="gaussian")
        pf = ttk.Frame(left)
        pf.pack(fill="x")
        ttk.Label(pf, text="计算程序:").pack(side="left")
        ttk.Radiobutton(pf, text="Gaussian (二聚体投影法)", value="gaussian", variable=self.prog).pack(side="left")
        ttk.Radiobutton(pf, text="ADF (片段法)", value="adf", variable=self.prog).pack(side="left")
        pf2 = ttk.Frame(left)
        pf2.pack(fill="x")
        ttk.Radiobutton(pf2, text="软件内计算 EHT (无需外部程序，秒级)", value="eht", variable=self.prog).pack(side="left")
        pf3 = ttk.Frame(left)
        pf3.pack(fill="x")
        ttk.Radiobutton(pf3, text="软件内 ZINDO/MOO (VOTCA izindo 方法，毫秒级)", value="zindo",
                        variable=self.prog).pack(side="left")
        zf = ttk.LabelFrame(left, text="ZINDO/MOO 与简并能级")
        zf.pack(fill="x", pady=3)
        self.z_orb = FileEntry(zf, "ZINDO 单体", filetypes=[("Gaussian fchk", "*.fchk *.fch"), ("所有文件", "*.*")],
                               width=24)
        self.z_orb.pack(fill="x")
        zg = ttk.Frame(zf)
        zg.pack(fill="x")
        self.deg = field(zg, 0, "简并阈值 / meV (0 = 不合并)", 0, width=8)
        ttk.Label(zf, text="ZINDO 单体留空 = 软件内 INDO/S 自洽场；也可用 Gaussian ZINDO 单体计算的 fchk (与 VOTCA 相同)。"
                           "简并阈值用于 Gaussian 投影法与 ZINDO：近简并能级的耦合按 √(ΣJ²/N_A N_B) 合并",
                  foreground=PALETTE["muted"], wraplength=440).pack(anchor="w")

        g = ttk.LabelFrame(left, text="Gaussian 设置")
        g.pack(fill="x", pady=3)
        self.g_method = field(g, 0, "泛函", "B3LYP", values=FUNCTIONALS, width=16)
        self.g_basis = field(g, 1, "基组", "6-31G(d)", values=BASIS_SETS, width=16)
        self.g_nproc = field(g, 0, "核数", 8, width=6, col=2)
        self.g_mem = field(g, 1, "内存", "8GB", width=6, col=2)
        self.g_extra = field(g, 2, "额外关键词", "", width=16)
        a = ttk.LabelFrame(left, text="ADF 设置")
        a.pack(fill="x", pady=3)
        self.a_func = field(a, 0, "泛函", "PW91", values=list(ADF_FUNCTIONALS), width=16)
        self.a_basis = field(a, 1, "基组", "TZP", values=ADF_BASIS, width=16)
        self.a_style = field(a, 2, "版本", "自动 (按安装版本)", values=["自动 (按安装版本)", "AMS2020+", "ADF2019 及以前"],
                             width=16)
        self.a_quality = field(a, 3, "数值精度", "Good", values=["Normal", "Good", "VeryGood"], width=16)

        i = ttk.LabelFrame(left, text="输入结构")
        i.pack(fill="x", pady=3)
        self.dimer = FileEntry(i, "二聚体文件", filetypes=STRUCT_TYPES, command=S(self.auto_na))
        self.dimer.pack(fill="x")
        r = ttk.Frame(i)
        r.pack(fill="x")
        self.na = field(r, 0, "单体A原子数", "", width=8)
        ttk.Button(r, text="自动识别", command=S(lambda: self.auto_na(self.dimer.get()))).grid(row=0, column=2)
        ttk.Label(i, text="—— 或分别给出两个单体 ——").pack()
        self.ma = FileEntry(i, "单体 A", filetypes=STRUCT_TYPES)
        self.ma.pack(fill="x")
        self.mb = FileEntry(i, "单体 B", filetypes=STRUCT_TYPES)
        self.mb.pack(fill="x")
        self.outdir = FileEntry(left, "输出目录", mode="dir")
        self.outdir.pack(fill="x")
        nf = ttk.Frame(left)
        nf.pack(fill="x")
        self.name = field(nf, 0, "任务名", "dimer")
        ttk.Button(left, text="生成输入", style="Accent.TButton", command=S(self.generate)).pack(fill="x", pady=3)
        ttk.Button(left, text="▶ 在本机运行…", command=S(lambda: app.open_runner(self.outdir.get() or "ti"))).pack(fill="x")

        res = ttk.LabelFrame(left, text="结果读取")
        res.pack(fill="x", pady=3)
        self.fa = FileEntry(res, "A.fchk", filetypes=FCHK_TYPES, command=self.fill_prefix)
        self.fa.pack(fill="x")
        self.fb = FileEntry(res, "B.fchk", filetypes=FCHK_TYPES)
        self.fb.pack(fill="x")
        self.fab = FileEntry(res, "AB.fchk", filetypes=FCHK_TYPES)
        self.fab.pack(fill="x")
        of = ttk.Frame(res)
        of.pack(fill="x")
        self.orbs = field(of, 0, "轨道", "HOMO-1,HOMO,LUMO,LUMO+1", width=24)
        self.adf_file = FileEntry(res, "ADF 结果", filetypes=[("ADF", "*.rkf *.t21 *.txt *.json"),
                                                            ("所有文件", "*.*")])
        self.adf_file.pack(fill="x")
        ttk.Button(res, text="计算 / 读取转移积分", style="Accent.TButton", command=S(self.calc)).pack(fill="x")

        cmp_ = ttk.LabelFrame(left, text="方法对比 (EHT / Gaussian / ADF)")
        cmp_.pack(fill="x", pady=4)
        ttk.Label(cmp_, text="二聚体目录 (含 dimers.json，可同时含 Gaussian fchk 与 ADF 结果)；留空则对比上面的单个二聚体",
                  foreground=PALETTE["muted"], wraplength=380).pack(anchor="w")
        self.cmp_dir = FileEntry(cmp_, "二聚体目录", mode="dir")
        self.cmp_dir.pack(fill="x")
        cf = ttk.Frame(cmp_)
        cf.pack(fill="x")
        self.fill_method = field(cf, 0, "填入迁移率所用方法", "自动 (精度最高)",
                                 values=["自动 (精度最高)", "Gaussian", "ADF", "EHT×比例因子", "EHT"], width=14)
        cb = ttk.Frame(cmp_)
        cb.pack(fill="x")
        ttk.Button(cb, text="对比三种方法", style="Accent.TButton", command=S(self.compare)).pack(side="left", fill="x",
                                                                                         expand=True)
        ttk.Button(cb, text="结果填入迁移率表", command=S(self.compare_to_mobility)).pack(side="left", fill="x", expand=True)
        ttk.Button(cmp_, text="直接读入迁移率表 (Gaussian / ADF 目录中的全部二聚体)", command=S(self.dir_to_mobility)
                   ).pack(fill="x", pady=(3, 0))
        ttk.Label(cmp_, text="对比完成后自动按所选方法填入迁移率表；“自动”按 Gaussian > ADF > EHT 选择",
                  foreground=PALETTE["muted"], wraplength=420).pack(anchor="w")
        an = ttk.LabelFrame(left, text="深入分析：多能级矩阵 / 原子对分解 (ti、tcal、CATNIP 方法)")
        an.pack(fill="x", pady=4)
        af = ttk.Frame(an)
        af.pack(fill="x")
        self.an_src = field(af, 0, "数据来源", "EHT (上方二聚体文件)",
                            values=["EHT (上方二聚体文件)", "Gaussian fchk (A、B、AB)"], width=22)
        self.an_n = field(af, 1, "能级数 n (H−n+1…L+n−1)", "2", values=["1", "2", "3", "4"], width=22)
        self.an_orb = field(af, 2, "原子对分解的轨道", "HOMO", values=["HOMO", "LUMO", "HOMO-1", "LUMO+1"], width=22)
        fch = [("fchk", "*.fchk *.fch"), ("所有文件", "*.*")]
        self.f_a = FileEntry(an, "单体 A fchk", filetypes=fch)
        self.f_b = FileEntry(an, "单体 B fchk", filetypes=fch)
        self.f_ab = FileEntry(an, "二聚体 fchk", filetypes=fch, command=lambda p: self._fill_ab(p))
        for w in (self.f_a, self.f_b, self.f_ab):
            w.pack(fill="x")
        ttk.Label(an, text="fchk 来源：本页“Gaussian”方式生成的三个转移积分单点计算 (G?_A.gjf、G?_B.gjf、G?_AB.gjf，"
                           "关键词含 nosymm iop(3/32=2)) 运行后 formchk 得到的 G?_A.fchk、G?_B.fchk、G?_AB.fchk；"
                           "“运行任务”会自动 formchk。不是重组能或几何优化的 fchk。选中二聚体 fchk 后自动补全 A、B。"
                           "单体与二聚体的原子顺序可以不同 (按坐标自动匹配)；原子对分解各项之和严格等于 J_eff",
                  foreground=PALETTE["muted"], wraplength=420).pack(anchor="w")
        ab_ = ttk.Frame(an)
        ab_.pack(fill="x")
        for txt, cmd in (("多能级矩阵", self.level_matrix), ("原子对分解", self.atom_pairs), ("导出 CSV…", self.export_apta)):
            ttk.Button(ab_, text=txt, command=S(cmd)).pack(side="left", fill="x", expand=True)
        self.an_last = None
        right = ttk.Frame(self, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.text = styled_text(right, app.mono, wrap="none", size=10, height=16)
        self.text.pack(fill="x")
        self.plot = PlotPanel(right, figsize=(7, 4))
        self.plot.pack(fill="both", expand=True, pady=(8, 0))
        self.cmp_rows, self.cmp_stats = None, None

    # ---------------- 深入分析 ----------------
    def _fill_ab(self, p):
        """选中 X_AB.fchk 时自动填入同目录的 X_A.fchk 与 X_B.fchk。"""
        p = Path(p)
        m = re.match(r"(.*)_AB(\.fch[k]?)$", p.name, re.I)
        if not m:
            return
        for w, suf in ((self.f_a, "_A"), (self.f_b, "_B")):
            q = p.with_name(m.group(1) + suf + m.group(2))
            if q.exists():
                w.set(q)

    def _an_data(self):
        from .ti_analysis import from_eht, from_fchk
        if self.an_src.get().startswith("Gaussian"):
            for w, src in ((self.f_a, self.fa), (self.f_b, self.fb), (self.f_ab, self.fab)):
                if not w.get() and src.get():                  # 沿用上方“读取 Gaussian 结果”中的文件
                    w.set(src.get())
            if not (self.f_a.get() and self.f_b.get() and self.f_ab.get()):
                raise ValueError("请选择单体 A、单体 B 与二聚体的 fchk 文件")
            d = from_fchk(self.f_a.get(), self.f_b.get(), self.f_ab.get())
            if d["reordered"]:
                self.app.log("二聚体原子顺序与单体不同，已按坐标自动匹配原子")
            return d
        if self.dimer.get():
            if not self.na.get():
                self.auto_na(self.dimer.get())
            at = read_geometry(self.dimer.get())
            a_, b_ = at[:int(self.na.get())], at[int(self.na.get()):]
        elif self.ma.get() and self.mb.get():
            a_, b_ = read_geometry(self.ma.get()), read_geometry(self.mb.get())
        else:
            raise ValueError("请在上方选择二聚体文件 (或两个单体文件)")
        from .crystal import check_dimer_geometry
        check_dimer_geometry(a_, b_)
        return from_eht(a_, b_)

    def level_matrix(self):
        from .ti_analysis import level_matrix
        d = self._an_data()
        rows = level_matrix(d, int(self.an_n.get()))
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"多能级转移积分矩阵 ({d['source']})；能量 eV，耦合 meV\n")
        t.insert("end", f"{'A':>8}{'B':>8}{'S':>10}{'e_i':>10}{'e_j':>10}{'J':>10}{'J_eff':>10}{'e_i,eff':>10}{'e_j,eff':>10}\n")
        for r in rows:
            t.insert("end", f"{r['A']:>8}{r['B']:>8}{r['S']:10.5f}{r['e_i_eV']:10.4f}{r['e_j_eV']:10.4f}{r['J_meV']:10.3f}"
                            f"{r['J_eff_meV']:10.3f}{r['e_i_eff_eV']:10.4f}{r['e_j_eff_eV']:10.4f}\n")
        t.insert("end", "J_eff = (J − ½(e_i + e_j)S)/(1 − S²)；e_eff 为对称正交化后的有效位点能 (Valeev 2006)\n")
        from .ti_analysis import degenerate_levels
        deg = degenerate_levels(d, int(self.an_n.get()))
        if deg:
            t.insert("end", "[注意] 近简并轨道: " + "，".join(f"单体 {w} 的 {a}/{b}" for w, a, b in deg) +
                     "。简并轨道可任意组合，单对耦合数值不唯一，应看简并子空间整体 (如 Σ|J_eff|²)\n")
        la = list(dict.fromkeys(r["A"] for r in rows))
        lb = list(dict.fromkeys(r["B"] for r in rows))
        M = np.array([[next(r["J_eff_meV"] for r in rows if r["A"] == x and r["B"] == y) for y in lb] for x in la])
        ax = self.plot.new_axes()
        vm = np.abs(M).max() or 1
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax.set_xticks(range(len(lb)))
        ax.set_xticklabels(lb, rotation=45, fontsize=8)
        ax.set_yticks(range(len(la)))
        ax.set_yticklabels(la, fontsize=8)
        for i in range(len(la)):
            for j in range(len(lb)):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(M[i, j]) > 0.6 * vm else "black")
        ax.set_xlabel("单体 B 轨道")
        ax.set_ylabel("单体 A 轨道")
        ax.set_title("有效转移积分 J_eff / meV")
        self.plot.fig.colorbar(im, ax=ax, shrink=0.8)
        self.plot.draw()
        self.an_last = ("levels", rows)
        return rows

    def atom_pairs(self):
        from .render import MolRenderer
        from .ti_analysis import apta, top_pairs
        from matplotlib import cm
        d = self._an_data()
        orb = self.an_orb.get()
        r = apta(d, orb)
        sa = [d["symbols"][k] for k in d["atomsA"]]
        sb = [d["symbols"][k] for k in d["atomsB"]]
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"原子对转移积分分析 ({d['source']}，{orb}–{orb})\n")
        t.insert("end", f"Σ u_αβ = {r['total_meV']:.3f} meV (J_eff = {r['J_eff_meV']:.3f} meV)；正贡献 {r['positive_meV']:.2f}，"
                        f"负贡献 {r['negative_meV']:.2f} meV\n贡献最大的原子对：\n")
        tops = top_pairs(r, sa, sb, 12)
        for x, y, v in tops:
            t.insert("end", f"   {x:>6} – {y:<6} {v:9.3f} meV\n")
        t.insert("end", "方法：Ozawa, Okada, Matsui, Sci. Technol. Adv. Mater. 2024, 25, 2354652 (tcal)\n")
        fig = self.plot.fig
        fig.clf()
        ax3 = fig.add_axes([0.0, 0.02, 0.5, 0.92])
        ax2 = fig.add_axes([0.58, 0.14, 0.36, 0.72])
        contrib = np.zeros(len(d["symbols"]))
        contrib[d["atomsA"]] = r["per_atom_A"]
        contrib[d["atomsB"]] = r["per_atom_B"]
        vm = np.abs(contrib).max() or 1
        cmap = matplotlib.colormaps["RdBu_r"]
        colors = [cmap(0.5 + 0.5 * c / vm) for c in contrib]
        rend = MolRenderer(ax3)
        rend.background = "白色"
        X = np.asarray(d["coords"])
        umax = max(abs(v) for *_, v in tops) or 1
        rend.cylinders = []
        U = r["U"]
        for k in np.argsort(-np.abs(U), axis=None)[:8]:
            i, j = np.unravel_index(k, U.shape)
            rend.cylinders.append((X[d["atomsA"][i]], X[d["atomsB"][j]], 0.12 * abs(U[i, j]) / umax + 0.03,
                                   "#C0392B" if U[i, j] > 0 else "#1F5FA6"))
        rend.set_scene(d["symbols"], X, colors=colors, reset=True)
        cen = X.mean(0)
        rel = X - cen
        normal = np.linalg.svd(rel - rel.mean(0))[2][2]
        side = np.cross(normal, [0, 0, 1.0]) if abs(normal[2]) < 0.9 else np.cross(normal, [0, 1.0, 0])
        rend.view_along(side + 0.3 * normal, normal)
        rend.title = f"{orb}：原子贡献 (红 +，蓝 −)"
        rend.draw()
        im = ax2.imshow(U, cmap="RdBu_r", vmin=-np.abs(U).max(), vmax=np.abs(U).max(), aspect="auto")
        ax2.set_xlabel("单体 B 原子序号")
        ax2.set_ylabel("单体 A 原子序号")
        ax2.set_title("u$_{αβ}$ / meV", fontsize=10)
        fig.colorbar(im, ax=ax2, shrink=0.8)
        self.plot.draw()
        self.an_last = ("apta", r, sa, sb)
        return r

    def export_apta(self, path=None):
        from .ti_analysis import write_apta_csv
        if not self.an_last:
            raise ValueError("请先进行多能级矩阵或原子对分解")
        p = path or filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return None
        if self.an_last[0] == "apta":
            write_apta_csv(p, *self.an_last[1:])
        else:
            import csv as _csv
            rows = self.an_last[1]
            with open(p, "w", newline="", encoding="utf-8-sig") as fh:
                w = _csv.writer(fh)
                w.writerow(list(rows[0]))
                for r_ in rows:
                    w.writerow(list(r_.values()))
        self.app.log(f"已导出 {p}")
        return p

    def settings(self):
        if self.prog.get() in ("eht", "xtb", "zindo"):
            return self.prog.get(), {}
        if self.prog.get() == "gaussian":
            return "gaussian", dict(method=self.g_method.get(), basis=self.g_basis.get(),
                                    nproc=int(self.g_nproc.get()), mem=self.g_mem.get(), extra=self.g_extra.get())
        return "adf", dict(functional=self.a_func.get(), basis=self.a_basis.get(),
                           style=self._adf_style(),
                           numerical_quality=self.a_quality.get())

    def compare(self):
        from .ticompare import compare_dimer, compare_dir, format_rows, statistics
        d = self.cmp_dir.get()
        xexe = None                                  # 方法对比只用 EHT、Gaussian 与 ADF
        if d:
            rows = None
        else:
            if not self.dimer.get():
                raise ValueError("请选择二聚体目录，或在上面给出二聚体文件")
            if not self.na.get():
                self.auto_na(self.dimer.get())
            pre = self.fa.get()[:-len("_A.fchk")] if self.fa.get().endswith("_A.fchk") else None
            adf = self.adf_file.get() or None
            rows = [compare_dimer(self.dimer.get(), int(self.na.get()), pre, None, label=Path(self.dimer.get()).stem,
                                  xtb_exe=xexe)]
            if adf:
                from .adf import read_adf_transfer_integrals
                r = read_adf_transfer_integrals(adf)
                rows[0]["ADF"] = {"hole": r.get("hole", {}).get("V_meV"), "electron": r.get("electron", {}).get("V_meV")}
        ams_ = self._ams_info()

        def _work():
            return compare_dir(d, xtb_exe=xexe, ams_info=ams_) if rows is None else rows

        self.app.run_bg("对比三种方法", _work, self._compare_done)

    def _compare_done(self, rows):
        from .ticompare import format_rows, statistics
        self.cmp_rows = rows
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", format_rows(rows) + "\n")
        for r in rows:
            for e in r["errors"]:
                t.insert("end", f"[警告] {r['label']} {e}\n")
        self.cmp_stats = {}
        for ref in ("Gaussian", "ADF"):
            st = statistics(rows, ref=ref, test="EHT")
            if st:
                self.cmp_stats[ref] = st
                for car, q in st.items():
                    t.insert("end", f"EHT vs {ref} ({'空穴' if car == 'hole' else '电子'}, {q['n']} 个): 相关系数 "
                                    f"{q['pearson_r']:.3f}, 比例因子 k = {q['scale_k']:.3f} (|J_{ref}| ≈ k|J_EHT|), "
                                    f"标定后平均相对偏差 {q['mean_rel_err_after_scaling'] * 100:.1f}%\n")
        st = statistics(rows, ref="Gaussian", test="ADF")
        for car, q in st.items():
            t.insert("end", f"ADF vs Gaussian ({'空穴' if car == 'hole' else '电子'}): 相关系数 {q['pearson_r']:.3f}, "
                            f"平均比值 {q['mean_ratio']:.3f}\n")
        self.draw_compare()
        self._auto_fill()

    def draw_compare(self):
        rows = self.cmp_rows
        fig = self.plot.fig
        fig.clf()
        cols = {"EHT": "#2B4A77", "Gaussian": "#C0392B", "ADF": "#2E8B57"}
        for k, car in enumerate(("hole", "electron")):
            ax = fig.add_subplot(1, 3, k + 1)
            x = np.arange(len(rows))
            for m_i, m in enumerate(("EHT", "Gaussian", "ADF")):
                vals = [abs((r.get(m) or {}).get(car) or np.nan) for r in rows]
                if not np.all(np.isnan(vals)):
                    ax.bar(x + (m_i - 1.5) * 0.2, np.nan_to_num(vals), 0.2, label=m, color=cols[m])
            ax.set_xticks(x)
            ax.set_xticklabels([r["label"] for r in rows], rotation=45, fontsize=8)
            ax.set_ylabel("|J| / meV")
            ax.set_title("空穴" if car == "hole" else "电子")
            if k == 0:
                ax.legend(fontsize=8)
        ax = fig.add_subplot(1, 3, 3)
        mx = 1
        for ref, mk in (("Gaussian", "o"), ("ADF", "s")):
            for car, col in (("hole", "#B22222"), ("electron", "#2B5DA8")):
                pts = [(abs(r[ref][car]), abs(r["EHT"][car])) for r in rows
                       if r.get(ref) and r[ref].get(car) is not None]
                if pts:
                    a, b = np.array(pts).T
                    ax.scatter(a, b, marker=mk, color=col, label=f"{ref} {'h' if car == 'hole' else 'e'}", s=28)
                    mx = max(mx, a.max(), b.max())
        ax.plot([0, mx * 1.05], [0, mx * 1.05], "k--", lw=1)
        ax.set_xlabel("|J| DFT / meV")
        ax.set_ylabel("|J| EHT / meV")
        ax.set_title("EHT 与 DFT 对照")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
        self.plot.draw()

    def _auto_fill(self):
        d = self.cmp_dir.get()
        if d and (Path(d) / "dimers.json").exists() and self.cmp_rows:
            try:
                self.compare_to_mobility()
                self.app.log("对比结果已自动填入“迁移率”页，可直接计算迁移率")
            except Exception as exc:  # noqa: BLE001
                self.app.log(f"自动填入迁移率表失败: {exc}")

    def single_to_mobility(self):
        """单个二聚体的计算结果 (EHT / ZINDO / Gaussian / ADF) 填入迁移率表：
        表中已有路径时写入选中行 (未选则全部行) 的 Ve / Vh；表为空时按二聚体质心矢量新建一行。"""
        last = self.app.data.get("last_J")
        if not last:
            raise ValueError("还没有转移积分结果：请先点“计算 / 读取转移积分”")
        mt = self.app.mob_tab
        if mt.tree.get_children():
            mt.fill_last_j()
            n = len(mt.tree.selection() or mt.tree.get_children())
            self.app.log(f"转移积分已写入迁移率表的 {n} 行")
        else:
            a, b = self._dimer_atoms()
            ca = np.mean([x[1:] for x in a], axis=0)
            cb = np.mean([x[1:] for x in b], axis=0)
            sc = mt._scale() / 1000.0
            mt.set_paths([{"name": Path(self.dimer.get() or "dimer").stem, "vector": list(np.round(cb - ca, 4)),
                           "Ve": abs(last["electron"]) * sc if last.get("electron") is not None else "",
                           "Vh": abs(last["hole"]) * sc if last.get("hole") is not None else ""}])
            self.app.log("已按二聚体质心矢量在迁移率表中新建一行 (晶体请用“结构”中的近邻矢量)")
        self.app.show_page("迁移率")

    def compare_to_mobility(self):
        if not self.cmp_rows or not self.cmp_dir.get():
            return self.single_to_mobility()
        meth = self.fill_method.get()
        info = json.loads((Path(self.cmp_dir.get()) / "dimers.json").read_text())
        st = self.cmp_stats.get("Gaussian") or self.cmp_stats.get("ADF") or {}
        kk = {c: st.get(c, {}).get("scale_k", 1.0) for c in ("hole", "electron")}
        rows = {int(r["label"][1:]): r for r in self.cmp_rows}

        def val(r, m):
            v = r.get("EHT" if m == "EHT×比例因子" else m) or {}
            if v.get("hole") is None or v.get("electron") is None:
                return None
            f = kk if m == "EHT×比例因子" else {"hole": 1.0, "electron": 1.0}
            return {c: v[c] * f[c] for c in ("hole", "electron")}
        eht_best = "EHT×比例因子" if st else "EHT"
        if meth.startswith("自动"):
            src = next((m for m in ("Gaussian", "ADF") if all(val(r, m) for r in rows.values())), eht_best)
        else:
            src = meth
        order = [src] + [m for m in ("Gaussian", "ADF", eht_best) if m != src]
        by, used = {}, {}
        for gid, r in rows.items():           # 个别组缺数据时用次一级方法补上
            for m in order:
                v = val(r, m)
                if v:
                    by[gid], used[gid] = v, m
                    break
        other = {g_: m for g_, m in used.items() if m != src}
        if other:
            self.app.log(f"以下组缺少 {src} 结果，改用其它方法: " + ", ".join(f"G{g_}:{m}" for g_, m in sorted(other.items())))
        mt = self.app.mob_tab
        sc = mt._scale() / 1000.0
        paths = []
        for i, nb in enumerate(info["neighbors"]):
            v = by.get(nb["group"]) or {}
            paths.append({"name": f"G{nb['group']}-{i + 1}", "vector": nb["vector"],
                          "Ve": abs(v["electron"]) * sc if v.get("electron") is not None else "",
                          "Vh": abs(v["hole"]) * sc if v.get("hole") is not None else ""})
        mt.set_paths(paths)
        self.app.log(f"已用 {src} 的转移积分填入迁移率表 ({len(paths)} 条路径)")
        return src

    def dir_to_mobility(self):
        """不做对比，直接把 Gaussian / ADF 目录中全部二聚体的转移积分读入迁移率表。"""
        d = self.cmp_dir.get() or self.outdir.get()
        if not d:
            d = filedialog.askdirectory(title=T("选择二聚体计算目录 (含 dimers.json)"))
        if not d:
            return
        if not (Path(d) / "dimers.json").exists():
            raise FileNotFoundError("所选目录中没有 dimers.json：请选择“结构/晶体”页导出二聚体时的目录")
        prog = MobilityTab.detect_program(d)
        if not prog:
            raise FileNotFoundError("目录中没有 Gaussian (G*_A.fchk / G*_AB.fchk) 或 ADF (G*.out / G*_full.t21) 结果")
        out = self.app.mob_tab.import_dir(d, prog)
        self.app.show_page("迁移率")
        return out

    def auto_na(self, path):
        if not path:
            return
        atoms = read_geometry(path)
        n_a, reordered, new = split_dimer(atoms)
        if reordered:
            newp = Path(path).with_name(Path(path).stem + "_sorted.xyz")
            write_xyz(newp, new, f"reordered dimer nA={n_a}")
            self.dimer.set(newp)
            self.app.log(f"原文件两个分子的原子交错排列，已重排并保存为 {newp}")
        self.na.set(n_a)

    def fill_prefix(self, path):
        p = Path(path)
        if p.stem.endswith("_A"):
            base = p.stem[:-2]
            self.fb.set(p.with_name(base + "_B" + p.suffix))
            self.fab.set(p.with_name(base + "_AB" + p.suffix))

    def generate(self):
        program, st = self.settings()
        if program in ("eht", "xtb", "zindo"):
            self.calc()      # EHT / xtb / ZINDO 不需要输入文件，直接计算
            return
        out = self.outdir.get() or "ti"
        dimer = self.dimer.get() or None
        kw = dict(dimer=dimer, n_a=int(self.na.get()) if dimer else None,
                  mono_a=self.ma.get() or None, mono_b=self.mb.get() or None, name=self.name.get())
        self.text.delete("1.0", "end")
        if program == "gaussian":
            stems = generate_ti_inputs(out, **kw, **st)
            for s in stems:
                self.text.insert("end", f"======== {s}.gjf (坐标省略) ========\n")
                self.text.insert("end", "\n".join((Path(out) / f"{s}.gjf").read_text().splitlines()[:8]) + "\n...\n")
            self.fa.set(Path(out) / f"{kw['name']}_A.fchk")
            self.fill_prefix(str(Path(out) / f"{kw['name']}_A.fchk"))
        else:
            r = generate_adf_inputs(out, **kw, **st)
            self.text.insert("end", Path(r["script"]).read_text())
            self.adf_file.set(r["result_file"])
        self.app.log(f"转移积分输入已生成于 {out}")

    def _dimer_atoms(self):
        if self.dimer.get():
            if not self.na.get():
                self.auto_na(self.dimer.get())
            at = read_geometry(self.dimer.get())
            return at[:int(self.na.get())], at[int(self.na.get()):]
        return read_geometry(self.ma.get()), read_geometry(self.mb.get())

    def _zindo(self, t):
        """ZINDO/MOO：输入在界面线程读取，计算放到后台线程，结果回到界面线程显示。"""
        from .zindo import MOO, dimer_projection
        a_, b_ = self._dimer_atoms()
        deg = float(self.deg.get() or 0) / 1000
        zorb = self.z_orb.get() or None

        def work():
            sa, sb = [x[0] for x in a_], [x[0] for x in b_]
            bb = b_
            if sa != sb:
                from .polar import match_atoms
                perm, _ = match_atoms(sa, np.array([x[1:] for x in a_]), sb, np.array([x[1:] for x in b_]))
                bb = [b_[k] for k in perm]
            Xa, Xb = np.array([x[1:] for x in a_], float), np.array([x[1:] for x in bb], float)
            moo = MOO(sa, Xa, zorb, degeneracy=deg)
            from .morphology import kabsch_batch
            _, _, rm = kabsch_batch(Xb[None], Xa)
            j = moo.coupling_atoms(Xa, Xb)
            jd = dimer_projection(sa, Xa, Xb, deg) if not zorb else None
            return dict(source=moo.source, rm=rm[0], j=j, jd=jd,
                        nlev={c: len(moo.levels(c)) for c in ("hole", "electron")})

        def done(r):
            j, jd = r["j"], r["jd"]
            t.insert("end", f"ZINDO/MOO 转移积分 (单体轨道：{r['source']})\n")
            if r["rm"] > 0.05:
                t.insert("end", f"[提示] B 与 A 的几何不完全相同 (叠合 RMSD {r['rm']:.2f} Å)，MOO 按刚性单体处理\n")
            t.insert("end", f"{'':<10}{'MOO J/meV':>12}" + (f"{'二聚体 INDO/S 投影':>16}" if jd else "") + "\n")
            for car, lab in (("hole", "HOMO (空穴)"), ("electron", "LUMO (电子)")):
                t.insert("end", f"{lab:<10}{1000 * j[car]:12.3f}" + (f"{1000 * jd[car]:16.3f}" if jd else "") + "\n")
            if deg > 0:
                t.insert("end", f"简并阈值 {deg * 1000:g} meV：空穴 {r['nlev']['hole']} 个、电子 "
                                f"{r['nlev']['electron']} 个能级参与合并\n")
            t.insert("end", "MOO 不做二聚体自洽场 (Kirkpatrick 2008)；与二聚体 INDO/S 投影相差通常在 10% 以内。"
                            "定量结论请与 DFT 投影法核对。\n"
                            "注意：ZINDO 的 J 是正交化轨道之间的耦合，可直接与 EHT 表中的 J_eff 比较；"
                            "EHT 的 J_eff 对能量零点敏感，二者差别较大时以 DFT (Gaussian / ADF) 为准。\n")
            self.app.data["last_J"] = {"hole": 1000 * j["hole"], "electron": 1000 * j["electron"]}
        self.app.run_bg("ZINDO/MOO 转移积分", work, done)

    def _eht(self):
        from .eht import eht_from_file
        if self.dimer.get():
            if not self.na.get():
                self.auto_na(self.dimer.get())
            return eht_from_file(self.dimer.get(), int(self.na.get()),
                                 orbitals=[o.strip() for o in self.orbs.get().split(",") if o.strip()])
        return eht_from_file(mono_a=self.ma.get(), mono_b=self.mb.get(),
                             orbitals=[o.strip() for o in self.orbs.get().split(",") if o.strip()])

    def _adf_style(self):
        st = self.a_style.get()
        if st.startswith("AMS"):
            return "ams"
        if st.startswith("ADF2019"):
            return "classic"
        from .programs import find_ams
        info = find_ams(self.app.config_data.get("ams_home") or None)
        if info:
            self.app.log(f"ADF 版本自动识别: {info['version']} → {'AMS2020+' if info['kind'] == 'ams' else 'ADF2019 及以前'} 格式")
            return info["kind"]
        self.app.log("[提示] 未找到 ADF 安装，按 AMS2020+ 格式生成；运行时若检测到旧版会自动转换")
        return "ams"

    def _ams_info(self):
        from .programs import find_ams
        return find_ams(self.app.config_data.get("ams_home") or None)

    def _xtb_exe(self):
        from .programs import find_xtb
        exe = find_xtb(self.app.config_data.get("xtb_exe") or None)
        if not exe:
            raise FileNotFoundError("没有找到 xtb，请在 工具 → 程序路径设置 中指定 (xtb ≥ 6.7 支持 DIPRO)")
        return exe

    def calc(self):
        t = self.text
        t.delete("1.0", "end")
        if self.prog.get() == "xtb":
            from .programs import xtb_dipro
            if self.dimer.get():
                if not self.na.get():
                    self.auto_na(self.dimer.get())
                at = read_geometry(self.dimer.get())
                a_, b_ = at[:int(self.na.get())], at[int(self.na.get()):]
            else:
                a_, b_ = read_geometry(self.ma.get()), read_geometry(self.mb.get())
            wd = Path(self.outdir.get() or Path(self.dimer.get() or self.ma.get()).parent) / "xtb_dipro"
            r = xtb_dipro(a_, b_, self._xtb_exe(), wd)
            t.insert("end", f"xtb DIPRO (GFN1-xTB)\n  HOMO–HOMO |J_eff| = {r['hole']:.1f} meV\n"
                            f"  LUMO–LUMO |J_eff| = {r['electron']:.1f} meV\n  窗口内总耦合: 空穴 {r['total_hole']:.1f}, "
                            f"电子 {r['total_electron']:.1f} meV\n  输出: {r['output']}\n")
            self.app.data["last_J"] = {"hole": r["hole"], "electron": r["electron"]}
            return
        if self.prog.get() == "zindo":
            self._zindo(t)
            return
        if self.prog.get() == "eht":
            if self.dimer.get() and not self.na.get():
                self.auto_na(self.dimer.get())
            from .eht import eht_from_file
            orbs = [o.strip() for o in self.orbs.get().split(",") if o.strip()]
            dimer, na, ma, mb = self.dimer.get(), self.na.get(), self.ma.get(), self.mb.get()   # 界面线程中读取输入

            def work():
                if dimer:
                    return eht_from_file(dimer, int(na), orbitals=orbs)
                return eht_from_file(mono_a=ma, mono_b=mb, orbitals=orbs)
            self.app.run_bg("软件内 EHT 转移积分", work, self._show_eht)
            return
        return self._calc_dft(t)

    def _show_eht(self, res):
        t = self.text
        t.delete("1.0", "end")
        for w in res["warnings"]:
                t.insert("end", f"[警告] {w}\n")
        if True:
            t.insert("end", f"软件内 EHT 计算 ({res['nbf']} 个基函数)\n")
            t.insert("end", f"{'轨道':<9}{'J_eff/meV':>12}{'J/meV':>12}{'S_AB':>11}{'e_A/eV':>10}{'e_B/eV':>10}\n")
            for lab, r in res["results"].items():
                if isinstance(r, dict):
                    t.insert("end", f"{lab:<9}{r['J_eff_meV']:12.3f}{r['J_raw_meV']:12.3f}{r['S_AB']:11.5f}"
                                    f"{r['e_A_eV']:10.4f}{r['e_B_eV']:10.4f}\n")
            t.insert("end", f"能级分裂法 ESD: 空穴 {res['results']['ESD_hole_meV']:.3f} meV, "
                            f"电子 {res['results']['ESD_electron_meV']:.3f} meV（对称二聚体且轨道非简并时才与投影法一致）\n"
                            "EHT 为半经验方法，建议用“三种方法对比”与 DFT 结果核对后再用于定量结论。\n")
            self.app.data["last_J"] = {"hole": res["results"]["HOMO"]["J_eff_meV"],
                                       "electron": res["results"]["LUMO"]["J_eff_meV"]}

    def _calc_dft(self, t):
        if self.prog.get() == "gaussian":
            res = calc_transfer_integrals(self.fa.get(), self.fb.get(), self.fab.get(),
                                          [o.strip() for o in self.orbs.get().split(",") if o.strip()],
                                          degeneracy=float(self.deg.get() or 0) / 1000)
            for w in res["warnings"]:
                t.insert("end", f"[警告] {w}\n")
            t.insert("end", f"{'轨道':<9}{'J_eff/meV':>12}{'J/meV':>12}{'S_AB':>11}{'e_A/eV':>10}{'e_B/eV':>10}\n")
            for lab, r in res["results"].items():
                t.insert("end", f"{lab:<9}{r['J_eff_meV']:12.3f}{r['J_raw_meV']:12.3f}{r['S_AB']:11.5f}"
                                f"{r['e_A_eV']:10.4f}{r['e_B_eV']:10.4f}\n")
            last = {"hole": res["results"].get("HOMO", {}).get("J_eff_meV"),
                    "electron": res["results"].get("LUMO", {}).get("J_eff_meV")}
            for lab, car in (("HOMO", "hole"), ("LUMO", "electron")):
                r = res["results"].get(lab, {})
                if "J_deg_meV" in r and max(r["deg_levels"]) > 1:
                    last[car] = r["J_deg_meV"]
        else:
            res = read_adf_transfer_integrals(self.adf_file.get(), self._ams_info())
            t.insert("end", f"{'通道':<28}{'V/meV':>11}{'Vtot/meV':>11}{'J/meV':>11}{'S':>10}"
                            f"{'e1/eV':>10}{'e2/eV':>10}\n")
            for ch, r in res.items():
                t.insert("end", f"{ch:<28}{r.get('V_meV', np.nan):11.3f}{r.get('Vtot_meV', np.nan):11.3f}"
                                f"{r.get('J_meV', np.nan):11.3f}{r.get('S', np.nan):10.5f}"
                                f"{r.get('e1_eV', np.nan):10.4f}{r.get('e2_eV', np.nan):10.4f}\n")
            last = {"hole": res.get("hole", {}).get("V_meV"), "electron": res.get("electron", {}).get("V_meV")}
        t.insert("end", "\n空穴传输取 HOMO / hole，电子传输取 LUMO / electron。\n"
                        "点“结果填入迁移率表”写入迁移率页 (表中已有路径时写入选中行)。\n")
        self.app.data["last_J"] = last


# ==============================================================================================
# 迁移率
# ==============================================================================================
class MobilityTab(ttk.Frame):
    """Marcus 跳跃 + 角分辨迁移率。μ_φ = e/(2k_BT) Σ r_i² W_i P_i cos²γ_i cos²(θ_i − φ)"""
    M3D = ["Wen 公式 (与二维极坐标图一致)", "迁移率张量 (无规行走，同 KMC)"]
    COLS = ("名称", "x/Å", "y/Å", "z/Å", "r/Å", "θ/°", "γ/°", "Ve", "Vh", "数目")

    def __init__(self, parent, app):
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.compare, self.last = [], None
        S = safe(app)
        panel = SectionPanel(self, width=560)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner
        p = ttk.LabelFrame(left, text="参数")
        p.pack(fill="x")
        self.lam_e = field(p, 0, "电子重组能 λe / eV", 0.2)
        self.lam_h = field(p, 1, "空穴重组能 λh / eV", 0.2)
        self.temp = field(p, 2, "温度 / K", "300")
        ttk.Label(p, text="(可写 300 或 200:400:20 或 250,300,350)", foreground=PALETTE["muted"]).grid(
            row=3, column=0, columnspan=2, sticky="w")
        self.plane = field(p, 0, "传输平面", "xy", values=["xy", "xz", "yz"], width=8, col=2)
        try:                                       # 换传输平面时，表中由 x, y, z 给出的路径重新计算 r / θ / γ
            self.plane.trace_add("write", lambda *a: self._refresh_angles())
        except Exception:  # noqa: BLE001
            pass
        self.unit = field(p, 1, "V 单位", "meV", values=["meV", "eV"], width=8, col=2)
        self.dim = field(p, 2, "维度 n", 3, values=["3", "2", "1"], width=8, col=2)
        self.dg = field(p, 4, "ΔG / eV", 0.0)

        self.tree = EditableTree(left, self.COLS, widths=[70, 52, 52, 52, 56, 56, 44, 64, 64, 40], height=7)
        self.tree.pack(fill="both", expand=True, pady=4)
        ttk.Label(left, text="双击编辑。θ 为平面内角度，γ 为与平面夹角；"
                             "只给 x,y,z 时 r/θ/γ 按“传输平面”自动计算。", foreground=PALETTE["muted"],
                  wraplength=540).pack(anchor="w")
        b = ttk.Frame(left)
        b.pack(fill="x")
        acts = [("添加行", self.add_row), ("删除选中", self.del_rows), ("清空", lambda: self.tree.clear()),
                ("填入最近一次 V", self.fill_last_j), ("从二聚体目录导入", self.import_dir),
                ("导入 CSV 表格", self.import_csv), ("加载配置", self.load_cfg), ("保存配置", self.save_cfg)]
        for i, (txt, cmd) in enumerate(acts):
            ttk.Button(b, text=txt, command=S(cmd)).grid(row=i // 4, column=i % 4, sticky="we", padx=1, pady=1)
        ttk.Button(left, text="计算迁移率 (电子 + 空穴)", style="Accent.TButton", command=S(self.compute)).pack(
            fill="x", pady=4)
        c = ttk.LabelFrame(left, text="角分辨迁移率极坐标图")
        c.pack(fill="x")
        self.label = field(c, 0, "体系名称", "Ir-BB")
        cb = ttk.Frame(c)
        cb.grid(row=1, column=0, columnspan=2, sticky="we")
        for txt, cmd in (("加入对比", self.add_compare), ("清空对比", self.clear_compare), ("导出图片…", self.export_fig)):
            ttk.Button(cb, text=txt, command=S(cmd)).pack(side="left", fill="x", expand=True)
        self.cmp_lab = ttk.Label(c, text="对比列表: (空)", foreground=PALETTE["muted"])
        self.cmp_lab.grid(row=2, column=0, columnspan=2, sticky="w")
        c3 = ttk.LabelFrame(left, text="三维各向异性迁移率")
        c3.pack(fill="x", pady=3)
        self.m3_model = field(c3, 0, "模型", self.M3D[0], values=self.M3D, width=26)
        cb3 = ttk.Frame(c3)
        cb3.grid(row=1, column=0, columnspan=2, sticky="we")
        ttk.Button(cb3, text="三维迁移率图", command=S(self.compute3d)).pack(side="left", fill="x", expand=True)
        ttk.Button(cb3, text="导出 (PNG + 可旋转 HTML)…", command=S(self.export3d)).pack(side="left", fill="x", expand=True)
        ttk.Label(c3, text="μ(n̂) = n̂ᵀMn̂：曲面半径为该方向的迁移率；彩色实线为传输平面内的截面 (即二维极坐标图)，"
                           "虚线为主轴，灰色箭头为晶轴 a、b、c (结构页打开晶体时)。需要路径的 x、y、z。",
                  foreground=PALETTE["muted"], wraplength=520).grid(row=2, column=0, columnspan=2, sticky="w")

        k = ttk.LabelFrame(left, text="KMC 模拟 (参照 VOTCA-CTP)")
        k.pack(fill="x", pady=4)
        kf = ttk.Frame(k)
        kf.pack(fill="x")
        self.k_src = field(kf, 0, "近邻图", "迁移率表", values=["迁移率表", "结构页晶体 (全部分子)"], width=16)
        self.k_car = field(kf, 1, "载流子", "电子", values=["电子", "空穴"], width=16)
        self.k_rate = field(kf, 2, "速率", "Marcus", values=["Marcus", "Jortner"], width=16)
        self.k_hw = field(kf, 3, "ħω / eV (Jortner)", 0.2, width=16)
        self.k_lout = field(kf, 4, "外重组能 λout / eV", 0.0, width=16)
        self.k_lmodel = field(kf, 5, "λout 模型", "常数", values=["常数", "球模型"], width=16)
        self.k_pekar = field(kf, 0, "Pekar 因子 cp", 0.05, width=8, col=2)
        self.k_rad = field(kf, 1, "球半径 R / Å", 5.0, width=8, col=2)
        self.k_field = field(kf, 2, "电场 V/cm (x y z)", "0 0 0", width=12, col=2)
        self.k_sigma = field(kf, 3, "无序 σ / eV", 0.0, width=8, col=2)
        self.k_cell = field(kf, 4, "无序超胞 N", 4, width=8, col=2)
        self.k_walk = field(kf, 5, "轨迹数 / 步数", "2000 3000", width=12, col=2)
        ttk.Button(k, text="运行 KMC", style="Accent.TButton", command=S(self.run_kmc)).pack(fill="x", pady=2)
        ttk.Label(k, text="零场: 由均方位移得到扩散张量 D，μ = D/kT；有场: μ = <v>·F̂/|F|。"
                          "有能量无序时载流子从 Boltzmann 平衡分布出发、先经平衡期，并按固定模拟时间统计 (系综平均)。",
                  foreground=PALETTE["muted"], wraplength=540).pack(anchor="w")
        kv = ttk.Frame(k)
        kv.pack(fill="x")
        ttk.Button(kv, text="KMC 精度自检 (与精确解对比)", command=S(self.kmc_selfcheck)).pack(side="left", fill="x", expand=True)
        ttk.Button(kv, text="导出自检结果…", command=S(self.export_selfcheck)).pack(side="left", fill="x", expand=True)
        self.selfcheck_rows = None

        right = ttk.Frame(self, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.plot = PlotPanel(right, figsize=(6, 4.6))
        self.plot.pack(fill="both", expand=True)
        self.text = styled_text(right, app.mono, height=15, wrap="none", size=9)
        self.text.pack(fill="x", pady=(8, 0))

    # ---------------- 表格 ----------------
    def _row(self, d):
        v = d.get("vector") or ["", "", ""]
        fmt = lambda x: "" if x in (None, "") else (f"{x:.4f}" if isinstance(x, float) else str(x))  # noqa: E731
        r, th, ga = d.get("r"), d.get("theta"), d.get("gamma")
        if d.get("vector"):                       # 有 x, y, z 时 r / θ / γ 总按矢量与当前传输平面计算 (忽略文件中的旧值)
            try:
                r, th, ga = (float(t) for t in plane_angles([float(t) for t in v], self.plane.get()))
            except Exception:  # noqa: BLE001
                pass
        return [d.get("name", ""), *(fmt(float(t)) if t != "" else "" for t in v), fmt(r),
                fmt(th), fmt(ga), fmt(d.get("Ve")), fmt(d.get("Vh")), d.get("count", 1)]

    def _refresh_angles(self):
        tree = getattr(self, "tree", None)
        if tree is None:
            return
        try:
            for iid in tree.get_children():
                vals = list(tree.item(iid, "values"))
                if all(str(t).strip() for t in vals[1:4]):
                    r, th, ga = plane_angles([float(t) for t in vals[1:4]], self.plane.get())
                    vals[4:7] = [f"{r:.4f}", f"{th:.4f}", f"{ga:.4f}"]
                    tree.item(iid, values=vals)
        except Exception:  # noqa: BLE001
            pass

    def set_paths(self, paths):
        self.tree.clear()
        for d in paths:
            self.tree.insert("", "end", values=self._row(d))
        stripe(self.tree)

    def add_row(self):
        n = len(self.tree.get_children()) + 1
        self.tree.insert("", "end", values=(f"P{n}", "", "", "", "8.0", "0", "0", "", "", "1"))
        stripe(self.tree)

    def del_rows(self):
        for i in self.tree.selection():
            self.tree.delete(i)

    def set_lambda(self, results):
        if "electron" in results:
            self.lam_e.set(f"{results['electron']['lambda_eV']:.5f}")
        if "hole" in results:
            self.lam_h.set(f"{results['hole']['lambda_eV']:.5f}")
        self.app.log("迁移率页 λe / λh 已更新")

    def _scale(self):
        return 1.0 if self.unit.get() == "eV" else 1000.0

    def fill_last_j(self):
        last = self.app.data.get("last_J")
        if not last:
            raise ValueError("还没有转移积分结果")
        sc = self._scale() / 1000.0
        for i in self.tree.selection() or self.tree.get_children():
            vals = list(self.tree.item(i, "values"))
            if last.get("electron") is not None:
                vals[7] = f"{abs(last['electron']) * sc:.6g}"
            if last.get("hole") is not None:
                vals[8] = f"{abs(last['hole']) * sc:.6g}"
            self.tree.item(i, values=vals)

    @staticmethod
    def detect_program(d):
        d = Path(d)
        if any(d.glob("G*_full.t21")) or any(d.glob("G*_full.txt")) or any(d.glob("G*_full.results")):
            return "adf"
        if any(d.glob("G*_AB.fchk")) or any(d.glob("G*_A.fchk")):
            return "gaussian"
        return None

    def import_dir(self, d=None, program=None):
        """从二聚体目录读取各组转移积分并填入表格；d、program 为空时询问目录、自动判断程序。"""
        if d is None:
            d = filedialog.askdirectory(title="选择二聚体计算目录 (含 dimers.json)")
            if not d:
                return None
        info = json.loads((Path(d) / "dimers.json").read_text())
        program = program or self.detect_program(d) or self.app.ti_tab.prog.get()
        from .programs import find_ams
        couplings, errors = collect_group_couplings(
            d, program, self.app.config_data.get("xtb_exe") or None,
            find_ams(self.app.config_data.get("ams_home") or None) if program == "adf" else None)
        sc = self._scale() / 1000.0
        paths, missing = [], set()
        for i, nb in enumerate(info["neighbors"]):
            c = couplings.get(nb["group"], {})
            if not c:
                missing.add(nb["group"])
            paths.append({"name": f"G{nb['group']}-{i + 1}", "vector": nb["vector"],
                          "Ve": abs(c["electron"]) * sc if c.get("electron") is not None else "",
                          "Vh": abs(c["hole"]) * sc if c.get("hole") is not None else ""})
        self.set_paths(paths)
        for g, e in errors.items():
            self.app.log(f"G{g} 读取失败: {e}")
        self.app.log(f"导入 {len(paths)} 个近邻，读取到 {len(couplings)} 组转移积分 ({program})"
                     + (f"；缺少组: {sorted(missing)}" if missing else ""))
        for gid, c in sorted(couplings.items()):
            self.app.log(f"  G{gid}: Vh = {c.get('hole') or 0:.2f} meV, Ve = {c.get('electron') or 0:.2f} meV")
        return couplings, errors

    def import_csv(self):
        """CSV/TSV，列名可为: name/Pathway, d/r, theta/Θ, gamma, Ve, Vh, x, y, z, count。"""
        import csv as _csv
        p = filedialog.askopenfilename(filetypes=[("表格", "*.csv *.txt *.tsv"), ("所有文件", "*.*")])
        if not p:
            return
        text = Path(p).read_text(encoding="utf-8-sig", errors="ignore")
        dialect = _csv.Sniffer().sniff(text.splitlines()[0], delimiters=",\t;")
        rows = list(_csv.DictReader(text.splitlines(), dialect=dialect))
        alias = {"name": ("name", "pathway", "path", "名称"), "r": ("r", "d", "d(å)", "d(a)", "distance"),
                 "theta": ("theta", "θ", "θ(°)", "Θ", "Θ(°)", "angle"), "gamma": ("gamma", "γ"),
                 "Ve": ("ve", "ve(ev)", "ve(mev)"), "Vh": ("vh", "vh(ev)", "vh(mev)"), "x": ("x",), "y": ("y",),
                 "z": ("z",), "count": ("count", "数目")}
        paths = []
        for r in rows:
            low = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
            d = {}
            for key, names in alias.items():
                for n in names:
                    if n.lower() in low and low[n.lower()] != "":
                        d[key] = low[n.lower()]
                        break
            if all(k in d for k in ("x", "y", "z")):
                d["vector"] = [d.pop("x"), d.pop("y"), d.pop("z")]
            paths.append(d)
        self.set_paths(paths)
        self.app.log(f"从 {Path(p).name} 导入 {len(paths)} 条路径（V 单位按当前设置: {self.unit.get()}）")

    def _paths(self):
        sc = self._scale()
        out = []
        for row in self.tree.rows():
            name, x, y, z, r, th, ga, ve, vh, cnt = (str(v).strip() for v in row)
            d = {"name": name, "count": int(float(cnt or 1))}
            if x and y and z:
                vec = [float(x), float(y), float(z)]
                d["vector"] = vec
                d["r"], d["theta"], d["gamma"] = (float(t) for t in plane_angles(vec, self.plane.get()))
            else:
                if not r:
                    raise ValueError(f"路径 {name} 缺少 r 或 x,y,z")
                d.update(r=float(r), theta=float(th or 0), gamma=float(ga or 0))
            d["Ve_eV"] = float(ve) / sc if ve else None
            d["Vh_eV"] = float(vh) / sc if vh else None
            out.append(d)
        if not out:
            raise ValueError("迁移率表为空")
        return out

    def _temps(self):
        s = self.temp.get().strip()
        if ":" in s:
            a, b, c = (float(v) for v in s.split(":"))
            return list(np.arange(a, b + 1e-9, c))
        return [float(v) for v in s.split(",")]

    def load_cfg(self):
        p = filedialog.askopenfilename(filetypes=[("json", "*.json")])
        if not p:
            return
        cfg = json.loads(Path(p).read_text())
        old = "lambda_eV" in cfg
        self.lam_e.set(cfg.get("lambda_e", cfg.get("lambda_eV", 0.2)))
        self.lam_h.set(cfg.get("lambda_h", cfg.get("lambda_eV", 0.2)))
        temps = cfg.get("temperatures") or [cfg.get("temperature", 300)]
        self.temp.set(",".join(str(t) for t in temps))
        self.dim.set(cfg.get("dimension", 3))
        self.dg.set(cfg.get("deltaG_eV", 0.0))
        self.plane.set(cfg.get("plane", "xy"))
        self.unit.set(cfg.get("V_unit", "meV"))
        paths = []
        for q in cfg["pathways"]:
            d = dict(q)
            if old:      # 旧格式: 单一 J
                j = q.get("J_meV", q.get("J_eV", 0) * 1000)
                self.unit.set("meV")
                d["Ve"] = d["Vh"] = j
            paths.append(d)
        self.set_paths(paths)

    def save_cfg(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("json", "*.json")])
        if not p:
            return
        sc = self._scale()
        paths = []
        for d in self._paths():
            q = {k: d[k] for k in ("name", "r", "theta", "gamma", "count")}
            if "vector" in d:
                q["vector"] = d["vector"]
            q["Ve"] = d["Ve_eV"] * sc if d["Ve_eV"] is not None else None
            q["Vh"] = d["Vh_eV"] * sc if d["Vh_eV"] is not None else None
            paths.append(q)
        cfg = {"lambda_e": float(self.lam_e.get()), "lambda_h": float(self.lam_h.get()), "temperatures": self._temps(),
               "dimension": int(self.dim.get()), "deltaG_eV": float(self.dg.get()), "plane": self.plane.get(),
               "V_unit": self.unit.get(), "pathways": paths}
        Path(p).write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        self.app.log(f"配置已保存 {p}")

    # ---------------- 计算 ----------------
    def compute(self):
        paths, temps = self._paths(), self._temps()
        lam = {"e": float(self.lam_e.get()), "h": float(self.lam_h.get())}
        dg, dim = float(self.dg.get()), int(self.dim.get())
        res = {}
        for T in temps:
            res[T] = {}
            for c, key in (("e", "Ve_eV"), ("h", "Vh_eV")):
                ps = [dict(d, V=d[key]) for d in paths if d[key] is not None]
                if ps:
                    res[T][c] = angular_mobility_2d(lam[c], ps, T, "V", dg_ev=dg)
                    scal = compute_mobility(lam[c], [dict(p_, J_eV=p_["V"]) for p_ in ps], T, dim, dg)
                    res[T][c]["scalar_mu"] = scal["mobility_cm2_Vs"]
                    if "mobility_principal_cm2_Vs" in scal:
                        res[T][c]["tensor_principal"] = scal["mobility_principal_cm2_Vs"]
        T0 = temps[0]
        r0 = res[T0]
        if not r0:
            raise ValueError("Ve、Vh 都为空")
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"T = {T0} K   λe = {lam['e']} eV   λh = {lam['h']} eV   传输平面 {self.plane.get()}\n")
        t.insert("end", f"{'路径':<10}{'r/Å':>8}{'θ/°':>9}{'γ/°':>7}{'We/s-1':>12}{'Pe':>8}{'Wh/s-1':>12}{'Ph':>8}\n")
        names = [d["name"] for d in paths]
        pe = {q["name"]: q for q in r0["e"]["pathways"]} if "e" in r0 else {}
        ph = {q["name"]: q for q in r0["h"]["pathways"]} if "h" in r0 else {}
        for d in paths:
            n = d["name"]
            e_, h_ = pe.get(n), ph.get(n)
            t.insert("end", f"{n:<10}{d['r']:8.3f}{d['theta']:9.2f}{d['gamma']:7.2f}"
                            f"{(e_['W_s-1'] if e_ else float('nan')):12.4e}{(e_['P'] if e_ else float('nan')):8.4f}"
                            f"{(h_['W_s-1'] if h_ else float('nan')):12.4e}{(h_['P'] if h_ else float('nan')):8.4f}\n")
        for c, lab in (("e", "电子"), ("h", "空穴")):
            if c in r0:
                q = r0[c]
                line = (f"\n{lab}: μmax = {q['mu_max']:.4g} cm²/Vs (φ = {q['phi_max_deg']:.1f}°)  μmin = {q['mu_min']:.4g}"
                        f"  各向异性 {q['anisotropy']:.3g}  |  三维标量公式 μ = {q['scalar_mu']:.4g}")
                if "tensor_principal" in q:
                    line += "  张量主值 " + ", ".join(f"{w:.4g}" for w in q["tensor_principal"])
                t.insert("end", line + "\n")
        if len(temps) > 1:
            t.insert("end", f"\n{'T/K':>8}{'μe,max':>14}{'μh,max':>14}\n")
            for T in temps:
                t.insert("end", f"{T:8.1f}{res[T].get('e', {}).get('mu_max', float('nan')):14.5g}"
                                f"{res[T].get('h', {}).get('mu_max', float('nan')):14.5g}\n")
        self.last = {"phi": (r0.get("e") or r0.get("h"))["phi"], "mu_e": r0["e"]["mu"] if "e" in r0 else None,
                     "mu_h": r0["h"]["mu"] if "h" in r0 else None, "lam_e": lam["e"] if "e" in r0 else None,
                     "lam_h": lam["h"] if "h" in r0 else None, "label": self.label.get()}
        self.app.data["mobility"] = res
        if len(temps) > 1:
            ax = self.plot.new_axes()
            for c, col, lab in (("e", "#2B5DA8", "电子 μmax"), ("h", "#B22222", "空穴 μmax")):
                ys = [res[T][c]["mu_max"] for T in temps if c in res[T]]
                if ys:
                    ax.plot(temps[:len(ys)], ys, "o-", color=col, label=lab)
            ax.set_xlabel("T / K")
            ax.set_ylabel("μ / cm$^2$ V$^{-1}$ s$^{-1}$")
            ax.legend()
        else:
            ax = self.plot.new_axes("polar")
            draw_angular(ax, **{k: v for k, v in self.last.items() if k != "label"})
        self.plot.draw()

    def kmc_selfcheck(self):
        """与有精确解的体系逐项比较 (布拉维格子、多分子晶胞、温度、弱场、能量无序)。"""
        from .validation import format_rows, kmc_validation
        st = self.app.struct_tab
        coup = self.app.data.get("crystal_couplings")
        cr = (st.crys, st.mols, coup, float(st.cut.get())) if (st.mols is not None and coup) else None

        def done(rows):
            from .validation import figure
            self.selfcheck_rows = rows
            t = self.text
            t.delete("1.0", "end")
            t.insert("end", "KMC 精度自检：与精确解逐项比较\n" + format_rows(rows) + "\n")
            t.insert("end", "精确解：布拉维格子 μ = Σ k r r / 2kT；多分子晶胞为周期格子均匀化解 (修正位移 χ)；"
                            "能量无序为一维环 Derrida 精确解的 Boltzmann 系综平均\n")
            ax = self.plot.new_axes()
            figure(rows, ax=ax)
            ax.set_title("KMC 与精确解对照")
            self.plot.draw()
        self.app.run_bg("KMC 精度自检", lambda: kmc_validation(quick=False, crystal=cr), done)

    def export_selfcheck(self, path=None):
        from .validation import figure, write_csv
        if not self.selfcheck_rows:
            raise ValueError("请先运行 KMC 精度自检")
        p = path or filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return None
        write_csv(self.selfcheck_rows, p)
        figure(self.selfcheck_rows, str(Path(p).with_suffix(".png")))
        self.app.log(f"已导出 {p} 与 {Path(p).with_suffix('.png').name} (300 dpi)")
        return p

    def run_kmc(self):
        from .kmc import analytic_bravais, graph_from_crystal, graph_from_paths, run_kmc
        car = "electron" if self.k_car.get() == "电子" else "hole"
        lam = float(self.lam_e.get() if car == "electron" else self.lam_h.get())
        T = float(self._temps()[0])
        walkers, steps = (int(x) for x in self.k_walk.get().split())
        F = [float(x) for x in self.k_field.get().split()]
        if self.k_src.get().startswith("结构"):
            st = self.app.struct_tab
            coup = self.app.data.get("crystal_couplings")
            if st.mols is None or not coup:
                raise ValueError("请先在“结构/晶体”页打开 CIF 并点击“软件内计算全部二聚体 J (EHT)”")
            graph = graph_from_crystal(st.crys, st.mols, coup, float(st.cut.get()),
                                       "atom" if st.crit.get().startswith("最短") else "centroid", car)
            if graph["missing_groups"]:
                self.app.log(f"[警告] 有近邻没有对应的 J (组 {graph['missing_groups']})，已忽略")
        else:
            key = "Ve_eV" if car == "electron" else "Vh_eV"
            graph = graph_from_paths(self._paths(), key)
        lmodel = "constant" if self.k_lmodel.get() == "常数" else "spheres"
        _kw = dict(method=self.k_rate.get().lower(), lam_out=float(self.k_lout.get()), lam_out_model=lmodel,
                   pekar=float(self.k_pekar.get()), radius=float(self.k_rad.get()), hw=float(self.k_hw.get()), field=F,
                   carrier=car, sigma=float(self.k_sigma.get()), supercell=int(self.k_cell.get()), walkers=walkers,
                   steps=steps)
        def _work():
            return run_kmc(graph, lam, T, **_kw)

        def _done(_v):
            res = _v
            t = self.text
            t.delete("1.0", "end")
            t.insert("end", f"KMC ({'电子' if car == 'electron' else '空穴'}, {self.k_rate.get()}, T={T} K, λ={lam} eV, "
                            f"位点 {res['n_sites']}, 跳跃 {res['hops']:,}, 模拟时间 {res['sim_time_s']:.3e} s)\n")
            mu, err = res["mu_tensor_cm2_Vs"], res["mu_err"]
            t.insert("end", "迁移率张量 / cm²V⁻¹s⁻¹ (± 统计误差)\n")
            for i in range(3):
                t.insert("end", "  " + "  ".join(f"{mu[i, j]:10.4g}±{err[i, j]:<8.2g}" for j in range(3)) + "\n")
            t.insert("end", "主值: " + ", ".join(f"{w:.4g}" for w in res["mu_principal"]) +
                            f"   平均 {res['mu_avg']:.4g} ± {res['mu_avg_err']:.2g}\n")
            if "mu_field_cm2_Vs" in res:
                t.insert("end", f"场迁移率 μ_F = {res['mu_field_cm2_Vs']:.4g} ± {res['mu_field_err']:.2g} cm²V⁻¹s⁻¹ "
                                f"(|F| = {np.linalg.norm(F):.3g} V/cm)\n")
            if graph["kind"] == "bravais" and float(self.k_sigma.get()) == 0 and not any(F) and lmodel == "constant" \
                    and self.k_rate.get() == "Marcus":
                ana = analytic_bravais(graph, lam, T, float(self.k_lout.get()))
                w = np.linalg.eigvalsh(ana)[::-1]
                t.insert("end", "解析张量主值 (对照): " + ", ".join(f"{x:.4g}" for x in w) + "\n")
            ax = self.plot.new_axes()
            tt, msd = res["msd_t"], res["msd_A2"]
            ax.plot(tt, msd, "o", ms=3, color="#2B4A77", label="KMC <|Δr|²>")
            if len(tt) > 2:
                k_ = np.polyfit(tt, msd, 1)
                ax.plot(tt, np.polyval(k_, tt), "-", color="#C0392B", label="线性拟合")
            ax.set_xlabel("t / s")
            ax.set_ylabel("MSD / Å²")
            ax.set_title("KMC 均方位移")
            ax.legend()
            self.plot.draw()
            self.app.data["kmc"] = res
        self.app.run_bg("运行 KMC", _work, _done)

    def show_strain_scan(self, rows, axis):
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"应变扫描 (沿 {axis}，EHT 转移积分，解析迁移率张量)\n")
        t.insert("end", f"{'应变/%':>8}{'近邻数':>8}{'Jh,max/meV':>12}{'Je,max/meV':>12}{'μh,max':>10}{'μe,max':>10}"
                        f"{'μh,avg':>10}{'μe,avg':>10}\n")
        f = lambda v: f"{v:10.4g}" if v is not None else f"{'—':>10}"  # noqa: E731
        for r in rows:
            t.insert("end", f"{r['strain_percent']:8.2f}{r['n_dimers']:8d}{r['J_max_hole']:12.3f}"
                            f"{r['J_max_electron']:12.3f}{f(r['mu_hole_max'])}{f(r['mu_electron_max'])}"
                            f"{f(r['mu_hole_avg'])}{f(r['mu_electron_avg'])}\n")
        ax = self.plot.new_axes()
        x = [r["strain_percent"] for r in rows]
        for key, col, lab in (("mu_electron_max", "#2B5DA8", "电子 μmax"), ("mu_hole_max", "#B22222", "空穴 μmax"),
                              ("mu_electron_avg", "#7FA3CF", "电子 μavg"), ("mu_hole_avg", "#E59866", "空穴 μavg")):
            y = [r[key] for r in rows]
            if all(v is not None for v in y):
                ax.plot(x, y, "o-" if "max" in key else "s--", color=col, label=lab)
        ax.set_xlabel(f"{axis} 方向应变 / %")
        ax.set_ylabel("μ / cm$^2$ V$^{-1}$ s$^{-1}$")
        ax.set_title("应变对迁移率的影响 (EHT)")
        ax.legend()
        self.plot.draw()

    def add_compare(self):
        if not self.last:
            raise ValueError("请先计算")
        self.compare.append(dict(self.last, label=self.label.get()))
        self.cmp_lab.config(text="对比列表: " + ", ".join(s["label"] for s in self.compare))

    def clear_compare(self):
        self.compare = []
        self.cmp_lab.config(text="对比列表: (空)")

    # ---------------- 三维 ----------------
    def _systems3d(self):
        from .mobility import mobility_3d
        from .plots import E_COLOR, H_COLOR
        paths = self._paths()
        if any("vector" not in d for d in paths):
            raise ValueError("三维图需要每条路径的 x、y、z (跳跃矢量)")
        T0 = self._temps()[0]
        dg = float(self.dg.get())
        model = "wen" if self.m3_model.get() == self.M3D[0] else "tensor"
        out = []
        for key, lam, lab, col in (("Ve_eV", float(self.lam_e.get()), "电子", E_COLOR),
                                   ("Vh_eV", float(self.lam_h.get()), "空穴", H_COLOR)):
            ps = [dict(d, V=d[key]) for d in paths if d[key] is not None]
            if ps:
                r = mobility_3d(lam, ps, T0, "V", model, dg)
                out.append({"label": f"{lab} (λ = {lam * 1000:.0f} meV)", "res": r, "color": col, "carrier": lab})
        if not out:
            raise ValueError("Ve、Vh 都为空")
        st = self.app.struct_tab
        lat = st.crys["lattice"] if getattr(st, "crys", None) is not None else None
        return out, lat, T0

    def compute3d(self):
        from .mobility import crystal_direction
        from .plots import draw_mobility_3d, view_azimuth
        systems, lat, T0 = self._systems3d()
        self.last3d = (systems, lat)
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"三维各向异性迁移率 ({self.m3_model.get()})，T = {T0:g} K\n")
        for s_ in systems:
            r = s_["res"]
            t.insert("end", f"\n{s_['carrier']}：主值 " + " / ".join(f"{v:.4g}" for v in r["principal"]) +
                            f" cm²/Vs，平均 {r['mu_avg']:.4g}，各向异性 μmax/μmin = {r['anisotropy']:.3g}\n")
            for k, (v, d) in enumerate(zip(r["principal"], r["axes"])):
                line = f"  μ{k + 1} = {v:9.4g}  方向 ({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f})"
                if lat is not None:
                    f, ang = crystal_direction(d, lat)
                    line += f"  ≈ [{f[0]:.2f} {f[1]:.2f} {f[2]:.2f}]，与 a/b/c 夹角 {ang[0]:.0f}°/{ang[1]:.0f}°/{ang[2]:.0f}°"
                t.insert("end", line + "\n")
        fig = self.plot.fig
        fig.clf()
        azim = view_azimuth(max(systems, key=lambda x: x["res"]["mu_max"])["res"])
        for i, s_ in enumerate(systems):
            ax = fig.add_subplot(1, len(systems), i + 1, projection="3d")
            draw_mobility_3d(ax, s_["res"], s_["color"], lat, self.plane.get(), s_["label"], n=41, azim=azim)
        self.plot.ax = ax
        self.plot.canvas.draw_idle()
        self.app.log("三维迁移率图：可用鼠标拖动旋转")

    def export3d(self, path=None):
        from .plots import figure_mobility_3d, mobility_3d_html
        systems, lat, _ = self._systems3d()
        p = path or filedialog.asksaveasfilename(defaultextension=".png", initialfile="mobility_3d.png",
                                                 filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("TIFF", "*.tif")])
        if not p:
            return None
        figure_mobility_3d(systems, p, lat, self.plane.get())
        h = str(Path(p).with_suffix(".html"))
        mobility_3d_html(h, systems, lat, self.plane.get(), title=f"三维各向异性迁移率 — {self.label.get()}")
        self.app.log(f"已导出 {p} (300 dpi) 与可在浏览器中旋转的 {Path(h).name}")
        return p

    def export_fig(self):
        systems = self.compare or ([dict(self.last, label=self.label.get())] if self.last else [])
        if not systems:
            raise ValueError("请先计算")
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                                                                            ("SVG", "*.svg"), ("TIFF", "*.tif")])
        if p:
            figure_angular(systems, p)
            self.app.log(f"已导出角分辨迁移率图 ({len(systems)} 个体系) → {p}")


# ==============================================================================================
# 输运理论与光物理扩展 (TLT / 能带 / 动态无序 / 振动分辨光谱 / π 堆积)
# ==============================================================================================
class AdvancedTab(ttk.Frame):
    SRC = ["结构页晶体 (EHT, 含符号)", "迁移率表 (|J|，按正号处理)"]

    def __init__(self, parent, app):
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.dyn, self.vib_sets, self.vib_last = None, [], None
        S = safe(app)
        panel = SectionPanel(self, width=470)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner
        c = ttk.LabelFrame(left, text="数据来源")
        c.pack(fill="x")
        self.src = field(c, 0, "近邻与 J", self.SRC[0], values=self.SRC, width=24)
        self.car = field(c, 1, "载流子", "电子", values=["电子", "空穴"], width=24)
        self.temp = field(c, 2, "温度 / K", 300, width=24)
        self.htau = field(c, 3, "ħ/τ_in / meV", 5, width=24)

        d = ttk.LabelFrame(left, text="动态无序 σ_J (EHT 刚体热振动采样)")
        d.pack(fill="x", pady=3)
        df = ttk.Frame(d)
        df.pack(fill="x")
        self.dt = field(df, 0, "平移 σ / Å", 0.10, width=8)
        self.dr = field(df, 0, "转动 σ / °", 1.5, width=8, col=2)
        self.dn = field(df, 1, "采样数", 30, width=8)
        ttk.Button(d, text="计算全部二聚体组的 σ_J", command=S(self.run_dyn)).pack(fill="x")

        t = ttk.LabelFrame(left, text="瞬态局域化理论 TLT (参考 TransLoc)")
        t.pack(fill="x", pady=3)
        tf = ttk.Frame(t)
        tf.pack(fill="x")
        self.sig = field(tf, 0, "σ_J / |J|", 0.3, width=8)
        self.use_dyn = tk.BooleanVar(value=False)
        ttk.Checkbutton(tf, text="使用动态无序结果", variable=self.use_dyn).grid(row=0, column=2, columnspan=2, sticky="w")
        self.tl = field(tf, 1, "超胞边长 L", 0, width=8)
        self.tn = field(tf, 1, "无序构型数", 10, width=8, col=2)
        self.tplane = field(tf, 2, "传输平面", "自动", values=["自动", "ab", "ac", "bc"], width=8)
        ttk.Label(t, text="L=0 时自动选择 (约 600 个位点)；空穴用 −H(HOMO)，电子用 +H(LUMO)",
                  foreground=PALETTE["muted"], wraplength=440).pack(anchor="w")
        ttk.Button(t, text="计算 TLT 迁移率", style="Accent.TButton", command=S(self.run_tlt)).pack(fill="x")

        b = ttk.LabelFrame(left, text="紧束缚能带 / 有效质量 (参考 effmass)")
        b.pack(fill="x", pady=3)
        ttk.Button(b, text="计算能带与有效质量", command=S(self.run_band)).pack(fill="x")

        v = ttk.LabelFrame(left, text="振动分辨磷光光谱 / 能隙定律")
        v.pack(fill="x", pady=3)
        vf = ttk.Frame(v)
        vf.pack(fill="x")
        self.vsrc = field(vf, 0, "黄昆因子来源", "分解页 λ1 (末态=中性/基态)",
                          values=["分解页 λ1 (末态=中性/基态)", "分解页 λ2", "FCHT log 文件"], width=22)
        self.vlog = FileEntry(v, "FCHT log", filetypes=[("log", "*.log *.out"), ("所有文件", "*.*")])
        self.vlog.pack(fill="x")
        vf2 = ttk.Frame(v)
        vf2.pack(fill="x")
        self.ekind = field(vf2, 0, "能量类型", "垂直发射能 E_v", values=["垂直发射能 E_v", "0-0 能量 E00"], width=14)
        self.evalue = field(vf2, 1, "能量 / eV", 1.65, width=14)
        self.vfwhm = field(vf2, 2, "展宽 FWHM / eV", 0.05, width=14)
        self.vc = field(vf2, 3, "电子耦合 C / meV (可空)", "", width=14)
        self.vlab = field(vf2, 4, "图例名称", "Ir-BB", width=14)
        vb = ttk.Frame(v)
        vb.pack(fill="x")
        for txt, cmd in (("计算光谱", self.run_vib), ("加入对比", self.add_vib), ("清空", self.clear_vib),
                         ("导出…", self.export_vib)):
            ttk.Button(vb, text=txt, command=S(cmd)).pack(side="left", fill="x", expand=True)

        p = ttk.LabelFrame(left, text="π 堆积几何 (面间距 / 滑移 / 夹角)")
        p.pack(fill="x", pady=3)
        ttk.Button(p, text="分析结构页全部近邻二聚体", command=S(self.run_stack)).pack(fill="x")

        right = ttk.Frame(self, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.plot = PlotPanel(right, figsize=(6.5, 4.6))
        self.plot.pack(fill="both", expand=True)
        self.text = styled_text(right, app.mono, height=15, wrap="none", size=9)
        self.text.pack(fill="x", pady=(8, 0))

    def _carrier(self):
        return "electron" if self.car.get() == "电子" else "hole"

    def _graph(self):
        from .kmc import graph_from_crystal, graph_from_paths
        car = self._carrier()
        st = self.app.struct_tab
        if self.src.get() == self.SRC[0]:
            coup = self.app.data.get("crystal_couplings")
            if st.mols is None or not coup:
                raise ValueError("请先在“结构/晶体”页打开 CIF 并点击“软件内计算全部二聚体 J (EHT)”")
            g = graph_from_crystal(st.crys, st.mols, coup, float(st.cut.get()),
                                   "atom" if st.crit.get().startswith("最短") else "centroid", car, signed=True)
            return g, st.crys["lattice"]
        mt = self.app.mob_tab
        key = "Ve_eV" if car == "electron" else "Vh_eV"
        return graph_from_paths(mt._paths(), key), None

    def _keep_axes(self, g):
        pl = self.tplane.get()
        if pl != "自动":
            return tuple("abc".index(x) for x in pl)
        w = np.zeros(3)
        for e in g["edges"]:
            if e[5] is not None:
                w += (np.array(e[5]) != 0) * e[3] ** 2
        return tuple(sorted(int(i) for i in np.argsort(w)[::-1][:2]))

    def run_dyn(self):
        from .transport import dynamic_disorder
        st = self.app.struct_tab
        if not st.dimer_res:
            raise ValueError("请先在“结构/晶体”页搜索近邻二聚体或计算全部二聚体 J")
        a_atoms = [(s_, *x) for s_, x in st.dimer_res["atoms_a"]]
        rows = {}
        T = float(self.temp.get())
        _a = (int(self.dn.get()), float(self.dt.get()), float(self.dr.get()), T)
        _groups = [(g["id"], [(s_, *x) for s_, x in g["members"][0]["atoms_b"]]) for g in st.dimer_res["groups"]]
        def _work():
            return {gid: dynamic_disorder(a_atoms, b_, *_a) for gid, b_ in _groups}

        def _done(_v):
            rows = _v
            self.dyn = rows
            t = self.text
            t.delete("1.0", "end")
            t.insert("end", f"动态无序 (EHT, T={T} K, 平移 σ={self.dt.get()} Å, 转动 σ={self.dr.get()}°)\n")
            t.insert("end", f"{'组':<5}{'<Jh>/meV':>10}{'σh/meV':>9}{'σh/|J|':>8}{'<Je>/meV':>10}{'σe/meV':>9}{'σe/|J|':>8}\n")
            for gid, r in rows.items():
                h, e = r["hole"], r["electron"]
                t.insert("end", f"G{gid:<4}{h['mean_meV']:10.2f}{h['std_meV']:9.2f}{h['ratio']:8.2f}"
                                f"{e['mean_meV']:10.2f}{e['std_meV']:9.2f}{e['ratio']:8.2f}\n")
            t.insert("end", "σ/|J| ≳ 0.5 时 J 涨落与平均值相当，宜用 TLT 而不是能带模型；J ≪ λ 时跳跃模型适用。\n")
            ax = self.plot.new_axes()
            car = self._carrier()
            for gid, r in rows.items():
                ax.hist(r[car]["samples"], bins=15, alpha=0.55, label=f"G{gid}")
            ax.set_xlabel("J / meV")
            ax.set_ylabel("采样数")
            ax.set_title(f"转移积分热涨落分布 ({'电子' if car == 'electron' else '空穴'})")
            ax.legend(fontsize=7)
            self.plot.draw()
        self.app.run_bg("动态无序 σ_J (EHT 刚体热振动采样)", _work, _done)

    def run_tlt(self):
        from .transport import tlt_mobility
        g, lat = self._graph()
        keep = self._keep_axes(g)
        L = int(float(self.tl.get() or 0)) or max(8, int(round(np.sqrt(600 / g["n_sites"]))))
        sizes = tuple(L if a in keep else 1 for a in range(3))
        car = self._carrier()
        sigma = float(self.sig.get())
        if self.use_dyn.get():
            if not self.dyn:
                raise ValueError("请先计算动态无序 σ_J")
            sigma = float(np.median([r[car]["ratio"] for r in self.dyn.values()]))
            self.app.log(f"TLT 使用动态无序结果: σ/|J| 中位数 = {sigma:.2f}")
        T = float(self.temp.get())
        _a = (float(self.htau.get()) / 1000, int(self.tn.get()))
        def _work():
            return tlt_mobility(g, T, car, sigma, _a[0], sizes, keep, _a[1])

        def _done(_v):
            r = _v
            t = self.text
            t.delete("1.0", "end")
            names = "".join(("abc" if lat is not None else "123")[a] for a in keep)
            t.insert("end", f"TLT ({'电子' if car == 'electron' else '空穴'}, T={T} K, σ/|J|={sigma:.2f}, "
                            f"ħ/τ={self.htau.get()} meV, 超胞 {sizes[0]}×{sizes[1]}×{sizes[2]}×{g['n_sites']} 位点, 平面 {names})\n")
            t.insert("end", "迁移率主值 / cm²V⁻¹s⁻¹: " + ", ".join(f"{w:.4g}" for w in r["mu_principal"][:2]) +
                            f"\n瞬态局域化长度 L ≈ {r['L_A']:.1f} Å，τ_in = {r['tau_s']:.2e} s\n")
            hop = self.app.data.get("mobility")
            if hop:
                T0 = next(iter(hop))
                c = "e" if car == "electron" else "h"
                if c in hop[T0]:
                    t.insert("end", f"对照: 跳跃模型 μmax = {hop[T0][c]['mu_max']:.4g} cm²V⁻¹s⁻¹ (迁移率页)\n")
            t.insert("end", "说明: TLT 适用于 J 与 λ 相当、热涨落强的高迁移率晶体；结果依赖 σ_J 与 ħ/τ_in，宜给出参数范围。\n")
            ax = self.plot.new_axes("polar")
            draw_angular(ax, r["phi"], r["mu_phi"] if car == "electron" else None, r["mu_phi"] if car == "hole" else None,
                         title=False)
            ax.set_title("TLT 角分辨迁移率 / cm$^2$V$^{-1}$s$^{-1}$", fontsize=10)
            self.plot.draw()
            self.app.data["tlt"] = r
        self.app.run_bg("计算 TLT 迁移率", _work, _done)

    def run_band(self):
        from .transport import band_mobility, band_path, effective_mass
        g, lat = self._graph()
        car = self._carrier()
        def _work():
            return (lambda e_: (e_, band_path(g, e_['reciprocal'], car)))(effective_mass(g, car, lat, n_grid=16 if g['n_sites'] > 2 else 24))

        def _done(_v):
            em, (segs, labs) = _v
            mus = band_mobility(em["masses_me"], float(self.htau.get()) / 1000)
            t = self.text
            t.delete("1.0", "end")
            t.insert("end", f"紧束缚能带 ({'电子' if car == 'electron' else '空穴'}, {g['n_sites']} 条能带)\n"
                            f"最低能带带宽 W = {em['bandwidth_eV'] * 1000:.1f} meV\n有效质量 (主轴, m_e): "
                            + ", ".join("∞" if not np.isfinite(m) else f"{m:.3g}" for m in em["masses_me"]) +
                            "\n常弛豫时间能带迁移率 eτ/m* (cm²V⁻¹s⁻¹): " + ", ".join(f"{m:.3g}" for m in mus) +
                            f"  (ħ/τ = {self.htau.get()} meV)\n能带模型仅在 W ≫ k_BT 且 σ_J 小时适用；"
                            "W ≲ λ 时应以跳跃 / TLT 结果为主。\n")
            fig = self.plot.fig
            fig.clf()
            for n, (seg, lab) in enumerate(zip(segs, labs)):
                ax = fig.add_subplot(1, 3, n + 1)
                ax.plot(np.linspace(0, 0.5, len(seg)), seg * 1000, color="#2B4A77")
                ax.set_xticks([0, 0.5])
                ax.set_xticklabels(["Γ", lab])
                if n == 0:
                    ax.set_ylabel("E / meV")
            fig.suptitle("紧束缚能带 (载流子能量)", fontsize=11)
            self.plot.draw()
        self.app.run_bg("计算能带与有效质量", _work, _done)

    def _hr(self):
        src = self.vsrc.get()
        if src.startswith("FCHT"):
            from .gaussian_fcht import read_fcht_log
            r = read_fcht_log(self.vlog.get())
            return [m["freq_cm-1"] for m in r["modes"]], [m["S"] for m in r["modes"]]
        sets = self.app.decomp_tab.sets
        key = "neutral" if "λ1" in src else "ion"
        if not sets or key not in sets:
            raise ValueError("请先在“分解/黄昆因子”页完成分解，或选择 FCHT log 文件")
        return [m["freq_cm-1"] for m in sets[key]["modes"]], [m["S"] for m in sets[key]["modes"]]

    def run_vib(self):
        from .vibronic import emission_spectrum, energy_gap_law
        f, hr = self._hr()
        e = float(self.evalue.get())
        kw = {"e_vert": e} if self.ekind.get().startswith("垂直") else {"e00": e}
        r = emission_spectrum(f, hr, T=float(self.temp.get()), fwhm_ev=float(self.vfwhm.get()), **kw)
        c = float(self.vc.get()) / 1000 if self.vc.get().strip() else None
        eg = energy_gap_law(f, hr, r["E00_eV"], c)
        r["egl"], r["label"] = eg, self.vlab.get()
        self.vib_last = r
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"振动分辨磷光光谱 ({r['label']}, T={self.temp.get()} K, 展宽 {self.vfwhm.get()} eV)\n"
                        f"E00 = {r['E00_eV']:.3f} eV, λ = Σ S ħω = {r['lambda_eV'] * 1000:.1f} meV, ΣS = {r['S_total']:.3f}\n"
                        f"峰位 {r['peak_eV']:.3f} eV ({r['peak_nm']:.1f} nm), FWHM = {r['fwhm_eV']:.3f} eV "
                        f"({r['fwhm_cm']:.0f} cm⁻¹)\n")
        t.insert("end", f"能隙定律 (受体模 ≥1000 cm⁻¹): S_M = {eg['S_M']:.3f}, ħω_M = {eg['hw_M_cm']:.0f} cm⁻¹, "
                        f"γ = {eg['gamma']:.2f}, k_nr/C² = {eg['knr_over_C2']:.3e} s⁻¹eV⁻²"
                        + (f", k_nr = {eg['knr_s-1']:.3e} s⁻¹" if "knr_s-1" in eg else "") + "\n")
        t.insert("end", "位移谐振子模型 (不含 Duschinsky 转动)；k_nr/C² 适合在体系之间作相对比较。\n")
        self.draw_vib([r] + [x for x in self.vib_sets if x["label"] != r["label"]])

    def draw_vib(self, sets):
        from .vibronic import NM_EV
        ax = self.plot.new_axes()
        for r in sets:
            ax.plot(NM_EV / r["E_eV"], r["I"], lw=1.8, label=f"{r['label']} (FWHM {r['fwhm_cm']:.0f} cm⁻¹)")
        ax.set_xlabel("波长 / nm")
        ax.set_ylabel("归一化强度")
        ax.set_title("振动分辨磷光光谱")
        ax.legend(fontsize=8)
        self.plot.draw()

    def add_vib(self):
        if not self.vib_last:
            raise ValueError("请先计算光谱")
        self.vib_sets = [x for x in self.vib_sets if x["label"] != self.vib_last["label"]] + [self.vib_last]
        self.draw_vib(self.vib_sets)
        if len(self.vib_sets) > 1:
            ref = self.vib_sets[0]["egl"]["knr_over_C2"]
            self.text.insert("end", "\n相对 k_nr (相同 C 假设): " + ", ".join(
                f"{x['label']} {x['egl']['knr_over_C2'] / ref:.3g}" for x in self.vib_sets) + "\n")

    def clear_vib(self):
        self.vib_sets = []

    def export_vib(self):
        sets = self.vib_sets or ([self.vib_last] if self.vib_last else [])
        if not sets:
            raise ValueError("请先计算光谱")
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if not p:
            return
        from matplotlib.figure import Figure as _F
        from .vibronic import NM_EV
        fig = _F(figsize=(8, 5.5), dpi=300)
        ax = fig.add_subplot(111)
        for r in sets:
            ax.plot(NM_EV / r["E_eV"], r["I"], lw=2, label=f"{r['label']} (FWHM {r['fwhm_cm']:.0f} cm$^{{-1}}$)")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized intensity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(p, dpi=300)
        import csv as _csv
        with open(Path(p).with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = _csv.writer(fh)
            for r in sets:
                w.writerow([r["label"], "E_eV", "nm", "I"])
                for e_, i_ in zip(r["E_eV"], r["I"]):
                    w.writerow(["", e_, NM_EV / e_, i_])
        self.app.log(f"已导出 {p}")

    def run_stack(self):
        from .crystal import stacking_geometry
        st = self.app.struct_tab
        if not st.dimer_res:
            raise ValueError("请先在“结构/晶体”页搜索近邻二聚体")
        a_atoms = [(s_, *x) for s_, x in st.dimer_res["atoms_a"]]
        coup = self.app.data.get("crystal_couplings") or {}
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"{'组':<5}{'r质心/Å':>9}{'面间距/Å':>10}{'夹角/°':>8}{'滑移/Å':>8}{'d最短/Å':>9}"
                        f"{'Jh/meV':>9}{'Je/meV':>9}  类型\n")
        rows = []
        f_ = lambda v, w=8, p=2: f"{v:{w}.{p}f}" if isinstance(v, float) else f"{'—':>{w}}"  # noqa: E731
        for g in st.dimer_res["groups"]:
            d = g["members"][0]
            r = stacking_geometry(a_atoms, [(s_, *x) for s_, x in d["atoms_b"]])
            c = coup.get(g["id"], {})
            t.insert("end", f"G{g['id']:<4}{d['r_centroid']:9.3f}{f_(r['d_plane'], 10)}{f_(r['angle'], 8, 1)}"
                            f"{f_(r['slip'], 8)}{f_(r.get('d_min'), 9)}{f_(c.get('hole'), 9)}{f_(c.get('electron'), 9)}"
                            f"  {r['type']}\n")
            rows.append((g["id"], r, c))
        t.insert("end", "面间距 / 夹角由两分子接触区 (距对方 < 4.5 Å 的非氢原子) 拟合平面得到。\n")
        pts = [(gid, r["slip"], abs(c.get(self._carrier(), np.nan))) for gid, r, c in rows if r["slip"] is not None]
        ax = self.plot.new_axes()
        if pts:
            for gid, x_, y_ in pts:
                ax.scatter([x_], [y_], color="#2B4A77")
                ax.annotate(f"G{gid}", (x_, y_), fontsize=8)
        ax.set_xlabel("滑移距离 / Å")
        ax.set_ylabel("|J| / meV (EHT)")
        ax.set_title("π 堆积滑移与转移积分")
        self.plot.draw()


# ==============================================================================================
# 迁移率方法对比
# ==============================================================================================
class CompareTab(ttk.Frame):
    SRC = ["结构页晶体 J (含符号)", "迁移率页的路径表", "CEDS 结果 (批量扫描页最近一次)", "无定形形貌 (形貌页)"]

    def __init__(self, parent, app):
        from .compare import DEFAULT_METHODS, METHOD_NAMES
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.result = None
        S = safe(app)
        panel = SectionPanel(self, width=470)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner

        c = ttk.LabelFrame(left, text="输入")
        c.pack(fill="x")
        self.src = field(c, 0, "近邻与 J", self.SRC[0], values=self.SRC, width=26)
        self.temp = field(c, 1, "温度 / K", 300, width=26)
        self.lam_h = field(c, 2, "λ 空穴 / eV (空 = 取迁移率页)", "", width=10)
        self.lam_e = field(c, 3, "λ 电子 / eV (空 = 取迁移率页)", "", width=10)
        ttk.Label(c, text="CEDS 来源使用 ⟨V⟩ 与逐键的 σ_V；其他来源的 σ_J 取下面的相对值；"
                          "“Marcus KMC + 极化位点能”需要先在“极化位点能”页计算并送到本页",
                  foreground=PALETTE["muted"], wraplength=440).grid(row=4, column=0, columnspan=4, sticky="w")

        d = ttk.LabelFrame(left, text="无序与弛豫")
        d.pack(fill="x", pady=3)
        self.sig_e = field(d, 0, "能量无序 σ / meV (0 = 不计)", 0, width=10)
        self.sig_j = field(d, 1, "耦合涨落 σ_J / |J|", 0.3, width=10)
        self.htau = field(d, 2, "ħ/τ / meV (TLT、能带)", 5, width=10)

        m = ttk.LabelFrame(left, text="Marcus–Levich–Jortner")
        m.pack(fill="x", pady=3)
        self.hw = field(m, 0, "有效量子模 ħω / meV", 180, width=10)
        self.lam_s = field(m, 1, "经典部分占 λ 的比例 λ_s/λ", 0.3, width=10)
        ttk.Button(m, text="由重组能分解估计 ħω 与 λ_s", command=S(self.mlj_from_decomp)).grid(
            row=2, column=0, columnspan=4, sticky="ew")

        k = ttk.LabelFrame(left, text="KMC")
        k.pack(fill="x", pady=3)
        self.walk = field(k, 0, "载流子数 / 步数", "400 2500", width=12)
        self.field_ = field(k, 1, "场迁移率的电场 / V cm⁻¹", "1e4", width=12)

        mm = ttk.LabelFrame(left, text="方法")
        mm.pack(fill="x", pady=3)
        self.use = {}
        for i, key in enumerate(DEFAULT_METHODS):
            v = tk.BooleanVar(value=True)
            ttk.Checkbutton(mm, text=METHOD_NAMES[key], variable=v).grid(row=i // 2, column=i % 2, sticky="w")
            self.use[key] = v
        self.with_ceds = tk.BooleanVar(value=True)
        ttk.Checkbutton(mm, text="附加 CEDS 结果 (同一温度)", variable=self.with_ceds).grid(
            row=len(DEFAULT_METHODS) // 2 + 1, column=0, columnspan=2, sticky="w")

        b = ttk.Frame(left)
        b.pack(fill="x", pady=4)
        ttk.Button(b, text="一键对比全部方法", style="Accent.TButton", command=S(self.run)).pack(
            side="left", fill="x", expand=True)
        ttk.Button(b, text="导出 CSV 与图…", command=S(self.export)).pack(side="left", fill="x", expand=True)
        ttk.Label(left, text="同一套 J、λ、σ 输入下比较跳跃 (Marcus / MLJ / Miller–Abrahams，解析与 KMC)、"
                             "瞬态局域化与能带模型；每个方法按 2J/λ、σ/kT、σ_J/|J| 给出适用性。",
                  foreground=PALETTE["muted"], wraplength=440).pack(anchor="w")

        right = ttk.Frame(self, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.plot = PlotPanel(right, figsize=(6.5, 4.6))
        self.plot.pack(fill="both", expand=True)
        self.text = styled_text(right, app.mono, height=17, wrap="none", size=9)
        self.text.pack(fill="x", pady=(8, 0))

    def _lam(self, car):
        v = (self.lam_h if car == "hole" else self.lam_e).get().strip()
        if not v:
            mt = self.app.mob_tab
            v = (mt.lam_h if car == "hole" else mt.lam_e).get().strip()
        if not v:
            raise ValueError("请给出重组能 λ (本页或迁移率页)")
        return float(v)

    def mlj_from_decomp(self):
        from .compare import mlj_from_modes
        sets = getattr(self.app.decomp_tab, "sets", None) or {}
        modes = [s["modes"] for s in sets.values() if s and s.get("modes")]
        if not modes:
            raise ValueError("请先在“分解/黄昆因子”页完成重组能分解")
        hw, flow = mlj_from_modes(modes)
        self.hw.set(f"{hw * 1000:.0f}")
        self.lam_s.set(f"{flow:.2f}")
        self.app.log(f"MLJ 参数由分解得到：ħω_eff = {hw * 1000:.0f} meV，低频 (< 1000 cm⁻¹) 部分占 λ 的 {flow:.0%}")

    def _ceds_source(self, T):
        sc = self.app.scan_tab
        rows = getattr(sc, "ce_md_rows", None) or sc.ce_rows
        if sc.ce is None or not (rows or sc.ce_results):
            raise ValueError("请先在“批量扫描工作流”页运行 CEDS (全晶体 MD、局域 MC 的 μ(T) 或 σ_V)")
        if rows:
            row = min(rows, key=lambda r: abs(r["T"] - T))
            return sc.ce, row["res"], row, "md" if rows is getattr(sc, "ce_md_rows", None) else "mc"
        res = sc.ce_results.get(T) or next(iter(sc.ce_results.values()))
        return sc.ce, res, None, "mc"

    def _inputs(self, car, T):
        from .kmc import graph_from_crystal, graph_from_paths
        src = self.SRC.index(self.src.get())
        st = self.app.struct_tab
        if src == 3:
            m = self.app.morph_tab.m
            if m is None or getattr(m, "J", None) is None:
                raise ValueError("请先在“无定形形貌 (MD)”页读取形貌并计算转移积分")
            return m.graph(car), float(self.sig_j.get()), None
        if src == 2:
            ce, res, row, last = self._ceds_source(T)
            g = ce.graph(res, car)
            sig = {"edges": [res[gid][car]["std"] / 1000.0 for gid in g["edge_group"]]}
            return g, sig, ce.crys["lattice"]
        if src == 0:
            coup = self.app.data.get("crystal_couplings")
            if st.mols is None or not coup:
                raise ValueError("请先在“结构/晶体”页打开 CIF 并计算全部二聚体 J")
            g = graph_from_crystal(st.crys, st.mols, coup, float(st.cut.get()),
                                   "atom" if st.crit.get().startswith("最短") else "centroid", car, signed=True)
            return g, float(self.sig_j.get()), st.crys["lattice"]
        key = "Ve_eV" if car == "electron" else "Vh_eV"
        return graph_from_paths(self.app.mob_tab._paths(), key), float(self.sig_j.get()), None

    def _polar(self, car, g):
        if g.get("kind") == "amorphous":
            m = self.app.morph_tab.m
            if getattr(m, "E", None) is None:
                return None
            return {"offsets": list(m.E[car]), "sigma": 0.0}
        d = (self.app.data.get("polar_sites") or {}).get(car)
        if not d:
            return None
        if len(d["offsets"]) != g["n_sites"]:
            raise ValueError("极化位点能的分子数与近邻图的位点数不同 (请对同一晶体计算)")
        return d

    def _ceds_extra(self, car, T):
        sc = self.app.scan_tab
        out = []
        for rows, lab in ((getattr(sc, "ce_md_rows", None), "CEDS 全晶体 MD + TLT"), (sc.ce_rows, "CEDS 局域 MC + TLT")):
            if not rows:
                continue
            r = min(rows, key=lambda x: abs(x["T"] - T))
            if abs(r["T"] - T) < 1e-6 and f"mu_{car}" in r:
                out.append({"key": "ceds", "name": lab, "family": "瞬态局域化", "mu": r[f"mu_{car}"],
                            "mu2": r.get(f"mu2_{car}", float("nan")), "valid": "适用",
                            "why": "耦合涨落由 DFT 耦合曲面与分子运动给出"})
        return out

    def run(self):
        from .compare import compare_methods, plot_comparison, table_text
        T = float(self.temp.get())
        methods = tuple(k for k, v in self.use.items() if v.get())
        if not methods:
            raise ValueError("请至少选择一种方法")
        walkers, steps = (int(x) for x in self.walk.get().split())
        jobs = {}
        for car in ("hole", "electron"):
            lam = self._lam(car)
            g, sig, lat = self._inputs(car, T)
            fs = float(self.lam_s.get() or 0.3)
            sE = float(self.sig_e.get() or 0) / 1000
            if not sE and g.get("kind") == "amorphous" and getattr(self.app.morph_tab.m, "E", None) is not None:
                sE = float(np.std(self.app.morph_tab.m.E[car]))          # 同 σ 的不相关高斯无序作对照
            jobs[car] = dict(graph=g, lam=lam, T=T, carrier=car, methods=methods, sigma_E=sE,
                             sigma_J=sig, hbar_tau=float(self.htau.get()) / 1000, mlj_hw=float(self.hw.get()) / 1000,
                             mlj_lam_s=fs * lam, field=float(self.field_.get() or 0),
                             walkers=walkers, steps=steps, lattice=lat,
                             extra=self._ceds_extra(car, T) if self.with_ceds.get() else (),
                             polar=self._polar(car, g))

        def work():
            return {car: compare_methods(**kw) for car, kw in jobs.items()}

        def done(res):
            self.result = res
            self.text.delete("1.0", "end")
            self.text.insert("end", f"迁移率方法对比 ({self.src.get()})\n\n" + table_text(res))
            ax = self.plot.new_axes()
            plot_comparison(ax, res)
            ax.set_title(f"迁移率方法对比，T = {T:g} K (浅色 = 不适用)", fontsize=10)
            self.plot.draw()
            self.app.data["method_comparison"] = res
            self.app.log(f"迁移率方法对比完成：{len(methods)} 种方法 × 2 种载流子")
        self.app.run_bg("迁移率方法对比", work, done)

    def export(self, path=None):
        from .compare import export_csv
        if not self.result:
            raise ValueError("请先运行对比")
        p = path or filedialog.asksaveasfilename(defaultextension=".csv", initialfile="mobility_methods.csv",
                                                 filetypes=[("CSV", "*.csv")])
        if not p:
            return None
        export_csv(self.result, p)
        self.plot.fig.savefig(str(Path(p).with_suffix(".png")), dpi=300)
        self.app.log(f"已导出 {p} 与 {Path(p).with_suffix('.png').name} (300 dpi)")
        return p


class PolarTab(ttk.Frame):
    """极化位点能：原子电荷 + Thole 诱导偶极 (微静电模型)。"""

    def __init__(self, parent, app):
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.res, self.md, self.levels, self.ce = None, None, None, None
        S = safe(app)
        panel = SectionPanel(self, width=470)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner

        c = ttk.LabelFrame(left, text="晶体与单体")
        c.pack(fill="x")
        self.cif = FileEntry(c, "晶体 CIF", filetypes=[("CIF", "*.cif")], width=26)
        self.cif.pack(fill="x")
        self.mono = FileEntry(c, "单体 (可选)", filetypes=STRUCT_TYPES + [("项目", "*.json")], width=26)
        self.mono.pack(fill="x")
        f = ttk.Frame(c)
        f.pack(fill="x")
        self.frame_ = field(f, 0, "分子坐标系", "all", values=["all", "core"], width=8)

        q = ttk.LabelFrame(left, text="原子电荷 (Gaussian 输出，可选)")
        q.pack(fill="x", pady=3)
        logs = [("Gaussian 输出", "*.log *.out"), ("所有文件", "*.*")]
        self.q_n = FileEntry(q, "中性", filetypes=logs, width=26)
        self.q_n.pack(fill="x")
        self.q_c = FileEntry(q, "阳离子", filetypes=logs, width=26)
        self.q_c.pack(fill="x")
        self.q_a = FileEntry(q, "阴离子", filetypes=logs, width=26)
        self.q_a.pack(fill="x")
        ttk.Button(q, text="生成 ESP 电荷与极化率的 Gaussian 输入…", command=S(self.make_gjf)).pack(fill="x")
        ttk.Label(q, text="不给出时：中性用 OPLS-AA 芳香 C/H 电荷，离子用 EHT 前线轨道布居 (只适合碳氢化合物)",
                  foreground=PALETTE["muted"], wraplength=440).pack(anchor="w")

        a = ttk.LabelFrame(left, text="极化率与团簇")
        a.pack(fill="x", pady=3)
        self.amean = field(a, 0, "分子平均极化率 / Å³ (空 = 不缩放)", "", width=10)
        self.radii = field(a, 1, "团簇半径 / Å (加 26 30 更精确，约 20 分钟)", "14 18 22", width=14)
        self.ip = field(a, 2, "气相 IE / eV (可选)", "", width=10)
        self.ea = field(a, 3, "气相 EA / eV (可选)", "", width=10)
        ttk.Button(a, text="由中性输出的 polar 结果填入极化率", command=S(self.alpha_from_log)).grid(
            row=4, column=0, columnspan=4, sticky="ew")

        m = ttk.LabelFrame(left, text="MD 快照 (热涨落 σ_E)")
        m.pack(fill="x", pady=3)
        self.traj = FileEntry(m, "MD 轨迹", filetypes=[("npz", "*.npz")], width=26)
        self.traj.pack(fill="x")
        g = ttk.Frame(m)
        g.pack(fill="x")
        self.md_nf = field(g, 0, "帧数 / 每帧分子数", "24 4", width=10)
        self.md_r = field(g, 1, "快照团簇半径 / Å", 14, width=10)
        self.md_T = field(g, 2, "无轨迹时运行 MD：T / K、采样 ps", "300 10", width=10)
        self.md_sc = field(g, 3, "MD 超胞", "4 3 2", width=10)

        b = ttk.Frame(left)
        b.pack(fill="x", pady=4)
        ttk.Button(b, text="计算晶体极化能", style="Accent.TButton", command=S(self.run)).pack(
            side="left", fill="x", expand=True)
        ttk.Button(b, text="MD 快照 σ_E", command=S(self.run_md)).pack(side="left", fill="x", expand=True)
        b2 = ttk.Frame(left)
        b2.pack(fill="x")
        ttk.Button(b2, text="送到方法对比页", command=S(self.to_compare)).pack(side="left", fill="x", expand=True)
        ttk.Button(b2, text="导出 CSV 与图…", command=S(self.export)).pack(side="left", fill="x", expand=True)
        ttk.Label(left, text="把晶体中一个分子换成带电态：静电 (电荷–四极) + 诱导 (Thole 原子极化率，按 1/R 外推)，"
                             "得到固态极化能 P±、各独立分子的位点能差；MD 快照给出位点能的热涨落 σ_E。",
                  foreground=PALETTE["muted"], wraplength=440).pack(anchor="w")

        right = ttk.Frame(self, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.plot = PlotPanel(right, figsize=(6.5, 4.2))
        self.plot.pack(fill="both", expand=True)
        self.text = styled_text(right, app.mono, height=17, wrap="none", size=9)
        self.text.pack(fill="x", pady=(8, 0))

    # ---- 输入 ----
    def _cif(self):
        p = self.cif.get() or (str(self.app.current_structure or "") if str(self.app.current_structure or "")
                               .lower().endswith(".cif") else "")
        if not p:
            raise ValueError("请选择晶体 CIF (或先在“结构/晶体”页打开)")
        return p

    def _ceds(self):
        from .ceds import CEDS
        from .cli import _read_monomer
        from .crystal import find_molecules, read_cif
        p, mp, fr = self._cif(), self.mono.get(), self.frame_.get()
        key = (p, mp, fr)
        if self.ce is None or self.ce[0] != key:
            crys = read_cif(p)
            mols = find_molecules(crys)
            mono = _read_monomer(mp, fr) if mp else None
            if mono and sorted(mono[0]) != sorted(mols[0]["symbols"]):
                raise ValueError("单体的组成与晶体中的分子不同")
            self.ce = (key, CEDS(crys, mols, cutoff=4.0, monomer=mono, frame=fr))
        return self.ce[1]

    def _charges(self, sym, Y):
        from .polar import charge_set
        return charge_set(sym, Y, self.q_n.get() or None, self.q_c.get() or None, self.q_a.get() or None)

    def _table(self, ce):
        from .polar import scale_alpha
        v = self.amean.get().strip()
        if not v:
            return None, ""
        table, s = scale_alpha(ce.sym, ce.Y, float(v))
        return table, f"原子极化率缩放 ×{s:.3f} (分子平均极化率 {float(v):.1f} Å³)"

    def _opt(self, w):
        v = w.get().strip()
        return float(v) if v else None

    def make_gjf(self):
        from .polar import write_charge_inputs
        ce = self._ceds()
        d = filedialog.askdirectory(title=T("选择输出目录"))
        if not d:
            return None
        files = write_charge_inputs(ce.sym, ce.Y, str(Path(d) / Path(self._cif()).stem))
        self.app.log("已生成 " + ", ".join(Path(f).name for f in files))
        return files

    def alpha_from_log(self):
        from .polar import read_polarizability
        if not self.q_n.get():
            raise ValueError("请先选择中性分子的 Gaussian 输出 (含 polar 结果)")
        A = read_polarizability(self.q_n.get())
        self.amean.set(f"{np.trace(A) / 3:.2f}")
        self.app.log(f"DFT 分子平均极化率 {np.trace(A) / 3:.1f} Å³")

    # ---- 计算 ----
    def run(self):
        from .polar import site_energies, solid_levels
        ce = self._ceds()
        ch = self._charges(ce.sym, ce.Y)
        table, note = self._table(ce)
        radii = [float(x) for x in self.radii.get().replace(",", " ").split()]
        ip, ea = self._opt(self.ip), self._opt(self.ea)

        def work():
            return site_energies(ce.crys["lattice"], ce.frames, ce.sym, ce.Y, ch, radii, table)

        def done(res):
            self.res, self.levels = res, solid_levels(res, ip, ea)
            self._table_scale = table
            self._show(note)
            self.app.log(f"晶体极化能：P+ = {res['P_plus']:.2f} eV，P− = {res['P_minus']:.2f} eV")
        self.app.run_bg("晶体极化能", work, done)

    def run_md(self):
        from .polar import load_traj, md_site_energies, traj_from_md
        ce = self._ceds()
        table, note = self._table(ce)
        nf, nc = (int(x) for x in self.md_nf.get().split())
        R = float(self.md_r.get())
        path = self.traj.get()

        def work():
            if path:
                tr = load_traj(path)
            else:
                from .rigidmd import RigidCrystalMD, run_temperature
                T_, ps = (float(x) for x in self.md_T.get().split())
                sc = tuple(int(x) for x in self.md_sc.get().replace(",", " ").split())
                md = RigidCrystalMD(ce.crys, ce.frames, (ce.sym, ce.Y), sc)
                tr = traj_from_md(md, run_temperature(md, T_, dict(prod_ps=ps)))
            ch = self._charges(tr["symbols"], tr["Y"])
            return md_site_energies(tr, ch, nf, nc, R, nk=len(ce.frames), alpha_table=table)

        def done(md):
            self.md = md
            self._show(note)
            self.app.log(f"MD 快照位点能：σ_E 空穴 {md['sigma_h'] * 1000:.0f} meV，电子 {md['sigma_e'] * 1000:.0f} meV")
        self.app.run_bg("MD 快照位点能", work, done)

    def _show(self, note=""):
        from .polar import plot_extrapolation, plot_md_hist, summary_text
        self.text.delete("1.0", "end")
        if self.res is not None:
            txt = summary_text(self.res, self.levels, self.md)
        else:
            txt = (f"MD 快照 (R = {self.md['R']:g} Å)：σ_E 空穴 {1000 * self.md['sigma_h']:.0f} meV，"
                   f"电子 {1000 * self.md['sigma_e']:.0f} meV")
        self.text.insert("end", (note + "\n" if note else "") + txt)
        self.plot.fig.clf()
        n = (self.res is not None) + (self.md is not None)
        k = 1
        if self.res is not None:
            plot_extrapolation(self.plot.fig.add_subplot(1, n, k), self.res)
            k += 1
        if self.md is not None:
            plot_md_hist(self.plot.fig.add_subplot(1, n, k), self.md)
        self.plot.fig.tight_layout()
        self.plot.draw()

    def site_data(self):
        """供方法对比页使用：{载流子: {"offsets", "sigma"}}。"""
        if self.res is None and self.md is None:
            return None
        out = {}
        for car, k in (("hole", "h"), ("electron", "e")):
            offs = self.res[f"offsets_{k}"] if self.res is not None else list(np.zeros(self.md["nk"]))
            out[car] = {"offsets": list(offs), "sigma": self.md[f"sigma_{k}"] if self.md is not None else 0.0}
        return out

    def to_compare(self):
        d = self.site_data()
        if not d:
            raise ValueError("请先计算晶体极化能或 MD 快照 σ_E")
        self.app.data["polar_sites"] = d
        cp = self.app.cmp_tab
        if "kmc_polar" in cp.use:
            cp.use["kmc_polar"].set(True)
        self.app.show_page("迁移率方法对比")
        self.app.log("极化位点能已送到方法对比页 (方法“Marcus KMC + 极化位点能”)")

    def export(self, path=None):
        from .polar import export_csv
        if self.res is None:
            raise ValueError("请先计算晶体极化能")
        p = path or filedialog.asksaveasfilename(defaultextension=".csv", initialfile="polarization.csv",
                                                 filetypes=[("CSV", "*.csv")])
        if not p:
            return None
        export_csv(self.res, p, self.md, self.levels)
        self.plot.fig.savefig(str(Path(p).with_suffix(".png")), dpi=300)
        self.app.log(f"已导出 {p} 与 {Path(p).with_suffix('.png').name} (300 dpi)")
        return p


class MorphTab(ttk.Frame):
    """无定形 / MD 形貌：GROMACS 形貌 → 分子映射 → 近邻 → 转移积分 → 位点能 → KMC。"""
    GEOM = ["叠合后的参考单体 (qm)", "MD 原子坐标 (md)"]
    EMODE = ["静电 (原子电荷)", "静电 + Thole 诱导", "不计位点能"]
    JMETH = ["EHT (软件内)", "ZINDO/MOO (VOTCA izindo)"]
    PICK = ["按 |J| 分布挑代表性对 (校准用)", "按序号取前 N 对"]

    def __init__(self, parent, app):
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.m, self.kres = None, {}
        S = safe(app)
        panel = SectionPanel(self, width=470)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner

        c = ttk.LabelFrame(left, text="形貌与参考单体")
        c.pack(fill="x")
        self.gro = FileEntry(c, "形貌文件", filetypes=[("GROMACS / PDB", "*.gro *.pdb"), ("所有文件", "*.*")], width=26)
        self.gro.pack(fill="x")
        self.traj = FileEntry(c, "轨迹 (可选)", filetypes=[("轨迹", "*.xtc *.trr *.dcd"), ("所有文件", "*.*")], width=26)
        self.traj.pack(fill="x")
        self.ref = FileEntry(c, "参考单体", filetypes=STRUCT_TYPES + [("项目", "*.json")], width=26)
        self.ref.pack(fill="x")
        f = ttk.Frame(c)
        f.pack(fill="x")
        self.frame_ = field(f, 0, "帧序号", 0, width=8)
        self.resn = field(f, 1, "残基名 (空 = 自动)", "", width=8)
        self.elmap = field(f, 2, "原子名 → 元素 (如 CA=C)", "", width=14)
        self.geom = field(f, 3, "几何", self.GEOM[0], values=self.GEOM, width=22)
        self.cut = field(f, 4, "近邻截断 / Å", 4.0, width=8)
        self.jm = field(f, 5, "转移积分方法", self.JMETH[0], values=self.JMETH, width=22)
        self.jdeg = field(f, 6, "简并阈值 / meV (ZINDO)", 0, width=8)
        ttk.Button(c, text="读取、映射、近邻与转移积分", style="Accent.TButton", command=S(self.load)).pack(fill="x")

        j = ttk.LabelFrame(left, text="DFT 转移积分与校准 (可选)")
        j.pack(fill="x", pady=3)
        jf = ttk.Frame(j)
        jf.pack(fill="x")
        self.maxp = field(jf, 0, "导出 N 对 (0 = 全部)", 20, width=8)
        self.pick = field(jf, 1, "选取方式", self.PICK[0], values=self.PICK, width=24)
        b = ttk.Frame(j)
        b.pack(fill="x")
        ttk.Button(b, text="导出 Gaussian 二聚体输入…", command=S(self.export_dimers)).pack(side="left", fill="x", expand=True)
        ttk.Button(b, text="读回并直接替换…", command=S(self.collect)).pack(side="left", fill="x", expand=True)
        ttk.Button(j, text="用 DFT 结果校准全部对 (空穴、电子各拟合 k)…", style="Accent.TButton",
                   command=S(self.calibrate)).pack(fill="x", pady=(2, 0))
        kf = ttk.Frame(j)
        kf.pack(fill="x")
        self.kh = field(kf, 0, "手动 k 空穴", "", width=8)
        self.ke = field(kf, 1, "手动 k 电子", "", width=8)
        ttk.Button(kf, text="应用", command=S(self.apply_k)).grid(row=0, column=2, rowspan=2, sticky="ns", padx=4)
        self.calib_lbl = ttk.Label(j, text="DFT 校准：先用 EHT / ZINDO 算全部对，导出约 20 对代表性二聚体做 DFT，再校准",
                                   foreground=PALETTE["muted"], wraplength=440, justify="left")
        self.calib_lbl.pack(fill="x")
        ttk.Button(j, text="读入转移积分 CSV (pair, J_hole_meV, J_electron_meV)…", command=S(self.import_j)).pack(fill="x")

        e = ttk.LabelFrame(left, text="位点能")
        e.pack(fill="x", pady=3)
        self.emode = field(e, 0, "方式", self.EMODE[0], values=self.EMODE, width=20)
        self.er = field(e, 1, "截断半径 / Å (空 = 默认)", "", width=8)
        ttk.Label(e, text="原子电荷取“极化位点能”页的设置 (没有则用默认电荷)", foreground=PALETTE["muted"],
                  wraplength=440).grid(row=2, column=0, columnspan=4, sticky="w")
        ttk.Button(e, text="计算位点能", command=S(self.energies)).grid(row=3, column=0, columnspan=4, sticky="ew")

        k = ttk.LabelFrame(left, text="KMC")
        k.pack(fill="x", pady=3)
        self.temp = field(k, 0, "温度 / K", 300, width=8)
        self.lam_h = field(k, 1, "λ 空穴 / eV (空 = 取迁移率页)", "", width=8)
        self.lam_e = field(k, 2, "λ 电子 / eV (空 = 取迁移率页)", "", width=8)
        self.field_ = field(k, 3, "场迁移率的电场 / V cm⁻¹ (0 = 不算)", "1e5", width=8)
        self.walk = field(k, 4, "载流子数 / 步数", "1000 4000", width=12)
        b2 = ttk.Frame(k)
        b2.grid(row=5, column=0, columnspan=4, sticky="ew")
        ttk.Button(b2, text="KMC 迁移率", command=S(self.run_kmc)).pack(side="left", fill="x", expand=True)
        ttk.Button(b2, text="送到方法对比页", command=S(self.to_compare)).pack(side="left", fill="x", expand=True)
        ttk.Button(left, text="导出 CSV 与图…", command=S(self.export)).pack(fill="x", pady=(4, 0))
        ttk.Label(left, text="与 VOTCA-CTP 相同的流程：MD 形貌中的每个分子用参考单体刚性叠合，按最短原子距离找近邻，"
                             "计算转移积分与位点能，在整个周期盒子上做 KMC。",
                  foreground=PALETTE["muted"], wraplength=440).pack(anchor="w")

        right = ttk.Frame(self, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.plot = PlotPanel(right, figsize=(6.5, 5.0))
        self.plot.pack(fill="both", expand=True)
        self.text = styled_text(right, app.mono, height=14, wrap="none", size=9)
        self.text.pack(fill="x", pady=(8, 0))

    def _need(self):
        if self.m is None:
            raise ValueError("请先读取形貌")
        return self.m

    def _lam(self, car):
        v = (self.lam_h if car == "hole" else self.lam_e).get().strip()
        if not v:
            mt = self.app.mob_tab
            v = (mt.lam_h if car == "hole" else mt.lam_e).get().strip()
        return float(v) if v else None

    def _show(self):
        m = self.m
        lines = [m.summary()]
        for car, (r0, r1, rf) in self.kres.items():
            lab = T("空穴") if car == "hole" else T("电子")
            ln = f"{lab} KMC：只有 J 的无序 μ = {r0['mu_avg']:.3g}"
            if r1:
                ln += f"；加位点能 μ = {r1['mu_avg']:.3g} ± {3 * r1['err']:.2g}"
            if rf:
                ln += f"；场迁移率 μ = {rf['mu_avg']:.3g}"
            lines.append(ln + " cm²/Vs")
        self.text.delete("1.0", "end")
        self.text.insert("end", "\n".join(lines))
        from .morphology import plot_overview
        self.plot.fig.clf()
        plot_overview(self.plot.fig, m)
        self.plot.draw()

    def load(self):
        from .cli import _element_map, _read_monomer
        from .morphology import Morphology, read_trajectory
        if not self.gro.get() or not self.ref.get():
            raise ValueError("请选择形貌文件与参考单体")
        fr_i = int(self.frame_.get() or 0)
        args = (self.gro.get(), self.traj.get() or None, {fr_i})
        sym, Y = _read_monomer(self.ref.get())
        geom = "md" if self.geom.get() == self.GEOM[1] else "qm"
        cut = float(self.cut.get())
        emap = _element_map(self.elmap.get())
        resn = self.resn.get().strip() or None
        zin = self.jm.get() == self.JMETH[1]
        zorb = self.app.ti_tab.z_orb.get() or None
        zdeg = float(self.jdeg.get() or 0) / 1000

        def work():
            frames = read_trajectory(*args)
            if not frames:
                raise ValueError(f"形貌中没有这一帧 (序号 {fr_i})")
            m = Morphology(frames[0], sym, Y, resn, emap, geom)
            m.neighbors(cut)
            if zin:
                m.couplings_zindo(zorb, zdeg)
            else:
                m.couplings_eht()
            return m

        def done(m):
            self.m, self.kres = m, {}
            self._show()
            self.app.log(f"形貌：{m.M} 个分子，{len(m.pairs)} 个近邻对 ({m.J_source} 转移积分)")
        self.app.run_bg("读取形貌与转移积分", work, done)

    def export_dimers(self):
        m = self._need()
        d = filedialog.askdirectory(title=T("选择输出目录"))
        if not d:
            return None
        n = int(self.maxp.get() or 0)
        if n and self.pick.get() == self.PICK[0] and getattr(m, "J_raw", None) is not None:
            idx = m.calibration_pairs(n)
        else:
            idx = range(min(n, len(m.pairs))) if n else None
        made = m.export_dimers(d, idx)
        self.app.log(f"已为 {len(made)} 个近邻对写出 Gaussian 投影法输入")
        return made

    def calibrate(self, d=None):
        m = self._need()
        d = d or filedialog.askdirectory(title=T("选择 Gaussian 结果目录"))
        if not d:
            return None
        ref = m.read_dft(d)
        if not ref:
            raise ValueError(T("该目录中没有完成的二聚体 (pair_xxxxx/dimer_A、B、AB.fchk)"))
        res = m.calibrate(ref)
        self._calib_done()
        return res

    def apply_k(self):
        m = self._need()
        kh = float(self.kh.get()) if str(self.kh.get()).strip() else 1.0
        ke = float(self.ke.get()) if str(self.ke.get()).strip() else 1.0
        res = m.calibrate_scale(kh, ke)
        self._calib_done()
        return res

    def _calib_done(self):
        txt = self.m.calibration_text()
        self.calib_lbl.configure(text=txt)
        self._show()
        self.app.log("DFT 校准：\n" + txt)

    def collect(self, d=None):
        m = self._need()
        d = d or filedialog.askdirectory(title=T("选择 Gaussian 结果目录"))
        if not d:
            return None
        n = m.collect_dft(d)
        self._show()
        self.app.log(f"读回 {n} 对 Gaussian 转移积分")
        return n

    def import_j(self, p=None):
        m = self._need()
        p = p or filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not p:
            return None
        n = m.import_couplings(p)
        self._show()
        self.app.log(f"读入 {n} 个转移积分")
        return n

    def energies(self):
        m = self._need()
        mode = ["static", "polar", "none"][self.EMODE.index(self.emode.get())]
        if mode == "none":
            m.E = None
            self._show()
            return
        pt = self.app.pol_tab
        ch = pt._charges(m.ref_sym, m.ref_Y)
        R = float(self.er.get()) if self.er.get().strip() else None

        def work():
            m.site_energies(ch, mode, R)
            return m

        def done(_):
            self._show()
            self.app.log(f"形貌位点能：σ 空穴 {1000 * m.E['hole'].std():.0f} meV，电子 {1000 * m.E['electron'].std():.0f} meV")
        self.app.run_bg("形貌位点能", work, done)

    def run_kmc(self):
        m = self._need()
        if getattr(m, "J", None) is None:
            raise ValueError("没有转移积分")
        T_ = float(self.temp.get())
        F = float(self.field_.get() or 0)
        wk, st = (int(x) for x in self.walk.get().split())
        lams = {c: self._lam(c) for c in ("hole", "electron")}
        if not any(lams.values()):
            raise ValueError("请给出重组能 λ (本页或迁移率页)")
        hasE = getattr(m, "E", None) is not None

        def work():
            out = {}
            for car, lam in lams.items():
                if not lam:
                    continue
                r0 = m.kmc(car, lam, T_, 0.0, wk, st, use_energies=False)
                r1 = m.kmc(car, lam, T_, 0.0, wk, st) if hasE else None
                rf = m.kmc(car, lam, T_, F, wk, st) if (F and hasE) else None
                out[car] = (r0, r1, rf)
            return out

        def done(res):
            self.kres = res
            self._show()
            self.app.log("形貌 KMC 完成")
        self.app.run_bg("形貌 KMC", work, done)

    def to_compare(self):
        self._need()
        cp = self.app.cmp_tab
        cp.src.set(cp.SRC[3])
        if "kmc_polar" in cp.use:
            cp.use["kmc_polar"].set(True)
        self.app.show_page("迁移率方法对比")
        self.app.log("方法对比页已切换到无定形形貌 (位点能按分子给出)")

    def export(self, path=None):
        m = self._need()
        p = path or filedialog.asksaveasfilename(defaultextension=".csv", initialfile="morphology_sites.csv",
                                                 filetypes=[("CSV", "*.csv")])
        if not p:
            return None
        prefix = str(Path(p).with_suffix("")).removesuffix("_sites")
        files = m.export(prefix)
        self.plot.fig.savefig(prefix + ".png", dpi=300)
        self.app.log("已导出 " + ", ".join(Path(f).name for f in files) + f" 与 {Path(prefix).name}.png")
        return files


# ==============================================================================================
# 分子间相互作用能 / 能量框架 / 晶格能
# ==============================================================================================
class InteractionTab(ttk.Frame):
    SRC = ["结构页全部近邻二聚体", "单个二聚体文件"]
    METHODS = ["UFF 力场 (秒级, 定性)"]

    def __init__(self, parent, app):
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.energies, self.method_label = {}, ""
        S = safe(app)
        panel = SectionPanel(self, width=450)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner
        c = ttk.LabelFrame(left, text="数据来源")
        c.pack(fill="x")
        cf = ttk.Frame(c)
        cf.pack(fill="x")
        self.src = field(cf, 0, "二聚体", self.SRC[0], values=self.SRC, width=24)
        self.dimer = FileEntry(c, "二聚体文件", filetypes=STRUCT_TYPES, command=lambda p: self._auto_na(p))
        self.dimer.pack(fill="x")
        cf2 = ttk.Frame(c)
        cf2.pack(fill="x")
        self.na = field(cf2, 0, "单体A原子数", "", width=8)

        m = ttk.LabelFrame(left, text="快速计算 (软件内 UFF 力场)")
        m.pack(fill="x", pady=3)
        mf = ttk.Frame(m)
        mf.pack(fill="x")
        self.method = field(mf, 0, "方法", self.METHODS[0], values=self.METHODS, width=24)
        self.charges = FileEntry(m, "原子电荷(可选)", filetypes=[("Gaussian 输出", "*.log *.out"), ("所有文件", "*.*")])
        self.charges.pack(fill="x")
        ttk.Label(m, text="UFF 可加入静电项：选择单体的 Gaussian 输出 (pop=mk 的 ESP 电荷优先，否则 Mulliken)",
                  foreground=PALETTE["muted"], wraplength=420).pack(anchor="w")
        ttk.Button(m, text="计算相互作用能", style="Accent.TButton", command=S(self.compute)).pack(fill="x", pady=(4, 0))

        g = ttk.LabelFrame(left, text="Gaussian 抵消校正 (Counterpoise, 用于发表)")
        g.pack(fill="x", pady=3)
        gf = ttk.Frame(g)
        gf.pack(fill="x")
        self.g_func = field(gf, 0, "泛函", "B3LYP", values=["B3LYP", "PBE0", "wB97XD", "M062X", "CAM-B3LYP"], width=14)
        self.g_basis = field(gf, 1, "基组", "6-31G(d)", values=["6-31G(d)", "6-31G(d,p)", "6-311G(d,p)", "def2SVP",
                                                               "def2TZVP"], width=14)
        self.g_disp = field(gf, 2, "色散校正", "GD3BJ", values=["GD3BJ", "GD3", "无"], width=14)
        self.g_cpu = field(gf, 3, "核数 / 内存", "8", width=6)
        self.g_mem = field(gf, 3, "", "8GB", width=6, col=2)
        self.g_dir = FileEntry(g, "输出目录", mode="dir")
        self.g_dir.pack(fill="x")
        gb = ttk.Frame(g)
        gb.pack(fill="x")
        ttk.Button(gb, text="生成 Gaussian 输入", command=S(self.gen_cp)).pack(side="left", fill="x", expand=True)
        ttk.Button(gb, text="读取 Gaussian 结果", command=S(self.read_cp)).pack(side="left", fill="x", expand=True)
        ttk.Label(g, text="生成后可在“运行任务”中运行，结束后自动读取；含 Ir 等重原子时请在高斯命令中改用赝势基组",
                  foreground=PALETTE["muted"], wraplength=420).pack(anchor="w")

        f = ttk.LabelFrame(left, text="能量框架 (参考 CrystalExplorer)")
        f.pack(fill="x", pady=3)
        ff = ttk.Frame(f)
        ff.pack(fill="x")
        self.f_cell = field(ff, 0, "晶胞数 n×n×n", "2", values=["1", "2", "3"], width=8)
        self.f_rmax = field(ff, 1, "最粗圆柱半径 / Å", "0.6", width=8)
        self.f_min = field(ff, 2, "忽略 |E| < (kJ/mol)", "2", width=8)
        self.f_mol = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="显示分子", variable=self.f_mol).pack(anchor="w")
        fb = ttk.Frame(f)
        fb.pack(fill="x")
        ttk.Button(fb, text="绘制能量框架", command=S(self.draw_framework)).pack(side="left", fill="x", expand=True)
        ttk.Button(fb, text="保存图片…", command=S(self.save_framework)).pack(side="left", fill="x", expand=True)
        ttk.Button(fb, text="导出 CSV…", command=S(self.export_csv)).pack(side="left", fill="x", expand=True)

        right = ttk.Frame(self, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        top = ttk.Frame(right, style="Bg.TFrame")
        top.pack(fill="both", expand=True)
        self.view = ViewerPanel(top, figsize=(4.6, 4.2))
        self.view.grid(row=0, column=0, sticky="nsew")
        self.plot = PlotPanel(top, figsize=(4.2, 4.2))
        self.plot.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        top.columnconfigure((0, 1), weight=1, uniform="half")          # 结构图与能量图各占一半，不互相挤压
        top.rowconfigure(0, weight=1)
        self.view.canvas.mpl_connect("scroll_event", self._scroll)
        self.view.canvas.mpl_connect("button_press_event", self._press)
        self.view.canvas.mpl_connect("motion_notify_event", self._motion)
        self.view.canvas.mpl_connect("button_release_event", self._release)
        self._drag = None
        self.text = styled_text(right, app.mono, height=12, wrap="none", size=9)
        self.text.pack(fill="x", pady=(8, 0))

    # ---- 视图交互 ----
    def _scroll(self, e):
        self.view.zoom(1.15 if e.button == "up" else 1 / 1.15)

    def _press(self, e):
        self._drag = (e.x, e.y) if e.x is not None else None

    def _motion(self, e):
        if not self._drag or e.x is None:
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self._drag = (e.x, e.y)
        self.view.renderer.rotate(dx, dy)
        self.view.renderer.draw(fast=True)
        self.view.canvas.draw_idle()

    def _release(self, e):
        if self._drag:
            self._drag = None
            self.view.renderer.draw()             # 松开鼠标后恢复完整绘制
            self.view.canvas.draw_idle()

    # ---- 计算 ----
    def _auto_na(self, p):
        """按成键拆分二聚体；原子顺序不是 A 在前、B 在后时，另存为 *_sorted.xyz 并改用它。"""
        if not p:
            return
        n_a, reordered, new = split_dimer(read_geometry(p))
        if reordered:
            newp = Path(p).with_name(Path(p).stem + "_sorted.xyz")
            write_xyz(newp, new, f"reordered dimer nA={n_a}")
            self.dimer.set(newp)
            self.app.log(f"二聚体原子顺序已重排为 A 在前、B 在后: {newp.name}")
        self.na.set(str(n_a))

    def _pairs(self):
        """[(组号, atoms_a, atoms_b, r)]。"""
        if self.src.get() == self.SRC[0]:
            st = self.app.struct_tab
            if not st.dimer_res:
                raise ValueError("请先在“结构/晶体”页打开 CIF 并搜索近邻二聚体")
            a = [(s_, *x) for s_, x in st.dimer_res["atoms_a"]]
            return [(g["id"], a, [(s_, *x) for s_, x in g["members"][0]["atoms_b"]], g["members"][0]["r_centroid"])
                    for g in st.dimer_res["groups"]]
        if not self.dimer.get():
            raise ValueError("请选择二聚体文件")
        if not self.na.get():
            self._auto_na(self.dimer.get())
        at = read_geometry(self.dimer.get())
        n = int(self.na.get())
        ca = np.mean([x[1:] for x in at[:n]], 0)
        cb = np.mean([x[1:] for x in at[n:]], 0)
        return [(1, at[:n], at[n:], float(np.linalg.norm(cb - ca)))]

    def compute(self):
        from .interaction import read_charges, uff_interaction, xtb_interaction
        pairs = self._pairs()
        use_xtb = self.method.get().startswith("xtb")
        q = None
        if not use_xtb and self.charges.get():
            q, kind = read_charges(self.charges.get())
            self.app.log(f"使用 {kind} 电荷 ({len(q)} 个原子)")
        exe = None
        if use_xtb:
            from .programs import find_xtb
            exe = find_xtb(self.app.config_data.get("xtb_exe") or None)
            if not exe:
                raise FileNotFoundError("没有找到 xtb，请在 工具 → 程序路径设置 中指定")
        wd = Path(self.app.config_data.get("workdir") or Path.home()) / "mobilis_xtb_interaction"

        def work():
            cache, out = {}, {}
            for gid, a, b, r in pairs:
                e = xtb_interaction(a, b, exe, wd, cache=cache) if use_xtb else uff_interaction(a, b, q)
                e["r"] = r
                out[gid] = e
            return out

        def done(out):
            self.energies = out
            self.method_label = "GFN2-xTB" if use_xtb else ("UFF + 静电" if q is not None else "UFF (仅范德华)")
            self.show()
        self.app.run_bg("计算分子间相互作用能", work, done)

    def gen_cp(self):
        from .interaction import prepare_counterpoise_jobs, write_counterpoise_gjf
        d = self.g_dir.get()
        if not d:
            raise ValueError("请选择输出目录")
        kw = dict(functional=self.g_func.get(), basis=self.g_basis.get(), dispersion=self.g_disp.get(),
                  nproc=int(self.g_cpu.get()), mem=self.g_mem.get())
        if self.src.get() == self.SRC[0]:
            st = self.app.struct_tab
            if not st.dimer_res:
                raise ValueError("请先在“结构/晶体”页搜索近邻二聚体")
            names = prepare_counterpoise_jobs(st.dimer_res, d, **kw)
            import json as _json
            (Path(d) / "dimers.json").write_text(_json.dumps({"neighbors": [
                {"group": x["group"], "vector": list(map(float, x["vector"]))} for x in st.dimer_res["dimers"]]},
                ensure_ascii=False))
        else:
            (_, a, b, _r), = self._pairs()
            write_counterpoise_gjf(Path(d) / "dimer_cp.gjf", a, b, **kw)
            from .jobs import write_windows_bat
            write_windows_bat(d, ["dimer_cp"], name="run_cp.bat")
            names = ["dimer_cp"]
        self.app.log(f"已在 {d} 生成 {len(names)} 个抵消校正输入 ({kw['functional']}/{kw['basis']}"
                     f"{'' if kw['dispersion'] == '无' else ' + ' + kw['dispersion']})，以及 run_cp.bat")
        if messagebox.askyesno(T("完成"), T("已生成 Gaussian 抵消校正输入。现在打开“运行任务”窗口吗？")):
            self.app.open_runner(d)

    def read_cp(self, d=None):
        from .interaction import collect_counterpoise, read_counterpoise
        d = d or self.g_dir.get()
        if not d:
            raise ValueError("请选择 Gaussian 结果所在目录")
        out, err = collect_counterpoise(d)
        one = Path(d) / "dimer_cp.log"
        if not out and one.exists():
            out = {1: read_counterpoise(one)}
        if not out:
            raise ValueError("目录中没有可读取的 Counterpoise 结果 (G<n>_cp.log)" + (f"：{err}" if err else ""))
        rs = {}
        st = self.app.struct_tab
        if st.dimer_res:
            rs = {g["id"]: g["members"][0]["r_centroid"] for g in st.dimer_res["groups"]}
        for gid, e in out.items():
            e["r"] = rs.get(gid, float("nan"))
        for gid, e in err.items():
            self.app.log(f"G{gid}: {e}")
        self.energies = out
        self.method_label = f"Gaussian 抵消校正 ({len(out)} 组)"
        self.show()
        return out

    # ---- 显示 ----
    def show(self):
        from .interaction import lattice_energy
        E = self.energies
        coup = self.app.data.get("crystal_couplings") or {}
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"分子间相互作用能 ({self.method_label})，单位 kJ/mol；负值为吸引\n")
        t.insert("end", f"{'组':<5}{'r质心/Å':>9}{'E_int':>10}{'vdW':>10}{'静电':>10}{'BSSE':>9}{'Jh/meV':>9}{'Je/meV':>9}\n")
        f_ = lambda v, w=10, p=2: f"{v:{w}.{p}f}" if isinstance(v, (int, float)) and v == v else f"{'—':>{w}}"  # noqa: E731
        for gid in sorted(E):
            e, c = E[gid], coup.get(gid, {})
            t.insert("end", f"G{gid:<4}{f_(e.get('r'), 9)}{f_(e['total'])}{f_(e.get('vdw'))}{f_(e.get('elec'))}"
                            f"{f_(e.get('BSSE'), 9)}{f_(c.get('hole'), 9)}{f_(c.get('electron'), 9)}\n")
        st = self.app.struct_tab
        if self.src.get() == self.SRC[0] and st.dimer_res:
            el, miss = lattice_energy(st.dimer_res, E)
            t.insert("end", f"\n晶格能近似 E_latt ≈ ½ Σ 近邻 E_int = {el:.1f} kJ/mol "
                            f"({len(st.dimer_res['dimers'])} 个近邻，截断 {st.cut.get()} Å)"
                            + (f"；缺少组 {miss}" if miss else "") + "\n")
            self.app.data["lattice_energy"] = el
        if self.method_label.startswith("UFF"):
            t.insert("end", "UFF 为经验力场，通常高估 π 堆积结合能，只宜作组间相对比较；发表请用 Gaussian 抵消校正结果。\n")
        self.app.data["interaction"] = E
        ax = self.plot.new_axes()
        ids = sorted(E)
        vals = [E[g]["total"] for g in ids]
        ax.bar([f"G{g}" for g in ids], vals, color=["#1E88E5" if v < 0 else "#E53935" for v in vals])
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_ylabel("E$_{int}$ / kJ mol$^{-1}$")
        ax.set_title("各二聚体组的相互作用能")
        if coup:
            ax2 = ax.twinx()
            ax2.plot([f"G{g}" for g in ids], [abs(coup.get(g, {}).get("hole") or np.nan) for g in ids], "o-",
                     color="#D98E04", label="|J$_h$|")
            ax2.plot([f"G{g}" for g in ids], [abs(coup.get(g, {}).get("electron") or np.nan) for g in ids], "s--",
                     color="#2E8B57", label="|J$_e$|")
            ax2.set_ylabel("|J| / meV")
            ax2.legend(fontsize=8, loc="upper right")
        self.plot.draw()
        if self.src.get() == self.SRC[0] and st.mols is not None:
            self.draw_framework()

    def draw_framework(self):
        from .interaction import framework_edges
        st = self.app.struct_tab
        if st.mols is None or not st.dimer_res:
            raise ValueError("能量框架需要晶体：请在“结构/晶体”页打开 CIF 并搜索近邻")
        if not self.energies:
            raise ValueError("请先计算或读取相互作用能")
        n = int(self.f_cell.get())
        crit = "atom" if st.crit.get().startswith("最短") else "centroid"
        edges, cents = framework_edges(st.crys, st.mols, self.energies, float(st.cut.get()), crit, (n, n, n))
        emin = float(self.f_min.get())
        emax = max([abs(e) for _, _, e in edges] + [1e-9])
        rmax = float(self.f_rmax.get())
        cyl = [(p, q, rmax * abs(e) / emax, "#1E88E5" if e < 0 else "#E53935") for p, q, e in edges if abs(e) >= emin]
        lat = np.asarray(st.crys["lattice"])
        syms, xyz = [], []
        if self.f_mol.get():
            for T_ in np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)]):
                for m in st.mols:
                    syms += m["symbols"]
                    xyz.append(np.asarray(m["cart"]) + T_ @ lat)
        r = self.view.renderer
        r.style, r.background = "stick", st.bg_var.get()
        r.cylinders = cyl
        r.title = T("能量框架") + f" ({len(cyl)})  " + T("蓝: 吸引  红: 排斥")
        r.set_scene(syms, np.vstack(xyz) if xyz else np.zeros((0, 3)),
                    colors=["#B0B7C3"] * len(syms), lattice=lat, cell_range=((0, n), (0, n), (0, n)), reset=True)
        r.view_along(lat[2] + 0.35 * lat[0] - 0.25 * lat[1], lat[1])
        r.draw()
        self.view.canvas.draw_idle()
        self.app.log(f"能量框架: {len(cyl)} 根圆柱 (|E| ≥ {emin} kJ/mol)，最粗对应 {emax:.1f} kJ/mol")

    def save_framework(self):
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                                                                            ("TIFF", "*.tif")])
        if p:
            self.view.renderer.save(p, dpi=300)
            self.app.log(f"已保存 {p}")

    def export_csv(self):
        if not self.energies:
            raise ValueError("请先计算或读取相互作用能")
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        import csv as _csv
        coup = self.app.data.get("crystal_couplings") or {}
        with open(p, "w", newline="", encoding="utf-8-sig") as fh:
            w = _csv.writer(fh)
            w.writerow(["group", "r_centroid_A", "E_int_kJmol", "vdW", "elec", "BSSE", "Jh_meV", "Je_meV", "method"])
            for gid in sorted(self.energies):
                e, c = self.energies[gid], coup.get(gid, {})
                w.writerow([gid, e.get("r"), e["total"], e.get("vdw"), e.get("elec"), e.get("BSSE"), c.get("hole"),
                            c.get("electron"), self.method_label])
        self.app.log(f"已导出 {p}")


# ==============================================================================================
# 耦合代理模型与堆积设计
# ==============================================================================================
class SurrogateTab(ttk.Frame):
    PLOTS = ["V(Δx,Δy) 耦合图", "一维堆积迁移率图", "可分离性 (两张网格逐点)", "验证 parity 图", "距离衰减 ln|V|–z"]
    KINDS = {"一维滑移堆积": "1d", "二维砖墙 (两个滑移近邻)": "2d"}

    def __init__(self, parent, app):
        from .surrogate import METHODS, OBJECTIVES
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.records, self.sources, self.model, self.valres, self.best = [], [], None, None, None
        self.monomer = None
        S = safe(app)
        panel = SectionPanel(self, width=460)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner

        a = ttk.LabelFrame(left, text="单体与扫描数据")
        a.pack(fill="x")
        self.mono = FileEntry(a, "单体结构", filetypes=STRUCT_TYPES, command=lambda p: self.load_monomer(p))
        self.mono.pack(fill="x")
        af = ttk.Frame(a)
        af.pack(fill="x")
        self.sym = field(af, 0, "对称性", "D2h (只扫第一象限)", values=["D2h (只扫第一象限)", "无"], width=20)
        ab = ttk.Frame(a)
        ab.pack(fill="x")
        for txt, cmd in (("导入扫描目录…", self.add_dir), ("导入 CSV…", self.add_csv), ("清空", self.clear)):
            ttk.Button(ab, text=txt, command=S(cmd)).pack(side="left", fill="x", expand=True)
        self.src_lbl = ttk.Label(a, text="尚无数据", foreground=PALETTE["muted"], wraplength=430)
        self.src_lbl.pack(anchor="w")
        ef = ttk.Frame(a)
        ef.pack(fill="x")
        self.e_z = field(ef, 0, "EHT 滑移网格 z / Å", "3.5 3.8", width=16)
        self.e_dx = field(ef, 1, "Δx 最大 / 步长", "6 0.25", width=16)
        self.e_dy = field(ef, 2, "Δy 最大 / 步长", "3 0.25", width=16)
        self.e_zs = field(ef, 3, "z 扫描 起 止 步", "3.3 4.2 0.1", width=16)
        ttk.Button(a, text="软件内 EHT 扫描 (秒级，用于预览与 M3)", command=S(self.eht_scan)).pack(fill="x")

        m = ttk.LabelFrame(left, text="代理模型")
        m.pack(fill="x", pady=3)
        mf = ttk.Frame(m)
        mf.pack(fill="x")
        self.method = MappedVar("M2", {k: v for k, v in METHODS.items()})
        ttk.Label(mf, text="模型").grid(row=0, column=0, sticky="w")
        ttk.Combobox(mf, textvariable=self.method, values=list(METHODS), state="readonly", width=34).grid(
            row=0, column=1, sticky="we")
        mb = ttk.Frame(m)
        mb.pack(fill="x")
        for txt, cmd in (("拟合模型", self.fit), ("可分离性检验", self.separability), ("保存模型…", self.save_model),
                         ("载入模型…", self.load_model)):
            ttk.Button(mb, text=txt, command=S(cmd)).pack(side="left", fill="x", expand=True)
        ttk.Button(m, text="导出全部数据 (CSV)…", command=S(self.export_csv)).pack(fill="x")

        v = ttk.LabelFrame(left, text="验证 (ADF 验证集)")
        v.pack(fill="x", pady=3)
        vf = ttk.Frame(v)
        vf.pack(fill="x")
        self.v_n = field(vf, 0, "构型数 / 随机种子", "40", width=8)
        self.v_seed = field(vf, 0, "", "2026", width=8, col=2)
        self.v_t21 = FileEntry(v, "单体片段 mono.t21", filetypes=[("TAPE21", "*.t21"), ("所有文件", "*.*")])
        self.v_t21.pack(fill="x")
        ttk.Label(v, text="请选扫描时所用的 mono.t21：共用同一套单体轨道，V 的符号才与扫描数据可比",
                  foreground=PALETTE["muted"], wraplength=430).pack(anchor="w")
        vb = ttk.Frame(v)
        vb.pack(fill="x")
        ttk.Button(vb, text="生成 ADF 验证集…", command=S(self.gen_valid)).pack(side="left", fill="x", expand=True)
        ttk.Button(vb, text="读取验证结果并评估…", command=S(self.eval_valid)).pack(side="left", fill="x", expand=True)

        d = ttk.LabelFrame(left, text="堆积反向设计")
        d.pack(fill="x", pady=3)
        df = ttk.Frame(d)
        df.pack(fill="x")
        self.kind = field(df, 0, "堆积类型", "一维滑移堆积", values=list(self.KINDS), width=22)
        self.obj = MappedVar("max_mu", OBJECTIVES)
        ttk.Label(df, text="优化目标").grid(row=1, column=0, sticky="w")
        ttk.Combobox(df, textvariable=self.obj, values=list(OBJECTIVES), state="readonly", width=30).grid(
            row=1, column=1, sticky="we")
        self.car = field(df, 2, "载流子", "空穴", values=["空穴", "电子"], width=22)
        self.lam_h = field(df, 3, "空穴重组能 λh / eV", "0.095", width=22)
        self.lam_e = field(df, 4, "电子重组能 λe / eV", "0.13", width=22)
        self.temp = field(df, 5, "温度 / K", "300", width=22)
        self.zr = field(df, 6, "z 范围 / Å (空=数据范围)", "", width=22)
        self.clash = field(df, 7, "碰撞阈值 d/(rA+rB)", "0.75", width=22)
        self.aniso = field(df, 8, "各向异性上限 / 迁移率下限", "", width=22)
        db = ttk.Frame(d)
        db.pack(fill="x")
        ttk.Button(db, text="优化堆积", style="Accent.TButton", command=S(self.optimize)).pack(side="left", fill="x",
                                                                                      expand=True)
        ttk.Button(db, text="导出最优构型 (xyz + ADF)…", command=S(self.export_best)).pack(side="left", fill="x",
                                                                                     expand=True)
        self.show_crys = tk.BooleanVar(value=True)
        ttk.Checkbutton(d, text="在图中标出“结构/晶体”页晶体的共面二聚体", variable=self.show_crys).pack(anchor="w")

        p = ttk.LabelFrame(left, text="作图")
        p.pack(fill="x", pady=3)
        pf = ttk.Frame(p)
        pf.pack(fill="x")
        self.plot_kind = field(pf, 0, "图", self.PLOTS[0], values=self.PLOTS, width=22)
        self.plot_z = field(pf, 1, "作图 z / Å", "3.5", width=22)
        pb = ttk.Frame(p)
        pb.pack(fill="x")
        ttk.Button(pb, text="作图", command=S(self.draw)).pack(side="left", fill="x", expand=True)
        ttk.Button(pb, text="导出图片 (300 dpi)…", command=S(self.save_fig)).pack(side="left", fill="x", expand=True)

        right = ttk.Frame(self, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.plot = PlotPanel(right, figsize=(7, 5))
        self.plot.pack(fill="both", expand=True)
        self.text = styled_text(right, app.mono, height=12, wrap="none", size=9)
        self.text.pack(fill="x", pady=(8, 0))

    # ---------------- 数据 ----------------
    def _carrier(self):
        return "hole" if self.car.get() == "空穴" else "electron"

    def _symm(self):
        return "D2h" if self.sym.get().startswith("D2h") else "none"

    def load_monomer(self, p=None):
        from .surrogate import molecular_frame
        p = p or self.mono.get()
        at = read_geometry(p)
        sym = [a[0] for a in at]
        Y, plan, _, _ = molecular_frame(sym, [a[1:] for a in at])
        self.monomer = (sym, Y)
        hv = Y[[e != "H" for e in sym]]
        self.app.log(f"单体 {Path(p).name}: {len(sym)} 个原子，长轴 {np.ptp(hv[:, 0]):.2f} Å、短轴 {np.ptp(hv[:, 1]):.2f} Å，"
                     f"重原子最大偏离平面 {plan:.3f} Å" + ("  [警告] 偏离平面 > 0.1 Å，z 的定义不够严格" if plan > 0.1 else ""))
        return self.monomer

    def _need_monomer(self):
        if self.monomer is None:
            if not self.mono.get():
                raise ValueError("请先选择单体结构 (与扫描所用的同一个，例如 pentacene.log)")
            self.load_monomer()
        return self.monomer

    def _update_src(self):
        n = len(self.records)
        self.src_lbl.config(text=(T("数据") + f": {n} " + T("个构型") + " · " + "; ".join(self.sources)) if n else T("尚无数据"))

    def add_dir(self, d=None):
        from .surrogate import load_scan_dir
        d = d or filedialog.askdirectory(title=T("选择扫描目录 (含 manifest.csv 与 outputs/)"),
                                         initialdir=self.app.config_data.get("workdir") or None)
        if not d:
            return None
        recs, missing = load_scan_dir(d)
        self.records += recs
        self.sources.append(f"{Path(d).name} ({len(recs)})")
        self._update_src()
        self.app.log(f"导入 {Path(d).name}: {len(recs)} 个构型" + (f"，{missing} 个没有结果 (未计算或失败)" if missing else ""))
        return recs

    def add_csv(self, p=None):
        from .surrogate import load_csv
        p = p or filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not p:
            return None
        recs = load_csv(p)
        self.records += recs
        self.sources.append(f"{Path(p).name} ({len(recs)})")
        self._update_src()
        return recs

    def clear(self):
        self.records, self.sources, self.model = [], [], None
        self._update_src()

    def eht_scan(self):
        from .surrogate import eht_scan
        sym, Y = self._need_monomer()
        zs = [float(x) for x in self.e_z.get().split()]
        dxm, dxs = map(float, self.e_dx.get().split())
        dym, dys = map(float, self.e_dy.get().split())
        dxv = np.round(np.arange(0, dxm + 1e-9, dxs), 4)
        dyv = np.round(np.arange(0, dym + 1e-9, dys), 4)
        if self._symm() != "D2h":
            dxv = np.round(np.arange(-dxm, dxm + 1e-9, dxs), 4)
            dyv = np.round(np.arange(-dym, dym + 1e-9, dys), 4)
        zline = []
        if self.e_zs.get().strip():
            z0, z1, st = map(float, self.e_zs.get().split())
            zline = np.round(np.arange(z0, z1 + 1e-9, st), 4)

        def work():
            r = eht_scan(sym, Y, zs, dxv, dyv)
            if len(zline):
                r += eht_scan(sym, Y, zline, [0.0], [0.0])
            return r

        def done(recs):
            self.records += recs
            self.sources.append(f"EHT ({len(recs)})")
            self._update_src()
            self.fit()
        self.app.run_bg("软件内 EHT 扫描", work, done)

    # ---------------- 模型 ----------------
    def fit(self):
        from .surrogate import CouplingModel
        if not self.records:
            raise ValueError("请先导入扫描数据或进行 EHT 扫描")
        mono = self.monomer
        if mono is None and self.mono.get():
            mono = self.load_monomer()
        self.model = CouplingModel(self.records, self._symm(), mono).fit()
        m = self.model
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"代理模型：{len(self.records)} 个构型，滑移网格 z = {', '.join(f'{z:.2f}' for z in sorted(m.grids))} Å，"
                        f"z_ref = {m.z_ref:.2f} Å，对称性 {m.symmetry}\n")
        for c in ("hole", "electron"):
            b = m.beta[c]
            t.insert("end", f"  β({'空穴' if c == 'hole' else '电子'}) = {b['value']:.3f} Å⁻¹ ({b['source']})")
            if c in m.calib:
                cal = m.calib[c]
                t.insert("end", f"；M3 标定 a = {cal['a']:.3f}, δβ = {cal['dbeta']:.3f} Å⁻¹, R² = {cal['R2']:.4f}")
            t.insert("end", "\n")
        rg = m.data_range()
        t.insert("end", f"  数据范围: z {rg['z'][0]:.2f}–{rg['z'][1]:.2f} Å, Δx {rg['dx'][0]:.2f}–{rg['dx'][1]:.2f} Å, "
                        f"Δy {rg['dy'][0]:.2f}–{rg['dy'][1]:.2f} Å\n")
        if not m.monomer:
            t.insert("end", "  [提示] 未指定单体结构：无法判断碰撞，也不能使用 M3\n")
        self.app.data["coupling_model"] = m
        self.plot_z.set(f"{m.z_ref:.2f}")
        self.draw()
        return m

    def _model(self):
        if self.model is None:
            self.fit()
        return self.model

    def separability(self):
        m = self._model()
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", "可分离性检验 V(z,Δx,Δy) = A·e^{−β(z−z_ref)}·F(Δx,Δy)：相邻两张滑移网格逐点比较\n")
        for c in ("hole", "electron"):
            for s in m.separability(c):
                t.insert("end", f"  {'空穴' if c == 'hole' else '电子'} {s['z1']:.2f}→{s['z2']:.2f} Å: 比例因子 k = {s['k']:.4f}，"
                                f"等效 β = {s['beta_eff']:.3f} Å⁻¹，r = {s['r']:.5f}，相对残差 {s['rel_resid']:.3f}，"
                                f"|V| 大区比值离散 {s['ratio_cv']:.3f}，变号点 {s['sign_flips']}/{s['n']}\n")
        if not any(m.separability(c) for c in ("hole", "electron")):
            t.insert("end", "  需要至少两张不同 z 的滑移网格\n")
        t.insert("end", "判据：r > 0.99、比值离散小、几乎无变号点 → 滑移图形状不随距离改变，可写成距离项 × 滑移项；"
                        "否则应使用 M2 (局部衰减，含距离–滑移耦合)\n")
        self.plot_kind.set(self.PLOTS[2])
        self.draw()

    def save_model(self):
        m = self._model()
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("模型", "*.json")])
        if p:
            m.to_json(p)
            self.app.log(f"模型已保存: {p} (含网格、β、标定、单体与全部数据，可复现)")

    def load_model(self, p=None):
        from .surrogate import CouplingModel
        p = p or filedialog.askopenfilename(filetypes=[("模型", "*.json")])
        if not p:
            return None
        self.model = CouplingModel.from_json(p)
        self.records = self.model.records
        self.monomer = self.model.monomer
        self.sources = [Path(p).name]
        self._update_src()
        self.app.log(f"已载入模型 {Path(p).name}")
        self.draw()
        return self.model

    def export_csv(self):
        from .surrogate import save_csv
        if not self.records:
            raise ValueError("没有数据")
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            save_csv(self.records, p)
            self.app.log(f"已导出 {len(self.records)} 个构型: {p}")

    # ---------------- 验证 ----------------
    def gen_valid(self, d=None):
        from .surrogate import lhs_points, write_adf_scan
        sym, Y = self._need_monomer()
        d = d or filedialog.askdirectory(title=T("选择验证集输出目录"))
        if not d:
            return None
        m = self.model
        rg = m.data_range() if m else {"z": (3.5, 3.8), "dx": (-6, 6), "dy": (-3, 3)}
        z0, z1 = rg["z"] if rg["z"][1] > rg["z"][0] else (rg["z"][0], rg["z"][0] + 0.3)
        pts = lhs_points(int(self.v_n.get()), (z0, z1), ((z0 - 0.2, z0), (z1, z1 + 0.4)), rg["dx"], rg["dy"],
                         int(self.v_seed.get()))
        r = write_adf_scan(d, sym, Y, pts, mono_t21=self.v_t21.get() or None, prefix="valid")
        self.app.log(f"验证集: {r['n']} 个构型 (碰撞跳过 {r['clash']})，内插 z∈[{z0:.2f},{z1:.2f}] Å，其余为外推；"
                     f"在 ADFjobs 中运行 {Path(r['batch']).name}" + ("；需先运行 mono.run 生成 mono.t21" if r["need_mono"] else ""))
        return r

    def eval_valid(self, d=None):
        from .surrogate import load_scan_dir
        m = self._model()
        d = d or filedialog.askdirectory(title=T("选择验证集目录"))
        if not d:
            return None
        recs, missing = load_scan_dir(d)
        if not recs:
            raise ValueError("验证集目录中还没有结果")
        self.valres = m.validate(recs)
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"验证集 {Path(d).name}: {len(recs)} 个构型" + (f" ({missing} 个缺结果)" if missing else "") + "，单位 meV\n")
        t.insert("end", f"{'模型':<4}{'载流子':<6}{'组':<5}{'n':>4}{'MAE':>9}{'RMSE':>9}{'最大':>9}{'R²':>9}{'符号一致':>9}\n")
        for (meth, c, grp), r in sorted(self.valres.items()):
            t.insert("end", f"{meth:<4}{'空穴' if c == 'hole' else '电子':<6}{grp:<5}{r['n']:>4}{r['MAE']:9.2f}{r['RMSE']:9.2f}"
                            f"{r['max']:9.2f}{r['R2']:9.4f}{r['sign'] * 100:8.0f}%\n")
        self.plot_kind.set(self.PLOTS[3])
        self.draw()
        return self.valres

    # ---------------- 设计 ----------------
    def optimize(self):
        from .surrogate import OBJECTIVES, optimize_packing
        m = self._model()
        kind = self.KINDS[self.kind.get()]
        obj = self.obj.get()
        if kind == "1d" and obj in ("max_avg", "min_aniso"):
            raise ValueError("该目标只适用于二维砖墙堆积")
        zr = tuple(map(float, self.zr.get().split())) if self.zr.get().strip() else None
        extra = self.aniso.get().strip()
        kw = dict(kind=kind, objective=obj, carrier=self._carrier(), lam=float(self.lam_h.get()),
                  lam_e=float(self.lam_e.get()), T=float(self.temp.get()), method=self.method.get(), z_range=zr,
                  clash_min=float(self.clash.get()))
        if extra and obj == "max_avg":
            kw["aniso_max"] = float(extra)
        if extra and obj == "min_aniso":
            kw["mu_floor"] = float(extra)

        def done(o):
            self.best = o
            t = self.text
            t.delete("1.0", "end")
            p = o["params"]
            t.insert("end", f"堆积优化：{self.kind.get()}，目标 = {OBJECTIVES[obj]}，模型 {o['method']}，"
                            f"{o['evaluations']} 次评估\n")
            if kind == "1d":
                t.insert("end", f"  最优滑移：Δx = {p[0]:.3f} Å，Δy = {p[1]:.3f} Å，z = {p[2]:.3f} Å\n")
            else:
                t.insert("end", f"  近邻 1：Δx = {p[0]:.3f}, Δy = {p[1]:.3f} Å；近邻 2：Δx = {p[2]:.3f}, Δy = {p[3]:.3f} Å；"
                                f"z = {p[4]:.3f} Å\n")
            for c in ("hole", "electron"):
                if c in o:
                    r = o[c]
                    t.insert("end", f"  {'空穴' if c == 'hole' else '电子'}: V = " + ", ".join(f"{v:.1f}" for v in r["V_meV"]) +
                                    f" meV，μ_max = {r['mu_max']:.3f} cm²V⁻¹s⁻¹" +
                                    (f"，μ_avg = {r['mu_avg']:.3f}，各向异性 {r['anisotropy']:.2f}" if kind == "2d" else "") + "\n")
            if o["clash_penalty"] > 0:
                t.insert("end", "  [警告] 最优解仍有原子碰撞，请放宽范围或检查单体\n")
            t.insert("end", "说明：设计结果基于代理模型，建议导出最优构型用 ADF 复核 (按钮“导出最优构型”)。\n")
            if kind == "1d":
                self.plot_z.set(f"{p[2]:.2f}")
            self.draw()
        self.app.run_bg("堆积优化", lambda: optimize_packing(m, **kw), done)

    def export_best(self, d=None):
        from .surrogate import write_adf_scan
        if not self.best:
            raise ValueError("请先优化堆积")
        sym, Y = self._need_monomer()
        d = d or filedialog.askdirectory(title=T("选择输出目录"))
        if not d:
            return None
        p = self.best["params"]
        cfgs = [("opt", p[2], p[0], p[1])] if self.best["kind"] == "1d" else \
            [("opt1", p[4], p[0], p[1]), ("opt2", p[4], p[2], p[3])]
        r = write_adf_scan(d, sym, Y, cfgs, mono_t21=self.v_t21.get() or None, prefix="opt")
        self.app.log(f"最优构型已导出到 {d}: xyz/ 与 batch_opt.run ({r['n']} 个二聚体)，可在 ADFjobs 中复核 V")
        return r

    # ---------------- 作图 ----------------
    def draw(self):
        if self.model is None:
            return
        m = self.model
        kind = self.plot_kind.get()
        fig = self.plot.fig
        fig.clf()
        ax = fig.add_subplot(111)
        car = self._carrier()
        lab = "空穴" if car == "hole" else "电子"
        if kind in self.PLOTS[:2]:
            from .surrogate import crystal_slips, landscape
            z = float(self.plot_z.get())
            q = "V" if kind == self.PLOTS[0] else "mu"
            lam = float(self.lam_h.get() if car == "hole" else self.lam_e.get())
            xs, ys, Z = landscape(m, z, car, self.method.get() if self.method.get() != "M3" else "M2", n=121, quantity=q,
                                  lam=lam, T=float(self.temp.get()), clash_min=float(self.clash.get()))
            import matplotlib
            cmap = matplotlib.colormaps["RdBu_r" if q == "V" else "viridis"].copy()
            cmap.set_bad("#BDBDBD")
            if q == "V":
                vm = np.nanmax(np.abs(Z))
                im = ax.pcolormesh(xs, ys, Z.T, cmap=cmap, vmin=-vm, vmax=vm, shading="auto")
                ax.contour(xs, ys, np.nan_to_num(Z.T), levels=[0], colors="k", linewidths=0.8)
                fig.colorbar(im, ax=ax, label=f"V$_{{{'h' if car == 'hole' else 'e'}}}$ / meV")
                ax.set_title(f"{lab}耦合 V(Δx, Δy)，z = {z:.2f} Å (黑线：V = 0 节线；灰：碰撞)")
            else:
                im = ax.pcolormesh(xs, ys, Z.T, cmap=cmap, shading="auto")
                fig.colorbar(im, ax=ax, label="μ$_{stack}$ / cm$^2$V$^{-1}$s$^{-1}$")
                ax.set_title(f"一维滑移堆积{lab}迁移率，z = {z:.2f} Å，λ = {lam:.3f} eV")
            if self.best and self.best["kind"] == "1d":
                p = self.best["params"]
                ax.plot(p[0], p[1], marker="*", ms=16, color="#FFD54F", mec="k", label=f"最优 (z={p[2]:.2f} Å)")
            elif self.best:
                p = self.best["params"]
                ax.plot([p[0], p[2]], [p[1], p[3]], "*", ms=14, color="#FFD54F", mec="k", label="最优 (近邻 1、2)")
            st = self.app.struct_tab
            if self.show_crys.get() and st.dimer_res:
                for s in crystal_slips(st.dimer_res):
                    if s["cofacial"]:
                        ax.plot(s["dx"], s["dy"], "o", ms=8, mfc="none", mec="#2E8B57", mew=2)
                        ax.annotate(f"G{s['group']} (z={s['z']:.2f})", (s["dx"], s["dy"]), fontsize=8, color="#1B5E20",
                                    xytext=(4, 4), textcoords="offset points")
            ax.set_xlabel("Δx (长轴) / Å")
            ax.set_ylabel("Δy (短轴) / Å")
            ax.set_aspect("equal")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, loc="upper right")
        elif kind == self.PLOTS[2]:
            seps = m.separability(car)
            if not seps:
                ax.text(0.5, 0.5, "需要两张不同 z 的滑移网格", ha="center", transform=ax.transAxes)
            for s in seps:
                gx, gy, G1 = m.grids[s["z1"]][car]
                G2 = m.splines[(s["z2"], car)](gx, gy)
                ax.scatter(G1.ravel(), G2.ravel(), s=8, alpha=0.6, label=f"{s['z1']:.2f}→{s['z2']:.2f} Å, k={s['k']:.3f}, r={s['r']:.4f}")
                lim = np.abs(G1).max()
                ax.plot([-lim, lim], [-lim * s["k"], lim * s["k"]], "k--", lw=1)
            ax.axhline(0, color="#999", lw=0.6)
            ax.axvline(0, color="#999", lw=0.6)
            ax.set_xlabel("V(z$_1$) / meV")
            ax.set_ylabel("V(z$_2$) / meV")
            ax.set_title(f"可分离性检验 ({lab})")
            if seps:
                ax.legend(fontsize=8)
        elif kind == self.PLOTS[3]:
            if not self.valres:
                ax.text(0.5, 0.5, "请先读取验证结果", ha="center", transform=ax.transAxes)
            else:
                lim = 0
                for (meth, c, grp), r in sorted(self.valres.items()):
                    if c != car or grp == "全部":
                        continue
                    t_, p_ = r["pairs"]
                    lim = max(lim, np.abs(t_).max(), np.abs(p_).max())
                    ax.scatter(t_, p_, s=18, marker="o" if grp == "内插" else "^", alpha=0.75,
                               label=f"{meth} {grp} (MAE {r['MAE']:.2f} meV)")
                ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
                ax.set_xlabel("V$_{DFT}$ / meV")
                ax.set_ylabel("V$_{模型}$ / meV")
                ax.set_title(f"验证集 parity 图 ({lab})")
                ax.legend(fontsize=7)
        else:
            line = sorted((r["z"], r[car]) for r in m.records if abs(r["dx"]) < 1e-6 and abs(r["dy"]) < 1e-6 and r[car])
            if line:
                z, vv = np.array(line).T
                ax.semilogy(z, np.abs(vv), "o", label="数据 (Δx = Δy = 0)")
                zz = np.linspace(z.min(), z.max(), 100)
                b = m.beta[car]["value"]
                v0 = float(np.abs(m.predict(m.z_ref, 0, 0, car, "M1")))
                ax.semilogy(zz, v0 * np.exp(-b * (zz - m.z_ref)), "-", label=f"β = {b:.3f} Å$^{{-1}}$")
                ax.legend()
            ax.set_xlabel("z / Å")
            ax.set_ylabel(f"|V$_{{{'h' if car == 'hole' else 'e'}}}$| / meV")
            ax.set_title("距离衰减")
        fig.tight_layout()
        self.plot.draw()

    def save_fig(self):
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("TIFF", "*.tif")])
        if p:
            self.plot.fig.savefig(p, dpi=300, bbox_inches="tight")
            self.app.log(f"已保存 {p}")


# ==============================================================================================
# 元素周期表选择窗口
# ==============================================================================================
def _ptable_positions():
    """标准长式周期表：{元素: (行, 列)}，镧系 / 锕系在第 8、9 行 (行列从 0 开始)。"""
    from .substitution import ELEMENTS
    pos, z = {}, 0
    layout = [(1, [1, 18]), (2, [1, 2] + list(range(13, 19))), (3, [1, 2] + list(range(13, 19))),
              (4, list(range(1, 19))), (5, list(range(1, 19))), (6, [1, 2, "L"] + list(range(4, 19))),
              (7, [1, 2, "A"] + list(range(4, 19)))]
    for period, cols in layout:
        for col in cols:
            if col in ("L", "A"):
                row = 8 if col == "L" else 9
                for k in range(15):
                    pos[ELEMENTS[z]] = (row, 2 + k)
                    z += 1
            else:
                pos[ELEMENTS[z]] = (period - 1, col - 1)
                z += 1
    return pos


class PeriodicTableDialog(tk.Toplevel):
    """点击元素后调用 callback(元素符号)。"""

    def __init__(self, master, callback, title="选择取代元素"):
        super().__init__(master)
        from matplotlib.colors import to_rgb
        from .eht import EHT_PARAMS
        from .render import JMOL
        from .substitution import COV, Z
        self.title(T(title))
        self.callback = callback
        try:
            self.transient(master)
            App.set_icon(self)
        except Exception:  # noqa: BLE001
            pass
        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        grid = ttk.Frame(body)
        grid.pack()
        self.info = ttk.Label(body, text=T("点击元素进行取代；粗体 = 软件内 EHT 有参数"), foreground=PALETTE["muted"])
        self.info.pack(anchor="w", pady=(8, 0))
        self.buttons = {}
        for el, (r, c) in _ptable_positions().items():
            rgb = np.array(to_rgb(JMOL.get(el, "#C8C8C8")))
            bg = "#%02x%02x%02x" % tuple(int(255 * v) for v in (rgb * 0.45 + 0.55))
            bold = el in EHT_PARAMS
            b = tk.Button(grid, text=el, width=3, relief="groove", bg=bg, activebackground="#FFE38A",
                          font=("", sz(10), "bold" if bold else "normal"), command=lambda e=el: self.pick(e))
            b.grid(row=r + (1 if r >= 8 else 0), column=c, padx=1, pady=1, sticky="nsew")
            b.bind("<Enter>", lambda ev, e=el: self.info.config(
                text=f"{Z[e]}  {e}    {T('共价半径')} {COV[e]:.2f} Å    EHT: {T('有') if e in EHT_PARAMS else T('无')}"))
            self.buttons[el] = b
        ttk.Label(grid, text="").grid(row=8, column=0)          # 镧系 / 锕系前空一行
        for r, lab in ((9, "La–Lu"), (10, "Ac–Lr")):
            ttk.Label(grid, text=lab, foreground=PALETTE["muted"]).grid(row=r, column=0, columnspan=2, sticky="e")

    def pick(self, el):
        self.callback(el)
        self.destroy()


# ==============================================================================================
# 波函数分析：轨道浏览器 / 电荷与 Mayer 键级 / ESP 表面 / NBO
# ==============================================================================================
class WavefunctionTab(ttk.Frame):
    ESP_SRC = ["Gaussian cubegen (严格)", "已有 cube 文件 (密度 + ESP)", "原子电荷近似 (定性)"]
    IG_SRC = ["当前 fchk (实际密度)", "结构页选中的二聚体 (前分子密度)", "结构文件 (前分子密度)"]
    GRID = {"粗 (0.35 Å)": 0.35, "中 (0.25 Å)": 0.25, "细 (0.18 Å)": 0.18}

    def __init__(self, parent, app):
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.wfn, self.cube, self.esp, self.charges, self.nbo = None, None, None, None, None
        S = safe(app)
        panel = SectionPanel(self, width=460)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner
        f = ttk.LabelFrame(left, text="Gaussian 波函数 (fchk)")
        f.pack(fill="x")
        self.fchk = FileEntry(f, "fchk 文件", filetypes=[("fchk", "*.fchk *.fch"), ("所有文件", "*.*")],
                              command=lambda p: S(self.load)())
        self.fchk.pack(fill="x")
        self.info = ttk.Label(f, text="", foreground=PALETTE["muted"], wraplength=430)
        self.info.pack(anchor="w")

        o = ttk.LabelFrame(left, text="轨道浏览器")
        o.pack(fill="x", pady=3)
        cols = ("#", "轨道", "能量/eV", "占据")
        self.orb = ttk.Treeview(o, columns=cols, show="headings", height=8, selectmode="browse")
        for c_, w_ in zip(cols, (50, 90, 90, 60)):
            self.orb.heading(c_, text=T(c_))
            self.orb.column(c_, width=w_, anchor="center")
        self.orb.pack(fill="x")
        self.orb.bind("<Double-1>", lambda e: S(self.show_orbital)())
        of = ttk.Frame(o)
        of.pack(fill="x")
        self.spin = field(of, 0, "自旋", "alpha", values=["alpha", "beta"], width=10)
        self.iso = field(of, 0, "等值", "0.02", width=8, col=2)
        self.grid_q = field(of, 1, "格点", "中 (0.25 Å)", values=list(self.GRID), width=10)
        ob = ttk.Frame(o)
        ob.pack(fill="x")
        for txt, cmd in (("显示选中轨道", self.show_orbital), ("HOMO", lambda: self.show_orbital("HOMO")),
                         ("LUMO", lambda: self.show_orbital("LUMO")), ("导出 cube…", self.export_cube)):
            ttk.Button(ob, text=txt, command=S(cmd)).pack(side="left", fill="x", expand=True)
        ttk.Label(o, text="双击轨道显示等值面 (蓝 +，红 −)；cube 可用 VMD / Multiwfn 打开",
                  foreground=PALETTE["muted"]).pack(anchor="w")

        c = ttk.LabelFrame(left, text="电荷与 Mayer 键级")
        c.pack(fill="x", pady=3)
        ttk.Button(c, text="由 fchk 计算 Mulliken 电荷 + Mayer 键级", command=S(self.mulliken)).pack(fill="x")
        self.chg_log = FileEntry(c, "或读取 log", filetypes=[("Gaussian 输出", "*.log *.out"), ("所有文件", "*.*")])
        self.chg_log.pack(fill="x")
        ttk.Button(c, text="读取 log 中的电荷 (ESP / NPA / Mulliken)", command=S(self.read_log_charges)).pack(fill="x")
        self.bo_min = field(ttk.Frame(c), 0, "标注键级 ≥", "0.5", width=8)
        self.bo_min_w = self.bo_min

        e = ttk.LabelFrame(left, text="分子表面静电势 (ESP)")
        e.pack(fill="x", pady=3)
        ef = ttk.Frame(e)
        ef.pack(fill="x")
        self.esp_src = field(ef, 0, "ESP 来源", self.ESP_SRC[0], values=self.ESP_SRC, width=24)
        self.esp_iso = field(ef, 1, "密度等值 / a.u.", "0.001", width=24)
        self.dens_cube = FileEntry(e, "密度 cube", filetypes=[("cube", "*.cub *.cube"), ("所有文件", "*.*")])
        self.esp_cube = FileEntry(e, "ESP cube", filetypes=[("cube", "*.cub *.cube"), ("所有文件", "*.*")])
        self.dens_cube.pack(fill="x")
        self.esp_cube.pack(fill="x")
        eb = ttk.Frame(e)
        eb.pack(fill="x")
        ttk.Button(eb, text="计算 ESP 表面", style="Accent.TButton", command=S(self.run_esp)).pack(side="left", fill="x",
                                                                                              expand=True)
        ttk.Button(eb, text="导出表面数据…", command=S(self.export_esp)).pack(side="left", fill="x", expand=True)
        ttk.Label(e, text="cubegen 随 Gaussian 安装 (与 g16.exe 同目录)，计算严格的静电势；原子电荷近似只适合定性比较",
                  foreground=PALETTE["muted"], wraplength=430).pack(anchor="w")

        n = ttk.LabelFrame(left, text="NBO 分析 (Gaussian pop=nbo 输出)")
        n.pack(fill="x", pady=3)
        self.nbo_log = FileEntry(n, "NBO 输出", filetypes=[("Gaussian 输出", "*.log *.out"), ("所有文件", "*.*")])
        self.nbo_log.pack(fill="x")
        nf = ttk.Frame(n)
        nf.pack(fill="x")
        self.e2_min = field(nf, 0, "E(2) ≥ / kcal·mol⁻¹", "1.0", width=10)
        self.nbo_na = field(nf, 1, "单体 A 原子数 (二聚体，可空)", "", width=10)
        ttk.Button(n, text="读取 NBO", command=S(self.read_nbo)).pack(fill="x")

        ig = ttk.LabelFrame(left, text="IGM / IGMH 弱相互作用 (π–π 堆积、氢键等)")
        ig.pack(fill="x", pady=3)
        gf = ttk.Frame(ig)
        gf.pack(fill="x")
        self.ig_src = field(gf, 0, "密度来源", self.IG_SRC[0], values=self.IG_SRC, width=26)
        self.ig_frag = field(gf, 1, "片段", "自动 (按成键连通性)", values=["自动 (按成键连通性)", "前 N 个原子为片段 A"], width=26)
        self.ig_n = field(gf, 2, "N (片段 A 原子数)", "", width=26)
        self.ig_iso = field(gf, 3, "δg_inter 等值 / a.u.", "0.005", width=26)
        self.ig_h = field(gf, 4, "格点间距 / Å", "0.15", values=["0.10", "0.15", "0.20", "0.25"], width=26)
        self.ig_sign = field(gf, 5, "IGMH 形式", "原文实现 (Lu & Chen 2022)", values=["原文实现 (Lu & Chen 2022)",
                                                                              "数学严格形式 (勘误)"], width=26)
        self.struct_file = FileEntry(ig, "结构文件", filetypes=STRUCT_TYPES)
        self.struct_file.pack(fill="x")
        gb = ttk.Frame(ig)
        gb.pack(fill="x")
        for txt, cmd in (("计算 δg_inter 等值面", self.run_igm), ("散点图", self.igm_scatter), ("导出 cube…", self.export_igm)):
            ttk.Button(gb, text=txt, command=S(cmd)).pack(side="left", fill="x", expand=True)
        ttk.Label(ig, text="蓝：强吸引 (氢键等)；绿：范德华 / π–π；红：位阻排斥。IGM 只需结构 (秒级，可直接分析晶体二聚体)，"
                           "IGMH 用 fchk 的实际密度。自由原子密度由本程序的 LDA 原子计算得到",
                  foreground=PALETTE["muted"], wraplength=430).pack(anchor="w")

        am = ttk.LabelFrame(left, text="AIM 拓扑 (临界点与键径)")
        am.pack(fill="x", pady=3)
        af_ = ttk.Frame(am)
        af_.pack(fill="x")
        self.aim_src = field(af_, 0, "密度来源", self.IG_SRC[0], values=self.IG_SRC, width=26)
        ab_ = ttk.Frame(am)
        ab_.pack(fill="x")
        ttk.Button(ab_, text="搜索临界点", style="Accent.TButton", command=S(self.run_aim)).pack(side="left", fill="x", expand=True)
        ttk.Button(ab_, text="导出 CSV…", command=S(self.export_aim)).pack(side="left", fill="x", expand=True)
        ttk.Label(am, text="键临界点 (橙)、环临界点 (绿)、笼临界点 (紫)；分子间键径加粗。按 Poincaré–Hopf 关系检查临界点是否找全；"
                           "能量 E ≈ V/2 (Espinosa) 原为氢键提出，用于 π–π 时仅供参考",
                  foreground=PALETTE["muted"], wraplength=430).pack(anchor="w")
        self.igm, self.aim, self.cur = None, None, None

        right = ttk.Frame(self, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.view = ViewerPanel(right, figsize=(6.2, 5.0))
        self.view.pack(fill="both", expand=True)
        rb = ttk.Frame(right, style="Bg.TFrame")
        rb.pack(fill="x")
        ttk.Button(rb, text="保存图片…", command=S(self.save_image)).pack(side="right")
        self.text = styled_text(right, app.mono, height=11, wrap="none", size=9)
        self.text.pack(fill="x", pady=(6, 0))
        self.view.canvas.mpl_connect("scroll_event", self._scroll)
        self.view.canvas.mpl_connect("button_press_event", self._press)
        self.view.canvas.mpl_connect("motion_notify_event", self._motion)
        self.view.canvas.mpl_connect("button_release_event", self._release)
        self._drag, self._cbar = None, None

    # ---- 视图 ----
    def _scroll(self, e):
        self.view.zoom(1.15 if e.button == "up" else 1 / 1.15)

    def _press(self, e):
        self._drag = (e.x, e.y) if e.x is not None else None

    def _motion(self, e):
        if not self._drag or e.x is None:
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self._drag = (e.x, e.y)
        self.view.renderer.rotate(dx, dy)
        self._draw(fast=True)

    def _release(self, e):
        if self._drag:
            self._drag = None
            self._draw()

    def _draw(self, fast=False):
        self.view.renderer.draw(fast=fast)
        self.view.canvas.draw_idle()

    def _scene(self, reset=False, colors=None, title=""):
        sym, X = self.cur if self.cur is not None else (self.wfn.symbols, self.wfn.coords * BOHR2ANG)
        r = self.view.renderer
        r.background = "白色"
        r.title = title
        r.set_scene(sym, X, colors=colors, reset=reset)
        if reset:
            nrm = np.linalg.svd(X - X.mean(0))[2][2]
            r.view_along(nrm + 0.6 * np.cross(nrm, [0, 0, 1.0]) if abs(nrm[2]) < 0.95 else nrm)
        self._colorbar(None)
        self._draw()

    def _colorbar(self, spec):
        fig = self.view.fig
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:  # noqa: BLE001
                pass
            self._cbar = None
        if spec:
            from matplotlib import cm
            from matplotlib.colors import Normalize
            cax = fig.add_axes([0.9, 0.18, 0.025, 0.62])
            cmap = spec[2]
            if cmap == "bgr":
                from matplotlib.colors import LinearSegmentedColormap
                cmap = LinearSegmentedColormap.from_list("bgr", ["#0000FF", "#00FF00", "#FF0000"])
            sm = cm.ScalarMappable(norm=Normalize(spec[0], spec[1]), cmap=cmap)
            cb = fig.colorbar(sm, cax=cax)
            cb.set_label(spec[3], fontsize=8)
            cb.ax.tick_params(labelsize=7)
            self._cbar = cax

    # ---- 读取 ----
    def load(self):
        from .wavefunction import Wavefunction
        p = self.fchk.get()
        if not p:
            raise ValueError("请选择 fchk 文件")
        w = Wavefunction(p)
        self.wfn = w
        self.cur = None
        self.fill_orbitals()
        homo = w.ea[w.nalpha - 1] * HARTREE2EV
        lumo = w.ea[w.nalpha] * HARTREE2EV if w.nalpha < len(w.ea) else float("nan")
        self.info.config(text=f"{len(w.Z)} {T('原子')}，{w.nbf} {T('个基函数')}，{T('电荷')} {w.charge}，"
                              f"{T('多重度')} {w.mult}{'，' + T('开壳层') if w.open_shell else ''}\n"
                              f"HOMO {homo:.3f} eV，LUMO {lumo:.3f} eV，{T('能隙')} {lumo - homo:.3f} eV")
        self._scene(reset=True, title=Path(p).name)
        self.app.log(f"已读取波函数 {Path(p).name}")
        return w

    def fill_orbitals(self):
        w = self.wfn
        self.orb.delete(*self.orb.get_children())
        spin = self.spin.get() if w.open_shell else "alpha"
        orbs = w.orbitals(spin)
        nocc = w.nalpha if spin == "alpha" else w.nbeta
        lo, hi = max(0, nocc - 15), min(len(orbs), nocc + 15)
        for o_ in orbs[lo:hi]:
            self.orb.insert("", "end", iid=str(o_["index"] - 1),
                            values=(o_["index"], o_["label"], f"{o_['energy_eV']:.3f}", f"{o_['occ']:.0f}"))
        stripe(self.orb)
        try:
            self.orb.selection_set(str(nocc - 1))
            self.orb.see(str(nocc - 1))
        except Exception:  # noqa: BLE001
            pass

    def _need(self):
        if self.wfn is None:
            self.load()
        return self.wfn

    # ---- 轨道 ----
    def show_orbital(self, which=None):
        w = self._need()
        spin = self.spin.get() if w.open_shell else "alpha"
        nocc = w.nalpha if spin == "alpha" else w.nbeta
        if which == "HOMO":
            idx = nocc - 1
        elif which == "LUMO":
            idx = nocc
        else:
            sel = self.orb.selection()
            idx = int(sel[0]) if sel else nocc - 1
        iso = abs(float(self.iso.get()))
        sp = self.GRID.get(self.grid_q.get(), 0.25)

        def work():
            from .wavefunction import isosurface
            cube = w.mo_cube(idx, spin, spacing_A=sp)
            return cube, isosurface(cube, iso), isosurface(cube, -iso)

        def done(res):
            cube, (vp, fp), (vn, fn) = res
            self.cube = cube
            lab = w.orbitals(spin)[idx]
            r = self.view.renderer
            r.meshes = [dict(verts=vp, faces=fp, color="#3B6FD8", alpha=0.72),
                        dict(verts=vn, faces=fn, color="#D84B3B", alpha=0.72)]
            r.annotations = []
            self._scene(title=f"{lab['label']} (#{idx + 1}, {lab['energy_eV']:.3f} eV)  ±{iso}")
            self.text.delete("1.0", "end")
            self.text.insert("end", f"{lab['label']}：轨道 {idx + 1}，能量 {lab['energy_eV']:.4f} eV，占据 {lab['occ']:.0f}\n"
                                    f"格点 {cube['shape']}，等值 ±{iso}，三角面 {len(fp) + len(fn)}\n")
        self.app.run_bg("计算分子轨道格点", work, done)

    def export_cube(self, path=None):
        if self.cube is None:
            raise ValueError("请先显示一个轨道")
        from .wavefunction import write_cube
        p = path or filedialog.asksaveasfilename(defaultextension=".cube", filetypes=[("cube", "*.cube")])
        if p:
            write_cube(p, self.cube)
            self.app.log(f"已导出 {p}")
        return p

    # ---- 电荷与键级 ----
    def mulliken(self):
        w = self._need()
        r = w.mulliken_mayer()
        self.show_charges(r["charges"], "Mulliken (fchk)", r["bond_orders"], r["spin"] if w.open_shell else None)
        return r

    def read_log_charges(self):
        from .interaction import read_charges
        if not self.chg_log.get():
            raise ValueError("请选择 Gaussian 输出文件")
        text = Path(self.chg_log.get()).read_text(errors="ignore")
        q, kind = None, ""
        k = text.rfind("Summary of Natural Population Analysis")
        if k >= 0:
            from .wavefunction import read_nbo
            q = np.array([v for _, _, v in read_nbo(self.chg_log.get())["npa"]])
            kind = "NPA"
        if q is None or not len(q):
            q, kind = read_charges(self.chg_log.get())
        w = self._need()
        if len(q) != len(w.Z):
            raise ValueError(f"电荷数 ({len(q)}) 与 fchk 原子数 ({len(w.Z)}) 不同")
        self.show_charges(q, f"{kind} ({Path(self.chg_log.get()).name})", None, None)

    def show_charges(self, q, label, B=None, spin=None):
        from matplotlib import cm
        w = self.wfn
        self.charges = np.asarray(q)
        vm = max(np.abs(q).max(), 1e-6)
        cmap = matplotlib.colormaps["bwr_r"]
        colors = [cmap(0.5 + 0.5 * v / vm)[:3] for v in q]
        r = self.view.renderer
        r.meshes = []
        X = w.coords * BOHR2ANG
        ann = [(X[i], f"{q[i]:+.2f}", "#1F2937") for i in range(len(q)) if w.symbols[i] != "H" or len(q) < 30]
        if B is not None:
            thr = float(self.bo_min.get())
            from .viewer import bonds_for
            for i, j in bonds_for(w.symbols, X):
                if B[i, j] >= thr:
                    ann.append(((X[i] + X[j]) / 2, f"{B[i, j]:.2f}", "#7B3F00"))
        r.annotations = ann
        self._scene(colors=colors, title=f"{label}：原子电荷 (红 +，蓝 −)" + ("；键级 (棕色)" if B is not None else ""))
        self._colorbar((-vm, vm, "bwr_r", "电荷 / e"))
        self._draw()
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"{label}：电荷总和 {np.sum(q):.4f}\n{'原子':>8}{'电荷':>10}" + (f"{'自旋':>10}" if spin is not None else "") + "\n")
        for i, v in enumerate(q):
            t.insert("end", f"{w.symbols[i]}{i + 1:>5}{v:10.4f}" + (f"{spin[i]:10.4f}" if spin is not None else "") + "\n")
        if B is not None:
            from .viewer import bonds_for
            t.insert("end", "\nMayer 键级 (成键原子对)：\n")
            for i, j in bonds_for(w.symbols, X):
                t.insert("end", f"  {w.symbols[i]}{i + 1}–{w.symbols[j]}{j + 1}  {B[i, j]:.4f}\n")

    # ---- ESP ----
    def run_esp(self):
        from .wavefunction import esp_surface, find_cubegen, read_cube, run_cubegen
        src = self.esp_src.get()
        iso = float(self.esp_iso.get())
        if src.startswith("已有"):
            if not (self.dens_cube.get() and self.esp_cube.get()):
                raise ValueError("请选择密度 cube 与 ESP cube (同一格点)")
            dpath, epath = self.dens_cube.get(), self.esp_cube.get()
            w = self.wfn

            def work():
                return esp_surface(read_cube(dpath), read_cube(epath), iso)
        elif src.startswith("Gaussian"):
            w = self._need()
            exe = find_cubegen(self.app.config_data.get("gaussian_exe") or None)
            if not exe:
                raise FileNotFoundError("没有找到 cubegen (随 Gaussian 安装，与 g16.exe 同目录)：请在“程序路径设置”中指定 "
                                        "Gaussian，或改用“已有 cube 文件”/“原子电荷近似”")
            fchk = Path(w.path)

            def work():
                d = fchk.with_name(fchk.stem + "_dens.cube")
                e = fchk.with_name(fchk.stem + "_esp.cube")
                run_cubegen(exe, fchk, "density=scf", d, "-2", cwd=fchk.parent)
                run_cubegen(exe, fchk, "potential=scf", e, "-2", cwd=fchk.parent)
                return esp_surface(read_cube(d), read_cube(e), iso)
        else:
            w = self._need()
            q = self.charges if self.charges is not None else w.mulliken_mayer()["charges"]

            def work():
                return esp_surface(w.density_cube(spacing_A=0.25), None, iso, charges=q)

        def done(es):
            from matplotlib import cm
            self.esp = es
            v = es["esp"]
            vm = max(abs(v.min()), abs(v.max()))
            fv = v[es["faces"]].mean(1)
            r = self.view.renderer
            r.meshes = [dict(verts=es["verts"], faces=es["faces"], face_rgb=matplotlib.colormaps["bwr"](0.5 + 0.5 * fv / vm)[:, :3],
                             alpha=0.88)]
            r.annotations = [(e_["xyz"], f"{e_['value']:.1f}", "#9B2C1F" if e_["kind"] == "max" else "#1F4E99")
                             for e_ in es["extrema"]]
            if self.wfn is not None:
                self._scene(title=f"ESP (kcal/mol, {es['method']})")
            else:
                self._draw()
            self._colorbar((-vm, vm, "bwr", "ESP / kcal·mol⁻¹"))
            self._draw()
            st = es["stats"]
            t = self.text
            t.delete("1.0", "end")
            t.insert("end", f"分子表面静电势 ({es['method']}，密度等值 {iso} a.u.)\n"
                            f"表面积 {st['area_A2']:.1f} Å²（正 {st['pos_area_A2']:.1f}，负 {st['neg_area_A2']:.1f}）\n"
                            f"最小 {st['min']:.2f}，最大 {st['max']:.2f}，平均 {st['mean']:.2f} kcal/mol，σ²tot {st['sigma2_tot']:.2f}\n"
                            f"表面极值点 ({len(es['extrema'])} 个)：\n")
            for e_ in es["extrema"]:
                t.insert("end", f"  {'极大' if e_['kind'] == 'max' else '极小'} {e_['value']:8.2f} kcal/mol  "
                                f"({e_['xyz'][0]:.2f}, {e_['xyz'][1]:.2f}, {e_['xyz'][2]:.2f}) Å\n")
        self.app.run_bg("计算分子表面静电势", work, done)

    def export_esp(self, path=None):
        if self.esp is None:
            raise ValueError("请先计算 ESP 表面")
        p = path or filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return None
        import csv as _csv
        with open(p, "w", newline="", encoding="utf-8-sig") as fh:
            wr = _csv.writer(fh)
            wr.writerow(["kind", "x_A", "y_A", "z_A", "ESP_kcal_mol"])
            for e_ in self.esp["extrema"]:
                wr.writerow([e_["kind"], *np.round(e_["xyz"], 4), round(e_["value"], 4)])
            wr.writerow([])
            for k, v in self.esp["stats"].items():
                wr.writerow([k, v])
        self.app.log(f"已导出 {p}")
        return p

    # ---- NBO ----
    def read_nbo(self):
        from .wavefunction import read_nbo
        if not self.nbo_log.get():
            raise ValueError("请选择含 NBO 结果的 Gaussian 输出")
        r = read_nbo(self.nbo_log.get())
        self.nbo = r
        thr = float(self.e2_min.get())
        na = int(self.nbo_na.get()) if self.nbo_na.get().strip() else None
        rows = [x for x in r["E2"] if x["E2"] >= thr]
        if na:
            side = lambda atoms: {a <= na for a in atoms}                          # noqa: E731
            rows = [x for x in rows if x["donor_atoms"] and x["acceptor_atoms"] and side(x["donor_atoms"]) != side(x["acceptor_atoms"])
                    and len(side(x["donor_atoms"])) == 1 and len(side(x["acceptor_atoms"])) == 1]
        rows.sort(key=lambda x: -x["E2"])
        t = self.text
        t.delete("1.0", "end")
        t.insert("end", f"NBO 二阶微扰能 E(2) ≥ {thr} kcal/mol" + ("，仅分子间 (A↔B)" if na else "") +
                        f"：{len(rows)} 项 (共 {len(r['E2'])})\n{'供体':<28}{'受体':<28}{'E(2)':>8}{'ΔE/a.u.':>10}{'F/a.u.':>9}\n")
        for x in rows[:80]:
            t.insert("end", f"{x['donor'][:27]:<28}{x['acceptor'][:27]:<28}{x['E2']:8.2f}{x['dE']:10.2f}{x['F']:9.3f}\n")
        if na and rows:
            t.insert("end", f"分子间 E(2) 合计 {sum(x['E2'] for x in rows):.2f} kcal/mol (反映分子间电荷转移相互作用)\n")
        if r["npa"]:
            t.insert("end", "\n自然电荷 (NPA)：" + "  ".join(f"{s}{i}:{q:+.3f}" for s, i, q in r["npa"][:40]) + "\n")
        if self.wfn is not None and r["npa"] and len(r["npa"]) == len(self.wfn.Z):
            self.show_charges(np.array([q for *_, q in r["npa"]]), "NPA (NBO)")
            t.insert("1.0", "")
        return rows

    # ---- IGM / IGMH / AIM ----
    def _density_source(self, src):
        """返回 (符号, 坐标 Å, Z, 坐标 bohr, provider 或 None, 说明)。"""
        from .substitution import Z as ZN
        from .topology import wfn_provider
        if src.startswith("当前 fchk"):
            w = self._need()
            return w.symbols, w.coords * BOHR2ANG, w.Z, w.coords, wfn_provider(w), f"fchk {Path(w.path).name}"
        if src.startswith("结构页"):
            st = self.app.struct_tab
            if not st.dimer_res:
                raise ValueError("请先在“结构/晶体”页搜索近邻二聚体 (可在表中选中一行)")
            kids = list(st.tree.get_children())
            sel = [kids.index(i) for i in st.tree.selection() if i in kids] if kids else []
            d = st.dimer_res["dimers"][sel[0] if sel else 0]
            atoms = [(s_, *x) for s_, x in st.dimer_res["atoms_a"]] + [(s_, *x) for s_, x in d["atoms_b"]]
            name = f"{st.group_name(d['group'])} ({d['r_centroid']:.2f} Å)"
        else:
            if not self.struct_file.get():
                raise ValueError("请选择结构文件 (二聚体 xyz / gjf / log)")
            atoms = read_geometry(self.struct_file.get())
            name = Path(self.struct_file.get()).name
        sym = [a[0] for a in atoms]
        XA = np.array([a[1:] for a in atoms], float)
        return sym, XA, np.array([ZN[s_] for s_ in sym]), XA / BOHR2ANG, None, name

    def _fragments(self, sym, XA):
        from .topology import fragments_by_connectivity
        if self.ig_frag.get().startswith("前"):
            n = int(self.ig_n.get())
            return np.array([0] * n + [1] * (len(sym) - n))
        return fragments_by_connectivity(sym, XA)

    def run_igm(self):
        from .topology import igm_analysis
        sym, XA, Z, R, prov, name = self._density_source(self.ig_src.get())
        frag = self._fragments(sym, XA)
        if len(set(frag.tolist())) < 2:
            raise ValueError("只识别出一个片段：二聚体请用“前 N 个原子为片段 A”指定")
        iso = float(self.ig_iso.get())
        h = float(self.ig_h.get())
        sign = -1 if self.ig_sign.get().startswith("原文") else 1

        def work():
            return igm_analysis(Z, R, frag, provider=prov, spacing_A=h, iso=iso, sign=sign)

        def done(res):
            from .topology import igm_surface
            self.igm = res
            self.cur = (sym, XA)
            v, f, col, val = igm_surface(res, iso)
            r = self.view.renderer
            r.meshes = [dict(verts=v, faces=f, face_rgb=col, alpha=0.85)] if len(f) else []
            r.annotations = []
            self._scene(reset=True, title=f"{res['method']}: δg_inter = {iso} ({name})")
            self._colorbar((-0.05, 0.05, "bgr", "sign(λ2)ρ / a.u."))
            self._draw()
            t = self.text
            t.delete("1.0", "end")
            t.insert("end", f"{res['method']}，{name}，片段数 {len(set(frag.tolist()))}，格点间距 {h} Å，计算点 {res['n_points']}\n"
                            f"δG_inter = ∫δg_inter dV = {res['dG_inter']:.4f} a.u. (越大表示片段间相互作用越强，用于同类体系比较)\n"
                            f"δg_inter 最大 {res['dg_max']:.4f} a.u.；等值面三角形 {len(f)}"
                            + (f"；sign(λ2)ρ 范围 [{val.min():.4f}, {val.max():.4f}] a.u." if len(val) else "") + "\n")
            if not len(f):
                t.insert("end", f"[提示] 等值 {iso} 下没有等值面，可改为 {res['dg_max'] / 2:.4f}\n")
            if res["method"] == "IGMH":
                t.insert("end", "IGMH 形式：" + ("原文实现 (作图效果更好)" if sign < 0 else "数学严格形式") + "\n")
        self.app.run_bg("计算 δg_inter 格点", work, done)

    def igm_scatter(self):
        if self.igm is None:
            raise ValueError("请先计算 δg_inter")
        from .topology import bgr_colors
        win = tk.Toplevel(self)
        win.title(T("δg_inter 散点图"))
        pp = PlotPanel(win, figsize=(5.5, 4.2))
        pp.pack(fill="both", expand=True)
        ax = pp.new_axes()
        x, y = self.igm["scatter"]
        ax.scatter(x, y, s=2, c=bgr_colors(x), rasterized=True)
        ax.set_xlim(-0.05, 0.05)
        ax.set_xlabel("sign(λ$_2$)ρ / a.u.")
        ax.set_ylabel("δg$_{inter}$ / a.u.")
        ax.set_title(self.igm["method"])
        pp.draw()
        return pp

    def export_igm(self, path=None):
        if self.igm is None:
            raise ValueError("请先计算 δg_inter")
        from .wavefunction import write_cube
        p = path or filedialog.asksaveasfilename(defaultextension=".cube", filetypes=[("cube", "*.cube")])
        if not p:
            return None
        p = Path(p)
        write_cube(p.with_name(p.stem + "_dg_inter.cube"), self.igm["dg_inter"])
        write_cube(p.with_name(p.stem + "_sl2r.cube"), self.igm["sl2r"])
        self.app.log(f"已导出 {p.stem}_dg_inter.cube 与 {p.stem}_sl2r.cube (可用 VMD 作图)")
        return p

    def run_aim(self):
        from .topology import aim_analysis
        sym, XA, Z, R, prov, name = self._density_source(self.aim_src.get())
        frag = self._fragments(sym, XA)
        if prov is None:
            from .topology import Promolecule
            prov = Promolecule(Z, R)
        kin = self.aim_src.get().startswith("当前 fchk")

        def work():
            return aim_analysis(Z, R, prov, frag=frag if len(set(frag.tolist())) > 1 else None, kinetic=kin)

        def done(res):
            self.aim = res
            self.cur = (sym, XA)
            r = self.view.renderer
            r.meshes = []
            colors = {"BCP": "#E08E0B", "RCP": "#2E9E44", "CCP": "#7D3C98"}
            ann, cyl = [], []
            bcps = [c for c in res["cps"] if c["type"] == "BCP"]
            for c in res["cps"]:
                inter = c["type"] == "BCP" and c.get("atoms") and frag[c["atoms"][0]] != frag[c["atoms"][1]]
                ann.append((c["xyz_bohr"] * BOHR2ANG, f"{c['rho']:.3f}" if inter else "", colors[c["type"]]))
            for c, path in zip(bcps, res["paths"]):
                inter = c.get("atoms") and frag[c["atoms"][0]] != frag[c["atoms"][1]]
                step = 3
                for p_, q_ in zip(path[:-step:step], path[step::step]):
                    cyl.append((p_, q_, 0.05 if inter else 0.025, "#B8860B" if inter else "#C9A227"))
            r.annotations = ann
            r.cylinders = cyl
            self._scene(reset=True, title=f"AIM ({name})")
            r.cylinders = []
            t = self.text
            t.delete("1.0", "end")
            cn = res["counts"]
            t.insert("end", f"AIM 拓扑 ({name})：原子核 {cn['NCP']}，键临界点 {cn['BCP']}，环 {cn['RCP']}，笼 {cn['CCP']}；"
                            f"Poincaré–Hopf n − b + r − c = {res['poincare_hopf']}" + (" ✓\n" if res["ph_ok"] else " (≠ 1，可能有遗漏)\n"))
            t.insert("end", f"{'键临界点':<14}{'ρ':>9}{'∇²ρ':>9}{'ε':>8}" + (f"{'G':>9}{'V':>9}{'H':>9}{'E≈V/2':>9}" if kin else "") + "\n")
            rows = sorted(bcps, key=lambda c: (not (c.get("atoms") and frag[c["atoms"][0]] != frag[c["atoms"][1]]), -c["rho"]))
            for c in rows:
                a_ = c.get("atoms")
                lab = f"{sym[a_[0]]}{a_[0] + 1}–{sym[a_[1]]}{a_[1] + 1}" if a_ else "?"
                inter = a_ and frag[a_[0]] != frag[a_[1]]
                t.insert("end", f"{('*' if inter else ' ') + lab:<14}{c['rho']:9.4f}{c['lap']:9.4f}{c.get('ellipticity', 0):8.3f}"
                                + (f"{c['G']:9.4f}{c['V']:9.4f}{c['H']:9.4f}{c['E_kcal']:9.2f}" if kin else "") + "\n")
            t.insert("end", "* 为分子间键临界点；ρ、∇²ρ、G、V、H 为原子单位，E 为 kcal/mol\n")
            self._redraw_aim(cyl)
        self.app.run_bg("搜索临界点", work, done)

    def _redraw_aim(self, cyl):
        r = self.view.renderer
        r.cylinders = cyl
        self._draw()

    def export_aim(self, path=None):
        if self.aim is None:
            raise ValueError("请先搜索临界点")
        p = path or filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return None
        import csv as _csv
        with open(p, "w", newline="", encoding="utf-8-sig") as fh:
            wr = _csv.writer(fh)
            wr.writerow(["type", "atoms", "x_A", "y_A", "z_A", "rho", "lap", "ellipticity", "G", "V", "H", "E_kcal"])
            for c in self.aim["cps"]:
                a_ = c.get("atoms")
                wr.writerow([c["type"], f"{a_[0] + 1}-{a_[1] + 1}" if a_ else "", *np.round(c["xyz_bohr"] * BOHR2ANG, 4),
                             c["rho"], c["lap"], c.get("ellipticity", ""), c.get("G", ""), c.get("V", ""), c.get("H", ""),
                             c.get("E_kcal", "")])
        self.app.log(f"已导出 {p}")
        return p

    def save_image(self):
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                                                                            ("TIFF", "*.tif")])
        if p:
            self.view.fig.savefig(p, dpi=300)
            self.app.log(f"已保存 {p} (300 dpi)")


# ==============================================================================================
# 批量扫描工作流 (参数网格、共用片段、续算、解析、对称检验、验证、模型、能量、报告、归档)
# ==============================================================================================
class ScanTab(ttk.Frame):
    KINDS = ["参数网格 (距离 / 滑移 / 旋转)", "随机验证集 (拉丁超立方)", "对称等价检验", "晶体近邻二聚体"]
    ENERGY = {"ADF": "ADF 能量分解 (总成键能)", "CP": "Gaussian 抵消校正", "UFF": "UFF (软件内，定性)"}
    PREVIEW = ["自动", "滑移图 V(Δx, Δy)", "距离衰减 ln|V|–z", "V–S 关系", "相互作用能图"]
    CE_SRC = ["扫描项目 (局域 / 可分离)", "软件内 EHT", "局域扫描 + 乘积模型 (T 对)"]
    FRAMES = ["全部重原子", "共轭主体 (最大稠环体系)"]

    def __init__(self, parent, app):
        from .scanflow import BASES, XC_LINES
        super().__init__(parent, style="Bg.TFrame")
        self.app = app
        self.proj = None
        S = safe(app)
        panel = SectionPanel(self, width=470)
        panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left = panel.inner

        a = ttk.LabelFrame(left, text="扫描项目")
        a.pack(fill="x")
        self.mono = FileEntry(a, "单体结构 (优化后)", filetypes=STRUCT_TYPES)
        self.mono.pack(fill="x")
        af = ttk.Frame(a)
        af.pack(fill="x")
        self.pname = field(af, 0, "项目名称 (英文)", "pentacene", width=24)
        self.pframe = field(af, 1, "分子坐标系", self.FRAMES[0], values=self.FRAMES, width=24)
        ab = ttk.Frame(a)
        ab.pack(fill="x")
        ttk.Button(ab, text="新建项目…", command=S(self.new_project)).pack(side="left", fill="x", expand=True)
        ttk.Button(ab, text="打开项目…", command=S(self.open_project)).pack(side="left", fill="x", expand=True)
        ttk.Button(ab, text="打开文件夹", command=S(lambda: self._open_folder(self._need().root))).pack(
            side="left", fill="x", expand=True)
        self.plbl = ttk.Label(a, text="尚未打开项目", foreground=PALETTE["muted"], wraplength=440)
        self.plbl.pack(anchor="w")

        lv = ttk.LabelFrame(left, text="计算层级 (每个层级一个共用的单体片段 mono.t21)")
        lv.pack(fill="x", pady=3)
        lf = ttk.Frame(lv)
        lf.pack(fill="x")
        self.xc = field(lf, 0, "泛函", "PW91", values=list(XC_LINES), width=10)
        self.basis = field(lf, 0, "基组", "TZP", values=BASES, width=8, col=2)
        self.d3 = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf, text="D3(BJ)", variable=self.d3).grid(row=0, column=4, sticky="w")
        lb = ttk.Frame(lv)
        lb.pack(fill="x")
        ttk.Button(lb, text="添加层级", command=S(self.add_level)).pack(side="left", fill="x", expand=True)
        ttk.Button(lb, text="导入已有 mono.t21…", command=S(self.import_t21)).pack(side="left", fill="x", expand=True)
        ttk.Button(lb, text="运行单体片段", command=S(self.run_monomer)).pack(side="left", fill="x", expand=True)
        lf2 = ttk.Frame(lv)
        lf2.pack(fill="x")
        ttk.Label(lf2, text="当前层级").grid(row=0, column=0, sticky="w", padx=2)
        self.level = tk.StringVar(value="")
        self.level_cb = ttk.Combobox(lf2, textvariable=self.level, values=[], state="readonly", width=30)
        self.level_cb.grid(row=0, column=1, sticky="we", padx=2)

        n = ttk.LabelFrame(left, text="新建扫描")
        n.pack(fill="x", pady=3)
        nf = ttk.Frame(n)
        nf.pack(fill="x")
        self.kind = field(nf, 0, "类型", self.KINDS[0], values=self.KINDS, width=28)
        self.sname = field(nf, 1, "扫描名称 (英文)", "slip35", width=28)
        ttk.Label(n, text="取值写法：起点:终点:步长 (如 0:6:0.25)，或用逗号分隔的数值，或单个值",
                  foreground=PALETTE["muted"], wraplength=440).pack(anchor="w")
        g = ttk.Frame(n)
        g.pack(fill="x")
        self.p_z = field(g, 0, "z / Å", "3.5", width=12)
        self.p_dx = field(g, 0, "Δx / Å", "0:6:0.25", width=12, col=2)
        self.p_dy = field(g, 1, "Δy / Å", "0:3:0.25", width=12)
        self.p_roll = field(g, 1, "φ (绕长轴 x) / °", "0", width=12, col=2)
        self.p_pitch = field(g, 2, "θ (绕短轴 y) / °", "0", width=12)
        self.p_gamma = field(g, 2, "γ (绕法线 z) / °", "0", width=12, col=2)
        self.clash = field(g, 3, "接触阈值 d/(rA+rB)", "0.75", width=12)
        self.chunk = field(g, 3, "每个批处理的构型数", "100", width=12, col=2)
        self.reduce = tk.BooleanVar(value=True)
        ttk.Checkbutton(n, text="按单体点群约简 (对称等价构型只算一次，结果自动展开)", variable=self.reduce).pack(anchor="w")
        v = ttk.Frame(n)
        v.pack(fill="x")
        ttk.Label(v, text="验证集", foreground=PALETTE["muted"]).grid(row=0, column=0, sticky="w")
        self.v_ni = field(v, 1, "内插个数 / z 范围", "20", width=6)
        self.v_zi = field(v, 1, "", "3.5 3.8", width=14, col=2)
        self.v_ne = field(v, 2, "外推个数 / z 范围", "20", width=6)
        self.v_ze = field(v, 2, "", "3.3 3.5; 3.8 4.2", width=14, col=2)
        self.v_dxy = field(v, 3, "Δx 范围 / Δy 范围", "-6 6", width=6)
        self.v_dy = field(v, 3, "", "-3 3", width=14, col=2)
        self.v_seed = field(v, 4, "随机种子", "2026", width=6)
        c = ttk.Frame(n)
        c.pack(fill="x")
        self.c_bases = field(c, 0, "对称检验基准 z Δx Δy φ", "3.5 1.2 0.7 0; 3.6 2.0 0.4 25", width=30)
        ttk.Label(n, text="晶体近邻二聚体：先在“结构/晶体”页打开 CIF 并搜索近邻",
                  foreground=PALETTE["muted"], wraplength=440).pack(anchor="w")
        nb_ = ttk.Frame(n)
        nb_.pack(fill="x")
        ttk.Button(nb_, text="预览构型数", command=S(self.preview_count)).pack(side="left", fill="x", expand=True)
        ttk.Button(nb_, text="生成扫描", style="Accent.TButton", command=S(self.create_scan)).pack(
            side="left", fill="x", expand=True)
        ttk.Button(nb_, text="导入已有扫描目录…", command=S(self.import_scan)).pack(side="left", fill="x", expand=True)

        r = ttk.LabelFrame(left, text="运行与结果 (对右侧选中的扫描)")
        r.pack(fill="x", pady=3)
        rb = ttk.Frame(r)
        rb.pack(fill="x")
        for txt, cmd in (("运行前检查", self.precheck), ("▶ 运行", self.run_scan), ("收集结果", self.collect)):
            ttk.Button(rb, text=txt, command=S(cmd)).pack(side="left", fill="x", expand=True)
        rb2 = ttk.Frame(r)
        rb2.pack(fill="x")
        for txt, cmd in (("对称检验结果", self.sym_result), ("换当前层级重算", self.clone), ("打开扫描文件夹", self.open_scan)):
            ttk.Button(rb2, text=txt, command=S(cmd)).pack(side="left", fill="x", expand=True)
        gf = ttk.Frame(r)
        gf.pack(fill="x")
        self.g_func = field(gf, 0, "Gaussian 泛函 / 基组", "B3LYP", width=10)
        self.g_basis = field(gf, 0, "", "6-31G(d)", width=10, col=2)
        self.g_stride = field(gf, 1, "每隔几个构型取一个", "1", width=10)
        rb3 = ttk.Frame(r)
        rb3.pack(fill="x")
        ttk.Button(rb3, text="Gaussian 抵消校正能量输入", command=S(self.gauss_cp)).pack(side="left", fill="x", expand=True)
        ttk.Button(rb3, text="Gaussian 转移积分输入", command=S(self.gauss_ti)).pack(side="left", fill="x", expand=True)

        an = ttk.LabelFrame(left, text="分析、报告与归档")
        an.pack(fill="x", pady=3)
        af2 = ttk.Frame(an)
        af2.pack(fill="x")
        self.lam_h = field(af2, 0, "重组能 λh / λe (eV)", "", width=8)
        self.lam_e = field(af2, 0, "", "", width=8, col=2)
        self.temp = field(af2, 1, "温度 / K", "300", width=8)
        self.z0 = field(af2, 1, "参考 z0 / Å", "3.5", width=8, col=2)
        self.energy = MappedVar("ADF", self.ENERGY)
        ttk.Label(af2, text="相互作用能").grid(row=2, column=0, sticky="w")
        ttk.Combobox(af2, textvariable=self.energy, values=list(self.ENERGY), state="readonly", width=24).grid(
            row=2, column=1, columnspan=3, sticky="we")
        self.reorg_dir = FileEntry(an, "重组能目录 (四点法输出，可选)", mode="dir")
        self.reorg_dir.pack(fill="x")
        anb = ttk.Frame(an)
        anb.pack(fill="x")
        ttk.Button(anb, text="生成报告", style="Accent.TButton", command=S(self.report)).pack(side="left", fill="x", expand=True)
        ttk.Button(anb, text="归档…", command=S(self.archive)).pack(side="left", fill="x", expand=True)
        self.with_out = tk.BooleanVar(value=True)
        ttk.Checkbutton(anb, text="含 ADF 原始输出", variable=self.with_out).pack(side="left")

        ce = ttk.LabelFrame(left, text="CEDS 动态无序与 μ(T) (晶体取自“结构/晶体”页)")
        ce.pack(fill="x", pady=3)
        cf = ttk.Frame(ce)
        cf.pack(fill="x")
        self.ce_src = field(cf, 0, "耦合曲面", self.CE_SRC[0], values=self.CE_SRC, width=26)
        self.ce_T = field(cf, 1, "温度列表 / K", "100 150 200 250 300", width=26)
        self.ce_n = field(cf, 2, "MC 样本数 / 每对扫描构型数", "1500", width=10)
        self.ce_nloc = field(cf, 2, "", "150", width=10, col=2)
        self.ce_pairs = field(cf, 4, "扫描只做这些组 (空=全部)", "", width=10)
        self.ce_htau = field(cf, 3, "ħ/τ_in / meV", "5", width=10)
        self.ce_ueq = field(cf, 3, "目标 U_eq / Å² (可空)", "", width=10, col=2)
        self.ce_fallback = tk.BooleanVar(value=False)
        ttk.Checkbutton(ce, text="缺少 DFT 曲面的二聚体用 EHT 补上 (只作预览)", variable=self.ce_fallback).pack(anchor="w")
        csv_t = [("CSV", "*.csv"), ("所有文件", "*.*")]
        self.ce_gcsv = FileEntry(ce, "G 网格表", filetypes=csv_t, width=26)
        self.ce_gcsv.pack(fill="x")
        self.ce_fdx = FileEntry(ce, "Δx 扫描表", filetypes=csv_t, width=26)
        self.ce_fdx.pack(fill="x")
        self.ce_lcsv = FileEntry(ce, "局域结果表", filetypes=csv_t, width=26)
        self.ce_lcsv.pack(fill="x")
        ttk.Label(ce, text="以上三项只用于“局域扫描 + 乘积模型”：T 对用乘积模型 (G 网格 × Δx 扫描)，平行对用局域扫描 "
                           "(项目中的数据，或没有 ADF 输出文件时直接给局域扫描的 results.csv)",
                  foreground=PALETTE["muted"], wraplength=440).pack(anchor="w")
        mf = ttk.Frame(ce)
        mf.pack(fill="x")
        self.ce_sc = field(mf, 0, "MD 超胞 na nb nc", "4 3 2", width=10)
        self.ce_mdps = field(mf, 0, "MD 采样 / ps", "30", width=6, col=2)
        self.ce_mono = FileEntry(ce, "MD 单体", filetypes=STRUCT_TYPES + [("项目", "*.json")], width=26)
        self.ce_mono.pack(fill="x")
        self.ce_mdhalf = tk.BooleanVar(value=False)
        ttk.Checkbutton(ce, text="局域扫描范围按全晶体 MD 的涨落确定 (±3σ；晶体没有原子位移参数时推荐)",
                        variable=self.ce_mdhalf).pack(anchor="w")
        self.ce_traj = tk.BooleanVar(value=False)
        ttk.Checkbutton(ce, text="保存 MD 轨迹 (npz，存到扫描项目的 md 目录)", variable=self.ce_traj).pack(anchor="w")
        cb = ttk.Frame(ce)
        cb.pack(fill="x")
        ttk.Button(cb, text="晶体二聚体 → 局域扫描", command=S(self.ceds_local)).pack(side="left", fill="x", expand=True)
        ttk.Button(cb, text="运行 CEDS (σ_V)", command=S(self.ceds_run)).pack(side="left", fill="x", expand=True)
        cb2 = ttk.Frame(ce)
        cb2.pack(fill="x")
        ttk.Button(cb2, text="μ(T)：CEDS + TLT", style="Accent.TButton", command=S(self.ceds_muT)).pack(
            side="left", fill="x", expand=True)
        ttk.Button(cb2, text="导出 CEDS 结果…", command=S(self.ceds_export)).pack(side="left", fill="x", expand=True)
        cb3 = ttk.Frame(ce)
        cb3.pack(fill="x")
        ttk.Button(cb3, text="全晶体 MD：σ_V 与 μ(T)", style="Accent.TButton", command=S(self.ceds_md)).pack(
            side="left", fill="x", expand=True)
        cb4 = ttk.Frame(ce)
        cb4.pack(fill="x")
        ttk.Button(cb4, text="MD 快照 → ADF 批处理", command=S(self.ceds_snapshots)).pack(side="left", fill="x", expand=True)
        ttk.Button(cb4, text="快照检验：直接计算 vs 曲面", command=S(self.ceds_validate)).pack(
            side="left", fill="x", expand=True)
        ttk.Label(ce, text="全晶体 MD：超胞中所有分子作刚体运动 (OPLS-AA + 静电)，代替局域 MC 的“邻居固定”近似；"
                           "每个温度 4×3×2 超胞约 10 分钟。MD 单体留空时取扫描项目的单体 (DFT 优化)，再不然取晶体中的分子",
                  foreground=PALETTE["muted"], wraplength=440).pack(anchor="w")
        ttk.Label(ce, text="局域扫描的范围由 MC 涨落 (±2.5σ) 自动确定；扫描名称取“新建扫描”中的名称，层级取“当前层级”",
                  foreground=PALETTE["muted"], wraplength=440).pack(anchor="w")
        self.ce, self.ce_results, self.ce_rows, self.ce_scale = None, {}, None, 1.0
        self.ce_md_rows = None
        self.ce_validation = None

        right = ttk.Frame(self, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        cols = ("扫描", "用途", "层级", "请求", "计算", "完成", "失败", "待算", "筛除")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=7, selectmode="browse")
        for col, w in zip(cols, (150, 80, 150, 60, 60, 60, 60, 60, 60)):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", lambda e: S(self.preview)())
        pv = ttk.Frame(right)
        pv.pack(fill="x", pady=(4, 0))
        self.pv_kind = field(pv, 0, "预览", self.PREVIEW[0], values=self.PREVIEW, width=20)
        self.pv_car = field(pv, 0, "载流子", "空穴", values=["空穴", "电子"], width=8, col=2)
        ttk.Button(pv, text="作图", command=S(self.preview)).grid(row=0, column=4, padx=4)
        ttk.Button(pv, text="导出图片 (300 dpi)…", command=S(self.save_fig)).grid(row=0, column=5)
        self.plot = PlotPanel(right, figsize=(7, 4.2))
        self.plot.pack(fill="both", expand=True, pady=(4, 0))
        self.text = styled_text(right, app.mono, height=11, wrap="none", size=9)
        self.text.pack(fill="x", pady=(6, 0))

    # ---------------- 工具 ----------------
    def _need(self):
        if self.proj is None:
            raise ValueError("请先新建或打开扫描项目")
        return self.proj

    def _sel(self):
        s = self.tree.selection()
        if not s:
            raise ValueError("请在右侧表格中选择一个扫描")
        return s[0]

    def _level(self):
        p = self._need()
        lv = self.level.get()
        if lv not in p.meta["levels"]:
            raise ValueError("请先添加或选择计算层级")
        return lv

    def show(self, txt):
        self.text.delete("1.0", "end")
        self.text.insert("end", T(txt))

    @staticmethod
    def _open_folder(path):
        import subprocess
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def refresh(self):
        p = self._need()
        rep = p.meta["monomer"]["report"]
        levels = list(p.meta["levels"])
        self.level_cb.configure(values=levels)
        if self.level.get() not in levels:
            self.level.set(levels[0] if levels else "")
        lv_txt = "; ".join(f"{k} ({'✓' if p.t21(k).exists() else T('片段未算')})" for k in levels) or T("无")
        self.plbl.config(text=f"{p.meta['name']}: {rep['formula']}, {rep['point_group']} · {T('层级')}: {lv_txt}\n{p.root}")
        from .scananalysis import ROLE_ZH, scan_roles
        roles = {k: ROLE_ZH.get(v, v) for k, v in scan_roles(p).items()}
        sel = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for n in p.scans():
            try:
                info = p.scan_info(n)
                recs = p.collect(n, write=False)
            except Exception as exc:  # noqa: BLE001
                self.tree.insert("", "end", iid=n, values=(n, "?", "", "", "", "", "", "", str(exc)[:40]))
                continue
            comp = [x for x in recs if not x["rep"]]
            st = [sum(x["state"] == s for x in comp) for s in ("done", "failed", "pending", "excluded")]
            self.tree.insert("", "end", iid=n, values=(n, T(roles[n]), info.get("level", ""), len(recs),
                                                      len(comp) - st[3], *st))
        if sel and self.tree.exists(sel[0]):
            self.tree.selection_set((sel[0],))

    # ---------------- 项目与层级 ----------------
    def new_project(self):
        from .scanflow import ScanProject, _ascii_path_warning
        mono = self.mono.get()
        if not mono:
            raise ValueError("请先选择单体结构 (优化后的 .log / .out / .xyz / .gjf)")
        name = re.sub(r"[^A-Za-z0-9_.\-]", "", self.pname.get()) or "scan"
        parent = filedialog.askdirectory(title=T("选择项目所在的文件夹 (建议纯英文路径)"),
                                         initialdir=self.app.config_data.get("workdir") or None)
        if not parent:
            return
        frame = "core" if self.FRAMES.index(self.pframe.get()) == 1 else "all"
        self.proj = ScanProject.create(Path(parent) / name, mono, name, frame=frame)
        self.app.config_data["scan_project"] = str(self.proj.root)
        save_config(self.app.config_data)
        rep = self.proj.meta["monomer"]["report"]
        msg = (f"已新建扫描项目 {self.proj.root}\n单体 {rep['formula']}：分子坐标系中的点群 {rep['point_group']} "
               f"(操作 {', '.join(rep['ops'])})，重原子最大偏离平面 {rep['planarity_A']:.3f} Å，"
               f"长 × 宽 {rep['length_A']:.2f} × {rep['width_A']:.2f} Å；分子坐标系：{self.pframe.get()}")
        w = _ascii_path_warning(self.proj.root)
        self.app.log(msg + ("\n[警告] " + w if w else ""))
        self.show(msg + "\n\n下一步：添加计算层级 (默认 PW91 / TZP)，运行单体片段，然后生成扫描")
        self.refresh()

    def open_project(self, d=None):
        from .scanflow import ScanProject
        d = d or filedialog.askdirectory(title=T("选择扫描项目文件夹 (含 project.json)"),
                                         initialdir=self.app.config_data.get("scan_project") or None)
        if not d:
            return
        self.proj = ScanProject(d)
        self.app.config_data["scan_project"] = str(self.proj.root)
        save_config(self.app.config_data)
        self.app.log(f"已打开扫描项目 {self.proj.root}")
        self.refresh()
        self.show(self.proj.readme())

    def add_level(self):
        p = self._need()
        lv = p.add_level(self.xc.get(), self.basis.get(), self.d3.get())
        self.refresh()
        self.level.set(lv)
        self.app.log(f"已添加层级 {lv}：{p.level_dir(lv)}\\mono.run (在 ADFjobs 中运行，生成 mono.t21)，"
                     f"或点“运行单体片段”")

    def import_t21(self):
        p = self._need()
        lv = self._level()
        f = filedialog.askopenfilename(title=T("选择已算好的单体片段 mono.t21"),
                                       filetypes=[("TAPE21", "*.t21"), ("所有文件", "*.*")])
        if f:
            p.import_fragment(lv, f)
            self.app.log(f"已把 {f} 复制为层级 {lv} 的片段文件；请确认它与项目单体用的是同一个结构和计算级别")
            self.refresh()

    def run_monomer(self):
        p = self._need()
        self.app.open_runner(str(p.level_dir(self._level())))

    # ---------------- 新建扫描 ----------------
    def _configs(self):
        from .scanflow import grid_configs
        return grid_configs({"z": self.p_z.get(), "dx": self.p_dx.get(), "dy": self.p_dy.get(), "gamma": self.p_gamma.get(),
                             "pitch": self.p_pitch.get(), "roll": self.p_roll.get()})

    @staticmethod
    def _pairs(s):
        out = []
        for part in str(s).replace("，", ";").replace(",", " ").split(";"):
            v = [float(x) for x in part.split()]
            if v:
                if len(v) != 2:
                    raise ValueError(f"区间应写成“下限 上限”，多个区间用分号分隔 (收到 {part.strip()})")
                out.append(tuple(v))
        return out

    def _blocks(self):
        dx, dy = self._pairs(self.v_dxy.get())[0], self._pairs(self.v_dy.get())[0]
        bl = []
        zi = self._pairs(self.v_zi.get())
        ni = int(self.v_ni.get() or 0)
        if ni and zi:
            bl.append({"tag": "interp", "n": ni, "z": zi[0], "dx": dx, "dy": dy})
        ze = self._pairs(self.v_ze.get())
        ne = int(self.v_ne.get() or 0)
        if ne and ze:
            w = np.array([hi - lo for lo, hi in ze], float)
            k = np.floor(ne * w / w.sum()).astype(int)
            k[0] += ne - k.sum()
            bl += [{"tag": "extrap", "n": int(m), "z": zr, "dx": dx, "dy": dy} for zr, m in zip(ze, k) if m]
        return bl

    def _bases(self):
        out = []
        for part in self.c_bases.get().replace("，", ";").split(";"):
            v = [float(x) for x in part.replace(",", " ").split()]
            if v:
                v += [0.0] * (4 - len(v))
                out.append({"z": v[0], "dx": v[1], "dy": v[2], "roll": v[3]})
        if not out:
            raise ValueError("请给出至少一个对称检验基准构型 (z Δx Δy φ)")
        return out

    def preview_count(self):
        from .scanflow import canonical, cfg_key
        p = self._need()
        k = self.KINDS.index(self.kind.get())
        if k == 0:
            cfgs = self._configs()
            reps = {cfg_key(canonical(c, p.ops)[0]) for c in cfgs} if self.reduce.get() else {cfg_key(c) for c in cfgs}
            msg = f"请求 {len(cfgs)} 个构型，对称约简后需计算 {len(reps)} 个 (接触筛除在生成时进行)"
        elif k == 1:
            msg = "验证集：" + "，".join(f"{b['tag']} {b['n']} 个 (z {b['z'][0]}–{b['z'][1]} Å)" for b in self._blocks())
        elif k == 2:
            msg = f"{len(self._bases())} 个基准构型，每个生成至多 {len(p.ops) + 3} 个对称变体"
        else:
            res = getattr(self.app.struct_tab, "dimer_res", None)
            msg = f"{len(res['groups'])} 组晶体近邻二聚体" if res else "尚未搜索晶体近邻"
        self.app.log(msg)
        self.show(msg)

    def create_scan(self):
        p = self._need()
        lv = self._level()
        name = self.sname.get().strip()
        k = self.KINDS.index(self.kind.get())
        clash, chunk = float(self.clash.get()), int(self.chunk.get())
        if k == 0:
            info = p.add_scan(name, self._configs(), lv, kind="grid", clash_min=clash, chunk=chunk,
                              reduce=self.reduce.get(),
                              info={"spec": {"z": self.p_z.get(), "dx": self.p_dx.get(), "dy": self.p_dy.get(),
                                             "gamma": self.p_gamma.get(), "pitch": self.p_pitch.get(), "roll": self.p_roll.get()}})
        elif k == 1:
            info = p.add_validation(name, lv, self._blocks(), seed=int(self.v_seed.get()), clash_min=clash, chunk=chunk)
        elif k == 2:
            info = p.add_check(name, lv, self._bases(), chunk=chunk)
        else:
            res = getattr(self.app.struct_tab, "dimer_res", None)
            if not res:
                raise ValueError("请先在“结构/晶体”页打开 CIF 并搜索近邻二聚体")
            info = p.add_crystal(name, lv, res, chunk=chunk)
        msg = (f"已生成扫描 {name}：请求 {info['n_requested']} 个构型，计算 {info['n_computed']} 个，"
               f"对称映射 {info['n_symmetry']} 个，接触过近筛除 {info['n_clash']} 个；"
               f"批处理 {', '.join(info['batches'])} (所有构型共用 {p.t21(lv).name}，层级 {lv})")
        if not p.t21(lv).exists():
            msg += "\n[提示] 本层级的单体片段尚未计算：先运行 monomer 下的 mono.run"
        self.app.log(msg)
        self.show(msg)
        self.refresh()
        self.tree.selection_set((name,))

    def import_scan(self):
        p = self._need()
        d = filedialog.askdirectory(title=T("选择已有扫描目录 (make_dimers.py 生成，含 manifest.csv)"))
        if d:
            lv = self.level.get() if self.level.get() in p.meta["levels"] else None
            n = p.import_scan(d, level=lv)
            self.app.log(f"已导入扫描 {n} ({d})")
            self.refresh()

    # ---------------- 运行与结果 ----------------
    def precheck(self):
        p = self._need()
        n = self._sel()
        r = p.precheck(n)
        lines = [f"{n}: {r['n_done']} / {r['n_total']} 已完成，待算 {r['n_pending']}，失败 {r['n_failed']}"]
        if r["est_MB"]:
            lines.append(f"预计还需磁盘约 {r['est_MB']:.0f} MB")
        lines += ["[错误] " + e for e in r["errors"]] + ["[提示] " + w for w in r["warnings"]]
        lines.append("可以运行" if r["ok"] else "请先处理上面的错误")
        lines.append("批处理：" + ", ".join(r["batches"]) + " (ADFjobs 打开 .run；已完成的构型会自动跳过，可随时中断续算)")
        self.show("\n".join(lines))
        self.app.log(lines[0])
        self.refresh()
        return r

    def run_scan(self):
        r = self.precheck()
        if not r["ok"]:
            raise ValueError("运行前检查未通过：" + "；".join(r["errors"]))
        self.app.open_runner(str(self._need().scan_dir(self._sel())))

    def open_scan(self):
        self._open_folder(self._need().scan_dir(self._sel()))

    def collect(self):
        p = self._need()
        n = self._sel()

        def work():
            return p.collect(n)

        def done(recs):
            comp = [x for x in recs if not x["rep"]]
            ok = [x for x in comp if x["state"] == "done"]
            mism = sum(1 for x in ok if x.get("sign_mismatch_hole") or x.get("sign_mismatch_electron"))
            lines = [f"{n}: {len(ok)} / {len(comp)} 个构型已解析，结果已写入 results.csv (含对称展开的 {len(recs) - len(comp)} 个)"]
            if mism:
                lines.append(f"[提示] {mism} 个构型 ADF 打印的 V 与由 J、S、e1、e2 重算的 V 符号不同，已按重算值取符号")
            lines.append(f"{'id':<14}{'z':>6}{'Δx':>7}{'Δy':>7}{'φ':>7}{'V_h':>10}{'V_e':>10}{'S_h':>10}{'E_int':>9}")
            for x in ok[:400]:
                e = x.get("E_total")
                lines.append(f"{x['id']:<14}{x['z']:6.2f}{x['dx']:7.2f}{x['dy']:7.2f}{x['roll']:7.1f}{x['hole']:10.2f}"
                             f"{x['electron']:10.2f}{(x.get('S_hole') or 0):10.5f}{'' if e is None else f'{e:9.2f}'}")
            self.show("\n".join(lines))
            self.app.log(lines[0])
            self.refresh()
            self.preview()
        self.app.run_bg("收集扫描结果", work, done)

    def sym_result(self):
        from .scanflow import symmetry_check
        p = self._need()
        recs = p.collect(self._sel(), write=False)
        if not any(r.get("set") == "check" for r in recs):
            from .scananalysis import _group_imported_checks
            recs = _group_imported_checks(p, recs)
        rows, ch = symmetry_check(recs)
        if not rows:
            raise ValueError("该扫描没有已完成的对称检验构型")
        lines = [f"{'组':<4}{'z, Δx, Δy, φ':<28}{'变体':>5}{'V_h':>10}{'最大偏差':>10}{'V_e':>10}{'最大偏差':>10}"]
        for r in rows:
            b = r["base"]
            lines.append(f"{r['group']:<4}{b['z']:.2f}, {b['dx']:.2f}, {b['dy']:.2f}, {b['roll']:.1f}".ljust(32) +
                         f"{r['n_ops']:>5}{(r['V_hole'] or 0):10.3f}{(r['maxdev_hole'] or 0):10.3f}"
                         f"{(r['V_electron'] or 0):10.3f}{(r['maxdev_electron'] or 0):10.3f}")
        for c, d in ch.items():
            if d:
                lines.append(f"{T('空穴' if c == 'hole' else '电子')}: B {T('绕自身二重轴的特征标')} " + ", ".join(f"{op} {v:+d}" for op, v in d.items()))
        lines.append("对称等价构型的 V (含符号) 应完全相同：偏差 < 0.1 meV 说明所有二聚体共用同一套片段轨道，符号可比")
        self.show("\n".join(lines))

    def clone(self):
        p = self._need()
        n, lv = self._sel(), self._level()
        if p.scan_info(n).get("level") == lv:
            raise ValueError("请在“当前层级”中选择另一个层级 (先添加)，再换层级重算")
        info = p.clone(n, lv)
        self.app.log(f"已生成 {info['name']}：同一批 {info['n_computed']} 个构型，层级 {lv} (用于泛函 / 基组对照，结果进入报告第 10 节)")
        self.refresh()

    def _gauss_args(self):
        return dict(functional=self.g_func.get().strip(), basis=self.g_basis.get().strip(),
                    stride=max(1, int(self.g_stride.get() or 1)))

    def gauss_cp(self):
        p = self._need()
        out, k = p.gaussian_cp(self._sel(), **self._gauss_args())
        self.app.log(f"已生成 {k} 个 Gaussian 抵消校正输入 ({self.g_func.get()}-D3(BJ)/{self.g_basis.get()})：{out}；"
                     f"算完后报告中选择“Gaussian 抵消校正”作为相互作用能来源")
        self.app.open_runner(str(out))

    def gauss_ti(self):
        p = self._need()
        out, k = p.gaussian_ti(self._sel(), **self._gauss_args())
        self.app.log(f"已生成 {k} 个构型的 Gaussian 转移积分输入 (A 只算一次)：{out}；结果进入报告的层级对照")
        self.app.open_runner(str(out))

    # ---------------- 预览图 ----------------
    def preview(self):
        from .scananalysis import _full_map, _parallel, computed, distance_fit, grid_matrix, grid_zs
        if self.proj is None or not self.tree.selection():
            return
        p = self.proj
        n = self._sel()
        recs = p.collect(n, write=False)
        c = "hole" if self.pv_car.get() == "空穴" else "electron"
        kind = self.PREVIEW.index(self.pv_kind.get())
        zs = grid_zs(recs)
        if kind == 0:
            if zs:
                kind = 1
            elif len({round(r["z"], 3) for r in computed(recs)}) >= 3 and all(abs(r["dx"]) + abs(r["dy"]) < 1e-6 for r in computed(recs)):
                kind = 2
            else:
                kind = 3
        fig = self.plot.fig
        fig.clf()
        ax = fig.add_subplot(111)
        self.plot.ax = ax
        col = "#2f6db3" if c == "hole" else "#c4472f"
        if kind in (1, 4):
            if not zs:
                raise ValueError("该扫描没有完整的 Δx–Δy 网格")
            z = zs[0]
            key = c if kind == 1 else "E_total"
            sub = [dict(r, hole=r.get(key)) for r in recs if r.get(key) is not None]
            xs, ys, M = grid_matrix(sub, z, "hole")
            X, Yv, F = _full_map(xs, ys, M)
            from matplotlib.colors import TwoSlopeNorm
            if kind == 1:
                vm = np.nanmax(np.abs(F)) or 1
                im = ax.pcolormesh(X, Yv, F.T, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vm, vm), shading="nearest")
                if np.isfinite(F).all():
                    ax.contour(X, Yv, F.T, [0], colors="k", linewidths=0.8)
                lab = f"V ({T('空穴') if c == 'hole' else T('电子')}) / meV"
            else:
                im = ax.pcolormesh(X, Yv, F.T, cmap="Blues_r", shading="nearest")
                lab = "E_int / kcal mol⁻¹"
            fig.colorbar(im, ax=ax, shrink=0.8).set_label(lab)
            ax.set_aspect("equal")
            ax.set_xlabel("Δx / Å")
            ax.set_ylabel("Δy / Å")
            ax.set_title(f"{n}, z = {z:.2f} Å" + (f" ({T('另有')} {len(zs) - 1} {T('个 z')})" if len(zs) > 1 else ""))
        elif kind == 2:
            d = distance_fit(recs, c)
            if not d:
                raise ValueError("该扫描没有足够的共面 Δx = Δy = 0 构型 (至少 3 个 z)")
            zz = np.array(d["z"])
            ax.semilogy(zz, np.abs(d["V"]), "o", color=col)
            ax.semilogy(zz, d["A_z0"] * np.exp(-d["beta"] * (zz - d["z0"])), "-", color="0.3")
            ax.set_xlabel("z / Å")
            ax.set_ylabel("|V| / meV")
            ax.set_title(f"β = {d['beta']:.3f} Å⁻¹, R² = {d['R2']:.4f}")
        else:
            rs = [r for r in computed(recs) if r.get(c) is not None and r.get(f"S_{c}") is not None]
            if not rs:
                raise ValueError("该扫描还没有结果")
            S_ = np.array([r[f"S_{c}"] for r in rs])
            V = np.array([r[c] for r in rs])
            ax.plot(S_, V, "o", ms=4, color=col)
            if S_ @ S_ > 0:
                k = V @ S_ / (S_ @ S_)
                xx = np.array([S_.min(), S_.max()])
                ax.plot(xx, k * xx, "-", color="0.3")
                ax.set_title(f"V = kS: k = {k:.0f} meV, r = {np.corrcoef(S_, V)[0, 1]:.4f}" if len(V) > 2 else "")
            ax.set_xlabel("S")
            ax.set_ylabel("V / meV")
        self.plot.draw()

    def save_fig(self):
        f = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if f:
            self.plot.fig.savefig(f, dpi=300, facecolor="white")
            self.app.log(f"已保存 {f}")

    # ---------------- 报告与归档 ----------------
    def report(self):
        from .scananalysis import make_report
        p = self._need()
        lam = None
        if self.lam_h.get().strip() or self.lam_e.get().strip():
            lam = {c: float(v.get()) for c, v in (("hole", self.lam_h), ("electron", self.lam_e)) if v.get().strip()}
        rd = self.reorg_dir.get() or None
        T_, z0, en = float(self.temp.get()), float(self.z0.get()), self.energy.get()
        lv = self.level.get() if self.level.get() in p.meta["levels"] else None
        grid_lv = {p.scan_info(n).get("level") for n in p.scans() if p.meta["scans"][n].get("kind") == "grid"}
        lv = lv if lv in grid_lv else None

        def work():
            return make_report(p, lam=lam, reorg_dir=rd, T=T_, energy=en, z0=z0, tr=T, level=lv)

        def done(res):
            self.show("\n".join(res["summary"]) + f"\n\n{T('报告')}: {res['html']}\n{T('图')}: {res['html'].parent / 'figs'}"
                      f"\n{T('模型')}: {p.root / 'model'}\n{T('验证')}: {p.root / 'validation'}")
            self.app.log(f"报告已生成：{res['html']}")
            try:
                import webbrowser
                webbrowser.open(res["html"].as_uri())
            except Exception:  # noqa: BLE001
                pass
        self.app.run_bg("生成扫描报告", work, done)

    def archive(self):
        p = self._need()
        f = filedialog.asksaveasfilename(defaultextension=".zip", initialdir=str(p.root),
                                         initialfile=f"archive_{p.meta['name']}.zip", filetypes=[("ZIP", "*.zip")])
        if not f:
            return
        inc = self.with_out.get()

        def work():
            return p.archive(f, include_outputs=inc)

        def done(res):
            path, n = res
            self.app.log(f"已归档 {n} 个文件：{path} ({Path(path).stat().st_size / 1e6:.1f} MB)")
        self.app.run_bg("归档扫描项目", work, done)

    # ---------------- CEDS ----------------
    def _temps(self):
        T = [float(x) for x in self.ce_T.get().replace(",", " ").split()]
        if not T:
            raise ValueError("请给出温度")
        return T

    def _ceds(self):
        from .ceds import CEDS, PairPotential
        st = self.app.struct_tab
        if st.crys is None or st.mols is None:
            raise ValueError("请先在“结构/晶体”页打开晶体 (CIF)")
        mono, chars = None, None
        if self.proj is not None and sorted(self.proj.symbols) == sorted(st.mols[0]["symbols"]):
            mono = (self.proj.symbols, self.proj.Y)
            chars = self._dft_characters()
        frame = self.proj.frame if self.proj is not None else "all"
        key = (id(st.crys), float(st.cut.get()), id(self.proj), self.ce_scale)
        if self.ce is None or getattr(self, "_ce_key", None) != key:
            self.ce = CEDS(st.crys, st.mols, cutoff=float(st.cut.get()), monomer=mono, characters=chars,
                           potential=PairPotential(scale=self.ce_scale), frame=frame)
            self._ce_key = key
            for n_ in self.ce.notes:
                self.app.log("[提示] " + n_)
        return self.ce

    def _dft_characters(self):
        """项目中有 DFT 对称检验时，用它给出的特征标 (否则由 EHT 计算)。"""
        from .scanflow import symmetry_check
        p = self.proj
        recs = [r for n in p.scans() if p.meta["scans"][n].get("kind") == "check" for r in p.collect(n, write=False)]
        if not recs:
            return None
        _, ch = symmetry_check(recs)
        if all(ch.get(c) for c in ("hole", "electron")) and all(v != 0 for c in ch for v in ch[c].values()):
            return {c: dict(ch[c], E=1) for c in ch}
        return None

    def _surfaces(self, ce):
        from .ceds import EHTSurface, surfaces_from_project
        src = self.CE_SRC.index(self.ce_src.get())
        if src == 1:
            s = {m: EHTSurface(ce.sym, ce.Y, mirror=m) for m in (False, True)}
            return ({p["gid"]: s[bool(p.get("mirror"))] for p in ce.pairs},
                    {p["gid"]: s[False].describe() for p in ce.pairs})
        if src == 2:
            from .ceds import combined_surfaces, local_surfaces_from_csv, product_surfaces
            g, f, lc = self.ce_gcsv.get(), self.ce_fdx.get(), self.ce_lcsv.get()
            if not g and not lc and self.proj is None:
                raise ValueError("请给出 G 网格表 (乘积模型) 或局域结果表，或打开扫描项目")
            prod = product_surfaces(ce, g, f or None) if g else {}
            loc = local_surfaces_from_csv(ce, lc) if lc else {}
            lv = self.level.get() if self.proj is not None and self.level.get() in self.proj.meta["levels"] else None
            return combined_surfaces(ce, self.proj, lv, prod, self.ce_fallback.get(), loc)
        p = self._need()
        lv = self.level.get() if self.level.get() in p.meta["levels"] else None
        return surfaces_from_project(p, ce, level=lv, eht_fallback=self.ce_fallback.get())

    def _maybe_calibrate(self, ce, T):
        if not self.ce_ueq.get().strip():
            return ""
        n = max(200, min(800, int(self.ce_n.get() or 1500) // 2))
        hist = ce.calibrate(float(self.ce_ueq.get()), T=T, n_samples=n)
        self.ce_scale = ce.pot.scale
        self._ce_key = self._ce_key[:3] + (self.ce_scale,)
        return (f"势能缩放 {hist[0][0]:.3f} → {hist[-1][0]:.3f}，U_eq {hist[0][1]:.4f} → {hist[-1][1]:.4f} Å² "
                f"(目标 {float(self.ce_ueq.get()):.4f})\n")

    def ceds_local(self):
        p = self._need()
        lv = self._level()
        name = self.sname.get().strip()
        T0 = self._temps()[-1]
        ce = self._ceds()
        n = int(self.ce_nloc.get() or 70)

        nmc = max(200, min(600, int(self.ce_n.get() or 1500) // 2))
        use_md = self.ce_mdhalf.get()
        sc = tuple(int(x) for x in self.ce_sc.get().replace(",", " ").split())

        def work():
            self._maybe_calibrate(ce, T0)
            half = ce.suggest_half(T=T0, n_samples=nmc)
            if use_md:
                from .rigidmd import md_half
                md, fr, _ = self._md_frames(ce, (p.symbols, p.Y), sc, T0)
                half = md_half(ce, md, fr, set(half))
            sel = self.ce_pairs.get().replace(",", " ").split()
            if sel:
                want = {int(x) for x in sel}
                half = {g: h for g, h in half.items() if g in want}
            cen = {g: c for g, c in ce.centers().items() if g in half}
            return half, p.add_local(name, lv, cen, n=n, half=half)

        def done(res):
            half, info = res
            lines = [f"已生成局域扫描 {name}：{len(half)} 个晶体二聚体，共 {info['n_computed']} 个构型 (层级 {lv})",
                     f"{'组':<5}" + "".join(f"{k:>9}" for k in ("z", "dx", "dy", "gamma", "pitch", "roll")) + "   (±半宽)"]
            for g, h in half.items():
                lines.append(f"G{g:<4}" + "".join(f"{h[k]:9.2f}" for k in ("z", "dx", "dy", "gamma", "pitch", "roll")))
            lines.append("在 ADFjobs 中运行该扫描的批处理，收集结果后选“扫描项目”曲面运行 CEDS")
            self.show("\n".join(lines))
            self.app.log(lines[0])
            self.refresh()
        self.app.run_bg("生成 CEDS 局域扫描", work, done)

    def ceds_run(self):
        from .ceds import plot_hist, summary_table
        ce = self._ceds()
        surf, why = self._surfaces(ce)
        T0 = self._temps()[-1]
        n = int(self.ce_n.get() or 1500)
        car = "hole" if self.pv_car.get() == "空穴" else "electron"

        def work():
            note = self._maybe_calibrate(ce, T0)
            return note, ce.run(surf, T0, n_samples=n)

        def done(r):
            note, res = r
            self.ce_results = {T0: res}
            self.ce_rows = None
            self._ce_last = "mc"
            ue = [np.mean(v["mc"]["U_eq"]) for v in res.values() if v.get("mc")]
            lines = [f"CEDS，T = {T0:g} K，{n} 个样本/二聚体；特征标 {ce.characters}", note.rstrip()]
            lines += [f"G{g}: {w}" for g, w in why.items() if g in res and "skipped" not in res[g]]
            if ue:
                lines.append(f"刚体涨落的 U_eq (重原子平均) ≈ {np.mean(ue):.4f} Å²：与晶体学 U_eq 对照，必要时填目标 U_eq 校准")
            for c in ("hole", "electron"):
                lines += ["", f"[{T('空穴') if c == 'hole' else T('电子')}]", summary_table(ce, res, c)]
            self.show("\n".join(x for x in lines if x is not None))
            ax = self.plot.new_axes()
            plot_hist(ax, ce, res, car)
            ax.set_title(f"CEDS, T = {T0:g} K")
            self.plot.draw()
        self.app.run_bg("CEDS 采样", work, done)

    def ceds_muT(self):
        from .ceds import plot_mu_T, temperature_series
        ce = self._ceds()
        surf, why = self._surfaces(ce)
        temps = self._temps()
        n = int(self.ce_n.get() or 1500)
        ht = float(self.ce_htau.get()) / 1000

        def work():
            note = self._maybe_calibrate(ce, temps[-1])
            rows = temperature_series(ce, surf, temps, hbar_tau=ht, n_samples=n)
            return note, rows

        def done(r):
            note, rows = r
            self.ce_rows = rows
            self.ce_results = {row["T"]: row["res"] for row in rows}
            self._ce_last = "mc"
            lines = [f"CEDS + TLT (ħ/τ = {ht * 1000:g} meV)", note.rstrip(),
                     f"{'T/K':>6}{'μh':>10}{'μh(σ=0)':>10}{'Lh/Å':>8}{'μe':>10}{'μe(σ=0)':>10}{'Le/Å':>8}"]
            for row in rows:
                lines.append(f"{row['T']:6.0f}{row['mu_hole']:10.3g}{row['mu_static_hole']:10.3g}{row['L_hole']:8.1f}"
                             f"{row['mu_electron']:10.3g}{row['mu_static_electron']:10.3g}{row['L_electron']:8.1f}")
            if len(rows) >= 2:
                for c in ("hole", "electron"):
                    T_ = np.log([row["T"] for row in rows])
                    mu = np.log([row[f"mu_{c}"] for row in rows])
                    lines.append(f"{T('空穴') if c == 'hole' else T('电子')}: μ ∝ T^{np.polyfit(T_, mu, 1)[0]:.2f}")
            self.show("\n".join(x for x in lines if x))
            ax = self.plot.new_axes()
            plot_mu_T(ax, rows)
            self.plot.draw()
        self.app.run_bg("CEDS + TLT 迁移率–温度", work, done)

    def _md_monomer(self, ce):
        p = self.ce_mono.get()
        if p:
            from .cli import _read_monomer
            mono = _read_monomer(p)
            if sorted(mono[0]) != sorted(ce.sym):
                raise ValueError("MD 单体的组成与晶体中的分子不同")
            return mono
        return ce.sym, ce.Y

    def ceds_md(self):
        from .ceds import summary_table
        from .rigidmd import RigidCrystalMD, md_summary, md_temperature_series, plot_md_mu_T
        ce = self._ceds()
        surf, why = self._surfaces(ce)
        if not surf:
            raise ValueError("没有可用的耦合曲面")
        temps = self._temps()
        ht = float(self.ce_htau.get()) / 1000
        sc = tuple(int(x) for x in self.ce_sc.get().replace(",", " ").split())
        if len(sc) != 3 or min(sc) < 1:
            raise ValueError("MD 超胞应为三个正整数，如 4 3 2")
        prod = float(self.ce_mdps.get() or 30)
        mono = self._md_monomer(ce)
        save = str(self.proj.root / "md") if (self.ce_traj.get() and self.proj is not None) else None

        def work():
            md = RigidCrystalMD(ce.crys, ce.frames, mono, sc)
            rows, _ = md_temperature_series(ce, surf, temps, md=md, prod_ps=prod, hbar_tau=ht, save_dir=save)
            return md, rows

        def done(r):
            md, rows = r
            self.ce_md_rows = rows
            self._ce_last = "md"
            lines = [f"全晶体刚体 MD + TLT (ħ/τ = {ht * 1000:g} meV)：超胞 {'×'.join(map(str, sc))} = {md.N} 个分子；"
                     f"{md.ff_note}；晶格能 {md.lattice_energy():.2f} kcal/mol",
                     f"每个温度 5 ps 平衡 + {prod:g} ps 采样；平均结构移回实验结构 (力场只决定涨落)"]
            lines += [f"G{g}: {w}" for g, w in why.items() if g in surf]
            lines += ["", md_summary(rows),
                      "τc：耦合自相关时间 (各二聚体平均)，ħ/τc 可与 TLT 的 ħ/τ 参数对照"]
            for c in ("hole", "electron"):
                lines += ["", f"[{T('空穴') if c == 'hole' else T('电子')}] T = {rows[-1]['T']:g} K",
                          summary_table(ce, rows[-1]["res"], c)]
            if save:
                lines.append(f"轨迹已保存到 {save}")
            self.show("\n".join(lines))
            ax = self.plot.new_axes()
            plot_md_mu_T(ax, rows)
            self.plot.draw()
            self.app.log(f"全晶体 MD 完成：{len(rows)} 个温度")
        self.app.run_bg(f"全晶体 MD ({len(temps)} 个温度)", work, done)

    def _md_frames(self, ce, mono, sc, T):
        """优先使用项目 md 目录中已保存的同温度、同超胞轨迹，否则运行 MD (与“全晶体 MD”相同的设置与随机数种子)。"""
        from .rigidmd import RigidCrystalMD, load_frames, run_temperature
        md = RigidCrystalMD(ce.crys, ce.frames, mono, sc)
        f = self.proj.root / "md" / f"md_{T:g}K.npz"
        if f.exists():
            fr = load_frames(f)
            if fr["c"].shape[1] == md.N:
                return md, fr, str(f)
        return md, run_temperature(md, T, dict(prod_ps=float(self.ce_mdps.get() or 30))), "md"

    def ceds_snapshots(self):
        from .rigidmd import snapshot_configs
        p = self._need()
        lv = self._level()
        name = self.sname.get().strip()
        if not name:
            raise ValueError("请在“新建扫描”中填写扫描名称")
        ce = self._ceds()
        T0 = self._temps()[-1]
        sc = tuple(int(x) for x in self.ce_sc.get().replace(",", " ").split())
        n = int(self.ce_nloc.get() or 120)
        sel = self.ce_pairs.get().replace(",", " ").split()
        gids = {int(x) for x in sel} if sel else {q["gid"] for q in ce.pairs if q["r_centroid"] < 7.0}
        mono = (p.symbols, p.Y)

        def work():
            md, fr, src = self._md_frames(ce, mono, sc, T0)
            cfgs = snapshot_configs(ce, md, fr, gids, n_per_pair=n)
            return src, p.add_md_snapshots(name, lv, cfgs, info={"T": T0, "n_per_pair": n, "pairs": sorted(gids),
                                                                  "supercell": list(sc), "trajectory": src})

        def done(r):
            src, info = r
            lines = [f"MD 快照 {name}：{len(gids)} 个二聚体 × {n} 个构型 = {info['n_computed']} 个 ADF 任务 (层级 {lv}，T = {T0:g} K)",
                     f"轨迹：{'新运行的 MD' if src == 'md' else src}",
                     f"批处理：{', '.join(info['batches'])}",
                     "在 ADFjobs 中运行这些批处理，收集结果后选中该扫描，点“快照检验”与耦合曲面比较"]
            self.show("\n".join(lines))
            self.app.log(lines[0])
            self.refresh()
        self.app.run_bg("MD 快照", work, done)

    def ceds_validate(self):
        from .rigidmd import plot_validation, snapshot_validation, validation_table
        p = self._need()
        name = self._sel()
        if p.meta["scans"][name].get("kind") != "md":
            raise ValueError("请在右侧表格中选择一个 MD 快照扫描")
        ce = self._ceds()
        surf, why = self._surfaces(ce)
        recs = [r for r in p.collect(name) if r.get("state") == "done" and not r.get("rep")]
        if not recs:
            raise ValueError("该扫描还没有算完的构型")
        val = snapshot_validation(ce, recs, surf)
        if not val:
            raise ValueError("没有同时具有快照数据和耦合曲面的二聚体")
        self.ce_validation = val
        self._ce_last = "validation"
        lines = [f"MD 快照检验 {name}：{len(recs)} 个构型；σ 比值 = σ(曲面) / σ(直接计算)，区间为自助法 95% 置信区间", ""]
        lines += [f"G{g}: {why.get(g, '')}" for g in val]
        lines += ["", validation_table(val)]
        self.show("\n".join(lines))
        fig = self.plot.fig
        fig.clf()
        axes = fig.subplots(1, 2)
        plot_validation(axes, val)
        self.plot.draw()

    def ceds_export(self):
        """导出最近一次的结果：快照检验、全晶体 MD 或局域 MC。"""
        from .ceds import export_csv
        last = getattr(self, "_ce_last", None)
        if last is None:
            raise ValueError("请先运行 CEDS")
        name = {"validation": "snapshot_validation.csv", "md": "ceds_md.csv"}.get(last, "ceds_results.csv")
        f = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=name, filetypes=[("CSV", "*.csv")])
        if not f:
            return
        if last == "validation":
            from .rigidmd import export_validation_csv
            export_validation_csv(self.ce_validation, f)
        elif last == "md":
            from .rigidmd import export_md_csv
            export_md_csv(self.ce, self.ce_md_rows, f)
        else:
            export_csv(self.ce, self.ce_results, f, self.ce_rows)
        self.app.log(f"已导出 {f}")


class CitationDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        from .citations import text as cite_text
        self.title(T("引用文献"))
        self.geometry("900x600")
        try:
            App.set_icon(self)
        except Exception:  # noqa: BLE001
            pass
        ttk.Label(self, text="发表使用本软件得到的结果时，请引用本软件及所用功能对应的方法文献：", padding=8).pack(anchor="w")
        self.txt = styled_text(self, app.mono, height=28, wrap="word", size=9)
        self.txt.pack(fill="both", expand=True, padx=8)
        self.content = cite_text(T)
        self.txt.insert("end", self.content)
        ttk.Button(self, text="复制全部", command=self.copy).pack(anchor="e", padx=8, pady=6)

    def copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.content)


class ProgramsDialog(tk.Toplevel):
    """设置 → 程序路径：Gaussian、ADF/AMS、xtb 的安装位置，可自动检测。"""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title(T("程序路径设置"))
        self.geometry("760x420")
        self.configure(background=PALETTE["bg"])
        S = safe(app)
        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)
        cfg = app.config_data
        rows = [("gaussian_exe", "Gaussian 程序 (g16.exe / g09.exe)", "open"),
                ("scratch", "Gaussian 临时目录", "dir"),
                ("ams_home", "ADF / AMS 安装目录 (如 C:\\AMS2024.104)", "dir")]
        self.entries, self.status = {}, {}
        for r, (key, lab, mode) in enumerate(rows):
            ttk.Label(body, text=lab).grid(row=2 * r, column=0, columnspan=3, sticky="w", pady=(8, 0))
            e = FileEntry(body, "", mode=mode)
            e.grid(row=2 * r + 1, column=0, sticky="we")
            e.set(cfg.get(key, ""))
            ttk.Button(body, text="自动检测", command=S(lambda k=key: self.detect(k))).grid(row=2 * r + 1, column=1, padx=4)
            st = ttk.Label(body, text="", foreground=PALETTE["muted"])
            st.grid(row=2 * r + 1, column=2, sticky="w")
            self.entries[key], self.status[key] = e, st
        body.columnconfigure(0, weight=1)
        nf = ttk.Frame(body)
        nf.grid(row=8, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.nscm = tk.StringVar(value=str(cfg.get("adf_nscm", "")))
        ttk.Label(nf, text="ADF 并行核数 (NSCM，空 = ADF 默认)").pack(side="left")
        ttk.Entry(nf, textvariable=self.nscm, width=6).pack(side="left", padx=6)
        ttk.Label(nf, text="大分子 (如红荧烯二聚体 140 原子) 建议设为 CPU 物理核数", foreground=PALETTE["muted"]).pack(side="left")
        ttk.Label(body, text="ADF 任务由 AMS 自带的 bash (或 Git-Bash / 系统 sh) 执行，并自动设置 AMSHOME、AMSBIN、"
                             "AMSRESOURCES、SCMLICENSE；路径中避免中文。", foreground=PALETTE["muted"], wraplength=700
                  ).grid(row=9, column=0, columnspan=3, sticky="w", pady=8)
        bb = ttk.Frame(body)
        bb.grid(row=10, column=0, columnspan=3, sticky="e")
        ttk.Button(bb, text="全部检测", command=S(self.detect_all)).pack(side="left", padx=4)
        ttk.Button(bb, text="保存", style="Accent.TButton", command=S(self.save)).pack(side="left")
        self.detect_all(quiet=True)

    def detect(self, key, quiet=False):
        from .programs import find_ams, find_xtb
        e, st = self.entries[key], self.status[key]
        if key == "gaussian_exe":
            info = find_gaussian(e.get() or None)
            if info:
                e.set(info["exe"])
            st.config(text=T("已找到") if info else T("未找到"))
        elif key == "ams_home":
            info = find_ams(e.get() or None)
            if info:
                e.set(info["home"])
                st.config(text=T("已找到") + f" ({info['kind']}, shell: {'OK' if info['shell'] else T('无')})")
            else:
                st.config(text=T("未找到"))
        elif key == "xtb_exe":
            x = find_xtb(e.get() or None)
            if x:
                e.set(x)
            st.config(text=T("已找到") if x else T("未找到"))

    def detect_all(self, quiet=False):
        for k in ("gaussian_exe", "ams_home"):
            self.detect(k, quiet)

    def save(self):
        for k, e in self.entries.items():
            self.app.config_data[k] = e.get()
        n = self.nscm.get().strip()
        if n and not n.isdigit():
            raise ValueError("ADF 并行核数应为正整数")
        self.app.config_data["adf_nscm"] = n
        save_config(self.app.config_data)
        self.app.log("程序路径已保存")
        self.destroy()


# ==============================================================================================
# 运行高斯任务窗口
# ==============================================================================================
class RunDialog(tk.Toplevel):
    def __init__(self, app, directory=""):
        super().__init__(app)
        self.app = app
        self.title(T("运行计算任务 (Gaussian / ADF)"))
        try:
            self.transient(app)
            self.lift()
            App.set_icon(self)
        except Exception:  # noqa: BLE001
            pass
        self.geometry("900x620")
        self.configure(background=PALETTE["bg"])
        self.queue, self.events = None, []
        S = safe(app)
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        cfg = app.config_data
        g = ttk.LabelFrame(body, text="高斯程序")
        g.pack(fill="x")
        self.exe = FileEntry(g, "g16 / g09 程序", filetypes=[("程序", "*.exe g16 g09"), ("所有文件", "*.*")])
        self.exe.pack(fill="x")
        self.exe.set(cfg.get("gaussian_exe", ""))
        self.scr = FileEntry(g, "临时目录", mode="dir")
        self.scr.pack(fill="x")
        self.scr.set(cfg.get("scratch", ""))
        gb = ttk.Frame(g)
        gb.pack(fill="x")
        ttk.Button(gb, text="自动检测", command=S(self.detect)).pack(side="left")
        self.info = ttk.Label(gb, text="", foreground=PALETTE["muted"])
        self.info.pack(side="left", padx=8)
        af = ttk.LabelFrame(body, text="ADF / AMS (含 ADF2016 等旧版)")
        af.pack(fill="x", pady=(6, 0))
        self.ams = FileEntry(af, "安装目录", mode="dir")
        self.ams.pack(fill="x")
        self.ams.set(cfg.get("ams_home", ""))
        ab = ttk.Frame(af)
        ab.pack(fill="x")
        ttk.Button(ab, text="自动检测", command=S(self.detect_ams)).pack(side="left")
        ttk.Button(ab, text="选择 adf 程序文件…", command=S(self.pick_adf_exe)).pack(side="left", padx=4)
        ttk.Button(ab, text="测试 ADF 环境", command=S(self.test_ams)).pack(side="left", padx=4)
        self.ams_info = ttk.Label(ab, text="", foreground=PALETTE["muted"])
        self.ams_info.pack(side="left", padx=8)
        d = ttk.LabelFrame(body, text="任务")
        d.pack(fill="both", expand=True, pady=6)
        self.dir = FileEntry(d, "任务目录", mode="dir", command=S(lambda p: self.load()))
        self.dir.pack(fill="x")
        cols = ("输入文件", "类型", "状态", "信息")
        self.tree = ttk.Treeview(d, columns=cols, show="headings", height=10, selectmode="extended")
        for c, w in zip(cols, (230, 80, 90, 420)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True)
        bb = ttk.Frame(d)
        bb.pack(fill="x", pady=4)
        ttk.Button(bb, text="刷新", command=S(self.load)).pack(side="left")
        ttk.Button(bb, text="全选", command=S(lambda: self.tree.selection_set(self.tree.get_children()))).pack(side="left")
        ttk.Button(bb, text="▶ 运行选中任务", style="Accent.TButton", command=S(self.run)).pack(side="left", padx=6)
        ttk.Button(bb, text="■ 停止", command=S(self.stop)).pack(side="left")
        ttk.Button(bb, text="打开任务文件夹", command=S(self.open_folder)).pack(side="left", padx=6)
        self.watch_btn = ttk.Button(bb, text="监视 ADFjobs 结果", command=S(self.toggle_watch))
        self.watch_btn.pack(side="left")
        self.status_lbl = ttk.Label(bb, text="", foreground=PALETTE["muted"])
        self.status_lbl.pack(side="left", padx=6)
        ttk.Label(bb, text="ADF2016 等旧版也可在 ADFjobs 中打开 batch_adf.run 一次算完全部二聚体", foreground=PALETTE["muted"]
                  ).pack(side="left", padx=6)
        self.log = styled_text(body, app.mono, dark=True, height=8, size=9)
        self.log.pack(fill="x")
        if directory:
            self.dir.set(directory)
            self.load()
        self.detect(quiet=True)
        self.detect_ams(quiet=True)
        self._after = self.after(400, self.poll)

    def detect(self, quiet=False):
        info = find_gaussian(self.exe.get() or None)
        if info:
            self.exe.set(info["exe"])
            self.info.config(text=f"formchk: {info['formchk'] or '未找到（将不生成 fchk）'}")
            self.app.config_data["gaussian_exe"] = info["exe"]
            save_config(self.app.config_data)
        elif not quiet:
            raise FileNotFoundError("没有找到高斯程序，请手动选择 g16.exe（例如 C:\\\\G16W\\\\g16.exe）")
        return info

    def batch_progress(self, force=False):
        """运行 ADF 批量脚本时，每 5 秒统计已生成的结果文件，显示“已完成 x / y 个二聚体”。"""
        import time as _t
        g = getattr(self, "_running", None)
        if not g or not str(g).endswith(".sh"):
            return None
        if not force and _t.time() - getattr(self, "_last_prog", 0) < 5:
            return None
        self._last_prog = _t.time()
        from .programs import adf_result_files
        res = adf_result_files(g)
        done = sum(1 for r in res if r.exists())
        txt = f"{T('进度')}: {done} / {len(res)} {T('个二聚体已完成')}"
        if self.tree.exists(g):
            vals = list(self.tree.item(g, "values"))
            vals[3] = txt
            self.tree.item(g, values=vals)
        try:
            self.status_lbl.config(text=txt)
        except Exception:  # noqa: BLE001
            pass
        return done, len(res)

    def auto_read(self, results):
        """任务结束后自动读取结果：二聚体目录 (含 dimers.json) 填入迁移率表；单个 ADF 任务显示转移积分。"""
        from .adf import read_adf_transfer_integrals
        from .programs import adf_result_files, find_ams
        info = find_ams(self.ams.get() or self.app.config_data.get("ams_home") or None)
        dirs = []
        for g, ok, _ in results:
            p = Path(g)
            if (p.parent / "scan.json").exists() or p.name == "run_mono_adf.sh":
                if ok:
                    self.write(f"{p.name}: " + T("扫描批处理已结束；在“批量扫描工作流”页点“收集结果”读取"))
                continue
            if (p.parent / "dimers.json").exists():
                if ok and p.parent not in dirs:
                    dirs.append(p.parent)
            elif ok and p.suffix == ".sh":
                for f in adf_result_files(p):
                    try:
                        r = read_adf_transfer_integrals(f, info)
                        self.write(f"{f.name}: Vh = {r['hole']['V_meV']:.2f} meV, Ve = {r['electron']['V_meV']:.2f} meV")
                        self.app.data["last_J"] = {"hole": r["hole"]["V_meV"], "electron": r["electron"]["V_meV"]}
                    except Exception as exc:  # noqa: BLE001
                        self.write(f"{f.name}: 读取失败 {exc}")
        cp_dirs = {Path(g).parent for g, ok, _ in results if ok and Path(g).name.endswith("_cp.gjf")}
        for d in cp_dirs:
            try:
                out = self.app.inter_tab.read_cp(str(d))
                self.write(f"已自动读取 {len(out)} 组抵消校正相互作用能 (“分子间相互作用能”页)")
            except Exception as exc:  # noqa: BLE001
                self.write(f"读取抵消校正结果失败: {exc}")
        dirs = [d for d in dirs if d not in cp_dirs]
        for d in dirs:
            prog = MobilityTab.detect_program(d)
            if not prog:
                continue
            out = self.app.mob_tab.import_dir(str(d), prog)
            if out:
                coup, err = out
                self.write(f"已自动读取 {d.name}: {len(coup)} 组转移积分已填入“迁移率”页" +
                           (f"；失败 {sorted(err)}" if err else ""))
                for gid, c in sorted(coup.items()):
                    self.write(f"  G{gid}: Vh = {c.get('hole') or 0:.2f} meV, Ve = {c.get('electron') or 0:.2f} meV")

    def toggle_watch(self):
        """监视任务目录：在 ADFjobs 中运行批量脚本时，每 10 秒检查进度，全部完成后自动读取结果。"""
        self._watching = not getattr(self, "_watching", False)
        self.watch_btn.config(text=T("停止监视") if self._watching else T("监视 ADFjobs 结果"))
        if self._watching:
            self.write("开始监视任务目录 (每 10 秒检查一次)，ADFjobs 中的计算完成后自动读取结果")
            self.watch_step()

    def watch_step(self):
        from .programs import adf_result_files
        d = self.dir.get()
        if not d or not getattr(self, "_watching", False):
            return None
        scripts = [p for p in list(Path(d).glob("batch_*.sh")) + list(Path(d).glob("run_*_adf.sh"))
                   if p.name != "run_all_adf.sh"]
        res = [r for s_ in scripts for r in adf_result_files(s_)]
        done = sum(1 for r in res if r.exists())
        self.status_lbl.config(text=T("监视中") + f": {done} / {len(res)}")
        if res and done == len(res):
            self._watching = False
            self.watch_btn.config(text=T("监视 ADFjobs 结果"))
            self.write(f"ADFjobs 计算全部完成 ({done} 个二聚体)，自动读取结果")
            self.auto_read([(str(s_), True, "") for s_ in scripts])
            self.load()
            return True
        try:
            self.after(10000, self.watch_step)
        except Exception:  # noqa: BLE001
            pass
        return False

    def detect_ams(self, quiet=False):
        from .programs import find_ams
        info = find_ams(self.ams.get() or None)
        if info:
            self.ams.set(info["home"])
            self.ams_info.config(text=f"ADF {info['version']} ({'AMS' if info['kind'] == 'ams' else T('旧版')}), "
                                      f"shell: {'OK' if info['shell'] else T('未找到')}")
            self.app.config_data["ams_home"] = info["home"]
            save_config(self.app.config_data)
        elif not quiet:
            from .programs import explain_ams
            raise FileNotFoundError(explain_ams(self.ams.get()))
        return info

    def pick_adf_exe(self):
        """直接选择安装目录 bin 中的 adf / ams 程序 (adf.exe、adf、ams.exe …)。"""
        try:
            p = filedialog.askopenfilename(parent=self, title=T("选择 ADF 安装目录 bin 中的 adf 或 ams 程序"),
                                           filetypes=[("ADF / AMS", "adf* ams*"), ("所有文件", "*.*")])
        except Exception:  # noqa: BLE001
            p = None
        if p:
            self.ams.set(p)
            self.detect_ams()

    def open_folder(self):
        import subprocess as _sp
        d = self.dir.get()
        if not d:
            raise ValueError("请先选择任务目录")
        if sys.platform.startswith("win"):
            os.startfile(d)  # noqa: S606
        else:
            _sp.Popen(["xdg-open", d])

    def test_ams(self):
        from .programs import test_adf_environment
        info = self.detect_ams()
        wd = Path(self.dir.get() or Path.home()) / "adf_env_test"
        self.write(f"正在用 H2 单点测试 ADF {info['version']} (目录 {wd}) …")
        ok, tail, dt = test_adf_environment(info, wd)
        self.write(("ADF 环境正常" if ok else "ADF 环境测试失败") + f" ({dt:.1f} s)\n{tail}")
        if not ok:
            raise RuntimeError("ADF 环境测试失败。运行窗口中已列出 shell、ADFBIN、许可证和 ADF 的输出；"
                               "请把这些内容发给开发者，或先在 ADFjobs 中运行 adf_env_test.run 对照")

    def load(self):
        d = self.dir.get()
        if not d:
            return
        self.tree.delete(*self.tree.get_children())
        from .jobs import order_jobs, output_name
        adf = [p for p in list(Path(d).glob("run_*_adf.sh")) + list(Path(d).glob("batch_*.sh"))
               if p.name != "run_all_adf.sh"]
        for g in order_jobs(list(Path(d).glob("*.gjf")) + adf):
            if g.suffix == ".sh":
                from .programs import adf_result_files
                res_ = adf_result_files(g)
                nd = sum(1 for x in res_ if x.exists())
                st = "已有输出" if res_ and nd == len(res_) else "待运行"
            else:
                out = output_name(g)
                st = "已有输出" if out.exists() and "Normal termination" in out.read_text(errors="ignore")[-3000:] \
                    else "待运行"
            info_ = ""
            if g.suffix == ".sh":
                info_ = (f"{nd} / {len(res_)} 个二聚体已完成" if g.name.startswith("batch_") else "") + \
                        ("；也可在 ADFjobs 中运行 " + g.with_suffix(".run").name if g.with_suffix(".run").exists() else "")
            self.tree.insert("", "end", iid=str(g), values=(g.name, "ADF" if g.suffix == ".sh" else "Gaussian", st, info_))
        self.tree.selection_set([i for i in self.tree.get_children() if self.tree.item(i, "values")[2] == T("待运行")])
        probs = path_problems(d)
        self.write("\n".join(f"[警告] {p}" for p in probs) if probs else f"已载入 {len(self.tree.get_children())} 个输入文件")

    def write(self, msg):
        self.log.insert("end", str(msg).rstrip() + "\n")
        self.log.see("end")

    def run(self):
        from .programs import find_ams
        jobs = list(self.tree.selection())
        if not jobs:
            raise ValueError("请选择要运行的任务")
        from .gaussian_fcht import diagnose_fcht
        produced = [c for j in jobs if j.endswith(".gjf") and "fcht" not in Path(j).read_text(errors="ignore").lower()
                    for c in re.findall(r"^%chk=(\S+)", Path(j).read_text(errors="ignore"), re.M | re.I)]
        bad = {}
        for j in jobs:
            if j.endswith(".gjf") and "fcht" in Path(j).read_text(errors="ignore").lower():
                pr = diagnose_fcht(j, produced)
                if pr:
                    bad[Path(j).name] = pr
        if bad:
            raise ValueError("以下 FCHT 任务所需的 chk 有问题，未开始运行：\n" + "\n".join(
                f"{n}:\n" + "\n".join(f"  · {p}" for p in pr) for n, pr in bad.items()) +
                "\n\n可先运行重组能任务 (opt freq) 生成 chk，或在“分解/黄昆因子”页改用“内置”分解 (读取 fchk)")
        need_g = any(j.endswith(".gjf") for j in jobs)
        info = self.detect(quiet=not need_g) if need_g else find_gaussian(self.exe.get() or None)
        ams = find_ams(self.ams.get() or self.app.config_data.get("ams_home") or None)
        if ams:
            self.app.config_data["ams_home"] = ams["home"]
            if str(self.app.config_data.get("adf_nscm", "")).strip():
                ams["nscm"] = str(self.app.config_data["adf_nscm"]).strip()
        if any(j.endswith(".sh") for j in jobs) and not ams:
            raise FileNotFoundError("没有找到 ADF/AMS 安装目录，请在 工具 → 程序路径设置 中指定")
        probs = path_problems(self.dir.get())
        if probs and not messagebox.askyesno("路径警告", "\n".join(probs) + "\n\nWindows 版高斯很可能因此失败，仍要继续吗？"):
            return
        scratch = self.scr.get() or str(Path(self.dir.get()) / "scratch")
        self.app.config_data["scratch"] = self.scr.get()
        save_config(self.app.config_data)
        self.queue = JobQueue(info, jobs, scratch, lambda g, st, m: self.events.append((g, st, m)),
                              lambda r: self.events.append((None, "done", r)), ams_info=ams)
        self.queue.start()
        self.write(f"开始运行 {len(jobs)} 个任务，程序 {(info or {}).get('exe', '—')}"
                   f"{'，ADF: ' + ams['home'] if ams else ''}，临时目录 {scratch}")

    def stop(self):
        if self.queue:
            self.queue.stop()
            self.write("已请求停止（当前任务将被终止）")

    def poll(self):
        while self.events:
            g, st, m = self.events.pop(0)
            if g is None:
                ok = sum(1 for _, o, _ in m if o)
                self.write(f"全部结束：成功 {ok} / {len(m)}")
                self.app.log(f"计算任务完成：成功 {ok} / {len(m)}")
                self.auto_read(m)
                continue
            if self.tree.exists(g):
                vals = list(self.tree.item(g, "values"))
                vals[2], vals[3] = st, m.splitlines()[0][:120] if m else ""
                self.tree.item(g, values=vals)
            self._running = g if st == "运行中" else (None if getattr(self, "_running", None) == g else
                                                      getattr(self, "_running", None))
            if m and st == "失败":
                self.write(f"{Path(g).name} 失败:\n{m}")
                if "fcht" in Path(g).name.lower():
                    from .gaussian_fcht import explain_fcht_failure
                    ex = explain_fcht_failure(Path(g).with_suffix(".log"))
                    if ex:
                        self.write(ex)
        self.batch_progress()
        try:
            self._after = self.after(400, self.poll)
        except Exception:  # noqa: BLE001
            pass

    def destroy(self):
        try:
            self.after_cancel(self._after)
        except Exception:  # noqa: BLE001
            pass
        super().destroy()


# ==============================================================================================
# 主窗口
# ==============================================================================================
class FunctionWindow(tk.Toplevel):
    """功能窗口：由菜单打开，窗口内左侧为分区标签页，右侧为图与结果；关闭时只隐藏，保留已填的设置与结果。"""

    def __init__(self, app, name):
        super().__init__(app)
        self.app, self.name = app, name
        self.title(f"{T(name)} — {APP_NAME}")
        App.set_icon(self)
        try:
            from .theme import px
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            w, h = min(px(1320), int(sw * 0.92)), min(px(840), int(sh * 0.88))
            self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
            self.configure(background=PALETTE["bg"])
            self.transient(app)
        except Exception:  # noqa: BLE001
            pass
        head = ttk.Frame(self, style="PageHead.TFrame", padding=(14, 8, 14, 2))
        head.pack(fill="x")
        grp = next((g for g, pages in App.NAV if name in pages), "")
        ttk.Label(head, text=f"{T(grp)}  ›  {T(name)}" if grp else name, style="PageCrumb.TLabel").pack(anchor="w")
        self.title_lbl = ttk.Label(head, text=name, style="PageTitle.TLabel")
        self.title_lbl.pack(anchor="w")
        ttk.Label(head, text=App.PAGE_INFO.get(name, ""), style="PageDesc.TLabel").pack(anchor="w")
        self.body = ttk.Frame(self, style="Bg.TFrame")
        self.body.pack(fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self.bind("<Escape>", lambda e: self.withdraw())

    def set_subtitle(self, text):
        self.title_lbl.configure(text=text)
        self.title(f"{T(text)} — {APP_NAME}")


def section_panel(tab):
    """功能页中的分区面板 (SectionPanel)。"""
    from .theme import SectionPanel as _SP
    stack = [tab]
    while stack:
        w = stack.pop()
        if isinstance(w, _SP):
            return w
        try:
            stack += list(w.winfo_children())
        except Exception:  # noqa: BLE001
            pass
    return None


class App(tk.Tk):
    STEPS = ["结构/晶体", "重组能", "IP/EA", "分解/黄昆因子", "激发态/光谱", "转移积分", "迁移率", "输运理论与光物理扩展",
             "分子间相互作用能", "耦合代理模型与堆积设计", "波函数分析", "批量扫描工作流", "极化位点能", "无定形形貌 (MD)",
             "迁移率方法对比"]
    NAV = [("结构", ["结构/晶体", "分子间相互作用能"]),
           ("单分子性质", ["重组能", "IP/EA", "分解/黄昆因子", "激发态/光谱", "波函数分析"]),
           ("电子耦合", ["转移积分", "批量扫描工作流", "耦合代理模型与堆积设计"]),
           ("迁移率", ["迁移率", "极化位点能", "无定形形貌 (MD)", "输运理论与光物理扩展", "迁移率方法对比"])]
    PAGE_INFO = {
        "结构/晶体": "打开分子或晶体 (CIF)，搜索近邻二聚体，软件内 EHT 计算全部二聚体的转移积分，原子 / 基团取代",
        "分子间相互作用能": "二聚体结合能 (UFF / Gaussian 抵消校正)、能量框架与晶格能",
        "重组能": "四点法重组能：生成 Gaussian 输入并解析结果",
        "IP/EA": "电离势、电子亲和势、空穴 / 电子抽取势与前线轨道",
        "分解/黄昆因子": "重组能按简正模式、内坐标与原子分解；黄昆因子与 Duschinsky 矩阵",
        "激发态/光谱": "S0 / S1 / T1 优化与 TD-DFT：吸收与发射光谱、Stokes 位移、辐射与非辐射速率",
        "波函数分析": "轨道、电荷与键级、静电势表面、NBO、IGM / IGMH 与 AIM 拓扑",
        "转移积分": "Gaussian 投影法、ADF 片段法与软件内 EHT；多能级矩阵、原子对分析与三种方法对比",
        "批量扫描工作流": "二聚体参数网格的 ADF 批量计算与分析；CEDS 耦合曲面、局域 MC、全晶体 MD 与快照检验",
        "耦合代理模型与堆积设计": "由滑移扫描建立耦合代理模型，在不碰撞约束下优化分子堆积",
        "迁移率": "Marcus 跳跃迁移率：标量公式、迁移率张量、角分辨迁移率与 KMC (Marcus / Jortner / Miller–Abrahams)",
        "输运理论与光物理扩展": "瞬态局域化理论、紧束缚能带与有效质量、动态无序、振动分辨光谱与 π 堆积几何",
        "极化位点能": "原子电荷 + Thole 诱导偶极：固态极化能 P±、固态 IE / EA、独立分子的位点能差与 MD 热涨落 σ_E",
        "无定形形貌 (MD)": "GROMACS 形貌 → 分子映射 → 近邻 → 转移积分 (EHT / Gaussian) → 位点能 → 周期盒子 KMC (同 VOTCA-CTP)",
        "迁移率方法对比": "同一套 J、λ、σ 输入下比较全部迁移率模型，并按输运区间判断各模型的适用性",
    }

    # 功能页 → (属性名, 类)；除“结构/晶体”外都在首次使用时创建 (启动更快)
    TAB_CLASSES = {}

    def __init__(self):
        super().__init__()
        self.config_data = load_config()
        global _APP
        _APP = self
        i18n.set_lang(self.config_data.get("lang", "zh"))
        from . import theme as _theme
        _theme.TRANSLATOR = T
        self.restart = False
        _theme.set_theme(self.config_data.get("theme", "专业浅色"))
        self.title(f"{APP_NAME} · {T(APP_CN)}  v{__version__}")
        self.set_icon(self)
        self.ui, self.mono = apply_theme(self, float(self.config_data.get("font_scale", 1.0)))
        try:
            from .theme import px
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"{min(px(1480), int(sw * 0.94))}x{min(px(920), int(sh * 0.9))}")
        except Exception:  # noqa: BLE001
            self.geometry("1440x900")
        self.minsize(900, 600)
        self.data = {}
        self.current_structure = None
        self._tabs, self._wins = {}, {}
        self.page_var = MappedVar(self.STEPS[0], {p: T(p) for p in self.STEPS})

        sbar = ttk.Frame(self, style="Card.TFrame")
        sbar.pack(side="bottom", fill="x")
        self.progress = _ttk.Progressbar(sbar, mode="indeterminate", length=160)
        self.progress.pack(side="right", padx=8, pady=2)
        self.status = ttk.Label(sbar, text="就绪", style="Status.TLabel", anchor="w")
        self.status.pack(side="left", fill="x", expand=True)
        self._busy = None
        self.toolbar = ttk.Frame(self, style="Toolbar.TFrame", padding=(4, 3))
        self.toolbar.pack(fill="x")
        self.struct_bar = ttk.Label(self, text=T("未加载结构 —— 文件 → 打开结构"), style="StructBar.TLabel", anchor="w")
        self.struct_bar.pack(fill="x")
        pw = ttk.PanedWindow(self, orient="vertical")
        pw.pack(fill="both", expand=True)
        self.struct_tab = StructureTab(pw, self)
        self._tabs["结构/晶体"] = self.struct_tab
        self.logbox = styled_text(pw, self.mono, dark=True, height=5, size=9)
        pw.add(self.struct_tab, weight=8)
        pw.add(self.logbox, weight=1)
        try:                                            # 结构面板顶部的“打开文件 / 说明”已由工具栏与标题栏代替
            self.struct_tab.panel.header.pack_forget()
            self.status.configure(text="  " + T("鼠标: 左/右键拖动旋转 · 滚轮缩放 · 中键平移 · 左键单击选分子 (Ctrl/Shift 多选)"))
        except Exception:  # noqa: BLE001
            pass
        self._build_toolbar()
        self.build_menu()
        if self.config_data.get("nav_hidden"):
            self.toggle_nav(False)
        self.after_idle(self._reflow_panels)
        self.log("欢迎使用。文件 → 打开结构；计算从 Gaussian / ADF / 迁移率 菜单开始，结果在 分析 菜单中读取；"
                 "程序位置在 任务 → 程序路径设置 中指定")

    # ---------------- 功能窗口 ----------------
    def _get_tab(self, name):
        tab = self._tabs.get(name)
        if tab is not None:
            return tab
        attr, cls = self.TAB_CLASSES[name]
        win = FunctionWindow(self, name)
        tab = cls(win.body, self)
        tab.pack(fill="both", expand=True)
        self._tabs[name], self._wins[name] = tab, win
        try:
            win.withdraw()
            sp = section_panel(tab)
            if sp is not None:
                self.after_idle(sp.reflow)
        except Exception:  # noqa: BLE001
            pass
        return tab

    @property
    def tabs(self):
        return [self._get_tab(n) for n in self.STEPS]

    def show_page(self, name, section=None, subtitle=None):
        """打开功能窗口 (或主窗口的结构面板)，可选切换到某个分区标签页。"""
        tab = self._get_tab(name)
        self.page_var.set(name)
        if name == "结构/晶体":
            self.toggle_nav(True)
            sp = self.struct_tab.panel
            win = None
        else:
            win = self._wins[name]
            sp = section_panel(tab)
            try:
                win.set_subtitle(subtitle or name)
                win.deiconify()
                win.lift()
                win.focus_set()
            except Exception:  # noqa: BLE001
                pass
        if section is not None and sp is not None:
            sp.select(section)
        return tab

    def open_ti(self, prog):
        """转移积分窗口，按菜单预选程序 (gaussian / adf / eht / zindo)。"""
        names = {"gaussian": "转移积分 — Gaussian 二聚体投影法", "adf": "转移积分 — ADF 片段法",
                 "eht": "转移积分 — 软件内 EHT", "zindo": "转移积分 — 软件内 ZINDO/MOO"}
        tab = self.show_page("转移积分", subtitle=names.get(prog))
        tab.prog.set(prog)
        return tab

    def open_run_gaussian(self, calc=None):
        from .gui_run import RunGaussianDialog
        return RunGaussianDialog.open(self, calc)

    def open_flow(self):
        from .gui_flow import FlowWindow
        return FlowWindow.open(self)

    def open_run_adf(self, calc=None):
        from .gui_run import RunADFDialog
        return RunADFDialog.open(self, calc)

    def set_structure_title(self, text):
        self.struct_bar.configure(text=text)
        self.title(f"{APP_NAME} · {T(APP_CN)}  v{__version__}  —  {text.split('  ·')[0]}")

    def _reflow_panels(self):
        """分区面板：过宽的横排控件折行，面板宽度随内容调整 (英文界面 / 高分屏)。"""
        from .theme import _PANELS
        for ref in list(_PANELS):
            p = ref()
            if p is not None:
                p.reflow()

    def toggle_nav(self, show=None):
        """显示 / 隐藏主窗口右侧的结构面板 (Ctrl+B)；状态记入配置，下次启动保持。"""
        panel = self.struct_tab.panel
        visible = bool(panel.winfo_manager())
        show = (not visible) if show is None else show
        if show and not visible:
            panel.pack(side="right", fill="y", padx=(0, 6), pady=6, before=self.struct_tab.plot)
        elif not show and visible:
            panel.pack_forget()
        self.config_data["nav_hidden"] = not show
        save_config(self.config_data)
        try:
            self._panel_var.set(bool(show))
        except Exception:  # noqa: BLE001
            pass

    def _build_toolbar(self):
        st, tb = self.struct_tab, self.toolbar
        S = safe(self)

        def btn(text, cmd, tip=None):
            b = ttk.Button(tb, text=text, style="Tool.TButton", command=S(cmd), width=0)   # 宽度随文字
            b.pack(side="left", padx=1)
            if tip:
                from .theme import ToolTip
                ToolTip(b, T(tip))
            return b

        def sep():
            ttk.Frame(tb, style="ToolSep.TFrame", width=1).pack(side="left", fill="y", padx=6, pady=2)

        btn("打开", lambda: st.open_file(), "打开结构 / 晶体 (xyz / gjf / log / fchk / cif)   Ctrl+O")
        btn("图片", st.save_image, "保存高清结构图   Ctrl+S")
        sep()
        for ax_, v in (("a", 0), ("b", 1), ("c", 2)):
            btn(ax_, lambda v=v: st.view_along(v), f"沿 {ax_} 轴观察")
        btn("复位", lambda: st.redraw(keep_view=False), "重置视角")
        sep()
        for txt, var, cmd, tip in (("测量", st.meas_mode, st.toggle_measure, "测量：依次点 2 / 3 / 4 个原子"),
                                   ("取代", st.sub_mode, st.toggle_sub, "原子取代模式：点击要取代的原子")):
            c = ttk.Checkbutton(tb, text=txt, variable=var, style="Tool.Toolbutton", command=S(cmd), width=0)
            c.pack(side="left", padx=1)
            from .theme import ToolTip
            ToolTip(c, T(tip))
        self._panel_var = tk.BooleanVar(value=True)
        c = ttk.Checkbutton(tb, text="结构面板", variable=self._panel_var, style="Tool.Toolbutton", width=0,
                            command=lambda: self.toggle_nav(self._panel_var.get()))
        c.pack(side="left", padx=1)
        sep()
        btn("近邻二聚体", lambda: (self.show_page("结构/晶体", section="晶体二聚体"), st.search()),
            "搜索近邻二聚体并按对称等价分组")
        btn("EHT → μ", st.eht_all, "软件内 EHT 计算全部二聚体转移积分并填入迁移率表")
        sep()
        btn("Gaussian…", self.open_run_gaussian, "运行 Gaussian (重组能 / 激发态 / 转移积分 / 抵消校正 / ESP …)   Ctrl+G")
        btn("ADF…", self.open_run_adf, "运行 ADF (片段法转移积分 / 晶体二聚体批处理)")
        btn("迁移率…", lambda: self.show_page("迁移率"), "Marcus / KMC / 三维迁移率")
        sep()
        btn("流程图", self.open_flow, "流程图：把优化 → 重组能 → 转移积分 → 迁移率等多个阶段串成一个任务运行   Ctrl+L")
        btn("▶ 任务", lambda: self.open_runner(), "任务管理器：运行 / 停止 / 查看 Gaussian 与 ADF 任务   Ctrl+J")
        ttk.Button(tb, text="English" if i18n.LANG == "zh" else "中文", style="Tool.TButton",
                   command=self.toggle_lang).pack(side="right", padx=2)
        ttk.Button(tb, text="A+", style="Tool.TButton", width=3, command=lambda: self.change_font(+1)).pack(side="right")
        ttk.Button(tb, text="A−", style="Tool.TButton", width=3, command=lambda: self.change_font(-1)).pack(side="right")

    sync = False          # 测试时设为 True：后台任务同步执行

    @staticmethod
    def set_icon(win):
        """窗口与任务栏图标 (PNG 用于所有平台，ICO 用于 Windows)。"""
        assets = Path(__file__).resolve().parent / "assets"
        try:
            win._icon_img = tk.PhotoImage(file=str(assets / "mobilis_256.png"))
            win.iconphoto(True, win._icon_img)
        except Exception:  # noqa: BLE001
            pass
        if sys.platform.startswith("win"):
            try:
                win.iconbitmap(default=str(assets / "mobilis.ico"))
            except Exception:  # noqa: BLE001
                pass

    def run_bg(self, label, work, done):
        """在后台线程执行 work()，界面不冻结；完成后在界面线程调用 done(结果)。"""
        if self.sync:
            done(work())
            return
        if self._busy:
            messagebox.showinfo(T("请稍候"), T("正在计算: ") + T(self._busy))
            return
        import threading
        import time as _time
        self._busy = label
        box, t0 = {}, _time.time()
        self.status.configure(text="  " + T("正在计算: ") + T(label))
        try:
            self.progress.start(12)
            self.configure(cursor="watch")
        except Exception:  # noqa: BLE001
            pass

        def target():
            try:
                box["res"] = work()
            except Exception as exc:  # noqa: BLE001
                box["err"], box["tb"] = exc, traceback.format_exc()
        th = threading.Thread(target=target, daemon=True)
        th.start()

        def poll():
            if th.is_alive():
                self.after(80, poll)
                return
            self._busy = None
            try:
                self.progress.stop()
                self.configure(cursor="")
            except Exception:  # noqa: BLE001
                pass
            if "err" in box:
                self.log(box["tb"])
                messagebox.showerror(T("出错了"), str(box["err"]))
                return
            try:
                done(box["res"])
                self.log(f"{label} 完成 ({_time.time() - t0:.1f} s)")
            except Exception as exc:  # noqa: BLE001
                self.log(traceback.format_exc())
                messagebox.showerror(T("出错了"), str(exc))
        self.after(80, poll)

    def build_menu(self):
        """菜单：文件 / 编辑 / 构建 / 视图 / Gaussian / ADF / 迁移率 / 分析 / 任务 / 窗口 / 帮助。"""
        from .render import BACKGROUNDS, STYLES
        m = tk.Menu(self)
        st = self.struct_tab
        S = safe(self)

        def menu(label):
            mm = tk.Menu(m, tearoff=0)
            m.add_cascade(label=T(label), menu=mm)
            return mm

        def cmd(mm, label, fn, accel=None):
            mm.add_command(label=T(label), command=S(fn), accelerator=accel or "")

        def cascade(mm, label):
            s2 = tk.Menu(mm, tearoff=0)
            mm.add_cascade(label=T(label), menu=s2)
            return s2

        page = lambda n, sec=None: (lambda: self.show_page(n, section=sec))  # noqa: E731
        panel = lambda sec: (lambda: self.show_page("结构/晶体", section=sec))  # noqa: E731

        # 文件
        fm = menu("文件")
        cmd(fm, "打开结构 / 晶体…", st.open_file, "Ctrl+O")
        rec = tk.Menu(fm, tearoff=0, postcommand=lambda: self._fill_recent(rec))
        fm.add_cascade(label=T("最近打开"), menu=rec)
        self._recent_menu = rec
        fm.add_separator()
        cmd(fm, "导出结构 (取代后 / 参考分子)…", st.export_sub)
        cmd(fm, "导出选中分子 xyz…", st.export_selected)
        cmd(fm, "选中两分子 → 二聚体 xyz…", st.export_pair)
        cmd(fm, "整个场景另存为 xyz…", st.save_xyz)
        cmd(fm, "保存结构图片…", st.save_image, "Ctrl+S")
        fm.add_separator()
        cmd(fm, "设置工作目录…", self.set_workdir)
        fm.add_separator()
        cmd(fm, "退出", self.destroy)
        # 编辑
        em = menu("编辑")
        em.add_checkbutton(label=T("原子取代模式"), variable=st.sub_mode, command=S(st.toggle_sub))
        em.add_checkbutton(label=T("测量模式 (距离 / 键角 / 二面角)"), variable=st.meas_mode, command=S(st.toggle_measure))
        em.add_separator()
        cmd(em, "撤销取代", st.undo_sub, "Ctrl+Z")
        cmd(em, "清除选择", st.clear_sub, "Esc")
        # 构建
        bm = menu("构建")
        rg = cascade(bm, "显示范围 / 超胞")
        for lab, r0, r1 in (("1×1×1", 0, 1), ("2×2×2", 0, 2), ("3×3×3", 0, 3), ("中心 3×3×3", -1, 2)):
            rg.add_command(label=T(lab), command=S(lambda a=r0, b=r1: st.set_range(a, b)))
        rg.add_separator()
        rg.add_command(label=T("自定义范围…"), command=panel("显示范围"))
        cmd(bm, "晶胞缩放 / 应变…", panel("晶胞大小"))
        cmd(bm, "原子 / 基团取代…", panel("原子取代"))
        cmd(bm, "拆分为分子 (分子列表)…", panel("分子列表"))
        bm.add_separator()
        cmd(bm, "近邻二聚体…", panel("晶体二聚体"))
        cmd(bm, "搜索近邻二聚体", lambda: (panel("晶体二聚体")(), st.search()))
        cmd(bm, "应变扫描 (EHT → μ)", st.strain_scan)
        # 视图
        vm = menu("视图")
        sm = cascade(vm, "显示样式")
        for k in STYLES:
            sm.add_radiobutton(label=T(k), value=T(k), variable=st.style_var, command=S(st.redraw))
        bg = cascade(vm, "背景")
        for k in BACKGROUNDS:
            bg.add_radiobutton(label=T(k), value=T(k), variable=st.bg_var, command=S(st.redraw))
        vm.add_checkbutton(label=T("透视"), variable=st.persp, command=S(st.redraw))
        vm.add_checkbutton(label=T("显示氢"), variable=st.show_h, command=S(st.redraw))
        vm.add_checkbutton(label=T("原子编号"), variable=st.show_lab, command=S(st.redraw))
        vm.add_checkbutton(label=T("显示每个晶胞的边框"), variable=st.all_cells, command=S(st.redraw))
        vm.add_separator()
        vm.add_radiobutton(label=T("晶胞"), value="cell", variable=st.mode, command=S(st.rebuild))
        vm.add_radiobutton(label=T("近邻簇"), value="cluster", variable=st.mode, command=S(st.rebuild))
        vm.add_separator()
        va = cascade(vm, "观察方向")
        for lab, v in (("a", 0), ("b", 1), ("c", 2), ("x", "x"), ("y", "y"), ("z", "z")):
            va.add_command(label=f"{T('沿')} {lab}", command=S(lambda v=v: st.view_along(v)))
        cmd(vm, "重置视角", lambda: st.redraw(keep_view=False))
        vm.add_separator()
        cmd(vm, "显示 / 隐藏结构面板", lambda: self.toggle_nav(), "Ctrl+B")
        fs = cascade(vm, "字体")
        for k, v in FONT_SCALES.items():
            fs.add_command(label=f"{T(k)} ({int(v * 100)}%)", command=S(lambda v=v: self.set_font_scale(v)))
        fs.add_separator()
        fs.add_command(label=T("放大"), accelerator="Ctrl +", command=lambda: self.change_font(+1))
        fs.add_command(label=T("缩小"), accelerator="Ctrl −", command=lambda: self.change_font(-1))
        th = cascade(vm, "主题配色")
        from .theme import THEME, THEMES
        self._theme_var = tk.StringVar(value=THEME)
        for k in THEMES:
            th.add_radiobutton(label=T(k), value=k, variable=self._theme_var,
                               command=lambda k=k: self.set_theme(k))
        lg = cascade(vm, "语言 / Language")
        lg.add_command(label="中文", command=lambda: self.set_lang("zh"))
        lg.add_command(label="English", command=lambda: self.set_lang("en"))
        # Gaussian (一个运行对话框 + 各功能页)
        gm = menu("Gaussian")
        cmd(gm, "运行 Gaussian…", self.open_run_gaussian, "Ctrl+G")
        gm.add_separator()
        for lab, calc in (("结构优化 + 频率…", "opt"), ("四点法重组能…", "reorg"), ("激发态 S0 / S1 / T1…", "excited"),
                          ("二聚体转移积分 (投影法)…", "ti"), ("相互作用能 (抵消校正)…", "cp"),
                          ("ESP 电荷与极化率…", "esp"), ("FCHT 黄昆因子…", "fcht")):
            cmd(gm, lab, lambda c=calc: self.open_run_gaussian(c))
        # ADF
        am = menu("ADF")
        cmd(am, "运行 ADF…", self.open_run_adf)
        am.add_separator()
        cmd(am, "二聚体转移积分 (片段法)…", lambda: self.open_run_adf("ti"))
        cmd(am, "晶体全部近邻二聚体 (批处理)…", lambda: self.open_run_adf("crystal"))
        am.add_separator()
        cmd(am, "批量扫描 / CEDS / 主动学习…", page("批量扫描工作流"))
        cmd(am, "测试 ADF 环境…", lambda: ProgramsDialog(self))
        # 迁移率 (软件内计算，不需要外部程序)
        mm_ = menu("迁移率")
        cmd(mm_, "软件内 EHT：全部二聚体 J → 迁移率表", st.eht_all)
        cmd(mm_, "转移积分 (软件内 EHT)…", lambda: self.open_ti("eht"))
        cmd(mm_, "转移积分 (软件内 ZINDO/MOO)…", lambda: self.open_ti("zindo"))
        mm_.add_separator()
        cmd(mm_, "迁移率 (Marcus / KMC / 三维)…", page("迁移率"))
        cmd(mm_, "瞬态局域化 TLT / 能带 / 动态无序…", page("输运理论与光物理扩展"))
        cmd(mm_, "极化位点能 (P±、σ_E)…", page("极化位点能"))
        cmd(mm_, "无定形形貌 (GROMACS → KMC)…", page("无定形形貌 (MD)"))
        cmd(mm_, "迁移率方法对比…", page("迁移率方法对比"))
        mm_.add_separator()
        cmd(mm_, "相互作用能 (UFF) 与能量框架…", page("分子间相互作用能", "快速计算"))
        cmd(mm_, "耦合代理模型与堆积设计…", page("耦合代理模型与堆积设计"))
        # 分析：读取已完成的计算
        an = menu("分析")
        cmd(an, "重组能 (读取 Gaussian log)…", page("重组能", "结果分析"))
        cmd(an, "IP / EA / 前线轨道…", page("IP/EA"))
        cmd(an, "重组能分解 / 黄昆因子…", page("分解/黄昆因子"))
        cmd(an, "激发态与光谱…", page("激发态/光谱", "读取结果"))
        cmd(an, "转移积分读取与三种方法对比…", page("转移积分", "结果读取"))
        cmd(an, "波函数分析 (轨道 / 电荷 / ESP / NBO / IGM)…", page("波函数分析"))
        cmd(an, "相互作用能 (Gaussian 结果) 与能量框架…", page("分子间相互作用能", "Gaussian"))
        cmd(an, "扫描结果 / CEDS / 主动学习…", page("批量扫描工作流", "运行与结果"))
        # 任务
        jm = menu("任务")
        cmd(jm, "新建流程 (流程图)…", self.open_flow, "Ctrl+L")
        cmd(jm, "打开流程…", lambda: self.open_flow().load())
        cmd(jm, "任务管理器 (运行 / 停止 / 状态)…", lambda: self.open_runner(), "Ctrl+J")
        cmd(jm, "程序路径设置…", lambda: ProgramsDialog(self))
        # 窗口
        wm = tk.Menu(m, tearoff=0, postcommand=lambda: self._fill_windows(wm))
        m.add_cascade(label=T("窗口"), menu=wm)
        # 帮助
        hm = menu("帮助")
        cmd(hm, "关于", self.about)
        cmd(hm, "引用文献…", lambda: CitationDialog(self))
        self.configure(menu=m)
        keys = (("<Control-o>", lambda e: S(st.open_file)()), ("<Control-s>", lambda e: S(st.save_image)()),
                ("<Control-b>", lambda e: self.toggle_nav()), ("<Control-B>", lambda e: self.toggle_nav()),
                ("<Control-g>", lambda e: S(self.open_run_gaussian)()), ("<Control-j>", lambda e: self.open_runner()),
                ("<Control-l>", lambda e: S(self.open_flow)()),
                ("<Control-z>", lambda e: S(st.undo_sub)()), ("<Escape>", lambda e: S(st.clear_sub)()),
                ("<Control-plus>", lambda e: self.change_font(+1)), ("<Control-equal>", lambda e: self.change_font(+1)),
                ("<Control-minus>", lambda e: self.change_font(-1)), ("<Control-0>", lambda e: self.set_font_scale(1.0)))
        for seq, fn in keys:
            try:
                self.bind(seq, fn)
            except Exception:  # noqa: BLE001
                pass

    def _fill_windows(self, menu):
        menu.delete(0, "end")
        menu.add_command(label=T("主窗口 (结构)"), command=lambda: (self.lift(), self.focus_set()))
        if self._wins:
            menu.add_separator()
        for name, win in self._wins.items():
            menu.add_command(label=T(name), command=lambda n=name: self.show_page(n))
        if self._wins:
            menu.add_separator()
            menu.add_command(label=T("关闭全部功能窗口"), command=lambda: [w.withdraw() for w in self._wins.values()])

    def set_workdir(self, d=None):
        """工作目录：所有文件选择框的下拉列表会列出其中 (及下一层子目录) 的匹配文件，浏览时也从这里开始。"""
        if d is None:
            d = filedialog.askdirectory(parent=self, title=T("选择工作目录"), initialdir=self.config_data.get("workdir") or None)
        if d:
            self.config_data["workdir"] = str(d)
            save_config(self.config_data)
            self.log(f"工作目录: {d}")
            self.status.configure(text="  " + T("工作目录") + f": {d}")

    def recent_structures(self):
        ok = (".xyz", ".gjf", ".com", ".log", ".out", ".fchk", ".cif")
        def usable(p):
            try:
                return Path(p).is_file()
            except OSError:              # 已删除的临时目录、断开的网络盘等 (Windows: 拒绝访问)
                return False
        rec = self.config_data.get("recent_files", [])
        out = [p for p in rec if isinstance(p, str) and p.lower().endswith(ok) and usable(p)][:12]
        gone = [p for p in rec if isinstance(p, str) and p.lower().endswith(ok) and not usable(p)]
        if gone:                          # 顺手清理失效的记录
            self.config_data["recent_files"] = [p for p in rec if p not in gone]
            save_config(self.config_data)
        return out

    def _fill_recent(self, menu):
        menu.delete(0, "end")
        items = self.recent_structures()
        if not items:
            menu.add_command(label=T("(无)"), state="disabled")
        for p in items:
            menu.add_command(label=p, command=lambda p=p: (self.show_page("结构/晶体"), self.struct_tab.open_path(p)))

    def set_font_scale(self, scale):
        scale = min(max(float(scale), 0.8), 1.8)
        self.ui, self.mono = set_scale(self, scale)
        self.config_data["font_scale"] = round(scale, 2)
        save_config(self.config_data)
        self.status.configure(text="  " + T("字体大小") + f": {int(round(scale * 100))}%")

    def change_font(self, step):
        steps = sorted(set(list(FONT_SCALES.values()) + [1.05, 1.4, 1.6, 1.8, 0.8]))
        cur = float(self.config_data.get("font_scale", 1.0))
        nxt = [v for v in steps if (v > cur + 1e-6 if step > 0 else v < cur - 1e-6)]
        if nxt:
            self.set_font_scale(nxt[0] if step > 0 else nxt[-1])

    def set_theme(self, name):
        """切换主题配色：保存设置后重建主窗口 (与切换语言相同)。"""
        if name == self.config_data.get("theme", "专业浅色"):
            return
        self.config_data["theme"] = name
        save_config(self.config_data)
        self.restart = True
        self.destroy()

    def set_lang(self, lang):
        if lang != i18n.LANG:
            self.toggle_lang()

    def about(self):
        messagebox.showinfo(T("关于"), f"{APP_NAME} {T(APP_CN)}  v{__version__}\n{T(APP_SUBTITLE)}\n\n"
                            + T("开发者") + f"：{__author__}\n" + T("单位") + f"：{T(__affiliation__.split(' (')[0])}\n\n"
                            + T("重组能 · IP/EA · 黄昆因子 · 激发态 · 转移积分 (Gaussian / ADF / EHT) · "
                                "Marcus / KMC / TLT 迁移率"))

    def toggle_lang(self):
        self.config_data["lang"] = "en" if i18n.LANG == "zh" else "zh"
        save_config(self.config_data)
        self.restart = True
        self.destroy()

    def open_runner(self, directory=""):
        try:
            return RunDialog(self, directory or "")
        except Exception as exc:  # noqa: BLE001
            self.log(traceback.format_exc())
            messagebox.showerror("出错了", str(exc))

    def log(self, msg):
        text = T(str(msg).rstrip())
        self.logbox.insert("end", text + "\n")
        self.logbox.see("end")
        last = text.splitlines()[-1] if text else ""
        self.status.configure(text="  " + last[:160])


App.TAB_CLASSES = {
    "重组能": ("reorg_tab", ReorgTab), "IP/EA": ("ipea_tab", IPEATab), "分解/黄昆因子": ("decomp_tab", DecompTab),
    "激发态/光谱": ("photo_tab", PhotoTab), "转移积分": ("ti_tab", TITab), "迁移率": ("mob_tab", MobilityTab),
    "输运理论与光物理扩展": ("adv_tab", AdvancedTab), "分子间相互作用能": ("inter_tab", InteractionTab),
    "耦合代理模型与堆积设计": ("sur_tab", SurrogateTab), "波函数分析": ("wfn_tab", WavefunctionTab),
    "批量扫描工作流": ("scan_tab", ScanTab), "极化位点能": ("pol_tab", PolarTab), "无定形形貌 (MD)": ("morph_tab", MorphTab),
    "迁移率方法对比": ("cmp_tab", CompareTab)}
for _name, (_attr, _cls) in App.TAB_CLASSES.items():          # app.reorg_tab 等：首次访问时创建对应的功能窗口
    setattr(App, _attr, property(lambda self, n=_name: self._get_tab(n)))


def compress_ranges(idx):
    """[1,2,3,5] → "1-3,5" """
    out, start, prev = [], None, None
    for i in sorted(idx):
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = i
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else str(start))
    return ",".join(out)


def main():
    enable_hidpi()
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Mobilis.Qianyue.App")
        except Exception:  # noqa: BLE001
            pass
    while True:
        app = App()
        app.mainloop()
        if not getattr(app, "restart", False):
            break


if __name__ == "__main__":
    main()

