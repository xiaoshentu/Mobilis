"""界面主题：配色、字体、ttk 样式、可滚动侧栏、文本框样式、高分屏适配。"""
import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import matplotlib

# ----------------------------------------------------------------------------------------------
# 主题 (参照目前主流计算化学 / 材料模拟软件：中性灰的窗口框架、白色工作区、一个强调色、状态色只表示状态；
#        界面字号 9 pt (Windows 标准，Segoe UI / 微软雅黑 UI)，日志与输入文件用等宽字体；图表底色始终为白色，便于直接出图)
# ----------------------------------------------------------------------------------------------
THEMES = {
    "专业浅色": dict(                     # 默认：中性灰框架 + 蓝色强调 (Materials Studio / Office 一类)
        bg="#EEF0F3", card="#FFFFFF", border="#C9CED6", primary="#1F5FA8", primary_hover="#2B70C0",
        primary_light="#DCE8F6", accent="#C0392B", text="#1E2329", muted="#5E6670", header="#28323E",
        header_text="#FFFFFF", log_bg="#FBFBFC", log_fg="#2B3038", stripe="#F6F7F9", nav="#E6E9EE",
        nav_hover="#DCE1E8", toolbar="#F7F8FA", tool_hover="#E3E8EF", tool_sel="#CFDDF0", toolsep="#CDD2D9",
        structbar="#2F3B4A", structbar_text="#FFFFFF", button="#F7F8FA", button_border="#BFC6D0",
        button_hover="#EAF1FA", tab="#E4E7EC", tab_hover="#F0F2F5", heading="#EEF0F3", scroll="#CBD1DA",
        field="#FFFFFF", select="#CFE0F5", plot_bg="#FFFFFF", ok="#2E8B57", warn="#C98A04", bad="#C0392B"),
    "经典灰": dict(                      # 浅灰框架 + 橄榄色标题栏
        bg="#ECECEC", card="#FFFFFF", border="#BEBEBE", primary="#5D6B1F", primary_hover="#6F7F27",
        primary_light="#E6EAD2", accent="#B03A2E", text="#1F1F1F", muted="#5F5F5F", header="#4A5418",
        header_text="#FFFFFF", log_bg="#FFFFFF", log_fg="#202020", stripe="#F5F5F2", nav="#E4E4E4",
        nav_hover="#D8D8D8", toolbar="#F2F2F2", tool_hover="#E2E5D6", tool_sel="#D5DCB8", toolsep="#C6C6C6",
        structbar="#9C9A2E", structbar_text="#FFFFFF", button="#F4F4F4", button_border="#B4B4B4",
        button_hover="#EEF0E2", tab="#E2E2E2", tab_hover="#EDEDED", heading="#ECECEC", scroll="#C8C8C8",
        field="#FFFFFF", select="#DDE4BF", plot_bg="#FFFFFF", ok="#2E7D32", warn="#B7860B", bad="#B03A2E"),
    "深色": dict(                         # 长时间使用时减少眩光 (Maestro / VS Code 一类)；图表仍为白底
        bg="#1E2228", card="#262B33", border="#3A414C", primary="#4C8DDB", primary_hover="#5E9BE3",
        primary_light="#2C405B", accent="#E06C5A", text="#E3E7ED", muted="#98A2AE", header="#15181D",
        header_text="#FFFFFF", log_bg="#181B20", log_fg="#D3D9E1", stripe="#2B3139", nav="#22262D",
        nav_hover="#2E343D", toolbar="#2A2F37", tool_hover="#353C46", tool_sel="#2F4A6B", toolsep="#454D59",
        structbar="#33404F", structbar_text="#FFFFFF", button="#313741", button_border="#4A525E",
        button_hover="#3A424E", tab="#2A2F37", tab_hover="#323843", heading="#2D333C", scroll="#4A525E",
        field="#1F242B", select="#2F4A6B", plot_bg="#FFFFFF", ok="#5CB97A", warn="#E0A43A", bad="#E06C5A"),
}
THEME = "专业浅色"
PALETTE = dict(THEMES[THEME])
BASE_PT = 9                       # 界面基础字号 (pt)；标题 +3，页内小字与正文相同


def set_theme(name):
    """切换配色 (原地更新 PALETTE，已导入 PALETTE 的模块随之生效；界面需重建才能完全应用)。"""
    global THEME
    THEME = name if name in THEMES else "专业浅色"
    PALETTE.clear()
    PALETTE.update(THEMES[THEME])
    return THEME


UI_FONTS = ["Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC",
            "WenQuanYi Micro Hei", "Segoe UI", "Helvetica"]
UI_FONTS_EN = ["Segoe UI", "Microsoft YaHei UI", "Helvetica Neue", "Noto Sans", "DejaVu Sans", "Helvetica"]
MONO_FONTS = ["Cascadia Mono", "Consolas", "Sarasa Mono SC", "Menlo", "DejaVu Sans Mono", "Courier New"]


SCALE = 1.0                      # 界面字体缩放比例
FONT_SCALES = {"小": 0.9, "标准": 1.0, "大": 1.15, "特大": 1.3, "超大": 1.5}
_TEXTS = []                      # 已创建的文本框 (随缩放更新字号)
_PANELS = []                     # 左侧可滚动面板 (随缩放更新宽度)


DPI = 1.0                        # 屏幕缩放 (Windows 150% → 1.5)；像素尺寸须乘它，字号 (磅) 由 Tk 自动缩放


def sz(n):
    return max(7, int(round(n * SCALE)))


def px(n):
    """以 100% 屏幕缩放、标准字号设计的像素尺寸 → 当前屏幕与字号下的像素。"""
    return int(round(n * SCALE * DPI))


def set_scale(root, scale):
    """运行中改变字体大小：重新配置 ttk 样式、已命名字体、文本框与侧栏宽度。"""
    ui, mono = apply_theme(root, scale)
    for ref in list(_TEXTS):
        t = ref()
        if t is None:
            continue
        try:
            t.configure(font=(mono, sz(t._base_size)))
        except Exception:  # noqa: BLE001
            pass
    for ref in list(_PANELS):
        p = ref()
        if p is None:
            continue
        try:
            p._fit()
        except Exception:  # noqa: BLE001
            pass
    return ui, mono


def enable_hidpi():
    """Windows 高分屏下避免界面模糊。"""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass


def _pick(families, candidates, default):
    fams = set(families)
    return next((f for f in candidates if f in fams), default)


def _font_chain(names):
    from .viewer import font_chain
    return font_chain(names)


def apply_theme(root, scale=None):
    """配置 ttk 样式，返回 (界面字体名, 等宽字体名)。"""
    global SCALE, DPI
    if scale is not None:
        SCALE = float(scale)
    try:
        DPI = max(1.0, float(root.winfo_fpixels("1i")) / 96.0)
    except Exception:  # noqa: BLE001
        DPI = 1.0
    P = PALETTE
    B = BASE_PT
    try:
        fams = tkfont.families(root)
    except Exception:  # noqa: BLE001
        fams = []
    try:
        from .i18n import LANG
    except Exception:  # noqa: BLE001
        LANG = "zh"
    ui = _pick(fams, UI_FONTS_EN if LANG == "en" else UI_FONTS, "TkDefaultFont")   # 英文界面用 Segoe UI
    mono = _pick(fams, MONO_FONTS, "TkFixedFont")
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        try:
            tkfont.nametofont(name).configure(family=ui, size=sz(B))
        except Exception:  # noqa: BLE001
            pass
    root.configure(background=P["bg"])
    for opt, val in (("*TCombobox*Listbox.background", P["field"]), ("*TCombobox*Listbox.foreground", P["text"]),
                     ("*TCombobox*Listbox.selectBackground", P["select"]), ("*TCombobox*Listbox.selectForeground", P["text"]),
                     ("*Menu.background", P["card"]), ("*Menu.foreground", P["text"]),
                     ("*Menu.activeBackground", P["select"]), ("*Menu.activeForeground", P["text"])):
        try:                                     # 下拉列表与弹出菜单 (Windows 菜单栏为系统原生，不受影响)
            root.option_add(opt, val)
        except Exception:  # noqa: BLE001
            pass

    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except Exception:  # noqa: BLE001
        pass
    st.configure(".", background=P["bg"], foreground=P["text"], font=(ui, sz(B)), bordercolor=P["border"],
                 lightcolor=P["card"], darkcolor=P["border"], troughcolor=P["bg"], focuscolor=P["primary_light"])
    st.configure("TFrame", background=P["card"])
    st.configure("Bg.TFrame", background=P["bg"])
    st.configure("Card.TFrame", background=P["card"])
    st.configure("TLabel", background=P["card"])
    st.configure("Card.TLabel", background=P["card"])
    st.configure("Muted.TLabel", background=P["card"], foreground=P["muted"], font=(ui, sz(B)))
    st.configure("Header.TFrame", background=P["header"])
    st.configure("Header.TLabel", background=P["header"])
    st.configure("Brand.TLabel", background=P["header"], foreground="#F2A541", font=(ui, sz(B + 6), "bold"))
    st.configure("HeaderSep.TFrame", background="#35507A")
    st.configure("Title.TLabel", background=P["header"], foreground=P["header_text"], font=(ui, sz(B + 6), "bold"))
    st.configure("Subtitle.TLabel", background=P["header"], foreground="#B8C7DE", font=(ui, sz(B)))
    st.configure("Status.TLabel", background=P["card"], foreground=P["muted"], font=(ui, sz(B)), padding=(10, 3))

    # 左侧导航与页眉
    st.configure("Nav.TFrame", background=P["nav"])
    st.configure("NavGroup.TLabel", background=P["nav"], foreground=P["muted"], font=(ui, sz(B), "bold"),
                 padding=(14, 10, 8, 2))
    st.configure("Nav.TButton", background=P["nav"], foreground=P["text"], font=(ui, sz(B)), anchor="w",
                 padding=(20, 5, 8, 5), relief="flat", borderwidth=0, bordercolor=P["nav"], focuscolor=P["nav"])
    st.map("Nav.TButton", background=[("active", P["nav_hover"])], bordercolor=[("active", P["nav_hover"])])
    st.configure("NavActive.TButton", background=P["card"], foreground=P["primary"], font=(ui, sz(B), "bold"),
                 anchor="w", padding=(20, 5, 8, 5), relief="flat", borderwidth=0, bordercolor=P["card"],
                 focuscolor=P["card"])
    st.map("NavActive.TButton", background=[("active", P["card"])])
    st.configure("NavMark.TFrame", background=P["primary"])
    st.configure("NavBlank.TFrame", background=P["nav"])
    st.configure("PageHead.TFrame", background=P["bg"])
    # 主窗口：工具栏 + 结构标题栏
    st.configure("Toolbar.TFrame", background=P["toolbar"])
    st.configure("Tool.TButton", background=P["toolbar"], foreground=P["text"], padding=(8, 3), relief="flat",
                 borderwidth=1, bordercolor=P["toolbar"], lightcolor=P["toolbar"], darkcolor=P["toolbar"], font=(ui, sz(B)))
    st.map("Tool.TButton", background=[("pressed", P["primary_light"]), ("active", P["tool_hover"])],
           bordercolor=[("active", P["button_border"])])
    st.configure("Tool.Toolbutton", background=P["toolbar"], foreground=P["text"], padding=(8, 3), font=(ui, sz(B)))
    st.map("Tool.Toolbutton", background=[("selected", P["tool_sel"]), ("active", P["tool_hover"])])
    st.configure("ToolSep.TFrame", background=P["toolsep"])
    st.configure("StructBar.TLabel", background=P["structbar"], foreground=P["structbar_text"], font=(ui, sz(B), "bold"),
                 padding=(10, 4))
    st.configure("DlgBar.TFrame", background=P["toolbar"])
    st.configure("DlgBar.TLabel", background=P["toolbar"])
    st.configure("PageTitle.TLabel", background=P["bg"], foreground=P["primary"], font=(ui, sz(B + 3), "bold"))
    st.configure("PageCrumb.TLabel", background=P["bg"], foreground=P["muted"], font=(ui, sz(B)))
    st.configure("PageDesc.TLabel", background=P["bg"], foreground=P["muted"], font=(ui, sz(B)))

    st.configure("TLabelframe", background=P["card"], bordercolor=P["border"], relief="solid", padding=8)
    st.configure("TLabelframe.Label", background=P["card"], foreground=P["primary"], font=(ui, sz(B), "bold"))
    st.configure("Section.TLabelframe", background=P["card"], borderwidth=0, relief="flat", padding=(2, 4))
    st.configure("Section.TLabelframe.Label", background=P["card"])
    st.configure("TNotebook", background=P["card"], borderwidth=0)
    st.configure("TNotebook.Tab", padding=(12, 5), font=(ui, sz(B)))
    st.map("TNotebook.Tab", foreground=[("selected", P["primary"])], font=[("selected", (ui, sz(B), "bold"))])

    # 普通按钮：浅底 + 细边框，一眼能看出可以点击 (纯文字样式容易被当成标签)
    st.configure("TButton", background=P["button"], foreground=P["text"], padding=(10, 3), relief="solid",
                 borderwidth=1, bordercolor=P["button_border"], lightcolor=P["button"], darkcolor=P["button"],
                 wraplength=px(400))
    st.map("TButton", background=[("pressed", P["primary_light"]), ("active", P["button_hover"])],
           bordercolor=[("active", P["primary"])])
    st.configure("Accent.TButton", background=P["primary"], foreground="#FFFFFF", font=(ui, sz(B), "bold"),
                 padding=(12, 5), bordercolor=P["primary"], lightcolor=P["primary"], darkcolor=P["primary"])
    st.map("Accent.TButton", background=[("pressed", P["header"]), ("active", P["primary_hover"])],
           foreground=[("disabled", P["muted"])])
    st.configure("Chip.TButton", background=P["header"], foreground="#DCE6F4", padding=(10, 3),
                 bordercolor=P["header"], lightcolor=P["header"], darkcolor=P["header"], relief="flat",
                 font=(ui, sz(B)))
    st.map("Chip.TButton", background=[("active", P["primary_hover"])], foreground=[("active", "#FFFFFF")])

    for w in ("TEntry", "TCombobox", "TSpinbox"):
        st.configure(w, fieldbackground=P["field"], foreground=P["text"], padding=3, bordercolor=P["border"],
                     arrowcolor=P["primary"], insertcolor=P["text"], selectbackground=P["select"], selectforeground=P["text"])
        st.map(w, bordercolor=[("focus", P["primary"])], lightcolor=[("focus", P["primary"])],
               fieldbackground=[("readonly", P["field"]), ("disabled", P["bg"])], foreground=[("disabled", P["muted"])])
    # 长文字自动换行 (英文界面中的长选项 / 按钮)，不撑宽左侧面板
    st.configure("TCheckbutton", background=P["card"], indicatorsize=px(13), wraplength=px(400))
    st.configure("TRadiobutton", background=P["card"], indicatorsize=px(13), wraplength=px(400))
    st.map("TCheckbutton", background=[("active", P["card"])], indicatorcolor=[("selected", P["primary"])])
    st.map("TRadiobutton", background=[("active", P["card"])], indicatorcolor=[("selected", P["primary"])])

    st.configure("TNotebook", background=P["bg"], borderwidth=0, tabmargins=(8, 6, 8, 0))
    st.configure("Hidden.TNotebook", background=P["bg"], borderwidth=0, tabmargins=0)
    try:
        st.layout("Hidden.TNotebook.Tab", [])          # 隐藏标签页，改用菜单 / 下拉框切换
    except Exception:  # noqa: BLE001
        pass
    st.configure("TNotebook.Tab", background=P["tab"], foreground=P["muted"], padding=(14, 5), font=(ui, sz(B)),
                 bordercolor=P["border"])
    st.map("TNotebook.Tab", background=[("selected", P["card"]), ("active", P["tab_hover"])],
           foreground=[("selected", P["primary"])], font=[("selected", (ui, sz(B), "bold"))])

    st.configure("Treeview", background=P["card"], fieldbackground=P["card"], foreground=P["text"], rowheight=px(24),
                 bordercolor=P["border"])
    st.map("Treeview", background=[("selected", P["select"])], foreground=[("selected", P["text"])])
    st.configure("Treeview.Heading", background=P["heading"], foreground=P["primary"], font=(ui, sz(B), "bold"),
                 relief="flat", padding=4)
    st.map("Treeview.Heading", background=[("active", P["primary_light"])])
    st.configure("Vertical.TScrollbar", background=P["scroll"], troughcolor=P["bg"], arrowcolor=P["muted"],
                 bordercolor=P["bg"])
    st.configure("TPanedwindow", background=P["bg"])

    matplotlib.rcParams.update({
        "font.sans-serif": [ui, "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        # 直接列出字体族：matplotlib 按字逐个回退，雅黑没有的字符 (如上标负号 ⁻) 用 DejaVu Sans 显示
        "font.family": _font_chain([ui, "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]),
        "axes.unicode_minus": False, "font.size": (B + 1) * SCALE, "axes.edgecolor": "#8A93A0", "axes.labelcolor": "#1E2329", "text.color": "#1E2329",
        "axes.titleweight": "bold", "axes.titlecolor": "#1E2329", "axes.grid": False,
        "xtick.color": "#475569", "ytick.color": "#475569", "figure.facecolor": P["plot_bg"], "axes.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF",
        "axes.prop_cycle": matplotlib.cycler(color=[P["primary"], P["accent"], "#2E8B57", "#D98E04",
                                                    "#7B5EA7", "#2C9FB0"]),
    })
    return ui, mono


TRANSLATOR = None


class TText(tk.Text):
    """插入的文字自动经过界面语言翻译。"""

    def insert(self, index, chars, *args):
        if TRANSLATOR is not None and isinstance(chars, str):
            chars = TRANSLATOR(chars)
        return super().insert(index, chars, *args)


def styled_text(parent, mono, dark=False, **kw):
    import weakref
    parent = route_parent(parent)
    P = PALETTE
    base = kw.pop("size", 10)
    t = TText(parent, font=(mono, sz(base)), relief="flat", borderwidth=0, padx=10, pady=8,
                highlightthickness=1, highlightbackground=P["border"], highlightcolor=P["primary"],
                background=P["log_bg"] if dark else P["card"], foreground=P["log_fg"] if dark else P["text"],
                insertbackground=P["primary"], selectbackground=P["primary_light"], **kw)
    t._base_size = base
    try:
        _TEXTS.append(weakref.ref(t))
    except TypeError:
        pass
    return t


class ScrollPanel(ttk.Frame):
    """固定宽度、可用滚轮滚动的左侧控制栏；把控件放进 .inner。"""

    def __init__(self, parent, width=420):
        import weakref
        super().__init__(parent, style="Card.TFrame")
        P = PALETTE
        self._base_width = width
        try:
            _PANELS.append(weakref.ref(self))
        except TypeError:
            pass
        self.canvas = tk.Canvas(self, width=px(width), background=P["card"], highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Card.TFrame", padding=(12, 10))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.inner.bind("<Enter>", lambda e: self._wheel(True))
        self.inner.bind("<Leave>", lambda e: self._wheel(False))

    def _on_inner(self, _e=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._fit()

    def reflow(self):
        """把超出面板宽度的“横排一行”控件 (pack side=left) 自动折成多行；标签与其后的输入框保持在同一行。"""
        try:
            if not self.winfo_exists():              # 已销毁的窗口 (如切换语言后重建主窗口)
                return
        except Exception:  # noqa: BLE001
            return
        limit = px(self._base_width) - px(30)
        try:
            self.inner.update_idletasks()
            _reflow(self.inner, limit)
        except Exception:  # noqa: BLE001
            pass
        self._fit()

    def _fit(self):
        """面板宽度 = max(设计宽度, 内容所需宽度)，但不超过窗口宽度的一半；避免英文 / 高分屏下控件被截断。"""
        want = px(self._base_width)
        try:
            need = self.inner.winfo_reqwidth()
            cap = max(want, int(self.winfo_toplevel().winfo_width() * 0.42))
            if self.winfo_toplevel().winfo_width() <= 1:
                cap = max(want, int(self.winfo_screenwidth() * 0.4))
            want = min(max(want, need), cap)
        except Exception:  # noqa: BLE001
            pass
        if abs(int(float(self.canvas.cget("width"))) - want) > 2:
            self.canvas.configure(width=want)

    def _wheel(self, on):
        if on:
            self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))
            self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
            self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        else:
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.canvas.unbind_all(ev)


def _short_title(text, n=16):
    """分区标题 → 标签页名：去掉括号里的说明，过长时截断。"""
    t = str(text or "").strip()
    for sep in (" (", "（", "：", ": ", " —"):
        if sep in t[1:]:
            t = t[:t.index(sep, 1)]
    return t if len(t) <= n else t[:n - 1] + "…"


class SectionPanel(ttk.Frame):
    """分区面板：页面代码照常往 .inner 里放控件，每个 LabelFrame (分区) 自动成为一个标签页，
    分区之后的松散控件 (按钮等) 归入该分区；第一个分区之前的控件放在标签页上方的公共区域。"""

    def __init__(self, parent, width=440):
        super().__init__(parent, style="Card.TFrame")
        self._width = width
        self.header = ttk.Frame(self, style="Card.TFrame", padding=(12, 10, 12, 2))
        self.header.pack(fill="x")
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=(4, 6))
        self.inner = ttk.Frame(self)                 # 不显示：只作为路由标记
        self.inner._router = self
        self._cur = None
        self.pages = []
        self.orig = []                              # 分区的原始 (中文) 标题，用于按名称切换

    def route(self, cls_name, kw, orig=None):
        """返回 (实际父控件, kw)。由 gui 的控件包装层在创建控件时调用；orig 为翻译前的分区标题。"""
        if cls_name in ("LabelFrame", "Labelframe"):
            title = _short_title(kw.get("text", ""))
            page = ScrollPanel(self.nb, width=self._width)
            self.nb.add(page, text=title)
            self.pages.append((title, page))
            self.orig.append(str(orig or title))
            self._cur = page.inner
            kw = dict(kw, text="", style="Section.TLabelframe")
            return page.inner, kw
        return (self._cur if self._cur is not None else self.header), kw

    def target(self):
        return self._cur if self._cur is not None else self.header

    def select(self, key):
        """按标签页序号或标题中的文字切换。"""
        for i, (title, page) in enumerate(self.pages):
            o = self.orig[i] if i < len(self.orig) else title
            if key == i or (isinstance(key, str) and key and (key in title or key in o)):
                try:
                    self.nb.select(page)
                except Exception:  # noqa: BLE001
                    pass
                return page
        return None

    def reflow(self):
        for _, p in self.pages:
            p.reflow()


def route_parent(parent):
    """tk 原生控件 (如文本框) 放进分区面板时，取当前分区作为父控件。"""
    r = getattr(parent, "__dict__", {}).get("_router")
    return r.target() if r is not None else parent


def _reflow(w, limit):
    for k in w.winfo_children():
        cls = k.winfo_class()
        if cls in ("TFrame", "TLabelframe", "Frame", "Labelframe"):
            _reflow(k, limit - 20)
        elif cls in ("TLabel", "TCheckbutton", "TRadiobutton", "TButton") and k.winfo_reqwidth() > limit:
            try:                                                      # 过长的说明文字 / 选项换行
                cur = int(float(str(k.cget("wraplength") or 0) or 0))
                want = max(px(200), limit - (10 if cls == "TLabel" else px(40)))
                k.configure(wraplength=str(min(cur, want) if cur > 0 else want))   # str: 不再按 DPI 放大
            except Exception:  # noqa: BLE001
                pass
        elif cls == "Text" and k.winfo_reqwidth() > limit:
            try:
                cw = tkfont.Font(font=k.cget("font")).measure("0") or 8
                k.configure(width=max(20, int((limit - 30) / cw)))
            except Exception:  # noqa: BLE001
                pass
    if w.grid_slaves():
        _reflow_grid(w, limit)
    kids = w.pack_slaves()
    if len(kids) < 2 or any(k.pack_info().get("side") != "left" for k in kids):
        return
    if w.winfo_reqwidth() <= limit:
        return
    units, i = [], 0                    # 标签 + 紧随的输入控件 视为一个单元
    while i < len(kids):
        if kids[i].winfo_class() == "TLabel" and i + 1 < len(kids) and kids[i + 1].winfo_class() != "TLabel":
            units.append(kids[i:i + 2])
            i += 2
        else:
            units.append(kids[i:i + 1])
            i += 1
    info = {k: k.pack_info() for k in kids}

    def wid(k):
        pi = info[k]
        padx = pi.get("padx", 0)
        padx = sum(padx) if isinstance(padx, (tuple, list)) else int(padx) * 2
        return k.winfo_reqwidth() + padx
    rows, cur, used = [], [], 0
    for u in units:
        uw = sum(wid(k) for k in u)
        if cur and used + uw > limit:
            rows.append(cur)
            cur, used = [], 0
        cur += u
        used += uw
    rows.append(cur)
    if len(rows) < 2:
        return
    for k in kids:
        k.pack_forget()
    ncol = 0
    for r, row in enumerate(rows):
        for c, k in enumerate(row):
            pi = info[k]
            k.grid(row=r, column=c, sticky="w" if k.winfo_class() == "TLabel" else "we",
                   padx=pi.get("padx", 0), pady=pi.get("pady", 0))
            if k.winfo_class() != "TLabel":
                w.columnconfigure(c, weight=1)
        ncol = max(ncol, len(row))


def _reflow_grid(w, limit):
    """多列表单 (标签 / 输入框 成对并排) 太宽时改为每行一对；多列复选框改为单列。"""
    if w.winfo_reqwidth() <= limit * 1.06:
        return
    kids = w.grid_slaves()
    info = {k: k.grid_info() for k in kids}
    ncol = max(int(i["column"]) + int(i["columnspan"]) for i in info.values())
    col0 = [k for k in kids if int(info[k]["column"]) == 0 and int(info[k]["columnspan"]) == 1]
    b = 2 if col0 and sum(k.winfo_class() == "TLabel" for k in col0) * 2 >= len(col0) else 1
    if ncol > b:
        _regrid(w, kids, info, ncol, b)
    if b == 2:
        w.update_idletasks()
        if w.winfo_reqwidth() > limit * 1.06:
            _squeeze_pairs(w, limit)


def _squeeze_pairs(w, limit):
    """标签 + 输入框 两列表单仍太宽：标签换行 (≤ 55%)，输入框按剩余宽度缩窄。"""
    lab_w = int(limit * 0.55)
    for k in w.grid_slaves():
        i = k.grid_info()
        if int(i["columnspan"]) > 1:
            continue
        cls = k.winfo_class()
        try:
            if int(i["column"]) == 0 and cls == "TLabel" and k.winfo_reqwidth() > lab_w:
                k.configure(wraplength=str(lab_w))
            elif int(i["column"]) == 1 and cls in ("TEntry", "TCombobox", "TSpinbox"):
                cw = tkfont.Font(font=k.cget("font") or "TkTextFont").measure("0") or 8
                room = limit - min(lab_w, max((s.winfo_reqwidth() for s in w.grid_slaves(row=int(i["row"]), column=0)),
                                              default=0)) - px(40)
                if k.winfo_reqwidth() > room:
                    k.configure(width=max(6, int(room / cw)))
        except Exception:  # noqa: BLE001
            pass


def _regrid(w, kids, info, ncol, b):
    units = {}
    for k in kids:
        i = info[k]
        r, c, cs = int(i["row"]), int(i["column"]), int(i["columnspan"])
        key = (r, c // b, 0) if cs <= b and (c % b) + cs <= b else (r, c // b, 1)
        units.setdefault(key, []).append(k)
    order = sorted(units)
    for c in range(ncol):
        w.columnconfigure(c, weight=0)
    for new_r, key in enumerate(order):
        for k in units[key]:
            i = info[k]
            span = key[2] == 1
            k.grid_configure(row=new_r, column=0 if span else int(i["column"]) % b,
                             columnspan=b if span else int(i["columnspan"]))
    w.columnconfigure(b - 1, weight=1)


def stripe(tree):
    """表格隔行底色。"""
    for i, iid in enumerate(tree.get_children()):
        tree.item(iid, tags=("odd" if i % 2 else "even",))
    tree.tag_configure("odd", background=PALETTE["stripe"])
    tree.tag_configure("even", background=PALETTE["card"])



class ToolTip:
    """鼠标悬停 0.5 s 后显示说明。"""

    def __init__(self, widget, text):
        self.w, self.text, self.tip, self.job = widget, text, None, None
        try:
            widget.bind("<Enter>", self._enter, add="+")
            widget.bind("<Leave>", self._leave, add="+")
            widget.bind("<ButtonPress>", self._leave, add="+")
        except Exception:  # noqa: BLE001
            pass

    def _enter(self, _e=None):
        self.job = self.w.after(500, self._show)

    def _leave(self, _e=None):
        if self.job:
            self.w.after_cancel(self.job)
            self.job = None
        if self.tip:
            self.tip.destroy()
            self.tip = None

    def _show(self):
        x = self.w.winfo_rootx() + 12
        y = self.w.winfo_rooty() + self.w.winfo_height() + 4
        self.tip = tk.Toplevel(self.w)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#FFFBEA", foreground="#1F2937",
                 relief="solid", borderwidth=1, wraplength=int(360 * SCALE), font=("", sz(9)), padx=6, pady=4).pack()
