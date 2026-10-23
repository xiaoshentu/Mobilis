"""界面主题：配色、字体、ttk 样式、可滚动侧栏、文本框样式、高分屏适配。"""
import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import matplotlib

PALETTE = {
    "bg": "#F3F5F9",          # 窗口底色
    "card": "#FFFFFF",        # 卡片 / 面板
    "border": "#D8DEE9",
    "primary": "#2B4A77",     # 主色
    "primary_hover": "#3A5F96",
    "primary_light": "#DCE6F4",
    "accent": "#C0392B",
    "text": "#1F2937",
    "muted": "#6B7280",
    "header": "#1E3354",
    "header_text": "#FFFFFF",
    "log_bg": "#1E293B",
    "log_fg": "#E2E8F0",
    "stripe": "#F7F9FC",
    "nav": "#E8EDF5",         # 左侧导航
    "nav_hover": "#DCE4F0",
}
UI_FONTS = ["Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC",
            "WenQuanYi Micro Hei", "Segoe UI", "Helvetica"]
MONO_FONTS = ["Cascadia Mono", "Consolas", "Sarasa Mono SC", "Menlo", "DejaVu Sans Mono", "Courier New"]


SCALE = 1.0                      # 界面字体缩放比例
FONT_SCALES = {"小": 0.9, "标准": 1.0, "大": 1.15, "特大": 1.3, "超大": 1.5}
_TEXTS = []                      # 已创建的文本框 (随缩放更新字号)
_PANELS = []                     # 左侧可滚动面板 (随缩放更新宽度)


def sz(n):
    return max(7, int(round(n * SCALE)))


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
            p.canvas.configure(width=int(p._base_width * SCALE))
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


def apply_theme(root, scale=None):
    """配置 ttk 样式，返回 (界面字体名, 等宽字体名)。"""
    global SCALE
    if scale is not None:
        SCALE = float(scale)
    P = PALETTE
    try:
        fams = tkfont.families(root)
    except Exception:  # noqa: BLE001
        fams = []
    ui = _pick(fams, UI_FONTS, "TkDefaultFont")
    mono = _pick(fams, MONO_FONTS, "TkFixedFont")
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        try:
            tkfont.nametofont(name).configure(family=ui, size=sz(10))
        except Exception:  # noqa: BLE001
            pass
    root.configure(background=P["bg"])

    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except Exception:  # noqa: BLE001
        pass
    st.configure(".", background=P["bg"], foreground=P["text"], font=(ui, sz(10)), bordercolor=P["border"],
                 lightcolor=P["card"], darkcolor=P["border"], troughcolor=P["bg"], focuscolor=P["primary_light"])
    st.configure("TFrame", background=P["card"])
    st.configure("Bg.TFrame", background=P["bg"])
    st.configure("Card.TFrame", background=P["card"])
    st.configure("TLabel", background=P["card"])
    st.configure("Card.TLabel", background=P["card"])
    st.configure("Muted.TLabel", background=P["card"], foreground=P["muted"], font=(ui, sz(9)))
    st.configure("Header.TFrame", background=P["header"])
    st.configure("Header.TLabel", background=P["header"])
    st.configure("Brand.TLabel", background=P["header"], foreground="#F2A541", font=(ui, sz(16), "bold"))
    st.configure("HeaderSep.TFrame", background="#35507A")
    st.configure("Title.TLabel", background=P["header"], foreground=P["header_text"], font=(ui, sz(16), "bold"))
    st.configure("Subtitle.TLabel", background=P["header"], foreground="#B8C7DE", font=(ui, sz(10)))
    st.configure("Status.TLabel", background=P["card"], foreground=P["muted"], font=(ui, sz(9)), padding=(10, 3))

    # 左侧导航与页眉
    st.configure("Nav.TFrame", background=P["nav"])
    st.configure("NavGroup.TLabel", background=P["nav"], foreground=P["muted"], font=(ui, sz(9), "bold"),
                 padding=(14, 10, 8, 2))
    st.configure("Nav.TButton", background=P["nav"], foreground=P["text"], font=(ui, sz(10)), anchor="w",
                 padding=(20, 5, 8, 5), relief="flat", borderwidth=0, bordercolor=P["nav"], focuscolor=P["nav"])
    st.map("Nav.TButton", background=[("active", P["nav_hover"])], bordercolor=[("active", P["nav_hover"])])
    st.configure("NavActive.TButton", background=P["card"], foreground=P["primary"], font=(ui, sz(10), "bold"),
                 anchor="w", padding=(20, 5, 8, 5), relief="flat", borderwidth=0, bordercolor=P["card"],
                 focuscolor=P["card"])
    st.map("NavActive.TButton", background=[("active", P["card"])])
    st.configure("NavMark.TFrame", background=P["primary"])
    st.configure("NavBlank.TFrame", background=P["nav"])
    st.configure("PageHead.TFrame", background=P["bg"])
    st.configure("PageTitle.TLabel", background=P["bg"], foreground=P["primary"], font=(ui, sz(13), "bold"))
    st.configure("PageCrumb.TLabel", background=P["bg"], foreground=P["muted"], font=(ui, sz(9)))
    st.configure("PageDesc.TLabel", background=P["bg"], foreground=P["muted"], font=(ui, sz(9)))

    st.configure("TLabelframe", background=P["card"], bordercolor=P["border"], relief="solid", padding=8)
    st.configure("TLabelframe.Label", background=P["card"], foreground=P["primary"], font=(ui, sz(10), "bold"))

    st.configure("TButton", background=P["card"], foreground=P["text"], padding=(10, 5), relief="flat",
                 bordercolor=P["border"])
    st.map("TButton", background=[("pressed", P["primary_light"]), ("active", "#EEF2F8")],
           bordercolor=[("active", P["primary"])])
    st.configure("Accent.TButton", background=P["primary"], foreground="#FFFFFF", font=(ui, sz(10), "bold"),
                 padding=(12, 7), bordercolor=P["primary"])
    st.map("Accent.TButton", background=[("pressed", P["header"]), ("active", P["primary_hover"])],
           foreground=[("disabled", "#CBD5E1")])
    st.configure("Chip.TButton", background=P["header"], foreground="#DCE6F4", padding=(10, 3),
                 bordercolor=P["header"], font=(ui, sz(9)))
    st.map("Chip.TButton", background=[("active", P["primary_hover"])], foreground=[("active", "#FFFFFF")])

    for w in ("TEntry", "TCombobox", "TSpinbox"):
        st.configure(w, fieldbackground=P["card"], padding=4, bordercolor=P["border"], arrowcolor=P["primary"])
        st.map(w, bordercolor=[("focus", P["primary"])], lightcolor=[("focus", P["primary"])])
    st.configure("TCheckbutton", background=P["card"])
    st.configure("TRadiobutton", background=P["card"])
    st.map("TCheckbutton", background=[("active", P["card"])], indicatorcolor=[("selected", P["primary"])])
    st.map("TRadiobutton", background=[("active", P["card"])], indicatorcolor=[("selected", P["primary"])])

    st.configure("TNotebook", background=P["bg"], borderwidth=0, tabmargins=(8, 6, 8, 0))
    st.configure("Hidden.TNotebook", background=P["bg"], borderwidth=0, tabmargins=0)
    try:
        st.layout("Hidden.TNotebook.Tab", [])          # 隐藏标签页，改用菜单 / 下拉框切换
    except Exception:  # noqa: BLE001
        pass
    st.configure("TNotebook.Tab", background="#E4E9F2", foreground=P["muted"], padding=(16, 7), font=(ui, sz(10)),
                 bordercolor=P["border"])
    st.map("TNotebook.Tab", background=[("selected", P["card"]), ("active", "#EEF2F8")],
           foreground=[("selected", P["primary"])], font=[("selected", (ui, sz(10), "bold"))])

    st.configure("Treeview", background=P["card"], fieldbackground=P["card"], rowheight=sz(26),
                 bordercolor=P["border"])
    st.map("Treeview", background=[("selected", P["primary_light"])], foreground=[("selected", P["text"])])
    st.configure("Treeview.Heading", background="#E8EDF5", foreground=P["primary"], font=(ui, sz(9), "bold"),
                 relief="flat", padding=4)
    st.map("Treeview.Heading", background=[("active", P["primary_light"])])
    st.configure("Vertical.TScrollbar", background="#D5DCE8", troughcolor=P["bg"], arrowcolor=P["muted"],
                 bordercolor=P["bg"])
    st.configure("TPanedwindow", background=P["bg"])

    matplotlib.rcParams.update({
        "font.sans-serif": [ui, "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10 * SCALE, "axes.edgecolor": "#94A3B8", "axes.labelcolor": P["text"],
        "axes.titleweight": "bold", "axes.titlecolor": P["primary"], "axes.grid": False,
        "xtick.color": "#475569", "ytick.color": "#475569", "figure.facecolor": P["card"],
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
        self.canvas = tk.Canvas(self, width=int(width * SCALE), background=P["card"], highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Card.TFrame", padding=(12, 10))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.inner.bind("<Enter>", lambda e: self._wheel(True))
        self.inner.bind("<Leave>", lambda e: self._wheel(False))

    def _wheel(self, on):
        if on:
            self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))
            self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
            self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        else:
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.canvas.unbind_all(ev)


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
