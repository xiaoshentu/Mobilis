"""程序运行对话框：运行 Gaussian / 运行 ADF。

组织方式：
    标签页：计算 / 泛函与基组 / 溶剂 / 资源 / 附加输入 / 预览输入
    底部：任务名 + 运行 / 写出输入文件 / 恢复默认 / 关闭
“运行”= 写出输入文件后交给任务管理器 (运行任务窗口) 排队计算；“写出输入文件”只生成文件 (可拿到服务器上算)。
输入文件由各计算模块原有的生成函数写出，与功能页、命令行生成的完全相同。
"""
import tempfile
import time
from pathlib import Path

import numpy as np

from . import gui as G
from .gui import T, filedialog, messagebox, tk, ttk

PALETTE = G.PALETTE


# ----------------------------------------------------------------------------------------------
# 从主窗口取结构
# ----------------------------------------------------------------------------------------------
def current_molecule(app):
    """主窗口当前结构中的一个分子：单分子文件直接用；晶体中选中一个分子时用它，否则用参考分子。"""
    st = app.struct_tab
    if st.atoms is not None:
        return list(st.atoms), Path(app.current_structure or "molecule").stem
    if st.mols:
        if len(st.sel) == 1:
            return st._sel_atoms(st.sel), st.scene[st.sel[0]]["label"]
        ref = int(st.ref.get() or 1) - 1
        m = st.mols[ref]
        return [(s, *map(float, x)) for s, x in zip(m["symbols"], m["cart"])], f"{Path(app.current_structure).stem}_m{ref + 1}"
    raise ValueError("主窗口中还没有结构：请先 文件 → 打开结构")


def current_dimer(app):
    """主窗口中选中的两个分子 (左键点第一个，Ctrl+左键点第二个)。返回 (A 原子, B 原子)。"""
    st = app.struct_tab
    if len(st.sel) != 2:
        raise ValueError("请在主窗口中恰好选中两个分子 (左键点第一个，Ctrl+左键点第二个)，"
                         "或把“来源”改为二聚体文件 / 晶体全部近邻二聚体")
    return st._sel_atoms([st.sel[0]]), st._sel_atoms([st.sel[1]])


def _new_files(d, t0, exts=(".gjf", ".com", ".run", ".sh", ".bat", ".json", ".xyz", ".in")):
    d = Path(d)
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts and p.stat().st_mtime >= t0 - 1)


# ----------------------------------------------------------------------------------------------
# 通用对话框
# ----------------------------------------------------------------------------------------------
class RunProgramDialog(tk.Toplevel):
    PROGRAM = ""
    CFG = ""
    CALCS = []            # [(key, 名称, 说明)]
    DEFAULTS = {}
    MAPPED = {}           # 取值需要翻译显示的下拉框 {变量名: 全部选项}
    _instances = {}

    @classmethod
    def open(cls, app, calc=None):
        dlg = cls._instances.get((cls, id(app)))
        try:
            alive = dlg is not None and bool(dlg.winfo_exists())
        except Exception:  # noqa: BLE001
            alive = False
        if not alive:
            dlg = cls(app)
            cls._instances[(cls, id(app))] = dlg
        else:
            dlg.deiconify()
            dlg.lift()
        if calc:
            dlg.calc.set(calc)
            dlg._calc_changed()
        return dlg

    def __init__(self, app, stage=None, on_ok=None):
        """stage: 流程图中的一个阶段 (dict)。给出时对话框编辑该阶段的设置，底部按钮为 确定 / 恢复默认 / 取消。"""
        super().__init__(app)
        self.app = app
        self.stage, self.on_ok = stage, on_ok
        S = G.safe(app)
        self.title(T(f"运行 {self.PROGRAM}") + f" — {G.APP_NAME}" if stage is None
                   else T(f"流程阶段 · {self.PROGRAM}") + f" — {G.APP_NAME}")
        G.App.set_icon(self)
        try:
            from .theme import px
            self.geometry(f"{px(860)}x{px(620)}")
            self.configure(background=PALETTE["card"])
            self.transient(app)
        except Exception:  # noqa: BLE001
            pass
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.v = {}
        saved = app.config_data.get(self.CFG, {}) if stage is None else dict(stage.get("settings", {}))
        for k, d in self.DEFAULTS.items():
            val = saved.get(k, d)
            if isinstance(d, bool):
                self.v[k] = tk.BooleanVar(value=bool(val))
            elif k in self.MAPPED:                   # 下拉选项：显示译文，读取时仍为原始取值
                self.v[k] = G.MappedVar(str(val), {o: T(o) for o in self.MAPPED[k]})
            else:
                self.v[k] = tk.StringVar(value=str(val))
        labels = {k: T(lab) for k, lab, _ in self.CALCS}
        self.calc = G.MappedVar(saved.get("calc", self.CALCS[0][0]), labels)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.build_tabs(S)
        pv = ttk.Frame(self.nb, padding=8)
        self.nb.add(pv, text="预览输入")
        top = ttk.Frame(pv)
        top.pack(fill="x")
        ttk.Label(top, text="文件").pack(side="left")
        self.pv_file = tk.StringVar()
        self.pv_box = ttk.Combobox(top, textvariable=self.pv_file, state="readonly", width=40)
        self.pv_box.pack(side="left", padx=6)
        self.pv_box.bind("<<ComboboxSelected>>", lambda e: self._show_preview_file())
        ttk.Button(top, text="刷新", command=S(self.preview)).pack(side="left")
        ttk.Button(top, text="复制到剪贴板", command=S(self._copy_preview)).pack(side="left", padx=4)
        self.pv_text = G.styled_text(pv, app.mono, height=18, wrap="none", size=9)
        self.pv_text.pack(fill="both", expand=True, pady=(6, 0))
        self._pv_files = {}
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._tab_changed())
        bar = ttk.Frame(self, style="DlgBar.TFrame", padding=(10, 6))
        bar.pack(fill="x", side="bottom")
        tl = ttk.Frame(bar, style="DlgBar.TFrame")
        tl.pack(fill="x")
        ttk.Label(tl, text="任务名 (英文、无空格)", style="DlgBar.TLabel").pack(side="left")
        ttk.Entry(tl, textvariable=self.v["name"], width=36).pack(side="left", padx=6, fill="x", expand=True)
        bb = ttk.Frame(bar, style="DlgBar.TFrame")
        bb.pack(fill="x", pady=(6, 0))
        if stage is None:
            btns = (("运行", self.run, "Accent.TButton"), ("写出输入文件", self.write_inputs, "TButton"),
                    ("恢复默认", self.restore_defaults, "TButton"), ("关闭", self.close, "TButton"))
        else:
            btns = (("确定", self.ok, "Accent.TButton"), ("恢复默认", self.restore_defaults, "TButton"),
                    ("取消", self.destroy, "TButton"))
            self.protocol("WM_DELETE_WINDOW", self.destroy)
            try:
                self.grab_set()
            except Exception:  # noqa: BLE001
                pass
        for txt, fn, sty in btns:
            ttk.Button(bb, text=txt, style=sty, command=S(fn)).pack(side="left", expand=True, padx=4)
        if saved.get("addin") and hasattr(self, "addin"):
            self.addin.insert("1.0", saved["addin"])
        self._calc_changed()
        try:
            self.after_idle(self._reflow)
        except Exception:  # noqa: BLE001
            pass

    def _reflow(self):
        from .theme import _reflow, px
        for i in range(len(self.nb.tabs())):
            try:
                w = self.nametowidget(self.nb.tabs()[i])
                _reflow(w, px(820))
            except Exception:  # noqa: BLE001
                pass

    # ---- 子类实现 ----
    def build_tabs(self, S):
        raise NotImplementedError

    def generate(self, outdir):
        """写出输入文件到 outdir，返回 (文件列表, 说明)。"""
        raise NotImplementedError

    def sources(self, calc):
        return []

    # ---- 公共部件 ----
    def page(self, title):
        f = ttk.Frame(self.nb, padding=(12, 10))
        self.nb.add(f, text=title)
        return f

    def row(self, parent, r, label, key, values=None, width=24, col=0, readonly=False):
        ttk.Label(parent, text=label).grid(row=r, column=col, sticky="w", padx=(0, 8), pady=3)
        if values is not None:
            w = ttk.Combobox(parent, textvariable=self.v[key], values=values, width=width,
                             state="readonly" if readonly else "normal")
        else:
            w = ttk.Entry(parent, textvariable=self.v[key], width=width)
        w.grid(row=r, column=col + 1, sticky="we", pady=3)
        parent.columnconfigure(col + 1, weight=1)
        return w

    def check(self, parent, r, label, key, col=0):
        c = ttk.Checkbutton(parent, text=label, variable=self.v[key])
        c.grid(row=r, column=col, columnspan=2, sticky="w", pady=3)
        return c

    def calc_key(self):
        return self.calc.get()

    def _calc_changed(self):
        key = self.calc_key()
        desc = next((d for k, _, d in self.CALCS if k == key), "")
        try:
            self.desc.configure(text=T(desc))
            src = self.sources(key)
            self.src_box.configure(values=src)          # 包装层显示译文
            if self.v["source"].get() not in src and src:
                self.v["source"].set(src[0])
            self._source_changed()
        except Exception:  # noqa: BLE001
            pass
        self._show_options(key)

    def _show_options(self, key):
        pass

    def _source_changed(self):
        pass

    def source(self):
        return self.v["source"].get()

    def values(self):
        d = {k: v.get() for k, v in self.v.items()}
        d["calc"] = self.calc_key()
        if hasattr(self, "addin"):
            d["addin"] = self.addin.get("1.0", "end").strip()
        return d

    def ok(self):
        """流程阶段模式：保存设置并关闭。"""
        v = self.values()
        v["name"] = self.job_name()
        self.stage["settings"] = v
        if self.on_ok:
            self.on_ok(self.stage)
        self.destroy()

    def save(self):
        if self.stage is not None:
            return
        self.app.config_data[self.CFG] = self.values()
        G.save_config(self.app.config_data)

    def restore_defaults(self):
        for k, d in self.DEFAULTS.items():
            if k in ("name", "outdir"):
                continue
            self.v[k].set(d if isinstance(d, bool) else str(d))
        self._calc_changed()
        self.app.log(f"运行 {self.PROGRAM}：已恢复默认设置")

    def close(self):
        try:
            self.save()
        except Exception:  # noqa: BLE001
            pass
        self.withdraw()

    def job_name(self):
        n = self.v["name"].get().strip()
        if not n:
            try:
                n = current_molecule(self.app)[1]
            except Exception:  # noqa: BLE001
                n = "job"
            n = "".join(c if c.isalnum() or c in "-_" else "_" for c in n)[:40] or "job"
            self.v["name"].set(n)
        if any(ord(c) > 127 or c.isspace() for c in n):
            raise ValueError("任务名只能用英文字母、数字、- 和 _ (Gaussian / ADF 不支持中文和空格)")
        return n

    def output_dir(self, ask=True):
        d = self.v["outdir"].get().strip()
        if not d and ask:
            base = self.app.config_data.get("workdir") or str(Path.home())
            d = filedialog.askdirectory(parent=self, title=T("选择输出目录"), initialdir=base)
            if d:
                self.v["outdir"].set(d)
        if not d:
            raise ValueError("请在“资源”标签页选择输出目录")
        probs = G.path_problems(d)
        if probs:
            self.app.log("[警告] " + "；".join(probs))
        return Path(d)

    def _add_input(self, files):
        add_input(files, self.addin.get("1.0", "end") if hasattr(self, "addin") else "")

    # ---- 动作 ----
    def write_inputs(self, ask=True):
        self.save()
        d = self.output_dir(ask=ask)
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        note = self.generate(d) or ""
        files = _new_files(d, t0)
        self._add_input(files)
        G.record_recent(str(d))
        self.app.log(f"运行 {self.PROGRAM}：已在 {d} 写出 {len(files)} 个文件 ({T(dict((k, n) for k, n, _ in self.CALCS)[self.calc_key()])})"
                     + (f"；{note}" if note else ""))
        return d, files

    def run(self):
        d, files = self.write_inputs()
        rd = self.app.open_runner(str(d))
        if rd is None:
            return d
        try:
            rd.dir.set(str(d))
            rd.load()
        except Exception:  # noqa: BLE001
            pass
        n = len([f for f in files if f.suffix.lower() in (".gjf", ".sh")])
        if n and messagebox.askyesno(T("运行"), T(f"已写出输入文件。现在在本机依次运行 {n} 个任务吗？\n"
                                                  "(选“否”可在任务管理器中再选择要运行的任务)")):
            rd.run()
        return d

    def preview(self):
        tmp = Path(tempfile.mkdtemp(prefix="mobilis_preview_"))
        name = self.job_name()
        try:
            self.generate(tmp)
        finally:
            pass
        files = _new_files(tmp, 0)
        self._add_input(files)
        self._pv_files = {str(f.relative_to(tmp)): f for f in files}
        keys = list(self._pv_files)
        self.pv_box.configure(values=keys)
        if keys:
            first = next((k for k in keys if k.endswith((".gjf", ".run"))), keys[0])
            self.pv_file.set(first)
        self._show_preview_file()
        self.app.log(f"预览：{name} 将生成 {len(keys)} 个文件")

    def _show_preview_file(self):
        f = self._pv_files.get(self.pv_file.get())
        self.pv_text.delete("1.0", "end")
        if f is not None:
            self.pv_text.insert("end", f.read_text(errors="replace"))

    def _copy_preview(self):
        self.clipboard_clear()
        self.clipboard_append(self.pv_text.get("1.0", "end"))

    def _tab_changed(self):
        try:
            if self.nb.index(self.nb.select()) == len(self.nb.tabs()) - 1:
                G.safe(self.app)(self.preview)()
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------------------------
# 运行 Gaussian
# ----------------------------------------------------------------------------------------------
class RunGaussianDialog(RunProgramDialog):
    PROGRAM = "Gaussian"
    CFG = "run_gaussian"
    CALCS = [
        ("opt", "结构优化 + 频率", "中性分子几何优化与频率 (检查虚频)。生成 1 个 gjf。"),
        ("reorg", "四点法重组能", "中性 / 阳离子 / 阴离子优化与交叉单点 (Link1 串联)，用于重组能 λ 与 IP / EA。"
                               "算完后在 分析 → 重组能 中读取。"),
        ("excited", "激发态 S0 / S1 / T1", "S0、T1 优化 (UB3LYP)、S1 优化 (TD-DFT) 与吸收 / 发射单点，用于光谱、Stokes 位移、"
                                        "辐射速率。算完后在 分析 → 激发态与光谱 中读取。"),
        ("ti", "二聚体转移积分 (投影法)", "单体 A、B 与二聚体 AB 三个单点 (nosymm，保存 chk / fchk)，用二聚体投影法求 J。"
                                    "选“晶体全部近邻二聚体”时为每个等价组生成一套，算完后在 分析 → 转移积分 中读入迁移率表。"),
        ("cp", "相互作用能 (抵消校正)", "二聚体结合能的 counterpoise 计算 (含色散校正)，用于能量框架与晶格能。"),
        ("esp", "ESP 电荷与极化率", "中性 / 阳离子 / 阴离子在中性几何下的单点 (pop=mk)，中性加 polar。"
                                  "用于全晶体 MD 的静电与极化位点能。"),
        ("fcht", "FCHT 黄昆因子", "在已算完的四点法重组能目录中生成 freq(readfc,fcht) 输入，得到黄昆因子与 Duschinsky 矩阵。"),
    ]
    MOL_SRC = ["主窗口当前结构", "结构文件"]
    DIMER_SRC = ["主窗口选中的两个分子", "晶体全部近邻二聚体", "二聚体文件"]
    MAPPED = {"source": MOL_SRC + DIMER_SRC + ["四点法重组能目录"], "carriers": ["空穴 + 电子", "空穴", "电子"],
              "disp": ["GD3BJ", "GD3", "无"]}
    DEFAULTS = dict(name="", outdir="", source="主窗口当前结构", file="", reorg_dir="",
                    charge="0", mult="1", ion_mult="2", carriers="空穴 + 电子", freq=True, nstates="10",
                    freq_excited=False, method="B3LYP", basis="6-31G(d)", sp_method="", sp_basis="",
                    ex_basis="6-311G(d,p)", ex_sp_basis="def2-TZVP", disp="GD3BJ", ecp="SDD", solvent="",
                    nproc="8", mem="8GB", program="g16", extra="")

    def sources(self, calc):
        if calc in ("ti", "cp"):
            return self.DIMER_SRC
        if calc == "fcht":
            return ["四点法重组能目录"]
        return self.MOL_SRC

    def build_tabs(self, S):
        from .gaussian_io import SOLVENTS
        from .reorganization import BASIS_SETS, FUNCTIONALS
        c = self.page("计算")
        ttk.Label(c, text="计算类型").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        cb = ttk.Combobox(c, textvariable=self.calc, values=[lab for _, lab, _ in self.CALCS], state="readonly", width=34)
        cb.grid(row=0, column=1, sticky="we", pady=3)
        cb.bind("<<ComboboxSelected>>", lambda e: self._calc_changed())
        self.desc = ttk.Label(c, text="", foreground=PALETTE["muted"], wraplength=640, justify="left")
        self.desc.grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 8))
        ttk.Label(c, text="来源").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        self.src_box = ttk.Combobox(c, textvariable=self.v["source"], state="readonly", width=34)
        self.src_box.grid(row=2, column=1, sticky="we", pady=3)
        self.src_box.bind("<<ComboboxSelected>>", lambda e: self._source_changed())
        self.file_row = G.FileEntry(c, "文件", filetypes=G.STRUCT_TYPES)
        self.file_row.var = self.v["file"]
        self.file_row.box.configure(textvariable=self.v["file"])
        self.file_row.grid(row=3, column=0, columnspan=4, sticky="we", pady=3)
        self.dir_row = G.FileEntry(c, "重组能目录", mode="dir")
        self.dir_row.var = self.v["reorg_dir"]
        self.dir_row.box.configure(textvariable=self.v["reorg_dir"])
        self.dir_row.grid(row=4, column=0, columnspan=4, sticky="we", pady=3)
        self.src_note = ttk.Label(c, text="", foreground=PALETTE["muted"], wraplength=640, justify="left")
        self.src_note.grid(row=5, column=0, columnspan=4, sticky="w", pady=(0, 6))
        o = ttk.LabelFrame(c, text="电荷与自旋 / 计算选项", padding=8)
        o.grid(row=6, column=0, columnspan=4, sticky="we", pady=4)
        self.opt_rows = {}
        self.opt_rows["charge"] = self.row(o, 0, "电荷", "charge", width=8)
        self.opt_rows["mult"] = self.row(o, 0, "自旋多重度", "mult", width=8, col=2)
        self.opt_rows["ion_mult"] = self.row(o, 1, "离子多重度", "ion_mult", width=8)
        self.opt_rows["carriers"] = self.row(o, 1, "载流子", "carriers", values=["空穴 + 电子", "空穴", "电子"],
                                             width=14, col=2, readonly=True)
        self.opt_rows["nstates"] = self.row(o, 2, "TD 激发态数", "nstates", width=8)
        self.opt_rows["freq"] = self.check(o, 3, "优化后做频率 (重组能分解、FCHT 需要)", "freq")
        self.opt_rows["freq_excited"] = self.check(o, 4, "激发态也做频率 (振动分辨光谱需要，耗时较长)", "freq_excited")
        c.columnconfigure(1, weight=1)

        f = self.page("泛函与基组")
        self.row(f, 0, "泛函", "method", values=FUNCTIONALS)
        self.row(f, 1, "基组", "basis", values=BASIS_SETS)
        self.row(f, 2, "单点泛函 (重组能，空 = 同上)", "sp_method", values=[""] + FUNCTIONALS)
        self.row(f, 3, "单点基组 (重组能，空 = 同上)", "sp_basis", values=[""] + BASIS_SETS)
        self.row(f, 4, "激发态优化基组", "ex_basis", values=BASIS_SETS)
        self.row(f, 5, "激发态单点基组", "ex_sp_basis", values=BASIS_SETS + ["def2-TZVP"])
        self.row(f, 6, "色散校正 (抵消校正)", "disp", values=["GD3BJ", "GD3", "无"])
        self.row(f, 7, "重原子赝势基组", "ecp", values=["SDD", "LANL2DZ", "def2-TZVP"])
        ttk.Label(f, text="含 Ir、Pt 等重原子且使用 Pople 基组时自动改为 GenECP：重原子用赝势基组，其余原子用上面的基组",
                  foreground=PALETTE["muted"], wraplength=640, justify="left").grid(row=8, column=0, columnspan=2,
                                                                                    sticky="w", pady=(8, 0))
        s = self.page("溶剂")
        self.row(s, 0, "PCM 溶剂 (空 = 气相)", "solvent", values=SOLVENTS)
        ttk.Label(s, text="溶剂只用于重组能、结构优化与激发态；转移积分、抵消校正与 ESP 电荷按晶体环境在气相中计算",
                  foreground=PALETTE["muted"], wraplength=640, justify="left").grid(row=1, column=0, columnspan=2,
                                                                                    sticky="w", pady=(8, 0))
        r = self.page("资源")
        self.row(r, 0, "核数 (%nprocshared)", "nproc", width=8)
        self.row(r, 1, "内存 (%mem)", "mem", width=8)
        self.row(r, 2, "Gaussian 程序", "program", values=["g16", "g09"], width=8)
        out = G.FileEntry(r, "输出目录", mode="dir")
        out.var = self.v["outdir"]
        out.box.configure(textvariable=self.v["outdir"])
        out.grid(row=3, column=0, columnspan=2, sticky="we", pady=(10, 3))
        ttk.Label(r, text="“运行”会把任务交给任务管理器 (任务 → 任务管理器) 在本机依次计算；也可只“写出输入文件”，"
                          "连同生成的 run_*.bat 拷到计算服务器上运行",
                  foreground=PALETTE["muted"], wraplength=640, justify="left").grid(row=4, column=0, columnspan=2,
                                                                                    sticky="w", pady=(8, 0))
        a = self.page("附加输入")
        self.row(a, 0, "附加关键词 (加在 route 行末)", "extra", width=40)
        ttk.Label(a, text="附加输入行 (加在每个计算步骤的末尾，如 gen 基组、modredundant)").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 2))
        self.addin = G.styled_text(a, self.app.mono, height=8, size=9)
        self.addin.grid(row=2, column=0, columnspan=2, sticky="nsew")
        a.rowconfigure(2, weight=1)
        a.columnconfigure(1, weight=1)

    def _show_options(self, key):
        show = {"opt": {"charge", "mult"}, "reorg": {"charge", "mult", "ion_mult", "carriers", "freq"},
                "excited": {"charge", "mult", "nstates", "freq_excited"}, "ti": {"charge", "mult"},
                "cp": set(), "esp": set(), "fcht": {"carriers"}}.get(key, set())
        for k, w in getattr(self, "opt_rows", {}).items():
            try:
                w.configure(state="normal" if k in show else "disabled")
                if k == "carriers":
                    w.configure(state="readonly" if k in show else "disabled")
            except Exception:  # noqa: BLE001
                pass

    def _source_changed(self):
        src = self.source()
        st = self.app.struct_tab
        file_on = src in ("结构文件", "二聚体文件")
        for w, on in ((self.file_row, file_on), (self.dir_row, src == "四点法重组能目录")):
            try:                                        # 不需要的行隐藏 (只显示当前计算用得到的选项)
                if on:
                    w.grid()
                else:
                    w.grid_remove()
            except Exception:  # noqa: BLE001
                pass
        if src == "主窗口当前结构":
            note = ("主窗口为晶体时：选中一个分子则用它，否则用结构面板“晶体二聚体”中的参考分子"
                    if st.mols else "使用主窗口中打开的分子")
        elif src == "主窗口选中的两个分子":
            note = f"当前选中 {len(st.sel)} 个分子 (左键点第一个，Ctrl+左键点第二个)"
        elif src == "晶体全部近邻二聚体":
            n = len(st.dimer_res["groups"]) if st.dimer_res else 0
            note = (f"结构面板中已搜索到 {n} 个等价组" if n else
                    "请先在 构建 → 搜索近邻二聚体 (或工具栏“近邻二聚体”)")
        elif src == "二聚体文件":
            note = "二聚体 xyz / gjf：原子按 A 在前、B 在后排列 (自动识别)"
        elif src == "四点法重组能目录":
            note = "选择“四点法重组能”写出的目录 (含 *_neutral.chk 等)；任务名需与当时一致"
        else:
            note = "xyz / gjf / log / fchk"
        try:
            self.src_note.configure(text=T(note))
        except Exception:  # noqa: BLE001
            pass

    def generate(self, outdir):
        v = dict(self.values(), name=self.job_name())
        note = gaussian_generate(outdir, v, snapshot(self.app))
        if v["calc"] == "fcht" and v.get("reorg_dir"):
            self.v["outdir"].set(v["reorg_dir"])
        return note


# ----------------------------------------------------------------------------------------------
# 运行 ADF
# ----------------------------------------------------------------------------------------------
class RunADFDialog(RunProgramDialog):
    PROGRAM = "ADF"
    CFG = "run_adf"
    CALCS = [
        ("ti", "二聚体转移积分 (片段法)", "两个单体片段 + 二聚体，ADF 片段轨道法给出 V、J、S 与位点能。"
                                    "算完后在 分析 → 转移积分 中读取，或由任务管理器自动读取。"),
        ("crystal", "晶体全部近邻二聚体 (批处理)", "为结构面板中搜索到的每个等价组生成二聚体，并写成一个批处理脚本 batch_adf.run："
                                          "单体片段只算一次，断点续算；可在 ADFjobs 中打开运行。"),
    ]
    MAPPED = {"source": ["主窗口选中的两个分子", "二聚体文件", "晶体全部近邻二聚体"],
              "style": ["自动 (按安装版本)", "AMS2020+", "ADF2019 及以前"]}
    DEFAULTS = dict(name="", outdir="", source="主窗口选中的两个分子", file="", functional="PW91", basis="TZP",
                    style="自动 (按安装版本)", quality="Good", chunk="0", extra="")

    def sources(self, calc):
        return ["晶体全部近邻二聚体"] if calc == "crystal" else ["主窗口选中的两个分子", "二聚体文件"]

    def build_tabs(self, S):
        from .adf import ADF_BASIS, ADF_FUNCTIONALS
        c = self.page("计算")
        ttk.Label(c, text="计算类型").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        cb = ttk.Combobox(c, textvariable=self.calc, values=[lab for _, lab, _ in self.CALCS], state="readonly", width=34)
        cb.grid(row=0, column=1, sticky="we", pady=3)
        cb.bind("<<ComboboxSelected>>", lambda e: self._calc_changed())
        self.desc = ttk.Label(c, text="", foreground=PALETTE["muted"], wraplength=640, justify="left")
        self.desc.grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 8))
        ttk.Label(c, text="来源").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        self.src_box = ttk.Combobox(c, textvariable=self.v["source"], state="readonly", width=34)
        self.src_box.grid(row=2, column=1, sticky="we", pady=3)
        self.src_box.bind("<<ComboboxSelected>>", lambda e: self._source_changed())
        self.file_row = G.FileEntry(c, "二聚体文件", filetypes=G.STRUCT_TYPES)
        self.file_row.var = self.v["file"]
        self.file_row.box.configure(textvariable=self.v["file"])
        self.file_row.grid(row=3, column=0, columnspan=4, sticky="we", pady=3)
        self.src_note = ttk.Label(c, text="", foreground=PALETTE["muted"], wraplength=640, justify="left")
        self.src_note.grid(row=4, column=0, columnspan=4, sticky="w", pady=(0, 6))
        self.chunk_row = self.row(c, 5, "每个批处理脚本的二聚体数 (0 = 全部放在一个脚本)", "chunk", width=8)
        c.columnconfigure(1, weight=1)
        f = self.page("泛函与基组")
        self.row(f, 0, "泛函", "functional", values=list(ADF_FUNCTIONALS))
        self.row(f, 1, "基组", "basis", values=ADF_BASIS)
        self.row(f, 2, "数值精度", "quality", values=["Normal", "Good", "VeryGood"])
        self.row(f, 3, "输入格式", "style", values=["自动 (按安装版本)", "AMS2020+", "ADF2019 及以前"])
        ttk.Label(f, text="ADF2016 等旧版用 ADF2019 及以前的格式；“自动”按 任务 → 程序路径设置 中的 ADF 安装判断",
                  foreground=PALETTE["muted"], wraplength=640, justify="left").grid(row=4, column=0, columnspan=2,
                                                                                    sticky="w", pady=(8, 0))
        r = self.page("资源")
        out = G.FileEntry(r, "输出目录", mode="dir")
        out.var = self.v["outdir"]
        out.box.configure(textvariable=self.v["outdir"])
        out.grid(row=0, column=0, columnspan=2, sticky="we", pady=3)
        ttk.Label(r, text="并行核数、临时目录与 ADF 安装位置在 任务 → 程序路径设置 中设定 (所有 ADF 任务共用)",
                  foreground=PALETTE["muted"], wraplength=640, justify="left").grid(row=1, column=0, columnspan=2,
                                                                                    sticky="w", pady=(8, 0))
        r.columnconfigure(1, weight=1)
        a = self.page("附加输入")
        self.row(a, 0, "附加输入 (加入 ADF 输入块)", "extra", width=40)

    def _show_options(self, key):
        try:
            self.chunk_row.configure(state="normal" if key == "crystal" else "disabled")
        except Exception:  # noqa: BLE001
            pass

    def _source_changed(self):
        st = self.app.struct_tab
        src = self.source()
        try:
            if src == "二聚体文件":
                self.file_row.grid()
            else:
                self.file_row.grid_remove()
        except Exception:  # noqa: BLE001
            pass
        if src == "主窗口选中的两个分子":
            note = f"当前选中 {len(st.sel)} 个分子 (左键点第一个，Ctrl+左键点第二个)"
        elif src == "晶体全部近邻二聚体":
            n = len(st.dimer_res["groups"]) if st.dimer_res else 0
            note = f"结构面板中已搜索到 {n} 个等价组" if n else "请先在 构建 → 搜索近邻二聚体"
        else:
            note = "二聚体 xyz / gjf：原子按 A 在前、B 在后排列 (自动识别)"
        try:
            self.src_note.configure(text=T(note))
        except Exception:  # noqa: BLE001
            pass

    def generate(self, outdir):
        v = dict(self.values(), name=self.job_name())
        return adf_generate(outdir, v, snapshot(self.app))


# ----------------------------------------------------------------------------------------------
# 输入生成 (与对话框分离：对话框、流程图与测试共用)。v = 对话框的取值，ctx = 结构快照 (见 snapshot)
# ----------------------------------------------------------------------------------------------
def snapshot(app):
    """在界面线程中取下主窗口结构的快照，供后台线程 / 流程使用 (不再访问界面控件)。"""
    ctx = {"ams_home": app.config_data.get("ams_home"), "molecule": None, "dimer": None, "dimer_res": None,
           "crystal": None}
    st = app.struct_tab
    try:
        ctx["molecule"] = current_molecule(app)
    except Exception:  # noqa: BLE001
        pass
    try:
        ctx["dimer"] = current_dimer(app)
    except Exception:  # noqa: BLE001
        pass
    ctx["dimer_res"] = st.dimer_res
    if st.mols:
        ctx["crystal"] = (st.crys, st.mols)
        try:
            ctx["cutoff"], ctx["criterion"] = float(st.cut.get()), \
                ("atom" if st.crit.get().startswith("最短") else "centroid")
            ctx["ref"] = int(st.ref.get() or 1) - 1
        except Exception:  # noqa: BLE001
            pass
    return ctx


from .run_inputs import (_dimer, _molecule, _need_dimer_res, add_input, adf_generate, adf_style,  # noqa: E402,F401
                         gaussian_generate)
