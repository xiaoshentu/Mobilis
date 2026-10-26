"""流程图窗口：把多个计算阶段串成一个任务，一次运行。

左侧：开始节点 + 依次相连的阶段框 (颜色表示状态)；单击选中，双击打开该阶段的设置
       (Gaussian / ADF 阶段打开与“运行 Gaussian / 运行 ADF”相同的对话框，底部为 确定 / 取消)。
右侧：添加阶段、模板、上移 / 下移 / 删除。
底部：任务标题、输出目录、运行 / 停止 / 打开 / 保存 / 关闭，以及运行记录。
"""
import queue
import time
from pathlib import Path

from . import flow as F
from . import gui as G
from .gui import T, filedialog, messagebox, tk, ttk

PALETTE = G.PALETTE

GAUSS_STAGES = [("opt", "结构优化", "主窗口当前结构"), ("reorg", "重组能 (四点法)", "主窗口当前结构"),
                ("excited", "激发态", "主窗口当前结构"), ("ti", "晶体全部近邻 J", "晶体全部近邻二聚体"),
                ("ti", "二聚体 J", "主窗口选中的两个分子"), ("cp", "相互作用能 (BSSE)", "晶体全部近邻二聚体"),
                ("esp", "ESP 电荷", "主窗口当前结构"), ("fcht", "FCHT 黄昆因子", "四点法重组能目录")]
ADF_STAGES = [("crystal", "晶体全部近邻 J", "晶体全部近邻二聚体"), ("ti", "二聚体 J", "主窗口选中的两个分子")]
LOCAL_STAGES = ["dimers", "eht", "mobility", "kmc"]

TEMPLATES = {
    "EHT 快速筛选：二聚体 → EHT → Marcus → KMC": [
        ("dimers", {}), ("eht", {}), ("mobility", {}), ("kmc", {})],
    "Gaussian 完整流程：优化 → 重组能 → 二聚体 → 晶体 J → 迁移率": [
        ("gaussian", {"calc": "opt", "source": "主窗口当前结构"}),
        ("gaussian", {"calc": "reorg", "source": "主窗口当前结构"}),
        ("dimers", {}), ("gaussian", {"calc": "ti", "source": "晶体全部近邻二聚体"}), ("mobility", {})],
    "ADF 流程：Gaussian 重组能 → 二聚体 → ADF 晶体 J → 迁移率": [
        ("gaussian", {"calc": "reorg", "source": "主窗口当前结构"}),
        ("dimers", {}), ("adf", {"calc": "crystal", "source": "晶体全部近邻二聚体"}), ("mobility", {})],
}

# 软件内阶段的设置项：(键, 标签, 类型 / 选项)
LOCAL_FIELDS = {
    "dimers": [("cutoff", "截断距离 (Å)", float), ("criterion", "判据", ["atom", "centroid"]), ("ref", "参考分子 (序号)", int)],
    "eht": [("cutoff", "截断距离 (Å)", float), ("criterion", "判据", ["atom", "centroid"]), ("ref", "参考分子 (序号)", int),
            ("use_flow_dimers", "使用前面阶段搜索的二聚体", bool)],
    "mobility": [("T", "温度 (K)", float), ("use_flow_lambda", "λ 取自前面的重组能阶段", bool),
                 ("lam_h", "空穴 λ (eV，没有重组能阶段时)", float), ("lam_e", "电子 λ (eV，没有重组能阶段时)", float)],
    "kmc": [("carrier", "载流子", ["hole", "electron"]), ("T", "温度 (K)", float),
            ("use_flow_lambda", "λ 取自前面的重组能阶段", bool), ("lam_h", "空穴 λ (eV)", float),
            ("lam_e", "电子 λ (eV)", float), ("walkers", "载流子数", int), ("steps", "步数", int), ("seed", "随机数种子", int)],
}
CHOICE_LABEL = {"atom": "最短原子间距", "centroid": "质心距离", "hole": "空穴", "electron": "电子"}

STATUS_COLOR = {"pending": "border", "running": "primary", "done": "ok", "failed": "bad", "skipped": "muted"}
STATUS_TEXT = {"pending": "", "running": "运行中…", "done": "完成", "failed": "失败", "skipped": "跳过"}


class FlowWindow(tk.Toplevel):
    _inst = {}

    @classmethod
    def open(cls, app):
        w = cls._inst.get(id(app))
        try:
            alive = w is not None and bool(w.winfo_exists())
        except Exception:  # noqa: BLE001
            alive = False
        if not alive:
            w = cls(app)
            cls._inst[id(app)] = w
        else:
            w.deiconify()
            w.lift()
        return w

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        S = self.S = G.safe(app)
        self.title(T("流程图") + f" — {G.APP_NAME}")
        G.App.set_icon(self)
        from .theme import px
        self.px = px
        try:
            self.geometry(f"{px(1080)}x{px(720)}")
            self.configure(background=PALETTE["bg"])
            self.transient(app)
        except Exception:  # noqa: BLE001
            pass
        self.protocol("WM_DELETE_WINDOW", self.close)
        cfg = app.config_data.get("flow", {})
        self.stages = cfg.get("stages") or [F.new_stage(k, s) for k, s in TEMPLATES[next(iter(TEMPLATES))]]
        self.status = ["pending"] * len(self.stages)
        self.msgs = [""] * len(self.stages)
        self.sel = None
        self.runner = None
        self.q = queue.Queue()
        self.flow_path = None

        head = ttk.Frame(self, style="PageHead.TFrame", padding=(14, 8, 14, 4))
        head.pack(fill="x")
        ttk.Label(head, text=f"{T('任务')}  ›  {T('流程图')}", style="PageCrumb.TLabel").pack(anchor="w")
        ttk.Label(head, text="流程图 (多步串联计算)", style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(head, text="把多个计算阶段串成一个任务：前一阶段的结果 (优化结构、λ、各二聚体 J) 自动交给后一阶段。"
                             "双击阶段修改设置。", style="PageDesc.TLabel").pack(anchor="w")

        bar = ttk.Frame(self, style="DlgBar.TFrame", padding=(10, 6))
        bar.pack(fill="x", side="bottom")
        r1 = ttk.Frame(bar, style="DlgBar.TFrame")
        r1.pack(fill="x")
        self.title_var = tk.StringVar(value=cfg.get("title", "mobility_flow"))
        self.dir_var = tk.StringVar(value=cfg.get("outdir", ""))
        ttk.Label(r1, text="任务标题 (英文、无空格)", style="DlgBar.TLabel").pack(side="left")
        ttk.Entry(r1, textvariable=self.title_var, width=24).pack(side="left", padx=6)
        ttk.Label(r1, text="输出目录", style="DlgBar.TLabel").pack(side="left", padx=(10, 0))
        ttk.Entry(r1, textvariable=self.dir_var, width=36).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(r1, text="浏览…", command=S(self.browse)).pack(side="left")
        r2 = ttk.Frame(bar, style="DlgBar.TFrame")
        r2.pack(fill="x", pady=(6, 0))
        self.run_btn = ttk.Button(r2, text="运行", style="Accent.TButton", command=S(self.run))
        self.run_btn.pack(side="left", expand=True, padx=4)
        for txt, fn in (("停止", self.stop), ("检查流程", self.check), ("打开流程…", self.load), ("保存流程…", self.save_as),
                        ("打开输出目录", self.open_dir), ("关闭", self.close)):
            ttk.Button(r2, text=txt, command=S(fn)).pack(side="left", expand=True, padx=4)

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=6)
        left = ttk.Frame(body)
        body.add(left, weight=3)
        self.canvas = tk.Canvas(left, background=PALETTE["field"], highlightthickness=1,
                                highlightbackground=PALETTE["border"], borderwidth=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Double-Button-1>", self._dclick)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.log_text = G.styled_text(left, app.mono, height=8, size=9)
        self.log_text.pack(fill="x", pady=(6, 0))

        right = ttk.Frame(body, padding=(6, 0))
        body.add(right, weight=1)
        self._build_palette(right)
        self.bind("<Delete>", lambda e: S(self.delete)())
        self._after = self.after(150, self._poll)

    # ---- 右侧：添加阶段 ----
    def _build_palette(self, f):
        S = self.S
        g = ttk.LabelFrame(f, text="模板", padding=6)
        g.pack(fill="x", pady=(0, 6))
        self.tpl = tk.StringVar(value=next(iter(TEMPLATES)))
        cb = ttk.Combobox(g, textvariable=self.tpl, values=list(TEMPLATES), state="readonly", width=34)
        cb.pack(fill="x")
        ttk.Button(g, text="使用模板 (替换当前流程)", command=S(self.use_template)).pack(fill="x", pady=(4, 0))
        for title, items, kind in (("添加 Gaussian 阶段", GAUSS_STAGES, "gaussian"), ("添加 ADF 阶段", ADF_STAGES, "adf")):
            g = ttk.LabelFrame(f, text=title, padding=6)
            g.pack(fill="x", pady=(0, 6))
            for i, (calc, lab, src) in enumerate(items):
                ttk.Button(g, text=lab, command=S(lambda k=kind, c=calc, s=src: self.add(k, {"calc": c, "source": s}))
                           ).grid(row=i // 2, column=i % 2, sticky="we", padx=2, pady=2)
            g.columnconfigure(0, weight=1)
            g.columnconfigure(1, weight=1)
        g = ttk.LabelFrame(f, text="添加软件内阶段", padding=6)
        g.pack(fill="x", pady=(0, 6))
        for i, k in enumerate(LOCAL_STAGES):
            ttk.Button(g, text=F.STAGE_TYPES[k][0], command=S(lambda k=k: self.add(k))).grid(
                row=i // 2, column=i % 2, sticky="we", padx=2, pady=2)
        g.columnconfigure(0, weight=1)
        g.columnconfigure(1, weight=1)
        g = ttk.LabelFrame(f, text="所选阶段", padding=6)
        g.pack(fill="x")
        for i, (txt, fn) in enumerate((("设置…", self.edit), ("删除", self.delete), ("上移", lambda: self.move(-1)),
                                       ("下移", lambda: self.move(+1)))):
            ttk.Button(g, text=txt, command=S(fn)).grid(row=i // 2, column=i % 2, sticky="we", padx=2, pady=2)
        g.columnconfigure(0, weight=1)
        g.columnconfigure(1, weight=1)

    # ---- 画流程图 ----
    def draw(self):
        c, px = self.canvas, self.px
        c.delete("all")
        w = max(c.winfo_width(), px(300))
        bw, bh, gap = min(px(380), w - px(60)), px(54), px(30)
        x0 = (w - bw) // 2
        cx = w // 2
        y = px(18)
        P = PALETTE
        fam = "Microsoft YaHei UI" if G.sys.platform.startswith("win") else "TkDefaultFont"
        try:
            import tkinter.font as tkfont
            base = tkfont.nametofont("TkDefaultFont").actual()
            fam, size = base["family"], abs(int(base["size"])) or 9
        except Exception:  # noqa: BLE001
            size = 9
        f_b, f_n = (fam, size, "bold"), (fam, size)
        r = px(16)
        c.create_oval(cx - r, y, cx + r, y + 2 * r, fill=P["primary"], outline="")
        c.create_text(cx, y + r, text=T("开始"), fill="#FFFFFF", font=(fam, max(size - 1, 7), "bold"))
        y += 2 * r
        self._boxes = []
        for n, st in enumerate(self.stages):
            c.create_line(cx, y, cx, y + gap, fill=P["muted"], width=max(1, px(1.5)), arrow="last")
            y += gap
            stat = self.status[n] if n < len(self.status) else "pending"
            col = P.get(STATUS_COLOR[stat], P["border"])
            sel = n == self.sel
            c.create_rectangle(x0, y, x0 + bw, y + bh, fill=P["primary_light"] if sel else P["card"],
                               outline=P["primary"] if sel else P["border"], width=px(2) if sel else 1)
            c.create_rectangle(x0, y, x0 + px(6), y + bh, fill=col, outline="")
            c.create_text(x0 + px(16), y + px(8), anchor="nw", text=f"{n + 1}.  {T(F.stage_title(st))}",
                          fill=P["text"], font=f_b)
            sub = T(F.stage_summary(st))
            c.create_text(x0 + px(16), y + px(30), anchor="nw", text=sub, fill=P["muted"], font=f_n)
            if stat != "pending":
                c.create_text(x0 + bw - px(10), y + px(8), anchor="ne", text=T(STATUS_TEXT[stat]),
                              fill=col, font=f_b)
            self._boxes.append((x0, y, x0 + bw, y + bh))
            y += bh
        if not self.stages:
            c.create_text(cx, y + px(40), text=T("在右侧添加阶段，或选择一个模板"), fill=P["muted"], font=f_n)
        c.configure(scrollregion=(0, 0, w, y + px(30)))

    def _hit(self, e):
        x, y = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        for n, (a, b, c, d) in enumerate(getattr(self, "_boxes", [])):
            if a <= x <= c and b <= y <= d:
                return n
        return None

    def _click(self, e):
        self.sel = self._hit(e)
        self.draw()

    def _dclick(self, e):
        self.sel = self._hit(e)
        if self.sel is not None:
            self.S(self.edit)()

    # ---- 编辑 ----
    def _reset_status(self):
        self.status = ["pending"] * len(self.stages)
        self.msgs = [""] * len(self.stages)

    def _busy(self):
        if self.runner is not None and self.runner.thread is not None and self.runner.thread.is_alive():
            raise RuntimeError("流程正在运行，请先停止")

    def add(self, kind, settings=None):
        self._busy()
        s = {}
        if kind in ("gaussian", "adf"):
            from .gui_run import RunADFDialog, RunGaussianDialog
            cls = RunGaussianDialog if kind == "gaussian" else RunADFDialog
            s = dict(cls.DEFAULTS)
            s.update({k: v for k, v in self.app.config_data.get(cls.CFG, {}).items() if k in cls.DEFAULTS or k == "addin"})
            s["name"] = s.get("name") or self.title_var.get() or "job"
        s.update(settings or {})
        st = F.new_stage(kind, s)
        pos = len(self.stages) if self.sel is None else self.sel + 1
        self.stages.insert(pos, st)
        self.sel = pos
        self._reset_status()
        self.draw()
        return st

    def use_template(self):
        self._busy()
        self.stages, self.sel = [], None
        for k, s in TEMPLATES[self.tpl.get()]:
            self.add(k, s)
        self.sel = None
        self.draw()

    def delete(self):
        self._busy()
        if self.sel is None:
            raise ValueError("请先单击选中一个阶段")
        del self.stages[self.sel]
        self.sel = min(self.sel, len(self.stages) - 1) if self.stages else None
        self._reset_status()
        self.draw()

    def move(self, d):
        self._busy()
        if self.sel is None:
            raise ValueError("请先单击选中一个阶段")
        j = self.sel + d
        if 0 <= j < len(self.stages):
            self.stages[self.sel], self.stages[j] = self.stages[j], self.stages[self.sel]
            self.sel = j
            self._reset_status()
            self.draw()

    def edit(self):
        if self.sel is None:
            raise ValueError("请先单击选中一个阶段")
        st = self.stages[self.sel]
        if st["type"] in ("gaussian", "adf"):
            from .gui_run import RunADFDialog, RunGaussianDialog
            cls = RunGaussianDialog if st["type"] == "gaussian" else RunADFDialog
            return cls(self.app, stage=st, on_ok=lambda s: self.draw())
        return LocalStageDialog(self, st)

    # ---- 文件 ----
    def browse(self):
        d = filedialog.askdirectory(parent=self, title=T("选择输出目录"),
                                    initialdir=self.dir_var.get() or self.app.config_data.get("workdir") or str(Path.home()))
        if d:
            self.dir_var.set(d)

    def save_as(self, path=None):
        path = path or filedialog.asksaveasfilename(parent=self, title=T("保存流程"), defaultextension=".mflow",
                                                    filetypes=[(T("Mobilis 流程"), "*.mflow"), (T("全部文件"), "*.*")])
        if path:
            F.save_flow(path, self.stages, self.title_var.get())
            self.flow_path = path
            self.log(f"已保存流程：{path}")
        return path

    def load(self, path=None):
        self._busy()
        path = path or filedialog.askopenfilename(parent=self, title=T("打开流程"),
                                                  filetypes=[(T("Mobilis 流程"), "*.mflow *.json"), (T("全部文件"), "*.*")])
        if path:
            title, self.stages = F.load_flow(path)
            if title:
                self.title_var.set(title)
            self.sel, self.flow_path = None, path
            self._reset_status()
            self.draw()
            self.log(f"已打开流程：{path} ({len(self.stages)} 个阶段)")
        return path

    def open_dir(self):
        d = self.job_dir()
        if not d.exists():
            raise FileNotFoundError(f"输出目录还不存在：{d}")
        import os
        import subprocess
        if G.sys.platform.startswith("win"):
            os.startfile(str(d))  # noqa: S606
        else:
            subprocess.Popen(["open" if G.sys.platform == "darwin" else "xdg-open", str(d)])

    def remember(self):
        self.app.config_data["flow"] = {"stages": self.stages, "title": self.title_var.get(), "outdir": self.dir_var.get()}
        try:
            G.save_config(self.app.config_data)
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        self.remember()
        self.withdraw()

    # ---- 运行 ----
    def log(self, text):
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

    def job_dir(self):
        t = self.title_var.get().strip() or "mobility_flow"
        if any(ord(ch) > 127 or ch.isspace() for ch in t):
            raise ValueError("任务标题只能用英文字母、数字、- 和 _")
        d = self.dir_var.get().strip()
        if not d:
            self.browse()
            d = self.dir_var.get().strip()
        if not d:
            raise ValueError("请选择输出目录")
        return Path(d) / t

    def check(self):
        probs = F.check_flow(self.stages)
        from .gui_run import snapshot
        ctx = snapshot(self.app)
        if any(s["type"] in ("dimers", "eht", "kmc") for s in self.stages) and not ctx.get("crystal"):
            probs.insert(0, "流程中有晶体阶段，但主窗口还没有打开晶体 (CIF)")
        if any(s["type"] == "gaussian" and s["settings"].get("source") == "主窗口当前结构" for s in self.stages) \
                and not ctx.get("molecule") and not ctx.get("crystal"):
            probs.insert(0, "流程中的 Gaussian 阶段使用主窗口当前结构，但主窗口还没有结构")
        if any(s["type"] in ("gaussian", "adf") and s["settings"].get("source") == "晶体全部近邻二聚体" for s in self.stages) \
                and not ctx.get("dimer_res") and not any(s["type"] in ("dimers", "eht") for s in self.stages):
            probs.insert(0, "需要近邻二聚体：请加入“搜索近邻二聚体”阶段")
        if probs:
            self.log("检查流程：\n  " + "\n  ".join(T(p) for p in probs))
        else:
            self.log("检查流程：没有发现问题")
        return probs, ctx

    def run(self):
        self._busy()
        if not self.stages:
            raise ValueError("流程中还没有阶段")
        probs, ctx = self.check()
        hard = [p for p in probs if "还没有" in p or "需要" in p]
        if hard and not self.app.sync:
            if not messagebox.askyesno(T("流程图"), T("检查发现以下问题，仍然运行吗？\n\n") + "\n".join(T(p) for p in hard)):
                return None
        d = self.job_dir()
        if d.exists() and any(d.iterdir()) and not self.app.sync:
            if not messagebox.askyesno(T("流程图"), T(f"输出目录 {d} 已有文件，继续运行会覆盖同名文件。继续吗？")):
                return None
        self.remember()
        d.mkdir(parents=True, exist_ok=True)
        F.save_flow(d / f"{d.name}.mflow", self.stages, self.title_var.get())
        from .jobs import find_gaussian
        from .programs import find_ams
        need_g = any(s["type"] == "gaussian" for s in self.stages)
        need_a = any(s["type"] == "adf" for s in self.stages)
        gi = find_gaussian(self.app.config_data.get("gaussian_exe") or None) if need_g else None
        ai = find_ams(self.app.config_data.get("ams_home") or None) if need_a else None
        if need_g and not gi:
            raise FileNotFoundError("没有找到 Gaussian：请在 任务 → 程序路径设置 中指定 g16 / g09")
        if need_a and not ai:
            raise FileNotFoundError("没有找到 ADF / AMS：请在 任务 → 程序路径设置 中指定安装目录")
        self._reset_status()
        self.draw()
        self.log(f"运行流程 {d.name}：{len(self.stages)} 个阶段，输出在 {d}")
        self.app.log(f"流程图：开始运行 {d.name} ({len(self.stages)} 个阶段)")
        q = self.q
        self.t0 = time.time()
        self.runner = F.FlowRunner(self.stages, d, ctx, gaussian=gi, ams=ai,
                                   on_status=lambda n, s, m: q.put(("status", n, s, m)),
                                   on_log=lambda t: q.put(("log", t)),
                                   on_done=lambda c, ok: q.put(("done", c, ok)))
        if self.app.sync:
            self.runner.run()
            self._poll(once=True)
        else:
            self.runner.start()
            try:
                self.run_btn.state(["disabled"])
            except Exception:  # noqa: BLE001
                pass
        return self.runner

    def stop(self):
        if self.runner is not None:
            self.runner.stop()
            self.log("正在停止 (当前计算结束后停止)…")

    def _poll(self, once=False):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "log":
                    self.log(item[1])
                elif item[0] == "status":
                    _, n, s, m = item
                    if n < len(self.status):
                        self.status[n], self.msgs[n] = s, m
                    self.draw()
                elif item[0] == "done":
                    self._finished(item[1], item[2])
        except queue.Empty:
            pass
        except tk.TclError:
            return
        if not once:
            try:
                self._after = self.after(200, self._poll)
            except Exception:  # noqa: BLE001
                pass

    def destroy(self):
        try:
            self.after_cancel(self._after)
        except Exception:  # noqa: BLE001
            pass
        super().destroy()

    def _finished(self, ctx, ok):
        try:
            self.run_btn.state(["!disabled"])
        except Exception:  # noqa: BLE001
            pass
        dt = time.time() - getattr(self, "t0", time.time())
        self.log(("流程完成" if ok else "流程未完成") + f" ({dt:.1f} s)；结果汇总：flow_results.json")
        self.app.log(f"流程图：{'完成' if ok else '未完成'}，结果在 {self.runner.workdir}")
        self.result_ctx = ctx
        if ctx.get("couplings") and ctx.get("dimer_res"):      # 把 J 与 λ 填入迁移率页，便于继续分析
            try:
                st, res, coup = self.app.struct_tab, ctx["dimer_res"], ctx["couplings"]
                st.dimer_res = res
                self.app.data["crystal_couplings"] = coup
                mt = self.app.mob_tab
                sc = mt._scale() / 1000.0
                mt.set_paths([{"name": f"{st.group_name(d['group'])}-{i + 1}", "vector": d["vector"],
                               "Ve": abs(coup[d["group"]]["electron"] or 0) * sc,
                               "Vh": abs(coup[d["group"]]["hole"] or 0) * sc}
                              for i, d in enumerate(res["dimers"]) if d["group"] in coup])
                lam = ctx.get("lambda") or {}
                if "hole" in lam:
                    mt.lam_h.set(f"{lam['hole']:.4f}")
                if "electron" in lam:
                    mt.lam_e.set(f"{lam['electron']:.4f}")
                self.log("各二聚体 J" + (" 与 λ" if lam else "") + " 已填入“迁移率”页，可继续做角分辨 / 三维分析")
            except Exception as exc:  # noqa: BLE001
                self.log(f"[提示] 结果未能填入迁移率页：{exc}")


class LocalStageDialog(tk.Toplevel):
    """软件内阶段 (二聚体 / EHT / 迁移率 / KMC) 的设置。"""

    def __init__(self, win, stage):
        super().__init__(win)
        self.win, self.stage = win, stage
        self.title(T("流程阶段") + " · " + T(F.STAGE_TYPES[stage["type"]][0]))
        try:
            self.transient(win)
            self.configure(background=PALETTE["card"])
        except Exception:  # noqa: BLE001
            pass
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=F.STAGE_TYPES[stage["type"]][1], style="Muted.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.vars = {}
        s = stage["settings"]
        for r, (k, lab, typ) in enumerate(LOCAL_FIELDS[stage["type"]], 1):
            if typ is bool:
                v = tk.BooleanVar(value=bool(s.get(k, True)))
                ttk.Checkbutton(body, text=lab, variable=v).grid(row=r, column=0, columnspan=2, sticky="w", pady=3)
            elif isinstance(typ, list):
                v = G.MappedVar(str(s.get(k, typ[0])), {o: T(CHOICE_LABEL.get(o, o)) for o in typ})
                ttk.Label(body, text=lab).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=3)
                ttk.Combobox(body, textvariable=v, values=typ, state="readonly", width=16).grid(row=r, column=1, sticky="we")
            else:
                v = tk.StringVar(value=str(s.get(k, "")))
                ttk.Label(body, text=lab).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=3)
                ttk.Entry(body, textvariable=v, width=18).grid(row=r, column=1, sticky="we")
            self.vars[k] = (v, typ)
        body.columnconfigure(1, weight=1)
        bb = ttk.Frame(self, style="DlgBar.TFrame", padding=(10, 6))
        bb.pack(fill="x", side="bottom")
        S = G.safe(win.app)
        ttk.Button(bb, text="确定", style="Accent.TButton", command=S(self.ok)).pack(side="left", expand=True, padx=4)
        ttk.Button(bb, text="取消", command=self.destroy).pack(side="left", expand=True, padx=4)
        self.bind("<Return>", lambda e: S(self.ok)())
        self.bind("<Escape>", lambda e: self.destroy())

    def ok(self):
        out = {}
        for k, (v, typ) in self.vars.items():
            val = v.get()
            if typ in (int, float):
                try:
                    val = typ(val)
                except ValueError:
                    raise ValueError(f"“{k}” 应为数字") from None
            out[k] = val
        self.stage["settings"].update(out)
        self.win._reset_status()
        self.win.draw()
        self.destroy()
