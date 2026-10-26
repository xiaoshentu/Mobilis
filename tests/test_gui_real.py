"""真实窗口的界面测试 (需要显示器；无显示环境自动跳过)。

smoke_gui.py 用桩对象逐个调用按钮回调，检查计算逻辑；本测试打开真正的 Tk 窗口，检查：
  1. 中文 / 英文界面下 15 个页面都能显示，左侧控制栏的内容没有被截断；
  2. 字体放大 / 缩小后界面不报错；
  3. 一条真实流程：打开 CIF → 搜索二聚体 → EHT 计算全部 J 并填入迁移率页 → 计算迁移率；
  4. 沿 a / b / c 观察后晶格不变 (曾被原地归一化成 1 Å)；
  5. 菜单栏与工具栏的每一项都能执行 (没有程序错误；“请先…”之类的提示不算错误)；
  6. 运行 Gaussian / 运行 ADF 对话框：每种计算类型都能写出输入文件，预览页能显示；
  7. 流程图：模板、添加 / 移动 / 删除阶段、阶段设置对话框 (确定)、运行软件内流程并把 J 填入迁移率页、保存 / 打开。
用法: python tests/test_gui_real.py [--keep]   (--keep: 结束后不关闭窗口，便于目测)
"""
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import tkinter as tk
except ImportError:
    print("跳过 (Python 没有 tkinter)")
    sys.exit(0)
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    print("跳过 (没有显示器；Linux 上可用 xvfb-run 运行)")
    sys.exit(0)
try:
    tk.Tk().destroy()
except tk.TclError as exc:
    print(f"跳过 (无法打开窗口: {exc})")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp())
from mobility_tool import gui  # noqa: E402
from mobility_tool import theme  # noqa: E402

gui.CONFIG = TMP / "config.json"            # 不改动用户自己的设置
ERRORS = []
gui._messagebox.showerror = lambda t, m, **k: ERRORS.append(f"{t}: {m}")
gui._messagebox.showinfo = lambda *a, **k: None
gui._messagebox.showwarning = lambda *a, **k: None
gui._messagebox.askyesno = lambda *a, **k: False
gui._filedialog.askopenfilename = lambda **k: ""
gui._filedialog.asksaveasfilename = lambda **k: ""
gui._filedialog.askdirectory = lambda **k: ""
EXPECTED = (ValueError, FileNotFoundError, RuntimeError)     # 给用户的操作提示
UNEXPECTED = []
_safe = gui.safe


def _safe_recording(app):
    deco = _safe(app)

    def outer(fn):
        def wrapper(*a, **k):
            try:
                return fn(*a, **k)
            except EXPECTED:
                return deco(fn)(*a, **k)
            except Exception:  # noqa: BLE001
                import traceback
                UNEXPECTED.append(f"{getattr(fn, '__name__', fn)}: {traceback.format_exc()}")
        return wrapper
    return outer


gui.safe = _safe_recording


def report_callback_exception(self, exc, val, tb):
    import traceback
    ERRORS.append("".join(traceback.format_exception(exc, val, tb)))


tk.Tk.report_callback_exception = report_callback_exception


def new_app(lang, font_scale=1.0):
    gui.save_config({"lang": lang, "font_scale": font_scale})
    app = gui.App()
    app.sync = True
    app.geometry("1440x900+0+0")
    app.update()
    app._reflow_panels()
    app.update()
    return app


def clipped_pages(app):
    """分区面板内容比面板宽 → 右侧被截断。逐个打开功能窗口、逐个切换分区标签页后测量。返回 [(页面, 分区, 需要宽度, 实际宽度)]。"""
    bad, n = [], 0
    for name in app.STEPS:
        tab = app.show_page(name)
        app.update()
        sp = gui.section_panel(tab) if name != "结构/晶体" else app.struct_tab.panel
        win = app._wins.get(name)
        if win is not None:
            win.deiconify()
            win.lift()
        for i, (title, page) in enumerate(sp.pages):
            sp.nb.select(page)
            for _ in range(20):               # Windows 上新窗口要几轮事件才完成布局 (宽度为 1 = 还没显示)
                app.update()
                if page.canvas.winfo_width() > 1:
                    break
                time.sleep(0.05)
            need, have = page.inner.winfo_reqwidth(), page.canvas.winfo_width()
            if have <= 1:
                continue                      # 窗口没有显示出来 (例如被最小化)，无法测量
            n += 1
            if need > have * 1.15 + 10:
                bad.append((name, title, need, have))
        if name in app._wins:
            app._wins[name].withdraw()
    if n < 10:
        bad.append(("只测量到", n, "个分区", "窗口没有显示出来"))
    return bad


def check(cond, msg):
    if ERRORS:
        raise SystemExit(f"[失败] {msg}:\n" + "\n".join(ERRORS))
    if not cond:
        raise SystemExit(f"[失败] {msg}")
    print(f"[通过] {msg}")


def test_pages():
    for lang in ("en", "zh"):                 # 先英文后中文：同一时间只保留一个主窗口
        app = new_app(lang)
        bad = clipped_pages(app)
        check(not bad, f"{'中文' if lang == 'zh' else '英文'}界面 15 个页面显示，控制栏无截断"
              + (f" —— 截断: {bad}" if bad else ""))
        app.change_font(+1)
        app.change_font(+1)
        app.update()
        app.change_font(-1)
        app.change_font(-1)
        app.update()
        check(True, f"{'中文' if lang == 'zh' else '英文'}界面字体放大 / 缩小")
        if lang == "en":
            app.destroy()
        else:
            ZH.append(app)


ZH = []


def test_workflow():
    from test_crystal import make_cif
    app = ZH[0] if ZH else new_app("zh")
    cif = TMP / "t.cif"
    make_cif(cif)
    gui._filedialog.askopenfilename = lambda **k: str(cif)
    st = app.struct_tab
    app.show_page("结构/晶体")
    st.open_file()
    app.update()
    check(st.scene, "打开 CIF 并显示晶胞")
    st.cut.set("4.5")
    st.search()
    app.update()
    n = len(st.tree.get_children())
    check(n > 0, f"搜索近邻二聚体 ({n} 个)")
    st.eht_all()                                    # 软件内 EHT 计算全部二聚体 J → 迁移率表
    app.show_page("迁移率")
    app.update()
    rows = app.mob_tab.tree.get_children()
    check(len(rows) > 0, f"EHT 计算全部二聚体 J 并填入迁移率表 ({len(rows)} 行)")
    app.mob_tab.compute()
    app.update()
    out = app.mob_tab.text.get("1.0", "end")
    check("cm" in out or "μ" in out, "计算迁移率并显示结果")
    if "--keep" in sys.argv:
        app.mainloop()
    else:
        app.destroy()


def _menu_items(menu, path=()):
    for i in range((menu.index("end") or -1) + 1):
        kind = menu.type(i)
        if kind == "cascade":
            sub = menu.nametowidget(menu.entrycget(i, "menu"))
            try:
                pc = sub.cget("postcommand")
                if pc:
                    sub.tk.eval(pc)
            except Exception:  # noqa: BLE001
                pass
            yield from _menu_items(sub, path + (menu.entrycget(i, "label"),))
        elif kind in ("command", "checkbutton", "radiobutton"):
            yield path + (menu.entrycget(i, "label"),), menu, i


def test_menus_and_toolbar():
    from test_crystal import make_cif
    app = new_app("zh")
    cif = TMP / "m.cif"
    make_cif(cif)
    st = app.struct_tab
    st.open_path(str(cif))
    st.cut.set("4.5")
    st.search()
    lat0 = st.crys["lattice"].copy()
    for v in (0, 1, 2, "x"):
        st.view_along(v)
    import numpy as np
    check(np.allclose(lat0, st.crys["lattice"]), "沿 a / b / c / x 观察后晶格矢量不变")
    menubar = app.nametowidget(app.cget("menu"))
    skip = ("退出", "English", "中文", "关于", "Exit", "关闭全部功能窗口") + tuple(theme.THEMES)   # 切换主题会重建窗口
    n = 0
    for path, menu, i in list(_menu_items(menubar)):
        if path[-1] in skip:
            continue
        menu.invoke(i)
        app.update()
        n += 1
        if UNEXPECTED:
            raise SystemExit("[失败] 菜单 " + " → ".join(path) + ":\n" + "\n".join(UNEXPECTED))
    ERRORS.clear()
    check(n > 60, f"菜单栏全部 {n} 项执行无程序错误")
    for w in app.toolbar.winfo_children():
        if w.winfo_class() in ("TButton", "TCheckbutton") and w.cget("text") not in ("English", "中文"):
            w.invoke()
            app.update()
    if UNEXPECTED:
        raise SystemExit("[失败] 工具栏:\n" + "\n".join(UNEXPECTED))
    ERRORS.clear()
    check(True, "工具栏全部按钮执行无程序错误")
    # 运行 Gaussian：每种计算类型写出输入文件
    st.sel = [0, 1]
    from mobility_tool.gui_run import RunADFDialog, RunGaussianDialog
    g = RunGaussianDialog.open(app)
    out = TMP / "gauss"
    for key, lab, _ in RunGaussianDialog.CALCS:
        if key == "fcht":
            continue
        g.calc.set(key)
        g._calc_changed()
        src = {"ti": "晶体全部近邻二聚体", "cp": "主窗口选中的两个分子"}.get(key, "主窗口当前结构")
        g.v["source"].set(src)
        g._source_changed()
        g.v["name"].set(f"t_{key}")
        g.v["outdir"].set(str(out / key))
        d, files = g.write_inputs(ask=False)
        gjf = [f for f in files if f.suffix == ".gjf"]
        check(gjf and not ERRORS, f"运行 Gaussian · {lab}：写出 {len(gjf)} 个 gjf")
    g.calc.set("ti")
    g._calc_changed()
    g.v["source"].set("主窗口选中的两个分子")
    g.preview()
    check(g.pv_text.get("1.0", "end").strip().startswith("%chk"), "运行 Gaussian · 预览输入")
    # FCHT：在四点法重组能目录中生成
    g.calc.set("fcht")
    g._calc_changed()
    g.v["reorg_dir"].set(str(out / "reorg"))
    g.v["name"].set("t_reorg")
    d, files = g.write_inputs(ask=False)
    check(any(f.suffix == ".gjf" for f in files) and not ERRORS, "运行 Gaussian · FCHT 黄昆因子")
    a = RunADFDialog.open(app)
    for key, lab, _ in RunADFDialog.CALCS:
        a.calc.set(key)
        a._calc_changed()
        a.v["source"].set("晶体全部近邻二聚体" if key == "crystal" else "主窗口选中的两个分子")
        a.v["name"].set(f"t_{key}")
        a.v["outdir"].set(str(TMP / "adf" / key))
        d, files = a.write_inputs(ask=False)
        check(any(f.suffix in (".run", ".sh") for f in files) and not ERRORS, f"运行 ADF · {lab}：写出 {len(files)} 个文件")
    app.destroy()


def test_flow_window():
    from test_crystal import make_cif
    from mobility_tool import gui_flow
    from mobility_tool.gui_run import RunGaussianDialog
    app = new_app("zh")
    cif = TMP / "f.cif"
    make_cif(cif)
    app.struct_tab.open_path(str(cif))
    w = app.open_flow()
    app.update()
    w.tpl.set(next(iter(gui_flow.TEMPLATES)))
    w.use_template()
    check(len(w.stages) == 4, "流程图 · 使用模板 (二聚体 → EHT → Marcus → KMC)")
    w.sel = 0
    dlg = w.edit()
    dlg.vars["cutoff"][0].set("4.5")
    dlg.ok()
    w.sel = 1
    dlg = w.edit()
    dlg.vars["cutoff"][0].set("4.5")
    dlg.ok()
    w.sel = 3
    d = w.edit()
    d.vars["walkers"][0].set("300")
    d.ok()
    w.sel = 3
    w.add("gaussian", {"calc": "reorg", "source": "主窗口当前结构"})
    g = w.edit()
    check(isinstance(g, RunGaussianDialog) and g.stage is w.stages[4], "流程图 · 双击 Gaussian 阶段打开运行 Gaussian 设置")
    g.v["method"].set("PBE0")
    g.ok()
    check(w.stages[4]["settings"]["method"] == "PBE0" and w.stages[4]["settings"]["calc"] == "reorg",
          "流程图 · 阶段设置对话框“确定”保存设置")
    w.move(-1)
    check(w.stages[3]["type"] == "gaussian", "流程图 · 上移阶段")
    w.delete()
    check(len(w.stages) == 4 and not ERRORS, "流程图 · 删除阶段")
    w.dir_var.set(str(TMP / "flows"))
    w.title_var.set("t_flow")
    w.run()
    app.update()
    check(w.status == ["done"] * 4, f"流程图 · 运行软件内流程 ({w.status})")
    check(len(app.mob_tab.tree.get_children()) > 0, "流程图 · 结果 J 填入迁移率页")
    p = w.save_as(str(TMP / "t.mflow"))
    w.stages = []
    w.load(p)
    check(len(w.stages) == 4 and (TMP / "flows/t_flow/flow_results.json").exists(), "流程图 · 保存 / 打开流程文件")
    for th in ("经典灰", "深色"):
        theme.set_theme(th)
        w.draw()
    theme.set_theme("专业浅色")
    app.destroy()


if __name__ == "__main__":
    test_pages()
    test_workflow()
    test_menus_and_toolbar()
    test_flow_window()
    print("真实窗口界面测试全部通过")
