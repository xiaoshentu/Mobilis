"""无显示环境下的界面冒烟测试：用桩对象替代 tkinter，逐个调用各按钮回调。"""
import sys, types, json, tempfile, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from fake_programs import install as _install_fakes
_install_fakes()

SHOWN = []
def _rec(k):
    for key in ("text", "title"):
        if isinstance(k.get(key), str): SHOWN.append(k[key])
    if isinstance(k.get("values"), (list, tuple)): SHOWN.extend(v for v in k["values"] if isinstance(v, str))
GEOM_ERR = []
class W:
    def __init__(self, *a, **k):
        self._kw = k; _rec(k)
        m = a[0] if a and isinstance(a[0], W) else k.get("master")
        self._master = m if isinstance(m, W) else None
        self._mgr = None
    def _manage(self, how):
        # 与真实 Tk 相同：同一个容器内的子控件不能混用 pack 与 grid
        m = self._master
        if m is None:
            return
        if m._mgr is None:
            m._mgr = how
        elif m._mgr != how and {m._mgr, how} == {"pack", "grid"}:
            GEOM_ERR.append(f"{type(m).__name__}({m._kw.get('text', '')!r}) 中混用 {m._mgr} 与 {how}："
                            f"{type(self).__name__}({self._kw.get('text', '')!r})")
    def pack(self, *a, **k): self._manage("pack")
    pack_configure = pack
    def grid(self, *a, **k): self._manage("grid")
    grid_configure = grid
    def place(self, *a, **k): self._manage("place")
    def configure(self, *a, **k): _rec(k)
    config = configure
    def heading(self, *a, **k): _rec(k)
    def add(self, *a, **k): _rec(k)
    def __getattr__(self, n):
        def f(*a, **k):
            _rec(k)
        return f
class Var:
    def __init__(self, value=None, **k): self.v = value
    def get(self): return self.v
    def set(self, v): self.v = v
class StringVar(Var):
    def __init__(self, value="", **k): self.v = value
class BooleanVar(Var): pass
class Text(W):
    def __init__(self, *a, **k): super().__init__(); self.buf = ""
    def insert(self, idx, s): self.buf += s
    def delete(self, *a): self.buf = ""
    def get(self, *a): return self.buf
class Treeview(W):
    def __init__(self, *a, **k): super().__init__(); self._rows = {}; self._n = 0; self._sel = ()
    def insert(self, parent, idx, values=(), iid=None, **k):
        self._n += 1; iid = iid or f"I{self._n}"; self._rows[iid] = list(values); SHOWN.extend(str(v) for v in values); return iid
    def selection_set(self, items): self._sel = tuple(items)
    def exists(self, iid): return iid in self._rows
    def get_children(self): return tuple(self._rows)
    def delete(self, *iids):
        for i in iids: self._rows.pop(i, None)
    def item(self, iid, key=None, values=None, **kw):
        if values is not None: self._rows[iid] = list(values); SHOWN.extend(str(v) for v in values)
        if values is not None or kw: return
        return {"values": self._rows[iid]} if key is None else self._rows[iid]
    def selection(self): return self._sel
    def index(self, iid): return list(self._rows).index(iid)
class Tk(W):
    def __init__(self): super().__init__()

tk = types.ModuleType("tkinter")
class Toplevel(W):
    def __init__(self, *a, **k): super().__init__()
class Menu(W):
    def add_command(self, **k): _rec(k); MENU.append(k)
    def add_cascade(self, **k): _rec(k)
MENU = []
for n, o in dict(Tk=Tk, StringVar=StringVar, BooleanVar=BooleanVar, Text=Text, Frame=W, Canvas=W,
                 Toplevel=Toplevel, Menu=Menu, Button=W, Label=W).items(): setattr(tk, n, o)
fnt = types.ModuleType("tkinter.font"); fnt.families = lambda *a: ["Microsoft YaHei UI", "Consolas"]
fnt.nametofont = lambda n: W(); tk.font = fnt
ttk = types.ModuleType("tkinter.ttk")
for n in ["Frame", "Label", "Entry", "Button", "Combobox", "Checkbutton", "Radiobutton", "LabelFrame",
          "Notebook", "PanedWindow", "Style", "Scrollbar", "Spinbox"]:
    setattr(ttk, n, type(n, (W,), {}))
ttk.Treeview = Treeview
ttk.Progressbar = type("Progressbar", (W,), {})
fd = types.ModuleType("tkinter.filedialog"); mb = types.ModuleType("tkinter.messagebox")
ERR = []
mb.showerror = lambda t, m: ERR.append(m); mb.showinfo = lambda t, m: print("  [info]", m.splitlines()[0])
mb.askyesno = lambda *a, **k: True
tk.ttk, tk.filedialog, tk.messagebox = ttk, fd, mb
sys.modules.update({"tkinter.font": fnt, "tkinter": tk, "tkinter.ttk": ttk, "tkinter.filedialog": fd, "tkinter.messagebox": mb})

import matplotlib; matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
bk = types.ModuleType("matplotlib.backends.backend_tkagg")
class Canvas(FigureCanvasAgg):
    def __init__(self, fig, master=None): super().__init__(fig)
    def get_tk_widget(self): return W()
    def draw_idle(self): self.draw()
bk.FigureCanvasTkAgg = Canvas; bk.NavigationToolbar2Tk = W
sys.modules["matplotlib.backends.backend_tkagg"] = bk
_use = matplotlib.use; matplotlib.use = lambda *a, **k: None

from mobility_tool import gui
gui.CONFIG = Path(tempfile.mkdtemp()) / "config.json"      # 不改动用户自己的设置 (最近文件等)
from test_crystal import make_cif
from test_decomposition import write_freq_fchk, model_hessian
from test_all import write_fchk, fake_log
from mobility_tool.gaussian_io import read_xyz, BOHR2ANG

app = gui.App(); app.sync = True
_nh = app.config_data.get("nav_hidden", False)
app.toggle_nav(False); assert app.config_data["nav_hidden"] is True
app.toggle_nav(True); assert app.config_data["nav_hidden"] is False
app.toggle_nav(not _nh)
print("[通过] 导航栏隐藏 / 显示")
if GEOM_ERR:
    raise SystemExit("[失败] 同一容器混用 pack 与 grid (真实 Tk 会报错)：\n  " + "\n  ".join(GEOM_ERR))
print("[通过] 全部页面的布局：没有在同一容器中混用 pack 与 grid")
tmp = Path(tempfile.mkdtemp())
def pick(*paths):
    it = iter(paths); fd.askopenfilename = fd.askdirectory = fd.asksaveasfilename = lambda **k: str(next(it))
def check(step):
    if ERR: raise SystemExit(f"[失败] {step}: {ERR}")
    print(f"[通过] {step}")

# ① 结构 / 晶体
make_cif(tmp / "t.cif"); pick(tmp / "t.cif"); app.struct_tab.open_file(); check("打开 CIF 并绘制晶胞")
app.struct_tab.cut.set("4.5"); app.struct_tab.search(); check("搜索二聚体")
app.struct_tab.tree._sel = (app.struct_tab.tree.get_children()[2],); app.struct_tab.redraw(); check("高亮选中二聚体")
pick(tmp / "dimer_sel.xyz"); app.struct_tab.save_xyz(); check("保存选中二聚体 xyz")
pick(tmp / "dimers"); app.struct_tab.export(); check("导出二聚体 + 生成 Gaussian 输入")
app.struct_tab.to_mobility(); assert len(app.mob_tab.tree.rows()) == 8; check("近邻矢量送到迁移率表")
from types import SimpleNamespace as NS
st = app.struct_tab
st.mode.set("cell"); st.rng["a"][1].set("2"); st.rebuild(keep_view=False)
assert len(st.scene) == 8, len(st.scene); check("显示范围 a:0→2 (8 个分子)")
st.set_range(-1, 2); assert len(st.scene) == 4 * 27; st.set_range(0, 1); st.rng["a"][1].set("2"); st.rebuild()
check("中心 3×3×3 显示 (108 个分子) 与范围切换")
ax = st.plot.ax; st.plot.canvas.draw(); rend = st.rend
z0 = rend.zoom; st.on_scroll(NS(inaxes=ax, button="up")); assert rend.zoom > z0; check("滚轮缩放")
R0 = rend.R.copy(); st.on_press(NS(button=3, inaxes=ax, x=100, y=100, key=None))
st.on_motion(NS(x=160, y=120)); st.on_release(NS(button=3, xdata=None, ydata=None))
assert not np.allclose(R0, rend.R) and st.sel == []; check("右键拖动旋转 (不触发选择)")
st.on_press(NS(button=1, inaxes=ax, x=100, y=100, key=None)); st.on_motion(NS(x=100, y=150))
st.on_release(NS(button=1, xdata=None, ydata=None)); check("左键拖动旋转")
st.on_press(NS(button=2, inaxes=ax, x=100, y=100, key=None)); p0 = rend.pan.copy(); st.on_motion(NS(x=130, y=100))
st.on_release(NS(button=2, xdata=None, ydata=None)); assert not np.allclose(p0, rend.pan); check("中键平移")
def click(k, key=None):
    scr = rend._screen[0]
    st.on_press(NS(button=1, inaxes=ax, x=0, y=0, key=key)); st.on_release(NS(button=1, xdata=scr[k, 0], ydata=scr[k, 1]))
dep = rend._screen[1]
k1 = int(np.argmax(np.where(np.array(st.scene_syms) != "H", dep, -1e9))); click(k1)
assert st.sel == [int(st.scene_owner[k1])], st.sel; check(f"左键单击选分子 → {st.scene[st.sel[0]]['label']}")
for k2 in np.argsort(-dep):
    if st.scene_owner[k2] != st.scene_owner[k1] and rend.pick(*rend._screen[0][k2]) is not None \
            and st.scene_owner[rend.pick(*rend._screen[0][k2])] != st.scene_owner[k1]:
        break
click(int(k2), key="control")
assert len(st.sel) == 2; check("Ctrl+左键多选")
pick(tmp / "one.xyz"); st.export_selected(); assert len(read_xyz(tmp / "one.xyz")) == 24; check("导出选中分子 xyz")
pick(tmp / "pair.xyz"); st.export_pair(); assert app.ti_tab.na.get() == 12 and app.ti_tab.dimer.get().endswith("pair.xyz")
check("选中两分子导出二聚体并填入转移积分页")
st.mtree._sel = st.mtree.get_children()[3:4]; st.on_tree_select(); assert st.sel == [3]; check("分子列表选择同步到三维图")
for sty in ("棍状", "空间填充", "线框", "球棍"):
    st.style_var.set(sty); st.redraw()
st.bg_var.set("白色"); st.persp.set(False); st.redraw(); st.view_along(0); st.view_along("z")
pick(tmp / "struct.png"); st.save_image(); assert (tmp / "struct.png").exists(); st.bg_var.set("深蓝渐变")
check("四种显示样式 / 背景 / 沿 a、z 观察 / 保存高清图片")
pick(ROOT / "examples/benzene.xyz"); app.struct_tab.open_file(); check("打开 xyz 分子")

# ② 重组能
app.reorg_tab.geom.set(ROOT / "examples/benzene.xyz"); app.reorg_tab.outdir.set(tmp / "reorg")
app.reorg_tab.method.set("PBE0 em=gd3bj"); app.reorg_tab.solvent.set("dichloromethane"); app.reorg_tab.generate()
assert "scrf=(pcm,solvent=dichloromethane)" in app.reorg_tab.text.buf; check("生成重组能输入 (含 PCM 溶剂)")
E = dict(nn=-232.0, cn=-231.65, an=-231.99, nc=-231.996, cc=-231.653, na=-231.997, aa=-231.9935)
(tmp/"reorg/benzene_neutral.log").write_text(fake_log([E["nn"], E["cn"], E["an"]]))
(tmp/"reorg/benzene_cation.log").write_text(fake_log([E["cc"], E["nc"]]))
(tmp/"reorg/benzene_anion.log").write_text(fake_log([E["aa"], E["na"]]))
app.reorg_tab.analyze(); app.reorg_tab.send(); check(f"读取重组能并送到迁移率页 (λe={app.mob_tab.lam_e.get()}, λh={app.mob_tab.lam_h.get()})")
assert "IP(v)" in app.reorg_tab.text.buf and app.ipea_tab.result; check("重组能页同时输出 IP/EA 和 HOMO/LUMO")

# ③ IP / EA：手动填 7 个能量（图片中的流程）
for k in app.ipea_tab.vars: app.ipea_tab.vars[k].set("")
for k, v in dict(E0_N=-232.0, Ec_C=-231.653, Ea_A=-231.9935, Ec_N=-231.65, Ea_N=-231.99, E0_C=-231.996,
                 E0_A=-231.997).items():
    app.ipea_tab.vars[k].set(str(v))
app.ipea_tab.compute(); assert "λh" in app.ipea_tab.text.buf; check("手动输入 7 个能量计算 IP/EA/HEP/EEP")
pick(tmp / "reorg"); app.ipea_tab.from_dir(); check("IP/EA 页从目录读取")
pick(tmp / "ipea.csv"); app.ipea_tab.export(); check("导出 IP/EA CSV")

# ③ 重组能分解（合成 fchk 放进重组能目录，测试自动查找）
atoms = read_xyz(ROOT / "examples/benzene.xyz"); sym = [a[0] for a in atoms]; xn = np.array([a[1:] for a in atoms])
hn, bn = model_hessian(sym, xn); dq = np.random.default_rng(0).normal(size=bn.shape[0]) * 0.01
xc = xn + (bn.T @ np.linalg.pinv(bn @ bn.T) @ dq).reshape(-1, 3) * BOHR2ANG; hc, _ = model_hessian(sym, xc)
write_freq_fchk(tmp/"reorg/benzene_neutral.fchk", sym, xn, hn); write_freq_fchk(tmp/"reorg/benzene_cation.fchk", sym, xc, hc)
pick(tmp/"reorg"); app.decomp_tab.autofind(); app.decomp_tab.run(); check("重组能分解")
for st in ("neutral", "ion"):
    app.decomp_tab.state_var.set(st)
    for k in app.decomp_tab.PLOTS:
        app.decomp_tab.kind.set(k); app.decomp_tab.draw()
        app.decomp_tab.plot.fig.savefig(tmp / f"decomp_{st}_{app.decomp_tab.PLOTS.index(k)}.png", dpi=70)
check("五种分解图 × 两个态")
pick(tmp/"csv"); app.decomp_tab.export(); check("导出分解 CSV + Origin 格式")
app.decomp_tab.state_var.set("neutral"); pick(tmp/"figs"); (tmp/"figs").mkdir(); app.decomp_tab.export_fig()
assert list((tmp/"figs").glob("Fig3_style_*.png")); check("导出论文图 (a)(b) + Fig. 3 三联图")
# Gaussian FCHT 路线
from test_fcht import make_fcht_log
rng = np.random.default_rng(2); nm = 30
ffr = np.sort(rng.uniform(300, 3200, nm)); qq, _ = np.linalg.qr(rng.normal(size=(nm, nm)))
pick(tmp/"reorg"); app.decomp_tab.gen_fcht(); assert (tmp/"reorg/benzene_fcht_cation_to_neutral.gjf").exists()
check("在重组能目录生成 FCHT 输入")
(tmp/"reorg/benzene_fcht_cation_to_neutral.log").write_text(make_fcht_log(ffr, ffr, rng.uniform(0, .1, nm), qq, rng.normal(size=nm)))
(tmp/"reorg/benzene_fcht_neutral_to_cation.log").write_text(make_fcht_log(ffr, ffr, rng.uniform(0, .1, nm), qq.T, rng.normal(size=nm)))
pick(tmp/"reorg"); app.decomp_tab.autofind(); app.decomp_tab.source.set("fcht"); app.decomp_tab.run()
for st in ("neutral", "ion"):
    app.decomp_tab.state_var.set(st)
    for k in app.decomp_tab.PLOTS[:3]:
        app.decomp_tab.kind.set(k); app.decomp_tab.draw()
app.decomp_tab.plot.fig.savefig(tmp/"dus.png", dpi=60)
check("FCHT 结果分解 + Duschinsky 图")
app.decomp_tab.kind.set(app.decomp_tab.PLOTS[3])
try:
    app.decomp_tab.draw(); raise SystemExit("[失败] 应提示内坐标需要内置方法")
except ValueError as e:
    assert "内置" in str(e); print("[通过] FCHT 模式下请求内坐标图给出正确提示")
pick(tmp/"fchtcsv"); app.decomp_tab.export(); pick(tmp/"figs"); app.decomp_tab.export_fig(); check("FCHT 导出数据与论文图")
adv0 = app.adv_tab; adv0.vsrc.set("分解页 λ1 (末态=中性/基态)"); adv0.run_vib(); adv0.add_vib()
adv0.vlab.set("Ir-NN"); adv0.evalue.set("1.72"); adv0.run_vib(); adv0.add_vib(); assert "相对 k_nr" in adv0.text.buf
pick(tmp/"vib.png"); adv0.export_vib(); assert (tmp/"vib.csv").exists(); check("振动分辨磷光光谱 / 能隙定律 / 对比导出")

# ④ 转移积分
shuffled = [atoms[i] for i in range(12)]  # 构造交错原子顺序的二聚体，测试自动识别
d2 = read_xyz(ROOT / "examples/benzene_dimer.xyz"); mixed = [d2[i] for i in [0, 12, 1, 13] + list(range(2, 12)) + list(range(14, 24))]
from mobility_tool.gaussian_io import write_xyz; write_xyz(tmp/"mixed.xyz", mixed)
app.ti_tab.auto_na(str(tmp/"mixed.xyz")); assert app.ti_tab.na.get() == 12; check("自动识别并重排二聚体原子")
app.ti_tab.outdir.set(tmp/"ti"); app.ti_tab.generate(); check("生成 Gaussian 转移积分输入")
app.ti_tab.prog.set("adf"); app.ti_tab.generate(); check("生成 ADF 输入")
app.ti_tab.a_style.set("ADF2019 及以前"); app.ti_tab.generate(); check("生成 ADF 旧版输入")
(tmp/"ti/adf.txt").write_text("V(hole) -2.679E-02\nJ(hole) -3.4988E-02\nS(hole) 4.9852E-02\ne1(hole) -0.165788\n"
                              "e2(hole) -0.16579\nV(electron) 1.3129E-02\n")
app.ti_tab.adf_file.set(tmp/"ti/adf.txt"); app.ti_tab.calc(); check(f"读取 ADF 结果 (J={app.data['last_J']})")
from scipy.linalg import eigh
rng = np.random.default_rng(1); n = 15; x = rng.normal(size=(n, n)) * .25; S = x @ x.T + 2*np.eye(n)
dd = 1/np.sqrt(np.diag(S)); S *= np.outer(dd, dd); F = rng.normal(size=(n, n))*.2; F = (F+F.T)/2 + np.diag(np.linspace(-1, .5, n))
eA, cA = eigh(F[:8, :8], S[:8, :8]); eB, cB = eigh(F[8:, 8:], S[8:, 8:]); eD, cD = eigh(F, S)
cs = rng.normal(size=(5, 3))
write_fchk(tmp/"x_A.fchk", cA, eA, 4, cs[:3]); write_fchk(tmp/"x_B.fchk", cB, eB, 3, cs[3:]); write_fchk(tmp/"x_AB.fchk", cD, eD, 7, cs)
app.ti_tab.prog.set("gaussian"); app.ti_tab.fill_prefix(str(tmp/"x_A.fchk")); app.ti_tab.fa.set(tmp/"x_A.fchk")
app.ti_tab.calc(); check("读取 Gaussian fchk 计算 J")
app.ipea_tab.orbfile.set(tmp/"x_A.fchk"); app.ipea_tab.read_orb(); check("从 fchk 读取 HOMO/LUMO")

# ⑤ 迁移率
app.mob_tab.tree._sel = app.mob_tab.tree.get_children(); app.mob_tab.fill_last_j(); app.mob_tab.compute()
app.mob_tab.plot.fig.savefig(tmp/"mob_polar.png", dpi=70); check("迁移率 + 角分布图")
app.mob_tab.compute3d(); assert "主值" in app.mob_tab.text.buf and len(app.mob_tab.plot.fig.axes) >= 2
app.mob_tab.m3_model.set(app.mob_tab.M3D[1]); app.mob_tab.compute3d()
pick(tmp/"mob3d.png"); app.mob_tab.export3d(); assert (tmp/"mob3d.png").exists() and (tmp/"mob3d.html").exists()
check("三维各向异性迁移率图 (两种模型)、导出 PNG 与可旋转 HTML")
(tmp/"t3.csv").write_text("Pathway,d,Theta,Ve,Vh\nP1,8.709,0,0.07873,0.00117\nP2,10.138,87.671,0.01381,0.00938\n"
                         "P3,8.490,-152.952,0.02277,0.00434\n")
mt = app.mob_tab; mt.unit.set("eV"); pick(tmp/"t3.csv"); mt.import_csv(); mt.lam_e.set("0.130"); mt.lam_h.set("0.225")
mt.temp.set("300"); mt.compute()
mu_e = mt.app.data["mobility"][300.0]["e"]["mu_max"]; mu_h = mt.app.data["mobility"][300.0]["h"]["mu_max"]
assert abs(mu_e - 10.84) < 0.02 and abs(mu_h - 0.058) < 0.001, (mu_e, mu_h)
check(f"导入论文 Table 3 (Ir-BB) → μe,max={mu_e:.2f} μh,max={mu_h:.4f} (论文 10.84 / 0.058)")
mt.label.set("Ir-BB"); mt.add_compare(); mt.add_compare(); pick(tmp/"fig6.png"); mt.export_fig()
assert (tmp/"fig6.png").exists(); check("导出 Fig. 6 风格对比图")
app.mob_tab.temp.set("200:400:50"); app.mob_tab.compute(); app.mob_tab.plot.fig.savefig(tmp/"mob_T.png", dpi=70)
check("温度扫描")
pick(tmp/"cfg.json"); app.mob_tab.save_cfg(); pick(tmp/"cfg.json"); app.mob_tab.load_cfg(); app.mob_tab.compute()
check("保存/加载迁移率配置")
for i in app.mob_tab.tree.get_children(): pass
# ⑦ 激发态 / 光谱
from test_photophys import fake_log as td_log, write_shell_fchk
pt = app.photo_tab; pt.geom.set(ROOT/"examples/benzene.xyz"); pt.outdir.set(tmp/"exc"); pt.generate()
assert (tmp/"exc/benzene_T1.gjf").exists(); check("生成 S0/S1/T1 激发态输入")
at = read_xyz(ROOT/"examples/benzene.xyz"); s1g = [(a[0], a[1] + .03, a[2], a[3]) for a in at]
cfgs = [("21", "22", 0.69)]
(tmp/"exc/benzene_S0.log").write_text(" 21 alpha electrons       21 beta electrons\n" + td_log([at], [[]], [("Singlet", 2.41, 0.001, cfgs), ("Singlet", 3.2, 0.4, cfgs)]))
(tmp/"exc/benzene_S1.log").write_text(td_log([at, s1g], [[("Singlet", 2.3, 0.01, cfgs)]] * 2, [("Singlet", 2.10, 0.02, cfgs)]))
(tmp/"exc/benzene_T1.log").write_text(td_log([at, s1g], [[("Triplet", 1.8, 0, cfgs)]] * 2, [("Triplet", 1.65, 0, cfgs)]))
pt.label.set("Ir-BB"); pt.analyze(); assert "Stokes" in pt.text.buf and "HOMO→LUMO" in pt.text.buf
check("读取激发态结果 (Table 4 格式, Stokes 位移, 组态 HOMO→LUMO)")
pt.label.set("Ir-NN"); pt.analyze(); pick(tmp/"fig9.png"); pt.export_spectra(); assert (tmp/"fig9.csv").exists()
check("两体系光谱对比并导出 Fig. 9 风格图 + 数据")
pt.state_overlays(); pt.plot.fig.savefig(tmp/"fig7.png", dpi=60); check("S0–S1 / S0–T1 叠合图 (Fig. 7)")
pt.ga.set(ROOT/"examples/benzene.xyz"); pt.gb.set(tmp/"one.xyz") if False else pt.gb.set(ROOT/"examples/benzene.xyz"); pt.calc_rmsd()
pt.rates(); assert "1.500e+05" in pt.text.buf; check("RMSD 与 k_r/k_nr (Φ=0.21, τ=1.4 μs → 1.50e5)")
# 轨道组成
from scipy.linalg import eigh as _eigh
rng2 = np.random.default_rng(4); nbf = 12; xx = rng2.normal(size=(nbf, nbf)) * .2; SS = xx @ xx.T + 2 * np.eye(nbf)
dd2 = 1 / np.sqrt(np.diag(SS)); SS *= np.outer(dd2, dd2); FF = rng2.normal(size=(nbf, nbf)); ee, cc = _eigh((FF + FF.T) / 2, SS)
write_shell_fchk(tmp/"ir.fchk", cc, ee, 3, [0, 1, 0, -2, 0, 0], [1, 1, 2, 2, 3, 4], [6, 77, 1, 1],
                 np.array([[0, 0, 0], [0, 0, 2.0], [0, 1.1, -.5], [5, 5, 5]]) / 0.529177)
ip = app.ipea_tab; ip.orbfile.set(tmp/"ir.fchk"); ip.auto_frag(); assert "Metal(Ir)" in ip.frag_text.buf
ip.comp_method.set("Mulliken"); ip.comp(); assert "HOMO" in ip.text.buf; check("轨道片段组成 (自动划分 Ir / 配体, Mulliken)")
# 运行窗口 (假高斯)
import shutil as _sh
from fake_programs import fake_gaussian
fake_exe = fake_gaussian(tmp / "fakebin")              # Windows 下为 g16.bat (调用当前 Python)
rd = gui.RunDialog(app, str(tmp/"exc")); rd.exe.set(fake_exe); rd.load()
rd.tree._sel = rd.tree.get_children(); rd.run(); rd.queue.thread.join(30); rd.poll()
assert "成功 3 / 3" in rd.log.buf, rd.log.buf; check("运行窗口用（假）高斯依次运行 S0→S1→T1 并 formchk")
# ---- EHT、三种方法对比、应变、KMC
sx = app.struct_tab; pick(tmp / "t.cif"); sx.open_file(); sx.cut.set("4.5"); sx.eht_all()
assert app.data["crystal_couplings"] and app.mob_tab.tree.rows()[0][7] != ""; check("软件内 EHT 计算全部二聚体并填入迁移率表")
mt = app.mob_tab; mt.unit.set("meV"); mt.lam_e.set("0.2"); mt.lam_h.set("0.2"); mt.temp.set("300")
mt.k_src.set("结构页晶体 (全部分子)"); mt.k_walk.set("600 1500"); mt.run_kmc()
assert "迁移率张量" in mt.text.buf; check("KMC (晶体近邻图, Marcus)")
mt.k_src.set("迁移率表"); mt.k_rate.set("Jortner"); mt.k_lout.set("0.05"); mt.run_kmc(); check("KMC (迁移率表, Jortner)")
mt.k_rate.set("Marcus"); mt.k_lout.set("0"); mt.run_kmc(); assert "解析张量主值" in mt.text.buf; check("KMC 与解析张量对照")
mt.k_field.set("1e4 0 0"); mt.run_kmc(); assert "场迁移率" in mt.text.buf; mt.k_field.set("0 0 0"); check("KMC 有外场 → 场迁移率")
mt.k_src.set("结构页晶体 (全部分子)"); mt.k_sigma.set("0.03"); mt.k_cell.set("3"); mt.run_kmc(); mt.k_sigma.set("0")
check("KMC 含能量无序 (周期超胞)")
sx.strain["c"].set("-2"); sx.apply_strain(); assert abs(sx.crys["cell"][2] - sx.crys0["cell"][2] * 0.98) < 1e-9
sx.reset_strain(); check("晶胞缩放 / 恢复")
sx.scan_axis.set("c"); sx.scan_vals.set("-2:2:2"); sx.strain_scan(); assert "应变扫描" in mt.text.buf; check("应变扫描 → 迁移率")
ti = app.ti_tab; ti.prog.set("eht"); ti.dimer.set(tmp / "pair.xyz"); ti.na.set(""); ti.calc()
assert "软件内 EHT" in ti.text.buf and app.data["last_J"]["hole"] is not None; check("转移积分页：软件内 EHT 计算")
gdir = tmp / "dimers"
ti.cmp_dir.set(gdir); ti.compare(); assert "EHT" in ti.text.buf; check("三种方法对比 (目录)")
ti.fill_method.set("EHT×比例因子"); ti.compare_to_mobility(); check("对比结果 (EHT×比例因子) 填入迁移率表")
ti.cmp_dir.set(""); ti.dimer.set(tmp / "pair.xyz"); ti.compare(); check("三种方法对比 (单个二聚体)")
ti.prog.set("gaussian")

# ---- 程序路径 / ADF 运行 / xtb / 扩展页 / 菜单
app.config_data.update({"ams_home": "/tmp/fakeams"})
import os as _os
def _same_path(a, b):        # Windows: "/tmp/fakeams" 与 "C:\\tmp\\fakeams" 是同一目录
    return _os.path.normcase(_os.path.abspath(str(a))) == _os.path.normcase(_os.path.abspath(str(b)))
pd = gui.ProgramsDialog(app); pd.detect_all()
assert _same_path(pd.entries["ams_home"].get(), "/tmp/fakeams"), pd.entries["ams_home"].get()
assert "xtb_exe" not in pd.entries
pd.nscm.set("4"); pd.save(); assert app.config_data["adf_nscm"] == "4"
check("程序路径设置 (检测 AMS；ADF 并行核数；已移除 xtb)")
ti.prog.set("adf"); ti.a_style.set("AMS2020+"); ti.outdir.set(tmp / "adfjobs"); ti.dimer.set(tmp / "pair.xyz"); ti.na.set("12"); ti.generate()
rd2 = gui.RunDialog(app, str(tmp / "adfjobs")); rd2.load(); assert any(i.endswith(".sh") for i in rd2.tree.get_children())
from mobility_tool.programs import find_shell as _find_shell
HAVE_SH = not sys.platform.startswith("win") or bool(_find_shell())   # 模拟 ADF 是 sh 脚本：Windows 上需要 Git for Windows 的 bash
if not HAVE_SH:
    print("  [跳过] 本机没有 bash (Git for Windows)，不运行模拟 ADF 脚本 (以下几项同样跳过)")
else:
    rd2.tree._sel = rd2.tree.get_children(); rd2.run(); rd2.queue.thread.join(30); rd2.poll()
    assert "成功 1 / 1" in rd2.log.buf, rd2.log.buf; check("运行窗口执行 ADF 任务 (AMS 环境变量 + 片段顺序)")
ti.cmp_dir.set(gdir); ti.compare(); assert "xtb" not in ti.text.buf.split("\n")[0]; check("方法对比 (EHT / Gaussian / ADF，不含 xtb)")
adv = app.adv_tab
sx.cut.set("4.5"); sx.eht_all(); adv.dn.set("6"); adv.run_dyn(); assert "σh/|J|" in adv.text.buf; check("动态无序 σ_J (EHT 采样)")
adv.tl.set("6"); adv.tn.set("3"); adv.run_tlt(); assert "TLT" in adv.text.buf; check("TLT 迁移率 (晶体图，多分子晶胞)")
adv.use_dyn.set(True); adv.run_tlt(); adv.use_dyn.set(False); check("TLT 使用动态无序结果")
adv.run_band(); assert "有效质量" in adv.text.buf; check("紧束缚能带 / 有效质量")
adv.src.set(adv.SRC[1]); adv.run_band(); adv.run_tlt(); adv.src.set(adv.SRC[0]); check("能带与 TLT (迁移率表来源)")
adv.run_stack(); assert "面间距" in adv.text.buf; check("π 堆积几何分析")
gui_ok = True
for item in MENU:
    if "退出" in str(item.get("label", "")) or "Exit" in str(item.get("label", "")): continue
assert any("程序路径" in str(m.get("label")) for m in MENU); check(f"菜单栏 ({len(MENU)} 个菜单项)")
app.show_page("迁移率"); assert app.page_var.get() == "迁移率"; check("页面下拉切换")

# ---- 字体大小 / ADF2016
from mobility_tool import theme as _th
app.set_font_scale(1.3); assert abs(_th.SCALE - 1.3) < 1e-9 and _th.sz(10) == 13
app.change_font(+1); assert _th.SCALE > 1.3; app.change_font(-1); app.change_font(-1); assert _th.SCALE < 1.3
app.set_font_scale(1.0); assert app.config_data["font_scale"] == 1.0; check("字体放大 / 缩小 / 恢复 (实时生效并保存)")
assert any("字体" in str(m.get("label", "")) or "%" in str(m.get("label", "")) for m in MENU); check("字体菜单")
app.config_data["ams_home"] = "/tmp/fake2016/ADF2016.101"
ti.prog.set("adf"); ti.a_style.set("自动 (按安装版本)"); ti.outdir.set(tmp / "adf2016"); ti.dimer.set(tmp / "pair.xyz"); ti.na.set("12")
ti.generate()
from mobility_tool.adf import script_style, generate_adf_inputs
assert script_style(tmp / "adf2016" / "run_dimer_adf.sh") == "classic"; check("ADF 版本“自动”→ 按 ADF2016 生成旧版脚本")
generate_adf_inputs(tmp / "adf2016", dimer=tmp / "pair.xyz", n_a=12, name="old", style="ams")   # 模拟以前生成的 AMS 脚本
(tmp / "adf2016" / "run_all_adf.sh").write_text("#!/bin/sh\n")
rd3 = gui.RunDialog(app, str(tmp / "adf2016")); rd3.ams.set("/tmp/fake2016"); rd3.detect_ams(); rd3.load()
names = [Path(i).name for i in rd3.tree.get_children()]
assert "run_all_adf.sh" not in names and len(names) == 2, names; check("运行窗口列出 ADF 任务 (排除 run_all)")
if HAVE_SH:
    rd3.test_ams(); assert "ADF 环境正常" in rd3.log.buf; check("测试 ADF 环境 (H2 单点)")
    rd3.tree._sel = rd3.tree.get_children(); rd3.run(); rd3.queue.thread.join(30); rd3.poll()
    assert "成功 2 / 2" in rd3.log.buf, rd3.log.buf; check("ADF2016 运行两个任务 (AMS 格式脚本自动转换为旧版)")
    ti.adf_file.set(tmp / "adf2016" / "dimer_full.t21"); ti.calc(); assert "hole" in ti.text.buf; check("读取 ADF2016 TAPE21 结果 (无需 PLAMS)")
ti.prog.set("gaussian")

# ---- 选择 adf 程序文件 / .run 文件
pick("/tmp/fake2016/ADF2016.101/bin/adf"); rd3.ams.set(""); rd3.pick_adf_exe()
assert rd3.ams.get().endswith("ADF2016.101"), rd3.ams.get(); check("选择 bin 中的 adf 程序文件 → 识别安装目录")
assert (tmp / "adf2016" / "dimer_adf.run").exists(); check("ADF2016 同时生成 ADFjobs 可直接运行的 .run")
rd3.ams.set("/tmp/emptyhome_x")
try:
    rd3.detect_ams(); raise SystemExit("[失败] 应给出说明")
except FileNotFoundError as e:
    assert "路径不存在" in str(e); print("[通过] 识别失败时说明原因")
rd3.ams.set("/tmp/fake2016")

# ---- 晶体二聚体一次批量计算 + 自动读取 + 监视 ADFjobs + GBK 编码
import mobility_tool.gui as _g
_g.messagebox.askyesno = lambda *a, **k: False
sx = app.struct_tab
make_cif(tmp / "t2.cif"); pick(tmp / "t2.cif"); sx.open_file(); sx.cut.set("4.5"); sx.search()
ti.prog.set("adf"); ti.a_style.set("自动 (按安装版本)")
pick(tmp / "batchdir"); sx.export()
assert (tmp / "batchdir" / "batch_adf.run").exists() and (tmp / "batchdir" / "batch_adf.sh").exists()
assert (tmp / "batchdir" / "batch_adf.run").read_bytes().decode("gbk"); check("导出全部二聚体 → 一个 batch_adf.run (GBK, ADFjobs 可直接运行)")
ti.text.delete("1.0", "end"); ti.outdir.set(tmp / "adf_gbk"); ti.dimer.set(tmp / "pair.xyz"); ti.na.set("12"); ti.generate()
assert "ADF" in ti.text.buf; check("显示 UTF-8 脚本内容不再出现 gbk 解码错误")
if HAVE_SH:
    rd4 = gui.RunDialog(app, str(tmp / "batchdir")); rd4.ams.set("/tmp/fake2016"); rd4.detect_ams(); rd4.load()
    kids = list(rd4.tree.get_children()); assert any(k.endswith("batch_adf.sh") for k in kids), kids
    app.mob_tab.set_paths([]); rd4.tree._sel = [k for k in kids if k.endswith("batch_adf.sh")]; rd4.run(); rd4.queue.thread.join(60); rd4.poll()
    assert "已自动读取" in rd4.log.buf, rd4.log.buf
    rows = app.mob_tab.tree.rows(); assert rows and all(r[-1] not in ("", None) for r in rows[:1]), rows[:2]
    check(f"运行 batch_adf.sh 一次算完 → 自动读取并填入迁移率表 ({len(rows)} 个近邻)")
    pick(tmp / "batchdir2"); sx.export()
    from mobility_tool.programs import run_adf_script, find_ams as _fa
    rd5 = gui.RunDialog(app, str(tmp / "batchdir2")); rd5.ams.set("/tmp/fake2016"); rd5.detect_ams()
    rd5._watching = True; assert rd5.watch_step() is False                      # 尚未计算
    run_adf_script(_fa("/tmp/fake2016"), tmp / "batchdir2" / "batch_adf.sh")   # 模拟在 ADFjobs 中运行完毕
    rd5._watching = True; assert rd5.watch_step() is True and "已自动读取" in rd5.log.buf
    check("监视 ADFjobs：计算完成后自动读取")
ti.prog.set("gaussian")

for f in gdir.glob("G*_A.gjf"):
    g = f.stem[:-2]
    for s in ("A", "B", "AB"): shutil.copy(tmp/f"x_{s}.fchk", gdir/f"{g}_{s}.fchk")
pick(gdir); app.mob_tab.import_dir(); app.mob_tab.compute(); check("从二聚体目录批量导入 J 并计算")
print(app.mob_tab.text.buf[-420:])
shutil.copytree(tmp, "/tmp/gsm", dirs_exist_ok=True)

# ---- 文件选择 / 测量 / 提示 / 相互作用能 / 能量框架
fe = gui.FileEntry(gui.ttk.Frame(None), "测试", filetypes=[("log", "*.log *.out"), ("所有文件", "*.*")])
okl = tmp / "ok.log"; okl.write_text("...\n Normal termination of Gaussian 16\n"); badl = tmp / "bad.log"; badl.write_text("running\n")
fe.set(okl)
assert gui.file_status(str(okl))[0] == "✓" and gui.file_status(str(badl))[0] == "…" and gui.file_status(str(tmp / "no.log"))[0] == "✗"
app.set_workdir(str(tmp)); gui.record_recent(str(okl))
c = fe.candidates(); assert str(okl) in c and str(badl) in c and all(p.endswith((".log", ".out")) for p in c)
check(f"文件选择框: 最近文件 + 工作目录下拉 ({len(c)} 项) + 状态标记 (✓ 正常结束 / … 未完成 / ✗ 不存在)")
pick(tmp / "t.cif"); app.struct_tab.open_file(); assert str(tmp / "t.cif") in app.recent_structures()
app._fill_recent(gui.tk.Menu()); check("文件 → 最近打开 / 设置工作目录")
st = app.struct_tab; st.meas_mode.set(True); st.toggle_measure()
for k in (0, 1, 2, 3):
    st.add_measure_point(k)
assert st.last_measure[0] == "dihedral" and st.rend.marks == [0, 1, 2, 3]
st.add_measure_point(5); st.add_measure_point(6); assert st.last_measure[0] == "distance"; st.meas_mode.set(False); st.toggle_measure()
check("测量模式: 距离 / 键角 / 二面角")
assert "泛函" in gui.TIPS and "σ_J / |J|" in gui.TIPS; check(f"悬停提示 ({len(gui.TIPS)} 个参数)")
it = app.inter_tab; st.cut.set("4.5"); st.search(); it.src.set(it.SRC[0]); it.method.set(it.METHODS[0]); it.compute()
assert it.energies and "晶格能近似" in it.text.buf; check(f"UFF 相互作用能 ({len(it.energies)} 组) + 晶格能")
it.draw_framework(); assert it.view.renderer.cylinders; check(f"能量框架 ({len(it.view.renderer.cylinders)} 根圆柱)")
qlog = tmp / "q.log"; qlog.write_text(" ESP charges:\n   1\n" + "".join(f"  {i+1}  C  {(-1)**i*0.05:.3f}\n" for i in range(len(st.dimer_res["atoms_a"]))) + " Sum of ESP charges = 0\n")
it.charges.set(qlog); it.compute(); assert "静电" in it.method_label; it.charges.set(""); check("UFF + ESP 静电项")
it.g_dir.set(tmp / "cp"); it.gen_cp(); assert (tmp / "cp" / "G1_cp.gjf").exists() and "Fragment=2" in (tmp / "cp" / "G1_cp.gjf").read_text()
check("生成 Gaussian 抵消校正输入 (全部近邻组)")
for f in (tmp / "cp").glob("G*_cp.gjf"):
    f.with_suffix(".log").write_text(" BSSE energy =  0.0021\n complexation energy = -6.10 kcal/mole (raw)\n"
                                     " complexation energy = -4.80 kcal/mole (corrected)\n Normal termination of Gaussian 16\n")
rd6 = gui.RunDialog(app, str(tmp / "cp")); rd6.auto_read([(str(f), True, "") for f in (tmp / "cp").glob("G*_cp.gjf")])
assert it.method_label.startswith("Gaussian") and abs(next(iter(it.energies.values()))["total"] + 20.08) < 0.01
check("运行结束后自动读取抵消校正结果")
it.src.set(it.SRC[1]); it.dimer.set(tmp / "pair.xyz"); it.na.set(""); it.method.set(it.METHODS[0]); it.compute()
assert len(it.energies) == 1; it.src.set(it.SRC[0]); check("单个二聚体文件的相互作用能")
pick(tmp / "inter.csv"); it.export_csv(); assert (tmp / "inter.csv").exists(); check("导出相互作用能 CSV")


# ---- 对比后自动填表 / 直接读入 / 黄昆因子导入 / FCHT 检查
ti = app.ti_tab; ti.cmp_dir.set(gdir); app.mob_tab.set_paths([]); ti.fill_method.set("自动 (精度最高)"); ti.compare()
rows_ = app.mob_tab.tree.rows(); assert rows_ and rows_[0][-1] not in ("", None); check(f"方法对比完成后自动填入迁移率表 ({len(rows_)} 条路径)")
assert ti.compare_to_mobility() == "Gaussian"; check("“自动”选择精度最高且所有组都有结果的方法 (Gaussian)")
app.mob_tab.set_paths([]); ti.dir_to_mobility(); assert app.mob_tab.tree.rows() and app.page_var.get() == "迁移率"
check("直接读入迁移率表 (Gaussian 目录全部二聚体)")
hr = tmp / "hr.dat"; hr.write_text("# mode freq S\n1 120 0.30\n2 760 0.08\n3 1580 0.42\n")
dt_ = app.decomp_tab; dt_.state_var.set("neutral"); r_ = dt_.import_hr(str(hr)); assert len(r_["modes"]) == 3 and "已读取" in dt_.text.buf
adv_ = app.adv_tab; adv_.vsrc.set("分解页 λ1 (末态=中性/基态)"); adv_.run_vib(); check("读取黄昆因子结果 → 作图 / 振动分辨光谱")
from mobility_tool.gaussian_fcht import fcht_input
fjd = tmp / "fchtjob"; fjd.mkdir(exist_ok=True); (fjd / "x_fcht_cation_to_neutral.gjf").write_text(fcht_input("x_cation.chk", "x_neutral.chk"))
rd7 = gui.RunDialog(app, str(fjd)); rd7.load(); rd7.tree._sel = rd7.tree.get_children()
try:
    rd7.run(); raise SystemExit("[失败] 应拦下缺少 chk 的 FCHT 任务")
except ValueError as e:
    assert "x_cation.chk 不在任务目录" in str(e)
check("运行前检查 FCHT 所需 chk (缺失时不启动)")
(fjd / "x_fcht_cation_to_neutral.log").write_text(" FileIO operation on non-existent file.\n Error termination in NtrErr:\n")
dt_.source.set("fcht"); dt_.l1.set(fjd / "x_fcht_cation_to_neutral.log"); dt_.l2.set("")
try:
    dt_.run(); raise SystemExit("[失败]")
except ValueError as e:
    assert "FileIO operation on non-existent file" in str(e) and "内置" in str(e)
dt_.source.set("builtin"); check("读取失败的 FCHT 输出时给出原因与建议")


# ---- 原子取代
st = app.struct_tab
pick(ROOT / "examples/benzene.xyz"); st.open_file(); st.sub_mode.set(True); st.toggle_sub()
kC = st.scene_syms.index("C"); st.toggle_sub_atom(kC)
dlg = gui.PeriodicTableDialog(app, lambda el: st.apply_sub(new=el)); assert len(dlg.buttons) == 118; dlg.pick("N")
assert len(st.atoms) == 11 and [a[0] for a in st.atoms].count("N") == 1; check("单分子: 周期表选元素 C–H → N (自动删除 H, 118 个元素)")
st.undo_sub(); assert len(st.atoms) == 12; check("撤销取代")
kH = st.scene_syms.index("H"); st.toggle_sub_atom(kH); st.sub_group.set("CH3"); st.apply_group(); assert len(st.atoms) == 15
check("单分子: H → −CH₃ 基团取代")
p_ = st.export_sub(tmp / "sub.gjf"); assert "opt freq" in p_.read_text(); st.sub_to_reorg()
assert app.reorg_tab.geom.get().endswith("substituted_molecule.xyz") and app.page_var.get() == "重组能"; check("导出 gjf / 送去重组能计算")
pick(tmp / "t.cif"); st.open_file(); st.sub_mode.set(True); st.toggle_sub()
k0 = next(k for k in range(len(st.scene_syms)) if st.scene_owner[k] == 0 and st.scene_syms[k] == "C")
st.toggle_sub_atom(k0); st.apply_sub(new="N")
assert all(len(m["symbols"]) == 11 for m in st.mols); check("晶体: 对称等价位置全部取代 (4 个分子)")
st.cut.set("4.5"); st.search(); assert st.dimer_res["groups"]; st.eht_all(); check("取代后搜索二聚体并计算 EHT")
st.export_sub(tmp / "sub.cif"); from mobility_tool.crystal import read_cif as _rc, find_molecules as _fm
assert {len(m["symbols"]) for m in _fm(_rc(tmp / "sub.cif"))} == {11}; check("导出取代后的 CIF (P1) 可重新读入")
st.undo_sub(); assert all(len(m["symbols"]) == 12 for m in st.mols)
st.sub_sym.set(False); st.rebuild(); st.sub_mode.set(True); st.toggle_sub()
k1 = next(k for k in range(len(st.scene_syms)) if st.scene_owner[k] == 1 and st.scene_syms[k] == "H")
k2 = next(k for k in range(len(st.scene_syms)) if st.scene_owner[k] == 2)
st.toggle_sub_atom(k1)
try:
    st.toggle_sub_atom(k2); raise SystemExit("[失败] 应拒绝跨分子选择")
except ValueError:
    pass
st.apply_sub(new="Cl"); assert [len(m["symbols"]) for m in st.mols].count(12) == 4 and sum(m["symbols"].count("Cl") for m in st.mols) == 1
st.sub_sym.set(True); st.sub_mode.set(False); st.toggle_sub(); check("晶体: 仅所选分子取代 / 拒绝跨分子选择")


# ---- 转移积分深入分析 (ti / tcal / CATNIP) / 几何检查 / ADF 片段共用与进度
sys.path.insert(0, str(ROOT / "tests"))
from fake_fchk import write_fake_fchk
ti = app.ti_tab; ti.prog.set("eht"); ti.dimer.set(tmp / "pair.xyz"); ti.na.set("12"); ti.an_src.set("EHT (上方二聚体文件)")
rows = ti.level_matrix(); assert len(rows) == 16 and "J_eff" in ti.text.buf; check("多能级转移积分矩阵 (EHT, H−1…L+1)")
r_ = ti.atom_pairs(); assert abs(r_["total_meV"] - r_["J_eff_meV"]) < 1e-6; check("原子对分解：Σu_αβ = J_eff (EHT)")
ti.export_apta(tmp / "apta.csv"); assert "atom_A" in (tmp / "apta.csv").read_text(encoding="utf-8-sig"); check("导出原子对分解 CSV")
from mobility_tool.gaussian_io import read_xyz as _rx
pa = _rx(tmp / "pair.xyz"); A_, B_ = pa[:12], pa[12:]
write_fake_fchk(tmp / "gA.fchk", A_); write_fake_fchk(tmp / "gB.fchk", B_); write_fake_fchk(tmp / "gAB.fchk", B_ + A_)   # 二聚体中 B 在前
ti.an_src.set("Gaussian fchk (A、B、AB)"); ti.f_a.set(tmp / "gA.fchk"); ti.f_b.set(tmp / "gB.fchk"); ti.f_ab.set(tmp / "gAB.fchk")
r2 = ti.atom_pairs(); assert abs(r2["J_eff_meV"] - r_["J_eff_meV"]) < 1e-3; check("Gaussian fchk：原子顺序不同时自动匹配，结果与 EHT 一致")
ti.an_src.set("EHT (上方二聚体文件)")
ti.prog.set("gaussian"); ti.mb.set(""); ti.ma.set(""); same = tmp / "same.xyz"
from mobility_tool.gaussian_io import write_xyz as _wx
_wx(same, A_ + A_, "same molecule twice"); ti.dimer.set(same); ti.na.set("12"); ti.outdir.set(tmp / "badti")
try:
    ti.generate(); raise SystemExit("[失败] 应拦截重合的二聚体")
except ValueError as e:
    assert "同一个分子" in str(e)
ti.dimer.set(tmp / "pair.xyz"); check("生成输入前的几何检查 (同一分子选两次被拦截)")
from mobility_tool.crystal import read_cif as _rc2, find_molecules as _fm2, find_dimers as _fd2
from mobility_tool.workflow import prepare_dimer_jobs as _pdj
rc = _rc2(ROOT / "tests/data/Rubrene.cif"); rm_ = _fm2(rc); rr = _fd2(rc, rm_, cutoff=4.0)
_pdj(rr, tmp / "rub", "adf", functional="PW91", basis="TZP", style="classic")
assert "1 个单体片段 + 4 个二聚体" in (tmp / "rub" / "batch_adf.sh").read_text(); check("红荧烯 ADF 批量：4 组二聚体共用 1 个单体片段")
rd8 = gui.RunDialog(app, str(tmp / "rub")); rd8._running = str(tmp / "rub" / "batch_adf.sh")
assert rd8.batch_progress(force=True) == (0, 4); check("ADF 批量运行进度 (0 / 4 个二聚体)")
from mobility_tool.programs import ams_env as _ae
assert _ae({"home": "/tmp/fake2016/ADF2016.101", "kind": "classic", "nscm": "6"})["NSCM"] == "6"; check("ADF 并行核数写入 NSCM")


# ---- 按方向选择二聚体 (Deng 2015) / 光谱基组写法 / 开发者信息
st = app.struct_tab; pick(ROOT / "tests/data/Rubrene.cif"); st.open_file()
st.sel_mode.set("按方向 (传输平面内，Deng 2015)"); st.plane.set("自动 (分子层)"); st.crit.set("最短原子间距"); st.cut.set("4.0"); st.search()
labs = [g["label"] for g in st.dimer_res["groups"]]; assert labs == ["P", "T1", "T2"], labs
check("红荧烯按方向选择 (bc 分子层)：P、T1、T2")
st.crit.set("质心距离"); st.cut.set("15"); st.search(); assert len(st.dimer_res["groups"]) == 6
kids = list(st.tree.get_children()); st.tree._sel = [k for k, d in zip(kids, st.dimer_res["dimers"]) if abs(d["r_centroid"] - 12.93) < 0.05]
st.remove_selected_groups(); labs = [g["label"] for g in st.dimer_res["groups"]]; assert labs == ["P", "T1", "T2", "L3"], labs
check("质心 15 Å：P、T1、T2、L1–L3；移除 L1、L2 后剩 4 个 (与论文数目一致)")
st.eht_all(); assert len(app.data["crystal_couplings"]) == 4; check("EHT 计算沿用按方向选择的 4 个二聚体")
from mobility_tool.workflow import prepare_dimer_jobs as _pdj2
_pdj2(st.dimer_res, tmp / "rubpaper", "gaussian", method="B3LYP", basis="6-31G(d)")
import json as _js; jl = [g["label"] for g in _js.loads((tmp / "rubpaper" / "dimers.json").read_text())["groups"]]
assert jl == ["P", "T1", "T2", "L3"]; check("导出 dimers.json 带二聚体标签")
st.sel_mode.set("全部近邻 (按对称等价分组)"); st.crit.set("最短原子间距"); st.cut.set("4.0")
ph = app.photo_tab
from mobility_tool.photophys import generate_excited_inputs as _gei
_gei(ROOT / "examples/benzene.xyz", tmp / "exc_v13", name="bz")
assert "Def2TZVP" in (tmp / "exc_v13" / "bz_S0.gjf").read_text() and "def2-TZVP" not in (tmp / "exc_v13" / "bz_S0.gjf").read_text()
check("激发态输入：def2-TZVP 自动写为 Gaussian 格式 Def2TZVP")
import shutil as _sh; (tmp / "excerr").mkdir(exist_ok=True); _sh.copy(ROOT / "tests/data/G2_A_S0_td_error.out", tmp / "excerr" / "G2_A_S0.out")
from mobility_tool.photophys import analyze_excited as _ae2
try:
    _ae2(tmp / "excerr", "G2_A"); raise SystemExit("[失败]")
except ValueError as e:
    assert "Def2" in str(e) and "语法错误" in str(e)
check("读取失败的 TD 输出：指出基组写法错误")
app.about(); assert "Yarui Shi" in gui.APP_DEVELOPER and "安阳师范学院" in gui.APP_DEVELOPER; check("关于窗口显示开发者 Yarui Shi (安阳师范学院)")


# ---- KMC 精度自检 / fchk 自动补全
mt = app.mob_tab; mt.kmc_selfcheck(); assert mt.selfcheck_rows and "精度自检" in mt.text.buf
mt.export_selfcheck(tmp / "kmc_check.csv"); assert (tmp / "kmc_check.csv").exists() and (tmp / "kmc_check.png").exists()
assert max(abs(r[5]) for r in mt.selfcheck_rows) < 3.5; check(f"KMC 精度自检 ({len(mt.selfcheck_rows)} 项，全部在统计误差内) 与导出")
ti = app.ti_tab; ti.f_a.set(""); ti.f_b.set("")
import shutil as _sh3; _sh3.copy(tmp / "gA.fchk", tmp / "pp_A.fchk"); _sh3.copy(tmp / "gB.fchk", tmp / "pp_B.fchk"); _sh3.copy(tmp / "gAB.fchk", tmp / "pp_AB.fchk")
ti._fill_ab(str(tmp / "pp_AB.fchk")); assert ti.f_a.get().endswith("pp_A.fchk") and ti.f_b.get().endswith("pp_B.fchk")
check("选中二聚体 fchk 自动补全单体 A、B")


# ---- 波函数分析 (轨道 / 电荷与 Mayer 键级 / ESP / NBO) 与引用文献
from fake_fchk import write_fake_fchk as _wff
from mobility_tool.gaussian_io import read_xyz as _rx2
_wff(tmp / "bz_wfn.fchk", _rx2(ROOT / "examples/benzene.xyz"))
wt = app.wfn_tab; wt.fchk.set(tmp / "bz_wfn.fchk"); wfn = wt.load(); assert len(wt.orb.get_children()) > 10
check("读取 fchk 并列出轨道 (HOMO / LUMO 标注)")
wt.show_orbital("HOMO"); assert len(wt.view.renderer.meshes) == 2 and len(wt.view.renderer.meshes[0]["faces"]) > 100
wt.export_cube(tmp / "homo.cube"); from mobility_tool.wavefunction import read_cube as _rc3
cc = _rc3(tmp / "homo.cube"); assert cc["data"].shape == wt.cube["data"].shape and abs(cc["data"] - wt.cube["data"]).max() < 1e-4
check("轨道等值面 (±) 与 cube 导出 / 重新读入")
r_ = wt.mulliken(); assert abs(r_["total_charge"]) < 1e-6 and wt.view.renderer.annotations; check("Mulliken 电荷 + Mayer 键级 (图中标注)")
wt.esp_src.set(wt.ESP_SRC[2]); wt.run_esp(); assert wt.esp and wt.esp["extrema"]; wt.export_esp(tmp / "esp.csv")
check(f"ESP 表面 (原子电荷近似)：{len(wt.esp['extrema'])} 个极值点")
from mobility_tool.wavefunction import write_cube as _wc
_wc(tmp / "d.cube", wfn.density_cube(0.3)); dc = _rc3(tmp / "d.cube"); dc2 = dict(dc, data=dc["data"] * 0 + 0.01 * (dc["data"] > 1e-3))
_wc(tmp / "e.cube", dict(dc, data=np.where(dc["data"] > 0, -0.02, 0.0) + 0.0))
wt.esp_src.set(wt.ESP_SRC[1]); wt.dens_cube.set(tmp / "d.cube"); wt.esp_cube.set(tmp / "e.cube"); wt.run_esp()
assert wt.esp["method"].startswith("ESP cube"); check("ESP 表面 (读入密度 + ESP cube)")
nbo_txt = """ SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS

     Threshold for printing:   0.50 kcal/mol
                                                          E(2)  E(j)-E(i) F(i,j)
         Donor NBO (i)                     Acceptor NBO (j)       kcal/mol   a.u.    a.u.
 ===================================================================================================

 within unit  1
   1. BD (   2) C   1 - C   2      /  10. BD*(   2) C   3 - C   4            20.16    0.28    0.068
   3. LP (   1) N   7              /  12. BD*(   1) C  13 - H  14             2.10    0.95    0.040
 from unit  1 to unit  2
   5. BD (   2) C   3 - C   4      /  40. BD*(   2) C  15 - C  16             0.85    0.30    0.014

 NATURAL BOND ORBITALS (Summary):
 Summary of Natural Population Analysis:

                                     Natural Population
             Natural    -----------------------------------------------
    Atom No    Charge         Core      Valence    Rydberg      Total
 -----------------------------------------------------------------------
      C    1   -0.20000      1.99900     4.18000    0.02100     6.20000
      H    2    0.20000      0.00000     0.79800    0.00200     0.80000
 =======================================================================
"""
(tmp / "nbo.log").write_text(nbo_txt); wt.nbo_log.set(tmp / "nbo.log"); wt.e2_min.set("0.5")
rows_ = wt.read_nbo(); assert len(rows_) == 3 and rows_[0]["E2"] == 20.16 and wt.nbo["npa"][0][2] == -0.2
wt.nbo_na.set("12"); rows2 = wt.read_nbo(); assert [r["E2"] for r in rows2] == [2.10, 0.85], rows2   # N7(A)→C13–H14(B)、C3–C4(A)→C15–C16(B)
wt.nbo_na.set(""); check("NBO：E(2) 表、自然电荷、二聚体分子间筛选")
cd_ = gui.CitationDialog(app); assert "Mayer" in cd_.content and "Valeev" in cd_.content and "Shi, Y. Mobilis" in cd_.content
check("帮助 → 引用文献")


# ---- IGM / IGMH 弱相互作用与 AIM 拓扑
from mobility_tool.gaussian_io import read_xyz as _rx4, write_xyz as _wx4
bz4 = _rx4(ROOT / "examples/benzene.xyz"); dimer4 = bz4 + [(s_, x + 1.0, y, z + 3.6) for s_, x, y, z in bz4]
_wx4(tmp / "bzdimer.xyz", dimer4, "slipped benzene dimer"); _wff(tmp / "bzdimer.fchk", dimer4)
wt = app.wfn_tab; wt.fchk.set(tmp / "bzdimer.fchk"); wt.load()
wt.ig_src.set(wt.IG_SRC[0]); wt.ig_h.set("0.25"); wt.run_igm(); assert wt.igm["method"] == "IGMH" and wt.igm["dG_inter"] > 0
check(f"IGMH (fchk)：δG_inter = {wt.igm['dG_inter']:.3f}")
wt.ig_src.set(wt.IG_SRC[2]); wt.struct_file.set(tmp / "bzdimer.xyz"); wt.ig_h.set("0.20"); wt.run_igm()
assert wt.igm["method"].startswith("IGM") and len(wt.view.renderer.meshes[0]["faces"]) > 100
pp_ = wt.igm_scatter(); wt.export_igm(tmp / "igm.cube"); assert (tmp / "igm_dg_inter.cube").exists()
check("IGM (结构文件)：等值面、散点图、cube 导出")
st = app.struct_tab; pick(ROOT / "tests/data/Rubrene.cif"); st.open_file(); st.sel_mode.set("全部近邻 (按对称等价分组)"); st.cut.set("4.0"); st.search()
wt.ig_src.set(wt.IG_SRC[1]); wt.ig_h.set("0.25"); wt.run_igm(); assert wt.igm["dG_inter"] > 0
check("IGM 直接分析晶体二聚体 (红荧烯)")
wt.aim_src.set(wt.IG_SRC[2]); wt.run_aim(); assert wt.aim["ph_ok"]; wt.export_aim(tmp / "aim.csv")
check(f"AIM (前分子密度)：{wt.aim['counts']}，Poincaré–Hopf 成立")
_wff(tmp / "h2.fchk", [("H", 0, 0, 0), ("H", 0, 0, 0.74)]); wt.fchk.set(tmp / "h2.fchk"); wt.load()
wt.aim_src.set(wt.IG_SRC[0]); wt.run_aim(); c0 = wt.aim["cps"][0]; assert wt.aim["counts"]["BCP"] == 1 and "G" in c0
check("AIM (fchk 实际密度)：H2 键临界点与 G / V / H")

# ---- 批量扫描工作流
import webbrowser as _wb; _wb.open = lambda *a, **k: True
from test_v17 import _pentacene
from fake_adf import run_scan as _run_scan
_pentacene(tmp / "pent.xyz")
sc = app.scan_tab
sc.mono.set(tmp / "pent.xyz"); sc.pname.set("pent"); pick(tmp / "sp"); sc.new_project(); assert sc.proj.ops and len(sc.proj.ops) == 8
sc.xc.set("PW91"); sc.basis.set("TZP"); sc.add_level(); sc.xc.set("PBE"); sc.basis.set("DZP"); sc.d3.set(True); sc.add_level()
for lv_ in sc.proj.meta["levels"]:
    (sc.proj.level_dir(lv_) / "mono.t21").write_bytes(b"x")
sc.level.set("PW91_TZP"); check("新建扫描项目、添加两个层级")
sc.kind.set(sc.KINDS[0])
for nm, z_, dx_, dy_ in (("zscan", "3.2:4.4:0.2", "0", "0"), ("slip35", "3.5", "0:6:1", "0:3:1"), ("slip38", "3.8", "0:6:1", "0:3:1")):
    sc.sname.set(nm); sc.p_z.set(z_); sc.p_dx.set(dx_); sc.p_dy.set(dy_); sc.preview_count(); sc.create_scan()
sc.kind.set(sc.KINDS[1]); sc.sname.set("valid"); sc.v_ni.set("6"); sc.v_ne.set("6"); sc.preview_count(); sc.create_scan()
sc.kind.set(sc.KINDS[2]); sc.sname.set("symcheck"); sc.create_scan()
assert set(sc.tree.get_children()) == {"zscan", "slip35", "slip38", "valid", "symcheck"}
check("生成网格 / 验证集 / 对称检验扫描")
for nm in sc.proj.scans():
    _run_scan(sc.proj, nm)
for nm in ("zscan", "slip35", "valid", "symcheck"):
    sc.tree.selection_set([nm]); r_ = sc.precheck(); assert r_["ok"] and r_["n_pending"] == 0; sc.collect()
sc.tree.selection_set(["symcheck"]); sc.sym_result(); assert "C2x" in sc.text.buf
sc.tree.selection_set(["slip35"])
for k_ in [x for x in sc.PREVIEW if "ln" not in x]:
    sc.pv_kind.set(k_); sc.preview()
sc.tree.selection_set(["zscan"]); sc.pv_kind.set(sc.PREVIEW[0]); sc.preview(); assert "β" in sc.plot.ax.get_title()
check("运行前检查、收集结果、对称检验、预览图")
sc.tree.selection_set(["slip35"]); sc.level.set("PBE_DZP_D3BJ"); sc.clone(); _run_scan(sc.proj, "slip35_PBE_DZP_D3BJ")
pick(tmp / "sp" / "pent" / "scans" / "slip35"); sc.gauss_cp(); sc.gauss_ti()
check("换层级重算、Gaussian 抵消校正 / 转移积分输入")
sc.level.set("PW91_TZP"); sc.lam_h.set("0.092"); sc.lam_e.set("0.132"); sc.report()
assert (sc.proj.root / "report" / "report.html").exists() and (sc.proj.root / "model" / "model.json").exists()
pick(tmp / "arch.zip"); sc.archive(); assert (tmp / "arch.zip").exists()
check("扫描报告与归档")
sc2 = gui.ScanTab(None, app); pick(sc.proj.root); sc2.open_project(); assert len(sc2.tree.get_children()) == 6
check("重新打开扫描项目")

# ---- CEDS
st = app.struct_tab; pick(ROOT / "tests/data/pentacene_90K.cif"); st.open_file(); st.cut.set("4.0")
sc.ce_src.set(sc.CE_SRC[1]); sc.ce_n.set("150"); sc.ce_T.set("150 300"); sc.ce_htau.set("5")
sc.ceds_run(); assert "G1" in sc.text.buf and 300.0 in sc.ce_results
sc.ceds_muT(); assert sc.ce_rows and sc.ce_rows[1]["mu_hole"] > 0 and "μ ∝ T^" in sc.text.buf
pick(tmp / "ceds.csv"); sc.ceds_export(); assert "sigma_meV" in (tmp / "ceds.csv").read_text(encoding="utf-8-sig")
check("CEDS (EHT 曲面)：σ_V、μ(T) 与导出")
sc.sname.set("cedsloc"); sc.ce_nloc.set("45"); sc.level.set("PW91_TZP"); sc.ceds_local()
assert "cedsloc" in sc.tree.get_children(); _run_scan(sc.proj, "cedsloc"); sc.tree.selection_set(["cedsloc"]); sc.collect()
sc.ce_src.set(sc.CE_SRC[0]); sc.ce_T.set("300"); sc.ceds_run(); assert "local" in sc.text.buf
sc.ce_ueq.set("0.012"); sc.ceds_run(); assert "势能缩放" in sc.text.buf and sc.ce_scale < 1.0; sc.ce_ueq.set("")
check("CEDS：局域扫描 → 局域曲面 (项目数据) → σ_V；U_eq 校准")

# ---- 全晶体刚体 MD (局域结果表曲面；小超胞、短轨迹)
pick(ROOT / "tests/data/pentacene_RT.cif"); st.open_file(); st.cut.set("4.0")
sc.ce_src.set(sc.CE_SRC[2]); sc.ce_lcsv.set(ROOT / "tests/data/pentacene_cedsP_results.csv")
sc.ce_T.set("300"); sc.ce_sc.set("2 2 1"); sc.ce_mdps.set("0.5"); sc.ce_traj.set(True)
sc.ceds_md()
assert sc.ce_md_rows and sc.ce_md_rows[0]["U_eq"] > 0 and "ħ/τc" in sc.text.buf and "[结果表]" in sc.text.buf
assert list((sc.proj.root / "md").glob("md_300K.npz"))
pick(tmp / "ceds_md.csv"); sc.ceds_export(); assert "hbar_tau_MD_meV" in (tmp / "ceds_md.csv").read_text(encoding="utf-8-sig")
sc.ce_traj.set(False)
check("全晶体 MD：σ_V、μ(T)、轨迹保存与导出")
sc.sname.set("mdsnap"); sc.ce_nloc.set("8"); sc.ce_pairs.set("3"); sc.level.set("PW91_TZP"); sc.ceds_snapshots()
assert "mdsnap" in sc.tree.get_children() and "md_300K.npz" in sc.text.buf
_run_scan(sc.proj, "mdsnap"); sc.tree.selection_set(["mdsnap"]); sc.collect(); sc.ceds_validate()
assert "σ 比值" in sc.text.buf and sc.ce_validation and 3 in sc.ce_validation
pick(tmp / "snapval.csv"); sc.ceds_export(); assert "sigma_ratio" in (tmp / "snapval.csv").read_text(encoding="utf-8-sig")
sc.ce_pairs.set("")
check("MD 快照 → ADF 批处理 → 快照检验与导出")

# ---- 迁移率方法对比与左侧导航
app.show_page("迁移率方法对比"); assert app.page_var.get() == "迁移率方法对比" and app.cmp_tab is app.tabs[-1]
assert [p_ for _g, ps_ in app.NAV for p_ in ps_].count("迁移率方法对比") == 1
assert sorted(p_ for _g, ps_ in app.NAV for p_ in ps_) == sorted(app.STEPS)
cp = app.cmp_tab; cp.src.set(cp.SRC[2]); cp.temp.set("300"); cp.lam_h.set("0.092"); cp.lam_e.set("0.132")
cp.sig_e.set("30"); cp.walk.set("100 600"); cp.run()
assert cp.result and "瞬态局域化" in cp.text.buf and "适用性" in cp.text.buf
assert any(r_["key"] == "ceds" for r_ in cp.result["hole"]["rows"])
pick(tmp / "methods.csv"); cp.export(); assert "applicability" in (tmp / "methods.csv").read_text(encoding="utf-8-sig")
cp.src.set(cp.SRC[0]); cp.sig_e.set("0"); cp.run(); assert "Marcus" in cp.text.buf
check("迁移率方法对比 (CEDS 与晶体 J 两种来源)、导出；左侧导航包含全部页面")

# ---- 极化位点能 → 方法对比
app.show_page("极化位点能"); pt = app.pol_tab
pt.cif.set(ROOT / "tests/data/pentacene_RT.cif"); pt.radii.set("7 9 11"); pt.ip.set("6.59"); pt.ea.set("1.39"); pt.run()
assert pt.res and "P+" in pt.text.buf and "固态电离能" in pt.text.buf and len(pt.res["offsets_h"]) == 2
pt.md_nf.set("2 2"); pt.md_r.set("8"); pt.md_T.set("300 0.2"); pt.md_sc.set("2 2 1"); pt.run_md()
assert pt.md and "σ_E" in pt.text.buf
pick(tmp / "pol.csv"); pt.export(); assert "P_plus_eV" in (tmp / "pol.csv").read_text(encoding="utf-8-sig")
pt.to_compare(); assert app.page_var.get() == "迁移率方法对比"
cp.src.set(cp.SRC[2]); cp.sig_e.set("30"); cp.run()
assert any(r_["key"] == "kmc_polar" and r_["mu"] > 0 for r_ in cp.result["hole"]["rows"])
check("极化位点能：晶体 P±、固态 IE/EA、MD 快照 σ_E、导出，送到方法对比页 (Marcus KMC + 极化位点能)")

# ---- 无定形形貌 (GROMACS) → 方法对比
app.show_page("无定形形貌 (MD)"); mt = app.morph_tab
mt.gro.set(ROOT / "tests/data/pentacene_300K.gro"); mt.ref.set(ROOT / "tests/data/pentacene_ref.xyz"); mt.frame_.set("1")
mt.load(); assert mt.m and mt.m.M == 48 and "近邻" in mt.text.buf
mt.energies(); assert mt.m.E is not None and "位点能" in mt.text.buf
mt.lam_h.set("0.092"); mt.lam_e.set("0.132"); mt.walk.set("100 500"); mt.field_.set("1e5"); mt.run_kmc()
assert "加位点能" in mt.text.buf and set(mt.kres) == {"hole", "electron"}
pick(tmp / "morph_sites.csv"); mt.export(); assert (tmp / "morph_pairs.csv").exists() and (tmp / "morph.png").exists()
mt.maxp.set("2"); mt.pick.set(mt.PICK[1]); pick(tmp / "dimers"); mt.export_dimers(); assert (tmp / "dimers/pair_00001/dimer_AB.gjf").exists()
mt.maxp.set("5"); mt.pick.set(mt.PICK[0]); pick(tmp / "dimers_cal"); mt.export_dimers()
assert len(list((tmp / "dimers_cal").glob("pair_*"))) == 5
mt.kh.set("2.2"); mt.ke.set("2.4"); mt.apply_k(); assert "手动" in mt.calib_lbl._kw.get("text", "") or "DFT 校准" in mt.text.buf
check("无定形形貌：按 |J| 分布挑校准用二聚体、手动比例系数")
mt.to_compare(); assert cp.src.get() == cp.SRC[3]
cp.sig_e.set("0"); cp.walk.set("80 400"); cp.run()
keys = {r_["key"] for r_ in cp.result["hole"]["rows"]}
assert "kmc_polar" in keys and "tlt" not in keys and "marcus_tensor" not in keys
check("无定形形貌：读取 .gro、映射、近邻与 EHT J、位点能、KMC、导出与 Gaussian 二聚体输入，送到方法对比页")

# ---- ZINDO/MOO：转移积分页与形貌页
from mobility_tool.gaussian_io import write_xyz as _wx
from mobility_tool.cli import _read_monomer as _rm
_s, _Y = _rm(ROOT / "tests/data/pentacene_ref.xyz")
_wx(tmp / "pent_dimer.xyz", [(a_, *x_) for a_, x_ in zip(_s, _Y)] + [(a_, *x_) for a_, x_ in zip(_s, _Y + [0, 1.2, 3.5])])
app.show_page("转移积分"); ti = app.ti_tab
ti.prog.set("zindo"); ti.dimer.set(tmp / "pent_dimer.xyz"); ti.na.set("36"); ti.deg.set("10"); ti.calc()
assert "ZINDO/MOO" in ti.text.buf and "二聚体 INDO/S" in ti.text.buf and app.data["last_J"]["hole"] != 0
mt.jm.set(mt.JMETH[1]); mt.load(); assert mt.m.J_source.startswith("ZINDO") and "ZINDO" in mt.text.buf
check("ZINDO/MOO：转移积分页 (含二聚体 INDO/S 投影与简并阈值)、形貌页的 ZINDO 耦合")

# ---- 英文界面：所有显示文字不应再含中文
import re as _re
cfg = tmp / "cfg_en.json"; cfg.write_text('{"lang": "en"}'); gui.CONFIG = cfg
SHOWN.clear()
en = gui.App(); en.sync = True
pick(tmp / "t.cif"); en.struct_tab.open_file(); en.struct_tab.cut.set("4.5"); en.struct_tab.search(); en.struct_tab.eht_all()
en.reorg_tab.geom.set(ROOT / "examples/benzene.xyz"); en.reorg_tab.outdir.set(tmp / "reorg_en"); en.reorg_tab.generate()
for f_ in ("benzene_neutral", "benzene_cation", "benzene_anion"):
    shutil.copy(tmp / "reorg" / f"{f_}.log", tmp / "reorg_en" / f"{f_}.log")
en.reorg_tab.analyze(); en.mob_tab.lam_e.set("0.2"); en.mob_tab.lam_h.set("0.2"); en.mob_tab.compute()
en.mob_tab.k_walk.set("300 500"); en.mob_tab.run_kmc()
en.mob_tab.compute3d()
en.ti_tab.prog.set("eht"); en.ti_tab.dimer.set(tmp / "pair.xyz"); en.ti_tab.calc()
en.ti_tab.text.buf_eht = en.ti_tab.text.buf
en.ti_tab.prog.set("zindo"); en.ti_tab.dimer.set(tmp / "pent_dimer.xyz"); en.ti_tab.na.set("36"); en.ti_tab.calc()
SHOWN.append(en.ti_tab.text.buf_eht)
en.photo_tab.resdir.set(tmp / "exc")
try:
    en.photo_tab.analyze()
except ValueError as e_:                       # 模拟输出中没有激发态：错误提示也应为英文
    SHOWN.append(gui.T(str(e_)))
en.decomp_tab.fn.set(tmp/"reorg/benzene_neutral.fchk"); en.decomp_tab.fi.set(tmp/"reorg/benzene_cation.fchk")
en.decomp_tab.source.set("builtin"); en.decomp_tab.run()
es = en.scan_tab; pick(sc.proj.root); es.open_project(); es.tree.selection_set(["slip35"]); es.precheck(); es.preview()
es.tree.selection_set(["symcheck"]); es.sym_result(); es.kind.set(es.KINDS[0]); es.preview_count()
es.tree.selection_set(["zscan"]); es.preview()
en.struct_tab.open_path(str(ROOT / "tests/data/pentacene_90K.cif")) if hasattr(en.struct_tab, "open_path") else None
es.ce_src.set(es.CE_SRC[1]); es.ce_n.set("100"); es.ce_T.set("300"); es.ceds_run(); es.ceds_muT()
es.ce_sc.set("2 2 1"); es.ce_mdps.set("0.5"); es.ceds_md()
ec = en.cmp_tab; ec.src.set(ec.SRC[2]); ec.lam_h.set("0.2"); ec.lam_e.set("0.2"); ec.sig_e.set("30"); ec.walk.set("80 400")
ec.temp.set("300"); ec.run(); en.show_page("迁移率方法对比")
ep = en.pol_tab; ep.cif.set(ROOT / "tests/data/pentacene_RT.cif"); ep.radii.set("7 9"); ep.ip.set("6.59"); ep.ea.set("1.39")
ep.run(); ep.md_nf.set("2 2"); ep.md_r.set("8"); ep.md_T.set("300 0.2"); ep.md_sc.set("2 2 1"); ep.run_md()
en.show_page("极化位点能")
em = en.morph_tab; em.gro.set(ROOT / "tests/data/pentacene_300K.gro"); em.ref.set(ROOT / "tests/data/pentacene_ref.xyz")
em.jm.set(em.JMETH[1]); em.load(); em.energies(); em.lam_h.set("0.092"); em.lam_e.set("0.132"); em.walk.set("60 300"); em.run_kmc()
ec.src.set(ec.SRC[3]); ec.sig_e.set("0"); ec.run(); en.show_page("无定形形貌 (MD)")
texts = SHOWN + [en.morph_tab.text.buf, en.pol_tab.text.buf, en.cmp_tab.text.buf, en.logbox.buf, en.reorg_tab.text.buf, en.mob_tab.text.buf, en.ti_tab.text.buf, en.photo_tab.text.buf,
                 en.decomp_tab.text.buf, es.text.buf]
import matplotlib.text as _mt
for tab in (en.mob_tab, en.decomp_tab, en.ti_tab, es, en.pol_tab, en.morph_tab, en.cmp_tab):
    tab.plot.canvas.draw_idle()
    texts += [x.get_text() for x in tab.plot.fig.findobj(_mt.Text)]
left = sorted({x for x in texts if _re.search(r"[\u4e00-\u9fff]", x) and x != "中文"})
print("[英文界面] 显示文字", len(texts), "条；仍含中文:", len(left))
for x in left[:400]:
    print("   ", repr(x)[:140])
assert en.mob_tab.lam_e.get() == "0.2" and en.struct_tab.crit.get() == "最短原子间距"
check("英文界面下下拉框仍返回内部取值")
