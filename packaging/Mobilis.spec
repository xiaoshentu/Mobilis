# PyInstaller 备选方案 (构建快，但保护弱：pyc 可被提取和反编译；正式发布请用 build_windows.bat 的 Nuitka 方案)
# 用法: pyinstaller packaging\Mobilis.spec --noconfirm
import os
root = os.path.abspath(os.path.join(SPECPATH, ".."))
a = Analysis([os.path.join(root, "run_gui.py")], pathex=[root],
             datas=[(os.path.join(root, "mobility_tool", "assets"), "mobility_tool/assets")],
             hiddenimports=["matplotlib.backends.backend_tkagg", "skimage.measure._marching_cubes_lewiner",
                            "mobility_tool.scanflow", "mobility_tool.scananalysis", "mobility_tool.ceds", "mobility_tool.rigidmd", "mobility_tool.compare", "mobility_tool.polar", "mobility_tool.morphology", "mobility_tool.zindo", "scipy.special", "scipy.stats._qmc", "scipy.optimize"],
             excludes=["tests", "pytest", "IPython"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Mobilis", console=False,
          icon=os.path.join(root, "mobility_tool", "assets", "mobilis.ico"))
coll = COLLECT(exe, a.binaries, a.datas, name="Mobilis")
