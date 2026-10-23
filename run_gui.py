"""Mobilis 迁跃 启动入口：先显示启动画面，再加载科学计算库 (打包后首次加载需要几秒)。"""
import sys
import tkinter as tk
from pathlib import Path


def _asset(name):
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    for cand in (base / "mobility_tool" / "assets" / name, Path(__file__).resolve().parent / "mobility_tool" / "assets" / name):
        if cand.exists():
            return cand
    return None


def splash():
    root = tk.Tk()
    root.overrideredirect(True)
    w, h = 420, 260
    root.geometry(f"{w}x{h}+{(root.winfo_screenwidth() - w) // 2}+{(root.winfo_screenheight() - h) // 2}")
    c = tk.Canvas(root, width=w, height=h, highlightthickness=0, bg="#0B1F3A")
    c.pack()
    for i in range(h):                     # 竖直渐变
        t = i / h
        col = "#%02x%02x%02x" % (int(11 + t * 47), int(31 + t * 79), int(58 + t * 107))
        c.create_line(0, i, w, i, fill=col)
    icon = _asset("mobilis_256.png")
    if icon:
        img = tk.PhotoImage(file=str(icon)).subsample(2, 2)
        root._img = img
        c.create_image(80, h // 2 - 10, image=img)
    c.create_text(160, 105, text="Mobilis", fill="white", anchor="w", font=("Segoe UI", 28, "bold"))
    c.create_text(162, 145, text="迁跃 · Charge transport & photophysics", fill="#B8C7DE", anchor="w",
                  font=("Microsoft YaHei UI", 11))
    c.create_text(162, 185, text="正在启动 / Loading…", fill="#DCE6F4", anchor="w", font=("Microsoft YaHei UI", 10))
    c.create_text(162, 225, text="开发者：Yarui Shi · 安阳师范学院", fill="#B8C7DE", anchor="w",
                  font=("Microsoft YaHei UI", 9))
    root.update()
    return root


if __name__ == "__main__":
    s = None
    try:
        s = splash()
    except Exception:  # noqa: BLE001  (无显示环境等)
        s = None
    from mobility_tool.gui import main     # 加载 numpy / scipy / matplotlib
    if s is not None:
        s.destroy()
    main()
