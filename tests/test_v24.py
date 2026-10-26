"""三维各向异性迁移率：与二维角分辨公式、迁移率张量的一致性，转动协变性，晶体学方向，图与 HTML 导出，命令行。"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool.mobility import (angular_mobility_2d, compute_mobility, crystal_direction,  # noqa: E402
                                    mobility_3d, mobility_on_sphere, plane_angles)

PATHS = [{"name": "a", "vector": [6.0, 0.0, 0.5], "V": 0.030, "count": 2},
         {"name": "b", "vector": [0.0, 7.5, 0.3], "V": 0.012, "count": 2},
         {"name": "d", "vector": [3.1, 3.9, 0.2], "V": 0.050, "count": 4},
         {"name": "c", "vector": [0.5, 0.4, 14.0], "V": 0.004, "count": 2}]


def test_slices_equal_2d_formula():
    r = mobility_3d(0.1, PATHS, 300.0, "V")
    for plane in ("xy", "xz", "yz"):
        p2 = []
        for p in PATHS:
            rr, th, ga = plane_angles(p["vector"], plane)
            p2.append(dict(p, r=rr, theta=th, gamma=ga))
        a = angular_mobility_2d(0.1, p2, 300.0, "V")
        ax = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
        u, v = (np.array(ax[c], float) for c in plane)
        d = np.outer(np.cos(a["phi"]), u) + np.outer(np.sin(a["phi"]), v)
        assert np.abs(np.einsum("ia,ab,ib->i", d, r["M"], d) - a["mu"]).max() < 1e-12 * a["mu_max"] + 1e-15


def test_tensor_model_and_sphere():
    t = mobility_3d(0.1, PATHS, 300.0, "V", model="tensor")
    c = compute_mobility(0.1, [dict(p, J_eV=p["V"]) for p in PATHS], 300.0)
    assert np.allclose(t["M"], c["mobility_tensor_cm2_Vs"], rtol=1e-12, atol=0)
    n, mu = mobility_on_sphere(t["M"], 41, 81)
    assert abs(mu.max() - t["mu_max"]) < 0.01 * t["mu_max"] and mu.min() >= t["mu_min"] - 1e-9
    assert abs(np.linalg.norm(t["dir_max"]) - 1) < 1e-12 and t["anisotropy"] > 1


def test_rotation_covariance():
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("zyx", [0.4, -1.0, 2.2]).as_matrix()
    r0 = mobility_3d(0.1, PATHS, 300.0, "V")
    r1 = mobility_3d(0.1, [dict(p, vector=(R @ np.array(p["vector"])).tolist()) for p in PATHS], 300.0, "V")
    assert np.allclose(r1["M"], R @ r0["M"] @ R.T, rtol=1e-10, atol=1e-14)
    assert np.allclose(r1["principal"], r0["principal"])


def test_crystal_direction():
    lat = np.array([[5.0, 0, 0], [0, 7.0, 0], [0, 0, 9.0]])
    f, ang = crystal_direction([0, -1.0, 0], lat)
    assert np.allclose(f, [0, 1, 0]) and abs(ang[1]) < 1e-9 and abs(ang[0] - 90) < 1e-9
    f, _ = crystal_direction(np.array([5.0, 7.0, 0]), lat)
    assert np.allclose(f, [1, 1, 0])


def test_figure_html_and_cli():
    from mobility_tool.plots import E_COLOR, H_COLOR, figure_mobility_3d, mobility_3d_html, view_azimuth
    d = tempfile.mkdtemp()
    sy = [{"label": "电子", "res": mobility_3d(0.13, PATHS, 300.0, "V"), "color": E_COLOR},
          {"label": "空穴", "res": mobility_3d(0.09, PATHS, 300.0, "V"), "color": H_COLOR}]
    lat = np.array([[6.3, 0, 0], [0.7, 7.8, 0], [0.6, 3.3, 14.1]])
    fig = figure_mobility_3d(sy, f"{d}/m.png", lat, "xy", dpi=80)
    assert os.path.getsize(f"{d}/m.png") > 5000 and len(fig.axes) >= 2
    a = view_azimuth(sy[1]["res"])
    dm = sy[1]["res"]["dir_max"]
    cam = np.array([np.cos(np.radians(a + 20)), np.sin(np.radians(a + 20))])
    assert abs(cam @ dm[:2]) < 1e-9                 # 视线 (去掉斜角) 垂直于快轴
    mobility_3d_html(f"{d}/m.html", sy, lat, "xy")
    h = Path(f"{d}/m.html").read_text(encoding="utf-8")
    assert "plotly" in h and '"surface"' in h and "scene2" in h
    csv = Path(d) / "paths.csv"
    csv.write_text("name,x,y,z,Ve,Vh,count\n" + "\n".join(
        f"{p['name']},{p['vector'][0]},{p['vector'][1]},{p['vector'][2]},{p['V'] * 1000},{p['V'] * 800},{p['count']}"
        for p in PATHS))
    p = subprocess.run([sys.executable, "-m", "mobility_tool", "mobility3d", str(csv), "--lam-e", "0.13", "--lam-h", "0.09",
                        "--scale", "1000", "--cif", os.path.join(ROOT, "tests/data/pentacene_RT.cif"),
                        "--plot", f"{d}/cli.png"], cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    assert "μ1" in p.stdout and os.path.exists(f"{d}/cli.png") and os.path.exists(f"{d}/cli.html")


if __name__ == "__main__":
    for f in (test_slices_equal_2d_formula, test_tensor_model_and_sphere, test_rotation_covariance,
              test_crystal_direction, test_figure_html_and_cli):
        f()
        print("ok", f.__name__)
