"""TLT、有效质量、振动分辨光谱、能隙定律、π 堆积、渲染器拾取、ADF 运行、xtb 解析。"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mobility_tool.kmc import graph_from_paths  # noqa: E402
from mobility_tool.mobility import HBAR_EVS, KB_EV  # noqa: E402
from mobility_tool.transport import HB2_ME, effective_mass, tlt_mobility  # noqa: E402
from mobility_tool.vibronic import emission_spectrum, energy_gap_law  # noqa: E402


def test_tlt_band():
    J, a, T, ht = 0.05, 5.0, 300.0, 0.005
    g = graph_from_paths([dict(vector=[a, 0, 0], V=J, count=2)], "V")
    r = tlt_mobility(g, T, "electron", sigma=0.0, hbar_tau=ht, sizes=(400, 1, 1), keep_axes=(0,), n_real=1)
    k = 2 * np.pi * np.arange(400) / 400 / a
    E, hv = 2 * J * np.cos(k * a), -2 * J * a * np.sin(k * a)
    w = np.exp(-(E - E.min()) / (KB_EV * T))
    w /= w.sum()
    drude = (w * hv ** 2).sum() / HBAR_EVS ** 2 * 1e-16 * (HBAR_EVS / ht) / (KB_EV * T)
    assert abs(r["mu_tensor_cm2_Vs"][0, 0] / drude - 1) < 1e-6
    r2 = tlt_mobility(g, T, "electron", sigma=0.5, hbar_tau=ht, sizes=(400, 1, 1), keep_axes=(0,), n_real=8)
    assert r2["mu_tensor_cm2_Vs"][0, 0] < 0.2 * drude
    em = effective_mass(g, "electron")
    assert abs(np.min(em["masses_me"]) - HB2_ME / (2 * J * a * a)) < 1e-3 and abs(em["bandwidth_eV"] - 4 * J) < 1e-6
    print(f"TLT OK: σ=0 时等于 Drude ({drude:.3f}), σ/J=0.5 降至 {r2['mu_tensor_cm2_Vs'][0, 0]:.3f}; "
          f"有效质量 {np.min(em['masses_me']):.3f} m_e (解析一致)")


def test_vibronic():
    r = emission_spectrum([1400], [1.0], e00=2.0, T=0, fwhm_ev=0.01, cube=False, npts=8192)
    peaks = [r["I"][np.argmin(abs(r["E_eV"] - (2.0 - n * 1400 / 8065.544)))] for n in range(4)]
    assert np.allclose(np.array(peaks) / peaks[0], [1, 1, 0.5, 1 / 6], atol=5e-3)
    k = [energy_gap_law([1300, 1550], [0.25, 0.35], gap)["knr_over_C2"] for gap in (1.7, 1.9, 2.1)]
    assert k[0] > k[1] > k[2]
    print("振动分辨光谱 OK (单模泊松级数), 能隙定律 OK (k_nr 随 ΔE 单调下降)")


def test_stacking_render():
    from mobility_tool.crystal import stacking_geometry
    from mobility_tool.gaussian_io import read_xyz
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from mobility_tool.render import MolRenderer
    a = read_xyz(ROOT / "examples/benzene_dimer.xyz")
    assert stacking_geometry(a[:12], a[12:])["type"] == "面对面 π 堆积"
    b = [(s, x + 3.5, y, z) for s, x, y, z in a[12:]]
    assert stacking_geometry(a[:12], b)["type"] == "滑移 π 堆积"
    fig = Figure(figsize=(4, 4), dpi=80)
    rr = MolRenderer(fig.add_axes([0, 0, 1, 1]))
    rr.set_scene([x[0] for x in a], [x[1:] for x in a], reset=True)
    rr.view_along([0, 0, 1])
    rr.draw()
    scr, depth = rr._screen[0], rr._screen[1]
    k = int(np.argmax(depth))
    assert rr.pick(*scr[k]) == k                                     # 最前面的原子被拾取
    for st in ("stick", "spacefill", "wireframe"):
        rr.style = st
        rr.draw()
    print("π 堆积分类 OK (面对面 / 滑移), 渲染器拾取 OK")


def test_programs():
    from mobility_tool.programs import parse_dipro
    txt = """ Fragment 1 HOMO   Fragment 2 HOMO
 --------------------------------------
 E_mon(orb) frag1 frag2             -12.966   -12.099 eV
 |J(AB)|:                0.238 eV
 S(AB):            -0.00998928
 |J(AB,eff)|:            0.113 eV
 :  total |J(AB,eff)| for hole transport (occ. MOs) :               0.113 eV  :
 :  total |J(AB,eff)| for charge transport (unocc. MOs) :           0.062 eV  :
 normal termination of dipro"""
    r = parse_dipro(txt)
    assert r["pairs"][("HOMO", "HOMO")] == 0.113 and r["total"]["electron"] == 0.062 and r["normal"]
    import shutil
    import threading
    if __import__("mobility_tool.programs").programs.find_shell():
        from mobility_tool.adf import generate_adf_inputs
        from mobility_tool.jobs import JobQueue
        from mobility_tool.programs import find_ams
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / "AMS2099"
            (home / "bin").mkdir(parents=True)
            (home / "atomicdata").mkdir()
            ams = home / "bin" / "ams"
            ams.write_text('#!/bin/sh\ncat > /dev/null\n[ -z "$AMSRESOURCES" ] && exit 1\nd="${AMS_JOBNAME}.results"; '
                           'mkdir -p "$d"; echo rkf > "$d/adf.rkf"\n')
            ams.chmod(0o755)
            info = find_ams(home)
            r = generate_adf_inputs(Path(d) / "job", dimer=ROOT / "examples/benzene_dimer.xyz", n_a=12)
            ev, done = [], threading.Event()
            JobQueue(None, [r["script"]], Path(d) / "scr", lambda g, s, m: ev.append(s), lambda x: done.set(),
                     ams_info=info).start()
            done.wait(30)
            assert ev[-1].startswith("完成"), ev
    print("xtb DIPRO 输出解析 OK, ADF 任务运行 OK (AMS 环境变量)")


if __name__ == "__main__":
    test_tlt_band()
    test_vibronic()
    test_stacking_render()
    test_programs()
