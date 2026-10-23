"""检验 Gaussian FCHT 输出解析（黄昆因子、Duschinsky 矩阵、位移矢量）与导出、作图。"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobility_tool.gaussian_fcht import CM2EV, export_fcht, fcht_input, generate_fcht_inputs, read_fcht_log  # noqa
from mobility_tool.plots import figure_hr_duschinsky, figure_reorg_decomposition  # noqa: E402


def dfmt(v):
    """高斯风格 D 指数格式，例如 0.123456D-01 / -.123456D+00"""
    m, e = f"{v:.6E}".split("E")
    m = float(m) / 10
    return f"{m:.6f}D{int(e) + 1:+03d}".replace("0.", ".") if v < 0 else f"{m:.6f}D{int(e) + 1:+03d}"


def gaussian_matrix(name, mat):
    out = [f" {name}"]
    n0, n1 = mat.shape
    for c0 in range(0, n1, 5):
        cols = range(c0, min(c0 + 5, n1))
        out.append("      " + "".join(f"{c + 1:14d}" for c in cols))
        for r in range(n0):
            out.append(f"{r + 1:6d}  " + "".join(f"{dfmt(mat[r, c]):>14s}" if mat[r, c] >= 0
                                                  else dfmt(mat[r, c]) for c in cols))
    return "\n".join(out)


def make_fcht_log(freq_final, freq_init, hr, jmat, kvec):
    n = len(freq_final)
    lines = [" -------------------------------", " #p geom=allcheck freq(readfc,fcht,readfcht)",
             " -------------------------------", " Deg. of freedom     %d" % n,
             " Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering",
             " activities (A**4/AMU), depolarization ratios for plane and unpolarized",
             " incident light, reduced masses (AMU), force constants (mDyne/A),",
             " and normal coordinates:"]
    for i in range(0, n, 3):
        lines.append("                     " + "  ".join(str(k + 1) for k in range(i, min(i + 3, n))))
        lines.append(" Frequencies --" + "".join(f"{f:12.4f}" for f in freq_final[i:i + 3]))
    lines += [" ==== Initial state ====", " and normal coordinates:"]
    for i in range(0, n, 3):
        lines.append(" Frequencies --" + "".join(f"{f:12.4f}" for f in freq_init[i:i + 3]))
    lines += ["", gaussian_matrix("Duschinsky Transformation", jmat), "", " Shift Vector"]
    lines += [f"{i + 1:6d}  {dfmt(v)}" for i, v in enumerate(kvec)]
    lines += ["", " Huang-Rhys Factors", " ------------------"]
    lines += [f" Mode num. {i + 1:3d} - Factor: {dfmt(s)}" for i, s in enumerate(hr)]
    lines += [" Normal termination of Gaussian 16"]
    return "\n".join(lines) + "\n"


def test_fcht():
    rng = np.random.default_rng(5)
    n = 7
    ff = np.sort(rng.uniform(200, 3200, n))
    fi = ff * 0.97
    hr = rng.uniform(0, 0.5, n) * np.array([1, 0, 1, 1e-5, 1, 1, 0.2])
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    kvec = rng.normal(size=n)
    with tempfile.TemporaryDirectory() as d:
        Path(d, "f.log").write_text(make_fcht_log(ff, fi, hr, q, kvec))
        res = read_fcht_log(Path(d, "f.log"))
        got_s = np.array([m["S"] for m in res["modes"]])
        got_f = np.array([m["freq_cm-1"] for m in res["modes"]])
        assert np.allclose(got_s, hr, rtol=1e-5, atol=1e-12) and np.allclose(got_f, ff, atol=1e-3)
        assert np.allclose(res["duschinsky"], q, atol=1e-6), np.abs(res["duschinsky"] - q).max()
        assert np.allclose(res["shift"], kvec, rtol=1e-5)
        assert abs(res["lambda_meV"] / ((hr * ff).sum() * CM2EV * 1000) - 1) < 1e-5
        files = export_fcht(res, d, "test")
        assert len(files) == 5
        back = np.loadtxt(Path(d, "test_矩阵.csv"), delimiter=",")
        assert np.allclose(back, q, atol=1e-6)
        figure_hr_duschinsky(got_f, got_s, res["duschinsky"], Path(d, "a.png"), dpi=80)
        figure_reorg_decomposition(got_f, [m["lambda_meV"] for m in res["modes"]], Path(d, "b.png"), fwhm=30,
                                   dpi=80)
        txt = fcht_input("x_cation.chk", "x_neutral.chk", clear_low_freq=100)
        assert "RedDim=ClearLowFreq=100.0" in txt and txt.splitlines()[0] == "%oldchk=x_neutral.chk"
        jobs = generate_fcht_inputs(d, "x")
        assert len(jobs) == 4 and Path(d, "x_fcht_cation_to_neutral.gjf").exists()
    print(f"FCHT 解析 OK: {n} 个模式, λ = {res['lambda_meV']:.2f} meV, Duschinsky {q.shape}, 导出 {len(files)} 个文件")


if __name__ == "__main__":
    test_fcht()
