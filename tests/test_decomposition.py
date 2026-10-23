"""用解析力场构造的 Hessian 检验重组能分解（运行: python tests/test_decomposition.py）"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobility_tool.decomposition import (build_internal_coords, decompose_reorg, normal_modes,  # noqa: E402
                                         wilson_b)
from mobility_tool.gaussian_io import BOHR2ANG, ELEMENTS, HARTREE2EV, MASSES, read_xyz  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "benzene.xyz"


def write_freq_fchk(path, symbols, coords_ang, hess, energy=-232.0):
    def arr(name, typ, vals):
        vals = list(vals)
        per, fmt = (6, "{:12d}") if typ == "I" else (5, "{:16.8E}")
        s = f"{name:<40s}   {typ}   N={len(vals):12d}\n"
        for i in range(0, len(vals), per):
            s += "".join(fmt.format(v) for v in vals[i:i + per]) + "\n"
        return s
    n3 = 3 * len(symbols)
    txt = "test\nFreq      RB3LYP                                                      6-31G(d)\n"
    txt += f"{'Number of atoms':<40s}   I     {len(symbols):12d}\n"
    txt += f"{'Total Energy':<40s}   R     {energy:22.15E}\n"
    txt += arr("Atomic numbers", "I", [ELEMENTS.index(s) for s in symbols])
    txt += arr("Current cartesian coordinates", "R", (np.asarray(coords_ang) / BOHR2ANG).ravel())
    txt += arr("Real atomic weights", "R", [MASSES[s] for s in symbols])
    txt += arr("Cartesian Force Constants", "R", hess[np.tril_indices(n3)])
    Path(path).write_text(txt)


def model_hessian(symbols, x_ang):
    ic = build_internal_coords(symbols, x_ang)
    b = wilson_b(np.asarray(x_ang) / BOHR2ANG, ic)
    k = np.concatenate([np.full(len(ic["bond"]), 0.45), np.full(len(ic["angle"]), 0.12),
                        np.full(len(ic["dihedral"]), 0.015)])
    return b.T @ (k[:, None] * b), b


def test_decomposition():
    atoms = read_xyz(EXAMPLE)
    sym = [a[0] for a in atoms]
    x_n = np.array([a[1:] for a in atoms])
    h_n, b_n = model_hessian(sym, x_n)
    rng = np.random.default_rng(3)
    dq = rng.normal(size=b_n.shape[0]) * 0.01
    dx = b_n.T @ np.linalg.pinv(b_n @ b_n.T) @ dq            # bohr, 纯内部位移
    x_c = x_n + dx.reshape(-1, 3) * BOHR2ANG
    h_c, _ = model_hessian(sym, x_c)
    expect1 = 0.5 * dx @ h_n @ dx * HARTREE2EV * 1000
    expect2 = 0.5 * dx @ h_c @ dx * HARTREE2EV * 1000
    # 离子几何随机转动 + 平移，检验叠合
    rot = Rotation.from_euler("xyz", [30, -50, 70], degrees=True).as_matrix()
    x_c_moved = x_c @ rot.T + np.array([3.0, -1.0, 2.0])
    h_c_moved = np.kron(np.eye(len(sym)), rot) @ h_c @ np.kron(np.eye(len(sym)), rot).T

    freq, *_ = normal_modes(h_n, [MASSES[s] for s in sym], x_n)
    assert len(freq) == 30 and freq.min() > 0, freq.min()

    with tempfile.TemporaryDirectory() as d:
        write_freq_fchk(f"{d}/n.fchk", sym, x_n, h_n, -232.0)
        write_freq_fchk(f"{d}/c.fchk", sym, x_c_moved, h_c_moved, -231.7)
        res = decompose_reorg(f"{d}/n.fchk", f"{d}/c.fchk")
    for key, exp in (("neutral", expect1), ("ion", expect2)):
        r = res[key]
        print(f"{key:8s} 期望 {exp:8.3f} meV | 简正模式 {r['lambda_normal_modes_meV']:8.3f} | "
              f"内坐标 {r['lambda_internal_meV']:8.3f} | 原子和 {sum(a['lambda_meV'] for a in r['atoms']):8.3f}")
        assert abs(r["lambda_normal_modes_meV"] - exp) < 0.01 * exp
        assert abs(r["lambda_internal_meV"] - exp) < 0.01 * exp
        assert abs(sum(a["lambda_meV"] for a in r["atoms"]) - exp) < 0.01 * exp
        s_sum = sum(m["S"] for m in r["modes"])
        top = sorted(r["modes"], key=lambda m: -m["lambda_meV"])[0]
        print(f"         ΣS = {s_sum:.4f}，贡献最大模式 {top['freq_cm-1']:.0f} cm-1 (S={top['S']:.4f})，"
              f"按类型: " + ", ".join(f"{k} {v:.2f}" for k, v in r["internal_by_type_meV"].items()))
    print("重组能分解 OK")


if __name__ == "__main__":
    test_decomposition()
