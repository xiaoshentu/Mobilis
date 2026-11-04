"""声子分辨的动态无序 (phonons)：声学和规则、q 空间求和与超胞力常数矩阵直接求逆一致、温度标度、量子极限、去除声子支。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mobility_tool import phonons as PH  # noqa: E402
from mobility_tool.ceds import CEDS  # noqa: E402
from mobility_tool.crystal import find_molecules, read_cif  # noqa: E402
from mobility_tool.rigidmd import KB, KCAL2MD, RigidCrystalMD  # noqa: E402

CIF = os.path.join(ROOT, "tests", "data", "pentacene_90K.cif")


class LinearSurface:
    """解析的线性耦合曲面 V = V0 + a·(q − q0)，用来检验求和本身。"""
    source = "test"
    frozen = []

    def __init__(self, q0, V0, a):
        self.q0, self.V0, self.a = np.asarray(q0, float), V0, np.asarray(a, float)

    def __call__(self, Q, carrier):
        Q = np.atleast_2d(Q)
        s = 1.0 if carrier == "hole" else 0.5
        return s * (self.V0 + (Q - self.q0) @ self.a)


def _setup(sc=(2, 2, 1)):
    c = read_cif(CIF)
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    md = RigidCrystalMD(c, ce.frames, (ce.sym, ce.Y), sc)
    lat = PH.RigidLattice(md)
    lat.relax()
    lat.force_constants()
    lat.modes()
    rng = np.random.default_rng(4)
    surf = {p["gid"]: LinearSurface(p["q0"], 30.0 / (1 + i), rng.normal(0, [20, 20, 20, 1, 1, 1]))
            for i, p in enumerate(ce.pairs)}
    return ce, md, lat, surf


def test_acoustic_and_direct_inverse():
    ce, md, lat, surf = _setup()
    f = lat.mode["freq_cm"]
    assert np.sort(np.abs(f[0]))[:3].max() < 0.5 and f[0][3:].min() > 5.0, f[0]
    assert (f[1:] > 1.0).all()
    assert lat.asym < 1e-3
    T = 250.0
    mc = PH.mode_couplings(lat, ce, surf, T=T)
    # 直接法：超胞力常数矩阵 H (6N×6N) 的伪逆给出经典位移协方差 ⟨u uᵀ⟩ = k_BT H⁺ (去掉 3 个整体平移)
    N = md.N
    H = np.zeros((N, 6, N, 6))
    for cell in md.cells:
        perm = np.array([md.index(tuple(np.array(cl) + np.array(cell)), k) for cl in md.cells for k in range(md.nk)])
        for k in range(md.nk):
            H[perm, :, md.index(cell, k), :] = lat.FC[k].transpose(1, 2, 0)
    H = H.reshape(6 * N, 6 * N)
    H = 0.5 * (H + H.T)
    w, v = np.linalg.eigh(H)
    keep = np.abs(w) > 1e-6 * np.abs(w).max()
    cov = (v[:, keep] / w[keep]) @ v[:, keep].T * KB * T              # Å² / rad²
    cov = cov.reshape(N, 6, N, 6)
    jac = PH.pair_jacobians(lat, ce)
    grad = PH.surface_gradient(ce, surf)
    for g, (J, kA, kB, t) in jac.items():
        nA, nB = md.index((0, 0, 0), kA), md.index(tuple(t), kB)
        C = np.block([[cov[nA, :, nA, :], cov[nA, :, nB, :]], [cov[nB, :, nA, :], cov[nB, :, nB, :]]])
        for car in ("hole", "electron"):
            a = grad[g][car][1] @ J
            s_direct = float(np.sqrt(a @ C @ a))
            s_modes = mc["pairs"][g][car]["sigma"]
            assert abs(s_modes - s_direct) < 1e-3 * max(s_direct, 0.05), (g, car, s_modes, s_direct)
    _ = KCAL2MD


def test_temperature_quantum_and_drop():
    ce, md, lat, surf = _setup()
    a = PH.mode_couplings(lat, ce, surf, T=100.0)
    b = PH.mode_couplings(lat, ce, surf, T=300.0)
    for g in a["pairs"]:
        sa, sb = a["pairs"][g]["hole"]["sigma"], b["pairs"][g]["hole"]["sigma"]
        assert abs(sb ** 2 / max(sa ** 2, 1e-30) - 3.0) < 1e-6 or sa < 1e-9
    hq = PH.mode_couplings(lat, ce, surf, T=3000.0, quantum=True)
    hc = PH.mode_couplings(lat, ce, surf, T=3000.0)
    lq = PH.mode_couplings(lat, ce, surf, T=5.0, quantum=True)
    lc = PH.mode_couplings(lat, ce, surf, T=5.0)
    for g in hq["pairs"]:
        x, y = hq["pairs"][g]["hole"]["sigma"], hc["pairs"][g]["hole"]["sigma"]
        assert abs(x / y - 1) < 0.01                                       # 高温：量子 → 经典
        assert lq["pairs"][g]["hole"]["sigma"] > 1.5 * lc["pairs"][g]["hole"]["sigma"]   # 低温：零点振动
    nb = len(b["branch"]["freq_gamma"])
    r = PH.as_ceds_results(b, drop_branches=range(nb))
    assert all(v["hole"]["std"] == 0.0 for v in r.values())
    full = PH.as_ceds_results(b)
    for g, v in full.items():
        assert abs(v["hole"]["std"] - b["pairs"][g]["hole"]["sigma"]) < 1e-9
    assert sum(b["branch"]["char_gamma"].sum(1) > 0.99) == nb
    assert 0 < b["hbar_tau_meV"]["hole"] < 30
    txt = PH.summary_text(b)
    assert "声子" in txt


def _write_phonopy_yaml(path, ce, md, lat, rot, bohr=False, seed=0):
    """把刚体晶格的声子写成 phonopy mesh.yaml 的格式 (原子顺序打乱、坐标折回晶胞、笛卡尔系转动、可用 bohr)，
    用来检验外部声子的读入与换算。"""
    import yaml
    from mobility_tool.rigidmd import MASS
    lat_c = np.asarray(ce.crys["lattice"], float)
    syms, xs, us = [], [], []
    for k, mol in enumerate(ce.mols):
        n0 = md.index((0, 0, 0), k)
        cart = np.asarray(mol["cart"], float)
        syms += list(mol["symbols"])
        xs.append(cart @ np.linalg.inv(lat_c))
        r = cart - md.c0[n0]
        eps = lat.mode["eps"][:, 6 * k:6 * k + 6, :]                    # (Nq, 6, nb)
        u = eps[:, None, :3, :] + np.cross(eps[:, None, 3:, :].transpose(0, 1, 3, 2),
                                            r[None, :, None, :]).transpose(0, 1, 3, 2)  # (Nq, natm, 3, nb)
        us.append(u)
    x = np.vstack(xs)
    u = np.concatenate(us, axis=1)
    m = np.array([MASS[e] for e in syms])
    q = lat.mode["q"]
    e = u * np.sqrt(m)[None, :, None, None] * np.exp(-2j * np.pi * x @ q.T).T[:, :, None, None]
    order = np.random.default_rng(seed).permutation(len(syms))
    scale = 1 / 0.529177210903 if bohr else 1.0
    L = lat_c @ rot.T * scale
    doc = {"lattice": L.tolist(),
           "points": [{"symbol": syms[i], "coordinates": (x[i] % 1.0).tolist(), "mass": float(m[i])} for i in order],
           "phonon": []}
    for iq in range(len(q)):
        bands = []
        for b in range(e.shape[3]):
            ev = e[iq, order, :, b] @ rot.T
            bands.append({"frequency": float(lat.mode["freq_cm"][iq, b] / 33.35640952),
                          "eigenvector": [[[float(z.real), float(z.imag)] for z in a] for a in ev]})
        doc["phonon"].append({"q-position": q[iq].tolist(), "weight": 1, "band": bands})
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh)


def test_external_phonopy_roundtrip():
    """刚体声子写成 phonopy 格式再读回 (转动的笛卡尔系、bohr、打乱的原子顺序)：σ_V 应完全复现。"""
    import tempfile
    ce, md, lat, surf = _setup()
    th = 0.7
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]) @ \
        np.array([[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)], [0, np.sin(0.3), np.cos(0.3)]])
    ref = PH.mode_couplings(lat, ce, surf, T=300.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mesh.yaml")
        _write_phonopy_yaml(path, ce, md, lat, rot, bohr=True)
        ext = PH.read_phonopy_yaml(path)
        el = PH.external_lattice(md, ce, ext)
    assert el.info["length_unit"] == "bohr" and el.info["map_dev_A"] < 1e-6
    assert np.abs(el.info["rigid_fraction"][1:] - 1).max() < 1e-6
    # 雅可比矩阵在实验结构处 (外部声子不做力场弛豫)；参考值也取实验结构处的雅可比
    lat0 = PH.RigidLattice(md)
    lat0.mode = lat.mode
    jac0 = PH.pair_jacobians(lat0, ce)
    ref0 = PH.mode_couplings(lat0, ce, surf, T=300.0, jac=jac0)
    got = PH.mode_couplings(el, ce, surf, T=300.0, jac=jac0)
    for g in ref0["pairs"]:
        for car in ("hole", "electron"):
            a, b = ref0["pairs"][g][car]["sigma"], got["pairs"][g][car]["sigma"]
            assert abs(a - b) < 1e-6 * max(a, 1e-3), (g, car, a, b)
    assert np.allclose(ref["freq_cm"][1:], got["freq_cm"][1:], atol=1e-4)


if __name__ == "__main__":
    test_acoustic_and_direct_inverse()
    test_temperature_quantum_and_drop()
    test_external_phonopy_roundtrip()
    print("ok")
