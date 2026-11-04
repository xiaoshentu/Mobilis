"""v1.21.2 回归测试：
1. ZINDO/MOO 的 J 与 B 分子的原子顺序无关 (晶体中滑移面 / 螺旋轴产生的分子原子顺序可能不同)；
2. dimers.json 记录晶体点群，迁移率张量按晶体对称性平均：单斜晶体 μ_ab = μ_bc = 0，Z = 2 时结果不变。"""
import json, os, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobility_tool.crystal import read_cif, find_molecules, find_dimers, export_dimers, mobility_pathways_from_dimers  # noqa: E402
from mobility_tool.mobility import compute_mobility  # noqa: E402
from mobility_tool.zindo import MOO  # noqa: E402

CIF = os.path.join(ROOT, "tests", "data", "pentacene_RT.cif")


def test_moo_atom_order():
    c = read_cif(CIF); m = find_molecules(c); res = find_dimers(c, m, 0, 4.0)
    sa = [s for s, x in res["atoms_a"]]; Xa = np.array([x for s, x in res["atoms_a"]])
    moo = MOO(sa, Xa)
    g = res["groups"][0]
    sb = [s for s, x in g["members"][0]["atoms_b"]]; Xb = np.array([x for s, x in g["members"][0]["atoms_b"]])
    j0 = moo.coupling_atoms(Xa, Xb, sym_b=sb)
    perm = np.random.default_rng(0).permutation(len(sb))
    j1 = moo.coupling_atoms(Xa, Xb[perm], sym_b=[sb[k] for k in perm])
    for car in ("hole", "electron"):
        assert abs(abs(j0[car]) - abs(j1[car])) < 1e-6, (car, j0, j1)
    print("MOO 与原子顺序无关:", {k: round(1000 * v, 2) for k, v in j0.items()})


def test_point_group_symmetrization():
    c = read_cif(CIF); m = find_molecules(c); res = find_dimers(c, m, 0, 4.0)
    d = tempfile.mkdtemp(); export_dimers(res, d)
    info = json.load(open(os.path.join(d, "dimers.json")))
    assert info["point_ops"] and len(info["point_ops"]) >= 2
    J = {g["id"]: 50.0 / (1 + g["id"]) for g in res["groups"]}
    paths = mobility_pathways_from_dimers(info, J)
    r0 = compute_mobility(0.1, paths, 300, 3)
    r1 = compute_mobility(0.1, paths, 300, 3, point_ops=info["point_ops"])
    assert np.allclose(r0["mobility_tensor_cm2_Vs"], r1["mobility_tensor_cm2_Vs"], atol=1e-6 * np.abs(r0["mobility_tensor_cm2_Vs"]).max())
    # 人为打破对称：只保留一个近邻 → 平均后张量满足晶体对称
    r2 = compute_mobility(0.1, paths[:1], 300, 3, point_ops=info["point_ops"])
    M = np.array(r2["mobility_tensor_cm2_Vs"])
    for R in info["point_ops"]:
        R = np.array(R)
        assert np.allclose(R @ M @ R.T, M, atol=1e-12 + 1e-9 * np.abs(M).max())
    print("点群对称化：", len(info["point_ops"]), "个操作")




def test_corrected_angular_formula():
    """修正式 = 无规行走张量的投影；Wen 原式 = 每条路径多乘跳跃概率 P_i。"""
    from mobility_tool.mobility import angular_mobility_2d, mobility_3d, plane_angles
    paths = [{"name": "a", "vector": [6.0, 0.0, 0.5], "V": 0.030, "count": 2},
             {"name": "d", "vector": [3.1, 3.9, 0.2], "V": 0.050, "count": 4},
             {"name": "c", "vector": [0.5, 0.4, 14.0], "V": 0.004, "count": 2}]
    p2 = [dict(p, **dict(zip(("r", "theta", "gamma"), plane_angles(p["vector"], "xy")))) for p in paths]
    t = angular_mobility_2d(0.1, p2, 300.0, "V")
    w = angular_mobility_2d(0.1, p2, 300.0, "V", model="wen")
    M = mobility_3d(0.1, paths, 300.0, "V", "tensor")["M"]
    d = np.c_[np.cos(t["phi"]), np.sin(t["phi"]), 0 * t["phi"]]
    assert np.abs(np.einsum("ia,ab,ib->i", d, M, d) - t["mu"]).max() < 1e-12 * t["mu_max"]
    assert t["model"] == "tensor" and w["model"] == "wen" and w["mu_max"] < t["mu_max"]
    print(f"修正式 / Wen 原式 (μ_max): {t['mu_max'] / w['mu_max']:.2f}")


def test_polar_symmetry_equivalent_sites():
    """polar：对称等价分子只算一次 (蒽 P2₁/a 的两个分子等价，并五苯 P-1 的两个分子不等价)。"""
    import os
    from mobility_tool import polar as P
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    for name, want in (("pentacene_90K.cif", [0, 1]), ("Rubrene.cif", [0, 0, 0, 0])):
        c = read_cif(os.path.join(data, name))
        ce = CEDS(c, find_molecules(c), cutoff=4.0)
        ch = P.charge_set(ce.sym, ce.Y)
        eq, _ = P.symmetry_equivalent_sites(c["lattice"], ce.frames, ce.Y, ce.sym, (ch["q0"], ch["dq_h"], ch["dq_e"]))
        assert eq == want, (name, eq)
    # site_energies 对等价分子沿用结果
    c = read_cif(os.path.join(data, "Rubrene.cif"))
    ce = CEDS(c, find_molecules(c), cutoff=4.0)
    msgs = []
    res = P.site_energies(c["lattice"], ce.frames, ce.sym, ce.Y, P.charge_set(ce.sym, ce.Y), [10.0, 12.0],
                          on_progress=msgs.append)
    assert sum("对称等价" in m for m in msgs) == 3 and max(abs(x) for x in res["offsets_h"]) < 1e-12


if __name__ == "__main__":
    test_moo_atom_order(); test_point_group_symmetrization(); test_corrected_angular_formula(); test_polar_symmetry_equivalent_sites(); print("ok")
