"""检验 CIF 读取、分子拼接、二聚体搜索与等价合并（运行: python tests/test_crystal.py）"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobility_tool.crystal import (cell_matrix, export_dimers, find_dimers, find_molecules,  # noqa: E402
                                   mobility_pathways_from_dimers, read_cif)
from mobility_tool.gaussian_io import read_xyz  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def make_cif(path):
    atoms = read_xyz(EXAMPLES / "benzene.xyz")
    xyz = np.array([a[1:] for a in atoms])
    xyz = xyz @ Rotation.from_euler("xyz", [35, 20, 60], degrees=True).as_matrix().T
    cell = (9.0, 10.5, 8.2, 90.0, 105.0, 90.0)
    lat = cell_matrix(*cell)
    frac = (xyz @ np.linalg.inv(lat)) + np.array([0.98, 0.25, 0.10])   # 质心靠近晶胞角，原子会被拆开
    lines = ["data_test", "_symmetry_space_group_name_H-M 'P 21/c'",
             *[f"_cell_{k} {v}" for k, v in zip(["length_a", "length_b", "length_c", "angle_alpha",
                                                  "angle_beta", "angle_gamma"], cell)],
             "loop_", "_symmetry_equiv_pos_as_xyz", "'x, y, z'", "'-x, y+1/2, -z+1/2'", "'-x, -y, -z'",
             "'x, -y+1/2, z+1/2'",
             "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
             "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_occupancy"]
    for i, (a, f) in enumerate(zip(atoms, frac % 1.0)):
        lines.append(f"{a[0]}{i + 1} {a[0]} {f[0]:.5f}(3) {f[1]:.5f}(2) {f[2]:.5f} 1")
    Path(path).write_text("\n".join(lines) + "\n")


def test_crystal():
    with tempfile.TemporaryDirectory() as d:
        make_cif(f"{d}/t.cif")
        crys = read_cif(f"{d}/t.cif")
        assert len(crys["symbols"]) == 48, len(crys["symbols"])
        mols = find_molecules(crys)
        assert len(mols) == 4 and all(len(m["symbols"]) == 12 for m in mols)
        # 拼出的分子应保持苯环几何 (C-C 1.397 Å)
        c = mols[0]["cart"][[i for i, s in enumerate(mols[0]["symbols"]) if s == "C"]]
        dcc = np.sort(np.linalg.norm(c[:, None] - c[None], axis=-1).ravel())[6:18]
        assert np.allclose(dcc, 1.3968, atol=2e-3), dcc
        res = find_dimers(crys, mols, cutoff=4.5)
        print(f"晶胞原子 {len(crys['symbols'])}，分子 {len(mols)} 个，近邻 {len(res['dimers'])} 个，"
              f"等价组 {len(res['groups'])} 个")
        for g in res["groups"]:
            m = g["members"][0]
            print(f"  组 G{g['id']}: 数目 {len(g['members'])}  质心距 {m['r_centroid']:.3f} Å  "
                  f"最短原子距 {m['d_min']:.3f} Å")
        # 平移 ±a 的近邻必须在同一组
        ta = [dd for dd in res["dimers"] if dd["mol"] == res["ref"] and abs(dd["translation"][0]) == 1
              and dd["translation"][1:] == [0, 0]]
        if len(ta) == 2:
            assert ta[0]["group"] == ta[1]["group"]
        files = export_dimers(res, d)
        paths = mobility_pathways_from_dimers(f"{d}/dimers.json", {g["id"]: 10.0 for g in res["groups"]})
        assert len(paths) == len(res["dimers"]) and len(files) == len(res["groups"])
    print("晶体 / 二聚体 OK")


if __name__ == "__main__":
    test_crystal()
