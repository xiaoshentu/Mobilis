"""快速 EHT 与完整算法一致、MO 系数转动、CIF 向量化去重、性能、图标与打包文件。"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from mobility_tool.eht import (build_basis, clear_cache, eht, eht_transfer_integrals,  # noqa: E402
                               eht_transfer_integrals_full, rotate_coefficients)

IRC = [("Ir", 0, 0, 0), ("N", 2.0, 0, 0), ("N", -2.0, 0, 0), ("O", 0, 2.05, 0), ("C", 0, -2.0, .3),
       ("Cl", .2, .1, 2.3), ("H", 2.5, 0.9, 0)]


def test_fast_eht():
    for ang, shift in (([30, 70, -20], [3.9, .8, .5]), ([0, 0, 0], [0, 0, 3.6]), ([120, -45, 10], [2.5, 2.5, 2.5])):
        R = Rotation.from_euler("xyz", ang, degrees=True).as_matrix()
        b = [(s, *(np.array(p) @ R.T + shift)) for s, *p in IRC]
        clear_cache()
        f = eht_transfer_integrals_full(IRC, b, orbitals=("HOMO-1", "HOMO", "LUMO", "LUMO+1"))["results"]
        n = eht_transfer_integrals(IRC, b, orbitals=("HOMO-1", "HOMO", "LUMO", "LUMO+1"))["results"]
        for k in ("HOMO-1", "HOMO", "LUMO", "LUMO+1"):
            assert abs(f[k]["J_eff_meV"] - n[k]["J_eff_meV"]) < 1e-8 and abs(f[k]["S_AB"] - n[k]["S_AB"]) < 1e-12
        assert abs(f["ESD_hole_meV"] - n["ESD_hole_meV"]) < 1e-6 and abs(f["ESD_electron_meV"] - n["ESD_electron_meV"]) < 1e-6
    sym, x = [a[0] for a in IRC], np.array([a[1:] for a in IRC])
    c0, m0 = x.mean(0), eht(sym, x)
    sh, nbf, _ = build_basis(sym, x)
    R = Rotation.from_euler("xyz", [12, -40, 77], degrees=True).as_matrix()
    m2 = eht(sym, (x - c0) @ R.T + c0)
    Crot = rotate_coefficients(sh, m0["C"], R)
    assert min(abs(Crot[:, k] @ m2["S"] @ m2["C"][:, k]) for k in range(nbf)) > 1 - 1e-8
    print("快速 EHT OK: 与完整算法一致 (J、S、ESD)，含 d 轨道的 MO 系数转动精确")


def test_cif_and_speed():
    from test_crystal import make_cif
    from mobility_tool.crystal import find_dimers, find_molecules, read_cif
    from mobility_tool.workflow import eht_couplings_for_crystal
    with tempfile.TemporaryDirectory() as d:
        make_cif(f"{d}/t.cif")
        txt = Path(f"{d}/t.cif").read_text()          # 重复写一遍对称操作: 去重后原子数不应变化
        Path(f"{d}/dup.cif").write_text(txt.replace("'x, y, z'", "'x, y, z'\n'x, y, z'\n'-x, -y, -z'"))
        c1, c2 = read_cif(f"{d}/t.cif"), read_cif(f"{d}/dup.cif")
        assert len(c1["symbols"]) == len(c2["symbols"]) == 48
        m = find_molecules(c1)
        clear_cache()
        t = time.time()
        res, coup = eht_couplings_for_crystal(c1, m, 4.5)
        dt = time.time() - t
        ref = find_dimers(c1, m, cutoff=4.5)
        a_atoms = [(s, *x) for s, x in ref["atoms_a"]]
        g = ref["groups"][0]
        full = eht_transfer_integrals_full(a_atoms, [(s, *x) for s, x in g["members"][0]["atoms_b"]])["results"]
        assert abs(full["HOMO"]["J_eff_meV"] - coup[g["id"]]["hole"]) < 1e-8
    print(f"CIF 向量化去重 OK (重复对称操作不产生重复原子); 晶体 EHT {len(coup)} 组 {dt:.2f} s, 与完整算法一致")


def test_assets_packaging():
    a = ROOT / "mobility_tool" / "assets"
    for f in ("mobilis.ico", "mobilis_256.png", "mobilis_32.png"):
        assert (a / f).exists(), f
    from PIL import Image
    assert (256, 256) in Image.open(a / "mobilis.ico").info["sizes"]
    p = ROOT / "packaging"
    bat = (p / "build_windows.bat").read_bytes()
    assert b"\r\n" in bat and "nuitka".encode() in bat.lower()
    bat.decode("gbk")                                   # 中文 Windows cmd 可正确解析
    iss = (p / "Mobilis.iss").read_bytes()
    assert iss.startswith(b"\xef\xbb\xbf") and b"dist\\Mobilis" in iss
    print("图标 (ICO 7 种尺寸) 与打包文件 OK")


if __name__ == "__main__":
    test_fast_eht()
    test_cif_and_speed()
    test_assets_packaging()
