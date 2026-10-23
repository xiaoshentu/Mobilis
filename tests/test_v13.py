"""按方向选择二聚体 (红荧烯 P/T1/T2)、Gaussian 基组写法与出错诊断、开发者信息。"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mobility_tool  # noqa: E402


def test_paper_dimers():
    from mobility_tool.crystal import find_molecules, paper_dimers, read_cif, remove_groups
    c = read_cif(ROOT / "tests/data/Rubrene.cif")
    m = find_molecules(c)
    inv = np.linalg.inv(c["lattice"])
    r = paper_dimers(c, m, plane="auto", criterion="atom", cutoff=4.0)
    assert r["plane"] == "bc" and [g["label"] for g in r["groups"]] == ["P", "T1", "T2"]
    dirs = [np.round(np.abs(np.array(g["members"][0]["vector"]) @ inv), 2).tolist() for g in r["groups"]]
    assert dirs == [[0, 1, 0], [0, 0.5, 0.5], [0, 0.5, 0.5]] and all(len(g["members"]) == 2 for g in r["groups"])
    r2 = paper_dimers(c, m, plane="auto", criterion="centroid", cutoff=15.0)
    assert [g["label"] for g in r2["groups"]] == ["P", "T1", "T2", "L1", "L2", "L3"]
    r3 = remove_groups(r2, [4, 5])
    assert [g["label"] for g in r3["groups"]] == ["P", "T1", "T2", "L3"] and [g["id"] for g in r3["groups"]] == [1, 2, 3, 4]
    r4 = paper_dimers(c, m, plane="3d", criterion="atom", cutoff=4.0)
    assert [g["kind"] for g in r4["groups"]].count("I") == 4
    print("按方向选择 OK: 红荧烯 bc 分子层 P (b)、T1/T2 (b±c)/2；L (c)；层间 I；移除指定组")


def test_gaussian_text():
    from mobility_tool.gaussian_io import explain_gaussian_error, gaussian_basis_fix
    assert gaussian_basis_fix("# td B3LYP/def2-TZVP\nC 0\ndef2-svp\n") == "# td B3LYP/Def2TZVP\nC 0\nDef2SVP\n"
    assert gaussian_basis_fix("B3LYP/6-311G(d,p) cc-pVTZ") == "B3LYP/6-311G(d,p) cc-pVTZ"
    msg = explain_gaussian_error((ROOT / "tests/data/G2_A_S0_td_error.out").read_text())
    assert "语法错误" in msg and "Def2" in msg and "def2-TZVP" in msg
    from mobility_tool.photophys import generate_excited_inputs
    with tempfile.TemporaryDirectory() as d:
        generate_excited_inputs(ROOT / "examples/benzene.xyz", d, name="bz")
        for st in ("S0", "S1", "T1"):
            t = Path(d, f"bz_{st}.gjf").read_text()
            assert "Def2TZVP" in t and "def2-" not in t
    print("Gaussian OK: def2-TZVP → Def2TZVP；用户的 TD 报错被正确诊断")


def test_author():
    assert mobility_tool.__author__ == "Yarui Shi" and "安阳师范学院" in mobility_tool.__affiliation__
    print("开发者信息 OK:", mobility_tool.__author__, mobility_tool.__affiliation__)


if __name__ == "__main__":
    test_paper_dimers()
    test_gaussian_text()
    test_author()
