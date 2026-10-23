"""批量扫描工作流 (项目、层级、网格 / 验证 / 对称 / 层级对照扫描、续算检查、结果收集、模型、报告、归档)。
用 fake_adf 以 EHT 耦合 + UFF 能量模拟 ADF 输出 (格式与 ADF2016 一致)。"""
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from fake_adf import FakeADF, run_scan  # noqa: E402
from mobility_tool.scananalysis import (PaperModels, analytic, boltzmann, compare_levels, compare_maps,  # noqa: E402
                                        distance_fit, grid_zs, make_report, svd_separability, grid_matrix, validate)
from mobility_tool.scanflow import ScanProject, grid_configs, parse_adf_out, symmetry_check  # noqa: E402

PENT = ROOT / "examples" / "pentacene.xyz"


def _pentacene(path):
    """理想化并五苯 (C–C 1.40 Å，C–H 1.08 Å)。"""
    a, h = 1.40, 1.08
    s3 = np.sqrt(3) / 2
    C, H = [], []
    for k in range(5):
        x0 = (k - 2) * 2 * a * s3
        for p in ((0, a), (0, -a), (a * s3, a / 2), (a * s3, -a / 2), (-a * s3, a / 2), (-a * s3, -a / 2)):
            q = (round(x0 + p[0], 4), round(p[1], 4))
            if q not in C:
                C.append(q)
    for x, y in C:
        n = sum(1 for u, v in C if 0.1 < np.hypot(u - x, v - y) < 1.5)
        if n == 2:
            d = np.array([x, y]) - np.mean([[u, v] for u, v in C if 0.1 < np.hypot(u - x, v - y) < 1.5], 0)
            d = d / np.linalg.norm(d)
            H.append((x + h * d[0], y + h * d[1]))
    with open(path, "w") as fh:
        fh.write(f"{len(C) + len(H)}\npentacene\n")
        for x, y in C:
            fh.write(f"C {x:.5f} {y:.5f} 0.0\n")
        for x, y in H:
            fh.write(f"H {x:.5f} {y:.5f} 0.0\n")


def build(root):
    mono = Path(root) / "pent.xyz"
    _pentacene(mono)
    p = ScanProject.create(Path(root) / "proj", mono, "pentacene")
    assert p.meta["monomer"]["report"]["point_group"] == "D2h", p.meta["monomer"]["report"]["point_group"]
    lv = p.add_level("PW91", "TZP")
    lv2 = p.add_level("PBE", "DZP", d3=True)
    for L in (lv, lv2):
        (p.level_dir(L) / "mono.t21").write_bytes(b"fake")
        (p.level_dir(L) / "mono.out").write_text("NORMAL TERMINATION\n")
    p.add_scan("zscan", grid_configs({"z": "3.2:5.0:0.1", "dx": "0", "dy": "0"}), lv)
    for z in ("3.5", "3.8"):
        p.add_scan(f"slip{z.replace('.', '')}", grid_configs({"z": z, "dx": "0:6:0.5", "dy": "0:3:0.5"}), lv)
    p.add_validation("valid", lv, [{"tag": "interp", "n": 12, "z": (3.5, 3.8), "dx": (-6, 6), "dy": (-3, 3)},
                                   {"tag": "extrap", "n": 6, "z": (3.3, 3.5), "dx": (-6, 6), "dy": (-3, 3)},
                                   {"tag": "extrap", "n": 6, "z": (3.8, 4.2), "dx": (-6, 6), "dy": (-3, 3)}])
    p.add_check("symcheck", lv, [{"z": 3.5, "dx": 1.2, "dy": 0.7}, {"z": 3.6, "dx": 2.0, "dy": 0.4, "roll": 25.0}])
    p.clone("slip35", lv2)
    return p, lv, lv2


def test_workflow():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        p, lv, lv2 = build(d)
        info = p.scan_info("slip35")
        assert info["n_requested"] == 91 and info["n_computed"] == 91, info      # 第一象限：无对称约简
        full = p.add_scan("full", grid_configs({"z": "3.5", "dx": "-3:3:1", "dy": "-2:2:1"}), lv)
        fi = p.scan_info("full")
        assert fi["n_requested"] == 35 and fi["n_computed"] == 12, fi           # 4×3 第一象限代表
        bat = (p.scan_dir("slip35") / "batch_slip35_1.run").read_text()
        assert p.t21(lv).resolve().as_posix() in bat and "\r" not in bat and bat.isascii()
        assert bat.count(f"    mono {p.t21(lv).resolve().as_posix()}") == 91

        # 预检：缺片段文件
        (p.level_dir(lv2) / "mono.t21").unlink()
        pc = p.precheck("slip35_" + lv2)
        assert not pc["ok"] and any("mono.t21" in e for e in pc["errors"])
        (p.level_dir(lv2) / "mono.t21").write_bytes(b"fake")

        fake = FakeADF(p.symbols, p.Y)
        for n in p.scans():
            fails = ["valid_0003"] if n == "valid" else ()
            partial = ["slip38_0005"] if n == "slip38" else ()
            run_scan(p, n, fail_ids=fails, partial_ids=partial)
        pc = p.precheck("valid")
        assert pc["n_failed"] == 1 and "valid_0003" in pc["pending"]
        pc = p.precheck("slip38")
        assert pc["n_pending"] == 1
        run_scan(p, "valid")
        run_scan(p, "slip38")
        assert p.precheck("valid")["n_pending"] == 0

        # 解析：V 与 fake 的参考值一致
        r0 = p.manifest("slip35")[7]
        res = parse_adf_out(p.output_path("slip35", r0["id"]))
        ref, _ = fake.results(r0)
        assert abs(abs(res["carriers"]["hole"]["V"]) - abs(ref["HOMO"]["J_eff_meV"])) < 0.01
        assert "total" in res["eda"]

        recs = p.collect("full")
        assert len(recs) == 35 and all(r["state"] == "done" for r in recs)
        by = {(r["dx"], r["dy"]): r for r in recs}
        for (x, y), r in by.items():
            for c in ("hole", "electron"):
                assert abs(r[c] - by[(abs(x), abs(y))][c]) < 1e-6           # D2h 展开：V 相同含符号

        rows, ch = symmetry_check(p.collect("symcheck"))
        assert max(r["maxdev_hole"] for r in rows) < 0.05
        assert ch["hole"]["C2x"] == 1 and ch["electron"]["C2x"] == -1, ch

        train = p.collect("zscan") + p.collect("slip35") + p.collect("slip38")
        df = distance_fit(train, "hole")
        assert df and df["n"] >= 10 and 0.5 < df["beta"] < 5, df
        assert grid_zs(train) == [3.5, 3.8]
        sv = svd_separability(grid_matrix(train, 3.5, "hole")[2])
        assert 0.3 < sv["share"] <= 1
        cm = compare_maps(train, 3.5, 3.8, "hole")
        assert cm["n"] == 91 and cm["k"] > 0
        m = PaperModels(train)
        assert m.z_ref == 3.5
        for c in ("hole", "electron"):                      # 网格点上 M1 重现训练值
            r = next(x for x in train if x["z"] == 3.5 and x["dx"] == 1.5 and x["dy"] == 1.0 and x["state"] == "done")
            assert abs(m.predict(c, "M1", 3.5, 1.5, 1.0) - r[c]) < 1e-6
            r = next(x for x in train if x["z"] == 3.8 and x["dx"] == 2.0 and x["dy"] == 0.5 and x["state"] == "done")
            assert abs(m.predict(c, "M2", 3.8, -2.0, 0.5) - r[c]) < 1e-6
        # 解析式拟合：在已知参数的合成数据上还原
        P = np.array([80.0, 0.03, 1.1, 0.10, 0.6])
        syn = [dict(x, hole=float(analytic(P, x["dx"], x["dy"]))) for x in train if x["z"] == 3.5]
        from mobility_tool.scananalysis import analytic_fit
        af = analytic_fit(syn, 3.5, "hole")
        assert af["rmse"] < 1e-3 and abs(af["params"]["kx"] - 1.1) < 1e-3, af
        stats, preds = validate(m, p.collect("valid"))
        assert len(preds) == 24 and {s["model"] for s in stats} == {"M1", "M2", "M3"}
        assert sum(x["set"] == "interp" for x in preds) == 12
        lv_cmp = compare_levels(p.collect("slip35"), p.collect("slip35_" + lv2), "hole")
        assert lv_cmp and lv_cmp["n"] == 91 and lv_cmp["r"] > 0.999   # fake 与层级无关
        E = {r["id"]: r["E_total"] for r in p.collect("full") if not r["rep"]}
        bz = boltzmann(p.collect("full"), E, "hole", 300)
        assert bz["n"] == 35 and 1 <= bz["n_eff"] <= 35

        rep = make_report(p, lam={"hole": 0.092, "electron": 0.132})
        html = rep["html"].read_text()
        for s in ("距离衰减", "滑移图", "可分离性", "V = kS", "解析式", "随机构型验证", "泛函", "Boltzmann", "Marcus",
                  "对称"):
            assert s in html, s
        assert (p.root / "model" / "model.json").exists() and (p.root / "validation" / "predictions.csv").exists()
        assert len(rep["figs"]) >= 7
        z, n = p.archive()
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
        assert "all_structures.xyz" in names and "README.txt" in names and any(x.endswith(".out") for x in names)
        assert any(x.startswith("model/") for x in names)
        z2, n2 = p.archive(Path(d) / "small.zip", include_outputs=False)
        assert n2 < n
        p2 = ScanProject(p.root)                          # 重新打开
        assert set(p2.scans()) == set(p.scans())
        shutil.copy(rep["html"], Path(d) / "keep.html")
    print(f"批量扫描工作流 OK ({time.time() - t0:.0f} s)")


def test_import_legacy():
    """导入 make_dimers.py 生成的旧扫描文件夹 (manifest + outputs)。"""
    with tempfile.TemporaryDirectory() as d:
        p, lv, _ = build(d)
        src = p.scan_dir("slip35")
        fake_old = Path(d) / "scan_slip"
        shutil.copytree(src, fake_old)
        for f in fake_old.glob("*.done"):
            f.unlink()
        n = p.import_scan(fake_old, "old35", level=lv)
        run_scan(p, n)
        recs = p.collect(n)
        assert sum(r["state"] == "done" for r in recs) == 91
    print("导入旧扫描 OK")


def test_crystal_params():
    """晶体二聚体 → (z, Δx, Δy, γ, θ, φ)：任意取向放置的已知构型应以 RMSD ≈ 0 还原。"""
    from scipy.spatial.transform import Rotation
    from mobility_tool.scanflow import make_cfg
    with tempfile.TemporaryDirectory() as d:
        p, lv, _ = build(d)
        Q = Rotation.from_euler("zyx", [33, -21, 57], degrees=True).as_matrix()
        t0 = np.array([4.0, -2.0, 7.5])
        A = p.Y @ Q.T + t0
        groups = []
        truth = [make_cfg(3.45, 1.2, 0.6), make_cfg(4.9, 2.1, 3.7, gamma=0, pitch=0, roll=52.0),
                 make_cfg(1.2, 5.1, 3.9, gamma=-48.0, pitch=8.0, roll=60.0)]
        for k, c in enumerate(truth, 1):
            B = p.B_coords(c) @ Q.T + t0
            groups.append({"id": k, "label": f"T{k}", "members": [{"atoms_b": list(zip(p.symbols, B.tolist())),
                                                                   "r_centroid": float(np.linalg.norm(B.mean(0) - A.mean(0)))}]})
        res = {"atoms_a": list(zip(p.symbols, A.tolist())), "groups": groups}
        cfgs = p.crystal_configs(res)
        for c, tr_ in zip(cfgs, truth):
            assert c["rmsd_heavy_A"] < 1e-6, c
            assert abs(abs(c["z"]) - abs(tr_["z"])) < 1e-6 and abs(np.hypot(c["dx"], c["dy"]) - np.hypot(tr_["dx"], tr_["dy"])) < 1e-6
        info = p.add_crystal("cryst", lv, res)
        assert info["n_computed"] == 3
    print("晶体二聚体参数换算 OK (RMSD < 1e-6 Å)")


if __name__ == "__main__":
    test_crystal_params()
    test_workflow()
    test_import_legacy()
