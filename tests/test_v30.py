"""v1.19 回归测试：流程图 (多步串联计算)。
1. 软件内流程：搜索二聚体 → EHT → Marcus 迁移率 → KMC，结果写入 flow_results.json，Marcus 精确解与 KMC 一致；
2. Gaussian 流程 (假 g16)：优化 → 重组能 → FCHT，后一阶段使用前一阶段的优化结构，λ 自动读取；
3. 依赖检查、保存 / 打开流程文件、失败阶段之后的阶段被跳过；
4. 输入生成模块不依赖 tkinter (命令行 / 服务器可用)。
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from mobility_tool import flow as F  # noqa: E402


def _crystal(d):
    from test_crystal import make_cif
    from mobility_tool.crystal import find_molecules, read_cif
    make_cif(Path(d) / "t.cif")
    c = read_cif(Path(d) / "t.cif")
    return c, find_molecules(c)


def test_local_flow():
    with tempfile.TemporaryDirectory() as d:
        crys, mols = _crystal(d)
        st = [F.new_stage("dimers", {"cutoff": 4.5}), F.new_stage("eht", {"cutoff": 4.5}), F.new_stage("mobility"),
              F.new_stage("kmc", {"walkers": 800, "steps": 3000})]
        assert F.check_flow(st) == []
        seen = []
        r = F.FlowRunner(st, Path(d) / "w", {"crystal": (crys, mols)}, on_status=lambda n, s, m: seen.append((n, s)))
        assert r.run()
        assert [s for n, s in seen if s != "running"] == ["done"] * 4
        out = json.loads((Path(d) / "w/flow_results.json").read_text(encoding="utf-8"))
        assert out["ok"] and len(out["couplings_meV"]) >= 3
        ex = out["results"]["mobility"]["exact"]["hole"]["mu_avg"]
        km = out["results"]["kmc"]["hole"]["mu_avg"]
        assert abs(km / ex - 1) < 0.25, (km, ex)
        print(f"软件内流程 4 个阶段完成：精确 μ_h = {ex:.3g}，KMC {km:.3g} cm²/V·s")


def test_gaussian_flow():
    from fake_programs import fake_gaussian
    from mobility_tool.gaussian_io import read_xyz
    from mobility_tool.jobs import find_gaussian
    with tempfile.TemporaryDirectory() as d:
        info = find_gaussian(str(fake_gaussian(Path(d) / "g")))
        at = read_xyz(ROOT / "examples/benzene_dimer.xyz")[:12]
        base = dict(source="主窗口当前结构", method="B3LYP", basis="6-31G(d)", charge=0, mult=1, ion_mult=2,
                    carriers="空穴和电子", nproc=4, mem="4GB", program="g16", name="bz", addin="! test addin")
        st = [F.new_stage("gaussian", dict(base, calc="opt")), F.new_stage("gaussian", dict(base, calc="reorg")),
              F.new_stage("gaussian", dict(base, calc="fcht"))]
        r = F.FlowRunner(st, Path(d) / "w", {"molecule": (at, "bz")}, gaussian=info)
        assert r.run()
        lam = r.ctx["lambda"]
        assert lam["hole"] > 0 and lam["electron"] > 0, lam
        g = (Path(d) / "w/stage2_g_reorg/bz_neutral.gjf").read_text()
        assert "! test addin" in g
        assert list((Path(d) / "w/stage2_g_reorg").glob("bz_fcht_*.gjf"))
        # 失败：没有结构 → 第 1 阶段失败，其余跳过
        seen = []
        r = F.FlowRunner(st, Path(d) / "w2", {}, gaussian=info, on_status=lambda n, s, m: seen.append(s))
        assert not r.run() and seen.count("failed") == 1 and seen.count("skipped") == 2
        print(f"Gaussian 流程 (假 g16)：优化 → 重组能 (λ_h = {lam['hole'] * 1000:.1f} meV) → FCHT；失败后跳过后续阶段")


def test_check_and_files():
    st = [F.new_stage("gaussian", {"calc": "ti", "source": "晶体全部近邻二聚体"}), F.new_stage("mobility")]
    probs = F.check_flow(st)
    assert any("近邻二聚体" in p for p in probs)
    assert not any("J" in p and "mobility" in p for p in probs)
    assert F.check_flow([F.new_stage("mobility")])
    with tempfile.TemporaryDirectory() as d:
        F.save_flow(Path(d) / "a.mflow", st, "demo")
        title, st2 = F.load_flow(Path(d) / "a.mflow")
        assert title == "demo" and st2 == st
    code = "import sys; sys.modules['tkinter'] = None; import mobility_tool.flow, mobility_tool.run_inputs; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stderr
    print("依赖检查、流程文件保存 / 打开；输入生成与流程引擎不依赖 tkinter")


if __name__ == "__main__":
    test_local_flow()
    test_gaussian_flow()
    test_check_and_files()
