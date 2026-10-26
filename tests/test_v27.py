"""ceds al：主动学习耦合曲面的完整流程 (命令行)。用模拟 ADF (软件内 EHT) 代替真实计算：
第 1 轮 → 算完 → 拟合并写下一轮 … 直到收敛；再用 --al-surfaces 的 GP 曲面做 MD → σ_V → TLT。"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from fake_adf import run_scan  # noqa: E402
from mobility_tool.alsurface import ALRun  # noqa: E402
from mobility_tool.scanflow import ScanProject  # noqa: E402

CIF = os.path.join(ROOT, "tests", "data", "pentacene_RT.cif")
REF = os.path.join(ROOT, "tests", "data", "pentacene_ref.xyz")


def cli(*args):
    r = subprocess.run([sys.executable, "-m", "mobility_tool", *args], cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}, text=True,
                       timeout=1500)
    assert r.returncode == 0, r.stderr[-3000:] + r.stdout[-2000:]
    return r.stdout


def test_ceds_al_loop():
    d = tempfile.mkdtemp()
    proj_dir = os.path.join(d, "pent")
    pr = ScanProject.create(proj_dir, REF, name="pent")
    pr.add_level()
    common = ["ceds", "al", "--cif", CIF, "--project", proj_dir, "--name", "al1", "--pairs", "1,3",
              "--temps", "300", "--supercell", "3,2,1", "--md-equil", "0.5", "--md-prod", "3", "--al-pool", "150",
              "--al-init", "8", "--al-batch", "8", "--al-target", "0.05"]
    out = cli(*common)
    assert "第 1 轮 16 个 ADF 任务" in out
    run = ALRun(ScanProject(proj_dir), "al1")
    assert run.exists and len(run.state["rounds"]) == 1
    try:                                                     # 没算完时不能继续
        cli(*common)
        raise AssertionError("未完成时应拒绝")
    except AssertionError as e:
        assert "没有算完" in str(e)
    for k in range(8):
        run = ALRun(ScanProject(proj_dir), "al1")
        run_scan(ScanProject(proj_dir), run.state["rounds"][-1]["scan"])
        out = cli(*common)
        print(out.split("\n\n")[-2] if "\n\n" in out else out, flush=True)
        if "全部近邻对已收敛" in out:
            break
    else:
        raise AssertionError("8 轮内没有收敛:\n" + out)
    assert "G1" in out and "G3" in out and "收敛" in out
    run = ALRun(ScanProject(proj_dir), "al1")
    sel = [i for r in run.state["rounds"] for idx in r["selected"].values() for i in idx]
    n_total = len(sel)
    assert n_total <= 2 * (8 + 8 * 8)
    # 种子：用 al1 各轮已算完的数据启动新的主动学习任务 (换一段 MD 的候选池) → 应直接收敛或只需少量补点
    seeds = ",".join(r["scan"] for r in run.state["rounds"])
    c2 = [x if x != "al1" else "al2" for x in common] + ["--al-seed", seeds]
    out2 = cli(*c2)
    assert "种子数据" in out2, out2
    run2 = ALRun(ScanProject(proj_dir), "al2")
    n2 = sum(len(i) for r in run2.state["rounds"] for i in r["selected"].values())
    assert n2 <= 2 * 8, (n2, out2)
    # 种子扫描全部未算完：不阻塞，按常规第 1 轮布点
    from mobility_tool.scanflow import make_cfg
    pr2 = ScanProject(proj_dir)
    pr2.add_scan("empty_seed", [dict(make_cfg(3.6 + 0.05 * i, 0.0, 0.0), set="al", group="1") for i in range(3)],
                 next(iter(pr2.meta["levels"])), kind="al", reduce=False, clash_min=0.0)
    c3 = [x if x != "al1" else "al3" for x in common] + ["--al-seed", "empty_seed"]
    out3 = cli(*c3)
    assert "第 1 轮 16 个 ADF 任务" in out3, out3
    out = cli("ceds", "md", "--cif", CIF, "--project", proj_dir, "--al-surfaces", "al1", "--temps", "300",
              "--supercell", "3,2,1", "--md-equil", "0.5", "--md-prod", "3", "--sigma-only")
    assert "GP 曲面" in out and "μ" in out


if __name__ == "__main__":
    test_ceds_al_loop()
    print("ok test_ceds_al_loop")
