"""v1.21：以迁移率为目标的主动学习 (--al-goal mu) 与误差预算 (--al-budget)。
1. MuGoal 的线性化 μ 区间与 mu_posterior 的联合抽样区间一致 (同一组曲面)；
2. 命令行完整流程：--al-goal mu 每轮在全部近邻对中共选 batch × 对数 个点，μ 区间达到目标后停止；
3. 已有任务可在 sigma / mu 之间切换；--al-budget 输出误差预算表。"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from fake_adf import run_scan  # noqa: E402
from test_v27 import CIF, REF, cli  # noqa: E402
from mobility_tool.alsurface import ALRun, MuGoal, mu_posterior  # noqa: E402
from mobility_tool.scanflow import ScanProject  # noqa: E402


def test_goal_loop_and_budget():
    d = tempfile.mkdtemp()
    proj_dir = os.path.join(d, "pent")
    pr = ScanProject.create(proj_dir, REF, name="pent")
    pr.add_level()
    common = ["ceds", "al", "--cif", CIF, "--project", proj_dir, "--name", "g1", "--pairs", "1,3",
              "--temps", "300", "--supercell", "3,2,1", "--md-equil", "0.5", "--md-prod", "3", "--al-pool", "150",
              "--al-init", "8", "--al-batch", "6", "--al-goal", "mu", "--al-mu-target", "0.03", "--al-mu-draws", "0"]
    out = cli(*common)
    assert "以迁移率为目标" in out and "第 1 轮 16 个 ADF 任务" in out, out
    sizes = []
    for k in range(10):
        run = ALRun(ScanProject(proj_dir), "g1")
        assert run.goal == "mu"
        run_scan(ScanProject(proj_dir), run.state["rounds"][-1]["scan"])
        out = cli(*common)
        assert "TLT 迁移率" in out, out
        run = ALRun(ScanProject(proj_dir), "g1")
        if "全部近邻对已收敛" in out:
            break
        sizes.append(sum(len(v) for v in run.state["rounds"][-1]["selected"].values()))
    else:
        raise AssertionError("10 轮内没有收敛:\n" + out)
    assert all(s == 12 for s in sizes), sizes            # 每轮共 batch × 2 个点，按 μ 的方差在两对之间分配
    h = run.state["mu_history"][-1]
    assert h["converged"] and max(h["half"].values()) <= 0.03, h
    # 线性化区间 vs 联合抽样区间
    ce_args = None  # noqa: F841
    from mobility_tool.ceds import CEDS
    from mobility_tool.crystal import find_molecules, read_cif
    c = read_cif(CIF)
    ce = CEDS(c, find_molecules(c), cutoff=4.0, monomer=(run.proj.symbols, run.proj.Y), frame=run.proj.frame)
    surf = run.surfaces(ce)
    _, thermal = run.pool()
    mg = MuGoal(ce, surf, thermal, 300.0, 0.005)
    lin = mg.half_width()
    mp = mu_posterior(ce, surf, thermal, 300.0, n_draws=60, n_real=12)
    for car in lin:
        sd = np.std(mp[car]["draws"], ddof=1) * 1.96 / mp[car]["mu"]
        assert 0.5 < lin[car] / max(sd, 1e-9) < 2.0, (car, lin[car], sd)
        print(f"{car}: 线性化 ±{100 * lin[car]:.2f}%，抽样 ±{100 * sd:.2f}%")
    # 切换回 sigma，并输出误差预算
    out = cli(*[x if x != "mu" else "sigma" for x in common], "--al-budget", "5", "--al-budget-sigma-e", "30,40", "--force")
    assert "σ_V 的后验区间" in out and "误差预算" in out and "热运动样本" in out and "位点能无序" in out, out
    assert ALRun(ScanProject(proj_dir), "g1").goal == "sigma"


if __name__ == "__main__":
    test_goal_loop_and_budget()
    print("ok test_goal_loop_and_budget")
