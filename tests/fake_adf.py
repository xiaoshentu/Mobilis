"""模拟 ADF2016 片段轨道转移积分输出 (格式与真实 G1.out 相同)：V、J、S、e1、e2 由软件内 EHT 计算 (同一套单体轨道，
相位一致)，能量分解中的总成键能用 UFF。用于在没有 ADF 的环境中端到端检验扫描工作流。"""
import numpy as np

from mobility_tool.eht import eht_transfer_integrals, monomer, rotate_coefficients, build_basis
from mobility_tool.interaction import uff_interaction
from mobility_tool.surrogate import build_B, rotation


class FakeADF:
    def __init__(self, symbols, Y):
        self.sym, self.Y = list(symbols), np.asarray(Y, float)
        self.mono = monomer(self.sym, self.Y)
        self.A = [(s, *p) for s, p in zip(self.sym, self.Y)]

    def results(self, c):
        R = rotation(c["gamma"], c["roll"], c["pitch"])
        B = build_B(self.Y, c["z"], c["dx"], c["dy"], c["gamma"], c["roll"], c["pitch"])
        shells, _, _ = build_basis(self.sym, B)
        mb = dict(self.mono, shells=shells, C=rotate_coefficients(self.mono["shells"], self.mono["C"], R))
        Bat = [(s, *p) for s, p in zip(self.sym, B)]
        r = eht_transfer_integrals(self.A, Bat, esd=False, _mono_b=mb)["results"]
        e = uff_interaction(self.A, Bat)["total"] / 4.184
        return r, e

    def text(self, c):
        r, e = self.results(c)
        L = [" Charge transfer integrals relevant for hole or electron mobility calculations", "",
             " Electronic coupling V (also known as effective (generalized) transfer integrals J_eff)", ""]
        for lab, k in (("hole", "HOMO"), ("electron", "LUMO")):
            L.append(f" V for {lab} transfer:          {r[k]['J_eff_meV'] / 1000:.5f} eV")
        L.append("")
        for lab, k in (("hole", "HOMO"), ("electron", "LUMO")):
            L += [f" e1({lab}) Site energy {k} fragment 1:      {r[k]['e_A_eV']:.5f} eV",
                  f" e2({lab}) Site energy {k} fragment 2:      {r[k]['e_B_eV']:.5f} eV",
                  f" J({lab}) Charge transfer integral {k} fragment 1 - {k} fragment 2:      {r[k]['J_raw_meV'] / 1000:.5f} eV",
                  f" S({lab}) Overlap integral {k} fragment 1 - {k} fragment 2:       {r[k]['S_AB']:.5f}", ""]
        h = e / 627.5095
        def row(lab, v):
            return f"  {lab:<38s}{v / 627.5095:20.15f}{v / 627.5095 * 27.2114:15.4f}{v:17.2f}{v * 4.184:17.2f}"
        L += ["", row("Total Pauli Repulsion:", max(e, 0) * 2 + 1.0), row("Electrostatic Interaction:", -0.4),
              row("Total Steric Interaction:", 0.6), row("Total Orbital Interactions:", e - 0.6),
              f"Total Bonding Energy:{h:39.15f}{h * 27.2114:15.4f}{e:17.2f}{e * 4.184:17.2f}", "", " NORMAL TERMINATION"]
        return "\n".join(L) + "\n"


def run_scan(project, name, fail_ids=(), partial_ids=()):
    """模拟在 ADFjobs 中运行该扫描的全部批处理：写输出与完成标记。"""
    fa = FakeADF(project.symbols, project.Y)
    d = project.scan_dir(name)
    for r in project.manifest(name):
        if r["status"] != "ok" or r["rep"]:
            continue
        out = d / "outputs" / f"{r['id']}.out"
        if r["id"] in partial_ids:
            out.write_text(" ADF 2016 ... (interrupted)\n")
            continue
        if r["id"] in fail_ids:
            out.write_text(" ERROR: SCF did not converge\n")
            (d / "outputs" / f"{r['id']}.done").write_text("ok\n")
            continue
        out.write_text(fa.text(r))
        (d / "outputs" / f"{r['id']}.done").write_text("ok\n")
