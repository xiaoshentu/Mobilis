"""各功能所依据的方法文献 (帮助 → 引用文献)。发表使用本软件得到的结果时，请同时引用相应的方法文献。"""

SOFTWARE = ("Shi, Y. Mobilis: an integrated platform for charge transport and photophysics of organic semiconductors. "
            "Anyang Normal University (软件论文发表后请替换为正式引用)")

REFS = {
    "转移积分 (投影法 / 多能级 / 有效位点能)": [
        "Valeev, E. F.; Coropceanu, V.; da Silva Filho, D. A.; Salman, S.; Brédas, J.-L. J. Am. Chem. Soc. 2006, 128, 9882–9886.",
        "Baumeier, B.; Kirkpatrick, J.; Andrienko, D. Phys. Chem. Chem. Phys. 2010, 12, 11103–11113."],
    "原子对转移积分分析 (APTA)": [
        "Ozawa, K.; Okada, T.; Matsui, H. Sci. Technol. Adv. Mater. 2024, 25, 2354652."],
    "扩展休克尔 (EHT)": ["Hoffmann, R. J. Chem. Phys. 1963, 39, 1397–1412."],
    "Marcus 速率 / 跳跃迁移率": [
        "Marcus, R. A. Rev. Mod. Phys. 1993, 65, 599–610.",
        "Deng, W.-Q.; Sun, L.; Huang, J.-D.; Chai, S.; Wen, S.-H.; Han, K.-L. Nat. Protoc. 2015, 10, 632–642."],
    "动力学蒙特卡洛 (KMC)": [
        "Rühle, V.; Lukyanov, A.; May, F.; Schrader, M.; Vehoff, T.; Kirkpatrick, J.; Baumeier, B.; Andrienko, D. "
        "J. Chem. Theory Comput. 2011, 7, 3335–3345.",
        "Derrida, B. J. Stat. Phys. 1983, 31, 433–450. (KMC 精度自检中的一维无序精确解)"],
    "瞬态局域化理论 (TLT)": ["Fratini, S.; Mayou, D.; Ciuchi, S. Adv. Funct. Mater. 2016, 26, 2292–2315."],
    "全晶体刚体 MD (CEDS)": [
        "Jorgensen, W. L.; Maxwell, D. S.; Tirado-Rives, J. J. Am. Chem. Soc. 1996, 118, 11225–11236. (OPLS-AA)",
        "Fennell, C. J.; Gezelter, J. D. J. Chem. Phys. 2006, 124, 234104. (DSF 静电)",
        "Dullweber, A.; Leimkuhler, B.; McLachlan, R. J. Chem. Phys. 1997, 107, 5840–5851. (刚体转动积分)",
        "Leimkuhler, B.; Matthews, C. Appl. Math. Res. Express 2013, 2013, 34–56. (BAOAB Langevin)"],
    "极化位点能 (原子电荷 + Thole 诱导偶极)": [
        "Thole, B. T. Chem. Phys. 1981, 59, 341–350.",
        "van Duijnen, P. T.; Swart, M. J. Phys. Chem. A 1998, 102, 2399–2407.",
        "Tsiper, E. V.; Soos, Z. G. Phys. Rev. B 2003, 68, 085301.",
        "Rühle, V.; Lukyanov, A.; May, F.; Schrader, M.; Vehoff, T.; Kirkpatrick, J.; Baumeier, B.; Andrienko, D. "
        "J. Chem. Theory Comput. 2011, 7, 3335–3345. (VOTCA-CTP)",
        "Singh, U. C.; Kollman, P. A. J. Comput. Chem. 1984, 5, 129–145. (ESP 电荷)"],
    "无定形形貌 (MD → 转移积分 → 位点能 → KMC)": [
        "Rühle, V.; Lukyanov, A.; May, F.; Schrader, M.; Vehoff, T.; Kirkpatrick, J.; Baumeier, B.; Andrienko, D. "
        "J. Chem. Theory Comput. 2011, 7, 3335–3345. (VOTCA-CTP)",
        "Abraham, M. J.; Murtola, T.; Schulz, R.; Páll, S.; Smith, J. C.; Hess, B.; Lindahl, E. SoftwareX 2015, 1–2, 19–25. "
        "(GROMACS)",
        "Kabsch, W. Acta Crystallogr. A 1976, 32, 922–923. (刚性叠合)",
        "Michaud-Agrawal, N.; Denning, E. J.; Woolf, T. B.; Beckstein, O. J. Comput. Chem. 2011, 32, 2319–2327. "
        "(MDAnalysis，读取 xtc / trr)"],
    "ZINDO/S 与 MOO 转移积分 (VOTCA izindo)": [
        "Ridley, J.; Zerner, M. Theor. Chim. Acta 1973, 32, 111–134. (INDO/S)",
        "Mataga, N.; Nishimoto, K. Z. Phys. Chem. 1957, 13, 140–157.",
        "Kirkpatrick, J. Int. J. Quantum Chem. 2008, 108, 51–56. (MOO)",
        "Baumeier, B.; Kirkpatrick, J.; Andrienko, D. Phys. Chem. Chem. Phys. 2010, 12, 11103–11113. "
        "(二聚体投影与简并能级)",
        "INDO/S 参数：MOPAC (openmopac.net，Apache-2.0)，Zerner 的 ZINDO/S 参数。"],
    "能隙定律 / 振动分辨光谱": ["Englman, R.; Jortner, J. Mol. Phys. 1970, 18, 145–164."],
    "相互作用能 (抵消校正 / UFF / 能量框架)": [
        "Boys, S. F.; Bernardi, F. Mol. Phys. 1970, 19, 553–566.",
        "Rappé, A. K.; Casewit, C. J.; Colwell, K. S.; Goddard, W. A.; Skiff, W. M. J. Am. Chem. Soc. 1992, 114, 10024–10035.",
        "Turner, M. J.; Thomas, S. P.; Shi, M. W.; Jayatilaka, D.; Spackman, M. A. Chem. Commun. 2015, 51, 3735–3738."],
    "电荷与键级 (Mulliken / Mayer)": [
        "Mulliken, R. S. J. Chem. Phys. 1955, 23, 1833–1840.",
        "Mayer, I. Chem. Phys. Lett. 1983, 97, 270–274."],
    "IGM / IGMH 弱相互作用": [
        "Lefebvre, C.; Rubez, G.; Khartabil, H.; Boisson, J.-C.; Contreras-García, J.; Hénon, E. "
        "Phys. Chem. Chem. Phys. 2017, 19, 17928–17936. (IGM)",
        "Lu, T.; Chen, Q. J. Comput. Chem. 2022, 43, 539–555. (IGMH)",
        "Hirshfeld, F. L. Theor. Chim. Acta 1977, 44, 129–138."],
    "AIM 拓扑": [
        "Bader, R. F. W. Atoms in Molecules: A Quantum Theory; Oxford University Press: Oxford, 1990.",
        "Espinosa, E.; Molins, E.; Lecomte, C. Chem. Phys. Lett. 1998, 285, 170–173. (E ≈ V/2)"],
    "自由原子密度 (LDA)": ["Vosko, S. H.; Wilk, L.; Nusair, M. Can. J. Phys. 1980, 58, 1200–1211."],
    "NBO 分析": ["Reed, A. E.; Curtiss, L. A.; Weinhold, F. Chem. Rev. 1988, 88, 899–926."],
    "等值面 (marching cubes)": ["Lorensen, W. E.; Cline, H. E. Comput. Graph. 1987, 21, 163–169."],
    "量子化学程序": [
        "Frisch, M. J. et al. Gaussian 16, Gaussian, Inc., Wallingford CT, 2016.",
        "te Velde, G.; Bickelhaupt, F. M.; Baerends, E. J.; Fonseca Guerra, C.; van Gisbergen, S. J. A.; Snijders, J. G.; "
        "Ziegler, T. J. Comput. Chem. 2001, 22, 931–967. (ADF)"],
}


def text(tr=lambda x: x):
    colon = "：" if tr("本软件") == "本软件" else ":"
    out = [tr("本软件") + colon, "  " + tr(SOFTWARE), ""]
    for k, v in REFS.items():
        out.append(tr(k) + colon)
        out += ["  " + tr(r) for r in v]
        out.append("")
    return "\n".join(out)
