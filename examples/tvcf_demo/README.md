# TVCF 演示文件

苯分子的**模型力场** Hessian (不是量子化学计算结果)，只用于演示 `tvcf` / `tadf` 的输入格式与流程：
`s0.fchk`、`s1.fchk`、`t1.fchk` 是三个态的 freq fchk (几何与 Hessian)，`nac.txt` 是 S1–S0 非绝热耦合 (每个原子一行，bohr⁻¹，随机数)。
真实体系请用 Gaussian 的 opt+freq (S1 用 TD-DFT) 计算并 `formchk`，见 docs/tutorials/04_激发态速率与TADF.md。
