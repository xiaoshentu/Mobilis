# 04 激发态速率与 TADF

热振动关联函数 (TVCF) 方法在谐振近似下计算激发态衰减速率，包括频率变化与 Duschinsky 转动：

| 过程 | 需要的耦合 | 命令选项 |
|---|---|---|
| 辐射 k_r 与发射光谱 | 跃迁偶极 (或振子强度)，可加 Herzberg–Teller 导数 | `--mu` / `--fosc`，`--ht-dipole` |
| 内转换 k_IC | S1–S0 非绝热耦合矢量 | `--nac` |
| 系间窜越 k_ISC / 反系间窜越 k_RISC | 旋轨耦合 (可加振动导数) | `--soc`，`--ht-soc` |

## 1. 准备 Gaussian 计算

每个态做 opt + freq，并保存 fchk：

```
# S0:  #p B3LYP/6-31G(d) opt freq
# S1:  #p B3LYP/6-31G(d) TD(nstates=5,root=1) opt freq
# T1:  #p B3LYP/6-31G(d) opt freq   (多重度 3，或 TD(triplets,root=1))
formchk S0.chk S0.fchk
```

振子强度取 S1 优化结构上的 TD 输出；S1–T1 旋轨耦合可用 ORCA (`%tddft dosoc true end`) 或 PySOC 计算；
非绝热耦合可用 ORCA (`%tddft nacme true end`) 或 Gaussian 16 的 `TD(NAC)`，每个原子一行 x y z (bohr⁻¹)。
fchk 中的总能量给出绝热能隙，也可以用 `--de` 指定。

## 2. 单个过程：S1 → S0 的辐射与内转换

演示文件 (苯的模型力场，见 `tvcf_demo/README.md`) 可以直接运行：

```
python -m mobility_tool tvcf --initial tvcf_demo/s1.fchk --final tvcf_demo/s0.fchk --fosc 0.05 --nac tvcf_demo/nac.txt --temps 77,300 --out bz
```

输出每个温度的 k_r、发射峰与 k_IC，`bz_rates.csv` 与 `bz_spectrum.csv` (可直接用 Origin 作图)。

## 3. 单个过程：系间窜越

```
python -m mobility_tool tvcf --initial tvcf_demo/s1.fchk --final tvcf_demo/t1.fchk --soc 0.8 --temps 300
python -m mobility_tool tvcf --initial tvcf_demo/t1.fchk --final tvcf_demo/s1.fchk --soc 0.8 --triplet-initial --temps 300
```

第二条是反系间窜越 (T1 → S1，能隙为负，对三个子能级平均)。

## 4. 完整的 S1 / T1 动力学

```
python -m mobility_tool tadf --s0 tvcf_demo/s0.fchk --s1 tvcf_demo/s1.fchk --t1 tvcf_demo/t1.fchk --fosc 0.02 --soc 0.8 --soc-t1s0 0.3 --temps 300 --out tadf.json
```

一次算出 k_r、k_IC、k_ISC、k_RISC、T1 的非辐射衰减，再求解 S1/T1 速率方程，给出瞬时与延迟荧光寿命以及量子产率。

## 5. 注意

- 低于约 50 cm⁻¹ 的大振幅模式不满足谐振近似，可用 `--floor-cm 50` 把它们抬高。
- `--damping-fs` 控制展宽 (默认 500 fs，σ ≈ 2 meV)；大分子的结果对它不敏感，小分子的内转换速率可能敏感。
- 方法与验证：`tests/test_v52.py` 把 FC、HT 与促进模式的结果与精确的态求和对比 (含频率变化与 Duschinsky 转动)，
  并检验正反速率的细致平衡。
