# Mobilis · 迁跃 (v1.20.2)

**开发者：Yarui Shi（安阳师范学院 Anyang Normal University）**

源代码：https://github.com/xiaoshentu/Mobilis · DOI：[10.5281/zenodo.23593489](https://doi.org/10.5281/zenodo.23593489) · 许可证 GPL-3.0

**有机半导体电荷输运与光物理计算平台** — 重组能、IP/EA、黄昆因子、激发态、转移积分 (Gaussian / ADF / EHT / ZINDO)、
Marcus / KMC / 瞬态局域化 (TLT) 迁移率。

名字来源：Mobilis 取自拉丁文“可移动的”，也是 mobility（迁移率）的词根；中文“迁跃”取“迁移”与“跃迁 / 跳跃”，
对应电荷在分子间的跳跃输运与激发态跃迁两条主线。Python 包名仍为 `mobility_tool`，命令行用法不变。

基于 Marcus 跳跃理论的完整流程，图形界面和命令行都能用：

晶体/分子结构 → 重组能（四点法，7 个能量）→ IP/EA/HEP/EEP 与 HOMO/LUMO
→ 重组能分解（黄昆因子、Duschinsky 矩阵、内坐标、原子；内置或 Gaussian FCHT 两条路线）
→ 转移积分（Gaussian 二聚体投影法 或 ADF 片段法）→ 迁移率（标量公式、迁移率张量、角分布、温度扫描）

## 安装与启动

```bash
pip install numpy scipy matplotlib
python run_gui.py            # 图形界面（Windows 下也可双击 Mobilis.bat）
python -m mobility_tool -h   # 命令行
```

界面按计算流程组织：主窗口显示结构 (文件 → 打开结构)，模型在“构建”菜单中处理，计算从 **Gaussian → 运行 Gaussian…**、
**ADF → 运行 ADF…** 和“迁移率”菜单 (软件内计算) 开始，任务在“任务 → 任务管理器”中排队运行，结果在“分析”菜单中读取。
多步计算可用 **任务 → 新建流程…** (流程图) 串成一个任务：例如 优化 → 重组能 → 近邻二聚体 → 晶体转移积分 → 迁移率。

界面使用 Python 自带的 tkinter。Windows/macOS 官方 Python 已包含；Linux 需要 `sudo apt install python3-tk`。
直接读取 ADF 的二进制结果文件需要 PLAMS（用 AMS 自带的 `$AMSBIN/amspython run_gui.py` 启动即可，
或 `pip install plams`）。

## v1.20.2
- 修正：迁移率表中给出 x, y, z 的路径，r / θ / γ 一律按矢量与当前传输平面计算。此前若载入的配置文件里同时写了 θ = 0，
  会沿用这个值，角分辨图把所有路径当成同一方向 (μ_min ≈ 0)。更换传输平面时表中的 r / θ / γ 也随之更新。

## v1.20.1
- 界面与文档中不再引用其它商业软件的名称：橄榄色标题栏的配色主题改名为“经典灰”，结构背景改名为“深蓝渐变”。
  旧设置中保存的主题名会自动回到默认主题。

## v1.20.0
**Δ-学习耦合曲面** (`ceds al`，新任务默认开启)：
- GP 的均值不再是常数，而是 β₀ + β₁·V_EHT(q)：软件内 EHT (每个构型约 10 ms) 给出耦合随几何变化的主要形状，
  GP 只学习 DFT 与缩放后的 EHT 之差。β 由数据估计 (通用克里金 / REML)，自动处理 EHT 的整体低估与符号；
  β 的不确定度计入后验方差。EHT 的 B 分子用同一套单体轨道旋转得到，相位与 ADF 片段法一致。
- 并五苯回溯检验 (400 个 MD 快照的直接 ADF 计算作检验集，四个近邻对)：主动学习时每个近邻对 16 个 DFT 点即可使
  耦合 RMSE < 1 meV (普通 GP 需要 40 个)，σ_V 误差 < 5% 需要 24 个 (普通 GP 32 个)，TLT 迁移率误差 < 5% 需要 12 个
  (普通 GP 20 个)。按收敛判据自动停止时共用 120 个 DFT 二聚体，σ_V 误差 < 1%；普通 GP 用 164 个，误差最大 4%。
- 后验区间的标定：同一回测中，GP 给出的 σ_V 95% 区间偏窄 (Δ-GP 在 ×1.5 放大后只覆盖 72%)。Δ-GP 的区间改为放大 3.4 倍
  (CI_INFLATE_DELTA)，收敛判据因此偏保守；曲面本身不受影响。
- 每轮报告 EHT 基线与已算 DFT 的相关系数；r < 0.8 时提示 (曲面仍然正确，只是优势变小)。
- `--al-delta auto|on|off`：已有任务保持原设置，on / off 可随时切换 (已算的数据全部保留)。

**曲面不确定度 → 迁移率**：收敛后自动把 GP 后验传递到 TLT 迁移率 (`--al-mu-draws`，默认 30 次抽样)，
给出只含耦合曲面误差的 μ 95% 区间。Python 接口 `alsurface.mu_posterior`。

修正：
- smoke_gui 在 Windows 上的两处测试问题：程序路径按规范化路径比较 (/tmp/fakeams 与 C:\tmp\fakeams)；
  本机没有 bash 时跳过运行模拟 ADF 脚本的几项。

## v1.19.1
- 修正：“最近打开的结构”中有已删除的临时文件时，Windows 上 (拒绝访问) 每次打开“文件”菜单都报错。失效记录现在自动清除。
- 测试不再把临时文件写进用户自己的最近文件列表 (smoke_gui 此前在设置隔离配置之前就打开了 t.cif)。
- test_gui_real：先测英文界面再测中文，避免两个主窗口同时存在；Windows 上窗口尚未完成布局 (宽度为 1) 时等待后再测量，不再误报“截断”。

## v1.19.0
**流程图 (任务 → 新建流程…，Ctrl+L，工具栏“流程图”)**：
- 把多个阶段串成一个任务一次运行：Gaussian (结构优化 / 四点法重组能 / 激发态 / 转移积分 / BSSE / ESP / FCHT)、
  ADF (二聚体 / 晶体全部近邻)、软件内 (搜索近邻二聚体 / EHT 转移积分 / Marcus 迁移率 / KMC)。
- 前一阶段的结果自动交给后一阶段：优化后的结构 → 后续 Gaussian 阶段；四点法 λ → 迁移率 / KMC；各二聚体 J → 迁移率。
  完成后 J 与 λ 自动填入“迁移率”页，可继续做角分辨与三维分析；全部结果汇总在 flow_results.json。
- 双击阶段打开设置：Gaussian / ADF 阶段用与“运行 Gaussian / 运行 ADF”完全相同的对话框 (底部改为 确定 / 恢复默认 / 取消)。
- 三个模板：EHT 快速筛选；Gaussian 完整流程 (优化 → 重组能 → 二聚体 → 晶体 J → 迁移率)；Gaussian 重组能 + ADF 晶体 J。
- 运行前检查阶段依赖 (例如迁移率阶段之前没有 J)；任一阶段失败时后续阶段跳过并标红；可随时停止。
- 流程可保存为 *.mflow，服务器上也能运行：`python -m mobility_tool.flow 流程.mflow 工作目录 --cif 晶体.cif`。
- Marcus 迁移率阶段对多分子晶胞同时给出均匀化精确解 (见 v1.18.0)。

**界面字体与配色** (参照常用材料与量子化学建模软件的界面规范)：
- 视图 → 主题配色：专业浅色 (默认)、经典灰、深色。
- 浅灰界面框架 + 白色工作区，只用一种强调色 (按钮、选中项、链接)；绿 / 橙 / 红只用于状态 (完成 / 警告 / 失败)。
- 界面字体 9 pt (Windows：中文 Microsoft YaHei UI，英文 Segoe UI)；日志与输入文件预览用等宽字体；标题只比正文大 3 pt。
- 图始终白底深色文字 (便于直接用于论文)，不随深色主题变化。

其它：
- 输入生成移到 run_inputs.py，不依赖界面 (流程引擎、命令行与测试共用)。
- 关闭主窗口时取消任务管理器 / 流程图的定时刷新 (此前偶尔在终端打印 invalid command name … poll)。

## v1.18.0
界面按计算流程重新组织：
- **主窗口**只显示结构：工具栏 (打开、图片、观察方向、测量、取代、结构面板、近邻二聚体、EHT → μ、Gaussian…、ADF…、迁移率…、任务)，
  橄榄色标题栏显示名称 · 化学式 · 空间群 · 晶胞；显示、取代、显示范围、晶胞缩放、分子列表、二聚体搜索放在右侧可隐藏的结构面板 (Ctrl+B)，按标签页分区。
- **菜单**：文件 / 编辑 / 构建 / 视图 / Gaussian / ADF / 迁移率 / 分析 / 任务 / 窗口 / 帮助。
  “构建”处理模型；“分析”读取已完成的计算；“任务”为任务管理器与程序路径。
- **运行 Gaussian…** (Ctrl+G)：一个对话框完成设置。标签页：计算 (结构优化+频率 / 四点法重组能 / 激发态 S0·S1·T1 / 二聚体转移积分 /
  抵消校正 / ESP 电荷与极化率 / FCHT)、泛函与基组、溶剂、资源、附加输入、预览输入；底部为任务名与“运行 / 写出输入文件 / 恢复默认 / 关闭”。
  结构取自主窗口 (当前分子、选中的两个分子或晶体全部近邻二聚体) 或文件。
- **运行 ADF…**：片段法转移积分 (选中的两个分子 / 二聚体文件) 与晶体全部近邻二聚体批处理，同样的标签页与按钮。
- 其余功能各自在独立窗口中打开，窗口内按分区成标签页，关闭时保留设置与结果；首次使用时才创建，启动更快。
- 界面中去掉 xtb (命令行保留)。

速度：
- 旋转 / 缩放晶体时用轻量预览绘制 (约快 5 倍)，松开鼠标后恢复完整光照；渐变背景改为色带，完整绘制也更快。
- 转移积分 (EHT / ZINDO)、重组能分解改在后台线程计算，界面不再卡住。

修正：
- **“沿 a / b / c 观察”会把晶格矢量原地归一化为 1 Å**，之后的近邻搜索 (近邻距离 1、2、3 Å)、导出二聚体 (原子重叠报错)、应变扫描 (曲线不变) 全部出错。
  角分辨迁移率以晶格矢量为传输平面时也有同样问题。均已修正并加回归测试。
- KMC 精度自检第 D 项 (多分子晶胞) 的“精确解”只在每个分子的跳跃矢量成 ±v 对时成立；DNTT (P2₁) 等不满足，导致 −35% 的假偏差。
  改用周期格子随机行走的均匀化精确解 (修正位移 χ)，DNTT 现为 −2.7%，在统计误差内。
- 转移积分页“结果填入迁移率表”在只算了单个二聚体 (EHT / ZINDO / Gaussian / ADF) 时也能用。
- 测试用的假 Gaussian 在 Windows 上改为 .bat (此前 smoke_gui 在 Windows 上报 WinError 193)。
- matplotlib 字体按字回退：雅黑缺少的上标负号 (cm²V⁻¹s⁻¹) 改用 DejaVu Sans 显示，不再警告。

## v1.17.2
界面整理（高分屏与英文界面）：
- Windows 显示缩放为 125% / 150% 时，字体随之放大而左侧控制栏宽度不变，导致按钮、下拉框和说明文字被截断。现在面板宽度、表格列宽、换行宽度都按屏幕缩放计算，内容放不下时面板自动加宽 (不超过窗口的 42%)。
- 一行放不下的按钮自动折成两行；并排的“标签 + 输入框”表单放不下时改为每行一项；过长的说明文字和选项自动换行。
- 普通按钮加浅色底和边框，与文字标签区分开；复选框随屏幕缩放放大。
- 英文界面：“Browse” 等按钮不再被截成 “Brows”；结构页六个 “Along” 按钮改为 “View along a / b / c / x / y / z”；导航栏两个过长的页名缩短；部分下拉框 (代理模型、取代基团) 显示中文、且选英文选项后读回的值不对，已修正。
- 相互作用能页的结构图与能量图各占一半，不再把能量图挤成一条。
- 新增 `tests/test_gui_real.py`：打开真实窗口，检查中英文 15 个页面无截断、字体缩放，并跑通 CIF → EHT → 迁移率；没有显示器时自动跳过。

## v1.17.1
Windows 兼容性修正（中文 Windows 上 `tests\run_all.py` 的 18 项失败）：
- 中文 Windows 控制台为 GBK，输出 Å、²、⟨⟩、∇ 等字符时报 `UnicodeEncodeError`；现在无法显示的字符以 ? 代替，计算与输出文件不受影响。
- 未装 WSL 发行版时，`C:\Windows\System32\bash.exe` 只是 WSL 启动器；查找 shell 时跳过它，改用 AMS 自带 bash 或 Git-Bash，都没有则提示在 ADFjobs 中运行 .run 文件。
- 未安装 scikit-image 时，测试跳过 ESP 表面部分（等值面功能仍需 `pip install scikit-image`）。
- `tests/run_all.py --ioenc gbk` 可在任何系统上模拟中文控制台；CI 增加此项。

## v1.17.0

**对映体 (手性分子的中心对称晶体)**
- mer-Alq3、Ir(C^N)2(acac) 等手性分子在 P-1、P2₁/n 等中心对称晶体中，一半分子是另一半的镜像，不能只用转动 + 平移重合。
  现在自动识别：这些分子的坐标系取非真转动，二聚体参数只描述真转动部分并加 mirror 标记，B 用镜像单体构建；
  晶体结构的重建误差为 0 (此前 Alq3、Ir-OO 的单体–晶体分子 RMSD 达 1.8–2.4 Å)。
- ADF：对映体二聚体的 B 片段引用镜像单体的片段文件 mono_m.t21 (`scan level --mirror` 生成 mono_m.run；
  批处理与运行前检查会提示)。
- 全晶体刚体 MD：镜像分子用镜像体坐标，取向仍为真转动，积分方法不变 (NVE 能量守恒检验通过)；快照与主动学习构型带 mirror 标记，
  GP 不对对映体对做 A↔B 交换增广；极化位点能的 MD 快照同样处理。
- 力场：增加 Al、Ga、Mg、Li、Be 的 UFF 参数和 Al、Ir、Pt、Cu、Zn、Ru、Os、Au 等原子质量 (此前含 Al / Ir 的晶体无法做 MD)。
- `--md-charges LOG`：MD 静电改用 Gaussian ESP 电荷 (pop=mk，按单体原子顺序自动对应)；含 N、O、金属的分子建议使用。

## v1.16.1

- `ceds al --al-seed SCAN[,SCAN]`：把项目中已有扫描 (如先前的局部扫描) 里已算完的构型作为主动学习的初始训练数据；
  未算完的构型不计入、不阻塞。有足够种子数据的近邻对第 1 轮直接按 IMSE 补点，已收敛的不再计算。
- 红荧烯 (CCDC 605654, 293 K) 模拟检验：π 堆积对 G1 约 22 个、T 形对 G2 约 32 个 DFT 点即收敛 (原局部扫描方案 286 个)。

## v1.16.0

- 主动学习耦合面 (`ceds al`)：高斯过程 (各向异性 SE 核，ML-II 超参) 拟合 DFT 转移积分，利用晶体对称性扩充数据；
  每轮在热运动样本上按积分方差下降 (IMSE) 选出一批构型并写成 ADF 输入，算完 collect 后重新运行同一命令，
  软件重新拟合，并用后验联合采样给出 ⟨V⟩、σ_V 的置信区间，判断是否需要下一轮。全部近邻对收敛后
  `ceds md --al-surfaces NAME` 直接用 GP 耦合面做热涨落统计和 TLT。
- 区间校准：并五苯真实 DFT 回测中未放大的 95% 区间只覆盖 34/52 例，默认放大 1.5 倍 (约 90% 覆盖)。
- 并五苯回测：每对 24–40 个 DFT 点时 σ_V 误差 ≤ 4%、快照 RMSE ≤ 1.1 meV；同样点数随机选点中位误差 4–7%，最差 15–21%；
  原二次拟合用 273 个点，σ 误差最高 8%。

## v1.15.0

- 无定形形貌：DFT 校准。先用 EHT 或 ZINDO 算全部近邻对，按 |J| 分布挑出约 20 对代表性二聚体做 DFT，
  空穴、电子各拟合一个比例系数 k (过原点最小二乘)，全部对 J ← k·J_半经验，有 DFT 值的对直接用 DFT 值。
  也可手动给定 k。界面：“无定形形貌 (MD)”页 → DFT 转移积分与校准；命令行：
  `morph --export-dimers DIR --calib-pairs 20`，算完后 `morph --calibrate DIR` (或 `--calibrate-csv`、`--scale-h/--scale-e`)。
  导出的 _pairs.csv 同时给出校准前的半经验 J。
- `zindo bench`：空穴、电子分开统计 (k、R²、MAE、|J| > 20 meV 的相对误差及按 k 缩放后的相对误差)，图改为两个子图。
  并五苯 18 个构型：ZINDO 相对 B3LYP 空穴 k = 1.10、电子 k = 0.80，合并统计会掩盖这一差别。

## v1.14.3

- 左侧导航栏可隐藏：点左上角 ☰、按 Ctrl+B 或菜单 视图 → 显示 / 隐藏导航栏；状态在下次启动时保持。
- 导航栏宽度按最长的页面名自动确定，长名称不再被截断。

## v1.14.2

- 修正：迁移率方法对比页的 MLJ 栏在同一容器中混用 pack 与 grid，真实 Tk 下界面无法启动。
- 界面测试的模拟控件按真实 Tk 的规则检查布局 (同一容器不能混用 pack 与 grid)，此类错误在发布前即可发现。

## v1.14.1

**极化位点能的外推修正**
- 诱导项默认用最大半径的值加 Born 尾项外推 (ε 取 Lorentz–Lorenz 估计)；1/R 拟合及其斜率对应的 ε 作为对照。
  并五苯检验：从 22–34 Å 出发的 Born 外推相差 < 0.02 eV；小团簇 (≤ 22 Å) 的 1/R 拟合斜率不合理、会高估约 0.2 eV。
- 静电与交叉项在球形截断下条件收敛、随半径摆动：取最大几个半径的平均，摆动幅度作为位点能差的误差；
  位点能差小于两倍误差时给出提示。默认团簇半径 14–30 Å (命令行)；界面默认 14–22 Å 的快速估计。
- 并五苯 (B3LYP/6-31G(d) ESP 电荷，极化率缩放到 DFT 42.8 Å³)：P+ = 1.09 eV，P− = 0.90 eV，和 1.99 eV
  (Tsiper & Soos 2003：2.01 eV)；两套分子的位点能差 ±2 ± 12 meV，与零无显著差别 (此前 ≤ 22 Å 得到的 ±27 meV 是截断假象)。

## v1.14

**三维各向异性迁移率**（迁移率页“三维各向异性迁移率”；命令 `mobility3d`）
- μ(n̂) = n̂ᵀ M n̂ 画成三维曲面 (半径 = 该方向的迁移率)，可用鼠标旋转；两种模型：
  Wen 公式的三维推广 M = e/(2k_BT) Σ m W P r rᵀ (任一平面上的截面与二维极坐标图完全相同)，
  或无规行走迁移率张量 μ = e/(k_BT)·½ Σ m W r rᵀ (与 KMC 一致)。
- 图中标出主轴、晶轴 a、b、c 与传输平面内的截面；文本给出主值、主轴方向及其晶体学方向 [u v w] 与和 a、b、c 的夹角。
- 导出 300 dpi 图片与可在浏览器中旋转的 HTML。
- 出版图的字体按 font.family 逐字回退：拉丁字符用 Arial，中文自动用雅黑 / 黑体 (此前图中的中文可能显示为方框)。

## v1.13

**ZINDO/MOO 转移积分 (VOTCA-CTP 的 izindo 方法)**（模块 `mobility_tool/zindo.py`；命令 `zindo`）
- 软件内 INDO/S (ZINDO/S) 闭壳层自洽场：Zerner 参数 (取自 MOPAC)，Mataga–Nishimoto γ (f_γ = 1.2)，
  σ/π 重叠加权 1.267 / 0.585，Slater 轨道重叠按长椭球坐标精确积分。
- MOO：参考单体的 ZINDO 轨道只算一次，随分子刚性转动、平移，J = c_Aᵀ F_AB c_B，不做二聚体自洽场；
  每对约 2 ms。单体轨道可用软件内 INDO/S，或 Gaussian ZINDO 单体计算的 fchk (与 VOTCA 相同)。
- 二聚体 INDO/S 投影作为检验；并五苯 18 个构型上 MOO 与二聚体投影：斜率 0.985、R² 0.998。
- 转移积分页新增“软件内 ZINDO/MOO”；无定形形貌页与 `morph --coupling zindo` 可用 ZINDO 计算全部近邻对。
- `zindo bench-inputs`：由晶体生成检验集 (晶体近邻对 + 共面 / 滑移 / 转动构型) 的 Gaussian ZINDO 与 DFT 输入；
  `zindo bench`：汇总软件内 MOO、MOO (Gaussian 轨道)、二聚体 INDO/S、Gaussian ZINDO 投影、DFT 投影与 EHT。

**近简并能级的有效耦合 (VOTCA 约定)**：Gaussian 投影法与 ZINDO 均可设简并阈值，与 HOMO / LUMO 能量差在阈值内的
轨道组 L_A、L_B 的耦合按 J_eff = √(Σ J_ij² / (N_A N_B)) 合并 (与简并轨道的任意组合无关)。

## v1.12

**无定形形貌 (MD → KMC)**（新页面“无定形形貌 (MD)”；模块 `mobility_tool/morphology.py`；命令 `morph`、`cif2gro`）
- 读取 GROMACS .gro (多帧、三斜盒子) 与 .pdb (CRYST1)；装有 MDAnalysis 或 mdtraj 时也读 .xtc / .trr / .dcd。
- 分子按残基拆分，跨周期边界的分子按最小像补齐；原子名自动推断元素 (可给对照表)。
- 每个分子用 DFT 参考单体刚性叠合 (重原子 Kabsch；原子对应关系自动求出，联合原子力场也可)；几何可取叠合后的
  参考单体或 MD 原子坐标。
- 近邻：最短重原子间距 < 截断 (含周期像)。转移积分：软件内 EHT，或导出 Gaussian 投影法输入后读回，或读入 CSV。
- 位点能：原子电荷静电，或静电 + Thole 诱导 (polar 模块)；给出 σ、近邻 ΔE 分布与位点能空间关联 C(r)。
- KMC：整个周期盒子为一个晶胞，位点能逐个分子给出；零场扩散与场迁移率。
- 方法对比页可选“无定形形貌”作为来源：比较只有 J 无序、相关位点能 (形貌给出) 与同 σ 的不相关高斯无序；
  依赖平移对称性的方法 (标量公式、解析张量、TLT、能带) 不再计算。
- `cif2gro`：晶体超胞写成 .gro，可作为 GROMACS 的起始结构。

## v1.11

**极化位点能**（新页面“极化位点能”，位于“迁移率”组；模块 `mobility_tool/polar.py`；命令 `polar`）
- 微静电模型：原子电荷 + Thole 诱导偶极 (线性阻尼，van Duijnen–Swart 原子极化率)。把晶体中一个分子换成带电态，
  计算静电 (电荷–四极)、交叉项与诱导能，诱导项按 1/R 外推到 R → ∞，得到固态极化能 P±、固态 IE / EA 与输运能隙。
- 各晶体学独立分子分别计算，给出子格子之间的位点能差。
- 由全晶体 MD 快照计算位点能的热涨落 σ_E (各子格子分别扣除平均值)。
- 电荷：Gaussian ESP 电荷 (中性 / 阳离子 / 阴离子，原子顺序自动对应)；没有时用 OPLS-AA 芳香 C/H 电荷 +
  EHT 前线轨道布居。可一键生成三个单点输入 (pop=mk，中性任务含 polar)，并把原子极化率缩放到 DFT 分子极化率。
- KMC 支持确定的子格子位点能 (可与高斯无序叠加)；方法对比页新增“Marcus KMC + 极化位点能”。
- 命令行：`python -m mobility_tool polar --cif x.cif [--neutral n.log --cation c.log --anion a.log] [--polar-log n.log]
  [--traj md_300K.npz] --ip-gas 6.59 --ea-gas 1.39 --out pol.csv`；`--gjf 前缀` 只生成 Gaussian 输入。

## v1.10

**迁移率方法对比**（新页面“迁移率方法对比”；模块 `mobility_tool/compare.py`）
- 同一套输入 (近邻图中的 J、重组能 λ、能量无序 σ、耦合涨落 σ_J) 一次算出：Marcus 标量公式、Marcus 无规行走张量、
  Marcus KMC (零场扩散 / 能量无序 / 场迁移率)、Marcus–Levich–Jortner KMC、Miller–Abrahams KMC、Bässler GDM 解析估计、
  瞬态局域化理论 (TLT)、能带 + 常弛豫时间，并可附加 CEDS (局域 MC / 全晶体 MD) 的结果。
- 输入来源：结构页晶体 J、迁移率页路径表，或 CEDS 的 ⟨V⟩ 与逐键 σ_V。
- 每个方法按输运区间给出适用性 (适用 / 边界 / 不适用) 与理由：2J/λ (非绝热跳跃)、σ/kT (静态无序)、σ_J/|J| (动态无序)、
  Ioffe–Regel 极限 (能带)。输出对比表、对数坐标条形图 (不适用的方法以浅色显示) 与 CSV。
- MLJ 的有效量子模 ħω 与经典部分 λ_s 可由重组能分解 (分解页的模式) 自动估计。
- KMC 新增 Miller–Abrahams 速率 (ν0 取该对的 Marcus 零偏速率)。

**界面**：左侧导航按“结构 → 单分子性质 → 电子耦合 → 迁移率”分组，当前页面高亮；内容区上方显示所在分组、页面名称与一句功能说明。

## v1.9

**全晶体刚体 MD**（CEDS 区的“全晶体 MD：σ_V 与 μ(T)”按钮；命令行 `ceds md`；模块 `mobility_tool/rigidmd.py`）
- 晶体超胞 (默认 4×3×2) 中所有分子同时作刚体运动，代替局域 MC 的“邻居固定” (Einstein) 近似，包含声学 (集体) 模式。
- 力场：芳香 C/H 用 OPLS-AA (含原子电荷)，静电用阻尼移位力 (DSF)；其他元素暂用 UFF LJ、电荷 0。
  积分：BAOAB Langevin + 刚体转动的对称分裂 (Dullweber–Leimkuhler–McLachlan)；分子坐标系不是惯量主轴系时自动换算。
- 每帧把超胞中每个晶体二聚体的全部实例换算为 (z, Δx, Δy, γ, θ, φ)，代入与 MC 完全相同的耦合曲面 → ⟨V⟩、σ_V、参数分配；
  平均结构移回实验结构 (可用 `--raw-geometry` 改用力场平均结构)。
- 由耦合的自相关时间 τc 直接给出 ħ/τc，可与 TLT 的 ħ/τ 参数对照；可同时计算多个 ħ/τ (`--htau-list`)。
- 输出：U_eq (扣除超胞整体漂移)、实验结构的晶格能 (与升华焓比较)、平动 / 转动温度、力场平均结构相对实验结构的偏移；
  μ(T)、σ = 0 对照、μ ∝ T^n；CSV、图与 npz 轨迹 (`--save-traj`)。
- 曲面组合 (MC 与 MD 通用)：“局域扫描 + 乘积模型 (T 对)”—— 平行二聚体用局域扫描，T 对用乘积模型 (G 网格 × Δx 扫描)；
  没有 ADF 输出文件时可直接给局域扫描的 results.csv (`--local-csv`)。
- **对称性较低的分子 (如红荧烯，C2h)**：晶体中每个分子的坐标系与参考单体逐原子匹配，刚性单体放回晶体与实际分子重合
  (此前对非 D2h 分子，主轴符号约定可能使部分分子的坐标系转了一个非对称操作的 C2)。并五苯等 D2h 分子结果不变。
- **分子坐标系可选“共轭主体”**（新建扫描项目时选择；命令行 `scan new --frame core`）：原点与坐标轴由最大的稠环体系确定，
  不含苯基等取代基；红荧烯的 π 堆积二聚体给出面间距 3.70 Å、长轴滑移 6.15 Å。
- **局域扫描范围可按全晶体 MD 确定**（`ceds local --md-half`，界面中对应复选框）：±3σ_MD，晶体没有原子位移参数时使用。
- **MD 快照检验**（“MD 快照 → ADF 批处理”与“快照检验：直接计算 vs 曲面”按钮；命令行 `ceds snapshots` / `ceds validate`）：
  从 MD 轨迹中为每个晶体二聚体等间隔取构型 (默认每对 120 个)，写成扫描项目中的 ADF 批处理 (与局域扫描共用 mono.t21)；
  算完后逐点比较直接计算的 V 与耦合曲面在同一几何上的预测，给出 MAE、RMSE、R²、⟨V⟩、σ_V 及 σ 比值的自助法 95% 区间。
  快照扫描只用于检验，不参与曲面拟合。

```bash
python -m mobility_tool ceds snapshots --cif pentacene_RT.cif --project pent_ceds --name md_snap \
    --temps 300 --pairs 1,2,3,7 --n-snap 120            # 也可用 --traj md_300K.npz 直接使用已保存的轨迹
# 在 ADFjobs 中运行 scans/md_snap/batch_md_snap_*.run，然后：
python -m mobility_tool ceds validate --cif pentacene_RT.cif --project pent_ceds --name md_snap \
    --g-csv scan_Gresults.csv --fdx-csv results_roll_dx.csv --out md_snap_validation.csv
```

- 复现并五苯计算结果 (室温晶体)：

```bash
python -m mobility_tool ceds md --cif pentacene_RT.cif --monomer project.json \
    --g-csv scan_Gresults.csv --fdx-csv results_roll_dx.csv --local-csv ceds_P_results.csv \
    --temps "100 150 200 250 300" --htau-list 2,10 --out pentacene_md.csv
python -m mobility_tool ceds run --cif pentacene_RT.cif --g-csv scan_Gresults.csv --fdx-csv results_roll_dx.csv \
    --local-csv ceds_P_results.csv --samples 2000 --out pentacene_mc_einstein.csv            # MC 下限
python -m mobility_tool ceds run ... --ueq-from-cif --out pentacene_mc_calibrated.csv           # MC 上限 (U_eq 校准)
```

## v1.8

**CEDS：耦合–能量双曲面法**（“批量扫描工作流”页的 CEDS 区；命令行 `ceds`）—— 不做分子动力学的动态无序与 μ(T)
- 晶体局域势：二聚体两个分子在晶体环境 (其余近邻固定) 中作刚体运动，对势 UFF (含色散)；Metropolis MC 给出非谐、含 A–B 相关的
  平动 / 转动分布。平均结构取实验晶体结构，势能只决定涨落；输出刚体涨落的 U_eq，可按晶体学 U_eq 缩放对势 (校准涨落幅度)。
- 每个样本换算为二聚体参数 (z, Δx, Δy, γ, θ, φ)，由耦合曲面给出 V：局域扫描的二次多项式 (DFT)、共面网格可分离模型或 EHT。
- 先做 MC (无需 DFT)，按 ±3σ 自动确定每个晶体二聚体的 DFT 局域扫描范围 (“晶体二聚体 → 局域扫描”)，扫描共用同一个 mono.t21。
- 位点轨道相位：每个分子固定一套坐标系，B 需绕自身二重轴 g 才落到扫描区域时 V = χ(g)·V(q·g) (χ 取 DFT 对称检验或 EHT)，
  TLT 中各键符号一致。
- 输出 ⟨V⟩、σ_V、σ_V/|⟨V⟩|、σ_V² 在各参数上的分配 (找出最伤迁移率的运动)、曲面覆盖率；TLT 给出 μ(T) 与 σ = 0 的对照、μ ∝ T^n。
- 测试数据：并五苯晶体 (Mattheus et al., Acta Cryst. C 2001, 90 K；tests/data/pentacene_90K.cif)。

## v1.7

**批量扫描工作流**（新页面“批量扫描工作流”；命令行 `scan`）：为二聚体耦合 / 相互作用能论文批量生产数据
- 扫描项目：单体只读一次并转到分子坐标系 (x 长轴、y 短轴、z 法线；与 make_dimers.py 完全相同)，报告点群、对称操作偏差、平面性。
- 计算层级：每个层级 (泛函 PW91 / PBE / BP86 / BLYP / B3LYP / PBE0，基组 DZ–TZ2P，可选 D3(BJ)) 一个单体片段 mono.t21，
  该层级的所有二聚体共用同一个文件 (V 的符号严格可比)；也可导入已算好的 mono.t21。
- 扫描类型：参数网格 (z、Δx、Δy、γ、θ、φ 任意组合)；随机验证集 (拉丁超立方，内插 / 外推分组，固定种子可重现，接触过近的样本自动补足)；
  对称等价检验 (单体点群全部整体操作 + B 绕自身二重轴的变体)；晶体近邻二聚体 (换算为冻结单体参数并给出 RMSD)。
- 按单体点群约简：对称等价构型只算一次，结果自动展开到全部请求的构型；接触筛选 d/(rA+rB)。
- 分块批处理 (ADFjobs 用 .run，本软件“运行任务”用 .sh)：每个构型完成后写 .done 标记，可随时中断续算；
  运行前检查片段文件、路径 (中文 / 空格)、失败与中断的构型 (自动去掉标记以便重算)、磁盘空间。
- 结果收集：V (大小取 ADF 打印值，符号由 J、S、e1、e2 重算并标出不一致)、J、S、e1、e2 与 ADF 能量分解
  (Pauli、静电、轨道、总成键能，kcal/mol)；写 results.csv。
- 换层级重算：同一批构型换泛函 / 基组；Gaussian 抵消校正 (含 D3) 相互作用能输入与 Gaussian 投影法转移积分输入 (A 只算一次)。
- 报告 (HTML，图 300 dpi)：距离衰减 β；滑移图 (四象限、V = 0 等值线、分子轮廓)；距离与滑移的可分离性 (相邻 z 的逐点比例因子、
  等效 β、按 |Δy| 分箱的比值、变号点；滑移矩阵 SVD 秩一占比)；V = kS 拟合；五参数解析式及标准误差；
  模型 M1 / M2 / M3 与随机验证 (MAE、RMSE 内插 / 外推、最大误差、R²、符号一致率、parity 图)；不同层级的对照；
  相互作用能图与 Boltzmann 加权 <V>、√<V²>、有效构型数；重组能 (四点法目录或手动输入) 与晶体二聚体的 Marcus 速率。
  同时写 model/model.json、model_grid_z*.csv、validation/predictions.csv、validation_summary.txt。
- 归档：all_structures.xyz (每个二聚体一帧，注释行含参数与 V)、all_results.csv、README.txt 与项目全部文件 (可不含 ADF 原始输出)。
- 可导入 make_dimers.py 生成的已有扫描目录，直接参与分析与归档。

## v1.6

**弱相互作用与电子密度拓扑**（“波函数分析”页）
- IGM / IGMH：δg_inter 等值面按 sign(λ2)ρ 着色 (蓝：强吸引；绿：范德华 / π–π；红：位阻)，δG_inter 积分，散点图，cube 导出。
  密度来源：当前 fchk (IGMH，实际密度)、结构页选中的晶体二聚体或结构文件 (IGM，前分子密度，秒级，无需量子化学计算)。
  片段自动按成键划分或指定前 N 个原子；IGMH 提供原文实现与勘误中的数学严格形式。
- AIM：牛顿法搜索键 / 环 / 笼临界点，追踪键径 (分子间加粗)；键临界点的 ρ、∇²ρ、椭圆度、G、V、H 与 Espinosa 估计 E ≈ V/2；
  Poincaré–Hopf 关系自动检查临界点是否找全。
- 密度的解析梯度、Hessian 与动能密度 (s 到 f 基函数)，按空间分块并筛选远处壳层。
- 自由原子密度：自洽求解球对称 LDA (VWN) 原子 (atomdens.py)，H–Xe 及常用重元素预先计算随软件发布，其它元素首次使用时计算并缓存。
  (Multiwfn 使用其内置数值原子密度；两者都来自原子计算，数值可能略有差别。)

## v1.5

**波函数分析**（新页面“波函数分析”；直接由 Gaussian fchk 计算，不依赖 Multiwfn）
- 轨道浏览器：轨道能量、占据、HOMO/LUMO 标注；双击计算轨道格点并显示 ± 等值面；导出标准 cube 文件。
  基函数支持 s / p / sp / 5D / 6D / 7F / 10F，按 Gaussian 的顺序与归一化约定。
- Mulliken 电荷与 Mayer 键级 (含开壳层与赝势)：原子按电荷着色，键上标注键级；也可读取 log 中的 ESP / NPA / Mulliken 电荷。
- 分子表面静电势：电子密度 0.001 a.u. 等值面上的 ESP 色图、表面极值点与统计量 (面积、平均、σ²tot)。
  严格 ESP 由 Gaussian 自带的 cubegen 计算；也可读入已有 cube，或用原子电荷近似 (定性)。
- NBO：二阶微扰能 E(2) 表 (按阈值筛选；二聚体可只看分子间 A↔B 相互作用)、自然电荷。
- 帮助 → 引用文献：各功能依据的方法文献，一键复制。
- 功能思路参考 GXNU-MolStudio (GPLv3)，代码为独立实现，未使用其源代码。
- 依赖新增 scikit-image (等值面)；打包脚本已包含。

## v1.4

**KMC 正确性核对与修正**
- 核对结果：跳跃选择、等待时间、扩散张量估计、外场项均正确；零场布拉维格子、多分子晶胞 (红荧烯 Z=4)、温度依赖、
  弱场漂移迁移率 (Einstein 关系) 与精确解一致。
- 修正：有能量无序时，原实现让载流子在全部位点上均匀出发并按固定步数统计，会混入初始高能暂态，且在遍历性差时偏向慢区域。
  现改为 Boltzmann 平衡起点 + 平衡期 + 固定模拟时间统计 (系综平均)；与一维环能量无序的精确解 (Derrida 1983) 在
  σ = 30 / 60 / 90 meV 下一致。无无序时结果不变。
- 新增“KMC 精度自检”(迁移率页；命令行 kmc-validate)：12 项与精确解的逐项比较，导出 CSV 与 300 dpi 对照图，可作论文补充材料。
- 深入分析：说明 fchk 来源 (Gaussian 转移积分单点 G?_A / G?_B / G?_AB)，选中二聚体 fchk 后自动补全 A、B。

## v1.3

- **按方向选择二聚体**（结构页 → 选择方式 “按方向 (传输平面内，Deng 2015)”）：只保留传输平面 (自动识别分子层，或 ab / bc / ac / 三维)
  内的近邻；同一直线上只保留最近的一个；对称等价但方向不同的二聚体分别列出；命名 P (最近的 π 堆积方向)、T (其它紧密接触方向)、
  L (远程)、I (层间)。红荧烯 (Cmca) 在 bc 分子层得到 P (b)、T1、T2 ((b±c)/2)，质心 15 Å 时另有 L (沿 c 等)，与
  Deng et al., Nat. Protoc. 2015, 10, 632 的二聚体选择方式一致；可在二聚体表中“移除选中组”。EHT 计算、导出与迁移率表沿用该选择。
- **光谱计算出错修复**：Gaussian 不接受 def2-TZVP 这种写法 (QPErr 语法错误)；写出输入时自动转换为 Def2TZVP 等 Gaussian 写法。
  新增 Gaussian 常见错误诊断 (语法错误并显示出错位置、原子过近、SCF 不收敛、磁盘 / 内存不足、电荷与多重度、chk 缺失等)，
  读取激发态结果与运行任务失败时给出原因与建议。
- **开发者信息**：“帮助 → 关于 / 开发者信息”、标题栏、启动画面、命令行 --version、安装包发布者与 exe 文件属性。

## v1.2

**转移积分深入分析**（转移积分页 → 深入分析；按公开方法自行实现）
- 原子自动匹配（思路参考 CATNIP / QC_Tools）：Gaussian 单体与二聚体的原子顺序可以不同，按坐标匹配原子并映射分子轨道系数。
  软件的 Gaussian 转移积分计算统一改用此实现（以前原子顺序不一致时只给警告、结果错误）。
- 多能级转移积分矩阵（思路参考 kndo/ti；Valeev 2006）：H−n…L+n 全部组合的 S、J、J_eff 与正交化后的有效位点能，
  热图显示；检测近简并轨道并提示（简并轨道的单对耦合不唯一）。
- 原子对转移积分分析 APTA（tcal；Ozawa 等, Sci. Technol. Adv. Mater. 2024）：u_αβ 分解，Σu_αβ 严格等于 J_eff；
  三维图按原子贡献着色并用圆柱标出贡献最大的原子对，热图与 CSV 导出。
- 数据来源：软件内 EHT（秒级）或 Gaussian 的 A、B、AB fchk。

**修复与改进**
- 写出 Gaussian / ADF / 抵消校正输入前做几何检查：分子内原子重叠、同一分子选了两次、两分子交叠时拒绝生成并指出原子与距离。
- ADF 批量：原子顺序不同但对称等价的分子 (惯性主轴对齐 + 匈牙利匹配 + Kabsch 检验) 共用同一个单体片段，
  红荧烯由 4 次单体片段计算减为 1 次；新增“ADF 并行核数 (NSCM)”设置；运行时显示“已完成 x / y 个二聚体”；
  大体系 (二聚体 > 100 原子) 导出时提示计算量并建议 DZP 基组。
- 从界面中移除 xtb 方法（转移积分、方法对比、程序路径设置、相互作用能）。

## v1.1

**原子 / 基团取代**（“结构/晶体”页 → 原子取代）
- 打开“取代模式”后在三维图中点击原子选择 (再次点击取消)；“换成元素”弹出完整元素周期表 (118 个元素，
  粗体为软件内 EHT 有参数的元素，悬停显示原子序数与共价半径)；或从下拉框选择基团：
  −F −Cl −Br −I −CH₃ −CF₃ −OH −OCH₃ −NH₂ −CN −NO₂ −C≡CH。
- 调整键长：末端原子沿原键方向按共价半径调整；自动增减氢：价态降低时删除 H (芳环 C–H → N 得到吡啶型 N)，
  价态升高时在键角平分线外侧补 H；基团与母体局部平面共面，四面体基团转到离其它原子最远的取向。
- 晶体：用 CIF 的对称操作找出所有等价位置，先在参考分子上取代、再由对称操作复制到每个分子，结果严格保持
  晶体对称性；分子位于对称元素上 (如反演中心) 时，分子内的等价位置同时取代；也可“仅所选分子” (结果为 P1)。
  取代后检查与相邻分子的最近接触 (含周期像)，原堆积容纳不下时提示需要晶体结构优化。
- 撤销；导出 xyz / gjf / CIF (P1)；“送去重组能计算”把取代后的分子 (晶体中为参考分子) 填入“重组能”页，
  优化后可继续计算 IP/EA、激发态、转移积分；晶体取代后可直接重新搜索二聚体、计算 EHT、迁移率、相互作用能。

**说明**：耦合代理模型与堆积设计页 (第 10 页) 已包含，相关修正待完整扫描数据到位后进行。

## v1.0.1

- 按 ADF2016 真实输出格式读取转移积分 (“V for hole transfer: … eV”、“J(hole) Charge transfer integral …: … eV” 等)，
  同时读取 J、S、e1、e2，并用 V = (J − S(e1+e2)/2)/(1 − S²) 校验；批量结果直接从 G*.out 读入迁移率表。

## v1.0

**新功能**
- **分子间相互作用能**（新页面）：UFF 力场 (秒级，可加 ESP/Mulliken 静电)、xtb GFN2 (半定量)、
  Gaussian Counterpoise=2 (抵消校正 BSSE，供发表；生成全部近邻组输入，运行后自动读取)；
  **能量框架** (参考 CrystalExplorer：圆柱粗细 ∝ |E_int|，蓝=吸引、红=排斥) 与晶格能近似 E_latt ≈ ½ Σ E_int；
  能量与 |J| 并列作图、导出 CSV。
- **结构测量** (参考 GaussView)：测量模式下依次点原子，得到距离 / 键角 / 二面角，图中标注。
- **文件选择**：所有文件框改为下拉框，列出最近使用的同类文件和工作目录 (含下一层) 中的匹配文件，
  右侧显示状态 (✓ 正常结束 / … 未完成 / ✗ 出错或不存在)；“文件”菜单新增“设置工作目录”和“最近打开”。
- **参数提示**：鼠标悬停在常用参数上显示说明 (26 项)。
- **黄昆因子结果导入**：分解页可读取 FCHT 输出或 CSV / TXT / DAT 表格 (MOMAP、DUSHIN 等导出的频率与 S 两列，
  自动识别表头和序号列)，用于作图、导出和振动分辨光谱。

**修复**
- **ADF 结果无法读取**：新增直接解析 ADF 文本输出 (.out，与 parse_results.py 相同的途径)；按 文本导出 → .out →
  PLAMS → dmpkf 依次尝试，dmpkf 按节导出为空时改为整体导出；方法对比也传入 ADF 安装信息；全部失败时逐项说明原因。
- **转移积分没有进入迁移率表**：方法对比完成后自动填入；“填入迁移率所用方法”默认“自动 (精度最高)”
  (Gaussian > ADF > xtb > 标定后的 EHT，个别组缺数据时用次一级方法补上)；新增“直接读入迁移率表”，
  一次读入 Gaussian / ADF 目录中全部二聚体的结果。
- **ADF 环境测试**：列出 shell、ADFBIN、程序文件、许可证、临时目录、返回码与 ADF 输出，并生成 adf_env_test.run
  供在 ADFjobs 中对照。
- **FCHT (重组能分解) 出错 “FileIO operation on non-existent file”**：运行前检查两个 chk 是否在任务目录、
  是否来自正常结束的频率计算、是否在计算后被改写；读取失败的 FCHT 输出时给出原因与建议
  (或改用“内置”分解，直接读 fchk)。

## v0.9

- **一次计算全部二聚体**：结构页导出 ADF 时生成一个 `batch_adf.run` (与 make_dimers.py 的 batch_*.run 相同思路)，
  先算单体片段，再依次算全部二聚体；能通过纯转动 + 平移重合的分子共用同一个片段 (如并五苯的 mono/1、mono/2)，
  手性分子的镜像拷贝自动单独计算片段；已完成的步骤自动跳过 (可中断续算)；可按 chunk 拆分为多个文件。
  `.run` 为 GBK 编码，可在 ADFjobs 中直接运行；同名 `.sh` 可在本程序“运行任务”中运行。
- **运行完自动读取结果**：每个二聚体结束后用 dmpkf 导出文本；“运行任务”结束后自动读取各组 Vh / Ve 并填入“迁移率”页。
  在 ADFjobs 中运行时，可点“监视 ADFjobs 结果”，程序每 10 秒检查进度，全部完成后自动读取。
- **修复 `'gbk' codec can't decode byte ...`**：中文 Windows 默认 GBK 编码，而脚本为 UTF-8。现在所有未指定编码的
  文本读写统一按 UTF-8 写入、按 UTF-8 → GBK 依次尝试读取 (以前保存的 GBK 文件也能读)。此问题还曾导致
  Windows 下识别不出 ADF 结果文件标记，一并修复。

## v0.8.1

- **ADF2016 脚本格式**改为与在 ADFjobs 中验证可用的 `.run` 一致：所有文件用绝对路径 (正斜杠)，每步输出写入单独的 `.out`，
  已完成的步骤自动跳过 (可中断后续算)，BASIS 中 `createoutput None`，并加 `NOPRINT LOGFILE`。
  除 `run_*_adf.sh` 外同时生成 `*_adf.run` (GBK 编码)，可在 ADFjobs 中直接打开运行。
- **选择 ADF 安装目录**：文件夹对话框始终显示在当前窗口前面；可直接“选择 adf 程序文件…” (bin 中的 adf / adf.exe)；
  选了上一级目录时向下搜索三层；识别失败时说明检查了哪里、缺什么。运行窗口新增“打开任务文件夹”。
- **左上角**：图标移到左上角，旁边两行显示 “Mobilis 迁跃” 与副标题、版本号，点击图标显示“关于”。

## v0.8 新增

**字体大小**：标题栏 A− / A+ 按钮、菜单“字体”(小 / 标准 / 大 / 特大 / 超大) 或 Ctrl + 加号 / 减号 / 0。
界面文字、表格行高、文本框与侧栏宽度实时缩放，设置自动保存。

**右上角图标**：标题栏右侧显示缩小版 Mobilis 图标，点击查看“关于”。

**ADF2016 等旧版 ADF**：
- 转移积分页 ADF 版本默认“自动”，按检测到的安装生成对应格式（ADF2016 → `$ADFBIN/adf` 旧版输入，含 `End Input`）；
- 运行窗口新增 ADF 设置区：选择安装目录（选上一级目录也能找到）、显示版本与 shell、**测试 ADF 环境**（H2 单点）；
  任务列表标明 Gaussian / ADF，不再重复列出 run_all_adf.sh；
- 以前生成的 AMS 格式脚本在 ADF2016 上运行时自动按旧版格式重新生成；
- 每步计算后检查 TAPE21 / adf.rkf，失败时立即停止并指出是哪一步；
- 结束时用 ADF 自带 dmpkf 导出结果为文本，读取 TAPE21 不再需要 PLAMS（打包后的程序也能读）。

## v0.7 新增

**速度**（真实规模测试：单分子 94 原子、Pbca、晶胞 752 原子）：

| 操作 | v0.6 | v0.7 |
|---|---|---|
| 读取 CIF | 0.83 s | < 0.01 s |
| 软件内 EHT 计算全部二聚体 | 8.3 s | 1.3 s |
| 动态无序 σ_J (30 个构型) | 约 84 s | 5 s |
| 显示 6016 原子 (成键 / 绘制) | 0.08 / 0.48 s 每次 | 0.01 s 仅一次 / 0.31 s |

做法：EHT 只算分子间重叠块并缓存单体（与完整算法结果逐位一致）；刚体转动后 MO 系数用 p/d 轨道转动矩阵直接变换；
CIF 对称去重改用周期 KD-tree；成键按分子模板缓存；着色向量化、大场景自动降低细节；
消除选中分子时的重复绘制，窗口缩放与拖动限帧。耗时计算（EHT、应变扫描、KMC、动态无序、TLT、能带、方法对比）
在后台线程运行，状态栏显示进度条，界面不再卡住。

**图标与启动画面**：`mobility_tool/assets/` 中的 Mobilis 图标（两个分子间的跳跃箭头 + 磷光星点），用于窗口、任务栏、
安装包和快捷方式；打包后首次加载显示启动画面。

**独立安装程序**：`packaging\build_windows.bat` 一键用 Nuitka 把全部代码编译为机器码，再用 Inno Setup 生成
`Mobilis_Setup_0.7.0.exe`（安装向导、开始菜单 / 桌面快捷方式、卸载），发布文件中不含源代码。详见 `packaging/打包说明.md`。

## v0.6 新增

**ADF 任务可以直接运行**：以前 ADF 只生成 .sh 脚本，Windows 不能直接执行，运行窗口也只识别 .gjf，所以“没有反应”。
现在运行窗口同时列出 ADF 任务，由 AMS 自带的 bash（或 Git-Bash / 系统 sh）执行，并自动设置 AMSHOME、AMSBIN、
AMSRESOURCES、SCMLICENSE、SCM_TMPDIR；成功与否以是否生成 adf.rkf / TAPE21 判断，失败时给出日志末尾。
**工具 → 程序路径设置** 可指定或自动检测 Gaussian、ADF/AMS 安装目录 (如 C:\AMS2024.104) 和 xtb。

**全新的结构显示**：自绘渲染器——带明暗与高光的球体、双色圆柱键、深度排序、透视投影、
a/b/c 坐标指示；球棍 / 棍状 / 空间填充 / 线框四种样式，深蓝渐变 / 白色 / 浅灰渐变 / 黑色背景；
左/右键拖动旋转、滚轮缩放、中键平移、左键单击选分子；沿 a/b/c/x/y/z 观察；保存 300 dpi 图片。

**下拉菜单界面**：顶部菜单栏 (文件 / 结构 / 计算 / 分析 / 视图 / 工具 / 语言 / 帮助) 与右上角“页面”下拉框，取代原来的标签页。

**根据 GitHub 上的有机半导体计算工具加入的功能**（新页面“输运理论与光物理扩展”）：

| 参考 | 功能 |
|---|---|
| grimme-lab/xtb (DIPRO) | xtb DIPRO 转移积分，作为第四种方法加入方法对比与批量计算 |
| CiuK1469/TransLoc | 瞬态局域化理论 (TLT) 迁移率：周期超胞 + 热涨落 J，输出平面内迁移率张量与角分布 |
| 动态无序分析 | 用软件内 EHT 对二聚体做刚体热振动采样，得到 σ_J 及 σ/|J|，可直接作为 TLT 输入 |
| lucydot/effmass | 紧束缚能带、带宽、有效质量张量及常弛豫时间能带迁移率 |
| 生成函数法 (Turbomole radless 等) | 由黄昆因子计算振动分辨磷光光谱 (FWHM)，Englman–Jortner 能隙定律给出相对 k_nr |
| OCELOT (堆积描述符) | π 堆积几何：接触区面间距、夹角、滑移距离与堆积类型 |

检验：TLT 在无无序时严格等于 Drude 能带结果；有效质量与带宽与解析值一致；振动光谱重现单模泊松级数；
能隙定律随 ΔE 单调下降；渲染器拾取最前面的原子。

## v0.5 新增

**中英文界面**：右上角 “English / 中文” 按钮切换（窗口自动重建，设置会保存）；也可 `python -m mobility_tool gui --lang en
python -m mobility_tool tlt IrBB.cif --carrier electron --sigma 0.3 --htau 5 --plane ab
python -m mobility_tool vibronic IrBB_fcht_T1_to_S0.log --ev 1.65`。

**软件内计算转移积分 (EHT)**：转移积分页选“软件内计算 EHT”，或结构页“软件内计算全部二聚体 J (EHT)”，
不需要高斯/ADF，单个二聚体通常不到 1 秒。实现：Slater 轨道用 6 个高斯函数拟合 (1−重叠 < 1e-6)，Obara–Saika 解析重叠积分
(s/p/d，Ir/Pt 等 d 轨道为双 ζ)，加权 Wolfsberg–Helmholz Hamiltonian，单体轨道直接投影得到 J_eff，同时给出能级分裂 (ESD)。
内置 H、B、C、N、O、F、Al、Si、P、S、Cl、Se、Br、I、Cu、Ru、Os、Ir、Pt 的 Hoffmann 参数（可在 eht.py 的 EHT_PARAMS 中补充/修改）。
EHT 是半经验方法，适合大批量筛选和看趋势；**定量结论请先用“三种方法对比”与 DFT 核对**。

**三种方法对比**：对一个二聚体目录（crystal 导出的 dimers.json + 各组 Gaussian fchk / ADF 结果）同时给出 EHT、
Gaussian DIPRO、ADF 片段法的 J，画分组柱状图和 EHT–DFT 对照散点图，并给出相关系数和比例因子 k（|J_DFT| ≈ k|J_EHT|）。
可选择用哪种方法（或 “EHT×比例因子”）的结果填入迁移率表：先对几个二聚体做 DFT 标定，其余用标定后的 EHT。

**KMC 模拟（参照 VOTCA-CTP）**：迁移率页 “KMC 模拟” 区。近邻图可用迁移率表（布拉维格子）或结构页晶体中的全部分子
（每个分子都搜索近邻，按等价二聚体组赋 J）；速率可选 Marcus 或 Jortner（半经典，ħω 可调）；外重组能可用常数或球模型
（Pekar 因子、半径）；可加外电场和高斯能量无序 σ（周期超胞，保证细致平衡）。零场由均方位移得到扩散张量与迁移率张量，
有场时由平均速度得到场迁移率，均附统计误差；布拉维格子、无无序、无场时与解析张量对照（测试中两者在统计误差内一致）。

**晶胞显示与大小调整**：显示范围可按 a、b、c 分别设起止晶胞（可为负，如 −1→2 显示以原点为中心的 3×3×3），可显示每个晶胞的边框；
“晶胞大小调整”按百分比缩放 a、b、c（分子刚性平移、内部几何不变），“应变扫描”沿某方向逐点用 EHT 计算 J 并求迁移率，
画出迁移率–应变曲线。

## Windows 下高斯“双击 bat 不运行”的处理

最省事的办法是在界面右上角点“▶ 运行高斯任务”（或各页的“▶ 在本机运行”）：程序会自动寻找 g16.exe
（PATH、GAUSS_EXEDIR、C:\G16W 等），设置 GAUSS_EXEDIR / GAUSS_SCRDIR，按依赖顺序逐个运行、检查是否正常结束，
并自动 formchk。若要用 bat，生成的 run_*.bat 已改为：GBK 编码（中文系统下不会被错误解析）、自动切换到 bat 所在目录、
设置环境变量和临时目录、找不到 g16 时给出提示。如高斯不在 C:\G16W，请改 bat 第三行的 G16DIR。
**注意：路径中有中文或空格时 Windows 版高斯经常无法运行**，请把计算文件夹放在如 `D:\calc\IrBB` 的纯英文路径下；
程序在运行前会检查并提醒。

## 图形界面

**① 结构/晶体**：打开 xyz/gjf/log/fchk/cif 并三维显示。鼠标：滚轮缩放、右键拖动旋转、中键平移、
左键点选分子（Ctrl/Shift+左键多选），“重置视角”恢复。可显示 a×b×c 个晶胞；分子列表与三维图联动；
可把选中分子导出为 xyz，或把选中的两个分子直接导出为二聚体（A 在前）并填入转移积分页。打开 CIF 时程序展开对称操作，
把被晶胞边界切开的分子拼完整，显示晶胞框。“搜索近邻二聚体”按最短原子间距（或质心距离）找出
参考分子的全部近邻，并把对称等价的二聚体合并成组（G1、G2…），每组只需计算一次转移积分。
在表格中点选近邻即可在三维图中高亮该二聚体。“导出等价二聚体 + 生成转移积分输入”会按④页的设置
（Gaussian 或 ADF）为每组生成输入，并写出记录全部近邻矢量的 dimers.json。

**② 重组能**：下拉选择或手动输入泛函与基组（如 `B3LYP em=gd3bj`），可选 PCM 溶剂。体系含 Ir 等重原子
而基组为 Pople/Dunning 系列时自动改用 GenECP（轻原子用所选基组，重原子用 SDD，可修改）；def2 系列基组高斯自带赝势，不需要 GenECP。可以单独指定“单点泛函/基组”，
例如 B3LYP/6-311G(d,p) 优化、B3LYP/6-311G(2d,2p) 算能量：此时每个文件多一步同态单点，
全部 7 个能量都取单点级别。读取 log 时按 route 段切分 Link1 步骤，Windows 版高斯的输出同样可以识别，
并自动检查虚频、优化收敛和异常结束。同时生成 run_reorg.sh（Linux）和 run_reorg.bat（Windows）。

**③ IP / EA**：（另可计算 HOMO-1…LUMO+1 的片段组成，SCPA 或 Mulliken，可自动按“金属 / 各配体”划分片段，
对应论文 Table 1 的 FMO 部分）可从重组能目录自动读取，也可以手动填写 7 个能量（中性优化、正电优化、负电优化、
中性几何下的正/负离子、正/负离子几何下的中性分子），得到垂直/绝热 IP、EA，HEP、EEP，λh、λe
（eV、kJ/mol、kcal/mol），并从 fchk 或 log 读取 HOMO/LUMO、能隙和 Koopmans 近似值，画能级示意图。

**④ 重组能分解 / 黄昆因子**：两种方法可选，结果可以互相对照。

- 内置方法：直接读两个态 opt+freq 的 fchk（Hessian），按末态极小点的简正模式分解
  （λ1 用中性态模式，λ2 用离子态模式，与 Dushin 一致），给出 λ_i、黄昆因子 S_i、Duschinsky 矩阵、
  键长/键角/二面角分解和原子贡献三维图。不需要再跑高斯。
- Gaussian FCHT 方法：一键在重组能目录生成 `freq(readfc,fcht,readfcht)` + `print=(huangrhys,matrix=JK)`
  输入（可选 RedDim=ClearLowFreq），算完后读 log 得到黄昆因子、λ_i = S_i·hν_i、Duschinsky 矩阵和位移矢量。
  取代原来 gau_reorg.py → Origin → csv → 画图脚本的流程。

“导出论文图 (a)(b)”生成与 Huang_and_Duschinsky.py 同风格的 300 dpi 图（黄昆因子柱状图 + Duschinsky 热图）
以及重组能分解图；“导出数据”生成 黄昆因子.csv、矩阵.csv、重组能分解.csv、.hr.txt、.mat.txt，可直接用于 Origin。

**⑤ 激发态 / 光谱**：生成 S0 opt+freq、S1 opt (TD root=1)、T1 opt (TD triplets) 及各自的单点级别 TD 计算
（支持优化/单点不同基组、GenECP、PCM）；读取后给出吸收、荧光、磷光能量、波长、振子强度和主要组态（Table 4 格式），
Stokes 位移，S0–S1 / S0–T1 RMSD 及叠合图（Fig. 7），多个体系的吸收 + 磷光光谱对比图（Fig. 9，FWHM 可调）及数据，
以及由 PLQY 和寿命计算 k_r、k_nr（Table S5）。也可任选两个结构计算 RMSD。

**⑥ 转移积分**：选择 Gaussian（nosymm iop(3/32=2)，由三个 fchk 投影计算 J_eff，可算 HOMO-1…LUMO+1）
或 ADF（TransferIntegrals 关键词，支持 AMS2020+ 与 ADF2019 以前两种输入格式，泛函/基组可选）。
二聚体的原子顺序如果交错，“自动识别”会按成键拆分并重排保存。

**⑦ 迁移率**：按论文所用公式计算二维角分辨迁移率
μ_φ = e/(2k_BT) Σ r_i² W_i P_i cos²γ_i cos²(θ_i − φ)，电子、空穴同时计算（λe、λh、Ve、Vh 分列），
输出各路径速率 W 与概率 P、μmax 及方向、各向异性，并画电子/空穴叠加的极坐标图；可把多个体系加入对比并导出
Fig. 6 风格的多子图。路径可手动填写 r、θ、γ，也可以只给跳跃矢量由程序按所选传输平面计算，或直接导入
`名称,d,Theta,Ve,Vh` 格式的 CSV（见 examples/Ir-BB_table3.csv）。用论文 Table 3 的 Ir-BB 数据在 300 K 下
重现 μe,max = 10.84、μh,max = 0.058 cm²V⁻¹s⁻¹。原有的三维标量公式和迁移率张量结果一并给出。
表格中每行一个跳跃路径（双击编辑）。可从①直接导入近邻矢量，或“从二聚体目录导入”
一次性读入所有组的 J。温度可写单值、列表或 `200:400:20`。给出跳跃矢量时同时输出迁移率张量主值、
主轴和指定平面的角分布极坐标图。

界面采用统一的蓝灰配色、可滚动的左侧控制栏、右上角可点击的流程导航和底部状态栏；
Windows 高分屏下自动开启 DPI 适配以避免模糊。

## 命令行

```bash
# CEDS：动态无序与 μ(T)
python -m mobility_tool ceds local --cif pentacene.cif --project pent --name ceds_local   # 生成晶体二聚体的 DFT 局域扫描
python -m mobility_tool ceds run --cif pentacene.cif --project pent --temps "100 150 200 250 300" --out ceds.csv
python -m mobility_tool ceds run --cif pentacene.cif --source eht                       # 无 DFT 数据时的 EHT 预览

# 批量扫描工作流 (论文数据)
python -m mobility_tool scan new pent --mono pentacene.log
python -m mobility_tool scan level pent --xc PW91 --basis TZP          # 生成 monomer/PW91_TZP/mono.run
python -m mobility_tool scan grid pent zscan --z 3.2:5.0:0.1
python -m mobility_tool scan grid pent slip35 --z 3.5 --dx 0:6:0.25 --dy 0:3:0.25
python -m mobility_tool scan valid pent valid --interp 20 --zi 3.5,3.8 --extrap 20 --ze "3.3,3.5;3.8,4.2"
python -m mobility_tool scan check pent symcheck --bases "3.5 1.2 0.7 0; 3.6 2.0 0.4 25"
python -m mobility_tool scan precheck pent slip35                      # 续算前检查 / 修复完成标记
python -m mobility_tool scan collect pent
python -m mobility_tool scan report pent --lam-h 0.092 --lam-e 0.132 --energy ADF
python -m mobility_tool scan archive pent

# 重组能
python -m mobility_tool reorg-gen mol.xyz -o reorg --method "B3LYP em=gd3bj" --basis "6-31G(d)"
python -m mobility_tool reorg-analyze reorg
python -m mobility_tool reorg-gen mol.xyz -o reorg --basis "6-311G(d,p)" --sp-basis "6-311G(2d,2p)"
python -m mobility_tool reorg-decomp reorg/mol_neutral.fchk reorg/mol_cation.fchk --csv decomp

# IP / EA
python -m mobility_tool ipea --dir reorg
python -m mobility_tool ipea --energies E1 E2 E3 E4 E5 E6 E7 --orbitals mol_neutral.fchk
python -m mobility_tool orbitals mol_neutral.fchk

# Gaussian FCHT：黄昆因子 + Duschinsky 矩阵
python -m mobility_tool fcht-gen reorg --clear-low-freq 100
python -m mobility_tool fcht-read reorg/mol_fcht_cation_to_neutral.log -o figs

# 晶体 → 二聚体 → 转移积分 → 迁移率
python -m mobility_tool crystal xxx.cif --cutoff 4.0 -o dimers --program gaussian
cd dimers && bash run_ti.sh && cd ..
python -m mobility_tool mobility --dimers dimers --lam 0.18 --carrier hole

# 论文式角分辨迁移率（CSV: name,d,theta,Ve,Vh；V 为 eV）
python -m mobility_tool mobility2d examples/Ir-BB_table3.csv --lam-e 0.130 --lam-h 0.225 --temp 300 --plot fig6.png

# 激发态 / RMSD / 轨道组成 / 本机运行高斯
python -m mobility_tool excited-gen IrBB.xyz -o exc --basis "6-311G(d,p)" --sp-basis def2-TZVP --solvent dichloromethane
python -m mobility_tool excited-read exc IrBB
python -m mobility_tool rmsd S0.log T1.log
python -m mobility_tool orbcomp IrBB_neutral.fchk --method mulliken
python -m mobility_tool run exc --gaussian C:\G16W

# 软件内 EHT / 三种方法对比 / VOTCA 式 KMC / 应变扫描
python -m mobility_tool eht-ti --dimer pair.xyz --na 61
python -m mobility_tool compare-ti dimers
python -m mobility_tool kmc IrBB.cif --lam 0.13 --carrier electron --sigma 0.05 --field "1e4 0 0"
python -m mobility_tool strain-scan IrBB.cif --axis c --range=-3:3:1 --lam-e 0.13 --lam-h 0.225
python -m mobility_tool gui --lang en
python -m mobility_tool tlt IrBB.cif --carrier electron --sigma 0.3 --htau 5 --plane ab
python -m mobility_tool vibronic IrBB_fcht_T1_to_S0.log --ev 1.65

# 单个二聚体
python -m mobility_tool ti-gen --dimer d.xyz --na 36 -o ti --name P1
python -m mobility_tool ti-calc --prefix ti/P1
python -m mobility_tool adf-gen --dimer d.xyz --na 36 -o adf --functional PW91 --basis TZP
python -m mobility_tool adf-read adf/dimer_full.results/adf.rkf
```

## 测试

```bash
python tests/test_all.py            # 重组能解析、转移积分投影、迁移率公式
python tests/test_decomposition.py  # 用解析力场 Hessian 检验三种分解方式都给出精确的谐振 λ
python tests/test_crystal.py        # CIF 对称展开、跨边界分子拼接、等价二聚体合并
python tests/test_photophys.py      # 激发态解析、Stokes、RMSD、光谱、k_r/k_nr、轨道组成
python tests/test_v18.py            # CEDS：晶体二聚体参数、特征标、MC 温度标度与校准、局域曲面、TLT
python tests/test_v17.py            # 批量扫描：对称约简、续算检查、解析、对称检验、模型、验证、报告、归档、晶体参数
python tests/test_v16.py            # LDA 原子密度、密度导数、IGM / IGMH、AIM
python tests/test_v15.py            # 波函数：基函数归一化、Mulliken/Mayer、密度积分、cube、ESP 表面、NBO
python tests/test_v14.py            # KMC 精度自检、能量无序的系综统计与 MSD 记录
python tests/test_v13.py            # 按方向选择二聚体 (红荧烯)、基组写法与 Gaussian 出错诊断、开发者信息
python tests/test_v12.py            # 多能级矩阵、原子对分解、原子自动匹配、几何检查、红荧烯与 ADF 片段共用
python tests/test_v11.py            # 原子 / 基团取代：单分子、晶体对称取代、位点对称
python tests/test_v10.py            # 相互作用能、测量、ADF .out 解析、FCHT 诊断、黄昆因子导入
python tests/test_v09.py            # GBK/UTF-8 文本读写、ADF 批量 (片段共用 / 续算 / 自动读取)
python tests/test_v08.py            # ADF2016 识别 / 自动转换 / 环境测试 / 无 PLAMS 读取；字体缩放
python tests/test_v07.py            # 快速 EHT = 完整算法、MO 转动、CIF 去重、图标与打包文件
python tests/test_v06.py            # TLT、有效质量、振动光谱、能隙定律、π 堆积、渲染器、ADF 运行、xtb 解析
python tests/test_eht_kmc.py        # EHT 重叠积分/转移积分、KMC 与解析张量、Jortner 极限、应变
python tests/test_fcht.py           # FCHT 输出解析 (D 指数、粘连负数、分块矩阵)、导出与作图
python tests/smoke_gui.py           # 无显示环境下逐个调用界面按钮回调
```

## 注意事项

- FCHT 路线中黄昆因子默认与 log 中第一组频率（本任务读入的末态频率）配对，与 gau_reorg.py 相同；
  如有疑问可与内置方法的结果对照。用了 RedDim 时模式数会少于 3N-6。
- IP/EA 只在 7 个能量处于同一计算级别时才有意义；HOMO/LUMO 的 Koopmans 近似通常与 ΔSCF 的 IP/EA 有明显差别。

- 重组能分解是谐振近似，其总和与四点法 λ 的差别反映非谐性；离子态有虚频时结果不可靠。
- 内坐标分解在冗余内坐标下不唯一（各项可正可负），适合比较“哪类结构变化贡献大”，
  不宜过度解读单个数值。
- CIF 中的无序原子只保留占有率 ≥0.5 或 disorder_group 为 1 的部分，复杂无序请先手动处理；
  缺少氢原子的 CIF 需要先补氢。
- 等价二聚体按“全部分子间原子距离排序后相同”判定，适用于 Z'=1；Z'>1 时请对每个独立分子
  分别作为参考分子搜索。
- ADF 的 TransferIntegrals 节数值为 Hartree，程序已换算为 meV/eV。
- Gaussian 转移积分必须保留 nosymm 与 iop(3/32=2)；二聚体中 A 的原子必须在前。
- Marcus 公式适用于跳跃区（J 明显小于 λ）。
