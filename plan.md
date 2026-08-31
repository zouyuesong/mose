# cpp_admm 实现计划(Step 1 / 1.5 / 2)

- 日期:2026-08-18
- 目标:C++ 复刻 golang_mosek 的完整 baseline(CSV 加载 → QP 建模 → 求解 → CSV 输出 → 进程伪装),
  求解算法用参数化后端从四个开源库中选择:`--engine=osqp|qpalm|scs|proxqp`
- 参考:根目录 `admm.md`(调研)、`log/admm/basic.md`+`improving.md`(Go 版实测)、
  `golang_mosek/src/opt/`(Go 参考实现)
- 前置结论:OSQP / QPALM / SCS / ProxQP 四库均有 C/C++ API,全部进 cpp_admm;
  python_admm(codegen/快速矩阵)推迟,不在本计划内
- **步骤总览**:
  - **Step 1**(已完成):四库 baseline——正确性(目标/约束/逐笔)+ 两大家族 + 3 家对比矩阵。
  - **Step 1.5**(已完成,见 §8):均衡 = 数据预处理——把 scaleRows() 前移成 `optim_scaled/`,
    证明与引擎无关;含 ADMM/Newton/IPM 三族的理论推演与实测。
  - **Step 2**(见 §9):文献背书——ABIP+(2209.01793)与 cuNRTO(2603.02642),
    对照我们的均衡/容差/裁判方法,提取下一代改进方向。

---

## 1. 库 × 语言可用性(实查)

| 库 | C/C++ | Python | 引擎接入要点 |
|---|---|---|---|
| OSQP | ✓ libosqp(C API,零依赖) | ✓ | 原生同构 `min ½x'Px+q'x s.t. l≤Ax≤u`;polish/adaptive_rho 开 |
| QPALM | ✓ C 库(CMake) | ✓ | 同构;近端 ALM + 半光滑 Newton,高精度段强;依赖 BLAS/LAPACK |
| SCS | ✓ C 库 | ✓ | **不支持双边行 l≤Ax≤u**:ranged 行拆成 ≤ub 与 ≥lb 两行(或 l=−∞/u=+∞ 的单边) |
| ProxQP | ✓ C++ header-only(proxsuite) | ✓ | 同构;依赖 Eigen |
| PDLP | ✗ 仅 LP | ✗ | 不适用 |
| QPPAL / ciPALM | ✗ MATLAB | ✗ | 不适用 |
| GLADMM / reflected-Halpern / 学习型 α | ✗ | ✗ | 无库,只能自研(留在 Go 路线 Phase 3) |
| Moehle-Boyd 定制 prox | ✗ | ✗ | 无库,自研(Go 版已存在,热循环将来可移 C 内核) |

### 1.1 算法家族划分(为什么四库是"两大家族"+ MOSEK 共三家)

四库都解同一个凸 QP,但算法谱系不同——它们全部源自**增广拉格朗日框架**
(把约束折进目标、带罚参数 σ/ρ),分岔在"子问题怎么解":

```
约束 QP → 增广拉格朗日
  ├── 一阶 ADMM 家族:子问题再拆 x/z 两块,每步只做 稀疏矩阵向量乘 + 逐行 clip 投影
  │     x-更新解一个固定线性系统(一次分解反复用),z-更新是区间投影
  │     每步 O(nnz) 便宜,收敛 O(1/k)——尾部越来越慢
  │     = OSQP(QP 原生形式)/ SCS(先锥形化 SOCP,z-步是锥投影)/ Go 自研版
  │
  └── 近端 ALM + (半光滑)Newton 家族:子问题用 Newton 法解到相当精度(二阶)
        每个外层迭代要更新稀疏因子分解(贵),但主动集识别后局部线性收敛,
        步数与问题规模弱相关
        = QPALM / ProxQP(INRIA 独立实现,加 Ruiz 预条件)

(MOSEK 的内点法是第三家:自洽障碍函数,每步解大 KKT,步数最少、每步最贵。)
```

本题实测(optim,358 只股票,position/trade 为名义值 notional)正是这条规律:

| 家族 | 库 | 迭代 | 单步成本 | solve |
|---|---|---|---|---|
| 内点 | MOSEK 9.3 | 17 | ~10ms | 0.17s |
| 近端 ALM+Newton | QPALM | 806 | ~0.5ms | 0.43s |
| 一阶 ADMM | OSQP / Go 版 | 1万–5.2万 | ~0.09ms | 2.5–4.7s |
| 一阶 ADMM(锥形) | SCS | 1e6 顶格 | ~0.17ms | 不收敛 |

选型含义:**步数敏感选 Newton 族(QPALM),单步成本敏感选 ADMM 族**;
ADMM 族还可用"低精度迭代 + 高精度 polish"剪尾(见 report_baseline.md §7)。

## 2. 目录结构

```
cpp_admm/
├── CMakeLists.txt            # C++17; -DUSE_OSQP/QPALM/SCS/PROXQP=ON/OFF 各自开关
├── src/
│   ├── csv.hpp/.cpp          # 移植 opt/types.go:三个 CSV loader(按列名匹配,未知列忽略)
│   ├── model.hpp/.cpp        # 移植 opt/solver.go(STEP1 放宽 / breachWithZeroTraded /
│   │                         #   noTradeResult)+ backend_admm.go 的 buildADMMProblem
│   │                         #   (固定变量 presolve + canonical QP: ½x'Px+q'x, l≤Ax≤u, CSC)
│   ├── proctitle.cpp         # 移植 main/proctitle.go:prctl(PR_SET_NAME) + argv 块覆写
│   ├── main.cpp              # CLI 与 Go 版对齐:-i -c -w -o --engine=...
│   └── engines/
│       ├── engine.hpp        # 统一接口: build(problem) → solve → {x,status,iters,ms}
│       ├── osqp_engine.cpp
│       ├── qpalm_engine.cpp
│       ├── scs_engine.cpp    # ranged 行拆两行的锥编码在此层做
│       └── proxqp_engine.cpp
└── test/
    ├── run_sample.sh         # sample 冒烟 + 与 Go 版输出 diff
    └── run_optim.sh          # optim 冒烟 + awk 约束校验(照 log/run_real.md)
```

## 3. 移植对照表(抄 Go 哪里)

| C++ 模块 | Go 来源 | 注意点 |
|---|---|---|
| csv.cpp | types.go `LoadOptInputsFromCSV` 等 3 个 loader | 按列名取数(与列序无关),无 tag 列静默丢弃——与 gocsv 行为一致 |
| model.cpp | solver.go:48-133(`Solve` 编排两遍 relax + STEP1 + breach 判定) | breach 判定用数据侧预计算版(solver.go:95-119),不是建模时顺手算的旧版 |
| buildADMMProblem | backend_admm.go:135-371 | 保留:固定变量消去(backend_admm.go:147-154 的 1e-5 相对宽判定)、x/u/g/t 变量布局、四类约束行、relax 弹性变量;**不移植**:GeneralStart 成对行契约与 Woodbury(那是自研求解器专属,库不需要) |
| polish/certifyKKT | — | **不移植**。baseline 用各库原生精度(OSQP polish / QPALM·ProxQP Newton);KKT 认证是 Go 专属优势 |
| proctitle.cpp | proctitle.go 全文件 | C 无 reflect:用 `__libc_argv`/`environ` 全局符号定位 argv 块(同 PostgreSQL 手法);先 clone 路径再覆写(main.cpp 里照 main.go:65-70 的顺序) |
| main.cpp | main/main.go | flag 名与默认值对齐;--engine 默认 osqp |

## 4. 各引擎接入细节

统一输入(由 model.cpp 产出,内存 CSC):
```
n 变量(x/u/g/t 块), m 行;PDiag[](对角≥0), q[], A(CSC), l[], u[]
```

| 引擎 | 调用要点 |
|---|---|
| osqp | osqp_setup(data, settings);settings: polishing=ON, adaptive_rho=ON(默认), eps_abs/rel=1e-6/1e-8;P 用对角稀疏 |
| qpalm | qpalm_setup(data, settings);默认设置起步,eps=同上 |
| scs | 每条 ranged 行 [l,u] 拆两行:Ax≤u、−Ax≤−l(等式行 l=u 保留原样);l=−∞/u=+∞ 的单边行不拆 |
| proxqp | proxsuite::proxqp::sparse::QP(n,m);H=对角,A=CSC;eps=同上 |

结果回收:取 x 前 na(活跃品种数)位 + PresolveInfo 展开成 per-symbol trade(照 backend_admm.go:87-122 `resultFromReduced`)。

## 5. 阶段划分

| 阶段 | 内容 | 验收 |
|---|---|---|
| A | csv + model + proctitle + main + **OSQP 引擎**(单此即为完整 baseline) | sample/optim 可跑,输出与 Go 版同格式 |
| B | + QPALM + SCS + ProxQP 三引擎 | `--engine` 四选一全部可跑 |
| D | 基准矩阵:四库 × 两数据集(wall/迭代/目标差)写入 `log/admm/baselines.md` | 与 mosek 6580.2285 / Go ADMM 6579.7531 对比表 |

(C = python_admm,推迟;E = 选优反哺 Go,不在本计划)

## 6. 验证标准(与既有记录一致)

- 目标值(最大化形式):optim 相对 mosek 参考差 < 1e-4;sample 同理
- 约束校验:log/run_real.md 的 awk 全套(逐品种 trade/position box、gross/delta 贴界、63 条 barra NET)
- 输出 CSV 三列 `sym,targetTrade,targetPosition`,行序=input.csv 行序,可与 Go 版直接 diff
- 进程伪装:运行中 `ps -o args` 显示 jobd,无 CSV 路径泄露

## 7. 风险

| 风险 | 对策 |
|---|---|
| SCS 拆行后行数翻倍(~4.4k 行) | 仅影响 SCS 引擎;scs 引擎内做,不污染统一 builder |
| QPALM 需 BLAS/LAPACK、ProxQP 需 Eigen,构建链可能拉不到 | CMake 开关可关;容器/本机先试 `find_package`,缺则该引擎编译为 stub 并在 --help 标注 |
| 库对 ±Inf 边界的约定差异 | 集中在 engine 层转换(OSQP/QPALM/ProxQP 用 1e30 惯例,SCS 用 INFINITY) |
| optim 的混合尺度(1e-6 vs 5e7)在各库容差下的表现不一 | 每库 eps 相同(1e-6/1e-8)保公平;逐行缩放判据是 Go 自研专属,不强制库 |

---

## 8. Step 1.5:均衡 = 数据预处理(零引擎改动)

实现后追加的发现与交付:§7.6/§7.7 证明 `--scale-rows`/`optim_scaled` 是
**与求解算法无关的纯代数输入变换**。本节先给三族的理论推演,再给实测
与可复现产物。

### 8.1 三个问题:为什么 ADMM 有效、MOSEK 无效、Newton 会怎么样

**共同机制**:行/列均衡是对 QP 做左右对角缩放 `Ã = D₁⁻¹AD₂⁻¹`,可行集与
最优解数学不变,只改变 ①KKT 矩阵条件数 κ ②每行的绝对尺度(影响全局
eps_rel 的"锚")。它对几族算法的影响取决于收敛界**是否依赖 κ** 与
**停机判据的尺度敏感性**:

| 算法族 | 收敛界是否依赖 κ | 对均衡的预期 | 实测 |
|---|---|---|---|
| 一阶 ADMM(OSQP/SCS/Go 版) | **是**:O(1/ε·κ) 型,次线性、尾部拖尾 | **大幅提速**:每步 O(nnz) 便宜,κ 改善直接砍迭代数;且把全局 eps_rel 从"锚到 5e7 行"重锚到"每行自身尺度",低精度点不再假收敛 | **25×**(4.3s→0.17s) |
| 近端 ALM+Newton(QPALM/ProxQP) | **弱**(局部二次收敛,step 数弱相关) | **中度提速**:迭代数主要被内循环残差阈值拉着跑;行/列均衡改善子问题条件数 → 半光滑 Newton 更快到阈值;无均衡时高精度段因 1e-6 行与 5e7 行同锚而多送迭代 | **9× / 87×** |
| 内点 IPM(MOSEK) | **否**(自和谐牛顿,迭代数与 κ 基本无关) | **无感**:MOSEK 内部已自带 Ruiz 式均衡,喂不喂已均衡数据都走自己的预条件路径;只换 KKT 坐标系不换步数,仅求解侧时间微降 | **0.54s vs 0.54s**(18→23 iter,单步更便宜) |

**关键推论(给 cpp_admm 多引擎 harness)**:均衡对"迭代数敏感族"收益最大,
对"单步成本敏感族"无感;低精度加速只在 均衡 之后才诚实(无均衡时
eps_rel 放松 1e-3 会崩 122%)。**把均衡做成预处理层而非求解器开关**
(model.cpp/optim_scaled 双形态)是唯一可移植、全引擎受益的形态。

### 8.2 交付物(已写入 cpp_admm/report_scaled.md)

- `make_optim_scaled.py`(仓库根,纯标准库):input 每品种 ÷band、
  alpha/cost ×band、lambda/tlambda ×band²;constraint lb/ub ÷σ;weight
  ×band/σ。band、σ 定义同 model.cpp。
- `optim_scaled/{optim_inputs,optim_constraints,optim_weights}.20260520`
  + `scale_map.csv`(sym→band,解后还原乘回)。
- MOSEK:目标 6580.2285(gross/delta 贴界,逐笔 mean 3.2/max 50.8),
  wall 0.54s ≡ 原数据——**与 §8.1 理论预测一致(MOSEK 无感)**。
- cpp_admm 交叉验证:原数据+`--scale-rows` vs scaled 无 flag,三引擎
  逐笔差 0.0007–0.014(机器精度级),目标全部 6579.757(−0.007%)。

---

## 9. Step 2:文献对照(读了两篇)

### 9.1 ABIP+(arXiv 2209.01793,math.OC,邓琪等,含 Yinyu Ye)

**它是什么**:ABIP = ADMM-based Interior Point,用 ADMM 近似解障碍罚问题
的双环混合法,专为大规模 LP/锥优化设计。增强版(ABIP+)加了 6 个技巧:
自适应 barrier 参数(Wächter-Biegler 激进 / LOQO)、restart、内环新终止
准则、half update、presolve(PaPILO)+ 对角预条件、null-objective 特化;
数值:Netlib 105 例几何均值时降 5.8×,SVM/PageRank 上优于 PDLP。

**与我们的直接对照(三处呼应)**:

| ABIP+(论文) | 本项目 | 意义 |
|---|---|---|
| §3.5 预条件 = Pock-Chambolle(ℓ^p 范数开根,单步)+ Ruiz(ℓ^∞ 迭代,行/列 ∞-范数→1),`Ã=D₁⁻¹AD₂⁻¹`;集成 PaPILO presolve | §8 的 scaleRows/optim_scaled = 同型左右对角缩放 + 固定变量 presolve(model.cpp) | **标准文献背书**:均衡是求解器公认预条件,不是实验 hack |
| "停机准则满足但仍不准确" → 用"相对 MOSEK 目标误差<1%"判解出(表 15) | 全篇以 6580.2285(或 mosek 输出)为裁判,eps_rel=1e-8 才不漏 | **假收敛是共通问题**,我们的 eps_rel=1e-8 是更严的裁判 |
| "比 PDLP 更受条件数影响",κ_A²‖Q‖²/ε 复杂度,"高精度仍落后商业求解器" | ADMM 家族 25×/87× 提速依赖均衡;1e-8 高精度仍 0.17s 强于 Newton 族 | 印证"一阶法=低精度/大规模/结构化","IPM=中小规模高精度"分工 |

**对 cpp_admm 的适用性**:presolve 思想(冗余行去重、单变量 box 行下沉
为变量边界)可在 model.cpp 的 buildProblem 层复用——与 scaleRows 同层,
四引擎全受益;ABIP 的内环技巧(restart/half-update/均值点终止)只可能
落在 vendored osqp(改 osqp_renamed 内循环),qpalm/proxqp 是 Newton 族
结构上不适用。

### 9.2 cuNRTO(arXiv 2603.02642,cs.RO,Wang/Abdul/Theodorou)

**它是什么**:CUDA 上的非线性鲁棒轨迹优化(NRTO),用 DR(Douglas-
Rachford)拆分解 SOCP 内子问题(NRTO-DR)+ ADMM 全 ADMM 变体(NRTO-
FullADMM),跨 139.6×。

**相关性**:与我们无直接算法重合(非线性 SOCP + GPU 并行),且该路线在
optim 上已被实测证伪——SCS(成熟的锥投影 DR/ADMM 实现)在 optim 上
1e6 迭代顶格不收敛(§1.1),1074 变量的规模也吃不到 GPU 并行收益。
仅佐证"DR/ADMM 同源,ADMM 家族"这一分类术语,**不立引擎方向**。

### 9.3 对 Step 2 的收敛结论

- 文献两篇都不打开新引擎方向(ABIP 仍属"一阶/混合"家族,cpp_admm 的
  proxqp/qpalm 已是更高精度成员;cuNRTO 是 GPU 专用)。
- 价值在**术语与判据的权威对账**:我们的"均衡=数据预处理"、"1e-8 高精度
  裁判"、"假收敛"三结论全部有文献对应;差异记录——ABIP+ 的预条件用
  通用范数(Ruiz 目标=∞-范数→1),我们用**问题特定尺度**(band/σ 锚到
  trade/delta/gross 语义),后者是 §7.6 结论 2 容差重锚的根源,目的是让
  eps_rel=1% 真正等于 trade 的 1%。
- ABIP+ 的 5.8× 是"从裸 ABIP 起步叠全套技巧"的总量,且主要份额来自
  presolve+rescale;cpp_admm 这边 rescale 份额已被 scaleRows 拿走
  (9~87×),OSQP 的 adaptive-ρ/polish、QPALM/ProxQP 的 Newton 机制
  本身已是成熟端态——文献技巧对成熟库的增量天然小,故不立待办。
