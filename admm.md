# ADMM 求解 QP 调研报告

> 面向 `golang_mosek`(csvport 组合交易优化器)用 ADMM/一阶法替换 MOSEK 内点法的可行性调研。
> 日期:2026-08-17。配套实测记录:`log/admm/basic.md`(基础款)与 `log/admm/improving.md`(逐步优化)。

---

## 0. 结论(TL;DR)

1. csvport 的问题(`min ½x'Px + q'x s.t. l ≤ Ax ≤ u`,默认 `soft-lambda=0` 时 **P 为对角阵**)
   是 ADMM 的标准理想形式(OSQP 同构)。TUB/GUB 辅助变量与弹性变量在 ADMM 下可保留亦可折叠,规模不增。
2. 单次求解延迟:MOSEK 0.17s 中含 env 创建/license 检查/QP→SOCP 锥形化 + 内点迭代的固定开销;
   ADMM 每次迭代仅需稀疏矩阵向量乘 + 逐坐标 prox(问题结构下线性系统可用对角+Woodbury 精确求解),
   千变量级预期 **几~几十 ms**。参考 Moehle-Boyd 2021:~1000 证券 + 100 因子 10–100ms。
3. 部署:OSQP Apache 2.0 无 license、可静态链接;自研 ADMM 为纯 Go,单二进制,
   摆脱 `libmosek64.so.9.3`、`libcilkrts.so.5`、`mosek.lic`。
4. 精度:1e-4~1e-6 相对精度需求 → 纯 ADMM 到中精度快、高精度慢,需 **polish**(末段主动集精修)
   或 Newton 兜底(QPALM 思路)。
5. 推荐路径:① 基础款 OSQP 式 ADMM 跑通与 MOSEK 对拍 → ② 逐项加前沿改进(§3)
   → ③ 必要时删辅助变量/折叠弹性变量的定制 prox 版本(§4 Moehle-Boyd 框架)。

---

## 1. 为什么 csvport 的 QP 是 ADMM 理想对象

现有 solver.go 建模(最小化形式):

```
min   Σᵢ (λᵢ+tλᵢ)·xᵢ² + (λᵢ·posᵢ − αᵢ)·xᵢ + costᵢ·uᵢ − elasticPenalty·Σtₖ
s.t.  x ∈ [max(minTrade, minPos−pos), min(maxTrade, maxPos−pos)]   (变量界)
      u ≥ |x|            (两行/品种, 线性化交易成本)
      g ≥ |pos+x|        (两行/品种, 线性化 gross 敞口)
      NET:   Σw·x ∈ [lb−Σw·pos, ub−Σw·pos]
      NET_TRADE: Σw·x ∈ [lb, ub]
      GROSS: Σw·g ∈ [0, ub]
      GROSS_TRADE: Σw·u ∈ [0, ub]
      (relax 模式) 弹性 tₖ ≥ 违约量, tₖ ≥ 0
```

ADMM 视角的关键观察:

| 特征 | MOSEK 现状 | ADMM 收益 |
|---|---|---|
| Hessian = diag(λ+tλ) | 内点法每步解大 KKT | x-更新逐坐标解析;整个线性系统 = **对角 + 低秩(W'W, W 为 67 条约束行)**,用 Woodbury 精确求解,无需稀疏分解器 |
| TUB/GUB 行成对出现 (u±x, g±x) | +2n 变量 +4n 行 | 成对外积**相消**,对 A'A 只贡献对角元 → 线性系统保持对角+低秩 |
| 两遍 relax(pass 2 只改界) | 第二遍完全重解 | **warm start**(因子/初值复用) |
| 弹性变量 tₖ | 追加变量+行 | 可折叠进 z-投影(Lew et al. 2025),规模不增 |
| 日内滚动重解 | 每次冷启动+license | 结构不变仅数值变 → 分解/Σ 矩阵复用 |

## 2. 基础文献

- **Boyd, Parikh, Chu, Peleato, Eckstein (2011)**, *Distributed Optimization and Statistical Learning via the ADMM*, FnTML —— 一切 ADMM 工作的基准;收敛性、收敛判据、residual balancing 调 ρ 的原始出处(§3.4.1)。
- **Parikh & Boyd (2014)**, *Proximal Algorithms* —— prox 算子库(软阈值、区间投影),折叠辅助变量后的基本工具。
- **Eckstein & Yao (2012/2015)**, *ADMM for LP and SDP* / *On the O(1/n) convergence...* —— ADMM 与 Douglas-Rachford/PDHG 的等价关系、inexact/随机化推广。

## 3. 求解器与算法前沿(按相关度)

### 3.1 OSQP —— 与本项目同构的标准形式

**Stellato, Banjac, Goulart, Bemporad, Boyd, *OSQP: an operator splitting solver for QP*, Math. Prog. Comp. 12(4), 2020**

算法:`min ½x'Px+q'x s.t. l≤Ax≤u` 的 ADMM 分裂(x/z 两块),每迭代:
x-更新解 `(P+σI+ρA'A)x̃ = σx−q+ρA'(z−y)`(一次分解、迭代回代),
z-更新 `z=Π_[l,u](Ax̃+y)`,y-更新对偶上升;配 **σ 正则化、over-relaxation α(默认1.6)、
自适应 ρ(残差平衡)、Ruiz 对角均衡、polish(末段按主动集解等式 KKT 抬精度)、warm start**。
工程要点全部可移植;对我们额外的红利:由于 TUB/GUB 成对相消 + P 对角,
`P+σI+ρA'A = D + ρW'W`(D 对角、W 仅 67 行),Woodbury 精确求解,**连稀疏分解器都不需要**。

### 3.2 近端增广拉格朗日分支(一阶↔二阶的中间路线)

| 求解器 | 文献 | 核心技术 | 备注 |
|---|---|---|---|
| **QPALM** | Hermans, Themelis, Patrinos, arXiv:2010.02653; MPC 14, 497–541 (2022) | 近端 ALM + 半光滑 Newton 子问题 + 精确线搜(一维分段仿射零点);子问题可强制强凸 → R-线性收敛;active-set 式因子更新 | **论文专门测了组合优化/MPC 类 QP**,解全部 Maros-Meszaros;精度兜底首选 |
| QPPAL | Liang, Li, Sun, Toh, arXiv:2103.13108 | 两阶段:sGS 半近端 ALM 暖启 → 近端 ALM 精解 | 高维凸 QP 对比过 Gurobi/OSQP/QPALM;MATLAB 实现 |
| ciPALM | Yang, Liang, Chu, Toh, arXiv:2311.01976 | 相对误差准则的 inexact 近端 ALM | 单一容差参数,易调参 |
| DPALM | Dahal, Liu, Xu, arXiv:2311.09065 (MPC 2026) | 阻尼对偶步长保对偶有界,弱凸目标 | 非凸扩展 |

### 3.3 PDHG/PDLP 谱系(ADMM 的对偶形式)

| 工作 | 文献 | 要点 |
|---|---|---|
| **PDLP** | Applegate, Díaz, Hinder, Lu, Lubin, O'Donoghue, Schudy, *Practical Large-Scale LP via PDHG*, arXiv:2109.03744 (MPC 2023) | LP 的 solver-grade PDHG:presolve + **对角预处理(Ruiz/Pock-Chambolle 类)** + 自适应步长 + **自适应 restart** + primal weight 平滑 → O(1/k) 变实际线性;"一阶法工程化教科书" |
| **PDHCG-CQP** | Li, Huang, Liu, Ge, Ye, arXiv:2608.09159 (2026) | GPU 锥 QP:restarted averaged PDHG + **reflected-Halpern 加速** + 严格互补下的局部线性收敛证明;4.4e8 变量/8 GPU | 当前研究前沿 |
| PDHG 局部线性收敛 | Jiang, arXiv:2607.08035 (2026) | SDP 上 PDHG 在严格互补/非退化下 R-线性收敛的机理 |
| 双随机 PDHG | Xiao, Liu, arXiv:2605.17883 | 块随机化 + restart 下线性收敛 |

### 3.4 参数、加速与折叠技巧(逐项可抄的"前沿改进")

**(a) 预处理**
- Ruiz 对角均衡(OSQP/PDLP 内置);Pock-Chambolle 对角预条件;Giselsson-Boyd metric selection (JOTA 2017)。
  作用于我们:持仓 ~1e5–1e6、λ~1e-6、权重 0.2 级的**病态尺度**——预计是收益最大的单项改进。

**(b) 自适应 ρ 与残差平衡**
- Boyd et al. 2011 §3.4.1 的 residual balancing;OSQP 的按 primal/dual 残差比在线调 ρ(触发重分解,我们 Woodbury 下 Σ 重算仅 67³/3 ≈ 10⁵ flop,几乎免费);
- 正则化 Barzilai-Borwein 谱步长选 ρ:Xu, arXiv:2503.06185(用于 ℓ1 组合选择)。

**(c) 加速与 restart**
- Goldstein, O'Donoghue, Setzer (2014):Nesterov 型加速 ADMM(经典);
- GLADMM(Zhou, Hou, Cai, Sun, arXiv:2511.17157):Güler 型外推,部分收敛率 O(1/N^{3/2})→O(1/N²);
- **reflected-Halpern 锚定 + restart**(Liu, Cao, Yin, Wen, arXiv:2606.16552):KKT 残差非遍历 O(1/k),主动集识别后线性收敛——解释一阶法为何能到高精度的最新理论;
- PDLP 式 adaptive restart(基于间隙重启)是把 O(1/k) 变线性的关键工程手段。

**(d) 学习型调参(2026 新方向)**
- Lin, Goulart, Furieri, arXiv:2604.26932:学习 **over-relaxation α 的在线策略**,带收敛保证,明确兼容 OSQP 架构且调 α 不触发重分解;
- Prasad & Sharma, arXiv:2606.08638:(cu)PDLP 超参学习的泛化保证。

**(e) 松弛/弹性变量折叠**
- **Lew, Greiff, Subosits, Plancher, arXiv:2511.08451**(Eur. J. Control):带 slack 的 QP 用 ADMM 解而**不增问题规模**——只改 z-投影,等价于标准 ADMM。正对应我们 relax 模式的弹性 tₖ。

**(f) polish**
- OSQP 内置:末段取主动集,解一次等式约束 KKT,把中精度解抬到机器精度附近;失败则回退 ADMM 解。1e-4→1e-8 的最后一段靠它。

### 3.5 GPU/并行方向(规模上去后)

- cuPDLP/CuPy 了;PDHCG-CQP(§3.3);WarpMPC(arXiv:2607.11603,批量 ADMM + 展开 LDL);TurboMPC(arXiv:2606.24039,JAX-CUDA ADMM 内核,58× GPU 加速);GPU-SLS(arXiv:2604.07644,GPU ADMM QP 内核,2e5 变量 20ms)。品种数上万或日内高频滚动时再考虑。

## 4. 组合交易方向的直接同类工作

| 文献 | 内容 | 相关性 |
|---|---|---|
| **Moehle, Gindi, Boyd, Kochenderfer (2021)**, arXiv:2103.05455, *Portfolio Construction as Linearly Constrained Separable Optimization* | 可分目标+仿射约束 ADMM;~1000 证券 + 100 因子 **10–100ms**;支持最小交易量等非凸可分项(启发式);给出性能上界 | ★★★★★ 场景几乎同构;自研定制版的模板 |
| Butler & Kwon (2021), arXiv:2112.07464 | ADMM 可微 QP 层,比 OptNet(IPM)快约一个量级;组合优化示例 | 预测+优化端到端时有用 |
| Bourgeron, Lezmi, Roncalli (2019), robo-advisor 鲁棒配置 | 范数惩罚 MV 的 ADMM 求解 | 框架参考 |
| Xiao, Li, Jiang (2025), arXiv:2510.19614 | UBSR 组合优化的 ADMM + 半光滑 Newton 投影子问题 | 高维组合 ADMM 最新例 |
| Boyd, Moehle, Kohler, et al. (2017), *Multi-Period Trading via Convex Optimization* | 多期交易凸优化框架 | 背景 |

## 5. 候选方案对比总表

| | OSQP | QPALM | QPPAL | PDLP/PDHCG | 自研定制 ADMM |
|---|---|---|---|---|---|
| 算法 | ADMM+自适应ρ+Ruiz+polish | 近端 ALM+半光滑 Newton | sGS 两阶段近端 ALM | restarted PDHG | 按结构定制(见§1) |
| 单次延迟(千变量稀疏) | 最快档 | 快 | 快 | LP/锥导向 | 预期最快(无通用开销) |
| 精度 | polish 后高 | 高 | 高 | 中 | +polish 后高 |
| license | Apache 2.0 | 开源 | MATLAB | Apache 2.0 | — |
| Go 接入 | cgo 自封装/代码生成 | cgo | 不便 | cgo | **纯 Go 原生** |
| 维护成本 | 低 | 低 | — | 低 | 中(自担) |

**决策:自研纯 Go 基础款(OSQP 算法 + 我们特有的对角+Woodbury 线性求解),后续按 §3.4 逐项加改进;
OSQP/QPALM 作为对拍基准与兜底备选。**

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 病态尺度(持仓 1e5–1e6 vs λ 1e-6)收敛慢 | Ruiz 均衡 + 自适应 ρ(§3.4a/b) |
| 高精度段收敛慢 | polish(§3.4f);strong convexity 来自 diag(λ+tλ)>0,主动集识别后理论线性收敛 |
| soft-λ>0 引入稠密低秩 Q | 默认 0 不受影响;将来用辅助变量线性化(等价于已有 TUB 块) |
| ADMM 不可行证书弱 | pass-1/2 切换本就由数据侧 `constraintInBreachWithZeroTraded` 决定,不依赖求解器证书 |
| relax 模式弹性变量 | 先按 MOSEK 原样建模(验证等价),再按 §3.4e 折叠 |

## 7. 落地计划(对应 work.md)

1. `golang_mosek/src/admm/`:基础款 OSQP 式 ADMM(固定 ρ、α=1、无均衡化),线性系统用
   对角 D + 67 行 W 的 Woodbury 精确解;`opt.SolveADMM` 复用 STEP1 预处理与两遍 relax 编排;
   `--engine admm` 切换。跑 optim 数据,与 MOSEK 对拍目标值/逐品种 trades/用时 → `log/admm/basic.md`。
2. 逐项改进并迭代测试:① 自适应 ρ ② over-relaxation α ③ Ruiz 均衡 ④ polish
   ⑤ warm start(两遍 relax/滚动重解)⑥ 折叠弹性变量 ⑦ reflected-Halpern/restart(如仍有需要)
   → `log/admm/improving.md`。
3. 终态(可选):Moehle-Boyd 式定制(删 TUB/GUB,prox 直接吃 cost|x| 与 gross 投影)。

---

## 参考文献一览(按主题)

**基础**
1. Boyd, Parikh, Chu, Peleato, Eckstein. *Distributed Optimization and Statistical Learning via the ADMM*. FnTML 3(1), 2011.
2. Parikh, Boyd. *Proximal Algorithms*. FnT Optimization 1(3), 2014.
3. Eckstein, Yao. *Understanding the Convergence of the Alternating Direction Method of Multipliers: Theoretical and Computational Perspectives*. Pac. J. Optim., 2015.

**求解器**
4. Stellato, Banjac, Goulart, Bemporad, Boyd. *OSQP: an operator splitting solver for quadratic programs*. Math. Prog. Comp. 12, 2020.
5. Hermans, Themelis, Patrinos. *QPALM: A proximal augmented Lagrangian method for nonconvex QPs*. arXiv:2010.02653; MPC 14, 2022.
6. Liang, Li, Sun, Toh. *QPPAL: a two-phase proximal ALM for high dimensional convex QP*. arXiv:2103.13108.
7. Applegate et al. *Practical Large-Scale LP via Primal-Dual Hybrid Gradient*. arXiv:2109.03744; MPC 15, 2023.
8. Li, Huang, Liu, Ge, Ye. *GPU-Accelerated Conic QP with Local Linear Convergence under Strict Complementarity (PDHCG-CQP)*. arXiv:2608.09159, 2026.

**改进技巧**
9. Goldstein, O'Donoghue, Setzer. *Fast alternating direction optimization methods*. SIAM J. Imaging Sci., 2014.
10. Lew, Greiff, Subosits, Plancher. *Solving QPs with slack variables via ADMM without increasing the problem size*. arXiv:2511.08451; Eur. J. Control, 2026.
11. Lin, Goulart, Furieri. *Learning Over-Relaxation Policies for ADMM with Convergence Guarantees*. arXiv:2604.26932, 2026.
12. Liu, Cao, Yin, Wen. *Restarted Reflected Halpern Acceleration for Augmented Primal-Dual Methods*. arXiv:2606.16552, 2026.
13. Zhou, Hou, Cai, Sun. *Güler-type acceleration for proximal gradient, linearized ALM and linearized ADMM (GLADMM)*. arXiv:2511.17157, 2025.
14. Xu. *An adaptive ADMM with regularized spectral penalty for sparse portfolio selection*. arXiv:2503.06185, 2025.
15. Prasad, Sharma. *Parameter Tuning with Generalization Guarantees for GPU-Accelerated LP*. arXiv:2606.08638, 2026.

**组合交易应用**
16. Moehle, Gindi, Boyd, Kochenderfer. *Portfolio Construction as Linearly Constrained Separable Optimization*. arXiv:2103.05455, 2021.
17. Butler, Kwon. *Efficient differentiable QP layers: an ADMM approach*. arXiv:2112.07464, 2021.
18. Bourgeron, Lezmi, Roncalli. *Robust Asset Allocation for Robo-Advisors*. 2019.
19. Xiao, Li, Jiang. *ADMM for Utility-based Shortfall Risk Portfolio Optimization*. arXiv:2510.19614, 2025.
20. Boyd, Moehle, Kohler, et al. *Multi-Period Trading via Convex Optimization*. Found. Trends Optim., 2017.

**GPU/并行(展望)**
21. Hose et al. *WarpMPC: Large-Batch MPC on GPU via ADMM with Unrolled LDLᵀ*. arXiv:2607.11603, 2026.
22. Bravo-Palacios et al. *TurboMPC: Fast, Scalable, Differentiable MPC on GPU*. arXiv:2606.24039, 2026.
23. Fang, Chou. *GPU-SLS: Reachability-Constrained SLS on GPU*. arXiv:2604.07644, 2026.
