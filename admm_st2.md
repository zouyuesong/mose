# ADMM Stage 2 调研与方向规划(admm_st2)

- 日期:2026-08-21;承接 `admm.md`(Stage 1 调研)与 `summary_stage1_0821.md`(Stage 1 全部实验结论)
- 任务:①基于 Stage 1 结果判断剩余优化潜力在哪;②针对高潜力方向深挖调研;③收录 Stage 1 未覆盖的新方向

---

## 1. 现状盘点:钱已经花在哪,还剩在哪

### 1.1 Stage 1 收益来源分解(全部实测)

| 手段 | 量级 | 状态 |
|---|---|---|
| 行列均衡(scaleRows/optim_scaled) | osqp 25× / qpalm 9× / proxqp 87× / scs 不收敛→可用 | **已吃满**(这是问题特定变换,非库技巧) |
| 库内置均衡开关(qpalm 开 osqp 关) | 10–15% | 已吃满 |
| qpalm sigma_init=1 | −18%(33.3ms 全场最优) | 已吃满 |
| harness polish(KKT+AMD-LU) | 一致性 5000×(166→0.02);osqp 松容差通道 153→107ms | 已交付 |
| 容差放宽(均衡后) | qpalm ×10⁴=38ms/0.010%;osqp 1e-2=14.7ms/1.85% | 已吃满(无均衡时全部证伪) |
| adaptive-ρ/α/ρ 扫描 | osqp 生死项但已默认开;其余钝感 | 已穷尽 |

### 1.2 当前时间花在哪(关键:量化的剩余空间)

以 osqp 均衡版 107ms(1e-5+polish)为解剖对象,一次 solve 的构成:

| 成本项 | 量 | 说明 |
|---|---|---|
| **模型规模税** | 1074 变量(x/u/g)、1782 行 | TUB/GUB 线性化使变量 ×3、行 ×5(67 条真实约束外全是结构行);**u/g 可折进 prox 完全消除**(§2.1) |
| 每迭代线性求解 | QDLDL 稀疏分解回代 | (P+σI+ρA'A) 1074²、12.5k nnz;**我们结构下 = 对角 D + 67 行 W'W,Woodbury 精确解只需 67³/3≈1e5 flop**(admm.md §1 预言,QDLDL 不知道结构) |
| 迭代数 | 1175–2000(均衡后) | 一阶法地板;restart/主动集识别后线性收敛是唯一理论出路(§2.3) |
| setup(含在 totalMs) | ~20–30ms(loadMs 26ms + 库 setup) | **codegen 可消除**(§2.4) |
| polish | 6–8ms | 已 AMD 优化;Schur 判死 |

### 1.3 潜力排序结论

| 方向 | 预估收益 | 成本 | 判据 |
|---|---|---|---|
| **A. cpp 无库定制版**(结构折叠 + Woodbury) | **10–50ms**(对标 Moehle-Boyd:358 品种 < 其 1000 品种 10–100ms) | 高(2–4 周) | 每迭代成本降 3–5×(变量 1074→358、行 1782→67+投影),迭代数持平或略增即净赚 |
| **B. warm start / 滚动重解** | 重解场景 2–5×(迭代 1725→数百) | **极低**(vendored OSQP v1.0 已有 `osqp_warm_start`+`osqp_update_data_vec`,已 grep 确认) | 生产日内场景直接适用;单日快照数据可用 alpha 扰动模拟 |
| C. OSQP codegen(embedded) | 消 setup ~20–30ms | 低 | 官方特性,无内存管理器 |
| D. restart/reflected-Halpern | 主动集识别后线性收敛(迭代尾部砍半+) | 中(需自有求解器;fork OSQP 已判不做) | 理论已备(2606.16552),实现挂 A |
| E. 在线学习 α/ρ | ≤10%(设置层地形平坦) | 中 | headroom 存疑,Stage 1 已判缓 |

**判论:A 是唯一通往 10ms 的路;B 是生产价值最高、成本最低的快赢;C 顺手;D 挂 A 内部。**

---

## 2. 方向深挖(针对 §1.3 的 A/B/C/D)

### 2.1 Moehle-Boyd(arXiv 2103.05455)全文细节提取——定制版蓝图

Stage 1 只引了摘要;本次提取实现级细节(子代理精读 v2 全文):

**算法结构(SAP 形式)**:`min Σf_i(x_i) s.t. Ax=b`——**A 只含 k+1 行**
(因子暴露 + 现金),**所有不等式(仓位箱、最小交易量、整数股、card 成本)
全部用 +∞ 折进可分函数 dom f_i**;ADMM 分裂:x-更新=逐坐标 prox(闭式,
分段二次 prox 按 §6.2 系数平移法),z-更新=向仿射子空间投影(解
`[[I,A'],[A,0]][z;ν]=[x−λ;b]`,**分解一次缓存**,引 OSQP 论文 §3.1)。

**实测数字**:S&P 500(持仓 200–300 只),**692 个真实回测问题全部收敛**,
非凸版均值 **251ms**±159、凸松弛版 **152ms**±67;能力声明"~1000 证券 +
100 因子 10–100ms"。

**缩放**:两条路线——①**4 参数对角超参搜索**(他们的实验用
`D=diag(100·1, 3, 100·1), E=100·1`,纯手工);②equilibration(引 OSQP
即 Ruiz)。**我们 Stage 1 的 band/σ 均衡就是路线 ② 的问题特定版**,且已
实测 25–87×——定制版直接继承 scaleRows 作为初始化,优于他们的手工 4 参数。

**终止**:每 10 步检查;`r(x)=dist(x,dom f)<3e-4` 且目标 N=50 步无
>1e-5 改进才停——**非凸安全**的停机设计,值得抄到定制版(我们的凸情形
可收紧)。

**对我们的关键差异**:他们 A 只有等式(k+1 行);我们的 66 条 NET 是
双边不等式、GROSS 非可分——**不能折进 dom f**。定制版正确形态是
OSQP 式分裂(Az∈[l,u] 的 z-更新=逐行区间裁剪)+ Moehle-Boyd 式
x-更新(prox 吃掉 quadratic+cost|x|+箱,闭式):
```
x-update: prox_{f_i/ρ}(·) —— 358 个坐标,每个 O(1)(分段二次+软阈值+clip 三合一闭式)
z-update: 67 行区间 clip + GROSS 行(单条 Σw·g≤ub 的 waterfill 投影,O(n log n) 一次排序)
线性求解:Woodbury (D+ρW'W),W=67 行 —— 67³/3 分解每次 ρ 变更时 ~1e5 flop
```
即:**u/g 变量与 4n 条 TUB/GUB 行全部消失**,模型从 1074 变量/1782 行
回到 358 变量/67 行;GROSS 用 g-epigraph 只留在 z 空间(waterfill 一次)。
这正是 admm.md §7-3"终态"的精确化。

### 2.2 warm start / 滚动重解(方向 B,Stage 1 完全未测)

- **API 已就位**(本次 grep vendored 头文件确认):`osqp_warm_start(solver,x,y)` +
  `osqp_update_data_vec`(改 q/l/u **不触发重分解**)——OSQP v1.0 官方定位就是
  "matrix factorization can be cached to solve parametrized problems extremely
  efficiently"
- **生产场景匹配**:日内滚动重解只变 alpha/position/bounds(结构不变)——
  正是 update_data_vec 的教科书用例;两遍 relax 的 pass-2 也是同结构热启
  (Go 版 Settings.X0/Z0/Y0 支持但编排未接,improving.md §5-2 遗留)
- **量化预期**:冷启 1725 迭代;热启(解漂移 ~1% 带宽)通常降到数百迭代
  → 107ms → **30–50ms**;与均衡、polish 全部叠加正交
- **实验设计**(无需新数据):以 optim 解为 t0,扰动 alpha ±5%/±20% 生成
  t1..t10 的伪滚动序列,测 warm vs cold 的迭代/时间曲线;qpalm 亦有
  `warm_start` 设置可同步测
- 附带:C 的 codegen 同理消 setup(26ms loadMs + 库内 setup),两者合计
  可把单次 wall 从 ~135ms 压到 ~80ms(不换求解器)

### 2.3 restart / reflected-Halpern(方向 D,挂在 A 内)

arXiv 2606.16552(Liu-Cao-Yin-Wen,admm.md §3.4c 已列)本次精读摘要要点:
- 统一增广 primal-dual 框架(PDHG 型/增广 Chambolle-Pock 型度量),精确式
  有 degenerate proximal-point 表示、线性化式有预条件 forward-backward 表示
  ——**反射 Halpern 加速可直接在 primal-dual 变量上分析**
- **KKT 残差与目标间隙的非遍历 O(1/k)**(shadow iterates);"finite
  identification belongs to the shadow sequence rather than to the anchored
  state"——**实现要点:主动集识别要在未加速的影子序列上做,不在锚定态上**
- 识别后:affine-face 模型 + 局部 sharpness → **restart 锚点线性收敛**
- 对照 Stage 1:我们 polish 的乘子判据正是"识别后的 affine-face 精修";
  restart 是把它变成**迭代内**机制(边跑边锚),预期砍掉 1725 迭代的尾段
- **同组新文**(本次新发现):arXiv 2608.19847 *Fixed-Penalty Linearized
  Augmented Lagrangian with Classical Multiplier Updates*(Liu-Deng-Wang-Wen,
  2026-08)——线性化 ALM+经典乘子更新,正是定制版的算法骨架候选;建议
  作为 A 的实现参考文献(全文未读,列入 A 的开工阅读)

### 2.4 OSQP v1.0 codegen(方向 C)

官方文档(本次核对):codegen 生成"embeddable C code with no memory manager
required"——setup 的内存分配/符号分解全部前移到编译期;对 100ms 级
solve 是 ~20% wall 的直接节省。vendored 版已带(头文件 OSQP_EMBEDDED_MODE
路径确认)。风险:codegen 面向固定问题结构,我们的 relax 两遍与滚动更新
恰好也是固定结构,匹配。

---

### 2.5 NR-LALM 精读判词(arXiv 2608.19847,2026-08-28 补)

全文精读(HTML 版)+ 第四族对照引擎 `nativeralm` 实测(报告
report_st2.md §7)后的结论:

- **约束线性 ⇒ 论文核心机器恒为零**:线性化误差 d_k≡0,SOC 校正
  精确退化为恒等(勿实现),§3 随机 restart 是理论定位装置,与
  确定性凸 QP 无关;不等式处理论文明言 "remain open";
  m=1782>n=1074 使 uniform LICQ 结构性不成立
- **引擎实测负结果**:忠实版(P 折出 M,843k iter)与 pinP 变体
  (P 进 M,85.5k iter)被 native(1775)支配 48-475×;ρ 对忠实版
  完全无影响(β vs ‖P‖≈600 决定一切)
- **可搬三件套**:① Φ̂=L_ρ+ρ‖r‖² 免常数 Lyapunov 监控;
  ② argmin 残差输出规则(Theorem 2.12,救松容差尾巴);
  ③ "取 ℛ 最小迭代点"与 ρ/β 扫描方法论(其区间 [6,64] 建立在
  AA'=I 上,对本问题无意义)

## 3. 新方向(Stage 1 视野外,本次调研新增)

### 3.1 内部质量证书:凸包下界替代 MOSEK 外部裁判 ★新

Moehle-Boyd §4:把可分函数换成**凸包 f**,**解松弛问题得 d⋆≤p⋆**;
他们的 692 实例中 ADMM 解与下界差 **0–10bp(均值 0.6bp)**。
**对我们的意义**:Stage 1 全部精度裁判依赖外部参考 6580.2285(单日、
MOSEK 专有);定制版可**自带下界**(我们的 cost|x| 凸包=自身,箱凸包=
自身,唯一非凸是……我们问题本就凸——但 relax 模式/低容差停机点仍可用
对偶间隙做内部证书,免 MOSEK 依赖)。落地形态:定制版输出
`obj ± gap_certified`,生产可自动判"解可信度"。

### 3.2 Lew 等 slack 折叠 v3(arXiv 2511.08451,2026-07 更新)

Stage 1 记过 v1;v3 要点不变:带 slack 的 QP 用 ADMM 解而**不增规模**
——只改 z-投影。对应我们 relax 模式的弹性 t_k:**定制版 relax 档不需要
为弹性变量加行**,折进 z-投影即可(与 §2.1 的 GROSS waterfill 同一层)。

### 3.3 滚动场景的"求解器即状态机"(B 的延伸)

OSQP v1.0 的 update_data_vec + warm_start + codegen 三者组合的含义:
**求解器实例可常驻进程**(因子缓存 + 双热启),单次重解只剩"纯迭代"成本。
这改变延迟预算的口径:生产关心的不再是 cold 33ms(qpalm)而是 warm ~10ms
量级——**B 的实验结论会直接改写排行榜的口径**,值得单独出报告。

### 3.4 已扫排的方向(维持 Stage 1 结论)

- GPU/并行(1074 变量负收益);(d) 在线学习(地形平坦);(e) 弹性折叠
  (单日数据 relax 未触发,但定制版按 §3.2 内建);Schur 消元 polish
  (u/g 小 δ 数值判死);fork OSQP 内循环(被 qpalm 支配)。

---

## 4. 路线图与执行状态(2026-08-21 更新)

| 步骤 | 内容 | 状态 | 结果 |
|---|---|---|---|
| S2-0 | osqp warm-start 滚动(update_data_vec+warm_start,alpha 扰动伪滚动) | ✅ 完成 | **5% 漂移 2.2×**(331→150ms)、20% 1.7×;低于预估 2–5×;OSQP 默认 ρ 按时间触发(勘误);qpalm 无数据更新 API |
| S2-1 | native 无库引擎(OSQP v1.0 精确移植 + Woodbury + scaleRows) | ✅ 完成 | **冷解 80ms**(vs osqp library 144ms=1.8×),迭代 1775≈1725,**松容差 6.7ms 进 10ms**;kill 线通过;详见 cpp_admm/report_st2.md |
| S2-1' | codegen 试编 | 未做 | native 引擎零 setup 开销,codegen 的边际收益已被吸收,降级 |
| S2-2 | native + restart(影子序列) | ✅ 完成(2026-08-28) | **kill 线处决**:调优基映射上全劣化、纯锚定 44 iter 爆炸、γ≤0.25 理论边界实测吻合;意外收获 = 固定 ρ=300 针尖(68ms,非稳健)+ 悬崖 obj 侧缓解;详见 report_st2.md §8 |
| S2-2' | nativeralm 第四族(NR-LALM 对照) | ✅ 完成(2026-08-28) | **负结果**:忠实版 843k / pinP 85.5k iter,被 native(1775)支配 48-475×;相图+三件套见 report_st2.md §7 |
| S2-3 | relax 折叠 + 内部证书(滚动常驻已按用户决定裁掉) | ✅ 完成(2026-08-28) | **证书 gapCert 交付**(三态语义:小正/大正/负,表示差与求解误差定量分离);**折叠 = 救场技术**(显式 native 不收敛 + qpalm 段错误 → 折叠 89ms,KKT 自洽验证通过);边界发现:纯 ADMM 对生产弹性罚结构性不可用(对偶爬升 1e18 iter);详见 report_st2.md §9-10 |

依赖与风险:S2-1 是主投入,唯一重大风险是**折掉 u/g 后迭代数上升**
(约束从显式行变隐式 prox,收敛界变差)——Moehle-Boyd 同结构实测给了
反例支撑(251ms@500 品种含非凸),且我们比他们少了因子协方差块;
kill criteria 已写明。S2-0 无风险先行。

---

## 5. 参考文献(本次新增/深读)

1. Moehle, Gindi, Boyd, Kochenderfer. *Portfolio Construction as Linearly
   Constrained Separable Optimization*. arXiv:2103.05455v2.(全文精读:
   结构/数字/缩放/终止细节见 §2.1)
2. Liu, Cao, Yin, Wen. *Restarted Reflected Halpern Acceleration for Augmented
   Primal-Dual Methods*. arXiv:2606.16552.(影子序列识别 + restart 线性收敛)
3. Liu, Deng, Wang, Wen. *A Fixed-Penalty Linearized Augmented Lagrangian
   Method with Classical Multiplier Updates*. arXiv:2608.19847(2026-08).
   (全文精读 + nativeralm 引擎实测:**负结果判词**见 §2.5 / report_st2.md §7;
   可搬三件套:Φ̂ 监控、argmin 残差输出、扫描方法论)
4. Lew, Greiff, Subosits, Plancher. *Solving QPs with Slack Variables via
   ADMM without Increasing the Problem Size*. arXiv:2511.08451v3(2026-07
   更新;relax 弹性折叠)
5. OSQP v1.0 官方文档(codegen / warm start / update_data_vec;vendored
   头文件 API 已逐一确认存在)