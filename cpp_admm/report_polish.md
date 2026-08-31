# admm.md §3.4(f) polish 实验:ALM 族机制澄清 + harness 级 KKT/IR 精修

- 日期:2026-08-19
- 问题:polish(末段主动集精修)在四库上的现状、可移植性与增量。
- 数据 `/tmp/mose_run/f/`,脚本 `/tmp/mose_run/f_sweep.sh`;
  gap% 相对 MOSEK 6580.2285;pos diff 单位=名义值。

---

## 1. 前提澄清:ALM 两库没有"polish"

| 库 | 机制 | 是否可移植为事后 polish |
|---|---|---|
| qpalm | 无独立 polish;精度来自**半光滑 Newton 内环**本身;`reset_newton_iter`/`max_rank_update` 是过程性因子管理 | ✗ 不可(是求解过程,不是步骤) |
| proxqp | 无独立 polish;内环自带 `nb_iterative_refinement`(迭代精修 IR) | ✓ **IR 思想可移植**(已实现) |
| osqp | 内置 polish(`polishing=1`,我们一直开着) | ✓(对照基准) |
| scs | 完全没有 | —(受益方) |

**Phase 0 诊断(新暴露 EngineResult.polishStatus)**:osqp 内置 polish 在本问题
**全容差梯阶 1e-8→1e-2 全部静默失败**(libPolish=−1)——从 baseline 起开着的
这功能从未生效过,其"单步 KKT + 距离判据"被本问题结构(双边行多、停机点
中间态)击穿。

## 2. 新增实现(harness 级,引擎无关,`--polish=off|kkt|ir`)

`polishKKT`(model.cpp,~100 行),作用于**均衡坐标系**(行 O(1),tol 语义干净):

1. **主动集判据用乘子**(关键改进,EngineResult 新增 y 返回):`y_r` 符号定
   哪一侧、非零定是否激活 + 宽松贴界门(1e-3 相对)——纯距离判据(内置
   polish 用的)对双边行必错;
2. 等式 KKT `[[P+δI, A_act'],[A_act, −εI]]` 稀疏 LU 一次求解(δ=1e-6 处理
   u/g/t 零对角;**−εI=1e-10 处理线性相关主动行**——delta_ptf/delta_grp3
   重复行、TUB/GUB 折点成对行会让无正则 KKT 奇异,LU 直接 fail);
3. `ir` 档 = 同一因子上 3 轮迭代精修(proxqp IR 思想);
4. 校验:全行可行 + 目标不劣,失败**回退原解**(免费期权语义)。

## 3. 主矩阵(eps 梯阶 × polish;solve 含 polish 耗时)

### osqp(`--scale-rows --lib-scaling=off`)

| eps_rel | off | kkt/ir(两者结果相同) |
|---|---|---|
| 1e-8 | 153ms / 6579.7541 | **148ms / 6579.7582 ✓** |
| 1e-7 | 196 / 6579.7524 | 195 / 6579.7582 ✓ |
| 1e-6 | 121 / 6579.6956 | 125 / 6579.7582 ✓ |
| 1e-5 | 151 / 6579.1965 | **102ms / 6579.7582 ✓** |
| 1e-4~1e-2 | 92/22/20ms(松容差彩票) | ✗ 全部拒绝(回退=off) |

### qpalm(`--scale-rows --lib-scaling=on --sigma=1`)

| eps_rel | off | kkt/ir |
|---|---|---|
| 1e-8 | 34ms / 6579.7549 | 45.6ms / **6579.7581 ✓** |
| 1e-6 | 34 / 6579.7549 | 45.6 / 6579.7581 ✓ |
| 1e-4 | 31 / 6579.6342 | ✗ |
| 1e-3 / 1e-2 | 26 / 19 | ✗ |

### proxqp(`--scale-rows`)

| eps_rel | off | kkt/ir |
|---|---|---|
| 1e-8 | 85ms / 6579.7571 | 80ms / 6579.7582 ✓ |
| 1e-6 及更松 | 68 / 6579.7525… | ✗(乘子在 1e-6 已不可靠) |

### scs(`--scale-rows`)

| eps | off | kkt/ir |
|---|---|---|
| 1e-6 | 5.99s / 6579.757 | 6.00s / **6579.7582 ✓**(pos 同步修复,见 §4) |
| 1e-5 | 4.55s / 6579.749 | 4.47s / 6579.7581 ✓ |
| 1e-4 | 3.00s / 6579.081 | ✗ |

### 截断+polish(Go 版"中精度点"策略模拟,`--max-iter --no-relax`)

500/800/1000 iter 点:polish ✗;1200 点:polish ✓ 但状态"maximum iterations
reached"被编排层拒收;1500(solved inaccurate):✓ 6579.7582。**甜点窗口
窄(~1200–1725),收益 ≤30% 迭代,不如松容差通道**(1e-5+kkt 102ms)。
单步 polish 需要"停机点足够收敛"或主动集修正循环(= Go polishADMM 的
加固,明确不在本期范围;未来 cpp 无库版可自带)。

## 4. 核心发现:polish 把跨引擎分歧压缩 ~5000×(0.02–0.03)

所有 polish 成功点(osqp@1e-5、qpalm@1e-8、proxqp@1e-8、scs@1e-6)的解
两两收敛到**同一顶点的 3–4 位有效数字**:

| 对比 | pos absMax / p90 / mean |
|---|---|
| osqp@1e-5-kkt vs qpalm@1e-8-kkt | **0.021 / — / —** |
| qpalm@kkt vs proxqp@kkt | **0.031 / — / —** |
| qpalm@kkt vs scs@kkt | **0.031 / — / —** |
| (对照)未 polish:osqp vs qpalm off | 3.6(族内) |
| 未 polish 跨族 vs MOSEK | 166.6 |
| 全家族 polished vs MOSEK | 166.8(仍为平坦面地板) |

机理:各引擎停机点的**主动集在边缘行上有差异**(k = 1163~1199,乘子
±1e-8 阈值附近的边际行不同),故 KKT 精修解在同族顶点的邻域内,残余
0.02–0.03(名义值,~1e-4 band)。

**勘误(2026-08-19 复核)**:本节初版曾报"逐位一致 absMax=0.0",系当时
比对脚本的错误结论(与同时记录的各引擎 maxObj 存在 3e-5 级差异自相矛
盾,未察觉);复算历史文件确认真实差异即上表 0.02–0.03。结论强度从
"逐位"降级为"~5000× 收敛",定性含义不变:**跨 ADMM 族引擎的持仓可复现
性由 polish 免费拉近到 0.03 量级**;与 MOSEK 的 166.8 属两族最优面表示
差(IPM 解析中心 vs KKT 顶点),不可由 polish 消除。

## 5. ir vs kkt;polish 的两条求解路径

- ir(3 轮 IR)与 kkt 在所有成功点给出相同解与目标——KKT 残差经一次
  稀疏 LU 已到机器精度,IR 无增量(它防的是 δ/ε 正则化误差,本问题量级
  1e-10 不触发)。**结论:kkt 够用,ir 作为保险保留**。
- **排序很关键**:COLAMD 在此 KKT 蛇形结构上 lu.compute 要 10.3ms,换
  **AMD 排序降到 5.8ms**(整体 polish 11–13ms → **6.2–7.8ms**)。
- **Schur 消元路径(失败实验,保留在 `POLISH_SOLVER=schur` 供查)**:
  消去对角 D 解 SPD Schur 系统 S=A D⁻¹A'+εI 的方案在本结构上**数值不
  可行**——u/g 辅助变量 P=0、D=δ=1e-6,使 q_u/D_u~1e8 进入 RHS,浮点
  噪声经乘子空间放大回 x(实测 TUB 行违反 187;经紧行反解 u/g 后仍有
  GUB 违反 ~0.5,源于 x 本身被污染)。教训:**凡 P=0 辅助变量的小 δ
  正则化,消元法都不敌带主元的 LU**。

## 6. 更新后的推荐配置与排行榜(solve 含 polish,gap 全 0.0071%)

| 引擎 | 配置 | solve(含 polish) | 变化 |
|---|---|---|---|
| **qpalm** | `--scale-rows --lib-scaling=on --sigma=1`(无 polish) | **33.3ms** | 不变(一致性时 +kkt = **39.6ms**,AMD 后从 45.6 降) |
| qpalm | 同上 `--eps-rel=1e-6 --polish=kkt` | 39.6ms(polish 6.3ms) | 一致性版 |
| proxqp | `--scale-rows --polish=kkt` | 74ms(polish 7.3ms) | 一致 |
| **osqp** | `--scale-rows --lib-scaling=off --eps-rel=1e-5 --polish=kkt` | **107ms**(polish 7.8ms) | 比 COLAMD 版 114.6ms(total)再降 |
| scs | `--scale-rows --eps=1e-5 --polish=kkt` | 4.35s(polish 6.2ms) | 可用+一致 |

## 7. 复现

```bash
cd /home/logic/code/mose/cpp_admm
# osqp 新最优:松容差快跑 + kkt 精修
./build/csvport_cpp opt -i ../optim_inputs/optim_inputs.20260520 \
  -c ../optim_inputs/optim_constraints.20260520 \
  -w ../optim_inputs/optim_weights.20260520 -o /tmp/o.csv \
  --engine=osqp --scale-rows --lib-scaling=off --eps-rel=1e-5 --polish=kkt
# 一致性验证:换 --engine=qpalm --lib-scaling=on --sigma=1 --polish=kkt,
# diff 两个 csv 应逐位相同
bash /tmp/mose_run/f_sweep.sh
```

## 7.5 自审查修复记录(2026-08-21,B1/B2 及连环 bug)

自审查发现 proxqp/scs 的乘子从未真正接入(y 提取错误),修复过程又揪出
三个连环 bug,全部已修并复测:

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| B1 | proxqp `results.y` 长度=0(等式乘子),不等式乘子在 `results.z` | `QP(n,0,m)` 无等式块 | 改读 z;符号对拍 vs osqp:1740 同/0 反,同号无需取负 |
| B2 | scs 无 y 提取(拆行使 m 漂移) | ranged 行拆成两条非负锥行 | 记录 (srcRow,±1) 来源,解后合并 `y=y_u−y_l`(OSQP 约定);对拍 1574 同/0 反 |
| C1 | **不可行解被接受**(proxqp 1e-3 给 3729 却 hPolish=1) | violator-dump 重构时 `if (!feas && dbg)` 把 `return false` 圈进 debug 分支——无 DEBUG 时不可行漏过,继续比目标 | `if (!feas) { dump…; return false; }` |
| C2 | verify 基准与输出指标口径不一 | 输入停机点的 u 松弛(>\|x\|)使 objMin 基准虚高,cand 搭便车 | `polishRecoverAux` 同时应用于输入解(xref)与 cand,统一到真实 cost\|x\| 语义 |
| C3 | "成功但输出=输入" | C2 修复时 polishVerify 覆盖的是局部拷贝 xref,caller 的 x 未更新 | 成功后显式 `x = cand` |

复测结论(f4/ 目录):错误解全部正确回退;**proxqp 1e-6 从失败变成功**
(B1 红利:z 乘子在 1e-6 仍可靠);一致性维持同顶点量级
(osqp↔qpalm 0.021 / qpalm↔proxqp **0.009** / qpalm↔scs 0.046);
osqp 梯阶 1e-8~1e-5 全成功、1e-4 起回退(不变);排行榜数字不变。
教训(写给 cpp 无库版):polish 的验收必须 ①可行性短路严格无条件
②基准与输出同口径(辅助变量恢复)③成功路径写回 caller——三者在重构
中各断过一次。

## 8. 备注(遗留与修复记录)

- ~~solveMs 记账 bug(polish 耗时被覆盖)~~ **已修**:solveMs 现含 polish,
  并单列 `polishMs=` 字段。
- "maximum iterations reached + polish 成功"目前被编排层判失败(no-trade
  兜底)——若未来采用截断策略需放宽该判定;
- osqp 内置 polish 建议直接关(从未成功,白付 ~0ms 但状态误导):
  `--osqp-polish=off` 与 kkt 可同开(互不影响);
- `POLISH_DEBUG=1` 打印主动集规模/残差/violator;`POLISH_SOLVER=lu|schur`
  切换求解路径(默认 lu+AMD;schur 为失败实验存档)。