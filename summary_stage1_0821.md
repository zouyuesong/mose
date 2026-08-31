# Stage 1 研究总结(2026-08-17 ~ 08-21)

- 范围:cpp_admm 四库 baseline → 均衡发现 → 数据级预处理 → admm.md §3.4 改进族
  (a)(b)(c)(f) 实验 → 精度/可复现性分析 → 文献对照。
- 数据集:optim(358 品种,67 约束,23628 权重行;**position/trade 单位=名义值/日元**,
  非股数);参考解 = MOSEK 9.3.22 目标 6580.2285。
- 本文件是总览;细节见 §9 文件索引的六份报告。

---

## 0. TL;DR

| 结论 | 数字 |
|---|---|
| 均衡(行列预处理)是本问题**唯一数量级级**的加速手段 | osqp **25×**、qpalm 9×、proxqp **87×**、scs 不收敛→可用 |
| 均衡与算法无关,可前移为**纯数据预处理**(optim_scaled/) | 数据版 ≡ 引擎内开关版(逐笔差 0.0007–0.014) |
| MOSEK(IPM)对均衡**无感** | 0.54s vs 0.54s;理论=实践互相印证 |
| 无均衡时放容差换不到速度 | 三条放宽路径全部"平坦→悬崖",1e-3 崩 122% |
| 设置层剩余增量:qpalm `--sigma=1`(**33.3ms** 全场最快) | −18%;osqp 关内置 Ruiz −15% |
| harness 级 KKT polish:osqp 内置 polish **一直在静默失败**,我们的乘子判据版修好 | polish 6–8ms;跨引擎分歧 166.6→**0.02–0.03** |
| 跨引擎逐位/绝对 1e-3 持仓一致性**不可达**(平坦最优面) | rel-band<1e-3 已天然满足(8.7e-4) |

## 1. 问题结构与病根:三个尺度跨度(乘性叠加)

| # | 跨度 | 实测 | 后果 |
|---|---|---|---|
| ① | 跨品种有效交易带 xl/xu(=交易界∩仓位界,**非**原始 trade 界) | band 1~7.5e5,314 个不同值 | 逐品种结构行失衡 |
| ② | 跨行约束目标 | 66 条 NET 全 1e5 vs GROSS 5e7(**500×**) | eps_rel 锚被劫持:1e-3 时全局容差 5e4=delta 行自身尺度的一半 |
| ③ | 目标曲率 vs 界 | P 对角 1.7e-9 vs 界 1e5~5e7,KKT 跨度 **3e16** | osqp/qpalm 误报 primal/dual infeasible |

只治其一不行:仅行均衡→误报仍在;仅列均衡→容差悬崖仍在。这是 scaleRows 必须"列+行"同做的根因。

## 2. 均衡 = 数据预处理(核心交付)

**变换**(make_optim_scaled.py,逐项可对账):
- 列均衡(逐品种):x=b·x′,b=band → pos/minTrade/maxTrade/minPosition/maxPosition ÷b;alpha/cost ×b;lambda/tlambda ×b²(曲率吃带宽平方)
- 行均衡(逐约束):整行 ÷σ(σ=max(1,|lb|,|ub|);六类行各有定义,数据级只需 CSV 里两类)
- 交汇点:weight′ = weight × band_sym ÷ σ_constr(23628 条目逐一)
- 解后还原:trade = x′·band(scale_map.csv);**目标值换元保值,免还原**

**效果**:
| 引擎 | 无均衡 | 均衡后 | 精度 |
|---|---|---|---|
| osqp | 4.3s / 52,625 iter | 0.17s / 2,000 iter | 全部 −0.007% |
| qpalm | 0.41s / 806 | 0.045s / 89 | gross/delta 贴界 |
| proxqp | 6.7s / 10,520 | 0.078s / 60 | |
| scs | 1e6 iter 不收敛 | **5.78s Solved(被救活)** | |
| MOSEK | 0.54s | 0.54s(**无感**,18→23 iter 但单步更便宜) | |

**三族理论**解释为何 ADMM 大赚/Newton 中赚/IPM 无感:收敛界是否依赖 κ
(ADMM O(1/ε·κ) 是;Newton 局部二次弱相关;IPM 自和谐+内部自带 Ruiz)。
ABIP+(arXiv 2209.01793 §3.5)的 Pock-Chambolle/Ruiz 预条件是同型对角缩放,
佐证"均衡是求解器公认预处理,非实验 hack";其假收敛判据("vs MOSEK <1%")
与我们的 6580.2285 裁判同法。

## 3. 容差地形(均衡前后)

- **无均衡**:比例射线(abs:rel=100:1 同步 ×10..×1e5)/abs 专项/对角线,三条路同构——悬崖前平坦(osqp 迭代钉 52.4k,abs 不进停机判据),悬崖后崩(1e-3 崩 122%~79%)。唯一"快"的点全是崩的;还有轨迹彩票案例(同迭代数崩 92.8%)。
- **均衡后**:同一射线变平滑 Pareto——qpalm ×10⁴ = **38ms/0.010%**(几乎白拿 11×);osqp ×10⁵ = 17.3ms/0.96%;对角线 1e-2 = 14.7ms/1.85%(仅有的 <20ms 点,离 10ms 目标差一段)。

## 4. admm.md §3.4 改进族实验结果

### (a) 预处理族(report_presolve.md)
- **经典矩阵均衡不能替代问题特定均衡**:Ruiz/PC 只作用 A,治不了跨度③,
  M-only 全线错误解(obj=−1.6e15)且迭代数与 no-scale 持平
- Ruiz 叠加 scaleRows 普遍冲突(osqp 发散/qpalm dual infeasible/scs 超时);
  唯一例外 proxqp 44 iter(−27%)但墙钟持平,无实用价值
- 内置开关正确姿态:**qpalm 开 scaling=10(基线硬编码 0 是次优,−10%);
  osqp 关 scaling(−15%);scs normalize 是命脉(关掉 21× 慢);proxqp 保持**

### (b) 自适应 ρ/残差平衡 + (c) 加速/松弛(report_bc.md)
- **osqp adaptive-ρ 是生死项**:关掉 0.15s→43s+(287×)——且 (a) 的均衡是
  (b) 生效的前提(残差比可被干净读出)
- **qpalm sigma_init=1 → 40.6→33.3ms(−18%),全场新最优**,宽平台(1~10 均 ~34ms)
- proxqp 默认即最优(mu/rho 全劣,alpha_bcl 动了全超时,标记勿动)
- over-relaxation α 三库默认即顶点(osqp 1.6/scs 1.8);(c) 深度项
  (reflected-Halpern/restart)需 fork vendored OSQP,不做(osqp 已被 qpalm 支配)
- (d) 在线学习:逐迭代控制经 `osqp_update_settings`(α 可中途改)+`osqp_update_rho`
  +warm start 分段续解在 harness 层**可行**,但固定 α 地形平坦→headroom 存疑,暂缓
- (e) 弹性折叠:本数据 relax 未触发,无实验对象

### (f) polish(report_polish.md)
- 前提澄清:qpalm 无独立 polish(Newton 内环即精度);proxqp IR 思想可移植;scs 没有
- **Phase 0 意外发现:osqp 内置 polish 从 baseline 起一直静默失败**(距离判据被双边行击穿)
- harness polishKKT(乘子判据定主动集 + δ=1e-6 处理 u/g/t 零对角 + −ε=1e-10 处理
  线性相关主动行 + AMD 排序 SparseLU + 全行校验/失败回退):**6–8ms**
- **Schur 消元路径判死**(有价值失败):u/g 变量 P=0、D=δ=1e-6,q_u/D_u~1e8 进 RHS,
  浮点噪声经乘子空间污染 x。教训:P=0 辅助变量小 δ 正则化下,消元法不敌带主元 LU
- 截断迭代+polish 甜点窗口窄(~1200–1725),不如松容差通道
- **跨引擎一致性:polish 成功 ⟹ 分歧 166.6→0.02–0.03(~5000×,同一顶点 3–4 位)**

## 5. 精度与可复现性(report_diff_analysis.md + report_scaled.md §5.2)

| 对比场景 | pos absMax(名义值) |
|---|---|
| MOSEK 同二进制复跑 ×10(md5 全不同) | 0.0105(p50 对 3.1e-3;70/358 品种 >1e-3) |
| MOSEK 跨日/跨构建 | 0.013(目标差 1.9e-7%) |
| 开源引擎互比(未 polish) | 3.6(族内) |
| 开源引擎互比(**polish=kkt**) | **0.02–0.03** |
| 开源引擎 vs MOSEK | 166.6~166.8(平坦面地板,与容差无关) |

- 分布:p50=0.24、p90=10.2、max=167;分歧集中在大持仓/折点品种的平坦方向
  (8750.T 近零折点、3697.T 167=0.36% band)
- **abs<1e-3 不可达**(连 MOSEK 自身复跑都超线);**rel-band<1e-3 天然满足**
  (跨求解器 max 8.7e-4)——推荐验收口径
- 若需更强一致性:加微小正则消平坦(λ/τ 或 ℓ2 prox),以微小目标损失换唯一性
- 单位勘误:全程 position/trade 为**名义值(日元)**,各报告用词已统一修正

## 6. 最终排行榜与推荐配置(gap 全 0.007%,默认 eps 除非注明)

| 引擎 | 配置 | solve | 定位 |
|---|---|---|---|
| **qpalm** | `--scale-rows --lib-scaling=on --sigma=1` | **33.3ms** | 速度档 |
| qpalm | 上 + `--eps-rel=1e-6 --polish=kkt` | 39.6ms | 一致性档 |
| proxqp | `--scale-rows --polish=kkt` | 74ms | 稳健档 |
| **osqp** | `--scale-rows --lib-scaling=off --eps-rel=1e-5 --polish=kkt` | **107ms** | 比 COLAMD 版再降 |
| scs | `--scale-rows --eps-abs=1e-5 --polish=kkt` | 4.35s | 可用性档 |
| MOSEK | — | 140–190ms | 商业参考 |

速度-精度 Pareto 极值:osqp 均衡 1e-2 对角线 = 14.7ms/1.85%(<20ms 仅有的两点之一);
qpalm ×10⁴ = 38ms/0.010%(性价比之王)。

## 7. 负结果与勘误存档(同样重要)

1. 经典 Ruiz/PC 矩阵均衡:错误解(M-only 全线,§4a)
2. Ruiz 叠加均衡:三引擎崩,proxqp 无实用收益
3. Schur 消元 polish:数值不可行(u/g 小 δ 放大)
4. osqp 内置 polish:静默失败(诊断后才知)
5. **勘误**:report_polish.md §4 初版"四引擎逐位一致 absMax=0.0"系比对脚本错误,
   真实 0.02–0.03(与 maxObj 3e-5 差异矛盾当时未察觉);已按诚实口径更正两份报告
6. 无均衡放容差:三条路径全部证伪(含"轨迹彩票"不可复现案例)
7. cuNRTO(GPU DR/ADMM):非适用规模,仅佐证术语

## 8. 基础设施事实(复现必备)

- MOSEK tools 正确 URL:`https://download.mosek.com/stable/9.3.22/mosektoolslinux64x86.tar.bz2`
  (此前 403 是路径错写成 /mosek/tools/...);部署 `~/mosek/9.3/tools/platform/linux64x86/`
- Go 版 jobd 带 mosek 引擎:`CGO_CFLAGS=-I.../h CGO_LDFLAGS="-L.../bin -Wl,-rpath,..."
  go build -tags mosek`(module cache 只读,不能走 build.sh)
- cpp_admm 新增 CLI:`--presolve=none|ruiz|pc` `--lib-scaling=default|on|off`
  `--rho/--sigma/--alpha/--adaptive-rho` `--polish=off|kkt|ir` `--osqp-polish=`
  `--max-iter=N` `--no-relax`;调试环境变量 `POLISH_DEBUG` `POLISH_SOLVER=lu|schur`
- Eigen 陷阱:setFromTriplets **不 resize**(0×0 矩阵上调用直接 SEGV);
  SimplicialLDLT 只读下三角;KKT 分解用 **AMD** 不用 COLAMD(5.8 vs 10.3ms)

## 9. 文件索引

| 文件 | 内容 |
|---|---|
| cpp_admm/report_baseline.md | 四库 baseline + §7.4-7.7 容差/均衡全史 |
| cpp_admm/report_scaled.md | 均衡=数据预处理(变换推导/三跨度/MOSEK 实测/交叉验证/容差地形/pos 精度) |
| cpp_admm/report_presolve.md | (a) 预处理族(手动 Ruiz/PC + 内置开关) |
| cpp_admm/report_bc.md | (b)(c) 自适应 ρ/松弛旋钮 |
| cpp_admm/report_polish.md | (f) polish(机制澄清/实现/Schur 失败史/勘误) |
| cpp_admm/report_diff_analysis.md | MOSEK 复跑 ×10 非确定性 + 三口径精度台阶 |
| plan.md | Step 1/1.5/2 总览(§8 理论/§9 文献) |
| make_optim_scaled.py + optim_scaled/ | 数据级均衡生成器与产物(含 scale_map.csv) |
| /tmp/mose_run/{sweep,sweep2,sweep3,pre,bc,f}/ | 全部扫描原始数据与脚本 |

## 9.5 自审查与修复补录(2026-08-21)

代码自审查发现并修复:①proxqp 乘子取错向量(y=等式乘子恒空,应读 z;
符号与 osqp 同号,对拍 1740:0)②scs 拆行后乘子未合并回原行(已按
y=y_u−y_l 合并)③violator-dump 重构引入的控制流 bug(`return false` 被
圈进 debug 分支,无 POLISH_DEBUG 时**不可行 polish 解被接受**,如
proxqp@1e-3 输出 3729)④verify 基准含松弛 u 与输出 cost|x| 口径不一
(输入解也过 polishRecoverAux)⑤polishVerify 写局部拷贝致"成功但输出
未更新"。全部修复+复测:错误解回退、proxqp 1e-6 polish 新增成功、
跨引擎一致性 0.009–0.046、排行榜不变。详细记录 report_polish.md §7.5。
未修(低优先):sweep 脚本 status 正则对带空格状态断裂(只影响存档标注);
make_optim_scaled.py 未复制 step1Relax(单日数据未触发,跨数据集需注意)。

## 9.7 Stage 2 补录(2026-08-21,详见 admm_st2.md + cpp_admm/report_st2.md)

- S2-0 滚动 warm-start:osqp update_data_vec+warm_start 实测 **5% 漂移 2.2×**
  (331→150ms)、20% 1.7×;qpalm 无数据更新 API
- S2-1c **nativeprox**(ProxQP 移植,复用块机制):43.5ms vs 库 62.6 =
  **1.44×**,maxObj 逐位一致,零调试一次通过
- S2-1b **nativeqpalm**(QPALM 算法移植 + 品种块×3×3 + rank-k Woodbury
  Newton):优化后 **19.7ms** vs qpalm lib 34.5 = **1.75×,全场第一**
  (min-heap 线搜 avg-41-步替代全排序 + 布局固定 + 合并块应用;
  重建预算发散证伪);polish 兼容(一致性 0.009);松容差 15.3ms@1e-2;
  移植五坑全部源码取证修复(σ₀ 公式/块连续布局/sz×sz 求逆/Dinv 全宽
  泄漏/γ 增量)
- S2-1 **native 无库引擎**(OSQP v1.0 忠实移植 + 问题特定 Woodbury,
  ~450 行零依赖):迭代 1775≈osqp 1725,**冷解 80ms(1.8× 快于 library)**,
  松容差 1e-2 = **6.7ms 首次破 10ms**(精度语义与 osqp 同点同值);polish
  兼容(hPolish=1,跨引擎一致性 0.031)。移植三坑:y 非缩放对偶/z̃=Ax̃
  (linsys wrapper 覆写,源码-only 知识)/ρ 估计 dual 归一化含 ‖Px‖。
  性能关键:dot-product 归约 4 路手工展开(向量化)+ 行分类连续切片
  (120→45μs/iter)

## 10. 下一步(未做,按优先级)

1. ~~cpp 无库版~~ **已落地**(native 引擎,见 §9.7);剩余:状态持久化
   (滚动 80→~40ms)、restart 影子序列(超 qpalm 唯一路径)
2. 10ms 冲刺剩余路径:osqp rho/alpha 精调(地形已知崎岖)或无库版定制 prox
   (Moehle-Boyd:删 TUB/GUB,cost|x| 软阈值 + gross 排序投影进 z-步)
3. (d) 在线 α/ρ 控制:harness 分段续解通道已验证可行,headroom 实测后决定
4. 多日真实数据验证(当前全部结论基于单日 optim;扰动合成器可先造假数据)