# Stage 2 实验 report:S2-0 滚动 warm-start + S2-1 native 无库引擎

- 日期:2026-08-21;规划见 `admm_st2.md` §4;全部数据 `/tmp/mose_run/st2/`
- 环境:同 Stage 1;新增编译选项 `-march=native`(全仓)

---

## 1. S2-0:滚动 warm-start(结论:~2×,低于预估 2–5×)

`test/rolling_osqp.cpp`:持久 OSQP solver + `osqp_update_data_vec`(改 q,零重分解)
+ `osqp_warm_start(x,y)`,模拟日内重解(alpha ±jitter,序列 9 点,cold=每点重建):
- **jitter 5%:solve-only 均值 331→150ms(2.2×)**,warm 迭代 1275–2125 vs cold 2200–3575
- jitter 20%:290→168ms(1.7×),方差大(adaptive-ρ 偶发 6325 iter)
- 解的质量:warm 与 cold 同顶点(diff 2.36 缩放单位 ≈ 家族地板换算)✓
- 勘误:OSQP 默认 adaptive_rho 按**时间**触发(结果受机器负载影响);
  `adaptive_rho_interval=25` 可确定化但更慢(base 2200→4750 iter),保持默认
- 附带确认:qpalm 有 warm_start 但**无数据更新 API**(重解需重建 workspace)
  → 滚动场景 osqp/native 是正解

## 2. S2-1:native 无库引擎(结论:核心达成,冷解 80ms / 松容差 6.7ms)

`src/engines/native_engine.cpp`(~450 行,零库依赖):OSQP v1.0 算法的忠实
C 移植 + 问题特定 Woodbury。

### 2.1 结构(admm.md §1 预言的兑现)

- **行分裂**:[结构行(box/TUB/GUB) | 一般行(67 条)]——`Problem.generalStart`
  新标记;结构行对 A'A 只贡献对角(TUB/GUB 成对相消),一般行构成 rank-67 项
- **x-更新** = Woodbury 精确解 D+ρG'G:D 对角、k×k Cholesky(67³/3≈1e5 flop
  仅在 ρ 变更时,全程仅 6 次);每迭代 O(nnz),无 1782 行 KKT 稀疏分解
- **算法细节(逐行对照 vendored OSQP 源码校准)**:
  - v1.0 关键语义:**y 是非缩放对偶**(ρ 变更免重缩放);z̃=Ax̃(KKT 第二块
    被 linsys wrapper 覆写为 rhs_z+ρ⁻¹ν=Ax̃——读 qdldl_interface.c 才知,
    论文不写);rhs_z = z − y/ρ(add_scaled 的替换/累积累积语义陷阱)
  - 自适应 ρ = OSQP compute_rho_estimate:est=ρ·√(rp/pn ÷ rd/dn),偏离 5×
    才采用;dual 归一化含 ||Px||(初版漏掉导致 ρ 永不触发)
  - 终止 = OSQP 全局范数 ||Ax−z||∞、||Px+q+A'y||∞(≠ Go 版逐行判据——
    这是初版 35k 迭代的另一半原因)
- **性能三板斧**(120μs/iter → 45μs/iter):
  1. 行分类存储:box(1 nnz)/pair(2 nnz)/general(索引连续→切片访问)
  2. **归约循环手工 4 路累加**(dot-product 不重结合 gcc 不向量化:G1 57→16ms)
  3. `-march=native`(80→103ms 再降到 80ms 级;chol 反向替换改右视行连续)
- 踩坑记录(全部修复并验证):y 符号反 → 线性发散;KKT companion z̃ 误用 →
  指数发散;单步取证法(显式 dense KKT vs OSQP API max_iter=1 步进)定位

### 2.2 实测(optim,`--scale-rows`,默认 eps)

| 配置 | iter | solve | 备注 |
|---|---|---|---|
| **native** | **1775** | **80.1ms**(×3 稳定) | vs OSQP library 144ms = **1.8×** |
| native + `--polish=kkt` | 1775 | 87.9ms | hPolish=1;与 qpalm+kkt 一致性 0.031(同顶点)✓ |
| osqp library(对照) | 1725 | 144ms | 迭代数一致(差 3%:ρ 时机) |

**正确性**:native vs osqp 逐笔 absMax=**0.1**(同轨迹);vs MOSEK 165.9/p90 10.3
= 精确踩在家族地板;maxObj 6579.7543(−0.007%)✓

**Pareto(eps 阶梯)**:

| eps_rel | iter | solve | maxObj(osqp library 同点对照) |
|---|---|---|---|
| 1e-8 | 1775 | 79.8ms | 6579.754(144ms) |
| 1e-5 | 1175 | 53.7ms | 6579.197(102ms+polish) |
| 1e-4 | 1125 | 50.3ms | 6579.722 |
| **1e-3** | 200 | **10.7ms** | 6622.28(17.4ms) |
| **1e-2** | 100 | **6.7ms** | 6733.69(13.5ms) |

**<10ms 达成**(1e-2 档 6.7ms,精度语义与 osqp 完全同点同值——同为 1.85%
gap);松容差下 native 相对 library 的优势扩大(1.6×)。

### 2.3 kill criteria 判定(admm_st2.md §4)

**通过**:迭代数 1775 ≈ OSQP 1725(未恶化,远低于 3× kill 线);每迭代
45μs vs OSQP library 83μs(1.8×);冷解 80ms 落在预估 10–50ms 区间上沿。

### 2.4 排行榜更新(默认 eps,solve 口径)

| 引擎 | solve | 定位 |
|---|---|---|
| **qpalm**(lib-scaling=on, σ=1) | 33.3ms | Newton 族速度王 |
| **native**(OSQP 式+Woodbury) | **80.1ms** | 无库依赖/warm-start 就绪/松容差 6.7ms |
| native + polish | 87.9ms | 一致性档 |
| osqp library | 144ms | 被 native 支配 |
| proxqp | 67ms | |
| MOSEK | 140–190ms | |

## 3. 剩余工作(S2-2/S2-3 与优化空间)

1. **native 滚动 warm-start**:引擎状态当前 per-call 创建;暴露持久化
   (x,z,y + Woodbury 复用)后,按 S2-0 结论预计 80→~40ms 级
2. 每迭代 45μs → ~25μs:rhs/Ax 的 box+pair 散射与 wy 折叠(余量 ~1.6×)
3. S2-2 restart(reflected-Halpern 影子序列):把 1775 砍到数百是超 qpalm
   (33ms)的唯一路径;polish 已兼容,主动集识别判据可复用乘子
4. S2-3 relax 折叠 + 内部证书:native 内建(Lew z-投影折叠)

## 4. 复现

```bash
cd /home/logic/code/mose/cpp_admm
./build/csvport_cpp opt -i ../optim_inputs/optim_inputs.20260520 \
  -c ../optim_inputs/optim_constraints.20260520 \
  -w ../optim_inputs/optim_weights.20260520 -o /tmp/o.csv \
  --engine=native --scale-rows                     # 80ms
NATIVE_TIME=1 ... --engine=native                  # 分段计时
./build/rolling_osqp -i=... -c=... -w=... --jitter=0.05 --seq=8   # S2-0
```

---

## 5. S2 增补:nativeqpalm(Newton 族无库引擎,2026-08-24)

应用户要求把 Newton 族也做成 native:QPALM 算法(proximal ALM + 半光滑
Newton + 精确分段线搜)忠实移植 + 问题特定 Newton 求解器,
`--engine=nativeqpalm`(~740 行,仅依赖 Eigen 头)。

### 5.1 Newton 求解器结构(与 LADEL/CHOLMOD 的差异)

M = diag(Q)+1/γ + Σ_active σᵢaᵢaᵢ′:
- **active 结构行**(box/TUB/GUB)只碰单品种变量 → union-find 聚成
  343 个 ≤3×3 品种块(两遍连续布局 + varSlot O(1) 定位,O(nnz) 装配)
- **active general 行**(≤67)→ rank-k Woodbury:S=I+W Dinv W′ 的 k×k
  Cholesky + 缓存 Dinv·W′
- 重建成本 12ms/64 次(库版用 rank-update 几乎为 0——我们的差距来源)

### 5.2 移植五坑(全源码取证修复)

1. **σ₀ 数据驱动初始化**(initialize_sigma 公式,非裸 sigma_init)
2. **块成员必须两遍连续布局**(单遍 append 交错 → blkStart 读错成员,
   d 炸 1e15;靠 selftest |Md−b| 逼近 + 最差块 dump 定位)
3. **块逆只算左上 sz×sz**(4×4 框架补零位使 det≡0 → 全块 fallback)
4. **Dinv·w 必须全宽**(块逆非对角,泄漏到切片外的块友 u_j/g_j;
   切片截断版 dua 炸 1e22)
5. gamma 增项在 Qx 里增量维护(outer 的 x0←x 后 df 的 prox 项自消)

### 5.3 实测

| 配置 | iter | solve | 备注 |
|---|---|---|---|
| **nativeqpalm** | 86 | **32.2ms**(×3 稳定) | vs qpalm lib 34.5ms 持平;iter 86 vs 58(重建税) |
| nativeqpalm+polish | 86 | 38.4ms | hPolish=1;与 qpalm+kkt 一致性 0.0093(同顶点) |
| eps 阶梯 | 83/71/64/39 | 31/27/25/**15.3ms** | 1e-2 档 15.3ms(−0.35% gap 曲线同 qpalm) |

正确性:vs MOSEK absMax=166.7/p90=10.2(家族地板);vs qpalm lib 1.86(同顶点)。

### 5.4 全家族最终榜(冷解,`--scale-rows`,默认 eps)

| 引擎 | solve | 依赖 | 定位 |
|---|---|---|---|
| **nativeqpalm** | 32.2ms | 无(纯 C+Eigen 头) | Newton 族自有版,与 qpalm lib 持平且 polish 兼容 |
| qpalm lib(σ=1) | 34.5ms | LADEL+BLAS | 被 nativeqpalm 追平 |
| native(OSQP 式) | 80.1ms | 无 | 松容差王(1e-2=6.7ms) |
| proxqp | 67ms | Eigen/proxsuite | |
| osqp lib | 144ms | | |
| MOSEK | 140-190ms | 商业 | |

两族 native 均无 warm-start 障碍(状态在自家结构里)——滚动场景
(nativeqpalm 重建税消失)是下一个待测点。

---

## 6. 任务 B + A 补遗(2026-08-24):nativeprox 与 nativeqpalm 优化

### 6.1 nativeprox(ProxQP sparse 算法移植,任务 B)

`--engine=nativeprox`(~600 行,纯 C+Eigen 头):ProxQP 主循环忠实移植
(BCL 步长控制 + 半光滑 Newton + primal-dual 分段折线线搜),Newton =
同一套块+Woodbury(系数统一为 mu_in⁻¹)。关键语义(源码取证):
KKT (2,2) 块 active=−mu_in、inactive 为恒等行(dz=−z 显式);
M = diag(H)+rho+mu_in⁻¹Σa a′;初值 x=−(H+rho)⁻¹g(空 active set 等式
约束猜测)。

| | iter | solve | 备注 |
|---|---|---|---|
| **nativeprox** | 90 | **43.5ms**(×3 稳定) | vs proxqp 库 62.6ms = **1.43×** |
| + polish | 90 | 50.3ms | hPolish=1;polished 后与 nqp 差 84.8(边缘行判定噪声,目标差 2.8e-5) |

正确性:maxObj 6579.757111 与库**逐位一致**;vs 库逐笔 0.08(同轨迹);
vs MOSEK 166.6/p90 10.2(家族地板)✓。**一次通过,零调试**——受益于
nativeqpalm 已验证的块机制直接复用。

### 6.2 nativeqpalm 优化(任务 A):32.2 → **19.7ms**

| 优化 | 时间 | 说明 |
|---|---|---|
| 基线 | 32.2ms | tLS=17.5(线搜 qsort 全排序)+ tBuild=11.7 |
| ① std::sort + walk 探针 | 27.3ms | walk 实测 avg=41/max=458(断点 2000+)→ 全排序浪费 |
| ② **min-heap 弹出式线搜** | 23.5ms | make_heap O(nL) + 逐个弹出到首个正梯度(41 步),替代 O(nL log nL) 全排序;结果逐位同 |
| ③ 布局固定化 | 22.2ms | union-find 聚类用全量结构行固定一次(active 变化不再重算布局) |
| ④ 合并 Dinv·w 双循环 | 20.5ms | wdFull 与 dinvWt 同一遍块应用 |
| ⑤ 重建预算/结构行 skip | ✗ 回退 | semismooth Newton 不容忍 inexact 方向(TH=10 发散 1e15;库的 rank-1 update 是精确同步,非"旧矩阵")|

最终:**19.7ms**(×3 稳定),iter 86 不变,maxObj 6579.756085 不变;
+polish 26.0ms(与 qpalm+kkt 一致性 0.0093 同顶点)。
tLS 8.7 + tBuild 7.6 + 主循环 3.4。

### 6.3 全家族最终榜(冷解,--scale-rows,默认 eps,solve 口径)

| 引擎 | solve | 依赖 | vs 库版 |
|---|---|---|---|
| **nativeqpalm** | **19.7ms** | 无 | qpalm 34.5ms 的 **1.75×** |
| **nativeprox** | 43.5ms | 无 | proxqp 62.6ms 的 **1.44×** |
| native(OSQP 式) | 80.1ms | 无 | osqp 144ms 的 1.8× |
| qpalm lib(σ=1) | 34.5ms | LADEL+BLAS | — |
| proxqp lib | 62.6ms | Eigen/proxsuite | — |
| osqp lib | 144ms | | |
| MOSEK | 140-190ms | 商业 | — |

三族 native 全部快于其库版且零依赖;nativeqpalm 19.7ms 为全场第一
(松容差仍由 native 6.7ms@1e-2 称王)。

## 7. S2 增补 2:nativeralm(NR-LALM 第四族,2026-08-28,负结果)

arXiv 2608.19847(固定罚线性化 ALM + 经典乘子更新)的忠实移植 +
我们问题上的不等式适配(Ax=z 分裂 + 精确箱投影),作为第四族 native
对照引擎(`--engine=nativeralm`,src/engines/native_ralm_engine.cpp)。

**论文保真度**(全文精读,详见 admm_st2.md §2.3 判词):约束线性时
论文的核心机器(约束线性化二次误差)恒为零;不等式处理论文留白
("remain open");m=1782>n=1074 使 uniform LICQ 结构性不成立。引擎
保留其单循环骨架:`M p = -(∇f + A'(y+ρr))`,经典乘子更新,固定
(ρ,β),**P 不进 M**(Gauss-Newton 特征)。

### 7.1 相图(ρ×β 网格 49 点,eps 1e-6/1e-8,scaleRows)

| 变体 | 收敛区 | 最优迭代数 | 时间 | 精度(vs MOSEK) |
|---|---|---|---|---|
| 忠实版(P 在 M 外) | 仅 β≳1000 且极窄 | **843,475**(β=1000,ρ=1) | 33.6s | — |
| pinP 变体(P 进 M) | ρ≥10,β 任意 | **85,500**(ρ=10) | 3.40s | max 167.6 / p90 10.60 / mean 4.42(精确踩家族地板) |
| (对照)native | 默认 | 1,775 | 80ms | 同地板 |
| (对照)nativeqpalm | 默认 | 86 | 19.7ms | 同地板 |

三个决定性观察:
1. **ρ 完全无影响**:忠实版下 ρ∈[0.01,10] 跨三个量级,迭代数纹丝不动
   (300/675/5001 逐组相同)——因为 P 不在 M 时 x-步退化为步长 1/β 的
   梯度下降,稳定性只由 β vs ‖P‖ 决定;发散边界 β∈(300,1000) ⇒
   有效 ‖P‖≈600
2. **P 折出矩阵是灾难**:忠实版最好 843k 迭代 = native 的 475×;
   pinP(把 P 放回 M,其余不动)恢复到 85.5k = native 的 48×——
   OSQP 把 P 放进 M 正是它 1775 迭代的来源
3. **循环形状本身也慢**:pinP 与 native 共享同一类矩阵(P+βI+ρA'A),
   差距 48× 全来自循环形状(无 over-relaxation α、无 adaptive-ρ、
   z-邻近更新)

### 7.2 三件套(论文的真正可搬部分)

- **Φ̂ 监控**(Theorem 2.9 免常数版):单调下降实测(-13769 min),正常
- **argmin 残差输出**(Theorem 2.12 输出规则):本引擎轨迹平滑,
  松容差(1e-3/1e-2)下 bestIt==finalIt 从未触发——悬崖是
  native/nativeqpalm 机制(adaptive-ρ/罚提升)的现象,S2-2 在那边测
- ρ/β 比例扫描:论文区间 [6,64] 建立在 AA'=I 正交行上,对本问题
  无意义(见观察 1)

### 7.3 判词

**NR-LALM 引擎被全家族支配(48-475×),不再迭代**。论文对本项目的
价值 = 三件套(其中 argmin 输出规则待在 native/nativeqpalm 上验证),
引擎本体是干净的负结果:固定罚单循环 + Gauss-Newton 邻近步在
P 主导的凸 QP 上结构性不合适。正确性交叉验证:pinP@ρ=10 解的
maxObj 6579.760 与家族一致,vs MOSEK 踩地板。

## 8. S2-2:反射 Halpern 加速(2026-08-28,kill 线处决)

arXiv 2606.16552 Algorithm 1/2 忠实移植到 native 引擎
(`NATIVE_HALPERN=1` 系列环境开关,基映射 = native 单步 + 固定 ρ):
shadow `ŵ=F(w)` → 反射 `w̄=(1+γ)ŵ−γw` → 锚定
`w⁺=(w⁰+(k+1)w̄)/(k+2)`;restart wrapper 用论文 §6.1 三触发
(0.2 充分下降 / 0.8+停滞 必要下降 / 0.36·kTot epoch 过长),
锚点重置为当前状态;停机与输出在 shadow 上;argmin 残差输出规则
(NR-LALM 三件套)同时下沉。

### 8.1 实验矩阵(eps 1e-6/1e-8,scaleRows)

| 基映射(native 单步) | 无 Halpern | +Halpern γ 扫描 |
|---|---|---|
| α=1, 固定 ρ=0.1(裸 PFBS) | **>1M iter 不收敛** | γ=0.7:**6450**(唯一救场);γ=0/0.3/0.9 不收敛 |
| α=1.6, 固定 ρ=0.1 | >20k 不收敛 | γ=0.7:6450;γ=0.5:15650;其余不收敛 |
| α=1.6, 固定 ρ=100/300 | 5150 / **1650** | **全部爆掉**(γ≥0.5 违反 α-averaged 上限 γ≤(1−α)/α≈0.25;γ≤0.25 也劣化到 8850+) |
| α=1, 固定 ρ=300 | 6500 | γ=0.5/0.7/0.9:10075-12575(**全劣化**) |
| 纯锚定(关 restart) | — | **44 iter 内全部数值爆炸**(restart 实为安全阀) |
| 默认(α=1.6 + adaptive-ρ) | 1775 / 80.1ms | —(映射非固定,理论不适用) |

### 8.2 机理判读

1. **本问题的 ADMM 算子不在 Halpern 增益区**:调优过的基映射
   (ρ=300)上锚定+反射只添噪声;病态基映射(小 ρ 过阻尼)上
   Halpern 有救场作用但仍劣于直接调 ρ
2. **restart 触发过频**(5360 次/6450 iter ≈ 每 1.2 iter)是安全阀
   行为:锚定态在平坦方向漂移累积,靠频繁重置压制——论文的
   sharpness 线性收敛区(识别后)在本问题从未进入
3. **理论边界实测吻合**:α=1.6 基映射(0.8-averaged)的反射上限
   γ≤0.25 被精确验证(γ≥0.5 在好基映射上爆,差基映射上不爆只因
   本身过阻尼)
4. **kill 线**:需求 <1000 iter(1.8×),最好结果 6450——处决

### 8.3 意外收获(如实记录)

- **固定 ρ=300 针尖**:1650 iter / 68ms,超默认 80.1ms(1.18×),
  精确踩家族地板(167.2/10.50/4.40);但 ρ=280/310 即 2× 恶化
  (3350/3250)——**非稳健调参,只作奇点记录,不推荐**
- **松容差悬崖在 ρ=300 下 obj 侧大幅缓解**:1e-3 → +0.10%、
  1e-2 → +0.13%(默认 α/ρ 下为 −0.7%/−2.3~4%);但 pos 侧仍
  万级(平坦面表示差,与 §7 一致);1e-2 档 50 iter / 3.4ms
- **argmin 输出规则从未触发**(bestIt==final 全部实验):悬崖是
  真实早停,非尾部漂移——NR-LALM 三件套的第三件对本问题无效
- nativeqpalm 外层不做:外层 ALM 每 epoch 改 σ/γ(非固定映射),
  且 native 上的 kill 证据充分

### 8.4 判词

**S2-2 处决**。反射 Halpern 在本问题(均衡后、强凸 P、167 个
有效行)无增益:一阶算子的瓶颈不是收敛尾部而是每迭代成本
(nativeqpalm Newton 族 86 iter 19.7ms 已封顶该路线)。代码保留在
env 开关后供复现。

## 9. S2-3a:内部对偶间隙证书 gapCert(2026-08-28)

**目标**(Moehle-Boyd §4 思路):求解器自带 `obj ± gap`,免 MOSEK
外部裁判。实现:`dualGapCert(p,x,y)`(model.cpp)+ main.cpp 在
**引擎自己的停机点**上计算(polish 之前——(x,y) 配对一致性;
polish 的 aux 坐标恢复会破坏 y 的配对,实测证书炸到 4180-6594)
并加入 summary 行 `gapCert=`。

**公式**:`prim − dual`,
`dual(y) = −Σ_{P_jj>0} g_j²/(2P_jj) + Σ_{P_jj=0} g_j·x_j − Σ_r [y_r>0 ? y_r·u_r : y_r·l_r]`,
`g = q + A'y`。KKT 点处精确归零(互补恒等式验证);P-zero 坐标的
`g_j·x_j` 替代引入 ≈‖g_P0‖_∞·‖x‖_∞ 的松弛;inf 端越号乘子行跳过
(只会高估 gap,保守方向)。缩放空间下计算,gap 值缩放不变。

### 9.1 验证矩阵(三族 native + lib,eps_rel=eps_abs)

| 引擎 | eps | maxObj | gapCert | 判读 |
|---|---|---|---|---|
| nativeqpalm | 1e-2 | 6351.29(−3.5%) | **1.67e1** | 红旗 ✓(悬崖检出) |
| nativeqpalm | 1e-3 | 6572.57(−0.12%) | 1.19e0 | 黄旗 ✓ |
| nativeqpalm | 1e-6 | 6579.75 | **4.0e-3** | 认证 ✓ |
| native | 1e-2 | 6733.69(+2.3%,不可行点) | **−3.29e2** | 负值 = 不可行红旗 ✓ |
| native | 1e-6 | 6579.76 | 2.5e-2 | 认证 ✓ |
| nativeprox | 1e-2 | 6580.29(+0.01%,微不可行) | 1.39e2 | 红旗 ✓ |
| nativeprox | 1e-6 | 6579.76 | 2.5e-1 | 认证 ✓ |
| osqp lib | 1e-6/1e-8 | 6579.757 | 2.5e-5 | 认证 ✓(引擎无关) |
| qpalm lib | 1e-6/1e-8 | 6579.757 | −3.3e-5 | ≈0 ✓ |

### 9.2 语义与发现

1. **三态读法**:小正 = 解可信;大正 = 松容差悬崖(自动检出);
   **负 = 原始不可行性主导**(x 越界"赚"目标分,如 native@1e-2 的
   +2.3%)——本身就是第三种红旗
2. **表示差 vs 求解误差被定量分离**:紧容差下 gapCert(4e-3~2.5e-1)
   ≪ maxObj 与 MOSEK 的 0.47-0.48 差 → **该差是平坦面表示差,不是
   求解质量问题**(precision_tiers 的结构地板结论获得独立佐证)
3. 生产形态:`status=solved && gapCert < tol` 即可自动判定解可信度,
   无需外部参考;polish 前计算使证书 = 纯求解器自评

## 10. S2-3b:relax 弹性折叠(Lew 2511.08451,2026-08-28)

**基础设施**(全部落地):`Row::hinge` 字段;`RELAX_FOLD=1` 时
buildProblem 不再物化 t_k 变量与两条附加行,原行携带弹性罚
(σ_row 缩放由 scaleRows 同步 `hinge *= s`);native 引擎 z-更新对
hinge 行用**三段 prox**(盒内 z=v / 肩部 z=v∓c/ρ / 盒外边界);
`ELASTIC_PENALTY` env + `RELAX_FORCE` env(直通 relax 场景)实验钩子。

**数学等价性**:`min_t c·t s.t. dist(w'x,[lb,ub]) ≤ t ≡ c·dist(...)`,
三段 prox 是该 hinge 项的精确邻近算子。

### 10.1 实验矩阵(人为收紧 delta_ptf 造 breach,penalty=0.001)

| 公式 | 引擎 | 结果 |
|---|---|---|
| 显式(t 变量) | native | **100k iter 不收敛**(t = P=0 弱曲率 + 巨 rhs 偏移) |
| 显式(t 变量) | qpalm lib | **段错误**(弱曲率 aux 变量病理) |
| 显式(t 变量) | MOSEK | 解出(参考) |
| **折叠(hinge)** | native | **1750 iter / 89ms 解出** |

折叠解的**弹性 KKT 自洽验证通过**:delta 从当前 +2.51M 推到
−100k,超 ub 1e4 单位、付罚 10,乘子饱和在 c(互补松弛精确成立)。

### 10.2 两个边界发现

1. **ADMM 对偶爬升瓶颈**:hinge 行的 y 必须从 0 爬到 c = penalty·σ_row
   才能施满罚压;生产罚(100)× 组合级 σ_row(1e5+)→ c~1e7+,实测
   爬升 1.3e-4/iter → **1e18 iter 量级,纯 ADMM 对生产弹性罚结构性
   不可用(与折叠与否无关)**;温和罚(c~1e2)下折叠正常工作
2. **语义差**:生产显式 relax 内嵌"不比今天更差"硬帽(原行 ub→EPS),
   折叠是纯罚 QP(无帽)——两者最优解不同(folded 允许 delta>0 换
   alpha)。要精确复刻生产语义需"硬盒内单侧 hinge"(follow-up)

### 10.3 判词

折叠 = **救场技术**:显式公式让 native 不收敛、qpalm lib 段错误的
场景,折叠后 native 89ms 解出且 KKT 自洽。正确完成体 = 折叠 +
Newton 族(nativeqpalm 的 huber 罚框架天然适配 hinge,乘子不靠
爬升靠 Newton 步)——留作后续。gapCert 在 relax 问题上语义不适用
(负值,硬界侧项 vs 弹性乘子冲突),已注记。
