# cpp_admm baseline 报告:四大开源 QP 库跑 optim

- 日期:2026-08-18
- 代码:`cpp_admm/`(C++17,单二进制 `csvport_cpp`,`--engine` 切换后端)
- 对照:MOSEK 9.3.22(求解器 0.17s / wall 0.58–0.64s,`log/run_real.md`)、
  Go 自研 ADMM(wall 2.5s,`log/admm/improving.md`)
- 数据:optim(358 只东证股票,67 约束,23628 权重行)+ sample(23 品种,29 约束)
- 容差统一:eps_abs=1e-6 / eps_rel=1e-8(各库默认其余参数,3 次取值稳定)

---

## 1. optim 实测结果(核心交付)

| 引擎 | 算法 | 状态 | 迭代 | solve ms | maxObj | vs MOSEK 6580.2285 |
|---|---|---|---|---|---|---|
| **qpalm** | 近端 ALM + 半光滑 Newton | solved | **806** | **421–440** | 6579.7447 | −0.007% |
| **osqp** | ADMM(自适应ρ + polish) | solved | 52,625 | 4,624–4,791 | 6579.7569 | −0.007% |
| **proxqp** | 近端增广拉格朗日 + Newton | solved | 10,520 | 6,698–6,809 | 6579.7496 | −0.007% |
| scs | 锥形 ADMM(SOCP) | Solved/Inaccurate | 1,000,000(顶格) | ~168,900 | 5349.9080 | 差 18.8%,不可用 |

(加载 CSV ~26ms 未计入 solve;每引擎跑 3 次,迭代数与耗时完全稳定。)

### 结果校验(awk 独立重算)

| 校验项 | qpalm | osqp | proxqp |
|---|---|---|---|
| gross_ptf(Σ\|targetPos\|,ub 5e7) | 5e7 贴界 | 5e7 贴界 | 5e7 贴界 |
| delta_ptf(ub 1e5) | 100000.0 贴界 | 100000.007 | 100000.0002 |
| 逐品种 trade/position box 最大违约 | 0.014 | 0.071 | **0(严格界内)** |
| 63 条 barra NET 最大越界 | barra_JPN +0.031 | 全部满足(delta_ptf 0.006) | 全部满足 |
| vs mosek 逐品种 trade 差(max/mean,名义值) | 1264 / 60 | **167 / 4.4** | 1573 / 39 |

量级说明(单位:名义值/日元,position 为 notional 非股数):box 违约 ≤0.07
发生在 1e5 量级交易上(相对 ~1e-6),是各库默认容差噪声,生产按 0.5(名义值)
向内截断即可消除;proxqp 的解严格在界内。

## 2. sample 实测结果(回归对照)

| 引擎 | 状态 | 迭代 | solve ms | maxObj(参考 48.7250) |
|---|---|---|---|---|
| osqp | solved | 175 | 0.7–0.8 | 48.7248 |
| qpalm | solved | 26 | 1.0 | 48.7225 |
| proxqp | solved | 32 | 1.2–1.5 | 48.7250 |
| scs | Solved | 38,980 | ~143 | 48.7250 |

sample 上 osqp/proxqp 亚毫秒,比 Go 自研版(18.7ms)与 mosek(40ms)都快。

## 3. 排名与结论

optim 单次冷启动:

```
MOSEK 0.17s  <  qpalm 0.43s  <<  Go自研 2.5s  <  osqp 4.7s  <  proxqp 6.8s  <<<  scs (不收敛)
```

1. **QPALM 是唯一接近 MOSEK 的库**(0.43 vs 0.17s):近端 ALM 的 Newton 子问题
   在本题(对角 P + 稀疏行)每步贵但步数极少(806 步 vs ADMM 家族 5 万步)。
   其行列均衡 + active-set 式因子更新正是 admm.md §3.2 预判的"高精度段强者"。
2. **OSQP 慢在迭代数**:算法与 Go 自研版同源,5 万步 vs Go 版 1 万步 + 认证
   polish——证明 Go 版的结构化 Woodbury + KKT 认证 polish 组合确实优于
   通用 OSQP 实现(同为 ADMM,快 ~2 倍且解质量更好)。
3. **SCS 不适合本题**:纯锥投影 ADMM 对混合尺度 box 约束收敛极慢,1e6 步
   只到 inaccurate;锥形化还把行数翻倍(ranged 拆两行 + SOC epigraph)。
4. **无库在冷启动上打赢 MOSEK**,但 QPALM 已在一个数量级内,且零 license、
   可静态链接;规模再涨(数千品种)时内点法的 KKT 分解成本上升更快,
   一阶/半光滑族的相对优势会扩大。

## 4. 工程记录(踩坑,复现必备)

### 4.1 第三方构建(全部装到 /tmp/cppadmm/install)

| 库 | 版本 | 构建方式 | 备注 |
|---|---|---|---|
| osqp | v1.0.0 | cmake,装 libosqpstatic.a | API 是 v1.0 新式(OSQPCscMatrix 字段直填 + osqp_setup 九参) |
| scs | 2.1(cvxgrp master) | 自带 Makefile(`USE_LAPACK=0`),libscsdir.so/.a | 无 cmake;2.x 无 P 矩阵,二次项需 SOC epigraph 自建 |
| qpalm | Benny44 master | cmake + LADEL 子模块;缺 liblapacke 时 `apt-get download liblapacke liblapacke-dev libtmglib3` 解包到 lapacke_root/ 传入 CMAKE_LIBRARY_PATH + CFLAGS=-I | 需 `-DDLONG -DUSE_LADEL`;QPALMData 直填 ladel_sparse_matrix |
| proxsuite | v0.6.4 | cmake(需先装 Eigen;`-DBUILD_WITH_VECTORIZATION_SUPPORT=OFF` 避缺 Simde;submodule update --init) | header-only 使用,config.hpp 是 cmake 生成后才有 |

### 4.2 符号冲突(本报告最重要的坑)

**qpalm.so 与 osqp 都导出同名 C 符号**(`check_termination`、`scale_data`
等,qpalm 源自同一 OSQP 代码族)。同进程加载时,exe 内静态 osqp 的同名符号
劫持 qpalm 的自调用 → **栈崩溃在 qpalm_solve 内部,极难定位**(bt 显示
crash 帧在另一个库里)。解法:osqp 静态链接且 **objcopy 重命名全部非
osqp_ 前缀符号**(116 个,`osqpv1_` 前缀,脚本见下);scs 也因与 osqp
静态都定义全局 `oact`(sigaction scratch)冲突,scs 用动态库。

```bash
# osqp 符号重命名(/tmp/cppadmm/osqp_renamed/)
nm libosqpr.a | awk '$2~/[TDBC]/{print $3}' | sort -u | grep -v '^osqp_\|^OSQP' \
  | awk '{printf "%s osqpv1_%s\n",$1,$1}' > redefs.txt   # objcopy 2.38 格式:old<space>new
objcopy --redefine-syms=redefs.txt libosqpr.a
```

### 4.3 各库接入要点

| 库 | 要点 |
|---|---|
| osqp v1.0 | P 取**上三角** CSC;`osqp_get_solution` 的 4 个输出数组(x/y/两证书)必须全部分配,内部无条件写入 → 传 null 段错误;**必须 `check_dualgap=0`**,否则对偶目标为 -inf 时 gap=nan,残差 1e-10 也不判收敛(表现为顶满 max_iter) |
| qpalm | 头文件缺 `extern "C"` 包裹(qpalm.h),C++ 调用方需手动包;A 的 CSC **列内行序要排序**;数据结构用 `QPALMData` + `ladel_sparse_alloc` 自建;`settings->scaling=0` 规避 scale_data 崩溃(未深究,疑与 nz 数组未提供有关) |
| scs 2.1 | 无 P:二次项 0.5x'Px 编码为单个 SOC `‖[√P·x;(t−1)/2]‖₂ ≤ (t+1)/2`,目标加 0.5t;ranged 行拆两个 nonneg 行(等式也拆);锥行序必须 [zero/nonneg][SOC],head 行 b=+0.5(先写成 −0.5 时解出 obj=−27 的错误点) |
| proxqp | 行/列直接 Eigen 稀疏;`qp.init(H,g,nullopt,nullopt,C,l,u)`(无等式块);v0.6.4 无 SOLVED_BACKTRACKING/UNSOLVED 枚举 |

### 4.4 与 Go 版代码的对应

- csv.cpp ≈ types.go loaders;model.cpp ≈ solver.go 编排 + backend_admm.go
  buildADMMProblem(含固定变量 presolve;**未**移植 Go 专属的成对行契约/
  Woodbury/polish——库自带);
- proctitle.cpp ≈ proctitle.go(prctl + argv 块覆写,C 直接拿 argv);
- main.cpp ≈ main.go CLI(masking 前先存路径字符串)。

## 5. 复现

```bash
cd cpp_admm && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTHIRDPARTY=/tmp/cppadmm/install
make -j16
./csvport_cpp opt -i $IN/optim_inputs.20260520 -c $IN/optim_constraints.20260520 \
                 -w $IN/optim_weights.20260520 -o out.csv --engine=qpalm   # IN=../optim_inputs 绝对路径
```

## 6. 后续建议

1. QPALM 的 806 步 × 0.5ms/步 说明"半光滑 Newton + 稀疏因子更新"路线在本题
   有绝对优势 → Go 自研版值得借鉴其 **active-set 因子增量更新** 思路;
2. OSQP 5 万步 vs Go 版 1 万步:Go 版逐行缩放停止判据 + 认证 polish 是
   通用库没有的增益,保留;
3. scs 移出候选;proxqp 解最干净(0 违约)但慢,可作高精度验证器;
4. python_admm(codegen/实验矩阵)如需做,OSQP codegen 值得单独验证——
   其静态定制版可能显著快于 4.7s 的通用版。

---

## 7. 补充问答记录(2026-08-18)

### 7.1 §4.2 符号崩溃是什么?修复了吗?

**已修复**(四引擎同进程稳定运行为证)。成因:qpalm 作者从 OSQP 代码起家,
两库导出**同名 C 函数**(`check_termination`、`scale_data`、`mat_vec` 等
116 个)。exe 静态链接 osqp + 动态加载 qpalm.so 时,动态链接器解析 qpalm
内部对这些名字的调用会先查全局作用域——exe 里的静态 osqp 副本优先,
于是 qpalm 拿着自己的数据结构调进了 osqp 的同名函数,内存布局对不上
→ 段错误。极难定位:backtrace 把崩溃帧错误归属到另一个库(bt 显示 crash
在 libosqp.so,实际凶手是加载顺序)。解法:osqp 静态链接并 objcopy 重命名
116 个非 `osqp_` 前缀符号为 `osqpv1_`;scs 因同类冲突(`oact` 全局变量)
改动态链接。

### 7.2 四库算法确实不同,实为两大家族(+MOSEK 共三家)

都源自**增广拉格朗日**框架,分岔在"子问题怎么解":

| 家族 | 库 | 原理 | 本题表现 |
|---|---|---|---|
| **一阶 ADMM** | OSQP | 算子分裂,QP 原生形式,z-步逐行 clip | 5.2 万步 × ~0.09ms/步 |
| | SCS | 同为 ADMM,但先锥形化(SOCP),z-步锥投影 | 最差(混合尺度对纯锥投影是毒药) |
| **近端 ALM + 半光滑 Newton** | QPALM | Newton 子问题解到高精度 + active-set 式因子更新 | **806 步 × ~0.5ms/步,夺冠** |
| | ProxQP | 同族独立实现(INRIA)+ Ruiz 预条件 | 1.05 万步 |
| (内点,对照) | MOSEK | 自洽障碍,每步大 KKT | 17 步 × ~10ms |

规律:ADMM 步子便宜但步数多;Newton 步子贵但主动集识别后步数与规模弱
相关;内点步数最少、每步最贵。

### 7.3 是否只有 3 个正确?

是:qpalm/osqp/proxqp 目标值 6579.74–76 三者一致(与 mosek 差 0.007%),
绑定约束全相同(gross 贴 5e7、delta 贴 1e5)——同一个最优解,仅容差噪声
不同(逐品种 box 最大违约,名义值:proxqp **严格 0** < qpalm 0.014 < osqp 0.07)。
**scs 不是 bug 而是失效**:1e6 步顶格仍只到 Inaccurate,目标 5349.9
(差 18.8%);但它在 sample 上收敛且正确(38,980 步,obj 48.7250)——
结论是"SCS 不适合本题的规模/尺度结构",非"SCS 是错的库"。

### 7.4 ADMM 能否用低精度换时间?

能,且有三条现成证据:

1. **理论**:残差 O(1/k) 下降——前几百步吃掉大部分误差,尾部几千步只磨
   最后一两个数量级。sample 的 OSQP 日志实拍:200 步时 prim res 已 1.56e-05,
   离最优 obj 差在第 5 位有效数字;容差 1e-6→1e-3 迭代数大致按比例砍。
2. **Go 版已在这么做**:ADMM 跑到 6–10k 步(中精度)即交认证 polish
   (主动集精确解),不陪 ADMM 磨尾部(improving.md 实测 6k 步起 polish
   命中)——标准套路"低精度 ADMM + 高精度 polish"。
3. **本题真实精度需求很低**:目标函数在最优点附近极平(解差 0.007%,
   目标才差 0.0007%),交易量最终按手取整——1e-3 相对精度足够下单。
   **(此条仅在最优点邻域成立;全局低容差提前停机已被 §7.5 实验证伪)**

**坑**(Go 版血泪教训,basic.md §3.6):混合尺度(box 行 ~1e-6、gross 行
~5e7)下**不能统一放宽全局绝对容差**——放宽到 0.5 时禁交易品种能"合法"
漂 0.5(名义值),经 λ·pos 项虚增目标 21 万。必须用逐行/逐坐标**相对**容差放宽,
或放宽前先做固定变量 presolve(两版均已实现)。

量化这条权衡曲线的现成实验(半小时):OSQP 引擎在 optim 上扫
`--eps-abs={1e-3,1e-4,1e-5,1e-6}` × {polish on/off},记录迭代数/solve/
box 违约/目标差,与 qpalm 0.43s 同表对比。

### 7.5 低相对容差实验(§7.4 假设的实证检验——**证伪**)

按 §7.4 的设想对 ADMM 族做低精度换时间实验(OSQP 扫 `--eps-rel`,
eps_abs 固定 1e-6;SCS 只有单一 eps 旋钮,扫 `--eps-abs`),optim 数据:

| 引擎 | 容差 | 迭代 | solve | maxObj | vs 最优 6579.75 | 解质量 |
|---|---|---|---|---|---|---|
| osqp | 1e-8(基线) | 52,625 | 4.7s | 6579.757 | −0.007% | 贴界,良 |
| osqp | 1e-3 | 31,950 | 2.81s | **−1483** | 差 122% | 4 笔 box 违约;delta=97278/gross=4.51e7,**均未顶界** |
| osqp | 1e-2 | 325 | **38ms** | **−2219.7** | 差 134% | 34 笔违约;delta=86412,界外悬停 |
| osqp | 2e-2 | 325 | 38ms | −2219.7 | 同上 | 同上 |
| scs | 1e-3 | 1e6 顶格 | 171s | 5349.9 | 差 19% | 仍 Inaccurate |
| scs | 1e-2 | 3,040 | 522ms | 4159.0 | 差 37% | 130 笔 box 违约 |
| scs | 2e-2 | 1,780 | 308ms | 4516.3 | 差 31% | 77 笔违约 |

#### 结论(修正 §7.4 的乐观外推)

1. **全局相对容差放宽在本题被证伪**:OSQP 1e-2 确实快 122 倍(4.7s→39ms),
   但目标从 6579 崩到 −2220(34 个百分点)。SCS 同样:1e-2 提前收敛,
   目标差 37%。§7.4 引用的"200 步已到第 5 位有效数字"是 **sample(23 品种)
   的日志**,对 optim(358 品种)不成立——外推失效。
2. **失效机理**:本题最优解位于约束**顶点**(gross 贴 5e7、delta 贴 1e5);
   ADMM 的价值几乎全部产生在"从可行内部推向顶点"的最后一段。全局相对容差
   按 ~1e6 量级行缩放,1% 容差 = 每行 1e4 绝对松弛,迭代在离顶点很远处
   (delta=86412,还剩 14% 杠杆没用)就"合法"停机。目标函数在**收敛路径**
   上并不平坦(§7.4 说的平坦只在最优点邻域成立)。
3. OSQP 1e-3 的 32k 步目标仍为 −1483:说明 32k→52.6k 这段"尾部"迭代
   承载了全部 ~8000 的目标改善——尾部不是噪声,是收益本体。
4. polish 救不了:OSQP 内置 polish 从 325 步迭代点猜主动集必然猜错
   (失败即返回原迭代点);这反证 Go 版"认证 polish(可拒绝/重试)+ 逐行
   缩放容差"设计的必要性——低精度路线**必须**配可认证的精修,否则不可用。
5. 低精度解的正确用途只剩 **warm start**(39ms 的粗解喂给 qpalm/
   自研版作初值),而非直接交付。

即:在"最优在顶点 + 混合尺度"的本题上,ADMM 族可用的最低交付精度由
逐行缩放判据决定(Go 版做法),单一全局 eps_rel 的低精度换时间路线不通。

### 7.5.1 机理:为什么 1e-3/1e-2 相对容差导致 122%/37% 的目标崩塌

三层原因,层层放大:

**① 容差算术——相对容差乘的是行活动量的 inf-范数(≈5e7)**

OSQP 的停止判据(全局非 per-row):

```
|Ax−z| ≤ eps_abs + eps_rel · max(‖Ax‖∞, ‖z‖∞)
                    ↑ optim 里最大行 = gross_ptf 活动量 ≈ 4.5e7~5e7
```

- eps_rel=1e-2 → 绝对松弛 ≈ **4.5e5**,比一整条交易带宽(±1e5)还大 4 倍;
- eps_rel=1e-3 → 绝对松弛 ≈ **4.5e4**,相当于半个交易带宽;
- eps_rel=1e-8(基线)→ 松弛 ≈ 0.5(名义值;与 §1 实测 box 违约 0.07 自洽)。

即:混合尺度下,一条 5e7 量级的 gross 行替**所有行**(包括 ±1e-6 宽的
禁交易 box)定了全局松弛——这正是 Go 版改逐行缩放判据的同一个坑,
只是从"停止判据"侧变成了"容差旋钮"侧。

**② 几何——最优在约束顶点,最后 10–14% 的余量装着 >100% 的净目标**

实测停机点:1e-2 时 delta 差 13.6%、gross 差 9.6% 即"合法"停机;
而目标从停机点 −2220 推到顶点的 +6580——**最后一段顶点推进贡献了
8800,超过全部净目标**(占比 >100%,因为中间点本身是负的)。本题 P 的
对角元 ~1e-6,目标在可行域内**几乎线性**:离顶点差 x%,目标就约差 x%
的全额(6580/杠杆未用部分 ≈ 0.65/单位净敞口,与 alpha 量级吻合)。

**③ 对偶未学会"该动谁"——中间点可以是负的**

optim 的 133 笔显著交易全部是**约束驱动**的(单股边际本身为负,
靠组合层再配置获利,见 run_real.md 分析)。325 步时对偶变量 y 还没
收敛到正确的"再配置价格",迭代点可能在**卖好 alpha、买差 alpha** 的
错误方向上移动——线性成本 cost·|x| 当场支付,而收益要方向正确+
推到顶点才兑现,于是 −2220 这种"付了成本、搬错仓位"的中间值。

**为什么 sample 的日志(§7.4 曾引用)没暴露这个**:sample 的目标主体
在 23 品种各自的**无约束最优点**(早期就到),依赖顶点推进的只有
2 笔交易;optim 的目标**全部**在顶点推进段。同一算法、同一容差,
"目标分布在哪里"决定了低精度的杀伤力。

SCS 的 19%~37% 同机理,且它只有单一 eps 旋钮,连"一侧收紧"都做不到,
外加锥形化的行数膨胀放大了 ①。

### 7.6 修复实验:把相对容差锚到"trade"尺度(逐行均衡),结果和速度

实现(cpp_admm 新增 `--scale-rows`):**列均衡**(每变量除以自身交易带
band,同时变换 P/q,返回时把 x 乘回)+ **行均衡**(每行除以自身的
target 尺度:box 行=该品种交易带、delta/gross 行=约束目标 1e5/5e7)。
行均衡使全局 eps_rel 的锚从"最大行(5e7)"变成"每行自己的尺度",即
**eps_rel=1% 从此真正等于 trade 的 1%**;列均衡保证系统条件数正常
(否则 osqp/qpalm 对 ~1e-5 行 + ~1e5 列的近奇异系统误报
primal/dual infeasible)。可行集与最优解数学不变。

**核心结论 1:均衡本身 = 免费 9~87× 提速,零精度损失**

同一容差(eps_abs=1e-6 / eps_rel=1e-8),精度完全相同(delta=100000 贴界、
gross=5e7 贴界、逐笔 vs mosek 均值 4.4/最大 167,与基线一致),但:

| 引擎 | 无均衡 | 均衡后 | 加速 |
|---|---|---|---|
| osqp | 52,625 iter / 4.35s | 2,000 iter / **0.17s** | 25× |
| qpalm | 806 iter / 0.41s | 89 iter / **0.045s** | 9× |
| proxqp | 10,520 iter / 6.75s | 60 iter / **0.078s** | 87× |

**核心结论 2:均衡后容差扫描是一条诚实平滑的 Pareto 曲线,不再崩塌**

(eps_abs 固定 1e-6,扫 eps_rel;gap 为目标相对 mosek 6580.2285 的损失%)

| eps_rel | osqp(无均衡) | osqp(均衡) | qpalm(无) | qpalm(均衡) | proxqp(无) | proxqp(均衡) |
|---|---|---|---|---|---|---|
| 1e-8 | 0.007% / 4.3s | 0.007% / 0.17s | 0.007% / 0.41s | 0.007% / 0.045s | 0.007% / 6.7s | 0.007% / 0.078s |
| 1e-5 | 0.011% / 4.3s | 0.025% / 0.073s | 0.007% / 0.42s | 0.008% / 0.043s | 0.007% / 6.7s | 0.007% / 0.066s |
| 1e-4 | 0.011% / 4.3s | 0.19% / 0.038s | 0.007% / 0.42s | 0.008% / 0.041s | 0.007% / 6.5s | 0.009% / 0.058s |
| 1e-3 | **122%** / 2.6s | 0.96% / 0.018s | 0.007% / 0.42s | 0.057% / 0.035s | 0.008% / 5.5s | 0.28% / 0.047s |
| 1e-2 | **134%** / 0.037s | 1.85% / 0.014s | **45%** / 0.17s | 5.1% / 0.020s | 0.018% / 4.5s | 2.5% / 0.039s |

- 无均衡时 osqp 在 1e-3 即崩(−1483,delta 掉到 97278、gross 4.51e7 未贴界),
  1e-2 崩得更彻底(−2219,delta 86412);均衡后最松的 1e-2 也只损失 1.85%,
  **delta 仍 99827 贴界、gross 4.945e7**——顶点保住了。
- qpalm 无均衡 1e-2 也崩(45%,gross 掉到 4.22e7);均衡后 5.1%。
- proxqp 无均衡天生抗造(1e-2 仍 0.018%)——它内部自带的均衡/求解路径不同;
  均衡后更快。
- 逐笔精度(均衡,osqp):1e-8 均值 4.4/最大 167;1e-2 均值 3075/最大 2.8e4
  (单笔差到 28% 带宽,但目标只差 1.85%——目标对"是否贴顶点"敏感,对
  个股持仓量在平坦方向不敏感)。

**核心结论 3:OSQP 自带的 `--osqp-scaled-term`(scaled_termination=1)无效**

与无均衡结果完全一致(52625 iter 不降、1e-3 照崩)。内部均衡只改善
**条件数**,不改变**容差语义**——它不把锚换成 trade 尺度。只有手动
逐行/列均衡(即 Go 版逐行缩放判据的同一思路)才能重锚。

**结论**:ADMM 家族在此问题上的正确用法 = `--scale-rows` + eps_rel 1e-8
(默认):osqp 0.17s、qpalm 0.045s、proxqp 0.078s,全部 −0.007% 精度,直接
取代未均衡版本(4.3/0.41/6.7s)。低精度换时间在均衡后是真实可选但收益
递减的曲线(最松 1e-2 也才 1.85%),不再有"1% 容差崩 122%"的陷阱。

### 7.7 均衡 = 数据预处理(零引擎改动,`optim_scaled/`)

§7.6 的 `--scale-rows` 是否只是"输入变换"而与求解器无关?把它**前移成
数据预处理**,做成 `optim_scaled/`(mose/make_optim_scaled.py,纯标准库
Python):对 input 每品种按交易带 band 缩放,constraint 每行按自身目标
尺度 σ 缩放,weight 相应折算——数学上与 §7.6 的 scaleRows() 严格等价
(已在 §7.6 用三引擎证明其独立性)。

#### 7.7.1 变换(推导可对账)

每个品种 band_i = max(1, max(|xl|,|xu|)),xl=max(minTrade,minPos−pos)、
xu=min(maxTrade,maxPos−pos)(与 model.cpp buildProblem 同式):

- **input**:pos/minTrade/maxTrade/minPosition/maxPosition ÷band;
  alpha/cost ×band;lambda/tlambda ×band²(其实只是把"列均衡"直接写进
  数据,求解视角各行各列都是 O(1))。
- **constraint**:每行 lb/ub ÷ σ,σ=max(1,|lb|,|ub|)→ 所有约束
  delta/gross 都变成 [−1,1] 或 [0,1]。
- **weight**:每行 ×band_sym ÷ σ;gross 权重变成 ~1e-5 量级、delta 权重
  ~1e-3,其余因子权重 ~0.1 量级。
- 解出的是缩放坐标系,trade 需 ×band 还原(附 scale_map.csv)。

校验链(逐行代回):NET 行 Σ w·x 变换后 = 原行 ÷σ,GROSS 同理,目标
中的 lambda·pos·x 项与 x² 项自动配平,可行集与最优解数学不变。

#### 7.7.2 MOSEK 实测(optim_scaled vs optim_inputs)

| 项 | optim_inputs(原) | optim_scaled(预处理) |
|---|---|---|
| wall time ×3 | 0.54/0.58/0.54s | 0.51/0.54/0.57s |
| MOSEK 内点迭代 | 18(0.19s 终止) | 23(0.14s 终止) |
| 目标(乘回还原) | 6580.227933(reference 6580.2285) | 6580.247014(−0.00003%) |
| gross / delta 贴界 | 4.99996e7 / +99999.7 | 4.99995e7 / +99999.7 |
| 逐笔 vs reference | — | mean 3.2 / max 50.8(≈band 的 0.001%) |

MOSEK 是内部已充分均衡的 IPM,两种输入对它**都很快**(0.54s vs 0.54s),
26%的迭代差属于 IPM 对不同 KKT 坐标系的正常路径分叉,不构成速度损失
(求解器侧总时间 0.19→0.14s 甚至略降)。**对 MOSEK,均衡既无损、无增。**
这本身就证明均衡是**引擎无关的预处理**:它不会伤害任何求解器,
收益在"无内部均衡"的开源 ADMM 家族。

#### 7.7.3 cpp_admm 交叉验证(数据级缩放 ≡ 内置 --scale-rows)

| 引擎 | 原数据+`--scale-rows` | scaled 数据(无flag) | 逐笔差 |
|---|---|---|---|
| osqp | 0.21s | 0.26s | mean 0.014 / max 0.37 |
| qpalm | 0.08s | 0.085s | mean 0.0033 / max 0.13 |
| proxqp | 0.11s | 0.12s | mean 0.0007 / max 0.08 |

(逐笔差为解还原后 vs 原坐标,机器精度级;时间噪声 ~0.05s 量级)

三引擎目标全部 6579.757 ±0.001(−0.007%),gross=5e7、delta=1e5 贴界,
与 §7.6 均衡后一致。**同一个变换既能在引擎内(--scale-rows)实现,也能
写成数据预处理;两者解完全相同,收益完全相同。** 这正面回应了
"均衡是不是作弊/只对自家引擎有效":它是纯代数输入变换,换成 osqp/
qpalm/proxqp/MOSEK 任何求解器、放引擎内或放数据里,结论不变。

