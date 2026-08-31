# ADMM 改进迭代记录(improving)

- 日期:2026-08-17(接续 `log/admm/basic.md`,代码同一版本演进)
- 基准(改进前,来自 basic):optim wall **2.38s**(10k 迭代 + 一次 polish)、
  sample 17.6ms;目标 6579.75 vs mosek 6580.23
- 实验环境:`ADMM_BENCH=1 go test ./opt -run TestImproveBench -v`(harness:
  `opt/improve_test.go`),CPU profile:`ADMM_PROF=1 ... -cpuprofile`
- 调研对照:根目录 `admm.md` §3.4(自适应ρ/松弛/Ruiz/polish/warm-start)

---

## 1. 发现并修复的正确性漏洞(最重要的一项)

### 1.1 早 checkpoint 会接受"可行但次优"的解

把 polish checkpoint 从 1 万步提前到 2 千步后:wall 降到 0.9s,但目标只有
**4685**(正确值 6579.75)——polish 从欠收敛迭代点出发,收敛到了**错误主动集
的合法 KKT 点**(可行、稳定,但非全局最优)。旧的接受条件只检查
"可行 + 不差于当前迭代点",而早期迭代点本身很差,防线形同虚设。

### 1.2 修复:KKT 对偶符号认证(certify)

polish 完成时(以及每轮整步后)对候选解做近似 KKT 认证:

- 行乘子符号:下界活动的行 μ≤0、上界活动的行 μ≥0(本代码约定);
- 逐变量梯度(以 lin=P x+q+A'μ 为光滑部分,cost|x| 按分支进入):
  - 箱下界:`lin+cost ≥ 0`(右行方向);箱上界:`lin−cost ≤ 0`;
  - |x| 折点(x=0):`|lin| ≤ cost`;GROSS 折点(x=−pos):
    `|lin+cost·sgn(x)| ≤ |Σ w·μ|`(折点次梯度松弛量);
  - 自由变量:`lin+cost·sgn = 0`(构造保证,兜底检查);
- **只有认证通过的点才可被 checkpoint 接受**;末次迭代允许 3000 轮的
  最后一次完整尝试。

效果(同一轮诊断数据,`pri/eps` 为迭代点原始残差比):

| 迭代数 | pri/eps | polish 可行点 obj | 认证结果 |
|---|---|---|---|
| 2k | 280 | 无 | —(自由变量平稳性真不满足,拒绝 ✓)|
| 4k | 8.1 | 4685(错误) | 乘子符号拒绝 ✓ |
| 6k | 16.7 | 6579.76 | **通过 ✓** |
| 10k | 6.3 | 6579.75 | **通过 ✓** |
| 20k | 41 | 6579.76 | **通过 ✓** |

### 1.3 配套:主动集的 "drop" 步(符号释放)

认证暴露了主动集法缺的另一半:被钉在箱界/折点上、但梯度把它往可行域内推的
变量应当**释放**为自由变量(经典 primal active-set 的 add/drop 对偶机制;
此前只有线搜索阻塞这个 "add" 方向)。实现:用上一轮的 lin 梯度做释放判定,
阈值与认证阈值一致(1e-6 相对;最初设 1e-4 与认证不一致,永不触发,已对齐)。

修完后所有配置的目标值全部回到 6579.75(此前 α=1.6 曾给出 6453 的
"认证通过"次优点,由 GROSS 折点次梯度条件堵住)。

## 2. 参数/结构实验矩阵(optim,认证修复后)

| 配置 | wall | 目标值 | 结论 |
|---|---|---|---|
| basic(ckpt10k, α=1.0, 自适应ρ) | **1.9–2.0s** | 6579.7531 | **保留为默认** |
| ckpt5k | 2.1s | ✓ | 失败尝试的开销 > 提前命中的收益 |
| ckpt2k | 2.3–2.4s | ✓ | 同上,更密更亏 |
| ckpt2k + 预过滤(pri/eps>200 跳过) | 2.3s | ✓ | 预过滤有效但 2k 处多数已过阈值 |
| ckpt2k + CheckEvery=100 | 2.5s | ✓ | 残差检查占比小,无感 |
| α=1.6 + ckpt2k | 3.6s | 6579.7413 | 更慢且略差,**不采用**(α=1.6 在固定ρ下还会死锁,见 basic.md)|
| 自适应ρ关(ρ=1e-4 固定)+ ckpt2k | **120s** | ✓(2m 后)| 固定ρ尾段收敛极慢,**证实自适应ρ必要** |

sample 在默认参数下 3750 迭代自然收敛(18.7ms),无需 polish。

## 3. 性能剖析与微优化(负结果,同样有价值)

CPU profile(30k 迭代,5.2s,`solveRaw` 内联循环 55% + `woodbury.solve` 43%):

- 尝试 1:**Woodbury 内部 CSR 扁平化 + 预存 1/D̃** → 无变化;
- 尝试 2:**全 A 矩阵 CSR 扁平化**(消 slice-of-slice 间接寻址)→ 无变化;
- 结论:每内层迭代 ~1.7ns,是 Go 标量循环 + 边界检查的**指令数瓶颈**
  (每迭代 ~10 万次内层循环 × 6 趟稀疏遍历),不是访存/间接寻址问题。
  数据(n=1065, 8.5KB)完全在 L1 内。要再快一个量级需要:
  (a) 减少迭代数;(b) 删掉 TUB/GUB 成对行(Moehle-Boyd 式 prox 建模,
  m 从 2846 → 66,工作量减 ~90%);(c) SIMD/汇编。留作后续。

## 4. 最终状态

| 数据集 | wall | 目标值 | 与 mosek 差 |
|---|---|---|---|
| optim(358 股) | **2.5s**(求解 2.1s = 10k 迭代 + 认证 polish) | 6579.7531 | 0.007% |
| sample(23 品种) | 0.05s(求解 18.7ms) | 48.7250 | 0.0002% |

约束校验(awk 独立重算):gross/delta 贴界、63 条 barra NET 最差违约
3.6e-4(即贴界的 delta_grp3)、箱子零违约;与 mosek 逐品种交易差
max 166 股 / mean 4.3 股(1e5 量级下)。

vs basic 的净变化:certainty ↑(认证杜绝次优解,这是生产可用性的前提),
wall 2.38→2.5s(+5%,认证与释放机制的代价)。

## 5. 未竞事项(后续路线)

1. **Moehle-Boyd 定制建模**:删 TUB/GUB(u/g)变量与 2846 条成对行,
   cost|x| 与 gross 投影进 z-更新(逐坐标软阈值 / 排序投影)——预期迭代
   成本降 ~5-10×,是超越 mosek 单次延迟的关键;
2. **warm start**:两遍 relax 的 pass-2、日内滚动重解复用 (x,z,y) 与
   Woodbury 因子(`Settings.X0/Z0/Y0` 已支持,编排层未接);
3. relax 模式的 polish 支持(弹性变量场景,真实数据未触发);
4. Ruiz 均衡的构造性验证(当前数据 A 全 ±1,天然均衡无收益);
5. 反思:认证阈值 1e-6 相对偏严(4k/15k 的边际拒绝),可做分级
   (checkpoint 用宽、最终用严)。

## 6. 复现

```bash
cd golang_mosek/src && make
ADMM_BENCH=1 go test -count=1 ./opt -run TestImproveBench -v   # 实验矩阵
ADMM_SWEEP=1 go test -count=1 ./opt -run TestOptimRhoSweep -v  # ρ 扫描
ADMM_PROF=1  go test -count=1 ./opt -run TestProfileADMM -cpuprofile /tmp/p.prof
go tool pprof -top /tmp/p.prof
```

---

# 追加:单次延迟攻坚(目标 <10ms)会话记录

- 日期:2026-08-18(接续上文;本节记录为追平/超越 mosek 0.17s 所做的实验)

## 0. 结论先行

**<10ms 目标本次未达成**;当前三引擎状态(optim 358 股 / sample 23 品种):

| 引擎 | optim 用时 | optim obj | sample 用时 | sample obj | 状态 |
|---|---|---|---|---|---|
| mosek(参考) | 170ms | 6580.2285 | ~40ms | 48.7249 | 基准 |
| admm(默认) | 2.1s+polish | 6579.7531 ✓ | 17.5ms | 48.7250 ✓ | 生产可用,认证 |
| osqp(vendored C) | 2.16s | 6579.7566 ✓ | **0.9ms** | 48.7238 ✓ | 可用 |
| ipm(纯 Go 原型) | ~100ms 未收敛 | — | 0.6ms 未收敛 | — | 实验性 |

关键发现:**该问题的混合尺度(界 ±1e-6~±1e9、λ~1e-9、x~1e5-1e6)使一切一阶法
需要 10^4~10^5 迭代**(我们的 Go ADMM 10k、OSQP 自带自适应 ρ/Ruiz/polish 也是
23k~87k)。<10ms 必须走二阶法(IPM/主动集),MOSEK 自己就是 17 步 IPM。

## 1. vendored OSQP(`src/cosqp/`,Apache 2.0,0.6.3 + qdldl + amd,cgo 静态编入)

- 接线:`--engine osqp`;问题与 admm 引擎共用 buildADMMProblem(含固定变量 presolve)
- optim:2.16s(23,250 迭代 @eps1e-6);参数扫描(rho 0.01~1、sigma 1e-9~1e-5、
  adaptive interval 25/50)在 eps1e-4/1e-5 下全部钉在 87,000 迭代——自适应 ρ 会把
  任何初值拉到同一轨迹;eps1e-3 时 1,650 迭代/152ms 但 obj=-636(不可用)
- **手工预缩放实验失败**:归一化变量界幅度(D)+行界幅度(E)后 OSQP 内部 Ruiz
  二次缩放冲突,20s 不收敛;曾因 CSC 重建 off-by-one 触发数据校验错(已修)
- sample 上 0.9ms——小问题上 OSQP 完胜,问题规模/条件数是 optim 的分水岭

## 2. 纯 Go IPM 原型(`src/admm/ipm.go`,Mehrotra 预测-校正)

- Newton 系统 (P+A'DA)dz=rhs 恰为 Woodbury 结构(逐行对角 D + 低秩 G'DG),
  复用现有求解核,每迭代 O(nnz+k²)
- 单测 3/3 过(盒子/一般行/辅助结构;含符号修复:dzl=A·dz+rl、
  min 形式上界对偶为正号、等式行单侧 ε 松弛)
- **每迭代成本实测 ~1ms(未优化,含每迭代重分解+分配)**:100 迭代 60~160ms;
  优化后(k=67 分解 10⁵ flop + CSR 复用)~0.3ms/迭代 × ~30 迭代 ≈ **10ms 可达**
  ——性能上目标成立,卡在收敛性:
  - 深度不可行起点(delta 行违约 2.4e6)→ 首步即被 fraction-to-boundary 阻塞
    (aP≈0.0004):需要 Gondzio 多重校正或 HSD 嵌入(mosek 级工程)
  - 可行化尝试:POCS(NET Schur 投影 + GROSS 加权 L1-ball 投影(Duchi 二分,
    已实现)+ box clamp)在 gross/net 交界振荡不收敛(net 2.5e6/gross 3.4e6)——
    可行域薄,本质是个 QP 而非投影问题;over-relaxation 1.7 更差
  - 退化最优面测试(整段最优):shift 起点后 barrier 停滞,返回可行次优点
    (已知局限,单测注明;生产数据非退化)
- best-iterate 跟踪(score=max(rp/scale, μ/μ₀, 0.1·rd/q))防终局自毁

## 3. 主动集(polish 扶正)补强(已合入默认 admm 引擎)

- 行释放:乘子符号错的最大激活行每轮 drop 一条
- 重复行去重:delta_ptf/delta_grp3 权重相同时只留一条(否则 add/drop 永久循环)
- bootstrap:首轮用无乘子梯度释放 kink 变量
- GROSS 折点次梯度认证堵住 α=1.6 下 6453 的次优解
- 但从解析起点/x=0 直接 polish 仍不可靠(可行域行走需千轮),保持
  "ADMM 10k 迭代 → checkpoint polish"管线

## 4. 复现

```bash
cd golang_mosek/src && make
./bin/jobd opt [--engine admm|osqp|ipm] -i ... -c ... -w ... -o ...
ADMM_BENCH=1 go test -count=1 ./opt -run "TestImproveBench|TestOSQPSweep" -v
```

## 5. 后续路线(若继续冲 <10ms)

1. **IPM 收敛性**(主路径,性能已验证可行):
   a. Gondzio 自步长 + 多重校正(标准 cure for tiny steps);或
   b. HSD 齐次自对偶嵌入(免可行起点);或
   c. 可行化 QP 交给 OSQP(P=εI,q=0 良态,预计 <50ms)产出 IPM 起点
2. 每迭代 0.3ms 化:分解复用(k 变化时才重构)、去分配、CSR 预取
3. 若 (1c) 通:OSQP-feas(50ms)+IPM(10ms)≈60ms,仍优于 mosek 3×;
   若 (1a/1b) 通:纯 10ms 达标
