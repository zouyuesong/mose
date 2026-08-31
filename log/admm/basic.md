# ADMM 基础款跑通记录(basic)

- 日期:2026-08-17
- 代码:`golang_mosek/src/admm/`(纯 Go,无 CGO/license)+ `opt/backend_admm.go`
- 对照:MOSEK 9.3.22(`bin/csvport` 原版二进制 + `~/mosek/mosek.lic`),
  求解器内部 0.17s,进程 wall 0.58–0.64s(log/run_real.md,未改动)
- 调研报告:根目录 `admm.md`

---

## 1. 实现概要

### 1.1 算法(OSQP 式分裂,Stellato et al. 2020)

```
min ½z'Pz + q'z  s.t.  l ≤ Az ≤ u        (P 对角 PSD)
x̃ = (P+σI+ρA'A)⁻¹(σx − q + ρA'(z−y))     ← Woodbury 精确解,见下
z⁺ = Π_[l,u](αAx̃+(1−α)z+y)
y⁺ = y + αAx̃+(1−α)z − z⁺
```

默认参数:ρ=1e-4(自适应开)、σ=1e-6、α=1.0、epsAbs=1e-6、epsRel=1e-8、
停止判据为**逐行/逐坐标缩放容差**(见 §3.3)。

### 1.2 结构化 KKT 求解(本项目专属红利)

建模器保证行布局:`[0:GeneralStart)` 为 ±1 成对行(box 镜像、TUB 的 u±x、
GUB 的 g∓x——成对外积在 A'A 中相消,只贡献对角),之后为 k 条任意稀疏行
(组合约束/弹性行,k=67 for optim)。于是

```
P + σI + ρA'A = D + ρG'G    (D 对角,G 为 k×n)
```

用 Woodbury 精确求解:每迭代 O(nnz(A)) + 一次 k×k Cholesky 回代,
**不需要任何稀疏分解器**;自适应 ρ 触发重分解时 k³/3 ≈ 10⁵ flop,近乎免费。
单测含 Woodbury vs 稠密消元对拍(maxdiff 4.4e-15)。

### 1.3 与项目主体的接线

- `opt/solver.go` 重构为后端注册表;`Solve(engine, ...)` 两遍 relax 编排不变
  (STEP1 预处理、`constraintInBreachWithZeroTraded` 改由数据侧直接计算);
- `--engine admm`(默认,纯 Go)| `mosek`(build tag `mosek`,原算法逐行保留,
  需 CGO 头文件/库,本机无头文件未编译,参考结果由 `bin/csvport` 原版产出);
- 原 `Makefile` 构建 `./main/main.go` 在加入 proctitle.go 后本就编不过
  (`bin/jobd` 从未存在过),已改为 `./main`。

### 1.4 建模(presolve 增强)

`buildADMMProblem` 与 mosek 后端逐行对应(x/u/g/t 变量、四类约束、relax 弹性),
另做**固定变量消去**:交易 box 数值闭合(宽 ≤1e-5·max(1,|xl|))的品种直接
固定在 0 并把贡献折进常数(optim 有 3 只 λ=1 的禁交易标志股,见 §3.4)。

---

## 2. 结果对比

### 2.1 optim 真实数据(358 只东证股票,67 约束,23628 权重行)

| 指标 | MOSEK(参考) | ADMM 基础款 |
|---|---|---|
| 求解器内部耗时 | 0.17 s | 1.94 s(checkpoint@10k 迭代)|
| 进程 wall time | 0.58–0.64 s | **2.38 s**(含 CSV 加载/打印)|
| 峰值内存 | ~40 MB | ~30 MB(纯 Go,无 so/license)|
| 目标值(最大化) | 6580.2285 | 6579.7531(**差 0.007%**)|
| gross_ptf | 4.99996e7(贴界) | 4.999992e7(贴界)|
| delta_ptf / delta_grp3 | +99999.7(贴界) | +100000.0(贴界)|
| 63 条 barra NET | 0 违约 | 最差 3.6e-4(即 delta_grp3 贴界)|
| 逐品种 box 违约 | 0 | 0 |
| 逐品种 trade 差 | — | max 166 股(3697.T),均值 4.3 股(量级 1e5)|

补充:ADMM 迭代本体在 10k 步时 priRes=4.7、duaRes=3e-4,直接返回不可用;
**主动集 polish(每 1 万步 checkpoint 尝试一次)把解抬到上表精度**,polish
本身仅 ~0.02s。

### 2.2 sample 数据(23 品种)

| 指标 | MOSEK | ADMM 基础款 |
|---|---|---|
| wall time | 0.04–0.06 s | **~0.05 s**(求解 17.6ms / 3750 迭代)|
| 目标值 | 48.7249(vendored 9.3 实测) | 48.7250 |
| 显著交易 | USD_10Y5Y +5075.5 / USD_4Y2Y +389.94 | 一致(差 0.13%/同量级噪声)|

sample 无需 polish 即达标(逐行容差判据下自然收敛)。

---

## 3. 调试过程中发现并修复的问题(按发现顺序)

1. **proctitle 覆写 argv 的潜伏 bug**(原仓库):`setProcTitle` 清零 argv
   内存,而 pflag 解析出的字符串值是 os.Args 的**子串共享**,路径随之被毁
   ("open : invalid argument")。原版 `bin/csvport` 未编译 proctitle 故从未
   暴露。修复:masking 前对四个路径 `strings.Clone`。
2. **Woodbury 对角双重计入**:`dtilde` 误把 general 行的对角平方也累进 D̃,
   与低秩项 ρG'G 重复。修复:D̃ 只含成对行贡献。
3. **Woodbury 回代公式错误**:`−ρD⁻¹G's` 误写成逐元素 `−ρs·G·(r/D)`。
   由 vs 稠密对拍单测捕获(该单测从此保留)。
4. **对偶残差的 ρ 因子**:迭代用 OSQP 未缩放 y(=真对偶/ρ),残差必须是
   `Px+q+ρA'y`;漏掉 ρ 会把收敛中的解误判为已收敛(小测试上对偶残差恒差
   ρ 倍)。另修复过一次反向错误(改迭代本身)——单测与两变量手算轨迹定位。
5. **固定 ρ 的尺度失配**:sample 要 ρ≈1e-6(225 步收敛),optim 要 ρ≈1e-4;
   单一固定值无法跨数据集。基础款引入 OSQP 残差平衡自适应 ρ(±5×,
   触发 Woodbury 重分解)。
6. **全局 max 范数停止判据失效(混合尺度)**:optim 中禁交易股 box=±1e-6,
   全局容差 ~0.5(由 5e7 量级的行决定)下,这些股能"合法"偏离 box 0.5,
   经 −λ·pos·x 项(λ=1,pos~3e5)虚增目标 ~21 万。两层修复:
   (a) 停止判据改逐行/逐坐标缩放容差;(b) 固定变量 presolve 消去这些品种。
7. **polish 主动集的四个数值陷阱**:
   - 等式 KKT 直接稠密求解奇异(optim 的 delta_ptf 与 delta_grp3 行完全
     相同→秩亏):改 Schur 形式 `S = Weq P⁻¹ Weq' + δI`(行归一化 + δ 正则);
   - ADMM 迭代点在单边 box 外的 1e-1 噪声被误判为自由变量:分类时越界即
     吸附到界;kink 判定加绝对地板 1.0;
   - 步长比率出现负值(x~3e5、d~1e5 下 fp 舍入 ~1e-6):每步后显式夹回
     box;行方向零阈值改相对 1e-9·max(1,|val|);
   - GROSS 折点锯齿:|pos+x| 符号折叠的线性化在迭代点跨折点时失效,
     gross_ptf 在 5.02e7↔5.005e7 振荡。修复:线搜索把折点(x=−pos)作为
     阻塞者 + 分类时把贴近折点(1e-2 相对)的变量钉在折点上。
8. **polish 可靠性**:同一算法从不同迭代数起点的成功率不稳定(20k✓/50k✗/
   100k✗→✓/200k✗)。工程解法:**每 1 万步 checkpoint 尝试 polish,成功即停**
   ——optim 在 10k 步命中,总耗时从 17s 降到 2.4s,且解质量一致。

## 4. 结论与遗留

- **结果相似性目标达成**:optim 目标差 0.007%,约束全部满足,逐品种解一致;
- 基础款在 optim 冷启动上仍比 MOSEK 慢(2.4s vs 0.6s wall)——单次冷解不是
  ADMM 的主场,且当前每迭代 ~190µs 的实现有大量优化空间(见下);
- 遗留(进入 improving 阶段):
  1. 迭代成本:rhs/Ax/solve 三趟全行扫描可合并,G'v/G's 可预取 CSR;
  2. checkpoint 间隔 1 万步可自适应(5k/2k 也可能命中);
  3. α=1.6 over-relaxation(固定 ρ 下曾冻结,自适应 ρ 下待重测);
  4. Ruiz 均衡(已实现开关,本数据 A 全 ±1 无收益,待构造坏尺度数据验证);
  5. warm start:两遍 relax 的 pass-2、日内滚动重解;
  6. relax 模式路径未在真实数据触发过(optim/sample 账本均未破线),
     弹性变量的 polish 支持未写(当前 relax 下仅靠 ADMM 原生收敛)。

## 5. 复现命令

```bash
cd golang_mosek/src
make                                    # 或: go build -o bin/jobd ./main
./bin/jobd opt -i sample/input.csv -c sample/constraint.csv \
                -w sample/weight.csv -o /tmp/out.csv            # sample
./bin/jobd opt -i ../../optim_inputs/optim_inputs.20260520 \
                -c ../../optim_inputs/optim_constraints.20260520 \
                -w ../../optim_inputs/optim_weights.20260520 \
                -o /tmp/out2.csv                                 # optim
# mosek 对照(需 license):
LD_LIBRARY_PATH=./lib ./bin/csvport opt -i ... -o ref.csv
# sweep 实验(开发测试):ADMM_SWEEP=1 go test ./opt -run TestRhoSweep -v
```
