# admm.md §3.4(b)/(c):自适应 ρ/残差平衡 + 加速/松弛旋钮实验

- 日期:2026-08-19(接 report_presolve.md 的最优 baseline)
- 对象:§3.4(b) 自适应 ρ 与残差平衡、(c) 加速与 restart 中**设置层可暴露**的旋钮;
  全部在 `--scale-rows`、默认 eps(1e-6/1e-8)、各库最优 lib-scaling 下测
  (osqp=off、qpalm=on、proxqp/scs=default)。
- 新增 CLI:`--rho= --sigma= --alpha= --adaptive-rho=on|off`(引擎映射:
  osqp→rho/sigma/alpha/adaptive_rho;qpalm→sigma_init;proxqp→default_rho/
  default_mu_in/alpha_bcl;scs→rho_x/alpha/scale)。
- (c) 族中需改源码的深度项(Nesterov/GLADMM 外推、reflected-Halpern 锚定、
  PDLP 式 restart)四库均未暴露接口,本报告不做源码 fork,结论见 §5。
- 数据 `/tmp/mose_run/bc/`,脚本 `/tmp/mose_run/bc_sweep.sh`;
  gap% 相对 MOSEK 6580.2285。

---

## 1. 主结果(相对各自 baseline 的变化)

### qpalm —— **sigma_init=1:40.6 → 33.3ms(−18%),全场新最优**

| sigma_init | iter | solve |
|---|---|---|
| 0.1 | 63 | 35.5ms |
| **1** | **58** | **33.3ms** |
| 2 / 3 / 5 | 64 / 61 / 61 | 35.9 / 34.7 / 34.1ms |
| 10 | 59 | 34.2ms |
| 100 | 80 | 47.0ms |
| (base=默认) | 72 | 40.6ms |

σ_init=1 是个宽平台(1~10 都 ~34ms),两侧衰减。**新生产配置:
`--engine=qpalm --scale-rows --lib-scaling=on --sigma=1`**。

### osqp —— adaptive ρ(残差平衡)是命门,其余旋钮钝感

| 旋钮 | 结果 |
|---|---|
| **adaptive-rho=off**(默认 ρ=0.1) | **超时(>150s)**;ρ=1 时 534,825 iter/43.5s;ρ=10 时 53,300/4.1s——**4 个数量级惩罚** |
| adaptive-rho=on(默认),扫 ρ ∈ {1e-3..1000} | 地形崎岖:ρ=100 最好 1750/146.6ms(base 1725/153.5),ρ=30/200/500/1000 全在 2100~4200 iter——**自适应 ρ 在补偿初值,扫 ρ 无稳定收益** |
| σ ∈ {1e-8..1e-4} | 迭代钉死 1725,零敏感 |
| α(over-relaxation)∈ {1.0,1.3,**1.6 默认**,1.9} | 3125 / 2175 / **1725** / 3200 iter——默认已是该族最优 |

### proxqp —— 全部默认即最优,动则崩

| 旋钮 | 结果 |
|---|---|
| mu_in ∈ {1e-7..1e-3} | 全劣于默认(60 iter/67ms;最好替代 mu=1e-3 也要 105/121ms) |
| rho ∈ {1e-7..1e-3} | 无感(60~61 iter) |
| alpha_bcl ∈ {1.0,1.2,1.4} | **全部超时**——BCL 加速参数脆弱,不可动 |

### scs —— 默认即最优

| 旋钮 | 结果 |
|---|---|
| alpha ∈ {1.0,1.5,**1.8 默认**} | 51380 / 34300 / 34300 iter;α=2.0 直接失败 |
| scale ∈ {0.1,1,**5 默认**,20,100} | 464100 / 41540 / **34300** / 56600 / 271500——倒钟形,默认在顶 |
| rho_x ∈ {1e-4,1e-3,1e-2,1} | 34180~42360,弱敏感 |

## 2. 结论

1. **(b) 族的判决**:残差平衡(adaptive ρ)对一阶 ADMM **是生死项**——
   osqp 关掉它从 0.15s 变 43s+(287×),比任何其它旋钮都大一个数量级;
   但它已经默认开着,设置层无增量可挖。**QPALM 的 sigma_init 是唯一真
   增量:−18%,33.3ms 新全场最优**。ProxQP/SCS 的 (b) 旋钮默认已在最优。
2. **(c) 族的判决**:over-relaxation α 三库(osqp 1.6 / scs 1.8)默认即
   顶点,proxqp 的 BCL 参数动了就崩——**设置层无增量**。
3. 与 §3.4(a) 实验合流的最终排行榜(solve ms,默认 eps,gap 全部 0.007%):

| 引擎 | 配置 | solve | 来源 |
|---|---|---|---|
| **qpalm** | `--scale-rows --lib-scaling=on --sigma=1` | **33.3** | (b) |
| proxqp | `--scale-rows` | 67 | — |
| osqp | `--scale-rows --lib-scaling=off` | 153.5 | (a) |
| scs | `--scale-rows` | 5776 | (a) 可用性 |
| MOSEK | — | 140–190 | 参考 |

## 3. 机理注记

- osqp 的 adaptive ρ 就是 §3.4(b) 引的 Boyd residual balancing 工程化:
  每 25(默认)迭代按 primal/dual 残差比调 ρ。它对本问题如此关键,因为
  均衡后各行 O(1),残差比能被干净地读出——**(a) 的均衡是 (b) 生效的
  前提**(无均衡时锚被 5e7 行劫持,残差比本身就是错的)。
- qpalm 的 sigma_init 是其近端罚参数初值:均衡后 KKT 特征值集中,σ=1
  恰落在平台上;过大(100)则初期过罚,过小(0.1)则初期欠罚。
- proxqp alpha_bcl 超时机理未深挖(BCL 内层与主动集更新耦合),标记
  **勿动**。

## 4. 复现

```bash
cd /home/logic/code/mose/cpp_admm
./build/csvport_cpp opt -i ../optim_inputs/optim_inputs.20260520 \
  -c ../optim_inputs/optim_constraints.20260520 \
  -w ../optim_inputs/optim_weights.20260520 -o /tmp/o.csv \
  --engine=qpalm --scale-rows --lib-scaling=on --sigma=1   # 33.3ms 新最优
bash /tmp/mose_run/bc_sweep.sh                              # 全矩阵 ~8min
```

## 5. (c) 族深度项不做源码 fork 的理由

reflected-Halpern/PDLP-restart/GLADMM 需要改 ADMM 内循环(vendored OSQP
C 源码):①工程量级 = fork + 维护一个私有求解器;②目标引擎 osqp 在
排行榜上已被 qpalm 支配(154 vs 33ms),优化它的内循环收益上限不改变
选型;③OSQP 的 adaptive ρ 已含 restart 的工程本质(残差失衡即重锚)。
若未来要做,应直接在 **Go 自研 ADMM**(我们拥有源码)上实现,而不是
fork OSQP——那是 plan.md 之外的路线。