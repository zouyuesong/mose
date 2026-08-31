# admm.md §3.4(a) 预处理系列:四库 × 手动/内置两层均衡实验

- 日期:2026-08-19
- 问题:admm.md §3.4(a) 列的预处理系列(Ruiz 对角均衡 / Pock-Chambolle 预条件 /
  Giselsson-Boyd metric)在每个 baseline(四库,**开 `--scale-rows`**,不调
  eps=1e-6/1e-8)上的影响与提升。
- 新增能力(本次实现,全部在 harness 层,引擎无关):
  - `--presolve=ruiz`(迭代 Ruiz,行/列 ∞-范数→1,10 次;OSQP/PDLP 内置同款)
  - `--presolve=pc`(Pock-Chambolle α=1,行/列 ℓ₁ 范数开根,单步;
    按 ABIP+ arXiv 2209.01793 §3.5 的公式)
  - `--lib-scaling=on|off`(强制开/关各库内置预处理:osqp scaling、
    qpalm scaling、scs normalize、proxqp compute_preconditioner)
  - Giselsson-Boyd 未实现(需解子问题选 metric,PDLP/ABIP 也未用,注明)
- 数据:`/tmp/mose_run/pre/`(raw.txt + 23 个解 CSV);脚本
  `/tmp/mose_run/presolve_sweep.sh`。gap% 相对 MOSEK 6580.2285;
  posP90 = targetPosition 绝对差 p90(名义值,vs MOSEK 原数据输出)。

---

## 1. 主表:开 `--scale-rows` 下,手动档 M × 内置档 L

| 引擎 | M | L=on(内置开) | L=off(内置关) | 结论 |
|---|---|---|---|---|
| osqp | none | 2000 iter / 180ms | **1725 / 154ms** | 手动均衡后内置 Ruiz 多余,关掉 **-15%** |
| osqp | ruiz | **发散**(30s 截断) | **发散** | 双重均衡冲突(§2) |
| osqp | pc | 1800 / 170ms | 4150 / 355ms | PC 中性;叠加后不能关内置 |
| qpalm | none | **72 / 40.3ms** | 89 / 45.0ms | **内置 Ruiz 该开(基线硬编码 scaling=0 是次优)**,-10% |
| qpalm | ruiz | dual infeasible | dual infeasible | 崩 |
| qpalm | pc | 98 / 56ms | 87 / 48ms | PC 略负 |
| proxqp | none | 60 / 68.4ms | 60 / 69.7ms | 持平 |
| proxqp | ruiz | **44 iter** / 74.8ms | **发散**(120s) | 迭代全场最少(−27%),墙钟持平(单步变贵);且必须保留内置 |
| proxqp | pc | 59 / 75ms | 296 / 201ms | PC 中性;同上不能关内置 |
| scs | none | **34300 / 5.78s ✓** | 362760 / 61.2s | **SCS+均衡首次收敛!** 内置 normalize 是其命脉(关掉 21× 慢) |
| scs | ruiz | 超时 | 超时 | 崩 |
| scs | pc | 55800 / 9.4s | 超时 | PC 可行但劣于 none |

所有可行配置:gap 全部 0.007%,posP90 全部 10.2(SCS none-on 24.2、pc 12.8)——
**预处理不改变平坦面地板**(report_diff_analysis.md)。

## 2. 发现一:经典矩阵均衡不能替代问题特定均衡(M-only 全线错误解)

不开 `--scale-rows`、只用手动 Ruiz/PC(内置=default):

| 引擎 | ruiz-only | pc-only |
|---|---|---|
| osqp | 50950 iter/4.4s,**obj=−1.6e15(错误解)** | 37175/3.2s,**错误解** |
| qpalm | 821/430ms,**错误解** | 690/321ms,**错误解** |
| proxqp | 10846/7.2s,**错误解** | 超时 |
| scs | 超时 | 超时 |

迭代数与 no-scale baseline 几乎持平(osqp 50950 vs 52625、qpalm 821 vs 806),
即**经典均衡对收敛速度无帮助**,且解还错了。机理:Ruiz/PC 的经典定义只作用
**约束矩阵 A**;本问题的病根是**三跨度乘性叠加**(report_scaled.md §2.1),
其中跨度③(P 对角 λ~1e-9 vs 界 1e5~5e7)根本不在 A 里——A 均衡碰不到,
反而把 P/A 相对尺度进一步打乱。**admm.md §3.4(a)"预计是收益最大的单项
改进"在本问题上被限定为:必须以"同时变换 P/q 与行界"的问题特定形式
(= scaleRows/optim_scaled 的 band/σ)才成立;纯矩阵形式无效且有害。**

这与 ABIP+ 的对照:ABIP 的 rescale 有效是因为其对象是无目标曲率悬殊的
**LP**(P=0),我们的跨度③是 QP 特有。

## 3. 发现二:双重均衡(Ruiz on scaleRows)普遍冲突,但 proxqp 例外

osqp(1e6 发散)、qpalm(dual infeasible 证书误报)、scs(超时)——我们的
band/σ 已经把系统均衡到 O(1),再叠一层范数均衡会把小系数行(如 gross 权重
~1e-5)二次压缩,恶化数值。**唯一例外 proxqp+ruiz+内置 on:44 迭代**
(全场最少,60→44,−27%),机理推测:ProxQP 的近端 ALM 内环对残差比敏感,
再均衡改变了 ρ/σ 的有效平衡;但墙钟 75ms vs 68ms 持平(单步成本升),
**不构成实用提升**。

## 4. 发现三:各库内置预处理的正确姿态(开均衡后)

| 库 | 内置 | 建议 | 依据 |
|---|---|---|---|
| osqp | scaling=10 | **关**(`--lib-scaling=off`) | 154 vs 180ms,−15%;再均衡纯开销 |
| qpalm | scaling(基线硬编码 0) | **开**(=10) | 40.3 vs 45.0ms,−10%;**修掉基线次优** |
| proxqp | compute_preconditioner | 保持开 | 关掉则 ruiz/pc 叠加全崩,none 持平 |
| scs | normalize | **必须开** | 关掉 21× 慢(61s);这是 SCS 可用性的命脉 |

## 5. 发现四:SCS 被 scaleRows 救活

无均衡:1e6 迭代顶格不收敛(§1.1 家族表)。开均衡 + 内置 normalize:
**34300 iter / 5.78s,Solved,gap 0.007%**。均衡对 SCS 不是提速项,是
**能不能解**的门槛项。这与三跨度机理一致:锥投影 ADMM 对条件数最敏感。

## 6. 更新后的最优配置(baseline 修订建议)

| 引擎 | 配置 | solve | vs 原 baseline |
|---|---|---|---|
| qpalm | `--scale-rows --lib-scaling=on` | **40.3ms** | 45ms → −10% |
| proxqp | `--scale-rows`(不变) | 68.4ms | — |
| osqp | `--scale-rows --lib-scaling=off` | **153.5ms** | 180ms → −15% |
| scs | `--scale-rows`(内置默认开) | **5.78s** | 不收敛 → 可用 |

PC 叠加无正收益;Ruiz 叠加仅 proxqp 迭代好看、墙钟无益——**两档手动
范数均衡在问题特定均衡之上均无实用价值**,§3.4(a) 系列在本问题的最终
形态就是 scaleRows/optim_scaled 本身。

## 7. 复现

```bash
cd /home/logic/code/mose/cpp_admm
# 手动预处理 + 内置开关(全部默认 eps)
./build/csvport_cpp opt -i ../optim_inputs/optim_inputs.20260520 \
  -c ../optim_inputs/optim_constraints.20260520 \
  -w ../optim_inputs/optim_weights.20260520 -o /tmp/o.csv \
  --engine=qpalm --scale-rows --lib-scaling=on      # 新最优 40ms
# 全矩阵
bash /tmp/mose_run/presolve_sweep.sh                # ~10min(含发散超时)
```