# nativeqpalm 移植步骤记录(QPALM → 零依赖 C++ 引擎)

- 日期:2026-08-24;代码 `cpp_admm/src/engines/native_qpalm_engine.cpp`(~700 行,仅 Eigen 头依赖)
- 目标:把 QPALM 1.0(proximal ALM + 半光滑 Newton)做成无库 native 引擎,
  Newton 系统用问题特定结构(品种块 + rank-k Woodbury)替代 LADEL/CHOLMOD
- 结果:**19.7ms vs qpalm 库 34.5ms(1.75×)**,iter 86,polish 兼容

---

## 1. 源码调研清单(逐文件取证)

| 文件 | 提取内容 |
|---|---|
| `qpalm/src/qpalm.c:401-730` | 主循环三分支:终止 / 子问题收敛→outer 更新 / inner Newton 步 |
| `qpalm/src/iteration.c` | compute_residuals(Axys/z/yh/dphi)、initialize_sigma、update_sigma、update_gamma、update_primal_iterate |
| `qpalm/src/newton.c` | set_active_constraints(Axys 贴界判定)、enter/leave、FACTORIZE_KKT 路径(KKT 装配 + 迭代精修) |
| `qpalm/src/linesearch.c` | 精确分段线搜全式:delta/alpha 断点、J=P xor L 集合、a/b 累计、排序走断点 |
| `qpalm/src/termination.c` | is_solved(pri/dua 双残差 vs eps_abs+eps_rel·norm)、check_subproblem_termination(‖dphi‖≤eps_in) |
| `qpalm/src/solver_interface.c` | qpalm_form_kkt:[[Q+1/γ, A_act'];[A_act, −diag(σ⁻¹ 或 1)]] —— **非 active 行对角是 −1 不是 −σ⁻¹**(数值有界的关键) |
| `qpalm/include/constants.h` | 默认参数:σ_init=20、θ=0.25、δ=100、σ_max=1e9、γ=1e7、ρ=0.1、eps_in=1/1 |
| `qpalm/include/qpalm.h` | warm_start API(有)但无 update_data API(滚动场景不可用,native 版可自建) |

## 2. 算法骨架(移植后的 C++ 结构)

```
outer (proximal ALM,对 Φ(x)=f(x)+Σφ_σᵢ(bᵢ−aᵢ'x)+‖x−x0‖²/2γ):
  每轮: compute_residuals → 判全局终止(pri/dua)
    ├ 子问题收敛(‖dphi‖∞≤eps_in)→ outer 更新:
    │    y←yh;σ 对"违约且 active"行 ×max(1,δ| r |/(‖r‖∞+ε)) 封顶 σ_max;
    │    eps_in ← max(eps, ρ·eps_in);x0←x;γ 不动(默认 γ=γ_max)
    └ 否则 inner: 半光滑 Newton 一步
         active_i = (Axys_i ≤ bmin) ∨ (≥ bmax),Axys = Ax + y./σ
         解 (Q + 1/γ I + Σ_active σᵢaᵢaᵢ') d = −dphi
         τ = 精确分段线搜(断点排序走符号翻转)
         x += τd;Qx/Ax 增量维护
```

## 3. Newton 求解器(结构利用,替代稀疏 LDL)

M = diag(Q)+1/γ+Σ_active σᵢaᵢaᵢ′ 拆成:
- **品种块**:active 的 box/TUB/GUB 结构行只碰单品种变量 → union-find
  (over active pair 行)聚成 343 个 ≤3×3 块,两遍连续布局 + varSlot O(1) 定位,
  O(nnz) 单遍装配,分尺寸解析求逆(1×1/2×2 闭式、3×3 Eigen)
- **rank-k Woodbury**:active general 行(≤67 条,索引连续)→
  S = I + W·Dinv·W′ 的 k×k Cholesky(全宽 Dinv·w 行)+ 缓存 Dinv·W′
- 求解:Dinv b → W·v1 → chol 解 → 减 Dinv·W′·rhs;每 active-set 变化重建一次

## 4. 移植五坑(全部源码取证修复,坑坑实测定位)

| # | 坑 | 症状 | 定位手段 | 修复 |
|---|---|---|---|---|
| 1 | **σ₀ 不是裸 sigma_init** | 初始 dua 1710 vs 库 1388 | 与库 verbose 首行对拍 | initialize_sigma 公式:σ₀=clip(σ_init·max(1,\|f0\|)/max(1,0.5·dist2),1e-4,1e4) |
| 2 | **块成员单遍 append 交错** | 首步 d 炸 1e15,τ~1e-12 死循环 | selftest(\|M·d−b\|)+最差块 dump(Msub 对角 0.0008 vs 引擎 1e-7)| 两遍布局:先分块号再桶排,成员物理连续 |
| 3 | **4×4 框架求逆含零槽** | det≡0 → 全块 fallback 1e12 → 发散 | 同上,数值差恰 1e7=1/γ | 只对左上 sz×sz 求逆;分尺寸闭式 |
| 4 | **Dinv·w 按行切片截断** | dua 炸 1e22 | sparse-S 版自测 | 块逆非对角,Dinv·wᵢ 泄漏到切片外的块友(u_j/g_j)→ 必须全宽 |
| 5 | **γ 项维护** | prox 项错位 | 对照 QPALM "Qx contains Qx+1/γx" 注释 | Qx 增量含 1/γ;df=Qx+q−x0/γ 在 x=x0 时自消 |

另:全量验证用过的取证工具(test/nqp_unit.cpp:块+Woodbury vs dense LU,
1e-13 一致)保留可复跑。

## 5. 性能优化:32.2 → 19.7ms

| 步骤 | 时间 | 内容 |
|---|---|---|
| 基线(正确版) | 32.2ms | tLS=17.5(qsort 全排序断点)+ tBuild=11.7 |
| ① std::sort + walk 探针 | 27.3ms | 实测走断点 **avg=41/max=458**(断点 2000+)→ 全排序浪费 |
| ② **min-heap 弹出式线搜** | 23.5ms | make_heap O(nL)+弹到首个正梯度;结果逐位同 |
| ③ 布局固定化 | 22.2ms | union-find 用全量结构行固定一次(active 变化不再重排) |
| ④ 合并 Dinv·w 双循环 | 20.5ms | wdFull 与 dinvWt 同遍块应用 |
| ⑤ 重建预算(enter/leave≤阈值不重建) | **发散,回退** | semismooth Newton 不容忍 inexact 方向(库的 rank-1 update 是精确同步,非"用旧矩阵") |
| 最终 | **19.7ms** | tLS 8.7 + tBuild 7.6 + 主循环 3.4 |

## 6. 验证

- iter 86(vs 库 58;多出的 28 步 = 重建税与子问题终止简化)
- maxObj 6579.756085(库 6579.754899;−0.007% gap 同族)
- vs MOSEK(原数据参考):absMax=166.7/p90=10.2/mean=4.38 = **精确踩家族地板**
- vs qpalm 库逐笔 1.86(同顶点);+polish=kkt 后与 qpalm+kkt 一致性 **0.0093**
- eps 阶梯:1e-6→1e-2 = 31.3/27.1/24.7/15.3ms,曲线与 qpalm 库同点

## 7. 复现

```bash
cd /home/logic/code/mose/cpp_admm
./build/csvport_cpp opt -i ../optim_inputs/optim_inputs.20260520 \
  -c ../optim_inputs/optim_constraints.20260520 \
  -w ../optim_inputs/optim_weights.20260520 -o /tmp/o.csv \
  --engine=nativeqpalm --scale-rows            # 19.7ms
NATIVEQP_DEBUG=1 ...                           # 轨迹+分段计时
./build/nqp_unit <in> <con> <wt>               # Newton 单元对拍
```