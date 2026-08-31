# 精度分层测试(修订版):native 表 + lib 表 vs MOSEK 参尺

- 日期:2026-08-24(v2,修正 v1 的 MOSEK 列循环论证);数据
  `/tmp/mose_run/tiers/raw.csv`(native 84 组)+ `raw_lib.csv`(lib 84 组
  + mosek 3 点);脚本 `tiers_sweep.py` / `tiers_sweep_lib.py`
- 设置:`--scale-rows`,eps_abs = eps_rel = v,v ∈ {1,2,3,5}×10⁻ᵏ,k=2..8;
  误差 = targetPosition(名义值)vs **唯一参考** `mosek_engine_out.csv`
  (MOSEK 原数据默认容差单次输出)
- 时间口径:**solver 内部时间**。native ×3 = solveMs(全含:分解/重建/
  线搜);lib ×3 = solveMs + setupMs(**补齐此前被排除的 QDLDL 分解 /
  Ruiz 预条件**,首次公平可比);MOSEK = 自报 `MSK_DINF_OPTIMIZER_TIME`

## 0. 参尺自身的刻度(先声明,避免循环论证)

MOSEK 复跑(同二进制)vs 参考的漂移:run1/2 逐位(≤0.001),run3
max **0.17** / p90 0.032 / mean 0.010(名义值;源自 Go 侧 map 遍历序
→ IPM 浮点路径分岔,把停机点在平坦最优面上轻推)。含义:
- 任何引擎 vs MOSEK 的 pos 比较,有意义下限 ≈ **0.1~0.2(百位档)**
- v1 表中"MOSEK 每档达"是拿尺子量自己,作废;本版 MOSEK 误差列 =
  它自己的复跑漂移,只作为尺子刻度注记

## 1. MOSEK 的 eps 旋钮在本问题上失效(新发现,本轮实测)

接入 `MOSEK_INTPNT_EPS` env(经 PutDouParam 同设 REL_GAP/PFEAS/DFEAS,
golang jobd 已重编)后扫 1e-1~1e-8:**全部 17 iter、同解(误差 ~0.001)**,
solver 内部时间 98-141ms 波动(负载噪声,非精度响应)。机理:18 步内点
迭代的残差轨迹在第 6 iter 已塌到 1e-3 量级,presolve/正则化主导,三个
tol 松到 0.1 也触发不了提前停。**结论:MOSEK 无"松容差换速度"的 Pareto,
是一个固定点(17 iter / ~110-140ms)** ——分层表里它只有一列常数。

## 2. 表 A:native 引擎(时间 = solveMs,全含口径)

| 精度档(pos 误差 vs MOSEK) | native(OSQP 式) | nativeqpalm | nativeprox | MOSEK(固定点) |
|---|---|---|---|---|
| 千位:max < 1000 | 53.2ms @2e-5(max 218) | **16.0ms** @2e-4(max 270) | 36.3ms @1e-4(max 455) | ~110-140ms(max 0.17) |
| 百位:max < 100 | ✗ 地板 166* | ✗ 地板 167* | ✗ 地板 167* | 达(0.17,即尺子刻度) |
| 百位:p90 < 100 | 50.0ms @1e-4(p90 39) | 16.0ms @2e-4(p90 42) | 36.3ms @1e-4(p90 35) | 达(0.0007) |
| 十位:p90 < 10 | 59.0ms @1e-6(p90 9.3) | ✗ 地板 10.2** | ✗ 地板 10.2** | 达(0.0007) |
| 十位:mean < 10 | 53.2ms @2e-5(mean 9.4) | **17.1ms** @5e-5(mean 7.4) | 39.5ms @2e-5(mean 5.4) | 达(0.0002) |
| 个位:mean < 1 | ✗ 地板 4.38* | ✗* | ✗* | 达(0.0001,尺子自身) |
| 个位:p90 < 1 | ✗* | ✗* | ✗* | 达(0.0007) |

\* 结构地板 = 平坦最优面**表示差**(§5.2):紧容差下三家误差完全一致
(166-167/10.2/4.38),与 eps 无关(1e-4→1e-8 误差纹丝不动,时间 +
20%)。polish=kkt 后三家互差 0.009~0.085(钉到同一 KKT 顶点),但
vs MOSEK 的 166 不可消(它停在解析中心侧)。
\** p90 地板 10.2 恰在 10 之上(边缘行判定噪声);native 个别点踩到
9.3-9.7 进档。

## 3. 表 B:lib 引擎(时间 = solveMs + setupMs,补齐分解/预条件)

| 精度档 | osqp(+QDLDL setup) | qpalm(+setup) | proxqp(+Ruiz setup) | 对应 native 最快 |
|---|---|---|---|---|
| 千位:max < 1000 | 99.1+2.9=102ms @2e-5 | **29.5+2.2=31.7ms** @1e-4 | 53.7+2.4=56.1ms @5e-6 | nativeqpalm 16.0ms |
| 百位:max < 100 | ✗ 218 地板* | ✗ 218 地板* | ✗ 205 地板* | —(同地板) |
| 百位:p90 < 100 | 84.1+2.9=87ms @1e-4 | 31.7ms @1e-4 | 56.1ms @5e-6 | 16.0ms |
| 十位:p90 < 10 | 108.4+3.0=111ms @2e-6 | ✗ 13.0 地板 | ✗ 12.2 地板 | native 59.0ms |
| 十位:mean < 10 | 102ms @2e-5 | 31.0+2.2=33.2ms @3e-5 | 58.7+2.4=61.1ms @3e-6 | nativeqpalm 17.1ms |
| 个位:mean < 1 | ✗* | ✗* | ✗* | — |

\* 同 §2 脚注:lib 三家的地板与 native 三家相同(166-218/12-13/4-5),
再次印证地板是**问题结构属性**而非实现。

setup 实测(默认 eps):osqp 3.0 / qpalm 2.2 / proxqp 2.5ms——不大,
但 v1 排行榜未计,本版补齐后口径才与 native(全含)公平。

## 4. 读表结论

1. **同精度档 native 全面快于对应 lib**(补 setup 后):千位档
   nativeqpalm 16.0 vs qpalm-lib 31.7(2.0×);十位 mean 档 17.1 vs
   33.2(1.9×)——与默认 eps 排行榜的 1.75× 一致,优势在松容差区扩大
2. **nativeqpalm 是千/百位档全场王**(16ms,比次快 native 快 3.3×,
   比 MOSEK 固定点快 ~8×);十位 p90 档唯一达标者是 native(59ms)
3. **MOSEK 固定点 ~110-140ms**:精度无与伦比(0.17 内),但无松档可换
   速度;"到 MOSEK 精度"的最近路径 = 任何 native 紧容差 + 平坦面
   4.38 地板(或 polish 后引擎间 0.009)
4. **十位以下(含个位)所有 6 个开源引擎同地板**:继续收 eps 只加时间
   (nativeqpalm 16.6→19.8ms),误差不动——要个位一致只能改问题
   (正则消平坦)或换比较对象(引擎间 polish 对比)

## 5. 松容差悬崖形态(eps≥1e-3,补充)

- native(OSQP 式):1e-3 崩(gap −0.7%),1e-2 大崩(−2.3~−4%)
- nativeqpalm:1e-3 尚可(0.12%),1e-2 崩(3.5~72%)
- **nativeprox 最稳**:1e-2 仍 gap −0.001%(pos 万级但目标几乎不动)
- lib 三家与各自 native 同形态(同算法)

## 6. 复现

```bash
python3 /tmp/mose_run/tiers_sweep.py        # native 84 组
python3 /tmp/mose_run/tiers_sweep_lib.py    # lib 84 组 + mosek eps 验证
cd golang_mosek/src && MOSEK_INTPNT_EPS=1e-1 ./run.sh opt --engine=mosek ...
# mosekOptMs=<solver 内部时间> 在 stderr;eps 旋钮已接入(三 tol 同设)
```

增补(2026-08-28,S2-2/S2-3 轮):分层结果在本轮所有引擎改动后
**回归不变**(native 1775/79.6ms、nativeqpalm 86/19.4ms、家族地板
形态保持);summary 行新增 `gapCert=` 内部证书列(紧容差 ≈0/松容差
悬崖检出/负值不可行红旗,见 report_st2.md §9)。