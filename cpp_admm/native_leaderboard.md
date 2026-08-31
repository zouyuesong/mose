# native 全家族最终榜(冷解,--scale-rows,默认 eps=1e-6/1e-8,solve 口径)

- 日期:2026-08-24;数据见 cpp_admm/report_st2.md §2/§5/§6
- "native x" = 算法的零依赖 C++ 移植(仅 Eigen 头),Newton/线性系统用
  问题特定结构(品种块 + rank-k Woodbury);全部兼容 `--polish=kkt`

| native 引擎 | solve | 库版本 | 库版 solve | vs lib |
|---|---|---|---|---|
| **nativeqpalm**(QPALM 算法移植) | **19.7ms** | qpalm lib(σ=1) | 34.5ms | **1.75×** |
| **nativeprox**(ProxQP 算法移植) | **43.5ms** | proxqp lib | 62.6ms | **1.44×** |
| **native**(OSQP v1.0 算法移植) | **80.1ms** | osqp lib | 144ms | **1.80×** |

参考(非 native,solveMs 口径):proxqp lib 62.6+setup 2.5;qpalm lib
34.5+setup 2.2;osqp lib 144+setup 3.0;scs 5.8s;MOSEK **solver 内部
110-140ms**(wall 0.44-0.60s,差值为 env/license/Go 编组;其 eps 旋钮
在本问题上失效——1e-1~1e-8 全部 17 iter 同解,无松容差 Pareto,
见 report_precision_tiers.md §1;自身复跑漂移 max 0.17/mean 0.010
名义值 = 与它比较的刻度下限)。

补充:
- 松容差王:native(OSQP 式)@eps 1e-2 = **6.7ms**(gap 1.85%);
  nativeqpalm @1e-2 = 15.3ms
- 一致性档:三者 + `--polish=kkt` 后互差 0.009~0.085(同 KKT 顶点)
- 滚动场景:native 两族状态在自家结构里(库 qpalm 无数据更新 API,
  osqp 有 update_data_vec+warm_start 已实测 2.2×)——native 版 warm-start
  待做(本阶段按指示未做)

2026-08-28 增补(S2-2/S2-3 轮,详见 report_st2.md §7-§10):
- **nativeralm**(NR-LALM 第四族,负结果):忠实版 843k / pinP 版
  85.5k iter,被 native 支配 48-475×;不入榜,代码留作相图复现
- **反射 Halpern 加速处决**:调优基映射上全劣化;意外收获 = 固定
  ρ=300 针尖 68ms(非稳健,邻域 2× 恶化,只作奇点)
- **gapCert 内部证书**:summary 行新增,小正 = 解可信/大正 = 悬崖
  检出/负 = 不可行红旗;紧容差下 4e-3~2.5e-1 ≪ 0.48 的 MOSEK obj
  差,定量分离表示差与求解误差
- **relax 折叠(RELAX_FOLD=1)**:显式公式 native 不收敛 + qpalm lib
  段错误的弹性场景,折叠后 native 89ms 解出(KKT 自洽);纯 ADMM 对
  生产级罚结构性不可用(对偶爬升),正确完成体 = 折叠 + Newton 族
