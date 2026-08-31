# optim_inputs(真实数据)运行记录

- 日期:2026-08-17
- 代码:`golang_mosek/src`(MOSEK 9.3.22 vendored so,`./run.sh` 启动,license 位于 `~/mosek/mosek.lic`)
- 数据:`optim_inputs/optim_inputs.20260520`、`optim_constraints.20260520`、`optim_weights.20260520`
- 运行方式:
  ```
  ./run.sh opt -i ../../optim_inputs/optim_inputs.20260520 \
               -c ../../optim_inputs/optim_constraints.20260520 \
               -w ../../optim_inputs/optim_weights.20260520   -o out2.csv
  ```

---

## 1. 运行结果(日股组合,20260520)

### 问题规模与用时

| 项 | 数值 |
|---|---|
| 品种数 | 358(全部 `.T` 东证股票,grp=3) |
| 约束数 | 67(1 gross + 66 net) |
| weight 行数 | 23628(1074 行权重=1,其余为 barra 因子暴露) |
| 标量变量 / 约束行(MOSEK) | 1074 / 1499 |
| 求解器内部耗时 | 0.17 s |
| 进程 wall time / 峰值内存 | 0.58–0.64 s / ~40 MB |

正常一次求解成功,未触发 relax 模式(当前账本零交易时未违约:
当前 gross 4.675e7 < 上限 5e7)。

### 解后校验(awk 独立重算,全部通过)

| 校验项 | 结果 |
|---|---|
| 逐品种 trade ∈ [minTrade, maxTrade] | 0 违规 |
| 逐品种 targetPosition ∈ [minPosition, maxPosition] | 0 违规 |
| gross_ptf = Σ\|pos\| | 4.99996e7 ≤ 5e7(**贴上界,绑定**) |
| delta_ptf = Σ pos | +99999.7 ≤ 1e5(**贴上界,绑定**) |
| delta_grp3 | +99999.7(与 delta_ptf 相同,因全部品种属于 grp3) |
| 63 条 barra_* 因子净敞口 | 全部 ∈ [−1e5, +1e5] |
| delta_grp1 | weight 文件中无任何行 → 空约束(Σ=0),平凡满足 |

### 结果特征

- 显著交易(|x|>1)共 133 笔,净卖出合计约 −2.41e6;
  优化器把组合从当前 gross 4.675e7 加杠杆到 5e7 上限、净敞口顶到 +1e5 上界
  (alpha 驱动)。典型大单:卖出 7269.T −70.7 万、9962.T −32.8 万;
  买入 8308.T +55.2 万、5801.T +46.7 万。9983.T 直接压到 minPosition=−50 万。

---

## 2. 与 sample 数据的差异

### 2.1 描述的问题不同

| 维度 | sample | optim_inputs |
|---|---|---|
| 资产 | AUD/JPY/USD 利率互换远期蝶式(23 个) | 日本股票(358 只,东证 `.T`) |
| 头寸单位 | dv01(风险单位) | 股数/名义值(±50 万级,gross 上限 5e7) |
| 风险模型 | 无因子;按币种/曲线 bucket 的 gross/net dv01 约束 | **BARRA 多因子**:63 个风格/行业因子净暴露 ±1e5 |
| 分组约束 | 币种层、曲线 bucket 层 | `delta_grp1/grp3` 分组净敞口(input 的 `grp` 列) |
| 交易方向限制 | 个别品种 minTrade=maxTrade=0(禁交易) | 每只股票单边受限(minTrade 或 maxTrade 一侧为 0,只能单方向交易) |
| 交易约束类型 | 用到 NET_TRADE(3 条) | 只用 NET / GROSS,无 *_TRADE |
| 借券成本 | 无此概念 | 带 `rebate`/`rebateLocate`(卖空券息/定位费)字段 |

### 2.2 数据格式差异

三个文件的**核心列名完全兼容**(gocsv 按列名匹配,与列顺序无关),差异如下:

| 文件 | sample | optim_inputs | 兼容性 |
|---|---|---|---|
| input | `sym,position,alpha,cost,lambda,tlambda,minTrade,maxTrade,minPosition,maxPosition` | 多出 `grp`(sym 后)、`rebate,rebateLocate`(末尾),列顺序不同 | 可解析,**多出的 3 列因无 csv tag 被静默丢弃** |
| constraint | `name,lb,ub,type`;type 含 net/net_trade | 同 4 列;type 仅 `net/gross`(小写) | 兼容(solver 内部 ToUpper) |
| weight | 权重全是 ±1(成员指示) | 除 ±1 外,barra 行为**实数因子暴露**(实测 [−0.1818, +0.2818]) | 兼容,权重本来就是任意系数 |
| 文件名 | 固定名 `.csv` | 带日期后缀无扩展名(`*.20260520`) | 兼容(loader 不看扩展名) |

数据质量检查:weight 中 symbol 全部存在于 input universe(0 个悬空);
constraint 中 `delta_grp1` 在 weight 文件里**没有任何行**(当前无 grp=1 品种)。

---

## 3. 接口调整建议

按必要性排序:

1. **`rebate`/`rebateLocate` 应纳入成本模型**(当前静默丢弃)。股票卖空的真实
   成本 = 交易费 + 借券成本,建议在 `OPT_InputItem` 加字段,并把目标函数中的
   `cost·|x|` 扩展为对卖出方向叠加 `rebate` 项(或先只读入+透传,至少打日志
   提示被忽略)。
2. **空约束校验**:`delta_grp1` 无权重行,`AssembleConstraintDetails` 生成全零
   行。建议 loader/solver 对"约束在 weight 文件无任何行"打 warning 或跳过,
   避免误以为它真的约束了什么。
3. **未知列告警**:gocsv 对无 tag 列静默丢弃,上游改列名时不易发现。建议加载后
   对比 CSV 表头与 struct tag,列出被忽略的列。
4. **PrettyPrint 截断**:optim_inputs 会把 358 行输入表 + 2.3 万行权重表全部
   打到 stdout(本次 stdout 重定向后约 2.4 万行)。建议加 `--quiet` 或默认只打
   摘要(行数、前 N 行)。
5. **一致性校验可选增强**:重复 sym、weight 引用不存在的 constraint name
   (当前数据没有,但一旦出现同样被静默忽略)。
6. `grp` 列:分组约束已在 weight 文件里展开(delta_grp3 → 358 行权重 1),
   `grp` 列冗余,可继续忽略;若想自动生成分组约束再读它。
7. 输出无需调整:`sym,targetTrade,targetPosition` 对股票同样适用;
   main.go 中已有 TODO(输出约束违约统计),对真实数据更有价值,可顺带做。

结论:**现有接口无需修改即可正确求解 optim_inputs 数据**(格式兼容、
结果通过全部约束校验),上述调整属于健壮性/成本模型完善项。
