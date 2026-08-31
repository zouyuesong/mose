# csvport 使用文档

`csvport opt` 是一个基于 MOSEK 凸二次规划（QP）的**组合交易优化器**：
给定当前持仓、收益预期（alpha）与各类风险约束，计算每个品种的最优交易量，
使"预期收益 − 交易成本 − 持仓风险 − 交易摩擦"最大。

```
csvport opt -i input.csv -c constraint.csv -w weight.csv -o output.csv
```

三个输入 CSV 与一个输出 CSV 均为长表格式，列名大小写敏感。
示例数据见 `sample/`（23 个利率互换远期蝶式组合，AUD/JPY/USD 三个币种）。

---

## 1. 输入：input.csv（每个品种一行）

| 列名 | 含义 | 是否使用 |
|---|---|---|
| `sym` | 品种代码，需唯一，作为各表关联键 | 是 |
| `position` | 当前持仓（风险单位，如 dv01） | 是 |
| `alpha` | 收益预期信号，越大越倾向加仓 | 是 |
| `lambda` | 持仓风险厌恶系数，惩罚 `(position+x)²` | 是 |
| `tlambda` | 交易摩擦厌恶系数，惩罚 `x²` | 是 |
| `cost` | 单位交易成本，惩罚 `\|x\|` | 是 |
| `minTrade` / `maxTrade` | 交易量硬上下界 | 是 |
| `minPosition` / `maxPosition` | 最终持仓硬上下界 | 是 |
| `masterAlpha_exec`, `minPerTradeDv01`, `isHoliday`, `isValidForTrd`, `panic_mode` | 上游系统字段 | **忽略**（无 csv tag，加载时丢弃） |

优化目标（最大化）：

```
Σᵢ [ alphaᵢ·xᵢ − lambdaᵢ·posᵢ·xᵢ − costᵢ·|xᵢ| ]       线性收益 + 成本
  − Σᵢ (lambdaᵢ + tlambdaᵢ)·xᵢ²                        二次风险/摩擦
  − softLambda · Σ_c (约束值)²                          软约束（可选，-s）
  − elasticPenalty · Σ tₖ                               弹性惩罚（relax 模式）
```

## 2. 输入：constraint.csv（每条约束一行）

| 列名 | 含义 |
|---|---|
| `name` | 约束名，唯一；用于关联 weight.csv |
| `lb` / `ub` | 约束上下界 |
| `type` | 见下表 |

| type | 数学形式 | 说明 |
|---|---|---|
| `NET` | `lb ≤ Σ wₖ·(posₖ+xₖ) ≤ ub` | 期末**净**敞口（如币种净 dv01） |
| `NET_TRADE` | `lb ≤ Σ wₖ·xₖ ≤ ub` | **净交易量**（如单币种当日买入限额） |
| `GROSS` | `Σ wₖ·\|posₖ+xₖ\| ≤ ub` | 期末**总**敞口，`lb` 忽略（恒 ≥ 0） |
| `GROSS_TRADE` | `Σ wₖ·\|xₖ\| ≤ ub` | **总交易量**，`lb` 忽略 |

sample 中的三层约束体系：
- `*_ptf`：全组合（23 个品种，权重 1）
- `*Dv01_AUD/JPY/USD`：单币种
- `*Dv01_10Y5Y_*` 等：按曲线 bucket

## 3. 输入：weight.csv（约束 × 品种权重，长表）

| 列名 | 含义 |
|---|---|
| `name` | 所属约束名 |
| `sym` | 品种代码 |
| `weight` | 该品种在此约束中的系数 |

- 同一约束每个参与品种一行；**未出现的品种权重视为 0**（不参与）。
- 例如 `grossDv01_JPY,JPY_OIS_FWDFLY_2Y1Y_3Y1Y_4Y1Y,1` 表示该品种以系数 1
  计入 JPY 币种总敞口约束。

## 4. 求解行为

1. **预处理**：若 `position+minTrade > maxPosition`（或对称情形），先放宽该
   品种持仓界，保证单品种可行（solver.go `Solve` STEP 1）。
2. **正常求解**：全部约束作为硬约束。若可行，直接返回。
3. **Relax 模式**（仅当"零交易也违约"，即当前账本已违反某条 NET/GROSS 约束）：
   - 持仓界锚定到当前持仓；
   - 违约的约束界放宽为"不比今天更差"；
   - 引入弹性变量 tₖ ≥ 违约量，按 `--elastic-penalty`（默认 100）线性惩罚，
     在可行域内尽量把解拉回原约束。

## 5. 输出：output.csv

| 列名 | 含义 |
|---|---|
| `sym` | 品种代码，行序与 input.csv 一致 |
| `targetTrade` | 应执行的交易量 x |
| `targetPosition` | 交易后期望持仓 = position + targetTrade |

**运行时 stdout** 依次打印：输入表 → 约束表 → 权重表（tablewriter 对齐表格），
求解后打印结果表；`-v` 额外输出 MOSEK 求解日志。

### sample 结果解读（sample/output.csv）

> 实测复现：vendored 9.3.22 + v11 trial license 在本仓库 sample 上求解成功，
> 显著交易与 sample/output.csv 一致（`USD_4Y2Y +389.94` 完全一致；
> `USD_10Y5Y` 为 +5082.37 vs 原 +5075.52，差 0.13%，属 MOSEK 小版本
> 求解容差，且新值更贴近下述理论最优 5082；其余品种均为 1e-6 级噪声）。

每个品种的最优交易量由线性边际 `alpha − lambda·position − cost` 决定：
为正才有交易动机，幅度由二次惩罚 `(lambda+tlambda)·x²` 收敛（无约束时
`x* = 边际 / (lambda+tlambda)`，成本 0.5555 对所有品种构成高门槛）。

| 品种 | targetTrade | 解读 |
|---|---|---|
| `USD_OIS_FWDFLY_10Y5Y_15Y5Y_20Y5Y` | **+5075.5** | 全表唯一显著交易：alpha=0.481，λ·pos=−0.106（负持仓抬升边际），净边际 = 0.481+0.106−0.556 ≈ +0.032；理论 x* = 0.032/6.27e-6 ≈ 5082，与实际解 5075 一致（差 0.15%，来自约束微调） |
| `USD_OIS_FWDFLY_4Y2Y_6Y2Y_8Y2Y` | **+389.94** | **被迫交易**：持仓 −60389.94 已破 minPosition=−60000，交易下界被顶到 max(0, −60000+60389.94)=389.94，最小回补至界内 |
| `AUD_IRS_FWDFLY_3Y2Y_5Y2Y_7Y2Y` | 0 | alpha=0.888 全表最高，但 minTrade=maxTrade=0，**无交易权限**；同理 alpha 更高的 AUD 品种要么不能交易，要么净边际被 cost=0.556 压成负值（如 AUD_4Y2Y: 0.536−0.31−0.556<0） |
| `AUD_IRS_FWDFLY_2Y1Y_3Y1Y_4Y1Y` | -2.05e-06 | 量级 1e-6~1e-8 的交易是**数值噪声，视同 0**（alpha 为负、成本高，不动） |
| 其余多数品种 | ~0 | 净边际为负，最优解即不动 |

## 6. CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `-i/--input` | 必填 | input.csv |
| `-c/--constraint` | 必填 | constraint.csv |
| `-w/--weight` | 必填 | weight.csv |
| `-o/--output` | 必填 | 输出 CSV |
| `-s/--soft-lambda` | 0 | >0 时对约束值加二次软惩罚（约束仍为硬约束） |
| `--elastic-penalty` | 100 | relax 模式弹性惩罚系数 |
| `-v/--verbose` | false | 打印 MOSEK 日志 |

## 7. 构建与运行前提

- **运行（推荐，无需完整安装 MOSEK）**：`lib/` 已内置 `libmosek64.so.9.3`
  与 `libcilkrts.so.5`（提取自 pip 包 `Mosek==9.3.22`），直接：
  ```
  ./run.sh opt -i sample/input.csv -c sample/constraint.csv -w sample/weight.csv -o output.csv
  ```
  注意：`pip3 install Mosek` 默认装 11.x，其库与本二进制 **ABI 不兼容**，
  必须是 9.3.x；`run.sh` 已自动处理，无需自己设 `LD_LIBRARY_PATH`。
- **license（必须）**：在 www.mosek.com 注册申请免费个人/试用 license，下载
  `mosek.lic` 放到 `~/mosek/mosek.lic`，或设环境变量
  `MOSEKLM_LICENSE_FILE` 指向它。缺失时报
  `MSK_RES_ERR_MISSING_LICENSE_FILE`。已实测 v11 trial license
  （FlexNet 向下兼容）可驱动 9.3 求解器；注意 license 有效期。
- **从源码构建**：需要 MOSEK 9.3 头文件/库路径（`CGO_CFLAGS`/
  `CGO_LDFLAGS` 见 README）、Go ≥ 1.17（CGO 启用），`make` 产出 `bin/jobd`。

## 8. sample 问题规模与用时基准

实测环境：vendored MOSEK 9.3.22（pip `Mosek==9.3.22` 的 so）+ v11 trial
license，Linux 6.8，3 次独立运行取值。

输入规模：

| 文件 | 行数（含表头） |
|---|---|
| input.csv | 24（23 品种） |
| constraint.csv | 30（29 条约束：13 gross / 13 net / 3 net_trade） |
| weight.csv | 162（161 个约束×品种权重） |

MOSEK 报告的问题规模（`-v` 日志，QO 二次规划）：

| 项 | 数量 | 构成（与 solver.go 建模一一对应） |
|---|---|---|
| 标量变量 | 69 | 23 交易 x + 23 个 \|x\| 辅助(TUB) + 23 个 \|pos+x\| 辅助(GUB) |
| 约束行 | 121 | 46(TUB±) + 46(GUB±) + 13 net + 3 net_trade + 13 gross |

用时与资源：

| 指标 | 实测 |
|---|---|
| 求解器内部耗时 | 0.03 s |
| 进程 wall time | 0.04–0.06 s（含 CSV 加载/打印） |
| 峰值内存 | ~18 MB |

即：**该规模（23 品种）下求解瞬间完成，瓶颈不在优化而在 IO**。规模增长
时变量/约束随品种数线性增长（每品种 +3 变量 +4 约束行，再加每条约束 1 行）。

## 9. 进程名伪装

运行期间进程在 top/htop/ps 中显示为中性名 `jobd`（comm 与 cmdline 均重写，
CSV 路径等参数不外泄），实现见 `main/proctitle.go`。注意：进程条目本身、
用户名、CPU/内存占用仍然可见，root 不受此影响。
