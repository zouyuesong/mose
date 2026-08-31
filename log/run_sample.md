# sample 运行记录

- 日期:2026-08-17
- 代码:`golang_mosek/src`(MOSEK 9.3.22 vendored so,`./run.sh` 启动,license 位于 `~/mosek/mosek.lic`)
- 二进制:`bin/jobd`(csvport `opt` 子命令)
- 运行方式:
  ```
  ./run.sh opt -i sample/input.csv -c sample/constraint.csv -w sample/weight.csv -o out.csv
  ```

---

## 1. 运行结果(利率互换组合)

### 问题规模与用时

| 项 | 数值 |
|---|---|
| 品种数 | 23(AUD/JPY/USD IRS/OIS 远期蝶式) |
| 约束数 | 29(13 gross / 13 net / 3 net_trade) |
| 标量变量 / 约束行(MOSEK) | 69 / 121 |
| 求解器内部耗时 | 0.04 s |
| 进程 wall time | ~0.07 s |

### 结果与参考 output.csv 对比

- 全部 23 行求解成功,未触发 relax 模式。
- `USD_OIS_FWDFLY_10Y5Y_15Y5Y_20Y5Y` = **+5082.37**,与 USAGE.md 记录一致
  (无约束理论最优 x* = 0.032/6.27e-6 ≈ 5082)。
- `USD_OIS_FWDFLY_4Y2Y_6Y2Y_8Y2Y` = **+389.94**:当前持仓 −60389.94 已破
  minPosition=−60000,Solver STEP 1 放宽持仓界后最小回补至界内,与文档解读一致。
- 其余品种交易量均在 1e-6 量级(数值噪声,视同 0)。
- 与仓库参考 `sample/output.csv` 逐行对比:显著交易完全一致,差异仅在
  噪声行的格式/精度(参考文件存的是当年小版本容差下的解),复现通过。
