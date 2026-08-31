# 代码走读与改动清单(csvport 组合交易优化器)

- 对象:`golang_mosek/src`(模块 `csvport`,Go 1.17)
- 相关文档:`desc.md`(问题与模型)、`run_sample.md` / `run_real.md`(运行记录)

---

# 第一部分:代码详细解释

## 0. 代码地图

```
golang_mosek/src/
├── go.mod              模块 csvport, Go 1.17
│                       依赖: cobra(CLI) / gocsv(CSV) / tablewriter(表格) / mosek.go(CGO 绑定)
├── main/main.go        CLI 入口、子命令、flag、顶层流程
├── main/proctitle.go   进程名伪装(jobd)
└── opt/
    ├── types.go        数据结构 + CSV 加载 + 打印 + 关联(纯数据层)
    └── solver.go       QP 建模 + 两阶段求解编排(MOSEK 交互层)
```

依赖分工:`gocsv` 按**列名**(csv tag)反序列化,与列顺序无关;mosek.go 通过 CGO 调 `lib/libmosek64.so.9.3`。

## 1. main 包

### main.go

| 函数 | 行号 | 干什么 |
|---|---|---|
| `init()` | 31-49 | 注册 `opt` 子命令到 RootCmd;声明 flags:`-i/-c/-w/-o`(必填,MarkFlagRequired)、`-s` soft-lambda、`--elastic-penalty`(默认100)、`-v` |
| `CsvPortOptCmd.RunE` | 56-90 | **命令主体**,顺序:①`setComm/setProcTitle` 伪装进程名(必须在 cobra 解析完 flag 之后,因为会毁掉 os.Args 内存)→ ②`LoadOptInputsFromCSV`+PrettyPrint → ③constraints 同 → ④weights 同 → ⑤`opt.Solve(...)` → ⑥`res.PrettyPrint()` → ⑦`res.ToCSV(output)`。注意 inputs 分支(62-67 行)**先 PrettyPrint 再查 err**(加载失败会先打一张空表再报错,小瑕疵;constraints/weights 分支已改为先查错) |
| `RootCmd` | 94-97 | cobra 根命令 `csvport` |
| `main()` | 100-103 | `RootCmd.Execute()`,出错 exit 1 |

关键设计:即使 `Solve` 两遍都失败,res 也是"全不交易"的兜底结果(见 solver.go:393-400),**输出 CSV 照写**,然后 err 往上抛 → 失败安全(fail-safe to no-trade)。

### proctitle.go(本仓库新增,原版无)

| 函数 | 行号 | 干什么 |
|---|---|---|
| `setComm(name)` | 38-42 | `prctl(PR_SET_NAME)` 改内核任务名 → `/proc/pid/comm`(top/`ps -o comm` 显示的列),最多 15 字符+NUL |
| `stringDataPtr(s)` | 47-49 | 用 `reflect.StringHeader` 拿 string 底层数组地址——对 `os.Args` 元素,这正指向 loader 放在栈上的 argv 字符串块 |
| `setProcTitle(title)` | 58-77 | 把 argv[0] 起到最后一个 arg 结尾的**整块栈内存清零**,再写入 "jobd"。`/proc/pid/cmdline` 读的就是这块内存(htop/`ps -o args` 的来源),清零的尾巴让 cmdline 提前终止,不泄露 CSV 路径。与 PostgreSQL/Python setproctitle 同一手法;只伪装不隐身(root、`/proc/pid/exe` 仍可看穿) |

## 2. opt/types.go(数据层)

### 输入侧

| 函数/类型 | 行号 | 干什么 |
|---|---|---|
| `OPT_InputItem` | 17-28 | input.csv 一行。**无 csv tag 的列(masterAlpha_exec、minPerTradeDv01、isHoliday…)被 gocsv 静默丢弃**——optim 数据的 grp/rebate/rebateLocate 就是这么丢的 |
| `LoadOptInputsFromCSV` | 34-45 | open → `gocsv.UnmarshalFile` → `[]*OPT_InputItem`(**指针切片**,后文 Solve 会原地改它) |
| `(OPT_Inputs).PrettyPrint` | 48-74 | tablewriter 右对齐表格,`FormatFloat('G',4/5/2)` 截精度 |
| `ToColumns()` | 93-131 | 行视图 → 列视图(`Univ/Position/Alpha/...` 平行切片,索引 i 即品种 i)。**solver 只吃列视图**,符号→下标的映射在这一步固定 |

### 约束/权重侧

| 函数 | 行号 | 干什么 |
|---|---|---|
| `OPT_ConstraintItem` + `LoadOptConstraintsFromCSV` | 136-181 | name/lb/ub/type 四列加载 |
| `(OPT_Constrints).PrettyPrint` | 147-167 | 打约束表 |
| `OPT_WeightItem` + `LoadOptWeightsFromCSV` | 186-207 | 长表 (name,sym,weight) 加载 |
| `(OPT_Weights).PrettyPrint` | 210-229 | 打权重表(optim 时 2.3 万行全打,建议截断) |
| `(OPT_Weights).Tabulate()` | 233-246 | 长表拍平成 `map[约束名]map[sym]weight`,O(1) 查询;**重复 (name,sym) 行后者覆盖前者** |

### 关联与输出

| 函数 | 行号 | 干什么 |
|---|---|---|
| `AssembleConstraintDetails(univ, constraints, weights)` | 294-315 | **三表 join**:每条约束 × 权重表 → `TradeVariableIndices[k]`(品种在 univ 中的变量下标)+ `Weights[k]`。按 univ 顺序扫描,符号不存在于该约束则不参与(隐式 0)。产出 solver 直接可用的稀疏行结构 |
| `OPT_ResultItem` | 250-255 | 输出一行;`CurrentPosition` tag 为 `csv:"-"` **不落盘**(仅屏幕显示) |
| `(OPT_Result).PrettyPrint / ToCSV` | 261-281 / 363-374 | 结果表 / 写 output.csv |
| 三个 `ToCSV`(inputs/constraints/weights) | 320-359 | round-trip 序列化(调试用) |

## 3. opt/solver.go(核心,2 个函数)

### `Solve(...)` — 两阶段编排(solver.go:29-54)

```
STEP 1 预处理(31-38):逐品种原地放宽持仓界
   if pos+minTrade > maxPosition: maxPosition = pos+minTrade+ε   (ε=1e-10, 行11)
   if pos+maxTrade < minPosition: minPosition = pos+maxTrade−ε
   → 保证"至少存在单品种可行解"(sample 的 USD_4Y2Y 被 maxTrade 顶住的情形)

pass 1(39):solve(..., relaxMode=false)
   if 成功 || 不是"零交易违约" → 直接返回   ← 失败但非零交易违约也返回错误,不重试

pass 2(42-52,仅当 pass1 失败 && constraintInBreachWithZeroTraded):
   持仓界锚定当前持仓(44-50)→ solve(..., relaxMode=true)
```

要点:**inputs 是指针切片,STEP 1 和 pass 2 的界改动都是原地叠加**;是否重试由 pass 1 内部计算的 `constraintInBreachWithZeroTraded` 标志决定(账本本身已破线、任何交易救不回时才 relax)。

### `solve(...)` — 建模 + 单次 MOSEK 调用(solver.go:75-441)

按执行顺序分 9 段:

**① 准备(77-113)**
- `ToColumns()` + `AssembleConstraintDetails()`(77-78)
- 变量布局规划:`numvar` 个 x → TUB 块 `uᵢ`(87-95,**每品种必有**,因 cost·|x| 恒在目标里;`tradeToTUB/tubToTrade` 双向映射)→ GUB 块 `gᵢ`(98-113,**惰性创建**,只给出现在某条 GROSS 约束里的品种,map 去重——这就是"每品种 3 变量"的出处)
- 目标系数累加器:`quadCoefs map[i]map[j]float64`(Q 稀疏)、`linearCoefs map[j]float64`(c)

**② MOSEK 环境(117-133)**:`MakeEnv`/`MakeTask`,错误码经 `GetCodeDesc` 翻译成可读信息返回。

**③ x 变量与合并界(135-144)**:

```
xᵢ ∈ [ max(minTradeᵢ, minPosᵢ−posᵢ),  min(maxTradeᵢ, maxPosᵢ−posᵢ) ]
```

交易界与持仓界(移常数后)取交——两条硬约束压成一条变量界,不占约束行。

**④ TUB(147-164)**:每个 uᵢ,`u ≥ 0` + 两行 `u+x≥0, u−x≥0`(线性化 |x|);目标 `linearCoefs[uᵢ] = −costᵢ`(最大化会把 u 压到 |x|)。

**⑤ GUB(166-180)**:每个 gᵢ,`g ≥ 0` + 两行 `g−x≥pos, g+x≥−pos`(线性化 |pos+x|,pos 进约束右端常数)。

**⑥ 目标(182-194)**——注释即推导:
- `quadCoefs[i][i] = −(λᵢ+tλᵢ)`(来自 (λ/2)(pos+x)² 展开的 x² 项 + (tλ/2)x²;MOSEK 目标 ½xᵀQx,MAXIMIZE)
- `linearCoefs[i] = αᵢ − λᵢ·posᵢ`(α·x 收益项 + λ·pos·x 交叉项;α·pos、(λ/2)pos² 是常数已丢)

**⑦ 组合约束 switch(197-365)**,四种 type:

| type | 行号 | 行构造 | 界 | 零交易违约判定 | relax 附加 |
|---|---|---|---|---|---|
| NET | 199-267 | `Σwₖxₖ`,界减常数 `Σw·pos`(203-208) | `lb−c, ub−c` | `lb>0 ∥ ub<0`(209) | 界锚定到 ±ε(211-219)+ 弹性变量 |
| NET_TRADE | 268-281 | `Σwₖxₖ`(不减常数) | 原 lb/ub | 无(交易恒可选 0) | — |
| GROSS | 282-321 | `Σwₖgₖ ≤ ub`(g 是 |pos+x| 辅助变量) | lb>0 时打警告"不 honored"(292-294) | `ub < Σ|pos|w`(296) | ub 放宽到零交易值+ε + 弹性 |
| GROSS_TRADE | 336-361 | `Σwₖuₖ ≤ ub` | 同上 | — | — |
| default | 362-363 | `panic`(未知 type 直接崩) | | | |

**弹性变量机制**(223-255, 304-321,relax 模式专属):每个被放宽的界加 `t ≥ 0` 及一行 `原约束 ± t vs 原界`(t ≥ 违约量),目标 `linearCoefs[t] = −elasticPenalty`(默认 100/单位)——在放宽后的可行域内线性代价地尽量拉回原界。索引从 `gubIdx` 起递增(196)。

**soft-lambda**(257-267 等,`-s > 0` 时):对约束值加二次惩罚 `−softLambda·Σwᵢwⱼxᵢxⱼ`,直接写进 `quadCoefs`(GROSS/GROSS_TRADE 作用在辅助变量块上,NET 不含 pos 交叉项只加 Q——三处实现细节不完全对称)。

**⑧ 提交与求解(366-382)**:`PutQObjIJ`/`PutCJ` 灌系数 → `-v` 时挂日志流(379)→ `PutObjSense(MAXIMIZE)`(381)→ `Optimize()`(382)。**没有任何 IPAR/DPAR 调参,全默认** → QP 默认路径:MOSEK 内核自动做锥形化(QP→SOCP)→ presolve → 择对偶形 → 内点法(sample 69 变量 0.04s / optim 1074 变量 0.17s,均十几步迭代收敛)。

**⑨ 取解与容错(384-440)**:
- `GetRes() != RES_OK` → 打印 + **panic**(385-390)
- 结果先初始化为全不交易兜底(391-400)
- `GetSolSta(SOL_ITR)`(404):
  - `OPTIMAL` → 通过(412)
  - 原始/对偶不可行证书 → err(414-417)→ 上层决定是否 relax 重试
  - `UNKNOWN`:rescode 为 OK → 当退化情况放过(420-422);`RES_TRM_STALL` → 打警告后**接受 stalled 解**(426-427);其他 → err
- 成功则 `GetXx(SOL_ITR, nil)` 取全变量向量,**只有前 numvar 个是 x**(433-437),`TargetPosition = pos + x`
- 返回 `(res, constraintInBreachWithZeroTraded, err)` —— 标志供 `Solve` 决定 pass 2

## 4. 求解时完整调用流程(一次实际运行)

```
main()                                              main.go:100
└─ RootCmd.Execute → cobra 解析 flags
   └─ CsvPortOptCmd.RunE                            main.go:56
      ├─ setComm("jobd") / setProcTitle("jobd")     proctitle.go:38,58   ← 伪装,必须在 flag 解析后
      ├─ LoadOptInputsFromCSV  → PrettyPrint        types.go:34 / 48     ← gocsv 按列名,未知列丢弃
      ├─ LoadOptConstraintsFromCSV → PrettyPrint    types.go:170 / 147
      ├─ LoadOptWeightsFromCSV → PrettyPrint        types.go:196 / 210
      │
      └─ opt.Solve                                  solver.go:29
         ├─ STEP1: 原地放宽单品种持仓界             solver.go:31-38
         │
         ├─ solve #1 (relaxMode=false)              solver.go:75
         │  ├─ inputs.ToColumns()                   types.go:93
         │  ├─ weights.Tabulate()                   types.go:233      ← 长表→嵌套 map
         │  ├─ AssembleConstraintDetails()          types.go:294      ← 三表 join,符号→变量号
         │  ├─ mosek.MakeEnv / MakeTask             solver.go:118,126
         │  ├─ AppendVars(x) + PutVarBound(合并界)  solver.go:136-143
         │  ├─ TUB 块: u≥±x, c[u]=−cost             solver.go:147-164
         │  ├─ GUB 块: g≥±(x+pos)                   solver.go:166-180
         │  ├─ 目标: Q=−(λ+tλ), c=α−λ·pos           solver.go:188-194
         │  ├─ 约束 switch (NET/NET_TRADE/GROSS/GROSS_TRADE)  solver.go:197-365
         │  │   └─ (顺带算 constraintInBreachWithZeroTraded)
         │  ├─ PutQObjIJ / PutCJ / PutObjSense(max)  solver.go:367-381
         │  ├─ task.Optimize()  ──────────────► MOSEK 内核
         │  │      QP→SOCP → presolve → 选对偶(695×1931) → IPM 17 步, μ→1e-16
         │  ├─ GetSolSta / 容错分支                  solver.go:404-431
         │  └─ GetXx → 填 Trade/TargetPosition      solver.go:433-437
         │
         ├─ [仅当 #1 失败 && 零交易违约]:
         │    锚定持仓界 → solve #2 (relaxMode=true, 弹性变量生效)   solver.go:42-52
         │
      ├─ res.PrettyPrint()                          types.go:261
      └─ res.ToCSV(output.csv)                      types.go:363
         (两遍都失败时:res=全不交易兜底,CSV 照写,err 上抛 → exit 1)
```

sample 与 optim 都走"solve #1 一次成功"路径;触发 pass 2 需要构造"账本已破线且无交易可救"的数据(如某品种 minTrade=maxTrade=0 同时 gross 超限)。

## 5. 值得注意的实现细节(读代码时会疑惑的点)

1. **inputs 是 `[]*OPT_InputItem`**:STEP 1 与 pass 2 的界改动是**原地副作用**,会传导到后续 pass(叠加放宽)。
2. **`GetXx` 取全向量但只读前 numvar**:u/g/t 是辅助变量,结果里被忽略;u 的真实值=|x| 由目标自动压紧。
3. **NET 与 GROSS 的 soft-lambda 实现不对称**(261-266 vs 324-334):NET 版多减了含 pos 的线性项(对 Σw(pos+x) 整体惩罚的展开),GROSS 版只惩罚辅助变量块——语义上 GROSS 的软惩罚对象其实是 Σw·g 而非 Σw|pos+x|(g≥|pos+x| 但可能不等的间隙)。
4. **重复 (name,sym) 权重行**:Tabulate 后者覆盖前者,静默。
5. **未知 type 直接 panic**(363)而非报错——线上数据写错 type 会崩进程而不是走 relax。
6. **`RES_TRM_STALL` 被接受**、rescode OK 的 UNKNOWN 也放过:宁可拿一个近似解也不返回"不交易"(不交易的代价可能更大),这是交易系统的典型取舍。
7. **内点解的非顶点性质**:代码读 `SOL_ITR` 内点解,停在解析中心附近而非顶点,所以"不该动"的品种得到 1e-6 量级而非精确 0(sample 噪声行的根源);工业上可开 basis identification 或事后清零。

---

# 第二部分:新增改动清单(对比 `backup/src` 原版)

对比对象:`backup/src`(2026-08-12)→ `golang_mosek/src`(现版)。
**Go 代码逻辑上只加了一样东西(proctitle 伪装),其余全是注释/文档/运行环境。**

## ① 逻辑改动(仅 1 处新功能)

| 改动 | 内容 |
|---|---|
| **新增 `main/proctitle.go`(77 行)** | 进程名伪装:setComm(prctl) + setProcTitle(覆写 argv 内存块) → ps/htop 显示 `jobd` |
| main.go 配套 +4 行 | RunE 开头调用 `setComm/setProcTitle`;以及 constraints/weights 的 PrettyPrint 与 err 检查**顺序对调**(原版先打印后查错;inputs 那处仍未改) |
| Makefile | 构建产物名 `bin/csvport` → `bin/jobd` |

## ② 纯注释(0 逻辑变化)

- `opt/types.go`:全部 diff 都是 doc comment(类型/字段/函数注释),无一行执行代码变动
- `opt/solver.go`:41 行变动全是注释——Solve/solve 的大段算法说明、四个 case 的数学形式注释
- main.go 其余变动也是注释

## ③ 新增文件(工程/文档)

| 文件 | 内容 |
|---|---|
| `run.sh` | 设置 LD_LIBRARY_PATH 用 vendored 库启动,bin/jobd 不存在时回落 bin/csvport |
| `lib/libmosek64.so.9.3, libcilkrts.so.5` | 从 pip Mosek==9.3.22 提取的运行库 |
| `USAGE.md` | 中文使用文档(176 行,含 sample 解读、性能基准) |
| README | 补 vendored 运行说明、jobd 伪装说明 |

## ④ 未变

- `bin/csvport` 二进制**逐字节相同**(8月12 日的原版构建);sample/ 四个 CSV 相同;go.mod/go.sum 相同

## ⑤ 一个值得注意的事实

`bin/jobd` 并不存在 → `./run.sh` 实际执行的是 **`bin/csvport` 原版二进制**。即:此前所有测试运行(sample + optim)跑的都是**没有 proctitle 伪装的原始代码**;proctitle.go 的改动从未被编译进任何已运行的二进制,除非重新 `make`(会产出 jobd 并被 run.sh 优先使用)。

(仓库根的 `log/`、`optim_inputs/` 属于工作区新增,不在 src 对比范围内。)
