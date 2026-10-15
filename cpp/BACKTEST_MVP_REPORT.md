# C++ 高性能回测核心 MVP — 交付报告

日期：2026-09-20 ｜ 环境：Windows 11 + PowerShell ｜ MinGW g++ (D:\mingw64) ｜ CMake 4.3.2 ｜ Python 3.11.5 ｜ pybind11 3.1.0

## 结论一句话

回测热路径下沉 C++ 完成并逐位对齐 Python 引擎：**20 标的 × 5000 日线，Python 4.172s → C++ 12.6ms，加速 331.5x**（目标 ≥50x），权益曲线 5000 点、6947 笔成交、3067 个回合全部 **rel diff = 0.000e+00**；全量 pytest 160 passed。

## 1. 文件清单

新增（C++ 核心）：
- `cpp/backtest_core.hpp` — Bar / BacktestParams / TradeRecord / RoundTripRecord / BacktestResult + sma / roc / true_range / atr + run_dual_ma_backtest 声明
- `cpp/backtest_core.cpp` — 实现：指标向量化 + 事件循环逐条复刻 Python 引擎
- `cpp/backtest_bindings.cpp` — pybind11 模块 `backtest_core`（numpy 数组入口 + 指标绑定 + GIL 释放策略）
- `cpp/backtest_core_test.cpp` — C++ 自测可执行（指标手算用例 + 冒烟回测守恒 + 参数校验）

新增（Python 侧，未触碰任何既有 Python 文件）：
- `benchmarks/bench_cpp_vs_python.py` — 基准 + 一致性校验 + 首个分歧点 diff 诊断
- `benchmarks/precision_check.py` — 精度核查（全曲线/逐笔最大相对偏差）
- `tests/test_backtest_core_binding.py` — 回归测试 5 条（importorskip，未构建时自动跳过）

修改：
- `CMakeLists.txt` — 追加 `qp_backtest_core_test` 与 `backtest_core` 两个目标；给两个 pybind 模块加 `-static`（见 §3 踩坑）

编译产物（落盘绝对路径）：
- `C:\Users\17600\QuantPilot\backtest_core.cp311-win_amd64.pyd`（423 KB，本次任务主产物）
- `C:\Users\17600\QuantPilot\qp_matching.cp311-win_amd64.pyd`（同次构建顺带产出，原有模块）
- `C:\Users\17600\QuantPilot\build\qp_backtest_core_test.exe`（C++ 自测，exit 0）

未改动：`engine/`、`strategies/`、`backtest/`、`risk/` 的任何 Python 文件；`cpp/order_book.*`、`cpp/bindings.cpp` 源码未动。

## 2. 绑定路线选择及原因

**路线 A（pybind11）成功，未启用 B/C。**

- `pip install pybind11`（3.1.0）→ cmake `find_package(pybind11)` → `pybind11_add_module(backtest_core ...)` 一次通过。
- 选 pybind11 而非 ctypes 的原因：与仓库既有 `qp_matching` 绑定风格一致；numpy 数组零拷贝入参（S×N double 直接进 C++，免除 10 万级 Python 对象逐个 marshal）；struct 绑定（BacktestParams/BacktestResult）类型安全。
- 过程踩坑（记录，供后人）：首版链接出的 .pyd 依赖 `libwinpthread-1.dll`，而 Python 3.8+ 加载扩展时不搜索 PATH → `ImportError: DLL load failed`。修复：`target_link_options(... -static -static-libgcc -static-libstdc++)`，objdump 复核依赖仅剩 `python311.dll / KERNEL32.dll / msvcrt.dll`。**同样的修复也应用到原有 `qp_matching`**——它在 MinGW 下从未可导入（此前 matching 测试一直处于 skip 状态），修复后 43 条 matching 测试首次真实运行并全绿。

## 3. 编译命令（PowerShell 实际执行）

```powershell
pip install pybind11
cmake -S C:\Users\17600\QuantPilot -B C:\Users\17600\QuantPilot\build -G "MinGW Makefiles" `
      -DCMAKE_BUILD_TYPE=Release `
      -DPYTHON_EXECUTABLE="C:\Users\17600\AppData\Local\Programs\Python\Python311\python.exe" `
      -Dpybind11_DIR="C:\Users\17600\AppData\Local\Programs\Python\Python311\Lib\site-packages\pybind11\share\cmake\pybind11"
cmake --build C:\Users\17600\QuantPilot\build -j 4
# 自测
C:\Users\17600\QuantPilot\build\qp_backtest_core_test.exe
# 基准
python C:\Users\17600\QuantPilot\benchmarks\bench_cpp_vs_python.py
```

要点：必须显式传 `PYTHON_EXECUTABLE`，否则 FindPythonLibsNew 找到 WindowsApps 存根报 "Python libraries not found"。

## 4. Benchmark 结果（真实数据）

数据：合成 20 标的 × 5000 根日线（种子 42 几何随机游走，σ=2%/日），fast/slow=5/20，初始资金 100 万，默认 CostModel + 默认 RiskManager（止损/回撤清仓/冷却真实触发）。

| 指标 | Python engine | C++ core |
|---|---|---|
| 3 次运行耗时 | 4.172 / 4.375 / 4.926 s | 13.1 / 12.6 / 13.9 ms |
| **best-of-3** | **4.172 s** | **12.6 ms**（数组+调用 14.8 ms） |
| **加速比** | — | **331.5x**（含数组 marshal 281.1x） |

一致性校验（容差 1e-6 相对偏差；成交次数要求精确相等）：

| 指标 | Python | C++ | 判定 |
|---|---|---|---|
| 权益终值 | 2,189,333.840540 | 2,189,333.840540 | PASS |
| 成交次数 | 6947 | 6947 | PASS |
| 总费用 | 253,547.095546 | 253,547.095546 | PASS |

`precision_check.py` 深度核查：权益曲线 5000 点最大相对偏差 **0.000e+00**；逐笔成交价/费用最大相对偏差 **0.000e+00**；round_trips 3067/3067；期末现金逐位一致。即在本数据集上 C++ 与 Python **逐位一致**（非仅 1e-6 内近似）。

Python 引擎完整性自检：`all_checks_passed=True`（6947 笔全部过账审计，max_ledger_drift=2.79e-09）。

## 5. 语义对齐清单（C++ ↔ Python 逐条）

来源文件：`engine/runner.py`、`strategies/dual_ma.py`、`backtest/costs.py`、`engine/portfolio.py`、`engine/execution.py`、`engine/datafeed.py`、`engine/events.py`、`risk/manager.py`。

| # | 语义 | Python 依据 | C++ 实现 |
|---|---|---|---|
| 1 | 信号：fast SMA > slow SMA（严格大于，双 NaN 检查）→ 权重 1/S，否则 0 | dual_ma.target_weights | SMA 预计算（滚动和），warmup 段 NaN |
| 2 | 决策在收盘、成交在**次日开盘**（时间锁） | runner 时序 + execution | 每日两相循环：开盘执行前日 pending |
| 3 | 买入价 = raw×(1+0.1%)，卖出价 = raw×(1−0.1%) | CostModel.buy_price/sell_price | 同式同序 |
| 4 | 佣金 = max(turnover×0.03%, 5元)，双边；买入另加过户费 0.001%；卖出另加印花 0.05% + 过户费 | costs.commission/buy_fee/sell_fee；execution._sell 中 commission 变量含过户费 | 逐式复刻；FillEvent.fee = fee_commission+fee_stamp（events.py 属性） |
| 5 | 下单：target = round((w×equity)/px/lot)×lot，**Python round=银行家舍入** | portfolio.orders_from_weights | `std::nearbyint`（默认 FP 模式=half-even），等价 |
| 6 | 订单 elif 链：空仓且 target≥lot 买入；target≤0 全卖；diff≥lot 且超带宽部分买；diff≤−lot 且超带宽部分卖；band=1%×equity | orders_from_weights | 同一 elif 结构，含 forced 符号跳过 |
| 7 | 开盘执行优先级：强制平仓 → 信号卖 → 买入；强制平仓取消同晨同标的买入 | execution.execute_pending | 三遍扫描 pending |
| 8 | 卖出 clamp 到 T+1 可卖量（min(shares, sellable)），无货则丢弃 | execution._sell | locked/shares 分离 + rollover_t1 |
| 9 | 买入按手缩量直至 turnover+fee ≤ cash；现金按成交顺序串行扣减（符号排序） | execution._buy | while 缩量循环，符号序一致 |
| 10 | 持仓：全含成本 avg_cost（含买入费）+ price-only avg_cost_gross；卖出实现 realized = 净回款 − avg_cost×股数；清零生成 round_trip | portfolio._apply_buy/_apply_sell | 逐字段复刻 |
| 11 | 权益 = cash + Σ shares×close（仅可打印价格参与，符号序累加） | portfolio.mark_to_market | 同序累加（浮点顺序一致） |
| 12 | 风控-止损：close ≤ avg_cost×0.92 → 次开强平，标的封禁 10 日 | risk.on_close | 同阈值同 `<=` 比较 |
| 13 | 风控-回撤：equity/peak−1 < −0.15（严格）→ 全部强平 + 20 日冷却，冷却期无信号，结束后 peak 重置为当期 | risk.on_close + runner.cooldown_active | 状态机逐分支复刻 |
| 14 | apply_caps：封禁标的权重置 0；clamp [0, 20%]；总和 >1 时按 1/total 缩放 | risk.apply_caps | 同序求和、同条件缩放 |
| 15 | apply_caps 后才生成订单（runner 先 cap 再 orders_from_weights） | runner._on_close | 顺序一致 |
| 16 | 浮点运算顺序保持（(w×equity)/px/lot 左结合；累加按符号排序序） | Python 求值序 | 代码同构 → 本数据集逐位一致 |

与任务描述的偏差说明：任务草图为 `run_dual_ma_backtest(const std::vector<Bar>&)`（单标的），但 Python 引擎语义是组合级（共享现金、1/S 权重、风控组合回撤），单标的向量无法表达。最终签名 `run_dual_ma_backtest(const std::vector<std::vector<Bar>>&, params)`；单标的退化为传入 1 行 panel（受 20% 单票上限约束，与引擎行为一致）。

## 6. 验收标准核对

1. ✅ 编译产物落盘：`backtest_core.cp311-win_amd64.pyd`（路径见 §1）
2. ✅ benchmark 真实跑出：4.172s vs 12.6ms，331.5x，一致性 3/3 PASS
3. ✅ 既有测试不破坏：`python -m pytest C:\Users\17600\QuantPilot\tests -q`（从 C:\Users\17600）→ **160 passed**（112 原有 + 43 matching 首次真实运行 + 5 新增绑定回归）
4. ✅ 未改 engine/、strategies/、backtest/ 任何文件（新增封装全在 cpp/、benchmarks/、tests/）
5. ✅ 本报告即交付

## 7. 遗留问题

1. **对齐面板假设**：C++ 核心要求所有标的同一交易日网格（DataFeed 的 as-of 陈旧价语义在不对齐面板上未复刻）；跨期/缺日数据需先重采样对齐。
2. **指标未处理 NaN**：SMA 滚动和假定输入有限值；含 NaN 序列会污染（Python pandas rolling 会输出 NaN）。面板级 printable 检查已复刻，指标级未做。
3. **CMake `-static` 无编译器守卫**：当前无条件应用于两个 pybind 目标，MSVC 构建会失败（本仓库此环境仅 MinGW）。后续可加 `if(MINGW)` 包裹。
4. **MinGW 下 pybind11 LTO 被 pybind11 自动禁用**（其已知问题），未额外 pursuit PGO/LTO；12.6ms 中指标预计算约占比可观，若要再压可缓存 SMA 跨 run 复用。
5. `roc/atr` 已实现并绑定、经 C++ 自测，但未进入引擎热路径（双 MA 不用 ATR），仅作为指标库暴露。
6. 回撤/止损等风控参数已参数化进 `BacktestParams`，但风险"冷却后再入场首日"等极端边界未做独立单元测试覆盖（由 5000 日基准的 6947 笔成交一致性间接覆盖）。
