# QuantPilot

Honest backtesting and disciplined daily signals for China A-shares.

QuantPilot is **not** a money-printing robot and does not pretend to be one. It is an
engineering-first toolkit that simulates three classic strategies under realistic
market frictions (commission, stamp tax, slippage, T+1) and enforces explicit risk
rules, so that "what if I had followed this mechanical rule" can be answered with
numbers you can defend in an interview - including the unflattering ones.

Every metric in every report is produced by the same engine that charges costs on
every fill. On the default universe and window (below) **all three strategies
underperform the CSI 300**, which is exactly the kind of result a serious backtest
framework should be comfortable showing.

## Features

- **Pluggable strategies** behind one `Strategy` interface: dual MA crossover,
  cross-sectional momentum rotation, grid trading
- **Event-driven engine**: DataFeed, Strategy, RiskManager, Portfolio and
  ExecutionHandler cooperate over one FIFO event queue; a per-fill ledger audit
  aborts the run the moment money goes missing
- **C++17 matching core** (`qp_matching`): a deterministic two-sided limit
  order book - price-time priority, GFD limit orders, cancels, self-trade
  prevention - exposed to Python via pybind11 with the GIL released during
  matching; a pure Python reference book and an equivalence test suite pin
  its behaviour fill-for-fill
- **Realistic cost model**: commission 0.025% per side (5 CNY minimum), stamp tax
  0.1% on sells, 0.1% adverse slippage on both sides - always on
- **T+1 settlement** and 100-share board lots, decided-on-close / filled-at-next-open
  execution (no same-bar lookahead)
- **Risk overlay**: 20% single-name cap, -8% per-position stop loss, -15% portfolio
  drawdown guard that forces flat and imposes a cash cooldown
- **Self-contained HTML reports**: equity curve vs CSI 300, drawdown, Sharpe, CAGR,
  win rate, full trade log, in-sample / out-of-sample split and a backtest-integrity
  section (look-ahead defences, ledger reconciliation with PASS/FAIL badges)
- **Daily signal CLI**: BUY / SELL / HOLD plus suggested (pre-cap) weights per strategy
- **Parameterized momentum**: the rotation knobs (`--mom-top`, `--mom-lookback`,
  `--mom-every`, `--mom-no-negative`) are CLI flags, `grid_scan.py` sweeps
  (top_n x lookback x rebalance_every) combinations over cached data, and
  `make_charts.py` exports standalone equity/drawdown charts
- **Free data only**: akshare (no token), with deterministic offline sample data for
  demos and CI

## Quickstart

```bash
pip install -r requirements.txt

# run all strategies on the default universe, last 3 years, real data
python backtest.py

# one strategy, custom window and universe
python backtest.py --strategy momentum --start 2023-01-01 --end 2025-12-31 \
    --symbols 600519,000858,601318,000333,600036,510300

# fully offline demo on synthetic data (no network needed)
python backtest.py --offline --split 0

# daily signals as of the latest session
python signal_cli.py
python signal_cli.py --symbols 600519,510300 --date 2026-08-21

# ETF cross-asset rotation: top-1 of treasury / Nasdaq / S&P 500 / gold ETFs,
# rebalanced every 10 sessions, 70/30 in-sample/out-of-sample validation
python backtest.py --strategy momentum --symbols 511260,513100,513500,518880 \
    --start 2021-10-12 --end 2026-10-09 --mom-top 1 --mom-every 10 \
    --split 0.7 --out reports_oos_top1e10

# tests
pytest

# build the optional C++17 matching core (see "Matching engine")
pip install pybind11
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build
```

Reports land in `reports_output/backtest_<strategy>.html`. Cached market data lands
in `data/cache/` (gitignored).

## Project layout

```
data/        akshare loader (eastmoney primary, sina fallback), cache, sample generator
strategies/  Strategy interface + dual_ma / momentum / grid plugins
engine/      event-driven core: typed events, FIFO queue, data feed, portfolio,
             execution handler, runner
backtest/    cost model + compatibility facade over the engine core
risk/        position cap, stop loss, drawdown cooldown
report/      metrics + self-contained HTML rendering (incl. integrity section)
cpp/         C++17 matching core: limit order book, native tests, pybind11
             bindings, native micro-benchmark
benchmarks/  pure Python reference order book + throughput benchmark
signal_cli.py CLI: per-strategy daily signals
backtest.py  CLI: run backtests, write reports
grid_scan.py CLI: batch momentum (top_n x lookback x rebalance_every) sweeps
make_charts.py CLI: render equity/drawdown PNGs from report data
tests/       pytest suite (82 tests: costs, signals, engine constraints, risk rules,
             event core, V1/V2 equivalence, look-ahead defences, reconciliation,
             matching bindings + C++/Python equivalence)
```

## Architecture

Since v0.2 the engine is event-driven: five decoupled modules cooperate over one
FIFO queue, and only the runner knows how they plug together. Each session `d`
unfolds as:

```
  DataFeed
    |-- MarketEvent(open, d) ---------------------------------------+
    |     ExecutionHandler.execute_pending (a generator)             |
    |       yields FillEvent -> queue -> Portfolio.on_fill           |
    |       ...booked before the generator sizes the next order      |
    |                                                                |
    |-- MarketEvent(close, d) ---------------------------------------+
    |     Portfolio.mark_to_market -> equity / exposure point        |
    |     RiskManager.on_close -> forced OrderEvent(s)               |
    |     Strategy.on_bar -> SignalEvent                             |
    |     RiskManager.apply_caps -> capped weights                   |
    |     Portfolio.orders_from_weights -> OrderEvent(s)             |
    |     ExecutionHandler.enqueue (parked until d+1's open)         |
    v
  next session ...
```

| Module | Owns | Emits |
| --- | --- | --- |
| `engine/datafeed.py` | the only door market data walks through; as-of prices per session phase | `MarketEvent` |
| `strategies/` | decision logic: `on_bar` hook, defaulting to the classic `target_weights` interface | `SignalEvent` |
| `risk/manager.py` | position caps, stop loss, drawdown cooldown | capped weights, forced `OrderEvent`s |
| `engine/portfolio.py` | cash, positions, whole-lot order sizing, per-fill ledger audit | `OrderEvent`s |
| `engine/execution.py` | next-open fills, T+1 clamps, cash-affordability shrinking | `FillEvent` |

`backtest/engine.py` stays the public entry point: it re-exports `Engine`,
`BacktestResult`, `Position`, `Trade` and `RoundTrip` from the new core, so the
CLIs, the reports and the original test suite run unchanged.

Morning accounting is interleaved with execution: `execute_pending` is a
generator that pauses after each fill, the runner routes that fill through the
queue and books it in the portfolio, and only then does the generator size the
next order. A morning sell therefore frees cash that the very next buy's
affordability check can see - the sequential semantics of a daily loop,
expressed in event flow.

## Methodology

### Execution timeline (per trading day `d`)

1. Shares bought on the previous session become sellable (T+1 rollover).
2. Orders decided at the close of `d-1` fill at the **open of `d`** with adverse
   slippage. Forced risk exits execute first, then strategy sells, then buys.
3. Positions are marked to market at the close of `d`.
4. The risk overlay evaluates stop loss / drawdown rules on those close marks.
5. The strategy computes new target weights using data up to `d` only.

Because signals are computed on the close and filled at the next open, no trade ever
uses information from its own fill bar. Targets are converted to whole 100-share
lots; partial rebalances only fire when position drift exceeds 1% of equity, which
keeps mark-to-market wiggle from generating churn.

### Cost model (defaults, all configurable)

| Item | Value | Charged |
| --- | --- | --- |
| Commission | 0.025% per side, 5 CNY minimum | both sides |
| Stamp tax | 0.1% | sell side |
| Slippage | 0.1% adverse | both sides |

The stamp tax is kept at the historical headline 0.1% for conservatism even though
the statutory rate was cut in 2023; pass your own `CostModel` to change it. All
performance figures are **net** of these costs.

### Risk overlay

- Single-name target weight capped at 20% of equity
- A position closing 8% below its all-in average cost is force-sold at the next open
  (the symbol is then blocked for ~10 sessions)
- If portfolio equity falls more than 15% below its running peak, everything is
  liquidated at the next open and the book stays in cash for ~20 sessions before
  strategies may re-engage (the peak resets afterwards)

These rules are designed to make the book hard to blow up, not to add alpha.

## Backtest integrity

Two failure modes silently corrupt a backtest: using future data, and losing
track of money. The engine treats both as architectural properties, not
conventions.

### No look-ahead: a lock, not a promise

- **Temporal lock in the engine.** Orders are born only while processing a
  close-phase event and are consumed only by the next session's open phase, so
  a signal can structurally never fill on the bar that produced it. Strategies
  never receive the queue, the portfolio or the execution handler - their only
  output is a weight map - so the timeline cannot be bypassed from strategy
  code, even by accident.
- **As-of data feed.** Every `MarketEvent` is built exclusively from bars dated
  at or before its session; future prices are censored at the single door data
  walks through.
- **Causal strategies.** Plugins precompute rolling indicators over the panel
  once and query them as-of the decision date; a rolling statistic at index `t`
  depends only on data at or before `t`.

The suite proves this with probes rather than trust:

- **Prefix invariance**: rebuilding the whole pipeline from data truncated at a
  cutoff reproduces the full-data run's equity curve and trade list up to that
  cutoff (all three strategies, two cutoffs). Appending future data cannot
  change the past.
- **Future-spike probe**: a flat panel that quintuples after date `T` - with
  the strategies handed the full spiked panel - produces runs ending at `T`
  identical to the flat panel's. Any leak anywhere would front-run the spike.
- **Wire audits**: with event tracing on, every streamed price must equal the
  panel's own as-of value, signals and orders may only be born in the close
  phase, and every fill must print strictly after the order that produced it,
  at that morning's slipped open price.

### Money conservation: an audit after every fill

After every single fill the portfolio re-checks a double-entry style identity:

```
cash + inventory at price-only basis + fees paid
    == initial capital + realized gross gains
```

Average-cost division introduces float rounding of a few ulps per partial fill,
so the check runs against a small tolerance (the larger of 1e-6 CNY and 1e-8 of
capital) and records the worst observed drift; a drift beyond tolerance
**aborts the backtest** instead of publishing broken numbers. At the close of a
run, four more identities must hold: cash flow (capital + sells - buys -
costs), equity marking (cash + market value), and both PnL decompositions
(net: realized + unrealized; gross: realized + unrealized - costs). The HTML
report renders all of this in its **Backtest integrity** section with
PASS/FAIL badges, next to the IS/OOS split.

### Engine lineage

The event-driven core reproduces the retired daily-loop engine's arithmetic
operation-for-operation - same fill priority, same float ordering. A frozen
reference (captured from the v0.1 engine with the CLI recipe) pins metrics,
first-trade anchors, ledger integrity and determinism for all three bundled
strategies. On the capture platform the two engines agree bit-for-bit; any
residual difference on other platforms is ulp-level floating-point noise.

## Matching engine

Since v0.3.0 QuantPilot also ships a C++17 limit order book under `cpp/`,
exposed to Python as the `qp_matching` extension module (pybind11). It is a
standalone, deterministic, single-threaded matching core - the same honesty
rules as the backtester apply: no hidden concurrency, no floating point in
the matching path, and behaviour pinned by tests rather than by claims.

### Semantics

- Two-sided limit order book, **price-time priority**: each price level is a
  FIFO queue; takers consume liquidity best-price-first, oldest-first.
- Good-for-day **limit orders** and **cancels**; a submitted order returns
  its fills (as `Fill` records) plus its resting remainder in one call.
- **Self-trade prevention** (cancel-resting): an incoming order never trades
  against a resting order of the same `trader_id`; the older resting order is
  cancelled and matching continues with the next maker. Cancelled ids are
  reported in `SubmitResult.self_trade_cancels`.
- **Prices and quantities are 64-bit integers** (ticks and lots). No floats
  anywhere in the matching path, so replaying one order stream always yields
  bit-identical fill sequences.
- Invalid input (non-positive price/quantity, duplicate order id) raises
  `ValueError`; re-entrant submits from a fill callback raise `RuntimeError`
  instead of corrupting the book.

### Architecture and complexity

```
LimitOrderBook
  bids_: map<price, Level, greater>     // best bid = first level
  asks_: map<price, Level, less>        // best ask = first level
  Level { list<Order> fifo; int64 total_qty; }
  index_: unordered_map<order_id -> (side, price, list::iterator)>
```

- `submit` is **O(log P)** per price level touched (P = live price levels)
  plus **O(1)** per fill; resting a remainder is one map lookup + list push.
- `cancel` is **O(log P)**: hash lookup, list erase via stored iterator,
  level removal when it empties.
- The order index stores list iterators, which stay valid until the order is
  erased; a level is only removed once its queue is empty, so iterators can
  never dangle.
- One deliberate invariant: after every public operation the book is never
  crossed (best ask is strictly above best bid); the native tests assert it
  after every scenario.

### Using it from Python

```python
import qp_matching as qpm

book = qpm.LimitOrderBook()
result = book.submit(qpm.Side.BUY, 100_050, 200, trader_id=1)  # price in ticks
result.fills            # [Fill(...), ...] in execution order
result.rested           # True if any quantity is resting
result.resting_quantity # remaining size at rest
book.cancel(result.order_id)
book.best_bid(), book.best_ask(), book.spread()
book.bid_depth(5), book.ask_depth(5)      # [(price, qty), ...] best first
book.on_fill(callback)                    # optional per-fill callback
```

While no fill callback is registered, `submit`/`cancel` **release the GIL**
for the entire matching computation (a pytest test proves a busy Python
thread keeps running during a single long C++ sweep). With a callback
registered the GIL is held, and re-entrant orders from the callback are
rejected.

### Build

The C++ core needs a C++17 compiler, CMake >= 3.15 and (for the Python
module only) pybind11:

```bash
pip install pybind11
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build          # drops qp_matching*.so into the repo root (gitignored)
./build/qp_orderbook_test    # 130 native checks
./build/qp_orderbook_bench   # core-only micro-benchmark
python benchmarks/bench_matching.py
```

Without pybind11 the native test and benchmark binaries still build; CMake
prints a warning and skips only the Python module. `pytest` skips the
matching tests with a pointer here if the module has not been built.

### Measured throughput (single core)

Same op mix everywhere: ~70% resting limit orders (exponential offset from
the mid, mean 50 ticks), ~15% marketable orders (crossing 0-5 ticks), ~15%
cancels; quantities 1-100; 64 trader ids; roughly one in five ops produces
fills. One warmup round, then five timed rounds, medians reported. Platform:
one core of an Intel Xeon Platinum 8457C, g++ 11.4 (`-O3`), Python 3.13.15,
Linux.

| Implementation | orders/sec | fills/sec | note |
| --- | ---: | ---: | --- |
| C++17 core, native (`qp_orderbook_bench`, 5 x 200k ops) | ~6.6M | ~1.2M | no Python in the loop |
| C++17 core via `qp_matching` (5 x 100k ops, identical stream) | ~920k | ~172k | per-call binding + result conversion dominates |
| Pure Python reference book (same stream) | ~380k | ~72k | `benchmarks/py_limit_book.py` |

The honest reading: driven order-by-order from Python, the pybind11 module is
**~2.4x faster** than the pure Python book - per-call argument parsing and
building the returned `SubmitResult` dominate, not matching. The core itself
sustains **~6.6M orders/sec** (~17x the Python reference under the same
workload). Both numbers are reproducible with the commands above; they are
median-of-5 measurements on one shared CPU core, not peak marketing figures.

### Design tradeoffs

- **Determinism over parallelism.** No atomics, locks or thread pools: the
  core is single-threaded by design, which is what makes fill-by-fill
  equivalence testing (and future replay/audit tooling) meaningful. Throughput
  comes from data structures, not concurrency.
- **Scaled integers over doubles.** Matching engines should not round; ticks
  also keep the C++ and Python books bit-identical. The backtester's
  double-based accounting stays in Python, where the reference numbers live.
- **`std::map` levels over a flat array book.** Node-based maps cost cache
  misses but support unbounded price ranges and still hit millions of ops/sec;
  a flat price-indexed book would be faster but cap the price range.
- **Standalone over integrated.** The backtest engine still fills at next
  open through its own execution handler - the v1/v2 equivalence tests stay
  untouched. Wiring the real order book into the daily loop is deliberately
  left as an optional, separately-tested step; nothing in the 82-test suite
  depends on the extension module being built.
- **Cancel-resting STP** is the simplest deterministic self-trade policy:
  the resting (older) order dies, matching continues. Real venues offer
  cancel-newest/cancel-both/decrement variants; adding them is a policy
  parameter, not a redesign.

## Strategies

| Name | Rule | Default parameters |
| --- | --- | --- |
| `dual_ma` | hold each symbol while MA5 > MA20 (golden/dead cross state), equal 1/N weight per active name | fast 5, slow 20 |
| `momentum` | every N sessions rank symbols by trailing 20-session return, hold the top 5 at equal weight | lookback 20, top 5, rebalance 20 |
| `grid` | fixed range from the trailing 120 sessions (+/-10% buffer); allocation steps down from full at the bottom band to zero at the top | 8 levels |

The momentum knobs are exposed on the CLI (`--mom-top`, `--mom-lookback`,
`--mom-every`, `--mom-no-negative`); the ETF rotation below is exactly this
strategy configured as top-1, rebalance every 10 sessions.

## Anti-overfitting measures

- **Costs are always on.** There is no "gross of fees" mode to make equity curves
  prettier; friction is charged on every fill.
- **No parameter mining.** Defaults are round, textbook numbers (5/20, 20, 8), not
  grid-searched optima. Nothing in this repo was tuned on the results.
- **In-sample / out-of-sample split.** `backtest.py --split 0.7` reruns each strategy
  independently on the first 70% and last 30% of the window with fresh capital. The
  HTML report shows both; a collapse out-of-sample is treated as a warning, not
  hidden.
- **Next-open execution** removes the classic same-bar-close lookahead.
- **Fixed universe, stated up front.** The default symbols are liquid CSI 300
  constituents picked for data quality, not because they flatter any strategy.
- **Honest benchmarks.** Reports compare against the CSI 300 index over the exact
  same dates, and label synthetic-data demo runs prominently.

## Sample results (real data)

### ETF cross-asset rotation (the configuration used for daily signals)

Momentum over 511260 / 513100 / 513500 / 518880 (10-year treasury, Nasdaq-100,
S&P 500 and gold ETFs): top-1 at full weight, rebalanced every 10 sessions,
negative momentum still held - on an always-investable universe the best of a
bad bunch is the intended exposure. Window 2021-10-12 .. 2026-10-09
(1211 sessions), 1,000,000 CNY, the same frictions and risk rules as above,
70% in-sample / 30% out-of-sample. Reproduce with the ETF command in
Quickstart.

| Segment | Total return | CAGR | Max drawdown | Sharpe | Win rate |
| --- | --- | --- | --- | --- | --- |
| Full window (net) | +24.41% | +4.65% | -4.41% | 1.21 | 56.45% |
| Out-of-sample (last 30%) | +12.20% | +8.29% | -3.50% | 1.61 | 66.67% |

Over the same window the CSI 300 returned -11.60% with a -37.86% drawdown, so
the rotation added ~7.2 pp of annualized excess. Read it with the usual
scepticism: four instruments is a tiny universe, and a parameter set this
small is always one lucky window away from looking ordinary - the 70/30 split
and the out-of-sample segment above are the actual defence, not the point
estimates.

### Stock universe (default configuration)

Window 2023-08-24 .. 2026-08-24 (726 sessions), universe 600519 / 000858 / 601318 /
000333 / 600036 / 510300 (forward-adjusted daily bars via akshare), 1,000,000 CNY,
all frictions and risk rules enabled. CSI 300 total return over the same window:
**+22.55%** (max drawdown -17.9%).

| Strategy | Total return | CAGR | Max drawdown | Sharpe | Excess CAGR vs CSI 300 | Win rate |
| --- | --- | --- | --- | --- | --- | --- |
| dual_ma | -5.46% | -1.93% | -20.77% | -0.11 | -9.2 pp | 31.3% |
| momentum | +7.59% | +2.57% | -18.12% | 0.24 | -4.7 pp | 34.8% |
| grid | -7.38% | -2.63% | -14.51% | -0.24 | -9.9 pp | 50.9% |

Read the table the way it is meant to be read: over this window, mechanically
following any of these three rules was worse than holding the index. Momentum came
closest but still trailed after costs (14.5k CNY in fees and taxes on 1M capital);
grid had the mildest drawdown of the three (the risk overlay helped); dual_ma paid
the most friction (32.2k CNY) for a negative result. Expect most strategy/window
combinations to look like this - that is the honest baseline any real edge has to
beat, and precisely why the frictions are not optional here.

## Data sources

- Primary: [akshare](https://github.com/akfamily/akshare) - free, no token.
  Eastmoney endpoints first; a Sina mirror is used automatically if Eastmoney is
  unreachable. Stock bars are forward-adjusted (`qfq`).
- Benchmark: CSI 300 index (`000300`) daily closes.
- Offline fallback: `data/samples/` contains deterministic synthetic bars (random
  walk + alternating trend regimes, seeded) so the full pipeline - including
  `python backtest.py --offline` and the test suite - runs with zero network access.
  Any report built on this data is labelled as a demo, not market results.

## Testing

`pytest` runs 82 tests, all offline: the cost model, strategy signal logic,
engine constraints (next-open fills, T+1 locks, caps, stop loss, drawdown
cooldown), the sample-data generator, the event vocabulary and queue, the
event-driven core (data feed, portfolio, execution handler), V1/V2 equivalence
against frozen references, look-ahead defences (prefix invariance,
future-spike probe, wire audits), money conservation (hand-computed
identities, tamper detection) and the matching bindings (semantics, fills
callback, GIL release, fill-for-fill equivalence with the pure Python
reference book). The matching tests skip cleanly when `qp_matching` has not
been built; the C++ core additionally carries 130 native checks in
`cpp/order_book_test.cpp` (run via `./build/qp_orderbook_test`).

## Disclaimer

This software is for education and engineering review. Backtests are historical
simulations net of *estimated* costs; real fills can be worse. Nothing here is
investment advice, no result is a promise of future performance, and the author is
not liable for any trading decisions made with this code. Markets can and will
hurt mechanical strategies for long stretches.

## License

MIT - see [LICENSE](LICENSE).
