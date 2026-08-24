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
python signal.py
python signal.py --symbols 600519,510300 --date 2026-08-21

# tests
pytest
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
signal.py    CLI: per-strategy daily signals
backtest.py  CLI: run backtests, write reports
tests/       pytest suite (68 tests: costs, signals, engine constraints, risk rules,
             event core, V1/V2 equivalence, look-ahead defences, reconciliation)
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

## Strategies

| Name | Rule | Default parameters |
| --- | --- | --- |
| `dual_ma` | hold each symbol while MA5 > MA20 (golden/dead cross state), equal 1/N weight per active name | fast 5, slow 20 |
| `momentum` | every N sessions rank symbols by trailing 20-session return, hold the top 5 at equal weight | lookback 20, top 5, rebalance 20 |
| `grid` | fixed range from the trailing 120 sessions (+/-10% buffer); allocation steps down from full at the bottom band to zero at the top | 8 levels |

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

`pytest` runs 68 tests, all offline: the cost model, strategy signal logic,
engine constraints (next-open fills, T+1 locks, caps, stop loss, drawdown
cooldown), the sample-data generator, the event vocabulary and queue, the
event-driven core (data feed, portfolio, execution handler), V1/V2 equivalence
against frozen references, look-ahead defences (prefix invariance,
future-spike probe, wire audits) and money conservation (hand-computed
identities, tamper detection).

## Disclaimer

This software is for education and engineering review. Backtests are historical
simulations net of *estimated* costs; real fills can be worse. Nothing here is
investment advice, no result is a promise of future performance, and the author is
not liable for any trading decisions made with this code. Markets can and will
hurt mechanical strategies for long stretches.

## License

MIT - see [LICENSE](LICENSE).
