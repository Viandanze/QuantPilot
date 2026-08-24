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
- **Realistic cost model**: commission 0.025% per side (5 CNY minimum), stamp tax
  0.1% on sells, 0.1% adverse slippage on both sides - always on
- **T+1 settlement** and 100-share board lots, decided-on-close / filled-at-next-open
  execution (no same-bar lookahead)
- **Risk overlay**: 20% single-name cap, -8% per-position stop loss, -15% portfolio
  drawdown guard that forces flat and imposes a cash cooldown
- **Self-contained HTML reports**: equity curve vs CSI 300, drawdown, Sharpe, CAGR,
  win rate, full trade log, in-sample / out-of-sample split
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
backtest/    cost model + event-driven engine (T+1, next-open fills)
risk/        position cap, stop loss, drawdown cooldown
report/      metrics + self-contained HTML rendering
signal.py    CLI: per-strategy daily signals
backtest.py  CLI: run backtests, write reports
tests/       pytest suite (costs, signals, engine constraints, risk rules)
```

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

`pytest` runs 29 tests covering the cost model, strategy signal logic, engine
constraints (next-open fills, T+1 locks, caps, stop loss, drawdown cooldown) and the
sample-data generator, all offline.

## Disclaimer

This software is for education and engineering review. Backtests are historical
simulations net of *estimated* costs; real fills can be worse. Nothing here is
investment advice, no result is a promise of future performance, and the author is
not liable for any trading decisions made with this code. Markets can and will
hurt mechanical strategies for long stretches.

## License

MIT - see [LICENSE](LICENSE).
