#!/usr/bin/env python3
"""Parameter sensitivity scan for the momentum strategy (vectorbt-style grid).

Loads market data once and re-runs the engine in-process for every
combination of (top_n x rebalance_every x allow_negative), printing one
summary row per combo. Use it to find robust plateaus, not lucky spikes.

Examples:
    python grid_scan.py --symbols 513100,513500,518880,511260 \
        --start 2021-10-12 --end 2026-10-12
    python grid_scan.py --symbols 600519,000858,601318,000333,600036,510300 \
        --tops 1,2,3 --everies 5,10,20 --no-neg-scan
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import sysconfig

# Same stdlib-signal pin as backtest.py (root signal.py shadows the stdlib).
if "signal" not in sys.modules:
    _stdlib = pathlib.Path(sysconfig.get_paths()["stdlib"]) / "signal.py"
    if _stdlib.exists():
        _spec = importlib.util.spec_from_file_location("signal", _stdlib)
        _module = importlib.util.module_from_spec(_spec)
        sys.modules["signal"] = _module
        _spec.loader.exec_module(_module)

import pandas as pd

from backtest.costs import CostModel
from backtest.engine import Engine
from data import loader
from report.metrics import compute_metrics
from risk.manager import RiskConfig, RiskManager
from strategies import create

FETCH_BUFFER_DAYS = 365


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", default=",".join(loader.DEFAULT_SYMBOLS),
                        help="comma-separated symbols")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (default: 5y before end)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--lookback", type=int, default=20,
                        help="momentum lookback in sessions (default: %(default)s)")
    parser.add_argument("--tops", default="1,2,3",
                        help="comma-separated top_n values (default: %(default)s)")
    parser.add_argument("--everies", default="1,3,5,10,20",
                        help="comma-separated rebalance_every values (default: %(default)s)")
    parser.add_argument("--no-neg-scan", action="store_true",
                        help="scan allow_negative=True only (skip the False variant)")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.DateOffset(years=5)
    fetch_start = start - pd.Timedelta(days=FETCH_BUFFER_DAYS)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tops = sorted({int(t) for t in args.tops.split(",") if t.strip()})
    everies = sorted({int(e) for e in args.everies.split(",") if e.strip()})
    neg_states = [True] if args.no_neg_scan else [True, False]

    frames = loader.load_daily(symbols, fetch_start, end)
    bench = loader.load_benchmark(fetch_start, end)
    panel = loader.build_panel(frames)
    close_df = pd.DataFrame({s: panel[s]["close"] for s in symbols})
    trade_dates = close_df.loc[start:end].dropna(how="all").index
    print(f"data: {loader.DATA_SOURCE['mode']} | {len(trade_dates)} sessions "
          f"{trade_dates[0].date()} .. {trade_dates[-1].date()} | pool {len(symbols)}")
    print(f"{'top':>3} {'every':>5} {'negok':>5} {'total':>8} {'cagr':>8} "
          f"{'maxdd':>9} {'sharpe':>7} {'excess':>8} {'win':>8} {'turnover':>9}")

    for top in tops:
        for every in everies:
            for neg in neg_states:
                engine = Engine(
                    panel,
                    create("momentum", panel, lookback=args.lookback, top_n=top,
                           rebalance_every=every, allow_negative=neg),
                    CostModel(),
                    RiskManager(RiskConfig()),
                    benchmark=bench,
                    capital=args.capital,
                )
                result = engine.run(start, end)
                m = compute_metrics(
                    result.equity, result.benchmark, result.trades,
                    result.round_trips, exposure=result.exposure,
                )

                def p(x):
                    return f"{x * 100:7.2f}%" if x is not None and pd.notna(x) else "    n/a"

                print(f"{top:>3} {every:>5} {str(neg)[:5]:>5} {p(m['total_return'])} "
                      f"{p(m['cagr'])} {p(m['max_drawdown']):>9} {m['sharpe']:7.2f} "
                      f"{p(m.get('excess_cagr'))} {p(m['win_rate'])} "
                      f"{m.get('avg_turnover', float('nan')):9.4f}")


if __name__ == "__main__":
    main()
