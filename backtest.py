#!/usr/bin/env python3
"""QuantPilot backtest CLI.

Examples:
    python backtest.py                          # all strategies, last 3 years
    python backtest.py --strategy dual_ma
    python backtest.py --start 2023-01-01 --end 2025-12-31
    python backtest.py --offline --split 0      # synthetic demo, no network
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import pandas as pd

from backtest.costs import CostModel
from backtest.engine import Engine
from data import loader
from report.html_report import render_report
from report.metrics import compute_metrics
from risk.manager import RiskConfig, RiskManager
from strategies import available, create

log = logging.getLogger("quantpilot")

FETCH_BUFFER_DAYS = 365  # extra calendar days fetched before start for warmup


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run QuantPilot backtests and write HTML reports.",
        epilog="Signals are decided on the close and filled at the next open, "
               "net of commission, stamp tax and slippage.",
    )
    parser.add_argument("--strategy", default="all", choices=[*available(), "all"],
                        help="strategy name or 'all' (default: all)")
    parser.add_argument("--symbols", default=",".join(loader.DEFAULT_SYMBOLS),
                        help="comma-separated symbols (default: %(default)s)")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (default: 3y before end)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="initial CNY")
    parser.add_argument("--split", type=float, default=0.7,
                        help="in-sample fraction for IS/OOS validation, 0 disables")
    parser.add_argument("--offline", action="store_true",
                        help="force built-in synthetic sample data (skip akshare)")
    parser.add_argument("--out", default="reports_output", help="output directory")
    return parser.parse_args(argv)


def run_strategy(name, panel, bench, start, end, capital, trade_dates, split):
    def fresh_engine():
        return Engine(
            panel,
            create(name, panel),
            CostModel(),
            RiskManager(RiskConfig()),
            benchmark=bench,
            capital=capital,
        )

    result = fresh_engine().run(start, end)
    metrics = compute_metrics(
        result.equity, result.benchmark, result.trades, result.round_trips,
        exposure=result.exposure,
    )
    split_metrics = None
    if split and 0.0 < split < 1.0 and len(trade_dates) >= 40:
        cut = max(1, min(len(trade_dates) - 1, int(len(trade_dates) * split)))
        is_end, oos_start = trade_dates[cut - 1], trade_dates[cut]
        res_is = fresh_engine().run(start, is_end)
        res_oos = fresh_engine().run(oos_start, end)
        split_metrics = {
            "in_sample": compute_metrics(
                res_is.equity, res_is.benchmark, res_is.trades, res_is.round_trips,
                exposure=res_is.exposure),
            "out_of_sample": compute_metrics(
                res_oos.equity, res_oos.benchmark, res_oos.trades, res_oos.round_trips,
                exposure=res_oos.exposure),
        }
    return result, metrics, split_metrics


def fmt_row(name, m):
    def p(x):
        return f"{x * 100:7.2f}%" if x is not None and pd.notna(x) else "    n/a"
    return (f"{name:<10} {p(m['total_return'])} {p(m['cagr'])} {p(m['max_drawdown']):>9} "
            f"{m['sharpe']:7.2f} {p(m.get('excess_cagr'))} {p(m['win_rate'])}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.DateOffset(years=3)
    fetch_start = start - pd.Timedelta(days=FETCH_BUFFER_DAYS)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("no symbols given")

    frames = loader.load_daily(symbols, fetch_start, end, offline=args.offline)
    bench = loader.load_benchmark(fetch_start, end, offline=args.offline)
    panel = loader.build_panel(frames)
    data_mode = loader.DATA_SOURCE["mode"]

    close_df = pd.DataFrame({s: panel[s]["close"] for s in symbols})
    trade_dates = close_df.loc[start:end].dropna(how="all").index
    if len(trade_dates) < 60:
        raise SystemExit(
            f"only {len(trade_dates)} usable sessions in {start.date()}..{end.date()}; "
            "widen the window or run --offline"
        )
    log.info("data source: %s | %d sessions %s .. %s",
             data_mode, len(trade_dates), trade_dates[0].date(), trade_dates[-1].date())
    if data_mode != "akshare":
        log.warning("not real market data (%s): treat results as a pipeline demo only", data_mode)

    names = available() if args.strategy == "all" else [args.strategy]
    out_dir = pathlib.Path(args.out)
    rows = []
    for name in names:
        log.info("running strategy '%s' ...", name)
        result, metrics, split_metrics = run_strategy(
            name, panel, bench, start, end, args.capital, trade_dates, args.split
        )
        out_path = render_report(
            result, metrics, out_dir / f"backtest_{name}.html",
            split_metrics=split_metrics,
            extra_meta={"data_source": data_mode, "split": args.split or None},
        )
        log.info("wrote %s", out_path)
        rows.append((name, metrics))

    print()
    print(f"QuantPilot backtest summary - net of costs - {trade_dates[0].date()} .. "
          f"{trade_dates[-1].date()} - data: {data_mode}")
    print(f"{'strategy':<10} {'total':>8} {'cagr':>8} {'maxdd':>9} {'sharpe':>7} "
          f"{'excess':>8} {'win':>8}")
    for name, m in rows:
        print(fmt_row(name, m))
    print(f"\nreports written to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
