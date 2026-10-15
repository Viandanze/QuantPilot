#!/usr/bin/env python3
"""QuantPilot daily signal CLI.

Prints every strategy's desired exposure for each symbol as of a given date,
plus an actionable label derived from the target change vs the previous
session. Weights are pre-trade suggestions; the engine would cap any single
name at 20% of equity.

Examples:
    python signal_cli.py
    python signal_cli.py --symbols 600519,510300
    python signal_cli.py --symbols 600519 --date 2026-06-19
    python signal_cli.py --offline
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from data import loader
from strategies import available, create

log = logging.getLogger("quantpilot")

HISTORY_DAYS = 500  # calendar days of history fetched for indicators


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Print per-strategy target signals.")
    parser.add_argument("--symbols", default=",".join(loader.DEFAULT_SYMBOLS),
                        help="comma-separated symbols (default: %(default)s)")
    parser.add_argument("--date", default=None,
                        help="decision date YYYY-MM-DD (default: latest session)")
    parser.add_argument("--offline", action="store_true",
                        help="force built-in synthetic sample data (skip akshare)")
    return parser.parse_args(argv)


def classify(w_today: float, w_prev: float) -> str:
    if w_today > w_prev + 1e-12:
        return "BUY"
    if w_today < w_prev - 1e-12:
        return "SELL"
    return "HOLD"


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("no symbols given")

    requested = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    fetch_start = requested - pd.Timedelta(days=HISTORY_DAYS)
    frames = loader.load_daily(symbols, fetch_start, requested, offline=args.offline)
    panel = loader.build_panel(frames)
    data_mode = loader.DATA_SOURCE["mode"]

    close_df = pd.DataFrame({s: panel[s]["close"] for s in symbols})
    all_dates = close_df.dropna(how="all").index
    known = all_dates[all_dates <= requested]
    if len(known) < 30:
        raise SystemExit(f"not enough history up to {requested.date()}")
    asof = known[-1]
    prev = known[-2] if len(known) >= 2 else None

    print()
    print(f"QuantPilot daily signals - decision date {asof.date()}"
          f"{'' if prev is None else f' (prev session {prev.date()})'} - data: {data_mode}")
    print(f"{'strategy':<10} {'symbol':<8} {'action':<7} {'weight':>7}  note")
    print("-" * 88)
    for name in available():
        strategy = create(name, panel)
        today = strategy.target_weights(asof, 1_000_000.0)
        # fresh instance for the previous session so stateful strategies
        # (momentum, grid) are not polluted by today's evaluation
        yesterday = (
            create(name, panel).target_weights(prev, 1_000_000.0) if prev is not None else {}
        )
        for symbol in symbols:
            w = float(today.get(symbol, 0.0))
            w_prev = float(yesterday.get(symbol, 0.0))
            action = classify(w, w_prev)
            weight = min(w, 0.20)
            note = strategy.note(asof, symbol)
            print(f"{name:<10} {symbol:<8} {action:<7} {weight * 100:6.1f}%  {note}")
    print("-" * 88)
    print("Weights are pre-trade suggestions; the risk overlay caps single names at 20%")
    print("and can force exits (stop loss -8%, portfolio drawdown -15%). Educational")
    print("tool for disciplined backtesting - not investment advice.")


if __name__ == "__main__":
    main()
