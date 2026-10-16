#!/usr/bin/env python3
"""Generate presentation charts for the QuantPilot portfolio page.

Produces two PNGs under ``charts/``:

1. ``performance_overview.png`` - 3-year cumulative return of all strategies
   vs the CSI 300 benchmark (top panel) plus underwater drawdown curves
   (bottom panel), computed from a real-data backtest run.
2. ``matching_benchmark.png`` - single-core matching-engine throughput,
   C++17 core vs pure-Python reference, measured live by
   ``benchmarks/bench_matching.py``.

The backtest window is pinned to the on-disk cache range so no network
access is needed; the script aborts if the data did not come from akshare.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from data import loader  # noqa: E402
from backtest.engine import Engine  # noqa: E402
from backtest.costs import CostModel  # noqa: E402
from report.metrics import compute_metrics  # noqa: E402
from risk.manager import RiskConfig, RiskManager  # noqa: E402
from strategies import available, create  # noqa: E402

OUT_DIR = ROOT / "charts"
OUT_DIR.mkdir(exist_ok=True)

# Window pinned to the cached akshare data (cache files end 2026-08-24).
END = pd.Timestamp("2026-08-24")
START = END - pd.DateOffset(years=3)
FETCH_START = START - pd.Timedelta(days=365)

COLORS = {"dual_ma": "#2563eb", "grid": "#d97706", "momentum": "#059669"}
LABELS = {"dual_ma": "Dual MA", "grid": "Grid", "momentum": "Momentum"}
BENCH_COLOR = "#64748b"


def run_backtests() -> dict[str, object]:
    symbols = loader.DEFAULT_SYMBOLS
    frames = loader.load_daily(symbols, FETCH_START, END)
    bench = loader.load_benchmark(FETCH_START, END)
    mode = loader.DATA_SOURCE["mode"]
    if mode != "akshare":
        raise SystemExit(f"refusing to chart non-real data (mode={mode!r})")

    panel = loader.build_panel(frames)
    close_df = pd.DataFrame({s: panel[s]["close"] for s in symbols})
    trade_dates = close_df.loc[START:END].dropna(how="all").index
    print(f"data: akshare | {len(trade_dates)} sessions "
          f"{trade_dates[0].date()} .. {trade_dates[-1].date()}")

    results = {}
    for name in available():
        engine = Engine(
            panel, create(name, panel), CostModel(),
            RiskManager(RiskConfig()),
            benchmark=bench, capital=1_000_000.0,
        )
        result = engine.run(START, END)
        metrics = compute_metrics(
            result.equity, result.benchmark, result.trades,
            result.round_trips, exposure=result.exposure,
        )
        results[name] = (result, metrics)
        print(f"strategy {name}: total={metrics['total_return']:.2%} "
              f"maxdd={metrics['max_drawdown']:.2%} sharpe={metrics['sharpe']:.2f}")
    return results


def chart_performance(results: dict[str, object]) -> pathlib.Path:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.8,
        "axes.edgecolor": "#94a3b8",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11.5, "axes.titlesize": 14, "axes.titleweight": "bold",
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 9), dpi=160, sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.08},
    )

    first = next(iter(results.values()))[0]
    bench = first.benchmark
    bench_ret = bench / bench.iloc[0] - 1.0
    ax1.plot(bench_ret.index, bench_ret.values * 100, color=BENCH_COLOR,
             lw=1.6, ls="--", label="CSI 300 benchmark", zorder=2)

    for name, (result, metrics) in results.items():
        eq = result.equity
        ret = eq / eq.iloc[0] - 1.0
        label = (f"{LABELS.get(name, name)}  "
                 f"(total {metrics['total_return']:+.1%} · "
                 f"maxDD {metrics['max_drawdown']:.1%} · "
                 f"Sharpe {metrics['sharpe']:.2f})")
        ax1.plot(ret.index, ret.values * 100, color=COLORS[name], lw=1.9,
                 label=label, zorder=3)

        dd = eq / eq.cummax() - 1.0
        ax2.fill_between(dd.index, dd.values * 100, 0.0,
                         color=COLORS[name], alpha=0.30, lw=1.0,
                         label=LABELS.get(name, name))

    ax1.axhline(0, color="#94a3b8", lw=0.9, ls=":")
    ax1.set_title("QuantPilot — 3-Year Strategy Backtest vs CSI 300 (net of costs)")
    ax1.set_ylabel("Cumulative return (%)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f%%"))
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95,
               edgecolor="#e2e8f0", fontsize=10)

    ax2.axhline(0, color="#94a3b8", lw=0.9)
    ax2.set_title("Drawdown", fontsize=12, loc="left", pad=6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.legend(loc="lower left", frameon=True, framealpha=0.95,
               edgecolor="#e2e8f0", fontsize=9, ncol=3)

    fig.text(0.99, 0.005,
             "6 A-share symbols · akshare real data · 2023-08 → 2026-08 · "
             "signals on close, filled next open, net of commission/stamp/slippage",
             ha="right", va="bottom", fontsize=8.5, color="#64748b")

    out = OUT_DIR / "performance_overview.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def chart_matching() -> pathlib.Path:
    proc = subprocess.run(
        [sys.executable, "benchmarks/bench_matching.py",
         "--ops", "500000", "--rounds", "3"],
        capture_output=True, text=True, cwd=ROOT, timeout=900,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"bench failed:\n{proc.stderr[-2000:]}")

    medians = re.findall(r"median :\s+[\d.]+ s\s+([\d,]+) orders/s", proc.stdout)
    if len(medians) < 2:
        raise SystemExit("could not parse bench medians")
    cpp_ops = float(medians[0].replace(",", ""))
    py_ops = float(medians[1].replace(",", ""))
    speedup = cpp_ops / py_ops

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.8,
        "axes.edgecolor": "#94a3b8",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=160)
    names = ["C++17 core\n(qp_matching · pybind11 · GIL released)",
             "Pure Python\n(reference order book)"]
    values = [cpp_ops / 1e6, py_ops / 1e6]
    bars = ax.bar(names, values, width=0.52,
                  color=["#2563eb", "#94a3b8"], zorder=3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + values[0] * 0.02,
                f"{v:.2f}M ops/s", ha="center", va="bottom",
                fontsize=12.5, fontweight="bold",
                color="#0f172a")

    ax.set_title("QuantPilot Matching Engine — Single-Core Throughput")
    ax.set_ylabel("Throughput (M orders/sec, single core)")
    ax.set_ylim(0, values[0] * 1.22)
    ax.annotate(f"{speedup:.0f}× faster",
                xy=(0.5, values[0] * 0.86), xycoords=("axes fraction", "data"),
                ha="center", fontsize=15, fontweight="bold", color="#b91c1c")

    fig.text(0.99, 0.005,
             "adaptive realistic order stream, 500k ops/round (70% resting / 15% marketable / 15% cancels) · "
             "median of 3 timed rounds · identical fills cross-checked vs Python · CPU-limited sandbox",
             ha="right", va="bottom", fontsize=8.5, color="#64748b")

    out = OUT_DIR / "matching_benchmark.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    results = run_backtests()
    p1 = chart_performance(results)
    print(f"wrote {p1}")
    p2 = chart_matching()
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
