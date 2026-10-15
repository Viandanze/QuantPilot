#!/usr/bin/env python3
"""Benchmark: C++ backtest core vs the Python event-driven engine.

Generates a synthetic aligned panel (20 symbols x 5000 daily bars, seeded
random walk), runs the dual-MA backtest through

  (a) the Python engine (engine.runner.Engine + DualMAStrategy + CostModel
      + default RiskManager), and
  (b) the native C++ core (cpp/backtest_core.cpp via the backtest_core
      pybind11 module),

then reports wall times, the speedup, and a consistency check
(equity terminal value / trade count / total fees, tolerance 1e-6 relative
deviation). On mismatch it prints the first divergent trade and the first
divergent equity point to help pin down semantic vs numeric drift.

Usage (from anywhere):
    python benchmarks\\bench_cpp_vs_python.py
Requires the built extension: repository-root\\backtest_core*.pyd
(see cpp/BACKTEST_MVP_REPORT.md for the build commands).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import sysconfig
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pin_stdlib_signal() -> None:
    """Keep stdlib `signal` importable despite the root-level signal.py CLI.

    Same pinning the pytest suite uses (tests/conftest.py): the project ships
    a top-level signal.py which would shadow the stdlib module once the repo
    root is on sys.path. Pin the real stdlib module first so third-party
    imports (pandas et al.) keep working.
    """
    if "signal" in sys.modules:
        return
    stdlib = pathlib.Path(sysconfig.get_paths()["stdlib"]) / "signal.py"
    if not stdlib.exists():
        return
    spec = importlib.util.spec_from_file_location("signal", stdlib)
    module = importlib.util.module_from_spec(spec)
    sys.modules["signal"] = module
    spec.loader.exec_module(module)


pin_stdlib_signal()
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest.costs import CostModel  # noqa: E402
from engine.runner import Engine  # noqa: E402
from risk.manager import RiskConfig, RiskManager  # noqa: E402
from strategies.dual_ma import DualMAStrategy  # noqa: E402

N_SYMBOLS = 20
N_BARS = 5000
SEED = 42
CAPITAL = 1_000_000.0
FAST, SLOW = 5, 20
REPEATS = 3
TOLERANCE = 1e-6  # relative deviation allowed on scalar agreement checks


def make_panel() -> dict[str, pd.DataFrame]:
    """Synthetic aligned panel: 20 symbols x 5000 business days, sorted keys.

    Geometric random walk: close follows a drift+noise log walk; open is the
    previous close perturbed (so next-open fills see realistic gaps); high /
    low / volume round the bar out. Insertion order is the sorted symbol
    order, matching both the engine's sorted(panel.keys()) and the strategy's
    panel-key iteration so the C++ port's single ordering assumption holds.
    """
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2006-01-02", periods=N_BARS)
    panel: dict[str, pd.DataFrame] = {}
    for i in range(N_SYMBOLS):
        sym = f"S{i:02d}"
        log_ret = rng.normal(3.0e-4, 2.0e-2, N_BARS)
        close = 50.0 * np.exp(np.cumsum(log_ret))
        open_ = np.empty(N_BARS)
        open_[0] = close[0] * (1.0 + rng.normal(0.0, 5.0e-3))
        open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 5.0e-3, N_BARS - 1))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 5.0e-3, N_BARS))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 5.0e-3, N_BARS))
        volume = rng.integers(1_000_000, 5_000_000, N_BARS).astype(float)
        panel[sym] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close,
             "volume": volume},
            index=dates,
        )
    return panel


def run_python(panel: dict[str, pd.DataFrame]) -> tuple[object, float]:
    """Baseline: existing engine, dual_ma, default risk overlay. Returns (result, seconds)."""
    t0 = time.perf_counter()
    strategy = DualMAStrategy(panel, FAST, SLOW)
    engine = Engine(panel, strategy, CostModel(), RiskManager(RiskConfig()),
                    capital=CAPITAL)
    result = engine.run()
    return result, time.perf_counter() - t0


def run_cpp(panel: dict[str, pd.DataFrame], bt) -> tuple[object, float, float]:
    """C++ core via the pybind11 module. Returns (result, seconds_total, seconds_run).

    seconds_total includes building the S x N arrays; seconds_run is the
    pybind call only (arrays pre-built, conversion happens inside C++).
    """
    syms = sorted(panel)
    dates = panel[syms[0]].index
    opens = np.stack([panel[s]["open"].to_numpy() for s in syms])
    highs = np.stack([panel[s]["high"].to_numpy() for s in syms])
    lows = np.stack([panel[s]["low"].to_numpy() for s in syms])
    closes = np.stack([panel[s]["close"].to_numpy() for s in syms])
    volumes = np.stack([panel[s]["volume"].to_numpy() for s in syms])
    ts = dates.asi8.astype(np.int64)

    params = bt.BacktestParams()
    params.capital = CAPITAL
    params.fast = FAST
    params.slow = SLOW

    t0 = time.perf_counter()
    res = bt.run_dual_ma_backtest(opens, highs, lows, closes, volumes, ts, params)
    run_secs = time.perf_counter() - t0
    # arrays+call: re-time the full entry path (rebuild arrays, then call)
    # so the reported total includes the marshaling cost, honestly.
    t1 = time.perf_counter()
    opens = np.stack([panel[s]["open"].to_numpy() for s in syms])
    highs = np.stack([panel[s]["high"].to_numpy() for s in syms])
    lows = np.stack([panel[s]["low"].to_numpy() for s in syms])
    closes = np.stack([panel[s]["close"].to_numpy() for s in syms])
    volumes = np.stack([panel[s]["volume"].to_numpy() for s in syms])
    ts = dates.asi8.astype(np.int64)
    _ = bt.run_dual_ma_backtest(opens, highs, lows, closes, volumes, ts, params)
    total_secs = time.perf_counter() - t1
    return res, total_secs, run_secs


def first_divergent_trade(py_trades, cpp_trades, syms) -> str | None:
    """First (index, py, cpp) triple where the trade streams disagree."""
    for i in range(min(len(py_trades), len(cpp_trades))):
        pt, ct = py_trades[i], cpp_trades[i]
        if (pt.symbol != syms[ct.symbol] or pt.side != ct.side
                or pt.shares != ct.shares):
            return (
                f"trade #{i}: python={pt.symbol} {pt.side} {pt.shares}sh "
                f"@{pt.price:.6f} fee={pt.fee:.4f} on {pt.date} | "
                f"cpp={syms[ct.symbol]} {ct.side} {ct.shares}sh "
                f"@{ct.price:.6f} fee={ct.fee:.4f} ts={ct.ts}"
            )
        if abs(pt.price - ct.price) > TOLERANCE * max(1.0, abs(pt.price)) or \
           abs(pt.fee - ct.fee) > TOLERANCE * max(1.0, abs(pt.fee)):
            return (
                f"trade #{i}: numeric drift python=({pt.symbol} {pt.side} "
                f"{pt.shares}sh @{pt.price:.10f} fee={pt.fee:.6f}) vs "
                f"cpp=({syms[ct.symbol]} {ct.side} {ct.shares}sh "
                f"@{ct.price:.10f} fee={ct.fee:.6f})"
            )
    return None


def first_divergent_equity(py_equity: pd.Series, cpp_curve, cpp_ts) -> str | None:
    for i, value in enumerate(cpp_curve):
        j = min(i, len(py_equity) - 1)
        pv = float(py_equity.iloc[j])
        if abs(value - pv) > 1e-9 * max(1.0, abs(pv)):
            return (
                f"equity[{i}] {pd.Timestamp(cpp_ts[i]).date()}: "
                f"python={pv:.6f} cpp={value:.6f} rel={abs(value - pv) / max(1.0, abs(pv)):.3e}"
            )
    return None


def main() -> int:
    ext = sorted(ROOT.glob("backtest_core*.pyd"))
    if not ext:
        print("ERROR: backtest_core*.pyd not found in the repository root.")
        print("Build it first (see cpp/BACKTEST_MVP_REPORT.md).")
        return 2
    sys.path.insert(0, str(ROOT))
    import backtest_core as bt

    print(f"loading extension: {ext[0].name}")
    panel = make_panel()
    print(f"panel: {N_SYMBOLS} symbols x {N_BARS} bars, seed={SEED}, "
          f"fast/slow={FAST}/{SLOW}, capital={CAPITAL:,.0f}")

    py_times, cpp_run_times, cpp_total_times = [], [], []
    py_res = cpp_res = None
    for r in range(REPEATS):
        py_res, t = run_python(panel)
        py_times.append(t)
        cpp_res, t_total, t_run = run_cpp(panel, bt)
        cpp_total_times.append(t_total)
        cpp_run_times.append(t_run)
        print(f"  run {r + 1}/{REPEATS}: python {t:8.3f}s | "
              f"cpp {t_run * 1000:7.1f}ms (call) / {t_total * 1000:7.1f}ms (arrays+call)")

    py_best = min(py_times)
    cpp_best = min(cpp_run_times)
    cpp_best_total = min(cpp_total_times)
    speedup = py_best / cpp_best
    speedup_total = py_best / cpp_best_total

    # ---------------- consistency checks -----------------------------------
    py_final = float(py_res.equity.iloc[-1])
    cpp_final = cpp_res.final_equity
    py_trades = len(py_res.trades)
    cpp_trades = int(cpp_res.trade_count)
    py_fees = float(py_res.fee_summary["total"])
    cpp_fees = cpp_res.total_fees

    def rel(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, abs(a))

    checks = {
        "equity terminal value": (py_final, cpp_final,
                                  rel(py_final, cpp_final) <= TOLERANCE),
        "trade count": (py_trades, cpp_trades, py_trades == cpp_trades),
        "total fees": (py_fees, cpp_fees, rel(py_fees, cpp_fees) <= TOLERANCE),
    }

    syms = sorted(panel)
    print("\n===== benchmark =====")
    print(f"python engine best of {REPEATS}:   {py_best:10.3f} s  "
          f"(runs: {', '.join(f'{t:.3f}' for t in py_times)})")
    print(f"c++ core  best of {REPEATS}:      {cpp_best * 1000:10.1f} ms "
          f"(runs: {', '.join(f'{t * 1000:.1f}' for t in cpp_run_times)}, call only)")
    print(f"c++ core  arrays+call best:       {cpp_best_total * 1000:10.1f} ms")
    print(f"speedup (call only):   {speedup:10.1f}x")
    print(f"speedup (arrays+call): {speedup_total:10.1f}x")

    print("\n===== consistency (tolerance: 1e-6 relative / exact for counts) =====")
    print(f"{'metric':<22}{'python':>20}{'c++':>20}{'verdict':>10}")
    verdicts = []
    for name, (pv, cv, ok) in checks.items():
        if isinstance(pv, int) and isinstance(cv, int):
            print(f"{name:<22}{pv:>20}{cv:>20}{'PASS' if ok else 'FAIL':>10}")
        else:
            print(f"{name:<22}{pv:>20,.6f}{cv:>20,.6f}{'PASS' if ok else 'FAIL':>10}")
        verdicts.append(ok)

    if not all(verdicts):
        print("\n-- diff diagnostics --")
        diag = first_divergent_trade(py_res.trades, cpp_res.trades, syms)
        print("first divergent trade:", diag or "none (streams align up to min length)")
        diag = first_divergent_equity(py_res.equity, cpp_res.equity_curve,
                                      cpp_res.equity_ts)
        print("first divergent equity:", diag or "none")
        print("verdict: SEMANTIC divergence if trade streams differ at an "
              "index with different symbol/side/shares; numeric drift if only "
              "prices/fees differ.")

    py_integrity = py_res.integrity
    print(f"\npython engine integrity all_checks_passed: "
          f"{py_integrity['all_checks_passed']} "
          f"(fills_audited={py_integrity['fills_audited']}, "
          f"max_ledger_drift={py_integrity['max_ledger_drift']:.2e})")
    print(f"python total_return={py_res.equity.iloc[-1] / CAPITAL - 1.0:+.4%} | "
          f"cpp total_return={cpp_res.total_return:+.4%} | "
          f"cpp mdd={cpp_res.max_drawdown:.4f}")

    print("\nOVERALL:", "PASS" if all(verdicts) else "FAIL")
    return 0 if all(verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
