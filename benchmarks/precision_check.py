#!/usr/bin/env python3
"""Precision check: how close is the C++ core to the Python engine, exactly?

Runs the same panel through both engines once and reports maximum relative
deviations across the full equity curve and every trade's price and fee.
Complements bench_cpp_vs_python.py (which only gates on three scalars).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pin_stdlib_signal() -> None:
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
sys.path.insert(0, str(ROOT / "benchmarks"))

import numpy as np  # noqa: E402

from bench_cpp_vs_python import run_cpp, run_python  # noqa: E402


def main() -> int:
    import backtest_core as bt  # noqa: PLC0415

    panel = run_python.__globals__["make_panel"]()
    py_res, _ = run_python(panel)
    cpp_res, _, _ = run_cpp(panel, bt)

    eq_py = py_res.equity.to_numpy()
    eq_cpp = np.asarray(cpp_res.equity_curve)
    rel_eq = np.abs(eq_py - eq_cpp) / np.abs(eq_py)

    max_price_rel = 0.0
    max_fee_rel = 0.0
    for pt, ct in zip(py_res.trades, cpp_res.trades):
        max_price_rel = max(max_price_rel, abs(pt.price - ct.price) / abs(pt.price))
        max_fee_rel = max(max_fee_rel, abs(pt.fee - ct.fee) / max(pt.fee, 1e-12))

    fee_rel = abs(py_res.fee_summary["total"] - cpp_res.total_fees) / py_res.fee_summary["total"]
    final_rel = abs(float(py_res.equity.iloc[-1]) - cpp_res.final_equity) / abs(float(py_res.equity.iloc[-1]))

    print(f"equity curve  max rel diff : {rel_eq.max():.3e}  (n={len(eq_py)})")
    print(f"equity final  rel diff     : {final_rel:.3e}")
    print(f"total fees    rel diff     : {fee_rel:.3e}")
    print(f"per-trade price max rel    : {max_price_rel:.3e}")
    print(f"per-trade fee   max rel    : {max_fee_rel:.3e}")
    print(f"round_trips python/cpp     : {len(py_res.round_trips)} / {len(cpp_res.round_trips)}")
    print(f"final cash   py/cpp        : {py_res.trades[-1].cash_after:.6f} / "
          f"{cpp_res.trades[-1].cash_after:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
