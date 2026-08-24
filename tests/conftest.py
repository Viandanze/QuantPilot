"""Shared fixtures and helpers for the test suite."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import sysconfig

ROOT = pathlib.Path(__file__).resolve().parents[1]


def pin_stdlib_signal() -> None:
    """Keep stdlib `signal` importable despite the root-level signal.py CLI.

    The project deliberately ships a top-level ``signal.py`` entry point,
    which would shadow the standard library module once the project root is
    on ``sys.path``. Pin the real stdlib module first so third-party imports
    of ``signal`` keep working.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from strategies.base import Strategy


def make_panel(prices: dict[str, list[float]], start: str = "2024-01-02") -> dict[str, pd.DataFrame]:
    """Build a minimal OHLCV panel from close-price series (open = prev close)."""
    longest = max(len(v) for v in prices.values())
    dates = pd.bdate_range(start, periods=longest)
    panel = {}
    for symbol, arr in prices.items():
        idx = dates[: len(arr)]
        close = pd.Series(arr, index=idx, dtype=float)
        open_ = close.shift(1)
        open_.iloc[0] = close.iloc[0]
        high = pd.concat([open_, close], axis=1).max(axis=1)
        low = pd.concat([open_, close], axis=1).min(axis=1)
        panel[symbol] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000.0}
        )
    return panel


class AlwaysLong(Strategy):
    name = "always_long"

    def target_weights(self, date, equity):
        return {"TEST": 1.0}


class LongThenFlat(Strategy):
    name = "long_then_flat"

    def __init__(self, exit_from):
        self.exit_from = exit_from

    def target_weights(self, date, equity):
        return {"TEST": 1.0 if date < self.exit_from else 0.0}
