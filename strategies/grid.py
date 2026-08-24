"""Grid trading: buy lower bands, sell higher bands inside a fixed range."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy, asof_value


class GridStrategy(Strategy):
    """Ladder a symbol's allocation across a fixed price range.

    The range is estimated once, on the first decision date, from the trailing
    ``range_window`` sessions (min/max padded by ``range_buffer``), so it is
    fixed for the life of the run - no look-ahead, no re-fitting to later
    prices. Each symbol gets an equal capital slice; within that slice the
    target is ``levels`` units at the bottom band, decreasing linearly to zero
    units at or above the top band, which mechanically buys weakness and sells
    strength inside the range.
    """

    name = "grid"

    def __init__(
        self,
        panel: dict[str, pd.DataFrame],
        levels: int = 8,
        range_window: int = 120,
        range_buffer: float = 0.10,
        symbols=None,
    ):
        if levels < 2:
            raise ValueError("levels must be >= 2")
        self.symbols = list(symbols or panel.keys())
        self.levels, self.window, self.buffer = int(levels), int(range_window), float(range_buffer)
        self.closes = {s: panel[s]["close"].sort_index() for s in self.symbols}
        self.unit = 1.0 / len(self.symbols)
        self.bounds: dict[str, tuple[float, float]] = {}

    def _ensure_bounds(self, symbol: str, date: pd.Timestamp) -> tuple[float, float] | None:
        if symbol in self.bounds:
            return self.bounds[symbol]
        hist = self.closes[symbol].loc[: pd.Timestamp(date)].dropna()
        if len(hist) < max(30, self.window // 2):
            return None
        hist = hist.tail(self.window)
        low = float(hist.min()) * (1.0 - self.buffer)
        high = float(hist.max()) * (1.0 + self.buffer)
        if low <= 0 or high <= low:
            return None
        self.bounds[symbol] = (low, high)
        return low, high

    def units(self, price: float, low: float, high: float) -> int:
        """Grid units to hold at ``price``: ``levels`` at/below low, 0 at/above high."""
        if price >= high:
            return 0
        if price <= low:
            return self.levels
        ratio = (price - low) / (high - low)
        return self.levels - int(ratio * self.levels)

    def target_weights(self, date: pd.Timestamp, equity: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.symbols:
            px = asof_value(self.closes[s], date)
            if not np.isfinite(px) or px <= 0:
                out[s] = 0.0
                continue
            bounds = self._ensure_bounds(s, date)
            if bounds is None:
                out[s] = 0.0
                continue
            units = self.units(px, *bounds)
            out[s] = self.unit * units / self.levels
        return out

    def note(self, date: pd.Timestamp, symbol: str) -> str:
        px = asof_value(self.closes[symbol], date)
        if not np.isfinite(px):
            return "no data"
        bounds = self.bounds.get(symbol) or self._ensure_bounds(symbol, date)
        if bounds is None:
            return "warming up"
        low, high = bounds
        units = self.units(px, low, high)
        return f"{units}/{self.levels} units, range [{low:.2f}, {high:.2f}]"
