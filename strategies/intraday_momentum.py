"""60-minute intraday momentum rotation with an ATR stop, T+0 friendly.

Designed for the T+0 ETF universe in :mod:`data.etf`: those symbols may be
bought and sold the same session, so a fast momentum book on 60-minute bars
is executable without overnight risk. Causality rules are the same as the
daily plugins - every decision reads only bars at or before the decision
timestamp, and the engine fills the resulting weights at the next bar.
"""
from __future__ import annotations

import math

import pandas as pd

from strategies.base import Strategy, asof_row, asof_value


class IntradayMomentum(Strategy):
    """Rank symbols by trailing 60-minute-bar momentum; hold the leaders.

    Decision rules, all evaluated on bars at or before the decision time:

    - momentum per symbol = close_t / close_{t-lookback} - 1 on 60-minute
      closes (rolling over the panel, then read as-of the decision date)
    - the ``top_n`` symbols with the highest positive momentum each get an
      equal ``1/top_n`` slice, so the gross target never exceeds 1.0; when
      no symbol has positive momentum the book goes flat (defensive)
    - ATR stop: a pick whose current price has fallen to ``ref -
      atr_stop * ATR`` or below is returned with weight 0, where ``ref``
      is the entry reference price tracked internally per symbol (first
      price seen when the symbol enters the book) and ATR is a plain
      rolling mean of true range over ``atr_period`` bars; until enough
      history exists the stop cannot fire
    """

    name = "intraday_momentum"

    def __init__(
        self,
        panel: dict[str, pd.DataFrame],
        lookback: int = 12,
        top_n: int = 2,
        atr_period: int = 14,
        atr_stop: float = 2.0,
    ):
        if lookback < 1 or top_n < 1 or atr_period < 1:
            raise ValueError("lookback, top_n and atr_period must be positive")
        if atr_stop <= 0:
            raise ValueError("atr_stop must be positive")
        self.symbols = sorted(panel.keys())
        if not self.symbols:
            raise ValueError("panel must contain at least one symbol")
        self.lookback, self.top_n = lookback, top_n
        self.atr_period, self.atr_stop = atr_period, atr_stop

        closes = pd.DataFrame({s: panel[s]["close"] for s in self.symbols}).sort_index()
        self.closes = closes
        self.momentum = closes / closes.shift(lookback) - 1.0
        self.atr = {
            s: self._true_range(panel[s]).rolling(atr_period).mean()
            for s in self.symbols
        }
        self._entry_ref: dict[str, float] = {}
        self._last: dict[str, float] = {s: 0.0 for s in self.symbols}

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        """True range per bar: max(high-low, |high-prev_close|, |low-prev_close|)."""
        prev_close = df["close"].shift(1)
        ranges = pd.concat(
            [df["high"] - df["low"],
             (df["high"] - prev_close).abs(),
             (df["low"] - prev_close).abs()],
            axis=1,
        )
        return ranges.max(axis=1)

    def _stopped(self, symbol: str, date: pd.Timestamp) -> bool:
        """ATR stop test against the pre-decision entry reference (fail-open)."""
        ref = self._entry_ref.get(symbol)
        if ref is None:
            return False
        px = asof_value(self.closes[symbol], date)
        atr = asof_value(self.atr[symbol], date)
        if not math.isfinite(px) or not math.isfinite(atr) or atr <= 0:
            return False
        return px <= ref - self.atr_stop * atr

    def target_weights(self, date: pd.Timestamp, equity: float) -> dict[str, float]:
        date = pd.Timestamp(date)
        weights = {s: 0.0 for s in self.symbols}
        row = asof_row(self.momentum, date).dropna()
        ranked = sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))
        picks = [s for s, m in ranked[: self.top_n] if m > 0.0]
        if not picks:
            self._entry_ref = {}
            self._last = weights
            return weights
        weight = 1.0 / self.top_n
        refs: dict[str, float] = {}
        for s in picks:
            if self._stopped(s, date):
                continue  # weight stays 0 and the entry reference is dropped
            refs[s] = self._entry_ref.get(s, asof_value(self.closes[s], date))
            weights[s] = weight
        self._entry_ref = refs
        self._last = weights
        return weights

    def note(self, date: pd.Timestamp, symbol: str) -> str:
        if symbol not in self.momentum.columns:
            return "no data"
        m = asof_value(self.momentum[symbol], pd.Timestamp(date))
        if not math.isfinite(m):
            return "warming up"
        state = "held" if self._last.get(symbol, 0.0) > 0.0 else "watch"
        return f"mom {m:+.1%} {state} ({self.lookback}x60m)"
