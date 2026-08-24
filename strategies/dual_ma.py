"""Dual moving average crossover (classic trend following)."""
from __future__ import annotations

import pandas as pd

from strategies.base import Strategy, asof_row


class DualMAStrategy(Strategy):
    """Hold a symbol while its fast MA is above its slow MA.

    Long entry effectively happens one session after a golden cross, exit one
    session after a dead cross (next-open execution). Each active symbol gets
    an equal fixed slice of the universe weight, so single-name exposure stays
    modest before the risk overlay is even applied.
    """

    name = "dual_ma"

    def __init__(self, panel: dict[str, pd.DataFrame], fast: int = 5, slow: int = 20, symbols=None):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.symbols = list(symbols or panel.keys())
        self.fast, self.slow = int(fast), int(slow)
        closes = pd.DataFrame({s: panel[s]["close"] for s in self.symbols}).sort_index()
        self.ma_fast = closes.rolling(self.fast).mean()
        self.ma_slow = closes.rolling(self.slow).mean()
        self.weight = 1.0 / len(self.symbols)

    def target_weights(self, date: pd.Timestamp, equity: float) -> dict[str, float]:
        f_row = asof_row(self.ma_fast, date)
        s_row = asof_row(self.ma_slow, date)
        out: dict[str, float] = {}
        for s in self.symbols:
            f = f_row.get(s)
            sl = s_row.get(s)
            active = (
                f is not None
                and sl is not None
                and pd.notna(f)
                and pd.notna(sl)
                and f > sl
            )
            out[s] = self.weight if active else 0.0
        return out

    def note(self, date: pd.Timestamp, symbol: str) -> str:
        f_row = asof_row(self.ma_fast, date)
        s_row = asof_row(self.ma_slow, date)
        f, sl = f_row.get(symbol), s_row.get(symbol)
        if f is None or sl is None or pd.isna(f) or pd.isna(sl):
            return "warming up"
        return f"MA{self.fast}={f:.2f} vs MA{self.slow}={sl:.2f}"
