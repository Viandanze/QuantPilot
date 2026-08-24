"""Strategy interface shared by all plugins."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from engine.events import MarketEvent


def asof_row(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Last row of ``df`` at or before ``date`` (empty series if none)."""
    if len(df.index) == 0:
        return pd.Series(dtype=float)
    pos = df.index.searchsorted(pd.Timestamp(date), side="right") - 1
    if pos < 0:
        return pd.Series(dtype=float)
    return df.iloc[pos]


def asof_value(series: pd.Series, date: pd.Timestamp) -> float:
    """Last value of ``series`` at or before ``date`` (NaN if none)."""
    if len(series.index) == 0:
        return float("nan")
    pos = series.index.searchsorted(pd.Timestamp(date), side="right") - 1
    if pos < 0:
        return float("nan")
    return float(series.iloc[pos])


class Strategy(ABC):
    """Map information up to a decision date to target portfolio weights.

    Implementations must only read data at or before the decision date; the
    engine executes resulting weight changes at the next session open, so any
    look-ahead would silently corrupt the backtest. The bundled plugins
    precompute causal indicators (rolling means, trailing returns) over the
    panel once, then query them with as-of lookups at the decision date - a
    rolling statistic at index ``t`` depends only on data at or before ``t``.
    """

    name: str = "base"

    @abstractmethod
    def target_weights(self, date: pd.Timestamp, equity: float) -> dict[str, float]:
        """Return {symbol: target weight} for positions held from next open."""

    def on_bar(self, event: MarketEvent, equity: float) -> dict[str, float]:
        """Event-driven decision hook: return desired target weights.

        The engine invokes this exactly once per session, at the close, with
        an event whose prices are as-of the session date and nothing else:
        strategies never see the queue, the portfolio or the execution
        handler, so the close-decision / next-open-fill timeline cannot be
        bypassed from strategy code. The default implementation adapts the
        classic ``target_weights`` interface, which stays the decision core
        of the bundled plugins; new strategies may override ``on_bar``
        directly to consume the streamed events instead.
        """
        return self.target_weights(event.timestamp, equity)

    def note(self, date: pd.Timestamp, symbol: str) -> str:
        """Short human-readable state string for signal output (optional)."""
        return ""
