"""DataFeed: the only door market data walks through.

The feed streams the panel as one ``MarketEvent`` per session phase - open
first, then close. Both phases carry *as-of* prices: each symbol's last open
(resp. close) at or before the session date, NaN if the symbol has no
history yet. A symbol that skips a session keeps its last known price, the
same staleness convention the marking engine uses, and an order for it can
still print at that stale open.

Censoring happens here, at the single point where data enters the event
stream: an event dated ``d`` is built exclusively from bars dated ``<= d``,
so no consumer downstream - strategy, risk overlay, portfolio sizing - can
observe the future even by accident.
"""
from __future__ import annotations

import math

import pandas as pd

from engine.events import CLOSE, OPEN, MarketEvent


def _asof(series: pd.Series, ts: pd.Timestamp) -> float:
    """Last value of ``series`` at or before ``ts`` (NaN if none)."""
    if len(series.index) == 0:
        return float("nan")
    pos = series.index.searchsorted(pd.Timestamp(ts), side="right") - 1
    if pos < 0:
        return float("nan")
    return float(series.iloc[pos])


class DataFeed:
    """Deterministic, replayable stream of session events from a panel."""

    def __init__(self, panel: dict[str, pd.DataFrame], symbols: list[str] | None = None):
        self.symbols = sorted(panel.keys()) if symbols is None else list(symbols)
        self.opens = {s: panel[s]["open"].sort_index() for s in self.symbols}
        self.closes = {s: panel[s]["close"].sort_index() for s in self.symbols}
        self.dates = sorted(set().union(*(set(self.opens[s].index) for s in self.symbols)))
        self._dates: list[pd.Timestamp] = []
        self._cursor = 0
        self._pending_phase: str | None = None
        self._current: pd.Timestamp | None = None

    # ------------------------------------------------------------ lifecycle
    def reset(self, start=None, end=None) -> None:
        """Arm the feed over the session range [start, end] (inclusive)."""
        dates = pd.DatetimeIndex(self.dates)
        if start is not None:
            dates = dates[dates >= pd.Timestamp(start)]
        if end is not None:
            dates = dates[dates <= pd.Timestamp(end)]
        self._dates = list(dates)
        self._cursor = 0
        self._pending_phase = None
        self._current = None

    def next(self) -> MarketEvent | None:
        """Return the next session event, open phase before close phase."""
        if self._pending_phase == OPEN:  # close phase of the current session
            self._pending_phase = None
            prices = {s: _asof(self.closes[s], self._current) for s in self.symbols}
            return MarketEvent(self._current, CLOSE, prices)
        if self._cursor >= len(self._dates):
            return None
        self._current = self._dates[self._cursor]
        self._cursor += 1
        self._pending_phase = OPEN
        prices = {s: _asof(self.opens[s], self._current) for s in self.symbols}
        return MarketEvent(self._current, OPEN, prices)

    # ------------------------------------------------------------- queries
    def sessions(self, start=None, end=None) -> list[pd.Timestamp]:
        """Session dates in [start, end] without consuming the stream."""
        dates = pd.DatetimeIndex(self.dates)
        if start is not None:
            dates = dates[dates >= pd.Timestamp(start)]
        if end is not None:
            dates = dates[dates <= pd.Timestamp(end)]
        return list(dates)

    @staticmethod
    def printable(price: float) -> bool:
        return math.isfinite(price) and price > 0
