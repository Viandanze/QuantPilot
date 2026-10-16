"""Cross-sectional momentum rotation with periodic rebalancing."""
from __future__ import annotations

import pandas as pd

from strategies.base import Strategy, asof_row


class MomentumStrategy(Strategy):
    """Every ``rebalance_every`` sessions, rank symbols by trailing return.

    The top ``top_n`` symbols each get an equal slice (1/top_n of equity);
    everything else is sold. Between rebalance dates the previous targets are
    held unchanged. Ranking uses the trailing ``lookback``-session return up
    to the decision date only - no look-ahead.

    ``allow_negative`` (default True) keeps symbols with negative trailing
    return eligible for the top slots, so on always-investable universes
    (e.g. bond/gold/cross-border ETFs) the book still holds the best of a
    bad bunch. With False, non-positive-return names are skipped and their
    weight stays in cash until the next rebalance.

    ``symbols`` optionally restricts the ranking universe; it defaults to
    every symbol in the panel.
    """

    name = "momentum"

    def __init__(
        self,
        panel: dict[str, pd.DataFrame],
        lookback: int = 20,
        top_n: int = 5,
        rebalance_every: int = 20,
        symbols=None,
        allow_negative: bool = True,
    ):
        if lookback < 1 or top_n < 1 or rebalance_every < 1:
            raise ValueError("lookback, top_n and rebalance_every must be positive")
        self.symbols = sorted(symbols or panel.keys())
        self.lookback, self.top_n, self.every = lookback, top_n, rebalance_every
        self.allow_negative = allow_negative
        closes = pd.DataFrame({s: panel[s]["close"] for s in self.symbols}).sort_index()
        self.closes = closes
        self.returns = closes / closes.shift(lookback) - 1.0
        self.schedule = set(closes.index[lookback::rebalance_every])
        self._last: dict[str, float] | None = None
        self._ranks: dict[str, tuple[int, float]] = {}
        self._n_ranked = 0

    def _rebalance(self, date: pd.Timestamp) -> None:
        row = asof_row(self.returns, date).dropna()
        ranked = sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))
        self._ranks = {s: (i + 1, float(r)) for i, (s, r) in enumerate(ranked)}
        self._n_ranked = len(ranked)
        candidates = ranked if self.allow_negative else [
            (s, r) for s, r in ranked if r > 0.0]
        top = {s for s, _ in candidates[: self.top_n]}
        weight = 1.0 / self.top_n
        self._last = {s: (weight if s in top else 0.0) for s in self.symbols}

    def target_weights(self, date: pd.Timestamp, equity: float) -> dict[str, float]:
        if self._last is None or pd.Timestamp(date) in self.schedule:
            self._rebalance(pd.Timestamp(date))
        return dict(self._last)

    def note(self, date: pd.Timestamp, symbol: str) -> str:
        if not self._ranks:
            return "warming up"
        entry = self._ranks.get(symbol)
        if entry is None:
            return "no data"
        rank, ret = entry
        held = self._last.get(symbol, 0.0) > 0 if self._last else False
        state = "held" if held else "watch"
        return f"rank {rank}/{self._n_ranked} {state} (ret {ret:+.1%}/{self.lookback}d)"

    def next_rebalance(self, date: pd.Timestamp) -> pd.Timestamp | None:
        later = [d for d in self.schedule if d > pd.Timestamp(date)]
        return min(later) if later else None
