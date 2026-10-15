"""ETF rotation on absolute + relative momentum with a defensive cash mode."""
from __future__ import annotations

import pandas as pd

from strategies.base import Strategy, asof_row


class EtfRotation(Strategy):
    """Hold the strongest ETFs, but only while they are actually rising.

    This is *not* :class:`strategies.momentum.MomentumStrategy` under
    another name - the two differ in kind:

    - ``MomentumStrategy`` is pure relative momentum: it always holds the
      ``top_n`` best-ranked names even when every trailing return is
      negative, i.e. it keeps the "least bad" losers through a bear market.
      ``EtfRotation`` adds an absolute momentum filter: a symbol qualifies
      only if its trailing ``lookback`` return is strictly above
      ``abs_threshold`` (default 0.0, i.e. positive). When nothing
      qualifies the book goes fully to cash (defensive mode) instead of
      grinding through the drawdown.
    - ``MomentumStrategy`` re-ranks every ``rebalance_every`` sessions and
      holds the stale ranking in between; ``EtfRotation`` re-evaluates the
      ranking on every decision date, so a name that loses its
      absolute-momentum edge is dropped at the next close decision.

    Selection: among symbols whose trailing return clears the absolute
    filter, rank by return (descending, symbol name as the tie-break), keep
    the best ``top_n``, and give each a fixed ``1/top_n`` slice of equity.
    Fewer qualifying names means the remainder intentionally stays in cash,
    and the gross weight can never exceed 1.0. Trailing returns are computed
    once over the panel (the rolling statistic at index ``t`` uses closes at
    or before ``t``) and read with an as-of lookup at the decision date - no
    look-ahead.
    """

    name = "etf_rotation"

    def __init__(
        self,
        panel: dict[str, pd.DataFrame],
        lookback: int = 20,
        top_n: int = 2,
        abs_threshold: float = 0.0,
        symbols=None,
    ):
        if lookback < 1 or top_n < 1:
            raise ValueError("lookback and top_n must be positive")
        self.symbols = sorted(symbols or panel.keys())
        if not self.symbols:
            raise ValueError("panel must contain at least one symbol")
        self.lookback = int(lookback)
        self.top_n = int(top_n)
        self.abs_threshold = float(abs_threshold)
        closes = pd.DataFrame({s: panel[s]["close"] for s in self.symbols}).sort_index()
        self.closes = closes
        self.returns = closes / closes.shift(self.lookback) - 1.0
        self._last: dict[str, float] = {s: 0.0 for s in self.symbols}
        self._ranks: dict[str, tuple[int, float]] = {}
        self._n_ranked = 0

    def target_weights(self, date: pd.Timestamp, equity: float) -> dict[str, float]:
        date = pd.Timestamp(date)
        weights = {s: 0.0 for s in self.symbols}
        row = asof_row(self.returns, date).dropna()
        ranked = sorted(
            ((s, float(r)) for s, r in row.items()), key=lambda kv: (-kv[1], kv[0])
        )
        eligible = [(s, r) for s, r in ranked if r > self.abs_threshold]
        picks = {s for s, _ in eligible[: self.top_n]}
        weight = 1.0 / self.top_n
        for s in picks:
            weights[s] = weight
        self._last = weights
        self._ranks = {s: (i + 1, r) for i, (s, r) in enumerate(ranked)}
        self._n_ranked = len(ranked)
        return dict(weights)

    def note(self, date: pd.Timestamp, symbol: str) -> str:
        if symbol not in self.returns.columns:
            return "no data"
        entry = self._ranks.get(symbol)
        if entry is None:
            return "warming up"
        rank, ret = entry
        if self._last.get(symbol, 0.0) > 0.0:
            state = "held"
        elif ret <= self.abs_threshold:
            state = "filtered"
        else:
            state = "watch"
        return f"rank {rank}/{self._n_ranked} {state} (ret {ret:+.1%}/{self.lookback}d)"
