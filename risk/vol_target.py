"""Volatility-targeted position sizing overlay (research item 7, step 1).

``scale = target_vol / realized_vol_20d``, clipped to [min_scale, max_scale].
The sizer stays inactive (scale = max_scale) until at least ``lookback``
return observations exist, so backtests keep their early history intact;
fail-closed semantics belong to the pre-trade gate, not here.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import pandas as pd

from strategies.base import Strategy


@dataclass(frozen=True)
class VolTargetConfig:
    lookback: int = 20            # trading days of returns
    target_vol: float = 0.15      # annualized target volatility
    max_scale: float = 1.0
    min_scale: float = 0.0
    periods_per_year: int = 252


class VolTargetSizer:
    def __init__(self, config: VolTargetConfig | None = None):
        self.cfg = config or VolTargetConfig()

    def realized_vol(self, returns: pd.Series) -> float | None:
        """Annualized std of the last ``lookback`` returns; None if short."""
        r = pd.Series(returns).dropna().astype(float)
        if len(r) < self.cfg.lookback:
            return None
        window = r.iloc[-self.cfg.lookback:]
        vol = float(window.std(ddof=1)) * math.sqrt(self.cfg.periods_per_year)
        return vol

    def scale(self, returns: pd.Series) -> float:
        """Sizing multiplier; inactive (max_scale) on insufficient data."""
        vol = self.realized_vol(returns)
        if vol is None or not math.isfinite(vol) or vol <= 0.0:
            return self.cfg.max_scale
        raw = self.cfg.target_vol / vol
        return min(max(raw, self.cfg.min_scale), self.cfg.max_scale)


class ScaledStrategy(Strategy):
    """Wrap a strategy and scale its target weights by book vol targeting.

    The book-vol proxy is the equal-weight average daily return across the
    panel universe, computed strictly from closes at or before the decision
    date (no look-ahead).
    """

    def __init__(self, inner: Strategy, panel: dict[str, pd.DataFrame],
                 config: VolTargetConfig | None = None):
        self.inner = inner
        self.panel = panel
        self.sizer = VolTargetSizer(config)
        self.name = f"vol_scaled[{inner.name}]"

    def book_returns(self, date: pd.Timestamp) -> pd.Series:
        per_symbol = []
        for df in self.panel.values():
            px = df["close"].loc[:pd.Timestamp(date)]
            if len(px) >= 2:
                per_symbol.append(px.pct_change())
        if not per_symbol:
            return pd.Series(dtype=float)
        combined = pd.concat(per_symbol, axis=1)
        return combined.mean(axis=1, skipna=True).dropna()

    def book_scale(self, date: pd.Timestamp) -> float:
        return self.sizer.scale(self.book_returns(date))

    def target_weights(self, date: pd.Timestamp, equity: float) -> dict[str, float]:
        targets = self.inner.target_weights(date, equity)
        if not targets:
            return {}
        scale = self.book_scale(date)
        return {symbol: weight * scale for symbol, weight in targets.items()}

    def note(self, date: pd.Timestamp, symbol: str) -> str:
        base = self.inner.note(date, symbol)
        tag = f"vol_scale={self.book_scale(date):.2f}"
        return f"{base} | {tag}" if base else tag
