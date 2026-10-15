"""Bollinger-band mean reversion: buy the first break below the lower band."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy, asof_value


class MeanReversion(Strategy):
    """Buy the first close below the lower Bollinger band, exit back at the mean.

    Per symbol, a rolling ``window``-session mean and standard deviation of
    the close define the bands (MA +/- ``num_std`` * std). The first close
    that drops below the lower band - the previous close was not already
    below it - opens a ``max_weight`` position; the position is held while
    the close keeps sliding under the band, and is flattened on the first
    close back at or above the rolling mean (mean reversion complete).
    Continued slides are deliberately not chased: only fresh breaks trigger
    entries, so a long breakdown never pyramids into the falling knife.

    Causality: both rolling statistics at index ``t`` use closes at or
    before ``t`` only, the holding state at ``t`` is derived from those
    statistics plus the state at ``t - 1``, and the decision date reads the
    precomputed state with an as-of lookup - a decision can never see
    future data. When several symbols hold positions at once and the gross
    weight would exceed 1.0, every weight is scaled by the same factor
    (equal-ratio compression) so total exposure is capped at exactly 1.0.
    With fewer than ``window`` sessions of history the bands are undefined
    and every weight is 0.0.
    """

    name = "mean_reversion"

    def __init__(
        self,
        panel: dict[str, pd.DataFrame],
        window: int = 20,
        num_std: float = 2.0,
        max_weight: float = 0.25,
        symbols=None,
    ):
        if window < 2:
            raise ValueError("window must be >= 2")
        if num_std <= 0:
            raise ValueError("num_std must be positive")
        if not 0.0 < max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1]")
        self.symbols = list(symbols or panel.keys())
        if not self.symbols:
            raise ValueError("panel must contain at least one symbol")
        self.window = int(window)
        self.num_std = float(num_std)
        self.max_weight = float(max_weight)

        self.closes: dict[str, pd.Series] = {}
        self.ma: dict[str, pd.Series] = {}
        self.lower: dict[str, pd.Series] = {}
        self.weights: dict[str, pd.Series] = {}
        for s in self.symbols:
            close = panel[s]["close"].sort_index().astype(float)
            ma = close.rolling(self.window).mean()
            std = close.rolling(self.window).std()
            lower = ma - self.num_std * std
            self.closes[s] = close
            self.ma[s] = ma
            self.lower[s] = lower
            self.weights[s] = self._state_weights(close, lower, ma)

    def _state_weights(
        self, close: pd.Series, lower: pd.Series, ma: pd.Series
    ) -> pd.Series:
        """Causal per-session book state: in from a fresh band break to the mean.

        Row ``i`` of the result depends only on closes at or before ``i``:
        a fresh break (close below the band while the prior close was not)
        flips the state on, a close back at or above the rolling mean flips
        it off, anything else leaves the state unchanged.
        """
        close_v = close.to_numpy(dtype=float)
        lower_v = lower.to_numpy(dtype=float)
        ma_v = ma.to_numpy(dtype=float)
        below = close_v < lower_v            # NaN bands compare False -> warmup is flat
        prev_below = np.concatenate(([False], below[:-1]))
        held = np.zeros(len(close_v), dtype=bool)
        cur = False
        for i in range(len(close_v)):
            if cur:
                if close_v[i] >= ma_v[i]:    # back at/above the mean: reversion done
                    cur = False
            elif below[i] and not prev_below[i]:  # first close below the band
                cur = True
            held[i] = cur
        return pd.Series(np.where(held, self.max_weight, 0.0), index=close.index)

    def target_weights(self, date: pd.Timestamp, equity: float) -> dict[str, float]:
        date = pd.Timestamp(date)
        out: dict[str, float] = {}
        for s in self.symbols:
            w = asof_value(self.weights[s], date)
            out[s] = w if np.isfinite(w) else 0.0
        total = sum(out.values())
        if total > 1.0:                      # equal-ratio compression to <= 1.0
            scale = 1.0 / total
            out = {s: w * scale for s, w in out.items()}
        return out

    def note(self, date: pd.Timestamp, symbol: str) -> str:
        if symbol not in self.closes:
            return "no data"
        px = asof_value(self.closes[symbol], date)
        if not np.isfinite(px):
            return "no data"
        ma = asof_value(self.ma[symbol], date)
        if not np.isfinite(ma):
            return "warming up"
        lower = asof_value(self.lower[symbol], date)
        state = "held" if asof_value(self.weights[symbol], date) > 0 else "flat"
        return f"close {px:.2f} vs lower {lower:.2f} (MA {ma:.2f}), {state}"
