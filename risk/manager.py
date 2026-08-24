"""Portfolio discipline overlay: caps, stop loss, drawdown cooldown.

Rules (defaults):

- single-name target weight capped at 20% of equity
- stop loss: a position whose close falls to 8% below its all-in average
  cost is force-closed at the next open
- drawdown guard: if portfolio equity is more than 15% below its running
  peak, everything is liquidated at the next open and the book stays in
  cash for a cooldown (about 20 trading sessions), after which the peak
  resets and strategies may re-engage
- re-entry block: a symbol stopped out may not be re-bought for about 10
  sessions

All thresholds are evaluated on close marks, after costs, never on
hindsight. The point is to make the engine hard to blow up, not to add
alpha.
"""
from __future__ import annotations

from dataclasses import dataclass

import math


@dataclass
class RiskConfig:
    max_position_weight: float = 0.20
    stop_loss: float = -0.08
    max_drawdown: float = 0.15
    cooldown_days: int = 20
    reentry_cooldown: int = 10


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.cfg = config or RiskConfig()
        self.peak_equity: float | None = None
        self.cooldown_remaining: int = 0
        self.symbol_blocks: dict[str, int] = {}

    # ------------------------------------------------------------- queries
    @property
    def cooldown_active(self) -> bool:
        return self.cooldown_remaining > 0

    def blocked(self, symbol: str) -> bool:
        return self.symbol_blocks.get(symbol, 0) > 0

    # -------------------------------------------------------------- rules
    def on_close(self, equity: float, positions: dict, closes: dict) -> list[tuple[str, str]]:
        """Evaluate rules at the close; return forced sells as (symbol, reason).

        ``positions`` maps symbol -> object with ``shares`` and ``avg_cost``;
        ``closes`` maps symbol -> close price (NaN allowed).
        """
        forced: list[tuple[str, str]] = []

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining == 0:
                self.peak_equity = equity  # fresh peak after the cooldown
        self._tick_blocks()
        if self.cooldown_active:
            return forced

        self.peak_equity = equity if self.peak_equity is None else max(self.peak_equity, equity)
        drawdown = equity / self.peak_equity - 1.0
        if drawdown < -self.cfg.max_drawdown:
            for symbol, pos in positions.items():
                if getattr(pos, "shares", 0) > 0:
                    forced.append((symbol, "drawdown_flat"))
            self.cooldown_remaining = self.cfg.cooldown_days
            return forced

        stop_level = 1.0 + self.cfg.stop_loss
        for symbol, pos in positions.items():
            if getattr(pos, "shares", 0) <= 0:
                continue
            px = closes.get(symbol, float("nan"))
            if px is None or not math.isfinite(px):
                continue
            if px <= pos.avg_cost * stop_level:
                forced.append((symbol, "stop_loss"))
                self.symbol_blocks[symbol] = self.cfg.reentry_cooldown
        return forced

    def apply_caps(self, targets: dict[str, float]) -> dict[str, float]:
        """Clamp weights to the single-name cap, zero blocked symbols, keep sum <= 1."""
        cap = self.cfg.max_position_weight
        out: dict[str, float] = {}
        for symbol, weight in targets.items():
            if self.blocked(symbol):
                out[symbol] = 0.0
                continue
            out[symbol] = min(max(float(weight), 0.0), cap)
        total = sum(out.values())
        if total > 1.0 and total > 0.0:
            scale = 1.0 / total
            out = {s: w * scale for s, w in out.items()}
        return out

    # ------------------------------------------------------------ internals
    def _tick_blocks(self) -> None:
        for symbol in list(self.symbol_blocks):
            self.symbol_blocks[symbol] -= 1
            if self.symbol_blocks[symbol] <= 0:
                del self.symbol_blocks[symbol]
