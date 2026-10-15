"""Small-account guardrails: per-order weight cap and daily loss breaker.

Day trading a tiny book fails differently from a sized portfolio: one
over-sized clip or one bad morning can end the experiment. This overlay
adds two blunt rules on top of :mod:`risk.manager` without touching it:

- cap every single target weight at ``max_order_weight`` of equity
- trip a circuit breaker when the day's equity falls to ``daily_loss_limit``
  relative to the session-start equity

The guard is stateless on purpose: it only evaluates rules, and the caller
owns what "day start" means and what happens after a breaker trip.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SmallAccountConfig:
    # single order cap, fraction of equity. Calibrated 2026-09-21 for a
    # ~2000 CNY book trading board-lot ETFs: one lot (~228-269 CNY) is the
    # minimum tradeable unit and already ~11-13% of equity, so a 5% cap
    # would make every signal untradeable. 50% (half-book per name) keeps
    # a top-2 equal-weight 50/50 book intact while still forcing a
    # single-name concentrated signal down to half exposure. The daily
    # loss breaker remains the session-level bound.
    max_order_weight: float = 0.50
    daily_loss_limit: float = -0.03  # day return at/below this trips the breaker


class SmallAccountGuard:
    """Rule evaluator for order sizing caps and the daily loss circuit breaker."""

    def __init__(self, config: SmallAccountConfig | None = None):
        self.cfg = config or SmallAccountConfig()

    def cap_order(self, targets: dict[str, float]) -> dict[str, float]:
        """Clamp each weight into [0, max_order_weight], keep the sum <= 1.

        Negative weights are floored at 0 (no implicit shorting). If the
        clamped book still sums above 1.0 (many names at the cap), all
        weights are scaled down proportionally, mirroring
        ``RiskManager.apply_caps``.
        """
        cap = self.cfg.max_order_weight
        out = {s: min(max(float(w), 0.0), cap) for s, w in targets.items()}
        total = sum(out.values())
        if total > 1.0 and total > 0.0:
            scale = 1.0 / total
            out = {s: w * scale for s, w in out.items()}
        return out

    def on_equity_update(self, equity: float, day_start_equity: float) -> bool:
        """True when the day return is at/below ``daily_loss_limit``.

        The day return is ``equity / day_start_equity - 1``; a non-positive
        ``day_start_equity`` has no defined baseline and never trips.
        """
        if day_start_equity <= 0.0:
            return False
        day_return = equity / day_start_equity - 1.0
        return day_return <= self.cfg.daily_loss_limit
