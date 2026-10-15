"""Fail-closed pre-trade risk gate (research item 2, lean version).

Sits between strategy signals and any order submission. Every check fails
closed: a rejected intent never trades, and a failed audit write rejects the
order too. All violations of one intent are reported at once; every decision
is appended to a JSONL audit log when a path is configured.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PreTradeConfig:
    max_order_notional: float = 300_000.0   # CNY per order
    max_position_weight: float = 0.20       # single-name cap, mirrors RiskConfig
    max_price_deviation: float = 0.03       # |order_px / reference_px - 1|
    daily_max_loss: float = 0.05            # kill switch on session PnL
    max_orders_per_day: int = 20
    weight_epsilon: float = 1e-9


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str                              # "buy" | "sell"
    quantity: int
    price: float
    reference_price: float | None = None   # strategy's reference close


@dataclass(frozen=True)
class AccountState:
    equity: float
    cash: float
    position_weights: dict[str, float]     # symbol -> current weight
    orders_placed_today: int = 0
    day_pnl_pct: float = 0.0               # vs previous session close equity


@dataclass(frozen=True)
class Decision:
    approved: bool
    reasons: tuple[str, ...] = field(default=())


class PreTradeGate:
    def __init__(self, config: PreTradeConfig | None = None,
                 audit_path: str | Path | None = None):
        self.cfg = config or PreTradeConfig()
        self.audit_path = Path(audit_path) if audit_path else None

    # -------------------------------------------------------------- checks
    def check(self, order: OrderIntent, state: AccountState) -> Decision:
        cfg = self.cfg
        reasons: list[str] = []

        if order.side not in ("buy", "sell"):
            reasons.append("malformed_order")
        if order.quantity <= 0 or order.price <= 0:
            reasons.append("malformed_order")
        if state.equity <= 0:
            reasons.append("malformed_state")

        # account-level rules block every intent, buys and sells alike
        if state.day_pnl_pct <= -cfg.daily_max_loss:
            reasons.append("kill_switch_daily_loss")
        if state.orders_placed_today >= cfg.max_orders_per_day:
            reasons.append("order_rate_cap")

        notional = order.quantity * order.price
        if order.side == "buy":
            if notional > cfg.max_order_notional:
                reasons.append("order_notional_cap")
            if notional > state.cash:
                reasons.append("insufficient_cash")
            current = state.position_weights.get(order.symbol, 0.0)
            projected = current + notional / state.equity if state.equity > 0 else float("inf")
            if projected > cfg.max_position_weight + cfg.weight_epsilon:
                reasons.append("position_weight_cap")
            if (order.reference_price is not None and order.reference_price > 0
                    and abs(order.price / order.reference_price - 1.0) > cfg.max_price_deviation):
                reasons.append("price_deviation")
        elif order.side == "sell":
            if state.position_weights.get(order.symbol, 0.0) <= 0.0:
                reasons.append("no_position")

        approved = not reasons
        reasons = list(reasons)
        if not self._audit(order, state, approved):
            reasons.append("audit_write_failed")
            approved = False
        return Decision(approved=approved, reasons=tuple(reasons))

    # --------------------------------------------------------------- audit
    def _audit(self, order: OrderIntent, state: AccountState, approved: bool) -> bool:
        """Append one JSONL record; False means the audit itself failed."""
        if self.audit_path is None:
            return True
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "order": {
                "symbol": order.symbol, "side": order.side,
                "quantity": order.quantity, "price": order.price,
                "reference_price": order.reference_price,
            },
            "day_pnl_pct": state.day_pnl_pct,
            "orders_placed_today": state.orders_placed_today,
            "approved": approved,
        }
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except OSError:
            return False
