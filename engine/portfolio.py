"""Portfolio: cash, positions, and the accounting that must never break.

The portfolio owns every yuan in the simulation. It applies fills produced
by the execution handler, marks positions to market on close events,
converts strategy signals into whole-lot orders, and - new in the
event-driven core - audits its own ledger after every fill.

Ledger model (average cost, all-in basis):

- ``avg_cost``       buy basis per share including buy fees (drives the
                     stop-loss rule and round-trip PnL, exactly as in the
                     original engine)
- ``avg_cost_gross`` buy basis per share excluding fees (price only)

Conservation identity checked after every fill:

    cash + sum(shares * avg_cost_gross) + total_fees
        == initial_capital + sum(realized_gross)

Read it as double-entry bookkeeping: every yuan is either sitting in cash,
tied up in inventory at price-only cost, or has been paid away as fees; the
only other source of money is realized gross price gains. Average-cost
division introduces float rounding of a few ulps per partial fill, so the
check uses a small tolerance and records the worst observed drift, which
reports surface as the "max ledger drift" integrity figure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from engine.events import FillEvent, MarketEvent, OrderEvent

LOT = 100  # A-share board lot


@dataclass
class Position:
    shares: int = 0
    locked: int = 0          # shares bought today, not sellable until tomorrow
    avg_cost: float = 0.0    # all-in buy basis per share (price + buy fees)
    entry_date: pd.Timestamp | None = None
    cycle_realized: float = 0.0
    avg_cost_gross: float = 0.0  # price-only basis per share (audit only)

    @property
    def sellable(self) -> int:
        return max(self.shares - self.locked, 0)


@dataclass
class Trade:
    date: pd.Timestamp
    symbol: str
    side: str                 # "BUY" | "SELL"
    shares: int
    price: float              # executed price (after slippage)
    turnover: float
    fee: float
    fee_commission: float       # all non-stamp fees: commission + transfer
    fee_stamp: float
    realized_pnl: float       # sells only; buys record 0.0
    cash_after: float
    reason: str               # "signal" | "stop_loss" | "drawdown_flat"


@dataclass
class RoundTrip:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    pnl: float
    exit_reason: str


class Portfolio:
    """Account state: the single writer of cash and position records."""

    def __init__(
        self,
        capital: float,
        symbols: list[str],
        lot: int = LOT,
        rebalance_band: float = 0.01,
        audit_tolerance: float | None = None,
    ):
        """``rebalance_band``: minimum drift (fraction of equity) between
        current and target position value before a partial rebalance trade
        fires; entries from flat and full exits always execute. The band
        prevents pointless 100-share churn from daily mark-to-market drift.

        ``audit_tolerance``: absolute CNY tolerance for the per-fill ledger
        identity; defaults to a few ulps scaled by capital."""
        self.capital = float(capital)
        self.symbols = list(symbols)
        self.lot = int(lot)
        self.rebalance_band = float(rebalance_band)
        self.audit_tolerance = (
            max(1e-6, 1e-8 * self.capital) if audit_tolerance is None else float(audit_tolerance)
        )
        self.cash = float(capital)
        self.positions = {s: Position() for s in self.symbols}
        self.trades: list[Trade] = []
        self.round_trips: list[RoundTrip] = []
        self.equity_points: dict[pd.Timestamp, float] = {}
        self.exposure_points: dict[pd.Timestamp, float] = {}
        self.last_prices: dict[str, float] = {s: float("nan") for s in self.symbols}

        # running audit accumulators
        self.total_costs = 0.0
        self.total_realized_gross = 0.0
        self.total_buy_turnover = 0.0
        self.total_sell_turnover = 0.0
        self.fills_audited = 0
        self.max_ledger_drift = 0.0

    # --------------------------------------------------------- state queries
    def sellable(self, symbol: str) -> int:
        """Shares of ``symbol`` that may legally be sold today (T+1 guard)."""
        return self.positions[symbol].sellable

    @property
    def equity_at_last_mark(self) -> float:
        if not self.equity_points:
            return self.capital
        return self.equity_points[max(self.equity_points)]

    # ------------------------------------------------------ session lifecycle
    def rollover_t1(self) -> None:
        """Start of a session: yesterday's buys become sellable."""
        for pos in self.positions.values():
            pos.locked = 0

    def mark_to_market(self, event: MarketEvent) -> float:
        """Close-phase marking: record equity and exposure for the session."""
        closes = event.prices
        self.last_prices = dict(closes)
        marked = {
            s: self.positions[s].shares * closes[s]
            for s in self.symbols
            if math.isfinite(closes[s]) and closes[s] > 0
        }
        equity = self.cash + sum(marked.values())
        self.equity_points[event.timestamp] = equity
        self.exposure_points[event.timestamp] = (
            sum(marked.values()) / equity if equity > 0 else 0.0
        )
        return equity

    # ----------------------------------------------------------------- fills
    def on_fill(self, fill: FillEvent) -> None:
        if fill.side == "SELL":
            self._apply_sell(fill)
        else:
            self._apply_buy(fill)
        self._audit_ledger(fill)

    def _apply_sell(self, fill: FillEvent) -> None:
        pos = self.positions[fill.symbol]
        self.cash += fill.turnover - fill.fee
        realized = (fill.turnover - fill.fee) - pos.avg_cost * fill.shares
        realized_gross = fill.turnover - pos.avg_cost_gross * fill.shares
        self.total_realized_gross += realized_gross
        self.total_costs += fill.fee
        self.total_sell_turnover += fill.turnover
        pos.cycle_realized += realized
        pos.shares -= fill.shares
        if pos.shares == 0:
            self.round_trips.append(
                RoundTrip(fill.symbol, pos.entry_date, fill.timestamp, pos.cycle_realized, fill.reason)
            )
            self.positions[fill.symbol] = Position()
        self.trades.append(
            Trade(fill.timestamp, fill.symbol, "SELL", fill.shares, fill.price,
                  fill.turnover, fill.fee, fill.fee_commission, fill.fee_stamp,
                  realized, self.cash, fill.reason)
        )

    def _apply_buy(self, fill: FillEvent) -> None:
        pos = self.positions[fill.symbol]
        self.cash -= fill.turnover + fill.fee
        basis = pos.avg_cost * pos.shares + fill.turnover + fill.fee
        gross_basis = pos.avg_cost_gross * pos.shares + fill.turnover
        pos.shares += fill.shares
        pos.avg_cost = basis / pos.shares
        pos.avg_cost_gross = gross_basis / pos.shares
        pos.locked += fill.shares  # T+1: not sellable today
        if pos.entry_date is None:
            pos.entry_date = fill.timestamp
        self.total_costs += fill.fee
        self.total_buy_turnover += fill.turnover
        self.trades.append(
            Trade(fill.timestamp, fill.symbol, "BUY", fill.shares, fill.price,
                  fill.turnover, fill.fee, fill.fee_commission, 0.0, 0.0,
                  self.cash, "signal")
        )

    # ----------------------------------------------------------------- audit
    def _ledger_value(self) -> float:
        """cash + inventory at price-only cost + fees already paid."""
        inventory = sum(p.shares * p.avg_cost_gross for p in self.positions.values())
        return self.cash + inventory + self.total_costs

    def _audit_ledger(self, fill: FillEvent) -> None:
        """Per-fill conservation check (see module docstring)."""
        drift = abs(self._ledger_value() - (self.capital + self.total_realized_gross))
        self.fills_audited += 1
        if drift > self.max_ledger_drift:
            self.max_ledger_drift = drift
        if drift > self.audit_tolerance:
            raise AssertionError(
                f"ledger conservation broken after {fill.side} of {fill.symbol} on "
                f"{fill.timestamp.date()}: drift {drift:.6f} CNY exceeds tolerance "
                f"{self.audit_tolerance:.6f} CNY"
            )

    def integrity(self) -> dict:
        """End-of-run reconciliation report for the HTML integrity section."""
        market_value = 0.0
        unrealized_net = 0.0
        unrealized_gross = 0.0
        unmarked = 0
        for s in self.symbols:
            pos = self.positions[s]
            px = self.last_prices.get(s, float("nan"))
            if not (math.isfinite(px) and px > 0):
                if pos.shares > 0:
                    unmarked += 1
                continue
            market_value += pos.shares * px
            if pos.shares > 0:
                unrealized_net += (px - pos.avg_cost) * pos.shares
                unrealized_gross += (px - pos.avg_cost_gross) * pos.shares
        final_equity = self.cash + market_value
        total_pnl = final_equity - self.capital
        realized_net = sum(t.realized_pnl for t in self.trades)
        cash_flow_rhs = (
            self.capital + self.total_sell_turnover - self.total_buy_turnover - self.total_costs
        )
        tol = self.audit_tolerance

        checks = {
            "per_fill_ledger": self.max_ledger_drift <= tol,
            "cash_flow_identity": abs(self.cash - cash_flow_rhs) <= tol,
            "equity_mark_identity": abs(final_equity - self.equity_at_last_mark) <= tol,
            "pnl_net_decomposition": abs(total_pnl - (realized_net + unrealized_net)) <= tol,
            "pnl_gross_decomposition": (
                abs(total_pnl - (self.total_realized_gross + unrealized_gross - self.total_costs)) <= tol
            ),
        }
        return {
            "fills_audited": self.fills_audited,
            "audit_tolerance": tol,
            "max_ledger_drift": self.max_ledger_drift,
            "all_checks_passed": all(checks.values()),
            "checks": checks,
            "unmarked_open_positions": unmarked,
            "reconciliation": {
                "initial_capital": self.capital,
                "final_equity": final_equity,
                "final_cash": self.cash,
                "final_market_value": market_value,
                "total_pnl": total_pnl,
                "realized_pnl_net": realized_net,
                "unrealized_pnl_net": unrealized_net,
                "realized_pnl_gross": self.total_realized_gross,
                "unrealized_pnl_gross": unrealized_gross,
                "total_costs": self.total_costs,
                "cash_flow_identity_rhs": cash_flow_rhs,
            },
        }

    # ------------------------------------------------------- order generation
    def orders_from_weights(
        self,
        timestamp: pd.Timestamp,
        weights: dict[str, float],
        closes: dict[str, float],
        equity: float,
        forced_symbols: set[str],
    ) -> list[OrderEvent]:
        """Convert capped target weights into next-open orders.

        Entries from flat and full exits always fire; partial adjustments
        only when the value drift clears the rebalance band. Symbols with a
        forced risk exit this session are left alone (their exit order was
        already issued).
        """
        orders: list[OrderEvent] = []
        band_value = self.rebalance_band * equity
        for s in self.symbols:
            if s in forced_symbols:
                continue
            px = closes[s]
            if not math.isfinite(px) or px <= 0:
                continue
            w = float(weights.get(s, 0.0))
            pos = self.positions[s]
            target_shares = int(round((w * equity) / px / self.lot)) * self.lot
            current = pos.shares
            diff = target_shares - current
            target_value = target_shares * px
            current_value = current * px
            if current == 0:
                if target_shares >= self.lot:
                    orders.append(OrderEvent(timestamp, s, "BUY", diff, "signal"))
            elif target_shares <= 0:
                orders.append(OrderEvent(timestamp, s, "SELL", current, "signal"))
            elif diff >= self.lot and target_value - current_value > band_value:
                orders.append(OrderEvent(timestamp, s, "BUY", diff, "signal"))
            elif diff <= -self.lot and current_value - target_value > band_value:
                orders.append(OrderEvent(timestamp, s, "SELL", -diff, "signal"))
        return orders
