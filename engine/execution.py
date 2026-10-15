"""Simulated execution handler: pending orders become fills at next open.

The handler holds orders created during a session's close phase and executes
them against the *next* session's open-phase MarketEvent. This is the second
half of the engine's temporal lock: an order can physically not fill on the
bar that produced its signal, because the runner enqueues orders only after
the open phase of that session has already been consumed.

``execute_pending`` is a *generator*: it executes one order per resumption
and yields the resulting fill. The runner routes each yielded fill through
the event queue and lets the portfolio apply it before resuming the
generator, so morning sequencing is preserved exactly - forced exits free
cash before strategy sells, and sells free cash before buys size their
affordability check.

Execution rules, mirroring the original engine exactly:

- priority at the open: forced risk exits first, then strategy sells, then
  buys; a morning forced exit cancels same-morning entries for that symbol
- sells are clamped to the T+1 sellable amount and dropped if nothing is
  sellable (the next decision re-issues what is still wanted)
- buys shrink by whole lots until cash covers turnover plus fees
- fills are priced with adverse slippage and charged commission (min 5 CNY)
  plus transfer fee on both sides, and stamp tax on sells
- orders that cannot print (no known open price) are dropped, not deferred
"""
from __future__ import annotations

from collections.abc import Iterator

from engine.datafeed import DataFeed
from engine.events import FillEvent, MarketEvent, OrderEvent
from engine.portfolio import Portfolio

FORCED_REASONS = ("stop_loss", "drawdown_flat")


class SimulatedExecutionHandler:
    """Fill pending orders at the next session's open, net of frictions."""

    def __init__(self, portfolio: Portfolio, cost_model, lot: int = 100):
        self.portfolio = portfolio
        self.cost = cost_model
        self.lot = int(lot)
        self.pending: list[OrderEvent] = []

    def enqueue(self, order: OrderEvent) -> None:
        """Park an order for the next open-phase event."""
        self.pending.append(order)

    def execute_pending(self, event: MarketEvent) -> Iterator[FillEvent]:
        """Execute pending orders one per resumption, in priority order."""
        pending, self.pending = self.pending, []
        forced = [o for o in pending if o.reason in FORCED_REASONS]
        signal_sells = [o for o in pending if o.side == "SELL" and o.reason == "signal"]
        buys = [o for o in pending if o.side == "BUY"]
        forced_symbols = {o.symbol for o in forced}

        for order in forced:
            yield from self._sell(order, event)
        for order in signal_sells:
            yield from self._sell(order, event)
        for order in buys:
            if order.symbol in forced_symbols:
                continue  # a forced exit cancels same-morning entries
            yield from self._buy(order, event)

    # ------------------------------------------------------------ execution
    def _sell(self, order: OrderEvent, event: MarketEvent) -> Iterator[FillEvent]:
        raw = event.prices.get(order.symbol, float("nan"))
        if not DataFeed.printable(raw):
            return  # no printable open; the next decision will re-issue
        shares = min(order.shares, self.portfolio.sellable(order.symbol))
        if shares <= 0:
            return  # T+1 hard guard
        px = self.cost.sell_price(raw)
        turnover = shares * px
        commission = self.cost.commission(turnover) + turnover * self.cost.transfer_fee_rate
        stamp = turnover * self.cost.stamp_tax_rate
        yield FillEvent(
            event.timestamp, order.symbol, "SELL", shares, px, turnover,
            commission, stamp, order.reason,
        )

    def _buy(self, order: OrderEvent, event: MarketEvent) -> Iterator[FillEvent]:
        raw = event.prices.get(order.symbol, float("nan"))
        if not DataFeed.printable(raw) or order.shares < self.lot:
            return
        px = self.cost.buy_price(raw)
        shares = order.shares
        turnover = 0.0
        fee = 0.0
        affordable = False
        while shares >= self.lot:
            turnover = shares * px
            fee = self.cost.buy_fee(turnover)
            if turnover + fee <= self.portfolio.cash:
                affordable = True
                break
            shares -= self.lot
        if not affordable:
            return
        yield FillEvent(
            event.timestamp, order.symbol, "BUY", shares, px, turnover,
            fee, 0.0, "signal",
        )
