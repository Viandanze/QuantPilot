"""Event-driven backtest runner: wires the five modules over one queue.

Session timeline (per trading day ``d``), driven entirely by the order in
which events flow through the queue:

```
  DataFeed
    |-- MarketEvent(open, d)  -----------------------------------------+
    |       ExecutionHandler.execute_pending (a generator)              |
    |         yields FillEvent -> queue -> Portfolio.on_fill            |
    |         ...booked before the generator sizes the next order       |
    |                                                                  |
    |-- MarketEvent(close, d) -----------------------------------------+
    |       Portfolio.mark_to_market -> equity/exposure point          |
    |       RiskManager.on_close -> forced OrderEvent(s)               |
    |       Strategy.on_bar -> SignalEvent                             |
    |       RiskManager.apply_caps -> capped weights                   |
    |       Portfolio.orders_from_weights -> OrderEvent(s)             |
    |       ExecutionHandler.enqueue (parked until d+1's open)         |
    v
  next session ...
```

Because the open phase of ``d`` is fully drained before any close-phase
processing starts, and orders are only ever created during close-phase
processing, an order can structurally never fill on the session that
produced its signal - even if strategy code tried. Strategies receive
neither the queue, the portfolio nor the execution handler: their only
output is a weight map. This is the engine-level temporal lock.

The runner reproduces the original vector-style daily loop's arithmetic
operation-for-operation (same fill priority, same float ordering), so
metrics are identical to the previous engine, not merely similar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from engine.datafeed import DataFeed
from engine.events import (
    Event,
    EventQueue,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from engine.execution import SimulatedExecutionHandler
from engine.portfolio import LOT, Portfolio
from risk.manager import RiskConfig, RiskManager

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from backtest.costs import CostModel


@dataclass
class BacktestResult:
    equity: pd.Series
    exposure: pd.Series
    benchmark: pd.Series | None
    trades: list = field(default_factory=list)
    round_trips: list = field(default_factory=list)
    fee_summary: dict = field(default_factory=dict)
    final_positions: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    integrity: dict = field(default_factory=dict)
    event_trace: list[Event] | None = None


class Engine:
    """Event-driven daily backtest with the classic constructor signature.

    Five decoupled modules cooperate through the queue: DataFeed (market
    events), Strategy (signals), RiskManager (caps and forced exits),
    Portfolio (accounting and sizing), ExecutionHandler (fills). This class
    is the only place that knows how they plug together.
    """

    def __init__(
        self,
        panel: dict[str, pd.DataFrame],
        strategy,
        cost_model: CostModel | None = None,
        risk_manager: RiskManager | None = None,
        benchmark: pd.Series | None = None,
        capital: float = 1_000_000.0,
        lot: int = LOT,
        rebalance_band: float = 0.01,
    ):
        self.symbols = sorted(panel.keys())
        self.panel = panel
        self.strategy = strategy
        if cost_model is None:
            from backtest.costs import CostModel  # local import: no import cycle
            cost_model = CostModel()
        self.cost = cost_model
        self.risk = risk_manager or RiskManager(RiskConfig())
        self.benchmark = None if benchmark is None else benchmark.sort_index()
        self.capital = float(capital)
        self.lot = int(lot)
        self.rebalance_band = float(rebalance_band)
        self.datafeed = DataFeed(panel, self.symbols)

    # ------------------------------------------------------------------ run
    def run(self, start=None, end=None, trace_events: bool = False) -> BacktestResult:
        dates = self.datafeed.sessions(start, end)
        if len(dates) < 2:
            raise ValueError("not enough trading dates in the requested range")
        self.datafeed.reset(start, end)

        portfolio = Portfolio(self.capital, self.symbols, lot=self.lot,
                              rebalance_band=self.rebalance_band)
        execution = SimulatedExecutionHandler(portfolio, self.cost, self.lot)
        queue: EventQueue = EventQueue()
        trace: list[Event] | None = [] if trace_events else None

        session: dict = {"closes": None, "equity": None, "forced": set()}

        while True:
            market_event = self.datafeed.next()
            if market_event is None:
                break
            queue.put(market_event)
            while not queue.empty():
                event = queue.get()
                if trace is not None:
                    trace.append(event)

                if isinstance(event, MarketEvent):
                    if event.phase == "open":
                        self._on_open(event, portfolio, execution, queue, trace)
                    else:
                        self._on_close(event, portfolio, queue, session)
                elif isinstance(event, SignalEvent):
                    capped = self.risk.apply_caps(event.weights)
                    for order in portfolio.orders_from_weights(
                        event.timestamp, capped, session["closes"], session["equity"],
                        session["forced"],
                    ):
                        queue.put(order)
                elif isinstance(event, OrderEvent):
                    execution.enqueue(event)
                else:  # FillEvent
                    portfolio.on_fill(event)

        return self._assemble(portfolio, trace)

    # --------------------------------------------------------------- open phase
    def _on_open(self, event: MarketEvent, portfolio: Portfolio,
                 execution: SimulatedExecutionHandler, queue: EventQueue, trace) -> None:
        """Execute yesterday's orders at today's open, one fill at a time.

        ``execute_pending`` is a generator that pauses after each fill, so
        each fill can be routed through the queue and booked by the
        portfolio *before* the generator sizes the next order. The queue is
        empty at this point, so the put/get round trip below returns
        exactly the event that was put - yet fills still flow through the
        queue like every other event. This preserves the original engine's
        sequential morning accounting: a sell frees cash that the very
        next buy's affordability check can see.
        """
        portfolio.rollover_t1()
        fills = execution.execute_pending(event)
        for fill in fills:
            queue.put(fill)
            queued = queue.get()  # FIFO: the fill we just put, nothing else
            if trace is not None:
                trace.append(queued)
            portfolio.on_fill(queued)

    # ------------------------------------------------------------ close phase
    def _on_close(self, event: MarketEvent, portfolio: Portfolio,
                  queue: EventQueue, session: dict) -> None:
        closes = dict(event.prices)
        equity = portfolio.mark_to_market(event)
        forced = self.risk.on_close(equity, portfolio.positions, closes)

        session["closes"] = closes
        session["equity"] = equity
        session["forced"] = {s for s, _ in forced}

        for symbol, reason in forced:
            if portfolio.positions[symbol].shares > 0:
                queue.put(
                    OrderEvent(event.timestamp, symbol, "SELL",
                               portfolio.positions[symbol].shares, reason)
                )
        if not self.risk.cooldown_active:
            weights = self.strategy.on_bar(event, equity)
            queue.put(SignalEvent(event.timestamp, weights))

    # ------------------------------------------------------------- assembly
    def _assemble(self, portfolio: Portfolio, trace) -> BacktestResult:
        equity = pd.Series(portfolio.equity_points).sort_index()
        exposure = pd.Series(portfolio.exposure_points).sort_index()
        bench = None
        if self.benchmark is not None:
            bench = self.benchmark.reindex(equity.index, method="ffill")

        commission = sum(t.fee_commission for t in portfolio.trades)
        stamp = sum(t.fee_stamp for t in portfolio.trades)
        fee_summary = {
            "commission": commission,
            "stamp": stamp,
            "total": commission + stamp,
            "slippage_est": sum(t.turnover * self.cost.slippage_rate for t in portfolio.trades),
        }
        final_positions = {
            s: (p.shares, p.avg_cost)
            for s, p in portfolio.positions.items()
            if p.shares > 0
        }
        meta = {
            "strategy": getattr(self.strategy, "name", type(self.strategy).__name__),
            "symbols": list(self.symbols),
            "capital": self.capital,
            "lot": self.lot,
            "start": equity.index[0],
            "end": equity.index[-1],
        }
        return BacktestResult(
            equity=equity,
            exposure=exposure,
            benchmark=bench,
            trades=portfolio.trades,
            round_trips=portfolio.round_trips,
            fee_summary=fee_summary,
            final_positions=final_positions,
            meta=meta,
            integrity=portfolio.integrity(),
            event_trace=trace,
        )
