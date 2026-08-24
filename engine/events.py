"""Typed event vocabulary and the FIFO queue that carries it.

Every component of the backtest core communicates exclusively through the
four event types below; nothing else crosses module boundaries:

- ``MarketEvent``  one session's printable prices, emitted by the DataFeed in
                   two phases ("open", then "close"). Each phase carries
                   as-of prices only: the last open (resp. close) at or before
                   the session date, per symbol. This is the single point
                   where market data enters the event stream, so it is also
                   where data beyond the session date is censored.
- ``SignalEvent``  desired target weights, produced by a Strategy during
                   close-phase processing.
- ``OrderEvent``   a sized order derived from a signal (or from a forced risk
                   exit). Orders are created only during close-phase
                   processing, so they can only ever be consumed by the next
                   session's open phase - the temporal lock of the engine.
- ``FillEvent``    an executed trade produced by the ExecutionHandler during
                   an open phase, net of slippage and fees.

The queue itself is a deliberately dumb FIFO: all causality comes from the
order in which modules emit events, and only the runner moves events between
modules. Strategies never receive the queue, the portfolio, or the execution
handler - their only output is a weight map - so the close-decision /
next-open-fill timeline cannot be bypassed from strategy code.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

OPEN = "open"
CLOSE = "close"


@dataclass(frozen=True)
class Event:
    """Base type: every event is stamped with the session it belongs to."""

    timestamp: pd.Timestamp


@dataclass(frozen=True)
class MarketEvent(Event):
    """Printable prices for one session phase.

    ``prices`` maps symbol -> as-of price. For the open phase it is each
    symbol's last open at or before the session; for the close phase, each
    symbol's last close at or before the session. Symbols with no history
    yet map to NaN. Nothing dated after ``timestamp`` is ever present.
    """

    phase: str = OPEN
    prices: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalEvent(Event):
    """Desired target weights {symbol: weight} as of the close of a session."""

    weights: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderEvent(Event):
    """An order sized in whole lots, awaiting the next session's open."""

    symbol: str = ""
    side: str = ""        # "BUY" | "SELL"
    shares: int = 0
    reason: str = "signal"  # "signal" | "stop_loss" | "drawdown_flat"


@dataclass(frozen=True)
class FillEvent(Event):
    """An executed trade, net of slippage, commission and stamp tax."""

    symbol: str = ""
    side: str = ""        # "BUY" | "SELL"
    shares: int = 0
    price: float = 0.0    # executed price (after adverse slippage)
    turnover: float = 0.0
    fee_commission: float = 0.0
    fee_stamp: float = 0.0
    reason: str = "signal"

    @property
    def fee(self) -> float:
        return self.fee_commission + self.fee_stamp


class EventQueue:
    """Unbounded FIFO event channel between engine components."""

    def __init__(self) -> None:
        self._items: deque[Event] = deque()

    def put(self, event: Event) -> None:
        self._items.append(event)

    def get(self) -> Event:
        return self._items.popleft()

    def empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)
