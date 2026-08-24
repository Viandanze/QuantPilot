"""Event-driven backtest core.

The package deliberately keeps its ``__init__`` light (events only) so that
low-level modules - notably ``strategies.base``, which type-hints against
``MarketEvent`` - can import the event vocabulary without pulling in the
runner, the cost model or the risk overlay. Import the concrete modules
directly::

    from engine.runner import Engine            # event-driven backtest
    from engine.datafeed import DataFeed
    from engine.portfolio import Portfolio
    from engine.execution import SimulatedExecutionHandler
"""
from engine.events import (
    CLOSE,
    OPEN,
    Event,
    EventQueue,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)

__all__ = [
    "CLOSE",
    "OPEN",
    "Event",
    "EventQueue",
    "FillEvent",
    "MarketEvent",
    "OrderEvent",
    "SignalEvent",
]
