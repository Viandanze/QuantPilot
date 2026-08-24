"""Event vocabulary and queue tests: FIFO discipline, payload contracts."""
import pandas as pd
import pytest

from engine.events import (
    CLOSE,
    OPEN,
    EventQueue,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)


def test_queue_is_strict_fifo():
    queue = EventQueue()
    assert queue.empty() and len(queue) == 0
    a = MarketEvent(pd.Timestamp("2024-01-02"), OPEN, {"AAA": 10.0})
    b = SignalEvent(pd.Timestamp("2024-01-02"), {"AAA": 1.0})
    c = OrderEvent(pd.Timestamp("2024-01-02"), "AAA", "BUY", 100)
    for event in (a, b, c):
        queue.put(event)
    assert len(queue) == 3
    assert queue.get() is a
    assert queue.get() is b
    assert queue.get() is c
    assert queue.empty()


def test_events_are_frozen():
    event = OrderEvent(pd.Timestamp("2024-01-02"), "AAA", "BUY", 100, "signal")
    with pytest.raises(Exception):
        event.shares = 200  # type: ignore[misc]


def test_market_event_carries_phase_and_price_map():
    event = MarketEvent(pd.Timestamp("2024-01-02"), CLOSE, {"AAA": 10.5, "BBB": float("nan")})
    assert event.phase == CLOSE
    assert event.prices["AAA"] == pytest.approx(10.5)
    assert pd.isna(event.prices["BBB"])


def test_order_and_signal_payloads():
    order = OrderEvent(pd.Timestamp("2024-01-02"), "AAA", "SELL", 300, "stop_loss")
    assert (order.symbol, order.side, order.shares, order.reason) == ("AAA", "SELL", 300, "stop_loss")
    signal = SignalEvent(pd.Timestamp("2024-01-02"), {"AAA": 0.25})
    assert signal.weights == {"AAA": 0.25}


def test_fill_event_fee_sums_commission_and_stamp():
    fill = FillEvent(
        pd.Timestamp("2024-01-02"), "AAA", "SELL", 100, 99.9, 9_990.0,
        fee_commission=5.0, fee_stamp=9.99, reason="signal",
    )
    assert fill.fee == pytest.approx(14.99)


def test_events_share_the_timestamp_contract():
    stamp = pd.Timestamp("2024-01-02")
    for event in (
        MarketEvent(stamp, OPEN, {}),
        SignalEvent(stamp, {}),
        OrderEvent(stamp, "AAA", "BUY", 100),
        FillEvent(stamp, "AAA", "BUY", 100, 10.0, 1_000.0, 5.0, 0.0),
    ):
        assert event.timestamp == stamp
