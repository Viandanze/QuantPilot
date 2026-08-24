"""Event-core tests: data feed censoring, portfolio accounting, execution."""
import pandas as pd
import pytest

from backtest.costs import CostModel
from engine.datafeed import DataFeed
from engine.events import OPEN, FillEvent, MarketEvent, OrderEvent
from engine.execution import SimulatedExecutionHandler
from engine.portfolio import Portfolio


def mk_event(date, phase, prices):
    return MarketEvent(pd.Timestamp(date), phase, prices)


# ------------------------------------------------------------------ datafeed
def test_datafeed_emits_open_then_close_per_session():
    panel = make_panel({"AAA": [10.0, 11.0, 12.0]})
    feed = DataFeed(panel)
    feed.reset()
    events = [feed.next() for _ in range(6)]
    assert [e.phase for e in events] == [OPEN, CLOSE, OPEN, CLOSE, OPEN, CLOSE]
    assert [str(e.timestamp.date()) for e in events] == [
        "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03",
        "2024-01-04", "2024-01-04",
    ]
    assert feed.next() is None


def test_datafeed_prices_are_asof_and_censored():
    # AAA trades every session; BBB only on sessions 1 and 3 (skips session 2)
    dates = pd.bdate_range("2024-01-02", periods=3)
    aaa = pd.DataFrame(
        {"open": [10.0, 11.0, 12.0], "high": [12.0, 12.0, 13.0], "low": [9.0, 10.0, 11.0],
         "close": [10.5, 11.5, 12.5], "volume": [1e6] * 3}, index=dates,
    )
    bbb_idx = [dates[0], dates[2]]
    bbb = pd.DataFrame(
        {"open": [20.0, 22.0], "high": [21.0, 23.0], "low": [19.0, 21.0],
         "close": [20.5, 22.5], "volume": [1e6] * 2}, index=pd.DatetimeIndex(bbb_idx),
    )
    feed = DataFeed({"AAA": aaa, "BBB": bbb})
    feed.reset()

    open1, close1 = feed.next(), feed.next()
    assert open1.prices == {"AAA": 10.0, "BBB": 20.0}
    assert close1.prices == {"AAA": 10.5, "BBB": 20.5}

    open2, close2 = feed.next(), feed.next()
    # BBB skipped session 2: its last known prices carry forward (staleness
    # convention shared with the marking engine), never forward-looking data
    assert open2.prices == {"AAA": 11.0, "BBB": 20.0}
    assert close2.prices == {"AAA": 11.5, "BBB": 20.5}

    open3, close3 = feed.next(), feed.next()
    assert open3.prices == {"AAA": 12.0, "BBB": 22.0}
    assert close3.prices == {"AAA": 12.5, "BBB": 22.5}


def test_datafeed_nan_before_first_bar_and_reset_slicing():
    dates = pd.bdate_range("2024-01-02", periods=4)
    late = pd.DataFrame(
        {"open": [30.0], "high": [31.0], "low": [29.0], "close": [30.5], "volume": [1e6]},
        index=pd.DatetimeIndex([dates[2]]),
    )
    early = pd.DataFrame(
        {"open": [10.0, 10.0, 10.0, 10.0], "high": [11.0] * 4, "low": [9.0] * 4,
         "close": [10.5] * 4, "volume": [1e6] * 4}, index=dates,
    )
    feed = DataFeed({"EARLY": early, "LATE": late})
    feed.reset()
    o1, c1 = feed.next(), feed.next()
    assert pd.isna(o1.prices["LATE"]) and pd.isna(c1.prices["LATE"])
    o3, _ = feed.next(), feed.next()
    o5, _ = feed.next(), feed.next()
    assert o5.prices["LATE"] == 30.0
    assert o3.timestamp == dates[1]

    feed.reset(start=dates[2])
    assert feed.next().timestamp == dates[2]


# ----------------------------------------------------------------- portfolio
def test_portfolio_buy_then_sell_accounting():
    pf = Portfolio(100_000.0, ["AAA"])
    cost = CostModel()
    d1, d2 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")

    px = cost.buy_price(100.0)
    turnover = 500 * px
    fee = cost.buy_fee(turnover)
    pf.on_fill(FillEvent(d1, "AAA", "BUY", 500, px, turnover, fee, 0.0, "signal"))
    pos = pf.positions["AAA"]
    assert pf.cash == pytest.approx(100_000.0 - turnover - fee)
    assert pos.shares == 500 and pos.locked == 500
    assert pos.avg_cost == pytest.approx((turnover + fee) / 500)
    assert pos.avg_cost_gross == pytest.approx(turnover / 500)
    assert pos.entry_date == d1

    pf.rollover_t1()  # next session: shares become sellable
    spx = cost.sell_price(102.0)
    sturn = 500 * spx
    scomm = cost.commission(sturn)
    sstamp = sturn * cost.stamp_tax_rate
    pf.on_fill(FillEvent(d2, "AAA", "SELL", 500, spx, sturn, scomm, sstamp, "signal"))
    assert pf.positions["AAA"].shares == 0
    assert len(pf.round_trips) == 1
    trip = pf.round_trips[0]
    assert trip.pnl == pytest.approx((sturn - scomm - sstamp) - pos.avg_cost * 500)
    # ledger fully closed: cash explains itself from flows
    assert pf.cash == pytest.approx(100_000.0 - turnover - fee + sturn - scomm - sstamp)


def test_portfolio_ledger_audit_detects_tampering():
    pf = Portfolio(100_000.0, ["AAA"])
    cost = CostModel()
    px = cost.buy_price(100.0)
    turnover = 500 * px
    fee = cost.buy_fee(turnover)
    fill = FillEvent(pd.Timestamp("2024-01-02"), "AAA", "BUY", 500, px, turnover, fee, 0.0)
    pf.on_fill(fill)
    assert pf.fills_audited == 1 and pf.max_ledger_drift < 1e-9

    rogue = Portfolio(100_000.0, ["AAA"])
    rogue.on_fill(fill)
    rogue.cash += 5_000.0  # money printed out of thin air
    with pytest.raises(AssertionError, match="ledger conservation"):
        rogue.on_fill(
            FillEvent(pd.Timestamp("2024-01-03"), "AAA", "BUY", 100, px, 100 * px, fee, 0.0)
        )


def test_orders_from_weights_entry_exit_and_band():
    pf = Portfolio(1_000_000.0, ["AAA", "BBB"], rebalance_band=0.01)
    closes = {"AAA": 100.0, "BBB": 50.0}
    d = pd.Timestamp("2024-01-02")

    # entry from flat: 10% of 1m at 100 -> 1000 shares
    orders = pf.orders_from_weights(d, {"AAA": 0.10, "BBB": 0.0}, closes, 1_000_000.0, set())
    assert [(o.symbol, o.side, o.shares) for o in orders] == [("AAA", "BUY", 1000)]

    pf.positions["AAA"].shares = 1000
    # tiny drift below the band -> no churn
    orders = pf.orders_from_weights(d, {"AAA": 0.101}, closes, 1_000_000.0, set())
    assert orders == []
    # drift beyond the band -> rebalance
    orders = pf.orders_from_weights(d, {"AAA": 0.13}, closes, 1_000_000.0, set())
    assert [(o.symbol, o.side, o.shares) for o in orders] == [("AAA", "BUY", 300)]
    # full exit
    orders = pf.orders_from_weights(d, {"AAA": 0.0}, closes, 1_000_000.0, set())
    assert [(o.symbol, o.side, o.shares) for o in orders] == [("AAA", "SELL", 1000)]
    # forced symbols are excluded from signal orders
    orders = pf.orders_from_weights(d, {"AAA": 0.0}, closes, 1_000_000.0, {"AAA"})
    assert orders == []


# ----------------------------------------------------------------- execution
def test_execution_clamps_sells_to_t1_sellable():
    pf = Portfolio(100_000.0, ["AAA"])
    handler = SimulatedExecutionHandler(pf, CostModel())
    px = CostModel().buy_price(100.0)
    pf.on_fill(FillEvent(pd.Timestamp("2024-01-02"), "AAA", "BUY", 500, px, 500 * px,
                         CostModel().buy_fee(500 * px), 0.0))
    # same-session sell order (locked shares): nothing sellable, order drops
    handler.enqueue(OrderEvent(pd.Timestamp("2024-01-02"), "AAA", "SELL", 500))
    fills = handler.execute_pending(mk_event("2024-01-02", OPEN, {"AAA": 100.0}))
    assert fills == []

    pf.rollover_t1()
    handler.enqueue(OrderEvent(pd.Timestamp("2024-01-03"), "AAA", "SELL", 500))
    fills = handler.execute_pending(mk_event("2024-01-03", OPEN, {"AAA": 100.0}))
    assert len(fills) == 1 and fills[0].shares == 500


def test_execution_shrinks_buys_to_affordable_lots():
    pf = Portfolio(10_500.0, ["AAA"])
    handler = SimulatedExecutionHandler(pf, CostModel())
    handler.enqueue(OrderEvent(pd.Timestamp("2024-01-02"), "AAA", "BUY", 10_000))
    fills = handler.execute_pending(mk_event("2024-01-02", OPEN, {"AAA": 100.0}))
    assert len(fills) == 1
    fill = fills[0]
    assert fill.shares == 100  # only one lot is affordable at ~100.1 CNY
    assert fill.turnover + fill.fee <= 10_500.0
    # cannot afford even one lot at this price
    handler.enqueue(OrderEvent(pd.Timestamp("2024-01-03"), "AAA", "BUY", 100))
    assert handler.execute_pending(mk_event("2024-01-03", OPEN, {"AAA": 5_000.0})) == []


def test_execution_forced_exit_cancels_same_morning_buy():
    pf = Portfolio(100_000.0, ["AAA"])
    handler = SimulatedExecutionHandler(pf, CostModel())
    handler.enqueue(OrderEvent(pd.Timestamp("2024-01-02"), "AAA", "SELL", 100, "stop_loss"))
    handler.enqueue(OrderEvent(pd.Timestamp("2024-01-02"), "AAA", "BUY", 100))
    fills = handler.execute_pending(mk_event("2024-01-02", OPEN, {"AAA": 100.0}))
    # no position to sell, and the buy was cancelled by the forced exit
    assert fills == []
    assert handler.pending == []


def test_execution_drops_orders_without_printable_open():
    pf = Portfolio(100_000.0, ["AAA"])
    handler = SimulatedExecutionHandler(pf, CostModel())
    handler.enqueue(OrderEvent(pd.Timestamp("2024-01-02"), "AAA", "BUY", 100))
    fills = handler.execute_pending(mk_event("2024-01-02", OPEN, {"AAA": float("nan")}))
    assert fills == [] and handler.pending == []
