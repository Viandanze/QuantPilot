"""Look-ahead defences: three independent tripwires against future data.

1. **Prefix invariance** (the researcher's replay): rebuilding the whole
   pipeline from data that ends at a cutoff must reproduce, bar for bar,
   the prefix of the full-data run. A decision at session ``t`` may only
   depend on data at or before ``t``, so appending future data cannot
   change the past. Verified for all three bundled strategies on the
   built-in sample data at two cutoffs.

2. **Future-spike probe**: a synthetic panel that is flat until ``T`` and
   then quintuples. Any leak - in the feed, the strategy hooks, risk
   sizing or order generation - would front-run the spike and change
   trades before ``T``. The run ending at ``T`` must be identical with
   and without the spike, even though the panels diverge wildly after
   ``T``. The strategies are handed the *full spiked panel* on purpose:
   precomputed indicators must still answer as-of queries causally.

3. **Event-stream audits**: with ``trace_events=True`` the wire itself is
   inspected. Market events must carry the panel's own as-of prices
   (censoring at the door), signals and orders may only be born in the
   close phase, and every fill must print strictly after the order that
   produced it, in an open phase - the engine-level temporal lock,
   observed on the wire rather than trusted from the source.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from backtest.costs import CostModel
from backtest.engine import Engine
from data import loader
from risk.manager import RiskConfig, RiskManager
from strategies import create
from strategies.base import Strategy
from tests.conftest import make_panel

START = "2024-06-03"
END = "2025-12-31"
FETCH_BUFFER_DAYS = 365
CUTOFFS = ("2025-05-30", "2025-09-30")
STRATEGY_NAMES = ("dual_ma", "grid", "momentum")


def _trade_key(t):
    return (t.date, t.symbol, t.side, t.shares, t.price, t.fee,
            t.realized_pnl, t.cash_after, t.reason)


def _equity_and_trades(panel, strategy_name, start, end):
    engine = Engine(panel, create(strategy_name, panel), CostModel(),
                    RiskManager(RiskConfig()))
    res = engine.run(start=start, end=end)
    return res


# ------------------------------------------------------- prefix invariance
@pytest.fixture(scope="module")
def full_run():
    frames = loader.load_daily(
        loader.DEFAULT_SYMBOLS,
        pd.Timestamp(START) - pd.Timedelta(days=FETCH_BUFFER_DAYS),
        pd.Timestamp(END),
        offline=True,
    )
    panel = loader.build_panel(frames)
    return {name: _equity_and_trades(panel, name, START, END) for name in STRATEGY_NAMES}


@pytest.mark.parametrize("strategy_name", STRATEGY_NAMES)
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_truncated_data_reproduces_the_full_run_prefix(full_run, strategy_name, cutoff):
    """Rerun with data ending at the cutoff; the past must not change."""
    frames = loader.load_daily(
        loader.DEFAULT_SYMBOLS,
        pd.Timestamp(START) - pd.Timedelta(days=FETCH_BUFFER_DAYS),
        pd.Timestamp(cutoff),
        offline=True,
    )
    panel = loader.build_panel(frames)
    truncated = _equity_and_trades(panel, strategy_name, START, cutoff)
    full = full_run[strategy_name]

    prefix = full.equity.loc[: pd.Timestamp(cutoff)]
    assert list(truncated.equity.index) == list(prefix.index)
    assert (truncated.equity.values == prefix.values).all()

    full_trades = [t for t in full.trades if t.date <= pd.Timestamp(cutoff)]
    assert [_trade_key(t) for t in truncated.trades] == [_trade_key(t) for t in full_trades]


# ------------------------------------------------------------ future spike
def _wiggle_panel(n: int, sessions_dates=None) -> dict[str, pd.DataFrame]:
    """Deterministic wiggly flat panel (no drift) for the spike probe."""
    aaa = [100.0 + (i % 5) * 0.2 for i in range(n)]
    bbb = [50.0 + (i % 7) * 0.3 for i in range(n)]
    return make_panel({"AAA": aaa, "BBB": bbb}, start="2024-01-02")


def _spike_panel(base: dict[str, pd.DataFrame], spike_from: pd.Timestamp):
    """Copy of ``base`` with every price after ``spike_from`` quintupled."""
    spiked = {}
    for symbol, df in base.items():
        copy = df.copy()
        mask = copy.index > spike_from
        for col in ("open", "high", "low", "close"):
            copy.loc[mask, col] = copy.loc[mask, col] * 5.0
        spiked[symbol] = copy
    return spiked


@pytest.mark.parametrize("strategy_name", STRATEGY_NAMES)
def test_known_future_spike_cannot_be_front_run(strategy_name):
    """A quintuple spike after T must not alter any decision made by T."""
    flat = _wiggle_panel(260)
    dates = flat["AAA"].index
    t_index = 200
    spike_from = dates[t_index]

    spiked = _spike_panel(flat, spike_from)
    # sanity: the panels really do diverge after the spike date
    assert flat["AAA"]["close"].iloc[t_index + 1] * 5.0 == pytest.approx(
        spiked["AAA"]["close"].iloc[t_index + 1])
    assert (flat["AAA"]["close"].iloc[: t_index + 1]
            == spiked["AAA"]["close"].iloc[: t_index + 1]).all()

    run_flat = _equity_and_trades(flat, strategy_name, None, spike_from)
    run_spiked = _equity_and_trades(spiked, strategy_name, None, spike_from)

    assert list(run_flat.equity.index) == list(run_spiked.equity.index)
    assert (run_flat.equity.values == run_spiked.equity.values).all()
    assert ([_trade_key(t) for t in run_flat.trades]
            == [_trade_key(t) for t in run_spiked.trades])


# -------------------------------------------------------- event-stream audit
class FlipFlop(Strategy):
    """Alternating allocation between two symbols, decided on date alone."""

    name = "flip_flop"

    def target_weights(self, date, equity):
        on = (date - pd.Timestamp("2024-01-01")).days % 10 < 5
        return {"AAA": 1.0 if on else 0.0, "BBB": 0.0 if on else 0.6}


@pytest.fixture(scope="module")
def traced_run():
    panel = _wiggle_panel(60)
    engine = Engine(panel, FlipFlop(), CostModel(), RiskManager(RiskConfig()))
    return panel, engine.run(start=None, end=None, trace_events=True)


def test_market_events_carry_asof_prices(traced_run):
    """Every streamed price is the panel's own value at or before the date."""
    panel, res = traced_run
    trace = res.event_trace
    assert trace
    for event in trace:
        if type(event).__name__ != "MarketEvent":
            continue
        column = "open" if event.phase == "open" else "close"
        for symbol, price in event.prices.items():
            expected = panel[symbol][column].asof(event.timestamp)
            if math.isnan(expected):
                assert math.isnan(price)
            else:
                assert price == pytest.approx(expected, abs=1e-12)


def test_phases_signals_and_orders_are_time_locked(traced_run):
    """Decisions are born at the close; fills print at the next open."""
    _, res = traced_run
    trace = res.event_trace

    last_market = None            # most recent MarketEvent on the wire
    last_order_date: dict[tuple[str, str], pd.Timestamp] = {}
    cost = CostModel()
    saw_fill = saw_signal = saw_order = False

    for event in trace:
        kind = type(event).__name__
        if kind == "MarketEvent":
            assert event.phase in ("open", "close")
            if last_market is not None:
                # open(d) -> close(d) -> open(d+1) -> close(d+1) -> ...
                if last_market.phase == "open":
                    assert event.phase == "close"
                    assert event.timestamp == last_market.timestamp
                else:
                    assert event.phase == "open"
                    assert event.timestamp > last_market.timestamp
            last_market = event
        elif kind == "SignalEvent":
            saw_signal = True
            assert last_market.phase == "close"
            assert event.timestamp == last_market.timestamp
        elif kind == "OrderEvent":
            saw_order = True
            assert last_market.phase == "close"
            assert event.timestamp == last_market.timestamp
            last_order_date[(event.symbol, event.side)] = event.timestamp
        elif kind == "FillEvent":
            saw_fill = True
            # fills print in an open phase, on a strictly later session
            assert last_market.phase == "open"
            assert event.timestamp == last_market.timestamp
            key = (event.symbol, event.side)
            assert key in last_order_date, "fill without a preceding order"
            assert event.timestamp > last_order_date[key], \
                "fill printed on the session that produced its order"
            # the fill price is that morning's open, slipped adversarially
            raw = last_market.prices[event.symbol]
            slipped = (cost.buy_price(raw) if event.side == "BUY"
                       else cost.sell_price(raw))
            assert event.price == pytest.approx(slipped, abs=1e-12)
        else:  # pragma: no cover - unknown event type on the wire
            raise AssertionError(f"unexpected event on the wire: {kind!r}")

    assert saw_fill and saw_signal and saw_order, "trace is too thin to audit"
