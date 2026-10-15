"""Engine tests: next-open execution, costs, T+1, caps, forced exits."""
import pytest

from backtest.costs import CostModel
from backtest.engine import Engine, Position
from risk.manager import RiskConfig, RiskManager

from tests.conftest import AlwaysLong, LongThenFlat, make_panel


def full_trust_risk(**overrides):
    """Risk manager with caps/stops relaxed (engine mechanics in focus)."""
    cfg = RiskConfig(max_position_weight=1.0, stop_loss=-1.0, max_drawdown=1.0)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return RiskManager(cfg)


def test_t1_lock_property():
    pos = Position(shares=500, locked=500)
    assert pos.sellable == 0            # bought today -> nothing sellable
    pos.locked = 200
    assert pos.sellable == 300


def test_buy_fills_next_open_with_costs():
    panel = make_panel({"TEST": [100.0] * 5})
    engine = Engine(panel, AlwaysLong(), CostModel(), full_trust_risk(), capital=100_000.0)
    result = engine.run()

    assert result.equity.iloc[0] == pytest.approx(100_000.0)  # idle on day 1
    buys = [t for t in result.trades if t.side == "BUY"]
    assert len(buys) == 1
    trade = buys[0]
    assert trade.date == result.equity.index[1]               # next session open
    assert trade.price == pytest.approx(100.0 * 1.001)        # slippage
    assert trade.shares == 900                                # cash-affordable lots
    fee = max(900 * trade.price * 3.0e-4, 5.0) + 900 * trade.price * 1e-5  # commission + transfer
    assert trade.fee == pytest.approx(fee)
    assert trade.fee_stamp == 0.0                             # no stamp tax on buys
    assert trade.cash_after == pytest.approx(100_000.0 - 900 * trade.price - fee)

    last_close = 100.0
    expected_equity = trade.cash_after + 900 * last_close
    assert result.equity.iloc[-1] == pytest.approx(expected_equity)


def test_sell_deferred_one_session():
    panel = make_panel({"TEST": [100.0] * 6})
    dates = panel["TEST"].index
    engine = Engine(
        panel, LongThenFlat(exit_from=dates[1]), CostModel(), full_trust_risk(),
        capital=100_000.0,
    )
    result = engine.run()
    assert [(t.date, t.side) for t in result.trades] == [
        (dates[1], "BUY"),
        (dates[2], "SELL"),  # decided at close of d1, filled at open of d2
    ]
    sell = result.trades[1]
    assert sell.price == pytest.approx(100.0 * 0.999)
    assert sell.fee == pytest.approx(
        max(sell.turnover * 3.0e-4, 5.0) + sell.turnover * (5e-4 + 1e-5)  # commission + stamp + transfer
    )
    assert result.round_trips[0].pnl < 0  # round trip pays the spread + fees


def test_single_name_cap_enforced():
    panel = make_panel({"TEST": [100.0] * 5})
    engine = Engine(panel, AlwaysLong(), CostModel(), RiskManager(), capital=1_000_000.0)
    result = engine.run()
    shares, avg_cost = result.final_positions["TEST"]
    assert shares == 2_000                                 # 20% of 1m at 100 CNY
    assert avg_cost > 100.0                                # all-in basis includes costs
    assert shares * 100.0 <= 0.2 * 1_000_000.0 + 1e-6


def test_stop_loss_forces_exit_next_open():
    prices = [100.0] * 2 + [90.0] * 4
    panel = make_panel({"TEST": prices})
    engine = Engine(panel, AlwaysLong(), CostModel(), full_trust_risk(stop_loss=-0.08),
                    capital=1_000_000.0)
    result = engine.run()

    stops = [t for t in result.trades if t.reason == "stop_loss"]
    assert len(stops) == 1
    assert stops[0].date == result.equity.index[3]          # close d2 breach -> open d3
    assert "TEST" not in result.final_positions
    # blocked for re-entry while the remaining flat sessions pass
    assert len(result.trades) == 2


def test_drawdown_flat_and_cooldown_then_reentry():
    prices = [100.0] * 2 + [70.0] + [50.0] * 30
    panel = make_panel({"TEST": prices})
    engine = Engine(
        panel, AlwaysLong(), CostModel(),
        full_trust_risk(max_drawdown=0.15),  # drawdown guard ON, stop/cap relaxed
        capital=1_000_000.0,
    )
    result = engine.run()

    flats = [t for t in result.trades if t.reason == "drawdown_flat"]
    assert len(flats) == 1
    assert flats[0].date == result.equity.index[3]

    buy_dates = [t.date for t in result.trades if t.side == "BUY"]
    later_buys = [d for d in buy_dates if d > flats[0].date]
    assert later_buys == [result.equity.index[23]]  # 20-session cooldown, then re-entry


def test_cash_never_negative():
    prices = [100.0, 101.0, 99.0, 97.0, 103.0, 104.0, 98.0, 96.0, 105.0] * 4
    panel = make_panel({"TEST": prices})
    engine = Engine(panel, AlwaysLong(), CostModel(), RiskManager(), capital=50_000.0)
    result = engine.run()
    assert min(t.cash_after for t in result.trades) >= 0.0
    assert (result.equity > 0).all()
