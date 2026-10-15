"""Money conservation: every yuan stays on the books, after every fill.

The portfolio runs a double-entry style audit after *each* fill:

    cash + sum(shares * avg_cost_gross) + total_fees
        == initial_capital + sum(realized_gross)

and a set of end-of-run identities:

    final_equity == final_cash + final_market_value
    total_pnl    == realized_net + unrealized_net
    total_pnl    == realized_gross + unrealized_gross - total_costs
    final_cash   == capital + sells - buys - total_costs

This module pins those identities three ways: hand-computed arithmetic on
a fully manual fill sequence, an engine-integration run whose round trip
must close exactly, and a tamper probe proving the audit actually fires
when money goes missing.
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest.costs import CostModel
from backtest.engine import Engine
from engine.events import FillEvent, MarketEvent
from engine.portfolio import Portfolio
from risk.manager import RiskConfig, RiskManager
from tests.conftest import AlwaysLong, LongThenFlat, make_panel

COST = CostModel()


def _fill(date, symbol, side, shares, raw_price, commission, stamp=0.0):
    price = COST.buy_price(raw_price) if side == "BUY" else COST.sell_price(raw_price)
    turnover = shares * price
    return FillEvent(pd.Timestamp(date), symbol, side, shares, price, turnover,
                     commission, stamp, "signal")


# ------------------------------------------------- hand-computed arithmetic
def test_hand_computed_round_trip_conserves_every_yuan():
    pf = Portfolio(100_000.0, ["AAA"])

    # --- buy 100 shares at a raw open of 100 -----------------------------
    raw_buy = 100.0
    buy_px = COST.buy_price(raw_buy)             # 100.1 after slippage
    buy_turnover = 100 * buy_px                  # 10_010.0
    buy_fee = COST.buy_fee(buy_turnover)         # 5.0 min commission + 0.1001 transfer
    assert buy_px == pytest.approx(100.1)
    assert buy_turnover == pytest.approx(10_010.0)
    assert buy_fee == pytest.approx(5.1001)

    pf.on_fill(_fill("2024-01-02", "AAA", "BUY", 100, raw_buy, buy_fee))
    assert pf.cash == pytest.approx(100_000.0 - 10_015.1001)        # 89_984.8999
    pos = pf.positions["AAA"]
    assert pos.shares == 100 and pos.locked == 100                  # T+1
    assert pos.avg_cost == pytest.approx(100.151001)                # (10_010 + 5.1001) / 100
    assert pos.avg_cost_gross == pytest.approx(100.1)               # price-only basis

    # ledger identity right after the buy: nothing realized yet
    # 89_984.8999 cash + 10_010 inventory + 5.1001 fees == 100_000 capital
    assert pf._ledger_value() == pytest.approx(100_000.0)

    # --- sell all 100 shares at a raw open of 110 the next morning -------
    pf.rollover_t1()                                                # yesterday's buys unlock
    raw_sell = 110.0
    sell_px = COST.sell_price(raw_sell)         # 109.89 after slippage
    sell_turnover = 100 * sell_px                # 10_989.0
    sell_fee = COST.sell_fee(sell_turnover)      # 5.0 commission + 5.4945 stamp + 0.10989 transfer
    assert sell_px == pytest.approx(109.89)
    assert sell_turnover == pytest.approx(10_989.0)
    assert sell_fee == pytest.approx(10.60439)

    pf.on_fill(_fill("2024-01-03", "AAA", "SELL", 100, raw_sell,
                     COST.commission(sell_turnover) + sell_turnover * COST.transfer_fee_rate,
                     sell_turnover * COST.stamp_tax_rate))
    assert pf.cash == pytest.approx(89_984.8999 + 10_989.0 - 10.60439)  # 100_963.29551

    # cash-flow identity: capital + sells - buys - fees
    assert pf.cash == pytest.approx(100_000.0 + 10_989.0 - 10_010.0 - 15.70449)

    # realized PnL, gross and net, hand-computed
    assert pf.total_realized_gross == pytest.approx(10_989.0 - 10_010.0)   # 979.0
    trade = pf.trades[-1]
    assert trade.realized_pnl == pytest.approx((10_989.0 - 10.60439) - 100.151001 * 100)  # 963.29551
    assert pf.round_trips and pf.round_trips[0].pnl == pytest.approx(963.29551)

    # full ledger identity after the round trip:
    # 100_963.29551 cash + 0 inventory + 15.70449 fees == 100_000 + 979 realized_gross
    assert pf._ledger_value() == pytest.approx(100_000.0 + 979.0)

    # mark the book once so the equity identity has a mark to compare to
    pf.mark_to_market(MarketEvent(pd.Timestamp("2024-01-03"), "close", {"AAA": 110.0}))

    integrity = pf.integrity()
    assert integrity["all_checks_passed"] is True
    assert integrity["fills_audited"] == 2
    assert integrity["unmarked_open_positions"] == 0
    rec = integrity["reconciliation"]
    assert rec["final_equity"] == pytest.approx(100_963.29551)
    assert rec["total_pnl"] == pytest.approx(963.29551)
    assert rec["total_pnl"] == pytest.approx(rec["realized_pnl_net"] + rec["unrealized_pnl_net"])
    assert rec["total_pnl"] == pytest.approx(
        rec["realized_pnl_gross"] + rec["unrealized_pnl_gross"] - rec["total_costs"])
    assert rec["total_costs"] == pytest.approx(15.70449)


def test_missing_money_trips_the_per_fill_audit():
    """Corrupt the books between fills; the next audit must raise."""
    pf = Portfolio(100_000.0, ["AAA"])
    pf.on_fill(_fill("2024-01-02", "AAA", "BUY", 100, 100.0, 5.0))

    pf.cash += 1_000.0  # a yuan leak: fees forgotten, a data bug, ...
    with pytest.raises(AssertionError, match="ledger conservation"):
        pf.on_fill(_fill("2024-01-03", "AAA", "SELL", 100, 110.0, 15.989))

    # control: the identical sequence without tampering sails through
    clean = Portfolio(100_000.0, ["AAA"])
    clean.on_fill(_fill("2024-01-02", "AAA", "BUY", 100, 100.0, 5.0))
    clean.on_fill(_fill("2024-01-03", "AAA", "SELL", 100, 110.0, 5.0, 10.989))
    clean.mark_to_market(MarketEvent(pd.Timestamp("2024-01-03"), "close", {"AAA": 110.0}))
    assert clean.integrity()["all_checks_passed"] is True


# ------------------------------------------------- engine-level integration
@pytest.fixture(scope="module")
def round_trip_run():
    closes = [100.0 + i * 0.5 for i in range(40)]          # gently rising
    panel = make_panel({"TEST": closes})
    exit_from = panel["TEST"].index[30]
    engine = Engine(panel, LongThenFlat(exit_from), CostModel(),
                    RiskManager(RiskConfig()), capital=100_000.0)
    return engine.run(start=None, end=None)


def test_engine_round_trip_reconciles(round_trip_run):
    res = round_trip_run
    assert len(res.trades) == 2                      # one buy, one sell
    assert [t.side for t in res.trades] == ["BUY", "SELL"]
    assert len(res.round_trips) == 1

    integrity = res.integrity
    assert integrity["all_checks_passed"] is True
    assert integrity["fills_audited"] == len(res.trades)
    assert integrity["max_ledger_drift"] <= integrity["audit_tolerance"]
    assert integrity["unmarked_open_positions"] == 0

    rec = integrity["reconciliation"]
    # equity is pure cash once the position is closed
    assert rec["final_market_value"] == pytest.approx(0.0)
    assert rec["final_equity"] == pytest.approx(rec["final_cash"])
    # the round trip is the whole story
    assert rec["total_pnl"] == pytest.approx(res.round_trips[0].pnl)
    assert rec["total_pnl"] == pytest.approx(res.equity.iloc[-1] - 100_000.0)
    # both PnL decompositions hold to the cent
    assert rec["total_pnl"] == pytest.approx(
        rec["realized_pnl_net"] + rec["unrealized_pnl_net"], abs=1e-9)
    assert rec["total_pnl"] == pytest.approx(
        rec["realized_pnl_gross"] + rec["unrealized_pnl_gross"] - rec["total_costs"],
        abs=1e-9)
    # cash-flow identity
    assert rec["final_cash"] == pytest.approx(
        rec["initial_capital"] - rec["total_costs"]
        + sum(t.turnover for t in res.trades if t.side == "SELL")
        - sum(t.turnover for t in res.trades if t.side == "BUY"), abs=1e-9)


def test_open_position_marking_reconciles():
    """End the run holding inventory: market value and unrealized PnL."""
    closes = [100.0 + i * 0.5 for i in range(20)]
    panel = make_panel({"TEST": closes})
    engine = Engine(panel, AlwaysLong(), CostModel(), RiskManager(RiskConfig()),
                    capital=100_000.0)
    res = engine.run(start=None, end=None)

    assert len(res.trades) == 1 and res.trades[0].side == "BUY"
    integrity = res.integrity
    assert integrity["all_checks_passed"] is True

    rec = integrity["reconciliation"]
    assert rec["final_market_value"] > 0
    assert rec["final_equity"] == pytest.approx(rec["final_cash"] + rec["final_market_value"])
    # unrealized net marks against the all-in basis (price + buy fee)
    assert rec["unrealized_pnl_net"] == pytest.approx(
        rec["final_market_value"] - res.trades[0].turnover - res.trades[0].fee)
    # net PnL decomposes into realized (none) + unrealized
    assert rec["total_pnl"] == pytest.approx(
        rec["realized_pnl_net"] + rec["unrealized_pnl_net"], abs=1e-9)
    # ... and into gross gains minus costs paid
    assert rec["total_pnl"] == pytest.approx(
        rec["realized_pnl_gross"] + rec["unrealized_pnl_gross"] - rec["total_costs"],
        abs=1e-9)
    # equity equals the marked portfolio, to the last float
    assert float(res.equity.iloc[-1]) == pytest.approx(rec["final_equity"], abs=1e-9)
