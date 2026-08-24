"""Cost model tests: rates, minimums, stamp tax asymmetry, slippage."""
import pytest

from backtest.costs import CostModel


def test_commission_rate():
    cost = CostModel()
    assert cost.commission(100_000.0) == pytest.approx(25.0)  # 0.025%


def test_commission_minimum():
    cost = CostModel()
    assert cost.commission(1_000.0) == 5.0  # below minimum -> flat 5 CNY


def test_zero_turnover_is_free():
    cost = CostModel()
    assert cost.commission(0.0) == 0.0
    assert cost.buy_fee(0.0) == 0.0
    assert cost.sell_fee(0.0) == 0.0


def test_buy_fee_has_no_stamp_tax():
    cost = CostModel()
    assert cost.buy_fee(100_000.0) == pytest.approx(25.0)


def test_sell_fee_includes_stamp_tax():
    cost = CostModel()
    # commission 25 + stamp 0.1% * 100000 = 100
    assert cost.sell_fee(100_000.0) == pytest.approx(125.0)


def test_slippage_is_adverse_on_both_sides():
    cost = CostModel()
    assert cost.buy_price(100.0) == pytest.approx(100.1)
    assert cost.sell_price(100.0) == pytest.approx(99.9)


def test_fees_are_configurable():
    cost = CostModel(commission_rate=0.0, commission_min=0.0, stamp_tax_rate=0.0)
    assert cost.sell_fee(1_000_000.0) == 0.0
