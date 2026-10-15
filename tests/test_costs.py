"""Cost model tests: rates, minimums, fee asymmetry, slippage."""
import pytest

from backtest.costs import CostModel


def test_commission_rate():
    cost = CostModel()
    assert cost.commission(100_000.0) == pytest.approx(30.0)  # 0.03%


def test_commission_minimum():
    cost = CostModel()
    assert cost.commission(1_000.0) == 5.0  # below minimum -> flat 5 CNY


def test_zero_turnover_is_free():
    cost = CostModel()
    assert cost.commission(0.0) == 0.0
    assert cost.buy_fee(0.0) == 0.0
    assert cost.sell_fee(0.0) == 0.0


def test_buy_fee_commission_plus_transfer():
    cost = CostModel()
    # commission 30 + transfer 0.001% * 100000 = 1
    assert cost.buy_fee(100_000.0) == pytest.approx(31.0)


def test_sell_fee_commission_stamp_transfer():
    cost = CostModel()
    # commission 30 + stamp 0.05% * 100000 = 50 + transfer 1
    assert cost.sell_fee(100_000.0) == pytest.approx(81.0)


def test_stamp_tax_sell_only_at_current_regime():
    cost = CostModel()
    # sell - buy = stamp tax only (transfer fee cancels out)
    assert cost.sell_fee(100_000.0) - cost.buy_fee(100_000.0) == pytest.approx(50.0)


def test_slippage_is_adverse_on_both_sides():
    cost = CostModel()
    assert cost.buy_price(100.0) == pytest.approx(100.1)
    assert cost.sell_price(100.0) == pytest.approx(99.9)


def test_fees_are_configurable():
    cost = CostModel(commission_rate=0.0, commission_min=0.0,
                     stamp_tax_rate=0.0, transfer_fee_rate=0.0)
    assert cost.sell_fee(1_000_000.0) == 0.0
    assert cost.buy_fee(1_000_000.0) == 0.0

# ---------------------------------------------------------- dynamic slippage
def test_dynamic_defaults_to_static_when_impact_disabled():
    cost = CostModel()
    assert cost.buy_price_dynamic(100.0, 1_000.0, 10_000.0) == \
        pytest.approx(cost.buy_price(100.0))
    assert cost.sell_price_dynamic(100.0, 1_000.0, 10_000.0) == \
        pytest.approx(cost.sell_price(100.0))


def test_dynamic_scales_with_participation():
    cost = CostModel(volume_impact_rate=0.002)
    # participation 0.1 -> dynamic slip = 0.001 + 0.002 * 0.1 = 0.0012
    assert cost.buy_price_dynamic(100.0, 100.0, 1_000.0) == pytest.approx(100.12)
    assert cost.sell_price_dynamic(100.0, 100.0, 1_000.0) == \
        pytest.approx(100.0 * (1.0 - 0.0012))


def test_dynamic_falls_back_on_nonpositive_participation():
    cost = CostModel(volume_impact_rate=0.002)
    assert cost.buy_price_dynamic(100.0, 0.0, 1_000.0) == \
        pytest.approx(cost.buy_price(100.0))
    assert cost.buy_price_dynamic(100.0, -100.0, 1_000.0) == \
        pytest.approx(cost.buy_price(100.0))
    assert cost.buy_price_dynamic(100.0, 100.0, 0.0) == \
        pytest.approx(cost.buy_price(100.0))
    assert cost.sell_price_dynamic(100.0, 100.0, -5.0) == \
        pytest.approx(cost.sell_price(100.0))


def test_dynamic_falls_back_on_nan_inputs():
    cost = CostModel(volume_impact_rate=0.002)
    assert cost.buy_price_dynamic(100.0, float("nan"), 1_000.0) == \
        pytest.approx(cost.buy_price(100.0))
    assert cost.buy_price_dynamic(100.0, 100.0, float("nan")) == \
        pytest.approx(cost.buy_price(100.0))
    assert cost.sell_price_dynamic(100.0, float("nan"), float("nan")) == \
        pytest.approx(cost.sell_price(100.0))
