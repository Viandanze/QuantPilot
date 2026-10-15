"""Strategy logic tests: crossovers, momentum ranking, grid mapping."""
import numpy as np
import pandas as pd
import pytest

from strategies.dual_ma import DualMAStrategy
from strategies.grid import GridStrategy
from strategies.momentum import MomentumStrategy

from tests.conftest import make_panel


# ------------------------------------------------------------------ dual MA
def test_dual_ma_long_only_in_uptrend():
    down = list(np.linspace(100.0, 70.0, 30))
    up = list(np.linspace(70.0, 100.0, 15))
    panel = make_panel({"AAA": down + up, "BBB": [50.0] * 45})
    strategy = DualMAStrategy(panel, fast=5, slow=10)
    dates = panel["AAA"].index

    late = strategy.target_weights(dates[-1], 1_000.0)
    assert late["AAA"] == pytest.approx(0.5)   # fast above slow -> equal slice
    assert late["BBB"] == 0.0                  # flat series: fast == slow -> no entry

    mid = strategy.target_weights(dates[20], 1_000.0)
    assert mid["AAA"] == 0.0                   # still in the downtrend leg


def test_dual_ma_rejects_bad_windows():
    panel = make_panel({"AAA": [10.0] * 30})
    with pytest.raises(ValueError):
        DualMAStrategy(panel, fast=20, slow=5)


# ----------------------------------------------------------------- momentum
def test_momentum_picks_top_ranked_and_holds_between_rebalances():
    n = 30
    rising = [100.0 * (1.01 ** i) for i in range(n)]
    falling = [80.0 * (0.99 ** i) for i in range(n)]
    panel = make_panel({"AAA": rising, "BBB": [50.0] * n, "CCC": falling})
    strategy = MomentumStrategy(panel, lookback=5, top_n=1, rebalance_every=5)
    dates = panel["AAA"].index

    rebalance_day = dates[10]  # schedule is index[5::5]
    weights = strategy.target_weights(rebalance_day, 1_000.0)
    assert weights == {"AAA": 1.0, "BBB": 0.0, "CCC": 0.0}

    held = strategy.target_weights(dates[11], 1_000.0)  # not a scheduled day
    assert held == weights

    assert "rank 1/3" in strategy.note(rebalance_day, "AAA")


# --------------------------------------------------------------------- grid
def test_grid_unit_mapping_is_monotone():
    strategy = GridStrategy(make_panel({"GRID": [10.0] * 30}), levels=8)
    assert strategy.units(9.0, 10.0, 20.0) == 8     # at/below the low -> max units
    assert strategy.units(10.0, 10.0, 20.0) == 8
    assert strategy.units(15.0, 10.0, 20.0) == 4    # middle of the range
    assert strategy.units(20.0, 10.0, 20.0) == 0    # at/above the high -> flat
    ladder = [strategy.units(p / 10.0, 10.0, 20.0) for p in range(100, 201, 5)]
    assert all(a >= b for a, b in zip(ladder, ladder[1:]))


def test_grid_weight_follows_price_band():
    prices = [15.0] * 125 + [14.0]
    panel = make_panel({"GRID": prices})
    strategy = GridStrategy(panel, levels=8, range_window=120, range_buffer=0.10)
    dates = panel["GRID"].index

    # 120 flat closes at 15 -> bounds (13.5, 16.5)
    w_mid = strategy.target_weights(dates[119], 1_000.0)
    assert w_mid["GRID"] == pytest.approx(0.5)  # ratio 0.5 -> 4/8 units

    w_low = strategy.target_weights(dates[125], 1_000.0)  # price 14
    assert w_low["GRID"] == pytest.approx(7.0 / 8.0)

    # bounds are frozen after the first decision date
    assert strategy.bounds["GRID"] == (pytest.approx(13.5), pytest.approx(16.5))


def test_grid_warms_up_with_short_history():
    panel = make_panel({"GRID": [10.0] * 10})
    strategy = GridStrategy(panel, levels=8)
    weights = strategy.target_weights(panel["GRID"].index[-1], 1_000.0)
    assert weights["GRID"] == 0.0
