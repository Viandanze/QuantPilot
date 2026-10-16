from __future__ import annotations
"""Tests for the v2 strategies: Bollinger mean reversion and ETF rotation."""
import pytest

from strategies import STRATEGIES, create
from strategies.etf_rotation import EtfRotation
from strategies.mean_reversion import MeanReversion

from tests.conftest import make_panel


def _trend(start: float, rate: float, n: int) -> list[float]:
    return [start * ((1.0 + rate) ** i) for i in range(n)]


# 20 flat sessions build the window; the dip at index 20 is a fresh lower-band
# break (close 50 vs lower band ~75.1); 100 afterwards is back above the mean.
DIP_SERIES = [100.0] * 20 + [50.0] + [100.0] * 5


# ----------------------------------------------------------- mean reversion
def test_mean_reversion_buys_first_band_break_and_exits_at_mean():
    panel = make_panel({"AAA": DIP_SERIES})
    strategy = MeanReversion(panel, window=20, num_std=2.0, max_weight=0.25)
    dates = panel["AAA"].index

    before = strategy.target_weights(dates[19], 1_000.0)
    assert before == {"AAA": 0.0}                      # flat history: no break yet

    at_dip = strategy.target_weights(dates[20], 1_000.0)
    assert at_dip == {"AAA": 0.25}                     # first close below the band

    back = strategy.target_weights(dates[21], 1_000.0)
    assert back == {"AAA": 0.0}                        # close back at/above the mean


def test_mean_reversion_holds_while_price_stays_below_the_band():
    series = [100.0] * 20 + [50.0] * 3 + [100.0] * 2
    panel = make_panel({"AAA": series})
    strategy = MeanReversion(panel, window=20, num_std=2.0, max_weight=0.25)
    dates = panel["AAA"].index

    assert strategy.target_weights(dates[20], 1_000.0) == {"AAA": 0.25}
    # staying under the band is a slide, not a fresh break: the book is held
    assert strategy.target_weights(dates[21], 1_000.0) == {"AAA": 0.25}
    assert strategy.target_weights(dates[22], 1_000.0) == {"AAA": 0.25}
    assert strategy.target_weights(dates[23], 1_000.0) == {"AAA": 0.0}  # recovered


def test_mean_reversion_compresses_when_many_symbols_signal_together():
    panel = make_panel({s: DIP_SERIES for s in ["AAA", "BBB", "CCC", "DDD", "EEE"]})
    strategy = MeanReversion(panel, window=20, num_std=2.0, max_weight=0.25)
    dates = panel["AAA"].index

    weights = strategy.target_weights(dates[20], 1_000.0)   # 5 x 0.25 -> 1.25 gross
    assert all(w == pytest.approx(0.2) for w in weights.values())
    assert sum(weights.values()) <= 1.0 + 1e-9

    pair = MeanReversion(
        make_panel({"AAA": DIP_SERIES, "BBB": DIP_SERIES}),
        window=20, num_std=2.0, max_weight=0.25,
    )
    two = pair.target_weights(dates[20], 1_000.0)
    assert two["AAA"] == pytest.approx(0.25)           # 0.5 gross: no compression
    assert two["BBB"] == pytest.approx(0.25)


def test_mean_reversion_warms_up_with_short_history():
    panel = make_panel({"AAA": [100.0] * 15})
    strategy = MeanReversion(panel, window=20, num_std=2.0)
    weights = strategy.target_weights(panel["AAA"].index[-1], 1_000.0)
    assert weights == {"AAA": 0.0}


def test_mean_reversion_decision_is_invariant_to_future_data():
    recovered = DIP_SERIES
    crashed = [100.0] * 20 + [50.0] + [10.0] * 5        # diverges right after t
    dates = make_panel({"AAA": recovered})["AAA"].index

    for probe in (19, 20):                              # last flat day, dip day
        t = dates[probe]
        full = MeanReversion(make_panel({"AAA": recovered}), window=20)
        trim = MeanReversion(make_panel({"AAA": crashed}), window=20)
        assert full.target_weights(t, 1_000.0) == trim.target_weights(t, 1_000.0)

    dip = dates[20]
    full = MeanReversion(make_panel({"AAA": recovered}), window=20)
    trim = MeanReversion(make_panel({"AAA": crashed}), window=20)
    assert full.target_weights(dip, 1_000.0) == {"AAA": 0.25}
    assert trim.target_weights(dip, 1_000.0) == {"AAA": 0.25}


def test_mean_reversion_rejects_bad_params():
    panel = make_panel({"AAA": [100.0] * 30})
    with pytest.raises(ValueError):
        MeanReversion(panel, window=1)
    with pytest.raises(ValueError):
        MeanReversion(panel, num_std=0.0)
    with pytest.raises(ValueError):
        MeanReversion(panel, max_weight=0.0)


# ------------------------------------------------------------- etf rotation
def test_etf_rotation_picks_the_strongest_by_relative_momentum():
    panel = make_panel({
        "AAA": _trend(100.0, 0.01, 30),    # strongest positive momentum
        "BBB": _trend(100.0, 0.002, 30),   # middle
        "CCC": _trend(100.0, -0.01, 30),   # falling
    })
    dates = panel["AAA"].index

    top2 = EtfRotation(panel, lookback=5, top_n=2)
    assert top2.target_weights(dates[10], 1_000.0) == {
        "AAA": pytest.approx(0.5), "BBB": pytest.approx(0.5), "CCC": 0.0,
    }

    top1 = EtfRotation(panel, lookback=5, top_n=1)
    assert top1.target_weights(dates[10], 1_000.0) == {
        "AAA": 1.0, "BBB": 0.0, "CCC": 0.0,
    }


def test_etf_rotation_absolute_filter_goes_defensive():
    falling = make_panel({
        "AAA": _trend(100.0, -0.01, 30),
        "BBB": _trend(100.0, -0.005, 30),
        "CCC": _trend(100.0, -0.02, 30),
    })
    defensive = EtfRotation(falling, lookback=5, top_n=2)
    weights = defensive.target_weights(falling["AAA"].index[10], 1_000.0)
    assert weights == {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0}   # all cash: bear mode

    # a flat series returns exactly 0.0, which is not strictly above 0.0
    mixed = make_panel({"AAA": _trend(100.0, 0.01, 30), "BBB": [100.0] * 30})
    single = EtfRotation(mixed, lookback=5, top_n=2)
    weights = single.target_weights(mixed["AAA"].index[10], 1_000.0)
    assert weights == {"AAA": pytest.approx(0.5), "BBB": 0.0}
    assert sum(weights.values()) < 1.0                  # leftover stays in cash


def test_etf_rotation_weight_sum_is_capped_at_one():
    panel = make_panel({
        "AAA": _trend(100.0, 0.01, 30),
        "BBB": _trend(100.0, 0.008, 30),
        "CCC": _trend(100.0, 0.005, 30),
    })
    dates = panel["AAA"].index
    weights = EtfRotation(panel, lookback=5, top_n=2).target_weights(dates[10], 1_000.0)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert sum(weights.values()) <= 1.0 + 1e-9


def test_etf_rotation_warms_up_with_short_history():
    panel = make_panel({"AAA": _trend(100.0, 0.01, 15)})
    strategy = EtfRotation(panel, lookback=20, top_n=2)
    weights = strategy.target_weights(panel["AAA"].index[-1], 1_000.0)
    assert weights == {"AAA": 0.0}


def test_etf_rotation_decision_is_invariant_to_future_data():
    base = {
        "AAA": _trend(100.0, 0.01, 30),
        "BBB": _trend(100.0, -0.01, 30),
    }
    flipped = {
        # everything after index 20 is reversed - AAA crashes, BBB rockets
        "AAA": _trend(100.0, 0.01, 21) + _trend(90.0, -0.05, 9),
        "BBB": _trend(100.0, -0.01, 21) + _trend(110.0, 0.05, 9),
    }
    dates = make_panel(base)["AAA"].index

    for probe in (19, 20):
        t = dates[probe]
        w_base = EtfRotation(make_panel(base), lookback=5, top_n=2)
        w_flip = EtfRotation(make_panel(flipped), lookback=5, top_n=2)
        assert w_base.target_weights(t, 1_000.0) == w_flip.target_weights(t, 1_000.0)

    t = dates[20]
    assert EtfRotation(make_panel(base), lookback=5, top_n=2).target_weights(
        t, 1_000.0
    ) == {"AAA": pytest.approx(0.5), "BBB": 0.0}


def test_etf_rotation_rejects_bad_params():
    panel = make_panel({"AAA": [100.0] * 30})
    with pytest.raises(ValueError):
        EtfRotation(panel, lookback=0)
    with pytest.raises(ValueError):
        EtfRotation(panel, top_n=0)


# -------------------------------------------------------------- registration
def test_v2_strategies_are_registered():
    assert STRATEGIES["mean_reversion"] is MeanReversion
    assert STRATEGIES["etf_rotation"] is EtfRotation
    panel = make_panel({"AAA": DIP_SERIES})
    assert isinstance(create("mean_reversion", panel), MeanReversion)
    assert isinstance(create("etf_rotation", panel), EtfRotation)
