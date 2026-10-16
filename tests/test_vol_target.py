from __future__ import annotations
"""Vol-target sizer and ScaledStrategy wrapper tests (offline, synthetic)."""
import numpy as np
import pandas as pd
import pytest

from risk.vol_target import ScaledStrategy, VolTargetConfig, VolTargetSizer
from strategies.base import Strategy


def _returns(n: int, seed: int, daily_vol: float) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, daily_vol, n))


def test_insufficient_history_keeps_full_size():
    sizer = VolTargetSizer()
    assert sizer.scale(pd.Series([0.01, -0.01])) == 1.0


def test_zero_vol_caps_at_max_scale():
    sizer = VolTargetSizer()
    assert sizer.scale(pd.Series([0.0] * 25)) == 1.0


def test_high_vol_scales_down():
    sizer = VolTargetSizer(VolTargetConfig(target_vol=0.10))
    # 1% daily vol -> ~15.9% annualized -> scale ~0.63, robustly below 1
    scale = sizer.scale(_returns(60, seed=7, daily_vol=0.01))
    assert 0.3 < scale < 0.9


def test_low_vol_clips_at_max():
    sizer = VolTargetSizer(VolTargetConfig(target_vol=0.60))
    assert sizer.scale(_returns(60, seed=3, daily_vol=0.01)) == 1.0


def test_scale_monotonic_in_target_vol():
    low = VolTargetSizer(VolTargetConfig(target_vol=0.05))
    high = VolTargetSizer(VolTargetConfig(target_vol=0.10))
    series = _returns(60, seed=11, daily_vol=0.01)
    assert low.scale(series) < high.scale(series)


class _FixedStrategy(Strategy):
    name = "fixed"

    def target_weights(self, date, equity):
        return {"A": 0.5, "B": 0.5}


def _panel() -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2026-01-01", periods=40, freq="B")
    frame = pd.DataFrame({"close": np.linspace(100.0, 110.0, 40)}, index=idx)
    return {"A": frame.copy(), "B": frame.copy()}


def test_scaled_strategy_scales_weights_and_notes():
    wrapped = ScaledStrategy(_FixedStrategy(), _panel(),
                             VolTargetConfig(target_vol=0.0005))
    date = pd.Timestamp("2026-02-20")
    out = wrapped.target_weights(date, 1_000_000)
    assert out["A"] == pytest.approx(out["B"])
    assert 0.0 < out["A"] < 0.5
    assert "vol_scale" in wrapped.note(date, "A")


def test_scaled_strategy_inactive_without_history():
    wrapped = ScaledStrategy(_FixedStrategy(), _panel())
    out = wrapped.target_weights(pd.Timestamp("2026-01-05"), 1_000_000)
    assert out["A"] == pytest.approx(0.5)
    assert out["B"] == pytest.approx(0.5)


def test_scaled_strategy_empty_targets_passthrough():
    class _Empty(Strategy):
        name = "empty"

        def target_weights(self, date, equity):
            return {}

    wrapped = ScaledStrategy(_Empty(), _panel())
    assert wrapped.target_weights(pd.Timestamp("2026-02-20"), 1_000_000) == {}
