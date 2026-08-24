"""Sample data generator tests: determinism, shape, committed CSVs."""
import pandas as pd
import pytest

from data import samples


def test_generator_is_deterministic():
    end = pd.Timestamp("2026-08-21")
    a = samples.generate_sample("510300", days=200, end=end)
    b = samples.generate_sample("510300", days=200, end=end)
    pd.testing.assert_frame_equal(a, b)


def test_generated_bars_are_sane():
    df = samples.generate_sample("600519", days=300, end=pd.Timestamp("2026-08-21"))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing
    assert df.index.name == "date"
    assert df[["open", "high", "low", "close"]].gt(0).all().all()
    body = df[["open", "close"]]
    assert (df["high"] >= body.max(axis=1) - 0.011).all()
    assert (df["low"] <= body.min(axis=1) + 0.011).all()
    assert df["close"].pct_change().abs().max() < 0.10  # clipped daily moves


def test_committed_sample_csvs_load():
    for symbol in ["600519", "000858", "601318", "000333", "600036", "510300", "000300"]:
        df = samples.load_sample(symbol)
        assert len(df) >= 800
        assert df["close"].notna().all()
        assert (df["close"] > 0).all()
