"""Regression tests for the native C++ backtest core (backtest_core module).

These tests are skipped when the extension has not been built (e.g. on CI
machines without pybind11/MinGW). When present they pin two guarantees:

1. Indicator parity: sma matches pandas rolling mean, roc/atr match
   hand-computed references.
2. Engine parity: the C++ dual-MA backtest reproduces the Python
   event-driven engine's trade stream and equity curve on a small
   deterministic panel (exact trade count, 1e-9 relative on scalars).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

bt = pytest.importorskip(
    "backtest_core",
    reason="backtest_core extension not built (see cpp/BACKTEST_MVP_REPORT.md)",
)

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

sys.path.insert(0, str(ROOT))

from backtest.costs import CostModel  # noqa: E402
from engine.runner import Engine  # noqa: E402
from risk.manager import RiskConfig, RiskManager  # noqa: E402
from strategies.dual_ma import DualMAStrategy  # noqa: E402


def _small_panel(n_bars: int = 300, n_symbols: int = 3, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_bars)
    panel: dict[str, pd.DataFrame] = {}
    for i in range(n_symbols):
        sym = f"S{i:02d}"
        log_ret = rng.normal(3.0e-4, 1.5e-2, n_bars)
        close = 50.0 * np.exp(np.cumsum(log_ret))
        open_ = np.empty(n_bars)
        open_[0] = close[0]
        open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 3.0e-3, n_bars - 1))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 3.0e-3, n_bars))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 3.0e-3, n_bars))
        volume = rng.integers(1_000_000, 5_000_000, n_bars).astype(float)
        panel[sym] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close,
             "volume": volume},
            index=dates,
        )
    return panel, dates


# ---------------------------------------------------------------- indicators
def test_sma_matches_pandas_rolling_mean():
    rng = np.random.default_rng(1)
    x = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 500))))
    for window in (5, 20):
        native = bt.sma(x.to_numpy(), window)
        ref = x.rolling(window).mean().to_numpy()
        np.testing.assert_allclose(native, ref, rtol=1e-12, atol=1e-12)
        assert np.isnan(native[: window - 1]).all()


def test_roc_matches_reference():
    x = np.array([100.0, 110.0, 121.0, 100.0])
    got = bt.roc(x, 2)
    assert np.isnan(got[0]) and np.isnan(got[1])
    np.testing.assert_allclose(got[2:], x[2:] / x[:-2] - 1.0, rtol=1e-12)


def test_true_range_and_atr_match_reference():
    bars = [
        (10.0, 12.0, 9.0, 11.0),
        (11.0, 13.0, 10.5, 12.5),
        (12.5, 12.6, 11.0, 11.2),
    ]
    opens = np.array([b[0] for b in bars])
    highs = np.array([b[1] for b in bars])
    lows = np.array([b[2] for b in bars])
    closes = np.array([b[3] for b in bars])
    tr = bt.true_range(opens, highs, lows, closes)
    np.testing.assert_allclose(tr, [3.0, 2.5, 1.6], rtol=1e-12)
    atr = bt.atr(opens, highs, lows, closes, 2)
    assert np.isnan(atr[0])
    np.testing.assert_allclose(atr[1:], [2.75, 2.05], rtol=1e-12)


# -------------------------------------------------------------- engine parity
def test_cpp_engine_matches_python_engine():
    panel, _ = _small_panel()
    strategy = DualMAStrategy(panel, 5, 20)
    py_res = Engine(panel, strategy, CostModel(), RiskManager(RiskConfig()),
                    capital=1_000_000.0).run()

    syms = sorted(panel)
    opens = np.stack([panel[s]["open"].to_numpy() for s in syms])
    highs = np.stack([panel[s]["high"].to_numpy() for s in syms])
    lows = np.stack([panel[s]["low"].to_numpy() for s in syms])
    closes = np.stack([panel[s]["close"].to_numpy() for s in syms])
    volumes = np.stack([panel[s]["volume"].to_numpy() for s in syms])
    ts = panel[syms[0]].index.asi8.astype(np.int64)
    params = bt.BacktestParams()
    cpp_res = bt.run_dual_ma_backtest(opens, highs, lows, closes, volumes,
                                      ts, params)

    assert len(py_res.trades) == cpp_res.trade_count > 0
    for pt, ct in zip(py_res.trades, cpp_res.trades):
        assert pt.symbol == syms[ct.symbol]
        assert pt.side == ct.side
        assert pt.shares == ct.shares
        assert pt.price == pytest.approx(ct.price, rel=1e-9)
        assert pt.fee == pytest.approx(ct.fee, rel=1e-9)
    np.testing.assert_allclose(py_res.equity.to_numpy(),
                               np.asarray(cpp_res.equity_curve),
                               rtol=1e-9, atol=1e-9)


def test_cpp_rejects_inverted_windows():
    panel, _ = _small_panel(n_bars=40, n_symbols=1)
    opens = np.stack([panel[s]["open"].to_numpy() for s in sorted(panel)])
    highs = opens * 1.01
    lows = opens * 0.99
    closes = np.stack([panel[s]["close"].to_numpy() for s in sorted(panel)])
    volumes = np.stack([panel[s]["volume"].to_numpy() for s in sorted(panel)])
    ts = panel[sorted(panel)[0]].index.asi8.astype(np.int64)
    params = bt.BacktestParams()
    params.fast, params.slow = 20, 5
    with pytest.raises(ValueError):
        bt.run_dual_ma_backtest(opens, highs, lows, closes, volumes, ts, params)
