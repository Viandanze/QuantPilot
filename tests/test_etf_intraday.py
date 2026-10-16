from __future__ import annotations
"""T+0 ETF universe, intraday momentum strategy and small-account guard tests.

All tests are offline: the HTTP layer of ``data.tx_http.fetch_minute`` is
stubbed through ``urllib.request.urlopen``, and every strategy test runs on
synthetic 60-minute panels.
"""
import io
import json
import math

import numpy as np
import pandas as pd
import pytest

from data.etf import T0_ETFS, etf_info, is_t0
from data.tx_http import fetch_minute
from risk.small_account import SmallAccountConfig, SmallAccountGuard
from strategies.intraday_momentum import IntradayMomentum


# --------------------------------------------------------------------- etf
def test_universe_has_four_t0_entries():
    assert set(T0_ETFS) == {"518880", "513100", "513500", "511260"}
    for info in T0_ETFS.values():
        assert info["t0"] is True
        assert info["min_lot"] == 100
        assert info["mic"] == "XSHG"


def test_is_t0_positive_and_negative():
    assert is_t0("518880") is True
    assert is_t0("600519") is False  # stock, not in the universe
    assert is_t0("999999") is False


def test_etf_info_returns_entry_and_raises_on_unknown():
    info = etf_info("513100")
    assert info["name"] == "Guotai Nasdaq 100 ETF"
    assert info["category"] == "cross_border"
    with pytest.raises(KeyError):
        etf_info("510300")


# ------------------------------------------------------------- fetch_minute
class _FakeResponse:
    def __init__(self, payload: dict):
        self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._buf.read()


def _mkline_payload(code: str, rows: list) -> dict:
    return {"code": 0, "data": {code: {code.split(".")[0] if "." in code else code: rows}}}


def test_fetch_minute_rejects_unsupported_period():
    with pytest.raises(ValueError):
        fetch_minute("518880", period="m30")


def test_fetch_minute_parses_and_dedups(monkeypatch):
    code = "sh518880"
    rows = [
        ["202609181000", 7.10, 7.20, 7.25, 7.05, 100000],
        ["202609181100", 7.20, 7.30, 7.35, 7.15, 120000],
        ["202609181100", 7.20, 7.30, 7.35, 7.15, 120000],  # duplicate
    ]
    payload = {"code": 0, "data": {code: {"m60": rows}}}

    def fake_urlopen(req, timeout=15):
        assert "kline/mkline" in req.full_url
        assert "m60" in req.full_url
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    df = fetch_minute("518880")
    assert len(df) == 2
    assert list(df.columns) == ["open", "close", "high", "low", "volume"]
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert df["close"].iloc[-1] == pytest.approx(7.30)


def test_fetch_minute_raises_on_empty(monkeypatch):
    code = "sh513100"
    payload = {"code": 0, "data": {code: {"m60": []}}}

    def fake_urlopen(req, timeout=15):
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ValueError):
        fetch_minute("513100")


# --------------------------------------------------------- intraday momentum
def _mk_panel(symbols: list[str], n: int = 40, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Synthetic 60-minute panel: hourly bars, mild random walk per symbol."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-09-10 10:00", periods=n, freq="h")
    panel = {}
    for i, s in enumerate(symbols):
        drift = 0.001 * (i + 1)
        closes = 10.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.002, n))
        highs = closes * 1.005
        lows = closes * 0.995
        opens = closes * (1.0 + rng.normal(0, 0.001, n))
        panel[s] = pd.DataFrame(
            {"open": opens, "close": closes, "high": highs, "low": lows, "volume": 1e5},
            index=idx,
        )
    return panel


def test_ctor_rejects_bad_params():
    panel = _mk_panel(["518880"])
    with pytest.raises(ValueError):
        IntradayMomentum(panel, lookback=0)
    with pytest.raises(ValueError):
        IntradayMomentum(panel, top_n=0)
    with pytest.raises(ValueError):
        IntradayMomentum(panel, atr_period=0)
    with pytest.raises(ValueError):
        IntradayMomentum(panel, atr_stop=0.0)
    with pytest.raises(ValueError):
        IntradayMomentum({})


def test_causality_future_bars_do_not_change_past_decision():
    panel = _mk_panel(["518880", "513100"], seed=3)
    strat = IntradayMomentum(panel, lookback=12, top_n=1)
    t = panel["518880"].index[30]
    before = strat.target_weights(t, 10_000.0)

    # corrupt every bar strictly after t; the t decision must not move
    for df in panel.values():
        df.iloc[df.index > t, df.columns.get_indexer(["close"])] = 99.0
    after = strat.target_weights(t, 10_000.0)
    assert before == after


def test_momentum_ranks_picks_top_n():
    panel = _mk_panel(["518880", "513100", "513500"], n=40, seed=5)
    strat = IntradayMomentum(panel, lookback=12, top_n=2)
    t = panel["518880"].index[-1]
    weights = strat.target_weights(t, 10_000.0)
    nonzero = {s: w for s, w in weights.items() if w > 0}
    assert len(nonzero) <= 2
    assert sum(weights.values()) <= 1.0 + 1e-12
    # every pick must carry the equal 1/top_n slice
    for w in nonzero.values():
        assert w == pytest.approx(1.0 / 2)


def test_nonpositive_momentum_goes_flat():
    # panel engineered so every symbol falls over the lookback window
    idx = pd.date_range("2026-09-10 10:00", periods=40, freq="h")
    rng = np.random.default_rng(9)
    panel = {}
    for s in ("518880", "513100"):
        closes = 10.0 * np.cumprod(1.0 - 0.002 + rng.normal(0, 0.001, 40))
        panel[s] = pd.DataFrame(
            {"open": closes, "close": closes, "high": closes * 1.001,
             "low": closes * 0.999, "volume": 1e5},
            index=idx,
        )
    strat = IntradayMomentum(panel, lookback=12, top_n=2)
    t = idx[-1]
    weights = strat.target_weights(t, 10_000.0)
    assert all(w == 0.0 for w in weights.values())


def test_atr_stop_flattens_crashed_symbol():
    # two rising symbols; after entry, one crashes far below the ATR stop
    idx = pd.date_range("2026-09-10 10:00", periods=40, freq="h")
    rng = np.random.default_rng(11)

    def _mk(closes):
        return pd.DataFrame(
            {"open": closes, "close": closes, "high": closes * 1.002,
             "low": closes * 0.998, "volume": 1e5},
            index=idx,
        )

    steady = 10.0 * np.cumprod(1.0 + 0.001 + rng.normal(0, 0.0005, 40))
    boom = 10.0 * np.cumprod(1.0 + 0.001 + rng.normal(0, 0.0005, 40))
    panel = {"518880": _mk(steady), "513100": _mk(boom)}
    strat = IntradayMomentum(panel, lookback=10, top_n=2, atr_period=10, atr_stop=1.0)

    t_entry = idx[25]
    w_entry = strat.target_weights(t_entry, 10_000.0)
    assert all(w > 0 for w in w_entry.values())  # both rising -> both held

    # crash 513100 by 10% over the next bars (way past 1xATR)
    for k in range(26, 40):
        boom[k] = boom[k - 1] * 0.985
    panel["513100"] = _mk(boom)
    strat2 = IntradayMomentum(panel, lookback=10, top_n=2, atr_period=10, atr_stop=1.0)
    strat2.target_weights(idx[25], 10_000.0)  # establish entry refs
    w_after = strat2.target_weights(idx[-1], 10_000.0)
    # the crashed symbol must be flat, the steady one may stay held
    assert w_after["513100"] == 0.0


def test_note_handles_unknown_symbol_and_warmup():
    panel = _mk_panel(["518880"], n=40, seed=13)
    strat = IntradayMomentum(panel, lookback=12, top_n=1)
    assert strat.note(pd.Timestamp("2026-09-10 10:00"), "599999") == "no data"
    idx = panel["518880"].index
    assert strat.note(idx[0], "518880") == "warming up"
    note = strat.note(idx[-1], "518880")
    assert "mom" in note


# --------------------------------------------------------- small account
def test_cap_order_clamps_to_max_weight():
    guard = SmallAccountGuard()
    out = guard.cap_order({"518880": 0.8, "513100": 0.02})
    assert out["518880"] == pytest.approx(0.50)  # clamped to default cap (2026-09-21 calibration)
    assert out["513100"] == pytest.approx(0.02)


def test_default_cap_keeps_two_name_half_book_tradable():
    # 2000 CNY book: one ETF lot is ~11-13% of equity, so a 50/50 two-name
    # target must survive the default 50% (half-book) cap untouched.
    guard = SmallAccountGuard()
    out = guard.cap_order({"513100": 0.5, "513500": 0.5})
    assert out["513100"] == pytest.approx(0.5)
    assert out["513500"] == pytest.approx(0.5)


def test_cap_order_floors_negatives():
    guard = SmallAccountGuard()
    out = guard.cap_order({"518880": -0.3})
    assert out["518880"] == 0.0


def test_cap_order_scales_when_many_names_at_cap():
    guard = SmallAccountGuard(SmallAccountConfig(max_order_weight=0.4))
    out = guard.cap_order({"A": 0.4, "B": 0.4, "C": 0.4})
    total = sum(out.values())
    assert total == pytest.approx(1.0)
    for w in out.values():
        assert w == pytest.approx(1.0 / 3.0)


def test_breaker_trips_at_limit():
    guard = SmallAccountGuard(SmallAccountConfig(daily_loss_limit=-0.03))
    assert guard.on_equity_update(9_699.0, 10_000.0) is True   # -3.01%
    assert guard.on_equity_update(9_700.0, 10_000.0) is True   # exactly -3%
    assert guard.on_equity_update(9_700.1, 10_000.0) is False  # -2.999%
    assert guard.on_equity_update(10_100.0, 10_000.0) is False  # profitable day


def test_breaker_ignores_invalid_baseline():
    guard = SmallAccountGuard()
    assert guard.on_equity_update(1.0, 0.0) is False
    assert guard.on_equity_update(1.0, -100.0) is False
