from __future__ import annotations
"""Multi-source daily failover and cross-validation tests (HTTP fully mocked).

Every network touchpoint is stubbed through ``urllib.request.urlopen`` with
the same ``_FakeResponse`` pattern used by ``tests/test_etf_intraday.py``;
no test reaches the real tencent or sina feeds.
"""
import io
import json

import pandas as pd
import pytest
import urllib.error

from data.multi_source import cross_validate, fetch_daily_any, fetch_daily_backup


class _FakeResponse:
    """Context-manager byte response standing in for urlopen results."""

    def __init__(self, body: bytes):
        self._buf = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._buf.read()


def _sina_rows() -> list[dict]:
    """Raw sina-style payload: unsorted, with one duplicated date."""
    return [
        {"day": "2026-09-15", "open": "10.10", "high": "10.60", "low": "10.00",
         "close": "10.50", "volume": "200000"},
        {"day": "2026-09-16", "open": "10.50", "high": "10.90", "low": "10.40",
         "close": "10.80", "volume": "210000"},
        {"day": "2026-09-16", "open": "10.50", "high": "10.90", "low": "10.40",
         "close": "10.80", "volume": "210000"},  # duplicate date
        {"day": "2026-09-14", "open": "9.90", "high": "10.20", "low": "9.80",
         "close": "10.00", "volume": "180000"},  # out of order
    ]


def _tx_payload() -> dict:
    """Raw tencent qfq-day payload for the primary-feed happy path."""
    code = "sh600519"
    rows = [
        ["2026-09-14", "100.00", "101.00", "102.00", "99.00", "100000"],
        ["2026-09-15", "101.00", "102.00", "103.00", "100.00", "110000"],
        ["2026-09-16", "102.00", "103.00", "104.00", "101.00", "120000"],
    ]
    return {"code": 0, "data": {code: {"qfqday": rows}}}


# ------------------------------------------------------------ fetch_daily_backup
def test_fetch_daily_backup_parses_dedups_and_sorts(monkeypatch):
    def fake_urlopen(req, timeout=15):
        assert "CN_MarketDataService.getKLineData" in req.full_url
        assert "symbol=sh600519" in req.full_url
        return _FakeResponse(json.dumps(_sina_rows()).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    df = fetch_daily_backup("600519", "2026-09-14", "2026-09-16")
    assert len(df) == 3
    assert list(df.columns) == ["open", "close", "high", "low", "volume"]
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df["close"].iloc[0] == pytest.approx(10.00)
    assert df["close"].iloc[-1] == pytest.approx(10.80)


def test_fetch_daily_backup_clips_to_requested_window(monkeypatch):
    def fake_urlopen(req, timeout=15):
        return _FakeResponse(json.dumps(_sina_rows()).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    df = fetch_daily_backup("600519", "2026-09-15", "2026-09-15")
    assert list(df.index.strftime("%Y-%m-%d")) == ["2026-09-15"]


def test_fetch_daily_backup_empty_response_raises(monkeypatch):
    def fake_urlopen(req, timeout=15):
        return _FakeResponse(json.dumps([]).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ValueError):
        fetch_daily_backup("600519", "2026-01-01", "2026-09-18")


def test_fetch_daily_backup_window_miss_raises(monkeypatch):
    def fake_urlopen(req, timeout=15):
        return _FakeResponse(json.dumps(_sina_rows()).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ValueError):
        fetch_daily_backup("600519", "2020-01-01", "2020-12-31")


# -------------------------------------------------------------- fetch_daily_any
def test_fetch_daily_any_prefers_primary(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=15):
        calls.append(req.full_url)
        assert "ifzq.gtimg.cn" in req.full_url
        return _FakeResponse(json.dumps(_tx_payload()).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    df, used = fetch_daily_any("600519", "2026-09-14", "2026-09-16")
    assert used == "tx"
    assert len(calls) == 1  # backup feed never contacted
    assert list(df.columns) == ["open", "close", "high", "low", "volume"]
    assert df["close"].iloc[-1] == pytest.approx(103.00)


def test_fetch_daily_any_fails_over_on_primary_urlerror(monkeypatch):
    def fake_urlopen(req, timeout=15):
        if "ifzq.gtimg.cn" in req.full_url:
            raise urllib.error.URLError("tencent down")
        return _FakeResponse(json.dumps(_sina_rows()).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    df, used = fetch_daily_any("600519", "2026-09-14", "2026-09-16")
    assert used == "backup"
    assert df["close"].iloc[-1] == pytest.approx(10.80)


def test_fetch_daily_any_fails_over_on_primary_valueerror(monkeypatch):
    # an empty primary payload (ValueError) must trigger the failover too
    def fake_urlopen(req, timeout=15):
        if "ifzq.gtimg.cn" in req.full_url:
            payload = {"code": 0, "data": {"sh600519": {"qfqday": []}}}
            return _FakeResponse(json.dumps(payload).encode("utf-8"))
        return _FakeResponse(json.dumps(_sina_rows()).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    df, used = fetch_daily_any("600519", "2026-09-14", "2026-09-16")
    assert used == "backup"
    assert len(df) == 3


def test_fetch_daily_any_raises_runtime_error_when_both_fail(monkeypatch):
    def fake_urlopen(req, timeout=15):
        if "ifzq.gtimg.cn" in req.full_url:
            raise urllib.error.URLError("tencent down")
        raise ValueError("sina empty payload")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError) as excinfo:
        fetch_daily_any("600519", "2026-09-14", "2026-09-16")
    msg = str(excinfo.value)
    assert "tx" in msg and "URLError" in msg and "tencent down" in msg
    assert "backup" in msg and "sina empty payload" in msg


def test_fetch_daily_any_rejects_unknown_source():
    with pytest.raises(ValueError):
        fetch_daily_any("600519", "2026-09-14", "2026-09-16", source="netease")


# -------------------------------------------------------------- cross_validate
def _panel(closes: dict[str, float]) -> pd.DataFrame:
    """Tiny close-only panel keyed by YYYY-MM-DD date strings."""
    idx = pd.to_datetime(sorted(closes))
    return pd.DataFrame(
        {"open": 1.0, "close": [closes[d] for d in sorted(closes)],
         "high": 1.0, "low": 1.0, "volume": 1.0},
        index=idx,
    )


def test_cross_validate_flags_only_deviating_dates():
    a = _panel({"2026-09-14": 100.0, "2026-09-15": 101.0, "2026-09-16": 102.0})
    b = _panel({"2026-09-14": 100.5, "2026-09-15": 107.0, "2026-09-16": 102.1})
    flagged = cross_validate(a, b, tol=0.01)
    assert flagged == ["2026-09-15"]  # 101 vs 107 is ~5.6% off; others < 1%


def test_cross_validate_all_match_returns_empty():
    a = _panel({"2026-09-14": 100.0, "2026-09-15": 101.0})
    b = _panel({"2026-09-14": 100.001, "2026-09-15": 101.001})
    assert cross_validate(a, b, tol=0.01) == []


def test_cross_validate_ignores_non_overlapping_dates():
    a = _panel({"2026-09-14": 100.0})
    b = _panel({"2026-09-20": 200.0})
    assert cross_validate(a, b) == []
