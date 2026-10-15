"""TradeCalendar tests (offline; network fetch validated separately)."""
import pandas as pd

from data.calendar import TradeCalendar


def _cal() -> TradeCalendar:
    days = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
            "2026-09-21", "2026-09-22"]
    return TradeCalendar(pd.to_datetime(days))


def test_is_trading_day():
    cal = _cal()
    assert cal.is_trading_day("2026-09-18")
    assert not cal.is_trading_day("2026-09-19")  # Saturday
    assert not cal.is_trading_day("2026-09-20")  # Sunday


def test_last_session_respects_asof():
    cal = _cal()
    assert cal.last_session("2026-09-18 08:00") == pd.Timestamp("2026-09-18")
    assert cal.last_session("2026-09-19") == pd.Timestamp("2026-09-18")
    assert cal.last_session("2026-01-01") is None


def test_last_completed_session_excludes_intraday_bar():
    cal = _cal()
    # intraday: today's bar is unfinished -> decision day is the prior session
    assert cal.last_completed_session("2026-09-18 08:00") == pd.Timestamp("2026-09-17")
    assert cal.last_completed_session("2026-09-18 14:59") == pd.Timestamp("2026-09-17")
    # after the 15:00 close, today's session is complete
    assert cal.last_completed_session("2026-09-18 15:00") == pd.Timestamp("2026-09-18")
    assert cal.last_completed_session("2026-09-19 10:00") == pd.Timestamp("2026-09-18")


def test_sessions_between_is_inclusive():
    cal = _cal()
    got = cal.sessions_between("2026-09-15", "2026-09-21")
    assert list(got.strftime("%Y-%m-%d")) == [
        "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18", "2026-09-21"]


def test_duplicate_and_unsorted_sessions_normalized():
    cal = TradeCalendar(pd.to_datetime(
        ["2026-09-17", "2026-09-14", "2026-09-17", "2026-09-15"]))
    assert list(cal.sessions.strftime("%Y-%m-%d")) == [
        "2026-09-14", "2026-09-15", "2026-09-17"]
