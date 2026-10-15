"""A-share trading-day calendar derived from exchange kline dates.

Zero new dependencies: the set of trading days is read off a liquid,
essentially always-traded instrument (Shanghai composite index by default),
so holiday tables never go stale. Fetch failures propagate - callers own
the retry/abort policy.
"""
from __future__ import annotations

import pandas as pd

from data.tx_http import fetch_daily

DEFAULT_CALENDAR_SOURCE = "sh000001"  # SSE composite index


class TradeCalendar:
    def __init__(self, sessions: pd.DatetimeIndex):
        self.sessions = pd.DatetimeIndex(sorted(pd.to_datetime(sessions).unique()))

    @classmethod
    def fetch(cls, beg: str, end: str,
              symbol: str = DEFAULT_CALENDAR_SOURCE) -> "TradeCalendar":
        df = fetch_daily(symbol, beg, end, bars=400)
        return cls(df.index)

    # ------------------------------------------------------------- queries
    def is_trading_day(self, date) -> bool:
        ts = pd.Timestamp(date)
        pos = self.sessions.searchsorted(ts)
        return pos < len(self.sessions) and self.sessions[pos] == ts

    def last_session(self, asof) -> pd.Timestamp | None:
        """Latest session at or before ``asof`` (date-level granularity)."""
        ts = pd.Timestamp(asof)
        pos = self.sessions.searchsorted(ts, side="right") - 1
        return self.sessions[pos] if pos >= 0 else None

    def last_completed_session(self, asof) -> pd.Timestamp | None:
        """Latest session whose close is final as of ``asof``.

        Intraday kline feeds include today's unfinished bar, so a session
        only counts as completed once ``asof`` is past 15:00 on that date;
        otherwise the last session strictly before today's date is returned.
        """
        ts = pd.Timestamp(asof)
        if ts.hour >= 15:
            return self.last_session(ts)
        return self.last_session(ts - pd.Timedelta(days=1))

    def sessions_between(self, start, end) -> pd.DatetimeIndex:
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        lo_pos = self.sessions.searchsorted(lo, side="left")
        hi_pos = self.sessions.searchsorted(hi, side="right")
        return self.sessions[lo_pos:hi_pos]
