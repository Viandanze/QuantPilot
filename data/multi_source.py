"""Multi-source daily kline fetching: tencent primary + sina backup failover.

Primary source is the tencent ifzq qfq feed (:func:`data.tx_http.fetch_daily`).
Backup source is the sina public kline endpoint (verified reachable on
2026-09-18 from the user's network; endpoint widely documented in community
API mirrors, sina publishes no official page):

    https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData
        ?symbol={sh|sz}{6-digit}&scale=240&ma=no&datalen={n}

It answers with a plain JSON array of {day, open, high, low, close, volume}
where ``scale=240`` aggregates 240-minute bars into one daily OHLCV row.
Known limitations versus the primary feed:
- it serves only the most recent ``datalen`` bars (cap 1023, ~4 years); a
  ``beg`` earlier than the covered window simply truncates to the first bar;
- prices are raw (unadjusted) while the primary feed is qfq, so pair the two
  with :func:`cross_validate` and a tolerance that absorbs ex-dividend gaps.

Zero third-party dependencies beyond pandas/numpy: urllib + json only.
"""
from __future__ import annotations

import json

import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from data.tx_http import fetch_daily, tx_code

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Referer": "https://finance.sina.com.cn/",
    "Accept": "*/*",
}


def fetch_daily_backup(symbol: str, beg: str, end: str,
                       datalen: int = 1023) -> pd.DataFrame:
    """Daily bars from the sina backup feed.

    Returns the same column contract as :func:`data.tx_http.fetch_daily`:
    a datetime-indexed frame (date index, ascending, unique) with float
    columns open/close/high/low/volume, clipped to ``[beg, end]``.

    Raises ValueError on an empty payload or when no row falls inside the
    requested window, mirroring the primary feed's empty-response policy.
    """
    code = tx_code(symbol)
    url = ("https://quotes.sina.cn/cn/api/json_v2.php/"
           f"CN_MarketDataService.getKLineData?symbol={code}&scale=240"
           f"&ma=no&datalen={datalen}")
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"empty backup kline response for {symbol}")
    parsed = [(item["day"], float(item["open"]), float(item["close"]),
               float(item["high"]), float(item["low"]), float(item["volume"]))
              for item in payload]
    df = pd.DataFrame(parsed, columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().drop_duplicates("date").set_index("date").sort_index()
    df = df.loc[(df.index >= pd.Timestamp(beg)) & (df.index <= pd.Timestamp(end))]
    if df.empty:
        raise ValueError(f"no backup kline rows for {symbol} in [{beg}, {end}]")
    return df


def fetch_daily_any(symbol: str, beg: str, end: str,
                    source: str = "auto") -> tuple[pd.DataFrame, str]:
    """Fetch daily bars with source failover; returns ``(df, used_source)``.

    ``source="auto"`` tries the tencent primary first and only falls back to
    the sina backup when the primary raises URLError/ValueError/TimeoutError.
    ``source="tx"`` / ``source="backup"`` pin one feed and propagate its
    exceptions unchanged. Under "auto", when both feeds fail a RuntimeError
    is raised summarising both errors.
    """
    if source not in ("auto", "tx", "backup"):
        raise ValueError(f"unknown source {source!r}; expected auto/tx/backup")
    if source == "tx":
        return fetch_daily(symbol, beg, end), "tx"
    if source == "backup":
        return fetch_daily_backup(symbol, beg, end), "backup"
    failures = []
    try:
        return fetch_daily(symbol, beg, end), "tx"
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        failures.append(f"tx: {type(exc).__name__}: {exc}")
    try:
        return fetch_daily_backup(symbol, beg, end), "backup"
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        failures.append(f"backup: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "all daily kline sources failed -> " + "; ".join(failures)
        ) from exc


def cross_validate(df_a: pd.DataFrame, df_b: pd.DataFrame,
                   tol: float = 0.01) -> list[str]:
    """Compare close prices on aligned dates; return deviating dates.

    A date is flagged (as a ``YYYY-MM-DD`` string) when the relative close
    difference between the two feeds exceeds ``tol`` (default 1%, computed
    against the larger absolute close). Purely advisory for feed-drift
    alerts -- callers own any blocking decision; non-overlapping dates are
    simply ignored.
    """
    common = df_a.index.intersection(df_b.index)
    if common.empty:
        return []
    a = df_a.loc[common, "close"].astype(float)
    b = df_b.loc[common, "close"].astype(float)
    rel = (a - b).abs() / np.maximum(a.abs(), b.abs())
    flagged = common[(rel > tol).to_numpy()]
    return [ts.strftime("%Y-%m-%d") for ts in flagged]
