"""Tencent ifzq kline HTTP client: daily qfq and 60-minute bars.

Zero third-party dependencies: urllib + json only. Chosen after eastmoney
push2his proved unreachable from the user's network; values were
cross-validated against the paper-broker quote feed on 2026-09-18
(600519 prev_close 1266.98 matched exactly).
"""
from __future__ import annotations

import json

import urllib.request

import pandas as pd

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Referer": "https://gu.qq.com/",
    "Accept": "*/*",
}


def tx_code(symbol: str) -> str:
    """Six-digit A-share code -> tencent prefixed code (sh600519 / sz000858).

    Prefixed inputs (sh/sz/bj, e.g. index sh000001) pass through unchanged.
    """
    if symbol[:2].lower() in ("sh", "sz", "bj"):
        return symbol.lower()
    return ("sh" if symbol.startswith(("6", "5", "9")) else "sz") + symbol


def fetch_daily(symbol: str, beg: str, end: str, bars: int = 320) -> pd.DataFrame:
    """Forward-adjusted daily bars; columns date/open/close/high/low/volume.

    Raises on empty responses; callers own the row-count/staleness policy.
    """
    code = tx_code(symbol)
    url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={code},day,{beg},{end},{bars},qfq")
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    node = (payload.get("data") or {}).get(code) or {}
    rows = node.get("qfqday") or node.get("day") or []
    if not rows:
        raise ValueError(f"empty kline response for {symbol}")
    parsed = [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
              for r in rows]
    df = pd.DataFrame(parsed, columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna().drop_duplicates("date").set_index("date").sort_index()


def fetch_minute(symbol: str, period: str = "m60", bars: int = 320) -> pd.DataFrame:
    """Raw intraday minute bars from the tencent mkline endpoint (m60).

    Returns columns open/close/high/low/volume indexed by bar datetime,
    deduplicated and sorted ascending. ``period`` only accepts "m60";
    ``bars`` caps the requested window. Raises on empty responses,
    mirroring :func:`fetch_daily`; callers own the staleness policy.
    """
    if period != "m60":
        raise ValueError(f"unsupported minute period {period!r}; only 'm60' is available")
    code = tx_code(symbol)
    # NOTE: the web.ifzq host 302-redirects this path to a dead endpoint;
    # the bare ifzq.gtimg.cn host answers directly with the same JSON.
    url = ("https://ifzq.gtimg.cn/appstock/app/kline/mkline"
           f"?param={code},{period},,{bars}")
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    node = (payload.get("data") or {}).get(code) or {}
    rows = node.get(period) or []
    if not rows:
        raise ValueError(f"empty minute response for {symbol}")
    parsed = [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
              for r in rows]
    df = pd.DataFrame(parsed, columns=["datetime", "open", "close", "high", "low", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.dropna().drop_duplicates("datetime").set_index("datetime").sort_index()
