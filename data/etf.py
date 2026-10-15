"""T+0 eligible ETF universe and lot rules for the small-account day book.

A-share ETFs are normally T+1, but commodity, cross-border and bond ETFs
settle T+0, so intraday round trips are only legal on that subset. This
module is the single source of truth for which symbols the intraday
strategy may trade and their lot size; entries are plain dicts so callers
can serialize or extend them without extra plumbing.
"""
from __future__ import annotations

T0_ETFS: dict[str, dict] = {
    "518880": {
        "name": "Huaxia Gold ETF",
        "mic": "XSHG",
        "t0": True,
        "min_lot": 100,
        "category": "commodity",
    },
    "513100": {
        "name": "Guotai Nasdaq 100 ETF",
        "mic": "XSHG",
        "t0": True,
        "min_lot": 100,
        "category": "cross_border",
    },
    "513500": {
        "name": "Bosera S&P 500 ETF",
        "mic": "XSHG",
        "t0": True,
        "min_lot": 100,
        "category": "cross_border",
    },
    "511260": {
        "name": "Guotai 10-Year Treasury Bond ETF",
        "mic": "XSHG",
        "t0": True,
        "min_lot": 100,
        "category": "bond",
    },
}


def is_t0(symbol: str) -> bool:
    """True when ``symbol`` is listed in the T+0 universe with t0 enabled."""
    info = T0_ETFS.get(symbol)
    return bool(info and info.get("t0"))


def etf_info(symbol: str) -> dict:
    """Universe entry for ``symbol``; raises KeyError for unknown symbols."""
    try:
        return T0_ETFS[symbol]
    except KeyError:
        raise KeyError(f"symbol {symbol!r} is not in the T+0 ETF universe") from None
