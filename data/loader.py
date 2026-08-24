"""Daily bar loader: akshare backend, on-disk cache, synthetic sample fallback.

The loader always normalizes bars to a DatetimeIndex with English columns
``open/high/low/close/volume``. Real data comes from akshare (free, no token,
forward-adjusted ``qfq``). If akshare is unavailable or a request fails, the
loader degrades to the built-in deterministic sample CSVs so demos and tests
keep working offline; ``DATA_SOURCE`` records which backend produced the last
load so reports can label it honestly.
"""
from __future__ import annotations

import logging
import pathlib
import time

import pandas as pd

from data import samples

log = logging.getLogger("quantpilot.data")

CACHE_DIR = pathlib.Path(__file__).resolve().parent / "cache"
BENCHMARK_SYMBOL = "000300"  # CSI 300 index
DEFAULT_SYMBOLS = ["600519", "000858", "601318", "000333", "600036", "510300"]

# Provenance of the most recent load: {"mode": "akshare"|"sample"|"mixed",
# "symbols": {sym: backend}}.
DATA_SOURCE: dict = {"mode": "unknown", "symbols": {}}

# akshare's eastmoney endpoints return Chinese column headers; keep the file
# pure ASCII by spelling them as unicode escapes (date/open/close/high/low/volume).
_CN_COLUMNS = {
    "\u65e5\u671f": "date",
    "\u5f00\u76d8": "open",
    "\u6536\u76d8": "close",
    "\u6700\u9ad8": "high",
    "\u6700\u4f4e": "low",
    "\u6210\u4ea4\u91cf": "volume",
}
_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize(raw: pd.DataFrame, english: bool = False) -> pd.DataFrame:
    df = raw.copy()
    if not english:
        df = df.rename(columns=_CN_COLUMNS)
    required = {"date", *_COLUMNS}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"unexpected columns {list(df.columns)}; missing {sorted(missing)}")
    df = df[["date", *_COLUMNS]]
    df["date"] = pd.to_datetime(df["date"])
    for col in _COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "close"]).sort_values("date")
    df = df.drop_duplicates(subset="date").set_index("date")
    return df


def _cache_path(name: str, start: pd.Timestamp, end: pd.Timestamp) -> pathlib.Path:
    return CACHE_DIR / f"{name}_{start:%Y%m%d}_{end:%Y%m%d}.csv"


def _from_cache(name: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    path = _cache_path(name, start, end)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def _to_cache(name: str, start: pd.Timestamp, end: pd.Timestamp, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_cache_path(name, start, end))


def _is_etf(symbol: str) -> bool:
    return symbol.startswith(("51", "15", "56", "58"))


def _sina_code(symbol: str) -> str:
    prefix = "sh" if symbol.startswith(("6", "5")) else "sz"
    return prefix + symbol


def _fetch_akshare(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Try eastmoney first, then the sina mirror; both are free and tokenless."""
    import akshare as ak  # imported lazily so offline environments stay usable

    kwargs = dict(
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )
    try:
        if _is_etf(symbol):
            return _normalize(ak.fund_etf_hist_em(symbol=symbol, **kwargs))
        return _normalize(ak.stock_zh_a_hist(symbol=symbol, **kwargs))
    except Exception as em_error:  # noqa: BLE001 - try the secondary source
        log.info("eastmoney source failed for %s (%s); trying sina", symbol, em_error)
    if _is_etf(symbol):
        raw = ak.fund_etf_hist_sina(symbol=_sina_code(symbol))
    else:
        raw = ak.stock_zh_a_daily(
            symbol=_sina_code(symbol),
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
    df = _normalize(raw, english=True)
    return df.loc[(df.index >= start) & (df.index <= end)]


def _fetch_index_akshare(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import akshare as ak

    try:
        raw = ak.index_zh_a_hist(
            symbol=BENCHMARK_SYMBOL,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        return _normalize(raw)
    except Exception:
        raw = ak.stock_zh_index_daily(symbol=f"sh{BENCHMARK_SYMBOL}")
        raw = raw.rename(columns={"date": "date"})
        return _normalize(raw, english=True)


def _sample_frame(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = samples.sample_path(symbol)
    if not path.exists():
        samples.generate_sample(symbol).to_csv(path)
    df = samples.load_sample(symbol)
    clipped = df.loc[(df.index >= start - pd.Timedelta(days=45)) & (df.index <= end)]
    if len(clipped) < 60:
        log.warning(
            "sample data for %s overlaps the requested window by only %d rows; "
            "using the full sample range instead",
            symbol,
            len(clipped),
        )
        return df
    return clipped


def load_daily(
    symbols: list[str],
    start,
    end,
    offline: bool = False,
    throttle: float = 0.6,
) -> dict[str, pd.DataFrame]:
    """Load daily bars for ``symbols`` between ``start`` and ``end``."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    frames: dict[str, pd.DataFrame] = {}
    backends: dict[str, str] = {}
    for symbol in symbols:
        df = None
        if not offline:
            cached = _from_cache(symbol, start, end)
            if cached is not None and len(cached) > 0:
                df = cached
                backends[symbol] = "akshare"
            else:
                try:
                    df = _fetch_akshare(symbol, start, end)
                    if df.empty:
                        raise ValueError("empty response")
                    _to_cache(symbol, start, end, df)
                    backends[symbol] = "akshare"
                    time.sleep(throttle)
                except Exception as exc:  # noqa: BLE001 - degrade, do not crash
                    log.warning("akshare unavailable for %s (%s); using sample data", symbol, exc)
        if df is None:
            if not samples.sample_path(symbol).exists() and symbol not in samples.SAMPLE_SPECS:
                raise ValueError(
                    f"cannot load {symbol}: akshare failed and no sample spec exists"
                )
            df = _sample_frame(symbol, start, end)
            backends[symbol] = "sample"
        frames[symbol] = df
    _refresh_mode()
    return frames


def _record_source(name: str, backend: str) -> None:
    DATA_SOURCE.setdefault("symbols", {})[name] = backend
    _refresh_mode()


def _refresh_mode() -> None:
    used = set(DATA_SOURCE.get("symbols", {}).values())
    DATA_SOURCE["mode"] = used.pop() if len(used) == 1 else ("mixed" if used else "unknown")


def load_benchmark(start, end, offline: bool = False) -> pd.Series:
    """Load the CSI 300 index close series; falls back to sample data."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cache_name = f"index_{BENCHMARK_SYMBOL}"
    df = None
    from_akshare = False
    if not offline:
        cached = _from_cache(cache_name, start, end)
        if cached is not None and len(cached) > 0:
            df = cached
            from_akshare = True
        else:
            try:
                df = _fetch_index_akshare(start, end)
                if df.empty:
                    raise ValueError("empty response")
                _to_cache(cache_name, start, end, df)
                from_akshare = True
            except Exception as exc:  # noqa: BLE001
                log.warning("akshare unavailable for benchmark (%s); using sample data", exc)
    if df is None:
        df = _sample_frame(BENCHMARK_SYMBOL, start, end)
    _record_source(BENCHMARK_SYMBOL, "akshare" if from_akshare else "sample")
    return df["close"].rename(BENCHMARK_SYMBOL)


def build_panel(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align per-symbol frames onto the union calendar, forward-filling gaps."""
    all_dates = sorted(set().union(*(set(f.index) for f in frames.values())))
    index = pd.DatetimeIndex(all_dates)
    panel = {s: f.reindex(index).ffill() for s, f in frames.items()}
    return panel
