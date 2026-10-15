"""Performance attribution: curve stats, round trips, and benchmark tilt.

Functional API over the pieces of a backtest result - an equity series, a
list of trade-like fills, an optional benchmark series - so nothing here
imports engine internals. Trade-like objects only need the attribute names
of ``engine.portfolio.Trade`` (date, symbol, side, shares, price, turnover,
fee, ...); the round-trip recognition mirrors the engine's own flat-to-flat
accounting semantics without importing its private helpers.

Conventions (all figures are net of fees: the engine charges fees before
equity is marked and before fills are recorded):

- ``periods_per_year`` defaults to 244, this project's A-share annualisation
  factor. Volatility-like figures scale by ``sqrt(periods_per_year)``;
  alpha scales by ``periods_per_year``.
- Risk-free rate is 0.
- ``years`` for CAGR is ``n_points / periods_per_year``, matching
  ``report.metrics``.
- Sharpe is the annualised mean daily return over annualised volatility
  (sample std, ddof=1). Zero volatility resolves to +/-inf by the sign of
  the mean (NaN when the mean is exactly 0) instead of raising.
- Sortino divides the annualised mean by the downside deviation: the
  full-sample RMS (ddof=0) of daily returns clipped at zero. A zero
  downside deviation with a positive mean resolves to +inf.
- Sharpe and Sortino need at least two return observations; fewer yield NaN.
- Drawdowns are negative fractions of the running peak. ``max_drawdown``
  is the most negative drawdown; ``max_drawdown_days`` is the longest run
  of consecutive sessions strictly below the running peak (same convention
  as ``report.metrics.max_dd_duration_sessions``).
- A round trip is one flat-to-flat position cycle per symbol: buys build an
  all-in basis (turnover + fees), each sell books
  ``(turnover - fee) - avg_cost * shares`` exactly like the engine, and the
  cycle closes when the position returns to zero. Partial sells are
  supported; a position still open after the last fill is not a round trip
  and is skipped. Realised PnL is recomputed from the fills; the recorded
  ``realized_pnl`` attribute is ignored so any fill stream works.
- ``entry_px`` is the first buy price of the cycle and ``exit_px`` the price
  of the flattening sell; multi-fill economics live in ``pnl`` and
  ``pnl_pct`` (net PnL over the all-in buy basis).
- Benchmark figures (beta, annualised alpha, excess return, tracking error)
  use only the dates present in both series. Anything that cannot be
  computed (too few common points, zero benchmark variance, zero base
  level) is reported as ``None`` rather than raised.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = [
    "STAT_KEYS",
    "TRIP_COLUMNS",
    "attribution_summary",
    "performance_stats",
    "round_trip_stats",
]

STAT_KEYS = (
    "total_return",
    "cagr",
    "annual_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "max_drawdown_days",
    "calmar",
)

TRIP_COLUMNS = (
    "symbol",
    "entry_date",
    "exit_date",
    "entry_px",
    "exit_px",
    "shares",
    "pnl",
    "pnl_pct",
    "holding_days",
    "cumulative_contribution",
)


# ------------------------------------------------------- curve-level stats
def performance_stats(equity: pd.Series, periods_per_year: int = 244) -> dict:
    """Curve-level statistics from an equity series.

    Fewer than two finite points yield an all-NaN dict instead of an
    exception. See the module docstring for every formula convention.
    """
    ppy = max(int(periods_per_year), 1)
    eq = equity if isinstance(equity, pd.Series) else pd.Series(equity)
    eq = eq.dropna().sort_index()
    n = len(eq)
    if n < 2:
        return {key: float("nan") for key in STAT_KEYS}

    values = eq.to_numpy(dtype=float)
    first, last = float(values[0]), float(values[-1])
    years = n / ppy

    total_return = last / first - 1.0 if first != 0.0 else float("nan")
    if first > 0.0 and last > 0.0:
        cagr = (last / first) ** (1.0 / years) - 1.0
    else:
        cagr = float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        rets = values[1:] / values[:-1] - 1.0
    rets = rets[np.isfinite(rets)]
    std = float(rets.std(ddof=1)) if rets.size >= 2 else float("nan")
    mean = float(rets.mean()) if rets.size else float("nan")

    annual_vol = std * math.sqrt(ppy) if math.isfinite(std) else float("nan")
    sharpe = _ratio_annualised(mean, std, ppy)
    sortino = _sortino(mean, rets, ppy)

    peak = np.maximum.accumulate(values)
    positive_peak = peak > 0.0
    if positive_peak.any():
        safe_peak = np.where(positive_peak, peak, 1.0)
        dd = np.where(positive_peak, values / safe_peak - 1.0, np.nan)
        max_drawdown = float(np.nanmin(dd))
    else:
        dd = np.full(n, np.nan)
        max_drawdown = float("nan")
    max_drawdown_days = _longest_below_peak(dd)

    if math.isfinite(cagr) and max_drawdown < 0.0:
        calmar = cagr / abs(max_drawdown)
    else:
        calmar = float("nan")

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "max_drawdown_days": max_drawdown_days,
        "calmar": calmar,
    }


def _ratio_annualised(mean: float, std: float, ppy: int) -> float:
    """Sharpe-style annualisation; zero volatility resolves to +/-inf/NaN."""
    if not (math.isfinite(mean) and math.isfinite(std)):
        return float("nan")
    if std == 0.0:
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return float("nan")
    return mean / std * math.sqrt(ppy)


def _sortino(mean: float, rets: np.ndarray, ppy: int) -> float:
    """Annualised mean over the full-sample RMS of returns clipped at zero."""
    if rets.size < 2 or not math.isfinite(mean):
        return float("nan")
    downside = np.minimum(rets, 0.0)
    ddev = float(np.sqrt(np.mean(downside**2)))
    if ddev == 0.0:
        return float("inf") if mean > 0.0 else float("nan")
    return mean / ddev * math.sqrt(ppy)


def _longest_below_peak(dd: np.ndarray) -> int:
    """Longest run of sessions strictly below the running peak."""
    longest = run = 0
    for value in dd:
        if value < -1e-12:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


# ---------------------------------------------------------- round trips
def round_trip_stats(trades: list, equity: pd.Series) -> pd.DataFrame:
    """Per-round-trip PnL table built from a chronological fill stream.

    Fills are stably sorted by date first (same-day order is preserved), so
    the caller may pass any fill order. One row is emitted each time a
    symbol's position returns to zero; rows appear in exit order. See the
    module docstring for the accounting and column conventions.
    """
    eq = equity if isinstance(equity, pd.Series) else pd.Series(equity)
    eq = eq.dropna()
    initial_capital = float(eq.iloc[0]) if len(eq) else float("nan")

    state: dict[str, dict] = {}
    rows: list[dict] = []
    ordered = sorted(trades, key=lambda t: t.date)

    for trade in ordered:
        side = str(trade.side).upper()
        if side not in ("BUY", "SELL"):
            continue
        book = state.setdefault(trade.symbol, _new_book())

        if side == "BUY":
            if book["shares"] == 0:
                book = state[trade.symbol] = {
                    "shares": 0,
                    "avg_cost": 0.0,
                    "entry_date": trade.date,
                    "entry_px": float(trade.price),
                    "realized": 0.0,
                    "basis": 0.0,
                    "bought": 0,
                }
            shares = int(trade.shares)
            cost = book["avg_cost"] * book["shares"] + float(trade.turnover) + float(trade.fee)
            book["shares"] += shares
            book["avg_cost"] = cost / book["shares"] if book["shares"] else book["avg_cost"]
            book["basis"] += float(trade.turnover) + float(trade.fee)
            book["bought"] += shares
            continue

        # SELL: engine arithmetic - net proceeds minus all-in basis of the
        # shares sold; the cycle closes when the position returns to zero.
        if book["shares"] <= 0:
            continue  # sell without inventory: nothing to attribute
        sold = min(int(trade.shares), book["shares"])
        realized = (float(trade.turnover) - float(trade.fee)) - book["avg_cost"] * sold
        book["realized"] += realized
        book["shares"] -= sold
        if book["shares"] == 0:
            pnl = book["realized"]
            basis = book["basis"]
            contrib = (
                pnl / initial_capital
                if math.isfinite(initial_capital) and initial_capital != 0.0
                else float("nan")
            )
            rows.append(
                {
                    "symbol": trade.symbol,
                    "entry_date": book["entry_date"],
                    "exit_date": trade.date,
                    "entry_px": book["entry_px"],
                    "exit_px": float(trade.price),
                    "shares": book["bought"],
                    "pnl": pnl,
                    "pnl_pct": pnl / basis if basis > 0 else float("nan"),
                    "holding_days": (trade.date - book["entry_date"]).days,
                    "cumulative_contribution": contrib,
                }
            )

    return pd.DataFrame(rows, columns=list(TRIP_COLUMNS))


def _new_book() -> dict:
    """Fresh per-symbol position book for round-trip recognition."""
    return {
        "shares": 0,
        "avg_cost": 0.0,
        "entry_date": None,
        "entry_px": float("nan"),
        "realized": 0.0,
        "basis": 0.0,
        "bought": 0,
    }


# ------------------------------------------------------------- summary
def attribution_summary(
    equity: pd.Series,
    trades: list,
    benchmark: pd.Series | None = None,
    periods_per_year: int = 244,
) -> dict:
    """One-call attribution bundle over curve, round trips and benchmark.

    Returns a dict with ``stats`` (see :func:`performance_stats`),
    ``top_winners`` / ``top_losers`` (up to five round-trip records sorted
    by net PnL), ``by_symbol`` (per-symbol PnL, win rate, average holding
    days) and the benchmark block (``beta``, ``alpha_annualized``,
    ``excess_return``, ``tracking_error``) - all ``None`` when no benchmark
    is given.
    """
    ppy = max(int(periods_per_year), 1)
    stats = performance_stats(equity, ppy)
    trips = round_trip_stats(trades, equity)

    if len(trips):
        records = trips.to_dict("records")
        by_pnl = sorted(records, key=lambda r: r["pnl"])
        top_winners = list(reversed(by_pnl[-5:]))
        top_losers = by_pnl[:5]
    else:
        top_winners, top_losers = [], []

    by_symbol: dict[str, dict] = {}
    for symbol, group in trips.groupby("symbol", sort=True):
        pnls = group["pnl"].astype(float)
        by_symbol[str(symbol)] = {
            "round_trips": int(len(group)),
            "pnl": float(pnls.sum()),
            "win_rate": float((pnls > 0).sum()) / len(group),
            "avg_holding_days": float(group["holding_days"].mean()),
        }

    out = {
        "stats": stats,
        "top_winners": top_winners,
        "top_losers": top_losers,
        "by_symbol": by_symbol,
        "beta": None,
        "alpha_annualized": None,
        "excess_return": None,
        "tracking_error": None,
    }
    if benchmark is not None:
        out.update(_benchmark_block(equity, benchmark, ppy))
    return out


def _benchmark_block(equity: pd.Series, benchmark: pd.Series, ppy: int) -> dict:
    """Beta / alpha / excess / tracking error on commonly dated points."""
    block = {
        "beta": None,
        "alpha_annualized": None,
        "excess_return": None,
        "tracking_error": None,
    }
    try:
        joined = pd.concat({"eq": equity, "bm": benchmark}, axis=1, join="inner").dropna()
    except Exception:  # incompatible indexes: no benchmark figures at all
        return block
    if len(joined) < 2:
        return block

    sv = joined["eq"].to_numpy(dtype=float)
    bv = joined["bm"].to_numpy(dtype=float)
    if sv[0] != 0.0 and bv[0] != 0.0:
        block["excess_return"] = (sv[-1] / sv[0] - 1.0) - (bv[-1] / bv[0] - 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        sr = sv[1:] / sv[:-1] - 1.0
        br = bv[1:] / bv[:-1] - 1.0
    keep = np.isfinite(sr) & np.isfinite(br)
    sr, br = sr[keep], br[keep]
    if sr.size < 2:
        return block

    var_b = float(np.var(br, ddof=1))
    if var_b > 0.0:
        cov = float(np.cov(sr, br, ddof=1)[0, 1])
        beta = cov / var_b
        block["beta"] = beta
        block["alpha_annualized"] = (float(np.mean(sr)) - beta * float(np.mean(br))) * ppy
    block["tracking_error"] = float(np.std(sr - br, ddof=1)) * math.sqrt(ppy)
    return block
