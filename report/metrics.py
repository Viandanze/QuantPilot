"""Performance metrics computed from engine outputs.

Conventions:

- 252 trading days per year
- Sharpe ratio uses the arithmetic mean of daily equity returns, risk-free
  rate 0 (documented simplification for CNY cash)
- win rate is computed over flat-to-flat round trips per symbol (a position
  opened from zero and later closed back to zero)
- all figures are net of commissions, stamp tax and slippage because the
  engine already charges them before equity is marked
"""
from __future__ import annotations

import math
from collections import Counter

import pandas as pd

TRADING_DAYS = 252


def max_dd_duration_sessions(dd: pd.Series) -> int:
    """Longest consecutive run of sessions below a running peak."""
    longest = run = 0
    for value in dd:
        if value < -1e-12:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def _series_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) < 2:
        return {"total": float("nan"), "cagr": float("nan"), "max_drawdown": float("nan")}
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    years = len(s) / TRADING_DAYS
    total = last / first - 1.0
    cagr = (last / first) ** (1.0 / years) - 1.0 if years > 0 and first > 0 else float("nan")
    dd = s / s.cummax() - 1.0
    return {"total": total, "cagr": cagr, "max_drawdown": float(dd.min())}


def compute_metrics(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    trades=(),
    round_trips=(),
    exposure: pd.Series | None = None,
) -> dict:
    eq = equity.dropna()
    if len(eq) < 2:
        raise ValueError("equity series is too short to evaluate")

    stats = _series_stats(eq)
    ret = eq.pct_change().dropna()
    std = float(ret.std(ddof=1)) if len(ret) > 1 else float("nan")
    mean = float(ret.mean()) if len(ret) > 0 else float("nan")
    sharpe = mean / std * math.sqrt(TRADING_DAYS) if std and std > 0 else float("nan")
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = stats["cagr"] / abs(max_dd) if max_dd < 0 and math.isfinite(stats["cagr"]) else float("nan")

    trades = list(trades)
    trips = list(round_trips)
    wins = [t for t in trips if t.pnl > 0]
    losses = [t for t in trips if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = sum(t.pnl for t in losses)
    reasons = Counter(t.reason for t in trades)

    out = {
        "start": eq.index[0],
        "end": eq.index[-1],
        "sessions": len(eq),
        "years": len(eq) / TRADING_DAYS,
        "total_return": stats["total"],
        "cagr": stats["cagr"],
        "ann_vol": std * math.sqrt(TRADING_DAYS) if math.isfinite(std) else float("nan"),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_dd_sessions": max_dd_duration_sessions(dd),
        "calmar": calmar,
        "avg_exposure": float(exposure.mean()) if exposure is not None and len(exposure) else float("nan"),
        "n_trades": len(trades),
        "n_buys": sum(1 for t in trades if t.side == "BUY"),
        "n_sells": sum(1 for t in trades if t.side == "SELL"),
        "turnover_total": float(sum(t.turnover for t in trades)),
        "fees_total": float(sum(t.fee for t in trades)),
        "commission_total": float(sum(t.fee_commission for t in trades)),
        "stamp_total": float(sum(t.fee_stamp for t in trades)),
        "stop_loss_exits": int(reasons.get("stop_loss", 0)),
        "drawdown_flats": int(reasons.get("drawdown_flat", 0)),
        "round_trips": len(trips),
        "win_rate": (len(wins) / len(trips)) if trips else None,
        "profit_factor": (gross_win / abs(gross_loss)) if gross_loss < 0 else None,
        "avg_trip_pnl": (sum(t.pnl for t in trips) / len(trips)) if trips else None,
    }

    if benchmark is not None:
        b = benchmark.dropna()
        if len(b) >= 2:
            bstats = _series_stats(b)
            out["bench_total"] = bstats["total"]
            out["bench_cagr"] = bstats["cagr"]
            out["bench_max_drawdown"] = bstats["max_drawdown"]
            out["excess_total"] = out["total_return"] - bstats["total"]
            out["excess_cagr"] = out["cagr"] - bstats["cagr"]
    return out
