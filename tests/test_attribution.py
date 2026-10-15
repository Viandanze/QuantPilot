"""Offline synthetic-data tests for the performance attribution module."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from engine.portfolio import Trade
from report.attribution import (
    TRIP_COLUMNS,
    attribution_summary,
    performance_stats,
    round_trip_stats,
)


def _trade(date, symbol, side, shares, price, fee, realized=0.0):
    """Build a minimal Trade-like fill (turnover = shares * price)."""
    return Trade(
        date=pd.Timestamp(date),
        symbol=symbol,
        side=side,
        shares=shares,
        price=price,
        turnover=shares * price,
        fee=fee,
        fee_commission=fee,
        fee_stamp=0.0,
        realized_pnl=realized,
        cash_after=0.0,
        reason="signal",
    )


# -------------------------------------------------------- performance_stats
def test_performance_stats_constant_growth_gives_extreme_sharpe():
    idx = pd.bdate_range("2024-01-02", periods=101)
    equity = pd.Series([100.0 * (1.001**i) for i in range(101)], index=idx)

    s = performance_stats(equity)

    # float noise may leave a tiny-but-nonzero std: accept inf or huge
    assert (not math.isfinite(s["sharpe"])) or s["sharpe"] > 100
    assert (not math.isfinite(s["sortino"])) or s["sortino"] > 100
    assert s["annual_vol"] == pytest.approx(0.0, abs=1e-12)
    assert s["total_return"] == pytest.approx(1.001**100 - 1)
    expected_cagr = (1.001**100) ** (244 / 101) - 1      # 101 points / 244 per year
    assert s["cagr"] == pytest.approx(expected_cagr)
    assert s["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert s["max_drawdown_days"] == 0
    assert math.isnan(s["calmar"])                       # no drawdown -> undefined


def test_performance_stats_known_drawdown_is_exact():
    equity = pd.Series([100.0, 110.0, 99.0, 121.0, 90.0, 108.0])

    s = performance_stats(equity)

    assert s["total_return"] == pytest.approx(0.08)
    assert s["max_drawdown"] == pytest.approx(90.0 / 121.0 - 1.0)
    assert s["max_drawdown_days"] == 2                   # 90 -> 108 below the 121 peak
    expected_cagr = (108.0 / 100.0) ** (244 / 6) - 1.0   # 6 points / 244 per year
    assert s["cagr"] == pytest.approx(expected_cagr)
    assert s["calmar"] == pytest.approx(expected_cagr / abs(90.0 / 121.0 - 1.0))

    # pin the annualisation formulas against an independent recomputation
    rets = np.array(
        [110 / 100 - 1, 99 / 110 - 1, 121 / 99 - 1, 90 / 121 - 1, 108 / 90 - 1]
    )
    assert s["annual_vol"] == pytest.approx(rets.std(ddof=1) * math.sqrt(244))
    assert s["sharpe"] == pytest.approx(rets.mean() / rets.std(ddof=1) * math.sqrt(244))
    downside = np.sqrt(np.mean(np.minimum(rets, 0.0) ** 2))
    assert s["sortino"] == pytest.approx(rets.mean() / downside * math.sqrt(244))


def test_performance_stats_single_point_returns_all_nan_without_raising():
    for series in (pd.Series([100.0]), pd.Series(dtype=float)):
        s = performance_stats(series)
        assert set(s.keys()) == {
            "total_return", "cagr", "annual_vol", "sharpe", "sortino",
            "max_drawdown", "max_drawdown_days", "calmar",
        }
        for value in s.values():
            assert isinstance(value, float) and math.isnan(value)


def test_performance_stats_two_points_gives_level_stats_but_nan_risk():
    s = performance_stats(pd.Series([100.0, 110.0]))

    assert s["total_return"] == pytest.approx(0.10)
    assert s["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert s["cagr"] == pytest.approx(1.1 ** (244 / 2) - 1)   # 2 points / 244
    # a single return observation cannot give a sample std
    assert math.isnan(s["annual_vol"])
    assert math.isnan(s["sharpe"])
    assert math.isnan(s["sortino"])


# --------------------------------------------------------- round_trip_stats
def test_round_trip_stats_two_trips_one_win_one_loss():
    equity = pd.Series(
        [1_000_000.0] * 3, index=pd.bdate_range("2024-01-02", periods=3)
    )
    trades = [
        _trade("2024-01-02", "600519", "BUY", 1000, 10.0, 10.0),
        _trade("2024-01-05", "600519", "SELL", 1000, 12.0, 5.0, realized=1985.0),
        _trade("2024-01-03", "000001", "BUY", 2000, 5.0, 8.0),
        _trade("2024-01-10", "000001", "SELL", 2000, 4.5, 6.0, realized=-1014.0),
    ]

    df = round_trip_stats(trades, equity)

    assert len(df) == 2
    assert list(df.columns) == list(TRIP_COLUMNS)

    win = df.iloc[0]
    assert win["symbol"] == "600519"
    assert win["entry_date"] == pd.Timestamp("2024-01-02")
    assert win["exit_date"] == pd.Timestamp("2024-01-05")
    assert win["entry_px"] == pytest.approx(10.0)
    assert win["exit_px"] == pytest.approx(12.0)
    assert win["shares"] == 1000
    assert win["pnl"] == pytest.approx(1985.0)            # (12000-5) - (10000+10)
    assert win["pnl_pct"] == pytest.approx(1985.0 / 10010.0)
    assert win["holding_days"] == 3
    assert win["cumulative_contribution"] == pytest.approx(1985.0 / 1_000_000.0)

    loss = df.iloc[1]
    assert loss["symbol"] == "000001"
    assert loss["pnl"] == pytest.approx(-1014.0)          # (9000-6) - (10000+8)
    assert loss["pnl_pct"] == pytest.approx(-1014.0 / 10008.0)
    assert loss["holding_days"] == 7
    assert loss["cumulative_contribution"] == pytest.approx(-1014.0 / 1_000_000.0)


def test_round_trip_stats_partial_fills_and_open_position_are_handled():
    equity = pd.Series(
        [1_000_000.0] * 5, index=pd.bdate_range("2024-01-02", periods=5)
    )
    trades = [
        # one clean winner
        _trade("2024-01-02", "AAA", "BUY", 300, 20.0, 12.0),
        _trade("2024-01-03", "AAA", "SELL", 300, 21.0, 9.0),    # (6300-9)-(6000+12)
        # scaled in once, out in two lots -> a single round trip
        _trade("2024-01-03", "BBB", "BUY", 1000, 10.0, 10.0),
        _trade("2024-01-04", "BBB", "SELL", 400, 11.0, 4.0),    # +392
        _trade("2024-01-05", "BBB", "SELL", 600, 9.5, 5.0),     # -311
        # never closed: must not appear as a round trip
        _trade("2024-01-05", "CCC", "BUY", 500, 8.0, 5.0),
    ]

    df = round_trip_stats(trades, equity)

    assert len(df) == 2
    assert list(df["symbol"]) == ["AAA", "BBB"]
    assert df.iloc[0]["pnl"] == pytest.approx(279.0)
    bbb = df.iloc[1]
    assert bbb["pnl"] == pytest.approx(81.0)              # (4400-4)+(5700-5)-(10000+10)
    assert bbb["shares"] == 1000
    assert bbb["entry_px"] == pytest.approx(10.0)
    assert bbb["exit_px"] == pytest.approx(9.5)
    assert bbb["holding_days"] == 2
    assert "CCC" not in set(df["symbol"])


def test_round_trip_stats_empty_trades_gives_empty_framed_table():
    df = round_trip_stats([], pd.Series([100.0, 101.0]))

    assert df.empty
    assert list(df.columns) == list(TRIP_COLUMNS)


# ----------------------------------------------------- attribution_summary
def test_attribution_summary_benchmark_moves_in_lockstep():
    idx = pd.bdate_range("2024-01-02", periods=30)
    equity = pd.Series([100.0 * (1.002**i) for i in range(30)], index=idx)
    benchmark = equity * 0.5                    # identical returns, half the level

    out = attribution_summary(equity, [], benchmark)

    assert out["beta"] == pytest.approx(1.0, abs=1e-9)
    assert out["alpha_annualized"] == pytest.approx(0.0, abs=1e-8)
    assert out["excess_return"] == pytest.approx(0.0, abs=1e-12)
    assert out["tracking_error"] == pytest.approx(0.0, abs=1e-10)
    assert out["stats"]["total_return"] == pytest.approx(1.002**29 - 1)
    assert out["top_winners"] == []
    assert out["top_losers"] == []
    assert out["by_symbol"] == {}


def test_attribution_summary_without_benchmark_leaves_exposure_none():
    out = attribution_summary(pd.Series([100.0, 101.0, 99.0, 103.0]), [])

    for key in ("beta", "alpha_annualized", "excess_return", "tracking_error"):
        assert out[key] is None
    assert out["stats"]["total_return"] == pytest.approx(0.03)


def test_attribution_summary_misaligned_benchmark_degrades_to_none():
    equity = pd.Series([100.0, 101.0], index=pd.bdate_range("2024-01-02", periods=2))
    benchmark = pd.Series([50.0, 51.0], index=pd.bdate_range("2030-01-02", periods=2))

    out = attribution_summary(equity, [], benchmark)

    assert out["beta"] is None
    assert out["alpha_annualized"] is None
    assert out["excess_return"] is None
    assert out["tracking_error"] is None


def test_attribution_summary_top5_truncates_and_orders_by_pnl():
    equity = pd.Series(
        [1_000_000.0] * 7, index=pd.bdate_range("2024-01-02", periods=7)
    )
    plan = [                       # net pnl = 1000 * (sell_px - 10), zero fees
        ("AAA", 10.5),             # +500
        ("BBB", 9.8),              # -200
        ("CCC", 10.3),             # +300
        ("DDD", 10.1),             # +100
        ("EEE", 9.95),             # -50
        ("FFF", 10.7),             # +700
        ("GGG", 9.1),              # -900
    ]
    trades = []
    for i, (symbol, sell_px) in enumerate(plan):
        day = equity.index[i]
        trades.append(_trade(day, symbol, "BUY", 1000, 10.0, 0.0))
        trades.append(_trade(day, symbol, "SELL", 1000, sell_px, 0.0))

    out = attribution_summary(equity, trades)

    assert len(out["top_winners"]) == 5                # 7 trips -> capped at 5
    assert [r["symbol"] for r in out["top_winners"]] == ["FFF", "AAA", "CCC", "DDD", "EEE"]
    assert [r["symbol"] for r in out["top_losers"]] == ["GGG", "BBB", "EEE", "DDD", "CCC"]
    assert set(out["by_symbol"]) == {sym for sym, _ in plan}
    assert out["by_symbol"]["FFF"]["pnl"] == pytest.approx(700.0)
    assert out["by_symbol"]["FFF"]["win_rate"] == pytest.approx(1.0)
    assert out["by_symbol"]["GGG"]["win_rate"] == pytest.approx(0.0)
    assert out["by_symbol"]["AAA"]["avg_holding_days"] == pytest.approx(0.0)
