"""V1/V2 arithmetic equivalence against frozen reference numbers.

The event-driven engine (``engine.runner.Engine``, re-exported through
``backtest.engine``) must reproduce the original daily-loop engine's
results operation-for-operation. The reference numbers below were
captured from the V1 engine at commit ``90d5802`` using the same recipe
as the CLI: built-in sample data, window 2024-01-01..2026-08-21, a
one-year fetch buffer for indicator warmup, default strategy parameters,
and the default cost and risk configuration.

On the capture platform the V2 engine reproduces these values
bit-for-bit - identical trade lists, equity curves, fee totals and
final positions. The tolerances below only absorb ulp-level
floating-point differences on other platforms; they must never absorb
behavioural drift. Counts are compared for exact equality and money
amounts within 1e-6 CNY.

The first trades of the dual_ma run double as a fine-grained anchor:
date, symbol, side, size, executed price, fee and reason must all match,
which pins the fill pipeline (next-open pricing, slippage, commission
minimum, stamp tax) and not just the aggregate curve.
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest.costs import CostModel
from backtest.engine import Engine
from data import loader
from report.metrics import compute_metrics
from risk.manager import RiskConfig, RiskManager
from strategies import create

START = "2024-01-01"
END = "2026-08-21"
FETCH_BUFFER_DAYS = 365  # same warmup as the backtest.py CLI

REFERENCE = {
    "dual_ma": {
        "sessions": 690,
        "years": 2.738095238095238,
        "total_return": -0.16276760188424777,
        "cagr": -0.06282212581710478,
        "ann_vol": 0.07226346373871248,
        "sharpe": -0.8629506326112549,
        "max_drawdown": -0.2204034253281223,
        "max_dd_sessions": 612,
        "calmar": -0.28503243869091094,
        "avg_exposure": 0.4316457367789349,
        "n_trades": 316,
        "n_buys": 150,
        "n_sells": 166,
        "turnover_total": 40449662.484,
        "fees_total": 30201.45588425,
        "commission_total": 10255.09121525,
        "stamp_total": 19946.364669,
        "stop_loss_exits": 3,
        "drawdown_flats": 3,
        "round_trips": 121,
        "win_rate": 0.2975206611570248,
        "profit_factor": 0.6664499289814011,
        "avg_trip_pnl": -1443.5583689834539,
        "final_equity": 837232.3981157523,
    },
    "momentum": {
        "sessions": 690,
        "years": 2.738095238095238,
        "total_return": -0.31819497188924895,
        "cagr": -0.13053958913001362,
        "ann_vol": 0.11686564781411726,
        "sharpe": -1.1400118784494475,
        "max_drawdown": -0.3871693144472509,
        "max_dd_sessions": 470,
        "calmar": -0.3371640888337981,
        "avg_exposure": 0.8577128371446451,
        "n_trades": 293,
        "n_buys": 147,
        "n_sells": 146,
        "turnover_total": 20266930.110999998,
        "fees_total": 15105.99688925,
        "commission_total": 5460.22032125,
        "stamp_total": 9645.776568,
        "stop_loss_exits": 23,
        "drawdown_flats": 15,
        "round_trips": 51,
        "win_rate": 0.21568627450980393,
        "profit_factor": 0.32962959499920064,
        "avg_trip_pnl": -6153.516824882333,
        "final_equity": 681805.0281107511,
    },
    "grid": {
        "sessions": 690,
        "years": 2.738095238095238,
        "total_return": -0.23249563231524795,
        "cagr": -0.09211774041117404,
        "ann_vol": 0.08793888395881976,
        "sharpe": -1.0564272250637023,
        "max_drawdown": -0.29264730300295616,
        "max_dd_sessions": 423,
        "calmar": -0.3147739256980049,
        "avg_exposure": 0.5892178133605537,
        "n_trades": 651,
        "n_buys": 324,
        "n_sells": 327,
        "turnover_total": 24743772.941,
        "fees_total": 18376.65731525,
        "commission_total": 6416.11283225,
        "stamp_total": 11960.544483,
        "stop_loss_exits": 31,
        "drawdown_flats": 8,
        "round_trips": 60,
        "win_rate": 0.48333333333333334,
        "profit_factor": 0.4326918715553373,
        "avg_trip_pnl": -3696.7623090458083,
        "final_equity": 767504.3676847521,
    },
}

COUNT_KEYS = (
    "sessions", "max_dd_sessions", "n_trades", "n_buys", "n_sells",
    "stop_loss_exits", "drawdown_flats", "round_trips",
)
MONEY_KEYS = ("turnover_total", "fees_total", "commission_total", "stamp_total")
RATIO_KEYS = (
    "years", "total_return", "cagr", "ann_vol", "sharpe", "max_drawdown",
    "calmar", "avg_exposure", "win_rate", "profit_factor", "avg_trip_pnl",
)

# (date, symbol, side, shares, executed price, total fee, reason)
DUAL_MA_FIRST_TRADES = [
    ("2024-01-02", "000858", "BUY", 1100, 157.77762, 43.388845, "signal"),
    ("2024-01-03", "000858", "SELL", 1100, 152.19765, 209.271769, "signal"),
    ("2024-01-05", "000333", "BUY", 2100, 77.38731, 40.628338, "signal"),
    ("2024-01-05", "510300", "BUY", 37400, 4.41441, 41.274733, "signal"),
    ("2024-01-08", "600519", "BUY", 100, 1171.4703, 29.286758, "signal"),
    ("2024-01-09", "601318", "BUY", 2100, 79.56949, 41.773982, "signal"),
    ("2024-01-16", "600036", "BUY", 4100, 39.63960, 40.63059, "signal"),
    ("2024-01-17", "600519", "SELL", 100, 1167.26157, 145.907696, "signal"),
]


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def panel():
    frames = loader.load_daily(
        loader.DEFAULT_SYMBOLS,
        pd.Timestamp(START) - pd.Timedelta(days=FETCH_BUFFER_DAYS),
        pd.Timestamp(END),
        offline=True,
    )
    return loader.build_panel(frames)


def _run(panel, strategy_name):
    engine = Engine(
        panel,
        create(strategy_name, panel),
        CostModel(),
        RiskManager(RiskConfig()),
        capital=1_000_000.0,
    )
    return engine.run(start=START, end=END)


@pytest.fixture(scope="module")
def results(panel):
    return {name: _run(panel, name) for name in REFERENCE}


# ------------------------------------------------------------------- tests
@pytest.mark.parametrize("strategy_name", sorted(REFERENCE))
def test_metrics_match_v1_reference(results, strategy_name):
    res = results[strategy_name]
    metrics = compute_metrics(
        res.equity, res.benchmark, res.trades, res.round_trips, exposure=res.exposure
    )
    ref = REFERENCE[strategy_name]

    for key in COUNT_KEYS:
        assert metrics[key] == ref[key], f"{strategy_name}.{key}"
    for key in MONEY_KEYS:
        assert metrics[key] == pytest.approx(ref[key], abs=1e-6), f"{strategy_name}.{key}"
    for key in RATIO_KEYS:
        assert metrics[key] == pytest.approx(ref[key], abs=1e-9), f"{strategy_name}.{key}"
    assert float(res.equity.iloc[-1]) == pytest.approx(ref["final_equity"], abs=1e-6)


def test_dual_ma_first_trades_anchored(results):
    trades = results["dual_ma"].trades
    assert len(trades) >= len(DUAL_MA_FIRST_TRADES)
    for expected, trade in zip(DUAL_MA_FIRST_TRADES, trades):
        date, symbol, side, shares, price, fee, reason = expected
        assert (str(trade.date.date()), trade.symbol, trade.side,
                trade.shares, trade.reason) == (date, symbol, side, shares, reason)
        assert trade.price == pytest.approx(price, abs=1e-6)
        assert trade.fee == pytest.approx(fee, abs=1e-6)


@pytest.mark.parametrize("strategy_name", sorted(REFERENCE))
def test_ledger_integrity_holds(results, strategy_name):
    integrity = results[strategy_name].integrity
    assert integrity["all_checks_passed"] is True
    assert integrity["max_ledger_drift"] <= integrity["audit_tolerance"]
    assert integrity["fills_audited"] == len(results[strategy_name].trades)


def test_repeated_run_is_deterministic(panel):
    first = _run(panel, "dual_ma")
    second = _run(panel, "dual_ma")
    assert list(first.equity.index) == list(second.equity.index)
    assert (first.equity.values == second.equity.values).all()
    assert len(first.trades) == len(second.trades)
    assert [(t.date, t.symbol, t.side, t.shares, t.price, t.fee)
            for t in first.trades] == \
           [(t.date, t.symbol, t.side, t.shares, t.price, t.fee)
            for t in second.trades]
