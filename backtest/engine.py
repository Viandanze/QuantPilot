"""Compatibility facade over the event-driven engine core.

The original vector-style daily loop lived here. It has been rewritten as
the five-module event-driven core in ``engine/`` (typed events + queue,
DataFeed, Portfolio, ExecutionHandler, RiskManager); this module keeps the
historical import path - ``from backtest.engine import Engine`` - and the
public result types stable so the CLI, the reports and the test suite work
unchanged.

Equivalence: the event-driven runner reproduces the retired loop's
arithmetic operation-for-operation (fill priority, T+1 clamps, lot
rounding, float ordering), so backtest metrics are identical to the
previous engine. ``tests/test_equivalence.py`` pins this against
snapshots captured from the original implementation before the rewrite.
"""
from __future__ import annotations

from engine.portfolio import Position, RoundTrip, Trade
from engine.runner import BacktestResult, Engine

__all__ = ["BacktestResult", "Engine", "Position", "RoundTrip", "Trade"]
