"""Risk manager tests: caps, stop-loss threshold, drawdown cooldown."""
import pytest

from backtest.engine import Position
from risk.manager import RiskConfig, RiskManager


def test_position_cap_and_blocked_symbols():
    manager = RiskManager()
    out = manager.apply_caps({"AAA": 0.5, "BBB": 0.1})
    assert out["AAA"] == pytest.approx(0.2)
    assert out["BBB"] == pytest.approx(0.1)

    manager.symbol_blocks["XXX"] = 3
    blocked = manager.apply_caps({"XXX": 0.2})
    assert blocked["XXX"] == 0.0


def test_total_weight_trimmed_to_one():
    manager = RiskManager(RiskConfig(max_position_weight=1.0))
    out = manager.apply_caps({"AAA": 0.7, "BBB": 0.6})
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["AAA"] > out["BBB"]


def test_stop_loss_threshold_is_inclusive():
    manager = RiskManager()
    hit = manager.on_close(100_000.0, {"S": Position(shares=100, avg_cost=100.0)}, {"S": 92.0})
    assert hit == [("S", "stop_loss")]

    miss = RiskManager().on_close(
        100_000.0, {"S": Position(shares=100, avg_cost=100.0)}, {"S": 92.01}
    )
    assert miss == []


def test_drawdown_guard_triggers_beyond_threshold():
    calm = RiskManager()
    calm.on_close(100.0, {}, {})   # set the peak
    calm.on_close(86.0, {}, {})    # -14% -> inside tolerance
    assert not calm.cooldown_active

    breach = RiskManager()
    breach.on_close(100.0, {}, {})
    breach.on_close(84.0, {}, {})  # -16% -> beyond the 15% line -> cooldown
    assert breach.cooldown_active


def test_cooldown_releases_and_resets_peak():
    manager = RiskManager(RiskConfig(cooldown_days=20))
    manager.on_close(100.0, {}, {})
    manager.on_close(80.0, {}, {})            # breach
    assert manager.cooldown_active
    for _ in range(20):
        manager.on_close(90.0, {}, {})        # cash-only cooldown
    assert not manager.cooldown_active
    assert manager.peak_equity == pytest.approx(90.0)

    # fresh peak means an 80 close is no longer a breach
    forced = manager.on_close(80.0, {}, {})
    assert forced == [] and not manager.cooldown_active


def test_symbol_reentry_block_expires():
    manager = RiskManager(RiskConfig(reentry_cooldown=10))
    manager.on_close(
        100_000.0, {"S": Position(shares=100, avg_cost=100.0)}, {"S": 90.0}
    )
    assert manager.blocked("S")
    for _ in range(10):
        manager.on_close(100_000.0, {}, {})
    assert not manager.blocked("S")
