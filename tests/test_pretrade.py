"""Pre-trade fail-closed gate tests: every rule, accumulation, audit."""
import json

from risk.pretrade import (AccountState, OrderIntent, PreTradeConfig,
                           PreTradeGate)


def _state(**over) -> AccountState:
    base = dict(equity=1_000_000.0, cash=500_000.0,
                position_weights={"600036": 0.10}, orders_placed_today=0,
                day_pnl_pct=0.0)
    base.update(over)
    return AccountState(**base)


def _buy(**over) -> OrderIntent:
    base = dict(symbol="600036", side="buy", quantity=1000, price=40.0,
                reference_price=40.0)
    base.update(over)
    return OrderIntent(**base)


def test_clean_buy_passes():
    assert PreTradeGate().check(_buy(), _state()).approved


def test_notional_cap_rejects():
    d = PreTradeGate().check(_buy(quantity=10_000), _state())  # 400k > 300k
    assert not d.approved and "order_notional_cap" in d.reasons


def test_cash_check_rejects():
    d = PreTradeGate().check(_buy(quantity=15_000), _state(cash=100_000.0))
    assert "insufficient_cash" in d.reasons


def test_position_weight_cap_rejects():
    # 0.10 held + 200k/1M = 0.30 > 0.20
    d = PreTradeGate().check(_buy(quantity=5000), _state())
    assert "position_weight_cap" in d.reasons


def test_price_deviation_rejects():
    d = PreTradeGate().check(_buy(price=45.0), _state())  # +12.5% vs ref
    assert "price_deviation" in d.reasons


def test_price_deviation_tolerates_small_buffer():
    gate = PreTradeGate(PreTradeConfig(max_price_deviation=0.03))
    assert gate.check(_buy(price=41.0), _state()).approved  # +2.5% ok


def test_kill_switch_blocks_buys_and_sells():
    gate = PreTradeGate()
    d = gate.check(_buy(), _state(day_pnl_pct=-0.05))
    assert not d.approved and "kill_switch_daily_loss" in d.reasons
    sell = OrderIntent(symbol="600036", side="sell", quantity=1000, price=40.0)
    d2 = gate.check(sell, _state(day_pnl_pct=-0.06))
    assert not d2.approved and "kill_switch_daily_loss" in d2.reasons


def test_order_rate_cap():
    d = PreTradeGate().check(_buy(), _state(orders_placed_today=20))
    assert "order_rate_cap" in d.reasons


def test_sell_ok_with_position_rejected_without():
    gate = PreTradeGate()
    sell = OrderIntent(symbol="600036", side="sell", quantity=1000, price=40.0)
    assert gate.check(sell, _state()).approved
    orphan = OrderIntent(symbol="300750", side="sell", quantity=1000, price=40.0)
    d = gate.check(orphan, _state())
    assert not d.approved and "no_position" in d.reasons


def test_malformed_orders_rejected():
    gate = PreTradeGate()
    for bad in (_buy(quantity=0), _buy(price=-1.0), _buy(side="nonsense")):
        d = gate.check(bad, _state())
        assert not d.approved and "malformed_order" in d.reasons


def test_violations_accumulate():
    d = PreTradeGate().check(_buy(quantity=50_000, price=50.0), _state(cash=1_000.0))
    assert len(d.reasons) >= 3


def test_audit_log_written(tmp_path):
    path = tmp_path / "audit" / "decisions.jsonl"
    gate = PreTradeGate(audit_path=path)
    gate.check(_buy(), _state())
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["approved"] is True
    assert rec["order"]["symbol"] == "600036"


def test_audit_failure_fails_closed(tmp_path):
    target = tmp_path / "audit.jsonl"
    target.mkdir()  # directory in place of file -> OSError on append
    gate = PreTradeGate(audit_path=target)
    d = gate.check(_buy(), _state())
    assert not d.approved and "audit_write_failed" in d.reasons
