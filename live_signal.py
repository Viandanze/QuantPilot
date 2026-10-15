#!/usr/bin/env python3
"""QuantPilot live paper-trading signal generator (inception: 2026-09-18).

Data: tencent ifzq fqkline HTTP API, forward-adjusted (qfq). Cross-validated
on 2026-09-18 against the paper-broker quote feed (600519 prev_close
1266.98 matches exactly). eastmoney direct is blocked on this host, so the
primary source here is tencent; the generator refuses to run on stale,
short or missing series.

Mirrors the backtest temporal lock: decide on the last COMPLETED close (T-1),
execute on T's session. dual_ma = live book (backtest-aligned 6-symbol
universe, 1/6 each). momentum (24-symbol liquid pool) and grid (6 symbols)
run as SHADOW signals (ledger only, no orders) until promoted.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

REPO = pathlib.Path(r"C:\Users\17600\QuantPilot")
sys.path.insert(0, str(REPO))

import pandas as pd

from strategies.dual_ma import DualMAStrategy
from strategies.grid import GridStrategy
from strategies.momentum import MomentumStrategy

OUT_DIR = pathlib.Path(r"C:\Users\17600\Coze\Drive\quantpliot\paper_live")

TODAY = pd.Timestamp("2026-09-18")
DECIDE = pd.Timestamp("2026-09-17")  # last completed session
BEG, END, BARS = "2026-02-20", "2026-09-18", 320
MIN_ROWS = 60

CORE = ["600519", "000858", "601318", "000333", "600036", "510300"]
POOL_EXTRA = [
    "300750", "002594", "600900", "601088", "601899", "600030", "601166",
    "300059", "600276", "300760", "603259", "002415", "002475", "601138",
    "300308", "600690", "600031", "601888",
]
POOL = CORE + POOL_EXTRA

CAPITAL = 1_000_000.0
LOT = 100
CASH_GUARD = 0.995

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Referer": "https://gu.qq.com/",
    "Accept": "*/*",
}


def _tx_code(symbol: str) -> str:
    return ("sh" if symbol.startswith(("6", "5", "9")) else "sz") + symbol


def fetch_daily(symbol: str) -> pd.DataFrame:
    """Fetch qfq daily bars from tencent ifzq; columns date/open/close/high/low/volume."""
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={_tx_code(symbol)},day,{BEG},{END},{BARS},qfq"
    )
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    node = (payload.get("data") or {}).get(_tx_code(symbol)) or {}
    rows = node.get("qfqday") or node.get("day") or []
    if not rows:
        raise ValueError(f"empty kline response for {symbol}")
    parsed = []
    for r in rows:
        # [date, open, close, high, low, volume, ...]
        parsed.append((r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
    df = pd.DataFrame(parsed, columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().drop_duplicates("date").set_index("date").sort_index()
    return df


def load_pool() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(POOL):
        try:
            df = fetch_daily(symbol, BEG, END, BARS)
        except Exception as exc:  # noqa: BLE001 - one bad name must not kill the run
            raise SystemExit(f"ABORT: fetch failed for {symbol}: {exc}") from exc
        if len(df) < MIN_ROWS:
            raise SystemExit(f"ABORT: {symbol} has only {len(df)} rows (<{MIN_ROWS})")
        if df.index.max() < TODAY - pd.Timedelta(days=7):
            raise SystemExit(f"ABORT: {symbol} data stale (last {df.index.max().date()})")
        frames[symbol] = df
        if i < len(POOL) - 1:
            time.sleep(0.2)
    return frames


def main() -> None:
    frames = load_pool()
    hist = {s: frames[s].loc[:DECIDE] for s in POOL}
    stale = {
        s: str(hist[s].index.max().date())
        for s in POOL
        if hist[s].empty or hist[s].index.max() < DECIDE - pd.Timedelta(days=7)
    }
    if stale:
        raise SystemExit(f"ABORT: stale/missing decision data {stale}")
    hist6 = {s: hist[s] for s in CORE}

    # ---------------- dual_ma: live book ----------------
    dm = DualMAStrategy(hist6)
    dm_w = dm.target_weights(DECIDE, CAPITAL)

    # ---------------- momentum: shadow ----------------
    mom = MomentumStrategy(hist, lookback=20, top_n=3, rebalance_every=20)
    mom_w = mom.target_weights(DECIDE, CAPITAL)

    # ---------------- grid: shadow ----------------
    grid = GridStrategy(hist6)
    grid_w = grid.target_weights(DECIDE, CAPITAL)

    closes = {s: float(hist[s]["close"].iloc[-1]) for s in POOL}

    orders: list[dict] = []
    for s in CORE:
        w = dm_w.get(s, 0.0)
        if w <= 0:
            continue
        px = closes[s]
        shares = int(round((w * CAPITAL) / px / LOT)) * LOT
        if shares < LOT:
            continue
        orders.append({"symbol": s, "weight": round(w, 4), "ref_close": px,
                       "shares": shares, "notional": round(shares * px, 2)})
    total = sum(r["notional"] for r in orders)
    while orders and total > CAPITAL * CASH_GUARD:
        last = orders[-1]
        last["shares"] -= LOT
        last["notional"] = round(last["shares"] * last["ref_close"], 2)
        if last["shares"] < LOT:
            orders.pop()
        total = sum(r["notional"] for r in orders)

    result = {
        "run_date": str(TODAY.date()),
        "decision_date": str(DECIDE.date()),
        "data_source": "tencent_ifzq_qfq",
        "core_sessions": {s: int(len(hist[s])) for s in CORE},
        "dual_ma_live": {
            "rule": "hold while MA5>MA20, 1/6 weight each (backtest-aligned)",
            "signals": {s: dm.note(DECIDE, s) for s in CORE},
            "orders": orders,
            "planned_notional": round(total, 2),
            "planned_invested_pct": round(total / CAPITAL, 4),
            "universe_note": "600519 kept; 100-share lot fits a 1/6 slice of 1M",
        },
        "momentum_shadow": {
            "config": {"pool_size": len(POOL), "lookback": 20, "top_n": 3,
                        "rebalance_every": 20},
            "targets": {s: round(mom_w.get(s, 0.0), 4) for s in POOL
                        if mom_w.get(s, 0.0) > 0},
            "ranks": {s: mom.note(DECIDE, s) for s in POOL},
        },
        "grid_shadow": {
            "targets": {s: round(grid_w.get(s, 0.0), 4) for s in CORE},
            "states": {s: grid.note(DECIDE, s) for s in CORE},
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{TODAY.date()}_signals.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
