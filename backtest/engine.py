"""Event-driven daily backtest engine.

Timeline for each trading day d:

1. Shares bought on the previous session are unlocked (T+1 rollover).
2. Orders decided at the close of d-1 execute at the open of d with adverse
   slippage. Forced risk exits run first, then strategy sells, then buys.
3. Positions are marked to market at the close of d.
4. The risk overlay evaluates stop loss / drawdown rules at the close of d.
5. The strategy computes new target weights from data up to d; the engine
   converts them into share deltas, which become the orders for the next open.

Because decisions are made on the close and filled at the next open, the
engine never trades on information from the future. The T+1 lock on freshly
bought shares is additionally tracked per position as a hard guard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from backtest.costs import CostModel
from risk.manager import RiskManager, RiskConfig

LOT = 100  # A-share board lot


@dataclass
class Position:
    shares: int = 0
    locked: int = 0          # shares bought today, not sellable until tomorrow
    avg_cost: float = 0.0    # all-in buy basis per share (price + buy fees)
    entry_date: pd.Timestamp | None = None
    cycle_realized: float = 0.0

    @property
    def sellable(self) -> int:
        return max(self.shares - self.locked, 0)


@dataclass
class Trade:
    date: pd.Timestamp
    symbol: str
    side: str                 # "BUY" | "SELL"
    shares: int
    price: float              # executed price (after slippage)
    turnover: float
    fee: float
    fee_commission: float
    fee_stamp: float
    realized_pnl: float       # sells only; buys record 0.0
    cash_after: float
    reason: str               # "signal" | "stop_loss" | "drawdown_flat"


@dataclass
class RoundTrip:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    pnl: float
    exit_reason: str


@dataclass
class BacktestResult:
    equity: pd.Series
    exposure: pd.Series
    benchmark: pd.Series | None
    trades: list[Trade] = field(default_factory=list)
    round_trips: list[RoundTrip] = field(default_factory=list)
    fee_summary: dict = field(default_factory=dict)
    final_positions: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _asof(series: pd.Series, ts: pd.Timestamp) -> float:
    if len(series.index) == 0:
        return float("nan")
    pos = series.index.searchsorted(pd.Timestamp(ts), side="right") - 1
    if pos < 0:
        return float("nan")
    return float(series.iloc[pos])


class _Book:
    """Mutable account state used during a run."""

    def __init__(self, cash: float, symbols: list[str]):
        self.cash = float(cash)
        self.positions = {s: Position() for s in symbols}
        self.trades: list[Trade] = []
        self.trips: list[RoundTrip] = []


class Engine:
    def __init__(
        self,
        panel: dict[str, pd.DataFrame],
        strategy,
        cost_model: CostModel | None = None,
        risk_manager: RiskManager | None = None,
        benchmark: pd.Series | None = None,
        capital: float = 1_000_000.0,
        lot: int = LOT,
        rebalance_band: float = 0.01,
    ):
        """``rebalance_band``: minimum drift (fraction of equity) between
        current and target position value before a partial rebalance trade
        fires; entries from flat and full exits always execute. The band
        prevents pointless 100-share churn from daily mark-to-market drift."""
        self.symbols = sorted(panel.keys())
        self.panel = panel
        self.opens = {s: panel[s]["open"].sort_index() for s in self.symbols}
        self.closes = {s: panel[s]["close"].sort_index() for s in self.symbols}
        self.dates = sorted(set().union(*(set(self.opens[s].index) for s in self.symbols)))
        self.strategy = strategy
        self.cost = cost_model or CostModel()
        self.risk = risk_manager or RiskManager(RiskConfig())
        self.benchmark = None if benchmark is None else benchmark.sort_index()
        self.rebalance_band = float(rebalance_band)
        self.capital = float(capital)
        self.lot = int(lot)

    # ------------------------------------------------------------------ run
    def run(self, start=None, end=None) -> BacktestResult:
        dates = pd.DatetimeIndex(self.dates)
        if start is not None:
            dates = dates[dates >= pd.Timestamp(start)]
        if end is not None:
            dates = dates[dates <= pd.Timestamp(end)]
        if len(dates) < 2:
            raise ValueError("not enough trading dates in the requested range")

        book = _Book(self.capital, self.symbols)
        pending: list[tuple[str, str, int, str]] = []
        equity_pts: dict[pd.Timestamp, float] = {}
        exposure_pts: dict[pd.Timestamp, float] = {}

        for d in dates:
            for pos in book.positions.values():
                pos.locked = 0  # T+1 rollover: yesterday's buys become sellable

            forced_orders = [o for o in pending if o[3] in ("stop_loss", "drawdown_flat")]
            signal_sells = [o for o in pending if o[1] == "SELL" and o[3] == "signal"]
            buys = [o for o in pending if o[1] == "BUY"]
            forced_syms = {o[0] for o in forced_orders}

            for order in forced_orders:
                self._execute_sell(book, order, d)
            for order in signal_sells:
                self._execute_sell(book, order, d)
            for order in buys:
                if order[0] in forced_syms:
                    continue  # a forced exit cancels same-morning entries
                self._execute_buy(book, order, d)

            closes = {s: _asof(self.closes[s], d) for s in self.symbols}
            marked = {
                s: book.positions[s].shares * closes[s]
                for s in self.symbols
                if math.isfinite(closes[s]) and closes[s] > 0
            }
            equity = book.cash + sum(marked.values())
            equity_pts[d] = equity
            exposure_pts[d] = (sum(marked.values()) / equity) if equity > 0 else 0.0

            forced_now = self.risk.on_close(equity, book.positions, closes)
            if self.risk.cooldown_active:
                targets: dict[str, float] = {}
            else:
                targets = self.strategy.target_weights(d, equity)
                targets = self.risk.apply_caps(targets)

            orders: list[tuple[str, str, int, str]] = []
            forced_set = {s for s, _ in forced_now}
            band_value = self.rebalance_band * equity
            for s, reason in forced_now:
                if book.positions[s].shares > 0:
                    orders.append((s, "SELL", book.positions[s].shares, reason))
            for s in self.symbols:
                if s in forced_set:
                    continue
                px = closes[s]
                if not math.isfinite(px) or px <= 0:
                    continue
                w = float(targets.get(s, 0.0))
                pos = book.positions[s]
                target_shares = int(round((w * equity) / px / self.lot)) * self.lot
                current = pos.shares
                diff = target_shares - current
                target_value = target_shares * px
                current_value = current * px
                if current == 0:
                    if target_shares >= self.lot:
                        orders.append((s, "BUY", diff, "signal"))
                elif target_shares <= 0:
                    orders.append((s, "SELL", current, "signal"))
                elif diff >= self.lot and target_value - current_value > band_value:
                    orders.append((s, "BUY", diff, "signal"))
                elif diff <= -self.lot and current_value - target_value > band_value:
                    orders.append((s, "SELL", -diff, "signal"))
            pending = orders

        equity = pd.Series(equity_pts).sort_index()
        exposure = pd.Series(exposure_pts).sort_index()
        bench = None
        if self.benchmark is not None:
            bench = self.benchmark.reindex(equity.index, method="ffill")

        commission = sum(t.fee_commission for t in book.trades)
        stamp = sum(t.fee_stamp for t in book.trades)
        fee_summary = {
            "commission": commission,
            "stamp": stamp,
            "total": commission + stamp,
            "slippage_est": sum(t.turnover * self.cost.slippage_rate for t in book.trades),
        }
        final_positions = {
            s: (p.shares, p.avg_cost) for s, p in book.positions.items() if p.shares > 0
        }
        meta = {
            "strategy": getattr(self.strategy, "name", type(self.strategy).__name__),
            "symbols": list(self.symbols),
            "capital": self.capital,
            "lot": self.lot,
            "start": equity.index[0],
            "end": equity.index[-1],
        }
        return BacktestResult(
            equity=equity,
            exposure=exposure,
            benchmark=bench,
            trades=book.trades,
            round_trips=book.trips,
            fee_summary=fee_summary,
            final_positions=final_positions,
            meta=meta,
        )

    # ------------------------------------------------------------ execution
    def _execute_sell(self, book: _Book, order: tuple, d: pd.Timestamp) -> None:
        symbol, _, shares, reason = order
        pos = book.positions[symbol]
        raw = _asof(self.opens[symbol], d)
        if not math.isfinite(raw) or raw <= 0:
            return  # no printable open; the next decision will re-issue
        shares = min(shares, pos.sellable)  # T+1 hard guard
        if shares <= 0:
            return
        px = self.cost.sell_price(raw)
        turnover = shares * px
        commission = self.cost.commission(turnover)
        stamp = turnover * self.cost.stamp_tax_rate
        fee = commission + stamp
        book.cash += turnover - fee
        realized = (turnover - fee) - pos.avg_cost * shares
        pos.cycle_realized += realized
        pos.shares -= shares
        if pos.shares == 0:
            book.trips.append(
                RoundTrip(symbol, pos.entry_date, d, pos.cycle_realized, reason)
            )
            book.positions[symbol] = Position()
        book.trades.append(
            Trade(d, symbol, "SELL", shares, px, turnover, fee, commission, stamp,
                  realized, book.cash, reason)
        )

    def _execute_buy(self, book: _Book, order: tuple, d: pd.Timestamp) -> None:
        symbol, _, shares, _ = order
        raw = _asof(self.opens[symbol], d)
        if not math.isfinite(raw) or raw <= 0 or shares < self.lot:
            return
        px = self.cost.buy_price(raw)
        turnover = 0.0
        fee = 0.0
        affordable = False
        while shares >= self.lot:
            turnover = shares * px
            fee = self.cost.buy_fee(turnover)
            if turnover + fee <= book.cash:
                affordable = True
                break
            shares -= self.lot
        if not affordable:
            return
        book.cash -= turnover + fee
        pos = book.positions[symbol]
        basis = pos.avg_cost * pos.shares + turnover + fee
        pos.shares += shares
        pos.avg_cost = basis / pos.shares
        pos.locked += shares  # T+1: not sellable today
        if pos.entry_date is None:
            pos.entry_date = d
        book.trades.append(
            Trade(d, symbol, "BUY", shares, px, turnover, fee, fee, 0.0, 0.0,
                  book.cash, "signal")
        )
