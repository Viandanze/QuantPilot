"""Pure Python reference limit order book.

This module implements exactly the same semantics as the C++17 core in
``cpp/order_book.{hpp,cpp}`` - price-time priority, GFD limit orders,
cancels, cancel-resting self-trade prevention, integer prices - in plain
Python. It exists for two reasons:

* cross-checking the C++ engine (identical order streams must produce
  identical fill sequences);
* providing an honest single-core throughput baseline for the benchmark in
  ``benchmarks/bench_matching.py``.

It deliberately avoids third-party dependencies (``sortedcontainers`` and
friends): bids/asks are price -> FIFO deque dictionaries with a sorted price
list maintained via :mod:`bisect`.
"""
from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PyFill:
    """One trade report; field names and order mirror the C++ ``Fill``."""

    taker_order_id: int
    maker_order_id: int
    maker_trader_id: int
    taker_side: str  # "BUY" / "SELL"
    price: int
    quantity: int
    taker_remaining: int
    maker_remaining: int

    def key(self) -> tuple:
        return (
            self.taker_order_id,
            self.maker_order_id,
            self.maker_trader_id,
            self.taker_side,
            self.price,
            self.quantity,
            self.taker_remaining,
            self.maker_remaining,
        )


@dataclass
class PyOrder:
    __slots__ = ("order_id", "trader_id", "side", "price", "quantity", "filled")

    order_id: int
    trader_id: int
    side: str  # "BUY" / "SELL"
    price: int
    quantity: int
    filled: int

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled


@dataclass
class PySubmitResult:
    """Outcome of one submit; mirrors the C++ ``SubmitResult``."""

    order_id: int
    fills: list[PyFill] = field(default_factory=list)
    self_trade_cancels: list[int] = field(default_factory=list)
    rested: bool = False
    resting_price: int = 0
    resting_quantity: int = 0


class PyLimitOrderBook:
    """Deterministic two-sided limit order book (reference implementation)."""

    def __init__(self, first_order_id: int = 1):
        self._next_order_id = max(1, first_order_id)
        # price -> deque[PyOrder]; deque front is the oldest order.
        self._bids: dict[int, deque] = {}
        self._asks: dict[int, deque] = {}
        # Sorted price lists. Bids ascending (best = last), asks ascending
        # (best = first).
        self._bid_prices: list[int] = []
        self._ask_prices: list[int] = []
        # order_id -> (is_bid, price, PyOrder)
        self._index: dict[int, tuple[bool, int, PyOrder]] = {}
        self.submitted_count = 0
        self.fill_count = 0
        self.cancel_count = 0
        self.self_trade_cancel_count = 0

    # ------------------------------------------------------------------ state

    def empty(self) -> bool:
        return not self._index

    def order_count(self) -> int:
        return len(self._index)

    def has_order(self, order_id: int) -> bool:
        return order_id in self._index

    def order_info(self, order_id: int):
        entry = self._index.get(order_id)
        if entry is None:
            return None
        _, _, order = entry
        return {
            "order_id": order.order_id,
            "trader_id": order.trader_id,
            "side": order.side,
            "price": order.price,
            "quantity": order.quantity,
            "filled": order.filled,
        }

    def best_bid(self):
        if not self._bid_prices:
            return None
        price = self._bid_prices[-1]
        return (price, sum(o.remaining for o in self._bids[price]))

    def best_ask(self):
        if not self._ask_prices:
            return None
        price = self._ask_prices[0]
        return (price, sum(o.remaining for o in self._asks[price]))

    def bid_qty_at(self, price: int) -> int:
        level = self._bids.get(price)
        return 0 if level is None else sum(o.remaining for o in level)

    def ask_qty_at(self, price: int) -> int:
        level = self._asks.get(price)
        return 0 if level is None else sum(o.remaining for o in level)

    def spread(self):
        if not self._bid_prices or not self._ask_prices:
            return None
        return self._ask_prices[0] - self._bid_prices[-1]

    def bid_depth(self, levels: int = 10) -> list[tuple[int, int]]:
        out = []
        for price in reversed(self._bid_prices):
            if len(out) >= levels:
                break
            out.append((price, sum(o.remaining for o in self._bids[price])))
        return out

    def ask_depth(self, levels: int = 10) -> list[tuple[int, int]]:
        out = []
        for price in self._ask_prices:
            if len(out) >= levels:
                break
            out.append((price, sum(o.remaining for o in self._asks[price])))
        return out

    # ------------------------------------------------------------------ mutate

    def submit(self, side: str, price: int, quantity: int, trader_id: int = 0,
               order_id: int | None = None) -> PySubmitResult:
        if price <= 0:
            raise ValueError("price must be a positive integer (ticks)")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if order_id is None:
            order_id = self._next_order_id
        elif order_id in self._index:
            raise ValueError("order_id is already in use")
        self._next_order_id = max(self._next_order_id, order_id + 1)
        self.submitted_count += 1

        result = PySubmitResult(order_id=order_id)
        remaining = quantity

        if side == "BUY":
            while remaining > 0 and self._ask_prices and self._ask_prices[0] <= price:
                level_price = self._ask_prices[0]
                level = self._asks[level_price]
                while level and remaining > 0:
                    maker = level[0]
                    if maker.trader_id == trader_id:
                        # Self-trade prevention: cancel the resting order.
                        result.self_trade_cancels.append(maker.order_id)
                        del self._index[maker.order_id]
                        level.popleft()
                        self.self_trade_cancel_count += 1
                        continue
                    trade_qty = min(remaining, maker.remaining)
                    remaining -= trade_qty
                    maker.filled += trade_qty
                    fill = PyFill(
                        taker_order_id=order_id,
                        maker_order_id=maker.order_id,
                        maker_trader_id=maker.trader_id,
                        taker_side="BUY",
                        price=level_price,
                        quantity=trade_qty,
                        taker_remaining=remaining,
                        maker_remaining=maker.remaining,
                    )
                    result.fills.append(fill)
                    self.fill_count += 1
                    if maker.remaining == 0:
                        del self._index[maker.order_id]
                        level.popleft()
                    else:
                        break  # taker exhausted
                if not level:
                    del self._asks[level_price]
                    self._ask_prices.pop(0)
        else:
            while remaining > 0 and self._bid_prices and self._bid_prices[-1] >= price:
                level_price = self._bid_prices[-1]
                level = self._bids[level_price]
                while level and remaining > 0:
                    maker = level[0]
                    if maker.trader_id == trader_id:
                        result.self_trade_cancels.append(maker.order_id)
                        del self._index[maker.order_id]
                        level.popleft()
                        self.self_trade_cancel_count += 1
                        continue
                    trade_qty = min(remaining, maker.remaining)
                    remaining -= trade_qty
                    maker.filled += trade_qty
                    fill = PyFill(
                        taker_order_id=order_id,
                        maker_order_id=maker.order_id,
                        maker_trader_id=maker.trader_id,
                        taker_side="SELL",
                        price=level_price,
                        quantity=trade_qty,
                        taker_remaining=remaining,
                        maker_remaining=maker.remaining,
                    )
                    result.fills.append(fill)
                    self.fill_count += 1
                    if maker.remaining == 0:
                        del self._index[maker.order_id]
                        level.popleft()
                    else:
                        break
                if not level:
                    del self._bids[level_price]
                    self._bid_prices.pop()

        if remaining > 0:
            self._rest(order_id, trader_id, side, price, quantity,
                       quantity - remaining)
            result.rested = True
            result.resting_price = price
            result.resting_quantity = remaining
        return result

    def _rest(self, order_id: int, trader_id: int, side: str, price: int,
              quantity: int, filled: int) -> None:
        order = PyOrder(order_id, trader_id, side, price, quantity, filled)
        if side == "BUY":
            level = self._bids.get(price)
            if level is None:
                level = deque()
                self._bids[price] = level
                insort(self._bid_prices, price)
            level.append(order)
        else:
            level = self._asks.get(price)
            if level is None:
                level = deque()
                self._asks[price] = level
                insort(self._ask_prices, price)
            level.append(order)
        self._index[order_id] = (side == "BUY", price, order)

    def cancel(self, order_id: int) -> bool:
        entry = self._index.pop(order_id, None)
        if entry is None:
            return False
        is_bid, price, order = entry
        levels = self._bids if is_bid else self._asks
        prices = self._bid_prices if is_bid else self._ask_prices
        level = levels[price]
        level.remove(order)  # first match by identity/equality; O(level)
        if not level:
            del levels[price]
            idx = bisect_left(prices, price)
            prices.pop(idx)
        self.cancel_count += 1
        return True
