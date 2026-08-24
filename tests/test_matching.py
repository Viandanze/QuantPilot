"""Tests for the qp_matching pybind11 bindings (C++17 limit order book).

The suite skips cleanly when the extension module has not been built - see
the Matching engine section of the README for build instructions:
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=...
    cmake --build build
"""
from __future__ import annotations

import random
import sys
import threading
import time
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))

qp = pytest.importorskip(
    "qp_matching", reason="qp_matching not built - see README 'Matching engine'"
)

from py_limit_book import PyLimitOrderBook  # noqa: E402

BUY = qp.Side.BUY
SELL = qp.Side.SELL


def cpp_fill_keys(result):
    return [
        (
            f.taker_order_id,
            f.maker_order_id,
            f.maker_trader_id,
            "BUY" if f.taker_side == BUY else "SELL",
            f.price,
            f.quantity,
            f.taker_remaining,
            f.maker_remaining,
        )
        for f in result.fills
    ]


class TestMatchingSemantics:
    def test_price_time_priority(self):
        book = qp.LimitOrderBook()
        book.submit(BUY, 100, 5, 1)
        book.submit(BUY, 100, 5, 2)
        result = book.submit(SELL, 100, 7, 3)
        assert [(f.maker_order_id, f.quantity, f.maker_remaining) for f in result.fills] == [
            (1, 5, 0),  # oldest first
            (2, 2, 3),  # then the younger one, partially
        ]
        assert not result.rested
        assert book.best_bid() == (100, 3)

    def test_partial_fill_and_resting_remainder(self):
        book = qp.LimitOrderBook()
        book.submit(BUY, 100, 10, 1)
        small = book.submit(SELL, 100, 4, 2)
        assert len(small.fills) == 1
        assert small.fills[0].maker_remaining == 6
        assert not small.rested

        big = book.submit(SELL, 100, 15, 3)
        assert len(big.fills) == 1
        assert big.fills[0].quantity == 6
        assert big.rested and big.resting_price == 100 and big.resting_quantity == 9
        assert book.best_ask() == (100, 9)

    def test_crossing_sweep_over_multiple_levels(self):
        book = qp.LimitOrderBook()
        book.submit(BUY, 98, 3, 1)
        book.submit(BUY, 99, 4, 2)
        book.submit(BUY, 100, 5, 3)
        result = book.submit(SELL, 98, 9, 4)
        assert [(f.price, f.quantity) for f in result.fills] == [
            (100, 5),  # best price first
            (99, 4),
        ]
        assert not result.rested
        assert book.best_bid() == (98, 3)
        assert book.bid_level_count() == 1

    def test_self_trade_prevention(self):
        book = qp.LimitOrderBook()
        resting = book.submit(BUY, 100, 5, 7)
        assert resting.rested

        crossing = book.submit(SELL, 99, 10, 7)  # same trader id
        assert crossing.fills == []
        assert list(crossing.self_trade_cancels) == [resting.order_id]
        assert crossing.rested and crossing.resting_price == 99
        assert not book.has_order(resting.order_id)
        assert book.best_bid() is None
        assert book.best_ask() == (99, 10)
        assert book.self_trade_cancel_count() == 1

        taker = book.submit(BUY, 99, 10, 8)  # a different trader may take it
        assert len(taker.fills) == 1
        assert taker.fills[0].maker_order_id == crossing.order_id

    def test_cancel_updates_levels_and_best(self):
        book = qp.LimitOrderBook()
        a = book.submit(BUY, 100, 5, 1)
        b = book.submit(BUY, 100, 7, 2)
        c = book.submit(BUY, 99, 3, 3)
        assert book.best_bid() == (100, 12)

        assert book.cancel(b.order_id) is True
        assert book.best_bid() == (100, 5)
        assert book.cancel(b.order_id) is False   # already gone
        assert book.cancel(987654) is False       # unknown

        assert book.cancel(a.order_id) is True
        assert book.best_bid() == (99, 3)
        assert book.bid_qty_at(100) == 0

        assert book.cancel(c.order_id) is True
        assert book.empty()

    def test_book_queries(self):
        book = qp.LimitOrderBook()
        assert book.empty() and book.best_bid() is None and book.best_ask() is None
        assert book.spread() is None

        book.submit(BUY, 98, 4, 1)
        book.submit(BUY, 97, 6, 2)
        book.submit(SELL, 101, 5, 3)
        book.submit(SELL, 102, 2, 4)

        assert book.best_bid() == (98, 4)
        assert book.best_ask() == (101, 5)
        assert book.spread() == 3
        assert book.bid_qty_at(97) == 6
        assert book.ask_qty_at(102) == 2
        assert book.bid_qty_at(105) == 0
        assert book.bid_depth(10) == [(98, 4), (97, 6)]
        assert book.ask_depth(10) == [(101, 5), (102, 2)]
        assert book.bid_depth(1) == [(98, 4)]
        assert book.order_count() == 4

    def test_order_info_and_order_ids(self):
        book = qp.LimitOrderBook(1000)
        first = book.submit(SELL, 105, 8, 42)
        assert first.order_id == 1000

        explicit = book.submit(SELL, 106, 3, 43, order_id=5000)
        assert explicit.order_id == 5000
        auto = book.submit(SELL, 107, 3, 44)
        assert auto.order_id == 5001  # auto ids never reuse explicit ones

        info = book.order_info(1000)
        assert (info.order_id, info.trader_id, info.price, info.quantity, info.filled) == (
            1000, 42, 105, 8, 0
        )
        assert info.side == SELL
        assert book.order_info(777) is None

        book.submit(BUY, 105, 5, 9)
        assert book.order_info(1000).filled == 5

    def test_invalid_arguments_raise(self):
        book = qp.LimitOrderBook()
        with pytest.raises(ValueError):
            book.submit(BUY, 0, 5)
        with pytest.raises(ValueError):
            book.submit(BUY, 100, 0)
        book.submit(BUY, 100, 5, 1, order_id=10)
        with pytest.raises(ValueError):
            book.submit(SELL, 100, 5, 2, order_id=10)  # duplicate id
        assert book.order_count() == 1

    def test_counters(self):
        book = qp.LimitOrderBook()
        book.submit(BUY, 100, 5, 1)
        book.submit(SELL, 99, 10, 1)  # STP cancel + rest
        book.submit(BUY, 99, 4, 2)    # one fill, maker partially filled
        assert book.submitted_count() == 3
        assert book.fill_count() == 1
        assert book.cancel_count() == 0
        assert book.self_trade_cancel_count() == 1


class TestFillCallback:
    def test_callback_sees_every_fill(self):
        book = qp.LimitOrderBook()
        seen = []
        book.on_fill(lambda f: seen.append((f.price, f.quantity, f.maker_order_id)))
        assert book.has_fill_callback()

        book.submit(BUY, 100, 5, 1)
        book.submit(BUY, 101, 5, 2)
        result = book.submit(SELL, 100, 7, 3)

        assert seen == [(101, 5, 2), (100, 2, 1)]
        assert len(result.fills) == len(seen)

    def test_reentrant_submit_from_callback_is_rejected(self):
        book = qp.LimitOrderBook()
        book.submit(BUY, 100, 5, 1)

        def bad_callback(fill):
            book.submit(SELL, 100, 1, 99)  # must not corrupt the book

        book.on_fill(bad_callback)
        with pytest.raises(RuntimeError):
            book.submit(SELL, 100, 2, 2)
        # The fill that triggered the callback stays applied and consistent.
        assert book.fill_count() == 1
        assert book.order_count() == 1
        assert book.best_bid() == (100, 3)


class TestGilRelease:
    def test_gil_released_during_matching(self):
        """A single C++ submit() call must release the GIL while it matches.

        We rest a large ask wall, then sweep it with one buy order - a single
        long C++ call. A busy Python thread can only make progress during
        that call if the GIL is released.
        """
        book = qp.LimitOrderBook()
        for price in range(100, 100 + 20000):
            book.submit(SELL, price, 5, trader_id=price)  # distinct makers

        stop = threading.Event()
        counter = {"n": 0}

        def spin():
            while not stop.is_set():
                counter["n"] += 1

        spinner = threading.Thread(target=spin)
        spinner.start()
        try:
            time.sleep(0.05)  # let the spinner get scheduled
            before = counter["n"]
            # One long C++ call: sweep the whole ask wall (20000 x 5 = 100000).
            result = book.submit(BUY, 10**6, 100000, trader_id=1)
            after = counter["n"]
        finally:
            stop.set()
            spinner.join()

        assert len(result.fills) == 20000
        assert book.empty()
        assert after - before > 100, (
            "Python thread made no progress during C++ matching - "
            "was the GIL actually released?"
        )


class TestEquivalenceWithPythonReference:
    def _run_stream(self, ops):
        """Replay `ops` on both books; return (cpp, py) observable behaviour."""
        cpp_book = qp.LimitOrderBook()
        py_book = PyLimitOrderBook()

        cpp_events = []
        py_events = []
        for op in ops:
            if op[0] == "new":
                _, side, price, qty, trader = op
                cpp_side = BUY if side == "BUY" else SELL
                cpp_res = cpp_book.submit(cpp_side, price, qty, trader)
                py_res = py_book.submit(side, price, qty, trader)

                cpp_events.append(("fills", cpp_fill_keys(cpp_res)))
                cpp_events.append(("rest", cpp_res.rested, cpp_res.resting_price,
                                   cpp_res.resting_quantity))
                cpp_events.append(("stp", tuple(cpp_res.self_trade_cancels)))
                py_events.append(("fills", [f.key() for f in py_res.fills]))
                py_events.append(("rest", py_res.rested, py_res.resting_price,
                                  py_res.resting_quantity))
                py_events.append(("stp", tuple(py_res.self_trade_cancels)))
            else:
                cpp_events.append(("cancel", cpp_book.cancel(op[1])))
                py_events.append(("cancel", py_book.cancel(op[1])))

        cpp_state = {
            "best_bid": cpp_book.best_bid(),
            "best_ask": cpp_book.best_ask(),
            "spread": cpp_book.spread(),
            "bids": cpp_book.bid_depth(50),
            "asks": cpp_book.ask_depth(50),
            "orders": cpp_book.order_count(),
        }
        py_state = {
            "best_bid": py_book.best_bid(),
            "best_ask": py_book.best_ask(),
            "spread": py_book.spread(),
            "bids": py_book.bid_depth(50),
            "asks": py_book.ask_depth(50),
            "orders": py_book.order_count(),
        }
        return cpp_events, py_events, cpp_state, py_state

    def test_random_streams_are_identical(self):
        """C++ and the Python reference must agree, fill for fill."""
        rng = random.Random(20240601)
        for trial in range(5):
            ops = []
            shadow = PyLimitOrderBook()
            submitted = 0
            for _ in range(600):
                bid, ask = shadow.best_bid(), shadow.best_ask()
                mid = (bid[0] + ask[0]) // 2 if bid and ask else 100_000 + trial
                u = rng.random()
                if u < 0.70:
                    side = rng.choice(("BUY", "SELL"))
                    offset = int(rng.expovariate(1 / 40)) + 1
                    price = mid - offset if side == "BUY" else mid + offset
                    op = ("new", side, price, rng.randint(1, 100), rng.randint(0, 15))
                    shadow.submit(op[1], op[2], op[3], op[4])
                    submitted += 1
                elif u < 0.85:
                    side = rng.choice(("BUY", "SELL"))
                    price = mid + rng.randint(0, 5) if side == "BUY" else mid - rng.randint(0, 5)
                    op = ("new", side, price, rng.randint(1, 50), rng.randint(0, 15))
                    shadow.submit(op[1], op[2], op[3], op[4])
                    submitted += 1
                else:
                    op = ("cancel", rng.randint(1, max(1, submitted)))
                    shadow.cancel(op[1])
                ops.append(op)

            cpp_events, py_events, cpp_state, py_state = self._run_stream(ops)
            assert cpp_events == py_events, f"behaviour diverged in trial {trial}"
            assert cpp_state == py_state, f"final book diverged in trial {trial}"

    def test_handpicked_edge_stream_is_identical(self):
        ops = [
            ("new", "BUY", 100, 10, 1),
            ("new", "BUY", 100, 10, 1),      # same trader, stacked
            ("new", "SELL", 100, 25, 1),     # STP cancels both bids, rests 25
            ("new", "BUY", 100, 5, 2),       # partial fill, rests
            ("cancel", 3),                   # cancel the resting ask
            ("new", "SELL", 99, 7, 3),       # crosses the resting buy
            ("new", "BUY", 100, 3, 3),       # same trader crosses own ask -> STP
            ("new", "SELL", 97, 50, 4),      # sweep everything, rests
            ("cancel", 999),                 # cancel of unknown id
        ]
        cpp_events, py_events, cpp_state, py_state = self._run_stream(ops)
        assert cpp_events == py_events
        assert cpp_state == py_state
        # Sanity: the stream really exercised matching.
        assert any(ev[0] == "fills" and ev[1] for ev in cpp_events)
        assert any(ev[0] == "stp" and ev[1] for ev in cpp_events)
