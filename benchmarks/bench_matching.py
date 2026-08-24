"""Single-core matching throughput benchmark: C++ core vs pure Python.

The same deterministic order stream is replayed against:

1. the C++17 core through the ``qp_matching`` pybind11 module (this measures
   end-to-end throughput including the Python call and result-conversion
   overhead, with the GIL released during matching);
2. the pure Python reference book in ``benchmarks/py_limit_book.py``.

The stream is adaptive and realistic: ~70% resting limit orders placed a few
ticks off the mid, ~15% marketable orders crossing by 0-5 ticks, ~15%
cancels, quantities 1-100, 64 distinct trader ids (so self-trade prevention
is exercised at a realistic rate). Prices are integer ticks. One warmup
round, then several timed rounds on a fresh book per round; medians are
reported.

Usage:
    python benchmarks/bench_matching.py [--ops N] [--rounds R] [--seed S]
"""
from __future__ import annotations

import argparse
import pathlib
import random
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    import qp_matching as qpm
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "qp_matching is not built. See the 'Matching engine' section of the "
        "README:\n  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "
        "-Dpybind11_DIR=\"$(python -m pybind11 --cmakedir)\"\n  cmake --build build"
    ) from exc

from py_limit_book import PyLimitOrderBook  # noqa: E402


def generate_stream(n_ops: int, seed: int) -> list[tuple]:
    """Build a realistic op stream, adapting prices to a shadow book's quotes."""
    rng = random.Random(seed)
    shadow = PyLimitOrderBook()
    ops: list[tuple] = []
    submitted = 0
    mid = 100_000
    for _ in range(n_ops):
        bid, ask = shadow.best_bid(), shadow.best_ask()
        if bid and ask:
            mid = (bid[0] + ask[0]) // 2
        u = rng.random()
        if u < 0.70:
            # Resting limit order a few ticks off the mid.
            side = rng.choice(("BUY", "SELL"))
            offset = int(rng.expovariate(1 / 50)) + 1
            price = mid - offset if side == "BUY" else mid + offset
            op = ("new", side, price, rng.randint(1, 100), rng.randint(0, 63))
            shadow.submit(op[1], op[2], op[3], op[4])
            submitted += 1
        elif u < 0.85:
            # Marketable limit order crossing by 0-5 ticks.
            side = rng.choice(("BUY", "SELL"))
            price = mid + rng.randint(0, 5) if side == "BUY" else mid - rng.randint(0, 5)
            op = ("new", side, price, rng.randint(1, 50), rng.randint(0, 63))
            shadow.submit(op[1], op[2], op[3], op[4])
            submitted += 1
        else:
            # Cancel a previously submitted order (may already be gone).
            op = ("cancel", rng.randint(1, max(1, submitted)))
            shadow.cancel(op[1])
        ops.append(op)
    return ops


def replay_cpp(ops: list[tuple]) -> tuple[int, int]:
    """Replay the stream on the C++ book; return (ops, fills)."""
    book = qpm.LimitOrderBook()
    submit = book.submit
    cancel = book.cancel
    buy, sell = qpm.Side.BUY, qpm.Side.SELL
    fills = 0
    for op in ops:
        if op[0] == "new":
            _, side, price, qty, trader = op
            fills += len(submit(buy if side == "BUY" else sell, price, qty, trader).fills)
        else:
            cancel(op[1])
    return len(ops), fills


def replay_python(ops: list[tuple]) -> tuple[int, int]:
    """Replay the stream on the pure Python book; return (ops, fills)."""
    book = PyLimitOrderBook()
    submit = book.submit
    cancel = book.cancel
    fills = 0
    for op in ops:
        if op[0] == "new":
            _, side, price, qty, trader = op
            fills += len(submit(side, price, qty, trader).fills)
        else:
            cancel(op[1])
    return len(ops), fills


def cross_check(ops: list[tuple]) -> int:
    """Replay on both books and assert identical behaviour; returns fills."""
    cpp_book = qpm.LimitOrderBook()
    py_book = PyLimitOrderBook()
    buy, sell = qpm.Side.BUY, qpm.Side.SELL
    for op in ops:
        if op[0] == "new":
            _, side, price, qty, trader = op
            cr = cpp_book.submit(buy if side == "BUY" else sell, price, qty, trader)
            pr = py_book.submit(side, price, qty, trader)
            assert len(cr.fills) == len(pr.fills), f"fill count diverged on {op}"
            for cf, pf in zip(cr.fills, pr.fills):
                assert (cf.maker_order_id, cf.price, cf.quantity,
                        cf.taker_remaining, cf.maker_remaining) == (
                    pf.maker_order_id, pf.price, pf.quantity,
                    pf.taker_remaining, pf.maker_remaining), f"fill diverged on {op}"
            assert cr.rested == pr.rested
            assert list(cr.self_trade_cancels) == list(pr.self_trade_cancels)
        else:
            assert cpp_book.cancel(op[1]) == py_book.cancel(op[1])
    assert cpp_book.best_bid() == py_book.best_bid()
    assert cpp_book.best_ask() == py_book.best_ask()
    return cpp_book.fill_count()


def timed_rounds(replay, ops: list[tuple], rounds: int) -> list[tuple[float, int, int]]:
    stats = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        n_ops, n_fills = replay(ops)
        stats.append((time.perf_counter() - t0, n_ops, n_fills))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=int, default=100_000,
                        help="order stream size per round (default 100000)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="timed rounds per implementation (default 5)")
    parser.add_argument("--seed", type=int, default=20240601, help="stream seed")
    args = parser.parse_args()

    print(f"qp_matching module: {qpm.__file__}")
    print(f"stream: {args.ops} ops/round, {args.rounds} timed rounds, seed {args.seed}")

    ops = generate_stream(args.ops, args.seed)
    total_fills = cross_check(ops)
    print(f"cross-check vs pure Python reference: OK "
          f"({len(ops)} ops, {total_fills} fills, identical behaviour)\n")

    # Warmup (JIT-free, but warms caches/allocators).
    replay_cpp(generate_stream(max(1000, args.ops // 10), args.seed + 1))
    replay_python(generate_stream(max(1000, args.ops // 10), args.seed + 1))

    cpp_stats = timed_rounds(replay_cpp, ops, args.rounds)
    py_stats = timed_rounds(replay_python, ops, args.rounds)

    def summarise(name: str, stats: list[tuple[float, int, int]]) -> tuple[float, float]:
        med_time = statistics.median(s[0] for s in stats)
        ops_rate = statistics.median(s[1] / s[0] for s in stats)
        fill_rate = statistics.median(s[2] / s[0] for s in stats)
        print(f"{name}:")
        for i, (seconds, n_ops, n_fills) in enumerate(stats, 1):
            print(f"  round {i}: {seconds:8.3f} s  {n_ops / seconds:12,.0f} orders/s"
                  f"  {n_fills / seconds:12,.0f} fills/s")
        print(f"  median : {med_time:8.3f} s  {ops_rate:12,.0f} orders/s"
              f"  {fill_rate:12,.0f} fills/s\n")
        return ops_rate, fill_rate

    cpp_ops, cpp_fills = summarise("C++17 core via qp_matching (pybind11, GIL released)",
                                   cpp_stats)
    py_ops, py_fills = summarise("pure Python reference order book", py_stats)

    print(f"speedup, C++ vs pure Python (orders/sec): {cpp_ops / py_ops:.1f}x")
    print(f"speedup, C++ vs pure Python (fills/sec) : {cpp_fills / py_fills:.1f}x")


if __name__ == "__main__":
    main()
