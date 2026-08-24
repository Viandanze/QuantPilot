// Single-core throughput micro-benchmark for the QuantPilot limit order book.
//
// This measures the C++ core with no Python in the loop: a synthetic but
// realistic order stream (adaptive prices around the live best quotes, mixed
// resting/marketable limit orders, cancels, self-trade-prevention hits) is
// replayed against a fresh book per round. One warmup round, then several
// timed rounds; the median rates are reported.
//
//   ./build/qp_orderbook_bench [rounds] [ops_per_round]
#include "order_book.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using qp::LimitOrderBook;
using qp::Side;

struct Op {
    enum Kind { NewOrder, CancelOrder } kind;
    Side side;
    long long price;
    long long quantity;
    unsigned long long trader_id;
    unsigned long long order_id;
};

// Mirrors the workload generator in benchmarks/bench_matching.py: the same
// op mix, adapted live against the book's own best quotes (the mid follows
// the quote midpoint while both sides exist, otherwise it stays put).
std::vector<Op> generate_stream(LimitOrderBook& book, int n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::exponential_distribution<double> expo(1.0 / 50.0);
    std::uniform_int_distribution<int> qty1(1, 100);
    std::uniform_int_distribution<int> qty2(1, 50);
    std::uniform_int_distribution<int> cross(0, 5);
    std::uniform_int_distribution<int> trader(0, 63);

    std::vector<Op> ops;
    ops.reserve(static_cast<std::size_t>(n));
    long long mid = 100000;
    unsigned long long submitted = 0;
    for (int i = 0; i < n; ++i) {
        const double u = uni(rng);
        const auto bid = book.best_bid();
        const auto ask = book.best_ask();
        if (bid && ask) {
            mid = (bid->first + ask->first) / 2;
        }
        if (u < 0.70) {
            // Resting limit order a few ticks off the mid.
            const bool is_buy = uni(rng) < 0.5;
            const long long offset = static_cast<long long>(expo(rng)) + 1;
            const long long price = is_buy ? mid - offset : mid + offset;
            ops.push_back(Op{Op::NewOrder, is_buy ? Side::Buy : Side::Sell, price,
                             qty1(rng), static_cast<unsigned long long>(trader(rng)), 0});
            ++submitted;
        } else if (u < 0.85) {
            // Marketable limit order that crosses by 0-5 ticks.
            const bool is_buy = uni(rng) < 0.5;
            const long long price = is_buy ? mid + cross(rng) : mid - cross(rng);
            ops.push_back(Op{Op::NewOrder, is_buy ? Side::Buy : Side::Sell, price,
                             qty2(rng), static_cast<unsigned long long>(trader(rng)), 0});
            ++submitted;
        } else {
            // Cancel a previously submitted order id (may already be gone).
            std::uniform_int_distribution<unsigned long long> pick(1, submitted);
            ops.push_back(Op{Op::CancelOrder, Side::Buy, 0, 0, 0, pick(rng)});
        }
    }
    return ops;
}

struct RoundStats {
    double seconds;
    unsigned long long ops;
    unsigned long long fills;
};

RoundStats run_round(int n, std::uint64_t seed) {
    LimitOrderBook book;
    const std::vector<Op> ops = generate_stream(book, n, seed);

    const auto t0 = std::chrono::steady_clock::now();
    unsigned long long fills = 0;
    for (const Op& op : ops) {
        if (op.kind == Op::NewOrder) {
            fills += book.submit(op.side, op.price, op.quantity, op.trader_id).fills.size();
        } else {
            book.cancel(op.order_id);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    return RoundStats{std::chrono::duration<double>(t1 - t0).count(),
                      static_cast<unsigned long long>(ops.size()), fills};
}

}  // namespace

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 5;
    const int ops_per_round = argc > 2 ? std::atoi(argv[2]) : 200000;

    run_round(ops_per_round / 10, 1);  // warmup

    std::vector<RoundStats> stats;
    for (int r = 0; r < rounds; ++r) {
        stats.push_back(run_round(ops_per_round, 100 + r));
    }

    std::vector<double> ops_rates;
    std::vector<double> fill_rates;
    for (const RoundStats& s : stats) {
        ops_rates.push_back(static_cast<double>(s.ops) / s.seconds);
        fill_rates.push_back(static_cast<double>(s.fills) / s.seconds);
    }
    std::sort(ops_rates.begin(), ops_rates.end());
    std::sort(fill_rates.begin(), fill_rates.end());

    const double median_ops = ops_rates[ops_rates.size() / 2];
    const double median_fills = fill_rates[fill_rates.size() / 2];
    unsigned long long total_fills = 0;
    for (const RoundStats& s : stats) {
        total_fills += s.fills;
    }

    std::printf("qp_orderbook_bench: %d rounds x %d ops\n", rounds, ops_per_round);
    for (std::size_t i = 0; i < stats.size(); ++i) {
        std::printf("  round %zu: %.3f s, %llu ops, %llu fills (%.0f ops/s)\n", i + 1,
                    stats[i].seconds, stats[i].ops, stats[i].fills,
                    static_cast<double>(stats[i].ops) / stats[i].seconds);
    }
    std::printf("median: %.0f orders/sec, %.0f fills/sec (%llu total fills)\n",
                median_ops, median_fills, total_fills);
    return 0;
}
