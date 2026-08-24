// Native unit tests for the QuantPilot limit order book.
//
// Lightweight assert harness: no external test framework, just CHECK macros
// and a non-zero exit code on failure. Build and run:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build --target qp_orderbook_test
//   ./build/qp_orderbook_test
#include "order_book.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                                \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

using qp::Fill;
using qp::LimitOrderBook;
using qp::OrderInfo;
using qp::Side;
using qp::SubmitResult;

std::string fill_key(const Fill& f) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%llu/%llu/%lld/%lld/%lld/%lld",
                  static_cast<unsigned long long>(f.maker_order_id),
                  static_cast<unsigned long long>(f.taker_order_id),
                  f.price, f.quantity, f.taker_remaining, f.maker_remaining);
    return std::string(buf);
}

// Invariant: after every public operation the book is never crossed
// (best ask strictly above best bid) and level totals are consistent.
void check_invariants(const LimitOrderBook& book, const char* where) {
    const auto bid = book.best_bid();
    const auto ask = book.best_ask();
    if (bid && ask) {
        if (!(ask->first > bid->first)) {
            std::fprintf(stderr, "crossed book at %s: bid %lld ask %lld\n", where,
                         static_cast<long long>(bid->first),
                         static_cast<long long>(ask->first));
            ++g_failures;
        }
    }
}

void test_price_time_priority() {
    LimitOrderBook book;
    // Two bids at the same price; the older one must fill first.
    SubmitResult r1 = book.submit(Side::Buy, 100, 5, 1);
    SubmitResult r2 = book.submit(Side::Buy, 100, 5, 2);
    CHECK(r1.rested && r2.rested);

    SubmitResult sell = book.submit(Side::Sell, 100, 7, 3);
    CHECK(sell.fills.size() == 2);
    CHECK(sell.fills[0].maker_order_id == 1);   // oldest first
    CHECK(sell.fills[0].quantity == 5);
    CHECK(sell.fills[0].maker_remaining == 0);
    CHECK(sell.fills[0].taker_remaining == 2);
    CHECK(sell.fills[1].maker_order_id == 2);   // then the younger one
    CHECK(sell.fills[1].quantity == 2);
    CHECK(sell.fills[1].maker_remaining == 3);
    CHECK(!sell.rested);
    CHECK(!book.has_order(1));
    CHECK(book.has_order(2));
    CHECK(book.best_bid().has_value());              // order 2 rests with 3 left
    CHECK(book.best_bid()->first == 100);
    CHECK(book.best_bid()->second == 3);
    CHECK(book.order_info(2)->filled == 2);
    check_invariants(book, "price_time_priority");
}

void test_partial_fill_and_rest() {
    LimitOrderBook book;
    book.submit(Side::Buy, 100, 10, 1);

    SubmitResult s1 = book.submit(Side::Sell, 100, 4, 2);
    CHECK(s1.fills.size() == 1);
    CHECK(s1.fills[0].quantity == 4);
    CHECK(s1.fills[0].maker_remaining == 6);  // maker partially filled
    CHECK(s1.fills[0].taker_remaining == 0);
    CHECK(!s1.rested);

    // The taker now needs more liquidity than is left at 100.
    SubmitResult s2 = book.submit(Side::Sell, 100, 15, 3);
    CHECK(s2.fills.size() == 1);
    CHECK(s2.fills[0].quantity == 6);
    CHECK(s2.fills[0].maker_remaining == 0);
    CHECK(s2.fills[0].taker_remaining == 9);
    CHECK(s2.rested);
    CHECK(s2.resting_price == 100);
    CHECK(s2.resting_quantity == 9);
    CHECK(book.best_ask().has_value());
    CHECK(book.best_ask()->first == 100);
    CHECK(book.best_ask()->second == 9);
    check_invariants(book, "partial_fill_and_rest");
}

void test_crossing_sweep_multiple_levels() {
    LimitOrderBook book;
    book.submit(Side::Buy, 98, 3, 1);
    book.submit(Side::Buy, 99, 4, 2);
    book.submit(Side::Buy, 100, 5, 3);

    SubmitResult sell = book.submit(Side::Sell, 98, 9, 4);
    CHECK(sell.fills.size() == 2);
    // Best price first: the bid at 100, then the bid at 99.
    CHECK(sell.fills[0].price == 100);
    CHECK(sell.fills[0].quantity == 5);
    CHECK(sell.fills[1].price == 99);
    CHECK(sell.fills[1].quantity == 4);
    CHECK(!sell.rested);
    // Only the bid at 98 survives.
    CHECK(book.best_bid().has_value());
    CHECK(book.best_bid()->first == 98);
    CHECK(book.best_bid()->second == 3);
    CHECK(book.bid_level_count() == 1);
    check_invariants(book, "crossing_sweep");
}

void test_self_trade_prevention() {
    LimitOrderBook book;
    SubmitResult resting = book.submit(Side::Buy, 100, 5, 7);
    CHECK(resting.rested);
    const std::uint64_t resting_id = resting.order_id;
    // An unrelated resting order at the same level must be unaffected.
    book.submit(Side::Buy, 100, 11, 9);
    CHECK(book.bid_qty_at(100) == 16);

    // Same trader crosses their own resting order: the resting order is
    // cancelled, matching continues with the next maker.
    SubmitResult sell = book.submit(Side::Sell, 99, 10, 7);
    CHECK(sell.fills.size() == 1);
    CHECK(sell.fills[0].maker_order_id == 2);   // the other trader's order
    CHECK(sell.fills[0].price == 100);
    CHECK(sell.fills[0].quantity == 10);
    CHECK(sell.fills[0].taker_remaining == 0);
    CHECK(sell.fills[0].maker_remaining == 1);
    CHECK(sell.self_trade_cancels.size() == 1);
    CHECK(sell.self_trade_cancels[0] == resting_id);
    CHECK(!sell.rested);
    CHECK(!book.has_order(resting_id));
    CHECK(book.best_bid()->first == 100);
    CHECK(book.best_bid()->second == 1);        // level total drops by the STP qty
    CHECK(book.bid_qty_at(100) == 1);
    CHECK(book.best_ask() == std::nullopt);
    CHECK(book.self_trade_cancel_count() == 1);
    check_invariants(book, "self_trade_prevention");

    // A different trader CAN take the remaining liquidity.
    SubmitResult sell2 = book.submit(Side::Sell, 100, 5, 8);
    CHECK(sell2.fills.size() == 1);
    CHECK(sell2.fills[0].maker_order_id == 2);
    CHECK(sell2.fills[0].quantity == 1);
    CHECK(sell2.fills[0].taker_remaining == 4);
    CHECK(sell2.rested);
    CHECK(book.best_ask()->first == 100);
    CHECK(book.best_ask()->second == 4);
    CHECK(book.order_count() == 1);
}

void test_cancel() {
    LimitOrderBook book;
    SubmitResult a = book.submit(Side::Buy, 100, 5, 1);
    SubmitResult b = book.submit(Side::Buy, 100, 7, 2);
    SubmitResult c = book.submit(Side::Buy, 99, 3, 3);

    CHECK(book.best_bid()->first == 100);
    CHECK(book.best_bid()->second == 12);

    CHECK(book.cancel(b.order_id));           // middle order at best level
    CHECK(book.best_bid()->second == 5);
    CHECK(book.bid_qty_at(100) == 5);
    CHECK(!book.cancel(b.order_id));          // double cancel fails
    CHECK(!book.cancel(999999));              // unknown id fails

    CHECK(book.cancel(a.order_id));           // empties the 100 level
    CHECK(book.bid_qty_at(100) == 0);
    CHECK(book.best_bid()->first == 99);
    CHECK(book.best_bid()->second == 3);
    CHECK(book.bid_level_count() == 1);

    CHECK(book.cancel(c.order_id));
    CHECK(book.empty());
    CHECK(book.order_count() == 0);
    check_invariants(book, "cancel");
}

void test_book_queries() {
    LimitOrderBook book;
    CHECK(book.empty());
    CHECK(book.best_bid() == std::nullopt);
    CHECK(book.best_ask() == std::nullopt);
    CHECK(book.spread() == std::nullopt);

    book.submit(Side::Buy, 98, 4, 1);
    book.submit(Side::Buy, 97, 6, 2);
    book.submit(Side::Sell, 101, 5, 3);
    book.submit(Side::Sell, 102, 2, 4);

    CHECK(book.best_bid()->first == 98);
    CHECK(book.best_bid()->second == 4);
    CHECK(book.best_ask()->first == 101);
    CHECK(book.best_ask()->second == 5);
    CHECK(book.spread().has_value());
    CHECK(*book.spread() == 3);
    CHECK(book.bid_qty_at(97) == 6);
    CHECK(book.ask_qty_at(102) == 2);
    CHECK(book.bid_qty_at(105) == 0);

    auto bids = book.bid_depth(5);
    auto asks = book.ask_depth(5);
    CHECK(bids.size() == 2);
    CHECK(bids[0].first == 98 && bids[0].second == 4);   // best first
    CHECK(bids[1].first == 97 && bids[1].second == 6);
    CHECK(asks.size() == 2);
    CHECK(asks[0].first == 101 && asks[0].second == 5);
    CHECK(asks[1].first == 102 && asks[1].second == 2);

    CHECK(bids.size() == book.bid_depth(1).size() + 1);  // level capping
    CHECK(book.order_count() == 4);
    check_invariants(book, "queries");
}

void test_order_info_and_ids() {
    LimitOrderBook book(1000);
    SubmitResult r = book.submit(Side::Sell, 105, 8, 42);
    CHECK(r.order_id == 1000);              // explicit first id honoured

    SubmitResult r2 = book.submit(Side::Sell, 106, 3, 43, 5000);  // explicit id
    CHECK(r2.order_id == 5000);
    SubmitResult r3 = book.submit(Side::Sell, 107, 3, 44);
    CHECK(r3.order_id == 5001);             // auto ids never reuse explicit ones

    std::optional<OrderInfo> info = book.order_info(1000);
    CHECK(info.has_value());
    CHECK(info->order_id == 1000);
    CHECK(info->trader_id == 42);
    CHECK(info->side == Side::Sell);
    CHECK(info->price == 105);
    CHECK(info->quantity == 8);
    CHECK(info->filled == 0);
    CHECK(book.order_info(777) == std::nullopt);

    // Partial fill updates the snapshot.
    book.submit(Side::Buy, 105, 5, 9);
    info = book.order_info(1000);
    CHECK(info->filled == 5);
    check_invariants(book, "order_info");
}

void test_invalid_input() {
    LimitOrderBook book;
    bool threw = false;
    try {
        book.submit(Side::Buy, 0, 5, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        book.submit(Side::Buy, 100, 0, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    book.submit(Side::Buy, 100, 5, 1, 10);
    threw = false;
    try {
        book.submit(Side::Sell, 100, 5, 2, 10);  // duplicate id
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(book.order_count() == 1);
}

void test_fill_callback() {
    std::vector<Fill> seen;
    LimitOrderBook book(1, [&seen](const Fill& f) { seen.push_back(f); });
    book.submit(Side::Buy, 100, 5, 1);
    book.submit(Side::Buy, 101, 5, 2);
    book.submit(Side::Sell, 100, 7, 3);

    CHECK(seen.size() == 2);
    CHECK(seen[0].price == 101 && seen[0].quantity == 5);
    CHECK(seen[1].price == 100 && seen[1].quantity == 2);
    CHECK(book.fill_count() == 2);
    CHECK(book.submitted_count() == 3);
    CHECK(book.cancel_count() == 0);
    check_invariants(book, "fill_callback");
}

void test_gfd_resting_semantics() {
    LimitOrderBook book;
    // A non-crossing GFD order simply rests at its own price.
    SubmitResult r = book.submit(Side::Buy, 90, 25, 1);
    CHECK(r.fills.empty());
    CHECK(r.rested);
    CHECK(r.resting_price == 90);
    CHECK(r.resting_quantity == 25);

    // Resting orders survive unrelated activity at other prices.
    book.submit(Side::Sell, 95, 10, 2);
    CHECK(book.has_order(r.order_id));
    CHECK(book.order_info(r.order_id)->quantity == 25);
    check_invariants(book, "gfd_resting");
}

}  // namespace

int main() {
    test_price_time_priority();
    test_partial_fill_and_rest();
    test_crossing_sweep_multiple_levels();
    test_self_trade_prevention();
    test_cancel();
    test_book_queries();
    test_order_info_and_ids();
    test_invalid_input();
    test_fill_callback();
    test_gfd_resting_semantics();

    if (g_failures == 0) {
        std::printf("qp_orderbook_test: all %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("qp_orderbook_test: %d of %d checks FAILED\n", g_failures, g_checks);
    return 1;
}
