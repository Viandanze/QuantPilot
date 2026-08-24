// QuantPilot matching core: a deterministic, single-threaded limit order book.
//
// Design notes
// ------------
// * Prices and quantities are signed 64-bit integers (ticks / lots). No
//   floating point anywhere in the matching path, so two runs of the same
//   order stream always produce bit-identical fill sequences.
// * Price-time priority: each price level holds a FIFO queue of resting
//   orders; takers consume liquidity best-price-first, oldest-first.
// * Self-trade prevention (STP): an incoming order never trades against a
//   resting order of the same trader. Policy is "cancel resting" - the older
//   resting order is cancelled and matching continues with the next maker.
// * Complexity: submit is O(log P) per price level touched (P = number of
//   live levels) plus O(1) per fill; cancel is O(log P) via the order index.
#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qp {

enum class Side : std::int8_t { Buy = 0, Sell = 1 };

inline constexpr Side opposite(Side side) {
    return side == Side::Buy ? Side::Sell : Side::Buy;
}

// One trade report produced by the matching engine.
struct Fill {
    std::uint64_t taker_order_id = 0;
    std::uint64_t maker_order_id = 0;
    std::uint64_t maker_trader_id = 0;
    Side taker_side = Side::Buy;
    std::int64_t price = 0;          // maker's resting price, in ticks
    std::int64_t quantity = 0;       // quantity traded in this fill
    std::int64_t taker_remaining = 0;  // taker quantity still unfilled after this fill
    std::int64_t maker_remaining = 0;  // maker quantity still unfilled after this fill
};

// Snapshot of a resting order.
struct OrderInfo {
    std::uint64_t order_id = 0;
    std::uint64_t trader_id = 0;
    Side side = Side::Buy;
    std::int64_t price = 0;
    std::int64_t quantity = 0;  // original quantity
    std::int64_t filled = 0;    // executed quantity so far
};

// Result of submitting one order.
struct SubmitResult {
    std::uint64_t order_id = 0;
    std::vector<Fill> fills;                       // in execution order
    std::vector<std::uint64_t> self_trade_cancels; // resting orders cancelled by STP
    bool rested = false;                           // true if any quantity is resting
    std::int64_t resting_price = 0;
    std::int64_t resting_quantity = 0;             // remaining quantity at rest
};

namespace detail {

struct Order {
    std::uint64_t order_id;
    std::uint64_t trader_id;
    Side side;
    std::int64_t price;
    std::int64_t quantity;
    std::int64_t filled = 0;

    std::int64_t remaining() const { return quantity - filled; }
};

struct Level {
    std::list<Order> queue;        // FIFO: front is the oldest order
    std::int64_t total_quantity = 0;
};

// Locates a resting order for O(log P) cancel. The list iterator stays valid
// until the order is erased; the price is looked up in the side map again,
// which is safe because a level only disappears once its queue is empty.
struct OrderRef {
    bool is_bid;
    std::int64_t price;
    std::list<Order>::iterator order;
};

}  // namespace detail

class LimitOrderBook {
public:
    using FillCallback = std::function<void(const Fill&)>;
    // Bids: highest price first. Asks: lowest price first.
    using BidMap = std::map<std::int64_t, detail::Level, std::greater<std::int64_t>>;
    using AskMap = std::map<std::int64_t, detail::Level, std::less<std::int64_t>>;

    explicit LimitOrderBook(std::uint64_t first_order_id = 1,
                            FillCallback on_fill = nullptr);

    // Submit a good-for-day limit order. If order_id is 0 the next automatic
    // id is assigned; otherwise the caller-supplied id must not be in use.
    // Throws std::invalid_argument on bad input.
    SubmitResult submit(Side side, std::int64_t price, std::int64_t quantity,
                        std::uint64_t trader_id = 0, std::uint64_t order_id = 0);

    // Cancel a resting order. Returns false for unknown / already filled ids.
    bool cancel(std::uint64_t order_id);

    void set_fill_callback(FillCallback on_fill) { on_fill_ = std::move(on_fill); }
    bool has_fill_callback() const { return static_cast<bool>(on_fill_); }

    // ---- book state queries -------------------------------------------------
    bool empty() const { return index_.empty(); }
    std::size_t order_count() const { return index_.size(); }
    bool has_order(std::uint64_t order_id) const {
        return index_.find(order_id) != index_.end();
    }
    std::optional<OrderInfo> order_info(std::uint64_t order_id) const;

    // Best bid/ask as (price, total quantity at that price); nullopt if empty.
    std::optional<std::pair<std::int64_t, std::int64_t>> best_bid() const;
    std::optional<std::pair<std::int64_t, std::int64_t>> best_ask() const;
    std::int64_t bid_qty_at(std::int64_t price) const;
    std::int64_t ask_qty_at(std::int64_t price) const;
    std::optional<std::int64_t> spread() const;

    // Top `levels` price levels per side, best first, as (price, quantity).
    std::vector<std::pair<std::int64_t, std::int64_t>> bid_depth(std::size_t levels) const;
    std::vector<std::pair<std::int64_t, std::int64_t>> ask_depth(std::size_t levels) const;
    std::size_t bid_level_count() const { return bids_.size(); }
    std::size_t ask_level_count() const { return asks_.size(); }

    // ---- counters (useful for benchmarks and tests) -------------------------
    std::uint64_t submitted_count() const { return submitted_; }
    std::uint64_t fill_count() const { return fills_; }
    std::uint64_t cancel_count() const { return cancelled_; }
    std::uint64_t self_trade_cancel_count() const { return stp_cancelled_; }

private:
    void rest(std::uint64_t order_id, std::uint64_t trader_id, Side side,
              std::int64_t price, std::int64_t quantity, std::int64_t filled,
              SubmitResult& result);
    bool remove_order(typename std::unordered_map<std::uint64_t, detail::OrderRef>::iterator it);

    BidMap bids_;
    AskMap asks_;
    std::unordered_map<std::uint64_t, detail::OrderRef> index_;
    FillCallback on_fill_;
    std::uint64_t next_order_id_;
    std::uint64_t submitted_ = 0;
    std::uint64_t fills_ = 0;
    std::uint64_t cancelled_ = 0;
    std::uint64_t stp_cancelled_ = 0;
    bool matching_ = false;  // guards against re-entrant calls from callbacks
};

}  // namespace qp
