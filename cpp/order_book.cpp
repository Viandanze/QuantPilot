#include "order_book.hpp"

#include <algorithm>

namespace qp {

namespace {

// RAII guard so `matching_` is cleared even if a fill callback throws.
struct MatchingGuard {
    bool& flag;
    explicit MatchingGuard(bool& f) : flag(f) { flag = true; }
    ~MatchingGuard() { flag = false; }
};

}  // namespace

LimitOrderBook::LimitOrderBook(std::uint64_t first_order_id, FillCallback on_fill)
    : on_fill_(std::move(on_fill)), next_order_id_(first_order_id == 0 ? 1 : first_order_id) {}

SubmitResult LimitOrderBook::submit(Side side, std::int64_t price, std::int64_t quantity,
                                    std::uint64_t trader_id, std::uint64_t order_id) {
    if (price <= 0) {
        throw std::invalid_argument("price must be a positive integer (ticks)");
    }
    if (quantity <= 0) {
        throw std::invalid_argument("quantity must be positive");
    }
    if (matching_) {
        throw std::runtime_error("re-entrant submit from a fill callback is not supported");
    }
    if (order_id == 0) {
        order_id = next_order_id_;
    } else if (index_.find(order_id) != index_.end()) {
        throw std::invalid_argument("order_id is already in use");
    }
    next_order_id_ = std::max(next_order_id_, order_id + 1);
    ++submitted_;

    SubmitResult result;
    result.order_id = order_id;
    std::int64_t remaining = quantity;

    MatchingGuard guard(matching_);
    if (side == Side::Buy) {
        // Consume asks from the lowest price upwards while they cross `price`.
        while (remaining > 0) {
            auto ask_it = asks_.begin();
            if (ask_it == asks_.end() || ask_it->first > price) {
                break;
            }
            const std::int64_t level_price = ask_it->first;
            auto& level = ask_it->second;
            auto maker = level.queue.begin();
            while (maker != level.queue.end() && remaining > 0) {
                if (maker->trader_id == trader_id) {
                    // Self-trade prevention: cancel the resting order.
                    result.self_trade_cancels.push_back(maker->order_id);
                    level.total_quantity -= maker->remaining();
                    index_.erase(maker->order_id);
                    ++stp_cancelled_;
                    maker = level.queue.erase(maker);
                    continue;
                }
                const std::int64_t trade_qty = std::min(remaining, maker->remaining());
                remaining -= trade_qty;
                maker->filled += trade_qty;
                level.total_quantity -= trade_qty;

                Fill fill;
                fill.taker_order_id = order_id;
                fill.maker_order_id = maker->order_id;
                fill.maker_trader_id = maker->trader_id;
                fill.taker_side = side;
                fill.price = level_price;
                fill.quantity = trade_qty;
                fill.taker_remaining = remaining;
                fill.maker_remaining = maker->remaining();
                result.fills.push_back(fill);
                ++fills_;
                if (on_fill_) {
                    on_fill_(fill);
                }
                if (maker->remaining() == 0) {
                    index_.erase(maker->order_id);
                    maker = level.queue.erase(maker);
                } else {
                    break;  // maker still has quantity, so the taker is exhausted
                }
            }
            if (level.queue.empty()) {
                asks_.erase(ask_it);
            }
        }
    } else {
        // Consume bids from the highest price downwards while they cross `price`.
        while (remaining > 0) {
            auto bid_it = bids_.begin();
            if (bid_it == bids_.end() || bid_it->first < price) {
                break;
            }
            const std::int64_t level_price = bid_it->first;
            auto& level = bid_it->second;
            auto maker = level.queue.begin();
            while (maker != level.queue.end() && remaining > 0) {
                if (maker->trader_id == trader_id) {
                    result.self_trade_cancels.push_back(maker->order_id);
                    level.total_quantity -= maker->remaining();
                    index_.erase(maker->order_id);
                    ++stp_cancelled_;
                    maker = level.queue.erase(maker);
                    continue;
                }
                const std::int64_t trade_qty = std::min(remaining, maker->remaining());
                remaining -= trade_qty;
                maker->filled += trade_qty;
                level.total_quantity -= trade_qty;

                Fill fill;
                fill.taker_order_id = order_id;
                fill.maker_order_id = maker->order_id;
                fill.maker_trader_id = maker->trader_id;
                fill.taker_side = side;
                fill.price = level_price;
                fill.quantity = trade_qty;
                fill.taker_remaining = remaining;
                fill.maker_remaining = maker->remaining();
                result.fills.push_back(fill);
                ++fills_;
                if (on_fill_) {
                    on_fill_(fill);
                }
                if (maker->remaining() == 0) {
                    index_.erase(maker->order_id);
                    maker = level.queue.erase(maker);
                } else {
                    break;
                }
            }
            if (level.queue.empty()) {
                bids_.erase(bid_it);
            }
        }
    }

    if (remaining > 0) {
        rest(order_id, trader_id, side, price, quantity, quantity - remaining, result);
    }
    return result;
}

void LimitOrderBook::rest(std::uint64_t order_id, std::uint64_t trader_id, Side side,
                          std::int64_t price, std::int64_t quantity, std::int64_t filled,
                          SubmitResult& result) {
    if (side == Side::Buy) {
        auto& level = bids_[price];
        level.queue.push_back(detail::Order{order_id, trader_id, side, price, quantity, filled});
        level.total_quantity += quantity - filled;
        auto order_it = std::prev(level.queue.end());
        index_[order_id] = detail::OrderRef{true, price, order_it};
    } else {
        auto& level = asks_[price];
        level.queue.push_back(detail::Order{order_id, trader_id, side, price, quantity, filled});
        level.total_quantity += quantity - filled;
        auto order_it = std::prev(level.queue.end());
        index_[order_id] = detail::OrderRef{false, price, order_it};
    }
    result.rested = true;
    result.resting_price = price;
    result.resting_quantity = quantity - filled;
}

bool LimitOrderBook::cancel(std::uint64_t order_id) {
    if (matching_) {
        throw std::runtime_error("re-entrant cancel from a fill callback is not supported");
    }
    auto it = index_.find(order_id);
    if (it == index_.end()) {
        return false;
    }
    if (!remove_order(it)) {
        return false;
    }
    ++cancelled_;
    return true;
}

bool LimitOrderBook::remove_order(
    std::unordered_map<std::uint64_t, detail::OrderRef>::iterator it) {
    const detail::OrderRef& ref = it->second;
    if (ref.is_bid) {
        auto level_it = bids_.find(ref.price);
        if (level_it == bids_.end()) {
            return false;  // defensive: index and book must agree
        }
        level_it->second.total_quantity -= ref.order->remaining();
        level_it->second.queue.erase(ref.order);
        if (level_it->second.queue.empty()) {
            bids_.erase(level_it);
        }
    } else {
        auto level_it = asks_.find(ref.price);
        if (level_it == asks_.end()) {
            return false;
        }
        level_it->second.total_quantity -= ref.order->remaining();
        level_it->second.queue.erase(ref.order);
        if (level_it->second.queue.empty()) {
            asks_.erase(level_it);
        }
    }
    index_.erase(it);
    return true;
}

std::optional<OrderInfo> LimitOrderBook::order_info(std::uint64_t order_id) const {
    auto it = index_.find(order_id);
    if (it == index_.end()) {
        return std::nullopt;
    }
    const detail::Order& order = *it->second.order;
    OrderInfo info;
    info.order_id = order.order_id;
    info.trader_id = order.trader_id;
    info.side = order.side;
    info.price = order.price;
    info.quantity = order.quantity;
    info.filled = order.filled;
    return info;
}

std::optional<std::pair<std::int64_t, std::int64_t>> LimitOrderBook::best_bid() const {
    if (bids_.empty()) {
        return std::nullopt;
    }
    const auto& level = bids_.begin()->second;
    return std::make_pair(bids_.begin()->first, level.total_quantity);
}

std::optional<std::pair<std::int64_t, std::int64_t>> LimitOrderBook::best_ask() const {
    if (asks_.empty()) {
        return std::nullopt;
    }
    const auto& level = asks_.begin()->second;
    return std::make_pair(asks_.begin()->first, level.total_quantity);
}

std::int64_t LimitOrderBook::bid_qty_at(std::int64_t price) const {
    auto it = bids_.find(price);
    return it == bids_.end() ? 0 : it->second.total_quantity;
}

std::int64_t LimitOrderBook::ask_qty_at(std::int64_t price) const {
    auto it = asks_.find(price);
    return it == asks_.end() ? 0 : it->second.total_quantity;
}

std::optional<std::int64_t> LimitOrderBook::spread() const {
    if (bids_.empty() || asks_.empty()) {
        return std::nullopt;
    }
    return asks_.begin()->first - bids_.begin()->first;
}

std::vector<std::pair<std::int64_t, std::int64_t>> LimitOrderBook::bid_depth(
    std::size_t levels) const {
    std::vector<std::pair<std::int64_t, std::int64_t>> out;
    out.reserve(std::min(levels, bids_.size()));
    for (const auto& [price, level] : bids_) {
        if (out.size() >= levels) {
            break;
        }
        out.emplace_back(price, level.total_quantity);
    }
    return out;
}

std::vector<std::pair<std::int64_t, std::int64_t>> LimitOrderBook::ask_depth(
    std::size_t levels) const {
    std::vector<std::pair<std::int64_t, std::int64_t>> out;
    out.reserve(std::min(levels, asks_.size()));
    for (const auto& [price, level] : asks_) {
        if (out.size() >= levels) {
            break;
        }
        out.emplace_back(price, level.total_quantity);
    }
    return out;
}

}  // namespace qp
