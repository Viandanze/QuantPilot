// pybind11 bindings for the QuantPilot matching core (module: qp_matching).
//
// The Python surface mirrors the C++ API:
//   book = qp_matching.LimitOrderBook()
//   result = book.submit(qp_matching.Side.BUY, price, quantity,
//                        trader_id=0, order_id=None)
//   result.fills / result.rested / result.resting_price / ...
//   book.cancel(order_id)
//   book.best_bid() / best_ask() / spread() / bid_qty_at(...) / depth(...)
//
// GIL policy: while a Python fill callback is registered the GIL is held for
// the duration of submit() (the callback needs it). Without a callback the
// GIL is released for the whole matching computation, so other Python
// threads keep running while the C++ core works.
#include "order_book.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

namespace py = pybind11;
using qp::Fill;
using qp::LimitOrderBook;
using qp::OrderInfo;
using qp::Side;
using qp::SubmitResult;

namespace {

// submit() wrapper that releases the GIL around the C++ call when no Python
// fill callback is registered. Result conversion happens with the GIL back.
// `order_id` accepts None (auto-assign) or an explicit positive id.
SubmitResult submit_py(LimitOrderBook& book, Side side, std::int64_t price,
                       std::int64_t quantity, std::uint64_t trader_id,
                       py::object order_id_obj) {
    std::uint64_t order_id = 0;
    if (!order_id_obj.is_none()) {
        order_id = order_id_obj.cast<std::uint64_t>();
    }
    if (book.has_fill_callback()) {
        return book.submit(side, price, quantity, trader_id, order_id);
    }
    py::gil_scoped_release nogil;
    return book.submit(side, price, quantity, trader_id, order_id);
}

// cancel() wrapper: same GIL policy as submit().
bool cancel_py(LimitOrderBook& book, std::uint64_t order_id) {
    if (book.has_fill_callback()) {
        return book.cancel(order_id);
    }
    py::gil_scoped_release nogil;
    return book.cancel(order_id);
}

// Optional trader id default.
constexpr std::uint64_t k_default_trader = 0;

}  // namespace

PYBIND11_MODULE(qp_matching, m) {
    m.doc() = "QuantPilot C++17 limit order book (price-time priority, "
              "self-trade prevention, deterministic single-threaded matching).";

    py::enum_<Side>(m, "Side")
        .value("BUY", Side::Buy)
        .value("SELL", Side::Sell)
        .def(py::init([](int v) {
            if (v == 0) return Side::Buy;
            if (v == 1) return Side::Sell;
            throw py::value_error("Side must be 0 (BUY) or 1 (SELL)");
        }));

    py::class_<Fill>(m, "Fill", "One trade report produced by the matching engine.")
        .def_readonly("taker_order_id", &Fill::taker_order_id)
        .def_readonly("maker_order_id", &Fill::maker_order_id)
        .def_readonly("maker_trader_id", &Fill::maker_trader_id)
        .def_readonly("taker_side", &Fill::taker_side)
        .def_readonly("price", &Fill::price,
                      "execution price in ticks (always the maker's resting price)")
        .def_readonly("quantity", &Fill::quantity)
        .def_readonly("taker_remaining", &Fill::taker_remaining)
        .def_readonly("maker_remaining", &Fill::maker_remaining)
        .def("__repr__", [](const Fill& f) {
            return "Fill(taker=" + std::to_string(f.taker_order_id) +
                   ", maker=" + std::to_string(f.maker_order_id) +
                   ", price=" + std::to_string(f.price) +
                   ", quantity=" + std::to_string(f.quantity) + ")";
        });

    py::class_<OrderInfo>(m, "OrderInfo", "Snapshot of a resting order.")
        .def_readonly("order_id", &OrderInfo::order_id)
        .def_readonly("trader_id", &OrderInfo::trader_id)
        .def_readonly("side", &OrderInfo::side)
        .def_readonly("price", &OrderInfo::price)
        .def_readonly("quantity", &OrderInfo::quantity)
        .def_readonly("filled", &OrderInfo::filled);

    py::class_<SubmitResult>(m, "SubmitResult", "Outcome of submitting one order.")
        .def_readonly("order_id", &SubmitResult::order_id)
        .def_readonly("fills", &SubmitResult::fills,
                      "fills in execution order (empty if the order only rested)")
        .def_readonly("self_trade_cancels", &SubmitResult::self_trade_cancels,
                      "resting order ids cancelled by self-trade prevention")
        .def_readonly("rested", &SubmitResult::rested,
                      "True if any quantity of this order is resting on the book")
        .def_readonly("resting_price", &SubmitResult::resting_price)
        .def_readonly("resting_quantity", &SubmitResult::resting_quantity);

    py::class_<LimitOrderBook>(m, "LimitOrderBook",
                               "Deterministic two-sided limit order book "
                               "(price-time priority, GFD orders, cancels, "
                               "cancel-resting self-trade prevention).")
        .def(py::init<>())
        .def(py::init<std::uint64_t>(), py::arg("first_order_id"),
             "start automatic order ids from `first_order_id` (default 1)")
        .def("submit", &submit_py, py::arg("side"), py::arg("price"),
             py::arg("quantity"), py::arg("trader_id") = k_default_trader,
             py::arg("order_id") = py::none(),
             "Submit a good-for-day limit order; returns a SubmitResult with "
             "the fills it generated. `trader_id` drives self-trade prevention; "
             "`order_id` (if given) must not be in use.")
        .def("cancel", &cancel_py, py::arg("order_id"),
             "Cancel a resting order; returns False for unknown/filled ids.")
        .def("on_fill", &LimitOrderBook::set_fill_callback, py::arg("callback"),
             "Register a callback(fill) invoked synchronously per fill. While "
             "a callback is registered the GIL is held during submit/cancel "
             "(and re-entrant submits from the callback are rejected).")
        .def("has_fill_callback", &LimitOrderBook::has_fill_callback)
        .def("has_order", &LimitOrderBook::has_order, py::arg("order_id"))
        .def("order_info", &LimitOrderBook::order_info, py::arg("order_id"),
             "OrderInfo for a resting order, or None.")
        .def("empty", &LimitOrderBook::empty)
        .def("order_count", &LimitOrderBook::order_count)
        .def("best_bid", &LimitOrderBook::best_bid,
             "(price, quantity) of the best bid, or None.")
        .def("best_ask", &LimitOrderBook::best_ask,
             "(price, quantity) of the best ask, or None.")
        .def("bid_qty_at", &LimitOrderBook::bid_qty_at, py::arg("price"))
        .def("ask_qty_at", &LimitOrderBook::ask_qty_at, py::arg("price"))
        .def("spread", &LimitOrderBook::spread,
             "best ask - best bid in ticks, or None.")
        .def("bid_depth", &LimitOrderBook::bid_depth, py::arg("levels") = 10,
             "Top bid levels (best first) as [(price, quantity), ...].")
        .def("ask_depth", &LimitOrderBook::ask_depth, py::arg("levels") = 10,
             "Top ask levels (best first) as [(price, quantity), ...].")
        .def("bid_level_count", &LimitOrderBook::bid_level_count)
        .def("ask_level_count", &LimitOrderBook::ask_level_count)
        .def("submitted_count", &LimitOrderBook::submitted_count)
        .def("fill_count", &LimitOrderBook::fill_count)
        .def("cancel_count", &LimitOrderBook::cancel_count)
        .def("self_trade_cancel_count", &LimitOrderBook::self_trade_cancel_count);
}
