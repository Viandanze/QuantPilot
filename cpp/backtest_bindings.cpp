// pybind11 bindings for the QuantPilot backtest core (module: backtest_core).
//
// Python surface:
//   import backtest_core as bt
//   p = bt.BacktestParams(); p.fast = 5; p.slow = 20
//   res = bt.run_dual_ma_backtest(opens, highs, lows, closes, volumes, ts, p)
//     - opens/highs/lows/closes/volumes: S x N float64 arrays (one row per
//       symbol, symbols ordered like the engine's sorted symbol list)
//     - ts: length-N int64 array of session timestamps (e.g. ns since epoch)
//   res.equity_curve / res.trades / res.total_return / res.max_drawdown / ...
//   bt.sma(x, window), bt.roc(x, period), bt.true_range(o, h, l, c), bt.atr(...)
//
// GIL policy: array marshaling happens with the GIL held; the whole C++
// simulation runs with the GIL released, so other Python threads keep
// running while the core works.
#include "backtest_core.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

namespace py = pybind11;
using namespace qp;

namespace {

constexpr const char* k_side_names[] = {"BUY", "SELL"};
constexpr const char* k_reason_names[] = {"signal", "stop_loss", "drawdown_flat"};

using Arr2D = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Arr1D = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrTs = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

// Builds the panel from S x N row-major arrays (one row per symbol).
std::vector<std::vector<Bar>> panel_from_arrays(const Arr2D& opens,
                                                const Arr2D& highs,
                                                const Arr2D& lows,
                                                const Arr2D& closes,
                                                const Arr2D& volumes,
                                                const ArrTs& timestamps) {
    auto o = opens.unchecked<2>();
    auto h = highs.unchecked<2>();
    auto l = lows.unchecked<2>();
    auto c = closes.unchecked<2>();
    auto v = volumes.unchecked<2>();
    auto t = timestamps.unchecked<1>();
    if (o.shape(0) != h.shape(0) || o.shape(0) != l.shape(0) ||
        o.shape(0) != c.shape(0) || o.shape(0) != v.shape(0)) {
        throw py::value_error("price arrays must share the same symbol count");
    }
    if (o.shape(1) != h.shape(1) || o.shape(1) != l.shape(1) ||
        o.shape(1) != c.shape(1) || o.shape(1) != v.shape(1)) {
        throw py::value_error("price arrays must share the same bar count");
    }
    if (t.shape(0) != o.shape(1)) {
        throw py::value_error("timestamps must have one entry per bar");
    }
    const int S = static_cast<int>(o.shape(0));
    const int N = static_cast<int>(o.shape(1));
    std::vector<std::vector<Bar>> panel(S, std::vector<Bar>(N));
    for (int s = 0; s < S; ++s) {
        for (int d = 0; d < N; ++d) {
            panel[s][d] = Bar{o(s, d), h(s, d), l(s, d), c(s, d),
                              static_cast<long long>(v(s, d)), t(d)};
        }
    }
    return panel;
}

BacktestResult run_py(const Arr2D& opens, const Arr2D& highs, const Arr2D& lows,
                      const Arr2D& closes, const Arr2D& volumes, const ArrTs& timestamps,
                      const BacktestParams& params) {
    auto panel = panel_from_arrays(opens, highs, lows, closes, volumes, timestamps);
    BacktestResult res;
    {
        py::gil_scoped_release nogil;
        res = run_dual_ma_backtest(panel, params);
    }
    return res;  // converted with the GIL held
}

std::vector<double> arr_to_vec(const Arr1D& a) {
    auto u = a.unchecked<1>();
    std::vector<double> out(static_cast<std::size_t>(u.shape(0)));
    for (py::ssize_t i = 0; i < u.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = u(i);
    }
    return out;
}

py::array_t<double> vec_to_arr(const std::vector<double>& v) {
    return py::array_t<double>(v.size(), v.data());
}

std::vector<Bar> bars_from_arrays(const Arr1D& opens, const Arr1D& highs,
                                  const Arr1D& lows, const Arr1D& closes) {
    auto o = arr_to_vec(opens);
    auto h = arr_to_vec(highs);
    auto l = arr_to_vec(lows);
    auto c = arr_to_vec(closes);
    if (o.size() != h.size() || o.size() != l.size() || o.size() != c.size()) {
        throw py::value_error("price arrays must have the same length");
    }
    std::vector<Bar> bars(o.size());
    for (std::size_t i = 0; i < o.size(); ++i) {
        bars[i] = Bar{o[i], h[i], l[i], c[i], 0, static_cast<std::int64_t>(i)};
    }
    return bars;
}

}  // namespace

PYBIND11_MODULE(backtest_core, m) {
    m.doc() = "QuantPilot C++17 backtest core: vectorized indicators and a "
              "dual-MA daily backtest that mirrors the Python event-driven "
              "engine operation-for-operation (next-open execution, T+1, lot "
              "sizing, A-share cost model, risk overlay).";

    py::class_<BacktestParams>(m, "BacktestParams",
                               "Fee/risk parameters. Defaults mirror "
                               "backtest.costs.CostModel and risk.manager.RiskConfig.")
        .def(py::init<>())
        .def_readwrite("capital", &BacktestParams::capital)
        .def_readwrite("commission_rate", &BacktestParams::commission_rate)
        .def_readwrite("commission_min", &BacktestParams::commission_min)
        .def_readwrite("stamp_tax_rate", &BacktestParams::stamp_tax_rate)
        .def_readwrite("transfer_fee_rate", &BacktestParams::transfer_fee_rate)
        .def_readwrite("slippage_rate", &BacktestParams::slippage_rate)
        .def_readwrite("fast", &BacktestParams::fast)
        .def_readwrite("slow", &BacktestParams::slow)
        .def_readwrite("max_position_weight", &BacktestParams::max_position_weight)
        .def_readwrite("stop_loss", &BacktestParams::stop_loss)
        .def_readwrite("max_drawdown", &BacktestParams::max_drawdown)
        .def_readwrite("cooldown_days", &BacktestParams::cooldown_days)
        .def_readwrite("reentry_cooldown", &BacktestParams::reentry_cooldown)
        .def_readwrite("rebalance_band", &BacktestParams::rebalance_band)
        .def_readwrite("lot", &BacktestParams::lot);

    py::class_<TradeRecord>(m, "Trade", "One executed trade (mirrors portfolio.Trade).")
        .def_readonly("ts", &TradeRecord::ts)
        .def_readonly("symbol", &TradeRecord::symbol,
                      "index into the sorted symbol list")
        .def_property_readonly(
            "side", [](const TradeRecord& t) { return std::string(k_side_names[t.side]); })
        .def_readonly("shares", &TradeRecord::shares)
        .def_readonly("price", &TradeRecord::price,
                      "executed price (after adverse slippage)")
        .def_readonly("turnover", &TradeRecord::turnover)
        .def_property_readonly(
            "fee",
            [](const TradeRecord& t) { return t.fee_commission + t.fee_stamp; })
        .def_readonly("fee_commission", &TradeRecord::fee_commission,
                      "commission + transfer fee")
        .def_readonly("fee_stamp", &TradeRecord::fee_stamp)
        .def_readonly("realized_pnl", &TradeRecord::realized_pnl)
        .def_readonly("cash_after", &TradeRecord::cash_after)
        .def_property_readonly("reason", [](const TradeRecord& t) {
            return std::string(k_reason_names[t.reason]);
        });

    py::class_<RoundTripRecord>(m, "RoundTrip",
                                "One completed round trip (mirrors portfolio.RoundTrip).")
        .def_readonly("symbol", &RoundTripRecord::symbol)
        .def_readonly("entry_ts", &RoundTripRecord::entry_ts)
        .def_readonly("exit_ts", &RoundTripRecord::exit_ts)
        .def_readonly("pnl", &RoundTripRecord::pnl)
        .def_property_readonly("exit_reason", [](const RoundTripRecord& r) {
            return std::string(k_reason_names[r.exit_reason]);
        });

    py::class_<BacktestResult>(m, "BacktestResult", "Result of a dual-MA backtest run.")
        .def_readonly("equity_ts", &BacktestResult::equity_ts)
        .def_readonly("equity_curve", &BacktestResult::equity_curve,
                      "one equity point per session close")
        .def_readonly("trades", &BacktestResult::trades)
        .def_readonly("round_trips", &BacktestResult::round_trips)
        .def_readonly("total_return", &BacktestResult::total_return)
        .def_readonly("max_drawdown", &BacktestResult::max_drawdown)
        .def_readonly("trade_count", &BacktestResult::trade_count)
        .def_readonly("total_fees", &BacktestResult::total_fees)
        .def_readonly("commission_total", &BacktestResult::commission_total)
        .def_readonly("stamp_total", &BacktestResult::stamp_total)
        .def_readonly("final_cash", &BacktestResult::final_cash)
        .def_readonly("final_equity", &BacktestResult::final_equity);

    m.def("run_dual_ma_backtest", &run_py,
          py::arg("opens"), py::arg("highs"), py::arg("lows"), py::arg("closes"),
          py::arg("volumes"), py::arg("timestamps"), py::arg("params"),
          "Run the dual-MA daily backtest over a S x N aligned panel. Symbols "
          "must be ordered like the engine's sorted symbol list. Runs with the "
          "GIL released.");

    m.def("sma",
          [](const Arr1D& x, int window) { return vec_to_arr(sma(arr_to_vec(x), window)); },
          py::arg("x"), py::arg("window"),
          "Simple moving average; NaN before the window is full.");

    m.def("roc",
          [](const Arr1D& x, int period) { return vec_to_arr(roc(arr_to_vec(x), period)); },
          py::arg("x"), py::arg("period"),
          "Rate of change (momentum): x[i]/x[i-period] - 1.");

    m.def("true_range",
          [](const Arr1D& opens, const Arr1D& highs, const Arr1D& lows,
             const Arr1D& closes) {
              return vec_to_arr(true_range(bars_from_arrays(opens, highs, lows, closes)));
          },
          py::arg("opens"), py::arg("highs"), py::arg("lows"), py::arg("closes"),
          "True range series; the first bar is high - low.");

    m.def("atr",
          [](const Arr1D& opens, const Arr1D& highs, const Arr1D& lows,
             const Arr1D& closes, int window) {
              return vec_to_arr(atr(bars_from_arrays(opens, highs, lows, closes), window));
          },
          py::arg("opens"), py::arg("highs"), py::arg("lows"), py::arg("closes"),
          py::arg("window"),
          "ATR: rolling mean of true range over `window` bars.");
}
