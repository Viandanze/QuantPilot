// QuantPilot backtest core: implementation.
//
// The simulation loop mirrors the Python event-driven engine step by step.
// Mapping table (Python -> here):
//   engine.datafeed.DataFeed (aligned panel)      -> direct indexing, this file
//   strategies.dual_ma.DualMAStrategy             -> sma() + weight rule below
//   risk.manager.RiskManager                      -> on_close()/apply_caps() below
//   engine.portfolio.Portfolio                    -> pos_/cash accounting below
//   engine.portfolio.orders_from_weights          -> order generation below
//   engine.execution.SimulatedExecutionHandler    -> open-phase fills below
//   backtest.costs.CostModel                      -> fee formulas below
//
// Float-ordering fidelity: every accumulation (marked equity, capped-weight
// sums, fee sums) iterates symbols in the same order the Python dicts do,
// and lot sizing uses std::nearbyint (round-half-even under the default FP
// mode), matching Python's round().
#include "backtest_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace qp {

std::vector<double> sma(const std::vector<double>& x, int window) {
    const std::size_t n = x.size();
    std::vector<double> out(n, std::numeric_limits<double>::quiet_NaN());
    if (window < 1 || n == 0 || static_cast<std::size_t>(window) > n) {
        return out;
    }
    const std::size_t w = static_cast<std::size_t>(window);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += x[i];
        if (i >= w) {
            sum -= x[i - w];
        }
        if (i >= w - 1) {
            out[i] = sum / static_cast<double>(window);
        }
    }
    return out;
}

std::vector<double> roc(const std::vector<double>& x, int period) {
    const std::size_t n = x.size();
    std::vector<double> out(n, std::numeric_limits<double>::quiet_NaN());
    if (period < 1 || n == 0 || static_cast<std::size_t>(period) >= n) {
        return out;
    }
    for (std::size_t i = static_cast<std::size_t>(period); i < n; ++i) {
        const double base = x[i - static_cast<std::size_t>(period)];
        if (std::isfinite(base) && base > 0.0) {
            out[i] = x[i] / base - 1.0;
        }
    }
    return out;
}

std::vector<double> true_range(const std::vector<Bar>& bars) {
    const std::size_t n = bars.size();
    std::vector<double> out(n, std::numeric_limits<double>::quiet_NaN());
    if (n == 0) {
        return out;
    }
    out[0] = bars[0].high - bars[0].low;
    for (std::size_t i = 1; i < n; ++i) {
        const double hl = bars[i].high - bars[i].low;
        const double hc = std::fabs(bars[i].high - bars[i - 1].close);
        const double lc = std::fabs(bars[i].low - bars[i - 1].close);
        out[i] = std::max(hl, std::max(hc, lc));
    }
    return out;
}

std::vector<double> atr(const std::vector<Bar>& bars, int window) {
    return sma(true_range(bars), window);
}

namespace {

inline bool printable(double v) { return std::isfinite(v) && v > 0.0; }

// Python's round(): nearest integer, ties to even. std::nearbyint under the
// default FE_TONEAREST mode does exactly that on the exact binary value.
inline long long py_round_half_even(double v) {
    return static_cast<long long>(std::nearbyint(v));
}

struct Pos {
    long long shares = 0;
    long long locked = 0;        // bought today, sellable tomorrow (T+1)
    double avg_cost = 0.0;       // all-in basis per share (price + buy fees)
    std::int64_t entry_ts = -1;  // "None" sentinel
    double cycle_realized = 0.0;
    double avg_cost_gross = 0.0;  // price-only basis per share
};

struct PendingOrder {
    int symbol;
    std::int8_t side;
    long long shares;
    std::int8_t reason;
};

}  // namespace

BacktestResult run_dual_ma_backtest(const std::vector<std::vector<Bar>>& panel,
                                    const BacktestParams& p) {
    const int S = static_cast<int>(panel.size());
    if (S == 0) {
        throw std::invalid_argument("empty panel");
    }
    const int N = static_cast<int>(panel[0].size());
    if (N < 2) {
        throw std::invalid_argument("not enough trading dates in the requested range");
    }
    for (int s = 0; s < S; ++s) {
        if (static_cast<int>(panel[s].size()) != N) {
            throw std::invalid_argument("panel bars are not aligned across symbols");
        }
    }
    if (p.fast < 1 || p.slow < 2 || p.fast >= p.slow) {
        throw std::invalid_argument("fast window must be shorter than slow window");
    }
    if (p.lot < 1) {
        throw std::invalid_argument("lot must be positive");
    }

    // Strategy precompute: rolling MAs per symbol (warmup NaN, like pandas).
    std::vector<std::vector<double>> ma_fast(S), ma_slow(S);
    for (int s = 0; s < S; ++s) {
        std::vector<double> closes(N);
        for (int d = 0; d < N; ++d) {
            closes[d] = panel[s][d].close;
        }
        ma_fast[s] = sma(closes, p.fast);
        ma_slow[s] = sma(closes, p.slow);
    }

    // Portfolio state (engine.portfolio.Portfolio).
    std::vector<Pos> pos(S);
    double cash = p.capital;
    double total_costs = 0.0;
    double total_realized_gross = 0.0;

    // Risk state (risk.manager.RiskManager).
    bool has_peak = false;
    double peak_equity = 0.0;
    int cooldown_remaining = 0;
    std::vector<int> symbol_blocks(S, 0);

    std::vector<PendingOrder> pending;
    pending.reserve(64);

    BacktestResult res;
    res.equity_ts.reserve(static_cast<std::size_t>(N));
    res.equity_curve.reserve(static_cast<std::size_t>(N));

    for (int d = 0; d < N; ++d) {
        // ---- open phase: execute yesterday's orders at today's open -------
        // portfolio.rollover_t1()
        for (int s = 0; s < S; ++s) {
            pos[s].locked = 0;
        }
        // execution.execute_pending: forced exits first, then signal sells,
        // then buys; a morning forced exit cancels same-morning entries.
        std::vector<char> is_forced_sym(S, 0);
        for (const auto& o : pending) {
            if (o.reason != kReasonSignal) {
                is_forced_sym[o.symbol] = 1;
            }
        }
        for (int pass = 0; pass < 3; ++pass) {
            for (const auto& o : pending) {
                if (pass == 0 && o.reason == kReasonSignal) {
                    continue;  // forced exits only
                }
                if (pass == 1 &&
                    (o.side != kSideSell || o.reason != kReasonSignal)) {
                    continue;  // signal sells only
                }
                if (pass == 2 && o.side != kSideBuy) {
                    continue;  // buys only
                }
                if (pass == 2 && is_forced_sym[o.symbol]) {
                    continue;  // a forced exit cancels same-morning entries
                }
                const int s = o.symbol;
                const double raw = panel[s][d].open;
                if (!printable(raw)) {
                    continue;  // no printable open; the next decision re-issues
                }
                Pos& P = pos[s];
                if (o.side == kSideSell) {
                    // execution._sell: clamp to T+1 sellable, drop if nothing.
                    const long long sellable =
                        P.shares > P.locked ? P.shares - P.locked : 0;
                    const long long shares = std::min(o.shares, sellable);
                    if (shares <= 0) {
                        continue;
                    }
                    const double px = raw * (1.0 - p.slippage_rate);
                    const double turnover = static_cast<double>(shares) * px;
                    const double commission =
                        (turnover <= 0.0
                             ? 0.0
                             : std::max(turnover * p.commission_rate,
                                        p.commission_min)) +
                        turnover * p.transfer_fee_rate;
                    const double stamp = turnover * p.stamp_tax_rate;
                    const double fee_total = commission + stamp;
                    // portfolio._apply_sell
                    cash += turnover - fee_total;
                    const double realized =
                        (turnover - fee_total) - P.avg_cost * static_cast<double>(shares);
                    total_realized_gross +=
                        turnover - P.avg_cost_gross * static_cast<double>(shares);
                    total_costs += fee_total;
                    P.cycle_realized += realized;
                    P.shares -= shares;
                    res.trades.push_back(TradeRecord{panel[s][d].ts, s, kSideSell,
                                                     shares, px, turnover, commission,
                                                     stamp, realized, cash, o.reason});
                    if (P.shares == 0) {
                        res.round_trips.push_back(RoundTripRecord{
                            s, P.entry_ts, panel[s][d].ts, P.cycle_realized, o.reason});
                        P = Pos{};
                    }
                } else {
                    // execution._buy: shrink by whole lots until affordable.
                    if (o.shares < static_cast<long long>(p.lot)) {
                        continue;
                    }
                    const double px = raw * (1.0 + p.slippage_rate);
                    long long shares = o.shares;
                    double turnover = 0.0;
                    double fee = 0.0;
                    bool affordable = false;
                    while (shares >= static_cast<long long>(p.lot)) {
                        turnover = static_cast<double>(shares) * px;
                        fee = (turnover <= 0.0
                                   ? 0.0
                                   : std::max(turnover * p.commission_rate,
                                              p.commission_min)) +
                              turnover * p.transfer_fee_rate;
                        if (turnover + fee <= cash) {
                            affordable = true;
                            break;
                        }
                        shares -= p.lot;
                    }
                    if (!affordable) {
                        continue;
                    }
                    // portfolio._apply_buy
                    cash -= turnover + fee;
                    const double basis =
                        P.avg_cost * static_cast<double>(P.shares) + turnover + fee;
                    const double gross_basis =
                        P.avg_cost_gross * static_cast<double>(P.shares) + turnover;
                    P.shares += shares;
                    P.avg_cost = basis / static_cast<double>(P.shares);
                    P.avg_cost_gross = gross_basis / static_cast<double>(P.shares);
                    P.locked += shares;  // T+1: not sellable today
                    if (P.entry_ts < 0) {
                        P.entry_ts = panel[s][d].ts;
                    }
                    total_costs += fee;
                    res.trades.push_back(TradeRecord{panel[s][d].ts, s, kSideBuy,
                                                     shares, px, turnover, fee, 0.0,
                                                     0.0, cash, kReasonSignal});
                }
            }
        }
        pending.clear();

        // ---- close phase: mark to market, risk overlay, next orders -------
        // portfolio.mark_to_market: only printable closes are marked; the
        // marked sum accumulates in the engine's (sorted) symbol order.
        double marked_sum = 0.0;
        for (int s = 0; s < S; ++s) {
            const double px = panel[s][d].close;
            if (printable(px)) {
                marked_sum += static_cast<double>(pos[s].shares) * px;
            }
        }
        const double equity = cash + marked_sum;
        res.equity_ts.push_back(panel[0][d].ts);
        res.equity_curve.push_back(equity);

        // risk.manager.RiskManager.on_close
        std::vector<std::pair<int, std::int8_t>> forced;  // (symbol, reason)
        if (cooldown_remaining > 0) {
            --cooldown_remaining;
            if (cooldown_remaining == 0) {
                peak_equity = equity;  // fresh peak after the cooldown
                has_peak = true;
            }
        }
        for (int s = 0; s < S; ++s) {  // _tick_blocks
            if (symbol_blocks[s] > 0 && --symbol_blocks[s] <= 0) {
                symbol_blocks[s] = 0;
            }
        }
        if (cooldown_remaining == 0) {
            if (!has_peak) {
                peak_equity = equity;
                has_peak = true;
            } else if (equity > peak_equity) {
                peak_equity = equity;
            }
            const double drawdown = equity / peak_equity - 1.0;
            if (drawdown < -p.max_drawdown) {
                for (int s = 0; s < S; ++s) {
                    if (pos[s].shares > 0) {
                        forced.emplace_back(s, kReasonDrawdownFlat);
                    }
                }
                cooldown_remaining = p.cooldown_days;
            } else {
                const double stop_level = 1.0 + p.stop_loss;
                for (int s = 0; s < S; ++s) {
                    if (pos[s].shares <= 0) {
                        continue;
                    }
                    const double px = panel[s][d].close;
                    if (!std::isfinite(px)) {
                        continue;
                    }
                    if (px <= pos[s].avg_cost * stop_level) {
                        forced.emplace_back(s, kReasonStopLoss);
                        symbol_blocks[s] = p.reentry_cooldown;
                    }
                }
            }
        }

        // runner._on_close: queue forced sells for held positions.
        for (const auto& f : forced) {
            if (pos[f.first].shares > 0) {
                pending.push_back(PendingOrder{f.first, kSideSell,
                                               pos[f.first].shares, f.second});
            }
        }

        // Strategy + caps + order generation (skipped while in cooldown).
        if (cooldown_remaining == 0) {
            // strategies.dual_ma: hold while fast MA > slow MA.
            const double unit_weight = 1.0 / static_cast<double>(S);
            std::vector<double> capped(S);
            double total = 0.0;
            for (int s = 0; s < S; ++s) {
                const double f = ma_fast[s][d];
                const double sl = ma_slow[s][d];
                const double w =
                    (std::isfinite(f) && std::isfinite(sl) && f > sl) ? unit_weight : 0.0;
                // risk.apply_caps: zero blocked symbols, clamp, keep sum <= 1.
                const double cw = symbol_blocks[s] > 0
                                      ? 0.0
                                      : std::min(std::max(w, 0.0), p.max_position_weight);
                capped[s] = cw;
                total += cw;
            }
            if (total > 1.0 && total > 0.0) {
                const double scale = 1.0 / total;
                for (int s = 0; s < S; ++s) {
                    capped[s] *= scale;
                }
            }
            // portfolio.orders_from_weights
            const double band_value = p.rebalance_band * equity;
            for (int s = 0; s < S; ++s) {
                bool skip = false;
                for (const auto& f : forced) {
                    if (f.first == s) {
                        skip = true;
                        break;
                    }
                }
                if (skip) {
                    continue;  // forced symbols keep their issued exit order
                }
                const double px = panel[s][d].close;
                if (!printable(px)) {
                    continue;
                }
                const double w = capped[s];
                const long long current = pos[s].shares;
                const long long target =
                    py_round_half_even((w * equity) / px / static_cast<double>(p.lot)) *
                    static_cast<long long>(p.lot);
                const long long diff = target - current;
                const double target_value = static_cast<double>(target) * px;
                const double current_value = static_cast<double>(current) * px;
                if (current == 0) {
                    if (target >= static_cast<long long>(p.lot)) {
                        pending.push_back(PendingOrder{s, kSideBuy, diff, kReasonSignal});
                    }
                } else if (target <= 0) {
                    pending.push_back(PendingOrder{s, kSideSell, current, kReasonSignal});
                } else if (diff >= static_cast<long long>(p.lot) &&
                           target_value - current_value > band_value) {
                    pending.push_back(PendingOrder{s, kSideBuy, diff, kReasonSignal});
                } else if (diff <= -static_cast<long long>(p.lot) &&
                           current_value - target_value > band_value) {
                    pending.push_back(PendingOrder{s, kSideSell, -diff, kReasonSignal});
                }
            }
        }
    }

    // ---- assembly (mirrors engine.runner._assemble + metrics) -------------
    res.trade_count = static_cast<long long>(res.trades.size());
    res.final_cash = cash;
    res.final_equity = res.equity_curve.empty() ? p.capital : res.equity_curve.back();
    res.total_return = res.final_equity / p.capital - 1.0;
    double peak = res.equity_curve.empty() ? 0.0 : res.equity_curve.front();
    double mdd = 0.0;
    for (const double eq : res.equity_curve) {
        if (eq > peak) {
            peak = eq;
        }
        if (peak > 0.0) {
            const double dd = (peak - eq) / peak;
            if (dd > mdd) {
                mdd = dd;
            }
        }
    }
    res.max_drawdown = mdd;
    for (const auto& t : res.trades) {
        res.commission_total += t.fee_commission;
        res.stamp_total += t.fee_stamp;
    }
    res.total_fees = res.commission_total + res.stamp_total;
    return res;
}

}  // namespace qp
