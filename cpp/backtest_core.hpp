// QuantPilot backtest core: vectorized indicators + dual-MA daily backtest.
//
// Design notes
// ------------
// * Hot path of the Python event-driven engine (engine/runner.py wired with
//   strategies/dual_ma.py, backtest/costs.py, engine/portfolio.py,
//   engine/execution.py and risk/manager.py) re-implemented in plain C++17.
//   The port mirrors the Python arithmetic operation-for-operation: same
//   fill priority at the open, same float ordering, Python-style
//   banker's rounding for lot sizing - so results match the engine
//   bit-for-bit up to indicator summation noise, not merely "similar".
// * Panel layout: one vector<Bar> per symbol, ordered like the engine's
//   sorted symbol list; every symbol shares the same session grid (aligned
//   dates). On an aligned panel the DataFeed's as-of staleness convention
//   degenerates to direct indexing, which this core assumes.
// * Indicators use plain contiguous loops over doubles (SIMD-friendly
//   under -O2/-O3; no external dependencies, standard library only).
// * Complexity: O(S*N) for indicators + O(S) per session for the event
//   simulation; no allocation in the per-session hot loop.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace qp {

// One daily bar. `ts` is the session timestamp (e.g. epoch nanoseconds),
// carried through to trades and equity points for exact Python diffing.
struct Bar {
    double open;
    double high;
    double low;
    double close;
    long long volume;
    std::int64_t ts;
};

// Fee/risk parameters. Fee defaults mirror backtest.costs.CostModel (the
// post-2023-08 A-share retail regime: 0.03% commission with a 5 CNY
// minimum, 0.05% stamp tax on sells, 0.001% transfer fee on both sides,
// 0.1% adverse slippage on both sides). Risk defaults mirror
// risk.manager.RiskConfig; lot/rebalance band mirror the engine defaults.
// All rates are fractions of turnover, not percentages.
struct BacktestParams {
    double capital = 1'000'000.0;
    double commission_rate = 3.0e-4;    // 0.03% per side
    double commission_min = 5.0;        // CNY, per trade
    double stamp_tax_rate = 5.0e-4;     // 0.05%, sell side only
    double transfer_fee_rate = 1.0e-5;  // 0.001%, both sides
    double slippage_rate = 1.0e-3;      // 0.1% adverse fill, both sides
    int fast = 5;                        // fast MA window (bars)
    int slow = 20;                       // slow MA window (bars)
    // risk overlay (risk.manager.RiskConfig defaults)
    double max_position_weight = 0.20;  // single-name cap on target weight
    double stop_loss = -0.08;           // close <= avg_cost * (1 + stop_loss)
    double max_drawdown = 0.15;         // equity drawdown guard
    int cooldown_days = 20;             // cash-only sessions after drawdown flat
    int reentry_cooldown = 10;          // sessions a stopped-out symbol is blocked
    // portfolio (engine defaults)
    double rebalance_band = 0.01;       // min value drift before partial rebalance
    int lot = 100;                      // A-share board lot
};

// Side / reason codes (plain ints: POD structs, cheap to marshal).
inline constexpr std::int8_t kSideBuy = 0;
inline constexpr std::int8_t kSideSell = 1;
inline constexpr std::int8_t kReasonSignal = 0;
inline constexpr std::int8_t kReasonStopLoss = 1;
inline constexpr std::int8_t kReasonDrawdownFlat = 2;

// One executed trade (mirrors engine.portfolio.Trade).
struct TradeRecord {
    std::int64_t ts;
    int symbol;              // index into the sorted symbol list
    std::int8_t side;        // kSideBuy / kSideSell
    long long shares;
    double price;            // executed price (after adverse slippage)
    double turnover;
    double fee_commission;   // commission + transfer fee (both sides)
    double fee_stamp;        // stamp tax (sell side only; 0 on buys)
    double realized_pnl;     // sells only; buys record 0.0
    double cash_after;
    std::int8_t reason;      // kReasonSignal / kReasonStopLoss / kReasonDrawdownFlat
};

// One completed round trip (mirrors engine.portfolio.RoundTrip).
struct RoundTripRecord {
    int symbol;
    std::int64_t entry_ts;
    std::int64_t exit_ts;
    double pnl;              // net of fees, on the position's all-in basis
    std::int8_t exit_reason;
};

struct BacktestResult {
    std::vector<std::int64_t> equity_ts;    // one point per session close
    std::vector<double> equity_curve;
    std::vector<TradeRecord> trades;
    std::vector<RoundTripRecord> round_trips;
    double total_return = 0.0;      // final_equity / capital - 1
    double max_drawdown = 0.0;      // max peak-to-trough on the equity curve
    long long trade_count = 0;
    double total_fees = 0.0;        // commission_total + stamp_total
    double commission_total = 0.0;  // sum of fee_commission (incl. transfer fee)
    double stamp_total = 0.0;
    double final_cash = 0.0;
    double final_equity = 0.0;
};

// ---- indicators (plain loops over contiguous doubles) ---------------------

// Simple moving average; NaN before the window is full. Matches
// pandas rolling(window).mean() on gap-free finite input.
std::vector<double> sma(const std::vector<double>& x, int window);

// Rate of change (momentum): x[i] / x[i - period] - 1. NaN for i < period
// or when the base value is not finite / not positive.
std::vector<double> roc(const std::vector<double>& x, int period);

// True range: max(high - low, |high - prev_close|, |low - prev_close|);
// the first bar is high - low.
std::vector<double> true_range(const std::vector<Bar>& bars);

// ATR: rolling mean of true range over `window` bars; NaN before warmup.
std::vector<double> atr(const std::vector<Bar>& bars, int window);

// ---- backtest --------------------------------------------------------------

// Dual moving-average crossover over a multi-symbol panel, mirroring the
// Python event-driven engine exactly: golden cross (fast MA > slow MA)
// targets an equal 1/S slice per symbol, dead cross targets flat;
// decisions on the close, fills at the next open with adverse slippage;
// whole-lot sizing with a rebalance band, T+1 sellability, affordability
// shrink, the full A-share cost model and the risk overlay (stop loss,
// drawdown flat + cooldown, re-entry blocks, single-name cap).
//
// `panel` must be ordered like the engine's sorted symbol list, and every
// symbol must share the same session grid (aligned dates, finite prices).
// Throws std::invalid_argument on bad input.
BacktestResult run_dual_ma_backtest(const std::vector<std::vector<Bar>>& panel,
                                    const BacktestParams& params);

}  // namespace qp
