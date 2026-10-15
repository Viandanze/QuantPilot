// Self-test harness for the QuantPilot backtest core (no external deps).
//
// Run: qp_backtest_core_test  (exit code 0 = all checks passed)
// Covers: indicator math on hand-computed cases, plus a deterministic
// two-symbol smoke backtest with shape and conservation checks.
#include "backtest_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "CHECK failed: %s (line %d)\n", #cond,      \
                         __LINE__);                                          \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

void test_sma() {
    std::vector<double> x(50, 7.5);
    auto m = qp::sma(x, 5);
    for (int i = 0; i < 50; ++i) {
        if (i < 4) {
            CHECK(std::isnan(m[i]));
        } else {
            CHECK(std::fabs(m[i] - 7.5) < 1e-12);
        }
    }
    // Window larger than the input: all NaN.
    auto m2 = qp::sma(x, 100);
    CHECK(std::all_of(m2.begin(), m2.end(), [](double v) { return std::isnan(v); }));
}

void test_roc() {
    std::vector<double> y{100.0, 110.0, 121.0};
    auto r = qp::roc(y, 2);
    CHECK(std::isnan(r[0]) && std::isnan(r[1]));
    CHECK(std::fabs(r[2] - 0.21) < 1e-12);
}

void test_true_range_atr() {
    std::vector<qp::Bar> bars{
        {10.0, 12.0, 9.0, 11.0, 0, 0},
        {11.0, 13.0, 10.5, 12.5, 0, 1},
        {12.5, 12.6, 11.0, 11.2, 0, 2},
    };
    auto tr = qp::true_range(bars);
    CHECK(std::fabs(tr[0] - 3.0) < 1e-12);
    CHECK(std::fabs(tr[1] - 2.5) < 1e-12);  // max(2.5, |13-11|, |10.5-11|)
    CHECK(std::fabs(tr[2] - 1.6) < 1e-12);  // max(1.6, |12.6-12.5|, |11-12.5|)
    auto a = qp::atr(bars, 2);
    CHECK(std::isnan(a[0]));
    CHECK(std::fabs(a[1] - 2.75) < 1e-12);
    CHECK(std::fabs(a[2] - (2.5 + 1.6) / 2.0) < 1e-12);
}

void test_smoke_backtest() {
    // Deterministic random-walk panel, two symbols, 250 sessions.
    std::mt19937_64 rng(7);
    std::normal_distribution<double> noise(0.0, 0.01);
    const int N = 250;
    std::vector<std::vector<qp::Bar>> panel(2, std::vector<qp::Bar>(N));
    for (int s = 0; s < 2; ++s) {
        double px = 50.0;
        for (int d = 0; d < N; ++d) {
            const double open = px * (1.0 + noise(rng));
            const double close = open * (1.0 + noise(rng));
            panel[s][d] = {open, std::max(open, close) * 1.001,
                           std::min(open, close) * 0.999, close, 1'000'000, d};
            px = close;
        }
    }
    qp::BacktestParams p;
    const auto res = qp::run_dual_ma_backtest(panel, p);

    CHECK(static_cast<int>(res.equity_curve.size()) == N);
    CHECK(static_cast<int>(res.equity_ts.size()) == N);
    CHECK(res.trade_count == static_cast<long long>(res.trades.size()));
    CHECK(std::fabs(res.total_return - (res.final_equity / p.capital - 1.0)) < 1e-12);
    CHECK(res.max_drawdown >= 0.0 && res.max_drawdown <= 1.0);
    // First session is always warmup: no fills can exist before the slow MA
    // window plus one session.
    if (!res.trades.empty()) {
        CHECK(res.trades.front().ts >= p.slow);
    }
    // Total fees decompose exactly.
    double fees = 0.0;
    for (const auto& t : res.trades) {
        fees += t.fee_commission + t.fee_stamp;
    }
    CHECK(std::fabs(fees - res.total_fees) < 1e-6);
    std::printf("smoke backtest: %lld trades, final equity %.2f, mdd %.4f, fees %.2f\n",
                res.trade_count, res.final_equity, res.max_drawdown, res.total_fees);
}

void test_validation() {
    qp::BacktestParams p;
    p.fast = 20;
    p.slow = 5;  // inverted windows must be rejected
    std::vector<std::vector<qp::Bar>> panel(1, std::vector<qp::Bar>(10));
    bool threw = false;
    try {
        (void)qp::run_dual_ma_backtest(panel, p);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    test_sma();
    test_roc();
    test_true_range_atr();
    test_smoke_backtest();
    test_validation();
    if (g_failures != 0) {
        std::fprintf(stderr, "backtest_core self-test FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("backtest_core self-test passed\n");
    return 0;
}
