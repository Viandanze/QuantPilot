"""Realistic A-share transaction cost model.

Defaults reflect a typical retail brokerage account and are deliberately
conservative:

- commission: 0.025% per side, with a 5 CNY minimum per trade
- stamp tax: 0.1% on the sell side (kept at the historical headline rate for
  conservatism; make it a parameter if you want to model the 2023 reduction)
- slippage: 0.1% adverse move on both sides (buys fill higher, sells lower)

All rates are fractions of turnover, not percentages.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_rate: float = 2.5e-4   # 0.025% per side
    commission_min: float = 5.0       # CNY, per trade
    stamp_tax_rate: float = 1.0e-3    # 0.1%, sell side only
    slippage_rate: float = 1.0e-3     # 0.1% adverse fill on both sides

    def buy_price(self, raw_price: float) -> float:
        """Executed buy price after adverse slippage."""
        return raw_price * (1.0 + self.slippage_rate)

    def sell_price(self, raw_price: float) -> float:
        """Executed sell price after adverse slippage."""
        return raw_price * (1.0 - self.slippage_rate)

    def commission(self, turnover: float) -> float:
        if turnover <= 0.0:
            return 0.0
        return max(turnover * self.commission_rate, self.commission_min)

    def buy_fee(self, turnover: float) -> float:
        """Fees charged on a buy (commission only, no stamp tax)."""
        return self.commission(turnover)

    def sell_fee(self, turnover: float) -> float:
        """Fees charged on a sell (commission + stamp tax)."""
        return self.commission(turnover) + turnover * self.stamp_tax_rate
