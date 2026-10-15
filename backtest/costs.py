"""Realistic A-share transaction cost model.

Defaults reflect the current (post-2023-08) retail fee regime, calibrated
against the paper broker's actual fill on 2026-09-18 (order #1965 charged
0.03% commission):

- commission: 0.03% per side, with a 5 CNY minimum per trade
- stamp tax: 0.05% on the sell side (halved from 0.1% on 2023-08-28)
- transfer fee: 0.001% on both sides (since the 2022-04 reduction)
- slippage: 0.1% adverse move on both sides (buys fill higher, sells lower)
- dynamic slippage (opt-in): fixed slippage_rate plus volume_impact_rate
  scaled by the fill's participation (volume / 20d average volume);
  volume_impact_rate defaults to 0.0, preserving legacy behaviour

All rates are fractions of turnover, not percentages.
"""
from __future__ import annotations

import math

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_rate: float = 3.0e-4    # 0.03% per side (matches paper broker)
    commission_min: float = 5.0        # CNY, per trade
    stamp_tax_rate: float = 5.0e-4     # 0.05%, sell side only (2023-08-28 cut)
    transfer_fee_rate: float = 1.0e-5  # 0.001%, both sides (2022-04 cut)
    slippage_rate: float = 1.0e-3      # 0.1% adverse fill on both sides
    volume_impact_rate: float = 0.0    # extra slippage per unit participation

    def buy_price(self, raw_price: float) -> float:
        """Executed buy price after adverse slippage."""
        return raw_price * (1.0 + self.slippage_rate)

    def sell_price(self, raw_price: float) -> float:
        """Executed sell price after adverse slippage."""
        return raw_price * (1.0 - self.slippage_rate)

    def _dynamic_slippage(self, volume: float, adv_volume: float) -> float:
        """Effective slippage for one fill: fixed rate + linear volume impact.

        participation = volume / adv_volume (fill size vs 20-day average
        volume). Falls back to the static ``slippage_rate`` whenever
        participation is non-positive or undefined (zero/negative/NaN
        inputs), so degenerate data degrades to the legacy behaviour.
        """
        if adv_volume <= 0.0 or not volume > 0.0:
            return self.slippage_rate
        participation = volume / adv_volume
        if math.isnan(participation):
            return self.slippage_rate
        return self.slippage_rate + self.volume_impact_rate * participation

    def buy_price_dynamic(self, raw_price: float, volume: float,
                          adv_volume: float) -> float:
        """Executed buy price under participation-aware dynamic slippage."""
        return raw_price * (1.0 + self._dynamic_slippage(volume, adv_volume))

    def sell_price_dynamic(self, raw_price: float, volume: float,
                           adv_volume: float) -> float:
        """Executed sell price under participation-aware dynamic slippage."""
        return raw_price * (1.0 - self._dynamic_slippage(volume, adv_volume))

    def commission(self, turnover: float) -> float:
        if turnover <= 0.0:
            return 0.0
        return max(turnover * self.commission_rate, self.commission_min)

    def buy_fee(self, turnover: float) -> float:
        """Fees charged on a buy (commission + transfer fee)."""
        return self.commission(turnover) + turnover * self.transfer_fee_rate

    def sell_fee(self, turnover: float) -> float:
        """Fees charged on a sell (commission + stamp tax + transfer fee)."""
        return (
            self.commission(turnover)
            + turnover * self.stamp_tax_rate
            + turnover * self.transfer_fee_rate
        )
