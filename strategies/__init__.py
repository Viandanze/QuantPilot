from __future__ import annotations
"""Strategy registry and factory."""

from strategies.base import Strategy
from strategies.dual_ma import DualMAStrategy
from strategies.etf_rotation import EtfRotation
from strategies.grid import GridStrategy
from strategies.mean_reversion import MeanReversion
from strategies.momentum import MomentumStrategy

STRATEGIES = {
    DualMAStrategy.name: DualMAStrategy,
    EtfRotation.name: EtfRotation,
    GridStrategy.name: GridStrategy,
    MeanReversion.name: MeanReversion,
    MomentumStrategy.name: MomentumStrategy,
}


def available() -> list[str]:
    return sorted(STRATEGIES)


def create(name: str, panel: dict, **params) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; available: {available()}")
    return STRATEGIES[name](panel, **params)
