"""Strategy registry and factory."""

from strategies.base import Strategy
from strategies.dual_ma import DualMAStrategy
from strategies.grid import GridStrategy
from strategies.momentum import MomentumStrategy

STRATEGIES = {
    DualMAStrategy.name: DualMAStrategy,
    MomentumStrategy.name: MomentumStrategy,
    GridStrategy.name: GridStrategy,
}


def available() -> list[str]:
    return sorted(STRATEGIES)


def create(name: str, panel: dict, **params) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; available: {available()}")
    return STRATEGIES[name](panel, **params)
