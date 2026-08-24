"""Data access layer: akshare-backed daily bars with offline sample fallback."""

from data.loader import (
    DEFAULT_SYMBOLS,
    BENCHMARK_SYMBOL,
    DATA_SOURCE,
    build_panel,
    load_benchmark,
    load_daily,
)
from data.samples import SAMPLE_SPECS, ensure_samples, generate_sample

__all__ = [
    "DEFAULT_SYMBOLS",
    "BENCHMARK_SYMBOL",
    "DATA_SOURCE",
    "build_panel",
    "load_benchmark",
    "load_daily",
    "SAMPLE_SPECS",
    "ensure_samples",
    "generate_sample",
]
