"""Deterministic synthetic daily bars for offline demos and unit tests.

The generator blends alternating trend regimes with Gaussian noise so the
strategies face plausible up/down structure instead of pure noise. Output is
fully deterministic for a given symbol seed and end date, which keeps unit
tests reproducible. This data is synthetic: it exists only so the whole
pipeline can run without network access, and it is clearly labelled as
"sample" wherever it is used.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

SAMPLE_DIR = pathlib.Path(__file__).resolve().parent / "samples"

# symbol -> (base price, daily volatility, rng seed)
SAMPLE_SPECS: dict[str, tuple[float, float, int]] = {
    "600519": (1500.0, 0.018, 101),
    "000858": (220.0, 0.020, 102),
    "601318": (72.0, 0.017, 103),
    "000333": (58.0, 0.019, 104),
    "600036": (41.0, 0.016, 105),
    "510300": (4.1, 0.013, 106),
    "000300": (3900.0, 0.012, 100),
}

# Alternating drift regimes (per day, in return space), each REGIME_LEN days.
REGIME_DRIFTS = (0.0011, -0.0009, 0.0006, -0.0012, 0.0009, -0.0005)
REGIME_LEN = 90


def generate_sample(symbol: str, days: int = 900, end=None) -> pd.DataFrame:
    """Return ``days`` synthetic OHLCV bars ending at ``end`` (default today)."""
    if symbol not in SAMPLE_SPECS:
        raise KeyError(f"no sample spec for symbol {symbol!r}")
    base, vol, seed = SAMPLE_SPECS[symbol]
    rng = np.random.default_rng(seed)
    if end is None:
        end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=pd.Timestamp(end), periods=days)

    closes = np.empty(days)
    price = base
    for i in range(days):
        drift = REGIME_DRIFTS[(i // REGIME_LEN) % len(REGIME_DRIFTS)]
        step = float(np.clip(drift + vol * rng.standard_normal(), -0.08, 0.08))
        price = max(price * (1.0 + step), base * 0.2)
        closes[i] = price

    opens = np.empty(days)
    highs = np.empty(days)
    lows = np.empty(days)
    for i in range(days):
        prev = closes[i - 1] if i > 0 else base
        opens[i] = prev * (1.0 + 0.3 * vol * rng.standard_normal())
        highs[i] = max(opens[i], closes[i]) * (1.0 + abs(0.6 * vol * rng.standard_normal()))
        lows[i] = min(opens[i], closes[i]) * (1.0 - abs(0.6 * vol * rng.standard_normal()))

    volumes = rng.lognormal(mean=13.0, sigma=0.35, size=days).astype(np.int64)
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )
    df.index.name = "date"
    return df.round(2)


def sample_path(symbol: str, out_dir: pathlib.Path = SAMPLE_DIR) -> pathlib.Path:
    return pathlib.Path(out_dir) / f"{symbol}.csv"


def ensure_samples(days: int = 900, end=None, out_dir=pathlib.Path(SAMPLE_DIR)) -> list[pathlib.Path]:
    """Write one CSV per sample symbol if not already present."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for symbol in SAMPLE_SPECS:
        path = sample_path(symbol, out_dir)
        if not path.exists():
            generate_sample(symbol, days=days, end=end).to_csv(path)
            written.append(path)
    return written


def load_sample(symbol: str, out_dir=pathlib.Path(SAMPLE_DIR)) -> pd.DataFrame:
    path = sample_path(symbol, out_dir)
    if not path.exists():
        raise FileNotFoundError(f"sample file missing: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate built-in sample CSVs.")
    parser.add_argument("--days", type=int, default=900)
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()
    for symbol in SAMPLE_SPECS:
        path = sample_path(symbol)
        if args.force or not path.exists():
            generate_sample(symbol, days=args.days).to_csv(path)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
