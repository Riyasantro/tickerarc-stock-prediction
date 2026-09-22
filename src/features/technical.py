"""Day 2 feature pipeline built on pandas + TA-Lib + chart heuristics."""

from __future__ import annotations

import pandas as pd

from src.features.chart_patterns import add_chart_pattern_features
from src.features.regime import add_regime_features
from src.features.talib_features import (
    add_all_candlestick_patterns,
    add_talib_indicators,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build technical, candlestick, and chart-pattern features without look-ahead."""
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy().sort_values("Date").reset_index(drop=True)
    data = add_talib_indicators(data)
    data = add_all_candlestick_patterns(data)
    data = add_chart_pattern_features(data)
    data = add_regime_features(data)
    return data
