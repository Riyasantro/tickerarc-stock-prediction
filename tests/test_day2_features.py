import numpy as np
import pandas as pd
import pytest

from src.features.chart_patterns import add_chart_pattern_features

talib = pytest.importorskip("talib")
from src.features.talib_features import add_all_candlestick_patterns, add_talib_indicators


def sample_ohlcv(rows: int = 140) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0008, 0.015, rows)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.003, rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.008, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.008, rows))
    volume = rng.integers(100_000, 2_000_000, rows)

    return pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=rows, freq="B"),
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


def test_talib_and_candlestick_features():
    result = add_all_candlestick_patterns(add_talib_indicators(sample_ohlcv()))
    assert "rsi_14" in result.columns
    assert "macd" in result.columns
    assert any(name.startswith("candle_cdl") for name in result.columns)


def test_chart_patterns_add_columns():
    result = add_chart_pattern_features(sample_ohlcv())
    expected = {
        "support_20", "resistance_20", "breakout_20", "breakdown_20",
        "double_top_30", "double_bottom_30", "head_shoulders_30",
        "inverse_head_shoulders_30", "ascending_triangle_20",
        "descending_triangle_20", "symmetrical_triangle_20",
        "rising_wedge_20", "falling_wedge_20", "bull_flag", "bear_flag",
    }
    assert expected.issubset(result.columns)
