import numpy as np
import pandas as pd

from src.features.technical import build_features


def synthetic_ohlcv(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, rows)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.003, rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0, 0.01, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0, 0.01, rows))
    volume = rng.integers(100_000, 1_000_000, rows)

    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def test_build_features_shape_and_columns():
    source = synthetic_ohlcv()
    result = build_features(source)

    assert len(result) == len(source)
    expected = {
        "return_1d",
        "sma_20",
        "ema_20",
        "rsi_14",
        "macd",
        "atr_14",
        "bb_upper",
        "bb_lower",
        "volume_ratio_20",
        "volatility_20d",
        "momentum_20d",
        "regime_trend_50",
        "regime_volatility_ratio",
        "drawdown_60d",
        "regime_risk_on",
        "regime_risk_off",
    }
    assert expected.issubset(result.columns)


def test_features_are_finite_after_warmup():
    result = build_features(synthetic_ohlcv())
    numeric = result.select_dtypes(include=["number"]).iloc[70:]
    assert np.isfinite(numeric.to_numpy()).all()
