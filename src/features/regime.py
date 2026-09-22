"""Causal market-regime and risk-context features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add causal trend, volatility, drawdown and volume-pressure features."""
    data = frame.copy().sort_values("Date").reset_index(drop=True)

    close = pd.to_numeric(data["Close"], errors="coerce")
    volume = pd.to_numeric(data["Volume"], errors="coerce").fillna(0.0)

    sma50 = close.rolling(50, min_periods=10).mean()
    sma200 = close.rolling(200, min_periods=40).mean()
    returns = close.pct_change()

    data["regime_trend_50"] = (
        close / sma50 - 1.0
    ).replace([np.inf, -np.inf], np.nan)

    data["regime_trend_200"] = (
        close / sma200 - 1.0
    ).replace([np.inf, -np.inf], np.nan)

    vol20 = returns.rolling(20, min_periods=10).std()
    vol60 = returns.rolling(60, min_periods=20).std()
    data["regime_volatility_ratio"] = (
        vol20 / vol60.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    rolling_peak = close.rolling(60, min_periods=20).max()
    data["drawdown_60d"] = (
        close / rolling_peak - 1.0
    ).replace([np.inf, -np.inf], np.nan)

    signed_volume = np.sign(returns.fillna(0.0)) * volume
    mean_volume = volume.rolling(20, min_periods=10).mean()
    data["volume_pressure_20d"] = (
        signed_volume.rolling(20, min_periods=10).sum()
        / mean_volume.replace(0.0, np.nan)
        / 20.0
    ).replace([np.inf, -np.inf], np.nan)

    data["return_skew_20d"] = returns.rolling(
        20,
        min_periods=10,
    ).skew()

    data["regime_risk_on"] = (
        (
            (data["regime_trend_50"] > 0)
            & (data["regime_trend_200"] > 0)
            & (data["regime_volatility_ratio"] <= 1.25)
        )
        .astype(float)
    )

    data["regime_risk_off"] = (
        (
            (data["regime_trend_50"] < 0)
            & (data["regime_trend_200"] < 0)
            & (data["regime_volatility_ratio"] > 1.0)
        )
        .astype(float)
    )

    numeric = data.select_dtypes(include=["number"]).columns
    data[numeric] = data[numeric].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return data
