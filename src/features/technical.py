"""Causal technical features for TickerArc Day 1."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _true_range(df: pd.DataFrame) -> pd.Series:
    previous_close = df["Close"].shift(1)
    ranges = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["return_1d"] = data["Close"].pct_change()
    data["return_5d"] = data["Close"].pct_change(5)
    data["return_10d"] = data["Close"].pct_change(10)
    data["return_20d"] = data["Close"].pct_change(20)
    return data


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for window in (10, 20, 50):
        data[f"sma_{window}"] = data["Close"].rolling(window).mean()
    return data


def add_exponential_averages(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for window in (10, 20):
        data[f"ema_{window}"] = data["Close"].ewm(span=window, adjust=False).mean()
    return data


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    data = df.copy()
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / (avg_loss + EPS)
    data["rsi_14"] = 100 - (100 / (1 + rs))
    return data


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    ema_fast = data["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = data["Close"].ewm(span=26, adjust=False).mean()
    data["macd"] = ema_fast - ema_slow
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]
    return data


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    data = df.copy()
    tr = _true_range(data)
    data["atr_14"] = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    data["atr_pct"] = data["atr_14"] / (data["Close"] + EPS)
    return data


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    data = df.copy()
    middle = data["Close"].rolling(window).mean()
    std = data["Close"].rolling(window).std(ddof=0)
    data["bb_mid"] = middle
    data["bb_upper"] = middle + 2 * std
    data["bb_lower"] = middle - 2 * std
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / (middle + EPS)
    data["bb_position"] = (data["Close"] - data["bb_lower"]) / (
        data["bb_upper"] - data["bb_lower"] + EPS
    )
    return data


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["volume_sma_20"] = data["Volume"].rolling(20).mean()
    data["volume_ratio_20"] = data["Volume"] / (data["volume_sma_20"] + EPS)
    signed_volume = np.sign(data["Close"].diff().fillna(0)) * data["Volume"]
    data["obv"] = signed_volume.cumsum()
    data["obv_change_20d"] = data["obv"].pct_change(20)
    return data


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    daily_returns = data["Close"].pct_change()
    data["volatility_10d"] = daily_returns.rolling(10).std()
    data["volatility_20d"] = daily_returns.rolling(20).std()
    data["volatility_60d"] = daily_returns.rolling(60).std()
    data["high_low_pct"] = (data["High"] - data["Low"]) / (data["Close"] + EPS)
    return data


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["momentum_10d"] = data["Close"] / (data["Close"].shift(10) + EPS) - 1
    data["momentum_20d"] = data["Close"] / (data["Close"].shift(20) + EPS) - 1
    data["distance_sma_20"] = data["Close"] / (data["sma_20"] + EPS) - 1
    data["distance_sma_50"] = data["Close"] / (data["sma_50"] + EPS) - 1
    return data


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the Day 1 feature set without forward-looking information."""
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy().sort_values("Date").reset_index(drop=True)
    data = add_returns(data)
    data = add_moving_averages(data)
    data = add_exponential_averages(data)
    data = add_rsi(data)
    data = add_macd(data)
    data = add_atr(data)
    data = add_bollinger_bands(data)
    data = add_volume_features(data)
    data = add_volatility_features(data)
    data = add_momentum_features(data)
    return data
