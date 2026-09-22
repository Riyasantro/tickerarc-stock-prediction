"""Causal heuristic chart-pattern features.

Chart patterns do not have one universal finite catalogue. TickerArc uses
explicit, reproducible heuristics for common price-structure patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_level(series: pd.Series, window: int, method: str) -> pd.Series:
    if method == "max":
        return series.rolling(window).max().shift(1)
    return series.rolling(window).min().shift(1)


def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    data = df.copy()
    data[f"support_{window}"] = _rolling_level(data["Low"], window, "min")
    data[f"resistance_{window}"] = _rolling_level(data["High"], window, "max")
    data[f"breakout_{window}"] = (
        data["Close"] > data[f"resistance_{window}"]
    ).astype(np.int8)
    data[f"breakdown_{window}"] = (
        data["Close"] < data[f"support_{window}"]
    ).astype(np.int8)
    return data


def _local_peaks(values: np.ndarray, distance: int = 2) -> list[int]:
    peaks: list[int] = []
    for i in range(distance, len(values) - distance):
        left = values[i - distance:i]
        right = values[i + 1:i + 1 + distance]
        if values[i] >= left.max() and values[i] >= right.max():
            peaks.append(i)
    return peaks


def _local_troughs(values: np.ndarray, distance: int = 2) -> list[int]:
    troughs: list[int] = []
    for i in range(distance, len(values) - distance):
        left = values[i - distance:i]
        right = values[i + 1:i + 1 + distance]
        if values[i] <= left.min() and values[i] <= right.min():
            troughs.append(i)
    return troughs


def add_reversal_patterns(
    df: pd.DataFrame, window: int = 30, tolerance: float = 0.025
) -> pd.DataFrame:
    data = df.copy()
    close = data["Close"].to_numpy(dtype=float)

    double_top = np.zeros(len(data), dtype=np.int8)
    double_bottom = np.zeros(len(data), dtype=np.int8)
    head_shoulders = np.zeros(len(data), dtype=np.int8)
    inverse_hs = np.zeros(len(data), dtype=np.int8)

    for end in range(window, len(data)):
        segment = close[end - window:end]
        peaks = _local_peaks(segment)
        troughs = _local_troughs(segment)

        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            first, second = segment[p1], segment[p2]
            if abs(first - second) / max(abs(first), 1e-12) <= tolerance:
                double_top[end] = 1

        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            first, second = segment[t1], segment[t2]
            if abs(first - second) / max(abs(first), 1e-12) <= tolerance:
                double_bottom[end] = 1

        if len(peaks) >= 3:
            p1, p2, p3 = peaks[-3:]
            a, b, c = segment[p1], segment[p2], segment[p3]
            if b > a and b > c and abs(a - c) / max(abs(b), 1e-12) <= tolerance:
                head_shoulders[end] = 1

        if len(troughs) >= 3:
            t1, t2, t3 = troughs[-3:]
            a, b, c = segment[t1], segment[t2], segment[t3]
            if b < a and b < c and abs(a - c) / max(abs(b), 1e-12) <= tolerance:
                inverse_hs[end] = 1

    data[f"double_top_{window}"] = double_top
    data[f"double_bottom_{window}"] = double_bottom
    data[f"head_shoulders_{window}"] = head_shoulders
    data[f"inverse_head_shoulders_{window}"] = inverse_hs
    return data


def add_triangle_patterns(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    data = df.copy()
    high_slope = data["High"].rolling(window).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
        if np.isfinite(x).all()
        else np.nan,
        raw=True,
    )
    low_slope = data["Low"].rolling(window).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
        if np.isfinite(x).all()
        else np.nan,
        raw=True,
    )

    scale = data["Close"].rolling(window).mean().abs() + 1e-12
    upper = high_slope / scale
    lower = low_slope / scale

    data[f"ascending_triangle_{window}"] = (
        (upper.abs() < 0.0008) & (lower > 0.0008)
    ).astype(np.int8)
    data[f"descending_triangle_{window}"] = (
        (upper < -0.0008) & (lower.abs() < 0.0008)
    ).astype(np.int8)
    data[f"symmetrical_triangle_{window}"] = (
        (upper < -0.0005) & (lower > 0.0005)
    ).astype(np.int8)
    return data


def add_wedge_patterns(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    data = df.copy()
    upper_slope = data["High"].rolling(window).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
        if np.isfinite(x).all()
        else np.nan,
        raw=True,
    )
    lower_slope = data["Low"].rolling(window).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
        if np.isfinite(x).all()
        else np.nan,
        raw=True,
    )
    scale = data["Close"].rolling(window).mean().abs() + 1e-12
    upper = upper_slope / scale
    lower = lower_slope / scale

    data[f"rising_wedge_{window}"] = (
        (upper > 0.0004) & (lower > 0.0004) & (upper < lower)
    ).astype(np.int8)
    data[f"falling_wedge_{window}"] = (
        (upper < -0.0004) & (lower < -0.0004) & (upper > lower)
    ).astype(np.int8)
    return data


def add_flag_features(
    df: pd.DataFrame, impulse_window: int = 5, flag_window: int = 8
) -> pd.DataFrame:
    data = df.copy()
    impulse = data["Close"].pct_change(impulse_window)
    flag_return = data["Close"].pct_change(flag_window)

    data["bull_flag"] = (
        (impulse.shift(flag_window) > 0.04)
        & (flag_return.abs() < 0.02)
    ).astype(np.int8)
    data["bear_flag"] = (
        (impulse.shift(flag_window) < -0.04)
        & (flag_return.abs() < 0.02)
    ).astype(np.int8)
    return data


def add_chart_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = add_support_resistance(data, window=20)
    data = add_reversal_patterns(data, window=30)
    data = add_triangle_patterns(data, window=20)
    data = add_wedge_patterns(data, window=20)
    data = add_flag_features(data)
    return data
