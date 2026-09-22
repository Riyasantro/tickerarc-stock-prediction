"""TA-Lib based feature engineering for TickerArc.

This module uses TA-Lib for technical indicators and dynamically loads every
TA-Lib candlestick recognition function whose name starts with CDL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

EPS = 1e-12


def _as_float(values: pd.Series) -> np.ndarray:
    return values.to_numpy(dtype=np.float64, copy=False)


def add_talib_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    open_ = _as_float(data["Open"])
    high = _as_float(data["High"])
    low = _as_float(data["Low"])
    close = _as_float(data["Close"])
    volume = _as_float(data["Volume"])

    data["sma_10"] = talib.SMA(close, timeperiod=10)
    data["sma_20"] = talib.SMA(close, timeperiod=20)
    data["sma_50"] = talib.SMA(close, timeperiod=50)
    data["ema_10"] = talib.EMA(close, timeperiod=10)
    data["ema_20"] = talib.EMA(close, timeperiod=20)

    data["rsi_14"] = talib.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    data["macd"] = macd
    data["macd_signal"] = macd_signal
    data["macd_hist"] = macd_hist

    data["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
    data["natr_14"] = talib.NATR(high, low, close, timeperiod=14)

    bb_upper, bb_mid, bb_lower = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    data["bb_upper"] = bb_upper
    data["bb_mid"] = bb_mid
    data["bb_lower"] = bb_lower
    data["bb_width"] = (bb_upper - bb_lower) / (bb_mid + EPS)
    data["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + EPS)

    data["adx_14"] = talib.ADX(high, low, close, timeperiod=14)
    data["plus_di_14"] = talib.PLUS_DI(high, low, close, timeperiod=14)
    data["minus_di_14"] = talib.MINUS_DI(high, low, close, timeperiod=14)
    data["cci_14"] = talib.CCI(high, low, close, timeperiod=14)

    slow_k, slow_d = talib.STOCH(
        high,
        low,
        close,
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    data["stoch_k"] = slow_k
    data["stoch_d"] = slow_d
    data["willr_14"] = talib.WILLR(high, low, close, timeperiod=14)

    data["roc_10"] = talib.ROC(close, timeperiod=10)
    data["mom_10"] = talib.MOM(close, timeperiod=10)
    data["trix_30"] = talib.TRIX(close, timeperiod=30)

    data["obv"] = talib.OBV(close, volume)
    data["adosc"] = talib.ADOSC(
        high, low, close, volume, fastperiod=3, slowperiod=10
    )
    data["mfi_14"] = talib.MFI(high, low, close, volume, timeperiod=14)
    data["sar"] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

    data["ht_trendmode"] = talib.HT_TRENDMODE(close)
    data["ht_sine"], data["ht_leadsine"] = talib.HT_SINE(close)

    data["return_1d"] = data["Close"].pct_change()
    data["return_5d"] = data["Close"].pct_change(5)
    data["return_10d"] = data["Close"].pct_change(10)
    data["return_20d"] = data["Close"].pct_change(20)
    data["volatility_10d"] = data["return_1d"].rolling(10).std()
    data["volatility_20d"] = data["return_1d"].rolling(20).std()
    data["volatility_60d"] = data["return_1d"].rolling(60).std()
    data["volume_sma_20"] = data["Volume"].rolling(20).mean()
    data["volume_ratio_20"] = data["Volume"] / (data["volume_sma_20"] + EPS)
    data["momentum_10d"] = close / (pd.Series(close).shift(10).to_numpy() + EPS) - 1
    data["momentum_20d"] = close / (pd.Series(close).shift(20).to_numpy() + EPS) - 1

    return data


def add_all_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add every TA-Lib CDL pattern as a separate feature column."""
    data = df.copy()
    open_ = _as_float(data["Open"])
    high = _as_float(data["High"])
    low = _as_float(data["Low"])
    close = _as_float(data["Close"])

    pattern_names = sorted(
        name
        for name in dir(talib)
        if name.startswith("CDL") and callable(getattr(talib, name))
    )

    for name in pattern_names:
        values = getattr(talib, name)(open_, high, low, close)
        data[f"candle_{name.lower()}"] = values.astype(np.int16)

    candle_columns = [c for c in data.columns if c.startswith("candle_cdl")]
    if candle_columns:
        candle_frame = data[candle_columns]
        data["candlestick_bullish_count"] = (candle_frame > 0).sum(axis=1)
        data["candlestick_bearish_count"] = (candle_frame < 0).sum(axis=1)
        data["candlestick_signal_score"] = np.sign(candle_frame).sum(axis=1)

    return data
