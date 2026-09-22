"""Free live/near-live market data layer using yfinance.

TickerArc refreshes this layer every 1 or 3 minutes from the Streamlit app.
It intentionally reports the provider timestamp and does not fabricate ticks
when a free source is delayed or unavailable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from config.universe import yahoo_symbol


def _flatten_download_columns(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if symbol in out.columns.get_level_values(-1):
            out = out.xs(symbol, axis=1, level=-1)
        elif symbol in out.columns.get_level_values(0):
            out = out.xs(symbol, axis=1, level=0)
        else:
            out.columns = out.columns.get_level_values(0)
    return out


def fetch_live_quotes(
    symbols: Iterable[str],
    interval: str = "1m",
    lookback: str = "5d",
) -> pd.DataFrame:
    """Fetch the most recent intraday quote and current-session volume proxy."""
    symbols = list(symbols)
    yahoo_symbols = [yahoo_symbol(s) for s in symbols]
    if not yahoo_symbols:
        return pd.DataFrame()

    frame = yf.download(
        yahoo_symbols,
        period=lookback,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    rows: list[dict[str, object]] = []
    for symbol, ysym in zip(symbols, yahoo_symbols, strict=True):
        data = _flatten_download_columns(frame, ysym)
        if data.empty or "Close" not in data.columns:
            continue
        data = data.dropna(subset=["Close"]).copy()
        if data.empty:
            continue
        data.index = pd.to_datetime(data.index)
        latest_ts = data.index[-1]
        latest_date = latest_ts.date()
        session = data[data.index.date == latest_date]
        if session.empty:
            session = data.tail(1)

        latest = session.iloc[-1]
        close = float(latest["Close"])
        open_price = (
            float(session["Open"].dropna().iloc[0])
            if session["Open"].notna().any()
            else np.nan
        )
        day_high = float(session["High"].max())
        day_low = float(session["Low"].min())
        day_volume = float(session["Volume"].fillna(0).sum())

        prior_sessions = data.loc[data.index.date < latest_date]
        if not prior_sessions.empty:
            previous_close = float(prior_sessions["Close"].dropna().iloc[-1])
        else:
            previous_close = float(close)

        change = close - previous_close
        change_pct = change / previous_close * 100 if previous_close else 0.0

        rows.append(
            {
                "symbol": symbol,
                "price": close,
                "previous_close": previous_close,
                "change": change,
                "change_pct": change_pct,
                "open": open_price,
                "day_high": day_high,
                "day_low": day_low,
                "session_volume": day_volume,
                "timestamp": latest_ts,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["fetched_at_utc"] = datetime.now(timezone.utc)
    return result.sort_values("symbol").reset_index(drop=True)


def fetch_intraday_history(
    symbol: str,
    period: str = "5d",
    interval: str = "1m",
) -> pd.DataFrame:
    """Fetch intraday OHLCV bars for online RL catch-up/replay."""
    frame = yf.download(
        yahoo_symbol(symbol),
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        return pd.DataFrame(
            columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
        )

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename_axis("timestamp").reset_index()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{symbol}: missing intraday fields {missing}")

    data = data[["timestamp", *required]].copy()
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce").fillna(0.0)
    for column in required[:4]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return (
        data.dropna(subset=["Open", "High", "Low", "Close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def fetch_daily_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    """Fetch daily history for model input and technical analysis."""
    frame = yf.download(
        yahoo_symbol(symbol),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError(f"No daily history returned for {symbol}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename_axis("Date").reset_index()
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True).dt.tz_convert(None)
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{symbol}: missing daily fields {missing}")
    frame = frame[columns].dropna(subset=["Open", "High", "Low", "Close"])
    frame["Volume"] = frame["Volume"].fillna(0)
    return frame.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
