"""Market-data downloader for TickerArc."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from config.universe import NIFTY50_SYMBOLS, yahoo_symbol

RAW_DIR = Path("data/raw")
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _normalise_columns(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        raise ValueError(f"No data returned for {symbol}")

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        if set(REQUIRED_COLUMNS).issubset(level0):
            data.columns = level0
        else:
            data = data.xs(data.columns.get_level_values(1)[0], axis=1, level=1)

    data = data.rename_axis("Date").reset_index()
    data["Date"] = pd.to_datetime(data["Date"], utc=True).dt.tz_convert(None)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    data = data[["Date", *REQUIRED_COLUMNS]].copy()
    data = data.sort_values("Date").drop_duplicates("Date")
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data["Volume"] = data["Volume"].fillna(0)

    for column in REQUIRED_COLUMNS[:4]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce").fillna(0)

    return data.reset_index(drop=True)


def download_history(
    symbol: str,
    *,
    period: str = "5y",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download one NSE symbol from Yahoo Finance."""
    frame = yf.download(
        yahoo_symbol(symbol),
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    return _normalise_columns(frame, symbol)


def save_history(data: pd.DataFrame, symbol: str, output_dir: Path = RAW_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = symbol.lower().replace("&", "_and_").replace("-", "_")
    path = output_dir / f"{filename}.parquet"
    data.to_parquet(path, index=False)
    return path


def download_universe(
    symbols: Iterable[str] = NIFTY50_SYMBOLS,
    *,
    period: str = "5y",
    interval: str = "1d",
    output_dir: Path = RAW_DIR,
) -> dict[str, Path]:
    """Download the selected universe; continue after individual failures."""
    paths: dict[str, Path] = {}
    failures: dict[str, str] = {}

    for symbol in symbols:
        try:
            data = download_history(symbol, period=period, interval=interval)
            paths[symbol] = save_history(data, symbol, output_dir)
            print(f"[OK] {symbol}: {len(data):,} rows -> {paths[symbol]}")
        except Exception as exc:  # noqa: BLE001
            failures[symbol] = str(exc)
            print(f"[FAIL] {symbol}: {exc}")

    if failures:
        print("\nDownload failures:")
        for symbol, reason in failures.items():
            print(f"- {symbol}: {reason}")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TickerArc market data")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--symbol", action="append", dest="symbols")
    args = parser.parse_args()

    symbols = args.symbols or NIFTY50_SYMBOLS
    download_universe(symbols, period=args.period, interval=args.interval)


if __name__ == "__main__":
    main()
