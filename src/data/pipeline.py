"""End-to-end Day 1 data pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config.universe import NIFTY50_SYMBOLS
from src.data.download import RAW_DIR, download_universe
from src.features.technical import build_features

PROCESSED_DIR = Path("data/processed")


def process_file(raw_path: Path, output_dir: Path = PROCESSED_DIR) -> Path:
    data = pd.read_parquet(raw_path)
    featured = build_features(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / raw_path.name
    featured.to_parquet(output_path, index=False)
    print(f"[FEATURES] {raw_path.stem}: {len(featured):,} rows -> {output_path}")
    return output_path


def process_downloaded_files(output_dir: Path = PROCESSED_DIR) -> list[Path]:
    return [process_file(path, output_dir) for path in sorted(RAW_DIR.glob("*.parquet"))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TickerArc Day 1 data pipeline")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        download_universe(NIFTY50_SYMBOLS, period=args.period, interval=args.interval)

    process_downloaded_files()


if __name__ == "__main__":
    main()
