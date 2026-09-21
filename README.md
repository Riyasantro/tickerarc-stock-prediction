# TickerArc

**Stock Analysis and Prediction System**

TickerArc is a stock market analysis and prediction project focused on combining market data, technical features, and deep learning to study future stock movement.

## Project scope

- Historical market data pipeline
- Price, volume, momentum, and volatility features
- Activity and volume-based stock classification
- LSTM-based future movement prediction
- Candlestick and chart-pattern analysis
- Clean black-and-white dashboard
- Agent-assisted stock analysis

## Development plan

The project is being built incrementally over five days:

1. Data collection and feature pipeline
2. LSTM prediction model
3. Market intelligence and agent tools
4. Dashboard
5. Testing, evaluation, and documentation

## Day 1 foundation

Day 1 establishes the reusable market-data and feature pipeline:

- NIFTY 50 stock universe in `config/universe.py`
- Yahoo Finance OHLCV downloader in `src/data/download.py`
- Causal technical features in `src/features/technical.py`
- End-to-end download and feature pipeline in `src/data/pipeline.py`
- Feature unit tests in `tests/test_features.py`

Generated market data is intentionally excluded from Git history so it can be reproduced locally.

## Run

```bash
python -m pip install -r requirements.txt
python -m src.data.pipeline
```

To generate features from already-downloaded raw parquet files:

```bash
python -m src.data.pipeline --skip-download
```
