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
- Reproducible feature pipeline in `src/data/pipeline.py`

## Day 2 feature engineering and model foundation

Day 2 extends the pipeline with:

- TA-Lib technical indicators
- The complete TA-Lib CDL candlestick-pattern family loaded dynamically
- Heuristic support/resistance, breakouts, reversals, triangles, wedges, and flag features
- A multi-task PyTorch LSTM backbone for return, direction, and volatility outputs
- An LSTM-DQN reinforcement-learning policy with Sell, Hold, and Buy actions
- A causal trading environment with transaction-cost penalties
- Replay-buffer training and target-network updates
- Unit tests for feature generation, model shapes, and environment behavior

Chart-pattern detection is deliberately implemented as reproducible heuristics rather than presented as a universal or exhaustive list, because chart-pattern taxonomies vary.

## Day 2 training

After generating a processed parquet file locally:

```bash
python -m src.data.pipeline
python scripts/train_rl.py --data data/processed/reliance.parquet
```

The RL checkpoint is written to:

```
models/tickerarc_lstm_dqn.pt
```

Generated datasets and model checkpoints are excluded from Git history by default.
