"""Evaluate a trained TickerArc LSTM-DQN checkpoint on a held-out time split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.rl.recurrent_dqn import RecurrentDQNAgent
from src.rl.trading_env import TradingEnv


DEFAULT_FEATURES = [
    "return_1d", "return_5d", "return_10d",
    "rsi_14", "macd", "macd_hist", "atr_14", "adx_14",
    "cci_14", "stoch_k", "stoch_d", "willr_14",
    "volume_ratio_20", "volatility_20d", "momentum_10d",
    "momentum_20d", "bb_position",
]


def max_drawdown(returns: list[float]) -> float:
    if not returns:
        return 0.0
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(equity)
    drawdowns = equity / peak - 1.0
    return float(drawdowns.min())


def evaluate(
    data_path: str,
    checkpoint_path: str,
    output_path: str,
) -> dict[str, float | int | str]:
    frame = pd.read_parquet(data_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    feature_columns = list(checkpoint["feature_columns"])
    sequence_length = int(checkpoint["sequence_length"])

    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing checkpoint features: {sorted(missing)}")

    frame = frame.dropna(
        subset=feature_columns + ["return_1d"]
    ).reset_index(drop=True)

    dates = pd.to_datetime(frame["Date"])
    latest = dates.max()
    eval_start = latest - pd.DateOffset(years=2)
    eval_positions = np.flatnonzero(
        dates.to_numpy() >= eval_start.to_datetime64()
    )
    if len(eval_positions) == 0:
        raise ValueError("No latest 2-year holdout rows are available.")

    eval_index = int(eval_positions[0])
    if eval_index < sequence_length:
        raise ValueError("Not enough pre-holdout context for the configured sequence length.")

    # Keep exactly one sequence of look-back context, then evaluate only from
    # the first row of the latest 2-year holdout onward.
    test_frame = frame.iloc[eval_index - sequence_length:].reset_index(drop=True)

    env = TradingEnv(
        test_frame,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
    )
    agent = RecurrentDQNAgent(input_size=len(feature_columns))
    agent.policy.load_state_dict(checkpoint["model_state_dict"])
    agent.target.load_state_dict(checkpoint["model_state_dict"])

    state = env.reset()
    strategy_returns: list[float] = []
    benchmark_returns: list[float] = []
    actions: list[int] = []

    done = False
    while not done:
        action = agent.select_action(state, epsilon=0.0)
        result = env.step(action)
        strategy_returns.append(result.info["equity_return"])
        benchmark_returns.append(result.info["next_return"])
        actions.append(action)
        state = result.observation
        done = result.done

    strategy_equity = float(np.prod(1.0 + np.asarray(strategy_returns)))
    benchmark_equity = float(np.prod(1.0 + np.asarray(benchmark_returns)))

    active = np.asarray(strategy_returns)
    wins = int((active > 0).sum())
    nonzero = int((active != 0).sum())

    metrics = {
        "data_file": data_path,
        "train_years": 24,
        "eval_years": 2,
        "evaluation_window": f"{eval_start.date()} to {latest.date()}",
        "checkpoint": checkpoint_path,
        "test_rows": int(len(strategy_returns)),
        "strategy_return": float(strategy_equity - 1.0),
        "buy_and_hold_return": float(benchmark_equity - 1.0),
        "outperformance": float(strategy_equity - benchmark_equity),
        "max_drawdown": max_drawdown(strategy_returns),
        "trade_actions": int(sum(action in (TradingEnv.ACTION_SELL, TradingEnv.ACTION_BUY) for action in actions)),
        "buy_actions": int(actions.count(TradingEnv.ACTION_BUY)),
        "sell_actions": int(actions.count(TradingEnv.ACTION_SELL)),
        "hold_actions": int(actions.count(TradingEnv.ACTION_HOLD)),
        "win_rate": float(wins / nonzero) if nonzero else 0.0,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="models/tickerarc_lstm_dqn.pt")
    parser.add_argument("--output", default="models/tickerarc_lstm_dqn_eval.json")
    args = parser.parse_args()

    metrics = evaluate(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )

    print("\nEvaluation")
    print("----------")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
