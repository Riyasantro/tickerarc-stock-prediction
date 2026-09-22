"""Train the Day 2 LSTM-DQN agent on one processed stock file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.trainer import EVAL_YEARS, TRAIN_YEARS

from src.rl.recurrent_dqn import RecurrentDQNAgent, ReplayBuffer, train_episode
from src.rl.trading_env import TradingEnv


DEFAULT_FEATURES = [
    "return_1d", "return_5d", "return_10d",
    "rsi_14", "macd", "macd_hist", "atr_14", "adx_14",
    "cci_14", "stoch_k", "stoch_d", "willr_14",
    "volume_ratio_20", "volatility_20d", "momentum_10d",
    "momentum_20d", "bb_position",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Processed parquet file")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--output", default="models/tickerarc_lstm_dqn.pt")
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    feature_columns = [name for name in DEFAULT_FEATURES if name in frame.columns]
    if not feature_columns:
        raise ValueError("No expected model features found in the processed dataframe.")

    frame = frame.dropna(subset=feature_columns + ["return_1d"]).reset_index(drop=True)
    dates = pd.to_datetime(frame["Date"])
    latest = dates.max()
    eval_start = latest - pd.DateOffset(years=EVAL_YEARS)
    train_start = eval_start - pd.DateOffset(years=TRAIN_YEARS)

    train = frame[
        (dates >= train_start) & (dates < eval_start)
    ].reset_index(drop=True)
    if len(train) <= args.sequence_length + 1:
        raise ValueError(
            f"Need at least {TRAIN_YEARS} years of training data "
            f"before the latest {EVAL_YEARS}-year holdout."
        )
    env = TradingEnv(
        train,
        feature_columns=feature_columns,
        sequence_length=args.sequence_length,
    )

    agent = RecurrentDQNAgent(input_size=len(feature_columns))
    buffer = ReplayBuffer()
    rewards: list[float] = []

    for episode in range(args.episodes):
        fraction = episode / max(args.episodes - 1, 1)
        epsilon = max(1.0 - 0.95 * fraction, 0.05)
        metrics = train_episode(env, agent, buffer, epsilon=epsilon)
        rewards.append(metrics["reward"])
        print(
            f"episode={episode + 1:03d} "
            f"reward={metrics['reward']:.6f} "
            f"loss={metrics['loss']:.6f} "
            f"epsilon={epsilon:.3f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": agent.policy.state_dict(),
            "feature_columns": feature_columns,
            "sequence_length": args.sequence_length,
            "episodes": args.episodes,
            "train_years": TRAIN_YEARS,
            "eval_years": EVAL_YEARS,
            "train_start": str(train_start.date()),
            "eval_start": str(eval_start.date()),
            "eval_end": str(latest.date()),
        },
        output,
    )
    print(f"saved={output}")
    print(f"mean_episode_reward={float(np.mean(rewards)):.6f}")


if __name__ == "__main__":
    main()
