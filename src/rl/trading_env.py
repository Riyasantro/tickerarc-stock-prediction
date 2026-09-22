"""Simple causal long/flat/short trading environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, float | int]


class TradingEnv:
    ACTION_SELL = 0
    ACTION_HOLD = 1
    ACTION_BUY = 2

    def __init__(
        self,
        features: pd.DataFrame,
        feature_columns: list[str],
        sequence_length: int = 60,
        transaction_cost: float = 0.0005,
    ) -> None:
        if len(features) <= sequence_length + 1:
            raise ValueError("Not enough rows for the selected sequence length.")
        missing = set(feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        if "return_1d" not in features.columns:
            raise ValueError("TradingEnv requires a return_1d column.")

        self.data = features.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.transaction_cost = transaction_cost
        self._cursor = sequence_length

        values = self.data[feature_columns].replace([np.inf, -np.inf], np.nan)
        self.values = values.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        self.returns = self.data["return_1d"].fillna(0.0).to_numpy(dtype=np.float32)

    def reset(self, start: int | None = None) -> np.ndarray:
        self._cursor = self.sequence_length if start is None else max(
            self.sequence_length, start
        )
        return self._observation()

    def _observation(self) -> np.ndarray:
        start = self._cursor - self.sequence_length
        return self.values[start:self._cursor]

    def step(self, action: int) -> StepResult:
        if action not in (self.ACTION_SELL, self.ACTION_HOLD, self.ACTION_BUY):
            raise ValueError("Action must be 0 (sell), 1 (hold), or 2 (buy).")

        position = {
            self.ACTION_SELL: -1,
            self.ACTION_HOLD: 0,
            self.ACTION_BUY: 1,
        }[action]
        next_return = float(self.returns[self._cursor])
        reward = position * next_return - self.transaction_cost * abs(position)

        self._cursor += 1
        done = self._cursor >= len(self.data) - 1

        return StepResult(
            observation=self._observation(),
            reward=float(reward),
            done=done,
            info={
                "position": position,
                "next_return": next_return,
                "equity_return": position * next_return,
            },
        )
