"""Simple baselines for final 2-year holdout benchmarking."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.trainer import (
    FEATURE_COLUMNS,
    MAX_TARGET_HORIZON,
    split_time_window,
)


def _row_dataset(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows: list[pd.DataFrame] = []
    test_rows: list[pd.DataFrame] = []

    for frame in frames.values():
        development, evaluation, _ = split_time_window(frame)

        train = development.iloc[:-MAX_TARGET_HORIZON].copy()
        test = evaluation.copy()

        train_rows.append(train)
        test_rows.append(test)

    if not train_rows or not test_rows:
        raise ValueError("No benchmark rows are available.")

    train_data = pd.concat(train_rows, ignore_index=True)
    test_data = pd.concat(test_rows, ignore_index=True)

    train_data = train_data.replace([np.inf, -np.inf], np.nan)
    test_data = test_data.replace([np.inf, -np.inf], np.nan)

    train_data = train_data.dropna(
        subset=FEATURE_COLUMNS + ["target_return_5d", "direction_class"]
    )
    test_data = test_data.dropna(
        subset=FEATURE_COLUMNS + ["target_return_5d", "direction_class"]
    )

    if train_data.empty or test_data.empty:
        raise ValueError("Benchmark dataset is empty after cleaning.")

    return train_data, test_data


def _metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    actual_direction = np.where(
        actual > 0.005,
        2,
        np.where(actual < -0.005, 0, 1),
    )
    predicted_direction = np.where(
        predicted > 0.005,
        2,
        np.where(predicted < -0.005, 0, 1),
    )

    return {
        "mae_5d": float(np.mean(np.abs(actual - predicted))),
        "direction_accuracy": float(
            np.mean(actual_direction == predicted_direction)
        ),
    }


def evaluate_baselines(
    frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float]]:
    """Compare zero-return and Ridge baselines on the untouched latest 2Y."""
    train_data, test_data = _row_dataset(frames)

    x_train = train_data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_data["target_return_5d"].to_numpy(dtype=np.float32)
    x_test = test_data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_data["target_return_5d"].to_numpy(dtype=np.float32)

    zero_prediction = np.zeros_like(y_test)
    ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    ridge.fit(x_train, y_train)
    ridge_prediction = ridge.predict(x_test).astype(np.float32)

    metrics = {
        "zero_return": _metrics(y_test, zero_prediction),
        "ridge": _metrics(y_test, ridge_prediction),
    }
    return metrics
