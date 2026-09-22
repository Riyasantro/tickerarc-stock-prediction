"""Time-ordered walk-forward evaluation for a selected stock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from src.models.global_lstm import MultiHorizonLSTM
from src.models.trainer import FEATURE_COLUMNS, HORIZONS, fit_scaler, prepare_frame


def walk_forward_evaluate(
    frame: pd.DataFrame,
    folds: int = 3,
    train_epochs: int = 3,
    sequence_length: int = 60,
) -> dict[str, float | int]:
    """Retrain on expanding historical windows and score later blocks only."""
    data = (
        prepare_frame(frame)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(
            subset=FEATURE_COLUMNS
            + [f"target_return_{h}d" for h in HORIZONS]
            + ["direction_class", "target_volatility_5d"]
        )
        .reset_index(drop=True)
    )

    n = len(data)
    if n < sequence_length + folds * 50:
        raise ValueError("Not enough data for the requested walk-forward folds.")

    results: list[dict[str, float]] = []

    for fold in range(folds):
        train_end = int(n * (0.55 + 0.10 * fold))
        test_end = int(n * (0.65 + 0.10 * fold)) if fold < folds - 1 else n
        if test_end <= train_end:
            continue

        scaler = fit_scaler(
            {"stock": data.iloc[:train_end].copy()},
            train_fraction=0.999,
        )
        train_scaled = scaler.transform(
            data.iloc[:train_end][FEATURE_COLUMNS]
        ).astype(np.float32)

        model = MultiHorizonLSTM(input_size=len(FEATURE_COLUMNS))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return_loss = torch.nn.HuberLoss()
        direction_loss = torch.nn.CrossEntropyLoss()

        model.train()
        for _ in range(train_epochs):
            for end in range(sequence_length, train_end):
                row = data.iloc[end]
                x = torch.tensor(
                    train_scaled[end - sequence_length:end][None, ...],
                    dtype=torch.float32,
                )
                y_return = torch.tensor(
                    [[
                        float(row[f"target_return_{h}d"])
                        for h in HORIZONS
                    ]],
                    dtype=torch.float32,
                )
                y_direction = torch.tensor(
                    [int(row["direction_class"])],
                    dtype=torch.long,
                )

                out = model(x)
                loss = (
                    return_loss(out["returns"], y_return)
                    + 0.5 * direction_loss(
                        out["direction_logits"], y_direction
                    )
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        predictions: list[float] = []
        actuals: list[float] = []
        correct = 0
        total = 0

        for end in range(train_end, test_end):
            context = data.iloc[:end][FEATURE_COLUMNS]
            context_scaled = scaler.transform(context).astype(np.float32)
            if len(context_scaled) < sequence_length:
                continue

            x = torch.tensor(
                context_scaled[-sequence_length:][None, ...],
                dtype=torch.float32,
            )

            with torch.no_grad():
                out = model(x)
                pred = float(out["returns"][0, 1].item())
                direction = int(
                    torch.argmax(out["direction_logits"], dim=1).item()
                )

            actual = float(data.iloc[end]["target_return_5d"])
            actual_class = int(data.iloc[end]["direction_class"])
            predictions.append(pred)
            actuals.append(actual)
            correct += int(direction == actual_class)
            total += 1

        if predictions:
            results.append(
                {
                    "mae_5d": float(mean_absolute_error(actuals, predictions)),
                    "direction_accuracy": float(correct / max(total, 1)),
                    "samples": float(total),
                }
            )

    if not results:
        raise ValueError("No valid walk-forward folds were produced.")

    return {
        "folds": len(results),
        "mae_5d": float(np.mean([r["mae_5d"] for r in results])),
        "direction_accuracy": float(
            np.mean([r["direction_accuracy"] for r in results])
        ),
        "samples": int(sum(r["samples"] for r in results)),
    }
