"""Global multi-stock LSTM training and inference utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.global_lstm import MultiHorizonLSTM

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "sma_10", "sma_20", "sma_50", "ema_10", "ema_20",
    "rsi_14", "macd", "macd_hist", "atr_14", "natr_14",
    "adx_14", "plus_di_14", "minus_di_14", "cci_14", "stoch_k", "stoch_d",
    "willr_14", "roc_10", "mom_10", "trix_30", "mfi_14", "adosc",
    "volume_ratio_20", "volatility_10d", "volatility_20d", "volatility_60d",
    "momentum_10d", "momentum_20d", "bb_width", "bb_position",
    "breakout_20", "breakdown_20", "double_top_30", "double_bottom_30",
    "head_shoulders_30", "inverse_head_shoulders_30",
    "ascending_triangle_20", "descending_triangle_20", "symmetrical_triangle_20",
    "rising_wedge_20", "falling_wedge_20", "bull_flag", "bear_flag",
    "candlestick_bullish_count", "candlestick_bearish_count",
    "candlestick_signal_score",
]

HORIZONS = (1, 5, 10)
TRAIN_YEARS = 24
EVAL_YEARS = 2
MAX_TARGET_HORIZON = max(HORIZONS)
SEQUENCE_LENGTH_DEFAULT = 60


@dataclass
class SequenceSample:
    x: np.ndarray
    y_return: np.ndarray
    y_direction: int
    y_volatility: float


class WindowDataset(Dataset):
    def __init__(self, samples: list[SequenceSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]
        return (
            torch.tensor(item.x, dtype=torch.float32),
            torch.tensor(item.y_return, dtype=torch.float32),
            torch.tensor(item.y_direction, dtype=torch.long),
            torch.tensor(item.y_volatility, dtype=torch.float32),
        )


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_values("Date").reset_index(drop=True)
    data = data.replace([np.inf, -np.inf], np.nan)

    missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    for horizon in HORIZONS:
        data[f"target_return_{horizon}d"] = (
            data["Close"].shift(-horizon) / data["Close"] - 1.0
        )

    future_abs = pd.concat(
        [data["return_1d"].shift(-i).abs() for i in range(1, 6)],
        axis=1,
    )
    data["target_volatility_5d"] = future_abs.mean(axis=1)
    data["direction_class"] = np.select(
        [
            data["target_return_5d"] > 0.005,
            data["target_return_5d"] < -0.005,
        ],
        [2, 0],
        default=1,
    ).astype(float)
    return data


def load_training_frames(
    data_dir: Path, symbols: list[str]
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = data_dir / (
            f"{symbol.lower().replace('&', '_and_').replace('-', '_')}.parquet"
        )
        if not path.exists():
            continue
        frames[symbol] = prepare_frame(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError(
            "No processed parquet files found. Run the Day 1 pipeline first."
        )
    return frames


def split_time_window(
    frame: pd.DataFrame,
    train_years: int = TRAIN_YEARS,
    eval_years: int = EVAL_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Split one stock into a chronological 24-year train window and latest 2-year holdout."""
    data = prepare_frame(frame)
    dates = pd.to_datetime(data["Date"])
    latest = dates.max()
    eval_start = latest - pd.DateOffset(years=eval_years)
    train_start = eval_start - pd.DateOffset(years=train_years)

    train = data[(dates >= train_start) & (dates < eval_start)].copy()
    evaluation = data[dates >= eval_start].copy()
    bounds = {
        "train_start": str(train_start.date()),
        "train_end": str((eval_start - pd.Timedelta(days=1)).date()),
        "eval_start": str(eval_start.date()),
        "eval_end": str(latest.date()),
    }
    return train, evaluation, bounds


def _clean_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
    )


def _eligible_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    train, _, _ = split_time_window(frame)
    if len(train) <= SEQUENCE_LENGTH_DEFAULT + MAX_TARGET_HORIZON + 20:
        return pd.DataFrame()
    # Do not allow a training target to cross into the holdout period.
    return train.iloc[:-MAX_TARGET_HORIZON].copy()


def fit_time_split_scaler(
    frames: dict[str, pd.DataFrame],
) -> tuple[StandardScaler, dict[str, dict[str, str]]]:
    chunks = []
    split_info: dict[str, dict[str, str]] = {}

    for symbol, frame in frames.items():
        train = _eligible_training_frame(frame)
        if train.empty:
            continue
        chunks.append(_clean_feature_frame(train))
        _, _, bounds = split_time_window(frame)
        split_info[symbol] = bounds

    if not chunks:
        raise ValueError(
            f"No stocks contain enough history for {TRAIN_YEARS} training years "
            f"plus {EVAL_YEARS} evaluation years."
        )

    scaler = StandardScaler()
    scaler.fit(pd.concat(chunks, ignore_index=True))
    return scaler, split_info


def build_time_split_samples(
    frames: dict[str, pd.DataFrame],
    scaler: StandardScaler,
    sequence_length: int = SEQUENCE_LENGTH_DEFAULT,
) -> list[SequenceSample]:
    samples: list[SequenceSample] = []

    for frame in frames.values():
        train = _eligible_training_frame(frame)
        if train.empty or len(train) <= sequence_length:
            continue

        values = scaler.transform(_clean_feature_frame(train)).astype(np.float32)

        for end in range(sequence_length, len(train)):
            row = train.iloc[end]
            target_cols = [
                *(f"target_return_{h}d" for h in HORIZONS),
                "target_volatility_5d",
                "direction_class",
            ]
            if not np.isfinite(row[target_cols].astype(float).to_numpy()).all():
                continue

            samples.append(
                SequenceSample(
                    x=values[end - sequence_length:end],
                    y_return=np.array(
                        [float(row[f"target_return_{h}d"]) for h in HORIZONS],
                        dtype=np.float32,
                    ),
                    y_direction=int(row["direction_class"]),
                    y_volatility=float(row["target_volatility_5d"]),
                )
            )

    return samples


def build_samples(
    frames: dict[str, pd.DataFrame],
    scaler: StandardScaler,
    sequence_length: int,
    train_fraction: float = 0.8,
) -> list[SequenceSample]:
    """Legacy fractional splitter retained for compatibility."""
    samples: list[SequenceSample] = []

    for frame in frames.values():
        split = int(len(frame) * train_fraction)
        values = scaler.transform(_clean_feature_frame(frame)).astype(np.float32)

        for end in range(sequence_length, split):
            row = frame.iloc[end]
            target_cols = [
                *(f"target_return_{h}d" for h in HORIZONS),
                "target_volatility_5d",
                "direction_class",
            ]
            if not np.isfinite(row[target_cols].astype(float).to_numpy()).all():
                continue

            samples.append(
                SequenceSample(
                    x=values[end - sequence_length:end],
                    y_return=np.array(
                        [float(row[f"target_return_{h}d"]) for h in HORIZONS],
                        dtype=np.float32,
                    ),
                    y_direction=int(row["direction_class"]),
                    y_volatility=float(row["target_volatility_5d"]),
                )
            )

    return samples

def fit_scaler(
    frames: dict[str, pd.DataFrame], train_fraction: float = 0.8
) -> StandardScaler:
    chunks = []
    for frame in frames.values():
        split = int(len(frame) * train_fraction)
        chunks.append(
            frame.iloc[:split][FEATURE_COLUMNS]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
    scaler = StandardScaler()
    scaler.fit(pd.concat(chunks, ignore_index=True))
    return scaler


def train_global_model(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    epochs: int = 8,
    sequence_length: int = SEQUENCE_LENGTH_DEFAULT,
    batch_size: int = 128,
    hidden_size: int = 128,
    lr: float = 1e-3,
    device: str = "auto",
    progress: Callable[[int, float], None] | None = None,
) -> dict[str, object]:
    """Train only on the 24-year window before the 2-year holdout."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler, split_info = fit_time_split_scaler(frames)
    samples = build_time_split_samples(frames, scaler, sequence_length)
    if not samples:
        raise ValueError("No valid 24-year LSTM training windows were produced.")

    loader = DataLoader(
        WindowDataset(samples),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    device_obj = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )

    model = MultiHorizonLSTM(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=hidden_size,
    ).to(device_obj)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    huber = nn.HuberLoss()
    cross_entropy = nn.CrossEntropyLoss()

    best_loss = float("inf")
    stale = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        count = 0

        for xb, yret, ydir, yvol in loader:
            xb = xb.to(device_obj)
            yret = yret.to(device_obj)
            ydir = ydir.to(device_obj)
            yvol = yvol.to(device_obj)

            out = model(xb)
            loss = (
                huber(out["returns"], yret)
                + 0.50 * cross_entropy(out["direction_logits"], ydir)
                + 0.25 * huber(out["volatility"], yvol)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * len(xb)
            count += len(xb)

        epoch_loss = total_loss / max(count, 1)
        if progress:
            progress(epoch + 1, epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            stale = 0
            torch.save(model.state_dict(), output_dir / "tickerarc_global_lstm.pt")
        else:
            stale += 1
            if stale >= 2:
                break

    joblib.dump(scaler, output_dir / "tickerarc_scaler.joblib")
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "horizons": HORIZONS,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "train_loss": best_loss,
        "device": str(device_obj),
        "train_years": TRAIN_YEARS,
        "eval_years": EVAL_YEARS,
        "split_info": split_info,
        "training_stocks": sorted(split_info),
    }
    (output_dir / "tickerarc_model_meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def evaluate_global_model(
    model: MultiHorizonLSTM,
    scaler: StandardScaler,
    frames: dict[str, pd.DataFrame],
    sequence_length: int = SEQUENCE_LENGTH_DEFAULT,
) -> dict[str, float | int | str]:
    """Evaluate the frozen model on the latest two years only."""
    model.eval()
    predictions: list[float] = []
    actuals: list[float] = []
    correct = 0
    total = 0
    stocks = 0
    eval_starts: list[str] = []
    eval_ends: list[str] = []
    device_obj = next(model.parameters()).device

    for frame in frames.values():
        data = prepare_frame(frame)
        _, evaluation, bounds = split_time_window(data)
        if evaluation.empty:
            continue

        eval_start = pd.Timestamp(bounds["eval_start"])
        dates = pd.to_datetime(data["Date"])
        eval_positions = np.flatnonzero(
            dates.to_numpy() >= eval_start.to_datetime64()
        )
        if len(eval_positions) == 0:
            continue

        symbol_samples = 0
        for end in eval_positions:
            if end + MAX_TARGET_HORIZON >= len(data) or end + 1 < sequence_length:
                continue

            row = data.iloc[end]
            target_cols = [
                *(f"target_return_{h}d" for h in HORIZONS),
                "target_volatility_5d",
                "direction_class",
            ]
            if not np.isfinite(row[target_cols].astype(float).to_numpy()).all():
                continue

            context = _clean_feature_frame(
                data.iloc[end - sequence_length + 1:end + 1]
            )
            scaled = scaler.transform(context).astype(np.float32)
            x = torch.tensor(scaled[None, ...], dtype=torch.float32, device=device_obj)

            with torch.no_grad():
                out = model(x)
                pred = float(out["returns"][0, 1].item())
                direction = int(torch.argmax(out["direction_logits"], dim=1).item())

            predictions.append(pred)
            actuals.append(float(row["target_return_5d"]))
            correct += int(direction == int(row["direction_class"]))
            total += 1
            symbol_samples += 1

        if symbol_samples:
            stocks += 1
            eval_starts.append(bounds["eval_start"])
            eval_ends.append(bounds["eval_end"])

    if not predictions:
        raise ValueError("No valid 2-year holdout samples were produced.")

    return {
        "train_years": TRAIN_YEARS,
        "eval_years": EVAL_YEARS,
        "evaluation_window": f"{min(eval_starts)} to {max(eval_ends)}",
        "stocks_evaluated": stocks,
        "samples": total,
        "mae_5d": float(mean_absolute_error(actuals, predictions)),
        "direction_accuracy": float(correct / max(total, 1)),
    }

def load_model(
    output_dir: Path,
    device: str = "auto",
) -> tuple[MultiHorizonLSTM, StandardScaler, dict[str, object]]:
    metadata = json.loads(
        (output_dir / "tickerarc_model_meta.json").read_text(
            encoding="utf-8"
        )
    )
    device_obj = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    model = MultiHorizonLSTM(
        input_size=len(metadata["feature_columns"]),
        hidden_size=int(metadata["hidden_size"]),
    ).to(device_obj)
    model.load_state_dict(
        torch.load(
            output_dir / "tickerarc_global_lstm.pt",
            map_location=device_obj,
            weights_only=True,
        )
    )
    model.eval()
    scaler = joblib.load(output_dir / "tickerarc_scaler.joblib")
    return model, scaler, metadata


def predict_latest(
    model: MultiHorizonLSTM,
    scaler: StandardScaler,
    frame: pd.DataFrame,
    metadata: dict[str, object],
) -> dict[str, float | str]:
    data = prepare_frame(frame)
    columns = list(metadata["feature_columns"])
    usable = (
        data[columns]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    sequence_length = int(metadata["sequence_length"])
    if len(usable) < sequence_length:
        raise ValueError("Not enough history for LSTM inference.")

    scaled = scaler.transform(usable.tail(sequence_length)).astype(np.float32)
    device_obj = next(model.parameters()).device

    with torch.no_grad():
        out = model(
            torch.tensor(
                scaled,
                dtype=torch.float32,
                device=device_obj,
            ).unsqueeze(0)
        )
        returns = out["returns"][0].cpu().numpy()
        probabilities = torch.softmax(
            out["direction_logits"], dim=1
        )[0].cpu().numpy()
        volatility = float(out["volatility"][0].cpu().item())

    direction_names = ["DOWN", "NEUTRAL", "UP"]
    direction_index = int(np.argmax(probabilities))

    return {
        "return_1d": float(returns[0]),
        "return_5d": float(returns[1]),
        "return_10d": float(returns[2]),
        "down_probability": float(probabilities[0]),
        "neutral_probability": float(probabilities[1]),
        "up_probability": float(probabilities[2]),
        "direction": direction_names[direction_index],
        "volatility_5d": volatility,
        "model_name": "TickerArc Multi-Horizon LSTM v1",
    }
