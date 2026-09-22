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
from sklearn.metrics import mean_absolute_error
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
    "regime_trend_50", "regime_trend_200",
    "regime_volatility_ratio", "drawdown_60d",
    "volume_pressure_20d", "return_skew_20d",
    "regime_risk_on", "regime_risk_off",
]

HORIZONS = (1, 5, 10)
TRAIN_YEARS = 24
VALIDATION_YEARS = 4
FIT_YEARS = TRAIN_YEARS - VALIDATION_YEARS
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
        .fillna(0.0)
    )


def split_development_window(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Split the 24-year development window into 20Y fit + 4Y validation."""
    development, _, bounds = split_time_window(frame)
    dates = pd.to_datetime(development["Date"])
    validation_start = pd.Timestamp(bounds["eval_start"]) - pd.DateOffset(
        years=VALIDATION_YEARS
    )
    fit = development[dates < validation_start].copy()
    validation = development[dates >= validation_start].copy()
    bounds = {
        **bounds,
        "fit_start": str(pd.to_datetime(fit["Date"]).min().date()) if not fit.empty else "",
        "fit_end": str((validation_start - pd.Timedelta(days=1)).date()),
        "validation_start": str(validation_start.date()),
        "validation_end": str((pd.Timestamp(bounds["eval_start"]) - pd.Timedelta(days=1)).date()),
    }
    return fit, validation, bounds


def _eligible_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    fit, _, _ = split_development_window(frame)
    if len(fit) <= SEQUENCE_LENGTH_DEFAULT + MAX_TARGET_HORIZON + 20:
        return pd.DataFrame()
    return fit.iloc[:-MAX_TARGET_HORIZON].copy()


def _clean_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0.0)
    )


def fit_development_scaler(
    frames: dict[str, pd.DataFrame],
) -> tuple[StandardScaler, dict[str, dict[str, str]]]:
    chunks = []
    split_info: dict[str, dict[str, str]] = {}

    for symbol, frame in frames.items():
        fit, validation, bounds = split_development_window(frame)
        if len(fit) <= SEQUENCE_LENGTH_DEFAULT + MAX_TARGET_HORIZON + 20:
            continue
        chunks.append(_clean_feature_frame(fit.iloc[:-MAX_TARGET_HORIZON]))
        split_info[symbol] = bounds

    if not chunks:
        raise ValueError(
            f"No stocks contain enough history for {FIT_YEARS} fit years, "
            f"{VALIDATION_YEARS} validation years and {EVAL_YEARS} evaluation years."
        )

    scaler = StandardScaler()
    scaler.fit(pd.concat(chunks, ignore_index=True))
    return scaler, split_info


def fit_full_development_scaler(
    frames: dict[str, pd.DataFrame],
) -> StandardScaler:
    chunks = []
    for frame in frames.values():
        development, _, _ = split_time_window(frame)
        if len(development) <= SEQUENCE_LENGTH_DEFAULT + MAX_TARGET_HORIZON + 20:
            continue
        chunks.append(_clean_feature_frame(development.iloc[:-MAX_TARGET_HORIZON]))

    if not chunks:
        raise ValueError("No valid 24-year development windows were produced.")

    scaler = StandardScaler()
    scaler.fit(pd.concat(chunks, ignore_index=True))
    return scaler


def _build_samples_for_range(
    ranges: list[tuple[pd.DataFrame, int, int]],
    scaler: StandardScaler,
    sequence_length: int,
) -> list[SequenceSample]:
    samples: list[SequenceSample] = []

    for data, start_index, end_index in ranges:
        if len(data) <= sequence_length:
            continue

        safe_values = data.copy()
        train_values = safe_values.iloc[:end_index]
        values = scaler.transform(_clean_feature_frame(train_values)).astype(np.float32)

        for end in range(max(sequence_length, start_index), end_index):
            row = safe_values.iloc[end]
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


def build_fit_samples(
    frames: dict[str, pd.DataFrame],
    scaler: StandardScaler,
    sequence_length: int = SEQUENCE_LENGTH_DEFAULT,
) -> list[SequenceSample]:
    ranges = []
    for frame in frames.values():
        fit, _, _ = split_development_window(frame)
        fit = fit.iloc[:-MAX_TARGET_HORIZON].copy()
        ranges.append((fit, sequence_length, len(fit)))
    return _build_samples_for_range(ranges, scaler, sequence_length)


def build_validation_samples(
    frames: dict[str, pd.DataFrame],
    scaler: StandardScaler,
    sequence_length: int = SEQUENCE_LENGTH_DEFAULT,
) -> list[SequenceSample]:
    ranges = []
    for frame in frames.values():
        fit, validation, _ = split_development_window(frame)
        if validation.empty:
            continue
        combined = pd.concat(
            [fit.tail(sequence_length), validation],
            ignore_index=True,
        )
        target_end = max(
            sequence_length,
            len(combined) - MAX_TARGET_HORIZON,
        )
        ranges.append(
            (combined, sequence_length, target_end)
        )

    # Validation uses only fitted scaler statistics. Validation targets never enter
    # the training gradient updates.
    samples = []
    for data, start, end in ranges:
        values = scaler.transform(_clean_feature_frame(data)).astype(np.float32)
        for local_end in range(start, end):
            row = data.iloc[local_end]
            target_cols = [
                *(f"target_return_{h}d" for h in HORIZONS),
                "target_volatility_5d",
                "direction_class",
            ]
            if not np.isfinite(row[target_cols].astype(float).to_numpy()).all():
                continue
            samples.append(
                SequenceSample(
                    x=values[local_end - sequence_length:local_end],
                    y_return=np.array(
                        [float(row[f"target_return_{h}d"]) for h in HORIZONS],
                        dtype=np.float32,
                    ),
                    y_direction=int(row["direction_class"]),
                    y_volatility=float(row["target_volatility_5d"]),
                )
            )
    return samples


def build_full_development_samples(
    frames: dict[str, pd.DataFrame],
    scaler: StandardScaler,
    sequence_length: int = SEQUENCE_LENGTH_DEFAULT,
) -> list[SequenceSample]:
    ranges = []
    for frame in frames.values():
        development, _, _ = split_time_window(frame)
        development = development.iloc[:-MAX_TARGET_HORIZON].copy()
        ranges.append((development, sequence_length, len(development)))
    return _build_samples_for_range(ranges, scaler, sequence_length)


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


def _loss_from_output(
    output: dict[str, torch.Tensor],
    yret: torch.Tensor,
    ydir: torch.Tensor,
    yvol: torch.Tensor,
) -> torch.Tensor:
    huber = nn.HuberLoss()
    cross_entropy = nn.CrossEntropyLoss()
    return (
        huber(output["returns"], yret)
        + 0.50 * cross_entropy(output["direction_logits"], ydir)
        + 0.25 * huber(output["volatility"], yvol)
    )


def _evaluate_samples(
    model: MultiHorizonLSTM,
    samples: list[SequenceSample],
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    if not samples:
        raise ValueError("No validation samples were produced.")

    loader = DataLoader(
        WindowDataset(samples),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    total_loss = 0.0
    count = 0
    correct = 0

    with torch.no_grad():
        for xb, yret, ydir, yvol in loader:
            xb = xb.to(device)
            yret = yret.to(device)
            ydir = ydir.to(device)
            yvol = yvol.to(device)
            output = model(xb)
            loss = _loss_from_output(output, yret, ydir, yvol)
            total_loss += float(loss.item()) * len(xb)
            count += len(xb)
            correct += int(
                (torch.argmax(output["direction_logits"], dim=1) == ydir).sum().item()
            )

    return {
        "loss": total_loss / max(count, 1),
        "direction_accuracy": correct / max(count, 1),
        "samples": float(count),
    }


def _fit_model(
    model: MultiHorizonLSTM,
    samples: list[SequenceSample],
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    progress: Callable[[int, float], None] | None = None,
) -> None:
    loader = DataLoader(
        WindowDataset(samples),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for xb, yret, ydir, yvol in loader:
            xb = xb.to(device)
            yret = yret.to(device)
            ydir = ydir.to(device)
            yvol = yvol.to(device)

            output = model(xb)
            loss = _loss_from_output(output, yret, ydir, yvol)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * len(xb)
            count += len(xb)

        epoch_loss = total_loss / max(count, 1)
        if progress:
            progress(epoch + 1, epoch_loss)


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
    """Select training duration on 20Y fit + 4Y validation, then fit final model on 24Y."""
    output_dir.mkdir(parents=True, exist_ok=True)

    development_scaler, split_info = fit_development_scaler(frames)
    fit_samples = build_fit_samples(frames, development_scaler, sequence_length)
    validation_samples = build_validation_samples(
        frames,
        development_scaler,
        sequence_length,
    )
    if not fit_samples or not validation_samples:
        raise ValueError("No valid fit/validation LSTM windows were produced.")

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

    best_loss = float("inf")
    best_epoch = 1
    patience = 2
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        _fit_model(
            model,
            fit_samples,
            epochs=1,
            batch_size=batch_size,
            optimizer=optimizer,
            device=device_obj,
        )
        metrics = _evaluate_samples(
            model,
            validation_samples,
            device_obj,
        )

        if progress:
            progress(epoch, metrics["loss"])

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise ValueError("Validation did not produce a checkpoint.")

    # Refit on the complete 24-year development window using the selected epoch count.
    full_scaler = fit_full_development_scaler(frames)
    full_samples = build_full_development_samples(
        frames,
        full_scaler,
        sequence_length,
    )
    final_model = MultiHorizonLSTM(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=hidden_size,
    ).to(device_obj)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )
    _fit_model(
        final_model,
        full_samples,
        epochs=best_epoch,
        batch_size=batch_size,
        optimizer=final_optimizer,
        device=device_obj,
    )

    torch.save(
        final_model.state_dict(),
        output_dir / "tickerarc_global_lstm.pt",
    )
    joblib.dump(full_scaler, output_dir / "tickerarc_scaler.joblib")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "horizons": HORIZONS,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "architecture": "LSTM + temporal attention pooling + multi-task heads",
        "train_loss": None,
        "device": str(device_obj),
        "train_years": TRAIN_YEARS,
        "fit_years": FIT_YEARS,
        "validation_years": VALIDATION_YEARS,
        "eval_years": EVAL_YEARS,
        "selected_epochs": best_epoch,
        "validation_loss": best_loss,
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
    """Evaluate the final 24-year development model on the latest 2-year test window."""
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
            x = torch.tensor(
                scaled[None, ...],
                dtype=torch.float32,
                device=device_obj,
            )

            with torch.no_grad():
                out = model(x)
                pred = float(out["returns"][0, 1].item())
                direction = int(
                    torch.argmax(out["direction_logits"], dim=1).item()
                )

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
        "model": "TickerArc Multi-Horizon LSTM v2",
        "train_years": TRAIN_YEARS,
        "fit_years": FIT_YEARS,
        "validation_years": VALIDATION_YEARS,
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
        "model_name": "TickerArc Multi-Horizon LSTM v2",
    }
