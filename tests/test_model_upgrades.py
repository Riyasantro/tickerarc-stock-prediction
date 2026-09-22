import numpy as np
import pandas as pd

from src.models.trainer import (
    EVAL_YEARS,
    FEATURE_COLUMNS,
    FIT_YEARS,
    TRAIN_YEARS,
    VALIDATION_YEARS,
    split_development_window,
    split_time_window,
)


def synthetic_model_frame(years: int = 31) -> pd.DataFrame:
    rows = years * 252
    dates = pd.bdate_range("1995-01-03", periods=rows)
    rng = np.random.default_rng(123)

    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, rows)))
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(100_000, 1_000_000, rows),
        }
    )

    for column in FEATURE_COLUMNS:
        frame[column] = rng.normal(0, 1, rows)

    return frame


def test_time_split_keeps_final_test_window():
    frame = synthetic_model_frame()

    development, evaluation, bounds = split_time_window(frame)

    assert TRAIN_YEARS == 24
    assert EVAL_YEARS == 2
    assert len(development) > len(evaluation)
    assert pd.to_datetime(development["Date"]).max() < pd.to_datetime(
        evaluation["Date"]
    ).min()
    assert bounds["eval_start"] < bounds["eval_end"]


def test_validation_window_is_inside_24_year_development_window():
    frame = synthetic_model_frame()

    fit, validation, bounds = split_development_window(frame)

    assert FIT_YEARS == 20
    assert VALIDATION_YEARS == 4
    assert len(fit) > len(validation)
    assert pd.to_datetime(fit["Date"]).max() < pd.to_datetime(
        validation["Date"]
    ).min()
    assert bounds["validation_start"] <= bounds["validation_end"]
