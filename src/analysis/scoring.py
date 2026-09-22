"""Market activity, volume, attention and potential scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _percentile_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return values.rank(pct=True).mul(100).clip(0, 100)


def build_activity_scores(live: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build cross-sectional activity, volume and attention proxy scores."""
    rows: list[dict[str, float | str]] = []
    for row in live.to_dict("records"):
        symbol = str(row["symbol"])
        hist = histories.get(symbol)
        if hist is None or len(hist) < 25:
            avg_volume = np.nan
            volatility = np.nan
            momentum = np.nan
            turnover = np.nan
            volume_ratio = np.nan
        else:
            avg_volume = float(hist["Volume"].tail(20).mean())
            daily_ret = hist["Close"].pct_change().tail(20)
            volatility = float(daily_ret.std())
            momentum = (
                float(hist["Close"].iloc[-1] / hist["Close"].iloc[-21] - 1)
                if len(hist) >= 21 else 0.0
            )
            turnover = float((hist["Close"] * hist["Volume"]).tail(20).mean())
            volume_ratio = float(row["session_volume"] / avg_volume) if avg_volume > 0 else 0.0

        rows.append(
            {
                **row,
                "avg_volume_20d": avg_volume,
                "volume_ratio": volume_ratio,
                "volatility_20d": volatility,
                "momentum_20d": momentum,
                "avg_turnover_20d": turnover,
                "day_range_pct": (
                    (float(row["day_high"]) - float(row["day_low"]))
                    / max(float(row["price"]), 1e-12)
                    * 100.0
                ),
                "abs_return_pct": abs(float(row["change_pct"])),
            }
        )

    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores

    volume_component = _percentile_score(np.log1p(scores["volume_ratio"].fillna(0)))
    move_component = _percentile_score(scores["abs_return_pct"])
    range_component = _percentile_score(scores["day_range_pct"])
    turnover_component = _percentile_score(
        np.log1p(scores["avg_turnover_20d"].fillna(0))
    )

    scores["activity_score"] = (
        0.40 * volume_component
        + 0.30 * move_component
        + 0.20 * range_component
        + 0.10 * turnover_component
    ).round(1)

    scores["volume_score"] = volume_component.round(1)
    scores["attention_score"] = (
        0.55 * volume_component + 0.45 * turnover_component
    ).round(1)

    scores["activity_band"] = pd.cut(
        scores["activity_score"],
        bins=[-np.inf, 35, 65, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    scores["volume_band"] = pd.cut(
        scores["volume_ratio"],
        bins=[-np.inf, 0.50, 1.25, np.inf],
        labels=["Low", "Normal", "High"],
    ).astype(str)
    scores["attention_band"] = pd.cut(
        scores["attention_score"],
        bins=[-np.inf, 35, 65, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    return scores.sort_values("activity_score", ascending=False).reset_index(drop=True)


def potential_score(
    direction_probability: float,
    predicted_return_5d: float,
    activity_score: float,
    volume_ratio: float,
    pattern_score: float,
    risk_penalty: float,
) -> float:
    """Heuristic 0-100 potential score; not a price target or guarantee."""
    direction = np.clip(direction_probability, 0, 1) * 100
    return_component = np.clip((predicted_return_5d + 0.05) / 0.10, 0, 1) * 100
    volume_component = np.clip(volume_ratio / 2.0, 0, 1) * 100
    pattern_component = (np.clip(pattern_score, -1, 1) + 1) * 50
    raw = (
        0.35 * direction
        + 0.25 * return_component
        + 0.15 * activity_score
        + 0.10 * volume_component
        + 0.15 * pattern_component
        - np.clip(risk_penalty, 0, 30)
    )
    return float(np.clip(raw, 0, 100))
