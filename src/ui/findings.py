"""Visual market findings for the TickerArc trading workspace."""
from __future__ import annotations

import numpy as np
import pandas as pd


_PATTERN_SPECS = (
    ("breakout_20", "Breakout", "bull"),
    ("breakdown_20", "Breakdown", "bear"),
    ("double_bottom_30", "Double Bottom", "bull"),
    ("double_top_30", "Double Top", "bear"),
    ("head_shoulders_30", "Head & Shoulders", "bear"),
    ("inverse_head_shoulders_30", "Inverse H&S", "bull"),
    ("ascending_triangle_20", "Ascending Triangle", "bull"),
    ("descending_triangle_20", "Descending Triangle", "bear"),
    ("symmetrical_triangle_20", "Symmetrical Triangle", "neutral"),
    ("rising_wedge_20", "Rising Wedge", "bear"),
    ("falling_wedge_20", "Falling Wedge", "bull"),
    ("bull_flag", "Bull Flag", "bull"),
    ("bear_flag", "Bear Flag", "bear"),
)


def _last_active_date(frame: pd.DataFrame, column: str) -> pd.Timestamp | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    active = np.flatnonzero(values.to_numpy() != 0)
    if len(active) == 0:
        return None
    return pd.Timestamp(frame.iloc[int(active[-1])]["Date"])


def _finding(
    finding_id: str,
    category: str,
    title: str,
    detail: str,
    tone: str,
    date: pd.Timestamp,
) -> dict[str, object]:
    return {
        "id": finding_id,
        "category": category,
        "title": title,
        "detail": detail,
        "tone": tone,
        "date": date,
    }


def build_visual_findings(
    frame: pd.DataFrame,
    prediction: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Build actionable findings with a chart date for focus/navigation."""
    if frame.empty:
        return []

    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    latest = data.iloc[-1]
    latest_date = pd.Timestamp(latest["Date"])
    findings: list[dict[str, object]] = []

    for column, title, tone in _PATTERN_SPECS:
        date = _last_active_date(data, column)
        if date is None:
            continue
        findings.append(
            _finding(
                finding_id=column,
                category="Pattern",
                title=title,
                detail=f"Detected on {date.strftime('%d %b %Y')}.",
                tone=tone,
                date=date,
            )
        )

    candle_hits: list[tuple[pd.Timestamp, str, str]] = []
    for column in data.columns:
        if not column.startswith("candle_cdl"):
            continue
        values = pd.to_numeric(data[column], errors="coerce").fillna(0)
        active = np.flatnonzero(values.to_numpy() != 0)
        if len(active) == 0:
            continue
        idx = int(active[-1])
        value = float(values.iloc[idx])
        direction = "bull" if value > 0 else "bear"
        name = column.replace("candle_cdl", "").replace("_", " ").strip().title()
        candle_hits.append((pd.Timestamp(data.iloc[idx]["Date"]), name, direction))

    candle_hits.sort(key=lambda item: item[0], reverse=True)
    for idx, (date, name, tone) in enumerate(candle_hits[:4]):
        findings.append(
            _finding(
                finding_id=f"candle_{idx}_{name.lower().replace(' ', '_')}",
                category="Candlestick",
                title=name,
                detail=f"TA-Lib pattern on {date.strftime('%d %b %Y')}.",
                tone=tone,
                date=date,
            )
        )

    close = float(latest["Close"])
    ema20 = float(
        pd.to_numeric(data["Close"], errors="coerce").ewm(span=20, adjust=False).mean().iloc[-1]
    )
    ema50 = float(
        pd.to_numeric(data["Close"], errors="coerce").ewm(span=50, adjust=False).mean().iloc[-1]
    )
    if close > ema20 > ema50:
        trend_title, trend_tone = "Uptrend structure", "bull"
        trend_detail = "Price is above EMA 20 and EMA 50, with EMA 20 above EMA 50."
    elif close < ema20 < ema50:
        trend_title, trend_tone = "Downtrend structure", "bear"
        trend_detail = "Price is below EMA 20 and EMA 50, with EMA 20 below EMA 50."
    else:
        trend_title, trend_tone = "Mixed trend structure", "neutral"
        trend_detail = "Price and EMA 20/50 are not aligned in one direction."

    findings.append(
        _finding(
            "trend_structure",
            "Trend",
            trend_title,
            trend_detail,
            trend_tone,
            latest_date,
        )
    )

    if "rsi_14" in data:
        rsi = float(pd.to_numeric(data["rsi_14"], errors="coerce").iloc[-1])
    else:
        delta = data["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float((100 - 100 / (1 + rs)).fillna(50).iloc[-1])

    if rsi >= 70:
        rsi_title, rsi_tone = "RSI overbought", "bear"
    elif rsi <= 30:
        rsi_title, rsi_tone = "RSI oversold", "bull"
    else:
        rsi_title, rsi_tone = "RSI neutral", "neutral"

    findings.append(
        _finding(
            "rsi_state",
            "Momentum",
            rsi_title,
            f"RSI 14 is {rsi:.1f}.",
            rsi_tone,
            latest_date,
        )
    )

    if "volume_ratio_20" in data:
        rel_volume = float(pd.to_numeric(data["volume_ratio_20"], errors="coerce").iloc[-1])
    else:
        avg_volume = float(pd.to_numeric(data["Volume"], errors="coerce").tail(20).mean())
        rel_volume = float(latest["Volume"]) / max(avg_volume, 1e-12)

    if rel_volume >= 1.5:
        volume_title, volume_tone = "High relative volume", "bull"
    elif rel_volume <= 0.6:
        volume_title, volume_tone = "Low relative volume", "neutral"
    else:
        volume_title, volume_tone = "Normal relative volume", "neutral"

    findings.append(
        _finding(
            "relative_volume",
            "Activity",
            volume_title,
            f"Latest volume is {rel_volume:.2f}x the 20-day average.",
            volume_tone,
            latest_date,
        )
    )

    support = float(latest.get("support_20", np.nan))
    resistance = float(latest.get("resistance_20", np.nan))
    if np.isfinite(support):
        findings.append(
            _finding(
                "support_level",
                "Levels",
                f"Support ₹{support:,.2f}",
                "20-day causal support level.",
                "bull",
                latest_date,
            )
        )
    if np.isfinite(resistance):
        findings.append(
            _finding(
                "resistance_level",
                "Levels",
                f"Resistance ₹{resistance:,.2f}",
                "20-day causal resistance level.",
                "bear",
                latest_date,
            )
        )

    if prediction:
        up = float(prediction.get("up_probability", 0.0))
        return_5d = float(prediction.get("return_5d", 0.0))
        direction = str(prediction.get("direction", "NEUTRAL"))
        if direction.upper() == "UP":
            tone = "bull"
        elif direction.upper() == "DOWN":
            tone = "bear"
        else:
            tone = "neutral"
        findings.insert(
            0,
            _finding(
                "forecast_5d",
                "Forecast",
                f"5D forecast {return_5d:+.2%}",
                f"Modeled direction {direction} with {up:.0%} upward probability.",
                tone,
                latest_date,
            ),
        )

    priority = {
        "Forecast": 0,
        "Pattern": 1,
        "Candlestick": 2,
        "Trend": 3,
        "Momentum": 4,
        "Activity": 5,
        "Levels": 6,
    }
    findings.sort(
        key=lambda item: (priority.get(str(item["category"]), 99), -pd.Timestamp(item["date"]).value)
    )
    return findings
