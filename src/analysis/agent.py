"""TickerArc analysis agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.analysis.scoring import potential_score


@dataclass
class AgentResult:
    symbol: str
    summary: str
    tools_used: list[str]
    potential: float


class TickerArcAgent:
    """Explicit tool-orchestration layer for traceable stock summaries."""

    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., object]] = {}

    def register(self, name: str, fn: Callable[..., object]) -> None:
        self.tools[name] = fn

    def analyze(
        self,
        symbol: str,
        snapshot: dict[str, float | str],
        prediction: dict[str, float | str],
        patterns: dict[str, object],
    ) -> AgentResult:
        tools_used = [
            "market_snapshot",
            "activity_scanner",
            "lstm_predictor",
            "candlestick_detector",
            "chart_pattern_detector",
        ]

        p_up = float(prediction.get("up_probability", 0.0))
        r5 = float(prediction.get("return_5d", 0.0))
        vol = float(snapshot.get("volume_ratio", 0.0))
        activity = float(snapshot.get("activity_score", 0.0))
        pattern_score = float(patterns.get("pattern_score", 0.0))
        risk = float(prediction.get("risk_penalty", 0.0))

        potential = potential_score(
            p_up, r5, activity, vol, pattern_score, risk
        )
        candle = patterns.get("candlestick", "No strong recent candle pattern")
        chart = patterns.get("chart", "No strong chart structure detected")

        summary = (
            f"{symbol} is currently in the "
            f"{str(snapshot.get('activity_band', 'Unknown')).lower()} activity band "
            f"with a {float(snapshot.get('change_pct', 0.0)):+.2f}% session move. "
            f"The LSTM projects a {r5:+.2%} 5-day return with "
            f"{p_up:.0%} modeled probability of an upward move. "
            f"Recent candlestick analysis: {candle}. "
            f"Chart structure: {chart}. "
            f"The heuristic potential score is {potential:.0f}/100; "
            "this summarizes model and market features and is not a guarantee "
            "of future performance."
        )
        return AgentResult(symbol, summary, tools_used, potential)
