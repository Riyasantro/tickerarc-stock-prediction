import numpy as np
import pandas as pd

from src.ui.findings import build_visual_findings


def test_visual_findings_include_forecast_and_current_technical_state():
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    close = pd.Series(np.linspace(100, 120, len(dates)))
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 1,
            "High": close + 1,
            "Low": close - 2,
            "Close": close,
            "Volume": np.full(len(dates), 1000.0),
            "rsi_14": np.full(len(dates), 55.0),
            "volume_ratio_20": np.full(len(dates), 1.2),
            "double_bottom_30": np.zeros(len(dates), dtype=np.int8),
            "double_top_30": np.zeros(len(dates), dtype=np.int8),
            "breakout_20": np.zeros(len(dates), dtype=np.int8),
            "breakdown_20": np.zeros(len(dates), dtype=np.int8),
        }
    )

    findings = build_visual_findings(
        frame,
        {
            "direction": "UP",
            "up_probability": 0.62,
            "return_5d": 0.018,
        },
    )

    titles = {str(item["title"]) for item in findings}
    assert any("5D forecast" in title for title in titles)
    assert "Uptrend structure" in titles
    assert "RSI neutral" in titles
    assert "Normal relative volume" in titles
