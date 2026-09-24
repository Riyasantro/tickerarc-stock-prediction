"""Stage 3 chart focus helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ui.trading_chart import make_trading_chart


def make_focused_trading_chart(
    frame: pd.DataFrame,
    symbol: str,
    range_name: str = "1Y",
    show_ema: bool = True,
    show_sr: bool = True,
    show_patterns: bool = True,
    focus_date: pd.Timestamp | str | None = None,
    focus_label: str | None = None,
):
    """Build the existing trading chart and focus the selected finding."""
    figure = make_trading_chart(
        frame,
        symbol,
        range_name,
        show_ema=show_ema,
        show_sr=show_sr,
        show_patterns=show_patterns,
    )
    if figure.data is None or not len(figure.data) or focus_date is None:
        return figure

    parsed = pd.to_datetime(focus_date, errors="coerce")
    if pd.isna(parsed):
        return figure

    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    parsed = pd.Timestamp(parsed)

    if parsed < data["Date"].min() or parsed > data["Date"].max():
        return figure

    window = pd.Timedelta(days=18)
    start = max(data["Date"].min(), parsed - window)
    end = min(data["Date"].max(), parsed + window)

    figure.update_xaxes(range=[start, end], row=1, col=1)
    figure.add_shape(
        type="line",
        x0=parsed,
        x1=parsed,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "#ffc857", "width": 2, "dash": "dash"},
    )

    close_row = data.loc[data["Date"] == parsed]
    if not close_row.empty:
        price = float(close_row.iloc[0]["Close"])
        figure.add_annotation(
            x=parsed,
            y=price,
            xref="x",
            yref="y",
            text=focus_label or "Focused finding",
            showarrow=True,
            arrowhead=2,
            ax=55,
            ay=-45,
            bgcolor="#101612",
            bordercolor="#ffc857",
            borderwidth=1,
            font={"color": "#ffc857", "size": 11},
        )
    return figure
