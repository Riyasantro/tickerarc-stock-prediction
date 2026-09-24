"""Interactive trading chart engine for TickerArc."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_PATTERN_SPECS = (
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
    ("breakout_20", "Breakout", "bull"),
    ("breakdown_20", "Breakdown", "bear"),
)


def _range_frame(frame: pd.DataFrame, range_name: str) -> pd.DataFrame:
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    days = {
        "1M": 31,
        "3M": 93,
        "6M": 186,
        "1Y": 365,
        "3Y": 365 * 3,
        "5Y": 365 * 5,
        "MAX": None,
    }.get(range_name)
    if days is not None and not data.empty:
        cutoff = data["Date"].max() - pd.Timedelta(days=days)
        data = data[data["Date"] >= cutoff]
    return data.reset_index(drop=True)


def _local_extrema(values: np.ndarray, mode: str, distance: int = 2) -> list[int]:
    result: list[int] = []
    for i in range(distance, len(values) - distance):
        left = values[i - distance:i]
        right = values[i + 1:i + 1 + distance]
        if mode == "low":
            ok = values[i] <= left.min() and values[i] <= right.min()
        else:
            ok = values[i] >= left.max() and values[i] >= right.max()
        if ok:
            result.append(i)
    return result


def _pattern_layers(
    data: pd.DataFrame,
) -> tuple[list[tuple[pd.Timestamp, float, str, str]], list[dict[str, object]]]:
    markers: list[tuple[pd.Timestamp, float, str, str]] = []
    shapes: list[dict[str, object]] = []

    dates = pd.to_datetime(data["Date"])
    lows = pd.to_numeric(data["Low"], errors="coerce").to_numpy(float)
    highs = pd.to_numeric(data["High"], errors="coerce").to_numpy(float)

    for column, label, direction in _PATTERN_SPECS:
        if column not in data.columns:
            continue
        active = np.flatnonzero(
            pd.to_numeric(data[column], errors="coerce").fillna(0).to_numpy() != 0
        )
        if len(active) == 0:
            continue

        idx = int(active[-1])
        price = lows[idx] if direction == "bull" else highs[idx]
        markers.append((dates.iloc[idx], float(price), label, direction))

        if column in {"double_bottom_30", "double_top_30"}:
            values = lows if direction == "bull" else highs
            pivots = _local_extrema(values, "low" if direction == "bull" else "high")
            if len(pivots) >= 2:
                p1, p2 = pivots[-2:]
                v1 = float(values[p1])
                v2 = float(values[p2])
                if abs(v1 - v2) <= 0.035 * max(abs(v1), abs(v2), 1e-9):
                    between = data.iloc[p1:p2 + 1]
                    neckline = (
                        float(between["High"].max())
                        if direction == "bull"
                        else float(between["Low"].min())
                    )
                    color = "#00e676" if direction == "bull" else "#ff5c5c"
                    shapes.extend(
                        [
                            {
                                "type": "line",
                                "x0": dates.iloc[p1],
                                "x1": dates.iloc[p2],
                                "y0": v1,
                                "y1": v2,
                                "line": {"color": color, "width": 2},
                            },
                            {
                                "type": "line",
                                "x0": dates.iloc[p1],
                                "x1": dates.iloc[-1],
                                "y0": neckline,
                                "y1": neckline,
                                "line": {"color": color, "width": 1, "dash": "dash"},
                            },
                        ]
                    )

    # Render a small, readable set of the latest TA-Lib candlestick detections.
    hits: list[tuple[pd.Timestamp, float, str, str]] = []
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
        price = lows[idx] if direction == "bull" else highs[idx]
        label = column.replace("candle_cdl", "").replace("_", " ").strip().title()
        hits.append((dates.iloc[idx], float(price), label, direction))

    hits.sort(key=lambda item: item[0])
    markers.extend(hits[-6:])
    return markers, shapes


def make_trading_chart(
    frame: pd.DataFrame,
    symbol: str,
    range_name: str = "1Y",
    show_ema: bool = True,
    show_sr: bool = True,
    show_patterns: bool = True,
) -> go.Figure:
    """Render a trading-terminal chart with candles, indicators and annotations."""
    if frame.empty:
        return go.Figure()

    base = frame.copy()
    base["Date"] = pd.to_datetime(base["Date"])
    base["EMA20"] = pd.to_numeric(base["Close"], errors="coerce").ewm(span=20, adjust=False).mean()
    base["EMA50"] = pd.to_numeric(base["Close"], errors="coerce").ewm(span=50, adjust=False).mean()

    if "support_20" not in base:
        base["support_20"] = pd.to_numeric(base["Low"], errors="coerce").rolling(20).min().shift(1)
    if "resistance_20" not in base:
        base["resistance_20"] = pd.to_numeric(base["High"], errors="coerce").rolling(20).max().shift(1)

    if "rsi_14" in base:
        base["RSI14"] = pd.to_numeric(base["rsi_14"], errors="coerce")
    else:
        delta = pd.to_numeric(base["Close"], errors="coerce").diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        base["RSI14"] = (100 - 100 / (1 + rs)).fillna(50)

    data = _range_frame(base, range_name)
    markers, shapes = _pattern_layers(data) if show_patterns else ([], [])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.66, 0.18, 0.16],
        subplot_titles=(symbol, "Volume", "RSI 14"),
    )

    fig.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            increasing_line_color="#00e676",
            increasing_fillcolor="#00e676",
            decreasing_line_color="#ff5c5c",
            decreasing_fillcolor="#ff5c5c",
            whiskerwidth=0.35,
            name=symbol,
        ),
        row=1,
        col=1,
    )

    if show_ema:
        for key, color, name, width in (
            ("EMA20", "#73f2ae", "EMA 20", 1.35),
            ("EMA50", "#aab6af", "EMA 50", 1.15),
        ):
            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data[key],
                    mode="lines",
                    name=name,
                    line={"color": color, "width": width},
                ),
                row=1,
                col=1,
            )

    if show_sr:
        for key, color, name in (
            ("support_20", "#2e9d68", "Support 20"),
            ("resistance_20", "#b2645d", "Resistance 20"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data[key],
                    mode="lines",
                    name=name,
                    line={"color": color, "dash": "dot", "width": 1},
                ),
                row=1,
                col=1,
            )

    colors = [
        "#00e676" if float(c) >= float(o) else "#ff5c5c"
        for o, c in zip(data["Open"], data["Close"], strict=True)
    ]
    fig.add_trace(
        go.Bar(
            x=data["Date"],
            y=data["Volume"],
            marker_color=colors,
            opacity=0.72,
            name="Volume",
            hovertemplate="%{x|%d %b %Y}<br>Volume %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["RSI14"],
            mode="lines",
            line={"color": "#00e676", "width": 1.4},
            name="RSI 14",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="#55645b", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#55645b", row=3, col=1)

    if markers:
        groups = {
            "bull": ("#00e676", "triangle-up", "Bullish patterns"),
            "bear": ("#ff5c5c", "triangle-down", "Bearish patterns"),
            "neutral": ("#ffc857", "diamond", "Neutral patterns"),
        }
        for direction, (color, marker_symbol, name) in groups.items():
            selected = [item for item in markers if item[3] == direction]
            if not selected:
                continue
            xs = [item[0] for item in selected]
            ys = []
            labels = [item[2] for item in selected]
            for date in xs:
                candle = data.loc[data["Date"] == date].iloc[-1]
                pad = max(float(candle["High"] - candle["Low"]), float(candle["Close"]) * 0.008, 1e-6)
                ys.append(
                    float(candle["Low"] - pad * 0.65)
                    if direction == "bull"
                    else float(candle["High"] + pad * 0.65)
                )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    name=name,
                    text=labels,
                    textposition="bottom center" if direction == "bull" else "top center",
                    marker={
                        "color": color,
                        "symbol": marker_symbol,
                        "size": 9,
                        "line": {"color": "#061009", "width": 1},
                    },
                    hovertemplate="%{text}<br>%{x|%d %b %Y}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        paper_bgcolor="#060806",
        plot_bgcolor="#060806",
        font={"color": "#e8f2eb"},
        height=760,
        margin={"l": 10, "r": 10, "t": 42, "b": 8},
        hovermode="x unified",
        dragmode="pan",
        shapes=shapes,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
        bargap=0.05,
    )
    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=False,
        zeroline=False,
        color="#738078",
        row=1,
        col=1,
    )
    for row in (2, 3):
        fig.update_xaxes(showgrid=False, zeroline=False, color="#738078", row=row, col=1)
    for row in (1, 2, 3):
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#162019",
            zeroline=False,
            color="#738078",
            row=row,
            col=1,
        )
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig


def mini_candlestick_svg(frame: pd.DataFrame, points: int = 18) -> str:
    """Compact SVG OHLC chart used inside instrument cards."""
    if frame is None or frame.empty:
        return ""
    data = frame.tail(points).copy()
    required = ["Open", "High", "Low", "Close"]
    if any(column not in data for column in required):
        return ""

    width, height = 168, 42
    highs = pd.to_numeric(data["High"], errors="coerce")
    lows = pd.to_numeric(data["Low"], errors="coerce")
    lo = float(lows.min())
    hi = float(highs.max())
    span = max(hi - lo, 1e-9)
    slot = width / max(len(data), 1)
    body_width = max(slot * 0.55, 2.0)
    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
    ]

    def y(value: float) -> float:
        return height - ((float(value) - lo) / span) * (height - 4) - 2

    for index, (_, row) in enumerate(data.iterrows()):
        open_ = float(row["Open"])
        close = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])
        x = index * slot + slot / 2
        color = "#00e676" if close >= open_ else "#ff5c5c"
        y_high = y(high)
        y_low = y(low)
        y_open = y(open_)
        y_close = y(close)
        body_y = min(y_open, y_close)
        body_h = max(abs(y_close - y_open), 1.25)
        svg_parts.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{y_high:.2f}" y2="{y_low:.2f}" '
            f'stroke="{color}" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<rect x="{x - body_width / 2:.2f}" y="{body_y:.2f}" '
            f'width="{body_width:.2f}" height="{body_h:.2f}" fill="{color}" rx="0.6"/>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)
