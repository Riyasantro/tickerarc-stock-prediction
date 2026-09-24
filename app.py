"""TickerArc single-entry Streamlit application.

Run:
    streamlit run app.py

First launch downloads historical data and trains the global LSTM if no local
model checkpoint exists. Later refreshes only fetch the live layer and run
inference; training is not repeated automatically.
"""

from __future__ import annotations

from datetime import datetime, time
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.universe import NIFTY50_SYMBOLS
from src.analysis.agent import TickerArcAgent
from src.analysis.scoring import build_activity_scores, potential_score
from src.data.download import download_universe
from src.data.live import fetch_intraday_history, fetch_live_quotes
from src.data.pipeline import PROCESSED_DIR, process_downloaded_files
from src.features.chart_patterns import add_chart_pattern_features
from src.features.talib_features import add_all_candlestick_patterns
from src.models.trainer import (
    EVAL_YEARS,
    FEATURE_COLUMNS,
    TRAIN_YEARS,
    evaluate_global_model,
    load_model,
    load_training_frames,
    predict_latest,
    train_global_model,
)
from src.models.benchmarks import evaluate_baselines
from src.models.walk_forward import walk_forward_evaluate
from src.rl.online_loop import OnlineRLManager
from src.ui.trading_chart import mini_candlestick_svg
from src.ui.trading_chart_focus import make_focused_trading_chart
from src.ui.findings import build_visual_findings

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "tickerarc_global_lstm.pt"
SCALER_PATH = MODEL_DIR / "tickerarc_scaler.joblib"
META_PATH = MODEL_DIR / "tickerarc_model_meta.json"
EVAL_PATH = MODEL_DIR / "tickerarc_holdout_eval.json"
HISTORY_MARKER = ROOT / "data" / "cache" / ".tickerarc_max_history"
LIVE_REFRESH_MINUTES = 1


st.set_page_config(
    page_title="TickerArc",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --bg: #060806;
    --panel: #0b0f0d;
    --panel-2: #101612;
    --border: #1f2b23;
    --text: #ecf7ef;
    --muted: #87958c;
    --green: #00e676;
    --red: #ff5c5c;
    --amber: #ffc857;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] {
    background: #080b09 !important;
    border-right: 1px solid var(--border);
}
[data-testid="stMetric"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; }
.stButton > button {
    background: var(--green) !important;
    color: #001a0c !important;
    border: 0 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
div[data-baseweb="select"] > div {
    background: var(--panel) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
[data-testid="stTabs"] [role="tab"] { color: var(--muted) !important; }
[data-testid="stTabs"] [aria-selected="true"] { color: var(--green) !important; }
.section-title {
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: .04em;
    margin: 1.1rem 0 .55rem 0;
}
.kicker {
    color: var(--green);
    font-size: .76rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    font-weight: 800;
}
.market-head {
    border: 1px solid var(--border);
    background: linear-gradient(180deg, #0d120f, #080b09);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.market-title {
    font-size: 2.25rem;
    line-height: 1;
    font-weight: 850;
    margin: .15rem 0 .35rem 0;
}
.market-subtitle { color: var(--muted); }
.pill {
    display: inline-block;
    padding: 4px 9px;
    border-radius: 999px;
    font-size: .73rem;
    font-weight: 800;
    margin: 0 5px 5px 0;
}
.pill-green { background: rgba(0,230,118,.12); color: var(--green); border: 1px solid rgba(0,230,118,.25); }
.pill-red { background: rgba(255,92,92,.12); color: var(--red); border: 1px solid rgba(255,92,92,.25); }
.pill-muted { background: #101612; color: var(--muted); border: 1px solid var(--border); }
.stock-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    min-height: 128px;
    transition: border-color .2s ease, transform .2s ease;
}
.stock-card:hover {
    border-color: rgba(0,230,118,.55);
    transform: translateY(-1px);
}
.stock-symbol { font-weight: 850; font-size: 1.02rem; }
.stock-price { font-weight: 750; font-size: 1.08rem; margin-top: 4px; }
.stock-meta { color: var(--muted); font-size: .75rem; margin-top: 2px; }
.stock-green { color: var(--green); }
.stock-red { color: var(--red); }
.finding {
    border-left: 2px solid var(--green);
    background: #0c110e;
    border-radius: 6px;
    padding: 9px 10px;
    margin-bottom: 7px;
}
.finding-label { color: var(--muted); font-size: .72rem; }
.finding-value { font-weight: 800; font-size: .9rem; }
.health-ok { color: var(--green); font-weight: 800; }
.health-warn { color: var(--amber); font-weight: 800; }
.small-muted { color: var(--muted); font-size: .79rem; }
.disclaimer { color: var(--muted); font-size: .74rem; padding: 8px 0; }
.terminal-bar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    background:#080d0a;
    border:1px solid var(--border);
    border-radius:12px;
    padding:10px 14px;
    margin-bottom:12px;
}
.terminal-brand { font-weight:850; letter-spacing:.08em; }
.terminal-status { color:var(--muted); font-size:.78rem; }
.live-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--green); margin-right:6px; }
.stock-header {
    display:flex; justify-content:space-between; align-items:end;
    gap:16px; margin:6px 0 10px 0;
}
.stock-name { font-size:2rem; font-weight:850; line-height:1; }
.stock-sub { color:var(--muted); font-size:.8rem; margin-top:5px; }
.price-big { font-size:1.55rem; font-weight:820; text-align:right; }
.range-bar {
    background:#080d0a; border:1px solid var(--border);
    border-radius:10px; padding:6px 10px 0 10px; margin:8px 0;
}
.finding-grid {
    background: #080d0a;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    margin: 10px 0 14px 0;
}
.finding-card {
    background: #0c120e;
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 10px 12px;
    min-height: 82px;
}
.finding-card-bull { border-left: 3px solid var(--green); }
.finding-card-bear { border-left: 3px solid var(--red); }
.finding-card-neutral { border-left: 3px solid var(--amber); }
.finding-category {
    color: var(--muted);
    font-size: .67rem;
    text-transform: uppercase;
    letter-spacing: .08em;
}
.finding-title {
    font-weight: 800;
    font-size: .92rem;
    margin-top: 2px;
}
.finding-detail {
    color: var(--muted);
    font-size: .73rem;
    margin-top: 3px;
}
.focused-label {
    color: var(--amber);
    font-weight: 750;
    font-size: .76rem;
    margin: 3px 0 8px 0;
}

</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=45, show_spinner=False)
def get_live_data(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Use 1-minute data first, then 5-minute data if needed."""
    try:
        result = fetch_live_quotes(symbols, interval="1m", lookback="5d")
        if not result.empty:
            return result
    except Exception:
        pass
    return fetch_live_quotes(symbols, interval="5m", lookback="5d")


@st.cache_data(ttl=900, show_spinner=False)
def load_processed_histories(
    symbols: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = PROCESSED_DIR / (
            f"{symbol.lower().replace('&', '_and_').replace('-', '_')}.parquet"
        )
        if path.exists():
            histories[symbol] = pd.read_parquet(path)
    return histories


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    if not (MODEL_PATH.exists() and SCALER_PATH.exists() and META_PATH.exists()):
        return None
    return load_model(MODEL_DIR)


@st.cache_resource(show_spinner=False)
def get_online_rl_manager() -> OnlineRLManager:
    return OnlineRLManager(MODEL_DIR)


def bootstrap_project() -> None:
    """Fetch maximum history, train on 24 years, and evaluate the next 2 years."""
    processed_files = list(PROCESSED_DIR.glob("*.parquet"))
    feature_refresh_needed = not processed_files

    if processed_files:
        try:
            sample = pd.read_parquet(
                processed_files[0],
                columns=["candlestick_signal_score", "regime_risk_on"],
            )
            feature_refresh_needed = sample.empty
        except Exception:
            feature_refresh_needed = True

    # Older versions downloaded only five years. Force one upgrade to max history.
    history_refresh_needed = not HISTORY_MARKER.exists()

    if history_refresh_needed:
        with st.status(
            "Downloading maximum available NIFTY 50 history…",
            expanded=True,
        ) as status:
            download_universe(
                NIFTY50_SYMBOLS,
                period="max",
                interval="1d",
            )
            HISTORY_MARKER.parent.mkdir(parents=True, exist_ok=True)
            HISTORY_MARKER.write_text("max-history-downloaded", encoding="utf-8")
            feature_refresh_needed = True
            status.update(
                label="Maximum available history downloaded",
                state="complete",
            )

    if feature_refresh_needed:
        with st.status(
            "Building technical and pattern features…",
            expanded=True,
        ) as status:
            process_downloaded_files()
            status.update(
                label="Feature dataset ready",
                state="complete",
            )

    model_rebuild_needed = not (
        MODEL_PATH.exists() and SCALER_PATH.exists() and META_PATH.exists()
    )
    if not model_rebuild_needed:
        try:
            metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
            model_rebuild_needed = (
                int(metadata.get("train_years", 0)) != TRAIN_YEARS
                or int(metadata.get("eval_years", 0)) != EVAL_YEARS
                or int(metadata.get("validation_years", 0)) != 4
                or metadata.get("architecture") != "LSTM + temporal attention pooling + multi-task heads"
            )
        except Exception:
            model_rebuild_needed = True

    if model_rebuild_needed:
        for path in (MODEL_PATH, SCALER_PATH, META_PATH, EVAL_PATH):
            path.unlink(missing_ok=True)

        with st.status(
            f"Training LSTM on {TRAIN_YEARS} years and reserving {EVAL_YEARS} years for evaluation…",
            expanded=True,
        ) as status:
            frames = load_training_frames(PROCESSED_DIR, NIFTY50_SYMBOLS)
            progress = st.progress(0)
            message = st.empty()

            def callback(epoch: int, loss: float) -> None:
                progress.progress(min(epoch / 8, 1.0))
                message.write(
                    f"Epoch {epoch}/8 · validation loss {loss:.5f}"
                )

            train_global_model(
                frames,
                MODEL_DIR,
                epochs=8,
                sequence_length=60,
                batch_size=128,
                hidden_size=128,
                progress=callback,
            )
            progress.progress(1.0)
            status.update(
                label=f"LSTM selected with 20Y fit + 4Y validation, then refit on {TRAIN_YEARS} years",
                state="complete",
            )

        with st.status(
            f"Evaluating the latest {EVAL_YEARS} years…",
            expanded=True,
        ) as status:
            frames = load_training_frames(PROCESSED_DIR, NIFTY50_SYMBOLS)
            model, scaler, metadata = load_model(MODEL_DIR)
            metrics = evaluate_global_model(
                model,
                scaler,
                frames,
                sequence_length=int(metadata["sequence_length"]),
            )
            metrics["baselines"] = evaluate_baselines(frames)
            EVAL_PATH.write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )
            status.update(
                label="2-year holdout evaluation complete",
                state="complete",
            )

        get_model_bundle.clear()
        get_online_rl_manager.clear()

def market_state() -> tuple[str, str]:
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    open_time = time(9, 15)
    close_time = time(15, 30)

    if now.weekday() >= 5:
        return "CLOSED", now.strftime("%Y-%m-%d %H:%M:%S IST")
    if open_time <= now.time() <= close_time:
        return "OPEN", now.strftime("%Y-%m-%d %H:%M:%S IST")
    return "CLOSED", now.strftime("%Y-%m-%d %H:%M:%S IST")


def pattern_summary(frame: pd.DataFrame) -> dict[str, object]:
    data = add_chart_pattern_features(
        add_all_candlestick_patterns(frame.copy())
    )
    latest = data.iloc[-1]

    candle_hits: list[str] = []
    signal_score = 0.0

    for column in [c for c in data.columns if c.startswith("candle_cdl")]:
        value = float(latest[column])
        if value == 0:
            continue
        label = (
            column.replace("candle_cdl", "")
            .replace("_", " ")
            .strip()
            .title()
        )
        direction = "bullish" if value > 0 else "bearish"
        candle_hits.append(f"{label} ({direction})")
        signal_score += 1 if value > 0 else -1

    chart_map = [
        ("breakout_20", "20-day breakout", 1),
        ("breakdown_20", "20-day breakdown", -1),
        ("double_top_30", "double top", -1),
        ("double_bottom_30", "double bottom", 1),
        ("head_shoulders_30", "head and shoulders", -1),
        ("inverse_head_shoulders_30", "inverse head and shoulders", 1),
        ("ascending_triangle_20", "ascending triangle", 1),
        ("descending_triangle_20", "descending triangle", -1),
        ("symmetrical_triangle_20", "symmetrical triangle", 0),
        ("rising_wedge_20", "rising wedge", -1),
        ("falling_wedge_20", "falling wedge", 1),
        ("bull_flag", "bull flag", 1),
        ("bear_flag", "bear flag", -1),
    ]

    chart_hits: list[str] = []
    for column, label, direction in chart_map:
        if column in latest.index and float(latest[column]) == 1:
            chart_hits.append(label)
            signal_score += direction

    return {
        "candlestick": (
            ", ".join(candle_hits[:4])
            if candle_hits
            else "No strong recent TA-Lib candlestick signal"
        ),
        "chart": (
            ", ".join(chart_hits[:4])
            if chart_hits
            else "No strong heuristic chart structure"
        ),
        "pattern_score": float(np.tanh(signal_score / 5.0)),
        "candle_hits": candle_hits,
        "chart_hits": chart_hits,
    }


def _range_frame(frame: pd.DataFrame, range_name: str) -> pd.DataFrame:
    data = frame.copy()
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])
    days_map = {
        "1M": 31,
        "3M": 93,
        "6M": 186,
        "1Y": 365,
        "3Y": 365 * 3,
        "5Y": 365 * 5,
        "MAX": None,
    }
    days = days_map.get(range_name)
    if days is not None and not data.empty:
        data = data[data["Date"] >= data["Date"].max() - pd.Timedelta(days=days)]
    return data.reset_index(drop=True)


def make_chart(
    frame: pd.DataFrame,
    symbol: str,
    range_name: str = "1Y",
    show_ema: bool = True,
    show_sr: bool = True,
) -> go.Figure:
    data = _range_frame(frame, range_name)
    if data.empty:
        return go.Figure()

    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["support"] = data["Low"].rolling(20).min().shift(1)
    data["resistance"] = data["High"].rolling(20).max().shift(1)
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI14"] = (100 - 100 / (1 + rs)).fillna(50)
    volume_colors = [
        "#00e676" if close >= open_ else "#ff5c5c"
        for open_, close in zip(data["Open"], data["Close"], strict=True)
    ]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.64, 0.18, 0.18],
        subplot_titles=(f"{symbol} · {range_name}", "Volume", "RSI 14"),
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
            name=symbol,
        ),
        row=1,
        col=1,
    )

    if show_ema:
        fig.add_trace(
            go.Scatter(
                x=data["Date"], y=data["EMA20"], mode="lines", name="EMA 20",
                line={"color": "#7ef2b1", "width": 1.5},
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["Date"], y=data["EMA50"], mode="lines", name="EMA 50",
                line={"color": "#b8c4bd", "width": 1.2},
            ),
            row=1, col=1,
        )

    if show_sr:
        fig.add_trace(
            go.Scatter(
                x=data["Date"], y=data["support"], mode="lines", name="Support",
                line={"color": "#4ea87a", "dash": "dot", "width": 1},
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["Date"], y=data["resistance"], mode="lines", name="Resistance",
                line={"color": "#ad766f", "dash": "dot", "width": 1},
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Bar(
            x=data["Date"], y=data["Volume"], name="Volume",
            marker_color=volume_colors, opacity=0.72,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["Date"], y=data["RSI14"], mode="lines", name="RSI 14",
            line={"color": "#00e676", "width": 1.5},
        ),
        row=3, col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="#6f7b74", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#6f7b74", row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#060806",
        plot_bgcolor="#060806",
        font={"color": "#e8f2eb"},
        height=720,
        margin={"l": 8, "r": 8, "t": 38, "b": 8},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
        xaxis_rangeslider_visible=False,
        bargap=0.05,
    )
    for row in (1, 2, 3):
        fig.update_xaxes(showgrid=False, zeroline=False, color="#738078", row=row, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#162019", zeroline=False, color="#738078", row=row, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig


def _sparkline_svg(history: pd.DataFrame, points: int = 34) -> str:
    if history is None or history.empty or "Close" not in history:
        return ""
    values = pd.to_numeric(history["Close"], errors="coerce").dropna().tail(points).to_numpy()
    if len(values) < 2:
        return ""
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1e-9)
    width, height = 160, 34
    coords = []
    for idx, value in enumerate(values):
        x = idx / (len(values) - 1) * width
        y = height - ((float(value) - lo) / span) * (height - 3) - 1
        coords.append(f"{x:.1f},{y:.1f}")
    stroke = "#00e676" if values[-1] >= values[0] else "#ff5c5c"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="34" preserveAspectRatio="none">'
        f'<polyline points="{" ".join(coords)}" fill="none" stroke="{stroke}" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def _select_stock(symbol: str) -> None:
    st.session_state.selected_symbol = symbol
    st.session_state.market_view = "stock"
    st.session_state.chart_focus = None

def _focus_finding(finding: dict[str, object]) -> None:
    st.session_state.chart_focus = {
        "date": str(finding["date"]),
        "title": str(finding["title"]),
        "id": str(finding["id"]),
    }

def _clear_chart_focus() -> None:
    st.session_state.chart_focus = None


def _sidebar_stock_changed() -> None:
    st.session_state.selected_symbol = st.session_state.stock_selector
    st.session_state.market_view = "stock"
    st.session_state.chart_focus = None


def build_market_categories(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return consistent cross-sectional screens used by the market navigator."""
    return {
        "Most Active": combined.sort_values(
            ["activity_score", "volume_ratio"],
            ascending=False,
        ).reset_index(drop=True),
        "Popular": combined.sort_values(
            ["attention_score", "activity_score"],
            ascending=False,
        ).reset_index(drop=True),
        "Model Signals": combined.sort_values(
            ["up_probability", "potential", "return_5d"],
            ascending=False,
        ).reset_index(drop=True),
        "Potential": combined.sort_values(
            ["potential", "up_probability", "activity_score"],
            ascending=False,
        ).reset_index(drop=True),
        "Low Attention": combined.sort_values(
            ["attention_score", "activity_score"],
            ascending=[True, False],
        ).reset_index(drop=True),
    }


def render_stock_cards(
    frame: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    title: str,
    subtitle: str,
    limit: int | None = 5,
    key_prefix: str = "stock",
) -> None:
    """Render interactive stock instruments; clicking a stock opens its workspace."""
    st.markdown(
        f'<div class="section-title">{title}</div>'
        f'<div class="small-muted" style="margin:-3px 0 8px 0">{subtitle}</div>',
        unsafe_allow_html=True,
    )
    if frame.empty:
        st.info("No stocks currently meet this screen.")
        return

    subset = frame if limit is None else frame.head(limit)

    for row_start in range(0, len(subset), 5):
        row_slice = subset.iloc[row_start : row_start + 5]
        cols = st.columns(5)
        for col, (_, row) in zip(cols, row_slice.iterrows(), strict=False):
            symbol = str(row["symbol"])
            price = float(row.get("price", 0.0))
            change = float(row.get("change_pct", 0.0))
            up_prob = float(row.get("up_probability", np.nan))
            potential = float(row.get("potential", np.nan))
            activity = float(row.get("activity_score", np.nan))
            volume_ratio = float(row.get("volume_ratio", np.nan))

            change_text = f"{change:+.2f}%"
            up_text = f"{up_prob * 100:.0f}%" if np.isfinite(up_prob) else "—"
            potential_text = (
                f"{potential:.0f}" if np.isfinite(potential) else "—"
            )
            activity_text = (
                f"{activity:.0f}" if np.isfinite(activity) else "—"
            )
            volume_text = (
                f"{volume_ratio:.2f}x"
                if np.isfinite(volume_ratio)
                else "—"
            )

            with col:
                st.markdown(
                    f'<div class="stock-card">'
                    f'<div class="stock-symbol">{symbol}</div>'
                    f'<div class="stock-price">₹{price:,.2f}</div>'
                    f'<div class="{"stock-green" if change >= 0 else "stock-red"}">'
                    f'{change_text} session</div>'
                    f'<div style="margin-top:6px">'
                    f'{mini_candlestick_svg(histories.get(symbol))}</div>'
                    f'<div class="stock-meta">Up {up_text} · Potential {potential_text}</div>'
                    f'<div class="stock-meta">Activity {activity_text} · Rel Vol {volume_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.button(
                    f"Open {symbol}",
                    key=f"{key_prefix}_{symbol}_{row_start}",
                    use_container_width=True,
                    on_click=_select_stock,
                    args=(symbol,),
                )


def render_market_finding_sidebar(
    combined: pd.DataFrame,
    live: pd.DataFrame,
    status: str,
    model_ready: bool,
    eval_ready: bool,
    rl_ready: bool,
) -> None:
    if combined.empty:
        return
    advances = int((combined["change_pct"] > 0).sum())
    declines = int((combined["change_pct"] < 0).sum())
    unchanged = int((combined["change_pct"] == 0).sum())
    active = combined.nlargest(1, "activity_score").iloc[0]
    model_leader = combined.nlargest(1, "up_probability").iloc[0]
    potential_leader = combined.nlargest(1, "potential").iloc[0]
    attention_leader = combined.nlargest(1, "attention_score").iloc[0]
    avg_vol = float(pd.to_numeric(combined["volatility_20d"], errors="coerce").mean())
    last_provider = pd.to_datetime(live["timestamp"], errors="coerce").max()

    with st.sidebar:
    st.markdown(
        '<div class="terminal-brand">TICKERARC</div>'
        '<div class="small-muted">NIFTY 50 MARKET</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**MARKET**")
    st.radio(
        "Market",
        ["Overview", "Most Active", "Popular", "Model Signals", "Potential", "Low Attention"],
        key="market_section",
        label_visibility="collapsed",
    )
    st.markdown("---")

    current_symbol = st.session_state.selected_symbol
    current_index = NIFTY50_SYMBOLS.index(current_symbol) if current_symbol in NIFTY50_SYMBOLS else 0
    st.selectbox(
        "Instrument",
        NIFTY50_SYMBOLS,
        index=current_index,
        key="stock_selector",
        on_change=_sidebar_stock_changed,
    )

    if st.session_state.market_view == "stock":
        if st.button("← Back to market", use_container_width=True):
            st.session_state.market_view = "market"
            st.session_state.chart_focus = None
            st.rerun()

try:
    bootstrap_project()
except Exception as exc:
    st.error(f"TickerArc initialization failed: {exc}")
    st.stop()

model_bundle = get_model_bundle()
if model_bundle is None:
    st.error("Model checkpoint is not available.")
    st.stop()

histories = load_processed_histories(tuple(NIFTY50_SYMBOLS))


def render_selected_instrument(
    selected_symbol: str,
    selected_quote: dict[str, object],
    row: pd.Series,
    history: pd.DataFrame,
    status: str,
    refresh_minutes: int,
    chart_range: str,
    show_ema: bool,
    show_sr: bool,
    show_patterns: bool,
) -> None:
    """Render the selected instrument workspace."""

    patterns = pattern_summary(history)
    prediction = row.to_dict()
    prediction["risk_penalty"] = min(
        float(prediction.get("volatility_5d", 0.0)) * 100,
        30,
    )
    result = TickerArcAgent().analyze(
        selected_symbol,
        row.to_dict(),
        prediction,
        patterns,
    )
    findings = build_visual_findings(history, row.to_dict())
    focus = st.session_state.get("chart_focus")
    focus_date = focus.get("date") if isinstance(focus, dict) else None
    focus_title = focus.get("title") if isinstance(focus, dict) else None

    provider_time = pd.to_datetime(selected_quote.get("timestamp"), errors="coerce")
    provider_text = provider_time.strftime("%H:%M:%S") if pd.notna(provider_time) else "—"
    change = float(row["change_pct"])
    change_class = "stock-green" if change >= 0 else "stock-red"

    st.markdown(
        f'<div class="stock-header">'
        f'<div><div class="stock-name">{selected_symbol}</div>'
        f'<div class="stock-sub">NSE · NIFTY 50 · {status}</div></div>'
        f'<div><div class="price-big">₹{float(row["price"]):,.2f}</div>'
        f'<div class="{change_class}" style="text-align:right">{change:+.2f}% today</div>'
        f'<div class="stock-sub" style="text-align:right">Updated {provider_text} IST</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric(
        "Price",
        f"₹{float(row['price']):,.2f}",
        f"{float(row['change_pct']):+.2f}%",
    )
    q2.metric("1D model", f"{float(row['return_1d'])*100:+.2f}%")
    q3.metric("5D model", f"{float(row['return_5d'])*100:+.2f}%")
    q4.metric("10D model", f"{float(row['return_10d'])*100:+.2f}%")
    q5.metric("Potential", f"{float(result.potential):.0f}/100")

    rc1, rc2 = st.columns([2.3, 1])
    with rc1:
        st.markdown('<div class="range-bar">', unsafe_allow_html=True)
        selected_range = st.radio(
            "Range",
            ["1M", "3M", "6M", "1Y", "3Y", "5Y", "MAX"],
            horizontal=True,
            key=f"range_{selected_symbol}",
            index=["1M", "3M", "6M", "1Y", "3Y", "5Y", "MAX"].index(st.session_state.chart_range),
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with rc2:
        cema, csr, cpat = st.columns(3)
        with cema:
            local_ema = st.checkbox("EMA", value=st.session_state.show_ema, key=f"ema_{selected_symbol}")
        with csr:
            local_sr = st.checkbox("S/R", value=st.session_state.show_sr, key=f"sr_{selected_symbol}")
        with cpat:
            local_patterns = st.checkbox("Patterns", value=st.session_state.show_patterns, key=f"patterns_{selected_symbol}")
    st.session_state.chart_range = selected_range
    st.session_state.show_ema = local_ema
    st.session_state.show_sr = local_sr
    st.session_state.show_patterns = local_patterns

    st.plotly_chart(
        make_focused_trading_chart(
            history,
            selected_symbol,
            selected_range,
            show_ema=local_ema,
            show_sr=local_sr,
            show_patterns=local_patterns,
            focus_date=focus_date,
            focus_label=focus_title,
        ),
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    st.markdown('<div class="section-title">Market findings</div>', unsafe_allow_html=True)
    if focus_title:
        fc1, fc2 = st.columns([5, 1])
        with fc1:
            st.markdown(
                f'<div class="focused-label">Focused on chart · {focus_title}</div>',
                unsafe_allow_html=True,
            )
        with fc2:
            st.button("Clear focus", key=f"clear_focus_{selected_symbol}", on_click=_clear_chart_focus, use_container_width=True)

    with st.container():
        st.markdown('<div class="finding-grid">', unsafe_allow_html=True)
        visible_findings = findings[:9]
        for start_idx in range(0, len(visible_findings), 3):
            cols = st.columns(3)
            for col, finding in zip(cols, visible_findings[start_idx:start_idx + 3], strict=False):
                tone = str(finding["tone"])
                css_tone = f"finding-card-{tone}" if tone in {"bull", "bear", "neutral"} else "finding-card-neutral"
                with col:
                    st.markdown(
                        f'<div class="finding-card {css_tone}">'
                        f'<div class="finding-category">{finding["category"]}</div>'
                        f'<div class="finding-title">{finding["title"]}</div>'
                        f'<div class="finding-detail">{finding["detail"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.button(
                        "Focus on chart",
                        key=f"finding_{selected_symbol}_{finding['id']}",
                        use_container_width=True,
                        on_click=_focus_finding,
                        args=(finding,),
                    )
        st.markdown('</div>', unsafe_allow_html=True)

    details_tab, signals_tab, performance_tab, rl_tab = st.tabs(
        ["Overview", "Patterns", "Forecast", "Strategy"]
    )

    with details_tab:
        d1, d2 = st.columns([1.35, 1])
        with d1:
            st.markdown("#### Session profile")
            s1, s2, s3 = st.columns(3)
            s1.metric(
                "Open",
                f"₹{float(selected_quote.get('open', row['price'])):,.2f}",
            )
            s2.metric(
                "High",
                f"₹{float(selected_quote.get('day_high', row['price'])):,.2f}",
            )
            s3.metric(
                "Low",
                f"₹{float(selected_quote.get('day_low', row['price'])):,.2f}",
            )
            st.markdown(
                f'<div class="small-muted">Session volume: '
                f'{float(selected_quote.get("session_volume", 0)):,.0f} · '
                f'Relative volume: {float(row.get("volume_ratio", 0)):.2f}x · '
                f'Activity score: {float(row.get("activity_score", 0)):.1f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("#### Forecast distribution")
            donut = go.Figure(
                go.Pie(
                    labels=["Down", "Neutral", "Up"],
                    values=[
                        float(row.get("down_probability", 0)),
                        float(row.get("neutral_probability", 0)),
                        float(row.get("up_probability", 0)),
                    ],
                    hole=0.68,
                    marker={"colors": ["#ff5c5c", "#5d6a62", "#00e676"]},
                )
            )
            donut.update_layout(
                paper_bgcolor="#060806",
                plot_bgcolor="#060806",
                font={"color": "#e8f2eb"},
                height=300,
                margin={"l": 10, "r": 10, "t": 15, "b": 10},
                showlegend=True,
            )
            st.plotly_chart(
                donut,
                use_container_width=True,
                config={"displaylogo": False},
            )
        with d2:
            st.markdown("#### TickerArc findings")
            st.write(result.summary)
            st.markdown(
                '<div class="disclaimer">Forecasts, probabilities and potential are model outputs and heuristics, not guaranteed future results.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("#### Market context")
            st.write(f"Session: **{status}**")
            st.write(f"Relative volume: **{float(row.get('volume_ratio', 0)):.2f}x**")
            st.write(f"Activity: **{float(row.get('activity_score', 0)):.0f}/100**")

    with signals_tab:
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("#### Candlestick patterns")
            candle_items = patterns["candle_hits"][:10] or [
                "No strong recent TA-Lib signal"
            ]
            for item in candle_items:
                st.markdown(
                    f'<span class="pill pill-green">{item}</span>',
                    unsafe_allow_html=True,
                )
        with p2:
            st.markdown("#### Chart patterns")
            chart_items = patterns["chart_hits"][:10] or [
                "No strong heuristic chart structure"
            ]
            for item in chart_items:
                st.markdown(
                    f'<span class="pill pill-muted">{item}</span>',
                    unsafe_allow_html=True,
                )

        st.markdown("#### Forecast distribution")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Up probability", f"{float(row['up_probability'])*100:.1f}%")
        f2.metric("Neutral probability", f"{float(row['neutral_probability'])*100:.1f}%")
        f3.metric("Down probability", f"{float(row['down_probability'])*100:.1f}%")
        f4.metric("5D volatility", f"{float(row['volatility_5d'])*100:.2f}%")

    with performance_tab:
        render_model_performance()

    online_result = None
    catchup_info = None
    try:
        online_manager = get_online_rl_manager()
        if online_manager.needs_catchup(
            selected_symbol,
            selected_quote["timestamp"],
            refresh_minutes,
        ):
            with st.spinner("Online RL: replaying missed intraday data…"):
                intraday = fetch_intraday_history(
                    selected_symbol,
                    period="5d",
                    interval="1m",
                )
                catchup_info = online_manager.catch_up(
                    selected_symbol,
                    intraday,
                    history,
                    refresh_minutes,
                )
        if status == "OPEN":
            online_result = online_manager.observe_live(
                selected_symbol,
                history.iloc[-1],
                selected_quote,
                refresh_minutes,
            )
    except Exception as exc:
        catchup_info = {
            "transitions": 0,
            "status": f"Online RL unavailable: {exc}",
        }

    with rl_tab:
        st.markdown("#### Strategy signal")
        r1, r2, r3, r4 = st.columns(4)
        if online_result is not None:
            r1.metric("Action", online_result.action_name)
            r2.metric("Position", "LONG" if online_result.position else "FLAT")
            r3.metric(
                "Interval move",
                f"{online_result.reward * 100:+.3f}%"
                if online_result.reward is not None
                else "Waiting",
            )
            r4.metric("Status", "Active")
        else:
            r1.metric("Action", "WAIT")
            r2.metric("Position", "FLAT")
            r3.metric("Interval move", "Waiting")
            r4.metric("Status", "Inactive")

        if catchup_info is not None and catchup_info.get("transitions", 0):
            st.caption(
                f"Strategy state synchronized from {int(catchup_info['transitions']):,} available market intervals."
            )

        st.markdown(
            '<div class="disclaimer">The strategy signal is a separate short-horizon policy layer. '
            'It is not the same as the multi-horizon forecast and does not guarantee future performance.</div>',
            unsafe_allow_html=True,
        )


@st.fragment(run_every="1min")
def live_dashboard() -> None:
    status, now_label = market_state()
    live = get_live_data(tuple(NIFTY50_SYMBOLS))

    if live.empty:
        st.error(
            "No live market data returned. The market-data provider may be unavailable or rate-limited."
        )
        return

    scores = build_activity_scores(live, histories)
    predictions = inference_universe(model_bundle, histories)
    combined = add_potential_scores(scores, predictions, histories)
    if combined.empty:
        st.warning("No combined market screen is available.")
        return

    selected_symbol = st.session_state.selected_symbol

    if st.session_state.market_view == "stock":
        selected_live_rows = live[live["symbol"] == selected_symbol]
        selected_history = histories.get(selected_symbol)
        stock_live = combined[combined["symbol"] == selected_symbol]

        if selected_history is None or selected_live_rows.empty or stock_live.empty:
            st.warning("Selected instrument is not currently available.")
            return

        render_selected_instrument(
            selected_symbol=selected_symbol,
            selected_quote=selected_live_rows.iloc[0].to_dict(),
            row=stock_live.iloc[0],
            history=selected_history,
            status=status,
            refresh_minutes=LIVE_REFRESH_MINUTES,
            chart_range=st.session_state.chart_range,
            show_ema=st.session_state.show_ema,
            show_sr=st.session_state.show_sr,
            show_patterns=st.session_state.show_patterns,
        )
        return

    advances = int((combined["change_pct"] > 0).sum())
    declines = int((combined["change_pct"] < 0).sum())
    h1 = combined.nlargest(1, "activity_score").iloc[0]
    h2 = combined.nlargest(1, "up_probability").iloc[0]
    h3 = combined.nlargest(1, "potential").iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Market", status)
    c2.metric("Advancers / Decliners", f"{advances} / {declines}")
    c3.metric("Activity leader", str(h1["symbol"]))
    c4.metric(
        "Forecast leader",
        str(h2["symbol"]),
        f"{float(h2['up_probability'])*100:.0f}%",
    )
    c5.metric(
        "Potential leader",
        str(h3["symbol"]),
        f"{float(h3['potential']):.0f}/100",
    )
    st.caption(now_label)

    categories = build_market_categories(combined)

    st.radio(
        "Market",
        ["Overview", *categories.keys()],
        key="market_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    section = st.session_state.market_section
    descriptions = {
        "Most Active": "Highest cross-sectional activity using volume, movement, range and turnover proxies.",
        "Popular": "Popularity is represented by the project's volume and turnover attention proxy.",
        "Model Signals": "Stocks currently highest on modeled upward probability and potential score.",
        "Potential": "Heuristic potential from model probability, expected return, activity, volume, patterns and risk penalty.",
        "Low Attention": "Lower-attention instruments kept separate from the active and model-led screens.",
    }

    if section == "Overview":
        for name, frame in categories.items():
            render_stock_cards(
                frame,
                histories,
                name,
                descriptions[name],
                limit=5,
                key_prefix=f"overview_{name.lower().replace(' ', '_')}",
            )
    else:
        render_stock_cards(
            categories[section],
            histories,
            section,
            descriptions[section],
            limit=None,
            key_prefix=f"category_{section.lower().replace(' ', '_')}",
        )


live_dashboard()

