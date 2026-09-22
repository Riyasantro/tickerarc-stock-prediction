"""TickerArc single-entry Streamlit application.

Run:
    streamlit run app.py

First launch downloads historical data and trains the global LSTM if no local
model checkpoint exists. Later refreshes only fetch the live layer and run
inference; training is not repeated automatically.
"""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.universe import NIFTY50_SYMBOLS
from src.analysis.agent import TickerArcAgent
from src.analysis.scoring import build_activity_scores, potential_score
from src.data.download import download_universe
from src.data.live import fetch_live_quotes
from src.data.pipeline import PROCESSED_DIR, process_downloaded_files
from src.features.chart_patterns import add_chart_pattern_features
from src.features.talib_features import add_all_candlestick_patterns
from src.models.trainer import (
    load_model,
    load_training_frames,
    predict_latest,
    train_global_model,
)
from src.models.walk_forward import walk_forward_evaluate

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "tickerarc_global_lstm.pt"
SCALER_PATH = MODEL_DIR / "tickerarc_scaler.joblib"
META_PATH = MODEL_DIR / "tickerarc_model_meta.json"


st.set_page_config(
    page_title="TickerArc",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #000 !important;
    color: #fff !important;
}
[data-testid="stSidebar"] {
    background: #050505 !important;
    border-right: 1px solid #222;
}
[data-testid="stMetric"] {
    background: #090909;
    border: 1px solid #252525;
    padding: 10px;
    border-radius: 4px;
}
.stButton > button {
    background: #fff !important;
    color: #000 !important;
    border: 1px solid #fff !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background: #090909;
    color: #fff;
    border-color: #333;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-top: 1rem;
}
.small-muted {
    color: #8e8e8e;
    font-size: 0.82rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=45, show_spinner=False)
def get_live_data(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Use 1-minute data first, then 5-minute data if the provider fails."""
    try:
        return fetch_live_quotes(symbols, interval="1m", lookback="5d")
    except Exception:
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


def bootstrap_project() -> None:
    """Fetch reproducible history, build features, and train once if needed."""
    if not any(PROCESSED_DIR.glob("*.parquet")):
        with st.status(
            "Downloading NIFTY 50 historical data…",
            expanded=True,
        ) as status:
            download_universe(NIFTY50_SYMBOLS, period="5y", interval="1d")
            status.update(
                label="Historical data downloaded",
                state="complete",
            )

        with st.status(
            "Building technical and pattern features…",
            expanded=True,
        ) as status:
            process_downloaded_files()
            status.update(
                label="Feature dataset ready",
                state="complete",
            )

    if not (MODEL_PATH.exists() and SCALER_PATH.exists() and META_PATH.exists()):
        with st.status(
            "Training the initial global LSTM…",
            expanded=True,
        ) as status:
            frames = load_training_frames(
                PROCESSED_DIR,
                NIFTY50_SYMBOLS,
            )
            progress = st.progress(0)
            message = st.empty()

            def callback(epoch: int, loss: float) -> None:
                progress.progress(min(epoch / 8, 1.0))
                message.write(
                    f"Epoch {epoch}/8 · training loss {loss:.5f}"
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
                label="LSTM model trained",
                state="complete",
            )

        get_model_bundle.clear()


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


def make_chart(frame: pd.DataFrame, symbol: str) -> go.Figure:
    data = frame.tail(120).copy()
    data["support"] = data["Low"].rolling(20).min().shift(1)
    data["resistance"] = data["High"].rolling(20).max().shift(1)

    figure = go.Figure(
        data=[
            go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                increasing_line_color="#fff",
                decreasing_line_color="#777",
                increasing_fillcolor="#fff",
                decreasing_fillcolor="#777",
                name=symbol,
            ),
            go.Scatter(
                x=data["Date"],
                y=data["support"],
                mode="lines",
                line={"color": "#555", "dash": "dot"},
                name="Support",
            ),
            go.Scatter(
                x=data["Date"],
                y=data["resistance"],
                mode="lines",
                line={"color": "#aaa", "dash": "dot"},
                name="Resistance",
            ),
        ]
    )
    figure.update_layout(
        paper_bgcolor="#000",
        plot_bgcolor="#000",
        font_color="#fff",
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            bgcolor="#000",
            font=dict(color="#fff"),
        ),
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#1b1b1b",
    )
    return figure


def inference_universe(
    model_bundle,
    histories: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    model, scaler, metadata = model_bundle
    rows: list[dict[str, object]] = []

    for symbol, frame in histories.items():
        try:
            rows.append(
                {
                    "symbol": symbol,
                    **predict_latest(
                        model,
                        scaler,
                        frame,
                        metadata,
                    ),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def add_potential_scores(
    scores: pd.DataFrame,
    predictions: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    out = scores.merge(predictions, on="symbol", how="left")
    if out.empty:
        return out

    pattern_scores = {}
    for symbol, frame in histories.items():
        try:
            pattern_scores[symbol] = pattern_summary(frame)["pattern_score"]
        except Exception:
            pattern_scores[symbol] = 0.0

    out["pattern_score"] = out["symbol"].map(pattern_scores).fillna(0.0)
    out["potential"] = out.apply(
        lambda row: potential_score(
            float(row.get("up_probability", 0.0)),
            float(row.get("return_5d", 0.0)),
            float(row.get("activity_score", 0.0)),
            float(row.get("volume_ratio", 0.0)),
            float(row.get("pattern_score", 0.0)),
            min(float(row.get("volatility_5d", 0.0)) * 100, 30),
        ),
        axis=1,
    ).round(1)
    return out


def render_table(frame: pd.DataFrame, title: str) -> None:
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )
    if frame.empty:
        st.info("No data available for this category.")
        return

    columns = [
        "symbol",
        "price",
        "change_pct",
        "activity_score",
        "volume_ratio",
        "attention_score",
        "direction",
        "up_probability",
        "return_5d",
        "potential",
    ]
    table = frame[[c for c in columns if c in frame.columns]].copy()
    table = table.rename(
        columns={
            "symbol": "Stock",
            "price": "Price",
            "change_pct": "Change %",
            "activity_score": "Activity",
            "volume_ratio": "Rel. Volume",
            "attention_score": "Attention",
            "direction": "LSTM Direction",
            "up_probability": "Up Prob.",
            "return_5d": "5D Return",
            "potential": "Potential",
        }
    )

    if "Price" in table:
        table["Price"] = table["Price"].round(2)
    if "Change %" in table:
        table["Change %"] = table["Change %"].round(2)
    if "Rel. Volume" in table:
        table["Rel. Volume"] = table["Rel. Volume"].round(2)
    if "Up Prob." in table:
        table["Up Prob."] = (
            table["Up Prob."].mul(100).round(1).astype(str) + "%"
        )
    if "5D Return" in table:
        table["5D Return"] = (
            table["5D Return"].mul(100).round(2).astype(str) + "%"
        )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


st.title("TickerArc")
st.caption("Stock Analysis and Prediction System")

with st.sidebar:
    st.markdown("### Controls")
    refresh_minutes = st.selectbox(
        "Refresh interval",
        [1, 3],
        index=1,
    )
    selected_symbol = st.selectbox(
        "Stock",
        NIFTY50_SYMBOLS,
        index=0,
    )

    st.markdown("---")
    st.markdown("### Model")
    st.write("Multi-Horizon LSTM v1")
    st.write("Horizons: 1D / 5D / 10D")
    st.write("RL policy: LSTM-DQN")

    if st.button("Rebuild model"):
        for path in (MODEL_PATH, SCALER_PATH, META_PATH):
            path.unlink(missing_ok=True)
        get_model_bundle.clear()
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


@st.fragment(run_every=f"{refresh_minutes}min")
def live_dashboard() -> None:
    status, now_label = market_state()
    live = get_live_data(tuple(NIFTY50_SYMBOLS))

    if live.empty:
        st.error(
            "No live market data returned. The free market-data provider "
            "may be unavailable or rate-limited."
        )
        return

    scores = build_activity_scores(live, histories)
    predictions = inference_universe(model_bundle, histories)
    combined = add_potential_scores(
        scores,
        predictions,
        histories,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market", status)
    c2.metric("Tracked", len(combined))
    c3.metric("Refresh", f"{refresh_minutes} min")
    c4.metric("Local time", now_label[-12:-6])

    high = (
        combined[combined["activity_band"] == "High"]
        .sort_values("activity_score", ascending=False)
        .head(10)
    )
    low = (
        combined[combined["activity_band"] == "Low"]
        .sort_values("activity_score")
        .head(10)
    )
    low_volume = (
        combined[combined["volume_band"] == "Low"]
        .sort_values("volume_ratio")
        .head(10)
    )
    low_attention = (
        combined[combined["attention_band"] == "Low"]
        .sort_values("attention_score")
        .head(10)
    )

    render_table(high, "High activity")
    render_table(low, "Low activity")
    render_table(low_volume, "Low volume")
    render_table(
        low_attention,
        "Low attention / popularity proxy",
    )

    st.markdown(
        '<div class="section-title">Selected stock</div>',
        unsafe_allow_html=True,
    )
    stock_live = combined[
        combined["symbol"] == selected_symbol
    ]
    history = histories.get(selected_symbol)

    if stock_live.empty or history is None:
        st.warning("Selected stock is not currently available.")
        return

    row = stock_live.iloc[0]
    patterns = pattern_summary(history)
    prediction = row.to_dict()

    # Feed actual pattern score into the analysis agent.
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

    left, right = st.columns([2.0, 1.0])
    with left:
        st.plotly_chart(
            make_chart(history, selected_symbol),
            use_container_width=True,
            config={"displaylogo": False},
        )

    with right:
        st.metric(
            "Current price",
            f"₹{float(row['price']):,.2f}",
            f"{float(row['change_pct']):+.2f}%",
        )
        st.metric(
            "Potential",
            f"{result.potential:.0f}/100",
        )
        st.metric(
            "LSTM direction",
            str(row.get("direction", "NEUTRAL")),
        )
        st.write(
            f"**Model:** "
            f"{row.get('model_name', 'TickerArc Multi-Horizon LSTM v1')}"
        )
        st.write(
            f"**Up probability:** "
            f"{float(row.get('up_probability', 0))*100:.1f}%"
        )
        st.write(
            f"**Neutral probability:** "
            f"{float(row.get('neutral_probability', 0))*100:.1f}%"
        )
        st.write(
            f"**Down probability:** "
            f"{float(row.get('down_probability', 0))*100:.1f}%"
        )

    p1, p2, p3, p4 = st.columns(4)
    current_price = float(row["price"])

    for column, horizon, return_key in [
        (p1, "1D", "return_1d"),
        (p2, "5D", "return_5d"),
        (p3, "10D", "return_10d"),
        (p4, "5D volatility", "volatility_5d"),
    ]:
        value = float(row[return_key])
        if horizon == "5D volatility":
            column.metric(
                horizon,
                f"{value*100:.2f}%",
            )
        else:
            column.metric(
                horizon,
                f"{value*100:+.2f}%",
                f"₹{current_price*(1+value):,.2f}",
            )

    a, b = st.columns(2)
    with a:
        st.markdown("**Candlestick patterns**")
        for item in patterns["candle_hits"][:8] or ["None detected"]:
            st.write(f"• {item}")

    with b:
        st.markdown("**Chart patterns**")
        for item in patterns["chart_hits"][:8] or ["None detected"]:
            st.write(f"• {item}")

    st.markdown("**Analysis Agent**")
    st.write(result.summary)
    st.caption(
        "The model uses the latest completed daily feature sequence; "
        "the live layer supplies the current market price/activity snapshot. "
        "Potential and prediction values are model/heuristic outputs, not "
        "guaranteed future results."
    )

    with st.expander("Advanced walk-forward evaluation"):
        st.write(
            "Expanding chronological folds are retrained on earlier data "
            "and scored only on later data. This is intentionally on-demand "
            "because retraining each fold is much heavier than live inference."
        )
        folds = st.slider(
            "Folds",
            min_value=2,
            max_value=4,
            value=3,
        )
        epochs = st.slider(
            "Epochs per fold",
            min_value=1,
            max_value=5,
            value=2,
        )

        if st.button("Run walk-forward evaluation"):
            with st.spinner("Running chronological folds…"):
                try:
                    metrics = walk_forward_evaluate(
                        history,
                        folds=folds,
                        train_epochs=epochs,
                    )
                    st.json(metrics)
                except Exception as exc:
                    st.error(str(exc))


live_dashboard()
