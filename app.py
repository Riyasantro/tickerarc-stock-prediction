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
    border-radius: 12px;    padding: 12px;    min-height: 128px;
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
    align-items:center;    background:#080d0a;    border:1px solid var(--border);
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

/* TradingView-inspired market terminal */
[data-testid="stSidebar"] { display:none !important; }
[data-testid="stAppViewContainer"] { background:#000 !important; }
[data-testid="stHeader"] { background:#000 !important; }
section.main > div { max-width:100% !important; padding-left:22px !important; padding-right:22px !important; }
.tv-topbar { display:flex;align-items:center;gap:20px;padding:7px 0 12px;border-bottom:1px solid #1b1d1b;margin-bottom:12px; }
.tv-brand { font-size:1.02rem;font-weight:900;letter-spacing:.08em;color:#f5f5f5;white-space:nowrap; }
.tv-brand-mark { font-size:1.15rem;margin-right:7px; }
.tv-nav-copy { color:#d9dfdb;font-size:.8rem;white-space:nowrap; }
.tv-nav-muted { color:#858f89;font-size:.8rem;white-space:nowrap; }
.tv-search { background:#202220;border:1px solid #303330;border-radius:18px;padding:7px 13px;color:#aeb6b2;font-size:.76rem; }
.market-breadcrumb { color:#76807b;font-size:.74rem;margin:4px 0 8px; }
.market-country { display:flex;align-items:center;gap:11px; }
.market-flag { font-size:1.9rem; }
.market-country-name { font-size:2.2rem;font-weight:900;letter-spacing:-.04em; }
.market-subline { color:#76807b;font-size:.76rem;margin-top:2px; }
.index-strip { display:flex;gap:9px;overflow:hidden;margin:18px 0 14px; }
.index-card { background:#101110;border:1px solid #282a28;border-radius:12px;padding:10px 12px;min-width:177px; }
.index-card-main { font-size:.74rem;color:#dfe5e1;font-weight:800; }
.index-card-value { font-size:1rem;font-weight:900;margin-top:2px; }
.index-up { color:#00d38a !important; } .index-down { color:#ff4b55 !important; }
.tv-panel-title { font-size:1.25rem;font-weight:880;margin:15px 0 8px;letter-spacing:-.02em; }
.tv-chart-shell { background:#000;padding-top:2px; }
.watchlist-panel { background:#090a09;border:1px solid #1f221f;border-radius:9px;overflow:hidden; }
.watchlist-head { padding:11px 11px;font-size:.82rem;font-weight:850;border-bottom:1px solid #202320;display:flex;justify-content:space-between; }
.watchlist-sub { color:#818a85;font-size:.63rem;font-weight:600; }
.watch-item { display:grid;grid-template-columns:1.05fr .9fr .62fr;gap:6px;padding:6px 8px;border-bottom:1px solid #181b18;align-items:center; }
.watch-item-symbol { font-weight:800;font-size:.69rem;color:#e4e9e5; }
.watch-item-price { text-align:right;color:#d2d8d4;font-size:.67rem; }
.watch-item-change { text-align:right;font-size:.67rem; }
.watch-item button { padding:0 !important;background:transparent !important;color:#e4e9e5 !important;border:0 !important;text-align:left !important;box-shadow:none !important; }
.watchlist-panel .stButton > button { background:transparent !important;color:#e4e9e5 !important;border:0 !important;padding:0 !important;box-shadow:none !important;text-align:left !important;font-size:.69rem !important;font-weight:800 !important; }
.watch-detail { padding:12px;border-top:1px solid #202320; }
.watch-detail-symbol { font-weight:900;font-size:.93rem; }
.watch-detail-price { font-weight:900;font-size:1.55rem;margin-top:2px; }
.breadth-wrap { background:#070807;border:1px solid #1c1f1c;border-radius:9px;padding:12px; }
.breadth-bar { display:flex;width:100%;height:12px;border-radius:8px;overflow:hidden;background:#20231f;margin:8px 0 4px; }
.breadth-adv { background:#18b990; } .breadth-dec { background:#ef3d4a; }
.breadth-labels { display:flex;justify-content:space-between;color:#858e89;font-size:.68rem; }
.list-panel { background:#070807;border-top:1px solid #1b1e1b;border-bottom:1px solid #1b1e1b; }
.quote-list-row { display:grid;grid-template-columns:1.55fr .9fr .75fr .62fr;gap:8px;align-items:center;padding:7px 2px;border-bottom:1px solid #151815;font-size:.71rem; }
.quote-list-row:last-child { border-bottom:0; }
.quote-symbol { font-weight:820;color:#e7ece8; }
.quote-price,.quote-change,.quote-tag { text-align:right;font-variant-numeric:tabular-nums; }
.quote-price { color:#d5dbd7; } .quote-tag { color:#7e8882;font-size:.63rem; }
.market-nav [data-testid="stRadio"] > div { gap:6px !important; }
.market-nav [role="radiogroup"] > label { border:1px solid #242724;background:#0d0e0d;border-radius:18px;padding:5px 10px !important; }
.market-nav [role="radiogroup"] > label:has(input:checked) { border-color:#4b6155;background:#16211a; }
.market-nav [role="radiogroup"] > label > div:last-child { color:#bfc8c2 !important;font-size:.71rem; }

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
def load_processed_histories(symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
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

    history_refresh_needed = not HISTORY_MARKER.exists()

    if history_refresh_needed:
        with st.status("Downloading maximum available NIFTY 50 history…", expanded=True) as status:
            download_universe(NIFTY50_SYMBOLS, period="max", interval="1d")
            HISTORY_MARKER.parent.mkdir(parents=True, exist_ok=True)
            HISTORY_MARKER.write_text("max-history-downloaded", encoding="utf-8")
            feature_refresh_needed = True
            status.update(label="Maximum available history downloaded", state="complete")

    if feature_refresh_needed:
        with st.status("Building technical and pattern features…", expanded=True) as status:
            process_downloaded_files()
            status.update(label="Feature dataset ready", state="complete")

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
                message.write(f"Epoch {epoch}/8 · validation loss {loss:.5f}")

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

        with st.status(f"Evaluating the latest {EVAL_YEARS} years…", expanded=True) as status:
            frames = load_training_frames(PROCESSED_DIR, NIFTY50_SYMBOLS)
            model, scaler, metadata = load_model(MODEL_DIR)
            metrics = evaluate_global_model(
                model,
                scaler,
                frames,
                sequence_length=int(metadata["sequence_length"]),            )
            metrics["baselines"] = evaluate_baselines(frames)
            EVAL_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            status.update(label="2-year holdout evaluation complete", state="complete")

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
    data = add_chart_pattern_features(add_all_candlestick_patterns(frame.copy()))
    latest = data.iloc[-1]
    candle_hits: list[str] = []
    signal_score = 0.0

    for column in [c for c in data.columns if c.startswith("candle_cdl")]:
        value = float(latest[column])
        if value == 0:
            continue
        label = column.replace("candle_cdl", "").replace("_", " ").strip().title()
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
        ("descending_triangle_20", "descending triangle", -1),        ("symmetrical_triangle_20", "symmetrical triangle", 0),
        ("rising_wedge_20", "rising wedge", -1),
        ("falling_wedge_20", "falling wedge", 1),        ("bull_flag", "bull flag", 1),
        ("bear_flag", "bear flag", -1),
    ]
    chart_hits: list[str] = []
    for column, label, direction in chart_map:
        if column in latest.index and float(latest[column]) == 1:
            chart_hits.append(label)
            signal_score += direction

    return {
        "candlestick": ", ".join(candle_hits[:4]) if candle_hits else "No strong recent TA-Lib candlestick signal",
        "chart": ", ".join(chart_hits[:4]) if chart_hits else "No strong heuristic chart structure",
        "pattern_score": float(np.tanh(signal_score / 5.0)),
        "candle_hits": candle_hits,
        "chart_hits": chart_hits,
    }

def _range_frame(frame: pd.DataFrame, range_name: str) -> pd.DataFrame:
    data = frame.copy()
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])
    days_map = {"1M": 31, "3M": 93, "6M": 186, "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5, "MAX": None}
    days = days_map.get(range_name)
    if days is not None and not data.empty:
        data = data[data["Date"] >= data["Date"].max() - pd.Timedelta(days=days)]
    return data.reset_index(drop=True)

def _select_stock(symbol: str) -> None:
    st.session_state.selected_symbol = symbol
    st.session_state.market_view = "stock"
    st.session_state.chart_focus = None

def _sidebar_stock_changed() -> None:
    st.session_state.selected_symbol = st.session_state.stock_selector
    st.session_state.market_view = "stock"
    st.session_state.chart_focus = None

def _focus_finding(finding: dict[str, object]) -> None:
    st.session_state.chart_focus = finding

def _clear_chart_focus() -> None:
    st.session_state.chart_focus = None

def build_market_categories(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "Most Active": combined.sort_values(["activity_score", "volume_ratio"], ascending=False),
        "Popular": combined.sort_values(["attention_score", "activity_score"], ascending=False),
        "Model Signals": combined.sort_values(["up_probability", "potential", "return_5d"], ascending=False),
        "Potential": combined.sort_values(["potential", "up_probability", "activity_score"], ascending=False),
        "Low Attention": combined.sort_values(["attention_score", "activity_score"], ascending=[True, False]),
    }

def render_stock_cards(
    frame: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    title: str,
    description: str,
    limit: int | None = 5,
    key_prefix: str = "cards",
) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.caption(description)
    view = frame.head(limit) if limit is not None else frame
    for row_start in range(0, len(view), 5):
        row = view.iloc[row_start:row_start + 5]
        cols = st.columns(len(row))
        for col, (_, item) in zip(cols, row.iterrows(), strict=True):
            symbol = str(item["symbol"])
            price = float(item["price"])
            change = float(item["change_pct"])
            up_prob = float(item.get("up_probability", 0.0))
            potential = float(item.get("potential", 0.0))
            activity = float(item.get("activity_score", 0.0))
            volume_ratio = float(item.get("volume_ratio", 0.0))
            with col:
                change_text = f"{change:+.2f}%"
                up_text = f"{up_prob * 100:.0f}%"
                potential_text = f"{potential:.0f}"
                activity_text = f"{activity:.0f}"
                volume_text = f"{volume_ratio:.2f}x"
                st.markdown(
                    f'<div class="stock-card">'
                    f'<div class="stock-symbol">{symbol}</div>'
                    f'<div class="stock-price">₹{price:,.2f}</div>'
                    f'<div class="{"stock-green" if change >= 0 else "stock-red"}">{change_text} session</div>'
                    f'<div style="margin-top:6px">{mini_candlestick_svg(histories.get(symbol))}</div>'
                    f'<div class="stock-meta">Up {up_text} · Potential {potential_text}</div>'                    f'<div class="stock-meta">Activity {activity_text} · Rel Vol {volume_text}</div>'
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

def add_potential_scores(
    scores: pd.DataFrame,
    predictions: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    out = scores.merge(predictions, on="symbol", how="left")
    if out.empty:
        return out

    pattern_scores: dict[str, float] = {}
    for symbol, frame in histories.items():
        latest = frame.iloc[-1]
        candle_signal = float(latest.get("candlestick_signal_score", 0.0))
        chart_signal = 0.0
        for column, direction in [
            ("breakout_20", 1),
            ("breakdown_20", -1),
            ("double_top_30", -1),
            ("double_bottom_30", 1),
            ("head_shoulders_30", -1),
            ("inverse_head_shoulders_30", 1),
            ("ascending_triangle_20", 1),
            ("descending_triangle_20", -1),
            ("rising_wedge_20", -1),
            ("falling_wedge_20", 1),
            ("bull_flag", 1),
            ("bear_flag", -1),
        ]:
            chart_signal += float(latest.get(column, 0.0)) * direction
        pattern_scores[symbol] = float(np.tanh((candle_signal / 5.0) + (chart_signal / 3.0)))

    out["pattern_score"] = out["symbol"].map(pattern_scores).fillna(0.0)
    out["potential"] = out.apply(
        lambda row: potential_score(
            float(row.get("up_probability", 0.0)),
            float(row.get("return_5d", 0.0)),            float(row.get("activity_score", 0.0)),
            float(row.get("volume_ratio", 0.0)),
            float(row.get("pattern_score", 0.0)),
            min(float(row.get("volatility_5d", 0.0)) * 100, 30),
        ),
        axis=1,
    ).round(1)
    return out

def render_model_performance() -> None:
    if not EVAL_PATH.exists():
        st.info("Holdout evaluation is not available yet.")
        return
    try:
        metrics = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    except Exception:
        st.warning("Holdout evaluation could not be read.")
        return
    holdout = metrics.get("aggregate", metrics)
    baseline = metrics.get("baselines", {})
    if not holdout:
        st.info("No evaluation metrics are recorded yet.")
        return

    st.markdown("#### 2-year holdout evaluation")
    labels, mae_values, acc_values = [], [], []
    for name, values in [("TickerArc LSTM", holdout)] + list(baseline.items()):
        if isinstance(values, dict):
            labels.append(name)
            mae_values.append(float(values.get("mae_5d", 0.0)))
            acc_values.append(float(values.get("direction_accuracy", 0.0)))

    p1, p2 = st.columns(2)
    with p1:
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=[v * 100 for v in mae_values],
                marker_color=["#00e676"] + ["#465149"] * max(len(labels) - 1, 0),
                text=[f"{v * 100:.2f}%" for v in mae_values],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="5D return MAE · lower is better",
            paper_bgcolor="#060806",
            plot_bgcolor="#060806",
            font={"color": "#e8f2eb"},
            margin={"l": 10, "r": 10, "t": 45, "b": 20},
            height=310,
            yaxis_title="MAE (%)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    with p2:
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=[v * 100 for v in acc_values],
                marker_color=["#00e676"] + ["#465149"] * max(len(labels) - 1, 0),
                text=[f"{v * 100:.1f}%" for v in acc_values],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Direction accuracy · 2Y holdout",
            paper_bgcolor="#060806",
            plot_bgcolor="#060806",
            font={"color": "#e8f2eb"},
            margin={"l": 10, "r": 10, "t": 45, "b": 20},
            height=310,
            yaxis_title="Accuracy (%)",
            yaxis={"range": [0, 100]},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

def inference_universe(model_bundle, histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    model, scaler, metadata = model_bundle
    rows: list[dict[str, object]] = []
    for symbol, frame in histories.items():
        try:
            rows.append({"symbol": symbol, **predict_latest(model, scaler, frame, metadata)})
        except Exception:
            continue
    return pd.DataFrame(rows)

def make_pattern_annotation_text(finding: dict[str, object]) -> str:
    title = str(finding.get("title", ""))
    detail = str(finding.get("detail", ""))
    return title if not detail else f"{title}<br><sup>{detail}</sup>"

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
    patterns = pattern_summary(history)
    prediction = row.to_dict()
    prediction["risk_penalty"] = min(float(prediction.get("volatility_5d", 0.0)) * 100, 30)
    result = TickerArcAgent().analyze(selected_symbol, row.to_dict(), prediction, patterns)
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
    q1.metric("Price", f"₹{float(row['price']):,.2f}", f"{float(row['change_pct']):+.2f}%")
    q2.metric("1D model", f"{float(row['return_1d']) * 100:+.2f}%")
    q3.metric("5D model", f"{float(row['return_5d']) * 100:+.2f}%")
    q4.metric("10D model", f"{float(row['return_10d']) * 100:+.2f}%")
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
        st.markdown("</div>", unsafe_allow_html=True)
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
            selected_symbol,            selected_range,
            show_ema=local_ema,
            show_sr=local_sr,
            show_patterns=local_patterns,
            focus_date=focus_date,
            focus_label=focus_title,
        ),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
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
        st.markdown("</div>", unsafe_allow_html=True)

    details_tab, signals_tab, performance_tab, rl_tab = st.tabs(["Overview", "Patterns", "Forecast", "Strategy"])

    with details_tab:
        d1, d2 = st.columns([1.35, 1])
        with d1:
            st.markdown("#### Session profile")
            s1, s2, s3 = st.columns(3)
            s1.metric("Open", f"₹{float(selected_quote.get('open', row['price'])):,.2f}")
            s2.metric("High", f"₹{float(selected_quote.get('day_high', row['price'])):,.2f}")
            s3.metric("Low", f"₹{float(selected_quote.get('day_low', row['price'])):,.2f}")
            st.markdown(
                f'<div class="small-muted">Session volume: {float(selected_quote.get("session_volume", 0)):,.0f} · '
                f'Relative volume: {float(row.get("volume_ratio", 0)):.2f}x · '
                f'Activity score: {float(row.get("activity_score", 0)):.1f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("#### Forecast distribution")
            donut = go.Figure(
                go.Pie(
                    labels=["Down", "Neutral", "Up"],
                    values=[float(row.get("down_probability", 0)), float(row.get("neutral_probability", 0)), float(row.get("up_probability", 0))],
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
            st.plotly_chart(donut, use_container_width=True, config={"displaylogo": False})
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
            candle_items = patterns["candle_hits"][:10] or ["No strong recent TA-Lib signal"]
            for item in candle_items:
                st.markdown(f'<span class="pill pill-green">{item}</span>', unsafe_allow_html=True)
        with p2:
            st.markdown("#### Chart patterns")
            chart_items = patterns["chart_hits"][:10] or ["No strong heuristic chart structure"]
            for item in chart_items:
                st.markdown(f'<span class="pill pill-muted">{item}</span>', unsafe_allow_html=True)

        st.markdown("#### Forecast distribution")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Up probability", f"{float(row['up_probability']) * 100:.1f}%")
        f2.metric("Neutral probability", f"{float(row['neutral_probability']) * 100:.1f}%")
        f3.metric("Down probability", f"{float(row['down_probability']) * 100:.1f}%")
        f4.metric("5D volatility", f"{float(row['volatility_5d']) * 100:.2f}%")

    with performance_tab:
        render_model_performance()

    online_result = None
    catchup_info = None
    try:
        online_manager = get_online_rl_manager()
        if online_manager.needs_catchup(selected_symbol, selected_quote["timestamp"], refresh_minutes):
            with st.spinner("Online RL: replaying missed intraday data…"):
                intraday = fetch_intraday_history(selected_symbol, period="5d", interval="1m")
                catchup_info = online_manager.catch_up(selected_symbol, intraday, history, refresh_minutes)
        if status == "OPEN":
            online_result = online_manager.observe_live(selected_symbol, history.iloc[-1], selected_quote, refresh_minutes)
    except Exception as exc:
        catchup_info = {"transitions": 0, "status": f"Online RL unavailable: {exc}"}

    with rl_tab:
        st.markdown("#### Strategy signal")
        r1, r2, r3, r4 = st.columns(4)
        if online_result is not None:
            r1.metric("Action", online_result.action_name)
            r2.metric("Position", "LONG" if online_result.position else "FLAT")
            r3.metric("Interval move", f"{online_result.reward * 100:+.3f}%" if online_result.reward is not None else "Waiting")
            r4.metric("Status", "Active")
        else:
            r1.metric("Action", "WAIT")
            r2.metric("Position", "FLAT")
            r3.metric("Interval move", "Waiting")
            r4.metric("Status", "Inactive")
        if catchup_info is not None and catchup_info.get("transitions", 0):
            st.caption(f"Strategy state synchronized from {int(catchup_info['transitions']):,} available market intervals.")
        st.markdown(
            '<div class="disclaimer">The strategy signal is a separate short-horizon policy layer. '
            'It is not the same as the multi-horizon forecast and does not guarantee future performance.</div>',
            unsafe_allow_html=True,
        )

def _market_proxy_series(histories: dict[str, pd.DataFrame], range_name: str = "1Y") -> pd.DataFrame:
    series = []
    for symbol, frame in histories.items():
        if frame is None or frame.empty or not {"Date", "Close"}.issubset(frame.columns):            continue
        data = frame[["Date", "Close"]].copy()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data = data.dropna().drop_duplicates("Date").set_index("Date")["Close"]
        if len(data) >= 20:
            series.append(data.rename(symbol))
    if not series:
        return pd.DataFrame(columns=["Date", "Value"])
    close = pd.concat(series, axis=1).sort_index().ffill()
    normalized = close.div(close.iloc[0].replace(0, np.nan)).mul(100)
    composite = normalized.mean(axis=1, skipna=True).dropna()
    days_map = {"1M":31, "3M":93, "6M":186, "1Y":365, "3Y":1095, "5Y":1825, "ALL":None}
    days = days_map.get(range_name)
    if days is not None and not composite.empty:
        composite = composite[composite.index >= composite.index.max() - pd.Timedelta(days=days)]
    return composite.rename("Value").reset_index().rename(columns={"index":"Date"})

def make_market_proxy_chart(histories: dict[str, pd.DataFrame], range_name: str = "1Y") -> go.Figure:
    data = _market_proxy_series(histories, range_name)
    fig = go.Figure()
    if data.empty:
        return fig
    rising = float(data["Value"].iloc[-1]) >= float(data["Value"].iloc[0])
    line_color = "#00e676" if rising else "#ff4b55"
    fill_color = "rgba(0,230,118,.08)" if rising else "rgba(255,75,85,.08)"
    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["Value"], mode="lines",
        name="NIFTY 50 basket", line={"color":line_color,"width":2.0},
        fill="tozeroy", fillcolor=fill_color,
        hovertemplate="%{x|%d %b %Y}<br>Basket index %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#000", plot_bgcolor="#000", height=385,
        margin={"l":0,"r":8,"t":5,"b":0}, showlegend=False,
        hovermode="x unified", dragmode="pan",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#707873")
    fig.update_yaxes(showgrid=True, gridcolor="#111511", zeroline=False, color="#707873", side="right")
    return fig

def render_tv_topbar() -> None:
    nav = ["Overview", "Most Active", "Popular", "Model Signals", "Potential", "Low Attention"]
    current = st.session_state.get("market_section", "Overview")
    st.markdown(
        '<div class="tv-topbar">'
        '<div class="tv-brand"><span class="tv-brand-mark">◼</span>TICKERARC</div>'
        '<div class="tv-nav-copy">Markets</div>'
        '<div class="tv-nav-copy">Screeners</div>'
        '<div class="tv-nav-copy">Models</div>'
        '<div class="tv-nav-muted">Analytics</div>'
        '<div class="tv-search">⌕&nbsp;&nbsp;Search symbol</div>'
        '<div style="flex:1"></div>'
        '<div class="tv-nav-muted">NIFTY 50</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.container():
        selected = st.radio(
            "Market",
            nav,
            index=nav.index(current) if current in nav else 0,
            key="market_section",
            horizontal=True,
            label_visibility="collapsed",
        )

def render_watchlist_panel(combined: pd.DataFrame) -> None:
    st.markdown('<div class="watchlist-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="watchlist-head"><span>Watchlist</span><span class="watchlist-sub">NIFTY 50 · LIVE</span></div>',
        unsafe_allow_html=True,
    )
    view = combined.sort_values(["activity_score","volume_ratio"], ascending=False).head(10)
    for _, item in view.iterrows():
        symbol = str(item["symbol"])
        price = float(item["price"])
        change = float(item["change_pct"])
        cls = "index-up" if change >= 0 else "index-down"
        c1, c2 = st.columns([1.05, 1.45])
        with c1:
            st.button(symbol.replace(".NS",""), key=f"watch_select_{symbol}", on_click=_select_stock, args=(symbol,), use_container_width=True)
        with c2:
            st.markdown(
                f'<div class="watch-item-price">₹{price:,.2f} '
                f'<span class="watch-item-change {cls}">{change:+.2f}%</span></div>',
                unsafe_allow_html=True,
            )
    selected_symbol = st.session_state.get("selected_symbol")
    row = combined[combined["symbol"] == selected_symbol]
    if not row.empty:
        item = row.iloc[0]
        change = float(item["change_pct"])
        cls = "index-up" if change >= 0 else "index-down"
        st.markdown(
            '<div class="watch-detail">'
            f'<div class="watch-detail-symbol">{selected_symbol.replace(".NS","")} · NSE</div>'
            f'<div class="watch-detail-price">₹{float(item["price"]):,.2f}</div>'
            f'<div class="{cls}" style="font-size:.77rem;font-weight:800;">{change:+.2f}% today</div>'
            f'<div style="color:#7e8882;font-size:.67rem;margin-top:6px;">'
            f'1D {float(item.get("change_pct",0)):+.2f}% · '
            f'5D model {float(item.get("return_5d",0))*100:+.2f}% · '
            f'Up {float(item.get("up_probability",0))*100:.0f}% · '
            f'Potential {float(item.get("potential",0)):.0f}/100'
            '</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

def _render_ranked_rows(frame: pd.DataFrame, metric: str, limit: int = 6) -> None:
    st.markdown('<div class="list-panel">', unsafe_allow_html=True)
    for _, item in frame.head(limit).iterrows():
        symbol = str(item["symbol"]).replace(".NS","")
        price = float(item["price"])
        change = float(item["change_pct"])
        cls = "index-up" if change >= 0 else "index-down"
        if metric == "Volume":
            val = float(item.get("session_volume", 0))
            tag = f"{val/1e6:.2f}M"
        elif metric == "Volatility":
            val = float(item.get("volatility_20d", 0))*100
            tag = f"{val:.2f}%"
        else:
            val = float(item.get("return_5d", 0))*100
            tag = f"{val:+.2f}%"
        st.markdown(
            f'<div class="quote-list-row"><div class="quote-symbol">{symbol}</div>'
            f'<div class="quote-price">₹{price:,.2f}</div>'
            f'<div class="quote-change {cls}">{change:+.2f}%</div>'
            f'<div class="quote-tag">{tag}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

def render_market_home(combined: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> None:
    status, now_label = market_state()
    advances = int((combined["change_pct"] > 0).sum())
    declines = int((combined["change_pct"] < 0).sum())
    unchanged = int((combined["change_pct"] == 0).sum())
    avg_up = float(pd.to_numeric(combined["up_probability"], errors="coerce").mean())
    proxy = _market_proxy_series(histories, st.session_state.get("market_range","1Y"))
    proxy_value = float(proxy["Value"].iloc[-1]) if not proxy.empty else 100.0
    proxy_start = float(proxy["Value"].iloc[0]) if not proxy.empty else 100.0
    st.markdown('<div class="market-breadcrumb">Markets &nbsp;/&nbsp; India</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="market-country"><span class="market-flag">🇮🇳</span>'
        '<div><div class="market-country-name">India</div>'
        f'<div class="market-subline">NIFTY 50 market · {status} · {now_label}</div></div></div>',
        unsafe_allow_html=True,    )

    card_specs = [
        ("NIFTY 50 basket", f"{proxy_value:.2f}", (proxy_value/proxy_start-1.0) if proxy_start else 0.0),
        ("Advancers / decliners", f"{advances} / {declines}", 0.0),
        ("Avg model up", f"{avg_up*100:.1f}%", 0.0),
        ("Activity", f"{float(combined['activity_score'].mean()):.0f}/100", 0.0),
        ("Potential", f"{float(combined['potential'].mean()):.0f}/100", 0.0),
    ]
    html = ['<div class="index-strip">']
    for label, value, change in card_specs:
        cls = "index-up" if change >= 0 else "index-down"
        sub = f"{change*100:+.2f}%" if label == "NIFTY 50 basket" else "TickerArc composite"
        html.append(
            f'<div class="index-card"><div class="index-card-main">{label}</div>'
            f'<div class="index-card-value">{value}</div><div class="{cls}" style="font-size:.65rem;">{sub}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    left, right = st.columns([4.65, 1.32], gap="small")
    with left:
        title_col, range_col = st.columns([3.4, 1.1])
        with title_col:
            st.markdown('<div class="tv-panel-title">Market overview</div>', unsafe_allow_html=True)
        with range_col:
            selected_range = st.selectbox(
                "Range", ["1M","3M","6M","1Y","3Y","5Y","ALL"],
                index=["1M","3M","6M","1Y","3Y","5Y","ALL"].index(st.session_state.get("market_range","1Y")),
                key="market_range",
                label_visibility="collapsed",
            )
            st.session_state.market_range = selected_range
        st.markdown('<div class="tv-chart-shell">', unsafe_allow_html=True)
        st.plotly_chart(
            make_market_proxy_chart(histories, selected_range),
            use_container_width=True,
            config={"displaylogo":False,"scrollZoom":True,"modeBarButtonsToRemove":["lasso2d","select2d"]},
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        render_watchlist_panel(combined)

    b1, b2 = st.columns([1.0, 1.35], gap="small")
    with b1:
        st.markdown('<div class="tv-panel-title">Market breadth</div>', unsafe_allow_html=True)
        total = max(advances + declines + unchanged, 1)
        adv_w, dec_w = advances/total, declines/total
        st.markdown(
            f'<div class="breadth-wrap">'
            f'<div style="font-size:.7rem;color:#9da7a1;">Advancing&nbsp; {advances:,} &nbsp;&nbsp; Declining&nbsp; {declines:,}</div>'
            f'<div class="breadth-bar"><div class="breadth-adv" style="width:{adv_w*100:.2f}%"></div>'
            f'<div class="breadth-dec" style="width:{dec_w*100:.2f}%"></div></div>'
            f'<div class="breadth-labels"><span>{adv_w*100:.0f}% advancing</span><span>{dec_w*100:.0f}% declining</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        stock_count = max(len(histories), 1)
        ma20 = sum(
            1 for f in histories.values()
            if f is not None and len(f) >= 20 and float(pd.to_numeric(f["Close"],errors="coerce").iloc[-1]) >= float(pd.to_numeric(f["Close"],errors="coerce").tail(20).mean())
        )
        ma50 = sum(
            1 for f in histories.values()
            if f is not None and len(f) >= 50 and float(pd.to_numeric(f["Close"],errors="coerce").iloc[-1]) >= float(pd.to_numeric(f["Close"],errors="coerce").tail(50).mean())
        )
        st.markdown(
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;">'
            f'<div class="breadth-wrap"><div style="color:#7f8883;font-size:.64rem;">Above 20D average</div><div style="font-size:1.2rem;font-weight:850;">{ma20/stock_count*100:.0f}%</div></div>'
            f'<div class="breadth-wrap"><div style="color:#7f8883;font-size:.64rem;">Above 50D average</div><div style="font-size:1.2rem;font-weight:850;">{ma50/stock_count*100:.0f}%</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown('<div class="tv-panel-title">New highs & lows</div>', unsafe_allow_html=True)
        st.markdown('<div class="list-panel">', unsafe_allow_html=True)
        for lb in (20, 60, 120, 250):
            highs = lows = 0
            for f in histories.values():
                if f is None or len(f) < lb + 1:
                    continue
                close = pd.to_numeric(f["Close"], errors="coerce").dropna()
                if len(close) < lb + 1:
                    continue
                latest = float(close.iloc[-1])
                prior = close.iloc[-lb-1:-1]
                highs += int(latest >= float(prior.max()))
                lows += int(latest <= float(prior.min()))
            st.markdown(
                f'<div class="quote-list-row"><div class="quote-symbol">{lb}D</div>'
                f'<div class="quote-tag">New highs</div><div class="quote-change index-up">{highs}</div>'
                f'<div class="quote-tag">New lows <span class="index-down">{lows}</span></div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="tv-panel-title">Highest volume stocks</div>', unsafe_allow_html=True)
    _render_ranked_rows(combined.sort_values("session_volume", ascending=False), "Volume")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="tv-panel-title">Most volatile stocks</div>', unsafe_allow_html=True)
        _render_ranked_rows(combined.sort_values("volatility_20d", ascending=False), "Volatility")
    with c2:
        st.markdown('<div class="tv-panel-title">Model signals</div>', unsafe_allow_html=True)
        _render_ranked_rows(combined.sort_values(["up_probability","potential"], ascending=False), "Model")

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown('<div class="tv-panel-title">Stock gainers</div>', unsafe_allow_html=True)
        _render_ranked_rows(combined.sort_values("change_pct", ascending=False), "Model", limit=7)
    with g2:
        st.markdown('<div class="tv-panel-title">Stock losers</div>', unsafe_allow_html=True)
        _render_ranked_rows(combined.sort_values("change_pct", ascending=True), "Model", limit=7)


@st.fragment(run_every="1min")
def live_dashboard() -> None:
    status, now_label = market_state()
    live = get_live_data(tuple(NIFTY50_SYMBOLS))
    if live.empty:
        st.error("No live market data returned. The market-data provider may be unavailable or rate-limited.")
        return

    scores = build_activity_scores(live, histories)
    predictions = inference_universe(model_bundle, histories)
    combined = add_potential_scores(scores, predictions, histories)
    if combined.empty:
        st.warning("No combined market screen is available.")
        return

    render_tv_topbar()

    selected_symbol = st.session_state.selected_symbol
    if st.session_state.market_view == "stock":
        selected_live_rows = live[live["symbol"] == selected_symbol]
        selected_history = histories.get(selected_symbol)
        stock_live = combined[combined["symbol"] == selected_symbol]
        if selected_history is None or selected_live_rows.empty or stock_live.empty:
            st.warning("Selected instrument is not currently available.")
            return

        left_stock, right_watch = st.columns([4.65, 1.32], gap="small")
        with left_stock:
            render_selected_instrument(
                selected_symbol,
                selected_live_rows.iloc[0].to_dict(),
                stock_live.iloc[0],
                selected_history,
                status,
                LIVE_REFRESH_MINUTES,
                st.session_state.chart_range,
                st.session_state.show_ema,
                st.session_state.show_sr,
                st.session_state.show_patterns,
            )
        with right_watch:
            render_watchlist_panel(combined)
        return

    categories = build_market_categories(combined)
    section = st.session_state.market_section
    if section == "Overview":
        render_market_home(combined, histories)
        return

    frame = categories.get(section, combined)
    st.markdown('<div class="market-breadcrumb">Markets / India / NIFTY 50</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="market-country-name">{section}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="market-subline">Cross-sectional NIFTY 50 screen with live quote activity and TickerArc model outputs.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="tv-panel-title">Stocks</div>', unsafe_allow_html=True)
    render_stock_cards(
        frame,
        histories,
        section,
        {
            "Most Active":"Highest cross-sectional activity using volume, movement, range and turnover proxies.",
            "Popular":"Popularity is represented by the project's volume and turnover attention proxy.",
            "Model Signals":"Stocks highest on modeled upward probability and potential score.",
            "Potential":"Heuristic potential from model probability, expected return, activity, volume, patterns and risk penalty.",
            "Low Attention":"Lower-attention instruments kept separate from active and model-led screens.",
        }.get(section, ""),
        limit=None,
        key_prefix=f"tv_{section.lower().replace(' ','_')}",
    )

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

for state_key, default in [
    ("market_view", "market"),
    ("selected_symbol", NIFTY50_SYMBOLS[0]),
    ("market_section", "Overview"),
    ("chart_focus", None),
    ("chart_range", "6M"),
    ("show_ema", True),
    ("show_sr", True),
    ("show_patterns", True),
    ("market_range", "1Y"),
]:
    if state_key not in st.session_state:
        st.session_state[state_key] = default

live_dashboard()