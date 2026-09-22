import numpy as np
import pandas as pd

from src.rl.online_loop import (
    ONLINE_FEATURE_SIZE,
    OnlineRLManager,
)


def _daily_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-09-18")],
            "Close": [100.0],
            "return_1d": [0.01],
            "return_5d": [0.02],
            "return_10d": [0.03],
            "rsi_14": [55.0],
            "macd": [0.5],
            "macd_hist": [0.1],
            "atr_14": [1.5],
            "adx_14": [20.0],
            "cci_14": [40.0],
            "stoch_k": [60.0],
            "stoch_d": [58.0],
            "willr_14": [-40.0],
            "volume_ratio_20": [1.1],
            "volatility_20d": [0.02],
            "momentum_10d": [0.01],
            "momentum_20d": [0.02],
            "bb_position": [0.6],
        }
    )


def test_online_manager_persists_live_state(tmp_path):
    manager = OnlineRLManager(tmp_path, sequence_length=10)

    quote = {
        "timestamp": pd.Timestamp("2026-09-21 09:30:00", tz="Asia/Kolkata"),
        "price": 100.2,
        "previous_close": 100.0,
        "change_pct": 0.2,
        "day_high": 100.4,
        "day_low": 99.8,
        "session_volume": 5000,
    }
    result = manager.observe_live(
        "TEST",
        _daily_row().iloc[-1],
        quote,
        cadence_minutes=1,
    )

    assert result.action_name in {"SELL", "HOLD", "BUY"}
    assert result.position in {0, 1}
    assert result.steps == 0
    assert (tmp_path / "tickerarc_online_lstm_dqn.pt").exists()
    assert (tmp_path / "tickerarc_online_rl_state.json").exists()

    restored = OnlineRLManager(tmp_path, sequence_length=10)
    restored_state = restored.runtime["symbols"]["TEST"]

    assert restored_state["pending"] is not None
    assert restored_state["last_timestamp"] is not None


def test_online_state_size():
    assert ONLINE_FEATURE_SIZE == 22
