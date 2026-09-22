"""Persistent online LSTM-DQN reinforcement loop for live TickerArc learning.

The loop treats each refresh as a small RL decision:
    state -> Sell/Hold/Buy -> wait for the next cadence -> reward -> DQN update

The model checkpoint, replay memory and per-symbol runtime state are persisted
locally so the process can stop and resume later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.rl.recurrent_dqn import ReplayBuffer, RecurrentDQNAgent


ONLINE_SEQUENCE_LENGTH = 60
ONLINE_FEATURE_SIZE = 22
REPLAY_LIMIT_ON_DISK = 1000
CATCHUP_MAX_DAYS = 5
FIRST_RUN_TRAIN_DAYS = 2
TRANSACTION_COST = 0.0005

BASE_FEATURES = [
    "return_1d",
    "return_5d",
    "return_10d",
    "rsi_14",
    "macd",
    "macd_hist",
    "atr_14",
    "adx_14",
    "cci_14",
    "stoch_k",
    "stoch_d",
    "willr_14",
    "volume_ratio_20",
    "volatility_20d",
    "momentum_10d",
    "momentum_20d",
    "bb_position",
]


ACTION_NAMES = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}


@dataclass
class OnlineResult:
    action: int
    action_name: str
    position: int
    reward: float | None
    loss: float | None
    trained: bool
    catchup_transitions: int
    replay_size: int
    steps: int
    timestamp: str
    status: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _parse_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _timestamp_key(value: Any) -> str:
    return _parse_timestamp(value).isoformat()


def _daily_vector(row: pd.Series) -> np.ndarray:
    close = max(_safe_float(row.get("Close")), 1e-8)

    values = [
        _safe_float(row.get("return_1d")),
        _safe_float(row.get("return_5d")),
        _safe_float(row.get("return_10d")),
        _safe_float(row.get("rsi_14")) / 100.0,
        _safe_float(row.get("macd")) / close,
        _safe_float(row.get("macd_hist")) / close,
        _safe_float(row.get("atr_14")) / close,
        _safe_float(row.get("adx_14")) / 100.0,
        _safe_float(row.get("cci_14")) / 200.0,
        _safe_float(row.get("stoch_k")) / 100.0,
        _safe_float(row.get("stoch_d")) / 100.0,
        _safe_float(row.get("willr_14")) / 100.0,
        _safe_float(row.get("volume_ratio_20")),
        _safe_float(row.get("volatility_20d")),
        _safe_float(row.get("momentum_10d")),
        _safe_float(row.get("momentum_20d")),
        _safe_float(row.get("bb_position")),
    ]
    return np.asarray(values, dtype=np.float32)


def _action_position(action: int, previous_position: int) -> int:
    if action == 0:
        return 0
    if action == 2:
        return 1
    return previous_position


def _prepare_intraday(
    frame: pd.DataFrame,
    cadence_minutes: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    data = frame.copy()
    data["timestamp"] = data["timestamp"].map(_parse_timestamp)
    data = data.set_index("timestamp").sort_index()
    rule = f"{cadence_minutes}min"

    data = data.resample(rule).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    data = data.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    data["Volume"] = data["Volume"].fillna(0.0)
    return data


def _intraday_components(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["Close"].astype(float)
    previous_close = close.shift(1)
    close_3 = close.shift(3)

    data["bar_return_1"] = (close / previous_close - 1.0).replace(
        [np.inf, -np.inf], np.nan
    )
    data["bar_return_3"] = (close / close_3 - 1.0).replace(
        [np.inf, -np.inf], np.nan
    )

    volume = data["Volume"].astype(float).fillna(0.0)
    median_volume = (
        volume.rolling(20, min_periods=3)
        .median()
        .replace(0.0, np.nan)
    )
    data["bar_volume_ratio"] = (
        volume / median_volume
    ).replace([np.inf, -np.inf], np.nan)

    session = pd.Series(data.index.date, index=data.index)
    data["session_high"] = data["High"].groupby(session).cummax()
    data["session_low"] = data["Low"].groupby(session).cummin()

    session_range = (
        data["session_high"] - data["session_low"]
    ).replace(0.0, np.nan)
    data["session_position"] = (
        (close - data["session_low"]) / session_range
    ).replace([np.inf, -np.inf], np.nan)

    return data.fillna(0.0)


def _daily_lookup(daily_history: pd.DataFrame) -> dict[Any, pd.Series]:
    if daily_history.empty:
        return {}

    data = daily_history.copy()
    dates = pd.to_datetime(data["Date"])
    lookup: dict[Any, pd.Series] = {}
    for index, date in enumerate(dates):
        lookup[date.date()] = data.iloc[index]
    return lookup


def _state_matrix_from_bars(
    daily_history: pd.DataFrame,
    bars: pd.DataFrame,
) -> np.ndarray:
    daily_map = _daily_lookup(daily_history)
    components = _intraday_components(bars)
    vectors: list[np.ndarray] = []

    last_daily: pd.Series | None = None
    for timestamp, row in components.iterrows():
        daily_row = daily_map.get(timestamp.date(), last_daily)
        if daily_row is None:
            continue
        last_daily = daily_row

        daily_values = _daily_vector(daily_row)
        daily_close = max(_safe_float(daily_row.get("Close")), 1e-8)
        session_return = (
            _safe_float(row["Close"]) / daily_close - 1.0
        )

        live_values = np.asarray(
            [
                _safe_float(row["bar_return_1"]),
                _safe_float(row["bar_return_3"]),
                _safe_float(row["bar_volume_ratio"]),
                _safe_float(row["session_position"]),
                session_return,
            ],
            dtype=np.float32,
        )

        vectors.append(
            np.concatenate(
                [daily_values, live_values],
                axis=0,
            )
        )

    if not vectors:
        return np.empty((0, ONLINE_FEATURE_SIZE), dtype=np.float32)

    return np.asarray(vectors, dtype=np.float32)


def _pad_sequence(
    values: np.ndarray,
    sequence_length: int = ONLINE_SEQUENCE_LENGTH,
) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("Cannot build an online RL sequence from empty data.")
    if len(values) >= sequence_length:
        return values[-sequence_length:].astype(np.float32)

    padding = np.repeat(
        values[:1],
        sequence_length - len(values),
        axis=0,
    )
    return np.concatenate([padding, values], axis=0).astype(np.float32)


class OnlineRLManager:
    """Persistent live RL manager shared by the Streamlit process."""

    def __init__(
        self,
        model_dir: Path,
        sequence_length: int = ONLINE_SEQUENCE_LENGTH,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = (
            self.model_dir / "tickerarc_online_lstm_dqn.pt"
        )
        self.state_path = (
            self.model_dir / "tickerarc_online_rl_state.json"
        )
        self.sequence_length = sequence_length

        self.buffer = ReplayBuffer(capacity=5_000)
        self.agent = RecurrentDQNAgent(
            input_size=ONLINE_FEATURE_SIZE,
            hidden_size=128,
            batch_size=64,
            gamma=0.99,
            target_update=250,
        )

        if self.checkpoint_path.exists():
            try:
                self.agent.load_checkpoint(
                    self.checkpoint_path,
                    buffer=self.buffer,
                )
            except Exception:
                # A corrupted/stale online checkpoint should not prevent the
                # dashboard from starting; the next save replaces it.
                self.buffer.clear()

        self.runtime = self._load_runtime_state()

    def _load_runtime_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"version": 1, "symbols": {}}
        try:
            state = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
            if not isinstance(state, dict):
                raise ValueError("Online RL state must be an object.")
            state.setdefault("version", 1)
            state.setdefault("symbols", {})
            return state
        except Exception:
            return {"version": 1, "symbols": {}}

    def _save(self) -> None:
        self.agent.save_checkpoint(
            self.checkpoint_path,
            buffer=self.buffer,
            replay_limit=REPLAY_LIMIT_ON_DISK,
        )
        self.state_path.write_text(
            json.dumps(
                self.runtime,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _symbol_state(self, symbol: str) -> dict[str, Any]:
        state = self.runtime["symbols"].setdefault(
            symbol,
            {
                "last_timestamp": None,
                "position": 0,
                "last_price": None,
                "last_session_volume": None,
                "recent_prices": [],
                "recent_volumes": [],
                "state_history": [],
                "pending": None,
                "trained_steps": 0,
                "catchup_transitions": 0,
            },
        )
        return state

    def needs_catchup(
        self,
        symbol: str,
        current_timestamp: Any,
        cadence_minutes: int,
    ) -> bool:
        state = self._symbol_state(symbol)
        if state["last_timestamp"] is None:
            return True

        last = _parse_timestamp(state["last_timestamp"])
        current = _parse_timestamp(current_timestamp)
        threshold = timedelta(
            minutes=max(1, cadence_minutes * 2)
        )
        return current - last > threshold

    def _epsilon(self) -> float:
        return max(
            0.05,
            0.20 * float(np.exp(-self.agent.steps / 5_000.0)),
        )

    def _add_transition(
        self,
        state_vector: np.ndarray,
        action: int,
        reward: float,
        next_vector: np.ndarray,
        done: bool = False,
        updates: int = 1,
    ) -> float | None:
        self.buffer.add(
            self.agent_transition(
                state_vector,
                action,
                reward,
                next_vector,
                done,
            )
        )
        loss: float | None = None
        for _ in range(max(0, updates)):
            current_loss = self.agent.update(self.buffer)
            if current_loss is not None:
                loss = current_loss
        return loss

    @staticmethod
    def agent_transition(
        state_vector: np.ndarray,
        action: int,
        reward: float,
        next_vector: np.ndarray,
        done: bool,
    ):
        from src.rl.recurrent_dqn import Transition

        return Transition(
            state=state_vector.astype(np.float32),
            action=int(action),
            reward=float(reward),
            next_state=next_vector.astype(np.float32),
            done=bool(done),
        )

    def catch_up(
        self,
        symbol: str,
        intraday_history: pd.DataFrame,
        daily_history: pd.DataFrame,
        cadence_minutes: int,
    ) -> dict[str, int | str]:
        """Replay available intraday bars since the last saved action.

        On a brand-new project, only the most recent two calendar days are
        replayed. When resuming, the available history is replayed from the
        last saved timestamp. The free provider may not retain arbitrarily old
        intraday bars, so the available window is used.
        """
        state = self._symbol_state(symbol)
        bars = _prepare_intraday(
            intraday_history,
            cadence_minutes,
        )
        if len(bars) <= cadence_minutes:
            return {
                "transitions": 0,
                "status": "No intraday history available for catch-up.",
            }

        state_matrix = _state_matrix_from_bars(
            daily_history,
            bars,
        )
        if len(state_matrix) != len(bars):
            usable = min(len(state_matrix), len(bars))
            bars = bars.iloc[-usable:]
            state_matrix = state_matrix[-usable:]

        if len(bars) <= 1:
            return {
                "transitions": 0,
                "status": "Not enough intraday bars for catch-up.",
            }

        latest = bars.index[-1]
        last_timestamp = state.get("last_timestamp")
        pending = state.get("pending")

        if last_timestamp is None:
            start_time = latest - timedelta(days=FIRST_RUN_TRAIN_DAYS)
            start_index = int(
                np.searchsorted(
                    bars.index.to_numpy(),
                    np.datetime64(start_time.to_datetime64()),
                    side="left",
                )
            )
            state["position"] = 0
            pending = None
        else:
            start_index = int(
                np.searchsorted(
                    bars.index.to_numpy(),
                    _parse_timestamp(last_timestamp).to_datetime64(),
                    side="left",
                )
            )
            start_index = max(0, start_index)

            # If the previous pending decision is still resolvable, settle it
            # before continuing through missed bars.
            if pending is not None:
                pending_ts = _parse_timestamp(pending["timestamp"])
                future_start = pending_ts + timedelta(minutes=cadence_minutes)
                future_index = int(
                    bars.index.searchsorted(
                        _parse_timestamp(future_start),
                        side="left",
                    )
                )
                if pending_ts < bars.index[0]:
                    state["pending"] = None
                    state["position"] = 0
                    pending = None
                    start_index = 0
                elif future_index < len(bars):
                    next_state = _pad_sequence(
                        state_matrix[: future_index + 1],
                        self.sequence_length,
                    )
                    future_price = _safe_float(
                        bars.iloc[future_index]["Close"]
                    )
                    pending_price = _safe_float(pending["price"])
                    realized_return = (
                        future_price / max(pending_price, 1e-8) - 1.0
                    )
                    position = int(pending["position"])
                    changed = int(
                        position != int(pending["previous_position"])
                    )
                    reward = (
                        position * realized_return
                        - TRANSACTION_COST * changed
                    )
                    loss = self._add_transition(
                        np.asarray(pending["state"], dtype=np.float32),
                        int(pending["action"]),
                        reward,
                        next_state,
                        False,
                        updates=1,
                    )
                    state["position"] = position
                    state["pending"] = None
                    start_index = future_index
                    state["last_timestamp"] = _timestamp_key(
                        bars.index[future_index]
                    )
                    state["last_price"] = future_price
                    state["recent_prices"] = [
                        _safe_float(value)
                        for value in bars["Close"].iloc[
                            max(0, future_index - 59) : future_index + 1
                        ]
                    ]
                    state["recent_volumes"] = [
                        _safe_float(value)
                        for value in bars["Volume"].iloc[
                            max(0, future_index - 59) : future_index + 1
                        ]
                    ]
                elif pending is not None:
                    state["state_history"] = state_matrix[-self.sequence_length :].tolist()
                    self._save()
                    return {
                        "transitions": 0,
                        "status": "Waiting for enough new bars to settle the saved RL action.",
                    }

        # If the saved state is older than the provider's available window,
        # restart the replay from the earliest available bar.
        if start_index >= len(bars) - 1:
            state["state_history"] = state_matrix[-self.sequence_length :].tolist()
            state["last_timestamp"] = _timestamp_key(latest)
            self._save()
            return {
                "transitions": 0,
                "status": "No new historical intervals to replay.",
            }

        # Build transitions from each available decision bar. The latest bar is
        # kept for the next live decision because it has no future answer yet.
        end_index = len(bars) - 2
        first_index = max(
            start_index,
            self.sequence_length - 1,
        )

        transitions = []
        for index in range(first_index, end_index + 1):
            if (
                last_timestamp is not None
                and _parse_timestamp(bars.index[index])
                <= _parse_timestamp(last_timestamp)
            ):
                continue

            current_state = _pad_sequence(
                state_matrix[: index + 1],
                self.sequence_length,
            )
            next_state = _pad_sequence(
                state_matrix[: index + 2],
                self.sequence_length,
            )
            current_price = _safe_float(bars.iloc[index]["Close"])
            next_price = _safe_float(bars.iloc[index + 1]["Close"])
            realized_return = (
                next_price / max(current_price, 1e-8) - 1.0
            )

            action = self.agent.select_action(
                current_state,
                epsilon=self._epsilon(),
            )
            previous_position = int(state["position"])
            position = _action_position(
                action,
                previous_position,
            )
            changed = int(position != previous_position)
            reward = (
                position * realized_return
                - TRANSACTION_COST * changed
            )

            transitions.append(
                (
                    current_state,
                    action,
                    reward,
                    next_state,
                    position,
                    bars.index[index],
                    next_price,
                )
            )
            state["position"] = position

        if transitions:
            for current_state, action, reward, next_state, position, timestamp, price in transitions:
                self.buffer.add(
                    self.agent_transition(
                        current_state,
                        action,
                        reward,
                        next_state,
                        False,
                    )
                )

            updates = min(
                100,
                max(1, len(transitions) // 8),
            )
            last_loss = None
            for _ in range(updates):
                last_loss = self.agent.update(self.buffer)
            state["trained_steps"] = int(self.agent.steps)
            state["catchup_transitions"] = (
                int(state.get("catchup_transitions", 0))
                + len(transitions)
            )
        else:
            last_loss = None

        state["state_history"] = state_matrix[-self.sequence_length :].tolist()
        state["last_timestamp"] = _timestamp_key(latest)
        state["last_price"] = _safe_float(bars.iloc[-1]["Close"])
        latest_session = pd.Series(
            bars["Volume"].astype(float).to_numpy(),
            index=bars.index,
        )
        latest_session_volume = float(
            latest_session[
                latest_session.index.date == latest_session.index[-1].date()
            ].sum()
        )
        state["last_session_volume"] = latest_session_volume
        state["recent_prices"] = [
            _safe_float(value)
            for value in bars["Close"].tail(self.sequence_length)
        ]
        state["recent_volumes"] = [
            _safe_float(value)
            for value in bars["Volume"].tail(self.sequence_length)
        ]
        state["pending"] = None
        self._save()

        return {
            "transitions": len(transitions),
            "status": (
                f"Replayed {len(transitions)} intraday intervals."
                + (
                    f" Last loss: {last_loss:.6f}."
                    if last_loss is not None
                    else ""
                )
            ),
        }

    def _live_state_vector(
        self,
        state: dict[str, Any],
        daily_row: pd.Series,
        quote: dict[str, Any],
    ) -> np.ndarray:
        daily_values = _daily_vector(daily_row)
        current_price = _safe_float(quote.get("price"))
        previous_close = max(
            _safe_float(quote.get("previous_close")),
            1e-8,
        )

        recent_prices = [
            _safe_float(value)
            for value in state.get("recent_prices", [])
        ]
        recent_volumes = [
            _safe_float(value)
            for value in state.get("recent_volumes", [])
        ]
        last_price = _safe_float(
            state.get("last_price"),
            current_price,
        )

        bar_return_1 = (
            current_price / max(last_price, 1e-8) - 1.0
        )
        base_price_3 = (
            recent_prices[-3]
            if len(recent_prices) >= 3
            else last_price
        )
        bar_return_3 = (
            current_price / max(base_price_3, 1e-8) - 1.0
        )

        session_volume = _safe_float(
            quote.get("session_volume")
        )
        previous_session_volume = _safe_float(
            state.get("last_session_volume")
        )
        bar_volume = (
            session_volume
            if previous_session_volume <= 0.0
            else max(
                session_volume - previous_session_volume,
                0.0,
            )
        )
        volume_reference = (
            float(np.median(recent_volumes[-20:]))
            if recent_volumes
            else 0.0
        )
        bar_volume_ratio = (
            bar_volume / volume_reference
            if volume_reference > 0.0
            else 0.0
        )

        day_high = _safe_float(quote.get("day_high"), current_price)
        day_low = _safe_float(quote.get("day_low"), current_price)
        session_position = (
            (current_price - day_low)
            / max(day_high - day_low, 1e-8)
        )

        session_return = current_price / previous_close - 1.0

        return np.concatenate(
            [
                daily_values,
                np.asarray(
                    [
                        bar_return_1,
                        bar_return_3,
                        bar_volume_ratio,
                        session_position,
                        session_return,
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(np.float32)

    def observe_live(
        self,
        symbol: str,
        daily_row: pd.Series,
        quote: dict[str, Any],
        cadence_minutes: int,
    ) -> OnlineResult:
        state = self._symbol_state(symbol)
        timestamp = _parse_timestamp(quote.get("timestamp"))
        price = _safe_float(quote.get("price"))

        if (
            state.get("pending") is not None
            and state.get("last_timestamp") is not None
            and timestamp <= _parse_timestamp(state["last_timestamp"])
        ):
            action = int(
                state["pending"]["action"]
                if state.get("pending")
                else RecurrentDQNAgent.__dict__.get("ACTION_HOLD", 1)
            )
            return OnlineResult(
                action=action,
                action_name=ACTION_NAMES.get(action, "HOLD"),
                position=int(state["position"]),
                reward=None,
                loss=None,
                trained=False,
                catchup_transitions=0,
                replay_size=len(self.buffer),
                steps=int(self.agent.steps),
                timestamp=_timestamp_key(timestamp),
                status="Waiting for the next market interval.",
            )

        reward = None
        loss = None
        trained = False

        pending = state.get("pending")
        if pending is not None:
            pending_timestamp = _parse_timestamp(pending["timestamp"])
            elapsed = timestamp - pending_timestamp
            if elapsed >= timedelta(minutes=cadence_minutes):
                pending_price = _safe_float(pending["price"])
                realized_return = (
                    price / max(pending_price, 1e-8) - 1.0
                )
                position = int(pending["position"])
                changed = int(
                    position != int(pending["previous_position"])
                )
                reward = (
                    position * realized_return
                    - TRANSACTION_COST * changed
                )

                sequence_history = [
                    np.asarray(item, dtype=np.float32)
                    for item in state.get("state_history", [])
                ]
                next_vector = self._live_state_vector(
                    state,
                    daily_row,
                    quote,
                )
                next_history = sequence_history + [next_vector]
                next_state = _pad_sequence(
                    np.asarray(next_history, dtype=np.float32),
                    self.sequence_length,
                )
                loss = self._add_transition(
                    np.asarray(pending["state"], dtype=np.float32),
                    int(pending["action"]),
                    reward,
                    next_state,
                    False,
                    updates=1,
                )
                trained = loss is not None
                state["position"] = position
                state["pending"] = None

        current_vector = self._live_state_vector(
            state,
            daily_row,
            quote,
        )
        history = [
            np.asarray(item, dtype=np.float32)
            for item in state.get("state_history", [])
        ]
        history.append(current_vector)
        history = history[-self.sequence_length :]
        state_sequence = _pad_sequence(
            np.asarray(history, dtype=np.float32),
            self.sequence_length,
        )

        action = self.agent.select_action(
            state_sequence,
            epsilon=self._epsilon(),
        )
        previous_position = int(state["position"])
        position = _action_position(
            action,
            previous_position,
        )

        state["pending"] = {
            "timestamp": _timestamp_key(timestamp),
            "price": price,
            "action": int(action),
            "previous_position": previous_position,
            "position": position,
            "state": state_sequence.tolist(),
        }
        state["position"] = position
        state["last_timestamp"] = _timestamp_key(timestamp)
        state["last_price"] = price
        state["last_session_volume"] = _safe_float(
            quote.get("session_volume")
        )
        state["state_history"] = [
            item.tolist() if isinstance(item, np.ndarray) else item
            for item in history
        ]

        self._save()

        return OnlineResult(
            action=int(action),
            action_name=ACTION_NAMES.get(action, "HOLD"),
            position=position,
            reward=reward,
            loss=loss,
            trained=trained,
            catchup_transitions=0,
            replay_size=len(self.buffer),
            steps=int(self.agent.steps),
            timestamp=_timestamp_key(timestamp),
            status=(
                "RL updated from the previous interval."
                if trained
                else "RL action recorded; waiting for the next interval."
            ),
        )
