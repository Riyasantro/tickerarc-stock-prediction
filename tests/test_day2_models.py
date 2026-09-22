import numpy as np
import pandas as pd
import torch

from src.models.lstm_encoder import LSTMActionQNetwork, MultiTaskLSTM
from src.rl.trading_env import TradingEnv


def test_lstm_model_shapes():
    model = MultiTaskLSTM(input_size=8, hidden_size=16, num_layers=2)
    x = torch.randn(4, 60, 8)
    out = model(x)

    assert out["return"].shape == (4, 3)
    assert out["direction_logits"].shape == (4, 3)
    assert out["volatility"].shape == (4,)


def test_lstm_action_q_shapes():
    model = LSTMActionQNetwork(input_size=8, hidden_size=16, num_layers=2)
    x = torch.randn(2, 60, 8)
    q_values = model(x)
    assert q_values.shape == (2, 3)


def test_trading_env_step_is_causal():
    rows = 100
    rng = np.random.default_rng(1)
    data = pd.DataFrame(
        {
            "return_1d": rng.normal(0, 0.01, rows),
            "f1": rng.normal(size=rows),
            "f2": rng.normal(size=rows),
        }
    )
    env = TradingEnv(data, ["f1", "f2"], sequence_length=20)
    state = env.reset()
    result = env.step(TradingEnv.ACTION_BUY)

    assert state.shape == (20, 2)
    assert result.observation.shape == (20, 2)
    assert np.isfinite(result.reward)
