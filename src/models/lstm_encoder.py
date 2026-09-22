"""LSTM prediction backbone used by TickerArc and the RL policy."""

from __future__ import annotations

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    """Sequence encoder for OHLCV-derived feature windows."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.output_size = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.norm = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        last = sequence[:, -1, :]
        return self.dropout(self.norm(last))


class MultiTaskLSTM(nn.Module):
    """LSTM with return, direction, and volatility prediction heads."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizons: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        latent = self.encoder.output_size
        self.return_head = nn.Sequential(
            nn.Linear(latent, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, forecast_horizons),
        )
        self.direction_head = nn.Linear(latent, 3)
        self.volatility_head = nn.Sequential(
            nn.Linear(latent, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(x)
        return {
            "return": self.return_head(latent),
            "direction_logits": self.direction_head(latent),
            "volatility": self.volatility_head(latent).squeeze(-1),
            "latent": latent,
        }


class LSTMActionQNetwork(nn.Module):
    """LSTM encoder followed by Q-values for Sell/Hold/Buy."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        actions: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        latent = self.encoder.output_size
        self.q_head = nn.Sequential(
            nn.Linear(latent, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.encoder(x))
