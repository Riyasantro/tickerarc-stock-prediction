"""Multi-horizon LSTM for TickerArc stock-return prediction."""

from __future__ import annotations

import torch
from torch import nn


class MultiHorizonLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        layers: int = 2,
        dropout: float = 0.20,
        horizons: int = 3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.return_head = nn.Linear(128, horizons)
        self.direction_head = nn.Linear(128, 3)
        self.volatility_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        seq, _ = self.lstm(x)
        latent = self.dropout(self.norm(seq[:, -1, :]))
        shared = self.shared(latent)
        return {
            "returns": self.return_head(shared),
            "direction_logits": self.direction_head(shared),
            "volatility": self.volatility_head(shared).squeeze(-1),
        }
