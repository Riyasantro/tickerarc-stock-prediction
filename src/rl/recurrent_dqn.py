"""LSTM-DQN implementation for Sell/Hold/Buy policy learning."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from src.models.lstm_encoder import LSTMActionQNetwork
from src.rl.trading_env import TradingEnv


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class RecurrentDQNAgent:
    """DQN agent whose Q-function uses an LSTM sequence encoder."""

    def __init__(
        self,
        input_size: int,
        device: str = "auto",
        hidden_size: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        target_update: int = 250,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy = LSTMActionQNetwork(input_size, hidden_size).to(self.device)
        self.target = LSTMActionQNetwork(input_size, hidden_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.steps = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            return random.randrange(3)

        tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy(tensor)
        return int(q_values.argmax(dim=1).item())

    def update(self, buffer: ReplayBuffer) -> float | None:
        if len(buffer) < self.batch_size:
            return None

        batch = buffer.sample(self.batch_size)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in batch],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in batch],
            dtype=torch.float32,
            device=self.device,
        )

        current_q = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy(next_states).argmax(dim=1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return float(loss.item())


def train_episode(
    env: TradingEnv,
    agent: RecurrentDQNAgent,
    buffer: ReplayBuffer,
    epsilon: float,
) -> dict[str, float]:
    state = env.reset()
    total_reward = 0.0
    losses: list[float] = []
    done = False

    while not done:
        action = agent.select_action(state, epsilon=epsilon)
        result = env.step(action)
        buffer.add(
            Transition(
                state=state,
                action=action,
                reward=result.reward,
                next_state=result.observation,
                done=result.done,
            )
        )
        loss = agent.update(buffer)
        if loss is not None:
            losses.append(loss)
        total_reward += result.reward
        state = result.observation
        done = result.done

    return {
        "reward": total_reward,
        "loss": float(np.mean(losses)) if losses else 0.0,
    }
