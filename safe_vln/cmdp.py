"""Framework-independent CMDP math for Safe-VLN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def compute_returns(
    values: Sequence[float], dones: Sequence[bool], gamma: float = 0.99
) -> list[float]:
    if len(values) != len(dones):
        raise ValueError("values and dones must have the same length")
    returns = [0.0] * len(values)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        running = float(values[index]) + gamma * (0.0 if dones[index] else running)
        returns[index] = running
    return returns


def compute_gae(
    signals: Sequence[float],
    value_predictions: Sequence[float],
    dones: Sequence[bool],
    *,
    next_value: float = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(advantages, returns)`` for reward or cost signals."""
    if not (len(signals) == len(value_predictions) == len(dones)):
        raise ValueError("signals, value_predictions and dones must have equal lengths")
    advantages = np.zeros(len(signals), dtype=np.float32)
    gae = 0.0
    following_value = float(next_value)
    for index in range(len(signals) - 1, -1, -1):
        nonterminal = 0.0 if dones[index] else 1.0
        delta = float(signals[index]) + gamma * following_value * nonterminal - float(value_predictions[index])
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
        following_value = float(value_predictions[index])
    returns = advantages + np.asarray(value_predictions, dtype=np.float32)
    return advantages, returns


def safe_advantage(reward_advantage, cost_advantage, multiplier: float):
    if multiplier < 0:
        raise ValueError("Lagrange multiplier must be non-negative")
    return (np.asarray(reward_advantage) - multiplier * np.asarray(cost_advantage)) / (1.0 + multiplier)


@dataclass
class LagrangeController:
    cost_limit: float = 0.1
    multiplier: float = 0.001
    learning_rate: float = 0.035
    max_multiplier: float = 100.0

    def update(self, mean_episode_cost: float) -> float:
        self.multiplier += self.learning_rate * (float(mean_episode_cost) - self.cost_limit)
        self.multiplier = min(self.max_multiplier, max(0.0, self.multiplier))
        return self.multiplier
