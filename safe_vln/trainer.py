"""Tensor-level constrained PPO update used by the Safe-VLN learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def normalize_advantage(value: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    if value.numel() <= 1:
        return value
    return (value - value.mean()) / (value.std(unbiased=False) + epsilon)


def constrained_ppo_loss(
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reward_advantages: torch.Tensor,
    cost_advantages: torch.Tensor,
    reward_values: torch.Tensor,
    cost_values: torch.Tensor,
    reward_returns: torch.Tensor,
    cost_returns: torch.Tensor,
    entropy: torch.Tensor,
    lagrange_multiplier: float,
    clip_ratio: float = 0.1,
    reward_value_coef: float = 0.5,
    cost_value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    normalize_advantages: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if normalize_advantages:
        reward_advantages = normalize_advantage(reward_advantages)
        cost_advantages = normalize_advantage(cost_advantages)
    safe_advantages = (reward_advantages - lagrange_multiplier * cost_advantages) / (1.0 + lagrange_multiplier)
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    policy_loss = -torch.minimum(ratio * safe_advantages, clipped_ratio * safe_advantages).mean()
    reward_value_loss = 0.5 * (reward_values - reward_returns).square().mean()
    cost_value_loss = 0.5 * (cost_values - cost_returns).square().mean()
    entropy_loss = -entropy.mean()
    total = (
        policy_loss
        + reward_value_coef * reward_value_loss
        + cost_value_coef * cost_value_loss
        + entropy_coef * entropy_loss
    )
    stats = {
        "loss/total": float(total.detach().item()),
        "loss/policy": float(policy_loss.detach().item()),
        "loss/reward_value": float(reward_value_loss.detach().item()),
        "loss/cost_value": float(cost_value_loss.detach().item()),
        "policy/entropy": float(entropy.detach().mean().item()),
        "policy/ratio": float(ratio.detach().mean().item()),
        "policy/safe_advantage": float(safe_advantages.detach().mean().item()),
    }
    return total, stats


@dataclass
class PPOConfig:
    clip_ratio: float = 0.1
    ppo_epochs: int = 4
    mini_batch_size: int = 16
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True


class SafePPOOptimizer:
    """Optimize already-evaluated on-policy tensors.

    The rollout/model service is responsible for re-evaluating the selected
    actions with the current model and assembling the tensors in ``batch``.
    Keeping this layer model-agnostic makes its CMDP math unit-testable without
    Isaac Sim or an 8B checkpoint.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, config: PPOConfig | None = None) -> None:
        self.optimizer = optimizer
        self.config = config or PPOConfig()

    def step(self, batch: dict[str, torch.Tensor], lagrange_multiplier: float):
        loss, stats = constrained_ppo_loss(
            **batch,
            lagrange_multiplier=lagrange_multiplier,
            clip_ratio=self.config.clip_ratio,
            normalize_advantages=self.config.normalize_advantages,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Safe-VLN PPO loss is not finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.config.max_grad_norm
        )
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "Safe-VLN PPO gradient norm is not finite; "
                "use BF16 training or lower the learning rate"
            )
        stats["optimizer/grad_norm"] = float(grad_norm.detach().item())
        self.optimizer.step()
        if any(
            not torch.isfinite(parameter).all()
            for parameter in parameters
            if parameter.requires_grad
        ):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "Safe-VLN PPO produced non-finite trainable parameters"
            )
        self.optimizer.zero_grad(set_to_none=True)
        return stats
