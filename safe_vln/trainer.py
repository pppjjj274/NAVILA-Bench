"""Tensor-level constrained PPO update used by the Safe-VLN learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


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
    action_logits: torch.Tensor | None = None,
    oracle_action_ids: torch.Tensor | None = None,
    oracle_mask: torch.Tensor | None = None,
    oracle_sample_weights: torch.Tensor | None = None,
    oracle_ce_coef: float = 0.0,
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
    oracle_loss = torch.zeros((), device=total.device, dtype=total.dtype)
    oracle_stop_loss = torch.zeros((), device=total.device, dtype=total.dtype)
    oracle_stop_accuracy = 0.0
    oracle_samples = 0
    oracle_stop_samples = 0
    if oracle_ce_coef:
        if action_logits is None or oracle_action_ids is None or oracle_mask is None:
            raise ValueError("oracle CE requires logits, action IDs, and a mask")
        mask = oracle_mask.to(dtype=torch.bool, device=action_logits.device)
        if mask.any():
            targets = oracle_action_ids.to(action_logits.device)[mask]
            per_sample_loss = F.cross_entropy(
                action_logits[mask],
                targets,
                reduction="none",
            )
            if oracle_sample_weights is None:
                weights = torch.ones_like(per_sample_loss)
            else:
                weights = oracle_sample_weights.to(
                    device=action_logits.device,
                    dtype=per_sample_loss.dtype,
                )[mask]
                if (
                    not torch.isfinite(weights).all()
                    or (weights <= 0).any()
                ):
                    raise ValueError("oracle sample weights must be finite and positive")
            # Divide by sample count, not by the sum of weights. Normalizing by
            # weights made a STOP weight of 5.0 cancel completely for the
            # batch-size-one training configuration used on a single A800.
            oracle_loss = (per_sample_loss * weights).mean()
            stop_mask = targets == 9
            oracle_samples = int(mask.sum().item())
            oracle_stop_samples = int(stop_mask.sum().item())
            if stop_mask.any():
                oracle_stop_loss = per_sample_loss[stop_mask].mean()
                predictions = action_logits[mask].argmax(dim=-1)
                oracle_stop_accuracy = float(
                    (predictions[stop_mask] == 9).float().mean().detach().item()
                )
            total = total + float(oracle_ce_coef) * oracle_loss
    stats = {
        "loss/total": float(total.detach().item()),
        "loss/policy": float(policy_loss.detach().item()),
        "loss/reward_value": float(reward_value_loss.detach().item()),
        "loss/cost_value": float(cost_value_loss.detach().item()),
        "policy/entropy": float(entropy.detach().mean().item()),
        "policy/ratio": float(ratio.detach().mean().item()),
        "policy/safe_advantage": float(safe_advantages.detach().mean().item()),
        "loss/oracle_ce": float(oracle_loss.detach().item()),
        "loss/oracle_stop_ce": float(oracle_stop_loss.detach().item()),
        "oracle/samples": oracle_samples,
        "oracle/stop_samples": oracle_stop_samples,
        "oracle/stop_accuracy": oracle_stop_accuracy,
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
