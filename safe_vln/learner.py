"""Critic warm-start, rollout evaluation, and checkpoint support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import torch

from .actions import ACTIONS
from .trainer import SafePPOOptimizer


def train_critic_epoch(model, samples, preprocessor, optimizer, *, max_samples: int | None = None):
    model.train()
    total_loss = 0.0
    count = 0
    for frames, metadata in samples:
        if max_samples is not None and count >= max_samples:
            break
        if "reward_return" not in metadata or "cost_return" not in metadata:
            continue
        state = preprocessor(frames, metadata["instruction"])
        output = model(state.input_ids, images=state.images)
        reward_target = torch.tensor([metadata["reward_return"]], device=output.reward_values.device)
        cost_target = torch.tensor([metadata["cost_return"]], device=output.cost_values.device)
        reward_loss = 0.5 * (output.reward_values - reward_target).square().mean()
        cost_loss = 0.5 * (output.cost_values - cost_target).square().mean()
        loss = reward_loss + cost_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("critic warm-start loss is not finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.reward_head.parameters()) + list(model.cost_head.parameters()), 0.5
        )
        optimizer.step()
        total_loss += float(loss.detach().item())
        count += 1
    return {"critic/loss": total_loss / max(count, 1), "critic/samples": count}


def evaluate_selected_actions(model, prepared_states, action_ids: torch.Tensor):
    logits = []
    reward_values = []
    cost_values = []
    for state in prepared_states:
        output = model(state.input_ids, images=state.images)
        logits.append(output.action_logits)
        reward_values.append(output.reward_values)
        cost_values.append(output.cost_values)
    logits = torch.cat(logits)
    distribution = torch.distributions.Categorical(logits=logits)
    return {
        "new_log_probs": distribution.log_prob(action_ids.to(logits.device)),
        "reward_values": torch.cat(reward_values),
        "cost_values": torch.cat(cost_values),
        "entropy": distribution.entropy(),
    }


def save_checkpoint(model, optimizer, output_dir, trainer_state: Mapping):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model.base_model, "save_pretrained"):
        model.base_model.save_pretrained(output_dir)
    model.save_safe_heads(output_dir)
    torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    (output_dir / "action_space.json").write_text(
        json.dumps([action.to_dict() for action in ACTIONS], indent=2), encoding="utf-8"
    )
    (output_dir / "trainer_state.json").write_text(json.dumps(dict(trainer_state), indent=2), encoding="utf-8")
