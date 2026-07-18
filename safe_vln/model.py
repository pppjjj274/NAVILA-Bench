"""NaViLA adapter with a canonical discrete actor and two value heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .actions import ACTIONS


@dataclass
class SafeActorCriticOutput:
    action_logits: torch.Tensor
    reward_values: torch.Tensor
    cost_values: torch.Tensor

    @property
    def distribution(self):
        return torch.distributions.Categorical(logits=self.action_logits)


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int = 512) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        parameters = next(self.parameters())
        hidden_state = hidden_state.to(device=parameters.device, dtype=parameters.dtype)
        return self.network(hidden_state).squeeze(-1).float()


class SafeNavilaActorCritic(nn.Module):
    """Score canonical response sequences and predict reward/cost values.

    ``prompt_input_ids`` must already contain NaViLA image placeholders and
    conversation formatting. Candidate responses are teacher-forced through
    the original causal LM. NaViLA expands every image placeholder into many
    visual embeddings, so candidate logits are deliberately indexed relative
    to the end of the expanded sequence rather than the text prompt length.
    """

    def __init__(self, base_model: nn.Module, tokenizer: Any, hidden_size: int | None = None) -> None:
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        if hidden_size is None:
            config = base_model.config
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is None and hasattr(base_model, "get_llm"):
                hidden_size = base_model.get_llm().config.hidden_size
            if hidden_size is None:
                raise ValueError("could not infer NaViLA hidden size")
        self.reward_head = ValueHead(int(hidden_size))
        self.cost_head = ValueHead(int(hidden_size))
        self._candidate_ids = [
            tokenizer(action.text, add_special_tokens=False, return_tensors="pt").input_ids[0]
            for action in ACTIONS
        ]
        if any(candidate.numel() == 0 for candidate in self._candidate_ids):
            raise ValueError("every canonical Safe-VLN action must tokenize to at least one token")

    def _forward_base(self, input_ids, attention_mask, images=None, **model_kwargs):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            output_hidden_states=True,
            return_dict=True,
            **model_kwargs,
        )

    def forward(self, prompt_input_ids: torch.Tensor, images=None, **model_kwargs) -> SafeActorCriticOutput:
        if prompt_input_ids.ndim != 2 or prompt_input_ids.shape[0] != 1:
            raise ValueError("candidate scoring currently expects one prompt per forward call")
        prompt_input_ids = prompt_input_ids.to(next(self.base_model.parameters()).device)
        candidate_scores = []
        state_hidden = None
        for candidate in self._candidate_ids:
            candidate = candidate.to(prompt_input_ids.device)
            full_ids = torch.cat((prompt_input_ids, candidate.unsqueeze(0)), dim=1)
            output = self._forward_base(
                full_ids,
                torch.ones_like(full_ids),
                images=images,
                **model_kwargs,
            )
            candidate_length = candidate.numel()
            # Causal logit at position t predicts token t+1. The visual token
            # expansion only changes the prefix length, while candidate tokens
            # remain the final ``candidate_length`` positions.
            token_logits = output.logits[:, -(candidate_length + 1) : -1]
            token_log_probs = token_logits.log_softmax(dim=-1)
            selected = token_log_probs.gather(-1, candidate.view(1, -1, 1)).squeeze(-1)
            candidate_scores.append(selected.mean(dim=-1))
            if state_hidden is None:
                state_hidden = output.hidden_states[-1][:, -(candidate_length + 1)]
        assert state_hidden is not None
        return SafeActorCriticOutput(
            action_logits=torch.stack(candidate_scores, dim=-1).float(),
            reward_values=self.reward_head(state_hidden),
            cost_values=self.cost_head(state_hidden),
        )

    def act(self, prompt_input_ids, images=None, deterministic: bool = False, **model_kwargs):
        output = self.forward(prompt_input_ids, images=images, **model_kwargs)
        distribution = output.distribution
        action = output.action_logits.argmax(dim=-1) if deterministic else distribution.sample()
        return {
            "action_id": int(action.item()),
            "log_prob": float(distribution.log_prob(action).item()),
            "reward_value": float(output.reward_values.item()),
            "cost_value": float(output.cost_values.item()),
            "action_probabilities": distribution.probs[0].detach().cpu().tolist(),
        }

    def save_safe_heads(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.reward_head.state_dict(), output_dir / "reward_critic.pt")
        torch.save(self.cost_head.state_dict(), output_dir / "cost_critic.pt")

    def load_safe_heads(self, checkpoint_dir: str | Path, map_location="cpu") -> None:
        checkpoint_dir = Path(checkpoint_dir)
        reward_state = torch.load(checkpoint_dir / "reward_critic.pt", map_location=map_location)
        cost_state = torch.load(checkpoint_dir / "cost_critic.pt", map_location=map_location)
        self.reward_head.load_state_dict(reward_state)
        self.cost_head.load_state_dict(cost_state)


def add_lora_adapters(model: nn.Module, *, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Attach PEFT LoRA adapters to common Llama attention/MLP projections."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("PEFT is required for Safe-VLN LoRA training") from exc
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, config)
