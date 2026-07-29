"""NaViLA adapter with a canonical discrete actor and two value heads."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from .actions import ACTIONS, NAVILA_ACTION_RESPONSES


@dataclass
class SafeActorCriticOutput:
    action_logits: torch.Tensor
    reward_values: torch.Tensor
    cost_values: torch.Tensor
    stop_logits: torch.Tensor | None = None
    motion_logits: torch.Tensor | None = None

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


ACTOR_ARCHITECTURE_CANDIDATE = "candidate-scoring"
ACTOR_ARCHITECTURE_HIERARCHICAL = "hierarchical-stop-motion"


class HierarchicalActorHead(nn.Module):
    """Factor ten actions into STOP probability and conditional motion."""

    def __init__(self, hidden_size: int, intermediate_size: int = 512) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
        )
        self.stop_head = nn.Linear(intermediate_size, 1)
        self.motion_head = nn.Linear(intermediate_size, 9)

    def forward(self, hidden_state: torch.Tensor):
        parameters = next(self.parameters())
        hidden_state = hidden_state.to(
            device=parameters.device, dtype=parameters.dtype
        )
        feature = self.projection(hidden_state)
        return self.stop_head(feature).squeeze(-1), self.motion_head(feature)

    @staticmethod
    def joint_log_probs(stop_logits, motion_logits):
        log_stop = -torch.nn.functional.softplus(-stop_logits)
        log_continue = -torch.nn.functional.softplus(stop_logits)
        motion = torch.log_softmax(motion_logits.float(), dim=-1)
        return torch.cat(
            (log_continue.float().unsqueeze(-1) + motion,
             log_stop.float().unsqueeze(-1)),
            dim=-1,
        )


class SafeNavilaActorCritic(nn.Module):
    """Score canonical response sequences and predict reward/cost values.

    ``prompt_input_ids`` must already contain NaViLA image placeholders and
    conversation formatting. Candidate responses are teacher-forced through
    the original causal LM. NaViLA expands every image placeholder into many
    visual embeddings, so candidate logits are deliberately indexed relative
    to the end of the expanded sequence rather than the text prompt length.
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        hidden_size: int | None = None,
        *,
        actor_architecture: str = ACTOR_ARCHITECTURE_CANDIDATE,
        stop_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if actor_architecture not in {
            ACTOR_ARCHITECTURE_CANDIDATE,
            ACTOR_ARCHITECTURE_HIERARCHICAL,
        }:
            raise ValueError(f"unsupported actor architecture: {actor_architecture}")
        if not 0.0 < float(stop_threshold) < 1.0:
            raise ValueError("stop threshold must be in (0, 1)")
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.actor_architecture = actor_architecture
        self.stop_threshold = float(stop_threshold)
        if hidden_size is None:
            config = base_model.config
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is None and hasattr(base_model, "get_llm"):
                hidden_size = base_model.get_llm().config.hidden_size
            if hidden_size is None:
                raise ValueError("could not infer NaViLA hidden size")
        self.reward_head = ValueHead(int(hidden_size))
        self.cost_head = ValueHead(int(hidden_size))
        self.actor_head = (
            HierarchicalActorHead(int(hidden_size))
            if actor_architecture == ACTOR_ARCHITECTURE_HIERARCHICAL
            else None
        )
        self._candidate_ids = [
            tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
            for text in NAVILA_ACTION_RESPONSES
        ]
        if any(candidate.numel() == 0 for candidate in self._candidate_ids):
            raise ValueError("every canonical Safe-VLN action must tokenize to at least one token")

    def _navila_model(self):
        model = self.base_model
        if hasattr(model, "get_base_model"):
            model = model.get_base_model()
        if (
            hasattr(model, "prepare_inputs_labels_for_multimodal")
            and hasattr(model, "llm")
        ):
            return model
        return None

    def _encode_navila_images_once(self, images):
        navila_model = self._navila_model()
        if navila_model is None or images is None:
            return None
        flattened = images
        if type(flattened) is list:
            flattened = torch.cat(flattened, dim=0)
        elif flattened.ndim == 5:
            flattened = flattened.flatten(0, 1)
        # The Safe-VLN LoRA adapters live in the language model.  The vision
        # tower and projector remain frozen, so their features can be shared
        # by all ten candidate actions without retaining an autograd graph.
        if hasattr(navila_model, "freezed_module_patch"):
            navila_model.freezed_module_patch()
        with torch.no_grad():
            return navila_model.encode_images(flattened).to(
                next(self.base_model.parameters()).device
            ).detach()

    def _forward_base(
        self,
        input_ids,
        attention_mask,
        images=None,
        image_features=None,
        output_suffix_length=None,
        project_logits=True,
        **model_kwargs,
    ):
        # NaViLA's outer LlavaLlamaModel.forward is coupled to its original
        # distributed language-model training loop: it expects time-token
        # config fields and always rescales a supervised LM loss.  Safe-VLN
        # needs logits/hidden states without LM labels, so prepare the same
        # multimodal embeddings and call the inner Llama directly.  PEFT
        # injects LoRA layers into this inner module, hence actor gradients are
        # retained during constrained PPO.
        navila_model = self._navila_model()
        if navila_model is not None:
            if hasattr(navila_model, "freezed_module_patch"):
                navila_model.freezed_module_patch()
            original_encode_images = None
            if image_features is not None:
                original_encode_images = navila_model.encode_images
                navila_model.encode_images = lambda unused_images: image_features
            try:
                (
                    prepared_input_ids,
                    position_ids,
                    prepared_attention_mask,
                    past_key_values,
                    inputs_embeds,
                    _,
                ) = navila_model.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    None,
                    attention_mask,
                    None,
                    None,
                    images,
                )
            finally:
                if original_encode_images is not None:
                    navila_model.encode_images = original_encode_images
            model_kwargs.setdefault("use_cache", False)
            if (
                output_suffix_length is not None
                and hasattr(navila_model.llm, "model")
                and hasattr(navila_model.llm, "lm_head")
            ):
                # LlamaForCausalLM normally materializes vocabulary logits for
                # every visual/text position and, with output_hidden_states,
                # retains every decoder layer.  Candidate scoring only needs
                # the final candidate tokens, so keep the backbone's final
                # state and project that small suffix through lm_head.
                backbone_output = navila_model.llm.model(
                    input_ids=prepared_input_ids,
                    attention_mask=prepared_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=False,
                    return_dict=True,
                    **model_kwargs,
                )
                suffix_hidden = backbone_output.last_hidden_state[
                    :, -output_suffix_length:
                ]
                suffix_logits = (
                    navila_model.llm.lm_head(suffix_hidden).float()
                    if project_logits
                    else None
                )
                return SimpleNamespace(
                    logits=suffix_logits,
                    hidden_states=(suffix_hidden,),
                )
            return navila_model.llm.forward(
                input_ids=prepared_input_ids,
                attention_mask=prepared_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict=True,
                **model_kwargs,
            )
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            output_hidden_states=True,
            return_dict=True,
            **model_kwargs,
        )

    def _forward_candidate(self, prompt_input_ids, images=None, **model_kwargs):
        if prompt_input_ids.ndim != 2 or prompt_input_ids.shape[0] != 1:
            raise ValueError("candidate scoring currently expects one prompt per forward call")
        prompt_input_ids = prompt_input_ids.to(next(self.base_model.parameters()).device)
        candidate_scores = []
        state_hidden = None
        image_features = self._encode_navila_images_once(images)
        for candidate in self._candidate_ids:
            candidate = candidate.to(prompt_input_ids.device)
            full_ids = torch.cat((prompt_input_ids, candidate.unsqueeze(0)), dim=1)
            output = self._forward_base(
                full_ids,
                torch.ones_like(full_ids),
                images=images,
                image_features=image_features,
                output_suffix_length=candidate.numel() + 1,
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

    def encode_state(self, prompt_input_ids, images=None, **model_kwargs):
        if prompt_input_ids.ndim != 2 or prompt_input_ids.shape[0] != 1:
            raise ValueError("Safe-VLN expects one prompt per forward call")
        prompt_input_ids = prompt_input_ids.to(
            next(self.base_model.parameters()).device
        )
        output = self._forward_base(
            prompt_input_ids,
            torch.ones_like(prompt_input_ids),
            images=images,
            output_suffix_length=1,
            project_logits=False,
            **model_kwargs,
        )
        return output.hidden_states[-1][:, -1]

    def _forward_hierarchical(self, prompt_input_ids, images=None, **model_kwargs):
        if self.actor_head is None:
            raise RuntimeError("hierarchical actor head is not initialized")
        state_hidden = self.encode_state(
            prompt_input_ids, images=images, **model_kwargs
        )
        stop_logits, motion_logits = self.actor_head(state_hidden)
        return SafeActorCriticOutput(
            action_logits=self.actor_head.joint_log_probs(
                stop_logits, motion_logits
            ),
            reward_values=self.reward_head(state_hidden),
            cost_values=self.cost_head(state_hidden),
            stop_logits=stop_logits,
            motion_logits=motion_logits,
        )

    def forward(self, prompt_input_ids, images=None, **model_kwargs):
        if self.actor_architecture == ACTOR_ARCHITECTURE_HIERARCHICAL:
            return self._forward_hierarchical(
                prompt_input_ids, images=images, **model_kwargs
            )
        return self._forward_candidate(
            prompt_input_ids, images=images, **model_kwargs
        )

    def act(self, prompt_input_ids, images=None, deterministic: bool = False, **model_kwargs):
        output = self.forward(prompt_input_ids, images=images, **model_kwargs)
        distribution = output.distribution
        if deterministic and output.stop_logits is not None:
            stop = torch.sigmoid(output.stop_logits) >= self.stop_threshold
            motion = output.motion_logits.argmax(dim=-1)
            action = torch.where(stop, torch.full_like(motion, 9), motion)
        else:
            action = (
                output.action_logits.argmax(dim=-1)
                if deterministic
                else distribution.sample()
            )
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
        if self.actor_head is not None:
            torch.save(self.actor_head.state_dict(), output_dir / "actor_head.pt")
        (output_dir / "actor_config.json").write_text(
            json.dumps(
                {
                    "architecture": self.actor_architecture,
                    "stop_threshold": self.stop_threshold,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_safe_heads(self, checkpoint_dir: str | Path, map_location="cpu") -> None:
        checkpoint_dir = Path(checkpoint_dir)
        reward_state = torch.load(checkpoint_dir / "reward_critic.pt", map_location=map_location)
        cost_state = torch.load(checkpoint_dir / "cost_critic.pt", map_location=map_location)
        self.reward_head.load_state_dict(reward_state)
        self.cost_head.load_state_dict(cost_state)

    def load_actor_head(self, checkpoint_dir: str | Path, map_location="cpu"):
        if self.actor_head is None:
            return
        path = Path(checkpoint_dir) / "actor_head.pt"
        if not path.is_file():
            raise RuntimeError(
                "hierarchical checkpoint is missing actor_head.pt"
            )
        self.actor_head.load_state_dict(torch.load(path, map_location=map_location))


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
