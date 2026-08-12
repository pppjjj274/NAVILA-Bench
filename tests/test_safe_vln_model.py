import json
from types import SimpleNamespace

import pytest
import torch

from safe_vln.actions import ACTIONS
from safe_vln.model import (
    ACTOR_ARCHITECTURE_HIERARCHICAL,
    ACTOR_ARCHITECTURE_FACTORIZED,
    FactorizedActorHead,
    HierarchicalActorHead,
    SafeActorCriticOutput,
    SafeNavilaActorCritic,
)


class _Tokenizer:
    def __init__(self):
        self.ids = {}

    def __call__(self, text, **kwargs):
        self.ids.setdefault(text, len(self.ids) + 1)
        return SimpleNamespace(input_ids=torch.tensor([[self.ids[text]]]))


class _ExpandedPrefixModel(torch.nn.Module):
    """Mimic NaViLA replacing image placeholders with extra embeddings."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(hidden_size=2)

    def forward(self, input_ids, **kwargs):
        expanded_length = input_ids.shape[1] + 3
        logits = torch.zeros(1, expanded_length, 16)
        target = int(input_ids[0, -1])
        logits[0, -2, target] = target / 2.0
        hidden = torch.zeros(1, expanded_length, 2)
        hidden[0, -2, 0] = 7.0
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


class _FirstCoordinate(torch.nn.Module):
    def forward(self, hidden_state):
        return hidden_state[:, 0]


class _NaVILABackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *, input_ids, inputs_embeds, **kwargs):
        self.calls += 1
        assert input_ids is None
        assert inputs_embeds is not None
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class _NaVILAInnerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _NaVILABackbone()
        self.lm_head = torch.nn.Linear(2, 16, bias=False)


class _NaVILAOuterModel(torch.nn.Module):
    """Mimic the NaViLA wrapper whose supervised forward is unsuitable here."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(hidden_size=2)
        self.llm = _NaVILAInnerModel()
        self.prepared = 0
        self.encoded = 0

    def encode_images(self, images):
        self.encoded += 1
        return torch.zeros(1, 1, 2)

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
    ):
        self.prepared += 1
        self.encode_images(images)
        expanded_length = input_ids.shape[1] + 3
        return (
            None,
            None,
            torch.ones(1, expanded_length, dtype=torch.long),
            None,
            torch.zeros(1, expanded_length, 2),
            None,
        )

    def forward(self, *args, **kwargs):
        raise AssertionError("Safe-VLN must bypass NaViLA's outer LM-loss forward")


def test_actor_indexes_candidates_relative_to_expanded_sequence_end():
    model = SafeNavilaActorCritic(_ExpandedPrefixModel(), _Tokenizer())
    model.reward_head = _FirstCoordinate()
    model.cost_head = _FirstCoordinate()
    output = model(torch.tensor([[2, -200, 3]]), images=[torch.zeros(1)])

    assert output.action_logits.argmax(dim=-1).item() == 9
    assert output.reward_values.item() == 7.0
    assert output.cost_values.item() == 7.0


def test_actor_bypasses_navila_supervised_loss_forward():
    base_model = _NaVILAOuterModel()
    model = SafeNavilaActorCritic(base_model, _Tokenizer())

    output = model(torch.tensor([[2, -200, 3]]), images=[torch.zeros(1)])

    assert output.action_logits.shape == (1, len(ACTIONS))
    assert base_model.prepared == len(ACTIONS)
    assert base_model.encoded == 1
    assert base_model.llm.model.calls == len(ACTIONS)


def test_value_only_forward_does_not_score_ten_actor_candidates():
    base_model = _NaVILAOuterModel()
    model = SafeNavilaActorCritic(base_model, _Tokenizer())

    values = model.forward_values(
        torch.tensor([[2, -200, 3]]), images=[torch.zeros(1)]
    )

    assert values["reward_values"].shape == (1,)
    assert values["cost_values"].shape == (1,)
    assert base_model.prepared == 1
    assert base_model.encoded == 1
    assert base_model.llm.model.calls == 1


def test_hierarchical_joint_probabilities_are_normalized():
    stop_logits = torch.tensor([torch.logit(torch.tensor(0.8))])
    motion_logits = torch.tensor([[2.0] + [0.0] * 8])
    log_probs = HierarchicalActorHead.joint_log_probs(
        stop_logits, motion_logits
    )
    probabilities = log_probs.exp()
    assert probabilities.shape == (1, 10)
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert probabilities[0, 9].item() == pytest.approx(0.8)
    assert probabilities[0, :9].sum().item() == pytest.approx(0.2)


def test_factorized_joint_probabilities_are_normalized():
    stop_logits = torch.tensor([torch.logit(torch.tensor(0.2))])
    direction_logits = torch.tensor([[2.0, 0.0, 0.0]])
    magnitude_logits = torch.zeros(1, 3, 3)
    log_probs = FactorizedActorHead.joint_log_probs(
        stop_logits, direction_logits, magnitude_logits
    )
    probabilities = log_probs.exp()
    assert probabilities.shape == (1, 10)
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert probabilities[0, 9].item() == pytest.approx(0.2)
    motion = FactorizedActorHead.motion_log_probs(
        direction_logits, magnitude_logits
    ).exp()
    assert motion.sum().item() == pytest.approx(1.0)


def test_hierarchical_actor_uses_one_multimodal_forward():
    base_model = _NaVILAOuterModel()
    model = SafeNavilaActorCritic(
        base_model,
        _Tokenizer(),
        actor_architecture=ACTOR_ARCHITECTURE_HIERARCHICAL,
    )
    output = model(torch.tensor([[2, -200, 3]]), images=[torch.zeros(1)])

    assert output.action_logits.shape == (1, len(ACTIONS))
    assert output.stop_logits.shape == (1,)
    assert output.motion_logits.shape == (1, 9)
    assert base_model.prepared == 1
    assert base_model.encoded == 1
    assert base_model.llm.model.calls == 1


def test_factorized_actor_uses_one_multimodal_forward():
    base_model = _NaVILAOuterModel()
    model = SafeNavilaActorCritic(
        base_model,
        _Tokenizer(),
        actor_architecture=ACTOR_ARCHITECTURE_FACTORIZED,
    )
    output = model(torch.tensor([[2, -200, 3]]), images=[torch.zeros(1)])

    assert output.action_logits.shape == (1, len(ACTIONS))
    assert output.stop_logits.shape == (1,)
    assert output.motion_logits.shape == (1, 9)
    assert output.direction_logits.shape == (1, 3)
    assert output.magnitude_logits.shape == (1, 3, 3)
    assert base_model.prepared == 1
    assert base_model.encoded == 1
    assert base_model.llm.model.calls == 1


def test_factorized_action_mapping_is_direction_major():
    direction = torch.tensor([[-10.0, -10.0, 10.0]])
    magnitude = torch.full((1, 3, 3), -10.0)
    magnitude[0, 2, 1] = 10.0
    prediction = FactorizedActorHead.motion_log_probs(
        direction, magnitude
    ).argmax()
    assert prediction == 7


def test_hierarchical_deterministic_stop_threshold():
    model = SafeNavilaActorCritic(
        _ExpandedPrefixModel(),
        _Tokenizer(),
        actor_architecture=ACTOR_ARCHITECTURE_HIERARCHICAL,
        stop_threshold=0.5,
    )

    def fixed_forward(*args, **kwargs):
        stop_logits = torch.tensor([0.0])
        motion_logits = torch.tensor([[0.0, 3.0] + [0.0] * 7])
        return SafeActorCriticOutput(
            action_logits=HierarchicalActorHead.joint_log_probs(
                stop_logits, motion_logits
            ),
            reward_values=torch.tensor([1.0]),
            cost_values=torch.tensor([2.0]),
            stop_logits=stop_logits,
            motion_logits=motion_logits,
        )

    model.forward = fixed_forward
    result = model.act(torch.tensor([[1]]), deterministic=True)
    assert result["action_id"] == 9
    assert len(result["action_probabilities"]) == 10
    assert sum(result["action_probabilities"]) == pytest.approx(1.0)


def test_hierarchical_checkpoint_saves_actor_contract(tmp_path):
    model = SafeNavilaActorCritic(
        _ExpandedPrefixModel(),
        _Tokenizer(),
        actor_architecture=ACTOR_ARCHITECTURE_HIERARCHICAL,
    )
    model.save_safe_heads(tmp_path)
    assert (tmp_path / "actor_head.pt").is_file()
    assert (tmp_path / "actor_config.json").is_file()

    restored = SafeNavilaActorCritic(
        _ExpandedPrefixModel(),
        _Tokenizer(),
        actor_architecture=ACTOR_ARCHITECTURE_HIERARCHICAL,
    )
    restored.load_actor_head(tmp_path)
    restored.load_safe_heads(tmp_path)


def test_factorized_checkpoint_saves_versioned_action_mapping(tmp_path):
    model = SafeNavilaActorCritic(
        _ExpandedPrefixModel(),
        _Tokenizer(),
        actor_architecture=ACTOR_ARCHITECTURE_FACTORIZED,
        stop_threshold=0.63,
    )
    model.save_safe_heads(tmp_path)
    config = json.loads(
        (tmp_path / "actor_config.json").read_text(encoding="utf-8")
    )
    assert config["schema_version"] == 2
    assert config["architecture"] == ACTOR_ARCHITECTURE_FACTORIZED
    assert config["stop_threshold"] == pytest.approx(0.63)
    assert config["action_factorization"]["action_mapping"] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
