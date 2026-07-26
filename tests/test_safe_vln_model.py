from types import SimpleNamespace

import torch

from safe_vln.actions import ACTIONS
from safe_vln.model import SafeNavilaActorCritic


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
