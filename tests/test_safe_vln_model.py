from types import SimpleNamespace

import torch

from safe_vln.actions import ACTIONS
from safe_vln.model import SafeNavilaActorCritic


class _Tokenizer:
    def __init__(self):
        self.ids = {action.text: index + 1 for index, action in enumerate(ACTIONS)}

    def __call__(self, text, **kwargs):
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


def test_actor_indexes_candidates_relative_to_expanded_sequence_end():
    model = SafeNavilaActorCritic(_ExpandedPrefixModel(), _Tokenizer())
    model.reward_head = _FirstCoordinate()
    model.cost_head = _FirstCoordinate()
    output = model(torch.tensor([[2, -200, 3]]), images=[torch.zeros(1)])

    assert output.action_logits.argmax(dim=-1).item() == 9
    assert output.reward_values.item() == 7.0
    assert output.cost_values.item() == 7.0
