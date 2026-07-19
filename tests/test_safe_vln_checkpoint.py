from types import SimpleNamespace

import pytest
import torch

from safe_vln.checkpoint import load_go2_inference_checkpoint


class TinyActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(2, 2)
        self.critic = torch.nn.Linear(2, 1)
        self.cost_critic = torch.nn.Linear(2, 1)


class OptimizerMustNotLoad:
    def load_state_dict(self, _state):
        raise AssertionError("inference checkpoint loader must not restore the optimizer")


class FakeRunner:
    def __init__(self, *, empirical_normalization=False):
        self.alg = SimpleNamespace(
            actor_critic=TinyActorCritic(), optimizer=OptimizerMustNotLoad()
        )
        self.empirical_normalization = empirical_normalization
        self.current_learning_iteration = -1
        if empirical_normalization:
            self.obs_normalizer = torch.nn.Linear(2, 2)
            self.critic_obs_normalizer = torch.nn.Linear(2, 2)


def _filled_model(value):
    model = TinyActorCritic()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)
    return model


def _save_checkpoint(path, model_state, **extra):
    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": {"must_not": "load"},
        "iter": 42,
        "infos": {"source": "test"},
    }
    checkpoint.update(extra)
    torch.save(checkpoint, path)


def test_legacy_checkpoint_loads_actor_and_preserves_initialized_cost_critic(tmp_path):
    source = _filled_model(3.0)
    legacy_state = {
        key: value for key, value in source.state_dict().items() if not key.startswith("cost_critic.")
    }
    path = tmp_path / "legacy.pt"
    _save_checkpoint(path, legacy_state)

    runner = FakeRunner()
    cost_before = {
        key: value.clone() for key, value in runner.alg.actor_critic.cost_critic.state_dict().items()
    }
    infos = load_go2_inference_checkpoint(runner, path, map_location="cpu")

    assert infos == {"source": "test"}
    assert runner.current_learning_iteration == 42
    assert torch.equal(runner.alg.actor_critic.actor.weight, source.actor.weight)
    for key, value in runner.alg.actor_critic.cost_critic.state_dict().items():
        assert torch.equal(value, cost_before[key])


def test_complete_checkpoint_loads_cost_critic_and_normalizers(tmp_path):
    source = _filled_model(4.0)
    source_obs_normalizer = torch.nn.Linear(2, 2)
    source_critic_normalizer = torch.nn.Linear(2, 2)
    with torch.no_grad():
        for parameter in source_obs_normalizer.parameters():
            parameter.fill_(5.0)
        for parameter in source_critic_normalizer.parameters():
            parameter.fill_(6.0)

    path = tmp_path / "complete.pt"
    _save_checkpoint(
        path,
        source.state_dict(),
        obs_norm_state_dict=source_obs_normalizer.state_dict(),
        critic_obs_norm_state_dict=source_critic_normalizer.state_dict(),
    )
    runner = FakeRunner(empirical_normalization=True)
    load_go2_inference_checkpoint(runner, path, map_location="cpu")

    assert torch.equal(runner.alg.actor_critic.cost_critic.weight, source.cost_critic.weight)
    assert torch.equal(runner.obs_normalizer.weight, source_obs_normalizer.weight)
    assert torch.equal(runner.critic_obs_normalizer.weight, source_critic_normalizer.weight)


@pytest.mark.parametrize("failure", ["missing_actor", "unexpected_key"])
def test_checkpoint_rejects_non_cost_model_mismatches(tmp_path, failure):
    state = _filled_model(2.0).state_dict()
    if failure == "missing_actor":
        state.pop("actor.weight")
        expected = "actor.weight"
    else:
        state["unknown.weight"] = torch.zeros(1)
        expected = "unknown.weight"

    path = tmp_path / f"{failure}.pt"
    _save_checkpoint(path, state)
    with pytest.raises(RuntimeError, match=expected):
        load_go2_inference_checkpoint(FakeRunner(), path, map_location="cpu")


def test_checkpoint_requires_normalizers_when_enabled(tmp_path):
    path = tmp_path / "missing_normalizers.pt"
    _save_checkpoint(path, _filled_model(1.0).state_dict())

    with pytest.raises(RuntimeError, match="normalization states"):
        load_go2_inference_checkpoint(
            FakeRunner(empirical_normalization=True), path, map_location="cpu"
        )
