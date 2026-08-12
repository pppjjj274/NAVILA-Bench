from types import SimpleNamespace

import pytest
import torch

from safe_vln.checkpoint import (
    CHECKPOINT_ROLE_CRITIC_ONLY,
    CHECKPOINT_ROLE_POLICY,
    POLICY_INTERFACE_NAVILA_GREEDY,
    POLICY_INTERFACE_SAFE_DISCRETE,
    SAFE_CHECKPOINT_CONTRACT_VERSION,
    load_go2_inference_checkpoint,
    require_safe_policy_checkpoint,
    safe_checkpoint_contract,
)


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


def test_legacy_checkpoint_without_actor_audit_is_critic_only():
    contract = safe_checkpoint_contract({"mode": "warmup-critics"})
    assert contract["checkpoint_role"] == CHECKPOINT_ROLE_CRITIC_ONLY
    assert contract["policy_interface"] == POLICY_INTERFACE_NAVILA_GREEDY
    with pytest.raises(RuntimeError, match="requires an independently audited"):
        require_safe_policy_checkpoint({}, context="test")


def test_policy_checkpoint_requires_independent_nonzero_audit_contract():
    state = {
        "checkpoint_contract_version": SAFE_CHECKPOINT_CONTRACT_VERSION,
        "checkpoint_role": CHECKPOINT_ROLE_POLICY,
        "policy_interface": POLICY_INTERFACE_SAFE_DISCRETE,
        "actor/accepted": True,
        "actor/audit_independent": True,
        "calibration_episode_ids": ["calibration"],
        "audit_episode_ids": ["audit"],
        "actor/audit_target_source": "original-navila-policy",
        "actor/minimum_stop_accuracy": 0.5,
        "actor/minimum_non_stop_macro_accuracy": 0.4,
    }
    assert require_safe_policy_checkpoint(state, context="test") == {
        "checkpoint_contract_version": SAFE_CHECKPOINT_CONTRACT_VERSION,
        "checkpoint_role": CHECKPOINT_ROLE_POLICY,
        "policy_interface": POLICY_INTERFACE_SAFE_DISCRETE,
        "actor_audit_independent": True,
    }
    state["actor/minimum_non_stop_macro_accuracy"] = 0.0
    with pytest.raises(RuntimeError, match="zero motion threshold"):
        require_safe_policy_checkpoint(state, context="test")
    state["actor/minimum_non_stop_macro_accuracy"] = 0.4
    state.pop("actor/audit_target_source")
    with pytest.raises(RuntimeError, match="audit target"):
        require_safe_policy_checkpoint(state, context="test")


def test_policy_checkpoint_rejects_unverifiable_independence_boolean():
    state = {
        "checkpoint_contract_version": SAFE_CHECKPOINT_CONTRACT_VERSION,
        "checkpoint_role": CHECKPOINT_ROLE_POLICY,
        "policy_interface": POLICY_INTERFACE_SAFE_DISCRETE,
        "actor/accepted": True,
        "actor/audit_independent": True,
        "actor/audit_target_source": "original-navila-policy",
        "actor/minimum_stop_accuracy": 0.5,
        "actor/minimum_non_stop_macro_accuracy": 0.4,
    }
    with pytest.raises(RuntimeError, match="independent Actor audit"):
        require_safe_policy_checkpoint(state, context="test")


def test_legacy_accepted_actor_without_disjoint_audit_is_not_deployable():
    state = {
        "actor/accepted": True,
        "calibration_episode_ids": ["same"],
        "audit_episode_ids": ["same"],
    }
    with pytest.raises(RuntimeError, match="role='diagnostic'"):
        require_safe_policy_checkpoint(state, context="test")
