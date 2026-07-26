from types import SimpleNamespace

import pytest

from safe_vln import train
from safe_vln.objective import build_objective_config, default_cost_profile


def _transition(episode_id, index, reward, cost, done):
    return {
        "episode_id": episode_id,
        "index": index,
        "old_log_prob": -0.1,
        "reward_value": 0.0,
        "cost_value": 0.0,
        "reward": reward,
        "cost": cost,
        "action_id": 6,
        "done": done,
    }


def test_rollout_loader_sorts_each_episode_and_computes_two_gaes(monkeypatch):
    source = [
        (None, _transition("one", 1, 2.0, 1.0, True)),
        (None, _transition("one", 0, 1.0, 0.0, False)),
        (None, _transition("two", 0, 3.0, 0.0, True)),
    ]
    monkeypatch.setattr(train, "iter_samples", lambda *args, **kwargs: iter(source))
    args = SimpleNamespace(
        rollout_dir="unused",
        split="train",
        max_samples=None,
        gamma=0.5,
        gae_lambda=1.0,
    )

    samples, episode_costs = train._load_on_policy_samples(args)

    assert [sample[1]["index"] for sample in samples[:2]] == [0, 1]
    assert samples[0][1]["reward_advantage"] == pytest.approx(2.0)
    assert samples[0][1]["cost_advantage"] == pytest.approx(0.5)
    assert samples[1][1]["reward_advantage"] == pytest.approx(2.0)
    assert episode_costs == {"one": 1.0, "two": 0.0}


def test_rollout_advantages_are_normalized_globally():
    samples = [
        (None, {"reward_advantage": 1.0, "cost_advantage": 0.0}),
        (None, {"reward_advantage": 3.0, "cost_advantage": 2.0}),
    ]

    train._normalize_rollout_advantages(samples)

    assert samples[0][1]["normalized_reward_advantage"] == pytest.approx(-1.0)
    assert samples[1][1]["normalized_reward_advantage"] == pytest.approx(1.0)
    assert samples[0][1]["normalized_cost_advantage"] == pytest.approx(-1.0)
    assert samples[1][1]["normalized_cost_advantage"] == pytest.approx(1.0)


def test_rollout_policy_version_validation():
    versioned = [(None, {"policy_version": 1})]
    train._validate_rollout_policy_version(versioned, 1)
    with pytest.raises(RuntimeError, match="do not match"):
        train._validate_rollout_policy_version(versioned, 0)

    unversioned = [(None, {"policy_version": None})]
    train._validate_rollout_policy_version(unversioned, 0)
    with pytest.raises(RuntimeError, match="only for policy version 0"):
        train._validate_rollout_policy_version(unversioned, 1)


def test_v2_objective_contract_requires_matching_checkpoint_and_samples():
    objective = build_objective_config(default_cost_profile())
    manifest = {
        "schema_version": "safe-vln-go2-v2",
        "objective_fingerprint": objective["fingerprint"],
        "objective_config": objective,
    }
    assert train._validate_objective_compatibility(
        manifest, {"objective_fingerprint": objective["fingerprint"]}
    ) == ("safe-vln-go2-v2", objective["fingerprint"])
    with pytest.raises(RuntimeError, match="do not match"):
        train._validate_objective_compatibility(
            manifest, {"objective_fingerprint": "other"}
        )

    samples = [
        (
            None,
            {
                "schema_version": "safe-vln-go2-v2",
                "objective_fingerprint": objective["fingerprint"],
                "policy_objective_fingerprint": objective["fingerprint"],
            },
        )
    ]
    train._validate_sample_objective(samples, manifest)
    samples[0][1]["objective_fingerprint"] = "other"
    with pytest.raises(RuntimeError, match="manifest"):
        train._validate_sample_objective(samples, manifest)


def test_legacy_v1_samples_may_omit_schema_but_cannot_mix_v2_metadata():
    manifest = {
        "schema_version": "safe-vln-go2-v1",
        "objective_fingerprint": None,
    }
    train._validate_sample_objective(
        [(None, {"schema_version": None, "objective_fingerprint": None})],
        manifest,
    )
    with pytest.raises(RuntimeError, match="do not match"):
        train._validate_sample_objective(
            [
                (
                    None,
                    {
                        "schema_version": "safe-vln-go2-v2",
                        "objective_fingerprint": "v2",
                    },
                )
            ],
            manifest,
        )
