from types import SimpleNamespace

import pytest

from safe_vln import train


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
