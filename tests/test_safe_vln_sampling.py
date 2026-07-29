import pytest

from safe_vln.sampling import (
    sampling_summary,
    select_balanced_critic,
    select_balanced_oracle,
)


def _sample(episode, index, action, *, scene="scene", cost_return=0.0):
    return (
        None,
        {
            "episode_id": str(episode),
            "index": index,
            "observation_key": f"episode{episode}/state{index:06d}",
            "scene_id": scene,
            "oracle_action_id": action,
            "reward_return": 1.0,
            "cost_return": cost_return,
            "cost_components": {},
        },
    )


def test_balanced_oracle_keeps_all_stops_and_all_episodes_deterministically():
    samples = [
        _sample("a", 0, 0, scene="one"),
        _sample("a", 1, 9, scene="one"),
        _sample("b", 0, 1, scene="two"),
        _sample("b", 1, 3, scene="two"),
        _sample("c", 0, 4, scene="three"),
        _sample("c", 1, 9, scene="three"),
        _sample("c", 2, 8, scene="three"),
    ]

    first = select_balanced_oracle(samples, max_samples=5, seed=123)
    second = select_balanced_oracle(samples, max_samples=5, seed=123)

    assert [
        (item[1]["episode_id"], item[1]["index"]) for item in first
    ] == [
        (item[1]["episode_id"], item[1]["index"]) for item in second
    ]
    assert sampling_summary(first) == {
        "samples": 5,
        "episodes": 3,
        "scenes": 3,
        "stop_samples": 2,
        "action_counts": {
            str(key): value
            for key, value in sorted(
                {
                    item[1]["oracle_action_id"]: sum(
                        other[1]["oracle_action_id"]
                        == item[1]["oracle_action_id"]
                        for other in first
                    )
                    for item in first
                }.items()
            )
        },
    }


def test_balanced_oracle_rejects_limit_that_cannot_cover_contract():
    samples = [
        _sample("a", 0, 0),
        _sample("a", 1, 9),
        _sample("b", 0, 1),
    ]
    with pytest.raises(ValueError, match="need 3"):
        select_balanced_oracle(samples, max_samples=2, seed=1)


def test_balanced_critic_prioritizes_hard_events():
    hard = _sample("hard", 0, 0, cost_return=2.0)
    hard[1]["cost_components"]["collision_event"] = 1.0
    samples = [
        hard,
        _sample("safe", 0, 1, cost_return=0.0),
        _sample("medium", 0, 3, cost_return=0.2),
    ]

    selected = select_balanced_critic(samples, max_samples=1, seed=1)

    assert selected == [hard]
