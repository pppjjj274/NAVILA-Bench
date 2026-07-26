import json

import pytest

from safe_vln.calibration import fit_cost_profile, read_calibration_records
from safe_vln.objective import (
    SCHEMA_VERSION,
    build_objective_config,
    default_cost_profile,
    graded_oracle_reward,
    validate_cost_profile,
)
from safe_vln.trajectory import SafeTrajectoryRecorder


@pytest.mark.parametrize(
    ("predicted", "oracle", "expected"),
    [
        (0, 0, 1.0),
        (0, 1, 0.5),
        (0, 2, 0.25),
        (0, 3, 0.0),
        (9, 8, -1.0),
        (8, 9, -0.5),
    ],
)
def test_graded_oracle_reward(predicted, oracle, expected):
    reward, components = graded_oracle_reward(predicted, oracle)
    assert reward == expected
    assert sum(components.values()) == expected


def test_invalid_action_reward_takes_precedence():
    assert graded_oracle_reward(9, 9, invalid_action=True)[0] == -1.0


def test_cost_profile_fingerprint_detects_mutation():
    profile = default_cost_profile()
    profile["soft_thresholds"]["near_safe_m"] = 0.9
    with pytest.raises(ValueError, match="fingerprint"):
        validate_cost_profile(profile)


def test_macro_dense_cost_aggregates_mean_and_peak_without_termination():
    objective = build_objective_config(default_cost_profile())
    recorder = SafeTrajectoryRecorder(
        episode_id="1",
        scene_id="scene",
        instruction="go",
        objective_config=objective,
        cost_limit=0.25,
    )
    recorder.begin(
        {
            "action_id": 8,
            "text": "move",
            "velocity_command": [0.5, 0.0, 0.0],
            "duration": 1.5,
        },
        5.0,
    )
    base = {
        "risk_components": {
            "contact": 0.0,
            "tilt": 0.0,
            "near_obstacle": 0.5,
            "blocked": 0.0,
            "speed_near": 0.0,
            "smoothness": 0.0,
        }
    }
    recorder.record_env_step({**base, "dense_risk": 0.2})
    recorder.record_env_step({**base, "dense_risk": 0.6})
    item = recorder.finish(distance_after=4.8)
    assert item["hard_violation"] is False
    assert item["hard_cost"] == 0.0
    assert item["dense_cost"] == pytest.approx(0.05 * 0.4 + 0.05 * 0.6)
    assert item["cost"] == pytest.approx(0.05)
    assert item["schema_version"] == SCHEMA_VERSION


def test_hard_events_are_counted_once_even_when_multiple_flags_fire():
    objective = build_objective_config(default_cost_profile())
    recorder = SafeTrajectoryRecorder(
        episode_id="1",
        scene_id="scene",
        instruction="go",
        objective_config=objective,
    )
    recorder.begin(
        {
            "action_id": 8,
            "text": "move",
            "velocity_command": [0.5, 0.0, 0.0],
            "duration": 1.5,
        },
        5.0,
    )
    recorder.record_env_step(
        {
            "dense_risk": 1.0,
            "risk_components": {
                "contact": 1.0,
                "tilt": 1.0,
                "near_obstacle": 1.0,
                "blocked": 0.0,
                "speed_near": 1.0,
                "smoothness": 0.0,
            },
            "unsafe_contact": True,
            "fall": True,
        }
    )
    item = recorder.finish(
        distance_after=5.0,
        unsafe_contact=True,
        fall=True,
        terminated=True,
    )
    assert item["hard_cost"] == 1.0
    assert item["dense_cost"] == 0.1
    assert item["cost"] == 1.1


def test_calibration_uses_bounded_quantiles_and_requires_episode_coverage(tmp_path):
    path = tmp_path / "records.jsonl"
    records = []
    for episode in range(80):
        for index in range(2):
            records.append(
                {
                    "episode_id": str(episode),
                    "hard_violation": False,
                    "max_unsafe_contact_force": 0.2,
                    "orientation_angle": 0.25,
                    "front_obstacle_distance_m": 0.6 + index * 0.1,
                    "planar_speed_mps": 0.5,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in records))
    profile = fit_cost_profile(
        read_calibration_records(path),
        calibration_episodes=80,
        minimum_recorded_episodes=20,
    )
    assert profile["hard_thresholds"]["contact_force_n"] == 1.0
    assert profile["hard_thresholds"]["orientation_rad"] == 0.8
    assert profile["soft_thresholds"]["contact_force_n"] == pytest.approx(0.2)
    assert profile["calibration"]["episodes"] == 80
