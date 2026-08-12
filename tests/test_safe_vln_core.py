import math

import numpy as np
import pytest
import torch
from PIL import Image

from safe_vln.actions import (
    ACTIONS,
    NAVILA_ACTION_RESPONSES,
    action_from_text,
    has_valid_policy_statistics,
    normalize_policy_response,
)
from safe_vln.cmdp import LagrangeController, compute_gae, compute_returns, safe_advantage
from safe_vln.trainer import constrained_ppo_loss
from safe_vln.trajectory import SafeTrajectoryRecorder
from safe_vln.native_history import sample_native_history


def test_canonical_action_space_and_legacy_parser():
    assert len(ACTIONS) == 10
    assert len(NAVILA_ACTION_RESPONSES) == len(ACTIONS)
    assert NAVILA_ACTION_RESPONSES[-1].startswith("I think I should stop")
    assert ACTIONS[7].velocity_command == (0.5, 0.0, 0.0)
    assert ACTIONS[7].duration == 1.0
    assert action_from_text("turn right 45 degrees")[0].action_id == 5
    assert action_from_text("unparseable action") == (ACTIONS[9], True)
    assert normalize_policy_response(
        "The next action is move forward 50 cm."
    )["policy_interface"] == "navila-greedy-text-v1"


def test_structured_response_uses_local_command_and_rejects_bad_values():
    result = normalize_policy_response(
        {"action_id": 6, "velocity_command": [99, 99, 99], "reward_value": float("nan"), "cost_value": 0.2}
    )
    assert result["velocity_command"] == [0.5, 0.0, 0.0]
    assert result["reward_value"] is None
    assert result["cost_value"] == 0.2
    assert normalize_policy_response({"action_id": 2.7})["invalid_action"] is True
    assert normalize_policy_response({"action_id": True})["invalid_action"] is True


def test_structured_response_records_normalized_action_probabilities():
    probabilities = [0.1] * len(ACTIONS)
    result = normalize_policy_response(
        {"action_id": 6, "action_probabilities": probabilities}
    )

    assert result["action_probabilities"] == pytest.approx(
        probabilities
    )
    assert normalize_policy_response(
        {"action_id": 6, "action_probabilities": [1.0] * len(ACTIONS)}
    )["action_probabilities"] is None
    assert normalize_policy_response(
        {"action_id": 6, "action_probabilities": [1.0, float("nan")]}
    )["action_probabilities"] is None
    assert normalize_policy_response(
        {"action_id": 6, "objective_fingerprint": "objective"}
    )["objective_fingerprint"] == "objective"


def test_native_history_marks_repeat_first_padding():
    entries = [
        {
            "image": Image.new("RGB", (2, 2), color=index),
            "metadata": {"frame_index": index, "physics_step": index * 25},
        }
        for index in range(3)
    ]
    sampled = sample_native_history(entries, num_frames=8)

    assert len(sampled) == 8
    assert sum(item["metadata"]["history_padding"] for item in sampled) == 5
    assert all(
        not item["metadata"]["strict_observation_state_alignment"]
        for item in sampled[:5]
    )
    assert sampled[-1]["metadata"]["frame_index"] == 2
    assert sampled[-1]["metadata"]["strict_observation_state_alignment"] is not False


def test_ppo_statistics_require_matching_objective_and_complete_values():
    output = normalize_policy_response(
        {
            "action_id": 8,
            "log_prob": math.log(0.1),
            "reward_value": 0.5,
            "cost_value": 0.2,
                "policy_version": 3,
                "policy_interface": "safe-vln-discrete-v1",
                "objective_fingerprint": "objective-a",
            "action_probabilities": [0.1] * len(ACTIONS),
        }
    )
    assert has_valid_policy_statistics(
        output, objective_fingerprint="objective-a"
    )
    assert not has_valid_policy_statistics(
        output, objective_fingerprint="objective-b"
    )
    output["policy_interface"] = "navila-greedy-text-v1"
    assert not has_valid_policy_statistics(
        output, objective_fingerprint="objective-a"
    )


def test_ppo_statistics_reject_probability_log_prob_mismatch():
    output = normalize_policy_response(
        {
            "action_id": 8,
            "log_prob": -1.0,
            "reward_value": 0.5,
            "cost_value": 0.2,
            "policy_version": 3,
            "policy_interface": "safe-vln-discrete-v1",
            "objective_fingerprint": "objective-a",
            "action_probabilities": [0.1] * len(ACTIONS),
        }
    )
    assert not has_valid_policy_statistics(
        output, objective_fingerprint="objective-a"
    )


def test_reward_cost_returns_and_lagrange_direction():
    assert compute_returns([1.0, 2.0], [False, True], gamma=0.5) == [2.0, 2.0]
    controller = LagrangeController(cost_limit=0.1, multiplier=0.2, learning_rate=0.5)
    assert controller.update(0.5) > 0.2
    assert controller.update(0.0) < 0.4
    np.testing.assert_allclose(safe_advantage([1.0], [0.5], 1.0), [0.25])


def test_gae_does_not_bootstrap_terminal_state():
    advantages, returns = compute_gae([1.0], [0.2], [True], next_value=100.0)
    np.testing.assert_allclose(advantages, [0.8])
    np.testing.assert_allclose(returns, [1.0])


def test_trajectory_separates_collision_cost_from_reward():
    recorder = SafeTrajectoryRecorder(
        episode_id="1", scene_id="scene", instruction="go", step_penalty=-0.01, cost_limit=0.0
    )
    recorder.begin(normalize_policy_response({"action_id": 7, "reward_value": 0.3, "cost_value": 0.1}), 5.0)
    recorder.count_env_step()
    item = recorder.finish(
        distance_after=4.5, unsafe_contact=True, terminated=True, termination_reason="unsafe_contact"
    )
    recorder.finalize()
    assert item["reward"] == pytest.approx(0.49)
    assert item["cost"] == 1.0
    assert recorder.transitions[0]["cost_return"] == 1.0
    assert recorder.summary({"success": 0.0})["constraint_satisfied"] is False


def test_trajectory_records_blocked_as_terminal_cost():
    recorder = SafeTrajectoryRecorder(
        episode_id="1", scene_id="scene", instruction="go", cost_limit=0.0
    )
    recorder.begin(normalize_policy_response({"action_id": 8}), 5.0)
    recorder.count_env_step()
    item = recorder.finish(
        distance_after=5.0,
        blocked=True,
        safety_diagnostics={"blocked_steps": 100, "blocked_displacement": 0.01},
        terminated=True,
        termination_reason="blocked",
    )
    recorder.finalize()

    assert item["cost"] == 1.0
    assert item["cost_components"]["blocked"] == 1.0
    assert item["safety_diagnostics"]["blocked_steps"] == 100
    assert item["termination_reason"] == "blocked"
    assert recorder.transitions[0]["cost_return"] == 1.0
    assert recorder.summary()["blocked_count"] == 1.0
    assert recorder.summary()["has_blocked"] is True


def test_hard_event_cost_is_not_diluted_by_long_episode():
    recorder = SafeTrajectoryRecorder(
        episode_id="1",
        scene_id="scene",
        instruction="go",
        cost_limit=0.25,
    )
    for index in range(5):
        recorder.begin(normalize_policy_response({"action_id": 8}), 5.0)
        recorder.count_env_step()
        recorder.finish(
            distance_after=5.0,
            unsafe_contact=index == 4,
            terminated=index == 4,
        )

    summary = recorder.summary()

    assert summary["cumulative_cost"] == 1.0
    assert summary["constraint_cost"] == 1.0
    assert summary["cost_normalization"] == "cumulative_episode_sum"
    assert summary["constraint_satisfied"] is False


def test_trajectory_records_failed_turn_as_non_terminal_diagnostic():
    recorder = SafeTrajectoryRecorder(
        episode_id="1", scene_id="scene", instruction="go", cost_limit=0.0
    )
    recorder.begin(normalize_policy_response({"action_id": 2}), 5.0)
    recorder.count_env_step()
    item = recorder.finish(
        distance_after=5.0,
        turn_blocked=True,
        safety_diagnostics={
            "turn_blocked": True,
            "turn_execution": {"execution_ratio": 0.0},
        },
        terminated=False,
        termination_reason=None,
    )
    assert item["cost"] == 0.0
    assert item["hard_violation"] is False
    assert item["cost_components"]["turn_blocked"] == 1.0
    assert item["cost_components"]["turn_tracking_failure"] == 1.0
    assert recorder.summary()["turn_blocked_count"] == 1.0
    assert recorder.summary()["turn_tracking_failure_count"] == 1.0
    assert recorder.summary()["has_turn_blocked"] is True
    assert recorder.summary()["has_turn_tracking_failure"] is True


def test_trajectory_truncation_closes_observation_chain():
    recorder = SafeTrajectoryRecorder(
        episode_id="1", scene_id="scene", instruction="go"
    )
    recorder.begin(normalize_policy_response({"action_id": 8}), 5.0)
    recorder.count_env_step()
    recorder.finish(distance_after=4.5)
    recorder.transitions[-1]["next_observation_key"] = "episode1/state000001"

    recorder.truncate_last("max_vlm_calls")
    recorder.finalize()

    transition = recorder.transitions[-1]
    assert transition["done"] is True
    assert transition["terminated"] is False
    assert transition["truncated"] is True
    assert transition["termination_reason"] == "max_vlm_calls"
    assert transition["next_observation_key"] is None


def test_trajectory_penalizes_and_summarizes_missed_stop():
    recorder = SafeTrajectoryRecorder(
        episode_id="1",
        scene_id="scene",
        instruction="go",
        step_penalty=-0.01,
        missed_stop_penalty=-0.5,
    )
    recorder.begin(normalize_policy_response({"action_id": 8}), 0.5)
    item = recorder.finish(distance_after=0.4, missed_stop=True)
    recorder.transitions[-1].update(
        {
            "in_goal_radius": True,
            "oracle_valid": True,
            "oracle_action_id": 9,
            "policy_action_id": 8,
            "goal_radius_m": 3.0,
        }
    )
    item = recorder.transitions[-1]
    recorder.finalize()
    summary = recorder.summary()
    assert item["reward"] == pytest.approx(-0.41)
    assert summary["entered_goal_radius"] is True
    assert summary["missed_stop_count"] == 1
    assert summary["oracle_stop_decisions"] == 1
    assert summary["model_stop_decisions"] == 0
    assert summary["stop_recall_in_goal"] == 0.0


def test_trajectory_accepts_oracle_reward_override_and_keeps_physical_progress():
    recorder = SafeTrajectoryRecorder(
        episode_id="1",
        scene_id="scene",
        instruction="go",
    )
    recorder.begin(normalize_policy_response({"action_id": 7}), 5.0)
    item = recorder.finish(
        distance_after=4.0,
        reward_override=0.0,
        reward_components={"oracle_action_match": 0.0},
        terminated=True,
        termination_reason="replay_exhausted",
    )
    recorder.finalize()

    assert item["reward"] == 0.0
    assert item["physical_progress"] == pytest.approx(1.0)
    assert item["reward_components"] == {"oracle_action_match": 0.0}
    assert recorder.transitions[0]["reward_return"] == 0.0


def test_constrained_ppo_loss_is_finite_and_backpropagates():
    new_log_probs = torch.tensor([-0.2, -0.4], requires_grad=True)
    reward_values = torch.tensor([0.1, 0.2], requires_grad=True)
    cost_values = torch.tensor([0.2, 0.3], requires_grad=True)
    loss, stats = constrained_ppo_loss(
        new_log_probs=new_log_probs,
        old_log_probs=torch.tensor([-0.25, -0.35]),
        reward_advantages=torch.tensor([1.0, -0.5]),
        cost_advantages=torch.tensor([0.2, 1.0]),
        reward_values=reward_values,
        cost_values=cost_values,
        reward_returns=torch.tensor([0.8, 0.1]),
        cost_returns=torch.tensor([0.0, 1.0]),
        entropy=torch.tensor([0.5, 0.5]),
        lagrange_multiplier=0.5,
    )
    assert math.isfinite(stats["loss/total"])
    loss.backward()
    assert new_log_probs.grad is not None
    assert reward_values.grad is not None
    assert cost_values.grad is not None


def test_constrained_ppo_adds_masked_oracle_cross_entropy():
    logits = torch.zeros((2, len(ACTIONS)), requires_grad=True)
    loss, stats = constrained_ppo_loss(
        new_log_probs=torch.tensor([-0.2, -0.2], requires_grad=True),
        old_log_probs=torch.tensor([-0.2, -0.2]),
        reward_advantages=torch.tensor([0.0, 0.0]),
        cost_advantages=torch.tensor([0.0, 0.0]),
        reward_values=torch.tensor([0.0, 0.0], requires_grad=True),
        cost_values=torch.tensor([0.0, 0.0], requires_grad=True),
        reward_returns=torch.tensor([0.0, 0.0]),
        cost_returns=torch.tensor([0.0, 0.0]),
        entropy=torch.tensor([0.0, 0.0]),
        lagrange_multiplier=0.0,
        action_logits=logits,
        oracle_action_ids=torch.tensor([3, 4]),
        oracle_mask=torch.tensor([True, False]),
        oracle_ce_coef=0.05,
    )
    assert stats["loss/oracle_ce"] == pytest.approx(math.log(len(ACTIONS)))
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_oracle_stop_weight_changes_ce_and_reports_stop_accuracy():
    logits = torch.zeros((2, len(ACTIONS)), requires_grad=True)
    logits.data[0, 0] = 2.0
    logits.data[1, 9] = 2.0
    _, stats = constrained_ppo_loss(
        new_log_probs=torch.tensor([0.0, 0.0], requires_grad=True),
        old_log_probs=torch.tensor([0.0, 0.0]),
        reward_advantages=torch.tensor([0.0, 0.0]),
        cost_advantages=torch.tensor([0.0, 0.0]),
        reward_values=torch.tensor([0.0, 0.0], requires_grad=True),
        cost_values=torch.tensor([0.0, 0.0], requires_grad=True),
        reward_returns=torch.tensor([0.0, 0.0]),
        cost_returns=torch.tensor([0.0, 0.0]),
        entropy=torch.tensor([0.0, 0.0]),
        lagrange_multiplier=0.0,
        action_logits=logits,
        oracle_action_ids=torch.tensor([0, 9]),
        oracle_mask=torch.tensor([True, True]),
        oracle_sample_weights=torch.tensor([1.0, 5.0]),
        oracle_ce_coef=1.0,
    )
    assert stats["oracle/samples"] == 2
    assert stats["oracle/stop_samples"] == 1
    assert stats["oracle/stop_accuracy"] == 1.0


def test_oracle_stop_weight_survives_batch_size_one():
    def oracle_ce(weight):
        _, stats = constrained_ppo_loss(
            new_log_probs=torch.tensor([0.0], requires_grad=True),
            old_log_probs=torch.tensor([0.0]),
            reward_advantages=torch.tensor([0.0]),
            cost_advantages=torch.tensor([0.0]),
            reward_values=torch.tensor([0.0], requires_grad=True),
            cost_values=torch.tensor([0.0], requires_grad=True),
            reward_returns=torch.tensor([0.0]),
            cost_returns=torch.tensor([0.0]),
            entropy=torch.tensor([0.0]),
            lagrange_multiplier=0.0,
            action_logits=torch.zeros((1, len(ACTIONS)), requires_grad=True),
            oracle_action_ids=torch.tensor([9]),
            oracle_mask=torch.tensor([True]),
            oracle_sample_weights=torch.tensor([weight]),
            oracle_ce_coef=1.0,
        )
        return stats["loss/oracle_ce"]

    assert oracle_ce(5.0) == pytest.approx(5.0 * oracle_ce(1.0))
