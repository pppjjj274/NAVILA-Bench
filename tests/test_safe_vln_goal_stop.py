import pytest

from safe_vln.goal_stop import GoalStopController


def test_policy_eval_never_overrides_or_patience_terminates():
    controller = GoalStopController(
        goal_radius=3.0,
        mode="policy",
        dataset_role="eval",
        missed_stop_patience=3,
    )
    decisions = [
        controller.resolve(8, 0.25, navigation_reward_valid=True)
        for _ in range(5)
    ]
    assert all(item.executed_action_id == 8 for item in decisions)
    assert all(item.missed_stop for item in decisions)
    assert not any(item.immediate_terminal for item in decisions)
    assert not any(item.terminate_after_execution for item in decisions)


def test_policy_train_terminates_after_third_consecutive_missed_stop():
    controller = GoalStopController(
        goal_radius=3.0,
        mode="policy",
        dataset_role="train",
        missed_stop_patience=3,
    )
    first = controller.resolve(8, 1.0, navigation_reward_valid=True)
    second = controller.resolve(7, 1.2, navigation_reward_valid=True)
    third = controller.resolve(6, 1.4, navigation_reward_valid=True)
    assert [first.consecutive_missed_stops, second.consecutive_missed_stops] == [1, 2]
    assert not first.terminate_after_execution
    assert not second.terminate_after_execution
    assert third.terminate_after_execution
    assert third.termination_reason == "missed_stop_patience"
    assert third.executed_action_id == 6


def test_leaving_goal_region_resets_missed_stop_patience():
    controller = GoalStopController(
        goal_radius=3.0,
        mode="policy",
        dataset_role="train",
    )
    controller.resolve(8, 1.0, navigation_reward_valid=True)
    outside = controller.resolve(8, 4.0, navigation_reward_valid=True)
    inside_again = controller.resolve(8, 2.0, navigation_reward_valid=True)
    assert outside.consecutive_missed_stops == 0
    assert inside_again.consecutive_missed_stops == 1


def test_shield_records_policy_action_but_executes_stop():
    controller = GoalStopController(
        goal_radius=3.0,
        mode="shield",
        dataset_role="eval",
    )
    decision = controller.resolve(8, 0.2, navigation_reward_valid=True)
    assert decision.policy_action_id == 8
    assert decision.executed_action_id == 9
    assert decision.shield_intervened
    assert decision.success
    assert decision.termination_reason == "shield_success"


@pytest.mark.parametrize(
    ("distance", "expected_success"),
    [(2.999, True), (3.001, False)],
)
def test_policy_stop_uses_episode_goal_radius(distance, expected_success):
    controller = GoalStopController(
        goal_radius=3.0,
        mode="policy",
        dataset_role="eval",
    )
    decision = controller.resolve(9, distance, navigation_reward_valid=True)
    assert decision.success is expected_success
    assert decision.failed_stop is (not expected_success)


def test_sensor_gate_forces_success_inside_goal():
    controller = GoalStopController(
        goal_radius=3.0, mode="sensor-gated", dataset_role="train"
    )
    decision = controller.resolve(8, 2.0, navigation_reward_valid=True)
    assert decision.executed_action_id == 9
    assert decision.success
    assert decision.goal_gate_reason == "goal_radius_stop"
    assert decision.termination_reason == "sensor_gated_success"


def test_sensor_gate_replaces_premature_stop_with_best_motion():
    controller = GoalStopController(
        goal_radius=3.0, mode="sensor-gated", dataset_role="train"
    )
    probabilities = [0.01] * 10
    probabilities[7] = 0.4
    probabilities[9] = 0.5
    decision = controller.resolve(
        9,
        4.0,
        navigation_reward_valid=True,
        action_probabilities=probabilities,
    )
    assert decision.executed_action_id == 7
    assert decision.shield_intervened
    assert not decision.immediate_terminal
    assert decision.goal_gate_reason == "premature_stop_rejected"


@pytest.mark.parametrize(
    ("reward_valid", "probabilities", "reason"),
    [
        (False, [0.1] * 10, "goal_distance_unavailable"),
        (True, None, "goal_gate_no_fallback"),
    ],
)
def test_sensor_gate_fails_closed(reward_valid, probabilities, reason):
    controller = GoalStopController(
        goal_radius=3.0, mode="sensor-gated", dataset_role="eval"
    )
    decision = controller.resolve(
        9,
        4.0,
        navigation_reward_valid=reward_valid,
        action_probabilities=probabilities,
    )
    assert decision.executed_action_id == 9
    assert decision.immediate_terminal
    assert not decision.success
    assert decision.termination_reason == reason
    assert decision.failed_stop is (
        reward_valid and probabilities is None
    )
