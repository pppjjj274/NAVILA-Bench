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
