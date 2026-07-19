import math

import numpy as np
import pytest
import torch

from safe_vln.actions import ACTIONS, action_from_text, normalize_policy_response
from safe_vln.cmdp import LagrangeController, compute_gae, compute_returns, safe_advantage
from safe_vln.trainer import constrained_ppo_loss
from safe_vln.trajectory import SafeTrajectoryRecorder


def test_canonical_action_space_and_legacy_parser():
    assert len(ACTIONS) == 10
    assert ACTIONS[7].velocity_command == (0.5, 0.0, 0.0)
    assert ACTIONS[7].duration == 1.0
    assert action_from_text("turn right 45 degrees")[0].action_id == 5
    assert action_from_text("unparseable action") == (ACTIONS[9], True)


def test_structured_response_uses_local_command_and_rejects_bad_values():
    result = normalize_policy_response(
        {"action_id": 6, "velocity_command": [99, 99, 99], "reward_value": float("nan"), "cost_value": 0.2}
    )
    assert result["velocity_command"] == [0.5, 0.0, 0.0]
    assert result["reward_value"] is None
    assert result["cost_value"] == 0.2


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
