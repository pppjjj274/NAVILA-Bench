import math

import pytest
import torch

from safe_vln.safety import (
    BlockedDetector,
    BlockedStatus,
    blocked_progress_risk,
    combined_step_risk,
    front_obstacle_distance,
    inverse_distance_risk,
    is_unsafe_contact_body,
    linear_risk,
    orientation_angle,
    smoothness_risk,
    unsafe_contact_diagnostics,
)


BODY_NAMES = ["base", "FL_hip", "FL_thigh", "FL_calf", "FL_foot"]


@pytest.mark.parametrize("body_name", BODY_NAMES[:-1])
def test_non_foot_body_contact_is_unsafe(body_name):
    forces = torch.zeros(3, len(BODY_NAMES), 3)
    body_index = BODY_NAMES.index(body_name)
    forces[1, body_index, 0] = 2.0

    unsafe, contact_forces, max_force = unsafe_contact_diagnostics(
        BODY_NAMES, forces, threshold=1.0
    )

    assert unsafe is True
    assert contact_forces == {body_name: 2.0}
    assert max_force == pytest.approx(2.0)


def test_foot_contact_is_excluded_from_unsafe_collision():
    forces = torch.zeros(3, len(BODY_NAMES), 3)
    forces[:, BODY_NAMES.index("FL_foot"), 2] = 100.0

    unsafe, contact_forces, max_force = unsafe_contact_diagnostics(
        BODY_NAMES, forces, threshold=1.0
    )

    assert unsafe is False
    assert contact_forces == {}
    assert max_force == 0.0
    assert is_unsafe_contact_body("RR_calf") is True
    assert is_unsafe_contact_body("RR_foot") is False


def test_orientation_angle_distinguishes_upright_and_fall():
    assert orientation_angle(torch.tensor([0.0, 0.0, -1.0])) == pytest.approx(0.0)
    tilted = torch.tensor([math.sin(1.0), 0.0, -math.cos(1.0)])
    assert orientation_angle(tilted) == pytest.approx(1.0)


def _run_stationary_forward(detector, steps):
    status = None
    position = torch.zeros(3)
    command = torch.tensor([0.5, 0.0, 0.0])
    for _ in range(steps):
        status = detector.update(command, position, position)
    return status


def test_blocked_triggers_on_100th_stationary_forward_step():
    detector = BlockedDetector(window_steps=100, min_displacement=0.10)

    status = _run_stationary_forward(detector, 99)
    assert status.blocked is False
    assert status.observed_steps == 99

    status = _run_stationary_forward(detector, 1)
    assert status.blocked is True
    assert status.observed_steps == 100
    assert status.displacement == 0.0


def test_sufficient_forward_displacement_does_not_trigger_blocked():
    detector = BlockedDetector(window_steps=100, min_displacement=0.10)
    command = torch.tensor([0.5, 0.0, 0.0])
    previous = torch.zeros(3)
    status = None

    for step in range(1, 121):
        current = torch.tensor([step * 0.01, 0.0, 0.0])
        status = detector.update(command, previous, current)
        previous = current

    assert status.blocked is False
    assert status.displacement == pytest.approx(1.0)


def test_turning_resets_blocked_window():
    detector = BlockedDetector(window_steps=100, min_displacement=0.10)
    _run_stationary_forward(detector, 60)

    status = detector.update(
        torch.tensor([0.0, 0.0, 0.5]), torch.zeros(3), torch.zeros(3)
    )
    assert status.observed_steps == 0

    status = _run_stationary_forward(detector, 99)
    assert status.blocked is False
    assert status.observed_steps == 99


def test_continuous_risks_are_bounded_at_thresholds():
    assert linear_risk(0.5, 0.5, 1.0) == 0.0
    assert linear_risk(0.75, 0.5, 1.0) == pytest.approx(0.5)
    assert linear_risk(2.0, 0.5, 1.0) == 1.0
    assert inverse_distance_risk(0.8, 0.25, 0.8) == 0.0
    assert inverse_distance_risk(0.25, 0.25, 0.8) == 1.0


def test_blocked_precursor_reaches_one_at_hard_event():
    halfway = blocked_progress_risk(
        BlockedStatus(False, 50, 0.0),
        window_steps=100,
        min_displacement=0.1,
    )
    terminal = blocked_progress_risk(
        BlockedStatus(True, 100, 0.0),
        window_steps=100,
        min_displacement=0.1,
    )
    assert halfway == pytest.approx(0.5)
    assert terminal == pytest.approx(1.0)


def test_command_smoothness_ignores_first_action_and_stop():
    forward = torch.tensor([0.5, 0.0, 0.0])
    turn = torch.tensor([0.0, 0.0, math.pi / 6.0])
    assert smoothness_risk(None, forward) == 0.0
    assert smoothness_risk(forward, torch.zeros(3)) == 0.0
    assert 0.0 < smoothness_risk(forward, turn) <= 1.0


def test_front_obstacle_distance_filters_side_and_vertical_hits():
    sensor = torch.zeros(3)
    hits = torch.tensor(
        [
            [0.6, 0.0, 0.0],
            [0.2, 1.0, 0.0],
            [0.2, 0.0, -1.0],
            [float("inf"), 0.0, 0.0],
        ]
    )
    identity_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert front_obstacle_distance(sensor, hits, identity_wxyz) == pytest.approx(0.6)


def test_combined_step_risk_uses_canonical_weights():
    components = {
        "contact": 1.0,
        "tilt": 0.0,
        "near_obstacle": 0.0,
        "blocked": 0.0,
        "speed_near": 0.0,
        "smoothness": 0.0,
    }
    weights = {
        "contact": 0.2,
        "tilt": 0.2,
        "near_obstacle": 0.25,
        "blocked": 0.2,
        "speed_near": 0.1,
        "smoothness": 0.05,
    }
    assert combined_step_risk(components, weights) == pytest.approx(0.2)
