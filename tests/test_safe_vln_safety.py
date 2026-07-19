import math

import pytest
import torch

from safe_vln.safety import (
    BlockedDetector,
    is_unsafe_contact_body,
    orientation_angle,
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
