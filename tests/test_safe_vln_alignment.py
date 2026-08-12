import math

import pytest

from safe_vln.alignment import (
    planar_aligned_settled_root_pose,
    refresh_attached_cameras,
    requires_strict_start_alignment,
)

from safe_vln.safety import yaw_from_quaternion_wxyz


class _FakeCamera:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def reset(self):
        self.events.append((self.name, "reset"))

    def update(self, dt, *, force_recompute=False):
        self.events.append((self.name, "update", dt, force_recompute))


class _FakeSimulation:
    def __init__(self, events):
        self.events = events

    def render(self):
        self.events.append(("simulation", "render"))


class _FakeScene:
    def __init__(self, sensors):
        self.sensors = sensors


def test_strict_start_alignment_covers_native_and_live_but_not_replay():
    assert requires_strict_start_alignment(safe_vln=True, safe_replay=False)
    assert not requires_strict_start_alignment(safe_vln=True, safe_replay=True)
    assert not requires_strict_start_alignment(safe_vln=False, safe_replay=False)


def test_planar_alignment_preserves_settled_height_and_restores_xy_yaw():
    settled_yaw = 0.2
    target_yaw = 1.0
    pose = planar_aligned_settled_root_pose(
        [1.2, 2.3, 0.31],
        [
            math.cos(settled_yaw / 2.0),
            0.0,
            0.0,
            math.sin(settled_yaw / 2.0),
        ],
        [1.0, 2.0, 0.17],
        [
            math.cos(target_yaw / 2.0),
            0.0,
            0.0,
            math.sin(target_yaw / 2.0),
        ],
    )

    assert pose[:3] == (1.0, 2.0, 0.31)
    assert yaw_from_quaternion_wxyz(pose[3:]) == pytest.approx(target_yaw)


def test_refresh_attached_cameras_renders_before_forced_buffer_update():
    events = []
    scene = _FakeScene(
        {
            "rgbd_camera": _FakeCamera("rgbd_camera", events),
            "viz_rgb_camera": _FakeCamera("viz_rgb_camera", events),
            "contact_forces": object(),
        }
    )

    refreshed = refresh_attached_cameras(scene, _FakeSimulation(events))

    assert refreshed == ("rgbd_camera", "viz_rgb_camera")
    assert events == [
        ("rgbd_camera", "reset"),
        ("viz_rgb_camera", "reset"),
        ("simulation", "render"),
        ("simulation", "render"),
        ("rgbd_camera", "update", 0.0, True),
        ("viz_rgb_camera", "update", 0.0, True),
    ]


def test_refresh_attached_cameras_is_noop_without_native_cameras():
    events = []
    refreshed = refresh_attached_cameras(
        _FakeScene({"contact_forces": object()}),
        _FakeSimulation(events),
    )
    assert refreshed == ()
    assert events == []
