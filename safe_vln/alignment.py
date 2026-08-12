"""Small helpers for synchronizing strict Safe-VLN observations."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any


NATIVE_CAMERA_SENSOR_NAMES = ("rgbd_camera", "viz_rgb_camera")


def planar_aligned_settled_root_pose(
    settled_position,
    settled_rotation_wxyz,
    start_position,
    start_rotation_wxyz,
) -> tuple[float, ...]:
    """Restore official x/y/yaw while retaining the settled support pose.

    Go2 is spawned above the floor and needs the excluded warm-up to settle its
    base height and small roll/pitch. Restoring the full spawn pose afterwards
    makes the first recorded action absorb another fall. A world-yaw correction
    preserves the settled tilt while returning the heading to the episode
    start.
    """

    vectors = (
        (settled_position, 3, "settled_position"),
        (settled_rotation_wxyz, 4, "settled_rotation_wxyz"),
        (start_position, 3, "start_position"),
        (start_rotation_wxyz, 4, "start_rotation_wxyz"),
    )
    resolved = {}
    for value, length, name in vectors:
        try:
            items = tuple(float(item) for item in value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain {length} numbers") from error
        if len(items) != length or not all(math.isfinite(item) for item in items):
            raise ValueError(f"{name} must contain {length} finite numbers")
        resolved[name] = items

    def normalized(quaternion, name):
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm <= 0.0:
            raise ValueError(f"{name} must be non-zero")
        return tuple(value / norm for value in quaternion)

    def yaw(quaternion):
        w, x, y, z = quaternion
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    settled_rotation = normalized(
        resolved["settled_rotation_wxyz"], "settled_rotation_wxyz"
    )
    target_rotation = normalized(
        resolved["start_rotation_wxyz"], "start_rotation_wxyz"
    )
    delta = yaw(target_rotation) - yaw(settled_rotation)
    delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
    cosine = math.cos(delta / 2.0)
    sine = math.sin(delta / 2.0)
    w, x, y, z = settled_rotation
    aligned_rotation = normalized(
        (
            cosine * w - sine * z,
            cosine * x - sine * y,
            cosine * y + sine * x,
            cosine * z + sine * w,
        ),
        "aligned_rotation_wxyz",
    )
    settled = resolved["settled_position"]
    start = resolved["start_position"]
    return (start[0], start[1], settled[2], *aligned_rotation)


def requires_strict_start_alignment(*, safe_vln: bool, safe_replay: bool) -> bool:
    """Return whether physics must be restored after excluded warm-up steps."""

    return bool(safe_vln and not safe_replay)


def refresh_attached_cameras(
    scene: Any,
    simulation: Any,
    *,
    sensor_names: Iterable[str] = NATIVE_CAMERA_SENSOR_NAMES,
) -> tuple[str, ...]:
    """Render and refresh attached camera buffers after a root-pose write.

    Writing an articulation root pose updates PhysX state but does not produce a
    new Replicator frame.  Reading the observation manager immediately after a
    teleport can therefore pair the new robot state with the previous RGB
    buffer.  Resetting only the present camera sensors, rendering twice (the
    same warm-up used by Isaac Lab's simulation reset), and forcing their
    buffers to recompute establishes a single post-teleport observation.
    """

    sensors = getattr(scene, "sensors", {})
    cameras = [
        (name, sensors[name])
        for name in sensor_names
        if name in sensors
    ]
    if not cameras:
        return ()

    for _, camera in cameras:
        camera.reset()
    simulation.render()
    simulation.render()
    for _, camera in cameras:
        camera.update(0.0, force_recompute=True)
    return tuple(name for name, _ in cameras)
