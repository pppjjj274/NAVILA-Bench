"""Pure Go2 safety predicates shared by Isaac evaluation and unit tests."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import torch


_UNSAFE_BODY_SUFFIXES = ("_hip", "_thigh", "_calf")


def is_unsafe_contact_body(name: str) -> bool:
    """Return whether contacts on this Go2 body count as unsafe."""

    return name == "base" or name.endswith(_UNSAFE_BODY_SUFFIXES)


def unsafe_contact_diagnostics(
    body_names: Sequence[str],
    net_forces_w_history: torch.Tensor,
    threshold: float,
) -> tuple[bool, dict[str, float], float]:
    """Find unsafe non-foot contacts from a single environment's force history."""

    if threshold <= 0:
        raise ValueError("contact threshold must be positive")
    if net_forces_w_history.ndim != 3 or net_forces_w_history.shape[-1] != 3:
        raise ValueError("contact force history must have shape [history, bodies, 3]")
    if net_forces_w_history.shape[1] != len(body_names):
        raise ValueError("contact force body dimension does not match body_names")

    unsafe_ids = [
        index for index, name in enumerate(body_names) if is_unsafe_contact_body(name)
    ]
    if not unsafe_ids:
        raise RuntimeError("Go2 contact sensor exposes no base/hip/thigh/calf bodies")

    force_norms = torch.linalg.vector_norm(
        net_forces_w_history[:, unsafe_ids], dim=-1
    )
    peak_forces = force_norms.amax(dim=0)
    triggered = {
        body_names[body_id]: float(peak_forces[index].item())
        for index, body_id in enumerate(unsafe_ids)
        if peak_forces[index] > threshold
    }
    return bool(triggered), triggered, float(peak_forces.max().item())


def orientation_angle(projected_gravity_b: torch.Tensor) -> float:
    """Compute absolute tilt angle from projected gravity in the robot frame."""

    if projected_gravity_b.shape[-1] != 3:
        raise ValueError("projected gravity must end with three coordinates")
    gravity_z = torch.clamp(-projected_gravity_b[..., 2], -1.0, 1.0)
    return float(torch.acos(gravity_z).abs().max().item())


@dataclass(frozen=True)
class BlockedStatus:
    blocked: bool
    observed_steps: int
    displacement: float


@dataclass(frozen=True)
class TurnExecutionStatus:
    """Whether a commanded turn produced the expected yaw change."""

    active: bool
    blocked: bool
    observed_steps: int
    expected_angle: float
    achieved_angle: float
    execution_ratio: float
    signed_yaw_delta: float = 0.0
    expected_yaw_sign: int = 0
    direction_mismatch: bool = False


def wrap_angle_radians(value: float) -> float:
    """Normalize an angle to ``[-pi, pi)``."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion_wxyz(quaternion: torch.Tensor | Sequence[float]) -> float:
    """Return planar yaw from an Isaac ``[w, x, y, z]`` quaternion."""
    values = torch.as_tensor(quaternion, dtype=torch.float64).reshape(-1)
    if values.numel() != 4 or not torch.isfinite(values).all():
        raise ValueError("robot quaternion must contain four finite values")
    norm = float(torch.linalg.vector_norm(values).item())
    if norm <= 0.0:
        raise ValueError("robot quaternion must be non-zero")
    w, x, y, z = (float(item) / norm for item in values.tolist())
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def turn_execution_diagnostics(
    command: torch.Tensor | Sequence[float],
    *,
    start_yaw: float | None,
    current_yaw: float | None,
    observed_steps: int,
    expected_steps: int,
    expected_angle: float,
    min_expected_angle: float = 0.18,
    minimum_execution_ratio: float = 0.25,
) -> TurnExecutionStatus:
    """Measure closed-loop turn execution without changing the 10 actions.

    A turn is considered blocked only after its requested duration has elapsed.
    The ratio is scaled by the requested angle so the 15/30/45 degree actions
    use the same proportional criterion. ``min_expected_angle`` is only a
    gate for whether to check a turn; it is not an absolute achieved-angle
    requirement. This avoids labeling a partially executed small turn as
    blocked merely because it falls below a fixed 0.18 radian floor.
    """
    command_tensor = torch.as_tensor(command, dtype=torch.float32).reshape(-1)
    if command_tensor.numel() < 3:
        raise ValueError("velocity command must contain vx, vy and wz")
    yaw_rate = float(command_tensor[2].item())
    expected = float(expected_angle)
    if not all(math.isfinite(value) for value in (expected, min_expected_angle, minimum_execution_ratio)):
        raise ValueError("turn execution thresholds must be finite")
    if expected < 0.0 or min_expected_angle < 0.0 or not 0.0 <= minimum_execution_ratio <= 1.0:
        raise ValueError("turn execution thresholds are out of range")
    is_turn = abs(yaw_rate) > 1e-6 and expected_steps > 0 and start_yaw is not None
    if not is_turn or current_yaw is None:
        return TurnExecutionStatus(
            active=False,
            blocked=False,
            observed_steps=int(observed_steps),
            expected_angle=expected,
            achieved_angle=0.0,
            execution_ratio=0.0,
        )
    signed_delta = wrap_angle_radians(float(current_yaw) - float(start_yaw))
    expected_sign = 1 if yaw_rate > 0.0 else -1
    aligned_delta = signed_delta * expected_sign
    achieved = max(0.0, aligned_delta)
    opposite_angle = max(0.0, -aligned_delta)
    ratio = achieved / max(expected, 1e-8)
    active = int(observed_steps) < int(expected_steps)
    direction_mismatch = bool(not active and opposite_angle > 0.02)
    blocked = bool(
        not active
        and expected >= min_expected_angle
        and achieved < expected * minimum_execution_ratio
    )
    return TurnExecutionStatus(
        active=active,
        blocked=blocked,
        observed_steps=int(observed_steps),
        expected_angle=expected,
        achieved_angle=achieved,
        execution_ratio=ratio,
        signed_yaw_delta=signed_delta,
        expected_yaw_sign=expected_sign,
        direction_mismatch=direction_mismatch,
    )


class BlockedDetector:
    """Detect insufficient planar displacement during sustained forward commands."""

    def __init__(
        self,
        *,
        window_steps: int,
        min_displacement: float,
        forward_threshold: float = 0.05,
    ) -> None:
        if window_steps <= 0:
            raise ValueError("blocked window_steps must be positive")
        if min_displacement <= 0:
            raise ValueError("blocked min_displacement must be positive")
        if forward_threshold < 0:
            raise ValueError("forward_threshold must be non-negative")
        self.window_steps = int(window_steps)
        self.min_displacement = float(min_displacement)
        self.forward_threshold = float(forward_threshold)
        self._positions: deque[torch.Tensor] = deque(maxlen=self.window_steps + 1)

    def reset(self) -> None:
        self._positions.clear()

    def update(
        self,
        command: torch.Tensor,
        previous_position: torch.Tensor,
        current_position: torch.Tensor,
    ) -> BlockedStatus:
        command_x = float(torch.as_tensor(command).reshape(-1)[0].item())
        if command_x <= self.forward_threshold:
            self.reset()
            return BlockedStatus(False, 0, 0.0)

        previous_xy = torch.as_tensor(previous_position)[:2].detach().clone()
        current_xy = torch.as_tensor(current_position)[:2].detach().clone()
        if not self._positions:
            self._positions.append(previous_xy)
        self._positions.append(current_xy)

        observed_steps = len(self._positions) - 1
        displacement = float(
            torch.linalg.vector_norm(self._positions[-1] - self._positions[0]).item()
        )
        blocked = (
            observed_steps >= self.window_steps
            and displacement < self.min_displacement
        )
        return BlockedStatus(blocked, observed_steps, displacement)


def linear_risk(value: float, soft_threshold: float, hard_threshold: float) -> float:
    """Map a scalar measurement to a bounded soft-to-hard risk."""
    value = float(value)
    soft_threshold = float(soft_threshold)
    hard_threshold = float(hard_threshold)
    if not math.isfinite(value):
        raise ValueError("risk measurement must be finite")
    if not 0 <= soft_threshold < hard_threshold:
        raise ValueError("risk thresholds must satisfy 0 <= soft < hard")
    return min(1.0, max(0.0, (value - soft_threshold) / (hard_threshold - soft_threshold)))


def inverse_distance_risk(distance: float, critical_distance: float, safe_distance: float) -> float:
    """Return one near an obstacle and zero at or beyond the safe distance."""
    distance = float(distance)
    critical_distance = float(critical_distance)
    safe_distance = float(safe_distance)
    if not math.isfinite(distance):
        raise ValueError("obstacle distance must be finite")
    if not 0 < critical_distance < safe_distance:
        raise ValueError("distance thresholds must satisfy 0 < critical < safe")
    return min(
        1.0,
        max(
            0.0,
            (safe_distance - distance) / (safe_distance - critical_distance),
        ),
    )


def blocked_progress_risk(
    status: BlockedStatus,
    *,
    window_steps: int,
    min_displacement: float,
) -> float:
    """Produce a continuous precursor that reaches one at a hard blocked event."""
    if window_steps <= 0 or min_displacement <= 0:
        raise ValueError("blocked risk scales must be positive")
    if status.observed_steps <= 0:
        return 0.0
    window_fraction = min(1.0, status.observed_steps / float(window_steps))
    expected_minimum = min_displacement * window_fraction
    progress_deficit = 1.0 - min(
        1.0, max(0.0, status.displacement / max(expected_minimum, 1e-8))
    )
    return window_fraction * progress_deficit


def smoothness_risk(
    previous_command: torch.Tensor | Sequence[float] | None,
    current_command: torch.Tensor | Sequence[float],
) -> float:
    """Normalize a high-level ``[vx, vy, wz]`` command discontinuity."""
    if previous_command is None:
        return 0.0
    previous = torch.as_tensor(previous_command, dtype=torch.float32).reshape(-1)
    current = torch.as_tensor(current_command, dtype=torch.float32).reshape(-1)
    if previous.numel() < 3 or current.numel() < 3:
        raise ValueError("velocity commands must contain vx, vy and wz")
    if torch.allclose(current[:3], torch.zeros(3), atol=1e-8):
        return 0.0
    scales = torch.tensor([0.5, 0.5, math.pi / 6.0])
    normalized = (current[:3].cpu() - previous[:3].cpu()) / scales
    return min(1.0, float(torch.linalg.vector_norm(normalized).item() / math.sqrt(3.0)))


def _quat_rotate_inverse_wxyz(quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    quaternion = torch.as_tensor(quaternion, dtype=vectors.dtype, device=vectors.device).reshape(4)
    norm = torch.linalg.vector_norm(quaternion)
    if float(norm.item()) <= 0:
        raise ValueError("robot quaternion must be non-zero")
    quaternion = quaternion / norm
    scalar = quaternion[0]
    axis = quaternion[1:]
    axis_batch = axis.expand_as(vectors)
    first_cross = torch.linalg.cross(axis_batch, vectors, dim=-1)
    second_cross = torch.linalg.cross(axis_batch, first_cross, dim=-1)
    return vectors - 2.0 * scalar * first_cross + 2.0 * second_cross


def front_obstacle_distance(
    sensor_position_w: torch.Tensor,
    ray_hits_w: torch.Tensor,
    robot_quaternion_w: torch.Tensor,
    *,
    horizontal_half_angle_deg: float = 45.0,
    vertical_half_angle_deg: float = 20.0,
) -> float | None:
    """Return the closest finite RayCaster hit inside a robot-forward sector."""
    position = torch.as_tensor(sensor_position_w)
    hits = torch.as_tensor(ray_hits_w, device=position.device, dtype=position.dtype)
    if position.shape[-1] != 3 or hits.ndim != 2 or hits.shape[-1] != 3:
        raise ValueError("RayCaster positions must have shapes [3] and [rays, 3]")
    vectors_w = hits - position.reshape(1, 3)
    finite = torch.isfinite(vectors_w).all(dim=-1)
    if not finite.any():
        return None
    vectors_b = _quat_rotate_inverse_wxyz(
        torch.as_tensor(robot_quaternion_w, device=hits.device, dtype=hits.dtype),
        vectors_w[finite],
    )
    horizontal = torch.linalg.vector_norm(vectors_b[:, :2], dim=-1)
    azimuth = torch.atan2(vectors_b[:, 1], vectors_b[:, 0]).abs()
    elevation = torch.atan2(vectors_b[:, 2], horizontal.clamp_min(1e-8)).abs()
    selected = (
        (vectors_b[:, 0] > 0)
        & (azimuth <= math.radians(horizontal_half_angle_deg))
        & (elevation <= math.radians(vertical_half_angle_deg))
    )
    if not selected.any():
        return None
    distances = torch.linalg.vector_norm(vectors_b[selected], dim=-1)
    positive = distances[distances > 1e-6]
    return None if positive.numel() == 0 else float(positive.min().item())


def combined_step_risk(components: dict[str, float], weights: dict[str, float]) -> float:
    expected = {
        "contact",
        "tilt",
        "near_obstacle",
        "blocked",
        "speed_near",
        "smoothness",
    }
    if set(components) != expected or set(weights) != expected:
        raise ValueError("step risk components and weights must use the canonical keys")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-6:
        raise ValueError("step risk weights must sum to one")
    values = {
        key: min(1.0, max(0.0, float(value))) for key, value in components.items()
    }
    return min(1.0, max(0.0, sum(values[key] * float(weights[key]) for key in expected)))


__all__ = [
    "BlockedDetector",
    "BlockedStatus",
    "TurnExecutionStatus",
    "blocked_progress_risk",
    "combined_step_risk",
    "front_obstacle_distance",
    "inverse_distance_risk",
    "is_unsafe_contact_body",
    "linear_risk",
    "orientation_angle",
    "smoothness_risk",
    "turn_execution_diagnostics",
    "unsafe_contact_diagnostics",
    "wrap_angle_radians",
    "yaw_from_quaternion_wxyz",
]
