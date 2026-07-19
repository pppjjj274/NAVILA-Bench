"""Pure Go2 safety predicates shared by Isaac evaluation and unit tests."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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
