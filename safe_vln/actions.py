"""Canonical macro action space used by the Go2 Safe-VLN policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class SafeAction:
    action_id: int
    text: str
    velocity_command: tuple[float, float, float]
    duration: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["velocity_command"] = list(self.velocity_command)
        return result


_TURN_RATE = math.pi / 6.0
ACTIONS: tuple[SafeAction, ...] = (
    SafeAction(0, "turn left 15 degrees", (0.0, 0.0, _TURN_RATE), 0.5),
    SafeAction(1, "turn left 30 degrees", (0.0, 0.0, _TURN_RATE), 1.0),
    SafeAction(2, "turn left 45 degrees", (0.0, 0.0, _TURN_RATE), 1.5),
    SafeAction(3, "turn right 15 degrees", (0.0, 0.0, -_TURN_RATE), 0.5),
    SafeAction(4, "turn right 30 degrees", (0.0, 0.0, -_TURN_RATE), 1.0),
    SafeAction(5, "turn right 45 degrees", (0.0, 0.0, -_TURN_RATE), 1.5),
    SafeAction(6, "move forward 25 centimeters", (0.5, 0.0, 0.0), 0.5),
    SafeAction(7, "move forward 50 centimeters", (0.5, 0.0, 0.0), 1.0),
    SafeAction(8, "move forward 75 centimeters", (0.5, 0.0, 0.0), 1.5),
    SafeAction(9, "stop", (0.0, 0.0, 0.0), 0.0),
)


def action_from_id(action_id: int) -> SafeAction:
    if isinstance(action_id, bool) or not isinstance(action_id, int):
        raise ValueError(f"action_id must be an integer, got {action_id!r}")
    if not 0 <= action_id < len(ACTIONS):
        raise ValueError(f"action_id must be in [0, {len(ACTIONS) - 1}], got {action_id}")
    return ACTIONS[action_id]


def action_from_text(text: str) -> tuple[SafeAction, bool]:
    """Parse legacy NaViLA text, returning ``(action, invalid_action)``."""
    if not isinstance(text, str):
        return ACTIONS[9], True

    normalized = " ".join(text.lower().strip().split())
    if "stop" in normalized:
        return ACTIONS[9], False

    match = re.search(r"\b(15|30|45)\b", normalized)
    angle = int(match.group(1)) if match else 15
    if "turn left" in normalized:
        return ACTIONS[{15: 0, 30: 1, 45: 2}[angle]], False
    if "turn right" in normalized:
        return ACTIONS[{15: 3, 30: 4, 45: 5}[angle]], False

    match = re.search(r"\b(25|50|75)\b", normalized)
    distance = int(match.group(1)) if match else 25
    if "move forward" in normalized or normalized.startswith("move"):
        return ACTIONS[{25: 6, 50: 7, 75: 8}[distance]], False
    return ACTIONS[9], True


def _finite_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_policy_response(response: Any) -> dict[str, Any]:
    """Normalize legacy text and structured Safe-VLN policy responses.

    Velocity and duration always come from the local canonical action table so
    an untrusted model server cannot alter the executable command.
    """
    structured = isinstance(response, Mapping)
    invalid_action = False

    if structured and "action_id" in response:
        try:
            action = action_from_id(int(response["action_id"]))
        except (TypeError, ValueError):
            action, invalid_action = ACTIONS[9], True
    else:
        text = response.get("action", "") if structured else response
        action, invalid_action = action_from_text(text)

    result = action.to_dict()
    result.update(
        {
            "invalid_action": invalid_action,
            "reward_value": _finite_optional_float(response.get("reward_value")) if structured else None,
            "cost_value": _finite_optional_float(response.get("cost_value")) if structured else None,
            "log_prob": _finite_optional_float(response.get("log_prob")) if structured else None,
            "policy_version": response.get("policy_version") if structured else None,
            "decision_id": response.get("decision_id") if structured else None,
        }
    )
    return result
