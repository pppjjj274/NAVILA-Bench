"""
Command Mapper — converts a structured mid-level action into low-level
velocity commands that the policy / simulation can consume.
"""

import math


def map_action_to_command(parsed_action: dict) -> dict:
    """Map a parsed mid-level action to low-level velocity commands.

    Args:
        parsed_action: dict with "action" (str) and "value" (float),
                       as returned by action_parser.parse_action().

    Returns:
        dict with keys "vx", "vy", "wz", "duration".
    """
    action = parsed_action.get("action", "STOP")
    value = parsed_action.get("value", 0.0)

    if action == "MOVE_FORWARD":
        vx = 0.5
        vy = 0.0
        wz = 0.0
        if vx == 0:
            duration = 0.0
        else:
            duration = value / vx
        return {"vx": vx, "vy": vy, "wz": wz, "duration": duration}

    elif action == "TURN_LEFT":
        wz = math.pi / 6.0  # positive angular velocity (left turn)
        angle_rad = math.radians(value)
        duration = angle_rad / abs(wz)
        return {"vx": 0.0, "vy": 0.0, "wz": wz, "duration": duration}

    elif action == "TURN_RIGHT":
        wz = -math.pi / 6.0  # negative angular velocity (right turn)
        angle_rad = math.radians(value)
        duration = angle_rad / abs(wz)
        return {"vx": 0.0, "vy": 0.0, "wz": wz, "duration": duration}

    elif action == "STOP":
        return {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 1.0}

    else:
        # Unknown action — fallback to STOP
        return {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 1.0}
