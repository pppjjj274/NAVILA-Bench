"""
Action Parser — converts VLM raw text output into a structured action dict.

Supports:
    MOVE_FORWARD <distance>
    TURN_LEFT <angle>
    TURN_RIGHT <angle>
    STOP

Any unrecognised or malformed output falls back to STOP with value 0.0.
"""

import re


def parse_action(text: str) -> dict:
    """Parse a raw VLM action string into a structured action dictionary.

    Args:
        text: Raw string from the VLM, e.g. "MOVE_FORWARD 0.5".

    Returns:
        dict with keys "action" (str) and "value" (float).
        Falls back to {"action": "STOP", "value": 0.0} on any parse error.
    """
    if not isinstance(text, str) or not text.strip():
        return _fallback()

    cleaned = text.strip().upper()

    # --- STOP ---
    if cleaned.startswith("STOP"):
        return {"action": "STOP", "value": 0.0}

    # --- MOVE_FORWARD <distance> ---
    m = re.match(r"^MOVE_FORWARD\s+([-+]?(?:\d+\.?\d*|\.\d+))$", cleaned)
    if m:
        try:
            dist = float(m.group(1))
            return {"action": "MOVE_FORWARD", "value": dist}
        except (ValueError, OverflowError):
            return _fallback()

    # --- TURN_LEFT <angle> ---
    m = re.match(r"^TURN_LEFT\s+([-+]?(?:\d+\.?\d*|\.\d+))$", cleaned)
    if m:
        try:
            angle = float(m.group(1))
            return {"action": "TURN_LEFT", "value": angle}
        except (ValueError, OverflowError):
            return _fallback()

    # --- TURN_RIGHT <angle> ---
    m = re.match(r"^TURN_RIGHT\s+([-+]?(?:\d+\.?\d*|\.\d+))$", cleaned)
    if m:
        try:
            angle = float(m.group(1))
            return {"action": "TURN_RIGHT", "value": angle}
        except (ValueError, OverflowError):
            return _fallback()

    return _fallback()


def _fallback() -> dict:
    return {"action": "STOP", "value": 0.0}
