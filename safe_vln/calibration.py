"""Fit bounded soft-risk thresholds from Go2 calibration diagnostics."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .objective import default_cost_profile, validate_cost_profile


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a quantile from no samples")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


def read_calibration_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path).expanduser()
    files = (
        sorted(input_path.glob("*.jsonl"))
        if input_path.is_dir()
        else [input_path]
    )
    records: list[dict[str, Any]] = []
    for file_path in files:
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError(f"unable to read calibration records: {file_path}") from error
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid calibration JSON at {file_path}:{line_number}"
                ) from error
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"calibration record at {file_path}:{line_number} must be an object"
                )
            records.append(dict(record))
    if not records:
        raise ValueError("no Safe-VLN calibration records were found")
    return records


def fit_cost_profile(
    records: Iterable[Mapping[str, Any]],
    *,
    calibration_episodes: int = 80,
    minimum_recorded_episodes: int = 20,
) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    episode_ids = {str(row.get("episode_id")) for row in rows if row.get("episode_id") is not None}
    if len(episode_ids) < minimum_recorded_episodes:
        raise ValueError(
            "calibration requires sensor records from at least "
            f"{minimum_recorded_episodes} episodes, found {len(episode_ids)}"
        )
    valid_ranges = [
        float(row["front_obstacle_distance_m"])
        for row in rows
        if row.get("front_obstacle_distance_m") is not None
    ]
    if len(valid_ranges) < 100:
        raise ValueError(
            "calibration requires at least 100 valid forward RayCaster samples"
        )
    non_hard = [row for row in rows if not bool(row.get("hard_violation", False))]
    if not non_hard:
        raise ValueError("calibration contains no non-hard safety samples")

    contact = [float(row["max_unsafe_contact_force"]) for row in non_hard]
    tilt = [float(row["orientation_angle"]) for row in non_hard]
    speed = [float(row["planar_speed_mps"]) for row in rows]

    profile = deepcopy(default_cost_profile())
    soft = profile["soft_thresholds"]
    soft["contact_force_n"] = _clip(_quantile(contact, 0.95), 0.1, 0.5)
    soft["orientation_rad"] = _clip(_quantile(tilt, 0.95), 0.20, 0.50)
    soft["near_critical_m"] = _clip(_quantile(valid_ranges, 0.01), 0.20, 0.35)
    safe_distance = _clip(_quantile(valid_ranges, 0.10), 0.50, 1.00)
    soft["near_safe_m"] = max(
        safe_distance, soft["near_critical_m"] + 0.20
    )
    soft["planar_speed_scale_mps"] = _clip(_quantile(speed, 0.95), 0.3, 0.8)
    profile["calibration"] = {
        "method": "bounded_quantiles",
        "episodes": int(calibration_episodes),
        "episodes_with_sensor_records": len(episode_ids),
        "records": len(rows),
        "valid_forward_ranges": len(valid_ranges),
        "quantiles": {
            "contact_force_non_hard_p95": _quantile(contact, 0.95),
            "orientation_non_hard_p95": _quantile(tilt, 0.95),
            "front_range_p01": _quantile(valid_ranges, 0.01),
            "front_range_p10": _quantile(valid_ranges, 0.10),
            "planar_speed_p95": _quantile(speed, 0.95),
        },
    }
    profile.pop("fingerprint", None)
    return validate_cost_profile(profile)


__all__ = ["fit_cost_profile", "read_calibration_records"]
