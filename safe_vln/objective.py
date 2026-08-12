"""Versioned Safe-VLN reward and cost objective definitions."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .actions import ACTIONS


SCHEMA_VERSION = "safe-vln-go2-v4"
LEGACY_OBJECTIVE_SCHEMA_VERSIONS = frozenset(
    {"safe-vln-go2-v2", "safe-vln-go2-v3"}
)
COST_PROFILE_VERSION = "safe-vln-go2-cost-profile-v1"
HARD_SAFETY_EVENTS = ("unsafe_contact", "fall", "blocked")
COST_NORMALIZATION = "cumulative_episode_sum"

REPLAY_REWARD_CONFIG = {
    "type": "graded_oracle_action",
    "exact": 1.0,
    "adjacent_magnitude": 0.5,
    "two_step_magnitude": 0.25,
    "different_family": 0.0,
    "wrong_stop": -1.0,
    "missed_stop": -0.5,
    "invalid_action": -1.0,
}

ONLINE_REWARD_CONFIG = {
    "type": "physical_navigation",
    "progress_scale": 1.0,
    "macro_step_penalty": -0.01,
    "success_reward": 10.0,
    "failed_stop_penalty": -1.0,
    "missed_stop_penalty": -0.5,
    "missed_stop_patience": 3,
    "subtract_safety_cost": False,
}

DEFAULT_COST_PROFILE = {
    "profile_version": COST_PROFILE_VERSION,
    "hard_thresholds": {
        "contact_force_n": 1.0,
        "orientation_rad": 0.8,
        "blocked_seconds": 2.0,
        "blocked_distance_m": 0.10,
        "forward_command_mps": 0.05,
    },
    "soft_thresholds": {
        "contact_force_n": 0.5,
        "orientation_rad": 0.35,
        "near_critical_m": 0.25,
        "near_safe_m": 0.8,
        "planar_speed_scale_mps": 0.5,
    },
    "ray_sector": {
        "horizontal_half_angle_deg": 45.0,
        "vertical_half_angle_deg": 20.0,
    },
    "risk_weights": {
        "contact": 0.20,
        "tilt": 0.20,
        "near_obstacle": 0.25,
        "blocked": 0.20,
        "speed_near": 0.10,
        "smoothness": 0.05,
    },
    "macro_aggregation": {
        "mean_scale": 0.05,
        "peak_scale": 0.05,
        "max_dense_cost": 0.10,
    },
    # SafeVLA constrains expected cumulative episode cost. Any hard event has
    # unit cost, so the default limit rejects collision/fall/blocked episodes
    # while leaving room for calibrated bounded dense-risk accumulation.
    "cost_limit": 0.25,
    "calibration": {
        "method": "defaults",
        "episodes": 0,
    },
}


def canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_cost_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(profile))
    if result.get("profile_version") != COST_PROFILE_VERSION:
        raise ValueError(
            f"unsupported cost profile version: {result.get('profile_version')!r}"
        )
    hard = result["hard_thresholds"]
    soft = result["soft_thresholds"]
    ray_sector = result["ray_sector"]
    weights = result["risk_weights"]
    aggregation = result["macro_aggregation"]
    expected_keys = {
        "hard_thresholds": {
            "contact_force_n",
            "orientation_rad",
            "blocked_seconds",
            "blocked_distance_m",
            "forward_command_mps",
        },
        "soft_thresholds": {
            "contact_force_n",
            "orientation_rad",
            "near_critical_m",
            "near_safe_m",
            "planar_speed_scale_mps",
        },
        "ray_sector": {
            "horizontal_half_angle_deg",
            "vertical_half_angle_deg",
        },
        "risk_weights": {
            "contact",
            "tilt",
            "near_obstacle",
            "blocked",
            "speed_near",
            "smoothness",
        },
        "macro_aggregation": {
            "mean_scale",
            "peak_scale",
            "max_dense_cost",
        },
    }
    for group_name, group in (
        ("hard_thresholds", hard),
        ("soft_thresholds", soft),
        ("ray_sector", ray_sector),
        ("risk_weights", weights),
        ("macro_aggregation", aggregation),
    ):
        if set(group) != expected_keys[group_name]:
            raise ValueError(
                f"{group_name} must contain exactly "
                f"{sorted(expected_keys[group_name])}"
            )
        for key, value in group.items():
            group[key] = _finite(value, f"{group_name}.{key}")
    result["cost_limit"] = _finite(result["cost_limit"], "cost_limit")
    if not 0 <= soft["contact_force_n"] < hard["contact_force_n"]:
        raise ValueError("contact soft threshold must be non-negative and below hard")
    if not 0 <= soft["orientation_rad"] < hard["orientation_rad"]:
        raise ValueError("orientation soft threshold must be non-negative and below hard")
    if not 0 < soft["near_critical_m"] < soft["near_safe_m"]:
        raise ValueError("near obstacle thresholds must satisfy 0 < critical < safe")
    if soft["planar_speed_scale_mps"] <= 0:
        raise ValueError("planar speed scale must be positive")
    if hard["blocked_seconds"] <= 0 or hard["blocked_distance_m"] <= 0:
        raise ValueError("blocked hard thresholds must be positive")
    if hard["forward_command_mps"] < 0:
        raise ValueError("forward command threshold must be non-negative")
    if not 0 < ray_sector["horizontal_half_angle_deg"] <= 180:
        raise ValueError("horizontal ray half-angle must be in (0, 180]")
    if not 0 < ray_sector["vertical_half_angle_deg"] <= 90:
        raise ValueError("vertical ray half-angle must be in (0, 90]")
    if any(value < 0 for value in weights.values()):
        raise ValueError("risk weights must be non-negative")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("risk weights must sum to one")
    if aggregation["mean_scale"] < 0 or aggregation["peak_scale"] < 0:
        raise ValueError("macro aggregation scales must be non-negative")
    expected_max = aggregation["mean_scale"] + aggregation["peak_scale"]
    if abs(expected_max - aggregation["max_dense_cost"]) > 1e-6:
        raise ValueError("max_dense_cost must equal mean_scale + peak_scale")
    if result["cost_limit"] < 0:
        raise ValueError("cost_limit must be non-negative")
    fingerprint_payload = deepcopy(result)
    fingerprint_payload.pop("fingerprint", None)
    expected_fingerprint = canonical_fingerprint(fingerprint_payload)
    supplied = result.get("fingerprint")
    if supplied is not None and supplied != expected_fingerprint:
        raise ValueError("cost profile fingerprint does not match its contents")
    result["fingerprint"] = expected_fingerprint
    return result


def default_cost_profile() -> dict[str, Any]:
    return validate_cost_profile(DEFAULT_COST_PROFILE)


def load_cost_profile(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return default_cost_profile()
    profile_path = Path(path).expanduser()
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read Safe-VLN cost profile: {profile_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Safe-VLN cost profile must be a JSON object")
    return validate_cost_profile(payload)


def save_cost_profile(profile: Mapping[str, Any], path: str | Path) -> Path:
    resolved = validate_cost_profile(profile)
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(output)
    return output


def build_objective_config(cost_profile: Mapping[str, Any]) -> dict[str, Any]:
    profile = validate_cost_profile(cost_profile)
    result = {
        "schema_version": SCHEMA_VERSION,
        "hard_safety_events": list(HARD_SAFETY_EVENTS),
        "cost_normalization": COST_NORMALIZATION,
        "replay_reward": deepcopy(REPLAY_REWARD_CONFIG),
        "online_reward": deepcopy(ONLINE_REWARD_CONFIG),
        "cost_profile": profile,
    }
    return validate_objective_config(result)


def validate_objective_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(config))
    if result.get("schema_version") not in {
        SCHEMA_VERSION,
        *LEGACY_OBJECTIVE_SCHEMA_VERSIONS,
    }:
        raise ValueError(
            f"unsupported Safe-VLN objective schema: "
            f"{result.get('schema_version')!r}"
        )
    result["cost_profile"] = validate_cost_profile(result["cost_profile"])
    if result.get("schema_version") == SCHEMA_VERSION:
        if result.get("hard_safety_events") != list(HARD_SAFETY_EVENTS):
            raise ValueError(
                "Safe-VLN v4 hard_safety_events must be exactly "
                f"{list(HARD_SAFETY_EVENTS)}"
            )
        if result.get("cost_normalization") != COST_NORMALIZATION:
            raise ValueError(
                "Safe-VLN v4 cost_normalization must be "
                f"{COST_NORMALIZATION!r}"
            )
    if not isinstance(result.get("replay_reward"), Mapping):
        raise ValueError("Safe-VLN replay_reward must be an object")
    if not isinstance(result.get("online_reward"), Mapping):
        raise ValueError("Safe-VLN online_reward must be an object")
    fingerprint_payload = deepcopy(result)
    fingerprint_payload.pop("fingerprint", None)
    expected_fingerprint = canonical_fingerprint(fingerprint_payload)
    supplied = result.get("fingerprint")
    if supplied is not None and supplied != expected_fingerprint:
        raise ValueError("objective fingerprint does not match its contents")
    result["fingerprint"] = expected_fingerprint
    return result


_ACTION_GROUPS = {
    **{action_id: ("left", action_id) for action_id in range(0, 3)},
    **{action_id: ("right", action_id - 3) for action_id in range(3, 6)},
    **{action_id: ("forward", action_id - 6) for action_id in range(6, 9)},
}


def graded_oracle_reward(
    predicted_action_id: int,
    oracle_action_id: int,
    *,
    invalid_action: bool = False,
) -> tuple[float, dict[str, float]]:
    if invalid_action:
        value = REPLAY_REWARD_CONFIG["invalid_action"]
        return value, {"invalid_action": value}
    if predicted_action_id == oracle_action_id:
        value = REPLAY_REWARD_CONFIG["exact"]
        return value, {"oracle_exact": value}
    if predicted_action_id == 9:
        value = REPLAY_REWARD_CONFIG["wrong_stop"]
        return value, {"wrong_stop": value}
    if oracle_action_id == 9:
        value = REPLAY_REWARD_CONFIG["missed_stop"]
        return value, {"missed_stop": value}
    predicted_group = _ACTION_GROUPS.get(predicted_action_id)
    oracle_group = _ACTION_GROUPS.get(oracle_action_id)
    if predicted_group is not None and predicted_group[0] == oracle_group[0]:
        magnitude_delta = abs(predicted_group[1] - oracle_group[1])
        if magnitude_delta == 1:
            value = REPLAY_REWARD_CONFIG["adjacent_magnitude"]
            return value, {"oracle_adjacent_magnitude": value}
        if magnitude_delta == 2:
            value = REPLAY_REWARD_CONFIG["two_step_magnitude"]
            return value, {"oracle_two_step_magnitude": value}
    value = REPLAY_REWARD_CONFIG["different_family"]
    return value, {"oracle_different_family": value}


assert len(ACTIONS) == 10


__all__ = [
    "COST_PROFILE_VERSION",
    "COST_NORMALIZATION",
    "DEFAULT_COST_PROFILE",
    "HARD_SAFETY_EVENTS",
    "LEGACY_OBJECTIVE_SCHEMA_VERSIONS",
    "ONLINE_REWARD_CONFIG",
    "REPLAY_REWARD_CONFIG",
    "SCHEMA_VERSION",
    "build_objective_config",
    "canonical_fingerprint",
    "default_cost_profile",
    "graded_oracle_reward",
    "load_cost_profile",
    "save_cost_profile",
    "validate_cost_profile",
    "validate_objective_config",
]
