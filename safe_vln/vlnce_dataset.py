"""Validated conversion of official VLN-CE episodes to Isaac coordinates."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import gzip
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .replay import habitat_heading_to_isaac, habitat_position_to_isaac


ISAAC_VLNCE_FORMAT = "safe-vln-isaac-vlnce-v1"
ISAAC_COORDINATE_SYSTEM = "isaac_xyz_z_up_wxyz"
VLNCE_ACTION_IDS = frozenset(range(4))


def scene_name(scene_id: str) -> str:
    return os.path.splitext(os.path.basename(scene_id))[0]


def balanced_episode_ids(
    episodes: Sequence[Mapping[str, Any]], *, seed: int
) -> list[str]:
    """Return deterministic round-robin IDs spanning every scene first."""

    by_scene: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        by_scene[scene_name(str(episode["scene_id"]))].append(
            str(episode["episode_id"])
        )
    generator = random.Random(seed)
    for identifiers in by_scene.values():
        generator.shuffle(identifiers)
    scenes = sorted(by_scene)
    generator.shuffle(scenes)
    ordered: list[str] = []
    round_index = 0
    while True:
        added = False
        for scene in scenes:
            if round_index < len(by_scene[scene]):
                ordered.append(by_scene[scene][round_index])
                added = True
        if not added:
            return ordered
        round_index += 1


def _position(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a 3D position")
    try:
        converted = habitat_position_to_isaac(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}: {value!r}") from error
    return list(converted)


def _convert_episode(
    episode: Mapping[str, Any], gt: Mapping[str, Any]
) -> dict[str, Any]:
    raw_episode_id = episode.get("episode_id")
    episode_id = str(raw_episode_id) if raw_episode_id is not None else ""
    scene_id = episode.get("scene_id")
    if not episode_id or not isinstance(scene_id, str) or not scene_id:
        raise ValueError("VLN-CE episode requires an ID and scene_id")
    goals = episode.get("goals")
    reference_path = episode.get("reference_path")
    locations = gt.get("locations")
    actions = gt.get("actions")
    if not isinstance(goals, list) or len(goals) != 1:
        raise ValueError(f"VLN-CE episode {episode_id} must have one goal")
    if not isinstance(goals[0], Mapping):
        raise ValueError(f"VLN-CE episode {episode_id} has an invalid goal")
    if not isinstance(reference_path, list) or not reference_path:
        raise ValueError(f"VLN-CE episode {episode_id} has no reference path")
    if not isinstance(locations, list) or not locations:
        raise ValueError(f"VLN-CE episode {episode_id} has no GT locations")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"VLN-CE episode {episode_id} has no GT actions")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in VLNCE_ACTION_IDS
        for value in actions
    ):
        raise ValueError(f"VLN-CE episode {episode_id} has invalid GT actions")
    forward_steps = gt.get("forward_steps", 0)
    if (
        isinstance(forward_steps, bool)
        or not isinstance(forward_steps, int)
        or forward_steps < 0
    ):
        raise ValueError(f"VLN-CE episode {episode_id} has invalid forward_steps")

    rotation = episode.get("start_rotation")
    if not isinstance(rotation, Sequence) or isinstance(rotation, (str, bytes)):
        raise ValueError(f"VLN-CE episode {episode_id} has invalid rotation")
    converted = deepcopy(dict(episode))
    converted["start_position"] = _position(
        episode.get("start_position"), f"episode {episode_id} start"
    )
    converted["start_rotation"] = list(habitat_heading_to_isaac(rotation))
    converted["goals"] = [
        {
            **dict(goals[0]),
            "position": _position(
                goals[0].get("position"), f"episode {episode_id} goal"
            ),
        }
    ]
    converted["reference_path"] = [
        _position(value, f"episode {episode_id} reference_path[{index}]")
        for index, value in enumerate(reference_path)
    ]
    converted["gt_locations"] = [
        _position(value, f"episode {episode_id} GT locations[{index}]")
        for index, value in enumerate(locations)
    ]
    converted["gt_actions"] = list(actions)
    converted["gt_forward_steps"] = forward_steps
    return converted


def convert_vlnce_payload(
    metadata: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    *,
    source_split: str,
    balanced_seed: int,
) -> dict[str, Any]:
    """Convert and scene-balance one official VLN-CE split."""

    if not isinstance(source_split, str) or not source_split.strip():
        raise ValueError("source_split must be a non-empty string")
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("VLN-CE metadata has no episodes")
    if not isinstance(ground_truth, Mapping):
        raise ValueError("VLN-CE ground truth must be an object")
    ground_truth_by_id = {
        str(key): value for key, value in ground_truth.items()
    }
    by_id: dict[str, Mapping[str, Any]] = {}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("VLN-CE episode must be an object")
        raw_episode_id = episode.get("episode_id")
        episode_id = str(raw_episode_id) if raw_episode_id is not None else ""
        if not episode_id:
            raise ValueError("VLN-CE episode requires a non-empty ID")
        if episode_id in by_id:
            raise ValueError(f"duplicate VLN-CE episode ID: {episode_id}")
        by_id[episode_id] = episode
    missing_gt = sorted(set(by_id) - set(ground_truth_by_id))
    if missing_gt:
        raise ValueError(f"VLN-CE ground truth is missing IDs: {missing_gt[:10]}")

    ordered_ids = balanced_episode_ids(episodes, seed=balanced_seed)
    converted = [
        _convert_episode(by_id[episode_id], ground_truth_by_id[episode_id])
        for episode_id in ordered_ids
    ]
    scenes = {scene_name(str(item["scene_id"])) for item in converted}
    dataset_role = "train" if source_split == "train" else "eval"
    payload = {
        "episodes": converted,
        "safe_vln_conversion": {
            "format": ISAAC_VLNCE_FORMAT,
            "source_split": source_split,
            "dataset_role": dataset_role,
            "source_coordinate_system": "habitat_xyz_y_up_xyzw",
            "coordinate_system": ISAAC_COORDINATE_SYSTEM,
            "ordering": "scene_balanced_round_robin",
            "balanced_seed": int(balanced_seed),
            "episode_count": len(converted),
            "scene_count": len(scenes),
        },
    }
    validate_isaac_vlnce_payload(payload, expected_role=dataset_role)
    return payload


def validate_isaac_vlnce_payload(
    payload: Mapping[str, Any],
    *,
    expected_role: str,
    expected_scene_count: int | None = None,
    require_source_hashes: bool = False,
) -> dict[str, Any]:
    """Fail closed if a native-camera dataset lacks split/coordinate provenance."""

    if expected_role not in {"train", "eval"}:
        raise ValueError("expected_role must be train or eval")
    provenance = payload.get("safe_vln_conversion")
    episodes = payload.get("episodes")
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "native Safe-VLN data lacks conversion provenance; convert the "
            "official VLN-CE split before collection"
        )
    if provenance.get("format") != ISAAC_VLNCE_FORMAT:
        raise ValueError("unsupported native VLN-CE conversion format")
    if provenance.get("coordinate_system") != ISAAC_COORDINATE_SYSTEM:
        raise ValueError("native VLN-CE dataset is not in Isaac coordinates")
    if provenance.get("source_coordinate_system") != "habitat_xyz_y_up_xyzw":
        raise ValueError("native VLN-CE provenance has an invalid source coordinate system")
    if provenance.get("dataset_role") != expected_role:
        raise ValueError(
            f"native VLN-CE role={provenance.get('dataset_role')!r} does not "
            f"match requested role={expected_role!r}"
        )
    source_split = provenance.get("source_split")
    if not isinstance(source_split, str) or not source_split:
        raise ValueError("native VLN-CE provenance has no source split")
    if expected_role == "train" and source_split != "train":
        raise ValueError("native VLN-CE training data must come from train split")
    if require_source_hashes:
        for field in ("source_metadata_sha256", "source_gt_sha256"):
            value = provenance.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"native VLN-CE provenance has invalid {field}")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("native VLN-CE dataset has no episodes")
    identifiers: set[str] = set()
    scenes: set[str] = set()
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("native VLN-CE episode must be an object")
        raw_episode_id = episode.get("episode_id")
        episode_id = str(raw_episode_id).strip() if raw_episode_id is not None else ""
        if not episode_id:
            raise ValueError("native VLN-CE episode requires a non-empty ID")
        if episode_id in identifiers:
            raise ValueError(f"duplicate native VLN-CE episode ID: {episode_id}")
        identifiers.add(episode_id)
        raw_scene_id = episode.get("scene_id")
        if not isinstance(raw_scene_id, str) or not raw_scene_id.strip():
            raise ValueError(f"episode {episode_id} has no scene_id")
        scene = scene_name(raw_scene_id)
        if not scene:
            raise ValueError(f"episode {episode_id} has no scene_id")
        scenes.add(scene)
        for field, length in (("start_position", 3), ("start_rotation", 4)):
            value = episode.get(field)
            if (
                not isinstance(value, list)
                or len(value) != length
                or any(not math.isfinite(float(item)) for item in value)
            ):
                raise ValueError(f"episode {episode_id} has invalid {field}")
        rotation = episode["start_rotation"]
        norm = math.sqrt(sum(float(item) ** 2 for item in rotation))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError(f"episode {episode_id} has a non-unit start_rotation")
        goals = episode.get("goals")
        if not isinstance(goals, list) or len(goals) != 1 or not isinstance(goals[0], Mapping):
            raise ValueError(f"episode {episode_id} must have one goal")
        radius = goals[0].get("radius")
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not math.isfinite(float(radius))
            or float(radius) <= 0.0
        ):
            raise ValueError(f"episode {episode_id} has an invalid goal radius")
        instruction = episode.get("instruction")
        if (
            not isinstance(instruction, Mapping)
            or not isinstance(instruction.get("instruction_text"), str)
            or not instruction["instruction_text"].strip()
        ):
            raise ValueError(f"episode {episode_id} has an invalid instruction")
        position_sequences = {
            "goal": [goals[0].get("position")],
            "reference_path": episode.get("reference_path"),
            "gt_locations": episode.get("gt_locations"),
        }
        for field, values in position_sequences.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"episode {episode_id} has no {field}")
            for index, value in enumerate(values):
                if (
                    not isinstance(value, list)
                    or len(value) != 3
                    or any(not math.isfinite(float(item)) for item in value)
                ):
                    raise ValueError(
                        f"episode {episode_id} has invalid {field}[{index}]"
                    )
        locations = episode.get("gt_locations")
        actions = episode.get("gt_actions")
        if (
            not isinstance(actions, list)
            or not actions
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value not in VLNCE_ACTION_IDS
                for value in actions
            )
        ):
            raise ValueError(f"episode {episode_id} has invalid GT actions")
        forward_steps = episode.get("gt_forward_steps")
        if (
            isinstance(forward_steps, bool)
            or not isinstance(forward_steps, int)
            or forward_steps < 0
        ):
            raise ValueError(f"episode {episode_id} has invalid forward_steps")
    if int(provenance.get("episode_count", -1)) != len(episodes):
        raise ValueError("native VLN-CE provenance episode count is stale")
    if int(provenance.get("scene_count", -1)) != len(scenes):
        raise ValueError("native VLN-CE provenance scene count is stale")
    if expected_scene_count is not None and len(scenes) != expected_scene_count:
        raise ValueError(
            f"native VLN-CE dataset has {len(scenes)} scenes; "
            f"expected {expected_scene_count}"
        )
    return {
        "episodes": len(episodes),
        "scenes": len(scenes),
        "dataset_role": expected_role,
        "source_split": provenance.get("source_split"),
    }


def load_isaac_vlnce_payload(
    path: str | Path,
    *,
    expected_role: str,
    expected_scene_count: int | None = None,
) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    try:
        with gzip.open(resolved, "rt", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read native VLN-CE dataset: {resolved}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("native VLN-CE dataset must be an object")
    validate_isaac_vlnce_payload(
        payload,
        expected_role=expected_role,
        expected_scene_count=expected_scene_count,
        require_source_hashes=True,
    )
    return dict(payload)


__all__ = [
    "ISAAC_COORDINATE_SYSTEM",
    "ISAAC_VLNCE_FORMAT",
    "VLNCE_ACTION_IDS",
    "balanced_episode_ids",
    "convert_vlnce_payload",
    "load_isaac_vlnce_payload",
    "scene_name",
    "validate_isaac_vlnce_payload",
]
