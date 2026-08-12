"""Pure split, sampling, loss, and audit utilities for the v5 actor."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Any, Sequence, TypeVar

import torch
import torch.nn.functional as F


T = TypeVar("T")


def metadata_of(item: Any) -> dict[str, Any]:
    if hasattr(item, "metadata"):
        return item.metadata
    if isinstance(item, tuple) and len(item) == 2:
        return item[1]
    if isinstance(item, dict):
        return item
    raise TypeError(f"cannot extract metadata from {type(item)!r}")


def identity_of(item: Any) -> str:
    if hasattr(item, "identity"):
        return "\0".join(str(value) for value in item.identity)
    metadata = metadata_of(item)
    return ":".join(
        (str(metadata.get("episode_id")), str(metadata.get("index")))
    )


def target_action_id(item: Any, target_field: str = "oracle_action_id") -> int:
    """Read one validated discrete Actor target from sample metadata."""

    if not target_field:
        raise ValueError("actor target field cannot be empty")
    value = metadata_of(item).get(target_field)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9:
        raise ValueError(
            f"invalid {target_field} Actor target: {value!r}"
        )
    return value


def stable_rank(item: Any, *, seed: int, namespace: str) -> bytes:
    return hashlib.sha256(
        f"{seed}:{namespace}:{identity_of(item)}".encode("utf-8")
    ).digest()


def split_actor_episodes(
    items: Sequence[T],
    *,
    seed: int,
    dev_episodes_per_scene: int = 1,
    target_field: str = "oracle_action_id",
) -> tuple[list[T], list[T]]:
    """Hold out complete episodes, prioritizing successful STOP examples."""
    if dev_episodes_per_scene <= 0:
        raise ValueError("dev episodes per scene must be positive")
    episodes: dict[str, list[T]] = defaultdict(list)
    episode_scene: dict[str, str] = {}
    for item in items:
        metadata = metadata_of(item)
        episode_id = str(metadata.get("episode_id"))
        scene_id = str(metadata.get("scene_id"))
        if episode_id in episode_scene and episode_scene[episode_id] != scene_id:
            raise ValueError("one episode cannot belong to multiple scenes")
        episode_scene[episode_id] = scene_id
        episodes[episode_id].append(item)
    by_scene: dict[str, list[str]] = defaultdict(list)
    for episode_id, scene_id in episode_scene.items():
        by_scene[scene_id].append(episode_id)
    dev_ids: set[str] = set()
    for scene_id, episode_ids in sorted(by_scene.items()):
        if len(episode_ids) <= dev_episodes_per_scene:
            raise ValueError(
                f"scene {scene_id} has no episode left for training"
            )

        def episode_key(episode_id: str):
            samples = episodes[episode_id]
            successful_stop = any(
                target_action_id(sample, target_field) == 9
                and bool(
                    metadata_of(sample).get("system_success", True)
                    or metadata_of(sample).get("policy_success", False)
                )
                for sample in samples
            )
            digest = hashlib.sha256(
                f"{seed}:dev:{scene_id}:{episode_id}".encode("utf-8")
            ).digest()
            return (not successful_stop, digest)

        selected = sorted(episode_ids, key=episode_key)[
            :dev_episodes_per_scene
        ]
        dev_ids.update(selected)
    train = [
        item
        for item in items
        if str(metadata_of(item).get("episode_id")) not in dev_ids
    ]
    dev = [
        item
        for item in items
        if str(metadata_of(item).get("episode_id")) in dev_ids
    ]
    if not train or not dev:
        raise ValueError("actor split produced an empty train or dev partition")
    return train, dev


def split_actor_partitions(
    items: Sequence[T],
    *,
    seed: int,
    calibration_episodes_per_scene: int = 1,
    audit_episodes_per_scene: int = 1,
    target_field: str = "oracle_action_id",
) -> tuple[list[T], list[T], list[T]]:
    """Create scene-stratified train/calibration/audit partitions."""
    if calibration_episodes_per_scene <= 0:
        raise ValueError("calibration episodes per scene must be positive")
    if audit_episodes_per_scene <= 0:
        raise ValueError("audit episodes per scene must be positive")
    held_out = calibration_episodes_per_scene + audit_episodes_per_scene
    episodes: dict[str, list[T]] = defaultdict(list)
    episode_scene: dict[str, str] = {}
    for item in items:
        metadata = metadata_of(item)
        episode_id = str(metadata.get("episode_id"))
        scene_id = str(metadata.get("scene_id"))
        if episode_id in episode_scene and episode_scene[episode_id] != scene_id:
            raise ValueError("one episode cannot belong to multiple scenes")
        episode_scene[episode_id] = scene_id
        episodes[episode_id].append(item)
    by_scene: dict[str, list[str]] = defaultdict(list)
    for episode_id, scene_id in episode_scene.items():
        by_scene[scene_id].append(episode_id)

    calibration_ids: set[str] = set()
    audit_ids: set[str] = set()
    insufficient: list[str] = []
    for scene_index, (scene_id, episode_ids) in enumerate(
        sorted(by_scene.items())
    ):
        if len(episode_ids) <= held_out:
            insufficient.append(
                f"{scene_id}={len(episode_ids)} (requires >= {held_out + 1})"
            )
            continue

        def episode_key(episode_id: str):
            samples = episodes[episode_id]
            successful_stop = any(
                target_action_id(sample, target_field) == 9
                and bool(
                    metadata_of(sample).get("system_success", True)
                    or metadata_of(sample).get("policy_success", False)
                )
                for sample in samples
            )
            digest = hashlib.sha256(
                f"{seed}:held-out:{scene_id}:{episode_id}".encode("utf-8")
            ).digest()
            return (not successful_stop, digest)

        selected = sorted(episode_ids, key=episode_key)[:held_out]
        if (
            calibration_episodes_per_scene == 1
            and audit_episodes_per_scene == 1
            and scene_index % 2
        ):
            selected.reverse()
        calibration_ids.update(selected[:calibration_episodes_per_scene])
        audit_ids.update(selected[calibration_episodes_per_scene:])
    if insufficient:
        raise ValueError(
            "scenes do not have enough episodes for strict three-way split: "
            + "; ".join(insufficient)
        )

    train = [
        item
        for item in items
        if str(metadata_of(item).get("episode_id"))
        not in calibration_ids | audit_ids
    ]
    calibration = [
        item
        for item in items
        if str(metadata_of(item).get("episode_id")) in calibration_ids
    ]
    audit = [
        item
        for item in items
        if str(metadata_of(item).get("episode_id")) in audit_ids
    ]
    if not train or not calibration or not audit:
        raise ValueError("actor split produced an empty partition")
    train_ids = {
        str(metadata_of(item).get("episode_id")) for item in train
    }
    if train_ids & calibration_ids or train_ids & audit_ids:
        raise AssertionError("actor partitions contain episode leakage")
    if calibration_ids & audit_ids:
        raise AssertionError("calibration and audit episodes overlap")
    return train, calibration, audit


def stratified_actor_schedule(
    items: Sequence[T],
    *,
    sample_count: int,
    stop_fraction: float,
    seed: int,
    hard_stop_negative_fraction: float = 0.0,
    hard_stop_negative_margin_m: float = 1.0,
    target_field: str = "oracle_action_id",
) -> list[T]:
    """Sample balanced actions and optionally prioritize STOP hard negatives."""
    if sample_count < 10:
        raise ValueError("actor schedule requires at least ten samples")
    if not 0.0 < float(stop_fraction) < 1.0:
        raise ValueError("actor stop fraction must be in (0, 1)")
    if not 0.0 <= float(hard_stop_negative_fraction) <= 1.0:
        raise ValueError("hard STOP-negative fraction must be in [0, 1]")
    if float(hard_stop_negative_margin_m) <= 0.0:
        raise ValueError("hard STOP-negative margin must be positive")
    buckets: dict[int, list[T]] = defaultdict(list)
    for item in items:
        action_id = target_action_id(item, target_field)
        buckets[action_id].append(item)
    missing = [action_id for action_id in range(10) if not buckets[action_id]]
    if missing:
        raise ValueError(f"actor data is missing action classes: {missing}")
    stop_count = max(1, min(sample_count - 1, round(sample_count * stop_fraction)))
    motion_count = sample_count - stop_count
    counts = {9: stop_count}
    for action_id in range(9):
        counts[action_id] = motion_count // 9
    for action_id in range(motion_count % 9):
        counts[action_id] += 1
    selected: list[T] = []
    for action_id, count in counts.items():
        pool = sorted(
            buckets[action_id],
            key=lambda item: stable_rank(
                item, seed=seed, namespace=f"class:{action_id}"
            ),
        )
        if action_id == 9 or hard_stop_negative_fraction == 0.0:
            selected.extend(pool[index % len(pool)] for index in range(count))
            continue
        hard_pool = [
            item
            for item in pool
            if _is_hard_stop_negative(
                item, hard_stop_negative_margin_m, target_field=target_field
            )
        ]
        hard_count = min(
            len(hard_pool), round(count * hard_stop_negative_fraction)
        )
        selected.extend(hard_pool[:hard_count])
        regular_pool = [item for item in pool if item not in hard_pool] or pool
        selected.extend(
            regular_pool[index % len(regular_pool)]
            for index in range(count - hard_count)
        )
    stops = [item for item in selected if target_action_id(item, target_field) == 9]
    motions = [item for item in selected if target_action_id(item, target_field) != 9]
    stops = sorted(
        enumerate(stops),
        key=lambda pair: hashlib.sha256(
            f"{seed}:stop-slot:{pair[0]}:{identity_of(pair[1])}".encode("utf-8")
        ).digest(),
    )
    motions = sorted(
        enumerate(motions),
        key=lambda pair: hashlib.sha256(
            f"{seed}:motion-slot:{pair[0]}:{identity_of(pair[1])}".encode("utf-8")
        ).digest(),
    )
    stops = [item for _, item in stops]
    motions = [item for _, item in motions]
    stop_positions = {
        min(
            sample_count - 1,
            round((index + 0.5) * sample_count / stop_count - 0.5),
        )
        for index in range(stop_count)
    }
    if len(stop_positions) != stop_count:
        stop_positions.update(
            position
            for position in range(sample_count)
            if position not in stop_positions
        )
        stop_positions = set(sorted(stop_positions)[:stop_count])
    schedule = []
    stop_iter = iter(stops)
    motion_iter = iter(motions)
    for position in range(sample_count):
        schedule.append(
            next(stop_iter) if position in stop_positions else next(motion_iter)
        )
    return schedule


def _is_hard_stop_negative(
    item: Any,
    margin_m: float,
    *,
    target_field: str = "oracle_action_id",
) -> bool:
    """Whether a non-STOP sample lies just outside its goal radius."""
    metadata = metadata_of(item)
    if target_action_id(item, target_field) == 9:
        return False
    try:
        distance = float(metadata["distance_before"])
        radius = float(metadata["goal_radius_m"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(distance)
        and math.isfinite(radius)
        and radius < distance <= radius + margin_m
    )


def hierarchical_actor_loss(stop_logits, motion_logits, targets):
    targets = targets.long()
    stop_targets = (targets == 9).float()
    stop_loss = F.binary_cross_entropy_with_logits(
        stop_logits.float(), stop_targets
    )
    motion_mask = targets != 9
    motion_loss = (
        F.cross_entropy(motion_logits[motion_mask].float(), targets[motion_mask])
        if bool(motion_mask.any())
        else motion_logits.sum() * 0.0
    )
    return stop_loss + motion_loss, stop_loss, motion_loss


def factorized_actor_loss(
    stop_logits,
    direction_logits,
    magnitude_logits,
    targets,
):
    """STOP + direction + cost-aware ordinal magnitude supervision."""
    targets = targets.long()
    stop_targets = (targets == 9).float()
    stop_loss = F.binary_cross_entropy_with_logits(
        stop_logits.float(), stop_targets
    )
    motion_mask = targets != 9
    if not bool(motion_mask.any()):
        zero = direction_logits.sum() * 0.0 + magnitude_logits.sum() * 0.0
        return stop_loss + zero, stop_loss, zero, zero, zero

    motion_targets = targets[motion_mask]
    direction_targets = torch.div(
        motion_targets, 3, rounding_mode="floor"
    )
    magnitude_targets = motion_targets.remainder(3)
    selected_direction_logits = direction_logits[motion_mask].float()
    selected_magnitude_logits = magnitude_logits[motion_mask].float()[
        torch.arange(
            direction_targets.shape[0],
            device=direction_targets.device,
        ),
        direction_targets,
    ]
    direction_loss = F.cross_entropy(
        selected_direction_logits, direction_targets
    )
    magnitude_hard_loss = F.cross_entropy(
        selected_magnitude_logits, magnitude_targets
    )
    magnitude_indices = torch.arange(
        3, device=magnitude_targets.device
    ).unsqueeze(0)
    distance = (magnitude_indices - magnitude_targets.unsqueeze(1)).abs()
    ordinal_weights = torch.where(
        distance == 0,
        torch.ones_like(distance, dtype=torch.float32),
        torch.where(
            distance == 1,
            torch.full_like(distance, 0.5, dtype=torch.float32),
            torch.full_like(distance, 0.25, dtype=torch.float32),
        ),
    )
    ordinal_targets = ordinal_weights / ordinal_weights.sum(
        dim=-1, keepdim=True
    )
    magnitude_ordinal_loss = -(
        ordinal_targets
        * F.log_softmax(selected_magnitude_logits, dim=-1)
    ).sum(dim=-1).mean()
    total = (
        stop_loss
        + direction_loss
        + 0.5 * magnitude_hard_loss
        + 0.5 * magnitude_ordinal_loss
    )
    return (
        total,
        stop_loss,
        direction_loss,
        magnitude_hard_loss,
        magnitude_ordinal_loss,
    )


def audit_hierarchical_actor(
    stop_probabilities: Sequence[float],
    motion_predictions: Sequence[int],
    targets: Sequence[int],
    *,
    stop_threshold: float = 0.5,
) -> dict[str, Any]:
    if not (
        len(stop_probabilities) == len(motion_predictions) == len(targets)
    ):
        raise ValueError("audit arrays must have equal lengths")
    if not targets:
        raise ValueError("actor audit requires samples")
    if not 0.0 < float(stop_threshold) < 1.0:
        raise ValueError("stop threshold must be in (0, 1)")
    class_samples: dict[int, int] = defaultdict(int)
    class_correct: dict[int, int] = defaultdict(int)
    confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
    stop_total = stop_true = false_stop = non_goal = 0
    finite = True
    for probability, motion, target in zip(
        stop_probabilities, motion_predictions, targets
    ):
        finite = finite and math.isfinite(float(probability))
        predicted_stop = float(probability) >= stop_threshold
        prediction = 9 if predicted_stop else int(motion)
        if not 0 <= prediction <= 9 or not 0 <= int(target) <= 9:
            raise ValueError("audit actions must be in [0, 9]")
        class_samples[int(target)] += 1
        class_correct[int(target)] += int(prediction == int(target))
        confusion_matrix[int(target)][prediction] += 1
        if int(target) == 9:
            stop_total += 1
            stop_true += int(predicted_stop)
        else:
            non_goal += 1
            false_stop += int(predicted_stop)
    per_class = {
        str(action_id): class_correct[action_id] / count
        for action_id, count in sorted(class_samples.items())
    }
    motion_accuracies = [
        per_class[str(action_id)]
        for action_id in range(9)
        if str(action_id) in per_class
    ]
    all_accuracies = [
        per_class[str(action_id)]
        for action_id in range(10)
        if str(action_id) in per_class
    ]
    return {
        "stop_recall": stop_true / stop_total if stop_total else None,
        "false_stop_rate_non_goal": false_stop / non_goal if non_goal else None,
        "non_stop_macro_accuracy": (
            sum(motion_accuracies) / len(motion_accuracies)
            if motion_accuracies
            else None
        ),
        "macro_accuracy": (
            sum(all_accuracies) / len(all_accuracies)
            if all_accuracies
            else None
        ),
        "per_class_accuracy": per_class,
        "class_samples": {
            str(action_id): count
            for action_id, count in sorted(class_samples.items())
        },
        "confusion_matrix": confusion_matrix,
        "stop_threshold": float(stop_threshold),
        "finite": finite,
    }


def calibrate_stop_threshold(
    stop_probabilities: Sequence[float],
    motion_predictions: Sequence[int],
    targets: Sequence[int],
    *,
    minimum_stop_recall: float,
    maximum_false_stop_rate: float,
    grid_step: float = 0.01,
) -> dict[str, Any]:
    """Choose a STOP threshold without touching the final audit partition."""
    if not 0.0 < grid_step < 1.0:
        raise ValueError("threshold grid step must be in (0, 1)")
    for name, value in (
        ("minimum stop recall", minimum_stop_recall),
        ("maximum false stop rate", maximum_false_stop_rate),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    thresholds = [
        round(index * grid_step, 10)
        for index in range(1, math.ceil(1.0 / grid_step))
        if round(index * grid_step, 10) < 1.0
    ]
    reports = [
        audit_hierarchical_actor(
            stop_probabilities,
            motion_predictions,
            targets,
            stop_threshold=threshold,
        )
        for threshold in thresholds
    ]
    feasible = [
        report
        for report in reports
        if report["stop_recall"] is not None
        and report["stop_recall"] >= minimum_stop_recall
        and report["false_stop_rate_non_goal"] is not None
        and report["false_stop_rate_non_goal"] <= maximum_false_stop_rate
    ]

    def feasible_key(report):
        return (
            float(report["macro_accuracy"]),
            -float(report["false_stop_rate_non_goal"]),
            -abs(float(report["stop_threshold"]) - 0.5),
        )

    if feasible:
        selected = max(feasible, key=feasible_key)
        accepted = True
    else:
        recall_eligible = [
            report
            for report in reports
            if report["stop_recall"] is not None
            and report["stop_recall"] >= minimum_stop_recall
        ]
        if recall_eligible:
            selected = min(
                recall_eligible,
                key=lambda report: (
                    float(
                        report["false_stop_rate_non_goal"]
                        if report["false_stop_rate_non_goal"] is not None
                        else 1.0
                    ),
                    -float(report["macro_accuracy"] or 0.0),
                    abs(float(report["stop_threshold"]) - 0.5),
                ),
            )
        else:
            selected = max(
                reports,
                key=lambda report: (
                    float(report["stop_recall"] or 0.0),
                    float(report["macro_accuracy"] or 0.0),
                ),
            )
        accepted = False
    compact_curve = [
        {
            "threshold": report["stop_threshold"],
            "stop_recall": report["stop_recall"],
            "false_stop_rate_non_goal": report[
                "false_stop_rate_non_goal"
            ],
            "macro_accuracy": report["macro_accuracy"],
        }
        for report in reports
    ]
    return {
        "accepted": accepted,
        "selected_threshold": selected["stop_threshold"],
        "selected_metrics": {
            key: selected[key]
            for key in (
                "stop_recall",
                "false_stop_rate_non_goal",
                "non_stop_macro_accuracy",
                "macro_accuracy",
            )
        },
        "grid_step": float(grid_step),
        "curve": compact_curve,
    }
