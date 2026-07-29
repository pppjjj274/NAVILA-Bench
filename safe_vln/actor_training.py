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


def stable_rank(item: Any, *, seed: int, namespace: str) -> bytes:
    return hashlib.sha256(
        f"{seed}:{namespace}:{identity_of(item)}".encode("utf-8")
    ).digest()


def split_actor_episodes(
    items: Sequence[T],
    *,
    seed: int,
    dev_episodes_per_scene: int = 1,
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
                int(metadata_of(sample).get("oracle_action_id", -1)) == 9
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


def stratified_actor_schedule(
    items: Sequence[T],
    *,
    sample_count: int,
    stop_fraction: float,
    seed: int,
) -> list[T]:
    """Sample with replacement: fixed STOP share, uniform motion classes."""
    if sample_count < 10:
        raise ValueError("actor schedule requires at least ten samples")
    if not math.isclose(stop_fraction, 0.25, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("v5 actor stop fraction must be exactly 0.25")
    if sample_count % 4:
        raise ValueError("v5 actor sample count must be divisible by four")
    buckets: dict[int, list[T]] = defaultdict(list)
    for item in items:
        action_id = int(metadata_of(item).get("oracle_action_id", -1))
        if 0 <= action_id <= 9:
            buckets[action_id].append(item)
    missing = [action_id for action_id in range(10) if not buckets[action_id]]
    if missing:
        raise ValueError(f"actor data is missing action classes: {missing}")
    stop_count = sample_count // 4
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
        selected.extend(pool[index % len(pool)] for index in range(count))
    stops = [item for item in selected if int(metadata_of(item)["oracle_action_id"]) == 9]
    motions = [item for item in selected if int(metadata_of(item)["oracle_action_id"]) != 9]
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
    schedule = []
    for group_index, stop in enumerate(stops):
        group = motions[group_index * 3 : group_index * 3 + 3]
        group.insert(group_index % 4, stop)
        schedule.extend(group)
    return schedule


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
    class_samples: dict[int, int] = defaultdict(int)
    class_correct: dict[int, int] = defaultdict(int)
    stop_total = stop_true = false_stop = non_goal = 0
    finite = True
    for probability, motion, target in zip(
        stop_probabilities, motion_predictions, targets
    ):
        finite = finite and math.isfinite(float(probability))
        predicted_stop = float(probability) >= stop_threshold
        prediction = 9 if predicted_stop else int(motion)
        class_samples[int(target)] += 1
        class_correct[int(target)] += int(prediction == int(target))
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
    return {
        "stop_recall": stop_true / stop_total if stop_total else None,
        "false_stop_rate_non_goal": false_stop / non_goal if non_goal else None,
        "non_stop_macro_accuracy": (
            sum(motion_accuracies) / len(motion_accuracies)
            if motion_accuracies
            else None
        ),
        "per_class_accuracy": per_class,
        "class_samples": {
            str(action_id): count
            for action_id, count in sorted(class_samples.items())
        },
        "finite": finite,
    }
