"""Deterministic balanced sampling for Safe-VLN training datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any, Iterable, Sequence, TypeVar

from .dataset import SafeVLNSampleRef


T = TypeVar("T")


def _metadata(item: Any) -> dict[str, Any]:
    if isinstance(item, SafeVLNSampleRef):
        return item.metadata
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
        return item[1]
    if isinstance(item, dict):
        return item
    raise TypeError(f"cannot extract Safe-VLN metadata from {type(item)!r}")


def _identity(item: Any) -> str:
    if isinstance(item, SafeVLNSampleRef):
        return "\0".join(item.identity)
    metadata = _metadata(item)
    return json.dumps(
        [
            str(metadata.get("episode_id", "")),
            int(metadata.get("index", -1)),
            str(metadata.get("observation_key", "")),
        ],
        separators=(",", ":"),
    )


def _rank(seed: int, namespace: str, item: Any) -> bytes:
    payload = f"{seed}:{namespace}:{_identity(item)}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def deterministic_shuffle(
    items: Iterable[T],
    *,
    seed: int,
    namespace: str,
) -> list[T]:
    """Return a stable pseudo-random order independent of Python hash state."""

    return sorted(items, key=lambda item: _rank(seed, namespace, item))


def _add_unique(
    selected: list[T],
    selected_ids: set[str],
    item: T,
) -> bool:
    identity = _identity(item)
    if identity in selected_ids:
        return False
    selected_ids.add(identity)
    selected.append(item)
    return True


def _round_robin_buckets(
    buckets: dict[Any, list[T]],
    *,
    selected: list[T],
    selected_ids: set[str],
    limit: int,
    seed: int,
    namespace: str,
) -> None:
    ordered = {
        key: deterministic_shuffle(
            values,
            seed=seed,
            namespace=f"{namespace}:{key}",
        )
        for key, values in buckets.items()
    }
    offsets = {key: 0 for key in ordered}
    keys = sorted(ordered, key=str)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            values = ordered[key]
            while offsets[key] < len(values):
                item = values[offsets[key]]
                offsets[key] += 1
                if _add_unique(selected, selected_ids, item):
                    progressed = True
                    break
            if len(selected) >= limit:
                return
        if not progressed:
            return


def sampling_summary(items: Sequence[Any]) -> dict[str, Any]:
    metadata = [_metadata(item) for item in items]
    actions = Counter(
        int(item["oracle_action_id"])
        for item in metadata
        if item.get("oracle_action_id") is not None
    )
    return {
        "samples": len(items),
        "episodes": len({str(item.get("episode_id")) for item in metadata}),
        "scenes": len({str(item.get("scene_id")) for item in metadata}),
        "stop_samples": actions.get(9, 0),
        "action_counts": {
            str(action_id): count for action_id, count in sorted(actions.items())
        },
    }


def select_balanced_oracle(
    items: Sequence[T],
    *,
    max_samples: int | None,
    seed: int,
) -> list[T]:
    """Select all STOP labels, every episode, then balance other actions.

    The input must already be restricted to strict, oracle-eligible samples.
    """

    eligible = [
        item
        for item in items
        if _metadata(item).get("oracle_action_id") is not None
    ]
    if not eligible:
        return []
    limit = len(eligible) if max_samples is None else min(max_samples, len(eligible))
    if limit <= 0:
        return []

    stops = [
        item
        for item in eligible
        if int(_metadata(item)["oracle_action_id"]) == 9
    ]
    by_episode: dict[str, list[T]] = defaultdict(list)
    for item in eligible:
        metadata = _metadata(item)
        if int(metadata["oracle_action_id"]) != 9:
            by_episode[str(metadata.get("episode_id"))].append(item)

    required = len(stops) + len(by_episode)
    if limit < required:
        raise ValueError(
            "balanced-oracle sampling requires room for every STOP transition "
            f"and one non-STOP transition per episode: need {required}, got {limit}"
        )

    selected: list[T] = []
    selected_ids: set[str] = set()
    for item in deterministic_shuffle(
        stops,
        seed=seed,
        namespace="oracle-stops",
    ):
        _add_unique(selected, selected_ids, item)

    episode_order = sorted(
        by_episode,
        key=lambda episode_id: hashlib.sha256(
            f"{seed}:episode:{episode_id}".encode("utf-8")
        ).digest(),
    )
    for episode_id in episode_order:
        candidates = deterministic_shuffle(
            by_episode[episode_id],
            seed=seed,
            namespace=f"oracle-episode:{episode_id}",
        )
        _add_unique(selected, selected_ids, candidates[0])

    by_action: dict[int, list[T]] = defaultdict(list)
    for item in eligible:
        action_id = int(_metadata(item)["oracle_action_id"])
        if action_id != 9:
            by_action[action_id].append(item)
    _round_robin_buckets(
        by_action,
        selected=selected,
        selected_ids=selected_ids,
        limit=limit,
        seed=seed,
        namespace="oracle-action",
    )
    if len(selected) != limit:
        raise RuntimeError(
            f"balanced-oracle selected {len(selected)} of requested {limit} samples"
        )
    return deterministic_shuffle(
        selected,
        seed=seed,
        namespace="oracle-training-order",
    )


def _risk_stratum(metadata: dict[str, Any]) -> int:
    components = metadata.get("cost_components") or {}
    hard_event = any(
        float(components.get(key, 0.0)) > 0
        for key in (
            "collision_event",
            "fall_event",
            "blocked_event",
            "unsafe_contact",
            "fall",
        )
    )
    if hard_event:
        return 3
    cost_return = float(metadata.get("cost_return", 0.0))
    if cost_return >= 0.5:
        return 2
    if cost_return > 0.05:
        return 1
    return 0


def select_balanced_critic(
    items: Sequence[T],
    *,
    max_samples: int | None,
    seed: int,
) -> list[T]:
    """Balance critic samples over episodes and empirical risk strata."""

    eligible = [
        item
        for item in items
        if _metadata(item).get("reward_return") is not None
        and _metadata(item).get("cost_return") is not None
    ]
    if not eligible:
        return []
    limit = len(eligible) if max_samples is None else min(max_samples, len(eligible))
    if limit <= 0:
        return []

    selected: list[T] = []
    selected_ids: set[str] = set()
    hard = [item for item in eligible if _risk_stratum(_metadata(item)) == 3]
    for item in deterministic_shuffle(hard, seed=seed, namespace="critic-hard"):
        if len(selected) >= limit:
            break
        _add_unique(selected, selected_ids, item)

    by_episode: dict[str, list[T]] = defaultdict(list)
    for item in eligible:
        by_episode[str(_metadata(item).get("episode_id"))].append(item)
    if len(selected) < limit and limit >= len(by_episode):
        for episode_id in sorted(by_episode):
            candidates = deterministic_shuffle(
                by_episode[episode_id],
                seed=seed,
                namespace=f"critic-episode:{episode_id}",
            )
            _add_unique(selected, selected_ids, candidates[0])
            if len(selected) >= limit:
                break

    by_stratum: dict[int, list[T]] = defaultdict(list)
    for item in eligible:
        by_stratum[_risk_stratum(_metadata(item))].append(item)
    _round_robin_buckets(
        by_stratum,
        selected=selected,
        selected_ids=selected_ids,
        limit=limit,
        seed=seed,
        namespace="critic-risk",
    )
    if len(selected) != limit:
        raise RuntimeError(
            f"balanced-critic selected {len(selected)} of requested {limit} samples"
        )
    return deterministic_shuffle(
        selected,
        seed=seed,
        namespace="critic-training-order",
    )


def select_balanced_ppo(
    items: Sequence[T],
    *,
    max_samples: int | None,
    seed: int,
) -> list[T]:
    """Balance PPO transitions over episodes, oracle actions, and cost risk."""

    if not items:
        return []
    limit = len(items) if max_samples is None else min(max_samples, len(items))
    if limit <= 0:
        return []

    selected: list[T] = []
    selected_ids: set[str] = set()
    stops = [
        item
        for item in items
        if _metadata(item).get("oracle_action_id") == 9
    ]
    for item in deterministic_shuffle(stops, seed=seed, namespace="ppo-stops"):
        if len(selected) >= limit:
            break
        _add_unique(selected, selected_ids, item)

    by_episode: dict[str, list[T]] = defaultdict(list)
    for item in items:
        by_episode[str(_metadata(item).get("episode_id"))].append(item)
    if limit >= len(by_episode):
        for episode_id in sorted(by_episode):
            candidates = deterministic_shuffle(
                by_episode[episode_id],
                seed=seed,
                namespace=f"ppo-episode:{episode_id}",
            )
            _add_unique(selected, selected_ids, candidates[0])

    buckets: dict[tuple[int, int], list[T]] = defaultdict(list)
    for item in items:
        metadata = _metadata(item)
        action_id = int(
            metadata.get("oracle_action_id", metadata.get("action_id", -1))
        )
        buckets[action_id, _risk_stratum(metadata)].append(item)
    _round_robin_buckets(
        buckets,
        selected=selected,
        selected_ids=selected_ids,
        limit=limit,
        seed=seed,
        namespace="ppo-action-risk",
    )
    if len(selected) != limit:
        raise RuntimeError(
            f"balanced-ppo selected {len(selected)} of requested {limit} samples"
        )
    return deterministic_shuffle(
        selected,
        seed=seed,
        namespace="ppo-training-order",
    )
