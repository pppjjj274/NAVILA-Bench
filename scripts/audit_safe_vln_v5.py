#!/usr/bin/env python3
"""Fail-closed acceptance audit for strictly aligned Safe-VLN v5 data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from safe_vln.dataset import iter_sample_refs
from safe_vln.live_render import (
    LIVE_SCHEMA_VERSION,
    NAVILA_HISTORY_SAMPLING_POLICY,
    NAVILA_VIDEO_FRAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--expected-episodes", type=int, default=500)
    parser.add_argument("--expected-scenes", type=int, default=61)
    parser.add_argument("--minimum-per-action", type=int, default=50)
    parser.add_argument("--minimum-stop", type=int, default=150)
    parser.add_argument("--expected-episode-ids")
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def read_expected_ids(path):
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [value for value in text.replace(",", " ").split() if value]
    if isinstance(payload, dict):
        payload = payload.get("episode_ids")
    if not isinstance(payload, list):
        raise ValueError("expected episode IDs must be a list")
    return {str(value) for value in payload}


def main():
    args = parse_args()
    root = Path(args.dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    refs = list(iter_sample_refs(root, args.split))
    episodes = {str(ref.metadata.get("episode_id")) for ref in refs}
    scenes = {str(ref.metadata.get("scene_id")) for ref in refs}
    action_counts = Counter(
        int(ref.metadata["oracle_action_id"])
        for ref in refs
        if ref.metadata.get("oracle_action_id") is not None
    )
    observation_keys = [
        str(ref.metadata.get("observation_key")) for ref in refs
    ]
    failures = []
    if manifest.get("schema_version") != LIVE_SCHEMA_VERSION:
        failures.append(
            f"schema={manifest.get('schema_version')} expected={LIVE_SCHEMA_VERSION}"
        )
    if manifest.get("dataset_role") != "train":
        failures.append("dataset_role must be train")
    if len(observation_keys) != len(set(observation_keys)):
        failures.append("duplicate observation_key values")
    for ref in refs:
        metadata = ref.metadata
        if metadata.get("schema_version") != LIVE_SCHEMA_VERSION:
            failures.append(f"sample schema mismatch: {ref.key}")
            break
        if not metadata.get("strict_observation_state_alignment", False):
            failures.append(f"sample is not strictly aligned: {ref.key}")
            break
        if metadata.get("history_sampling_policy") != NAVILA_HISTORY_SAMPLING_POLICY:
            failures.append(f"history policy mismatch: {ref.key}")
            break
        alignment = metadata.get("frame_alignment")
        if not isinstance(alignment, list) or len(alignment) != NAVILA_VIDEO_FRAMES:
            failures.append(f"frame alignment length mismatch: {ref.key}")
            break
        if any(
            not frame.get("history_padding", False)
            and not frame.get("strict_observation_state_alignment", False)
            for frame in alignment
        ):
            failures.append(f"physical frame is not strictly aligned: {ref.key}")
            break
    expected_ids = read_expected_ids(args.expected_episode_ids)
    if expected_ids is not None:
        missing = sorted(expected_ids - episodes)
        extra = sorted(episodes - expected_ids)
        if missing or extra:
            failures.append(
                f"episode ID mismatch: missing={missing[:10]} extra={extra[:10]}"
            )
    if not args.allow_small_dataset:
        if len(episodes) != args.expected_episodes:
            failures.append(
                f"episodes={len(episodes)} expected={args.expected_episodes}"
            )
        if len(scenes) != args.expected_scenes:
            failures.append(f"scenes={len(scenes)} expected={args.expected_scenes}")
        missing_actions = [
            action_id for action_id in range(10) if action_counts[action_id] == 0
        ]
        if missing_actions:
            failures.append(f"missing actions={missing_actions}")
        below = {
            action_id: action_counts[action_id]
            for action_id in range(10)
            if action_counts[action_id] < args.minimum_per_action
        }
        if below:
            failures.append(f"actions below minimum={below}")
        if action_counts[9] < args.minimum_stop:
            failures.append(
                f"STOP={action_counts[9]} minimum={args.minimum_stop}"
            )
    report = {
        "accepted": not failures,
        "schema_version": manifest.get("schema_version"),
        "samples": len(refs),
        "episodes": len(episodes),
        "scenes": len(scenes),
        "action_counts": {
            str(action_id): action_counts[action_id]
            for action_id in range(10)
        },
        "history_sampling_policy": NAVILA_HISTORY_SAMPLING_POLICY,
        "failures": failures,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
