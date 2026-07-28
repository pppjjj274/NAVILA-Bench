#!/usr/bin/env python3
"""Select deterministic, scene-balanced VLN-CE episode IDs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import random


def _scene_name(scene_id: str) -> str:
    return os.path.splitext(os.path.basename(scene_id))[0]


def balanced_episode_ids(episodes, *, seed: int) -> list[str]:
    by_scene = defaultdict(list)
    for episode in episodes:
        by_scene[_scene_name(str(episode["scene_id"]))].append(
            str(episode["episode_id"])
        )
    generator = random.Random(seed)
    for scene_ids in by_scene.values():
        generator.shuffle(scene_ids)
    scenes = sorted(by_scene)
    generator.shuffle(scenes)
    ordered = []
    round_index = 0
    while True:
        added = False
        for scene in scenes:
            if round_index < len(by_scene[scene]):
                ordered.append(by_scene[scene][round_index])
                added = True
        if not added:
            break
        round_index += 1
    return ordered


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--require-scene-count", type=int)
    parser.add_argument("--format", choices=("shell", "json"), default="shell")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    metadata_path = Path(args.metadata).expanduser()
    with gzip.open(metadata_path, "rt", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"metadata has no episode list: {metadata_path}")
    scene_count = len({_scene_name(str(item["scene_id"])) for item in episodes})
    if (
        args.require_scene_count is not None
        and scene_count != args.require_scene_count
    ):
        raise RuntimeError(
            f"expected {args.require_scene_count} scenes, found {scene_count}"
        )
    ordered = balanced_episode_ids(episodes, seed=args.seed)
    selected = ordered[args.offset : args.offset + args.count]
    if len(selected) != args.count:
        raise RuntimeError(
            f"requested {args.count} IDs at offset {args.offset}, "
            f"but only {len(selected)} are available"
        )
    if args.format == "json":
        print(json.dumps(selected))
    else:
        print(" ".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
