#!/usr/bin/env python3
"""Convert an official VLN-CE split to validated Isaac coordinates."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from safe_vln.vlnce_dataset import (
    convert_vlnce_payload,
    scene_name,
    validate_isaac_vlnce_payload,
)


def _read(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        return json.load(input_file)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-split")
    parser.add_argument("--balanced-seed", type=int, default=20260727)
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument("--usd-root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = Path(args.metadata).expanduser().resolve()
    gt_path = Path(args.gt).expanduser().resolve()
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite converted dataset: {output}")
    source_split = args.source_split or metadata_path.parent.name
    payload = convert_vlnce_payload(
        _read(metadata_path),
        _read(gt_path),
        source_split=source_split,
        balanced_seed=args.balanced_seed,
    )
    provenance = payload["safe_vln_conversion"]
    provenance.update(
        {
            "source_metadata": str(metadata_path),
            "source_gt": str(gt_path),
            "source_metadata_sha256": _sha256(metadata_path),
            "source_gt_sha256": _sha256(gt_path),
        }
    )
    if (
        args.expected_scenes is not None
        and provenance["scene_count"] != args.expected_scenes
    ):
        raise RuntimeError(
            f"source split has {provenance['scene_count']} scenes; "
            f"expected {args.expected_scenes}"
        )
    if args.usd_root:
        root = Path(args.usd_root).expanduser()
        scenes = {
            scene_name(str(episode["scene_id"]))
            for episode in payload["episodes"]
        }
        missing = sorted(
            str(root / scene / f"{scene}.usd")
            for scene in scenes
            if not (root / scene / f"{scene}.usd").is_file()
        )
        if missing:
            raise RuntimeError(f"missing Matterport USD scenes: {missing[:10]}")
    validate_isaac_vlnce_payload(
        payload,
        expected_role=provenance["dataset_role"],
        expected_scene_count=args.expected_scenes,
        require_source_hashes=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    with gzip.open(temporary, "wt", encoding="utf-8") as output_file:
        json.dump(payload, output_file, separators=(",", ":"), allow_nan=False)
    temporary.replace(output)
    print(json.dumps({"output": str(output), **provenance}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
