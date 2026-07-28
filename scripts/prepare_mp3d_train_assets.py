#!/usr/bin/env python3
"""Extract the MP3D train GLBs required by VLN-CE metadata from habitat zip."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import zipfile


def _scene_name(scene_id: str) -> str:
    return Path(scene_id).stem


def _load_required_scenes(metadata: Path) -> list[str]:
    with gzip.open(metadata, "rt", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"metadata has no episodes list: {metadata}")
    return sorted({_scene_name(str(episode["scene_id"])) for episode in episodes})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-list")
    parser.add_argument("--require-scene-count", type=int, default=61)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip_path).expanduser().resolve()
    metadata = Path(args.metadata).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    scenes = _load_required_scenes(metadata)
    if len(scenes) != args.require_scene_count:
        raise RuntimeError(
            f"expected {args.require_scene_count} scenes, found {len(scenes)}"
        )

    if args.scene_list:
        Path(args.scene_list).expanduser().write_text(
            "\n".join(scenes) + "\n", encoding="utf-8"
        )

    missing: list[str] = []
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for scene in scenes:
            target = output_root / scene / f"{scene}.glb"
            if target.is_file():
                skipped += 1
                continue
            candidates = [
                name
                for name in names
                if name.endswith(f"mp3d/{scene}/{scene}.glb")
                or name.endswith(f"MatterPort3D/mp3d/{scene}/{scene}.glb")
            ]
            if not candidates:
                missing.append(scene)
                continue
            member = sorted(candidates, key=len)[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".glb.incomplete")
            with archive.open(member) as source, temporary.open("wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            temporary.replace(target)
            extracted += 1

    if missing:
        raise RuntimeError(
            f"zip does not contain {len(missing)} required train GLBs: "
            + " ".join(missing)
        )
    print(
        json.dumps(
            {
                "required_scenes": len(scenes),
                "extracted": extracted,
                "skipped_existing": skipped,
                "output_root": str(output_root),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
