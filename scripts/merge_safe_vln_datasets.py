#!/usr/bin/env python3
"""Build an immutable strict dataset from complete episode directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import uuid

from safe_vln.live_render import LIVE_SCHEMA_VERSION
from safe_vln.dataset import iter_metadata
from safe_vln.objective import SCHEMA_VERSION, validate_objective_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--replace-episode",
        action="append",
        default=[],
        help="allow a later source to replace this episode ID",
    )
    return parser.parse_args()


def read_episode_manifests(source_dir):
    root = Path(source_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing dataset manifest: {root}")
    root_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if root_manifest.get("transactional_episodes") is not True:
        raise RuntimeError(f"source is not a transactional dataset: {root}")
    try:
        objective = validate_objective_config(
            root_manifest.get("objective_config")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"source has an invalid objective: {root}") from error
    if objective.get("fingerprint") != root_manifest.get(
        "objective_fingerprint"
    ):
        raise RuntimeError(f"source objective fingerprint is invalid: {root}")
    if root_manifest.get("schema_version") == LIVE_SCHEMA_VERSION and objective.get(
        "schema_version"
    ) != SCHEMA_VERSION:
        raise RuntimeError(f"source uses an obsolete objective contract: {root}")
    root_contract = (
        root_manifest.get("schema_version"),
        root_manifest.get("objective_fingerprint"),
        root_manifest.get("dataset_role"),
        root_manifest.get("split"),
    )
    records = []
    for path in sorted((root / "completed").glob("*/manifest.json")):
        episode_dir = path.parent
        episode_manifest = json.loads(path.read_text(encoding="utf-8"))
        episode_id = str(episode_manifest.get("episode_id", episode_dir.name))
        if episode_id != episode_dir.name:
            raise RuntimeError(f"episode directory/name mismatch: {path}")
        episode_contract = (
            episode_manifest.get("schema_version"),
            episode_manifest.get("objective_fingerprint"),
            episode_manifest.get("dataset_role"),
            episode_manifest.get("split"),
        )
        if episode_contract != root_contract:
            raise RuntimeError(f"episode contract does not match source: {path}")
        if episode_manifest.get("transactional_episode") is not True:
            raise RuntimeError(f"episode is not transactionally committed: {path}")
        records.append((episode_id, episode_dir, episode_manifest))
    if not records:
        raise RuntimeError(f"dataset has no completed episodes: {root}")
    if int(root_manifest.get("completed_episodes", -1)) != len(records):
        raise RuntimeError(f"source completed episode count is inconsistent: {root}")
    if int(root_manifest.get("total_samples", -1)) != sum(
        int(item.get("total_samples", -1)) for _, _, item in records
    ):
        raise RuntimeError(f"source sample count is inconsistent: {root}")
    native_contracts = {
        (
            metadata.get("vlnce_source_split"),
            metadata.get("vlnce_source_dataset_role"),
            metadata.get("vlnce_coordinate_system"),
            metadata.get("vlnce_source_metadata_sha256"),
            metadata.get("vlnce_source_gt_sha256"),
        )
        for metadata in iter_metadata(root, root_manifest.get("split", "train"))
        if metadata.get("observation_alignment") == "native_isaac_camera"
    }
    if len(native_contracts) > 1:
        raise RuntimeError(f"source mixes native VLN-CE contracts: {root}")
    native_contract = next(iter(native_contracts), None)
    if native_contract is not None and any(value is None for value in native_contract):
        raise RuntimeError(f"source has incomplete native VLN-CE provenance: {root}")
    return root, root_manifest, records, native_contract


def main():
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite dataset: {output}")
    replace = {str(value) for value in args.replace_episode}
    selected = {}
    contracts = set()
    native_contracts = set()
    objective_config = None
    for source in args.source_dir:
        root, root_manifest, records, native_contract = read_episode_manifests(
            source
        )
        contract = (
            root_manifest.get("schema_version"),
            root_manifest.get("objective_fingerprint"),
            root_manifest.get("dataset_role"),
            root_manifest.get("split"),
        )
        contracts.add(contract)
        native_contracts.add(native_contract)
        if objective_config is None:
            objective_config = root_manifest.get("objective_config")
        for episode_id, episode_dir, episode_manifest in records:
            if episode_id in selected and episode_id not in replace:
                raise RuntimeError(
                    f"duplicate episode {episode_id}; pass --replace-episode "
                    "for an intentional repair"
                )
            selected[episode_id] = (episode_dir, episode_manifest)
    if len(contracts) != 1:
        raise RuntimeError(f"incompatible dataset contracts: {sorted(contracts)}")
    schema, fingerprint, role, split = next(iter(contracts))
    if role != "train" or split != "train":
        raise RuntimeError("dataset merge requires train split and train role")
    if len(native_contracts) > 1:
        raise RuntimeError(
            "cannot merge native and non-native data or different VLN-CE sources"
        )

    staging = output.parent / f".{output.name}.incomplete-{uuid.uuid4().hex}"
    try:
        completed = staging / "completed"
        completed.mkdir(parents=True, exist_ok=False)
        total_samples = 0
        for episode_id in sorted(selected):
            episode_dir, episode_manifest = selected[episode_id]
            destination = completed / episode_id
            shutil.copytree(episode_dir, destination)
            total_samples += int(episode_manifest.get("total_samples", 0))
        manifest = {
            "schema_version": schema,
            "objective_fingerprint": fingerprint,
            "objective_config": objective_config,
            "dataset_role": role,
            "split": split,
            "transactional_episodes": True,
            "merged_sources": [str(Path(item).expanduser().resolve()) for item in args.source_dir],
            "replaced_episode_ids": sorted(replace & set(selected)),
            "completed_episodes": len(selected),
            "total_samples": total_samples,
            "episode_ids": sorted(selected),
        }
        native_contract = next(iter(native_contracts))
        if native_contract is not None:
            manifest["vlnce_source_contract"] = {
                key: value
                for key, value in zip(
                    (
                        "source_split",
                        "dataset_role",
                        "coordinate_system",
                        "source_metadata_sha256",
                        "source_gt_sha256",
                    ),
                    native_contract,
                )
            }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "episodes": len(selected),
                "samples": total_samples,
                "replaced_episode_ids": sorted(replace & set(selected)),
            }
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
