#!/usr/bin/env python3
"""Build a strict Actor-training dataset from native shards and Oracle sidecars."""

from __future__ import annotations

import argparse
from io import BytesIO
import copy
import json
from pathlib import Path
import tarfile

from safe_vln.live_render import LIVE_SCHEMA_VERSION, NAVILA_HISTORY_SAMPLING_POLICY
from safe_vln.objective import SCHEMA_VERSION, validate_objective_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", action="append", required=True)
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _read_labels(paths: list[str]) -> tuple[dict[tuple[str, int], dict], dict]:
    labels = {}
    source_contract = None
    for path in paths:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != "safe-vln-native-oracle-v1"
            or payload.get("teacher_source") != "habitat_navmesh_shortest_path"
            or payload.get("diagnostic_only") is not True
        ):
            raise ValueError(f"invalid diagnostic Oracle sidecar: {path}")
        current_contract = payload.get("source_contract")
        if not isinstance(current_contract, dict):
            raise ValueError(f"Oracle sidecar has no source contract: {path}")
        if source_contract is None:
            source_contract = current_contract
        elif current_contract != source_contract:
            raise ValueError("Oracle sidecars use different source contracts")
        for record in payload.get("records", []):
            key = (str(record["episode_id"]), int(record["transition_index"]))
            if key in labels:
                raise ValueError(f"duplicate Oracle label: {key}")
            labels[key] = record
    if not labels:
        raise ValueError("Oracle sidecars contain no records")
    return labels, source_contract


def _write_member(output: tarfile.TarFile, member: tarfile.TarInfo, payload: bytes) -> None:
    info = copy.copy(member)
    info.size = len(payload)
    output.addfile(info, BytesIO(payload))


def _patch_sample(
    metadata: dict,
    labels: dict[tuple[str, int], dict],
    source_contract: dict,
) -> dict:
    source_policy = metadata.get("history_sampling_policy")
    if source_policy != NAVILA_HISTORY_SAMPLING_POLICY:
        raise ValueError(
            "native Actor conversion requires the official NaViLA history "
            f"policy {NAVILA_HISTORY_SAMPLING_POLICY!r}; got {source_policy!r}. "
            "Recollect native-camera data with the patched sampler."
        )
    episode_id = str(metadata.get("episode_id"))
    index = metadata.get("index")
    if index is None:
        raise ValueError(f"sample {metadata.get('observation_key')} has no index")
    key = (episode_id, int(index))
    label = labels.pop(key, None)
    if label is None:
        raise ValueError(f"missing Oracle label for {key}")
    if label.get("observation_key") != metadata.get("observation_key"):
        raise ValueError(f"observation key mismatch for {key}")
    if label.get("isaac_pose_before") != metadata.get("isaac_pose_before"):
        raise ValueError(f"pose mismatch for {key}")
    if label.get("frame_alignment") != metadata.get("frame_alignment"):
        raise ValueError(f"frame alignment mismatch for {key}")
    if label.get("source_contract") != source_contract:
        raise ValueError(f"label source contract mismatch for {key}")
    sample_contract = {
        "source_split": metadata.get("vlnce_source_split"),
        "dataset_role": metadata.get("vlnce_source_dataset_role"),
        "coordinate_system": metadata.get("vlnce_coordinate_system"),
        "source_metadata_sha256": metadata.get("vlnce_source_metadata_sha256"),
        "source_gt_sha256": metadata.get("vlnce_source_gt_sha256"),
    }
    if sample_contract != source_contract:
        raise ValueError(f"sample source contract mismatch for {key}")
    metadata.update(
        {
            "schema_version": LIVE_SCHEMA_VERSION,
            "dataset_role": "train",
            "oracle_valid": bool(label.get("oracle_valid", False)),
            "oracle_eligible": bool(label.get("oracle_eligible", False)),
            "oracle_action_id": label.get("oracle_action_id"),
            "oracle_action": label.get("oracle_action"),
            "teacher_source": label.get("teacher_source"),
            "teacher_navmesh_metadata": label.get("navmesh_metadata"),
            "teacher_pose_alignment": {
                "isaac_pose_before": label.get("isaac_pose_before"),
                "frame_alignment": label.get("frame_alignment"),
            },
        }
    )
    return metadata


def _convert_source(
    source: Path,
    destination: Path,
    labels: dict,
    objective: dict,
    source_contract: dict,
) -> tuple[list[str], int]:
    episodes = []
    sample_count = 0
    manifests = sorted(source.glob("completed/*/manifest.json"))
    if not manifests:
        raise ValueError(f"source has no transactional episodes: {source}")
    for manifest_path in manifests:
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episode_id = str(source_manifest.get("episode_id"))
        if episode_id != manifest_path.parent.name:
            raise ValueError(f"episode directory/name mismatch: {manifest_path}")
        episode_dir = destination / "completed" / episode_id
        episode_dir.mkdir(parents=True, exist_ok=False)
        episode_samples = 0
        shards = sorted(manifest_path.parent.glob("train-*.tar"))
        if not shards:
            raise ValueError(f"episode has no shards: {manifest_path.parent}")
        for shard in shards:
            with tarfile.open(shard, "r") as archive:
                members = archive.getmembers()
                json_members = [
                    member
                    for member in members
                    if member.isfile() and member.name.endswith(".json")
                ]
                if not json_members:
                    raise ValueError(f"shard has no samples: {shard}")
                with tarfile.open(episode_dir / shard.name, "w") as output:
                    for member in members:
                        extracted = archive.extractfile(member) if member.isfile() else None
                        payload = extracted.read() if extracted is not None else b""
                        if member.isfile() and member.name.endswith(".json"):
                            metadata = _patch_sample(
                                json.loads(payload.decode("utf-8")),
                                labels,
                                source_contract,
                            )
                            payload = json.dumps(
                                metadata,
                                ensure_ascii=False,
                                sort_keys=True,
                                allow_nan=False,
                            ).encode("utf-8")
                            episode_samples += 1
                        _write_member(output, member, payload)
        if episode_samples != int(source_manifest.get("total_samples", -1)):
            raise ValueError(f"episode sample count mismatch: {manifest_path}")
        manifest = {
            **source_manifest,
            "schema_version": LIVE_SCHEMA_VERSION,
            "objective_fingerprint": objective["fingerprint"],
            "objective_config": objective,
            "dataset_role": "train",
            "split": "train",
            "transactional_episode": True,
            "episode_id": episode_id,
            "diagnostic_only": True,
            "teacher_source": "habitat_navmesh_shortest_path",
        }
        (episode_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        episodes.append(episode_id)
        sample_count += episode_samples
    if labels:
        raise ValueError(f"Oracle sidecar has unused labels: {sorted(labels)[:5]}")
    return episodes, sample_count


def main() -> int:
    args = _parser().parse_args()
    if len(args.source_dir) != len(args.labels):
        raise ValueError("provide exactly one --labels sidecar per --source-dir")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite dataset: {output}")
    manifests = [json.loads((Path(source).expanduser() / "manifest.json").read_text()) for source in args.source_dir]
    if any(
        item.get("schema_version") != LIVE_SCHEMA_VERSION
        or item.get("transactional_episodes") is not True
        or item.get("dataset_role") != "train"
        or item.get("split") != "train"
        for item in manifests
    ):
        raise ValueError("native Oracle conversion requires transactional v5 train data")
    fingerprints = {item.get("objective_fingerprint") for item in manifests}
    objectives = [item.get("objective_config") for item in manifests]
    if len(fingerprints) != 1 or not fingerprints or not objectives[0]:
        raise ValueError("source datasets have incompatible objective contracts")
    validated_objectives = [validate_objective_config(item) for item in objectives]
    if any(item.get("schema_version") != SCHEMA_VERSION for item in validated_objectives):
        raise ValueError("source dataset does not use the current objective contract")
    objective = validated_objectives[0]
    if any(item != objective for item in validated_objectives[1:]):
        raise ValueError("source datasets have different objective configurations")
    if objective["fingerprint"] != next(iter(fingerprints)):
        raise ValueError("source objective fingerprint is invalid")
    staging = output.parent / f".{output.name}.incomplete"
    if staging.exists():
        raise ValueError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        episodes = []
        total_samples = 0
        source_contracts = []
        for source, sidecar in zip(args.source_dir, args.labels):
            labels, source_contract = _read_labels([sidecar])
            source_contracts.append(source_contract)
            converted, count = _convert_source(
                Path(source).expanduser(),
                staging,
                labels,
                objective,
                source_contract,
            )
            episodes.extend(converted)
            total_samples += count
        if len(episodes) != len(set(episodes)):
            raise ValueError("source datasets contain duplicate episode IDs")
        if len({json.dumps(item, sort_keys=True) for item in source_contracts}) != 1:
            raise ValueError("source datasets use different VLN-CE provenance")
        manifest = {
            "schema_version": LIVE_SCHEMA_VERSION,
            "objective_fingerprint": next(iter(fingerprints)),
            "objective_config": objective,
            "dataset_role": "train",
            "split": "train",
            "transactional_episodes": True,
            "teacher_source": "habitat_navmesh_shortest_path",
            "diagnostic_only": True,
            "source_contract": source_contracts[0],
            "completed_episodes": len(episodes),
            "total_samples": total_samples,
            "episode_ids": sorted(episodes),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"output_dir": str(output), "episodes": len(episodes), "samples": total_samples}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
