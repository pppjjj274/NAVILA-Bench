#!/usr/bin/env python3
"""Fail-closed acceptance audit for strictly aligned Safe-VLN v5 data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import tarfile

from safe_vln.dataset import iter_sample_refs
from safe_vln.actions import has_valid_policy_statistics
from safe_vln.live_render import (
    LIVE_SCHEMA_VERSION,
    NAVILA_HISTORY_SAMPLING_POLICY,
    NAVILA_VIDEO_FRAMES,
)
from safe_vln.objective import SCHEMA_VERSION, validate_objective_config
from safe_vln.vlnce_dataset import ISAAC_COORDINATE_SYSTEM


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--expected-episodes",
        type=int,
        help=(
            "exact episode count; when omitted, full-dataset audits default "
            "to 500 while --allow-small-dataset skips the count check"
        ),
    )
    parser.add_argument(
        "--expected-scenes",
        type=int,
        help=(
            "exact scene count; when omitted, full-dataset audits default "
            "to 61 while --allow-small-dataset skips the count check"
        ),
    )
    parser.add_argument("--minimum-per-action", type=int, default=50)
    parser.add_argument("--minimum-stop", type=int, default=150)
    parser.add_argument("--expected-episode-ids")
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument("--require-on-policy", action="store_true")
    parser.add_argument(
        "--require-navila-teacher",
        action="store_true",
        help=(
            "require every sample to be a distillation demonstration from "
            "the original NaViLA greedy-text policy"
        ),
    )
    parser.add_argument(
        "--require-online-dagger",
        action="store_true",
        help="require recovery labels emitted by an online DAgger rollout",
    )
    parser.add_argument(
        "--allow-online-oracle",
        action="store_true",
        help="allow privileged dynamic-oracle fields for an explicit ablation",
    )
    parser.add_argument("--minimum-forward-after-turn", type=int, default=1)
    parser.add_argument("--expected-policy-version", type=int)
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
    if isinstance(payload, (str, int, float)) and not isinstance(payload, bool):
        payload = [payload]
    if isinstance(payload, dict):
        payload = payload.get("episode_ids")
    if not isinstance(payload, list):
        raise ValueError("expected episode IDs must be a list")
    return {str(value) for value in payload}


def _valid_pose(value):
    if not isinstance(value, dict):
        return False
    for field, length in (("position", 3), ("rotation_wxyz", 4)):
        vector = value.get(field)
        if not isinstance(vector, list) or len(vector) != length:
            return False
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            for item in vector
        ):
            return False
    return True


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def main():
    args = parse_args()
    if args.expected_episodes is not None and args.expected_episodes <= 0:
        raise ValueError("--expected-episodes must be positive")
    if args.expected_scenes is not None and args.expected_scenes <= 0:
        raise ValueError("--expected-scenes must be positive")
    root = Path(args.dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    refs = list(iter_sample_refs(root, args.split))
    episodes = {str(ref.metadata.get("episode_id")) for ref in refs}
    scenes = {str(ref.metadata.get("scene_id")) for ref in refs}
    action_count_field = (
        "actor_teacher_action_id"
        if args.require_navila_teacher
        else "oracle_action_id"
    )
    action_counts = Counter(
        int(ref.metadata[action_count_field])
        for ref in refs
        if ref.metadata.get(action_count_field) is not None
    )
    observation_keys = [
        str(ref.metadata.get("observation_key")) for ref in refs
    ]
    ppo_eligible = sum(bool(ref.metadata.get("ppo_eligible", False)) for ref in refs)
    recovery_categories = Counter(
        str(ref.metadata.get("recovery_category", "missing")) for ref in refs
    )
    policy_versions = {
        int(ref.metadata["policy_version"])
        for ref in refs
        if ref.metadata.get("policy_version") is not None
    }
    failures = []
    if manifest.get("schema_version") != LIVE_SCHEMA_VERSION:
        failures.append(
            f"schema={manifest.get('schema_version')} expected={LIVE_SCHEMA_VERSION}"
        )
    if manifest.get("dataset_role") != "train":
        failures.append("dataset_role must be train")
    if manifest.get("split") != args.split:
        failures.append(
            f"manifest split={manifest.get('split')!r} expected={args.split!r}"
        )
    if manifest.get("transactional_episodes") is not True:
        failures.append("dataset was not published as transactional episodes")
    try:
        manifest_completed_episodes = int(manifest["completed_episodes"])
    except (KeyError, TypeError, ValueError):
        manifest_completed_episodes = -1
    if manifest_completed_episodes != len(episodes):
        failures.append(
            "completed episode count does not match visible episode shards"
        )
    try:
        manifest_total_samples = int(manifest["total_samples"])
    except (KeyError, TypeError, ValueError):
        manifest_total_samples = -1
    if manifest_total_samples != len(refs):
        failures.append("manifest total_samples does not match readable samples")
    try:
        objective = validate_objective_config(manifest.get("objective_config"))
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"invalid objective configuration: {error}")
    else:
        if objective.get("fingerprint") != manifest.get("objective_fingerprint"):
            failures.append("manifest objective fingerprint is invalid")
        if objective.get("schema_version") != SCHEMA_VERSION:
            failures.append(
                f"objective schema={objective.get('schema_version')} "
                f"expected={SCHEMA_VERSION}"
            )
    if len(observation_keys) != len(set(observation_keys)):
        failures.append("duplicate observation_key values")
    # Metadata alone is not a usable multimodal sample. Verify every JSON
    # record has exactly eight image members, and reject duplicate tar member
    # names that tarfile lookup would otherwise resolve ambiguously.
    refs_by_shard = {}
    native_source_contracts = set()
    for ref in refs:
        refs_by_shard.setdefault(ref.shard_path, []).append(ref)
    for shard_path, shard_refs in refs_by_shard.items():
        with tarfile.open(shard_path, "r") as archive:
            member_counts = Counter(
                member.name for member in archive if member.isfile()
            )
        invalid_members = None
        invalid_key = None
        for ref in shard_refs:
            expected = [ref.metadata_name] + [
                f"{ref.key}.{index}.jpg"
                for index in range(NAVILA_VIDEO_FRAMES)
            ]
            invalid_members = {
                member: member_counts[member]
                for member in expected
                if member_counts[member] != 1
            }
            if invalid_members:
                invalid_key = ref.key
                break
        if invalid_members:
            failures.append(
                "sample tar members missing or duplicated: "
                f"{invalid_key} {invalid_members}"
            )
            break
    for ref in refs:
        metadata = ref.metadata
        if metadata.get("schema_version") != LIVE_SCHEMA_VERSION:
            failures.append(f"sample schema mismatch: {ref.key}")
            break
        if metadata.get("objective_fingerprint") != manifest.get(
            "objective_fingerprint"
        ):
            failures.append(f"sample objective mismatch: {ref.key}")
            break
        if metadata.get("ppo_eligible", False):
            policy_stats = {
                **metadata,
                "log_prob": metadata.get("old_log_prob"),
            }
            if not has_valid_policy_statistics(
                policy_stats,
                objective_fingerprint=manifest.get("objective_fingerprint"),
            ):
                failures.append(f"PPO policy statistics are inconsistent: {ref.key}")
                break
        if not metadata.get("strict_observation_state_alignment", False):
            failures.append(f"sample is not strictly aligned: {ref.key}")
            break
        if metadata.get("history_sampling_policy") != NAVILA_HISTORY_SAMPLING_POLICY:
            failures.append(f"history policy mismatch: {ref.key}")
            break
        diagnostics = metadata.get("safety_diagnostics")
        if not isinstance(diagnostics, dict) or diagnostics.get("contact_sensor_enabled") is not True:
            failures.append(f"contact sensor was not verified: {ref.key}")
            break
        has_online_oracle = bool(
            metadata.get("online_oracle_enabled", False)
            or metadata.get("oracle_eligible", False)
            or metadata.get("oracle_valid", False)
            or metadata.get("dynamic_oracle_action") is not None
        )
        if has_online_oracle and not args.allow_online_oracle:
            failures.append("live dynamic-oracle labels are diagnostic-only")
            break
        action_id = metadata.get("action_id")
        if (
            action_id is not None
            and 0 <= int(action_id) <= 8
            and not isinstance(diagnostics.get("turn_execution"), dict)
        ):
            failures.append(f"turn execution diagnostics missing: {ref.key}")
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
        if metadata.get("observation_alignment") == "native_isaac_camera":
            source_contract = (
                metadata.get("vlnce_source_split"),
                metadata.get("vlnce_source_dataset_role"),
                metadata.get("vlnce_coordinate_system"),
                metadata.get("vlnce_source_metadata_sha256"),
                metadata.get("vlnce_source_gt_sha256"),
            )
            native_source_contracts.add(source_contract)
            if source_contract[:3] != (
                "train",
                metadata.get("dataset_role"),
                ISAAC_COORDINATE_SYSTEM,
            ):
                failures.append(
                    f"native VLN-CE source contract mismatch: {ref.key}"
                )
                break
            if not all(_valid_sha256(value) for value in source_contract[3:]):
                failures.append(
                    f"native VLN-CE source hashes are invalid: {ref.key}"
                )
                break
            physical_frames = [
                frame
                for frame in alignment
                if not frame.get("history_padding", False)
            ]
            if any(
                not _valid_pose(frame.get("isaac_pose"))
                or not _valid_pose(frame.get("camera_pose"))
                for frame in physical_frames
            ):
                failures.append(
                    f"native frame pose provenance is invalid: {ref.key}"
                )
                break
    if len(native_source_contracts) > 1:
        failures.append("native samples mix multiple VLN-CE source contracts")
    by_episode = {}
    for ref in refs:
        by_episode.setdefault(str(ref.metadata.get("episode_id")), []).append(
            ref.metadata
        )
    for episode_id, rows in by_episode.items():
        try:
            rows.sort(key=lambda item: int(item["index"]))
            indices = [int(item["index"]) for item in rows]
        except (KeyError, TypeError, ValueError):
            failures.append(f"episode {episode_id} has invalid transition indices")
            break
        if (
            indices != list(range(len(rows)))
            or any(bool(item.get("done")) for item in rows[:-1])
            or not bool(rows[-1].get("done"))
        ):
            failures.append(
                f"episode {episode_id} is incomplete or non-contiguous"
            )
            break
        invalid_link = False
        for index, item in enumerate(rows):
            expected_next = (
                None
                if index == len(rows) - 1
                else rows[index + 1].get("observation_key")
            )
            if item.get("next_observation_key") != expected_next:
                failures.append(
                    f"episode {episode_id} has a broken observation link at "
                    f"transition {index}"
                )
                invalid_link = True
                break
        if invalid_link:
            break
    expected_ids = read_expected_ids(args.expected_episode_ids)
    if expected_ids is not None:
        missing = sorted(expected_ids - episodes)
        extra = sorted(episodes - expected_ids)
        if missing or extra:
            failures.append(
                f"episode ID mismatch: missing={missing[:10]} extra={extra[:10]}"
            )
    if args.require_on_policy:
        if not refs:
            failures.append("on-policy rollout contains no samples")
        if any(ref.metadata.get("collection_policy") != "vlm" for ref in refs):
            failures.append("on-policy rollout contains non-VLM samples")
        if any(not ref.metadata.get("policy_statistics_valid", False) for ref in refs):
            failures.append("on-policy rollout contains invalid policy statistics")
        if ppo_eligible == 0:
            failures.append("on-policy rollout contains no PPO-eligible samples")
        fingerprints = {
            ref.metadata.get("objective_fingerprint") for ref in refs
        }
        if fingerprints != {manifest.get("objective_fingerprint")}:
            failures.append("rollout objective fingerprints are inconsistent")
        if (
            args.expected_policy_version is not None
            and policy_versions != {args.expected_policy_version}
        ):
            failures.append(
                f"policy versions={sorted(policy_versions)} "
                f"expected={[args.expected_policy_version]}"
            )
    teacher_eligible = sum(
        bool(ref.metadata.get("actor_distillation_eligible", False))
        for ref in refs
    )
    if args.require_navila_teacher:
        invalid_teacher = [
            ref.key
            for ref in refs
            if not (
                ref.metadata.get("actor_distillation_eligible", False)
                and ref.metadata.get("actor_teacher_interface")
                == "navila-greedy-text-v1"
                and ref.metadata.get("policy_interface")
                == "navila-greedy-text-v1"
                and isinstance(ref.metadata.get("actor_teacher_action_id"), int)
                and 0 <= ref.metadata["actor_teacher_action_id"] <= 9
            )
        ]
        if invalid_teacher:
            failures.append(
                "dataset contains non-NaViLA Actor demonstrations: "
                f"{invalid_teacher[:5]}"
            )
    if args.require_online_dagger:
        if args.minimum_forward_after_turn < 1:
            failures.append("minimum-forward-after-turn must be positive")
        if any(not ref.metadata.get("online_dagger_eligible", False) for ref in refs):
            failures.append("online DAgger rollout contains ineligible samples")
        if recovery_categories["forward_after_turn"] < args.minimum_forward_after_turn:
            failures.append(
                "online DAgger rollout lacks sufficient forward-after-turn "
                f"recovery samples={recovery_categories['forward_after_turn']} "
                f"minimum={args.minimum_forward_after_turn}"
            )
    expected_episodes = (
        args.expected_episodes
        if args.expected_episodes is not None
        else 500
    )
    expected_scenes = (
        args.expected_scenes
        if args.expected_scenes is not None
        else 61
    )
    if not args.allow_small_dataset or args.expected_episodes is not None:
        if len(episodes) != expected_episodes:
            failures.append(
                f"episodes={len(episodes)} expected={expected_episodes}"
            )
    if not args.allow_small_dataset or args.expected_scenes is not None:
        if len(scenes) != expected_scenes:
            failures.append(f"scenes={len(scenes)} expected={expected_scenes}")
    if not args.allow_small_dataset:
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
        "expected_episodes": (
            expected_episodes
            if not args.allow_small_dataset or args.expected_episodes is not None
            else None
        ),
        "expected_scenes": (
            expected_scenes
            if not args.allow_small_dataset or args.expected_scenes is not None
            else None
        ),
        "action_counts": {
            str(action_id): action_counts[action_id]
            for action_id in range(10)
        },
        "action_count_source": action_count_field,
        "history_sampling_policy": NAVILA_HISTORY_SAMPLING_POLICY,
        "ppo_eligible_samples": ppo_eligible,
        "actor_distillation_eligible_samples": teacher_eligible,
        "policy_versions": sorted(policy_versions),
        "recovery_categories": dict(sorted(recovery_categories.items())),
        "failures": failures,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
