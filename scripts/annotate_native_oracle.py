#!/usr/bin/env python3
"""Attach fail-closed Habitat navmesh teacher labels to native-camera data.

The native Isaac collector records the actual Go2 pose but intentionally does
not query a privileged oracle.  This sidecar joins each transition with a
shortest-path action at that exact pose without rewriting the RGB tar shards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import uuid

from safe_vln.actions import action_from_id
from safe_vln.live_render import (
    HabitatRenderClient,
    NAVILA_HISTORY_SAMPLING_POLICY,
    NAVILA_VIDEO_FRAMES,
    isaac_position_to_habitat,
    isaac_yaw_to_habitat_yaw,
)
from safe_vln.vlnce_dataset import (
    ISAAC_COORDINATE_SYSTEM,
    load_isaac_vlnce_payload,
    scene_name,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", action="append", required=True)
    parser.add_argument("--r2r-data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--render-host", default="127.0.0.1")
    parser.add_argument("--render-port", type=int, default=54322)
    parser.add_argument("--render-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--episode-ids", nargs="+", type=str)
    parser.add_argument(
        "--allow-diagnostic-navmesh-teacher",
        action="store_true",
        help=(
            "Acknowledge that Habitat shortest-path labels are a diagnostic "
            "ablation, not the canonical original-NaViLA Actor target."
        ),
    )
    return parser


def _read_asset_episodes(path: str | Path) -> tuple[dict[str, dict], dict]:
    payload = load_isaac_vlnce_payload(
        path,
        expected_role="train",
        expected_scene_count=61,
    )
    episodes = payload["episodes"]
    result = {}
    for episode in episodes:
        if not isinstance(episode, dict) or episode.get("episode_id") is None:
            raise ValueError("R2R asset contains an invalid episode")
        key = str(episode["episode_id"])
        if key in result:
            raise ValueError(f"duplicate R2R episode id: {key}")
        result[key] = episode
    return result, dict(payload["safe_vln_conversion"])


def _goal_habitat_position(episode: dict) -> list[float]:
    goals = episode.get("goals")
    if not isinstance(goals, list) or len(goals) != 1:
        raise ValueError(f"episode {episode.get('episode_id')} has no unique goal")
    goal = goals[0].get("position")
    if not isinstance(goal, list) or len(goal) != 3:
        raise ValueError(f"episode {episode.get('episode_id')} has invalid goal")
    # The validated native asset stores Isaac xyz; Habitat uses x, z, -y.
    return [float(goal[0]), float(goal[2]), -float(goal[1])]


def _episode_files(source_dirs: list[str], episode_ids: set[str] | None, limit: int | None):
    files = []
    for source in source_dirs:
        files.extend(Path(source).expanduser().glob("episodes/episode_*.json"))
    files.sort(key=lambda path: (str(path.parent.parent), path.name))
    if episode_ids is not None:
        files = [path for path in files if path.stem.removeprefix("episode_") in episode_ids]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--episode-limit must be positive")
        files = files[:limit]
    if not files:
        raise ValueError("no native-camera episode summaries matched")
    return files


def annotate(args: argparse.Namespace) -> dict:
    if not args.allow_diagnostic_navmesh_teacher:
        raise ValueError(
            "native navmesh labels are diagnostic-only; pass "
            "--allow-diagnostic-navmesh-teacher for an explicit ablation"
        )
    asset, provenance = _read_asset_episodes(args.r2r_data_path)
    selected_ids = set(args.episode_ids) if args.episode_ids else None
    files = _episode_files(args.source_dir, selected_ids, args.episode_limit)
    client = HabitatRenderClient(
        args.render_host,
        args.render_port,
        timeout_seconds=args.render_timeout_seconds,
    )
    health = client.health()
    records = []
    invalid = 0
    for path in files:
        episode = json.loads(path.read_text(encoding="utf-8"))
        episode_id = str(episode.get("episode_id"))
        source_episode = asset.get(episode_id)
        if source_episode is None:
            raise ValueError(f"episode {episode_id} is absent from R2R asset")
        expected_scene = scene_name(str(source_episode["scene_id"]))
        if scene_name(str(episode.get("scene_id", ""))) != expected_scene:
            raise ValueError(
                f"episode {episode_id} scene does not match the converted train asset"
            )
        goal = _goal_habitat_position(source_episode)
        scene_id = source_episode["scene_id"]
        radius = float(source_episode["goals"][0].get("radius", 3.0))
        for index, transition in enumerate(episode.get("transitions", [])):
            source_contract = (
                transition.get("vlnce_source_split"),
                transition.get("vlnce_source_dataset_role"),
                transition.get("vlnce_coordinate_system"),
                transition.get("vlnce_source_metadata_sha256"),
                transition.get("vlnce_source_gt_sha256"),
            )
            expected_contract = (
                "train",
                "train",
                ISAAC_COORDINATE_SYSTEM,
                provenance["source_metadata_sha256"],
                provenance["source_gt_sha256"],
            )
            if source_contract != expected_contract:
                raise ValueError(
                    f"{path}: transition {index} does not match the exact "
                    "converted train source contract"
                )
            if (
                str(transition.get("episode_id")) != episode_id
                or scene_name(str(transition.get("scene_id", "")))
                != expected_scene
            ):
                raise ValueError(
                    f"{path}: transition {index} episode/scene identity mismatch"
                )
            if transition.get("history_sampling_policy") != NAVILA_HISTORY_SAMPLING_POLICY:
                raise ValueError(
                    f"{path}: transition {index} uses "
                    f"{transition.get('history_sampling_policy')!r}; "
                    f"expected {NAVILA_HISTORY_SAMPLING_POLICY!r}"
                )
            alignment = transition.get("frame_alignment")
            if not isinstance(alignment, list) or len(alignment) != NAVILA_VIDEO_FRAMES:
                raise ValueError(
                    f"{path}: transition {index} has invalid {NAVILA_VIDEO_FRAMES}-frame alignment"
                )
            pose = transition.get("isaac_pose_before")
            if not isinstance(pose, dict):
                raise ValueError(f"{path}: transition {index} has no Isaac pose")
            position = pose.get("position")
            if not isinstance(position, list) or len(position) != 3:
                raise ValueError(f"{path}: transition {index} has invalid pose")
            request_id = f"native-oracle-{episode_id}-{index}-{uuid.uuid4().hex}"
            rendered = client.render(
                {
                    "request_id": request_id,
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "physics_step": int(
                        transition.get("frame_alignment", [{}])[-1].get("physics_step", 0)
                    ),
                    "isaac_pose": pose,
                    "habitat_position": list(isaac_position_to_habitat(position)),
                    "habitat_yaw": isaac_yaw_to_habitat_yaw(float(pose["yaw"])),
                    "goal_position": goal,
                    "success_distance_m": radius,
                }
            )
            metadata = rendered.metadata
            action_id = metadata.get("dynamic_oracle_action", {})
            action_id = action_id.get("action_id") if isinstance(action_id, dict) else None
            valid = bool(metadata.get("oracle_valid", False) and action_id is not None)
            if valid:
                action = action_from_id(int(action_id))
                action_payload = {"action_id": action.action_id, "text": action.text}
            else:
                invalid += 1
                action_payload = None
            records.append(
                {
                    "episode_id": episode_id,
                    "source_episode": str(path),
                    "transition_index": index,
                    "observation_key": transition.get("observation_key"),
                    "strict_observation_state_alignment": bool(
                        transition.get("strict_observation_state_alignment", False)
                    ),
                    "frame_alignment": transition.get("frame_alignment"),
                    "isaac_pose_before": pose,
                    "oracle_valid": valid,
                    "oracle_eligible": valid,
                    "oracle_action_id": action_payload["action_id"] if action_payload else None,
                    "oracle_action": action_payload,
                    "teacher_source": "habitat_navmesh_shortest_path",
                    "source_contract": {
                        "source_split": provenance["source_split"],
                        "dataset_role": provenance["dataset_role"],
                        "coordinate_system": provenance["coordinate_system"],
                        "source_metadata_sha256": provenance[
                            "source_metadata_sha256"
                        ],
                        "source_gt_sha256": provenance["source_gt_sha256"],
                    },
                    "navmesh_metadata": {
                        key: value
                        for key, value in metadata.items()
                        if key
                        in {
                            "scene_id",
                            "physics_step",
                            "geodesic_distance",
                            "relative_path_bearing",
                            "navigation_reward_valid",
                            "oracle_invalid_reason",
                            "navmesh_snap_distance",
                            "nearest_navmesh_point",
                            "start_snap_valid",
                            "goal_snap_valid",
                            "request_id",
                        }
                    },
                }
            )
        print(f"annotated episode={episode_id} transitions={len(episode.get('transitions', []))}", flush=True)
    result = {
        "schema_version": "safe-vln-native-oracle-v1",
        "teacher_source": "habitat_navmesh_shortest_path",
        "diagnostic_only": True,
        "source_contract": {
            key: provenance[key]
            for key in (
                "source_split",
                "dataset_role",
                "coordinate_system",
                "source_metadata_sha256",
                "source_gt_sha256",
            )
        },
        "r2r_data_path": str(Path(args.r2r_data_path).expanduser().resolve()),
        "render_health": health,
        "episodes": sorted({record["episode_id"] for record in records}),
        "transitions": len(records),
        "oracle_valid": sum(bool(record["oracle_valid"]) for record in records),
        "oracle_invalid": invalid,
        "records": records,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(output)
    print(json.dumps({k: result[k] for k in ("schema_version", "episodes", "transitions", "oracle_valid", "oracle_invalid")}))
    return result


def main() -> int:
    args = _parser().parse_args()
    annotate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
