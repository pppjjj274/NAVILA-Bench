#!/usr/bin/env python3
"""Executable entry point for the Go2 Safe-VLN pipeline."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

from safe_vln import train as pipeline
from safe_vln.calibration import fit_cost_profile, read_calibration_records
from safe_vln.objective import save_cost_profile
from safe_vln.replay import DEFAULT_VLNCE_TRAIN_METADATA
from safe_vln.goal_stop import GOAL_STOP_MODES
from safe_vln.dataset import iter_sample_refs
from safe_vln.sampling import select_risk_episodes
from safe_vln.vlnce_dataset import load_isaac_vlnce_payload


def add_eval_args(parser):
    parser.add_argument(
        "--r2r-data-path",
        default="isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz",
    )
    parser.add_argument("--low-level-policy-dir", default="2024-09-25_23-22-02")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--vlm-host", default="localhost")
    parser.add_argument("--vlm-port", type=int, default=54321)
    parser.add_argument("--vlm-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--cost-limit", type=float)
    parser.add_argument("--safe-cost-profile")
    parser.add_argument("--blocked-seconds", type=float, default=2.0)
    parser.add_argument("--blocked-distance", type=float, default=0.10)
    parser.add_argument(
        "--turn-min-expected-angle",
        type=float,
        default=0.18,
        help="Minimum requested yaw (rad) before proportional turn checking.",
    )
    parser.add_argument("--turn-min-achieved-ratio", type=float, default=0.25)
    parser.add_argument("--dataset-dir")
    parser.add_argument("--safe-replay", action="store_true")
    parser.add_argument("--safe-replay-root")
    parser.add_argument("--safe-replay-annotations")
    parser.add_argument("--safe-replay-id", type=int)
    parser.add_argument("--safe-replay-ids", type=int, nargs="+")
    parser.add_argument(
        "--safe-replay-vlnce-metadata",
        default=str(DEFAULT_VLNCE_TRAIN_METADATA),
    )
    parser.add_argument("--safe-replay-vlnce-gt")
    parser.add_argument("--safe-replay-legacy-unpaired", action="store_true")
    parser.add_argument("--safe-policy-tag")
    parser.add_argument("--online-round", type=int)
    parser.add_argument("--safe-live-render", action="store_true")
    parser.add_argument("--render-host", default="127.0.0.1")
    parser.add_argument("--render-port", type=int, default=54322)
    parser.add_argument("--render-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--mp3d-scenes-root")
    parser.add_argument("--vlnce-episode-id", type=int)
    parser.add_argument("--vlnce-episode-ids", type=int, nargs="+")
    parser.add_argument("--vlnce-metadata")
    parser.add_argument("--vlnce-gt")
    parser.add_argument("--dataset-role", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--goal-stop-mode",
        choices=GOAL_STOP_MODES,
        default="policy",
    )
    parser.add_argument(
        "--collection-policy",
        choices=("vlm", "oracle"),
        default="vlm",
    )
    parser.add_argument("--allow-online-oracle", action="store_true")
    parser.add_argument("--missed-stop-penalty", type=float, default=-0.5)
    parser.add_argument("--missed-stop-patience", type=int, default=3)
    parser.add_argument("--max-vlm-calls", type=int)
    parser.add_argument("--max-episode-seconds", type=float)
    parser.add_argument("--calibration-dir")


def _python_launcher():
    loader = os.environ.get("GLIBC_LOADER")
    glibc_lib = os.environ.get("GLIBC_LIB")
    if loader and glibc_lib:
        conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
        library_path = (
            f"{glibc_lib}:{conda_prefix}/lib:/lib64:/usr/lib64"
        )
        return [loader, "--library-path", library_path, sys.executable]
    return [sys.executable]


def _resolve_episode_plan(args, episode_count):
    if getattr(args, "safe_live_render", False):
        if args.safe_replay:
            raise ValueError(
                "--safe-live-render cannot be combined with --safe-replay"
            )
        episode_ids = (
            list(args.vlnce_episode_ids)
            if getattr(args, "vlnce_episode_ids", None) is not None
            else (
                [args.vlnce_episode_id]
                if getattr(args, "vlnce_episode_id", None) is not None
                else []
            )
        )
        if (
            getattr(args, "vlnce_episode_id", None) is not None
            and getattr(args, "vlnce_episode_ids", None) is not None
        ):
            raise ValueError(
                "use only one of --vlnce-episode-id and --vlnce-episode-ids"
            )
        if not episode_ids:
            raise ValueError(
                "--safe-live-render requires a VLN-CE episode ID"
            )
        end = args.start_idx + len(episode_ids)
        if args.end_idx is not None and args.end_idx != end:
            raise ValueError(
                "--end-idx must equal --start-idx plus the number of "
                "VLN-CE episode IDs"
            )
        return episode_ids, end
    if args.safe_replay:
        replay_ids = (
            list(args.safe_replay_ids)
            if args.safe_replay_ids is not None
            else (
                [args.safe_replay_id]
                if args.safe_replay_id is not None
                else []
            )
        )
        if args.safe_replay_id is not None and args.safe_replay_ids is not None:
            raise ValueError(
                "use only one of --safe-replay-id and --safe-replay-ids"
            )
        if not replay_ids:
            raise ValueError(
                "--safe-replay requires --safe-replay-id or --safe-replay-ids"
            )
        end = args.start_idx + len(replay_ids)
        if args.end_idx is not None and args.end_idx != end:
            raise ValueError(
                "--end-idx must equal --start-idx plus the number of "
                "Safe-Replay IDs"
            )
        if end > episode_count:
            raise ValueError("Safe-Replay physical episode range is out of bounds")
        return replay_ids, end

    if args.safe_replay_id is not None or args.safe_replay_ids is not None:
        raise ValueError("replay IDs require --safe-replay")
    if (
        getattr(args, "vlnce_episode_id", None) is not None
        or getattr(args, "vlnce_episode_ids", None) is not None
    ):
        raise ValueError("VLN-CE episode IDs require --safe-live-render")
    end = (
        episode_count
        if args.end_idx is None
        else min(args.end_idx, episode_count)
    )
    return None, end


def _preflight_live_assets(args, episode_ids):
    metadata_path = Path(args.vlnce_metadata).expanduser()
    with gzip.open(metadata_path, "rt", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list):
        raise ValueError(f"VLN-CE metadata has no episodes list: {metadata_path}")
    by_id = {
        str(item.get("episode_id")): item
        for item in episodes
        if isinstance(item, dict)
    }
    requested = [str(value) for value in episode_ids]
    missing_ids = [value for value in requested if value not in by_id]
    if missing_ids:
        raise ValueError(
            f"VLN-CE metadata is missing requested IDs: {missing_ids[:10]}"
        )
    scenes_root = Path(args.mp3d_scenes_root).expanduser()
    missing_assets = []
    for scene_id in sorted(
        {
            os.path.splitext(
                os.path.basename(str(by_id[value]["scene_id"]))
            )[0]
            for value in requested
        }
    ):
        scene_root = scenes_root / scene_id
        for suffix in (".glb", ".navmesh"):
            path = scene_root / f"{scene_id}{suffix}"
            if not path.is_file():
                missing_assets.append(str(path))
    if missing_assets:
        preview = "\n".join(missing_assets[:10])
        raise RuntimeError(
            f"selected live-render episodes need {len(missing_assets)} missing "
            f"MP3D assets:\n{preview}"
        )


def run_episodes(args):
    if args.blocked_seconds <= 0:
        raise ValueError("--blocked-seconds must be positive")
    if args.blocked_distance <= 0:
        raise ValueError("--blocked-distance must be positive")
    if args.turn_min_expected_angle < 0:
        raise ValueError("--turn-min-expected-angle must be non-negative")
    if not 0 <= args.turn_min_achieved_ratio <= 1:
        raise ValueError("--turn-min-achieved-ratio must be in [0, 1]")
    if args.max_vlm_calls is not None and args.max_vlm_calls <= 0:
        raise ValueError("--max-vlm-calls must be positive")
    if args.max_episode_seconds is not None and args.max_episode_seconds <= 0:
        raise ValueError("--max-episode-seconds must be positive")
    if args.render_timeout_seconds <= 0:
        raise ValueError("--render-timeout-seconds must be positive")
    if args.vlm_timeout_seconds <= 0:
        raise ValueError("--vlm-timeout-seconds must be positive")
    if args.missed_stop_patience <= 0:
        raise ValueError("--missed-stop-patience must be positive")
    if not args.safe_replay and not args.safe_live_render:
        native_payload = load_isaac_vlnce_payload(
            args.r2r_data_path,
            expected_role=args.dataset_role,
            expected_scene_count=61 if args.dataset_role == "train" else None,
        )
        episodes = native_payload["episodes"]
    else:
        with gzip.open(args.r2r_data_path, "rt", encoding="utf-8") as file:
            episodes = json.load(file)["episodes"]
    replay_ids, end = _resolve_episode_plan(args, len(episodes))
    if args.safe_replay:
        if args.safe_replay_root is None:
            raise ValueError(
                "--safe-replay requires --safe-replay-root"
            )
    if args.safe_live_render:
        if not args.vlnce_metadata or not args.mp3d_scenes_root:
            raise ValueError(
                "--safe-live-render requires --vlnce-metadata and "
                "--mp3d-scenes-root"
            )
        metadata_split = Path(args.vlnce_metadata).expanduser().parent.name
        if metadata_split == "val_unseen" and args.dataset_role != "eval":
            raise ValueError(
                "val_unseen live-render data requires --dataset-role=eval"
            )
        if args.collection_policy == "oracle" and args.dataset_role != "train":
            raise ValueError(
                "--collection-policy=oracle is allowed only for train data"
            )
        if args.collection_policy == "oracle" and not args.allow_online_oracle:
            raise ValueError(
                "live dynamic Oracle is paused; use paired offline labels or "
                "pass --allow-online-oracle for an ablation"
            )
        if args.collection_policy == "oracle" and args.goal_stop_mode != "policy":
            raise ValueError(
                "--collection-policy=oracle requires --goal-stop-mode=policy"
            )
        _preflight_live_assets(args, replay_ids)
    if not 0 <= args.start_idx < end:
        raise ValueError("invalid episode range")

    for index in range(args.start_idx, end):
        command = [
            *_python_launcher(),
            "scripts/navila_eval.py",
            "--task=go2_matterport_vision",
            "--num_envs=1",
            "--history_length=9",
            f"--load_run={args.low_level_policy_dir}",
            "--headless",
            f"--r2r-data-path={args.r2r_data_path}",
            f"--episode_idx={index}",
            f"--vlm_host={args.vlm_host}",
            f"--vlm_port={args.vlm_port}",
            f"--vlm-timeout-seconds={args.vlm_timeout_seconds}",
            "--safe-vln",
            f"--safe-blocked-seconds={args.blocked_seconds}",
            f"--safe-blocked-distance={args.blocked_distance}",
            f"--safe-turn-min-expected-angle={args.turn_min_expected_angle}",
            f"--safe-turn-min-achieved-ratio={args.turn_min_achieved_ratio}",
            f"--goal-stop-mode={args.goal_stop_mode}",
            f"--collection-policy={args.collection_policy}",
            *(["--allow-online-oracle"] if args.allow_online_oracle else []),
            f"--missed-stop-penalty={args.missed_stop_penalty}",
            f"--missed-stop-patience={args.missed_stop_patience}",
        ]
        if args.cost_limit is not None:
            command.append(f"--safe-cost-limit={args.cost_limit}")
        if args.safe_cost_profile:
            command.append(f"--safe-cost-profile={args.safe_cost_profile}")
        if args.calibration_dir:
            calibration_dir = Path(args.calibration_dir).expanduser()
            calibration_dir.mkdir(parents=True, exist_ok=True)
            replay_label = (
                replay_ids[index - args.start_idx]
                if replay_ids is not None
                else index
            )
            command.append(
                "--safe-calibration-file="
                f"{calibration_dir / f'episode_{replay_label}.jsonl'}"
            )
        if args.safe_replay:
            replay_id = replay_ids[index - args.start_idx]
            command.extend(
                [
                    "--safe-replay",
                    f"--safe-replay-root={args.safe_replay_root}",
                    f"--safe-replay-id={replay_id}",
                ]
            )
            if args.safe_replay_annotations:
                command.append(
                    f"--safe-replay-annotations={args.safe_replay_annotations}"
                )
            if args.safe_replay_vlnce_metadata:
                command.append(
                    "--safe-replay-vlnce-metadata="
                    f"{args.safe_replay_vlnce_metadata}"
                )
            if args.safe_replay_vlnce_gt:
                command.append(
                    f"--safe-replay-vlnce-gt={args.safe_replay_vlnce_gt}"
                )
            if args.safe_replay_legacy_unpaired:
                command.append("--safe-replay-legacy-unpaired")
            if args.safe_policy_tag:
                command.append(f"--safe-policy-tag={args.safe_policy_tag}")
            if args.online_round is not None:
                command.append(f"--online-round={args.online_round}")
        elif args.safe_live_render:
            vlnce_id = replay_ids[index - args.start_idx]
            command.extend(
                [
                    "--safe-live-render",
                    f"--vlnce-episode-id={vlnce_id}",
                    f"--vlnce-metadata={args.vlnce_metadata}",
                    f"--mp3d-scenes-root={args.mp3d_scenes_root}",
                    f"--render-host={args.render_host}",
                    f"--render-port={args.render_port}",
                    (
                        "--render-timeout-seconds="
                        f"{args.render_timeout_seconds}"
                    ),
                    f"--dataset-role={args.dataset_role}",
                ]
            )
            if args.vlnce_gt:
                command.append(f"--vlnce-gt={args.vlnce_gt}")
            if args.safe_policy_tag:
                command.append(f"--safe-policy-tag={args.safe_policy_tag}")
            if args.online_round is not None:
                command.append(f"--online-round={args.online_round}")
        else:
            command.append("--enable_cameras")
        if args.dataset_dir:
            command.append(f"--safe-dataset-dir={args.dataset_dir}")
        if args.max_vlm_calls is not None:
            command.append(f"--max_vlm_calls={args.max_vlm_calls}")
        if args.max_episode_seconds is not None:
            command.append(
                f"--max_episode_seconds={args.max_episode_seconds}"
            )
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


def summarize(args):
    files = sorted(Path(args.measurement_dir).glob("*.json"))
    if not files:
        raise RuntimeError("no measurement JSON files found")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    rows = [
        payload.get("summary", payload)
        if isinstance(payload, dict)
        else {}
        for payload in payloads
    ]

    def mean(key):
        return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows)

    costs = sorted(
        float(row.get("constraint_cost", row.get("cumulative_cost", 0.0)))
        for row in rows
    )
    stop_recalls = [
        float(row["stop_recall_in_goal"])
        for row in rows
        if row.get("stop_recall_in_goal") is not None
    ]
    minimum_distances = [
        float(row["minimum_distance_to_goal_m"])
        for row in rows
        if row.get("minimum_distance_to_goal_m") is not None
    ]

    def percentile(value):
        return costs[round((len(costs) - 1) * value)]

    report = {
        "episodes": len(rows),
        "success_rate": mean("success"),
        "spl": mean("spl"),
        "safe_success_rate": mean("safe_success"),
        "safe_spl": mean("safe_spl"),
        "mean_reward": mean("total_high_level_reward"),
        "mean_cost": (
            sum(
                float(row.get("constraint_cost", row.get("cumulative_cost", 0.0)))
                for row in rows
            )
            / len(rows)
        ),
        "mean_cumulative_cost": mean("cumulative_cost"),
        "mean_hard_cost": mean("cumulative_hard_cost"),
        "mean_dense_cost": mean("cumulative_dense_cost"),
        "hard_violation_rate": sum(
            float(row.get("cumulative_hard_cost", 0.0) or 0.0) > 0
            for row in rows
        )
        / len(rows),
        "zero_cost_rate": sum(cost == 0 for cost in costs) / len(costs),
        "collision_rate": mean("has_collision"),
        "blocked_rate": mean("has_blocked"),
        "turn_tracking_failure_rate": mean("has_turn_tracking_failure"),
        # Deprecated output alias for readers of pre-v4 result summaries.
        "turn_blocked_rate": mean("has_turn_tracking_failure"),
        "constraint_satisfaction_rate": mean("constraint_satisfied"),
        "cost_p90": percentile(0.90),
        "cost_p95": percentile(0.95),
        "cost_p99": percentile(0.99),
        "cost_max": costs[-1],
        "policy_success_rate": mean("policy_success"),
        "system_success_rate": mean("system_success"),
        "policy_system_success_gap": (
            mean("system_success") - mean("policy_success")
        ),
        "entered_goal_rate": mean("entered_goal_radius"),
        "mean_goal_dwell_decisions": mean("goal_dwell_decisions"),
        "mean_oracle_stop_decisions": mean("oracle_stop_decisions"),
        "mean_model_stop_decisions": mean("model_stop_decisions"),
        "mean_missed_stop_count": mean("missed_stop_count"),
        "stop_recall_in_goal": (
            sum(stop_recalls) / len(stop_recalls) if stop_recalls else None
        ),
        "mean_minimum_distance_to_goal_m": (
            sum(minimum_distances) / len(minimum_distances)
            if minimum_distances
            else None
        ),
        "shield_intervention_rate": sum(
            float(row.get("shield_intervention_count", 0.0) or 0.0) > 0
            for row in rows
        )
        / len(rows),
        "goal_escape_rate": mean("goal_escape_after_entry"),
    }
    output = Path(args.output or Path(args.measurement_dir).parent / "safe_vln_summary.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def add_model_args(parser, *, train=False):
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--training-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--reset-critics", action="store_true")
    parser.add_argument(
        "--sampling-strategy",
        choices=(
            ("sequential", "balanced-ppo")
            if train
            else ("sequential", "balanced-critic")
        ),
        default="sequential",
    )
    parser.add_argument("--sampling-seed", type=int, default=20260729)
    if train:
        parser.add_argument("--rollout-dir", required=True)
        parser.add_argument("--actor-lr", type=float, default=1e-5)
        parser.add_argument("--cost-limit", type=float)
        parser.add_argument("--clip-ratio", type=float, default=0.1)
        parser.add_argument("--ppo-epochs", type=int, default=4)
        parser.add_argument("--mini-batch-size", type=int, default=16)
        parser.add_argument(
            "--gradient-accumulation-steps",
            type=int,
            default=8,
            help="Accumulate Safe-PPO micro-batches before one optimizer update.",
        )
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--gae-lambda", type=float, default=0.95)
        parser.add_argument("--lagrange-lr", type=float, default=0.035)
        parser.add_argument("--initial-lagrange-multiplier", type=float)
        parser.add_argument("--policy-version", type=int, default=0)
        parser.add_argument("--oracle-ce-coef", type=float, default=0.05)
        parser.add_argument("--oracle-stop-weight", type=float, default=5.0)
    else:
        parser.add_argument("--dataset-dir", required=True)
        parser.add_argument("--epochs", type=int, default=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Go2 Safe-VLN")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    add_eval_args(collect)
    evaluate = subparsers.add_parser("evaluate")
    add_eval_args(evaluate)
    calibrate = subparsers.add_parser("calibrate-safety")
    add_eval_args(calibrate)
    calibrate.add_argument("--output-profile", required=True)
    risk = subparsers.add_parser("select-risk-episodes")
    risk.add_argument("--dataset-dir", required=True)
    risk.add_argument("--output", required=True)
    risk.add_argument("--split", default="train")
    risk.add_argument("--per-stratum", type=int, default=20)
    risk.add_argument("--max-per-scene", type=int, default=2)
    warmup = subparsers.add_parser("warmup-critics")
    add_model_args(warmup)
    actor_warmup = subparsers.add_parser("warmup-actor")
    actor_warmup.add_argument("--model-path", required=True)
    actor_warmup.add_argument("--dataset-dir", required=True)
    actor_warmup.add_argument("--output-dir", required=True)
    actor_warmup.add_argument("--device", default="cuda")
    actor_warmup.add_argument(
        "--training-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    actor_warmup.add_argument("--split", default="train")
    actor_warmup.add_argument(
        "--actor-architecture",
        choices=(
            "hierarchical-stop-direction-magnitude",
            "hierarchical-stop-motion",
            "candidate-scoring",
        ),
        default="hierarchical-stop-motion",
    )
    actor_warmup.add_argument(
        "--actor-target-source",
        choices=("oracle", "navila-policy"),
        default="navila-policy",
        help=(
            "Supervise from strict dynamic-oracle labels, or distill the "
            "original NaViLA greedy policy recorded by native/live collection."
        ),
    )
    actor_warmup.add_argument("--actor-lr", type=float, default=1e-6)
    actor_warmup.add_argument("--head-lr", type=float, default=1e-4)
    actor_warmup.add_argument("--head-warmup-lr", type=float, default=3e-4)
    actor_warmup.add_argument("--head-warmup-epochs", type=int, default=20)
    actor_warmup.add_argument("--head-batch-size", type=int, default=256)
    actor_warmup.add_argument("--gradient-accumulation-steps", type=int, default=4)
    actor_warmup.add_argument("--max-grad-norm", type=float, default=0.5)
    actor_warmup.add_argument("--stop-fraction", type=float, default=0.10)
    actor_warmup.add_argument(
        "--hard-stop-negative-fraction", type=float, default=0.25
    )
    actor_warmup.add_argument(
        "--hard-stop-negative-margin-m", type=float, default=1.0
    )
    actor_warmup.add_argument("--stop-threshold", type=float, default=0.5)
    actor_warmup.add_argument(
        "--calibration-episodes-per-scene", type=int, default=1
    )
    actor_warmup.add_argument(
        "--audit-episodes-per-scene", type=int, default=1
    )
    actor_warmup.add_argument(
        "--dev-episodes-per-scene",
        type=int,
        help="deprecated alias overriding --audit-episodes-per-scene",
    )
    actor_warmup.add_argument(
        "--calibrate-stop-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    actor_warmup.add_argument(
        "--stop-threshold-grid-step", type=float, default=0.01
    )
    actor_warmup.add_argument("--allow-small-dataset", action="store_true")
    actor_warmup.add_argument("--epochs", type=int, default=1)
    actor_warmup.add_argument("--mini-batch-size", type=int, default=1)
    actor_warmup.add_argument("--max-samples", type=int)
    actor_warmup.add_argument("--oracle-stop-weight", type=float, default=5.0)
    actor_warmup.add_argument(
        "--sampling-strategy",
        choices=("sequential", "balanced-oracle", "stratified"),
        default="stratified",
    )
    actor_warmup.add_argument("--sampling-seed", type=int, default=20260729)
    actor_warmup.add_argument(
        "--minimum-stop-accuracy",
        type=float,
        default=0.5,
    )
    actor_warmup.add_argument(
        "--maximum-false-stop-rate",
        type=float,
        default=0.05,
    )
    actor_warmup.add_argument(
        "--minimum-non-stop-macro-accuracy",
        type=float,
        default=0.4,
    )
    dagger_actor = subparsers.add_parser("dagger-actor")
    dagger_actor.add_argument("--model-path", required=True)
    dagger_actor.add_argument("--checkpoint", required=True)
    dagger_actor.add_argument("--rollout-dir", required=True)
    dagger_actor.add_argument("--anchor-dataset-dir", required=True)
    dagger_actor.add_argument("--output-dir", required=True)
    dagger_actor.add_argument("--device", default="cuda")
    dagger_actor.add_argument(
        "--training-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    dagger_actor.add_argument("--split", default="train")
    dagger_actor.add_argument("--actor-lr", type=float, default=1e-6)
    dagger_actor.add_argument("--head-lr", type=float, default=1e-4)
    dagger_actor.add_argument("--gradient-accumulation-steps", type=int, default=4)
    dagger_actor.add_argument("--max-grad-norm", type=float, default=0.5)
    dagger_actor.add_argument("--epochs", type=int, default=1)
    dagger_actor.add_argument("--max-samples", type=int, default=4000)
    dagger_actor.add_argument("--online-fraction", type=float, default=0.60)
    dagger_actor.add_argument("--sampling-seed", type=int, default=20260802)
    dagger_actor.add_argument("--online-round", type=int, default=1)
    dagger_actor.add_argument("--allow-small-dataset", action="store_true")
    actor_audit = subparsers.add_parser("audit-actor")
    actor_audit.add_argument("--model-path", required=True)
    actor_audit.add_argument("--checkpoint", required=True)
    actor_audit.add_argument("--dataset-dir", required=True)
    actor_audit.add_argument("--output-dir", required=True)
    actor_audit.add_argument("--device", default="cuda")
    actor_audit.add_argument(
        "--training-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    actor_audit.add_argument("--split", default="train")
    actor_audit.add_argument("--dev-episodes-per-scene", type=int, default=1)
    actor_audit.add_argument("--allow-small-dataset", action="store_true")
    actor_audit.add_argument("--sampling-seed", type=int, default=20260729)
    actor_audit.add_argument(
        "--minimum-stop-accuracy", type=float, default=0.5
    )
    actor_audit.add_argument(
        "--maximum-false-stop-rate", type=float, default=0.05
    )
    actor_audit.add_argument(
        "--stop-threshold-grid-step", type=float, default=0.01
    )
    actor_audit.add_argument(
        "--goal-stop-contract",
        choices=("policy-v1", "sensor-gated-v1"),
        default="policy-v1",
    )
    actor_audit.add_argument("--certify", action="store_true")
    actor_audit.add_argument(
        "--minimum-non-stop-macro-accuracy", type=float, default=0.4
    )
    train = subparsers.add_parser("train")
    add_model_args(train, train=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--measurement-dir", required=True)
    summary.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "select-risk-episodes":
        selected = select_risk_episodes(
            iter_sample_refs(args.dataset_dir, args.split),
            per_stratum=args.per_stratum,
            max_per_scene=args.max_per_scene,
        )
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
        ids_path = output.with_suffix(".txt")
        ids_path.write_text(
            "\n".join(row["episode_id"] for row in selected) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"episodes": len(selected), "output": str(output), "ids": str(ids_path)}))
        return 0
    if args.command == "collect":
        if not args.dataset_dir:
            raise ValueError("collect requires --dataset-dir")
        return run_episodes(args)
    if args.command == "evaluate":
        return run_episodes(args)
    if args.command == "calibrate-safety":
        calibration_ids = (
            args.safe_replay_ids
            if args.safe_replay
            else args.vlnce_episode_ids
            if args.safe_live_render
            else None
        )
        if calibration_ids is None:
            raise ValueError(
                "calibrate-safety requires either --safe-replay-ids or "
                "--safe-live-render with --vlnce-episode-ids"
            )
        if len(calibration_ids) != 80:
            raise ValueError("calibrate-safety requires exactly 80 episodes")
        if not args.calibration_dir:
            raise ValueError("calibrate-safety requires --calibration-dir")
        calibration_dir = Path(args.calibration_dir).expanduser()
        if calibration_dir.exists() and any(calibration_dir.glob("*.jsonl")):
            raise ValueError(
                "calibration directory already contains JSONL records; "
                "use a new directory"
            )
        result = run_episodes(args)
        if result:
            return result
        calibration_files = sorted(calibration_dir.glob("*.jsonl"))
        if len(calibration_files) != 80:
            raise RuntimeError(
                f"expected 80 calibration files, found {len(calibration_files)}"
            )
        profile = fit_cost_profile(
            read_calibration_records(calibration_dir),
            calibration_episodes=80,
            minimum_recorded_episodes=20,
        )
        output = save_cost_profile(profile, args.output_profile)
        print(json.dumps({"cost_profile": str(output), "fingerprint": profile["fingerprint"]}))
        return 0
    if args.command == "warmup-critics":
        return pipeline.warmup(args)
    if args.command == "warmup-actor":
        return pipeline.warmup_actor(args)
    if args.command == "dagger-actor":
        return pipeline.dagger_actor(args)
    if args.command == "audit-actor":
        return pipeline.audit_actor(args)
    if args.command == "train":
        return pipeline.train(args)
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
