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
    parser.add_argument("--cost-limit", type=float)
    parser.add_argument("--safe-cost-profile")
    parser.add_argument("--blocked-seconds", type=float, default=2.0)
    parser.add_argument("--blocked-distance", type=float, default=0.10)
    parser.add_argument("--dataset-dir")
    parser.add_argument("--safe-replay", action="store_true")
    parser.add_argument("--safe-replay-root")
    parser.add_argument("--safe-replay-annotations")
    parser.add_argument("--safe-replay-id", type=int)
    parser.add_argument("--safe-replay-ids", type=int, nargs="+")
    parser.add_argument("--safe-policy-tag")
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
    end = (
        episode_count
        if args.end_idx is None
        else min(args.end_idx, episode_count)
    )
    return None, end


def run_episodes(args):
    if args.blocked_seconds <= 0:
        raise ValueError("--blocked-seconds must be positive")
    if args.blocked_distance <= 0:
        raise ValueError("--blocked-distance must be positive")
    if args.max_vlm_calls is not None and args.max_vlm_calls <= 0:
        raise ValueError("--max-vlm-calls must be positive")
    if args.max_episode_seconds is not None and args.max_episode_seconds <= 0:
        raise ValueError("--max-episode-seconds must be positive")
    with gzip.open(args.r2r_data_path, "rt", encoding="utf-8") as file:
        episodes = json.load(file)["episodes"]
    replay_ids, end = _resolve_episode_plan(args, len(episodes))
    if args.safe_replay:
        if args.safe_replay_root is None:
            raise ValueError(
                "--safe-replay requires --safe-replay-root"
            )
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
            f"--episode_idx={index}",
            f"--vlm_host={args.vlm_host}",
            f"--vlm_port={args.vlm_port}",
            "--safe-vln",
            f"--safe-blocked-seconds={args.blocked_seconds}",
            f"--safe-blocked-distance={args.blocked_distance}",
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
            if args.safe_policy_tag:
                command.append(f"--safe-policy-tag={args.safe_policy_tag}")
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

    costs = sorted(float(row.get("cumulative_cost", 0.0)) for row in rows)

    def percentile(value):
        return costs[round((len(costs) - 1) * value)]

    report = {
        "episodes": len(rows),
        "success_rate": mean("success"),
        "spl": mean("spl"),
        "safe_success_rate": mean("safe_success"),
        "safe_spl": mean("safe_spl"),
        "mean_reward": mean("total_high_level_reward"),
        "mean_cost": mean("cumulative_cost"),
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
        "constraint_satisfaction_rate": mean("constraint_satisfied"),
        "cost_p90": percentile(0.90),
        "cost_p95": percentile(0.95),
        "cost_p99": percentile(0.99),
        "cost_max": costs[-1],
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
    if train:
        parser.add_argument("--rollout-dir", required=True)
        parser.add_argument("--actor-lr", type=float, default=1e-5)
        parser.add_argument("--cost-limit", type=float)
        parser.add_argument("--clip-ratio", type=float, default=0.1)
        parser.add_argument("--ppo-epochs", type=int, default=4)
        parser.add_argument("--mini-batch-size", type=int, default=16)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--gae-lambda", type=float, default=0.95)
        parser.add_argument("--lagrange-lr", type=float, default=0.035)
        parser.add_argument("--initial-lagrange-multiplier", type=float, default=0.001)
        parser.add_argument("--policy-version", type=int, default=0)
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
    warmup = subparsers.add_parser("warmup-critics")
    add_model_args(warmup)
    train = subparsers.add_parser("train")
    add_model_args(train, train=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--measurement-dir", required=True)
    summary.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "collect":
        if not args.dataset_dir:
            raise ValueError("collect requires --dataset-dir")
        return run_episodes(args)
    if args.command == "evaluate":
        return run_episodes(args)
    if args.command == "calibrate-safety":
        if not args.safe_replay or args.safe_replay_ids is None:
            raise ValueError(
                "calibrate-safety requires --safe-replay and --safe-replay-ids"
            )
        if len(args.safe_replay_ids) != 80:
            raise ValueError("calibrate-safety requires exactly 80 replay episodes")
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
    if args.command == "train":
        return pipeline.train(args)
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
