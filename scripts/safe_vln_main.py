#!/usr/bin/env python3
"""Executable entry point for the Go2 Safe-VLN pipeline."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys

from safe_vln import train as pipeline


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
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument("--dataset-dir")


def run_episodes(args):
    with gzip.open(args.r2r_data_path, "rt", encoding="utf-8") as file:
        episodes = json.load(file)["episodes"]
    end = len(episodes) if args.end_idx is None else min(args.end_idx, len(episodes))
    if not 0 <= args.start_idx < end:
        raise ValueError("invalid episode range")

    for index in range(args.start_idx, end):
        command = [
            sys.executable,
            "scripts/navila_eval.py",
            "--task=go2_matterport_vision",
            "--num_envs=1",
            "--history_length=9",
            f"--load_run={args.low_level_policy_dir}",
            "--headless",
            "--enable_cameras",
            f"--episode_idx={index}",
            f"--vlm_host={args.vlm_host}",
            f"--vlm_port={args.vlm_port}",
            "--safe-vln",
            f"--safe-cost-limit={args.cost_limit}",
        ]
        if args.dataset_dir:
            command.append(f"--safe-dataset-dir={args.dataset_dir}")
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


def summarize(args):
    files = sorted(Path(args.measurement_dir).glob("*.json"))
    if not files:
        raise RuntimeError("no measurement JSON files found")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]

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
        "zero_cost_rate": sum(cost == 0 for cost in costs) / len(costs),
        "collision_rate": mean("has_collision"),
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
    parser.add_argument("--split", default="train")
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int)
    if train:
        parser.add_argument("--rollout-dir", required=True)
        parser.add_argument("--actor-lr", type=float, default=1e-5)
        parser.add_argument("--cost-limit", type=float, default=0.1)
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
    if args.command == "warmup-critics":
        return pipeline.warmup(args)
    if args.command == "train":
        return pipeline.train(args)
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
