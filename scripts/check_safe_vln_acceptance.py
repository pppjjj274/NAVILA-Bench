#!/usr/bin/env python3
"""Fail closed before constrained PPO when online actor evaluation regresses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "success_rate",
    "mean_cost",
    "hard_violation_rate",
    "blocked_rate",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    parser.add_argument("--minimum-success-rate", type=float, default=2 / 22)
    parser.add_argument("--maximum-mean-cost", type=float, default=0.9253261279243038)
    parser.add_argument("--maximum-hard-violation-rate", type=float, default=12 / 22)
    parser.add_argument("--maximum-blocked-rate", type=float, default=1 / 22)
    return parser.parse_args()


def read_summary(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return payload


def main():
    args = parse_args()
    baseline = read_summary(args.baseline)
    candidate = read_summary(args.candidate)
    missing = [key for key in METRICS if key not in candidate]
    if missing:
        raise ValueError(f"candidate summary lacks metrics: {missing}")
    checks = {
        "success_rate": float(candidate["success_rate"]) >= args.minimum_success_rate,
        "mean_cost": float(candidate["mean_cost"]) <= args.maximum_mean_cost,
        "hard_violation_rate": float(candidate["hard_violation_rate"])
        <= args.maximum_hard_violation_rate,
        "blocked_rate": float(candidate["blocked_rate"])
        <= args.maximum_blocked_rate,
    }
    report = {
        "accepted": all(checks.values()),
        "baseline": {key: baseline.get(key) for key in METRICS},
        "candidate": {key: candidate.get(key) for key in METRICS},
        "thresholds": {
            "minimum_success_rate": args.minimum_success_rate,
            "maximum_mean_cost": args.maximum_mean_cost,
            "maximum_hard_violation_rate": args.maximum_hard_violation_rate,
            "maximum_blocked_rate": args.maximum_blocked_rate,
        },
        "checks": checks,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
