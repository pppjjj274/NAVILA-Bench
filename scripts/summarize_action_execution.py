#!/usr/bin/env python3
"""Summarize closed-loop action execution calibration JSONL records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def _records(path: Path):
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file_path in files:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = defaultdict(list)
    total_records = 0
    verified_contact_records = 0
    for record in _records(args.input.expanduser()):
        total_records += 1
        if record.get("contact_sensor_enabled") is True:
            verified_contact_records += 1
        action_id = record.get("action_id")
        execution = record.get("turn_execution")
        if action_id is None or not isinstance(execution, dict):
            continue
        try:
            action_id = int(action_id)
        except (TypeError, ValueError, OverflowError):
            continue
        if not 0 <= action_id <= 5:
            continue
        # One terminal row represents one completed macro action.  Earlier
        # rows are intentionally ignored to avoid overweighting long actions.
        if execution.get("active", False):
            continue
        rows[int(action_id)].append(execution)

    report = {
        "records": total_records,
        "verified_contact_sensor_records": verified_contact_records,
        "contact_sensor_verified_rate": (
            verified_contact_records / total_records if total_records else None
        ),
        "actions": {},
        "turn_actions": list(range(6)),
    }
    for action_id in range(10):
        executions = rows.get(action_id, [])
        ratios = [float(item.get("execution_ratio", 0.0)) for item in executions]
        report["actions"][str(action_id)] = {
            "macros": len(executions),
            "turn_tracking_failures": sum(
                bool(item.get("blocked", False)) for item in executions
            ),
            # Deprecated aliases: this is a controller tracking diagnostic,
            # not a collision/blocked safety event.
            "turn_blocked": sum(bool(item.get("blocked", False)) for item in executions),
            "turn_tracking_failure_rate": (
                sum(bool(item.get("blocked", False)) for item in executions)
                / len(executions)
                if executions
                else None
            ),
            "turn_blocked_rate": (
                sum(bool(item.get("blocked", False)) for item in executions)
                / len(executions)
                if executions
                else None
            ),
            "mean_execution_ratio": sum(ratios) / len(ratios) if ratios else None,
            "min_execution_ratio": min(ratios) if ratios else None,
            "direction_mismatch": sum(
                bool(item.get("direction_mismatch", False)) for item in executions
            ),
        }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
