# siara_upper_vln_sandbox

**Independent sandbox** for the upper-level VLM-based navigation pipeline.

This directory is self-contained and does **not** modify any existing
project files.  It is safe to experiment here without affecting other
team members' code.

## What's inside

| File | Purpose |
|---|---|
| `action_parser.py` | Parse raw VLM text → structured `{action, value}` |
| `command_mapper.py` | Convert structured action → low-level `{vx, vy, wz, duration}` |
| `vlm_agent.py` | Mock VLM agent (TODO: replace with Qwen3-VL-8B-Instruct) |
| `run_upper_demo.py` | End-to-end dry-run of the upper pipeline |
| `run_with_policy.py` | Future entry point with policy integration (currently dry-run only) |
| `tests/` | Unit tests for parser and mapper |

## Quick start — dry-run

```bash
# Full pipeline demo (no arguments, no Isaac)
python siara_upper_vln_sandbox/run_upper_demo.py "go forward"

# With --dry-run flag
python siara_upper_vln_sandbox/run_with_policy.py \
    --instruction "turn right and go to the door" \
    --mock --dry-run
```

## Running tests

```bash
PYTHONPATH=siara_upper_vln_sandbox pytest siara_upper_vln_sandbox/tests -q
```

## Important warnings

- **Do NOT run Isaac Sim on the login node.**  Real simulation requires a
  GPU node.  Use `srun` or `sbatch` with an appropriate GPU partition.
- This sandbox currently uses a **mock VLM**.  Real Qwen3-VL-8B-Instruct
  inference has not been integrated yet.
- `run_with_policy.py` is **dry-run only** at this stage.  The
  `run_policy_control_loop()` function is a placeholder with TODO stubs.
