#!/usr/bin/env bash

# Preserve the physical GPU tokens selected by Slurm before worker processes
# narrow CUDA_VISIBLE_DEVICES. Hard-coding logical 0/1 discards Slurm's
# allocation (which may be physical GPUs 3/6 or UUIDs) and can make workers
# collide on a GPU they do not own.
safe_vln_capture_allocated_gpus() {
    local expected_count="$1"
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "Slurm did not provide CUDA_VISIBLE_DEVICES" >&2
        return 1
    fi
    IFS=',' read -r -a SAFE_VLN_ALLOCATED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
    if [[ "${#SAFE_VLN_ALLOCATED_GPUS[@]}" -ne "$expected_count" ]]; then
        echo "Expected $expected_count allocated GPUs, got ${#SAFE_VLN_ALLOCATED_GPUS[@]}: $CUDA_VISIBLE_DEVICES" >&2
        return 1
    fi
}

safe_vln_gpu_token() {
    local index="$1"
    if [[ "$index" -lt 0 || "$index" -ge "${#SAFE_VLN_ALLOCATED_GPUS[@]}" ]]; then
        echo "GPU worker index $index is outside the Slurm allocation" >&2
        return 1
    fi
    printf '%s' "${SAFE_VLN_ALLOCATED_GPUS[$index]}"
}

safe_vln_require_policy_checkpoint() {
    local python_executable="$1"
    local bench_root="$2"
    local checkpoint="$3"
    local expected_policy_version="${4:-}"
    PYTHONPATH="$bench_root" "$python_executable" -c '
import json
import sys
from pathlib import Path
from safe_vln.checkpoint import require_safe_policy_checkpoint

checkpoint = Path(sys.argv[1])
state_path = checkpoint / "trainer_state.json"
if not state_path.is_file():
    raise SystemExit(f"missing trainer_state.json: {state_path}")
state = json.loads(state_path.read_text(encoding="utf-8"))
require_safe_policy_checkpoint(state, context="Slurm policy launch")
version = int(state.get("policy_version", 0))
if sys.argv[2]:
    expected = int(sys.argv[2])
    if version != expected:
        raise SystemExit(
            f"checkpoint policy_version={version} does not match expected={expected}"
        )
print(
    f"validated Safe-VLN policy checkpoint: {checkpoint} "
    f"version={version}"
)
' "$checkpoint" "$expected_policy_version"
}
