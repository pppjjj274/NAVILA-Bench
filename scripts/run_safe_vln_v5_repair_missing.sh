#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT=${BENCH_ROOT:-/share/home/202430461770/NaVILA-Bench}
NAVILA_ROOT=${NAVILA_ROOT:-/share/home/202430461770/NaVILA}
DATA_ROOT=${DATA_ROOT:-/share/home/202430461770/NaVILA-Dataset}
RENDER_ENV=${RENDER_ENV:-/share/home/202430461770/.conda/envs/vlnce3}
ISAAC_ENV=${ISAAC_ENV:-/share/home/202430461770/.conda/envs/vlnce-isaac}
ISAACLAB_ROOT=${ISAACLAB_ROOT:-/share/home/202430461770/IsaacLab}
DATASET_DIR=${DATASET_DIR:-$BENCH_ROOT/outputs/safe_live_v5_oracle_500}
ORIGINAL_LOG_ROOT=${ORIGINAL_LOG_ROOT:-${DATASET_DIR}_logs}
LOG_ROOT=${LOG_ROOT:-$BENCH_ROOT/outputs/safe_live_v5_oracle_500_repair_logs}
REPAIR_IDS=${REPAIR_IDS:-"109 530 532 904"}
RENDER_PORT=${RENDER_PORT:-54322}
GLIBC_ROOT=${GLIBC_ROOT:-/share/software/spack/opt/spack/linux-rocky8-icelake/gcc-8.5.0/glibc-2.38-kbyap6e5vjwnkhmks7d4nbfh3fabixle}
GLIBC_LOADER="$GLIBC_ROOT/lib/ld-linux-x86-64.so.2"
GLIBC_LIB="$GLIBC_ROOT/lib"

TRAIN_META="$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
TRAIN_GT="$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz"
MP3D_ROOT="$DATA_ROOT/MP3D-mirror/extracted/MatterPort3D/mp3d"
COST_PROFILE="$BENCH_ROOT/checkpoints/safe_vln_v4_cost_profile.json"
EXPECTED_IDS_FILE="$ORIGINAL_LOG_ROOT/episode_ids.txt"

mkdir -p "$LOG_ROOT"
cd "$BENCH_ROOT"

read -r -a repair_ids <<<"$REPAIR_IDS"
if [[ ${#repair_ids[@]} -eq 0 ]]; then
    echo "REPAIR_IDS must contain at least one episode ID" >&2
    exit 2
fi

"$ISAAC_ENV/bin/python" - "$DATASET_DIR" "$EXPECTED_IDS_FILE" "${repair_ids[@]}" <<'PY'
import json
from pathlib import Path
import sys

dataset = Path(sys.argv[1])
expected_path = Path(sys.argv[2])
repair_ids = set(sys.argv[3:])
manifest_path = dataset / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"missing dataset manifest: {manifest_path}")
if not expected_path.is_file():
    raise SystemExit(f"missing original episode ID list: {expected_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_ids = set(expected_path.read_text(encoding="utf-8").replace(",", " ").split())
completed_ids = {str(value) for value in manifest.get("episode_ids", [])}
missing_ids = expected_ids - completed_ids
extra_ids = completed_ids - expected_ids
if manifest.get("schema_version") != "safe-vln-go2-v5":
    raise SystemExit(f"unexpected schema: {manifest.get('schema_version')}")
if manifest.get("dataset_role") != "train":
    raise SystemExit(f"unexpected dataset role: {manifest.get('dataset_role')}")
if manifest.get("completed_episodes") != 496:
    raise SystemExit(
        f"repair requires exactly 496 completed episodes; found "
        f"{manifest.get('completed_episodes')}"
    )
if missing_ids != repair_ids or extra_ids:
    raise SystemExit(
        f"repair set mismatch: missing={sorted(missing_ids)} "
        f"requested={sorted(repair_ids)} extra={sorted(extra_ids)}"
    )
for episode_id in repair_ids:
    if (dataset / "completed" / episode_id).exists():
        raise SystemExit(f"repair episode already committed: {episode_id}")
print(json.dumps({"repair_preflight": "accepted", "missing": sorted(missing_ids)}))
PY

renderer_pid=
cleanup() {
    if [[ -n "$renderer_pid" ]]; then
        kill "$renderer_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

PYTHONUNBUFFERED=1 PYTHONPATH="$BENCH_ROOT" \
"$RENDER_ENV/bin/python" scripts/habitat_render_server.py \
    --scenes-root "$MP3D_ROOT" --host 127.0.0.1 --port "$RENDER_PORT" \
    --gpu-device-id 0 >"$LOG_ROOT/renderer.log" 2>&1 &
renderer_pid=$!

renderer_ready=false
for _ in $(seq 1 180); do
    if ! kill -0 "$renderer_pid" 2>/dev/null; then
        echo "renderer exited before opening port $RENDER_PORT" >&2
        exit 1
    fi
    if /usr/bin/python3 -c \
        "import socket; s=socket.socket(); s.settimeout(0.2); raise SystemExit(s.connect_ex(('127.0.0.1',$RENDER_PORT)))" \
        >/dev/null 2>&1; then
        renderer_ready=true
        break
    fi
    sleep 2
done
if [[ "$renderer_ready" != true ]]; then
    echo "renderer did not open port $RENDER_PORT within 360 seconds" >&2
    exit 1
fi

export CONDA_PREFIX="$ISAAC_ENV"
export GLIBC_ROOT GLIBC_LOADER GLIBC_LIB
export GIT_PYTHON_REFRESH=quiet OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export http_proxy=${http_proxy:-http://login04:3128}
export https_proxy=${https_proxy:-http://login04:3128}
export HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
export HTTPS_PROXY=${HTTPS_PROXY:-$https_proxy}
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
export PYTHONPATH="$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce:$BENCH_ROOT/isaaclab_exts/omni.isaac.matterport:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_tasks:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_assets:$BENCH_ROOT"

failed_ids=()
for episode_id in "${repair_ids[@]}"; do
    echo "[SAFE-V5-REPAIR] collecting episode=$episode_id"
    set +e
    "$GLIBC_LOADER" \
        --library-path "$GLIBC_LIB:$ISAAC_ENV/lib:/lib64:/usr/lib64" \
        "$ISAAC_ENV/bin/python" scripts/safe_vln_main.py collect \
        --safe-live-render --vlnce-episode-ids "$episode_id" \
        --vlnce-metadata "$TRAIN_META" --vlnce-gt "$TRAIN_GT" \
        --mp3d-scenes-root "$MP3D_ROOT" --render-host 127.0.0.1 \
        --render-port "$RENDER_PORT" --render-timeout-seconds 120 \
        --dataset-role train --collection-policy oracle \
        --goal-stop-mode policy --safe-cost-profile "$COST_PROFILE" \
        --dataset-dir "$DATASET_DIR" \
        2>&1 | tee "$LOG_ROOT/episode_${episode_id}.log"
    collect_status=${PIPESTATUS[0]}
    set -e
    if [[ $collect_status -eq 0 \
        && -d "$DATASET_DIR/completed/$episode_id" ]]; then
        echo "[SAFE-V5-REPAIR] committed episode=$episode_id"
    else
        failed_ids+=("$episode_id")
        echo "[SAFE-V5-REPAIR] failed episode=$episode_id " \
            "status=$collect_status committed=$([[ -d "$DATASET_DIR/completed/$episode_id" ]] && echo yes || echo no)" >&2
    fi
done

if [[ ${#failed_ids[@]} -ne 0 ]]; then
    printf '[SAFE-V5-REPAIR] failed IDs: %s\n' "${failed_ids[*]}" >&2
    exit 1
fi

"$ISAAC_ENV/bin/python" scripts/audit_safe_vln_v5.py \
    --dataset-dir "$DATASET_DIR" \
    --expected-episode-ids "$EXPECTED_IDS_FILE" \
    --output "$LOG_ROOT/audit.json"

echo "[SAFE-V5-REPAIR] completed: dataset accepted with 500 episodes"
