#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT=${BENCH_ROOT:-/share/home/202430461770/NaVILA-Bench}
NAVILA_ROOT=${NAVILA_ROOT:-/share/home/202430461770/NaVILA}
DATA_ROOT=${DATA_ROOT:-/share/home/202430461770/NaVILA-Dataset}
RENDER_ENV=${RENDER_ENV:-/share/home/202430461770/.conda/envs/vlnce3}
ISAAC_ENV=${ISAAC_ENV:-/share/home/202430461770/.conda/envs/vlnce-isaac}
ISAACLAB_ROOT=${ISAACLAB_ROOT:-/share/home/202430461770/IsaacLab}
EPISODE_COUNT=${EPISODE_COUNT:-500}
EPISODE_OFFSET=${EPISODE_OFFSET:-80}
GLIBC_ROOT=${GLIBC_ROOT:-/share/software/spack/opt/spack/linux-rocky8-icelake/gcc-8.5.0/glibc-2.38-kbyap6e5vjwnkhmks7d4nbfh3fabixle}
GLIBC_LOADER="$GLIBC_ROOT/lib/ld-linux-x86-64.so.2"
GLIBC_LIB="$GLIBC_ROOT/lib"

TRAIN_META="$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
TRAIN_GT="$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz"
MP3D_ROOT="$DATA_ROOT/MP3D-mirror/extracted/MatterPort3D/mp3d"
COST_PROFILE="$BENCH_ROOT/checkpoints/safe_vln_v4_cost_profile.json"
if [[ "$EPISODE_COUNT" == 500 && "$EPISODE_OFFSET" == 80 ]]; then
    DATASET_DIR=${DATASET_DIR:-$BENCH_ROOT/outputs/safe_live_v5_oracle_500}
else
    DATASET_DIR=${DATASET_DIR:-$BENCH_ROOT/outputs/safe_live_v5_oracle_smoke_${EPISODE_COUNT}}
fi
LOG_ROOT=${LOG_ROOT:-${DATASET_DIR}_logs}

if [[ -e "$DATASET_DIR/manifest.json" || -d "$DATASET_DIR/completed" ]]; then
    echo "Refusing to append to existing v5 dataset: $DATASET_DIR" >&2
    exit 2
fi
mkdir -p "$LOG_ROOT"
cd "$BENCH_ROOT"

renderer_pid=
cleanup() {
    if [[ -n "$renderer_pid" ]]; then
        kill "$renderer_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

PYTHONUNBUFFERED=1 PYTHONPATH="$BENCH_ROOT" \
"$RENDER_ENV/bin/python" scripts/habitat_render_server.py \
    --scenes-root "$MP3D_ROOT" --host 127.0.0.1 --port 54322 \
    --gpu-device-id 0 >"$LOG_ROOT/renderer.log" 2>&1 &
renderer_pid=$!

for attempt in $(seq 1 180); do
    if ! kill -0 "$renderer_pid" 2>/dev/null; then
        echo "renderer exited before opening port 54322" >&2
        exit 1
    fi
    if /usr/bin/python3 -c \
        "import socket; s=socket.socket(); s.settimeout(0.2); raise SystemExit(s.connect_ex(('127.0.0.1',54322)))" \
        >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

ORACLE_IDS=$(
    "$RENDER_ENV/bin/python" scripts/select_vlnce_episode_ids.py \
        --metadata "$TRAIN_META" --count "$EPISODE_COUNT" \
        --offset "$EPISODE_OFFSET" \
        $(if [[ "$EPISODE_COUNT" -ge 61 ]]; then echo --require-scene-count 61; fi)
)
printf '%s\n' "$ORACLE_IDS" >"$LOG_ROOT/episode_ids.txt"

export CONDA_PREFIX="$ISAAC_ENV"
export GLIBC_ROOT GLIBC_LOADER GLIBC_LIB
export GIT_PYTHON_REFRESH=quiet OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
export PYTHONPATH="$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce:$BENCH_ROOT/isaaclab_exts/omni.isaac.matterport:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_tasks:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_assets:$BENCH_ROOT"

"$GLIBC_LOADER" \
    --library-path "$GLIBC_LIB:$ISAAC_ENV/lib:/lib64:/usr/lib64" \
    "$ISAAC_ENV/bin/python" scripts/safe_vln_main.py collect \
    --safe-live-render --vlnce-episode-ids $ORACLE_IDS \
    --vlnce-metadata "$TRAIN_META" --vlnce-gt "$TRAIN_GT" \
    --mp3d-scenes-root "$MP3D_ROOT" --render-host 127.0.0.1 \
    --render-port 54322 --dataset-role train --collection-policy oracle \
    --goal-stop-mode policy --safe-cost-profile "$COST_PROFILE" \
    --dataset-dir "$DATASET_DIR" 2>&1 | tee "$LOG_ROOT/collection.log"

AUDIT_ARGS=(
    --dataset-dir "$DATASET_DIR"
    --expected-episode-ids "$LOG_ROOT/episode_ids.txt"
    --output "$LOG_ROOT/audit.json"
)
if [[ "$EPISODE_COUNT" != 500 ]]; then
    AUDIT_ARGS+=(--allow-small-dataset)
fi
"$ISAAC_ENV/bin/python" scripts/audit_safe_vln_v5.py "${AUDIT_ARGS[@]}"
