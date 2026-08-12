#!/bin/bash
# Collect 80 strictly paired base-NaViLA episodes with Habitat RGB rendering.
# This is the fallback for nodes where an Isaac native camera is unavailable.
# Go2 physics and safety remain in Isaac; Habitat renders every RGB history at
# the matching physics pose. Two workers use 2 GPUs and 18 CPU cores total.
#
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v8-live
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --chdir=/share/home/202430461770/NaVILA-Bench
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

BENCH_ROOT=/share/home/202430461770/NaVILA-Bench
NAVILA_ROOT=/share/home/202430461770/NaVILA
DATA_ROOT=/share/home/202430461770/NaVILA-Dataset
ISAACLAB_ROOT=/share/home/202430461770/IsaacLab
NAVILA_ENV=/share/home/202430461770/.conda/envs/navila
RENDER_ENV=/share/home/202430461770/.conda/envs/vlnce3
ISAAC_ENV=/share/home/202430461770/.conda/envs/vlnce-isaac
GLIBC_ROOT=/share/software/spack/opt/spack/linux-rocky8-icelake/gcc-8.5.0/glibc-2.38-kbyap6e5vjwnkhmks7d4nbfh3fabixle
GLIBC_LOADER=$GLIBC_ROOT/lib/ld-linux-x86-64.so.2
GLIBC_LIB=$GLIBC_ROOT/lib
source "$BENCH_ROOT/scripts/slurm_gpu_env.sh"
safe_vln_capture_allocated_gpus 2
GPU0=$(safe_vln_gpu_token 0)
GPU1=$(safe_vln_gpu_token 1)

IDS_FILE=${SAFE_VLN_LIVE_IDS_FILE:-$BENCH_ROOT/outputs/safe_vln_v6_risk80.txt}
FINAL_DATASET=${SAFE_VLN_LIVE_DATASET:-$BENCH_ROOT/outputs/safe_vln_v8_live_render_2gpu}
WORK_ROOT=${SAFE_VLN_LIVE_WORK_ROOT:-$BENCH_ROOT/outputs/safe_vln_v8_live_render_workers}
LOG_ROOT=${SAFE_VLN_LIVE_LOG_ROOT:-$BENCH_ROOT/outputs/safe_vln_v8_live_render_logs}
COLLECTION_POLICY=${SAFE_VLN_COLLECTION_POLICY:-vlm}
ALLOW_ONLINE_ORACLE=${SAFE_VLN_ALLOW_ONLINE_ORACLE:-0}
GOAL_STOP_MODE=${SAFE_VLN_GOAL_STOP_MODE:-sensor-gated}
EXPECTED_EPISODES=${SAFE_VLN_LIVE_EXPECTED_EPISODES:-80}
ALLOW_SMALL_DATASET=${SAFE_VLN_LIVE_ALLOW_SMALL_DATASET:-0}
TRAIN_META=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
TRAIN_GT=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
MP3D_ROOT=$DATA_ROOT/MP3D-mirror/extracted/MatterPort3D/mp3d

if [[ -n "${SAFE_VLN_CHECKPOINT:-}" ]]; then
    echo "Base live-render collection requires original NaViLA; unset SAFE_VLN_CHECKPOINT" >&2
    exit 1
fi
[[ -s "$IDS_FILE" ]] || { echo "Missing episode ID file: $IDS_FILE" >&2; exit 1; }
[[ -f "$TRAIN_META" && -f "$TRAIN_GT" ]] || {
    echo "Missing VLN-CE train metadata" >&2
    exit 1
}
[[ -d "$MP3D_ROOT" ]] || { echo "Missing MP3D scene root: $MP3D_ROOT" >&2; exit 1; }
for path in "$FINAL_DATASET" "$WORK_ROOT/gpu0" "$WORK_ROOT/gpu1"; do
    if [[ -e "$path" && -n "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to append to existing dataset: $path" >&2
        exit 1
    fi
done

mapfile -t EPISODES < <(awk 'NF {print $1}' "$IDS_FILE")
TOTAL=${#EPISODES[@]}
[[ "$TOTAL" -eq "$EXPECTED_EPISODES" ]] || {
    echo "Expected exactly $EXPECTED_EPISODES episode IDs, found $TOTAL" >&2
    exit 1
}
MID=$((TOTAL / 2))

mkdir -p "$LOG_ROOT" "$BENCH_ROOT/outputs/slurm"
module load anaconda/3-2024.02.01
source /share/software/anaconda3/2024.02.01/etc/profile.d/conda.sh
conda activate "$NAVILA_ENV"
cd "$BENCH_ROOT"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export http_proxy=${http_proxy:-http://login04:3128}
export https_proxy=${https_proxy:-http://login04:3128}
export HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
export HTTPS_PROXY=${HTTPS_PROXY:-$https_proxy}
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost

worker_pids=()
cleanup() {
    for pid in "${worker_pids[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

run_worker() {
    local gpu="$1"
    local worker="$2"
    local first="$3"
    local last="$4"
    local vlm_port="$5"
    local render_port="$6"
    local worker_dir="$WORK_ROOT/gpu${worker}"
    local worker_log="$LOG_ROOT/gpu${worker}"
    local ids
    local count=$((last - first))
    ids=$(printf '%s\n' "${EPISODES[@]:$first:$count}" | tr '\n' ' ')

    mkdir -p "$worker_dir" "$worker_log"

    # Keep every Kit/Omniverse lock and renderer cache private to this worker.
    local omni_base="/tmp/${USER}/safe_vln_live_omni_${SLURM_JOB_ID}/worker${worker}"
    export OMNI_USER_CACHE_DIR="$omni_base/cache"
    export OMNI_USER_DATA="$omni_base/data"
    export XDG_CACHE_HOME="$omni_base/xdg"
    export XDG_CONFIG_HOME="$omni_base/config"
    export XDG_DATA_HOME="$omni_base/xdg-data"
    export OMNI_USER_CACHE="$OMNI_USER_CACHE_DIR"
    export OMNI_USER_CONFIG="$XDG_CONFIG_HOME"
    export OMNI_USER_LOGS="$omni_base/logs"
    export TMPDIR="$omni_base/tmp"
    export HOME="$omni_base/home"
    mkdir -p "$OMNI_USER_CACHE_DIR" "$OMNI_USER_DATA" "$XDG_CACHE_HOME" \
        "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$OMNI_USER_LOGS" "$TMPDIR" "$HOME"

    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$BENCH_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$BENCH_ROOT" \
        "$RENDER_ENV/bin/python" scripts/habitat_render_server.py \
        --scenes-root "$MP3D_ROOT" --host 127.0.0.1 --port "$render_port" \
        --gpu-device-id 0 >"$worker_log/renderer.log" 2>&1 &
    local renderer_pid=$!

    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$NAVILA_ROOT:$BENCH_ROOT" \
        "$NAVILA_ENV/bin/python" scripts/vlm_server.py \
        --model_path "$NAVILA_ROOT/checkpoints/navila-llama3-8b-8f" \
        --host 127.0.0.1 --port "$vlm_port" \
        >"$worker_log/vlm_server.log" 2>&1 &
    local vlm_pid=$!

    cleanup_worker() {
        kill "$renderer_pid" "$vlm_pid" 2>/dev/null || true
    }
    trap cleanup_worker EXIT INT TERM

    for port in "$vlm_port" "$render_port"; do
        local ready=0
        for attempt in $(seq 1 300); do
            if /usr/bin/python3 -c \
                "import socket; s=socket.socket(); s.settimeout(0.2); raise SystemExit(s.connect_ex(('127.0.0.1',$port)))" \
                >/dev/null 2>&1; then
                ready=1
                break
            fi
            sleep 2
        done
        [[ "$ready" -eq 1 ]] || {
            echo "Worker $worker service port $port did not become ready" >&2
            return 1
        }
    done

    export CONDA_PREFIX="$ISAAC_ENV" GLIBC_ROOT GLIBC_LOADER GLIBC_LIB
    export GIT_PYTHON_REFRESH=quiet OMNI_KIT_ACCEPT_EULA=YES
    export PYTHONPATH="$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce:$BENCH_ROOT/isaaclab_exts/omni.isaac.matterport:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_tasks:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_assets:$BENCH_ROOT"

    HOME="$HOME" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    XDG_CONFIG_HOME="$XDG_CONFIG_HOME" XDG_DATA_HOME="$XDG_DATA_HOME" \
    collection_args=(--collection-policy "$COLLECTION_POLICY")
    if [[ "$ALLOW_ONLINE_ORACLE" == "1" ]]; then
        collection_args+=(--allow-online-oracle)
    fi
    "$GLIBC_LOADER" --library-path "$GLIBC_LIB:$ISAAC_ENV/lib:/lib64:/usr/lib64" \
        "$ISAAC_ENV/bin/python" scripts/safe_vln_main.py collect \
        --safe-live-render --vlnce-episode-ids $ids \
        --vlnce-metadata "$TRAIN_META" --vlnce-gt "$TRAIN_GT" \
        --mp3d-scenes-root "$MP3D_ROOT" --render-host 127.0.0.1 \
        --render-port "$render_port" --render-timeout-seconds 120 \
        --vlm-host 127.0.0.1 --vlm-port "$vlm_port" --dataset-role train \
        "${collection_args[@]}" --goal-stop-mode "$GOAL_STOP_MODE" \
        --safe-policy-tag "navila-base-live-gpu${worker}" --online-round 0 \
        --max-vlm-calls 60 --dataset-dir "$worker_dir" \
        2>&1 | tee "$worker_log/collection.log"

    local completed
    completed=$(find "$worker_dir/completed" -mindepth 2 -maxdepth 2 -name manifest.json -type f 2>/dev/null | wc -l)
    [[ "$completed" -eq "$count" ]] || {
        echo "Worker $worker collected $completed/$count episodes" >&2
        return 1
    }
    echo "live_render_worker=$worker completed=$completed range=${first}:${last}"
}

run_worker "$GPU0" 0 0 "$MID" 54621 54622 >"$LOG_ROOT/gpu0.worker.out" 2>&1 &
worker_pids+=("$!")
run_worker "$GPU1" 1 "$MID" "$TOTAL" 54631 54632 >"$LOG_ROOT/gpu1.worker.out" 2>&1 &
worker_pids+=("$!")

status=0
for pid in "${worker_pids[@]}"; do
    wait "$pid" || status=1
done
[[ "$status" -eq 0 ]] || { echo "One or more live-render workers failed" >&2; exit 1; }

PYTHONPATH="$BENCH_ROOT" "$NAVILA_ENV/bin/python" scripts/merge_safe_vln_datasets.py \
    --source-dir "$WORK_ROOT/gpu0" --source-dir "$WORK_ROOT/gpu1" \
    --output-dir "$FINAL_DATASET"
audit_args=(--allow-online-oracle)
if [[ "$ALLOW_SMALL_DATASET" == "1" ]]; then
    audit_args+=(--allow-small-dataset)
fi
PYTHONPATH="$BENCH_ROOT" "$NAVILA_ENV/bin/python" scripts/audit_safe_vln_v5.py \
    --dataset-dir "$FINAL_DATASET" \
    --expected-episode-ids <(printf '%s\n' "${EPISODES[@]}") \
    --require-navila-teacher "${audit_args[@]}" --output "$LOG_ROOT/audit.json"
echo "safe_vln_v8_live_render_complete episodes=$TOTAL dataset=$FINAL_DATASET"
