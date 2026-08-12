#!/bin/bash
# Collect strict Safe-VLN critic data with two independent Isaac/VLM workers.
# Each worker owns one GPU and nine CPU cores; the final dataset is merged only
# after both workers pass their episode-completion checks.
#
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v7-critic-2gpu
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

IDS_FILE=${SAFE_VLN_CRITIC_IDS_FILE:-$BENCH_ROOT/outputs/safe_vln_v6_risk80.txt}
FINAL_DATASET=${SAFE_VLN_CRITIC_DATASET:-$BENCH_ROOT/outputs/safe_vln_v7_strict_critic_2gpu}
WORK_ROOT=${SAFE_VLN_CRITIC_WORK_ROOT:-$BENCH_ROOT/outputs/safe_vln_v7_strict_critic_workers}
LOG_ROOT=${SAFE_VLN_CRITIC_LOG_ROOT:-$BENCH_ROOT/outputs/safe_vln_v7_strict_critic_2gpu_logs}
VLM_PORT0=${SAFE_VLN_CRITIC_VLM_PORT0:-54321}
VLM_PORT1=${SAFE_VLN_CRITIC_VLM_PORT1:-54331}
RENDER_PORT0=${SAFE_VLN_CRITIC_RENDER_PORT0:-54322}
RENDER_PORT1=${SAFE_VLN_CRITIC_RENDER_PORT1:-54332}
POLICY_TAG=${SAFE_VLN_CRITIC_POLICY_TAG:-v7-strict-critic-2gpu}
MAX_VLM_CALLS=${SAFE_VLN_CRITIC_MAX_VLM_CALLS:-60}

TRAIN_META=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
TRAIN_GT=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
MP3D_ROOT=$DATA_ROOT/MP3D-mirror/extracted/MatterPort3D/mp3d

[[ -s "$IDS_FILE" ]] || { echo "Missing episode ID file: $IDS_FILE" >&2; exit 1; }
[[ -f "$TRAIN_META" && -f "$TRAIN_GT" ]] || { echo "Missing VLN-CE train metadata" >&2; exit 1; }
[[ -d "$MP3D_ROOT" ]] || { echo "Missing MP3D scene root: $MP3D_ROOT" >&2; exit 1; }
for path in "$FINAL_DATASET" "$WORK_ROOT/gpu0" "$WORK_ROOT/gpu1"; do
    if [[ -e "$path" && -n "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to append to existing dataset: $path" >&2
        exit 1
    fi
done

mapfile -t EPISODES < <(awk 'NF {print $1}' "$IDS_FILE")
TOTAL=${#EPISODES[@]}
[[ "$TOTAL" -ge 2 ]] || { echo "Need at least two episode IDs" >&2; exit 1; }
MID=$(( (TOTAL + 1) / 2 ))

mkdir -p "$LOG_ROOT" "$BENCH_ROOT/outputs/slurm"
module load anaconda/3-2024.02.01
source /share/software/anaconda3/2024.02.01/etc/profile.d/conda.sh
conda activate "$NAVILA_ENV"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export http_proxy=${http_proxy:-http://login04:3128}
export https_proxy=${https_proxy:-http://login04:3128}
export HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
export HTTPS_PROXY=${HTTPS_PROXY:-$http_proxy}
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost

renderer_pids=()
vlm_pids=()
cleanup() {
    for pid in "${renderer_pids[@]:-}" "${vlm_pids[@]:-}"; do
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
    local count
    ids=$(printf '%s\n' "${EPISODES[@]:$first:$((last - first))}" | tr '\n' ' ')
    count=$((last - first))
    mkdir -p "$worker_log"

    export CUDA_VISIBLE_DEVICES="$gpu"
    local renderer_pid=
    local vlm_pid=
    cleanup_worker() {
        [[ -n "$renderer_pid" ]] && kill "$renderer_pid" 2>/dev/null || true
        [[ -n "$vlm_pid" ]] && kill "$vlm_pid" 2>/dev/null || true
    }
    trap cleanup_worker EXIT INT TERM

    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$BENCH_ROOT" \
        "$RENDER_ENV/bin/python" scripts/habitat_render_server.py \
        --scenes-root "$MP3D_ROOT" --host 127.0.0.1 --port "$render_port" \
        --gpu-device-id 0 >"$worker_log/renderer.log" 2>&1 &
    renderer_pid=$!

    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$NAVILA_ROOT:$BENCH_ROOT" \
        "$NAVILA_ENV/bin/python" scripts/vlm_server.py \
        --model_path "$NAVILA_ROOT/checkpoints/navila-llama3-8b-8f" \
        --host 127.0.0.1 --port "$vlm_port" \
        >"$worker_log/vlm_server.log" 2>&1 &
    vlm_pid=$!

    for port in "$vlm_port" "$render_port"; do
        ready=0
        for attempt in $(seq 1 300); do
            if /usr/bin/python3 -c \
                "import socket; s=socket.socket(); s.settimeout(0.2); raise SystemExit(s.connect_ex(('127.0.0.1',$port)))" \
                >/dev/null 2>&1; then
                ready=1
                break
            fi
            sleep 2
        done
        [[ "$ready" -eq 1 ]] || { echo "Worker $worker port $port did not become ready" >&2; return 1; }
    done

    export CONDA_PREFIX="$ISAAC_ENV"
    export GLIBC_ROOT GLIBC_LOADER GLIBC_LIB
    export GIT_PYTHON_REFRESH=quiet OMNI_KIT_ACCEPT_EULA=YES
    export PYTHONPATH="$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce:$BENCH_ROOT/isaaclab_exts/omni.isaac.matterport:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_tasks:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_assets:$BENCH_ROOT"

    "$GLIBC_LOADER" --library-path "$GLIBC_LIB:$ISAAC_ENV/lib:/lib64:/usr/lib64" \
        "$ISAAC_ENV/bin/python" scripts/safe_vln_main.py collect \
        --safe-live-render --vlnce-episode-ids $ids \
        --vlnce-metadata "$TRAIN_META" --vlnce-gt "$TRAIN_GT" \
        --mp3d-scenes-root "$MP3D_ROOT" --render-host 127.0.0.1 \
        --render-port "$render_port" --render-timeout-seconds 120 \
        --vlm-host 127.0.0.1 --vlm-port "$vlm_port" --dataset-role train \
        --collection-policy vlm --goal-stop-mode sensor-gated \
        --safe-policy-tag "$POLICY_TAG" --max-vlm-calls "$MAX_VLM_CALLS" \
        --dataset-dir "$worker_dir" 2>&1 | tee "$worker_log/collection.log"

    local completed
    completed=$(find "$worker_dir/completed" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)
    [[ "$completed" -eq "$count" ]] || {
        echo "Worker $worker collected $completed/$count episodes" >&2
        return 1
    }
}

run_worker "$GPU0" 0 0 "$MID" "$VLM_PORT0" "$RENDER_PORT0" >"$LOG_ROOT/gpu0.worker.out" 2>&1 &
worker0_pid=$!
run_worker "$GPU1" 1 "$MID" "$TOTAL" "$VLM_PORT1" "$RENDER_PORT1" >"$LOG_ROOT/gpu1.worker.out" 2>&1 &
worker1_pid=$!

status=0
wait "$worker0_pid" || status=1
wait "$worker1_pid" || status=1
[[ "$status" -eq 0 ]] || { echo "One or more GPU workers failed" >&2; exit 1; }

PYTHONPATH="$BENCH_ROOT" "$NAVILA_ENV/bin/python" scripts/merge_safe_vln_datasets.py \
    --source-dir "$WORK_ROOT/gpu0" --source-dir "$WORK_ROOT/gpu1" \
    --output-dir "$FINAL_DATASET"

PYTHONPATH="$BENCH_ROOT" "$NAVILA_ENV/bin/python" scripts/audit_safe_vln_v5.py \
    --dataset-dir "$FINAL_DATASET" --expected-episode-ids <(printf '%s\n' "${EPISODES[@]}") \
    --allow-small-dataset --require-on-policy --output "$LOG_ROOT/audit.json"
