#!/bin/bash
# Collect the strict, on-policy VLM data used to warm up Safe-VLN critics.
# This is deliberately separate from run_safe_vln_v6_collect_risk80.sh:
# critic warmup starts from the base NaViLA checkpoint and therefore cannot
# require a Safe-VLN checkpoint that does not exist yet.
#
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v7-critic-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
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
safe_vln_capture_allocated_gpus 1
ALLOCATED_GPU=$(safe_vln_gpu_token 0)

IDS_FILE=${SAFE_VLN_CRITIC_IDS_FILE:-$BENCH_ROOT/outputs/safe_vln_v6_risk80.txt}
DATASET_DIR=${SAFE_VLN_CRITIC_DATASET:-$BENCH_ROOT/outputs/safe_vln_v7_strict_critic}
LOG_ROOT=${SAFE_VLN_CRITIC_LOG_ROOT:-$BENCH_ROOT/outputs/safe_vln_v7_strict_critic_logs}
VLM_PORT=${SAFE_VLN_CRITIC_VLM_PORT:-54321}
RENDER_PORT=${SAFE_VLN_CRITIC_RENDER_PORT:-54322}
POLICY_TAG=${SAFE_VLN_CRITIC_POLICY_TAG:-v7-strict-critic}
MAX_VLM_CALLS=${SAFE_VLN_CRITIC_MAX_VLM_CALLS:-60}

TRAIN_META=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
TRAIN_GT=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
MP3D_ROOT=$DATA_ROOT/MP3D-mirror/extracted/MatterPort3D/mp3d

[[ -s "$IDS_FILE" ]] || { echo "Missing episode ID file: $IDS_FILE" >&2; exit 1; }
[[ -f "$TRAIN_META" && -f "$TRAIN_GT" ]] || { echo "Missing VLN-CE train metadata" >&2; exit 1; }
[[ -d "$MP3D_ROOT" ]] || { echo "Missing MP3D scene root: $MP3D_ROOT" >&2; exit 1; }
if [[ -e "$DATASET_DIR" && -n "$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to append to critic dataset: $DATASET_DIR" >&2
    exit 1
fi

IDS=$(tr '\n' ' ' < "$IDS_FILE")
COUNT=$(wc -w < "$IDS_FILE")
mkdir -p "$LOG_ROOT" "$BENCH_ROOT/outputs/slurm"

module load anaconda/3-2024.02.01
source /share/software/anaconda3/2024.02.01/etc/profile.d/conda.sh
conda activate "$NAVILA_ENV"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export http_proxy=${http_proxy:-http://login04:3128}
export https_proxy=${https_proxy:-http://login04:3128}
export HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
export HTTPS_PROXY=${HTTPS_PROXY:-$https_proxy}
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost

renderer_pid=
vlm_pid=
cleanup() {
    if [[ -n "$renderer_pid" ]]; then kill "$renderer_pid" 2>/dev/null || true; fi
    if [[ -n "$vlm_pid" ]]; then kill "$vlm_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

cd "$BENCH_ROOT"
CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU" PYTHONPATH="$BENCH_ROOT" \
    "$RENDER_ENV/bin/python" scripts/habitat_render_server.py \
    --scenes-root "$MP3D_ROOT" --host 127.0.0.1 --port "$RENDER_PORT" \
    --gpu-device-id 0 >"$LOG_ROOT/renderer.log" 2>&1 &
renderer_pid=$!

CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU" PYTHONPATH="$NAVILA_ROOT:$BENCH_ROOT" \
    "$NAVILA_ENV/bin/python" scripts/vlm_server.py \
    --model_path "$NAVILA_ROOT/checkpoints/navila-llama3-8b-8f" \
    --host 127.0.0.1 --port "$VLM_PORT" \
    >"$LOG_ROOT/vlm_server.log" 2>&1 &
vlm_pid=$!

for port in "$VLM_PORT" "$RENDER_PORT"; do
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
    [[ "$ready" -eq 1 ]] || { echo "Service port $port did not become ready" >&2; exit 1; }
done

export CONDA_PREFIX="$ISAAC_ENV"
export GLIBC_ROOT GLIBC_LOADER GLIBC_LIB
export GIT_PYTHON_REFRESH=quiet OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce:$BENCH_ROOT/isaaclab_exts/omni.isaac.matterport:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_tasks:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_assets:$BENCH_ROOT"

"$GLIBC_LOADER" --library-path "$GLIBC_LIB:$ISAAC_ENV/lib:/lib64:/usr/lib64" \
    "$ISAAC_ENV/bin/python" scripts/safe_vln_main.py collect \
    --safe-live-render --vlnce-episode-ids $IDS \
    --vlnce-metadata "$TRAIN_META" --vlnce-gt "$TRAIN_GT" \
    --mp3d-scenes-root "$MP3D_ROOT" --render-host 127.0.0.1 \
    --render-port "$RENDER_PORT" --render-timeout-seconds 120 \
    --vlm-host 127.0.0.1 --vlm-port "$VLM_PORT" --dataset-role train \
    --collection-policy vlm --goal-stop-mode sensor-gated \
    --safe-policy-tag "$POLICY_TAG" --max-vlm-calls "$MAX_VLM_CALLS" \
    --dataset-dir "$DATASET_DIR" 2>&1 | tee "$LOG_ROOT/collection.log"

EPISODE_COUNT=$(find "$DATASET_DIR/completed" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)
[[ "$EPISODE_COUNT" -eq "$COUNT" ]] || {
    echo "Collected $EPISODE_COUNT/$COUNT completed episodes; refusing warmup" >&2
    exit 1
}

PYTHONPATH="$BENCH_ROOT" "$NAVILA_ENV/bin/python" scripts/audit_safe_vln_v5.py \
    --dataset-dir "$DATASET_DIR" --expected-episode-ids <(printf '%s\n' $IDS) \
    --allow-small-dataset --require-on-policy --output "$LOG_ROOT/audit.json"
