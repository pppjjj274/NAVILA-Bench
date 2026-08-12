#!/bin/bash
# Collect 80 strictly paired critic-warmup episodes from Isaac's native RGB
# camera.  By default the workers serve the original deterministic NaViLA
# Actor directly.  Critic values are training targets, not collection inputs,
# so loading a stale critic checkpoint here only adds a second visual forward
# and couples fresh data to an obsolete objective.
# Two array tasks use one IsaacSim process each on the known-compatible g02n06
# node (driver 550.78).  Each task gets one Slurm GPU and an isolated Kit
# cache/HOME, so both workers can share the node without a viewport/KVDB lock.
# Total allocation when both tasks run: 2 GPUs + 18 CPUs.
#
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v8-native
#SBATCH --nodes=1
#SBATCH --nodelist=g02n06
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH --chdir=/share/home/202430461770/NaVILA-Bench
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

BENCH_ROOT=/share/home/202430461770/NaVILA-Bench
NAVILA_ROOT=/share/home/202430461770/NaVILA
ISAAC_ENV=/share/home/202430461770/.conda/envs/vlnce-isaac
NAVILA_ENV=/share/home/202430461770/.conda/envs/navila
TRAIN_META=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
TRAIN_GT=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
USD_ROOT=$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce/assets/matterport_usd
GLIBC_ROOT=/share/software/spack/opt/spack/linux-rocky8-icelake/gcc-8.5.0/glibc-2.38-kbyap6e5vjwnkhmks7d4nbfh3fabixle
GLIBC_LOADER=$GLIBC_ROOT/lib/ld-linux-x86-64.so.2
GLIBC_LIB=$GLIBC_ROOT/lib
source "$BENCH_ROOT/scripts/slurm_gpu_env.sh"
safe_vln_capture_allocated_gpus 1
ALLOCATED_GPU=$(safe_vln_gpu_token 0)
OUTPUT_ROOT=${SAFE_VLN_NATIVE_OUTPUT_ROOT:-$BENCH_ROOT/outputs/safe_vln_v8_native_camera}
LOG_ROOT=${SAFE_VLN_NATIVE_LOG_ROOT:-$BENCH_ROOT/outputs/safe_vln_v8_native_camera_logs}
WORKER=${SLURM_ARRAY_TASK_ID:-0}
EPISODES_PER_WORKER=${SAFE_VLN_NATIVE_EPISODES_PER_WORKER:-40}
MAX_VLM_CALLS=${SAFE_VLN_NATIVE_MAX_VLM_CALLS:-60}
if [[ "$EPISODES_PER_WORKER" -le 0 ]]; then
    echo "SAFE_VLN_NATIVE_EPISODES_PER_WORKER must be positive" >&2
    exit 1
fi
if [[ "$MAX_VLM_CALLS" -le 0 ]]; then
    echo "SAFE_VLN_NATIVE_MAX_VLM_CALLS must be positive" >&2
    exit 1
fi
START_IDX=$((WORKER * EPISODES_PER_WORKER))
END_IDX=$(((WORKER + 1) * EPISODES_PER_WORKER))
VLM_PORT=$((54621 + WORKER))
DATASET_DIR=$OUTPUT_ROOT/gpu${WORKER}
WORKER_LOG=$LOG_ROOT/gpu${WORKER}
POLICY_TAG="navila-base-native-camera-gpu${WORKER}"

if [[ -n "${SAFE_VLN_CHECKPOINT:-}" ]]; then
    echo "Canonical native collection requires original NaViLA; unset SAFE_VLN_CHECKPOINT" >&2
    exit 1
fi
if [[ -e "$DATASET_DIR" && -n "$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite native-camera dataset: $DATASET_DIR" >&2
    exit 1
fi

module load anaconda/3-2024.02.01
source /share/software/anaconda3/2024.02.01/etc/profile.d/conda.sh
conda activate "$NAVILA_ENV"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep Omniverse/KVDB state private to this worker.  Both array tasks share
# $HOME over NFS; a shared Kit cache can make the second viewport segfault.
OMNI_CACHE_BASE=${SAFE_VLN_OMNI_CACHE_BASE:-/tmp/${USER}/safe_vln_omni_${SLURM_JOB_ID}}
export OMNI_USER_CACHE_DIR="$OMNI_CACHE_BASE/worker${WORKER}/cache"
export OMNI_USER_DATA="$OMNI_CACHE_BASE/worker${WORKER}/data"
export XDG_CACHE_HOME="$OMNI_CACHE_BASE/worker${WORKER}/xdg"
export TMPDIR="$OMNI_CACHE_BASE/worker${WORKER}/tmp"
export XDG_CONFIG_HOME="$OMNI_CACHE_BASE/worker${WORKER}/config"
export XDG_DATA_HOME="$OMNI_CACHE_BASE/worker${WORKER}/xdg-data"
export OMNI_USER_CACHE="$OMNI_USER_CACHE_DIR"
export OMNI_USER_CONFIG="$XDG_CONFIG_HOME"
export OMNI_USER_LOGS="$OMNI_CACHE_BASE/worker${WORKER}/logs"
KIT_HOME="$OMNI_CACHE_BASE/worker${WORKER}/home"
mkdir -p "$OMNI_USER_CACHE_DIR" "$OMNI_USER_DATA" "$XDG_CACHE_HOME" \
    "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$OMNI_USER_LOGS" "$TMPDIR" \
    "$KIT_HOME"
export PYTHONPATH="$NAVILA_ROOT:$BENCH_ROOT"
export http_proxy=${http_proxy:-http://login04:3128}
export https_proxy=${https_proxy:-http://login04:3128}
export HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
export HTTPS_PROXY=${HTTPS_PROXY:-$http_proxy}
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
mkdir -p "$DATASET_DIR" "$WORKER_LOG" "$BENCH_ROOT/outputs/slurm"
cd "$BENCH_ROOT"

# The bundled vln_ce_isaac_v1 file is val_unseen (11 scenes), not training
# data.  Convert the official 61-scene train split into Isaac coordinates and
# scene-balanced order in each worker's private temporary directory.  Passing
# this exact path through both launcher layers prevents the old silent fallback
# to the bundled evaluation episodes.
ISAAC_R2R="$TMPDIR/vlnce_train_isaac_balanced.json.gz"
"$NAVILA_ENV/bin/python" scripts/convert_vlnce_to_isaac.py \
    --metadata "$TRAIN_META" --gt "$TRAIN_GT" \
    --source-split train --balanced-seed 20260727 \
    --expected-scenes 61 --usd-root "$USD_ROOT" \
    --output "$ISAAC_R2R" \
    >"$WORKER_LOG/dataset_conversion.json"

vlm_pid=
cleanup() {
    [[ -n "$vlm_pid" ]] && kill "$vlm_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU" PYTHONUNBUFFERED=1 PYTHONPATH="$NAVILA_ROOT:$BENCH_ROOT" \
"$NAVILA_ENV/bin/python" scripts/vlm_server.py \
    --model_path "$NAVILA_ROOT/checkpoints/navila-llama3-8b-8f" \
    --safe_sampling_seed "$((20260802 + WORKER))" \
    --host 127.0.0.1 --port "$VLM_PORT" \
    >"$WORKER_LOG/vlm_server.log" 2>&1 &
vlm_pid=$!

ready=0
for attempt in $(seq 1 300); do
    if ! kill -0 "$vlm_pid" 2>/dev/null; then
        echo "VLM server exited before becoming ready; see $WORKER_LOG/vlm_server.log" >&2
        exit 1
    fi
    if /usr/bin/python3 -c \
        "import socket; s=socket.socket(); s.settimeout(0.2); raise SystemExit(s.connect_ex(('127.0.0.1',$VLM_PORT)))" \
        >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 2
done
[[ "$ready" -eq 1 ]] || { echo "VLM port $VLM_PORT did not become ready" >&2; exit 1; }

export CONDA_PREFIX="$ISAAC_ENV" GLIBC_ROOT GLIBC_LOADER GLIBC_LIB
export GIT_PYTHON_REFRESH=quiet OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce:$BENCH_ROOT/isaaclab_exts/omni.isaac.matterport:/share/home/202430461770/IsaacLab/source/extensions/omni.isaac.lab:/share/home/202430461770/IsaacLab/source/extensions/omni.isaac.lab_tasks:/share/home/202430461770/IsaacLab/source/extensions/omni.isaac.lab_assets:$BENCH_ROOT"

HOME="$KIT_HOME" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
XDG_CONFIG_HOME="$XDG_CONFIG_HOME" XDG_DATA_HOME="$XDG_DATA_HOME" \
CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU" \
"$GLIBC_LOADER" --library-path "$GLIBC_LIB:$ISAAC_ENV/lib:/lib64:/usr/lib64" \
    "$ISAAC_ENV/bin/python" scripts/safe_vln_main.py collect \
    --r2r-data-path "$ISAAC_R2R" \
    --start-idx "$START_IDX" --end-idx "$END_IDX" \
    --vlm-host 127.0.0.1 --vlm-port "$VLM_PORT" \
    --collection-policy vlm --goal-stop-mode sensor-gated \
    --safe-policy-tag "$POLICY_TAG" --online-round 1 \
    --max-vlm-calls "$MAX_VLM_CALLS" --dataset-dir "$DATASET_DIR" \
    2>&1 | tee "$WORKER_LOG/collection.log"

# Only atomically committed episodes count as complete.
completed=0
if [[ -d "$DATASET_DIR/completed" ]]; then
    completed=$(find "$DATASET_DIR/completed" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)
fi
if [[ "$completed" -ne "$EPISODES_PER_WORKER" ]]; then
    echo "Native-camera worker collected $completed/$EPISODES_PER_WORKER episode shards" >&2
    exit 1
fi
PYTHONPATH="$BENCH_ROOT" "$NAVILA_ENV/bin/python" scripts/audit_safe_vln_v5.py \
    --dataset-dir "$DATASET_DIR" --allow-small-dataset \
    --require-navila-teacher \
    --output "$WORKER_LOG/dataset_audit.json"
echo "native_camera_worker=$WORKER completed=$completed range=${START_IDX}:${END_IDX}"
