#!/bin/bash
# Collect 80 strictly paired Safe-VLN episodes in one 2-GPU allocation.
# The two Isaac/VLM workers share a single Slurm job with 2 GPUs + 18 CPUs;
# each worker is pinned to one GPU and gets an isolated Kit cache.
#
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v9-native
#SBATCH --nodes=1
#SBATCH --nodelist=g02n06
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
ISAAC_ENV=/share/home/202430461770/.conda/envs/vlnce-isaac
NAVILA_ENV=/share/home/202430461770/.conda/envs/navila
TRAIN_META=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
TRAIN_GT=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
USD_ROOT=$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce/assets/matterport_usd
GLIBC_ROOT=/share/software/spack/opt/spack/linux-rocky8-icelake/gcc-8.5.0/glibc-2.38-kbyap6e5vjwnkhmks7d4nbfh3fabixle
GLIBC_LOADER=$GLIBC_ROOT/lib/ld-linux-x86-64.so.2
GLIBC_LIB=$GLIBC_ROOT/lib
source "$BENCH_ROOT/scripts/slurm_gpu_env.sh"
safe_vln_capture_allocated_gpus 2
OUTPUT_ROOT=${SAFE_VLN_NATIVE_OUTPUT_ROOT:-$BENCH_ROOT/outputs/safe_vln_v9_native_camera_canonical}
LOG_ROOT=${SAFE_VLN_NATIVE_LOG_ROOT:-$BENCH_ROOT/outputs/safe_vln_v9_native_camera_logs}
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
if [[ -n "${SAFE_VLN_CHECKPOINT:-}" ]]; then
    echo "Canonical native collection requires original NaViLA; unset SAFE_VLN_CHECKPOINT" >&2
    exit 1
fi
for required in "$TRAIN_META" "$TRAIN_GT"; do
    [[ -f "$required" ]] || { echo "Missing official train data: $required" >&2; exit 1; }
done
if [[ -e "$OUTPUT_ROOT" && -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite native-camera dataset: $OUTPUT_ROOT" >&2
    exit 1
fi

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
export GIT_PYTHON_REFRESH=quiet
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce:$BENCH_ROOT/isaaclab_exts/omni.isaac.matterport:/share/home/202430461770/IsaacLab/source/extensions/omni.isaac.lab:/share/home/202430461770/IsaacLab/source/extensions/omni.isaac.lab_tasks:/share/home/202430461770/IsaacLab/source/extensions/omni.isaac.lab_assets:$BENCH_ROOT"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$BENCH_ROOT/outputs/slurm"
cd "$BENCH_ROOT"

WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# Run each worker in a subshell so its VLM server PID and Kit cleanup trap are
# private to that worker.  The parent job still owns both workers and waits for
# both to finish before returning the combined status.
run_worker() (
    local worker="$1"
    local gpu_token
    gpu_token=$(safe_vln_gpu_token "$worker")
    local start_idx=$((worker * EPISODES_PER_WORKER))
    local end_idx=$(((worker + 1) * EPISODES_PER_WORKER))
    local vlm_port=$((54621 + worker))
    local dataset_dir="$OUTPUT_ROOT/gpu${worker}"
    local worker_log="$LOG_ROOT/gpu${worker}"
    local cache_base="/tmp/${USER}/safe_vln_v9_omni_${SLURM_JOB_ID}/worker${worker}"
    local kit_home="$cache_base/home"
    local r2r_data="$cache_base/tmp/vlnce_train_isaac_balanced.json.gz"
    local vlm_pid=""

    cleanup_worker() {
        if [[ -n "$vlm_pid" ]]; then
            kill "$vlm_pid" 2>/dev/null || true
            wait "$vlm_pid" 2>/dev/null || true
        fi
    }
    trap cleanup_worker EXIT INT TERM

    mkdir -p "$dataset_dir" "$worker_log" "$cache_base/cache" \
        "$cache_base/data" "$cache_base/xdg" "$cache_base/config" \
        "$cache_base/xdg-data" "$cache_base/logs" "$cache_base/tmp" \
        "$kit_home"

    PYTHONPATH="$BENCH_ROOT" "$NAVILA_ENV/bin/python" \
        scripts/convert_vlnce_to_isaac.py \
        --metadata "$TRAIN_META" --gt "$TRAIN_GT" \
        --source-split train --balanced-seed 20260727 \
        --expected-scenes 61 --usd-root "$USD_ROOT" \
        --output "$r2r_data" \
        >"$worker_log/dataset_conversion.json"

    CUDA_VISIBLE_DEVICES="$gpu_token" OMP_NUM_THREADS=9 MKL_NUM_THREADS=9 \
    OPENBLAS_NUM_THREADS=9 NUMEXPR_NUM_THREADS=9 TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$NAVILA_ROOT:$BENCH_ROOT" \
    "$NAVILA_ENV/bin/python" scripts/vlm_server.py \
        --model_path "$NAVILA_ROOT/checkpoints/navila-llama3-8b-8f" \
        --safe_sampling_seed "$((20260802 + worker))" \
        --host 127.0.0.1 --port "$vlm_port" \
        >"$worker_log/vlm_server.log" 2>&1 &
    vlm_pid=$!

    local ready=0
    for _ in $(seq 1 300); do
        if ! kill -0 "$vlm_pid" 2>/dev/null; then
            echo "VLM server exited before becoming ready; see $worker_log/vlm_server.log" >&2
            return 1
        fi
        if /usr/bin/python3 -c \
            "import socket; s=socket.socket(); s.settimeout(.2); raise SystemExit(s.connect_ex(('127.0.0.1',$vlm_port)))" \
            >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 2
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "VLM port $vlm_port did not become ready" >&2
        return 1
    fi

    HOME="$kit_home" OMP_NUM_THREADS=9 MKL_NUM_THREADS=9 \
    OPENBLAS_NUM_THREADS=9 NUMEXPR_NUM_THREADS=9 TOKENIZERS_PARALLELISM=false \
    OMNI_USER_CACHE_DIR="$cache_base/cache" \
    OMNI_USER_DATA="$cache_base/data" \
    XDG_CACHE_HOME="$cache_base/xdg" \
    XDG_CONFIG_HOME="$cache_base/config" \
    XDG_DATA_HOME="$cache_base/xdg-data" \
    OMNI_USER_CACHE="$cache_base/cache" \
    OMNI_USER_CONFIG="$cache_base/config" \
    OMNI_USER_LOGS="$cache_base/logs" \
    CUDA_VISIBLE_DEVICES="$gpu_token" \
    CONDA_PREFIX="$ISAAC_ENV" GLIBC_ROOT="$GLIBC_ROOT" \
    GLIBC_LOADER="$GLIBC_LOADER" GLIBC_LIB="$GLIBC_LIB" \
    "$GLIBC_LOADER" --library-path "$GLIBC_LIB:$ISAAC_ENV/lib:/lib64:/usr/lib64" \
        "$ISAAC_ENV/bin/python" scripts/safe_vln_main.py collect \
        --r2r-data-path "$r2r_data" \
        --start-idx "$start_idx" --end-idx "$end_idx" \
        --vlm-host 127.0.0.1 --vlm-port "$vlm_port" \
        --collection-policy vlm --goal-stop-mode sensor-gated \
        --safe-policy-tag "navila-base-native-camera-gpu${worker}" --online-round 1 \
        --max-vlm-calls "$MAX_VLM_CALLS" --dataset-dir "$dataset_dir" \
        2>&1 | tee "$worker_log/collection.log"

    local completed
    completed=0
    if [[ -d "$dataset_dir/completed" ]]; then
        completed=$(find "$dataset_dir/completed" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)
    fi
    if [[ "$completed" -ne "$EPISODES_PER_WORKER" ]]; then
        echo "Worker $worker collected $completed/$EPISODES_PER_WORKER shards" >&2
        return 1
    fi
    PYTHONPATH="$BENCH_ROOT" "$NAVILA_ENV/bin/python" scripts/audit_safe_vln_v5.py \
        --dataset-dir "$dataset_dir" --allow-small-dataset \
        --require-navila-teacher \
        --output "$worker_log/dataset_audit.json"
    echo "native_camera_worker=$worker completed=$completed range=${start_idx}:${end_idx}"
)

run_worker 0 & worker0_pid=$!
run_worker 1 & worker1_pid=$!
WORKER_PIDS=("$worker0_pid" "$worker1_pid")
status=0
wait "$worker0_pid" || status=1
wait "$worker1_pid" || status=1
exit "$status"
