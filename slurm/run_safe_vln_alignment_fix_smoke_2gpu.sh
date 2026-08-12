#!/bin/bash
# Regression smoke test for the warmup alignment fix.
# GPU0: original Isaac RGB camera path.
# GPU1: strict Habitat live-render path using the patched wrapper.
#
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-align-fix
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=128G
#SBATCH --time=04:00:00
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

ISAAC_EPISODE=${SAFE_ALIGNMENT_ISAAC_EPISODE:-0}
LIVE_EPISODE=${SAFE_ALIGNMENT_LIVE_EPISODE:-6906}
MAX_CALLS=${SAFE_ALIGNMENT_MAX_VLM_CALLS:-10}
ISAAC_PORT=${SAFE_ALIGNMENT_ISAAC_PORT:-54521}
LIVE_PORT=${SAFE_ALIGNMENT_LIVE_PORT:-54531}
RENDER_PORT=${SAFE_ALIGNMENT_RENDER_PORT:-54532}
RUN_ROOT=$BENCH_ROOT/outputs/safe_vln_alignment_fix_smoke
ISAAC_LOG=$RUN_ROOT/isaac_camera
LIVE_LOG=$RUN_ROOT/strict_live
TRAIN_META=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
TRAIN_GT=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
MP3D_ROOT=$DATA_ROOT/MP3D-mirror/extracted/MatterPort3D/mp3d

if [[ -e "$RUN_ROOT" && -n "$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite smoke output: $RUN_ROOT" >&2
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
mkdir -p "$ISAAC_LOG" "$LIVE_LOG" "$BENCH_ROOT/outputs/slurm"
cd "$BENCH_ROOT"

isaac_vlm_pid=
live_vlm_pid=
renderer_pid=
cleanup() {
    for pid in "$isaac_vlm_pid" "$live_vlm_pid" "$renderer_pid"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$GPU0" PYTHONUNBUFFERED=1 PYTHONPATH="$NAVILA_ROOT:$BENCH_ROOT" \
"$NAVILA_ENV/bin/python" scripts/vlm_server.py \
    --model_path "$NAVILA_ROOT/checkpoints/navila-llama3-8b-8f" \
    --host 127.0.0.1 --port "$ISAAC_PORT" >"$ISAAC_LOG/vlm_server.log" 2>&1 &
isaac_vlm_pid=$!

CUDA_VISIBLE_DEVICES="$GPU1" PYTHONUNBUFFERED=1 PYTHONPATH="$NAVILA_ROOT:$BENCH_ROOT" \
"$NAVILA_ENV/bin/python" scripts/vlm_server.py \
    --model_path "$NAVILA_ROOT/checkpoints/navila-llama3-8b-8f" \
    --host 127.0.0.1 --port "$LIVE_PORT" >"$LIVE_LOG/vlm_server.log" 2>&1 &
live_vlm_pid=$!

CUDA_VISIBLE_DEVICES="$GPU1" PYTHONUNBUFFERED=1 PYTHONPATH="$BENCH_ROOT" \
"$RENDER_ENV/bin/python" scripts/habitat_render_server.py \
    --scenes-root "$MP3D_ROOT" --host 127.0.0.1 --port "$RENDER_PORT" \
    --gpu-device-id 0 >"$LIVE_LOG/renderer.log" 2>&1 &
renderer_pid=$!

for port in "$ISAAC_PORT" "$LIVE_PORT" "$RENDER_PORT"; do
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

export GIT_PYTHON_REFRESH=quiet OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce:$BENCH_ROOT/isaaclab_exts/omni.isaac.matterport:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_tasks:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_assets:$BENCH_ROOT"

run_isaac() {
    export CUDA_VISIBLE_DEVICES="$GPU0" CONDA_PREFIX="$ISAAC_ENV" GLIBC_ROOT GLIBC_LOADER GLIBC_LIB
    "$GLIBC_LOADER" --library-path "$GLIBC_LIB:$ISAAC_ENV/lib:/lib64:/usr/lib64" \
        "$ISAAC_ENV/bin/python" scripts/navila_eval.py \
        --task=go2_matterport_vision --num_envs=1 --history_length=9 \
        --load_run=2024-09-25_23-22-02 --headless --enable_cameras --safe-vln \
        --episode_idx="$ISAAC_EPISODE" --vlm_host=127.0.0.1 --vlm_port="$ISAAC_PORT" \
        --max_vlm_calls="$MAX_CALLS" --safe-policy-tag=alignment-fix-isaac \
        >"$ISAAC_LOG/run.log" 2>&1
}

run_strict_live() {
    export CUDA_VISIBLE_DEVICES="$GPU1" CONDA_PREFIX="$ISAAC_ENV" GLIBC_ROOT GLIBC_LOADER GLIBC_LIB
    "$GLIBC_LOADER" --library-path "$GLIBC_LIB:$ISAAC_ENV/lib:/lib64:/usr/lib64" \
        "$ISAAC_ENV/bin/python" scripts/navila_eval.py \
        --task=go2_matterport_vision --num_envs=1 --history_length=9 \
        --load_run=2024-09-25_23-22-02 --headless --safe-vln --safe-live-render \
        --vlnce-episode-id="$LIVE_EPISODE" --vlnce-metadata="$TRAIN_META" \
        --vlnce-gt="$TRAIN_GT" --mp3d-scenes-root="$MP3D_ROOT" \
        --render-host=127.0.0.1 --render-port="$RENDER_PORT" \
        --render-timeout-seconds=120 --vlm_host=127.0.0.1 --vlm_port="$LIVE_PORT" \
        --dataset-role=train --max_vlm_calls="$MAX_CALLS" \
        --safe-policy-tag=alignment-fix-strict-live \
        >"$LIVE_LOG/run.log" 2>&1
}

run_isaac & isaac_pid=$!
run_strict_live & live_pid=$!
status=0
wait "$isaac_pid" || status=1
wait "$live_pid" || status=1
echo "alignment_fix_smoke_exit_status=$status"
exit "$status"
