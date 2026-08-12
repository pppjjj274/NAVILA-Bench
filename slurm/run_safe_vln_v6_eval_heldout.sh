#!/bin/bash
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v6-eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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

CHECKPOINT=$SAFE_VLN_EVAL_CHECKPOINT
EVAL_TAG=$SAFE_VLN_EVAL_TAG
EVAL_COUNT=$SAFE_VLN_EVAL_COUNT
VLM_PORT=$SAFE_VLN_EVAL_VLM_PORT
RENDER_PORT=$SAFE_VLN_EVAL_RENDER_PORT
OUTPUT_DIR=$BENCH_ROOT/outputs/safe_vln_v6_eval_val_unseen_$EVAL_TAG
LOG_ROOT=$OUTPUT_DIR"_logs"
VAL_META=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz
VAL_GT=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz
MP3D_ROOT=$DATA_ROOT/MP3D-mirror/extracted/MatterPort3D/mp3d
COST_PROFILE=${SAFE_VLN_COST_PROFILE:-}

if [[ -z "$CHECKPOINT" || -z "$EVAL_TAG" || -z "$EVAL_COUNT" || -z "$VLM_PORT" || -z "$RENDER_PORT" ]]; then
    echo "Set SAFE_VLN_EVAL_CHECKPOINT, SAFE_VLN_EVAL_TAG, SAFE_VLN_EVAL_COUNT, SAFE_VLN_EVAL_VLM_PORT, and SAFE_VLN_EVAL_RENDER_PORT" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT/trainer_state.json" ]]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 1
fi
safe_vln_require_policy_checkpoint \
    "$NAVILA_ENV/bin/python" "$BENCH_ROOT" "$CHECKPOINT"
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite evaluation output: $OUTPUT_DIR" >&2
    exit 1
fi

mkdir -p "$LOG_ROOT" "$BENCH_ROOT/outputs/slurm"
cd "$BENCH_ROOT"

export http_proxy=http://login04:3128
export https_proxy=http://login04:3128
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost

renderer_pid=
vlm_pid=
COST_PROFILE_ARGS=()
if [[ -n "$COST_PROFILE" ]]; then
    COST_PROFILE_ARGS+=(--safe-cost-profile "$COST_PROFILE")
fi
cleanup() {
    if [[ -n "$renderer_pid" ]]; then kill "$renderer_pid" 2>/dev/null || true; fi
    if [[ -n "$vlm_pid" ]]; then kill "$vlm_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU" PYTHONUNBUFFERED=1 PYTHONPATH="$BENCH_ROOT" \
"$RENDER_ENV/bin/python" scripts/habitat_render_server.py \
    --scenes-root "$MP3D_ROOT" --host 127.0.0.1 --port "$RENDER_PORT" \
    --gpu-device-id 0 >"$LOG_ROOT/renderer.log" 2>&1 &
renderer_pid=$!

CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU" PYTHONUNBUFFERED=1 \
PYTHONPATH="$NAVILA_ROOT:$BENCH_ROOT" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$NAVILA_ENV/bin/python" scripts/vlm_server.py \
    --model_path "$NAVILA_ROOT/checkpoints/navila-llama3-8b-8f" \
    --safe_checkpoint "$CHECKPOINT" --host 127.0.0.1 --port "$VLM_PORT" \
    >"$LOG_ROOT/vlm_server.log" 2>&1 &
vlm_pid=$!

for port in "$VLM_PORT" "$RENDER_PORT"; do
    for attempt in $(seq 1 300); do
        if /usr/bin/python3 -c "import socket; s=socket.socket(); s.settimeout(0.2); raise SystemExit(s.connect_ex(('127.0.0.1',$port)))" >/dev/null 2>&1; then
            break
        fi
        if [[ "$attempt" -eq 300 ]]; then
            echo "Service port $port did not become ready" >&2
            exit 1
        fi
        sleep 2
    done
done

EVAL_IDS=$("$NAVILA_ENV/bin/python" scripts/select_vlnce_episode_ids.py \
    --metadata "$VAL_META" --count "$EVAL_COUNT" --offset 0 \
    --seed 20260801 --require-scene-count 11)
printf '%s\n' "$EVAL_IDS" >"$LOG_ROOT/episode_ids.txt"

export CONDA_PREFIX="$ISAAC_ENV"
export GIT_PYTHON_REFRESH=quiet OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce:$BENCH_ROOT/isaaclab_exts/omni.isaac.matterport:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_tasks:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_assets:$BENCH_ROOT"

"$GLIBC_LOADER" --library-path "$GLIBC_LIB:$ISAAC_ENV/lib:/lib64:/usr/lib64" \
    "$ISAAC_ENV/bin/python" scripts/safe_vln_main.py evaluate \
    --safe-live-render --vlnce-episode-ids $EVAL_IDS \
    --vlnce-metadata "$VAL_META" --vlnce-gt "$VAL_GT" \
    --mp3d-scenes-root "$MP3D_ROOT" --render-host 127.0.0.1 --render-port "$RENDER_PORT" \
    --render-timeout-seconds 120 --dataset-role eval --collection-policy vlm \
    --goal-stop-mode sensor-gated --safe-policy-tag "$EVAL_TAG" \
    "${COST_PROFILE_ARGS[@]}" --max-vlm-calls 60 \
    --vlm-host 127.0.0.1 --vlm-port "$VLM_PORT" --dataset-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_ROOT/evaluation.log"

EPISODE_COUNT=$(find "$OUTPUT_DIR/completed" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)
if [[ "$EPISODE_COUNT" -ne "$EVAL_COUNT" ]]; then
    echo "Completed $EPISODE_COUNT/$EVAL_COUNT evaluation episodes" >&2
    exit 1
fi

PYTHONPATH="$BENCH_ROOT" "$NAVILA_ENV/bin/python" scripts/safe_vln_main.py summarize \
    --measurement-dir "$OUTPUT_DIR/episodes" --output "$LOG_ROOT/summary.json"
