#!/bin/bash
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v6-risk80
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

CHECKPOINT=${SAFE_VLN_CHECKPOINT:?set SAFE_VLN_CHECKPOINT to an independently audited policy checkpoint}
RISK_IDS_FILE=${SAFE_VLN_RISK_IDS_FILE:-$BENCH_ROOT/outputs/safe_vln_v6_risk80.txt}
DATASET_DIR=${SAFE_VLN_DATASET_DIR:-$BENCH_ROOT/outputs/safe_vln_v7_rollout}
LOG_ROOT=${SAFE_VLN_LOG_ROOT:-$BENCH_ROOT/outputs/safe_vln_v7_rollout_logs}
EPISODE_LIMIT=${SAFE_VLN_EPISODE_LIMIT:-0}
EPISODE_IDS=${SAFE_VLN_EPISODE_IDS:-}
VLM_PORT=${SAFE_VLN_VLM_PORT:-54321}
RENDER_PORT=${SAFE_VLN_RENDER_PORT:-54322}
POLICY_TAG=${SAFE_VLN_POLICY_TAG:-v7-p0-seed20260801}
EXPECTED_POLICY_VERSION=${SAFE_VLN_EXPECTED_POLICY_VERSION:-0}
STOCHASTIC=${SAFE_VLN_STOCHASTIC:-1}
SAMPLING_SEED=${SAFE_VLN_SAMPLING_SEED:-20260801}
ONLINE_ROUND=${SAFE_VLN_ONLINE_ROUND:-1}
REQUIRE_ONLINE_DAGGER=${SAFE_VLN_REQUIRE_ONLINE_DAGGER:-0}
TRAIN_META=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
TRAIN_GT=$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
MP3D_ROOT=$DATA_ROOT/MP3D-mirror/extracted/MatterPort3D/mp3d
COST_PROFILE=${SAFE_VLN_COST_PROFILE:-}

if [[ ! -f "$CHECKPOINT/trainer_state.json" || ! -f "$CHECKPOINT/adapter_config.json" ]]; then
    echo "Missing completed audited policy checkpoint: $CHECKPOINT" >&2
    exit 1
fi
if [[ -e "$DATASET_DIR" && -n "$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to append to rollout dataset: $DATASET_DIR" >&2
    exit 1
fi
if [[ -n "$EPISODE_IDS" ]]; then
    RISK_IDS="$EPISODE_IDS"
elif [[ "$EPISODE_LIMIT" -gt 0 ]]; then
    RISK_IDS=$(head -n "$EPISODE_LIMIT" "$RISK_IDS_FILE" | tr '\n' ' ')
else
    RISK_IDS=$(tr '\n' ' ' < "$RISK_IDS_FILE")
fi
RISK_COUNT=$(wc -w <<<"$RISK_IDS")
if [[ "$RISK_COUNT" -lt 1 ]]; then
    echo "Risk ID file must contain at least one episode" >&2
    exit 1
fi
VLM_SAMPLING_ARGS=()
if [[ "$STOCHASTIC" == 1 ]]; then
    VLM_SAMPLING_ARGS+=(--no-safe_deterministic)
fi
COST_PROFILE_ARGS=()
if [[ -n "$COST_PROFILE" ]]; then
    COST_PROFILE_ARGS+=(--safe-cost-profile "$COST_PROFILE")
fi
ONLINE_ORACLE_ARGS=()
if [[ "${SAFE_VLN_ALLOW_ONLINE_ORACLE:-0}" == "1" ]]; then
    ONLINE_ORACLE_ARGS+=(--allow-online-oracle)
fi
AUDIT_DAGGER_ARGS=()
if [[ "$REQUIRE_ONLINE_DAGGER" == 1 ]]; then
    AUDIT_DAGGER_ARGS+=(--require-online-dagger)
fi
if [[ "${SAFE_VLN_ALLOW_ONLINE_ORACLE:-0}" == "1" ]]; then
    AUDIT_DAGGER_ARGS+=(--allow-online-oracle)
fi
safe_vln_require_policy_checkpoint \
    "$NAVILA_ENV/bin/python" "$BENCH_ROOT" "$CHECKPOINT" "$EXPECTED_POLICY_VERSION"

mkdir -p "$LOG_ROOT" "$BENCH_ROOT/outputs/slurm"
cd "$BENCH_ROOT"

export http_proxy=${http_proxy:-http://login04:3128}
export https_proxy=${https_proxy:-http://login04:3128}
export HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
export HTTPS_PROXY=${HTTPS_PROXY:-$https_proxy}

renderer_pid=
vlm_pid=
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
    --safe_checkpoint "$CHECKPOINT" \
    "${VLM_SAMPLING_ARGS[@]}" \
    --safe_sampling_seed "$SAMPLING_SEED" \
    --host 127.0.0.1 --port "$VLM_PORT" >"$LOG_ROOT/vlm_server.log" 2>&1 &
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
    if [[ "$ready" -ne 1 ]]; then
        echo "Service port $port did not become ready" >&2
        exit 1
    fi
done

export CONDA_PREFIX="$ISAAC_ENV"
export GLIBC_ROOT GLIBC_LOADER GLIBC_LIB
export GIT_PYTHON_REFRESH=quiet OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
export PYTHONPATH="$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce:$BENCH_ROOT/isaaclab_exts/omni.isaac.matterport:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_tasks:$ISAACLAB_ROOT/source/extensions/omni.isaac.lab_assets:$BENCH_ROOT"

"$GLIBC_LOADER" \
    --library-path "$GLIBC_LIB:$ISAAC_ENV/lib:/lib64:/usr/lib64" \
    "$ISAAC_ENV/bin/python" scripts/safe_vln_main.py collect \
    --safe-live-render --vlnce-episode-ids $RISK_IDS \
    --vlnce-metadata "$TRAIN_META" --vlnce-gt "$TRAIN_GT" \
    --mp3d-scenes-root "$MP3D_ROOT" --render-host 127.0.0.1 \
    --render-port "$RENDER_PORT" --render-timeout-seconds 120 \
    --vlm-host 127.0.0.1 --vlm-port "$VLM_PORT" \
    --dataset-role train --collection-policy vlm \
    --goal-stop-mode sensor-gated --safe-policy-tag "$POLICY_TAG" \
    --online-round "$ONLINE_ROUND" \
    "${COST_PROFILE_ARGS[@]}" "${ONLINE_ORACLE_ARGS[@]}" \
    --max-vlm-calls 60 \
    --dataset-dir "$DATASET_DIR" 2>&1 | tee "$LOG_ROOT/collection.log"

EPISODE_COUNT=$(find "$DATASET_DIR/completed" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)
if [[ "$EPISODE_COUNT" -ne "$RISK_COUNT" ]]; then
    echo "Collected $EPISODE_COUNT/$RISK_COUNT completed episodes; refusing audit and PPO" >&2
    exit 1
fi

"$ISAAC_ENV/bin/python" scripts/audit_safe_vln_v5.py \
    --dataset-dir "$DATASET_DIR" --expected-episode-ids <(printf '%s\n' $RISK_IDS) \
    --allow-small-dataset --require-on-policy --expected-policy-version "$EXPECTED_POLICY_VERSION" \
    "${AUDIT_DAGGER_ARGS[@]}" \
    --output "$LOG_ROOT/audit.json"
