#!/bin/bash
# Annotate a small native-camera shard with exact-pose Habitat navmesh labels.
# Two renderer workers use one GPU and nine CPUs each.
#
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v8-oracle
#SBATCH --nodes=1
#SBATCH --nodelist=g02n06
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2
#SBATCH --chdir=/share/home/202430461770/NaVILA-Bench
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

BENCH_ROOT=/share/home/202430461770/NaVILA-Bench
NAVILA_ROOT=/share/home/202430461770/NaVILA
RENDER_ENV=/share/home/202430461770/.conda/envs/vlnce3
source "$BENCH_ROOT/scripts/slurm_gpu_env.sh"
safe_vln_capture_allocated_gpus 2
GPU0=$(safe_vln_gpu_token 0)
GPU1=$(safe_vln_gpu_token 1)
: "${SAFE_VLN_NATIVE_ORACLE_SOURCE:?set SAFE_VLN_NATIVE_ORACLE_SOURCE to a current transactional native dataset}"
DATA_ROOT=$SAFE_VLN_NATIVE_ORACLE_SOURCE
OUTPUT_ROOT=${SAFE_VLN_NATIVE_ORACLE_OUTPUT:-$BENCH_ROOT/outputs/safe_vln_native_oracle_${SLURM_JOB_ID}}
R2R_DATA=${SAFE_VLN_NATIVE_ORACLE_R2R:-/tmp/${USER}/safe_vln_native_oracle_${SLURM_JOB_ID}/vlnce_train_isaac.json.gz}
EPISODES_PER_WORKER=${SAFE_VLN_NATIVE_ORACLE_EPISODES:-2}
MP3D_ROOT=${SAFE_VLN_NATIVE_ORACLE_MP3D_ROOT:-/share/home/202430461770/NaVILA-Dataset/MP3D-mirror/extracted/MatterPort3D/mp3d}
LOG_ROOT=${SAFE_VLN_NATIVE_ORACLE_LOG_ROOT:-$OUTPUT_ROOT/logs}

[[ -x "$RENDER_ENV/bin/python" ]] || { echo "missing renderer environment: $RENDER_ENV" >&2; exit 1; }
[[ -d "$MP3D_ROOT" ]] || { echo "missing MP3D root: $MP3D_ROOT" >&2; exit 1; }
[[ -d "$DATA_ROOT/gpu0" && -d "$DATA_ROOT/gpu1" ]] || { echo "missing native worker datasets: $DATA_ROOT" >&2; exit 1; }
[[ "$EPISODES_PER_WORKER" -gt 0 ]] || { echo "episode count must be positive" >&2; exit 1; }
if [[ -e "$OUTPUT_ROOT" && -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "refusing to overwrite oracle output: $OUTPUT_ROOT" >&2
    exit 1
fi

module load anaconda/3-2024.02.01
source /share/software/anaconda3/2024.02.01/etc/profile.d/conda.sh
conda activate "$RENDER_ENV"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$BENCH_ROOT/outputs/slurm"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$BENCH_ROOT"
export OUTPUT_ROOT
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost

if [[ -z "${SAFE_VLN_NATIVE_ORACLE_R2R:-}" ]]; then
    mkdir -p "$(dirname "$R2R_DATA")"
    "$RENDER_ENV/bin/python" scripts/convert_vlnce_to_isaac.py \
        --metadata "$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz" \
        --gt "$NAVILA_ROOT/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz" \
        --source-split train --balanced-seed 20260727 --expected-scenes 61 \
        --usd-root "$BENCH_ROOT/isaaclab_exts/omni.isaac.vlnce/assets/matterport_usd" \
        --output "$R2R_DATA" >"$LOG_ROOT/dataset_conversion.json"
fi
[[ -f "$R2R_DATA" ]] || { echo "missing converted R2R train asset: $R2R_DATA" >&2; exit 1; }

PIDS=()
cleanup() {
    for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

start_renderer() {
    local gpu="$1" port="$2" worker="$3"
    local base="/tmp/${USER}/safe_vln_native_oracle_${SLURM_JOB_ID}_${worker}"
    mkdir -p "$base"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$BENCH_ROOT" \
        "$RENDER_ENV/bin/python" scripts/habitat_render_server.py \
        --scenes-root "$MP3D_ROOT" --host 127.0.0.1 --port "$port" \
        --gpu-device-id 0 >"$LOG_ROOT/renderer${worker}.log" 2>&1 &
    PIDS+=("$!")
}

wait_port() {
    local port="$1"
    for _ in $(seq 1 180); do
        if /usr/bin/python3 -c \
            "import socket; s=socket.socket(); s.settimeout(.2); raise SystemExit(s.connect_ex(('127.0.0.1',$port)))" \
            >/dev/null 2>&1; then return 0; fi
        sleep 2
    done
    echo "renderer port $port did not become ready" >&2
    return 1
}

start_renderer "$GPU0" 54322 0
start_renderer "$GPU1" 54332 1
wait_port 54322
wait_port 54332

annotate_worker() {
    local worker="$1" port="$2"
    local gpu_token
    gpu_token=$(safe_vln_gpu_token "$worker")
    PYTHONPATH="$BENCH_ROOT" CUDA_VISIBLE_DEVICES="$gpu_token" \
        "$RENDER_ENV/bin/python" scripts/annotate_native_oracle.py \
        --source-dir "$DATA_ROOT/gpu${worker}" \
        --r2r-data-path "$R2R_DATA" \
        --episode-limit "$EPISODES_PER_WORKER" \
        --render-host 127.0.0.1 --render-port "$port" \
        --render-timeout-seconds 120 \
        --allow-diagnostic-navmesh-teacher \
        --output "$OUTPUT_ROOT/gpu${worker}.json" \
        >"$LOG_ROOT/annotate${worker}.log" 2>&1
}

annotate_worker 0 54322 & PIDS+=("$!")
annotate_worker 1 54332 & PIDS+=("$!")
status=0
for pid in "${PIDS[@]:2}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || exit 1

PYTHONPATH="$BENCH_ROOT" "$RENDER_ENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["OUTPUT_ROOT"])
reports = [json.loads(path.read_text()) for path in sorted(root.glob("gpu*.json"))]
payload = {
    "schema_version": "safe-vln-native-oracle-v1",
    "workers": len(reports),
    "episodes": sum(len(item["episodes"]) for item in reports),
    "transitions": sum(item["transitions"] for item in reports),
    "oracle_valid": sum(item["oracle_valid"] for item in reports),
    "oracle_invalid": sum(item["oracle_invalid"] for item in reports),
}
(root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload))
PY
