#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT=${BENCH_ROOT:-/share/home/202430461770/NaVILA-Bench}
NAVILA_ROOT=${NAVILA_ROOT:-/share/home/202430461770/NaVILA}
NAVILA_ENV=${NAVILA_ENV:-/share/home/202430461770/.conda/envs/navila}
DATASET_DIR=${DATASET_DIR:-$BENCH_ROOT/outputs/safe_live_v5_oracle_500}
OUTPUT_DIR=${OUTPUT_DIR:-$BENCH_ROOT/checkpoints/safe_vln_v5_actor_hierarchical_v2}
LOG_ROOT=${LOG_ROOT:-$BENCH_ROOT/outputs/safe_vln_v5_actor_hierarchical_v2_logs}
MAX_SAMPLES=${MAX_SAMPLES:-4000}
ACTOR_TARGET_SOURCE=${ACTOR_TARGET_SOURCE:-navila-policy}

mkdir -p "$LOG_ROOT"
cd "$BENCH_ROOT"
export CONDA_PREFIX="$NAVILA_ENV"
export PYTHONPATH="$NAVILA_ROOT:$BENCH_ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$NAVILA_ENV/bin/python" scripts/safe_vln_main.py warmup-actor \
    --model-path "$NAVILA_ROOT/checkpoints/navila-llama3-8b-8f" \
    --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_DIR" \
    --device cuda --training-dtype bfloat16 \
    --actor-architecture hierarchical-stop-motion \
    --actor-target-source "$ACTOR_TARGET_SOURCE" \
    --head-warmup-epochs 20 --head-warmup-lr 3e-4 \
    --head-batch-size 256 --actor-lr 1e-6 --head-lr 1e-4 \
    --epochs 1 --gradient-accumulation-steps 4 --max-grad-norm 0.5 \
    --max-samples "$MAX_SAMPLES" --stop-fraction 0.10 \
    --hard-stop-negative-fraction 0.25 --hard-stop-negative-margin-m 1.0 \
    --stop-threshold 0.5 --calibrate-stop-threshold \
    --stop-threshold-grid-step 0.01 \
    --calibration-episodes-per-scene 1 --audit-episodes-per-scene 1 \
    --sampling-strategy stratified --sampling-seed 20260729 \
    --minimum-stop-accuracy 0.5 \
    --maximum-false-stop-rate 0.05 \
    --minimum-non-stop-macro-accuracy 0.4 \
    2>&1 | tee "$LOG_ROOT/training.log"
