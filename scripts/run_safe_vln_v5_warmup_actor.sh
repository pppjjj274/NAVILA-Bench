#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT=${BENCH_ROOT:-/share/home/202430461770/NaVILA-Bench}
NAVILA_ROOT=${NAVILA_ROOT:-/share/home/202430461770/NaVILA}
NAVILA_ENV=${NAVILA_ENV:-/share/home/202430461770/.conda/envs/navila}
DATASET_DIR=${DATASET_DIR:-$BENCH_ROOT/outputs/safe_live_v5_oracle_500}
OUTPUT_DIR=${OUTPUT_DIR:-$BENCH_ROOT/checkpoints/safe_vln_v5_actor_hierarchical_v1}
LOG_ROOT=${LOG_ROOT:-$BENCH_ROOT/outputs/safe_vln_v5_actor_hierarchical_v1_logs}
MAX_SAMPLES=${MAX_SAMPLES:-4000}

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
    --head-warmup-epochs 20 --head-warmup-lr 3e-4 \
    --head-batch-size 256 --actor-lr 1e-6 --head-lr 1e-4 \
    --epochs 1 --gradient-accumulation-steps 4 --max-grad-norm 0.5 \
    --max-samples "$MAX_SAMPLES" --stop-fraction 0.25 \
    --stop-threshold 0.5 --dev-episodes-per-scene 1 \
    --sampling-seed 20260729 --minimum-stop-accuracy 0.5 \
    --maximum-false-stop-rate 0.05 \
    --minimum-non-stop-macro-accuracy 0.4 \
    2>&1 | tee "$LOG_ROOT/training.log"
