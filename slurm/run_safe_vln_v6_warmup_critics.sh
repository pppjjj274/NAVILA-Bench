#!/bin/bash
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v6-critics
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/share/home/202430461770/NaVILA-Bench
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

module load anaconda/3-2024.02.01
source /share/software/anaconda3/2024.02.01/etc/profile.d/conda.sh
conda activate /share/home/202430461770/.conda/envs/navila

BENCH_ROOT=/share/home/202430461770/NaVILA-Bench
NAVILA_ROOT=/share/home/202430461770/NaVILA
SOURCE_CHECKPOINT=${SAFE_VLN_CRITIC_SOURCE_CHECKPOINT:-}
DATASET_DIR=${SAFE_VLN_CRITIC_DATASET:-$BENCH_ROOT/outputs/safe_vln_v7_strict_critic}
OUTPUT_DIR=${SAFE_VLN_CRITIC_OUTPUT:-$BENCH_ROOT/checkpoints/safe_vln_v7_critics_warmup}

mkdir -p "$BENCH_ROOT/outputs/slurm"
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite critic output: $OUTPUT_DIR" >&2
    exit 1
fi

export PYTHONPATH="$NAVILA_ROOT:$BENCH_ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT_ARGS=()
if [[ -n "$SOURCE_CHECKPOINT" ]]; then
    CHECKPOINT_ARGS+=(--checkpoint "$SOURCE_CHECKPOINT" --reset-critics)
fi

python scripts/safe_vln_main.py warmup-critics \
    --model-path "$NAVILA_ROOT/checkpoints/navila-llama3-8b-8f" \
    "${CHECKPOINT_ARGS[@]}" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda --training-dtype bfloat16 \
    --critic-lr 1e-4 --epochs 1 \
    --sampling-strategy balanced-critic --sampling-seed 20260801 \
    --max-samples 4000
