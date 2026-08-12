#!/bin/bash
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v6-certify
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

BENCH=/share/home/202430461770/NaVILA-Bench
NAVILA=/share/home/202430461770/NaVILA
CHECKPOINT=${SAFE_VLN_ACTOR_CHECKPOINT:-$BENCH/checkpoints/safe_vln_v5_actor_factorized_v2}
DATASET_DIR=${SAFE_VLN_ACTOR_DATASET:-$BENCH/outputs/safe_live_v5_oracle_500}
OUTPUT=${SAFE_VLN_ACTOR_CERTIFIED_OUTPUT:-$BENCH/checkpoints/safe_vln_v5_actor_factorized_gated_v1}

mkdir -p "$BENCH/outputs/slurm"
if [[ -e "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite existing certification output: $OUTPUT" >&2
    exit 1
fi

export PYTHONPATH="$NAVILA:$BENCH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/safe_vln_main.py audit-actor \
    --model-path "$NAVILA/checkpoints/navila-llama3-8b-8f" \
    --checkpoint "$CHECKPOINT" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT" \
    --device cuda --training-dtype bfloat16 \
    --goal-stop-contract sensor-gated-v1 --certify \
    --minimum-non-stop-macro-accuracy 0.4 \
    --maximum-false-stop-rate 0.05 \
    --stop-threshold-grid-step 0.01 \
    --sampling-seed 20260729
