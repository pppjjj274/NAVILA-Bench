#!/bin/bash
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v6-ppo-v1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
CHECKPOINT=${SAFE_VLN_PPO_CHECKPOINT:-$BENCH_ROOT/checkpoints/safe_vln_v7_critics_warmup}
ROLLOUT_DIR=${SAFE_VLN_PPO_ROLLOUT_DIR:-$BENCH_ROOT/outputs/safe_vln_v7_rollout}
OUTPUT_DIR=${SAFE_VLN_PPO_OUTPUT_DIR:-$BENCH_ROOT/checkpoints/safe_vln_v7_ppo_v1}

if [[ ! -f "$ROLLOUT_DIR/manifest.json" ]]; then
    echo "Missing audited risk rollout: $ROLLOUT_DIR" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite PPO output: $OUTPUT_DIR" >&2
    exit 1
fi

mkdir -p "$BENCH_ROOT/outputs/slurm"
export PYTHONPATH="$NAVILA_ROOT:$BENCH_ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/safe_vln_main.py train \
    --model-path "$NAVILA_ROOT/checkpoints/navila-llama3-8b-8f" \
    --checkpoint "$CHECKPOINT" --rollout-dir "$ROLLOUT_DIR" \
    --output-dir "$OUTPUT_DIR" --device cuda --training-dtype bfloat16 \
    --actor-lr 1e-6 --critic-lr 1e-4 \
    --lagrange-lr 0.035 --ppo-epochs 1 --mini-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --sampling-strategy balanced-ppo --sampling-seed 20260801 \
    --max-samples 4000 --policy-version 0 \
    --oracle-ce-coef 0.0 --oracle-stop-weight 5
