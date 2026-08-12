#!/bin/bash
# Online DAgger actor recovery update.  Run certification next, before serving it.
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v6-dagger-actor
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/share/home/202430461770/NaVILA-Bench
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

if [[ "${SAFE_VLN_ALLOW_ONLINE_ORACLE:-0}" != "1" ]]; then
    echo "Online DAgger is a retired diagnostic ablation. Recollect strict base-NaViLA data and use the hierarchical distillation mainline; set SAFE_VLN_ALLOW_ONLINE_ORACLE=1 only for an explicit ablation." >&2
    exit 2
fi

module load anaconda/3-2024.02.01
source /share/software/anaconda3/2024.02.01/etc/profile.d/conda.sh
conda activate /share/home/202430461770/.conda/envs/navila

BENCH=/share/home/202430461770/NaVILA-Bench
NAVILA=/share/home/202430461770/NaVILA
CHECKPOINT=${SAFE_VLN_DAGGER_SOURCE_CHECKPOINT:-$BENCH/checkpoints/safe_vln_v6_actor_balanced_gated_v1}
ROLLOUT_DIR=${SAFE_VLN_DAGGER_ROLLOUT_DIR:-$BENCH/outputs/safe_vln_v6_dagger_r1_rollout}
ANCHOR_DIR=${SAFE_VLN_DAGGER_ANCHOR_DIR:-$BENCH/outputs/safe_live_v5_oracle_500}
OUTPUT_DIR=${SAFE_VLN_DAGGER_OUTPUT_DIR:-$BENCH/checkpoints/safe_vln_v6_dagger_actor_r1}

if [[ ! -f "$CHECKPOINT/trainer_state.json" ]]; then
    echo "Missing certified source checkpoint: $CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$ROLLOUT_DIR/manifest.json" || ! -f "$ANCHOR_DIR/manifest.json" ]]; then
    echo "Missing online rollout or static oracle anchor dataset" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite actor output: $OUTPUT_DIR" >&2
    exit 1
fi

mkdir -p "$BENCH/outputs/slurm"
export PYTHONPATH="$NAVILA:$BENCH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/safe_vln_main.py dagger-actor \
    --model-path "$NAVILA/checkpoints/navila-llama3-8b-8f" \
    --checkpoint "$CHECKPOINT" \
    --rollout-dir "$ROLLOUT_DIR" --anchor-dataset-dir "$ANCHOR_DIR" \
    --output-dir "$OUTPUT_DIR" --device cuda --training-dtype bfloat16 \
    --actor-lr 1e-6 --head-lr 1e-4 --gradient-accumulation-steps 4 \
    --max-grad-norm 0.5 --epochs 1 --max-samples 4000 \
    --online-fraction 0.60 --online-round 1 --sampling-seed 20260802 \
    --allow-small-dataset
