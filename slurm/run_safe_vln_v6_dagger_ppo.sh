#!/bin/bash
# This job is intentionally fail-closed: it starts only after gated evaluation passes.
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v6-dagger-ppo
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
    echo "Online-Oracle DAgger PPO is a retired diagnostic ablation. Use the audited hierarchical Safe-PPO path; set SAFE_VLN_ALLOW_ONLINE_ORACLE=1 only for an explicit ablation." >&2
    exit 2
fi

module load anaconda/3-2024.02.01
source /share/software/anaconda3/2024.02.01/etc/profile.d/conda.sh
conda activate /share/home/202430461770/.conda/envs/navila

BENCH=/share/home/202430461770/NaVILA-Bench
NAVILA=/share/home/202430461770/NaVILA
CHECKPOINT=${SAFE_VLN_PPO_CHECKPOINT:?set SAFE_VLN_PPO_CHECKPOINT to the certified DAgger actor}
ROLLOUT_DIR=${SAFE_VLN_PPO_ROLLOUT_DIR:?set SAFE_VLN_PPO_ROLLOUT_DIR to a fresh audited DAgger rollout}
CANDIDATE_SUMMARY=${SAFE_VLN_PPO_EVAL_SUMMARY:?set SAFE_VLN_PPO_EVAL_SUMMARY to the 22-episode gated evaluation summary}
BASELINE_SUMMARY=${SAFE_VLN_PPO_BASELINE_SUMMARY:-$BENCH/outputs/safe_vln_v6_eval_val_unseen_v0_logs/summary.json}
GATE_REPORT=${SAFE_VLN_PPO_GATE_REPORT:-$BENCH/outputs/safe_vln_v6_dagger_acceptance.json}
OUTPUT_DIR=${SAFE_VLN_PPO_OUTPUT_DIR:-$BENCH/checkpoints/safe_vln_v6_dagger_ppo_r1}
POLICY_VERSION=${SAFE_VLN_PPO_POLICY_VERSION:-0}

if [[ ! -f "$CHECKPOINT/trainer_state.json" || ! -f "$ROLLOUT_DIR/manifest.json" ]]; then
    echo "Missing certified checkpoint or audited rollout" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite PPO output: $OUTPUT_DIR" >&2
    exit 1
fi

mkdir -p "$BENCH/outputs/slurm"
export PYTHONPATH="$NAVILA:$BENCH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c \
    "import json; s=json.load(open('$CHECKPOINT/trainer_state.json')); assert s.get('actor/accepted') is True; assert s.get('actor/goal_stop_contract') == 'sensor-gated-v1'; assert int(s.get('policy_version', 0)) == int('$POLICY_VERSION')"

python scripts/audit_safe_vln_v5.py \
    --dataset-dir "$ROLLOUT_DIR" --allow-small-dataset --require-on-policy \
    --require-online-dagger --expected-policy-version "$POLICY_VERSION"

python scripts/check_safe_vln_acceptance.py \
    --baseline "$BASELINE_SUMMARY" --candidate "$CANDIDATE_SUMMARY" \
    --output "$GATE_REPORT"

python scripts/safe_vln_main.py train \
    --model-path "$NAVILA/checkpoints/navila-llama3-8b-8f" \
    --checkpoint "$CHECKPOINT" --rollout-dir "$ROLLOUT_DIR" \
    --output-dir "$OUTPUT_DIR" --device cuda --training-dtype bfloat16 \
    --actor-lr 1e-6 --critic-lr 1e-4 --cost-limit 0.25 \
    --lagrange-lr 0.035 --ppo-epochs 1 --mini-batch-size 1 \
    --sampling-strategy balanced-ppo --sampling-seed 20260802 \
    --max-samples 4000 --policy-version "$POLICY_VERSION" \
    --oracle-ce-coef 0.20 --oracle-stop-weight 5
