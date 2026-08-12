#!/bin/bash
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH --job-name=safe-vln-v5-factorized-v2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/share/home/202430461770/NaVILA-Bench
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

module load anaconda/3-2024.02.01
source /share/software/anaconda3/2024.02.01/etc/profile.d/conda.sh
conda activate /share/home/202430461770/.conda/envs/navila

mkdir -p outputs/slurm
bash scripts/run_safe_vln_v5_warmup_actor.sh
