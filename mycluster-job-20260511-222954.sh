#!/bin/bash
#SBATCH -A a_yifanliu
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH -J mycluster-job-20260511-222954
#SBATCH --nodes=1
#SBATCH -c 9
#SBATCH --time=600
#SBATCH --chdir=/share/home/202430461770/NaVILA-Bench
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --gres=gpu:1

source /etc/profile.d/lmod.sh
$GLIBC_RUN $CONDA_PREFIX/bin/python scripts/run_benchmark.py \
  --task=go2_matterport_vision \
  --low_level_policy_dir=2024-09-25_23-22-02