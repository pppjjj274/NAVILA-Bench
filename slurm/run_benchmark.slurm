#!/bin/bash

#SBATCH --job-name=navila-bench
#SBATCH --partition=gpuA800
#SBATCH -N 1
#SBATCH -w g02n04
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
module load anaconda/3-2024.02.01
source /share/software/anaconda3/2024.02.01/etc/profile.d/conda.sh
source /share/software/spack/share/spack/setup-env.sh
spack env activate glibc238
conda activate /share/home/202430461770/.conda/envs/vlnce-isaac

export GLIBC_ROOT=/share/software/spack/opt/spack/linux-rocky8-icelake/gcc-8.5.0/glibc-2.38-kbyap6e5vjwnkhmks7d4nbfh3fabixle
export GLIBC_LOADER=$GLIBC_ROOT/lib/ld-linux-x86-64.so.2
export GLIBC_LIB=$GLIBC_ROOT/lib
export GLIBC_RUN="$GLIBC_LOADER --library-path $GLIBC_LIB:$CONDA_PREFIX/lib:/lib64:/usr/lib64"
export http_proxy=http://login04:3128
export https_proxy=http://login04:3128
export HTTP_PROXY=http://login04:3128
export HTTPS_PROXY=http://login04:3128

export PYTHONPATH=$HOME/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce:$HOME/NaVILA-Bench/isaaclab_exts/omni.isaac.matterport:$PYTHONPATH
export PYTHONPATH=$HOME/IsaacLab/source/extensions/omni.isaac.lab:$HOME/IsaacLab/source/extensions/omni.isaac.lab_tasks:$HOME/IsaacLab/source/extensions/omni.isaac.lab_assets:$PYTHONPATH
export GIT_PYTHON_REFRESH=quiet
export OMNI_KIT_ACCEPT_EULA=YES
cd ~/NaVILA-Bench
export CUDA_VISIBLE_DEVICES=0
$GLIBC_LOADER \
  --library-path "$GLIBC_LIB:$CONDA_PREFIX/lib:/lib64:/usr/lib64" \
  "$CONDA_PREFIX/bin/python" scripts/run_benchmark.py \
  --task=go2_matterport_vision \
  --low_level_policy_dir=2024-09-25_23-22-02 \
  --vlm_host 127.0.0.1 \
  --vlm_port 55666 \
