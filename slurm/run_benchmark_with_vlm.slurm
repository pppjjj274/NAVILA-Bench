#!/bin/bash


#SBATCH --error=%j.err
module load anaconda/3-2024.02.01
source $(conda info --base)/etc/profile.d/conda.sh
conda activate navila
cd ~/NaVILA-Bench
 export PYTHONPATH=$HOME/NaVILA:$HOME/NaVILA-Bench:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python scripts/vlm_server.py \
  --model_path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --port 55666