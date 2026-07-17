#!/bin/bash
#SBATCH -A a_yifanliu
#SBATCH --partition=cpuXeon6458
#SBATCH --qos=normal
#SBATCH -J navila_benchmark_demo
#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --time=30
#SBATCH --chdir=/share/home/202430461770/NaVILA-Bench
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

source /etc/profile.d/lmod.sh
sbatch ~/job_scripts/navila_benchmark_demo.sbatch