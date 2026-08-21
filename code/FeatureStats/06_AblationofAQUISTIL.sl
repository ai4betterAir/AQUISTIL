#!/bin/bash
# tell this is a bash script
 
# Request 4 CPU, 500MB of memory and 10 minutes of walltime
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120GB
#SBATCH --time=12:10:00

#SBATCH -o X1_stage6.AblationofAQUISTIL_%j.txt
#SBATCH --chdir=/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/code/FeatureStats
 
# Specify where(partition) to run the job
#SBATCH --partition=GPU
 
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/ahmedmas/.conda/envs/ai4air/lib/python3.11/site-packages:/home/ahmedmas/.local/lib/python3.11/site-packages
export LD_LIBRARY_PATH=/home/ahmedmas/.conda/envs/ai4air/lib
export MPLCONFIGDIR=/tmp/aquistil_matplotlib

echo "$(date '+%F %T') starting stage 6 ablation on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "PWD=$(pwd)"
/home/ahmedmas/.conda/envs/ai4air/bin/python -u 6.AblationofAQUISTIL.py
echo "$(date '+%F %T') finished stage 6 ablation"
