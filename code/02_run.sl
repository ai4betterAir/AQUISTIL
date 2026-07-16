#!/bin/bash
# tell this is a bash script
set -euo pipefail
 
# Request 4 CPU, 500MB of memory and 10 minutes of walltime
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120GB
#SBATCH --time=12:10:00

#SBATCH -o output_main_models.txt
 
# Specify where(partition) to run the job
#SBATCH --partition=GPU

cd /mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/code

echo "Started main.py event model evaluation at $(date) on $(hostname)"
echo "Python: $(command -v python3)"
python3 --version

# -u flushes logs immediately so progress is visible with:
# tail -f output_main_models.txt
python3 -u main.py --regime event --skip-api-refresh

echo "Finished event evaluation at $(date)"



