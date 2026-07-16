#!/bin/bash
# tell this is a bash script
 
# Request 4 CPU, 500MB of memory and 10 minutes of walltime
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120GB
#SBATCH --time=12:10:00

#SBATCH -o X1_output_imputation.txt
 
# Specify where(partition) to run the job
#SBATCH --partition=GPU
 

# python3 main.py

python main.py --regime random

# MISSINGNESS_REGIMES = [
#     "random",
#     "short_gap",
#     "medium_gap",
#     "long_gap",
#     "event",
# ]

