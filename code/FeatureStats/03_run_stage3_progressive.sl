#!/bin/bash
# tell this is a bash script
 
# Request 4 CPU, 500MB of memory and 10 minutes of walltime
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120GB
#SBATCH --time=12:10:00

#SBATCH -o X1_stage3_progressive_output_%j.txt
#SBATCH --chdir=/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/code/FeatureStats
 
# Specify where(partition) to run the job
#SBATCH --partition=GPU
 

python3 03regional_selected_feature_progressive_evaluation.py
