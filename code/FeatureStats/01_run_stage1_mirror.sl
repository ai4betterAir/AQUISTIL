#!/bin/bash
# tell this is a bash script
 
# Request 4 CPU, 500MB of memory and 10 minutes of walltime
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120GB
#SBATCH --time=12:10:00

#SBATCH -o X1_stage1_mirror_output_%j.txt
#SBATCH --chdir=/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/code/FeatureStats
 
# Specify where(partition) to run the job
#SBATCH --partition=GPU
 
RESULTS_CSV="/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Feature_Selection/00RandomForest_Best_Individual_Feature_Selection/individual_results/all_random_forest_individual_feature_results.csv"
OUTPUT_ROOT="/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Feature_Selection"

export FEATURE_SELECTION_OUTPUT_ROOT="${OUTPUT_ROOT}"
export FS_OUTPUT_ROOT="${OUTPUT_ROOT}/00RandomForest_Best_Individual_Feature_Selection"
export RESULTS_CSV

if [ ! -f "${RESULTS_CSV}" ]; then
  python3 00run_random_forest_best_individual_feature.py
fi

python3 01plot_mirror_region_vs_site_edited.py
