#!/bin/bash
# tell this is a bash script
 
# Request 4 CPU, 500MB of memory and 10 minutes of walltime
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120GB
#SBATCH --time=12:10:00
# One independent Stage 2 run for each supported target.
#SBATCH --array=0-6

#SBATCH -o X1_stage2_regional_selection_output_%A_%a.txt
#SBATCH --chdir=/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/code/FeatureStats
 
# Specify where(partition) to run the job
#SBATCH --partition=GPU
set -euo pipefail

TARGETS=("PM2.5" "PM10" "OZONE" "NO" "NO2" "NOX" "CO")
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#TARGETS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID ${ARRAY_INDEX} is outside 0-$(( ${#TARGETS[@]} - 1 ))" >&2
    exit 1
fi

TARGET="${TARGET:-${TARGETS[ARRAY_INDEX]}}"
TARGET_DIR="${TARGET//./_}"
INPUT_DIR="/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/API_Input/Inputs"
OUTPUT_DIR="/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Feature_Selection/by_target/${TARGET_DIR}/02Regional_RF_SHAP_Selection"

echo "Target: ${TARGET}"
echo "Regions: ALL_AVAILABLE"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# An empty --regions value means every region found in INPUT_DIR.
python3 02regional_rf_shap_feature_selection.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --target "${TARGET}" \
    --regions "" \
    --min-sites 1
