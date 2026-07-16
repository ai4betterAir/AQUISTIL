#!/bin/bash
#SBATCH --job-name=feature_selection_pipeline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --partition=GPU
# One independent pipeline per target.  Keeping targets in separate array tasks
# avoids output-file collisions and lets the scheduler run them concurrently.
#SBATCH --array=0-6
#SBATCH --output=X1_feature_selection_pipeline_%A_%a.txt
#SBATCH --chdir=/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/code/FeatureStats

set -euo pipefail

# By default the array covers every target and an empty REGIONS value means all
# available regions.  TARGET and REGIONS can still be overridden for testing.
# Example (one target and region, using only array task 0):
#   sbatch --array=0 --export=ALL,TARGET=PM2.5,REGIONS='Lower Hunter' 00_run_feature_selection_pipeline.sl
export FEATURE_SELECTION_RUN_MODE="${FEATURE_SELECTION_RUN_MODE:-default}"
TARGETS=("PM2.5" "PM10" "OZONE" "NO" "NO2" "NOX" "CO")
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#TARGETS[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID ${ARRAY_INDEX} is outside 0-$(( ${#TARGETS[@]} - 1 ))" >&2
  exit 1
fi

TARGET="${TARGET:-${TARGETS[ARRAY_INDEX]}}"
REGIONS="${REGIONS:-}"
TARGET_DIR="${TARGET//./_}"
BASE_OUTPUT_ROOT="/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Feature_Selection"
OUTPUT_ROOT="${BASE_OUTPUT_ROOT}/by_target/${TARGET_DIR}"

if [[ "${FEATURE_SELECTION_RUN_MODE}" == "event" ]]; then
  EFFECTIVE_OUTPUT_ROOT="${OUTPUT_ROOT}/feature_selection_event"
else
  EFFECTIVE_OUTPUT_ROOT="${OUTPUT_ROOT}"
fi

echo "================================================================================"
echo "FEATURE SELECTION PIPELINE STARTED"
echo "Run mode: ${FEATURE_SELECTION_RUN_MODE}"
echo "Target: ${TARGET}"
echo "Regions: ${REGIONS}"
echo "Output root: ${EFFECTIVE_OUTPUT_ROOT}"
echo "Stages: 0 Random Forest -> 1 plots -> 2 RF-SHAP -> 3 progressive RF -> 4 basics"
echo "================================================================================"

# main_stats.py uses subprocess(..., check=True), so it starts each numbered
# stage only after the preceding stage exits successfully. Stages 1 and 3 also
# require the expected Stage 0 and Stage 2 CSVs, respectively.
python3 main_stats.py \
  --target "${TARGET}" \
  --regions "${REGIONS}" \
  --output-root "${OUTPUT_ROOT}"

SETTINGS_FILE="${EFFECTIVE_OUTPUT_ROOT}/main_stats_run_settings.json"
FINAL_FEATURES="${EFFECTIVE_OUTPUT_ROOT}/02Regional_RF_SHAP_Selection/summary_outputs/FINAL_selected_feature_combination_by_region.csv"
PROGRESSIVE_RESULT="${EFFECTIVE_OUTPUT_ROOT}/03Regional_Selected_Feature_Progressive_Evaluation/summary_outputs/regional_progressive_best_configuration_by_region.csv"

for required_file in "${SETTINGS_FILE}" "${FINAL_FEATURES}" "${PROGRESSIVE_RESULT}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "ERROR: required pipeline output is missing or empty: ${required_file}" >&2
    exit 1
  fi
  echo "Confirmed output: ${required_file}"
done

echo "================================================================================"
echo "FEATURE SELECTION PIPELINE FINISHED SUCCESSFULLY"
echo "================================================================================"
