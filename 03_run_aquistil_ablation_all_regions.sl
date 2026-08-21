#!/bin/bash
# Run the AQUISTIL ablation study across the requested region list.

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=200GB
#SBATCH --time=72:00:00
#SBATCH --partition=GPU
#SBATCH -o /mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/X1_output_aquistil_ablation_all_regions_%j.txt

set -euo pipefail

PROJECT_ROOT="/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL"
PROJECT_PYTHON="/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/.venv/bin/python"
METRICS_ROOT="${PROJECT_ROOT}/Outputs/Imputation_Results/Metrics"
ABLATION_METRICS_DIR="${METRICS_ROOT}/Metrics with Ablation"

cd "${PROJECT_ROOT}"

export AQUISTIL_EXPERIMENT_MODE="development_ablation"
export AQUISTIL_IMPUTATION_OUTPUT_DIR="${PROJECT_ROOT}/Outputs/Imputation_Results"
export AQUISTIL_ABLATION_METRICS_DIR="${ABLATION_METRICS_DIR}"
export PYTHONWARNINGS='ignore::UserWarning:sklearn.utils.parallel'
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-aquistil-ablation-all-regions-${SLURM_JOB_ID:-local}"

REGIONS=(
  "Sydney North-west"
  "Sydney South-west"
  "Upper Hunter"
  "Sydney East"
  "Northern Tablelands"
  "Southern Tablelands"
  "Lower Hunter"
  "Mid-North Coast"
  "Central Tablelands"
  "Illawarra"
  "Central Coast"
)

echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "Experiment mode: ${AQUISTIL_EXPERIMENT_MODE}"
echo "Metrics root: ${METRICS_ROOT}"
echo "Ablation metrics directory: ${ABLATION_METRICS_DIR}"
echo "Regions: ${REGIONS[*]}"
"${PROJECT_PYTHON}" --version
mkdir -p "${ABLATION_METRICS_DIR}"

# Main.py skips exact model/region/target tasks that are already complete in
# the combined metrics file. Missing regional wide-input CSVs are logged and
# skipped by main.py.
env -u PYTHONPATH -u LD_LIBRARY_PATH "${PROJECT_PYTHON}" code/main.py \
  --skip-api-refresh \
  --run-aquistil-ablations \
  --missingness-levels 0.05 0.10 0.20 0.30 \
  --regions "${REGIONS[@]}"

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  "${PROJECT_PYTHON}" code/tools/summarize_aquistil_ablation.py \
  --results-root "${PROJECT_ROOT}/Outputs/Imputation_Results" \
  --metrics-dir "${ABLATION_METRICS_DIR}" \
  --scope All

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  "${PROJECT_PYTHON}" code/tools/plot_aquistil_ablation_improvement.py \
  --metrics-dir "${ABLATION_METRICS_DIR}" \
  --scope Region_Macro

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  "${PROJECT_PYTHON}" code/tools/plot_aquistil_ablation_improvement.py \
  --metrics-dir "${ABLATION_METRICS_DIR}" \
  --scope Region_Micro

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  "${PROJECT_PYTHON}" code/tools/plot_aquistil_ablation_improvement.py \
  --metrics-dir "${ABLATION_METRICS_DIR}" \
  --scope Site
