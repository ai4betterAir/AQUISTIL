#!/bin/bash
# Complete and validate the four-region AQUISTIL development ablation study.

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=200GB
#SBATCH --time=32:10:00
#SBATCH --partition=GPU
#SBATCH -o /mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/X1_output_aquistil_ablation_%j.txt

set -euo pipefail

PROJECT_ROOT="/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL"
PROJECT_PYTHON="/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/.venv/bin/python"
METRICS_DIR="${PROJECT_ROOT}/Outputs/Imputation_Results/Metrics copy"

cd "${PROJECT_ROOT}"

export AQUISTIL_EXPERIMENT_MODE="development_ablation"
export AQUISTIL_IMPUTATION_OUTPUT_DIR="${PROJECT_ROOT}/Outputs/Imputation_Results"
export AQUISTIL_ABLATION_METRICS_DIR="${METRICS_DIR}"
export PYTHONWARNINGS='ignore::UserWarning:sklearn.utils.parallel'
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-aquistil-ablation-${SLURM_JOB_ID:-local}"

echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "Experiment mode: ${AQUISTIL_EXPERIMENT_MODE}"
echo "Combined metrics: ${METRICS_DIR}/aquistil_ablation_metrics.csv"
"${PROJECT_PYTHON}" --version

# config_spatial.py defines the complete protocol. main.py skips any exact
# model/region/target task already present in the combined assessment CSV.
env -u PYTHONPATH -u LD_LIBRARY_PATH "${PROJECT_PYTHON}" code/main.py \
  --skip-api-refresh

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  "${PROJECT_PYTHON}" code/tools/audit_aquistil_ablation_completeness.py \
  --metrics "${METRICS_DIR}/aquistil_ablation_metrics.csv"

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  "${PROJECT_PYTHON}" code/tools/summarize_aquistil_ablation.py \
  --results-root "${PROJECT_ROOT}/Outputs/Imputation_Results" \
  --metrics-dir "${METRICS_DIR}" \
  --scope All
