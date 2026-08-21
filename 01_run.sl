#!/bin/bash
# tell this is a bash script
 
# Frozen held-out AQUISTIL publication validation.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=200GB
#SBATCH --time=32:10:00

#SBATCH -o /mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/X1_output_frozen_validation_%j.txt
 
# Specify where(partition) to run the job
#SBATCH --partition=GPU
 
set -euo pipefail


PROJECT_PYTHON="/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/.venv/bin/python"
PROJECT_ROOT="/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL"
cd "${PROJECT_ROOT}"

export PYTHONWARNINGS='ignore::UserWarning:sklearn.utils.parallel'
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-aquistil-${SLURM_JOB_ID:-local}"
export AQUISTIL_EXPERIMENT_MODE="frozen_validation"
export AQUISTIL_IMPUTATION_OUTPUT_DIR="${PROJECT_ROOT}/Outputs/Final_Frozen_2026_08_17"
unset AQUISTIL_ABLATION_METRICS_DIR

echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "Frozen tag: aquistil-frozen-heldout-2026-08-17-r4"
echo "Python: ${PROJECT_PYTHON}"
"${PROJECT_PYTHON}" --version

# All experimental settings are frozen and validated inside config_spatial.py.
env -u PYTHONPATH -u LD_LIBRARY_PATH "${PROJECT_PYTHON}" code/main.py \
  --skip-api-refresh

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  "${PROJECT_PYTHON}" code/tools/analyze_frozen_validation.py \
  --results-root Outputs/Final_Frozen_2026_08_17
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  "${PROJECT_PYTHON}" code/tools/plot_frozen_validation.py \
  --results-root Outputs/Final_Frozen_2026_08_17
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  "${PROJECT_PYTHON}" code/tools/plot_aquistil_architecture.py \
  --output-dir Outputs/Final_Frozen_2026_08_17/Paper_Figures
