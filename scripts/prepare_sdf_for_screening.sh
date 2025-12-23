#!/bin/bash -l
#SBATCH --job-name=prepare_sdf_for_screening
#SBATCH --output=logs/prepare_sdf_for_screening_%A_%a.out
#SBATCH --error=logs/prepare_sdf_for_screening_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
# NOTE: Set your cluster-specific partition/account outside the repo:
#   #SBATCH --partition=YOUR_PARTITION
#   #SBATCH -A YOUR_ACCOUNT

set -euo pipefail
mkdir -p logs

# Avoid oversubscription in numeric libs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# (Optional) conda activation – keep generic
# source /path/to/miniconda/etc/profile.d/conda.sh
# conda activate your_env

PART=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")

# ---- repo-safe paths (edit to your environment) ----
DATA_ROOT="${DATA_ROOT:-/path/to/processed_parts}"          # contains part_XXX.parquet
OUT_ROOT="${OUT_ROOT:-/path/to/sdf_files}"                 # output root
IN_PARQUET="${DATA_ROOT}/part_${PART}.parquet"
OUT_DIR="${OUT_ROOT}/part_${PART}"
# ----------------------------------------------------

echo "=== RUNNING part_${PART} ==="
echo "Input:  ${IN_PARQUET}"
echo "Output: ${OUT_DIR}"
echo "Host:   $(hostname)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"

python -u data_preparation/prepare_sdf_for_screening.py \
  --in-parquet "${IN_PARQUET}" \
  --part "${PART}" \
  --out-dir "${OUT_DIR}" \
  --chunk-size-ids 1000 \
  --target-confs 50 \
  --progress-step 100 \
  --n-proc "${SLURM_CPUS_PER_TASK}" \
  --batch-size 64
