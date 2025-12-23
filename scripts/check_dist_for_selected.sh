#!/bin/bash -l
#SBATCH --job-name=qc_check_dist
#SBATCH --output=logs/qc_check_dist_%A_%a.out
#SBATCH --error=logs/qc_check_dist_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:10:00
#SBATCH --array=0-53
# NOTE: set partition/account outside the repo if needed
#   #SBATCH --partition=YOUR_PARTITION
#   #SBATCH -A YOUR_ACCOUNT

set -euo pipefail
mkdir -p logs

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Node: $(hostname)"
echo "[INFO] Start time: $(date)"

# ----------------------------
# Environment (optional)
# ----------------------------
# source /path/to/miniconda/etc/profile.d/conda.sh
# conda activate your_env

# ----------------------------
# Config (override via --export)
# ----------------------------
# Example:
# sbatch --array=0-53 --export=ALL,K=k4,BASE_DIR=/path/to/project_root,HYPOTHESIS=hypothesis/pharma_4.json hpc/qc_check_dist_for_selected.template.sh

K="${K:-k4}"   # k3 | k4 | k5
PART_IDX="${SLURM_ARRAY_TASK_ID}"

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
HYPOTHESIS="${HYPOTHESIS:-hypothesis/pharma_4_elements.json}"

SCRIPT="${SCRIPT:-data_preparation/qc/check_dist_for_selected.py}"

echo "[INFO] K=${K}"
echo "[INFO] PART_IDX=${PART_IDX}"
echo "[INFO] BASE_DIR=${BASE_DIR}"
echo "[INFO] HYPOTHESIS=${HYPOTHESIS}"
echo "[INFO] SCRIPT=${SCRIPT}"

# ----------------------------
# Run
# ----------------------------
python -u "${SCRIPT}" \
  --k "${K}" \
  --part-idx "${PART_IDX}" \
  --hypo-json "${HYPOTHESIS}"

echo "[INFO] End time: $(date)"
