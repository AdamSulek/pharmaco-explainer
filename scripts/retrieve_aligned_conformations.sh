#!/bin/bash -l
#SBATCH --job-name=qc_retrieve_positive_conformers
#SBATCH --output=logs/qc_retrieve_positive_conformers_%A_%a.out
#SBATCH --error=logs/qc_retrieve_positive_conformers_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-52
# NOTE: Set your cluster-specific partition/account outside the repo:
#   #SBATCH --partition=YOUR_PARTITION
#   #SBATCH -A YOUR_ACCOUNT

set -euo pipefail
mkdir -p logs

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Node: $(hostname)"
echo "[INFO] Start time: $(date)"

# (Optional) conda activation – keep generic
# source /path/to/miniconda/etc/profile.d/conda.sh
# conda activate your_env

# ----------------------------
# Config (override via --export)
# ----------------------------
# Example:
# sbatch --export=ALL,K=k4,BASE_DIR=/path/to/project_root hpc/qc_retrieve_positive_conformers.template.sh

K="${K:-k4}"                 # k3 | k4 | k5
PART_IDX="${SLURM_ARRAY_TASK_ID}"

# Repo-safe roots (edit to your environment)
BASE_DIR="${BASE_DIR:-/path/to/project_root}"
LABELS_ROOT="${LABELS_ROOT:-${BASE_DIR}/labels_out/${K}}"          # where labels exist
SDF_ROOT="${SDF_ROOT:-${BASE_DIR}/sdf_files}"                      # where source SDF parts exist
OUT_DIR="${OUT_DIR:-${BASE_DIR}/qc/positive_sdfs/${K}}"            # output for retrieved positives

# Python entrypoint (rename if needed)
SCRIPT="${SCRIPT:-data_preparation/retrieve_aligned_conformations.py}"
# If you keep the old name:
# SCRIPT="${SCRIPT:-qc/retrive_sdf_conformer.py}"

echo "[INFO] K=${K}"
echo "[INFO] PART_IDX=${PART_IDX}"
echo "[INFO] BASE_DIR=${BASE_DIR}"
echo "[INFO] LABELS_ROOT=${LABELS_ROOT}"
echo "[INFO] SDF_ROOT=${SDF_ROOT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] SCRIPT=${SCRIPT}"

# Run
python -u "${SCRIPT}" \
  --k "${K}" \
  --part-idx "${PART_IDX}" \
  --labels-root "${LABELS_ROOT}" \
  --sdf-root "${SDF_ROOT}" \
  --out-dir "${OUT_DIR}"

echo "[INFO] End time: $(date)"
