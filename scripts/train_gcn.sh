#!/bin/bash -l
#SBATCH --job-name=train_gcn
#SBATCH --output=logs/train_gcn_%j.out
#SBATCH --error=logs/train_gcn_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=16:10:00
# NOTE: set partition/account/gres outside the repo if needed
#   #SBATCH --partition=YOUR_PARTITION
#   #SBATCH -A YOUR_ACCOUNT
#   #SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Node: $(hostname)"
echo "[INFO] Start time: $(date)"
echo "[INFO] CPUs: ${SLURM_CPUS_PER_TASK}"

# ----------------------------
# Environment (optional)
# ----------------------------
# source /path/to/miniconda/etc/profile.d/conda.sh
# conda activate your_env
# export LD_PRELOAD="...:${LD_PRELOAD:-}"

PYTHON="${PYTHON:-python}"

# ----------------------------
# Config (override via --export)
# ----------------------------
# Required:
#   K=3|4|5
#
# Example:
# sbatch --export=ALL,K=4,BASE_DIR=/path/to/project_root,SPLIT_TYPE=split_distant_set hpc/train_gcn.template.sh

: "${K:?You must export K (e.g. 3,4,5)}"

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
SPLIT_TYPE="${SPLIT_TYPE:-split}"

INPUT_DIR="${INPUT_DIR:-${BASE_DIR}/data/k${K}/graph_data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BASE_DIR}/runs/checkpoints_gcn/k${K}/${SPLIT_TYPE}}"
RESULT_DIR="${RESULT_DIR:-${BASE_DIR}/runs/results_gcn/k${K}/${SPLIT_TYPE}}"
SPLIT_FILE_PATH="${SPLIT_FILE_PATH:-${BASE_DIR}/data/k${K}/k${K}_split.parquet}"
SPLIT_TYPE_ARG="${SPLIT_TYPE_ARG:-${SPLIT_TYPE}}"

SCRIPT="${SCRIPT:-src/training/train_gcn.py}"

echo "[INFO] K=${K}"
echo "[INFO] BASE_DIR=${BASE_DIR}"
echo "[INFO] INPUT_DIR=${INPUT_DIR}"
echo "[INFO] CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "[INFO] RESULT_DIR=${RESULT_DIR}"
echo "[INFO] SPLIT_FILE_PATH=${SPLIT_FILE_PATH}"
echo "[INFO] SPLIT_TYPE=${SPLIT_TYPE_ARG}"
echo "[INFO] SCRIPT=${SCRIPT}"

mkdir -p "${CHECKPOINT_DIR}" "${RESULT_DIR}"

CMD=( "${PYTHON}" -u "${SCRIPT}"
  --k "${K}"
  --input_dir "${INPUT_DIR}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --result_dir "${RESULT_DIR}"
  --split_file_path "${SPLIT_FILE_PATH}"
  --split_type "${SPLIT_TYPE_ARG}" )

echo "[INFO] Running:"
echo "${CMD[*]}"

"${CMD[@]}"

echo "[INFO] End time: $(date)"
