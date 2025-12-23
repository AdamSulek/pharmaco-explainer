#!/bin/bash -l
#SBATCH --job-name=gcn_run
#SBATCH --output=${PHARM_PROJECT_ROOT}/logs/gcn_run/%x_%j.out
#SBATCH --error=${PHARM_PROJECT_ROOT}/logs/gcn_run/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=70G
#SBATCH --time=11:00:00
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --gres=gpu:1

set -euo pipefail

set +u
source $HOME/miniconda/etc/profile.d/conda.sh # Path to your conda
conda activate savi-arm
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6:${LD_PRELOAD:-}"
set -u

PYTHON="${PYTHON:-python}"

: "${K:?You must export K (e.g. 3,4,5)}"

INPUT_DIR="${INPUT_DIR:-${PHARM_PROJECT_ROOT}/data/k${K}/graph_data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PHARM_PROJECT_ROOT}/results/checkpoints_gcn/k${K}}"
RESULT_DIR="${RESULT_DIR:-${PHARM_PROJECT_ROOT}/results/train/k${K}}"
SPLIT_FILE_PATH="${SPLIT_FILE_PATH:-${PHARM_PROJECT_ROOT}/data/k${K}/processed/final_dataset.parquet}"
SPLIT_TYPE="${SPLIT_TYPE:-split}"

mkdir -p "$INPUT_DIR" "$CHECKPOINT_DIR" "$RESULT_DIR"

echo "[INFO] K=${K}"
echo "[INFO] INPUT_DIR=${INPUT_DIR}"
echo "[INFO] CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "[INFO] RESULT_DIR=${RESULT_DIR}"
echo "[INFO] SPLIT_FILE_PATH=${SPLIT_FILE_PATH}"
echo "[INFO] SPLIT_TYPE=${SPLIT_TYPE}"

CMD=( "${PYTHON}" ${PHARM_PROJECT_ROOT}/src/training/models/train_gcn.py
  --k "${K}"
  --input_dir "${INPUT_DIR}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --result_dir "${RESULT_DIR}"
  --split_file_path "${SPLIT_FILE_PATH}"
  --split_type "${SPLIT_TYPE}" )

echo "[INFO] Running:"
echo "${CMD[*]}"

"${CMD[@]}"
