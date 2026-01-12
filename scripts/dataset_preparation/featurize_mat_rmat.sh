#!/bin/bash -l
#SBATCH --job-name=featurize_mat_rmat
#SBATCH --output=logs/featurize_mat_rmat_%A_%a.out
#SBATCH --error=logs/featurize_mat_rmat_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
# NOTE: set partition/account/gres outside the repo if needed
#   #SBATCH --partition=YOUR_PARTITION
#   #SBATCH -A YOUR_ACCOUNT
#   #SBATCH --gres=gpu:1

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
# export LD_PRELOAD="...:${LD_PRELOAD:-}"

# ----------------------------
# Config (override via --export)
# ----------------------------
# Example:
# sbatch --array=0-9 --export=ALL,BASE_DIR=/path/to/project_root,K=4,MODEL_TYPE=mat,MODE=no_pos \
#   hpc/featurize_mat_rmat.template.sh
#
# For positives with y=1:
# sbatch --array=0-9 --export=ALL,BASE_DIR=/path/to/project_root,K=4,MODEL_TYPE=rmat,MODE=with_pos,Y_VALUE=1 \
#   hpc/featurize_mat_rmat.template.sh

BASE_DIR="${BASE_DIR:-/path/to/project_root}"

K="${K:-4}"                       # dataset tag for naming/paths (optional)
MODEL_TYPE="${MODEL_TYPE:-mat}"    # mat | rmat
MODE="${MODE:-no_pos}"             # no_pos | with_pos
Y_VALUE="${Y_VALUE:-1}"            # used only when MODE=with_pos

# where each array task points to one folder (e.g. part index / shard index)
IDX="${SLURM_ARRAY_TASK_ID}"

SDF_DIR="${SDF_DIR:-${BASE_DIR}/sdf_files/k${K}_positive/${IDX}}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/pickle_dataloaders/${MODEL_TYPE}/k${K}_positive/${IDX}}"

SCRIPT="${SCRIPT:-src/data_preparation/featurize_mat_rmat.py}"
OUT_NAME="${OUT_NAME:-k${K}_${IDX}_${MODE}.p}"

echo "[INFO] BASE_DIR=${BASE_DIR}"
echo "[INFO] K=${K}"
echo "[INFO] IDX=${IDX}"
echo "[INFO] MODEL_TYPE=${MODEL_TYPE}"
echo "[INFO] MODE=${MODE}"
echo "[INFO] Y_VALUE=${Y_VALUE}"
echo "[INFO] SDF_DIR=${SDF_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] OUT_NAME=${OUT_NAME}"
echo "[INFO] SCRIPT=${SCRIPT}"
echo "[INFO] CPUs=${SLURM_CPUS_PER_TASK}"

mkdir -p "${OUT_DIR}"

# ----------------------------
# Run
# ----------------------------
if [[ "${MODE}" == "with_pos" ]]; then
  python -u "${SCRIPT}" \
    --model-type "${MODEL_TYPE}" \
    --mode "${MODE}" \
    --y-value "${Y_VALUE}" \
    --sdf-dir "${SDF_DIR}" \
    --output-dir "${OUT_DIR}" \
    --output-name "${OUT_NAME}"
else
  python -u "${SCRIPT}" \
    --model-type "${MODEL_TYPE}" \
    --mode "${MODE}" \
    --sdf-dir "${SDF_DIR}" \
    --output-dir "${OUT_DIR}" \
    --output-name "${OUT_NAME}"
fi

echo "[INFO] End time: $(date)"
