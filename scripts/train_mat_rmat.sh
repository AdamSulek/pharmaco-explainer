#!/bin/bash -l
#SBATCH --job-name=train_mat_rmat
#SBATCH --output=logs/train_mat_rmat_%j.out
#SBATCH --error=logs/train_mat_rmat_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=370G
#SBATCH --time=02:10:00
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
# Examples:
# sbatch --export=ALL,BASE_DIR=/path/to/project_root,MODEL=mat,K=k4,DIFFICULTY=normal \
#   hpc/train_mat_or_rmat.template.sh
#
# sbatch --export=ALL,BASE_DIR=/path/to/project_root,MODEL=rmat,K=k4,DIFFICULTY=hard,SPLIT_FILE=/path/to/split.parquet \
#   hpc/train_mat_or_rmat.template.sh
#
# sbatch --export=ALL,BASE_DIR=/path/to/project_root,MODEL=mat,K=k4,DIFFICULTY=easy,SPLIT_FILE=/path/to/split.parquet,POSITIVE_PICKLE_POS_PATH=/path/to/positives.p \
#   hpc/train_mat_or_rmat.template.sh

BASE_DIR="${BASE_DIR:-/path/to/project_root}"

MODEL="${MODEL:-mat}"                  # mat | rmat
K="${K:-k4}"                           # k3 | k4 | k5
DIFFICULTY="${DIFFICULTY:-normal}"     # normal | easy | hard | none
SUBSET="${SUBSET:-normal}"             # forwarded to Python (kept for compatibility)
SPLIT_FILE="${SPLIT_FILE:-}"           # optional (mainly for easy/hard)
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}}"  # root expected by Python
POSITIVE_PICKLE_POS_PATH="${POSITIVE_PICKLE_POS_PATH:-}"  # optional

SCRIPT="${SCRIPT:-src/training/train_mat_or_rmat.py}"

echo "[INFO] BASE_DIR=${BASE_DIR}"
echo "[INFO] MODEL=${MODEL}"
echo "[INFO] K=${K}"
echo "[INFO] DIFFICULTY=${DIFFICULTY}"
echo "[INFO] SUBSET=${SUBSET}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] SPLIT_FILE=${SPLIT_FILE}"
echo "[INFO] POSITIVE_PICKLE_POS_PATH=${POSITIVE_PICKLE_POS_PATH}"
echo "[INFO] SCRIPT=${SCRIPT}"

# ----------------------------
# Output layout (repo-safe)
# ----------------------------
BASE_TAG="${MODEL}_${K}_${DIFFICULTY}"
if [[ -n "${POSITIVE_PICKLE_POS_PATH}" ]]; then
  TAG="${BASE_TAG}_with_pos"
else
  TAG="${BASE_TAG}_normal"
fi

CKPT_DIR="${BASE_DIR}/runs/checkpoints/${TAG}"
RES_DIR="${BASE_DIR}/runs/results/${TAG}"
mkdir -p "${CKPT_DIR}" "${RES_DIR}"

CKPT_PATH="${CKPT_DIR}/best_model.pth"
RESULTS_PKL="${RES_DIR}/metrics.pkl"

echo "[INFO] TAG=${TAG}"
echo "[INFO] CKPT_PATH=${CKPT_PATH}"
echo "[INFO] RESULTS_PKL=${RESULTS_PKL}"

# ----------------------------
# Extra args (optional)
# ----------------------------
EXTRA_ARGS=()

if [[ -n "${SPLIT_FILE}" ]]; then
  EXTRA_ARGS+=( --data-split-file "${SPLIT_FILE}" )
fi

if [[ -n "${POSITIVE_PICKLE_POS_PATH}" ]]; then
  EXTRA_ARGS+=( --positive-pickle-pos-path "${POSITIVE_PICKLE_POS_PATH}" )
fi

# ----------------------------
# Run
# ----------------------------
CMD=( "${PYTHON}" -u "${SCRIPT}"
  --model "${MODEL}"
  --k "${K}"
  --data-root "${DATA_ROOT}"
  --subset "${SUBSET}"
  --difficulty "${DIFFICULTY}"
  --selection-metric val_auc
  --checkpoint_path "${CKPT_PATH}"
  --results-pickle "${RESULTS_PKL}"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=( "${EXTRA_ARGS[@]}" )
fi

echo "[INFO] Running command:"
echo "${CMD[*]}"

"${CMD[@]}"

echo "[INFO] End time: $(date)"
