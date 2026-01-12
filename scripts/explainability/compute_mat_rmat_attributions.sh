#!/bin/bash -l
#SBATCH --job-name=cam_run
#SBATCH --output=logs_cam/cam_run_%j.out
#SBATCH --error=logs_cam/cam_run_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=364G
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH -A plgsonata19-gpu-gh200
#SBATCH --gres=gpu:1

set -euo pipefail

LOG() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
DIE() { echo "[ERROR] $*" >&2; exit 1; }

REPO_ROOT="${REPO_ROOT:-/net/storage/pr3/plgrid/plggsanodrugs/cr/huggingmolecules}"
CONDA_SH="${CONDA_SH:-/net/storage/pr3/plgrid/plggsanodrugs/miniconda-arm/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-savi-arm-hm}"

METHOD="${METHOD:-vg}"
MODEL="${MODEL:-mat}"
CHECKPOINT="${CHECKPOINT:-}"
TEST_PICKLE="${TEST_PICKLE:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
POSITIVE_PICKLE_POS_PATH="${POSITIVE_PICKLE_POS_PATH:-}"
ONLY_POSITIVE="${ONLY_POSITIVE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_REL="${SCRIPT_REL:-experiments/src/training/check_mat_or_rmat_explainability.py}"
LOG_EVERY="${LOG_EVERY:-200}"

[[ -n "${CHECKPOINT}" ]] || DIE "CHECKPOINT is required"
[[ -n "${TEST_PICKLE}" ]] || DIE "TEST_PICKLE is required"
[[ -n "${OUTPUT_FILE}" ]] || DIE "OUTPUT_FILE is required"

LOG "Changing to repo root: ${REPO_ROOT}"
cd "${REPO_ROOT}"

[[ -f "${CONDA_SH}" ]] || DIE "conda.sh not found: ${CONDA_SH}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6:${LD_PRELOAD:-}"
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"

LOG "METHOD=${METHOD}"
LOG "MODEL=${MODEL}"
LOG "CHECKPOINT=${CHECKPOINT}"
LOG "TEST_PICKLE=${TEST_PICKLE}"
LOG "OUTPUT_FILE=${OUTPUT_FILE}"
LOG "POSITIVE_PICKLE_POS_PATH=${POSITIVE_PICKLE_POS_PATH}"
LOG "ONLY_POSITIVE=${ONLY_POSITIVE}"
LOG "PYTHON_BIN=${PYTHON_BIN}"
LOG "SCRIPT_REL=${SCRIPT_REL}"
LOG "LOG_EVERY=${LOG_EVERY}"

[[ -f "${SCRIPT_REL}" ]] || DIE "Python script not found (relative to repo root): ${SCRIPT_REL}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_REL}"
  --method "${METHOD}"
  --model "${MODEL}"
  --checkpoint "${CHECKPOINT}"
  --test-pickle "${TEST_PICKLE}"
  --output-file "${OUTPUT_FILE}"
  --log-every "${LOG_EVERY}"
)

if [[ -n "${POSITIVE_PICKLE_POS_PATH}" ]]; then
  CMD+=( --positive-pickle-pos-path "${POSITIVE_PICKLE_POS_PATH}" )
fi

if [[ "${ONLY_POSITIVE}" == "1" ]]; then
  CMD+=( --only-positive )
fi

LOG "Running: ${CMD[*]}"
"${CMD[@]}"
LOG "Done"
