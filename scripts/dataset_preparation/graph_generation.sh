#!/bin/bash -l
#SBATCH --job-name=graph_generation
#SBATCH --output=logs/graph_generation_%j.out
#SBATCH --error=logs/graph_generation_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:30:00
# NOTE: set partition/account outside the repo if required
#   #SBATCH --partition=YOUR_PARTITION
#   #SBATCH -A YOUR_ACCOUNT

set -euo pipefail
mkdir -p logs

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
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
# sbatch --export=ALL,K=4,BASE_DIR=/path/to/project_root hpc/graph_generation.template.sh

K="${K:-}"
if [[ -z "${K}" ]]; then
  echo "[ERROR] Environment variable K is not set."
  echo "        Use: sbatch --export=ALL,K=4 hpc/graph_generation.template.sh"
  exit 1
fi

BASE_DIR="${BASE_DIR:-/path/to/project_root}"

INPUT_PARQUET="${INPUT_PARQUET:-${BASE_DIR}/data/k${K}/ks${K}.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/data/k${K}/graph_data}"

SCRIPT="${SCRIPT:-src/data_preparation/graph_generation.py}"

echo "[INFO] K=${K}"
echo "[INFO] BASE_DIR=${BASE_DIR}"
echo "[INFO] INPUT_PARQUET=${INPUT_PARQUET}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] SCRIPT=${SCRIPT}"
echo "[INFO] CPUs=${SLURM_CPUS_PER_TASK}"

mkdir -p "${OUTPUT_DIR}"

# ----------------------------
# Run
# ----------------------------
python -u "${SCRIPT}" \
  --input_parquet "${INPUT_PARQUET}" \
  --output_dir "${OUTPUT_DIR}"

echo "[INFO] End time: $(date)"
