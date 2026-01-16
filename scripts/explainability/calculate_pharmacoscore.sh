#!/bin/bash -l
#SBATCH --job-name=calc_pharmacoscore
#SBATCH --output=logs/pharmacoscore/%x_%j.out
#SBATCH --error=logs/pharmacoscore/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=<partition>
#SBATCH --account=<account>

set -eo pipefail

MODEL="xgb"
INPUT="split_distant_set"
K_PAR="k3"
AGGREGATE="max"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --input) INPUT="$2"; shift 2;;
        --k|--k_par|--kpar) K_PAR="$2"; shift 2;;
        --aggregate) AGGREGATE="$2"; shift 2;;
        *) echo "[ERROR] Unknown parameter: $1"; exit 1;;
    esac
done

source $HOME/miniconda/etc/profile.d/conda.sh # Path to your conda
conda activate savi
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

echo "[INFO] Running PharmacoScore for: MODEL=${MODEL} | DATASET=${K_PAR} | SPLIT=${INPUT} | AGG=${AGGREGATE}"

python ${PHARM_PROJECT_ROOT}/src/explainability/calculate_pharmacoscore.py \
    --model "$MODEL" \
    --dataset "$K_PAR" \
    --split "$INPUT" \
    --aggregate "$AGGREGATE"

echo "[INFO] === FINISHED ==="