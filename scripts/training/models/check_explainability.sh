#!/bin/bash -l
#SBATCH --job-name=explain
#SBATCH --output=${PHARM_PROJECT_ROOT}/logs/explainability/%x_%j.out
#SBATCH --error=${PHARM_PROJECT_ROOT}/logs/explainability/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --time=16:00:00
#SBATCH --partition=general
#SBATCH --account=your-account-name

set -eo pipefail

MODEL="gcn"
INPUT="easy"
K_PAR="k3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --input) INPUT="$2"; shift 2;;
        --k|--k_par|--kpar) K_PAR="$2"; shift 2;;
        *) echo "[ERROR] Unknown parameter: $1"; exit 1;;
    esac
done

set +u
source $HOME/miniconda/etc/profile.d/conda.sh # Path to your conda
conda activate savi
set -u
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

echo "[INFO] MODEL=${MODEL} | INPUT=${INPUT} | K_PAR=${K_PAR}"

MODEL_UPPER=$(echo "${MODEL}" | tr '[:lower:]' '[:upper:]')

case "$MODEL_UPPER" in
    RF)
        python ${PHARM_PROJECT_ROOT}/src/training/models/check_explainability.py \
            --model RF --dataset "$K_PAR" --split "$INPUT"
        ;;
    XGB)
        python ${PHARM_PROJECT_ROOT}/src/training/models/check_explainability.py \
            --model XGB --dataset "$K_PAR" --split "$INPUT"
        ;;
    MLP)
        python ${PHARM_PROJECT_ROOT}/src/training/models/check_explainability.py \
            --model MLP --dataset "$K_PAR" --split "$INPUT"
        ;;
    MLP_VG)
        python ${PHARM_PROJECT_ROOT}/src/training/models/check_explainability.py \
            --model MLP_VG --dataset "$K_PAR" --split "$INPUT"
        ;;
    *)
        echo "[ERROR] Unknown MODEL=${MODEL_UPPER}"
        exit 1
        ;;
esac

echo "[INFO] === FINISHED ==="
