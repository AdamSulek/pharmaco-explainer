#!/bin/bash -l
#SBATCH --job-name=check_explainability_ecfp
#SBATCH --output=logs/explainability/%x_%j.out
#SBATCH --error=logs/explainability/%x_%j.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --partition=general
#SBATCH --account=your-account-name

set -eo pipefail

MODEL="xgb"
INPUT="split_distant_set" #all split_close_set split_distant_set
K_PAR="k3"
AGGREGATE="max" #mean, sum

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --input) INPUT="$2"; shift 2;;
        --k|--k_par|--kpar) K_PAR="$2"; shift 2;;
        *) echo "[ERROR] Unknown parameter: $1"; exit 1;;
    esac
done

set +u
source /net/storage/pr3/plgrid/plggsanodrugs/miniconda/etc/profile.d/conda.sh
conda activate savi
set -u
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

echo "[INFO] MODEL=${MODEL} | INPUT=${INPUT} | K_PAR=${K_PAR}"

MODEL_UPPER=$(echo "${MODEL}" | tr '[:lower:]' '[:upper:]')

case "$MODEL_UPPER" in
    RF)
        python ${PHARM_PROJECT_ROOT}/src/explainability/check_explainability_ecfp2.py \
            --model rf --dataset "$K_PAR" --split "$INPUT" --aggregate "$AGGREGATE"
        ;;
    XGB)
        python ${PHARM_PROJECT_ROOT}/src/explainability/check_explainability_ecfp2.py \
            --model xgb --dataset "$K_PAR" --split "$INPUT" --aggregate "$AGGREGATE"
        ;;
    MLP)
        python ${PHARM_PROJECT_ROOT}/src/explainability/check_explainability_ecfp2.py \
            --model mlp --dataset "$K_PAR" --split "$INPUT" --aggregate "$AGGREGATE"
        ;;
    MLP_VG)
        python ${PHARM_PROJECT_ROOT}/src/explainability/check_explainability_ecfp2.py \
            --model mlp_vg --dataset "$K_PAR" --split "$INPUT" --aggregate "$AGGREGATE"
        ;;
    *)
        echo "[ERROR] Unknown MODEL=${MODEL_UPPER}"
        exit 1
        ;;
esac

echo "[INFO] === FINISHED ==="
