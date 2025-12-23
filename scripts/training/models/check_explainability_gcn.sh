#!/bin/bash -l
#SBATCH --job-name=explain
#SBATCH --output=${PHARM_PROJECT_ROOT}/logs/explainability/%x_%j.out
#SBATCH --error=${PHARM_PROJECT_ROOT}/logs/explainability/%x_%j.err
#SBATCH --cpus-per-task=164
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --partition=<partition>
#SBATCH -A <A>>
#SBATCH --gres=gpu:1

set -eo pipefail

# ==============================
# -------- ARGUMENTS -----------
# ==============================

MODEL="gcn_vg"
INPUT="hard"
K_PAR="k5"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --input) INPUT="$2"; shift 2 ;;
        --k|--k_par|--kpar) K_PAR="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown parameter: $1"; exit 1 ;;
    esac
done

# ==============================
# -------- ENV CHECK ----------
# ==============================

if [[ -z "${PHARM_PROJECT_ROOT:-}" ]]; then
    echo "[ERROR] PHARM_PROJECT_ROOT is not set"
    exit 1
fi

echo "[INFO] PHARM_PROJECT_ROOT=${PHARM_PROJECT_ROOT}"
echo "[INFO] MODEL=${MODEL} | INPUT=${INPUT} | K_PAR=${K_PAR}"

# ==============================
# -------- CONDA SETUP ---------
# ==============================

set +u
source $HOME/miniconda/etc/profile.d/conda.sh # Path to your conda
conda activate savi-arm
set -u
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

# ==============================
# -------- RUN SCRIPT ----------
# ==============================

MODEL_UPPER=$(echo "${MODEL}" | tr '[:lower:]' '[:upper:]')

cd "${PHARM_PROJECT_ROOT}"

case "$MODEL_UPPER" in
    GCN)
        python ${PHARM_PROJECT_ROOT}/src/training/models/check_explainability_gcn.py \
            --model GCN \
            --dataset "$K_PAR" \
            --split "$INPUT"
        ;;
    GCN_VG)
        python ${PHARM_PROJECT_ROOT}/src/training/models/check_explainability_gcn.py \
            --model GCN_VG \
            --dataset "$K_PAR" \
            --split "$INPUT"
        ;;
    *)
        echo "[ERROR] Unknown MODEL=${MODEL_UPPER}"
        exit 1
        ;;
esac

echo "[INFO] === FINISHED ==="
