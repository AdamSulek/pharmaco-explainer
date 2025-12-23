#!/bin/bash -l
#SBATCH --job-name=train_model
#SBATCH --output=${PHARM_PROJECT_ROOT}/logs/train/%x_%j.out
#SBATCH --error=${PHARM_PROJECT_ROOT}/logs/train/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --partition=<partition>
#SBATCH --account=<account>

set -euo pipefail

set +u
source $HOME/miniconda/etc/profile.d/conda.sh # Path to your conda
conda activate savi
set -u
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

# --------- PARAMS ---------
MODEL="${1:-xgb}"
SPLIT="${2:-easy}"
DATASET="${3:-k3}"

INPUT_DIR="${INPUT_DIR:-${PHARM_PROJECT_ROOT}/data/${DATASET}/processed}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PHARM_PROJECT_ROOT}/results/checkpoints/${DATASET}}"

mkdir -p "$CHECKPOINT_DIR"

echo "[INFO] Training MODEL=$MODEL, SPLIT=$SPLIT, DATASET=$DATASET"
echo "[INFO] Checkpoint dir: $CHECKPOINT_DIR"

# wybór skryptu w zależności od modelu
case "$MODEL" in
    xgb)
        python ${PHARM_PROJECT_ROOT}/src/training/models/train_xgb.py \
            --dataset "$DATASET" \
            --split "$SPLIT"
        ;;
    rf)
        python ${PHARM_PROJECT_ROOT}/src/training/models/train_rf.py \
            --dataset "$DATASET" \
            --split "$SPLIT"
        ;;
    mlp)
        python ${PHARM_PROJECT_ROOT}/src/training/models/train_mlp.py \
            --dataset "$DATASET" \
            --split "$SPLIT"
        ;;
    *)
        echo "[ERROR] Unknown MODEL=$MODEL"
        exit 1
        ;;
esac

echo "[INFO] Training complete"
