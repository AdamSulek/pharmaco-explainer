#!/bin/bash -l
#SBATCH --job-name=finalize_dataset
#SBATCH --output=${PHARM_PROJECT_ROOT}/logs/processed/%x_%j.out
#SBATCH --error=${PHARM_PROJECT_ROOT}/logs/processed/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=<partition>
#SBATCH --account=<account>

set -eo pipefail

DATASET=${DATASET:-k3}
NUM_CHUNKS=${NUM_CHUNKS:-50}
FRAC=${FRAC:-0.05}

set +u
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate savi
set -u
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

mkdir -p ${PHARM_PROJECT_ROOT}/logs/processed

echo "=== RUNNING FINALIZE DATASET ==="
echo "Dataset:   ${DATASET}"
echo "NumChunks: ${NUM_CHUNKS}"
echo "Frac:      ${FRAC}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"

python -u ${PHARM_PROJECT_ROOT}/src/data_preprocessing/4_finalize_dataset.py \
    --dataset "${DATASET}" \
    --num-chunks "${NUM_CHUNKS}" \
    --frac "${FRAC}"
