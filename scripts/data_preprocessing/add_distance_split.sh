#!/bin/bash -l
#SBATCH --job-name=finalize_dataset
#SBATCH --output=logs/distance_split/%x_%j.out
#SBATCH --error=logs/distance_split/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=<partition>
#SBATCH --account=<account>

set -eo pipefail

DATASET=${DATASET:-k3}
FRAC=${FRAC:-0.05}

set +u
source $HOME/miniconda/etc/profile.d/conda.sh # Path to your conda
conda activate savi
set -u
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

echo "=== RUNNING FINALIZE DATASET ==="
echo "Dataset:   ${DATASET}"
echo "Frac:      ${FRAC}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"

python -u ${PHARM_PROJECT_ROOT}/src/data_preprocessing/add_distance_split.py \
    --dataset "${DATASET}" \
    --frac "${FRAC}"
