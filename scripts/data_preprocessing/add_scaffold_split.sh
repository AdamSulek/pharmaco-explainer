#!/bin/bash -l
#SBATCH --job-name=add_split
#SBATCH --output=logs/scaffold_split/%x_%j.out
#SBATCH --error=logs/scaffold_split/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=<partition>
#SBATCH --account=<account>

set -eo pipefail

DATASET=${DATASET:-k3}

set +u
source $HOME/miniconda/etc/profile.d/conda.sh # Path to your conda
conda activate savi
set -u
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

echo "=== RUNNING ADD_SPLIT ==="
echo "Dataset: ${DATASET}"
echo "CPUs:    ${SLURM_CPUS_PER_TASK}"

python -u ${PHARM_PROJECT_ROOT}/src/data_preprocessing/add_scaffold_split.py \
    --dataset "${DATASET}" \
    --seed 42
