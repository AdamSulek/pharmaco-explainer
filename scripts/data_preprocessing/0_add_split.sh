#!/bin/bash -l
#SBATCH --job-name=add_split
#SBATCH --output=${PHARM_PROJECT_ROOT}/logs/splits/%x_%j.out
#SBATCH --error=${PHARM_PROJECT_ROOT}/logs/splits/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=<partition>
#SBATCH --account=<account>

set -eo pipefail

DATASET=${DATASET:-k3}

set +u
source $HOME/miniconda/etc/profile.d/conda.sh #
conda activate savi
set -u
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

echo "=== RUNNING ADD_SPLIT ==="
echo "Dataset: ${DATASET}"
echo "CPUs:    ${SLURM_CPUS_PER_TASK}"

python -u ${PHARM_PROJECT_ROOT}/src/data_preprocessing/0_add_split.py \
    --dataset "${DATASET}" \
    --seed 42
