#!/bin/bash -l
#SBATCH --job-name=tanimoto
#SBATCH --output=${PHARM_PROJECT_ROOT}/logs/tanimoto/%x_%A_%a.out
#SBATCH --error=${PHARM_PROJECT_ROOT}/logs/tanimoto/%x_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --array=0-49

set -eo pipefail

DATASET=${DATASET:-k4}
PART=${SLURM_ARRAY_TASK_ID}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set +u
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate savi
set -u
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

mkdir -p ${PHARM_PROJECT_ROOT}/logs/tanimoto

echo "=== RUNNING Tanimoto PART ${PART} ==="
echo "Dataset: ${DATASET}"
echo "CPUs:    ${SLURM_CPUS_PER_TASK}"

python -u ${PHARM_PROJECT_ROOT}/src/data_preprocessing/3_add_tanimoto.py \
    --dataset "${DATASET}" \
    --part "${PART}" \
    --batch-size 50000
