#!/bin/bash -l
#SBATCH --job-name=generate_fps
#SBATCH --output=logs/generate_ecfp/%x_%j.out
#SBATCH --error=logs/generate_ecfp/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=<partition>
#SBATCH --account=<account>

set -eo pipefail

DATASET=${DATASET:-k3}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set +u
source $HOME/miniconda/etc/profile.d/conda.sh # Path to your conda
conda activate savi
set -u
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

BASE_DIR="../../data/${DATASET}"

echo "Dataset:     ${DATASET}"
echo "CPUs:        ${SLURM_CPUS_PER_TASK}"

python -u ${PHARM_PROJECT_ROOT}/src/data_preprocessing/generate_ecfp.py \
    --dataset "${DATASET}" \
    --fingerprint "X_ecfp_2" \
    --n-proc "${SLURM_CPUS_PER_TASK}"
