#!/bin/bash -l
#SBATCH --job-name=generate_fps
#SBATCH --output=${PHARM_PROJECT_ROOT}/logs/fps/k4/%a_generate_fps.out
#SBATCH --error=${PHARM_PROJECT_ROOT}/logs/fps/k4/%a_generate_fps.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=plgrid
#SBATCH -A plgsonata19-cpu
#SBATCH --array=0-51

set -eo pipefail

DATASET=${DATASET:-k3}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set +u
source /net/storage/pr3/plgrid/plggsanodrugs/miniconda/etc/profile.d/conda.sh
conda activate savi
set -u
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

PART=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
BASE_DIR="../../data/${DATASET}"

if [ ${SLURM_ARRAY_TASK_ID} -le 49 ]; then
    INPUT="${BASE_DIR}/splits/${DATASET}_negative_train_chunk_${PART}.parquet"
    OUT_FILE="${BASE_DIR}/fgp/${DATASET}_negative_train_chunk_${PART}.parquet"
elif [ ${SLURM_ARRAY_TASK_ID} -eq 50 ]; then
    INPUT="${BASE_DIR}/splits/${DATASET}_positive.parquet"
    OUT_FILE="${BASE_DIR}/fgp/${DATASET}_positive.parquet"
elif [ ${SLURM_ARRAY_TASK_ID} -eq 51 ]; then
    INPUT="${BASE_DIR}/splits/${DATASET}_negative_test.parquet"
    OUT_FILE="${BASE_DIR}/fgp/${DATASET}_negative_test.parquet"
fi

echo "=== RUNNING PART ${PART} ==="
echo "Dataset:     ${DATASET}"
echo "Input file:  ${INPUT}"
echo "Output file: ${OUT_FILE}"
echo "CPUs:        ${SLURM_CPUS_PER_TASK}"

python -u ${PHARM_PROJECT_ROOT}/src/data_preprocessing/2_generate_fingerprint.py \
    --dataset "${DATASET}" \
    --fingerprint "X_ecfp_2" \
    --n-proc "${SLURM_CPUS_PER_TASK}"
