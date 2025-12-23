#!/bin/bash -l
#SBATCH --job-name=pharmacophore_align
#SBATCH --output=logs/pharmacophore_align_%A_%a.out
#SBATCH --error=logs/pharmacophore_align_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
# NOTE: Set your cluster-specific partition/account outside the repo:
#   #SBATCH --partition=YOUR_PARTITION
#   #SBATCH -A YOUR_ACCOUNT

set -euo pipefail
mkdir -p logs

# (Optional) conda activation – keep generic
# source /path/to/miniconda/etc/profile.d/conda.sh
# conda activate your_env

# ----------------------------
# Config (override via --export)
# ----------------------------
# Example:
# sbatch --export=ALL,DATASET=enamine,PHARM_KIND=k4,HYPOTHESIS=hypothesis/k4.json hpc/pharmacophore_alignment_pipeline.template.sh

DATASET="${DATASET:-enamine}"            # e.g. enamine | savi | ...
PHARM_KIND="${PHARM_KIND:-k4}"           # k3 | k4 | k5
HYPOTHESIS="${HYPOTHESIS:-hypothesis/k4.json}"

# Repo-safe roots (edit to your environment)
BASE_DIR="${BASE_DIR:-/path/to/project_root}"
SDF_ROOT="${SDF_ROOT:-${BASE_DIR}/sdf_files/${DATASET}}"
PLOTS_ROOT="${PLOTS_ROOT:-${BASE_DIR}/plots/${PHARM_KIND}}"
OUT_ROOT="${OUT_ROOT:-${BASE_DIR}/labels_out/${PHARM_KIND}}"

SCRIPT="${SCRIPT:-data_preparation/pharmacophore_alignment_pipeline.py}"

IDX="${SLURM_ARRAY_TASK_ID}"

# ----------------------------
# Autodetect part_* directory
# ----------------------------
CANDIDATES=(
  "part_${IDX}"
  "$(printf 'part_%01d' "${IDX}")"
  "$(printf 'part_%02d' "${IDX}")"
  "$(printf 'part_%03d' "${IDX}")"
  "$(printf 'part_%04d' "${IDX}")"
  "$(printf 'part_%05d' "${IDX}")"
)

PART_NAME=""
for cand in "${CANDIDATES[@]}"; do
  if [[ -d "${SDF_ROOT}/${cand}" ]]; then
    PART_NAME="${cand}"
    break
  fi
done

# Fallback: regex match (handles arbitrary zero-padding)
if [[ -z "${PART_NAME}" ]]; then
  PART_NAME="$(ls -1 "${SDF_ROOT}" 2>/dev/null | grep -E "^part_0*${IDX}$" | head -n1 || true)"
fi

if [[ -z "${PART_NAME}" ]]; then
  echo "[WARN] No part directory for IDX=${IDX} in ${SDF_ROOT}. Tried: ${CANDIDATES[*]} and grep '^part_0*${IDX}$'. Skipping."
  exit 0
fi

PART_DIR="${SDF_ROOT}/${PART_NAME}"

echo "[INFO] Host: $(hostname)  Time: $(date)"
echo "[INFO] DATASET=${DATASET}"
echo "[INFO] PHARM_KIND=${PHARM_KIND}"
echo "[INFO] IDX=${IDX}  PART_NAME=${PART_NAME}"
echo "[INFO] PART_DIR=${PART_DIR}"
echo "[INFO] HYPOTHESIS=${HYPOTHESIS}"
echo "[INFO] PLOTS_ROOT=${PLOTS_ROOT}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] SCRIPT=${SCRIPT}"
echo "[INFO] CPUS=${SLURM_CPUS_PER_TASK}"

if [[ ! -d "${PART_DIR}" ]]; then
  echo "[WARN] Missing directory: ${PART_DIR} — skipping."
  exit 0
fi

python -u "${SCRIPT}" \
  --part-dir "${PART_DIR}" \
  --plots-root "${PLOTS_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --pharm "${PHARM_KIND}" \
  --hypo-json "${HYPOTHESIS}" \
  --cpus "${SLURM_CPUS_PER_TASK}" \
  --tol-core 1.0 \
  --tol-ar 2.0
