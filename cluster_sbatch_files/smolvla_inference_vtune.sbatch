#!/bin/bash
#SBATCH -J smolvla_infer_vtune_cpu
#SBATCH -p cpu_part
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH -t 00:20:00
#SBATCH -o /mnt/galactica/ocakir/logs/%x_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/%x_%j.err

set -euo pipefail

# --- Paths ---
WORKDIR="/mnt/galactica/ocakir/smolvla/lerobot"
LOGDIR="/mnt/galactica/ocakir/logs"
VTUNE_ROOT="/mnt/galactica/ocakir/intel/vtune/vtune/2025.8"
RESULTS_BASE="/mnt/galactica/ocakir/vtune_runs"
CONDA_SH="/mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh"

CHECKPOINT="outputs/train/my_smolvla3/checkpoints/last/pretrained_model"
DATASET_REPO="lerobot/svla_so101_pickplace"

# --- Prep ---
mkdir -p "${LOGDIR}" "${RESULTS_BASE}"
cd "${WORKDIR}"

# Conda env
source "${CONDA_SH}"
conda activate lerobot

# VTune env
source "${VTUNE_ROOT}/vtune-vars.sh"

# --- Sanity prints ---
echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "PWD=$PWD"
python -V
which python
which vtune
vtune --version
echo "================"

# --- VTune result directory (unique per job) ---
RUN_ID="smolvla_infer_cpu_${SLURM_JOB_ID}"
VTUNE_RESULT="${RESULTS_BASE}/${RUN_ID}"

# Optional: reduce HF noise / keep caches consistent
export HF_HOME="/mnt/galactica/ocakir/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

# CPU run
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# --- Run VTune collection ---
vtune -collect hotspots \
      -result-dir "${VTUNE_RESULT}" \
      -finalization-mode fast \
      -- \
      python /mnt/galactica/ocakir/smolvla/lerobot/prof/smolvla_inference.py \
        --checkpoint_path "${CHECKPOINT}" \
        --dataset_repo_id "${DATASET_REPO}" \
        --device cpu \
        --warmup_iters 5 \
        --profile_iters 20

echo "VTune result saved to: ${VTUNE_RESULT}"
echo "Done."
