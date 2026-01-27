#!/bin/bash
#SBATCH -J smolvla_ncu_batch
#SBATCH -p gpu_part
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH -t 00:30:00
#SBATCH -o /mnt/galactica/ocakir/logs/%x_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/%x_%j.err

set -euo pipefail

cd /mnt/galactica/ocakir/smolvla/lerobot

source /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

mkdir -p /mnt/galactica/ocakir/logs

mkdir -p /mnt/galactica/ocakir/tmp/ncu
chmod 700 /mnt/galactica/ocakir/tmp /mnt/galactica/ocakir/tmp/ncu
export TMPDIR=/mnt/galactica/ocakir/tmp/ncu

hostname
nvidia-smi || true
python -V
ncu --version

# Configure these as needed
NCU_BIN="${NCU_BIN:ncu}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/train/my_smolvla3/checkpoints/last/pretrained_model}"
DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/svla_so101_pickplace}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WARMUP_ITERS="${WARMUP_ITERS:-5}"
PROFILE_ITERS="${PROFILE_ITERS:-20}"
LAUNCH_COUNT="${LAUNCH_COUNT:-200}"
OUT_DIR="${OUT_DIR:-/mnt/galactica/ocakir/smolvla/lerobot/prof}"
OUT_TAG="${OUT_TAG:-ncu_roofline_batch${BATCH_SIZE}}"

mkdir -p "${OUT_DIR}"

${NCU_BIN} --nvtx --set roofline --target-processes all --replay-mode kernel --launch-count "${LAUNCH_COUNT}" \
  -o "${OUT_DIR}/${OUT_TAG}" \
  -- python /mnt/galactica/ocakir/smolvla/lerobot/prof/smolvla_inference.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --dataset_repo_id "${DATASET_REPO_ID}" \
    --device cuda \
    --batch_size "${BATCH_SIZE}" \
    --fresh_batch_each_iter \
    --time_preprocessor \
    --warmup_iters "${WARMUP_ITERS}" \
    --profile_iters "${PROFILE_ITERS}" \
    --nvtx_layers --nvtx_leaf_only

# python /mnt/galactica/ocakir/smolvla/lerobot/prof/export_ncu_and_plot.py \
#   --rep "${OUT_DIR}/${OUT_TAG}.ncu-rep"

# python /mnt/galactica/ocakir/smolvla/lerobot/prof/analyze_roofline_bounds.py \
#   --csv "${OUT_DIR}/${OUT_TAG}_bounds.csv"
