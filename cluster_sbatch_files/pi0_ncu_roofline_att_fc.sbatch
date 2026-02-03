#!/bin/bash
#SBATCH -J pi0_ncu_att_fc
#SBATCH -p gpu_part
#SBATCH -w fury0
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH -t 10:00:00
#SBATCH -o /mnt/galactica/ocakir/logs/%x_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/%x_%j.err

set -euo pipefail

mkdir -p /mnt/galactica/ocakir/logs
mkdir -p /mnt/galactica/ocakir/tmp/ncu
chmod 700 /mnt/galactica/ocakir/tmp /mnt/galactica/ocakir/tmp/ncu
export TMPDIR=/mnt/galactica/ocakir/tmp/ncu

cd /mnt/galactica/ocakir/smolvla/lerobot

source /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
echo "Active conda env: ${CONDA_DEFAULT_ENV:-unknown}"
conda info --envs || true

export HF_HOME=/mnt/galactica/ocakir/hf_cache
export HF_LEROBOT_HOME=/mnt/galactica/ocakir/hf_cache/lerobot
export PATH=/usr/local/cuda/bin:$PATH

hostname
nvidia-smi || true
python -V

/usr/local/cuda/bin/ncu --nvtx \
  --nvtx-include "regex:policy\\.model\\..*/" \
  --set roofline \
  --target-processes all \
  --replay-mode kernel \
  -o /mnt/galactica/ocakir/smolvla/lerobot/prof/ncu_roofline_nvtx_fury0_pi0_full_launch_bs16_lo \
   python /mnt/galactica/ocakir/smolvla/lerobot/prof/pi0_inference.py \
    --checkpoint_path lerobot/pi0_base \
    --device cuda \
    --warmup_iters 5 \
    --profile_iters 20 \
    --nvtx_layers \
    --nvtx_leaf_only \
    --fresh_batch_each_iter \
    --time_preprocessor \
    --batch_size 16
